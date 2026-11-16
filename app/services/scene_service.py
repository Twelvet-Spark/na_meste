"""Revision-checked scene saves and two bounded native frozen snapshots.

Camera mutation publication must call invalidate(camera_id) AFTER its DB commit.
A same-source reconnect is also checked on every token use. Physical movement
without a source/stream change cannot be detected; operators must review it.
"""
import base64
import logging
import math
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from uuid import UUID, uuid4

import cv2
from fastapi import HTTPException
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.config import settings
from app.database.database import immediate_session
from app.database.models import AttendanceCamera, Camera, CameraScene
from app.rtsp.reader import CameraStatus, FramePacket
from app.scene_geometry import GeometryError, MetricTransform, build_metric_transform, validate_geometry
from app.time_utils import attendance_iso, utcnow

logger = logging.getLogger('app.scene')
SNAPSHOT_TTL = 300
MAX_SNAPSHOTS = 2
MAX_SNAPSHOT_PIXELS = 16_000_000


def _error(status, code, message):
    return HTTPException(status, detail=dict(code=code, message=message))


def _token(value):
    if not isinstance(value, str):
        raise _error(422, 'SCENE_SNAPSHOT_INVALID', 'snapshot_id must be a UUID')
    try:
        return str(UUID(value))
    except ValueError as exc:
        raise _error(422, 'SCENE_SNAPSHOT_INVALID', 'snapshot_id must be a UUID') from exc


@dataclass(frozen=True, slots=True)
class FrozenSceneSnapshot:
    snapshot_id: str
    camera_id: int
    packet: FramePacket
    width: int
    height: int
    expires_mono: float
    # Identity matters even if a replacement reader incorrectly reuses stream_id.
    reader: object = field(default=None, repr=False, compare=False)


class SceneService:
    def __init__(self, engine, manager, *, monotonic=time.monotonic):
        self.engine = engine
        self.manager = manager
        self._monotonic = monotonic
        self._lock = threading.RLock()
        self._snapshots: dict[str, FrozenSceneSnapshot] = {}
        self._encoding: set[str] = set()
        # Counts unique native-image reservations even after token lookup expiry.
        # Values are refcounts: two users of one token still consume one image.
        self._native_leases: dict[str, int] = {}
        self._metric_cache: dict[int, tuple[int, MetricTransform | None]] = {}
        self._invalidation_callbacks = []

    def add_invalidation_callback(self, callback):
        """Subscribe camera-id invalidation (e.g. segmentation jobs/embeddings)."""
        with self._lock:
            if callback not in self._invalidation_callbacks:
                self._invalidation_callbacks.append(callback)

    def invalidate(self, camera_id):
        """Publication seam for committed source changes, disable and deletion."""
        with self._lock:
            self._snapshots = {key: item for key, item in self._snapshots.items()
                               if item.camera_id != camera_id}
            self._metric_cache.pop(camera_id, None)
            callbacks = tuple(self._invalidation_callbacks)
        for callback in callbacks:
            try:
                callback(camera_id)
            except Exception:
                # A listener must not turn an already committed camera mutation
                # into an apparent failed mutation. Token checks still fail closed.
                logger.exception('Scene invalidation listener failed')

    def _purge_expired(self):
        now = self._monotonic()
        self._snapshots = {key: item for key, item in self._snapshots.items() if now < item.expires_mono}

    @staticmethod
    def _camera(db, camera_id, *, token_use=False):
        camera = db.get(Camera, camera_id)
        if camera is None:
            raise _error(409 if token_use else 404, 'CAMERA_SOURCE_CHANGED' if token_use else 'CAMERA_NOT_FOUND',
                         'Camera no longer exists' if token_use else 'Camera not found')
        source = db.get(AttendanceCamera, camera_id)
        if source is None:
            raise _error(409 if token_use else 503, 'CAMERA_SOURCE_CHANGED' if token_use else 'CAMERA_SOURCE_UNAVAILABLE',
                         'Camera source configuration is unavailable')
        return camera, source

    def _fresh(self, packet):
        age = self._monotonic() - packet.received_mono
        return math.isfinite(age) and 0 <= age <= min(settings.frame_stall_timeout, settings.attendance_frame_max_age)

    def _get_snapshot_locked(self, snapshot_id, camera_id, db):
        self._purge_expired()
        snapshot = self._snapshots.get(snapshot_id)
        if snapshot is None:
            raise _error(410, 'SCENE_SNAPSHOT_EXPIRED', 'Snapshot is unknown, invalidated or expired; freeze a new frame')
        if camera_id is not None and snapshot.camera_id != camera_id:
            raise _error(409, 'SCENE_SNAPSHOT_MISMATCH', 'Snapshot belongs to another camera')
        camera, source = self._camera(db, snapshot.camera_id, token_use=True)
        reader = self.manager.get(snapshot.camera_id)
        current = None if reader is None else reader.buffer.latest()
        if (not camera.enabled or source.source_revision != snapshot.packet.source_revision
                or reader is not snapshot.reader or reader.status() is not CameraStatus.ONLINE
                or reader.source_revision != snapshot.packet.source_revision or current is None
                or current.source_revision != snapshot.packet.source_revision
                or current.stream_id != snapshot.packet.stream_id
                or current.frame.shape[:2] != (snapshot.height, snapshot.width)):
            self._snapshots.pop(snapshot_id, None)
            raise _error(409, 'CAMERA_SOURCE_CHANGED', 'Camera source or stream changed; freeze a new frame')
        return snapshot

    def get_snapshot(self, snapshot_id, camera_id=None):
        """Validate a frozen token now; native inference must recheck on completion.

        Frozen-packet age may reach the TTL; only its INITIAL capture requires a
        fresh live packet. No caller may mutate the original read-only frame.
        """
        identifier = _token(snapshot_id)
        try:
            with self._lock, Session(self.engine) as db:
                return self._get_snapshot_locked(identifier, camera_id, db)
        except SQLAlchemyError as exc:
            raise _error(503, 'SCENE_STORAGE_UNAVAILABLE', 'Scene database is unavailable') from exc

    @contextmanager
    def lease_snapshot(self, snapshot_id, camera_id=None):
        """Retain a validated native image until work exits, without holding locks.

        Expiry/source invalidation removes public lookup visibility immediately,
        but cannot release a running native call's unique-token capacity slot.
        Callers must still get_snapshot AFTER inference before publishing.
        """
        identifier = _token(snapshot_id)
        try:
            with self._lock, Session(self.engine) as db:
                snapshot = self._get_snapshot_locked(identifier, camera_id, db)
                self._native_leases[identifier] = self._native_leases.get(identifier, 0) + 1
        except SQLAlchemyError as exc:
            raise _error(503, 'SCENE_STORAGE_UNAVAILABLE', 'Scene database is unavailable') from exc
        try:
            yield snapshot
        finally:
            with self._lock:
                remaining = self._native_leases[identifier] - 1
                if remaining:
                    self._native_leases[identifier] = remaining
                else:
                    del self._native_leases[identifier]

    def snapshot(self, camera_id):
        identifier = str(uuid4())
        try:
            with self._lock, Session(self.engine) as db:
                self._purge_expired()
                camera, source = self._camera(db, camera_id)
                if not camera.enabled:
                    raise _error(409, 'CAMERA_DISABLED', 'Enable the camera before freezing a scene')
                if len(self._snapshots.keys() | self._encoding | self._native_leases.keys()) >= MAX_SNAPSHOTS:
                    raise _error(503, 'SCENE_BUSY', 'Two frozen snapshots are already retained; wait for expiry')
                reader = self.manager.get(camera_id)
                if reader is None or reader.status() is not CameraStatus.ONLINE:
                    raise _error(503, 'CAMERA_NOT_ONLINE', 'Camera is not online')
                packet = reader.buffer.latest()
                if (packet is None or packet.source_revision != source.source_revision
                        or reader.source_revision != source.source_revision or not self._fresh(packet)):
                    raise _error(503, 'SCENE_FRAME_UNAVAILABLE', 'No fresh frame for this camera source')
                frame = packet.frame
                if (frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype.name != 'uint8'
                        or frame.flags.writeable):
                    raise _error(503, 'SCENE_FRAME_INVALID', 'Expected an immutable native BGR frame')
                height, width = frame.shape[:2]
                if width <= 0 or height <= 0 or width * height > MAX_SNAPSHOT_PIXELS:
                    raise _error(503, 'SCENE_FRAME_TOO_LARGE', 'Frozen frames must be at most 16 megapixels')
                snapshot = FrozenSceneSnapshot(identifier, camera_id, packet, width, height,
                                               self._monotonic() + SNAPSHOT_TTL, reader)
                # Reserve before native encoding so concurrent requests cannot
                # retain more than two full-resolution images, including work.
                self._snapshots[identifier] = snapshot
                self._encoding.add(identifier)
            try:
                ok, encoded = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            except cv2.error as exc:
                raise _error(503, 'SCENE_SNAPSHOT_ENCODE_FAILED', 'Could not encode scene snapshot') from exc
            if not ok:
                raise _error(503, 'SCENE_SNAPSHOT_ENCODE_FAILED', 'Could not encode scene snapshot')
            jpeg_base64 = base64.b64encode(encoded).decode('ascii')
            with self._lock, Session(self.engine) as db:
                self._get_snapshot_locked(identifier, camera_id, db)
                if not self._fresh(packet):
                    raise _error(503, 'SCENE_FRAME_UNAVAILABLE', 'Frame became stale while encoding; freeze again')
                return dict(snapshot_id=identifier, camera_id=camera_id, jpeg_base64=jpeg_base64,
                            source_revision=packet.source_revision, stream_id=packet.stream_id,
                            seq=packet.seq, received_at=attendance_iso(packet.received_at),
                            width=width, height=height,
                            expires_in_seconds=math.ceil(snapshot.expires_mono - self._monotonic()))
        except SQLAlchemyError as exc:
            with self._lock:
                self._snapshots.pop(identifier, None)
            raise _error(503, 'SCENE_STORAGE_UNAVAILABLE', 'Scene database is unavailable') from exc
        except BaseException:
            with self._lock:
                self._snapshots.pop(identifier, None)
            raise
        finally:
            with self._lock:
                self._encoding.discard(identifier)

    def _transform(self, row):
        cached = self._metric_cache.get(row.camera_id)
        if cached is None or cached[0] != row.revision:
            # Stored/imported data uses the same authoritative contract.
            geometry = validate_geometry(row.geometry)
            transform = build_metric_transform(geometry)
            cached = (row.revision, transform)
            self._metric_cache[row.camera_id] = cached
        return cached[1]

    @staticmethod
    def _view(camera, source, row, transform=None):
        status = ('UNSET' if row is None else
                  'CURRENT' if camera.enabled and row.source_revision > 0
                  and row.source_revision == source.source_revision else 'STALE')
        metric = None
        if status == 'CURRENT' and transform is not None:
            width, height = transform.rectified_size()
            metric = dict(image_to_metres=transform.image_to_metres.tolist(),
                          metres_to_image=transform.metres_to_image.tolist(),
                          rectified_width=width, rectified_height=height)
        return dict(camera_id=camera.id, revision=0 if row is None else row.revision,
                    source_revision=None if row is None else row.source_revision,
                    current_source_revision=source.source_revision, status=status,
                    reference_width=None if row is None else row.reference_width,
                    reference_height=None if row is None else row.reference_height,
                    geometry=None if row is None else row.geometry,
                    updated_at=None if row is None else attendance_iso(row.updated_at), metric_transform=metric)

    def get(self, camera_id):
        try:
            with self._lock, Session(self.engine) as db:
                camera, source = self._camera(db, camera_id)
                row = db.get(CameraScene, camera_id)
                transform = None if row is None else self._transform(row)
                return self._view(camera, source, row, transform)
        except GeometryError as exc:
            raise _error(503, 'SCENE_GEOMETRY_INVALID', 'Saved scene geometry is invalid; review required') from exc
        except SQLAlchemyError as exc:
            raise _error(503, 'SCENE_STORAGE_UNAVAILABLE', 'Scene database is unavailable') from exc

    def get_metric_transform(self, camera_id):
        """Only current saved calibration may project live floor positions."""
        with self._lock:
            view = self.get(camera_id)
            return self._metric_cache[camera_id][1] if view['status'] == 'CURRENT' else None

    def save(self, camera_id, payload):
        if not isinstance(payload, dict) or set(payload) != {
                'snapshot_id', 'expected_revision', 'expected_source_revision', 'geometry'}:
            raise _error(422, 'SCENE_INPUT_INVALID', 'Supply only snapshot_id, expected_revision, expected_source_revision, geometry')
        for name, minimum in (('expected_revision', 0), ('expected_source_revision', 1)):
            if type(payload[name]) is not int or payload[name] < minimum:
                raise _error(422, 'SCENE_INPUT_INVALID', f'{name} must be an integer >= {minimum}')
        identifier = _token(payload['snapshot_id'])
        # Preserve source-CAS semantics even after the committed source callback
        # has evicted the old token. An unchanged-source expired token is 410.
        try:
            with Session(self.engine) as db:
                _camera, source = self._camera(db, camera_id, token_use=True)
                if source.source_revision != payload['expected_source_revision']:
                    raise _error(409, 'CAMERA_SOURCE_CHANGED', 'Camera source changed; freeze a new frame')
        except SQLAlchemyError as exc:
            raise _error(503, 'SCENE_STORAGE_UNAVAILABLE', 'Scene database is unavailable') from exc
        # Admission check precedes geometry work. No writer lock during math.
        self.get_snapshot(identifier, camera_id)
        try:
            geometry = validate_geometry(payload['geometry'])
            transform = build_metric_transform(geometry)
        except GeometryError as exc:
            raise _error(422, 'SCENE_GEOMETRY_INVALID', str(exc)) from exc
        try:
            with self._lock:
                with immediate_session(self.engine) as db:
                    snapshot = self._get_snapshot_locked(identifier, camera_id, db)
                    camera, source = self._camera(db, camera_id, token_use=True)
                    row = db.get(CameraScene, camera_id)
                    revision = 0 if row is None else row.revision
                    if revision != payload['expected_revision']:
                        raise _error(409, 'SCENE_CHANGED', 'Scene was edited; reconcile the draft before saving')
                    if (source.source_revision != payload['expected_source_revision']
                            or snapshot.packet.source_revision != source.source_revision):
                        raise _error(409, 'CAMERA_SOURCE_CHANGED', 'Camera source changed; freeze a new frame')
                    if row is None:
                        row = CameraScene(camera_id=camera_id)
                        db.add(row)
                    row.revision = revision + 1
                    row.source_revision = source.source_revision
                    row.reference_width, row.reference_height = snapshot.width, snapshot.height
                    row.geometry = geometry
                    row.updated_at = utcnow()
                    db.flush()
                    # Source writes are excluded by BEGIN IMMEDIATE; reader
                    # reconnects are not. Check again after the actual DML.
                    self._get_snapshot_locked(identifier, camera_id, db)
                    result = self._view(camera, source, row, transform)
                self._metric_cache[camera_id] = (revision + 1, transform)
                return result
        except SQLAlchemyError as exc:
            raise _error(503, 'SCENE_STORAGE_UNAVAILABLE', 'Scene was not saved; database unavailable') from exc
