"""Manual recording admission, clocks and lifecycle, independent of attendance."""
import hashlib
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from uuid import UUID, uuid4

from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict, PrivateAttr
from sqlalchemy import func
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session

from app.config import settings
from app.database.database import immediate_session
from app.database.models import (OperatorRequest, RecordingConnection, RecordingGap,
                                 RecordingSegment, RecordingSession)
from app.media.process_guard import MediaProcessGuard
from app.media.recorder import SegmentRecorder
from app.media.storage import ACTIVE, RecordingStorage
from app.rtsp.reader import Backoff
from app.services import camera_service
from app.time_utils import utcnow

logger = logging.getLogger("app.recordings")


def aware(value):
    return value.replace(tzinfo=timezone.utc) if value is not None and value.tzinfo is None else value


class SessionView(BaseModel):
    model_config = ConfigDict(frozen=True)
    id: str
    camera_id: int
    camera_name: str
    source_revision: int
    requested_at: datetime
    started_at: datetime | None
    deadline_at: datetime
    ended_at: datetime | None
    status: str
    reason_code: str | None
    bytes_recorded: int
    available_from: datetime | None
    available_to: datetime | None
    _replayed: bool = PrivateAttr(default=False)


@dataclass
class _Job:
    session_id: str
    source: object = field(repr=False)
    admission: int
    requested_at: datetime
    deadline: datetime
    started_mono: float
    duration: int
    recorder: SegmentRecorder | None = None
    thread: threading.Thread | None = None
    stop_event: threading.Event = field(default_factory=threading.Event)
    final_status: str = "STOPPED"
    reason: str | None = None
    stop_requested_at: datetime | None = None
    gap_id: str | None = None


class RecordingManager:
    def __init__(self, engine, root, ffmpeg_path, ffprobe_path, *, now=utcnow, monotonic=time.monotonic):
        self.engine, self.ffmpeg, self.ffprobe = engine, ffmpeg_path, ffprobe_path
        self.now, self.monotonic = now, monotonic
        self.lock = threading.RLock()
        self.jobs = {}
        self.closing = False
        self.storage = RecordingStorage(engine, root, now=now, monotonic=monotonic)
        self._maintenance_stop = threading.Event()
        try:
            self._recover()
            self._maintenance = threading.Thread(target=self._maintain, daemon=True, name="recording-storage")
            self._maintenance.start()
        except BaseException:
            self.storage.close()
            raise

    def _view(self, db, row):
        if row is None:
            raise HTTPException(404, "RECORDING_NOT_FOUND")
        total = db.query(func.coalesce(func.sum(RecordingSegment.bytes), 0)).filter_by(session_id=row.id).scalar()
        first, last = db.query(func.min(RecordingSegment.started_at), func.max(RecordingSegment.ended_at)).filter_by(
            session_id=row.id, state="READY").one()
        return SessionView(id=row.id, camera_id=row.camera_id, camera_name=row.camera_name,
            source_revision=row.source_revision, requested_at=aware(row.requested_at), started_at=aware(row.started_at),
            deadline_at=aware(row.deadline_at), ended_at=aware(row.ended_at), status=row.status,
            reason_code=row.reason_code, bytes_recorded=total, available_from=aware(first), available_to=aware(last))

    def get(self, session_id):
        with Session(self.engine) as db:
            return self._view(db, db.get(RecordingSession, str(session_id)))

    def list_sessions(self, *, camera_id=None, limit=50, offset=0):
        with Session(self.engine) as db:
            query = db.query(RecordingSession)
            if camera_id is not None:
                query = query.filter_by(camera_id=camera_id)
            return [self._view(db, row) for row in query.order_by(
                RecordingSession.requested_at.desc(), RecordingSession.id).offset(offset).limit(limit).all()]

    def start(self, camera_id, request_id, duration_seconds=86400) -> SessionView:
        if type(duration_seconds) is not int or not 60 <= duration_seconds <= 86400:
            raise HTTPException(422, "RECORDING_DURATION_INVALID")
        if type(camera_id) is not int or camera_id <= 0:
            raise HTTPException(422, "CAMERA_ID_INVALID")
        try:
            request_id = str(UUID(str(request_id)))
        except (TypeError, ValueError, AttributeError):
            raise HTTPException(422, "REQUEST_ID_INVALID") from None
        payload = {"camera_id": camera_id, "duration_seconds": duration_seconds}
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        with self.lock:
            if self.closing:
                raise HTTPException(503, "MEDIA_SHUTTING_DOWN")
            # Replay precedes infrastructure checks: lost-response retries never relaunch.
            with Session(self.engine) as db:
                prior = db.get(OperatorRequest, request_id)
                if prior:
                    if prior.operation != "recording.start" or prior.payload_sha256 != digest:
                        raise HTTPException(409, "REQUEST_ID_CONFLICT")
                    result = self._view(db, db.get(RecordingSession, request_id))
                    result._replayed = True
                    return result
            MediaProcessGuard.require()
            if not self.ffmpeg or not self.ffprobe:
                raise HTTPException(503, "FFMPEG_UNAVAILABLE")
            try:
                if not self.storage.enforce():
                    raise HTTPException(503, "STORAGE_FULL")
                with immediate_session(self.engine) as db:
                    prior = db.get(OperatorRequest, request_id)
                    if prior:
                        if prior.operation != "recording.start" or prior.payload_sha256 != digest:
                            raise HTTPException(409, "REQUEST_ID_CONFLICT")
                        result = self._view(db, db.get(RecordingSession, request_id))
                        result._replayed = True
                        return result
                    source = camera_service.camera_snapshot(db, camera_id)
                    if source is None:
                        raise HTTPException(404, "CAMERA_NOT_FOUND")
                    if not source.enabled:
                        raise HTTPException(409, "CAMERA_DISABLED")
                    if db.query(RecordingSession.id).filter(RecordingSession.camera_id == camera_id,
                                                           RecordingSession.status.in_(ACTIVE)).first():
                        raise HTTPException(409, "RECORDING_ALREADY_ACTIVE")
                    requested = self.now()
                    mono = self.monotonic()
                    deadline = requested + timedelta(seconds=duration_seconds)
                    # The same process-local sequence used by camera commits, under
                    # the same writer lock. It is an admission barrier, NEVER updated
                    # by later notifications. Disable/re-enable need no schema field.
                    admission = next(camera_service._publication_sequences)
                    db.add(RecordingSession(id=request_id, camera_id=camera_id, camera_name=source.name,
                        source_revision=source.source_revision, requested_at=requested, deadline_at=deadline,
                        status="STARTING"))
                    db.add(OperatorRequest(id=request_id, operation="recording.start", payload_sha256=digest,
                                           result={"id": request_id}))
                    db.flush()
                    initial_gap = str(uuid4())
                    db.add(RecordingGap(id=initial_gap, session_id=request_id,
                                        started_at=requested, reason_code="CONNECTING"))
                job = _Job(request_id, source, admission, requested, deadline, mono, duration_seconds,
                           gap_id=initial_gap)
                self.jobs[request_id] = job
                try:
                    job.recorder = self._launch(job)
                except Exception as exc:
                    reason = (exc.detail if isinstance(exc, HTTPException) and isinstance(exc.detail, str)
                              else "MEDIA_LAUNCH_FAILED")
                    if self._alive(job):
                        reason = "MEDIA_STOP_PENDING"
                        self._terminal(job, "STOPPING", reason)
                    else:
                        self._terminal(job, "FAILED", reason)
                        self.jobs.pop(request_id, None)
                    status = 409 if reason in ("SOURCE_CHANGED", "CAMERA_DISABLED") else 503
                    raise HTTPException(status, reason) from None
                job.thread = threading.Thread(target=self._run, args=(job,), daemon=True,
                                               name=f"recording-{camera_id}")
                try:
                    job.thread.start()
                except RuntimeError:
                    job.thread = None
                    try:
                        job.recorder.finish()
                    finally:
                        if self._alive(job):
                            self._terminal(job, "STOPPING", "MEDIA_STOP_PENDING")
                        else:
                            self._terminal(job, "FAILED", "MEDIA_LAUNCH_FAILED")
                            self.jobs.pop(request_id, None)
                    raise HTTPException(503, "MEDIA_LAUNCH_FAILED") from None
            except IntegrityError:
                raise HTTPException(409, "RECORDING_ALREADY_ACTIVE") from None
            except SQLAlchemyError:
                raise HTTPException(503, "MEDIA_DATABASE_UNAVAILABLE") from None
            except OSError:
                raise HTTPException(503, "MEDIA_STORAGE_UNAVAILABLE") from None
        return self.get(request_id)

    def _launch(self, job):
        connection_id = str(uuid4())
        directory = f"{job.session_id}/{connection_id}"
        with self.storage.lock:
            self.storage.path(job.session_id, exists=False, directory=True).mkdir(exist_ok=True)
            self.storage.path(directory, exists=False, directory=True).mkdir()
        with immediate_session(self.engine) as db:
            db.add(RecordingConnection(id=connection_id, session_id=job.session_id, directory=directory,
                                      source_revision=job.source.source_revision, created_at=self.now()))
        recorder = SegmentRecorder(self.engine, self.storage, connection_id, self.ffmpeg, self.ffprobe,
                                   now=self.now, monotonic=self.monotonic)
        job.recorder = recorder
        try:
            # Connection row is already durable. Hold SQLite's writer lock through
            # launch admission so a camera edit cannot slip between check and spawn.
            with immediate_session(self.engine) as db:
                current = camera_service.camera_snapshot(db, job.source.id)
                if current is None or not current.enabled:
                    raise HTTPException(409, "CAMERA_DISABLED")
                if current.source_revision != job.source.source_revision:
                    raise HTTPException(409, "SOURCE_CHANGED")
                if job.stop_event.is_set():
                    raise HTTPException(409, job.reason or "RECORDING_STOPPED")
                remaining = min(job.duration - (self.monotonic() - job.started_mono),
                                (job.deadline - self.now()).total_seconds())
                if remaining <= 0:
                    raise HTTPException(409, "RECORDING_DEADLINE")
                recorder.start(job.source, remaining)
            return recorder
        except BaseException:
            recorder.finish()
            raise

    def _request_stop(self, job, status, reason):
        with self.lock:
            if not job.stop_event.is_set():
                job.final_status, job.reason = status, reason
                job.stop_requested_at = self.now()
                job.stop_event.set()

    def _clock_stop(self, job):
        mono_elapsed = self.monotonic() - job.started_mono
        utc_elapsed = (self.now() - job.requested_at).total_seconds()
        if abs(utc_elapsed - mono_elapsed) > 2:
            self._request_stop(job, "INTERRUPTED", "CLOCK_DISCONTINUITY")
        elif mono_elapsed >= job.duration or self.now() >= job.deadline:
            self._request_stop(job, "COMPLETED", None)

    def _run(self, job):
        backoff = Backoff(settings.rtsp_reconnect_max_delay)
        retry_at = None
        gap_id = job.gap_id
        try:
            while not job.stop_event.is_set():
                self._clock_stop(job)
                if job.stop_event.is_set():
                    break
                recorder = job.recorder
                if recorder is None:
                    if self.monotonic() < retry_at:
                        job.stop_event.wait(.25)
                        continue
                    try:
                        with self.lock:
                            if job.stop_event.is_set():
                                break
                            job.recorder = self._launch(job)
                        retry_at = None
                    except HTTPException as exc:
                        if exc.detail in ("SOURCE_CHANGED", "CAMERA_DISABLED"):
                            self._request_stop(job, "INTERRUPTED", exc.detail)
                        else:
                            retry_at = self.monotonic() + backoff.next()
                    except (OSError, SQLAlchemyError):
                        retry_at = self.monotonic() + backoff.next()
                    if job.recorder is not None and job.recorder.closed:
                        job.recorder = None
                    elif retry_at is not None and self._alive(job):
                        self._request_stop(job, "INTERRUPTED", "MEDIA_STOP_PENDING")
                else:
                    recorder.scan()
                    if recorder.first_output_mono is not None and gap_id:
                        with immediate_session(self.engine) as db:
                            gap = db.get(RecordingGap, gap_id)
                            connection = db.get(RecordingConnection, recorder.connection_id)
                            gap.ended_at = connection.anchor_utc
                        gap_id = None
                        backoff.reset()
                    stall = (settings.frame_stall_timeout if recorder.first_output_mono is not None
                             else max(30, settings.rtsp_connect_timeout * 3))
                    if (recorder.process.poll() is not None
                            or self.monotonic() - recorder.last_progress >= stall):
                        gap_start = recorder.last_progress_utc
                        recorder.finish()
                        with Session(self.engine) as db:
                            last_end = db.query(func.max(RecordingSegment.ended_at)).filter_by(
                                connection_id=recorder.connection_id, state="READY").scalar()
                        if last_end is not None:
                            gap_start = min(last_end, self.now())
                        job.recorder = None
                        retry_at = self.monotonic() + backoff.next()
                        with immediate_session(self.engine) as db:
                            db.get(RecordingSession, job.session_id).status = "RECONNECTING"
                            if gap_id is None:
                                gap_id = str(uuid4())
                                db.add(RecordingGap(id=gap_id, session_id=job.session_id,
                                                    started_at=gap_start, reason_code="CONNECTION_LOST"))
                            else:
                                db.get(RecordingGap, gap_id).reason_code = "CONNECTION_LOST"
                job.stop_event.wait(.25)
        except Exception:
            self._request_stop(job, "FAILED", "MEDIA_PROCESS_FAILED")
            logger.warning("Recording worker failed: MEDIA_PROCESS_FAILED")
        finally:
            try:
                with immediate_session(self.engine) as db:
                    db.get(RecordingSession, job.session_id).status = "STOPPING"
                if job.recorder:
                    job.recorder.finish(recovering=job.reason == "CLOCK_DISCONTINUITY")
                self._terminal(job, job.final_status, job.reason)
            except Exception:
                logger.warning("Recording finalization failed: MEDIA_FINALIZATION_FAILED")
                try:
                    self._terminal(job, "STOPPING" if self._alive(job) else "FAILED",
                                   "MEDIA_STOP_PENDING" if self._alive(job) else "MEDIA_FINALIZATION_FAILED")
                except SQLAlchemyError:
                    pass  # Persisted active state is interrupted by next root owner.
            finally:
                with self.lock:
                    if not self._alive(job):
                        self.jobs.pop(job.session_id, None)

    @staticmethod
    def _alive(job):
        return job.recorder is not None and job.recorder.process is not None and job.recorder.process.poll() is None

    def _terminal(self, job, status, reason):
        ended = self.now()
        with immediate_session(self.engine) as db:
            row = db.get(RecordingSession, job.session_id)
            row.status, row.reason_code, row.ended_at = status, reason, ended
            if status == "STOPPING":
                row.ended_at = None
                return
            for gap in db.query(RecordingGap).filter_by(session_id=job.session_id, ended_at=None).all():
                gap.ended_at = ended
            if reason in ("STORAGE_FULL", "MEDIA_PROCESS_FAILED", "MEDIA_LAUNCH_FAILED"):
                db.add(RecordingGap(id=str(uuid4()), session_id=job.session_id,
                                    started_at=job.stop_requested_at or ended, ended_at=ended, reason_code=reason))

    def stop(self, session_id) -> SessionView:
        session_id = str(session_id)
        with self.lock:
            job = self.jobs.get(session_id)
            if job:
                self._request_stop(job, "STOPPED", None)
        if job and job.thread and job.thread is not threading.current_thread():
            job.thread.join(timeout=35)  # Never join while holding the lock its finally needs.
            if job.thread.is_alive():
                raise HTTPException(503, "MEDIA_STOP_PENDING")
        if job and (self._alive(job) or job.thread is None):
            try:
                if job.recorder:
                    job.recorder.finish()
            except Exception:
                raise HTTPException(503, "MEDIA_STOP_PENDING") from None
            self._terminal(job, job.final_status, job.reason)
            with self.lock:
                self.jobs.pop(session_id, None)
        return self.get(session_id)

    def source_changed(self, snapshot) -> None:
        with self.lock:
            for job in self.jobs.values():
                if snapshot.id != job.source.id or snapshot.publication_sequence <= job.admission:
                    continue
                if not snapshot.enabled:
                    self._request_stop(job, "INTERRUPTED", "CAMERA_DISABLED")
                elif snapshot.source_revision > job.source.source_revision:
                    self._request_stop(job, "INTERRUPTED", "SOURCE_CHANGED")
                # An archive-only edit is irrelevant to local media. A newer
                # enable does not erase a pending older disable after admission.

    def _maintain(self):
        while not self._maintenance_stop.wait(1):
            try:
                available = self.storage.enforce()
            except Exception:
                available = False
            if not available:
                with self.lock:
                    for job in self.jobs.values():
                        self._request_stop(job, "INTERRUPTED", "STORAGE_FULL")

    def _recover(self):
        with immediate_session(self.engine) as db:
            for row in db.query(RecordingSession).filter(RecordingSession.status.in_(ACTIVE)).all():
                row.status, row.reason_code, row.ended_at = "INTERRUPTED", "APPLICATION_RESTART", self.now()
            for gap in db.query(RecordingGap).filter_by(ended_at=None).all():
                gap.ended_at = self.now()
        try:
            MediaProcessGuard.require()
        except HTTPException:
            return  # Nonmedia UI remains available; don't mislabel files corrupt without a probe.
        if not self.ffprobe:
            return
        with Session(self.engine) as db:
            connections = [(row.id, row.directory) for row in db.query(RecordingConnection).all()]
        for connection_id, directory in connections:
            try:
                self.storage.path(directory, directory=True)
            except (OSError, ValueError):
                with immediate_session(self.engine) as db:
                    for segment in db.query(RecordingSegment).filter_by(connection_id=connection_id).all():
                        if segment.state not in ("EXPIRED", "DELETING"):
                            segment.state = "MISSING"
                continue
            recorder = SegmentRecorder(self.engine, self.storage, connection_id, self.ffmpeg, self.ffprobe,
                                       now=self.now, monotonic=self.monotonic)
            recorder.finish(recovering=True)
        self.storage.enforce()

    def snapshot(self) -> dict:
        policy = self.storage.snapshot()
        policy["oldest_available"] = aware(policy["oldest_available"])
        with Session(self.engine) as db:
            sessions = [self._view(db, row).model_dump(mode="json") for row in db.query(
                RecordingSession).filter(RecordingSession.status.in_(ACTIVE)).all()]
        return {"sessions": sessions, "storage": policy, "time_basis": "approximate_server_recording_time"}

    def shutdown(self) -> None:
        with self.lock:
            if self.closing:
                return
            self.closing = True
            jobs = list(self.jobs.values())
            for job in jobs:
                self._request_stop(job, "INTERRUPTED", "APPLICATION_SHUTDOWN")
        self._maintenance_stop.set()
        self._maintenance.join(timeout=35)
        for job in jobs:
            if job.thread:
                job.thread.join(timeout=35)
                if job.thread.is_alive():
                    raise RuntimeError("MEDIA_STOP_PENDING")
            if self._alive(job) or job.thread is None:
                self.stop(job.session_id)
        if self._maintenance.is_alive():
            raise RuntimeError("MEDIA_STOP_PENDING")
        self.storage.close()
