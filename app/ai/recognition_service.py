"""Single-owner batched person tracking and native, revision-bound face evidence.

Receipt timestamps describe server receipt, not camera exposure. Attendance proof
is short lived; ordinary identity uses independent retry, quality and vote policy.
Neither predicted positions nor replayed packets are new human observations.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field, replace
from datetime import datetime
from uuid import uuid4
from typing import TYPE_CHECKING

import numpy as np
from sqlalchemy.orm import Session

from app.ai.face_detector import ALIGNMENT_ID, FaceEngine
from app.ai.face_quality import FaceQuality, measure_face_quality, quality_reasons
from app.ai.face_recognition import EmbeddingStore, FaceMatch, is_normalized_embedding
from app.ai.model_bundle import ModelBundleError
from app.ai.tracker import ByteTracker, Track, TrackLifecycle
from app.config import settings
from app.services.attendance_evidence import encode_evidence
from app.rtsp.reader import FramePacket

if TYPE_CHECKING:
    from app.ai.detector import PersonDetector
    from sqlalchemy import Engine

logger = logging.getLogger("app.ai.service")
_CONFIRMATION_MAX_AGE = 0.75
_FACE_BATCH_LIMIT = 16


@dataclass(frozen=True, slots=True)
class RecognitionInput:
    camera_id: int
    packet: FramePacket
    attendance: bool = False


@dataclass(frozen=True, slots=True)
class FaceEvidence:
    frame_seq: int
    received_at: datetime
    received_mono: float
    candidate_employee_id: int | None
    candidate_employee_name: str | None
    score: float
    runner_up_employee_id: int | None
    runner_up_employee_name: str | None
    runner_up_score: float
    margin: float
    quality: FaceQuality
    store_revision: int
    confirmed_employee_id: int | None
    confirmed_employee_name: str | None
    supporting_frame_sequences: tuple[int, ...]
    supporting_received_monos: tuple[float, ...]
    supporting_received_ats: tuple[datetime, ...]
    confirmed_at_mono: float | None
    face_bbox: tuple[float, float, float, float] | None
    aligned_jpeg: bytes | None
    reason_codes: tuple[str, ...]


@dataclass
class DetectionResult:
    """One display-eligible person; bbox is normalized to frame dimensions."""
    track_id: int
    employee_id: int | None
    employee_name: str | None
    confidence: float | None
    bbox: tuple[float, float, float, float]
    state: str
    global_id: int | None = None
    association_uncertain: bool = False
    person_score: float = 0.
    fresh_face_evidence: FaceEvidence | None = None
    identity_reason_codes: tuple[str, ...] = ()
    visit_id: str | None = None


@dataclass
class FrameOutcome:
    detections: list[DetectionResult] = field(default_factory=list)
    tracking_generation: str = ""
    track_lifecycle: list[TrackLifecycle] = field(default_factory=list)
    # All detector person candidates, normalized, including recovery-only scores.
    person_regions: list[tuple[float, float, float, float]] = field(default_factory=list)
    observed_tracks: list[DetectionResult] = field(default_factory=list)
    face_deferred: int = 0
    stage_times_ms: dict[str, float] = field(default_factory=dict)
    face_unavailable: bool = False


@dataclass
class _FaceRun:
    candidate_id: int | None = None
    store_revision: int | None = None
    votes: list[tuple[int, float, datetime]] = field(default_factory=list)
    last_attempt: float = float("-inf")
    confirmed_at: float | None = None
    attendance: bool = False
    last_served: int = 0
    reason_codes: tuple[str, ...] = ()


@dataclass(slots=True)
class _FrameWork:
    input: RecognitionInput
    outcome: FrameOutcome
    tracks: list[Track] = field(default_factory=list)
    observed: bool = False
    evidence: dict[int, FaceEvidence] = field(default_factory=dict)
    reasons: dict[int, tuple[str, ...]] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class _FaceWork:
    input: RecognitionInput
    generation: str
    track_id: int
    store_revision: int
    quality: FaceQuality
    bbox: tuple[float, float, float, float] | None
    aligned: np.ndarray | None
    jpeg: bytes | None
    reasons: tuple[str, ...]


class RecognitionService:
    def __init__(
        self,
        detector: PersonDetector,
        face_engine: FaceEngine | None,
        store: EmbeddingStore | None,
        min_confidence: float = 0.60,
        min_confirmations: int = 3,
        retry_interval: float = 0.5,
        track_lost_timeout: float = 3.,
        min_hits: int = 2,
        tracker_high_threshold: float = .5,
        *,
        live_face_min_size: int | None = None,
        live_face_margin: float | None = None,
        live_face_window: float | None = None,
        live_face_recheck_interval: float | None = None,
    ):
        self.detector = detector
        self.faces = face_engine
        self.store = store
        self.min_confidence = min_confidence
        self.min_confirmations = min_confirmations
        self.retry_interval = retry_interval
        self.track_lost_timeout = track_lost_timeout
        self.min_hits = min_hits
        self.tracker_high_threshold = tracker_high_threshold
        self.live_face_min_size = (settings.live_face_min_size if live_face_min_size is None
                                   else live_face_min_size)
        self.live_face_margin = (settings.live_face_margin if live_face_margin is None
                                 else live_face_margin)
        self.live_face_window = (settings.live_face_window if live_face_window is None
                                 else live_face_window)
        self.live_face_recheck_interval = (
            settings.live_face_recheck_interval if live_face_recheck_interval is None
            else live_face_recheck_interval
        )
        self._trackers: dict[int, ByteTracker] = {}
        self._generations: dict[int, str] = {}
        self._last_frame_seq: dict[int, int] = {}
        self._last_packets: dict[int, tuple[int, str, int]] = {}
        self._face_runs: dict[tuple[int, int], _FaceRun] = {}
        self._store_revision = store.revision if store is not None else None
        self._expected_gallery_revision: int | None = None
        self._last_gallery_refresh = float("-inf")
        self._face_turn = 0
        self._face_camera_served: dict[int, int] = {}
        self._face_detection_failed = False
        self._face_recognition_failed = False
        self._lock = threading.Lock()

    def set_expected_gallery_revision(self, revision: int) -> None:
        """Set the durable admission token read by the worker before a batch.

        A committed employee edit invalidates votes even if publishing the new
        in-memory gallery failed. The worker rechecks the durable token after
        native inference, under publication_guard, before applying any outcome.
        """
        with self._lock:
            previous = (self._store_revision if self._expected_gallery_revision is None
                        else self._expected_gallery_revision)
            self._expected_gallery_revision = revision
            if revision != previous:
                for tracker in self._trackers.values():
                    for track in tracker.tracks:
                        track.clear_identity()
                self._face_runs.clear()
            self._invalidate_gallery()

    def refresh_gallery(self, engine: Engine, *, now: float | None = None) -> bool:
        """Retry a stale gallery at most once/second, outside publication_guard.

        The inference owner calls this after setting the durable revision and
        before process_batch. A failed refresh never disables person tracking.
        """
        with self._lock:
            if self.store is None:
                return False
            expected = self._expected_gallery_revision
            if expected is None or self.store.revision == expected:
                return True
            now = time.monotonic() if now is None else now
            if now - self._last_gallery_refresh < 1.:
                return False
            self._last_gallery_refresh = now
            try:
                with Session(engine) as db:
                    self.store.refresh(db)
            except Exception:
                logger.warning("GALLERY_UNAVAILABLE: durable gallery refresh failed")
            self._invalidate_gallery()
            return self.store.revision == expected

    def process_batch(self, inputs: list[RecognitionInput]) -> list[FrameOutcome]:
        """One detector call, camera-local observations and one bounded face chunk."""
        if not inputs:
            return []
        if len({item.camera_id for item in inputs}) != len(inputs):
            raise ValueError("Recognition batches require distinct cameras")
        with self._lock:
            self._invalidate_gallery()
            frames: list[_FrameWork] = []
            admitted: list[_FrameWork] = []
            for item in inputs:
                camera_id, packet = item.camera_id, item.packet
                if camera_id not in self._generations:
                    self._generations[camera_id] = str(uuid4())
                tracker = self._trackers.get(camera_id)
                if tracker is None:
                    tracker = ByteTracker(lost_timeout=self.track_lost_timeout,
                                          high_threshold=self.tracker_high_threshold)
                    self._trackers[camera_id] = tracker
                current = _FrameWork(
                    item, FrameOutcome(tracking_generation=self._generations[camera_id]),
                )
                frames.append(current)
                if packet.seq <= self._last_frame_seq.get(camera_id, -1):
                    # Cached boxes are display-only: no tracker update, observation,
                    # lifecycle transition or fresh face evidence for replayed pixels.
                    current.tracks = [track for track in tracker.tracks if track.state == "tracked"]
                else:
                    admitted.append(current)

            person_started = time.perf_counter()
            batches = self.detector.detect_batch([work.input.packet.frame for work in admitted]) if admitted else []
            if len(batches) != len(admitted):
                raise ModelBundleError("MODEL_CONTRACT_INVALID", "person", "Detector batch length")
            person_ms = (time.perf_counter() - person_started) * 1000.
            tracking_ms = face_detection_ms = 0.
            face_work: list[_FaceWork] = []
            admission_revision = self._store_revision
            face_failed = False
            for current, persons in zip(admitted, batches):
                item, outcome = current.input, current.outcome
                camera_id, packet = item.camera_id, item.packet
                tracking_started = time.perf_counter()
                tracker = self._trackers[camera_id]
                self._last_frame_seq[camera_id] = packet.seq
                self._last_packets[camera_id] = (packet.source_revision, packet.stream_id, packet.seq)
                height, width = packet.frame.shape[:2]
                outcome.person_regions = [self._normalized_box(bbox, width, height)
                                          for bbox, _score in persons]
                current.tracks = tracker.update(persons, now=packet.received_mono)
                current.observed = True
                outcome.track_lifecycle = tracker.lifecycle
                tracks = {track.track_id: track for track in tracker.tracks}
                for transition in outcome.track_lifecycle:
                    if transition.state in ("LOST", "AMBIGUOUS", "EXPIRED"):
                        self._face_runs.pop((camera_id, transition.track_id), None)
                        track = tracks.get(transition.track_id)
                        if track is not None:
                            track.clear_identity()
                heads = [self._head_region(bbox, width, height) for bbox, _score in persons]
                tracking_ms += (time.perf_counter() - tracking_started) * 1000.
                unavailable = self._identity_unavailable()
                for track in current.tracks:
                    run = self._face_runs.get((camera_id, track.track_id))
                    if run is not None and run.attendance != item.attendance:
                        # Attendance votes must never borrow an ordinary confirmation.
                        self._clear_run(track, run)
                        self._face_runs.pop((camera_id, track.track_id))
                    if item.attendance:
                        self._expire_confirmation(camera_id, track, packet.received_mono)
                    if unavailable is not None:
                        track.clear_identity()
                        self._face_runs.pop((camera_id, track.track_id), None)
                        current.reasons[track.track_id] = (unavailable,)
                        continue
                    if face_failed:
                        continue
                    if track.hits < self.min_hits:
                        continue
                    run = self._face_runs.setdefault(
                        (camera_id, track.track_id), _FaceRun(attendance=item.attendance),
                    )
                    interval = (settings.attendance_face_interval if item.attendance else
                                self.live_face_recheck_interval if track.employee_id is not None
                                else self.retry_interval)
                    if packet.received_mono - run.last_attempt >= interval:
                        face_detection_started = time.perf_counter()
                        try:
                            face_work.append(self._detect_face(
                                item, outcome.tracking_generation, track, heads, admission_revision,
                            ))
                        except Exception:
                            self._face_detection_failed = face_failed = True
                            logger.warning("FACE_MODEL_UNAVAILABLE: face detection failed", exc_info=True)
                        else:
                            self._face_detection_failed = False
                        finally:
                            face_detection_ms += (time.perf_counter() - face_detection_started) * 1000.

            face_recognition_started = time.perf_counter()
            by_camera = {current.input.camera_id: current for current in frames}
            if face_failed:
                # A shared session failed after potentially selecting other
                # cameras' faces. Finish tracking every frame, but publish none
                # of the partial identity work from this batch.
                face_work.clear()
            eligible: list[_FaceWork] = []
            for work in face_work:
                if work.reasons:
                    evidence = self._apply_face(work, None)
                    if evidence is not None:
                        by_camera[work.input.camera_id].evidence[work.track_id] = evidence
                else:
                    eligible.append(work)
            # Least-recently-served tracks rotate across the entire camera batch.
            # Deferred tracks retain their candidate and last actual attempt time.
            selected, deferred = self._select_face_work(eligible)
            for work in deferred:
                current = by_camera[work.input.camera_id]
                current.outcome.face_deferred += 1
                current.reasons[work.track_id] = ("FACE_DEFERRED",)
            if selected:
                # A concurrent gallery publication invalidates the admitted token,
                # even if matching against the new gallery would produce a winner.
                if self.store.revision != admission_revision:
                    for work in selected:
                        evidence = self._apply_face(
                            replace(work, reasons=("GALLERY_CHANGED",)), None,
                        )
                        if evidence is not None:
                            by_camera[work.input.camera_id].evidence[work.track_id] = evidence
                else:
                    try:
                        embeddings = np.asarray(self.faces.embed_aligned([work.aligned for work in selected]))
                        if embeddings.shape != (len(selected), 512):
                            raise ModelBundleError("MODEL_CONTRACT_INVALID", "face_recognition",
                                                   "Embedding batch shape")
                    except Exception:
                        self._face_recognition_failed = face_failed = True
                        logger.warning("FACE_MODEL_UNAVAILABLE: face recognition failed", exc_info=True)
                    else:
                        self._face_recognition_failed = False
                        for work, embedding in zip(selected, embeddings):
                            evidence = self._apply_face(work, embedding)
                            if evidence is not None:
                                by_camera[work.input.camera_id].evidence[work.track_id] = evidence
            face_recognition_ms = (time.perf_counter() - face_recognition_started) * 1000.
            self._invalidate_gallery()
            face_unavailable = self._face_detection_failed or self._face_recognition_failed
            if face_failed or face_unavailable:
                for current in frames:
                    self._clear_unavailable_faces(current, attempted=face_failed)
            for current in frames:
                item, outcome = current.input, current.outcome
                height, width = item.packet.frame.shape[:2]
                if current.observed:
                    # Shared model-stage wall time is apportioned across admitted
                    # cameras; it is not multiplied by the detector batch length.
                    outcome.stage_times_ms = {
                        "person": person_ms / len(admitted),
                        "tracking": tracking_ms / len(admitted),
                        "face_detection": face_detection_ms / len(admitted),
                        "face_recognition": face_recognition_ms / len(admitted),
                    }
                for track in current.tracks:
                    run = self._face_runs.get((item.camera_id, track.track_id))
                    if run is not None and run.attendance != item.attendance:
                        self._clear_run(track, run)
                        self._face_runs.pop((item.camera_id, track.track_id))
                    if item.attendance:
                        self._expire_confirmation(item.camera_id, track, item.packet.received_mono)
                    if self._identity_unavailable() is not None:
                        track.clear_identity()
                    result = self._to_result(
                        track, width, height, current.evidence.get(track.track_id),
                        current.reasons.get(
                            track.track_id,
                            run.reason_codes if run is not None and track.track_id not in current.evidence else (),
                        ),
                    )
                    if current.observed:
                        outcome.observed_tracks.append(result)
                    if track.hits >= self.min_hits:
                        outcome.detections.append(result)
                outcome.face_unavailable = (
                    face_unavailable or self._identity_unavailable() == "FACE_MODEL_UNAVAILABLE"
                    or any("FACE_MODEL_UNAVAILABLE" in result.identity_reason_codes
                           for result in outcome.observed_tracks or outcome.detections)
                )
            # Reconcile all cameras against one final gallery publication, not
            # separate revisions observed while returning individual outcomes.
            self._invalidate_gallery()
            return [self._finalize_outcome(current.input.camera_id, current.outcome)
                    for current in frames]

    def _clear_unavailable_faces(self, current: _FrameWork, *, attempted: bool) -> None:
        """Invalidate shared-session evidence without resetting real observations.

        A failure charges the existing retry cadence to the affected batch,
        including cached ordinary confirmations. Frames before the next due
        attempt retain unavailability without pushing that deadline forward.
        """
        current.evidence.clear()
        reasons = ("FACE_MODEL_UNAVAILABLE",)
        item, packet = current.input, current.input.packet
        for track in current.tracks:
            run = self._face_runs.setdefault(
                (item.camera_id, track.track_id), _FaceRun(attendance=item.attendance),
            )
            self._clear_run(track, run)
            run.reason_codes = reasons
            current.reasons[track.track_id] = reasons
            if not current.observed:
                continue
            if attempted:
                run.last_attempt = track.last_recog_attempt = packet.received_mono
            # This is explicit unavailable evidence, not a face observation or a
            # negative gallery match. Doorway must not reuse pre-failure votes.
            current.evidence[track.track_id] = FaceEvidence(
                frame_seq=packet.seq, received_at=packet.received_at,
                received_mono=packet.received_mono, candidate_employee_id=None,
                candidate_employee_name=None, score=0., runner_up_employee_id=None,
                runner_up_employee_name=None, runner_up_score=0., margin=0.,
                quality=FaceQuality(0., 0., None, False), store_revision=self._store_revision,
                confirmed_employee_id=None, confirmed_employee_name=None,
                supporting_frame_sequences=(), supporting_received_monos=(),
                supporting_received_ats=(), confirmed_at_mono=None,
                face_bbox=None, aligned_jpeg=None, reason_codes=reasons,
            )

    def reset_camera(self, camera_id: int) -> str:
        """Every worker-owned reset, including same-stream resets, changes UUID."""
        with self._lock:
            self._trackers.pop(camera_id, None)
            self._last_frame_seq.pop(camera_id, None)
            self._last_packets.pop(camera_id, None)
            self._face_camera_served.pop(camera_id, None)
            self._face_runs = {key: value for key, value in self._face_runs.items()
                               if key[0] != camera_id}
            generation = str(uuid4())
            self._generations[camera_id] = generation
            return generation

    def _select_face_work(self, eligible: list[_FaceWork]) -> tuple[list[_FaceWork], list[_FaceWork]]:
        groups: dict[int, list[_FaceWork]] = {}
        for work in eligible:
            groups.setdefault(work.input.camera_id, []).append(work)
        for camera_id, group in groups.items():
            group.sort(key=lambda work: (
                self._face_runs[(camera_id, work.track_id)].last_served, work.track_id,
            ))
        selected: list[_FaceWork] = []
        for attendance in (True, False):
            cameras = sorted(
                (camera_id for camera_id, group in groups.items()
                 if group[0].input.attendance == attendance),
                key=lambda camera_id: (self._face_camera_served.get(camera_id, 0), camera_id),
            )
            rank = 0
            while cameras and len(selected) < _FACE_BATCH_LIMIT:
                remaining = []
                for camera_id in cameras:
                    if len(selected) == _FACE_BATCH_LIMIT:
                        break
                    selected.append(groups[camera_id][rank])
                    if len(groups[camera_id]) > rank + 1:
                        remaining.append(camera_id)
                cameras = remaining
                rank += 1
        selected_keys = {(work.input.camera_id, work.track_id) for work in selected}
        deferred = [work for work in eligible
                    if (work.input.camera_id, work.track_id) not in selected_keys]
        return selected, deferred

    def _identity_unavailable(self) -> str | None:
        if self.faces is None:
            return "FACE_MODEL_UNAVAILABLE"
        if self.store is None:
            return "GALLERY_UNAVAILABLE"
        if (self._expected_gallery_revision is not None
                and self.store.revision != self._expected_gallery_revision):
            return "GALLERY_UNAVAILABLE"
        if (self.faces.model_id != self.store.model_id
                or self.faces.alignment_id != ALIGNMENT_ID):
            return "FACE_MODEL_MISMATCH"
        return None

    def _invalidate_gallery(self) -> None:
        revision = self.store.revision if self.store is not None else None
        if (revision != self._store_revision
                or (self._expected_gallery_revision is not None
                    and revision != self._expected_gallery_revision)):
            for camera_id, tracker in self._trackers.items():
                for track in tracker.tracks:
                    run = self._face_runs.get((camera_id, track.track_id))
                    if run is None:
                        track.clear_identity()
                    else:
                        self._clear_run(track, run)
                        run.store_revision = revision
                        run.reason_codes = ("GALLERY_CHANGED",)
            self._store_revision = revision

    def _expire_confirmation(self, camera_id: int, track: Track, now: float) -> None:
        run = self._face_runs.get((camera_id, track.track_id))
        if run is None:
            return
        if (track.employee_id is not None and
                (not run.votes or now - run.votes[-1][1] > _CONFIRMATION_MAX_AGE)):
            self._clear_run(track, run)
        else:
            run.votes[:] = [vote for vote in run.votes
                            if now - vote[1] <= settings.attendance_face_window]
            track.candidate_count = len(run.votes)

    @staticmethod
    def _clear_run(track: Track, run: _FaceRun) -> None:
        track.clear_identity()
        run.candidate_id = None
        run.votes.clear()
        run.confirmed_at = None
        run.reason_codes = ()

    def _detect_face(self, item, generation, track, heads, revision) -> _FaceWork:
        frame = item.packet.frame
        head = self._head_region(track.bbox, frame.shape[1], frame.shape[0])
        hx1, hy1, hx2, hy2 = head
        faces = (self.faces.detect_faces(frame[hy1:hy2, hx1:hx2])
                 if hx2 - hx1 >= 40 and hy2 - hy1 >= 40 else [])
        quality = FaceQuality(0., 0., None, False)
        bbox, aligned, jpeg = None, None, None
        reasons: list[str] = []
        if len(faces) != 1:
            reasons.append("NO_FACE" if not faces else "AMBIGUOUS_FACE")
        else:
            face = faces[0]
            raw_box = np.asarray(face.native_bbox)
            if (raw_box.shape != (4,) or not np.isfinite(raw_box).all()
                    or raw_box[2] <= raw_box[0] or raw_box[3] <= raw_box[1]):
                reasons.append("ASSOCIATION_UNCERTAIN")
            else:
                quality = measure_face_quality(replace(face, bbox=raw_box))
                bbox = (float(raw_box[0] + hx1), float(raw_box[1] + hy1),
                        float(raw_box[2] + hx1), float(raw_box[3] + hy1))
                center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
                if (raw_box[0] < 0 or raw_box[1] < 0
                        or raw_box[2] > hx2 - hx1 or raw_box[3] > hy2 - hy1
                        or not self._contains(head, center)
                        or sum(self._contains(region, center) for region in heads) != 1):
                    reasons.append("AMBIGUOUS_FACE")
            points = np.asarray(face.landmarks)
            if (points.shape != (5, 2) or not np.isfinite(points).all()
                    or np.linalg.matrix_rank(points - points.mean(axis=0)) < 2
                    or np.linalg.norm(points[1] - points[0]) < 1.
                    or (points < 0).any()
                    or (points[:, 0] > hx2 - hx1).any()
                    or (points[:, 1] > hy2 - hy1).any()):
                reasons.append("BAD_ALIGNMENT")
            if face.model_id != self.faces.model_id:
                reasons.append("FACE_MODEL_MISMATCH")
            reasons.extend(quality_reasons(
                quality,
                min_size=(settings.attendance_face_min_size if item.attendance else
                          self.live_face_min_size),
                det_score=settings.attendance_face_det_score if item.attendance else .8,
                blur_min=settings.attendance_face_blur_min if item.attendance else 20.,
            ))
            if not reasons:
                aligned = face.aligned
                jpeg = encode_evidence(before=None, after=None, face=aligned).face_jpeg
                if jpeg is None and item.attendance:
                    reasons.append("EVIDENCE_UNAVAILABLE")
        if track.association_uncertain:
            reasons.append("ASSOCIATION_UNCERTAIN")
        return _FaceWork(item, generation, track.track_id, revision, quality, bbox,
                         aligned, jpeg, tuple(dict.fromkeys(reasons)))

    def _apply_face(self, work: _FaceWork, embedding: np.ndarray | None) -> FaceEvidence | None:
        item, packet = work.input, work.input.packet
        camera_id, now = item.camera_id, packet.received_mono
        if (self._generations.get(camera_id) != work.generation
                or self._last_packets.get(camera_id) != (
                    packet.source_revision, packet.stream_id, packet.seq)):
            return None
        track = next((track for track in self._trackers[camera_id].tracks
                      if track.track_id == work.track_id and track.state == "tracked"), None)
        run = self._face_runs.get((camera_id, work.track_id))
        if track is None or run is None or run.attendance != item.attendance:
            return None
        reasons = list(work.reasons)
        self._invalidate_gallery()
        if work.store_revision != self._store_revision:
            reasons.append("GALLERY_CHANGED")
        unavailable = self._identity_unavailable()
        if unavailable is not None:
            reasons.append(unavailable)
        match = FaceMatch(None, 0., None, 0., 0., work.store_revision)
        candidate_name = runner_name = None
        if not reasons:
            if not is_normalized_embedding(embedding):
                reasons.append("BAD_EMBEDDING")
            else:
                match = self.store.match(embedding)
                candidate_name = self.store.employee_name(match.employee_id)
                runner_name = self.store.employee_name(match.runner_up_employee_id)
                threshold = settings.attendance_face_threshold if item.attendance else self.min_confidence
                margin = settings.attendance_face_margin if item.attendance else self.live_face_margin
                if match.employee_id is None or match.score < threshold:
                    reasons.append("LOW_FACE_SCORE")
                if match.margin < margin:
                    reasons.append("AMBIGUOUS_FACE")
                if (match.store_revision != work.store_revision
                        or match.store_revision != self.store.revision
                        or (match.employee_id is not None
                            and not self.store.is_active(match.employee_id))):
                    reasons.append("GALLERY_CHANGED")
                    candidate_name = runner_name = None
                if match.store_revision != work.store_revision:
                    # Never label new-gallery scores with the admitted old
                    # revision, even though the result is already disqualified.
                    match = FaceMatch(None, 0., None, 0., 0., work.store_revision)
        self._invalidate_gallery()
        if work.store_revision != self._store_revision:
            reasons.append("GALLERY_CHANGED")
        if track.association_uncertain:
            reasons.append("ASSOCIATION_UNCERTAIN")
        if run.store_revision != work.store_revision:
            self._clear_run(track, run)
            run.store_revision = self._store_revision
        previous_employee = track.employee_id
        if reasons:
            self._clear_run(track, run)
            track.best_unknown_sim = match.score
        else:
            if run.candidate_id != match.employee_id:
                self._clear_run(track, run)
                run.candidate_id = match.employee_id
            window = settings.attendance_face_window if item.attendance else self.live_face_window
            run.votes[:] = [vote for vote in run.votes if now - vote[1] <= window]
            if all(vote[0] != packet.seq for vote in run.votes):
                run.votes.append((packet.seq, now, packet.received_at))
            needed = settings.attendance_face_confirmations if item.attendance else self.min_confirmations
            run.votes[:] = run.votes[-needed:]
            track.candidate_id, track.candidate_count = run.candidate_id, len(run.votes)
            if len(run.votes) >= needed:
                track.employee_id = match.employee_id
                track.employee_conf = match.score
                track.best_unknown_sim = None
                if run.confirmed_at is None:
                    run.confirmed_at = now
                if previous_employee != track.employee_id:
                    logger.info("Camera %d: employee=%d similarity=%.3f revision=%d",
                                camera_id, track.employee_id, match.score, match.store_revision)
            elif track.employee_id is not None:
                # A positive ordinary recheck maintains the same confirmed owner;
                # contradictory/negative evidence takes the clear-run path above.
                track.employee_conf = match.score
            else:
                track.best_unknown_sim = match.score
        self._face_turn += 1
        run.last_served, run.last_attempt = self._face_turn, now
        self._face_camera_served[camera_id] = self._face_turn
        track.last_recog_attempt = now
        run.reason_codes = tuple(dict.fromkeys(reasons))
        return FaceEvidence(
            packet.seq, packet.received_at, now, match.employee_id,
            candidate_name, match.score, match.runner_up_employee_id, runner_name,
            match.runner_up_score, match.margin, work.quality, work.store_revision,
            track.employee_id, candidate_name if track.employee_id is not None else None,
            tuple(vote[0] for vote in run.votes), tuple(vote[1] for vote in run.votes),
            tuple(vote[2] for vote in run.votes), run.confirmed_at, work.bbox, work.jpeg,
            run.reason_codes,
        )

    def _finalize_outcome(self, camera_id: int, outcome: FrameOutcome) -> FrameOutcome:
        """Reconcile earlier/cached results after every model call has finished."""
        tracks = {track.track_id: track for track in self._trackers[camera_id].tracks}
        for result in outcome.observed_tracks or outcome.detections:
            evidence = result.fresh_face_evidence
            stale = evidence is not None and evidence.store_revision != self._store_revision
            if stale:
                reasons = tuple(dict.fromkeys((*evidence.reason_codes, "GALLERY_CHANGED")))
                result.fresh_face_evidence = replace(evidence, reason_codes=reasons)
                result.identity_reason_codes = tuple(dict.fromkeys(
                    (*result.identity_reason_codes, "GALLERY_CHANGED"),
                ))
            track = tracks.get(result.track_id)
            if result.employee_id is not None and (
                    stale or track is None or track.employee_id != result.employee_id):
                result.employee_id = None
                result.employee_name = None
                result.confidence = None
                result.state = "unknown"
        return outcome

    @staticmethod
    def _contains(region, point) -> bool:
        return region[0] <= point[0] < region[2] and region[1] <= point[1] < region[3]

    @staticmethod
    def _head_region(bbox, width, height):
        x1, y1, x2, y2 = bbox
        box_width, box_height = x2 - x1, y2 - y1
        head_ratio = .55 if box_width > 0 and box_height / box_width >= 1.6 else 1.
        return (max(0, min(width, int(x1 - box_width * .1))),
                max(0, min(height, int(y1))),
                max(0, min(width, int(x2 + box_width * .1))),
                max(0, min(height, int(y1 + box_height * head_ratio))))

    @staticmethod
    def _normalized_box(bbox, width, height):
        x1, y1, x2, y2 = bbox
        return (max(0., float(x1 / width)), max(0., float(y1 / height)),
                min(1., float(x2 / width)), min(1., float(y2 / height)))

    def _to_result(self, track, width, height, evidence=None, reasons=()) -> DetectionResult:
        unavailable = self._identity_unavailable()
        if track.employee_id is not None:
            state, confidence = "recognized", track.employee_conf
        elif track.last_recog_attempt > 0:
            state, confidence = "unknown", track.best_unknown_sim
        else:
            state, confidence = "detecting", None
        if unavailable is not None:
            state, confidence = "unknown", None
        identity_reasons = ((unavailable,) if unavailable is not None else
                            reasons or (evidence.reason_codes if evidence is not None else ()))
        return DetectionResult(
            track_id=track.track_id, employee_id=track.employee_id,
            employee_name=self.store.employee_name(track.employee_id) if self.store is not None else None,
            confidence=round(confidence, 3) if confidence is not None else None,
            bbox=self._normalized_box(track.bbox, width, height), state=state,
            association_uncertain=track.association_uncertain,
            person_score=track.score, fresh_face_evidence=evidence,
            identity_reason_codes=identity_reasons,
        )
