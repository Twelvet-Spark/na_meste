"""Worker-owned YOLO/local tracking and fresh, unambiguous employee evidence.

Receipt timestamps describe server receipt, not camera exposure. Attendance votes
are short-lived evidence; neither a cached identity nor a global body ID is proof
of a new crossing. Nonattendance cameras retain their configurable match policy.
"""
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from uuid import uuid4

import numpy as np

from app.ai.detector import PersonDetector
from app.ai.face_detector import FaceEngine
from app.ai.face_quality import FaceQuality, measure_face_quality, quality_reasons
from app.ai.face_recognition import EmbeddingStore, FaceMatch
from app.ai.tracker import ByteTracker, Track, TrackLifecycle
from app.config import settings
from app.services.attendance_evidence import encode_evidence

logger = logging.getLogger("app.ai.service")
_CONFIRMATION_MAX_AGE = 0.75


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
    just_confirmed: bool = False  # Legacy presence bridge, removed at step 5.
    global_id: int | None = None
    association_uncertain: bool = False
    person_score: float = 0.
    fresh_face_evidence: FaceEvidence | None = None


@dataclass
class FrameOutcome:
    detections: list[DetectionResult] = field(default_factory=list)
    new_recognitions: list[tuple[int, int, float]] = field(default_factory=list)
    tracking_generation: str = ""
    track_lifecycle: list[TrackLifecycle] = field(default_factory=list)
    # All YOLO person boxes, normalized, even low-score/min-hit-ineligible ones.
    person_regions: list[tuple[float, float, float, float]] = field(default_factory=list)


@dataclass
class _FaceRun:
    candidate_id: int | None = None
    votes: list[tuple[int, float, datetime]] = field(default_factory=list)
    last_attempt: float = float("-inf")
    confirmed_at: float | None = None


class RecognitionService:
    def __init__(
        self,
        detector: PersonDetector,
        face_engine: FaceEngine,
        store: EmbeddingStore,
        *,
        min_confidence: float = .45,
        min_confirmations: int = 2,
        retry_interval: float = 2.,
        track_lost_timeout: float = 3.,
        min_hits: int = 2,
        tracker_high_threshold: float = .5,
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
        self._trackers: dict[int, ByteTracker] = {}
        self._generations: dict[int, str] = {}
        self._face_runs: dict[tuple[int, int], _FaceRun] = {}
        self._last_frame_seq: dict[int, int] = {}
        self._store_revision = store.revision
        self._lock = threading.Lock()

    def process_frame(
        self, frame: np.ndarray, camera_id: int, now: float | None = None,
        *, frame_seq: int, received_at: datetime, attendance: bool = False,
    ) -> FrameOutcome:
        now = time.monotonic() if now is None else now
        height, width = frame.shape[:2]
        with self._lock:
            self._invalidate_gallery()
            if camera_id not in self._generations:
                self._generations[camera_id] = str(uuid4())
            tracker = self._trackers.get(camera_id)
            if tracker is None:
                tracker = ByteTracker(lost_timeout=self.track_lost_timeout,
                                      high_threshold=self.tracker_high_threshold)
                self._trackers[camera_id] = tracker
            outcome = FrameOutcome(tracking_generation=self._generations[camera_id])
            if frame_seq <= self._last_frame_seq.get(camera_id, -1):
                # Replayed pixels cannot advance tracking, votes or face freshness.
                for track in tracker.tracks:
                    if attendance:
                        self._expire_confirmation(camera_id, track, now)
                    if track.state == "tracked" and track.hits >= self.min_hits:
                        outcome.detections.append(self._to_result(track, width, height))
                return outcome
            self._last_frame_seq[camera_id] = frame_seq
            persons = self.detector.detect(frame)
            heads = [self._head_region(bbox, width, height) for bbox, _score in persons]
            outcome.person_regions = [self._normalized_box(bbox, width, height)
                                      for bbox, _score in persons]
            tracks = tracker.update(persons, now)
            outcome.track_lifecycle = tracker.lifecycle
            for transition in outcome.track_lifecycle:
                if transition.state in ("LOST", "AMBIGUOUS", "EXPIRED"):
                    self._face_runs.pop((camera_id, transition.track_id), None)
                    if attendance:
                        for track in tracker.tracks:
                            if track.track_id == transition.track_id:
                                track.clear_identity()
                                break
            for track in tracks:
                if attendance:
                    self._expire_confirmation(camera_id, track, now)
                if track.hits < self.min_hits:
                    continue
                evidence = None
                run = self._face_runs.setdefault((camera_id, track.track_id), _FaceRun())
                interval = settings.attendance_face_interval if attendance else self.retry_interval
                if (attendance or track.employee_id is None) and now - run.last_attempt >= interval:
                    run.last_attempt = now
                    track.last_recog_attempt = now
                    evidence = self._recognize(track, frame, heads, camera_id, now,
                                               frame_seq, received_at, attendance, run, outcome)
                outcome.detections.append(self._to_result(track, width, height, evidence))
            return outcome

    def reset_camera(self, camera_id: int) -> str:
        """Every worker-owned reset, including same-stream resets, changes UUID."""
        with self._lock:
            self._trackers.pop(camera_id, None)
            self._last_frame_seq.pop(camera_id, None)
            self._face_runs = {key: value for key, value in self._face_runs.items()
                               if key[0] != camera_id}
            generation = str(uuid4())
            self._generations[camera_id] = generation
            return generation

    def _invalidate_gallery(self) -> None:
        revision = self.store.revision
        if revision != self._store_revision:
            for tracker in self._trackers.values():
                for track in tracker.tracks:
                    track.clear_identity()
            self._face_runs.clear()
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

    @staticmethod
    def _clear_run(track: Track, run: _FaceRun) -> None:
        track.clear_identity()
        run.candidate_id = None
        run.votes.clear()
        run.confirmed_at = None

    def _recognize(self, track, frame, heads, camera_id, now, frame_seq, received_at,
                   attendance, run, outcome) -> FaceEvidence:
        head = self._head_region(track.bbox, frame.shape[1], frame.shape[0])
        hx1, hy1, hx2, hy2 = head
        faces = (self.faces.detect(frame[hy1:hy2, hx1:hx2])
                 if hx2 - hx1 >= 40 and hy2 - hy1 >= 40 else [])
        quality = FaceQuality(0., 0., None, False)
        match = FaceMatch(None, 0., None, 0., 0., self.store.revision)
        bbox, jpeg = None, None
        reasons = []
        if len(faces) != 1:
            reasons.append("NO_FACE" if not faces else "AMBIGUOUS_FACE")
        else:
            face = faces[0]
            quality = measure_face_quality(face)
            match = self.store.match(face.embedding)
            raw_box = np.asarray(face.bbox)
            if raw_box.shape != (4,) or not np.isfinite(raw_box).all():
                reasons.append("ASSOCIATION_UNCERTAIN")
            else:
                bbox = (float(raw_box[0] + hx1), float(raw_box[1] + hy1),
                        float(raw_box[2] + hx1), float(raw_box[3] + hy1))
                center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
                # Count raw current person heads, including undisplayed detections.
                if not self._contains(head, center) or sum(
                        self._contains(region, center) for region in heads) != 1:
                    reasons.append("AMBIGUOUS_FACE")
            if attendance:
                reasons.extend(quality_reasons(
                    quality, min_size=settings.attendance_face_min_size,
                    det_score=settings.attendance_face_det_score,
                    blur_min=settings.attendance_face_blur_min,
                ))
            if quality.aligned:
                jpeg = encode_evidence(before=None, after=None, face=face.aligned).face_jpeg
                if jpeg is None and attendance:
                    reasons.append("EVIDENCE_UNAVAILABLE")
            threshold = settings.attendance_face_threshold if attendance else self.min_confidence
            if match.employee_id is None or match.score < threshold:
                reasons.append("LOW_FACE_SCORE")
            if attendance and match.margin < settings.attendance_face_margin:
                reasons.append("AMBIGUOUS_FACE")
            if match.store_revision != self.store.revision:
                reasons.append("GALLERY_CHANGED")
            elif match.employee_id is not None and not self.store.is_active(match.employee_id):
                reasons.append("GALLERY_CHANGED")
        if track.association_uncertain:
            reasons.append("ASSOCIATION_UNCERTAIN")
        candidate_name = self.store.employee_name(match.employee_id)
        runner_name = self.store.employee_name(match.runner_up_employee_id)
        if match.store_revision != self.store.revision:
            # Never attach names from a newly published gallery to old scores.
            candidate_name = runner_name = None
            reasons.append("GALLERY_CHANGED")
        previous_employee = track.employee_id
        if reasons:
            self._clear_run(track, run)
            track.last_recog_attempt = now
            track.best_unknown_sim = match.score
        else:
            if run.candidate_id != match.employee_id:
                self._clear_run(track, run)
                run.candidate_id = match.employee_id
            if attendance:
                run.votes[:] = [vote for vote in run.votes
                                if now - vote[1] <= settings.attendance_face_window]
            run.votes.append((frame_seq, now, received_at))
            needed = settings.attendance_face_confirmations if attendance else self.min_confirmations
            # Keep only the evidence required for the current confirmation.
            run.votes[:] = run.votes[-needed:]
            track.candidate_id = run.candidate_id
            track.candidate_count = len(run.votes)
            track.last_recog_attempt = now
            if len(run.votes) >= needed:
                track.employee_id = match.employee_id
                track.employee_conf = match.score
                track.best_unknown_sim = None
                if run.confirmed_at is None:
                    run.confirmed_at = now
                if previous_employee != track.employee_id:
                    outcome.new_recognitions.append((track.track_id, track.employee_id, match.score))
                    logger.info("Camera %d: employee=%d similarity=%.3f revision=%d",
                                camera_id, track.employee_id, match.score, match.store_revision)
        return FaceEvidence(
            frame_seq, received_at, now, match.employee_id,
            candidate_name, match.score,
            match.runner_up_employee_id, runner_name,
            match.runner_up_score, match.margin, quality, match.store_revision,
            track.employee_id, candidate_name if track.employee_id is not None else None,
            tuple(vote[0] for vote in run.votes), tuple(vote[1] for vote in run.votes),
            tuple(vote[2] for vote in run.votes), run.confirmed_at, bbox, jpeg,
            tuple(dict.fromkeys(reasons)),
        )

    @staticmethod
    def _contains(region, point) -> bool:
        return region[0] <= point[0] < region[2] and region[1] <= point[1] < region[3]

    @staticmethod
    def _head_region(bbox, width, height):
        x1, y1, x2, y2 = bbox
        box_width, box_height = x2 - x1, y2 - y1
        head_ratio = .55 if box_height > 1.2 * box_width else 1.
        return (max(0, min(width, int(x1 - box_width * .05))),
                max(0, min(height, int(y1))),
                max(0, min(width, int(x2 + box_width * .05))),
                max(0, min(height, int(y1 + box_height * head_ratio))))

    @staticmethod
    def _normalized_box(bbox, width, height):
        x1, y1, x2, y2 = bbox
        return (max(0., float(x1 / width)), max(0., float(y1 / height)),
                min(1., float(x2 / width)), min(1., float(y2 / height)))

    def _to_result(self, track, width, height, evidence=None) -> DetectionResult:
        if track.employee_id is not None:
            state, confidence = "recognized", track.employee_conf
        elif track.last_recog_attempt > 0:
            state, confidence = "unknown", track.best_unknown_sim
        else:
            state, confidence = "detecting", None
        return DetectionResult(
            track_id=track.track_id, employee_id=track.employee_id,
            employee_name=self.store.employee_name(track.employee_id),
            confidence=round(confidence, 3) if confidence is not None else None,
            bbox=self._normalized_box(track.bbox, width, height), state=state,
            association_uncertain=track.association_uncertain,
            person_score=track.score, fresh_face_evidence=evidence,
        )
