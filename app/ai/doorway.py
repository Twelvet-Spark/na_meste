"""Receipt-timed, finite doorway crossings; owned exclusively by the AI worker.

A face is evidence for one local trajectory, never a substitute for a crossing.
The detector neither writes attendance nor consults commissioning: auto_eligible
is captured technical eligibility, to be revalidated transactionally at storage.
"""
import math
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from app.ai.face_quality import quality_reasons
from app.ai.recognition_service import FaceEvidence, FrameOutcome
from app.config import settings
from app.rtsp.reader import FramePacket
from app.services.attendance_evidence import encode_evidence

TrackKey = tuple[int, str, str, int]
Box = tuple[float, float, float, float]
_SIDE_BAND = .02
_PENDING_SECONDS = 2.
_CONFIRMATION_AGE = .75


@dataclass(frozen=True, slots=True)
class DoorwayCalibration:
    camera_id: int
    role: str
    revision: int
    source_revision: int
    line_start: tuple[float, float]
    line_end: tuple[float, float]
    inside_point: tuple[float, float]
    reference_width: int
    reference_height: int
    _normal: tuple[float, float] = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        if self.role not in ("ENTRY", "EXIT"):
            raise ValueError("Role must be ENTRY or EXIT")
        if (self.reference_width <= 0 or self.reference_height <= 0
                or self.revision < 0 or self.source_revision < 1):
            raise ValueError("Invalid reference dimensions or revisions")
        for name in ("line_start", "line_end", "inside_point"):
            point = tuple(getattr(self, name))
            if len(point) != 2 or any(not math.isfinite(v) or not 0 <= v <= 1 for v in point):
                raise ValueError("Coordinates must be finite normalized points")
            object.__setattr__(self, name, point)
        ax, ay = self.line_start
        bx, by = self.line_end
        length = math.hypot(bx - ax, by - ay)
        if length < .05 - 1e-12:
            raise ValueError("Gate endpoints must be at least .05 apart")
        normal = (-(by - ay) / length, (bx - ax) / length)
        distance = (self.inside_point[0] - ax) * normal[0] + (self.inside_point[1] - ay) * normal[1]
        if abs(distance) < .05 - 1e-12:
            raise ValueError("Interior must be at least .05 from the gate")
        if distance < 0:
            normal = (-normal[0], -normal[1])
        object.__setattr__(self, "_normal", normal)

    def signed_distance(self, point: tuple[float, float]) -> float:
        return ((point[0] - self.line_start[0]) * self._normal[0]
                + (point[1] - self.line_start[1]) * self._normal[1])


@dataclass(frozen=True, slots=True)
class CrossingEvidence:
    track_key: TrackKey
    crossing_index: int
    source_revision: int
    calibration_revision: int
    gallery_revision: int | None
    direction: str | None
    occurred_at: datetime
    occurred_mono: float
    confirmed_employee_id: int | None
    confirmed_employee_name: str | None
    face_evidence: tuple[FaceEvidence, ...]
    frame_sequences: tuple[int, ...]
    before_crop: Box | None
    after_crop: Box | None
    calibration: DoorwayCalibration
    before_jpeg: bytes | None
    after_jpeg: bytes | None
    face_jpeg: bytes | None
    auto_eligible: bool
    disqualifying_reasons: tuple[str, ...]
    # Frozen name/value pairs make queued evidence independent of later settings.
    thresholds: tuple[tuple[str, int | float], ...]

    @property
    def would_auto_accept(self) -> bool:
        return self.auto_eligible


@dataclass(slots=True)
class _Point:
    x: float
    y: float
    distance: float
    seq: int
    utc: datetime
    mono: float
    bbox: Box
    frame: np.ndarray


@dataclass(slots=True)
class _Bracket:
    utc: datetime
    mono: float
    sequences: tuple[int, int]
    before_crop: Box | None
    after_crop: Box | None
    before_jpeg: bytes | None
    after_jpeg: bytes | None


@dataclass(slots=True)
class _Pending:
    index: int
    direction: str | None
    bracket: _Bracket
    calibration: DoorwayCalibration
    gallery_revision: int | None
    faces: list[FaceEvidence]
    reasons: set[str]
    frame_sequences: tuple[int, ...]
    thresholds: tuple[tuple[str, int | float], ...]


@dataclass(slots=True)
class _Trajectory:
    key: TrackKey
    calibration: DoorwayCalibration
    previous: _Point | None = None
    side_run: int = 0
    side_hits: int = 0
    armed_side: int = 0
    bracket: _Bracket | None = None
    ordinal: int = 0
    last_seen: float = 0.
    gallery_revision: int | None = None
    faces: deque = field(default_factory=lambda: deque(maxlen=64))
    pending: list[_Pending] = field(default_factory=list)
    reasons: set[str] = field(default_factory=set)


@dataclass(slots=True)
class _Camera:
    stream: str
    generation: str
    source_revision: int
    calibration_revision: int
    last_seq: int = -1
    tracks: dict[int, _Trajectory] = field(default_factory=dict)


def _thresholds() -> tuple[tuple[str, int | float], ...]:
    names = ("frame_max_age", "max_frame_gap", "face_threshold", "face_margin",
             "face_confirmations", "face_window", "face_interval", "face_min_size",
             "face_det_score", "face_blur_min")
    return tuple((name, getattr(settings, f"attendance_{name}")) for name in names)


def _context(point: _Point) -> tuple[bytes | None, Box | None]:
    height, width = point.frame.shape[:2]
    x1, y1, x2, y2 = point.bbox
    padx, pady = (x2 - x1) * .35, (y2 - y1) * .15
    left, top = max(0, math.floor((x1 - padx) * width)), max(0, math.floor((y1 - pady) * height))
    right, bottom = min(width, math.ceil((x2 + padx) * width)), min(height, math.ceil((y2 + pady) * height))
    if right <= left or bottom <= top:
        return None, None
    crop = point.frame[top:bottom, left:right]
    jpeg = encode_evidence(before=crop, after=None).before_jpeg
    return jpeg, (left / width, top / height, right / width, bottom / height)


def _along(x: float, y: float, config: DoorwayCalibration) -> float:
    ax, ay = config.line_start
    bx, by = config.line_end
    return ((x - ax) * (bx - ax) + (y - ay) * (by - ay)) / ((bx - ax) ** 2 + (by - ay) ** 2)


def _intersection_fraction(previous: _Point, x: float, y: float,
                           config: DoorwayCalibration) -> float | None:
    distance = config.signed_distance((x, y))
    if previous.distance == distance:
        return None
    fraction = previous.distance / (previous.distance - distance)
    if not 0 <= fraction <= 1:
        return None
    ix = previous.x + fraction * (x - previous.x)
    iy = previous.y + fraction * (y - previous.y)
    return fraction if 0 <= _along(ix, iy, config) <= 1 else None


def _bracket(previous: _Point, current: _Point, config: DoorwayCalibration) -> _Bracket | None:
    fraction = _intersection_fraction(previous, current.x, current.y, config)
    if fraction is None:
        return None
    before, before_crop = _context(previous)
    after, after_crop = _context(current)
    return _Bracket(previous.utc + (current.utc - previous.utc) * fraction,
                    previous.mono + (current.mono - previous.mono) * fraction,
                    (previous.seq, current.seq), before_crop, after_crop, before, after)


class DoorwayDetector:
    def __init__(self):
        self._cameras: dict[int, _Camera] = {}

    def process(self, packet: FramePacket, camera_id: int, outcome: FrameOutcome,
                calibration: DoorwayCalibration) -> list[CrossingEvidence]:
        if calibration.camera_id != camera_id:
            raise ValueError("Calibration belongs to a different camera")
        events = []
        camera = self._cameras.get(camera_id)
        if packet.source_revision != calibration.source_revision:
            return events  # Old reader packets can never arm a replacement source.
        if time.monotonic() - packet.received_mono > settings.attendance_frame_max_age:
            if camera is not None:
                for track in camera.tracks.values():
                    events.extend(self._interrupt(track, "FRAME_GAP"))
            return events
        if camera is not None:
            reset_reason = None
            if (camera.stream != packet.stream_id or camera.generation != outcome.tracking_generation
                    or camera.source_revision != packet.source_revision):
                reset_reason = "STREAM_RESET"
            elif camera.calibration_revision != calibration.revision:
                reset_reason = "CALIBRATION_CHANGED"
            if reset_reason:
                events.extend(self.reset_camera(camera_id, reset_reason))
                camera = None
        if camera is None:
            camera = _Camera(packet.stream_id, outcome.tracking_generation,
                             packet.source_revision, calibration.revision)
            self._cameras[camera_id] = camera
        if packet.seq <= camera.last_seq:
            return events
        camera.last_seq = packet.seq
        ambiguous_ids = {item.track_id for item in outcome.track_lifecycle if item.state == "AMBIGUOUS"}
        # Lifecycle includes unmatched tracks; process it before any pending admission.
        for lifecycle in outcome.track_lifecycle:
            track = camera.tracks.get(lifecycle.track_id)
            if track is None:
                continue
            if lifecycle.state == "AMBIGUOUS":
                track.reasons.add("ASSOCIATION_UNCERTAIN")
                for pending in track.pending:
                    pending.reasons.add("ASSOCIATION_UNCERTAIN")
                detection = next((item for item in outcome.detections
                                  if item.track_id == lifecycle.track_id), None)
                possible = False
                if detection is not None:
                    x1, _y1, x2, y2 = detection.bbox
                    if track.previous is not None:
                        possible = _intersection_fraction(track.previous, (x1 + x2) / 2,
                                                          y2, calibration) is not None
                    if detection.fresh_face_evidence is not None:
                        track.faces.append(detection.fresh_face_evidence)
                events.extend(self._interrupt(track, "ASSOCIATION_UNCERTAIN", possible=possible))
            elif lifecycle.state == "LOST":
                events.extend(self._interrupt(track, None))
            elif lifecycle.state == "EXPIRED":
                events.extend(self._finish_track(track, lifecycle.reason or "TRACK_EXPIRED"))
                del camera.tracks[lifecycle.track_id]
        height, width = packet.frame.shape[:2]
        aspect_changed = abs((width / height) / (calibration.reference_width / calibration.reference_height) - 1) > .01
        for detection in outcome.detections:
            key = (camera_id, packet.stream_id, outcome.tracking_generation, detection.track_id)
            track = camera.tracks.get(detection.track_id)
            if track is None:
                track = _Trajectory(key, calibration)
                camera.tracks[detection.track_id] = track
            if track.previous is not None and packet.received_mono - track.last_seen > settings.attendance_max_frame_gap:
                events.extend(self._interrupt(track, "FRAME_GAP"))
            track.last_seen = packet.received_mono
            observation = detection.fresh_face_evidence
            if observation is not None:
                if track.gallery_revision is not None and track.gallery_revision != observation.store_revision:
                    track.reasons.add("GALLERY_CHANGED")
                    for pending in track.pending:
                        pending.reasons.add("GALLERY_CHANGED")
                track.gallery_revision = observation.store_revision
                track.reasons.update(reason for reason in observation.reason_codes
                                     if reason in ("GALLERY_CHANGED", "ASSOCIATION_UNCERTAIN"))
                track.faces.append(observation)
                for pending in track.pending:
                    pending.faces.append(observation)
                    if pending.gallery_revision is None:
                        pending.gallery_revision = observation.store_revision
                    elif pending.gallery_revision != observation.store_revision:
                        pending.reasons.add("GALLERY_CHANGED")
                    pending.reasons.update(reason for reason in observation.reason_codes
                                           if reason in ("GALLERY_CHANGED", "ASSOCIATION_UNCERTAIN"))
            if aspect_changed:
                track.reasons.add("ASPECT_MISMATCH")
                for pending in track.pending:
                    pending.reasons.add("ASPECT_MISMATCH")
            if detection.association_uncertain or detection.track_id in ambiguous_ids:
                track.reasons.add("ASSOCIATION_UNCERTAIN")
                for pending in track.pending:
                    pending.reasons.add("ASSOCIATION_UNCERTAIN")
                events.extend(self._interrupt(track, "ASSOCIATION_UNCERTAIN"))
                continue
            x1, y1, x2, y2 = detection.bbox
            point = _Point((x1 + x2) / 2, y2, 0., packet.seq, packet.received_at,
                           packet.received_mono, detection.bbox, packet.frame)
            point.distance = calibration.signed_distance((point.x, point.y))
            previous = track.previous
            if track.armed_side and previous is not None and track.bracket is None:
                # First toward-opposite sign change; neutral jitter retains this time.
                if (previous.distance * track.armed_side >= 0
                        and point.distance * track.armed_side <= 0):
                    track.bracket = _bracket(previous, point, calibration)
            side = 1 if point.distance >= _SIDE_BAND else -1 if point.distance <= -_SIDE_BAND else 0
            if side:
                track.side_hits = track.side_hits + 1 if side == track.side_run else 1
                track.side_run = side
            else:
                track.side_hits, track.side_run = 0, 0
            if side and track.side_hits >= 2:
                if track.armed_side and side != track.armed_side and track.bracket is not None:
                    direction = "ENTRY" if side > 0 else "EXIT"
                    if direction == calibration.role:
                        track.ordinal += 1
                        bracket = track.bracket
                        bracket.after_jpeg, bracket.after_crop = _context(point)
                        pending = _Pending(track.ordinal, direction, bracket, calibration,
                                           track.gallery_revision,
                                           [item for item in track.faces
                                            if item.received_mono >= bracket.mono - settings.attendance_face_window],
                                           set(track.reasons),
                                           tuple(dict.fromkeys((*bracket.sequences, point.seq))), _thresholds())
                        track.pending.append(pending)
                track.armed_side = side
                track.bracket = None
                # Faults from a previous movement cannot contaminate all future trips.
                track.reasons = {"ASPECT_MISMATCH"} if aspect_changed else set()
            track.previous = point
            events.extend(self._ready_pending(track, packet.received_mono))
        return events

    def reset_camera(self, camera_id: int, reason: str) -> list[CrossingEvidence]:
        camera = self._cameras.pop(camera_id, None)
        if camera is None:
            return []
        events = []
        for track in camera.tracks.values():
            events.extend(self._finish_track(track, reason))
        return events

    def expire(self, now_mono: float) -> list[CrossingEvidence]:
        events = []
        for camera in self._cameras.values():
            for track in camera.tracks.values():
                if track.previous is not None and now_mono - track.last_seen > settings.attendance_max_frame_gap:
                    events.extend(self._interrupt(track, "FRAME_GAP"))
                events.extend(self._ready_pending(track, now_mono))
        return events

    def _interrupt(self, track: _Trajectory, reason: str | None, *,
                   possible: bool = False) -> list[CrossingEvidence]:
        events = []
        if reason in ("FRAME_GAP", "ASSOCIATION_UNCERTAIN"):
            for pending in track.pending:
                pending.reasons.add(reason)
        point = track.previous
        near_gate = (point is not None and abs(point.distance) < _SIDE_BAND
                     and 0 <= _along(point.x, point.y, track.calibration) <= 1)
        if track.armed_side and point is not None and (near_gate or track.bracket is not None or possible):
            track.ordinal += 1
            if track.bracket is None:
                before, crop = _context(point)
                bracket = _Bracket(point.utc, point.mono, (point.seq, point.seq), crop, None, before, None)
            else:
                bracket = track.bracket
            reasons = set(track.reasons) | {"INCOMPLETE_CROSSING"}
            if reason:
                reasons.add(reason)
            pending = _Pending(track.ordinal, None, bracket, track.calibration,
                               track.gallery_revision, list(track.faces), reasons,
                               tuple(dict.fromkeys(bracket.sequences)), _thresholds())
            events.append(self._evidence(track.key, pending))
        track.previous, track.bracket = None, None
        track.side_run, track.side_hits, track.armed_side = 0, 0, 0
        track.faces.clear()
        track.reasons.clear()
        return events

    def _finish_track(self, track: _Trajectory, reason: str) -> list[CrossingEvidence]:
        # Known completed geometry keeps its captured eligibility. Partial geometry
        # must instead be emitted once as unknown direction, never interpolated.
        events = [self._evidence(track.key, pending) for pending in track.pending]
        track.pending.clear()
        events.extend(self._interrupt(track, reason))
        return events

    def _ready_pending(self, track: _Trajectory, now: float) -> list[CrossingEvidence]:
        events, waiting = [], []
        for pending in track.pending:
            evidence = self._evidence(track.key, pending)
            if evidence.auto_eligible or now >= pending.bracket.mono + _PENDING_SECONDS:
                events.append(evidence)
            else:
                waiting.append(pending)
        track.pending = waiting
        return events

    @staticmethod
    def _evidence(key: TrackKey, pending: _Pending) -> CrossingEvidence:
        thresholds = dict(pending.thresholds)
        window = thresholds["face_window"]
        crossing = pending.bracket.mono
        faces = tuple(pending.faces)
        reasons = set(pending.reasons)
        relevant = [item for item in faces if crossing - window <= item.received_mono <= crossing + window]
        before = [item for item in relevant if item.received_mono <= crossing]
        # A failed attempt before the line clears an earlier run, just as it does
        # in RecognitionService; no-face after the line need not erase proof that
        # really was valid at the crossing.
        candidates = ([before[-1]] if before else []) + [item for item in relevant if item.received_mono > crossing]
        valid = []
        observed_reasons = set()
        for item in candidates:
            gates = set(item.reason_codes)
            gates.update(quality_reasons(item.quality, min_size=thresholds["face_min_size"],
                                        det_score=thresholds["face_det_score"], blur_min=thresholds["face_blur_min"]))
            if item.score < thresholds["face_threshold"]:
                gates.add("LOW_FACE_SCORE")
            if item.margin < thresholds["face_margin"]:
                gates.add("AMBIGUOUS_FACE")
            observed_reasons.update(gates)
            sequences, supports = item.supporting_frame_sequences, item.supporting_received_monos
            if (gates or item.confirmed_employee_id is None or item.confirmed_at_mono is None
                    or len(set(sequences)) < thresholds["face_confirmations"]
                    or len(sequences) != len(supports)
                    or any(not crossing - window <= value <= crossing + window for value in supports)):
                continue
            if item.confirmed_at_mono <= crossing:
                supporting_before = [value for value in supports if value <= crossing]
                if not supporting_before or crossing - max(supporting_before) > _CONFIRMATION_AGE:
                    continue
            elif item.confirmed_at_mono > crossing + window:
                continue
            valid.append(item)
        # Competing confirmations can straddle the line; retain all in-window
        # confirmations, not only the last effective detection identity.
        confirmed_ids = {item.confirmed_employee_id for item in relevant
                         if item.confirmed_employee_id is not None and not item.reason_codes}
        if len(confirmed_ids) > 1:
            reasons.add("AMBIGUOUS_FACE")
        chosen = valid[-1] if valid else None
        if chosen is None:
            reasons.update(observed_reasons or {"NO_FACE"})
        if pending.direction is None:
            reasons.add("INCOMPLETE_CROSSING")
        if pending.bracket.before_jpeg is None or pending.bracket.after_jpeg is None:
            reasons.add("EVIDENCE_UNAVAILABLE")
        raw_identity = chosen or next((item for item in reversed(relevant)
                                       if item.confirmed_employee_id is not None), None)
        face_jpeg = (chosen.aligned_jpeg if chosen else
                     next((item.aligned_jpeg for item in reversed(relevant) if item.aligned_jpeg), None))
        return CrossingEvidence(
            track_key=key, crossing_index=pending.index,
            source_revision=pending.calibration.source_revision,
            calibration_revision=pending.calibration.revision,
            gallery_revision=pending.gallery_revision, direction=pending.direction,
            occurred_at=pending.bracket.utc, occurred_mono=crossing,
            confirmed_employee_id=raw_identity.confirmed_employee_id if raw_identity else None,
            confirmed_employee_name=raw_identity.confirmed_employee_name if raw_identity else None,
            face_evidence=faces, frame_sequences=pending.frame_sequences,
            before_crop=pending.bracket.before_crop, after_crop=pending.bracket.after_crop,
            calibration=pending.calibration, before_jpeg=pending.bracket.before_jpeg,
            after_jpeg=pending.bracket.after_jpeg, face_jpeg=face_jpeg,
            auto_eligible=chosen is not None and not reasons,
            disqualifying_reasons=tuple(sorted(reasons)), thresholds=pending.thresholds,
        )
