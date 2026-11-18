"""Motion-predicted two-stage tracking with conservative identity continuity."""
import logging
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.optimize import linear_sum_assignment

logger = logging.getLogger("app.ai.tracker")


def iou_xyxy(a, b) -> float:
    """IoU двух боксов в формате xyxy."""
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / (area_a + area_b - inter)


def _xyxy_to_xyah(bbox: np.ndarray) -> np.ndarray:
    height = max(float(bbox[3] - bbox[1]), 1e-6)
    return np.array([
        (bbox[0] + bbox[2]) / 2.,
        (bbox[1] + bbox[3]) / 2.,
        (bbox[2] - bbox[0]) / height,
        height,
    ], dtype=np.float64)


@dataclass(frozen=True, slots=True)
class TrackLifecycle:
    track_id: int
    state: str
    last_seen_mono: float
    reason: str | None = None


class Track:
    __slots__ = (
        "track_id", "bbox", "score", "state", "last_seen", "hits", "is_new",
        "employee_id", "employee_conf", "candidate_id", "candidate_count",
        "last_recog_attempt", "best_unknown_sim", "first_seen", "association_uncertain",
        "_mean", "_covariance", "_prediction_at",
    )

    def __init__(self, track_id: int, bbox: np.ndarray, score: float, now: float):
        self.track_id = track_id
        self.bbox = bbox
        self.score = score
        self.state = "tracked"           # tracked | lost
        self.last_seen = now
        self.hits = 1
        self.is_new = True
        self.first_seen = now
        self.association_uncertain = False
        # Both lost and ambiguous continuity invalidate face evidence.
        self.employee_id: Optional[int] = None
        self.employee_conf: Optional[float] = None
        self.candidate_id: Optional[int] = None
        self.candidate_count = 0
        self.last_recog_attempt: float = 0.0
        self.best_unknown_sim: Optional[float] = None
        measurement = _xyxy_to_xyah(bbox)
        height = measurement[3]
        self._mean = np.concatenate((measurement, np.zeros(4)))
        # Velocities are in native units per second, not per assumed frame.
        deviation = np.array([
            height / 10., height / 10., .01, height / 10.,
            height, height, .1, height,
        ])
        self._covariance = np.diag(deviation ** 2)
        self._prediction_at = now

    def clear_identity(self) -> None:
        self.employee_id = None
        self.employee_conf = None
        self.candidate_id = None
        self.candidate_count = 0
        self.last_recog_attempt = 0.0
        self.best_unknown_sim = None

    def _predict(self, now: float, lost_timeout: float) -> np.ndarray:
        """Advance internal motion only; never publish a predicted observation."""
        elapsed = min(max(now - self._prediction_at, 0.), lost_timeout)
        if elapsed > 0.:
            height = max(self._mean[3], 1e-6)
            transition = np.eye(8)
            transition[:4, 4:] = np.eye(4) * elapsed
            # Continuous white acceleration keeps uncertainty tied to elapsed
            # receipt time, including intervening frames with no detections.
            acceleration = np.array([height / 4., height / 4., .05, height / 4.]) ** 2
            noise = np.zeros((8, 8))
            noise[:4, :4] = np.diag(acceleration * elapsed ** 3 / 3.)
            cross_noise = np.diag(acceleration * elapsed ** 2 / 2.)
            noise[:4, 4:] = cross_noise
            noise[4:, :4] = cross_noise
            noise[4:, 4:] = np.diag(acceleration * elapsed)
            self._mean = transition @ self._mean
            self._covariance = transition @ self._covariance @ transition.T + noise
            self._prediction_at = now

        cx, cy, aspect, height = self._mean[:4]
        height = max(height, 1e-6)
        width = max(aspect * height, 1e-6)
        return np.array([
            cx - width / 2., cy - height / 2.,
            cx + width / 2., cy + height / 2.,
        ])

    def _observe(self, bbox: np.ndarray, score: float, now: float) -> None:
        measurement = _xyxy_to_xyah(bbox)
        height = max(self._mean[3], 1e-6)
        noise = np.diag(np.array([height / 20., height / 20., .01, height / 20.]) ** 2)
        innovation_covariance = self._covariance[:4, :4] + noise
        gain = np.linalg.solve(innovation_covariance, self._covariance[:4, :]).T
        self._mean += gain @ (measurement - self._mean[:4])
        self._covariance -= gain @ innovation_covariance @ gain.T
        self._covariance = (self._covariance + self._covariance.T) / 2.
        self.bbox = bbox
        self.score = score
        self.last_seen = now
        self.hits = self.hits + 1 if self.state == "tracked" else 1
        self.state = "tracked"


class ByteTracker:
    """Трекер одной камеры. update() вызывается на каждый обработанный AI-кадр."""

    def __init__(
        self,
        lost_timeout: float = 3.0,
        high_threshold: float = 0.5,
        low_threshold: float = 0.1,
        match_threshold: float = 0.3,
    ):
        self.lost_timeout = lost_timeout
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.match_threshold = match_threshold
        self.tracks: list[Track] = []
        self._next_id = 1
        self.lifecycle: list[TrackLifecycle] = []

    def update(
        self, detections: list[tuple[np.ndarray, float]], now: Optional[float] = None
    ) -> list[Track]:
        """Associate only unexpired, unambiguous tracks; hits are consecutive."""
        now = time.monotonic() if now is None else now
        self.lifecycle = [
            TrackLifecycle(t.track_id, "EXPIRED", t.last_seen, "TRACK_EXPIRED")
            for t in self.tracks if now - t.last_seen > self.lost_timeout
        ]
        self.tracks = [
            t for t in self.tracks if now - t.last_seen <= self.lost_timeout
        ]
        predictions = []
        for track in self.tracks:
            track.is_new = False
            track.association_uncertain = False
            predictions.append(track._predict(now, self.lost_timeout))

        high_idx = [i for i, (_b, s) in enumerate(detections) if s >= self.high_threshold]
        low_idx = [
            i for i, (_b, s) in enumerate(detections)
            if self.low_threshold <= s < self.high_threshold
        ]
        candidate_indices = high_idx + low_idx
        overlaps = np.empty((len(self.tracks), len(candidate_indices)))
        for row, predicted in enumerate(predictions):
            for column, index in enumerate(candidate_indices):
                overlaps[row, column] = iou_xyxy(predicted, detections[index][0])
        uncertain_tracks, uncertain_dets = self._ambiguity(overlaps)
        existing = [row for row in range(len(self.tracks)) if row not in uncertain_tracks]
        high_columns = [column for column in range(len(high_idx))
                        if column not in uncertain_dets]
        low_columns = [column for column in range(len(high_idx), len(candidate_indices))
                       if column not in uncertain_dets]
        matches = self._match(overlaps, existing, high_columns)
        matched_rows = {row for row, _column in matches}
        matches.extend(self._match(
            overlaps, [row for row in existing if row not in matched_rows], low_columns,
        ))
        matched_rows.update(row for row, _column in matches)
        matched_columns = {column for _row, column in matches}
        for row, column in matches:
            bbox, score = detections[candidate_indices[column]]
            self.tracks[row]._observe(bbox, score, now)

        for row, track in enumerate(self.tracks):
            if row in uncertain_tracks:
                track.association_uncertain = True
                track.clear_identity()
                track.state = "lost"
                track.hits = 0
                state, reason = "AMBIGUOUS", "ASSOCIATION_UNCERTAIN"
            elif row not in matched_rows:
                track.clear_identity()
                track.state = "lost"
                track.hits = 0
                state, reason = "LOST", "NO_DETECTION"
            else:
                state, reason = "TRACKED", None
            self.lifecycle.append(TrackLifecycle(
                track.track_id, state, track.last_seen, reason,
            ))

        # Quarantined high detections cannot spawn a trusted continuation either.
        for column in high_columns:
            if column in matched_columns:
                continue
            bbox, score = detections[candidate_indices[column]]
            track = Track(self._next_id, bbox, score, now)
            self.tracks.append(track)
            self._next_id += 1
            self.lifecycle.append(TrackLifecycle(track.track_id, "TRACKED", now))
        self.lifecycle.sort(key=lambda item: item.track_id)
        return [t for t in self.tracks if t.state == "tracked"]

    # ------------------------------------------------------------ внутреннее

    def _ambiguity(self, overlaps: np.ndarray) -> tuple[set[int], set[int]]:
        """Quarantine a whole association component before high/low splitting."""
        eligible = overlaps >= self.match_threshold

        def close_pair(scores: np.ndarray) -> bool:
            if len(scores) < 2:
                return False
            best = np.partition(scores, -2)[-2:]
            return bool(abs(best[1] - best[0]) <= .10 + 1e-12)

        tracks = {row for row in range(overlaps.shape[0])
                  if close_pair(overlaps[row, eligible[row]])}
        dets = {column for column in range(overlaps.shape[1])
                if close_pair(overlaps[eligible[:, column], column])}
        pending_tracks = list(tracks)
        pending_dets = list(dets)
        while pending_tracks or pending_dets:
            while pending_tracks:
                row = pending_tracks.pop()
                for column in np.flatnonzero(eligible[row]):
                    if column not in dets:
                        dets.add(column)
                        pending_dets.append(column)
            while pending_dets:
                column = pending_dets.pop()
                for row in np.flatnonzero(eligible[:, column]):
                    if row not in tracks:
                        tracks.add(row)
                        pending_tracks.append(row)
        return tracks, dets

    def _match(
        self, overlaps: np.ndarray, track_indices: list[int], det_indices: list[int],
    ) -> list[tuple[int, int]]:
        """Find the global minimum-cost gated assignment for one score stage."""
        if not track_indices or not det_indices:
            return []
        stage_overlaps = overlaps[np.ix_(track_indices, det_indices)]
        eligible = stage_overlaps >= self.match_threshold
        if not np.any(eligible):
            return []
        # A forbidden edge costs more than all valid assignments combined, so
        # filtering after assignment cannot sacrifice an available valid match.
        forbidden_cost = min(stage_overlaps.shape) + 1.
        cost = np.where(eligible, 1. - stage_overlaps, forbidden_cost)
        rows, columns = linear_sum_assignment(cost)
        return [
            (track_indices[row], det_indices[column])
            for row, column in zip(rows, columns) if eligible[row, column]
        ]
