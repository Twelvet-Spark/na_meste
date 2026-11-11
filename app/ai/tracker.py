"""Greedy two-stage IoU tracking with explicit, conservative continuity states."""
import logging
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

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
        # Identity may survive a short loss, but never an ambiguous association.
        self.employee_id: Optional[int] = None
        self.employee_conf: Optional[float] = None
        self.candidate_id: Optional[int] = None
        self.candidate_count = 0
        self.last_recog_attempt: float = 0.0
        self.best_unknown_sim: Optional[float] = None

    def clear_identity(self) -> None:
        self.employee_id = None
        self.employee_conf = None
        self.candidate_id = None
        self.candidate_count = 0
        self.last_recog_attempt = 0.0
        self.best_unknown_sim = None


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
        for track in self.tracks:
            track.is_new = False
            track.association_uncertain = False

        high_idx = [i for i, (_b, s) in enumerate(detections) if s >= self.high_threshold]
        low_idx = [
            i for i, (_b, s) in enumerate(detections)
            if self.low_threshold <= s < self.high_threshold
        ]
        uncertain_tracks, uncertain_dets = self._ambiguity(
            high_idx + low_idx, detections,
        )
        existing = [t for t in self.tracks if t.track_id not in uncertain_tracks]
        matches_1, rest_high = self._match(
            existing, [i for i in high_idx if i not in uncertain_dets], detections, now,
        )
        matched_ids = {t.track_id for t in matches_1}
        matches_2, _rest_low = self._match(
            [t for t in existing if t.track_id not in matched_ids],
            [i for i in low_idx if i not in uncertain_dets], detections, now,
        )
        matched_ids.update(t.track_id for t in matches_2)
        for track in self.tracks:
            if track.track_id in uncertain_tracks:
                track.association_uncertain = True
                track.clear_identity()
                track.state = "lost"
                track.hits = 0
                state, reason = "AMBIGUOUS", "ASSOCIATION_UNCERTAIN"
            elif track.track_id not in matched_ids:
                track.state = "lost"
                track.hits = 0
                state, reason = "LOST", "NO_DETECTION"
            else:
                state, reason = "TRACKED", None
            self.lifecycle.append(TrackLifecycle(
                track.track_id, state, track.last_seen, reason,
            ))

        # Ambiguous high detections cannot spawn a trusted continuation either.
        for index in rest_high:
            bbox, score = detections[index]
            track = Track(self._next_id, bbox, score, now)
            self.tracks.append(track)
            self._next_id += 1
            self.lifecycle.append(TrackLifecycle(track.track_id, "TRACKED", now))
        self.lifecycle.sort(key=lambda item: item.track_id)
        return [t for t in self.tracks if t.state == "tracked"]

    # ------------------------------------------------------------ внутреннее

    def _ambiguity(
        self, det_indices: list[int], detections: list[tuple[np.ndarray, float]],
    ) -> tuple[set[int], set[int]]:
        """Quarantine the involved association component before high/low splitting."""
        edges: list[tuple[int, int, float]] = []
        by_track: dict[int, list[float]] = {}
        by_detection: dict[int, list[float]] = {}
        for track in self.tracks:
            for index in det_indices:
                score = iou_xyxy(track.bbox, detections[index][0])
                if score >= self.match_threshold:
                    edges.append((track.track_id, index, score))
                    by_track.setdefault(track.track_id, []).append(score)
                    by_detection.setdefault(index, []).append(score)

        def close_pair(scores: list[float]) -> bool:
            ordered = sorted(scores)
            return any(b - a <= 0.10 + 1e-12 for a, b in zip(ordered, ordered[1:]))

        tracks = {key for key, scores in by_track.items() if close_pair(scores)}
        dets = {key for key, scores in by_detection.items() if close_pair(scores)}
        changed = True
        while changed:
            changed = False
            for track_id, index, _score in edges:
                if track_id in tracks or index in dets:
                    if track_id not in tracks or index not in dets:
                        tracks.add(track_id)
                        dets.add(index)
                        changed = True
        return tracks, dets

    def _match(
        self,
        tracks: list[Track],
        det_indices: list[int],
        detections: list[tuple[np.ndarray, float]],
        now: float,
    ) -> tuple[list[Track], list[int]]:
        """Жадное IoU-сопоставление треков с детекциями.

        Возвращает (совпавшие треки, индексы непроклассифицированных детекций).
        """
        if not tracks or not det_indices:
            return [], list(det_indices)

        pairs: list[tuple[float, Track, int]] = []
        for t in tracks:
            for j in det_indices:
                v = iou_xyxy(t.bbox, detections[j][0])
                if v >= self.match_threshold:
                    pairs.append((v, t, j))
        pairs.sort(key=lambda p: (-p[0], p[1].track_id, p[2]))

        matched: list[Track] = []
        used_tracks: set[int] = set()
        used_dets: set[int] = set()
        for v, t, j in pairs:
            if id(t) in used_tracks or j in used_dets:
                continue
            bbox, score = detections[j]
            t.bbox = bbox
            t.score = score
            t.last_seen = now
            t.hits = t.hits + 1 if t.state == "tracked" else 1
            t.state = "tracked"
            used_tracks.add(id(t))
            used_dets.add(j)
            matched.append(t)

        return matched, [j for j in det_indices if j not in used_dets]
