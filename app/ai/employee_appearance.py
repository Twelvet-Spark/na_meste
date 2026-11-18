"""Same-day body-appearance gallery of face-confirmed employees.

Appearance never creates an identity by itself: samples are learned only from tracks whose
employee was just confirmed by face, and are discarded when the workplace calendar day changes.
Used from the AI worker thread only, so it needs no lock.
"""
from collections import deque
from datetime import date, datetime, timezone

import numpy as np

from app.config import settings
from app.time_utils import local_date

LEARN_FACE_SECONDS = 10.
MAX_SAMPLES = 64
TOP_K = 3
REID_VOTES = 2


def _naive_utc(value: datetime) -> datetime:
    return value.astimezone(timezone.utc).replace(tzinfo=None) if value.tzinfo is not None else value


def _unit(embedding: np.ndarray) -> np.ndarray | None:
    vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm == 0.:
        return None
    return vector / norm


class EmployeeAppearanceGallery:
    def __init__(self):
        self._day: date | None = None
        self._samples: dict[int, deque[np.ndarray]] = {}

    def clear(self) -> None:
        """A changed employee/face gallery invalidates every learned body sample."""
        self._samples.clear()
        self._day = None

    def _roll(self, observed_at: datetime) -> None:
        day = local_date(_naive_utc(observed_at))
        if day != self._day:
            self._samples.clear()
            self._day = day

    def learn(self, employee_id: int, embedding: np.ndarray, observed_at: datetime) -> None:
        self._roll(observed_at)
        vector = _unit(embedding)
        if vector is not None:
            self._samples.setdefault(employee_id, deque(maxlen=MAX_SAMPLES)).append(vector)

    def match(self, embedding: np.ndarray, observed_at: datetime) -> int | None:
        """The employee whose appearance clearly matches, else None."""
        self._roll(observed_at)
        vector = _unit(embedding)
        if vector is None:
            return None
        scores = []
        for employee_id, samples in self._samples.items():
            similarities = np.stack(samples) @ vector
            top = np.sort(similarities)[-min(TOP_K, len(similarities)):]
            scores.append((float(top.mean()), employee_id))
        if not scores:
            return None
        scores.sort(reverse=True)
        best, employee_id = scores[0]
        if best < settings.reid_employee_threshold:
            return None
        if len(scores) > 1 and best - scores[1][0] < settings.reid_employee_margin:
            return None
        return employee_id
