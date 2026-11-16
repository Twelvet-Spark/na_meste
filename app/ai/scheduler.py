"""Worker-owned deadlines: distinct cameras, rotating ties, no catch-up queue."""
import math
from bisect import bisect_right


class DeadlineScheduler:
    def __init__(self, batch_size: int):
        self.batch_size = batch_size
        self.deadlines: dict[int, float] = {}
        self._last_tie = -1

    def retain(self, camera_ids, now: float) -> None:
        included = set(camera_ids)
        self.deadlines = {key: value for key, value in self.deadlines.items() if key in included}
        for camera_id in included:
            self.deadlines.setdefault(camera_id, now)

    def select(self, camera_ids, now: float, *, attendance_ids=()) -> list[int]:
        ids = sorted(set(camera_ids))
        if not ids:
            return []
        offset = bisect_right(ids, self._last_tie) % len(ids)
        rotation = ids[offset:] + ids[:offset]
        rank = {camera_id: index for index, camera_id in enumerate(rotation)}
        attendance = set(attendance_ids)
        due = [camera_id for camera_id in ids if self.deadlines[camera_id] <= now + 1e-9]
        # Higher-rate attendance deadlines win equal-time ties; rotating every
        # camera first can push their completion into the next 100ms slot.
        due.sort(key=lambda camera_id: (self.deadlines[camera_id], camera_id not in attendance, rank[camera_id]))
        selected = due[:self.batch_size]
        if selected:
            self._last_tie = selected[-1]
        return selected

    def advance(self, camera_id: int, fps: float, completed: float) -> int:
        """Advance the original phase, skipping slots already missed, never replaying."""
        period = 1. / fps
        previous = self.deadlines[camera_id]
        periods = max(1, math.floor((completed - previous) / period + 1e-9) + 1)
        self.deadlines[camera_id] = previous + periods * period
        return periods - 1

    def delay(self, now: float, maximum: float, *, camera_ids=None) -> float:
        deadlines = self.deadlines.values() if camera_ids is None else (
            self.deadlines[camera_id] for camera_id in camera_ids if camera_id in self.deadlines)
        earliest = min(deadlines, default=None)
        return maximum if earliest is None else min(maximum, max(0., earliest - now))

