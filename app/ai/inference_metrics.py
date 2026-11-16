"""Bounded per-camera receipt-to-completion measurements; no frame retention."""
from collections import deque
import math
import threading


class InferenceMetrics:
    def __init__(self):
        self._lock = threading.Lock()
        self._cameras: dict[int, dict] = {}

    def configure(self, camera_id, target_fps, now):
        with self._lock:
            state = self._cameras.setdefault(camera_id, {
                "samples": deque(maxlen=4096), "since": now, "bad_since": None,
                "good_since": None, "degraded": False, "skipped": 0,
                "face_deferred": 0, "reid_deferred": 0, "pending": None,
                "stage_times_ms": {}, "error_code": None,
                "queue_age_seconds": None, "processing_seconds": None, "completion_age_seconds": None,
            })
            state["target_fps"] = target_fps

    def pending(self, camera_id, received_mono):
        with self._lock:
            self._cameras[camera_id]["pending"] = received_mono

    def complete(self, camera_id, received_mono, completed, *, skipped=0,
                 face_deferred=0, stage_times_ms=None, error_code=None, started_mono=None):
        with self._lock:
            state = self._cameras[camera_id]
            state["pending"] = None
            state["error_code"] = error_code
            state["skipped"] += skipped
            state["face_deferred"] += face_deferred
            if error_code is None:
                state["samples"].append((completed, max(0., completed - received_mono)))
            state["stage_times_ms"] = dict(stage_times_ms or {})
            state["completion_age_seconds"] = max(0., completed - received_mono)
            state["queue_age_seconds"] = max(0., started_mono - received_mono) if started_mono is not None else None
            state["processing_seconds"] = max(0., completed - started_mono) if started_mono is not None else None

    def defer_reid(self, camera_id):
        with self._lock:
            self._cameras[camera_id]["reid_deferred"] += 1

    def drop(self, camera_id):
        with self._lock:
            self._cameras.pop(camera_id, None)

    def snapshot(self, camera_id, now):
        with self._lock:
            state = self._cameras.get(camera_id)
            if state is None:
                return None
            samples = state["samples"]
            while samples and samples[0][0] < now - 60.:
                samples.popleft()
            fps = sum(completed > now - 10. for completed, _age in samples) / 10.
            ages = sorted(age for _completed, age in samples)
            p95 = ages[max(0, math.ceil(.95 * len(ages)) - 1)] if ages else None
            unhealthy = fps < .9 * state["target_fps"] or (p95 is not None and p95 > 1.)
            if unhealthy:
                state["good_since"] = None
                if state["bad_since"] is None:
                    state["bad_since"] = now
                if now - state["bad_since"] >= 10.:
                    state["degraded"] = True
            else:
                state["bad_since"] = None
                if state["good_since"] is None:
                    state["good_since"] = now
                if now - state["good_since"] >= 10.:
                    state["degraded"] = False
            return {
                "processed_fps_10s": round(fps, 3), "target_fps": state["target_fps"],
                "completion_age_p95_seconds": p95,
                "queue_age_seconds": state["queue_age_seconds"],
                "processing_seconds": state["processing_seconds"],
                "completion_age_seconds": state["completion_age_seconds"],
                "pending_age_seconds": (max(0., now - state["pending"]) if state["pending"] is not None else None),
                "skipped_slots": state["skipped"], "face_deferred": state["face_deferred"],
                "reid_deferred": state["reid_deferred"], "stage_times_ms": dict(state["stage_times_ms"]),
                "reason_codes": (["CAPACITY_DEGRADED"] if state["degraded"] else [])
                    + ([state["error_code"]] if state["error_code"] else []),
            }

    def snapshots(self, now):
        with self._lock:
            camera_ids = list(self._cameras)
        result = {}
        for camera_id in camera_ids:
            measurement = self.snapshot(camera_id, now)
            if measurement is not None:
                result[camera_id] = measurement
        return result
