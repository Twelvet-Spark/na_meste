"""Thread-safe metrics and immutable fault notifications, never trajectory state."""
import copy
import math
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from uuid import uuid4

from app.config import settings
from app.time_utils import attendance_iso, utcnow

PROVIDER_KEYS = ("yolo", "live_face_detection", "live_face_recognition",
                 "enroll_face_detection", "enroll_face_recognition", "reid")


@dataclass(frozen=True, slots=True)
class CoverageFault:
    id: str
    started_at: datetime
    started_mono: float
    ended_at: datetime | None
    reason_codes: tuple[str, ...]


class AttendanceHealth:
    def __init__(self, *, now=utcnow, monotonic=time.monotonic):
        self._now, self._mono = now, monotonic
        self._lock = threading.RLock()
        self._pending = {}
        self._sources = {}
        self._ongoing = {}
        # Delivery and recovery are separate: acknowledged open notifications
        # still describe an unsafe interval. Recovery has a new notification
        # UUID but retains the original start/monotonic/reason identity.
        self._technical_faults = {}
        self._roles = {}
        self._providers = {key: [] for key in PROVIDER_KEYS}
        self._commission_valid = False
        self._clock_blocked = False
        self._clock_previous = None
        self._clock_floor = None
        self._healthy_since = None
        self._pending_writes = 0

    def publish_fault(self, fault: CoverageFault):
        with self._lock:
            if fault.ended_at is not None and fault.ended_at < fault.started_at:
                raise ValueError("Negative coverage interval")
            if fault.id in self._pending:
                if self._pending[fault.id] != fault:
                    raise ValueError("Fault notifications are immutable")
                return
            reasons = tuple(sorted(set(fault.reason_codes)))
            technical_reasons = tuple(reason for reason in reasons if reason != "COMMISSIONING_REQUIRED")
            if technical_reasons:
                # Invalidate at publication, not at the next snapshot. A whole
                # interruption may start and recover between two monitor polls.
                self._healthy_since = None
                identity = (fault.started_at, fault.started_mono, reasons)
                if fault.ended_at is None:
                    self._technical_faults[identity] = technical_reasons
                else:
                    self._technical_faults.pop(identity, None)
            self._pending[fault.id] = fault
            self._sources[fault.id] = (fault.id,)
            if len(self._pending) > 100:
                items = tuple(self._pending.values())
                identifiers = tuple(dict.fromkeys(source for key in self._pending for source in self._sources[key]))
                merged = CoverageFault(str(uuid4()), min(item.started_at for item in items),
                                       min(item.started_mono for item in items),
                                       None if any(item.ended_at is None for item in items) else
                                       max(item.ended_at for item in items),
                                       tuple(sorted({reason for item in items for reason in item.reason_codes})))
                self._pending = {merged.id: merged}
                self._sources = {merged.id: identifiers + (merged.id,)}

    def pending_faults(self):
        with self._lock:
            return tuple(self._pending.values())

    def fault_source_ids(self, identifier):
        with self._lock:
            return self._sources.get(identifier, (identifier,))

    def ack_fault(self, identifier):
        with self._lock:
            self._pending.pop(identifier, None)
            self._sources.pop(identifier, None)

    def transition(self, key, reasons, *, at=None, mono=None, recovered_at=None):
        """Publish only start/end/reason transitions; never overwrite a notice."""
        with self._lock:
            at = self._now() if at is None else at
            mono = self._mono() if mono is None else mono
            reasons = tuple(sorted(set(reasons)))
            previous = self._ongoing.get(key)
            if previous is not None and previous.reason_codes == reasons:
                return
            if previous is not None:
                closed = CoverageFault(str(uuid4()), previous.started_at, previous.started_mono,
                                       max(previous.started_at, recovered_at or at), previous.reason_codes)
                self.publish_fault(closed)
                del self._ongoing[key]
            if reasons:
                opened = CoverageFault(str(uuid4()), at, mono, None, reasons)
                self._ongoing[key] = opened
                self.publish_fault(opened)

    def set_providers(self, providers):
        with self._lock:
            self._providers = {key: list(providers.get(key, [])) for key in PROVIDER_KEYS}

    def configure_role(self, role, *, camera_id, source_revision, enabled, online, included):
        with self._lock:
            row = self._roles.get(role)
            if row is None or (row["camera_id"], row["source_revision"]) != (camera_id, source_revision):
                row = dict(camera_id=camera_id, source_revision=source_revision, stream_id=None,
                           tracking_generation=None, last_processed_at=None, last_mono=None,
                           frames=deque(), samples=deque(maxlen=600),
                           low_since=self._mono(), good_since=None, degraded=False)
                self._roles[role] = row
            row.update(enabled=enabled, online=online, included=included)

    def record_frame(self, role, packet, *, tracking_generation, started_mono, completed_mono):
        with self._lock:
            row = self._roles.get(role)
            if row is None or row["source_revision"] != packet.source_revision:
                return
            previous = row["last_mono"]
            reason = None
            if row["stream_id"] not in (None, packet.stream_id):
                reason = "STREAM_RESET"
            elif previous is not None and packet.received_mono - previous > settings.attendance_max_frame_gap:
                reason = "FRAME_GAP"
            if reason:
                start = row["last_processed_at"] or packet.received_at
                self.publish_fault(CoverageFault(str(uuid4()), min(start, packet.received_at),
                                                previous or packet.received_mono,
                                                max(start, packet.received_at), (reason,)))
            row.update(stream_id=packet.stream_id, tracking_generation=tracking_generation,
                       last_processed_at=packet.received_at, last_mono=packet.received_mono)
            while row["frames"] and row["frames"][0] < completed_mono - 10:
                row["frames"].popleft()
            row["frames"].append(completed_mono)
            row["samples"].append((completed_mono, max(0., started_mono - packet.received_mono),
                                   max(0., completed_mono - packet.received_mono)))

    def set_admission(self, *, commission_valid, clock_blocked):
        with self._lock:
            self._commission_valid, self._clock_blocked = commission_valid, clock_blocked

    def set_pending_writes(self, count):
        with self._lock:
            self._pending_writes = count

    def observe_clock(self, utc, mono):
        with self._lock:
            previous = self._clock_previous
            self._clock_previous = (utc, mono)
            if previous is not None and abs((utc - previous[0]).total_seconds() - (mono - previous[1])) > 1:
                self._clock_blocked = True
                self._clock_floor = max(utc, previous[0], self._clock_floor or utc)
                self.transition("clock", ("CLOCK_DISCONTINUITY",), at=min(utc, previous[0]), mono=previous[1])
                return False
            if "clock" in self._ongoing:
                self.transition("clock", (), at=utc, mono=mono, recovered_at=max(utc, self._clock_floor))
            return True

    @staticmethod
    def _p95(values):
        if not values:
            return None
        ordered = sorted(values)
        return ordered[max(0, math.ceil(len(ordered) * .95) - 1)]

    def snapshot(self):
        with self._lock:
            now, utc = self._mono(), self._now()
            roles, reasons = {}, set()
            models_ready = all(self._providers[key] for key in PROVIDER_KEYS[:5])
            if not models_ready:
                reasons.add("AI_UNAVAILABLE")
            for role in ("ENTRY", "EXIT"):
                row = self._roles.get(role)
                if row is None:
                    reasons.add("CAMERA_UNCONFIGURED")
                    continue
                while row["frames"] and row["frames"][0] < now - 10:
                    row["frames"].popleft()
                while row["samples"] and row["samples"][0][0] < now - 60:
                    row["samples"].popleft()
                fps = len(row["frames"]) / 10
                age = None if row["last_mono"] is None else max(0., now - row["last_mono"])
                start_p95 = self._p95([item[1] for item in row["samples"]])
                end_p95 = self._p95([item[2] for item in row["samples"]])
                bad = fps < 8 or start_p95 is None or start_p95 > 1 or end_p95 is None or end_p95 > 1
                if bad:
                    row["good_since"] = None
                    if row["low_since"] is None:
                        row["low_since"] = now
                    if now - row["low_since"] >= 10:
                        row["degraded"] = True
                else:
                    row["low_since"] = None
                    if row["good_since"] is None:
                        row["good_since"] = now
                    if now - row["good_since"] >= 10:
                        row["degraded"] = False
                failures = set()
                if not row["enabled"]:
                    failures.add("CAMERA_UNCONFIGURED")
                if not row["online"]:
                    failures.add("CAMERA_OFFLINE")
                if not row["included"]:
                    failures.add("AI_EXCLUDED")
                if age is None or age > settings.attendance_frame_max_age:
                    failures.add("FRAME_GAP")
                if row["degraded"]:
                    failures.add("PERFORMANCE_DEGRADED")
                elif bad:
                    failures.add("PERFORMANCE_WARMUP")
                self.transition(f"role:{role}", failures,
                                at=row["last_processed_at"] if "FRAME_GAP" in failures and row["last_processed_at"] else utc,
                                mono=row["last_mono"] if "FRAME_GAP" in failures and row["last_mono"] is not None else now)
                reasons.update(failures)
                roles[role] = dict(camera_id=row["camera_id"], stream_id=row["stream_id"],
                                   source_revision=row["source_revision"], tracking_generation=row["tracking_generation"],
                                   last_processed_at=attendance_iso(row["last_processed_at"]), frame_age_seconds=age,
                                   processed_fps=fps, processing_age_p95_seconds=start_p95,
                                   completion_age_p95_seconds=end_p95, reasons=sorted(failures))
            self.transition("models", () if models_ready else ("AI_UNAVAILABLE",), at=utc, mono=now)
            for fault_reasons in self._technical_faults.values():
                reasons.update(fault_reasons)
            if self._pending_writes:
                reasons.add("PERSISTENCE_ERROR")
            technical = not reasons and len(roles) == 2
            if not technical:
                self._healthy_since = None
            elif self._healthy_since is None:
                self._healthy_since = now
            automatic = technical and self._commission_valid and not self._clock_blocked and not self._pending
            admission_reasons = set(reasons)
            if not self._commission_valid:
                admission_reasons.add("COMMISSIONING_REQUIRED")
            if self._clock_blocked:
                admission_reasons.add("CLOCK_DISCONTINUITY")
            return dict(roles=roles, providers=copy.deepcopy(self._providers), reasons=sorted(admission_reasons),
                        technical_ready=technical, automatic_ready=automatic,
                        technically_healthy_seconds=0. if self._healthy_since is None else max(0., now - self._healthy_since),
                        pending_fault_count=len(self._pending), pending_write_count=self._pending_writes)
