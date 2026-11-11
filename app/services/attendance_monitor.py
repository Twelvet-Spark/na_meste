"""Single retry consumer and durable coverage monitor; never touches AI tracks."""
import logging
import threading
import time
from collections import deque
from uuid import uuid4

from app.config import settings
from app.database.models import AttendanceCamera, Camera
from app.rtsp.reader import CameraStatus
from app.services.attendance_queries import read_snapshot
from app.services.attendance_health import CoverageFault
from app.time_utils import utcnow

logger = logging.getLogger("app.attendance.monitor")


class AttendanceWriteQueue:
    def __init__(self, service, health):
        self.service, self.health = service, health
        self._lock = threading.RLock()
        self._items = deque()

    def pending(self):
        with self._lock:
            return tuple(self._items)

    def _append(self, crossing):
        if len(self._items) == 100:
            self.health.publish_fault(CoverageFault(
                str(uuid4()), crossing.occurred_at, crossing.occurred_mono,
                crossing.occurred_at, ("EVIDENCE_LOST",)))
            return
        self._items.append(crossing)
        self.health.set_pending_writes(len(self._items))

    def submit(self, crossing):
        """AI sole producer; new evidence never bypasses a queued head."""
        with self._lock:
            if self._items:
                self._append(crossing)
                return None
            try:
                return self.service.record_crossing(crossing)
            except Exception:
                logger.exception("Crossing not saved; retaining immutable evidence for retry")
                self.health.transition("persistence", ["PERSISTENCE_ERROR"],
                                       at=crossing.occurred_at, mono=crossing.occurred_mono)
                self._append(crossing)
                return None

    def retry_one(self):
        """Monitor sole consumer; one attempt, no retry loop or head replacement."""
        with self._lock:
            if not self._items:
                return False
            try:
                self.service.record_crossing(self._items[0])
            except Exception:
                logger.exception("Attendance FIFO head remains unsaved")
                return False
            self._items.popleft()
            self.health.set_pending_writes(len(self._items))
            if not self._items:
                self.health.transition("persistence", [])
            return True


class AttendanceMonitor(threading.Thread):
    def __init__(self, service, *, write_queue=None, manager=None, worker=None,
                 now=utcnow, monotonic=time.monotonic):
        super().__init__(name="attendance-monitor", daemon=True)
        self.service, self.health = service, service.health
        self.write_queue = write_queue
        self.manager, self.worker = manager, worker
        self._now, self._mono = now, monotonic
        self._stop_event = threading.Event()
        self._tick_lock = threading.Lock()
        self._last_heartbeat = float("-inf")
        self._last_retry = float("-inf")
        self._last_purge = float("-inf")
        self._started_runtime = False

    def startup(self):
        if self._started_runtime:
            return
        self.service.start_monitoring_process(self._now())
        self._started_runtime = True

    def _sample_readers(self):
        if self.manager is None:
            return
        def read(db):
            return [(row.role, row.camera_id, row.source_revision, row.enabled and enabled)
                    for row, enabled in db.query(AttendanceCamera, Camera.enabled).join(
                        Camera, Camera.id == AttendanceCamera.camera_id) if row.role is not None]
        configurations = read_snapshot(self.service.engine, read)
        allowed = settings.ai_camera_ids()
        for role in ("ENTRY", "EXIT"):
            matching = [row for row in configurations if row[0] == role and row[3]]
            if matching:
                _, camera_id, revision, enabled = matching[0]
                reader = self.manager.get(camera_id)
                online = (reader is not None and reader.is_alive() and reader.status() is CameraStatus.ONLINE
                          and reader.source_revision == revision)
                included = (settings.ai_enabled and self.worker is not None and self.worker.is_alive()
                            and (allowed is None or camera_id in allowed))
                self.health.configure_role(role, camera_id=camera_id, source_revision=revision,
                                           enabled=enabled, online=online, included=included)
            else:
                self.health.configure_role(role, camera_id=None, source_revision=0,
                                           enabled=False, online=False, included=False)

    def run(self):
        while not self._stop_event.is_set():
            self.tick()
            self._stop_event.wait(1.)

    def tick(self, *, stopped=False):
        with self._tick_lock:
            now, mono = self._now(), self._mono()
            try:
                if self.manager is not None and not self._started_runtime:
                    self.startup()
                self._sample_readers()
                if self.write_queue is not None and mono - self._last_retry >= 1 and not stopped:
                    self._last_retry = mono
                    self.write_queue.retry_one()
                snapshot = self.health.snapshot()
                faults = self.health.pending_faults()
                heartbeat = mono - self._last_heartbeat >= 5
                valid, blocked = self.service.persist_health(
                    [(fault, self.health.fault_source_ids(fault.id)) for fault in faults],
                    healthy=snapshot["technical_ready"] and not stopped,
                    reasons=snapshot["reasons"] + (["PROCESS_STOPPED"] if stopped else []),
                    observed_at=now, heartbeat=heartbeat, stopped=stopped,
                )
                # Only the successful outer coverage commit permits acknowledgment.
                for fault in faults:
                    self.health.ack_fault(fault.id)
                self.health.set_admission(commission_valid=valid, clock_blocked=blocked)
                self.health.transition("monitor", [])
                if heartbeat:
                    self._last_heartbeat = mono
                if mono - self._last_purge >= 3600:
                    self.service.purge_evidence(now)
                    self._last_purge = mono
            except Exception:
                logger.exception("Attendance coverage persistence failed; faults remain latched")
                self.health.transition("monitor", ["PERSISTENCE_ERROR"], at=now, mono=mono)
                self.health.set_admission(
                    commission_valid=False,
                    clock_blocked="CLOCK_DISCONTINUITY" in self.health.snapshot()["reasons"])

    def shutdown(self):
        if self.write_queue is not None:
            # Producer is already joined. Never skip an unsaved head to write a
            # later observation; every successfully drained head is tried once.
            for _ in range(len(self.write_queue.pending())):
                if not self.write_queue.retry_one():
                    break
            remaining = self.write_queue.pending()
            if remaining:
                self.health.publish_fault(CoverageFault(
                    str(uuid4()), min(item.occurred_at for item in remaining),
                    min(item.occurred_mono for item in remaining), None,
                    ("PERSISTENCE_ERROR", "EVIDENCE_LOST")))
        self.tick(stopped=True)

    def stop(self):
        self._stop_event.set()
        if self.ident is not None:
            self.join()
        self.shutdown()
