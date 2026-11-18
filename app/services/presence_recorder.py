"""In-memory presence segments and door crossings, flushed to SQLite by a background thread.

The worker thread only touches memory (`observe`, `record_crossing`); all database work
happens in `flush`, so recognition never waits on SQLite.
"""
import logging
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Literal

from sqlalchemy import update
from sqlalchemy.exc import SQLAlchemyError

from app.attendance_types import milliseconds
from app.config import settings
from app.database.database import immediate_session
from app.database.models import Camera, DoorCrossing, Employee, PresenceSegment
from app.time_utils import attendance_iso, day_bounds, local_date, utcnow

logger = logging.getLogger("app.services.presence")

SEGMENT_JOIN_SECONDS = 60.
MAX_PENDING_CROSSINGS = 1000
_JOIN = timedelta(seconds=SEGMENT_JOIN_SECONDS)


@dataclass(slots=True)
class _Segment:
    employee_id: int
    camera_id: int
    started_at: datetime
    last_seen_at: datetime
    face_seen: bool
    id: int | None = None
    version: int = 1
    persisted_version: int = 0

    @property
    def dirty(self) -> bool:
        return self.version != self.persisted_version


class PresenceRecorder:
    def __init__(self, engine, notifier=None, *, now=utcnow):
        self.engine, self.notifier, self._now = engine, notifier, now
        self._lock = threading.Lock()
        self._flush_lock = threading.Lock()
        self._open: dict[tuple[int, int], _Segment] = {}
        self._closing: list[_Segment] = []
        self._crossings: list[dict] = []
        self._dropped_crossings = 0
        self._last_flush_at = None
        self._last_error: str | None = None

    # -- worker thread: memory only ------------------------------------------------------------

    def observe(self, camera_id: int, employee_id: int, source: Literal["FACE", "REID"], observed_at) -> None:
        observed_at = milliseconds(observed_at)
        key = (employee_id, camera_id)
        with self._lock:
            segment = self._open.get(key)
            if segment is not None:
                if observed_at < segment.last_seen_at:
                    return
                if observed_at - segment.last_seen_at > _JOIN:
                    self._closing.append(segment)
                    segment = None
            if segment is None:
                self._open[key] = _Segment(employee_id, camera_id, observed_at, observed_at, source == "FACE")
                return
            segment.last_seen_at = observed_at
            segment.face_seen = segment.face_seen or source == "FACE"
            segment.version += 1

    def record_crossing(self, *, source_key: str, camera_id: int, employee_id: int, direction: str,
                        occurred_at) -> None:
        item = {"source_key": source_key, "camera_id": camera_id, "employee_id": employee_id,
                "direction": direction, "occurred_at": milliseconds(occurred_at)}
        with self._lock:
            self._crossings.append(item)
            overflow = len(self._crossings) - MAX_PENDING_CROSSINGS
            if overflow > 0:
                del self._crossings[:overflow]
                previously = self._dropped_crossings
                self._dropped_crossings += overflow
                if previously == 0 or previously // 100 != self._dropped_crossings // 100:
                    logger.warning("Pending door crossings exceeded %d; %d dropped so far",
                                   MAX_PENDING_CROSSINGS, self._dropped_crossings)

    def has_employee_work(self, employee_id: int) -> bool:
        with self._lock:
            return (any(key[0] == employee_id for key in self._open)
                    or any(s.employee_id == employee_id for s in self._closing)
                    or any(c["employee_id"] == employee_id for c in self._crossings))

    def has_camera_work(self, camera_id: int) -> bool:
        with self._lock:
            return (any(key[1] == camera_id for key in self._open)
                    or any(s.camera_id == camera_id for s in self._closing)
                    or any(c["camera_id"] == camera_id for c in self._crossings))

    def snapshot(self) -> dict:
        with self._lock:
            return {"open_segments": len(self._open), "pending_crossings": len(self._crossings),
                    "dropped_crossings": self._dropped_crossings,
                    "last_flush_at": attendance_iso(self._last_flush_at), "last_error": self._last_error}

    # -- flusher thread: all database work ----------------------------------------------------

    def flush(self) -> None:
        with self._flush_lock:
            self._flush()

    def _flush(self) -> None:
        with self._lock:
            segments = [s for s in (*self._closing, *self._open.values()) if s.dirty]
            captured = [(s, s.version, s.id, s.employee_id, s.camera_id, s.started_at, s.last_seen_at, s.face_seen)
                        for s in segments]
            crossings = list(self._crossings)
        notifications: list[str] = []
        stored: dict[int, int] = {}  # id(segment) -> new row id
        gone_employees: set[int] = set()
        gone_cameras: set[int] = set()
        try:
            if captured or crossings:
                with immediate_session(self.engine) as db:
                    wanted_employees = {c[3] for c in captured} | {c["employee_id"] for c in crossings}
                    wanted_cameras = {c[4] for c in captured} | {c["camera_id"] for c in crossings}
                    employees = {row.id: row.name for row in db.query(Employee.id, Employee.name).filter(
                        Employee.id.in_(wanted_employees))} if wanted_employees else {}
                    cameras = {row[0] for row in db.query(Camera.id).filter(
                        Camera.id.in_(wanted_cameras))} if wanted_cameras else set()
                    gone_employees = wanted_employees - set(employees)
                    gone_cameras = wanted_cameras - cameras
                    # Check before inserting, so both late replay and several cameras in one
                    # flush notify at most once. Earliest new segment wins within a flush.
                    for segment, _version, row_id, employee_id, camera_id, started, last_seen, face in sorted(
                            captured, key=lambda item: item[5]):
                        if employee_id in gone_employees or camera_id in gone_cameras:
                            continue
                        if row_id is None:
                            day_start = day_bounds(local_date(started))[0]
                            existing = db.query(PresenceSegment.id).filter(
                                PresenceSegment.employee_id == employee_id,
                                PresenceSegment.started_at >= day_start).first()
                            if existing is None:
                                notifications.append(
                                    f"На месте: {employees[employee_id]}, {attendance_iso(started)}")
                            row = PresenceSegment(employee_id=employee_id, camera_id=camera_id,
                                                  started_at=started, last_seen_at=last_seen, face_seen=face)
                            db.add(row)
                            db.flush()
                            stored[id(segment)] = row.id
                        else:
                            db.execute(update(PresenceSegment).where(PresenceSegment.id == row_id).values(
                                last_seen_at=last_seen, face_seen=face))
                    known = {row[0] for row in db.query(DoorCrossing.source_key).filter(
                        DoorCrossing.source_key.in_([c["source_key"] for c in crossings]))} if crossings else set()
                    recorded_at = milliseconds(self._now())
                    for item in crossings:
                        if (item["source_key"] in known or item["employee_id"] in gone_employees
                                or item["camera_id"] in gone_cameras):
                            continue
                        known.add(item["source_key"])
                        db.add(DoorCrossing(recorded_at=recorded_at, **item))
        except SQLAlchemyError:
            with self._lock:
                first_failure = self._last_error is None
                self._last_error = "PERSISTENCE_ERROR"
            if first_failure:
                logger.exception("Presence flush failed; observations are kept in memory")
            return

        if gone_employees or gone_cameras:
            logger.warning("Presence data for deleted employee/camera dropped")
        sent = {id(c) for c in crossings}
        now = self._now()
        with self._lock:
            for segment, version, row_id, employee_id, camera_id, *_ in captured:
                if employee_id in gone_employees or camera_id in gone_cameras:
                    segment.persisted_version = segment.version  # nothing can ever be stored for it
                    continue
                if id(segment) in stored:
                    segment.id = stored[id(segment)]
                segment.persisted_version = version
            self._crossings = [c for c in self._crossings if id(c) not in sent]
            self._closing = [s for s in self._closing if s.dirty]
            for key in [key for key, s in self._open.items()
                        if not s.dirty and now - s.last_seen_at > _JOIN]:
                del self._open[key]
            self._open = {key: s for key, s in self._open.items()
                          if key[0] not in gone_employees and key[1] not in gone_cameras}
            self._last_flush_at = now
            self._last_error = None
        if self.notifier is not None and settings.telegram_notify_presence:
            for text in notifications:
                try:
                    self.notifier.send_text(text)
                except Exception:
                    logger.exception("Presence notification failed after committed observation")


class PresenceFlusher(threading.Thread):
    def __init__(self, recorder: PresenceRecorder, *, interval: float = 1.):
        super().__init__(name="presence-flusher", daemon=True)
        self.recorder, self.interval = recorder, interval
        self._stop_event = threading.Event()

    def run(self) -> None:
        while not self._stop_event.wait(self.interval):
            try:
                self.recorder.flush()
            except Exception:
                logger.exception("Presence flusher iteration failed")

    def stop(self) -> None:
        self._stop_event.set()
        if self.is_alive():
            self.join(timeout=10)
        self.recorder.flush()
