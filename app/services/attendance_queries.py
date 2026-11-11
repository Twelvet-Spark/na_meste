"""One explicit SQLite read snapshot and one shared work-span trust evaluator."""
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, TypeVar

from sqlalchemy import Engine, or_
from sqlalchemy.orm import Session

from app.attendance_types import EventView, GapView, SessionView, milliseconds
from app.database.models import AttendanceCoverageGap, AttendanceEvent, AttendanceEvidence, WorkSession
from app.time_utils import attendance_iso

T = TypeVar("T")


def read_snapshot(engine: Engine, reader: Callable[[Session], T]) -> T:
    with engine.connect() as connection:
        connection.exec_driver_sql("BEGIN")
        try:
            with Session(bind=connection, expire_on_commit=False) as db:
                return reader(db)
        finally:
            connection.rollback()


@dataclass(frozen=True, slots=True)
class TrustBlocker:
    key: str
    employee_id: int | None
    started_at: datetime
    ended_at: datetime
    reason: str


def load_trust_blockers(db: Session, *, as_of: datetime) -> list[TrustBlocker]:
    blockers = []
    for row in db.query(AttendanceEvent).filter(or_(
        AttendanceEvent.status == "REVIEW", AttendanceEvent.projection_reasons != [],
    )):
        if row.occurred_at > as_of:
            continue
        reasons = row.projection_reasons or row.reason_codes or ["UNRESOLVED_CROSSING"]
        for reason in reasons:
            blockers.append(TrustBlocker(f"event:{row.id}", row.employee_id,
                                         row.occurred_at, as_of, reason))
    for row in db.query(AttendanceCoverageGap).filter_by(reviewed=False):
        if row.started_at > as_of:
            continue
        for reason in row.reason_codes:
            blockers.append(TrustBlocker(f"gap:{row.id}", None, row.started_at,
                                         row.ended_at if row.ended_at is not None else as_of, reason))
    return blockers


def overlaps(start: datetime, end: datetime, blocker: TrustBlocker, *, open_end=False) -> bool:
    if blocker.started_at == blocker.ended_at:
        return start <= blocker.started_at and (blocker.started_at < end or
                                                (open_end and blocker.started_at == end))
    return blocker.started_at < end and blocker.ended_at > start


def duration_ms(start: datetime, end: datetime) -> int:
    delta = milliseconds(end) - milliseconds(start)
    return max(0, delta.days * 86_400_000 + delta.seconds * 1000 + delta.microseconds // 1000)


def evaluate_session(session: WorkSession, blockers: list[TrustBlocker], *, as_of: datetime) -> SessionView:
    start = milliseconds(session.started_at)
    end = milliseconds(session.ended_at if session.ended_at is not None else as_of)
    reasons = set(session.reason_codes or [])
    for blocker in blockers:
        if blocker.employee_id not in (None, session.employee_id):
            continue
        if overlaps(start, end, blocker, open_end=session.ended_at is None):
            reasons.add(blocker.reason)
    disputed = bool(reasons) or session.status == "REVIEW"
    duration = duration_ms(start, end)
    closed = session.ended_at is not None
    return SessionView(
        entry_event_id=session.entry_event_id, exit_event_id=session.exit_event_id,
        employee_id=session.employee_id, employee_name=session.employee_name,
        started_at=attendance_iso(start), ended_at=attendance_iso(session.ended_at),
        status="REVIEW" if disputed else "CLOSED" if closed else "OPEN",
        reason_codes=sorted(reasons), duration_ms=duration,
        confirmed_ms=duration if closed and not disputed else 0,
        running_ms=duration if not closed and not disputed else 0,
        review_ms=duration if disputed else 0,
    )


def event_view(db: Session, row: AttendanceEvent) -> EventView:
    image = db.get(AttendanceEvidence, row.id)
    return EventView(
        **{key: getattr(row, key) for key in (
            "id", "source", "camera_id", "camera_name", "employee_id", "employee_name",
            "direction", "status", "revision", "reason_codes", "projection_reasons")},
        occurred_at=attendance_iso(row.occurred_at), recorded_at=attendance_iso(row.recorded_at),
        evidence=dict(before=bool(image and image.before_jpeg), after=bool(image and image.after_jpeg),
                      face=bool(image and image.face_jpeg), purged=bool(image and image.purged_at)),
        needs_review=row.status == "REVIEW" or bool(row.projection_reasons),
    )


def gap_view(row: AttendanceCoverageGap) -> GapView:
    return GapView(id=row.id, started_at=attendance_iso(row.started_at), ended_at=attendance_iso(row.ended_at),
                   reason_codes=row.reason_codes, reviewed=row.reviewed, revision=row.revision)
