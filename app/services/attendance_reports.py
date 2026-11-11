"""Workplace-date reports from one caller-owned SQLite read snapshot.

Trust is evaluated for each complete candidate session before splitting its span
at midnight. UNMONITORED is a day-completeness condition, not an invented gap or
an instruction to discard explicitly confirmed manual intervals.
"""
import csv
import io
from collections import defaultdict
from datetime import date, datetime, timedelta
from typing import Literal

from fastapi import HTTPException
from pydantic import BaseModel
from sqlalchemy import or_
from sqlalchemy.orm import Session

from app.attendance_types import milliseconds
from app.config import settings
from app.database.models import (
    AttendanceEmployeeReference, AttendanceEvent, AttendanceRuntime, Employee, WorkSession,
)
from app.services.attendance_queries import (
    duration_ms, evaluate_session, load_trust_blockers, overlaps,
)
from app.time_utils import attendance_iso, day_bounds


class AttendanceReportRow(BaseModel):
    date: date
    employee_id: int
    employee_name: str
    confirmed_ms: int = 0
    running_ms: int = 0
    review_ms: int = 0
    visit_count: int = 0
    first_entry: str | None = None
    last_exit: str | None = None
    pending_count: int = 0
    status: Literal["REVIEW", "RUNNING", "CONFIRMED"]
    coverage_reasons: list[str]


class AttendanceReportTotal(BaseModel):
    employee_id: int
    employee_name: str
    confirmed_ms: int = 0
    running_ms: int = 0
    review_ms: int = 0
    pending_count: int = 0


class AttendanceReport(BaseModel):
    from_date: date
    to_date: date
    utc_offset_minutes: int
    as_of: str
    finalizable: bool
    unresolved_crossings: int
    unreviewed_gaps: int
    rows: list[AttendanceReportRow]
    totals: list[AttendanceReportTotal]


def build_report(db: Session, from_date: date, to_date: date, employee_id: int | None,
                 *, as_of: datetime) -> AttendanceReport:
    """Materialize the report entirely inside the supplied explicit snapshot."""
    if to_date < from_date or (to_date - from_date).days >= 366:
        raise HTTPException(422, "Report range must contain 1 to 366 inclusive dates")
    as_of = milliseconds(as_of)
    lower, upper = day_bounds(from_date)[0], day_bounds(to_date)[1]
    runtime = db.get(AttendanceRuntime, 1)
    monitored_from = runtime.monitoring_started_at if runtime is not None else None
    employees = {item.id: item for item in db.query(Employee)}
    if employee_id is not None and employee_id not in employees:
        raise HTTPException(404, "Employee not found")

    sessions_query = db.query(WorkSession).filter(
        WorkSession.started_at < upper, WorkSession.started_at <= as_of,
        or_(WorkSession.ended_at.is_(None), WorkSession.ended_at >= lower),
    )
    events_query = db.query(AttendanceEvent).filter(
        AttendanceEvent.occurred_at < upper, AttendanceEvent.occurred_at <= as_of,
        or_(AttendanceEvent.occurred_at >= lower, AttendanceEvent.status == "REVIEW",
            AttendanceEvent.projection_reasons != []),
    )
    references_query = db.query(AttendanceEmployeeReference).join(
        AttendanceEvent, AttendanceEvent.id == AttendanceEmployeeReference.event_id,
    ).filter(AttendanceEvent.occurred_at >= lower, AttendanceEvent.occurred_at < upper,
             AttendanceEvent.occurred_at <= as_of)
    if employee_id is not None:
        sessions_query = sessions_query.filter(WorkSession.employee_id == employee_id)
        events_query = events_query.filter(AttendanceEvent.employee_id == employee_id)
        references_query = references_query.filter(AttendanceEmployeeReference.employee_id == employee_id)
    sessions = sessions_query.order_by(WorkSession.started_at, WorkSession.entry_event_id).all()
    events = events_query.order_by(AttendanceEvent.occurred_at, AttendanceEvent.id).all()
    references = references_query.order_by(AttendanceEvent.occurred_at, AttendanceEvent.id).all()
    selected = ({employee_id} if employee_id is not None else
                {item.id for item in employees.values() if item.active}
                | {item.employee_id for item in sessions}
                | {item.employee_id for item in events if item.employee_id is not None}
                | {item.employee_id for item in references})
    ordered_ids = sorted(selected)

    # A history row uses its saved name, never a join to the renamed employee.
    # For empty dates, prefer a recorded name from this report over today's name.
    names = {identifier: employees[identifier].name for identifier in ordered_ids}
    snapshots = {}
    for item in references:
        snapshots.setdefault(item.employee_id, item.employee_name)
    for item in events:
        if item.employee_id is not None and item.employee_name is not None:
            snapshots.setdefault(item.employee_id, item.employee_name)
    for item in sessions:
        snapshots[item.employee_id] = snapshots.get(item.employee_id, item.employee_name)
    names.update(snapshots)

    # Do not date-filter this query: an unresolved observation from an earlier
    # date still blocks until this single frozen as_of. Later evidence can also
    # quarantine the pre-midnight portion of an overlapping overnight session.
    blockers = load_trust_blockers(db, as_of=as_of)
    blockers = [item for item in blockers if item.employee_id is None or item.employee_id in selected]
    by_employee = defaultdict(list)
    global_blockers = []
    for blocker in blockers:
        if blocker.employee_id is None:
            global_blockers.append(blocker)
        else:
            by_employee[blocker.employee_id].append(blocker)
    evaluated = defaultdict(list)
    relevant_keys = {item.key for item in blockers if overlaps(lower, upper, item)}
    for session in sessions:
        applicable = global_blockers + by_employee[session.employee_id]
        view = evaluate_session(session, applicable, as_of=as_of)
        start = milliseconds(session.started_at)
        end = min(milliseconds(session.ended_at) if session.ended_at is not None else as_of, as_of)
        evaluated[session.employee_id].append((session, view, start, end))
        # Midnight closures also load for last_exit metadata. Only a candidate
        # span with actual report overlap can propagate its out-of-range faults.
        if max(start, lower) < min(end, upper):
            for blocker in applicable:
                if overlaps(start, end, blocker, open_end=session.ended_at is None):
                    relevant_keys.add(blocker.key)

    employee_events = defaultdict(list)
    for item in events:
        if item.employee_id is not None:
            employee_events[item.employee_id].append(item)
    rows = []
    totals = {identifier: AttendanceReportTotal(employee_id=identifier, employee_name=names[identifier])
              for identifier in ordered_ids}
    period_pending = defaultdict(set)
    # Completeness exists even when there are no employees to produce rows.
    finalizable = upper <= as_of and monitored_from is not None and monitored_from <= lower
    if relevant_keys:
        finalizable = False
    for offset in range((to_date - from_date).days + 1):
        day = from_date + timedelta(days=offset)
        start, end = day_bounds(day)
        day_complete = end <= as_of
        common_reasons = set()
        if monitored_from is None or monitored_from > start:
            common_reasons.add("UNMONITORED")
        if start > as_of:
            common_reasons.add("FUTURE_DATE")
        for identifier in ordered_ids:
            reasons = set(common_reasons)
            pending = set()
            applicable = global_blockers + by_employee[identifier]
            for blocker in applicable:
                if overlaps(start, end, blocker):
                    reasons.add(blocker.reason)
                    if blocker.key.startswith("event:"):
                        pending.add(blocker.key.removeprefix("event:"))
            item = AttendanceReportRow(date=day, employee_id=identifier, employee_name=names[identifier],
                                       status="CONFIRMED", coverage_reasons=[])
            historical_name = None
            sequence_conflicts = set()
            open_interval = False
            exits = []
            for session, view, session_start, session_end in evaluated[identifier]:
                if session.ended_at is not None and start <= session.ended_at < end and session.ended_at <= as_of:
                    exits.append(session.ended_at)
                if session_end <= start or session_start >= end:
                    continue
                if historical_name is None:
                    historical_name = session.employee_name
                amount = duration_ms(max(start, session_start), min(end, session_end))
                # evaluate_session alone decides which accounting bucket applies;
                # splitting cannot turn part of a quarantined session trustworthy.
                if view.status == "REVIEW":
                    item.review_ms += amount
                    reasons.update(view.reason_codes or ["SESSION_REVIEW"])
                    pending.add(session.entry_event_id)
                elif view.status == "OPEN":
                    item.running_ms += amount
                else:
                    item.confirmed_ms += amount
                if session.status == "REVIEW" or session.reason_codes:
                    sequence_conflicts.add(session.entry_event_id)
                if session.ended_at is None:
                    open_interval = True
                    finalizable = False
            day_events = [event for event in employee_events[identifier] if start <= event.occurred_at < end]
            accepted_entries = [event for event in day_events if event.status == "ACCEPTED" and event.direction == "ENTRY"]
            item.visit_count = sum(not event.projection_reasons and event.id not in sequence_conflicts
                                   for event in accepted_entries)
            item.first_entry = attendance_iso(accepted_entries[0].occurred_at) if accepted_entries else None
            item.last_exit = attendance_iso(max(exits)) if exits else None
            if historical_name is None:
                historical_name = next((event.employee_name for event in day_events if event.employee_name is not None), None)
            if historical_name is not None:
                item.employee_name = historical_name
            item.pending_count = len(pending)
            item.coverage_reasons = sorted(reasons)
            item.status = "REVIEW" if reasons else "RUNNING" if not day_complete or open_interval else "CONFIRMED"
            if item.status == "REVIEW":
                finalizable = False
            rows.append(item)
            total = totals[identifier]
            total.confirmed_ms += item.confirmed_ms
            total.running_ms += item.running_ms
            total.review_ms += item.review_ms
            period_pending[identifier].update(pending)
    for identifier, total in totals.items():
        total.pending_count = len(period_pending[identifier])
    return AttendanceReport(
        from_date=from_date, to_date=to_date, utc_offset_minutes=settings.attendance_utc_offset_minutes,
        as_of=attendance_iso(as_of), finalizable=finalizable,
        unresolved_crossings=len({key for key in relevant_keys if key.startswith("event:")}),
        unreviewed_gaps=len({key for key in relevant_keys if key.startswith("gap:")}),
        rows=rows, totals=list(totals.values()),
    )


def format_hms(value: int) -> str:
    """Floor integer milliseconds without rounding up or wrapping at 24 hours."""
    seconds = max(0, value) // 1000
    return f"{seconds // 3600:02d}:{seconds // 60 % 60:02d}:{seconds % 60:02d}"


def report_csv(report: AttendanceReport) -> bytes:
    """Serialize this same materialized report; never recalculate export totals."""
    output = io.StringIO(newline="")
    writer = csv.writer(output, quoting=csv.QUOTE_ALL)
    writer.writerow(("date", "employee_id", "employee_name", "confirmed_hms", "running_hms", "review_hms",
                     "visits", "first_entry", "last_exit", "pending", "status", "report_finalizable"))
    for item in report.rows:
        name = item.employee_name
        if name.startswith(("=", "+", "-", "@", "\t", "\r")):
            name = "'" + name
        writer.writerow((item.date.isoformat(), item.employee_id, name, format_hms(item.confirmed_ms),
                         format_hms(item.running_ms), format_hms(item.review_ms), item.visit_count,
                         item.first_entry or "", item.last_exit or "", item.pending_count, item.status,
                         str(report.finalizable).lower()))
    return output.getvalue().encode("utf-8-sig")
