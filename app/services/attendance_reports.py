"""Work-time views built from one caller-owned SQLite read snapshot, plus audited corrections.

Presence segments, door crossings and corrections are loaded once and fed to the pure
`compute_day`; nothing here decides what counts as work time.
"""
import csv
import io
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.attendance_types import CorrectionInput, RevokeInput, milliseconds
from app.config import settings
from app.database.database import immediate_session
from app.database.models import (
    Camera, DoorCrossing, Employee, ManualPresenceInterval, PresenceExclusion, PresenceSegment,
)
from app.services.work_time import Crossing, DayResult, Mark, Span, compute_day, subtract
from app.time_utils import attendance_iso, day_bounds, local_date

MAX_REPORT_DAYS = 366
CORRECTION_LIMIT = 500
_CORRECTION_MODELS = {"manual": ManualPresenceInterval, "exclusion": PresenceExclusion}
_RANGE_ERROR = "Укажите период от 1 до 366 дней"


@dataclass
class EmployeeInputs:
    segments: list[Mark] = field(default_factory=list)
    manual: list[Span] = field(default_factory=list)
    crossings: list[Crossing] = field(default_factory=list)
    exclusions: list[Span] = field(default_factory=list)


def _inputs(db: Session, lower: datetime, upper: datetime, employee_id: int | None = None) -> dict[int, EmployeeInputs]:
    """Everything overlapping [lower, upper), grouped by employee; revoked corrections are ignored."""
    result: dict[int, EmployeeInputs] = defaultdict(EmployeeInputs)

    def scoped(query, model):
        return query.filter(model.employee_id == employee_id) if employee_id is not None else query

    for row in scoped(db.query(PresenceSegment), PresenceSegment).filter(
            PresenceSegment.started_at < upper, PresenceSegment.last_seen_at >= lower):
        result[row.employee_id].segments.append(
            Mark(row.started_at, row.last_seen_at, bool(row.face_seen), "SEGMENT", row.camera_id))
    for row in scoped(db.query(DoorCrossing), DoorCrossing).filter(
            DoorCrossing.occurred_at >= lower, DoorCrossing.occurred_at < upper):
        result[row.employee_id].crossings.append(Crossing(row.occurred_at, row.direction, row.camera_id))
    for model, target in ((ManualPresenceInterval, "manual"), (PresenceExclusion, "exclusions")):
        for row in scoped(db.query(model), model).filter(
                model.revoked_at.is_(None), model.started_at < upper, model.ended_at > lower):
            getattr(result[row.employee_id], target).append(Span(row.started_at, row.ended_at))
    return result


def _day(data: EmployeeInputs, lower: datetime, upper: datetime, as_of: datetime) -> DayResult:
    return compute_day(segments=data.segments, manual=data.manual, crossings=data.crossings,
                       exclusions=data.exclusions, lower=lower, upper=upper, as_of=as_of)


def _ms(value: timedelta) -> int:
    return value // timedelta(milliseconds=1)


def _names(db: Session, ids) -> dict[int, str]:
    ids = list(ids)
    if not ids:
        return {}
    return {row.id: row.name for row in db.query(Employee.id, Employee.name).filter(Employee.id.in_(ids))}


def _source(mark: Mark | None) -> str | None:
    if mark is None:
        return None
    if mark.source == "SEGMENT":
        return "FACE" if mark.face_seen else "REID"
    return mark.source  # "MANUAL" or "DOOR"


def active_view(db: Session, as_of: datetime) -> dict:
    """Employees on site now: a live session of today ending within the live gap of `as_of`."""
    lower, upper = day_bounds(local_date(as_of))
    inputs = _inputs(db, lower, upper)
    names = _names(db, inputs)
    cameras = {row.id: row.name for row in db.query(Camera.id, Camera.name)}
    threshold = as_of - timedelta(minutes=settings.attendance_live_gap_minutes)
    items = []
    for employee_id, data in inputs.items():
        result = _day(data, lower, upper, as_of)
        if not result.sessions or result.sessions[-1].end < threshold:
            continue
        session, mark = result.sessions[-1], result.last_mark
        camera_id = mark.camera_id if mark is not None else None
        items.append({
            "employee_id": employee_id, "employee_name": names.get(employee_id, ""),
            "since": attendance_iso(session.start), "last_seen_at": attendance_iso(session.end),
            "last_camera_id": camera_id, "last_camera_name": cameras.get(camera_id),
            "today_ms": _ms(result.worked), "source": _source(mark),
        })
    items.sort(key=lambda item: item["employee_name"])
    return {"as_of": attendance_iso(as_of), "items": items}


def build_report(db: Session, from_date: date, to_date: date, employee_id: int | None, as_of: datetime) -> dict:
    days = (to_date - from_date).days + 1
    if not 1 <= days <= MAX_REPORT_DAYS:
        raise HTTPException(422, _RANGE_ERROR)
    if employee_id is not None and db.get(Employee, employee_id) is None:
        raise HTTPException(404, "Сотрудник не найден")
    lower, upper = day_bounds(from_date)[0], day_bounds(to_date)[1]
    inputs = _inputs(db, lower, upper, employee_id)
    names = _names(db, inputs)
    rows, totals = [], {}
    for offset in range(days):
        day = from_date + timedelta(days=offset)
        day_lower, day_upper = day_bounds(day)
        if day_lower > as_of:
            break
        for current, data in inputs.items():
            overlapping = EmployeeInputs(
                [m for m in data.segments if m.start < day_upper and m.end >= day_lower],
                [s for s in data.manual if s.start < day_upper and s.end > day_lower],
                [c for c in data.crossings if day_lower <= c.at < day_upper],
                [s for s in data.exclusions if s.start < day_upper and s.end > day_lower])
            result = _day(overlapping, day_lower, day_upper, as_of)
            if result.first is None:
                continue
            rows.append({
                "date": day.isoformat(), "employee_id": current, "employee_name": names.get(current, ""),
                "worked_ms": _ms(result.worked), "first_seen_at": attendance_iso(result.first),
                "last_seen_at": attendance_iso(result.last), "sessions": len(result.sessions),
                "method": result.method, "has_manual": bool(overlapping.manual),
                "face_seen": any(m.face_seen for m in overlapping.segments),
            })
            total = totals.setdefault(current, {"employee_id": current, "employee_name": names.get(current, ""),
                                                "worked_ms": 0, "days": 0})
            total["worked_ms"] += _ms(result.worked)
            total["days"] += 1
    rows.sort(key=lambda row: (row["date"], row["employee_name"], row["employee_id"]))
    return {"from_date": from_date.isoformat(), "to_date": to_date.isoformat(),
            "utc_offset_minutes": settings.attendance_utc_offset_minutes, "as_of": attendance_iso(as_of),
            "live_gap_minutes": settings.attendance_live_gap_minutes,
            "day_gap_minutes": settings.attendance_day_gap_minutes,
            "rows": rows, "totals": sorted(totals.values(), key=lambda t: (t["employee_name"], t["employee_id"]))}


def _hours_minutes(milliseconds_: int) -> tuple[str, int]:
    minutes = max(0, milliseconds_) // 60000
    return f"{minutes // 60}:{minutes % 60:02d}", minutes


def _csv_text(value: str) -> str:
    # Spreadsheet formula injection: a leading operator character is neutralized.
    return "'" + value if value.startswith(("=", "+", "-", "@", "\t", "\r")) else value


def report_csv(report: dict) -> bytes:
    """Serialize the same materialized report; never recalculate export totals."""
    output = io.StringIO(newline="")
    writer = csv.writer(output, delimiter=";", quoting=csv.QUOTE_ALL)
    writer.writerow(("Дата", "Сотрудник", "Отработано (ч:мм)", "Минут", "Первое появление",
                     "Последнее появление", "Сессий", "Расчёт", "Примечания"))
    for row in report["rows"]:
        hours, minutes = _hours_minutes(row["worked_ms"])
        notes = []
        if row["has_manual"]:
            notes.append("вручную")
        elif not row["face_seen"]:
            notes.append("только по внешности")
        writer.writerow((row["date"], _csv_text(row["employee_name"]), hours, minutes,
                         row["first_seen_at"][11:16], row["last_seen_at"][11:16], row["sessions"],
                         "Итог дня" if row["method"] == "FINAL" else "День идёт", ", ".join(notes)))
    return output.getvalue().encode("utf-8-sig")


def _correction_view(kind: str, row, employee_name: str | None) -> dict:
    return {"kind": kind.upper(), "id": row.id, "employee_id": row.employee_id, "employee_name": employee_name,
            "started_at": attendance_iso(row.started_at), "ended_at": attendance_iso(row.ended_at),
            "actor": row.actor, "reason": row.reason, "created_at": attendance_iso(row.created_at),
            "revoked_at": attendance_iso(row.revoked_at), "revoked_by": row.revoked_by,
            "revoke_reason": row.revoke_reason}


def timeline(db: Session, employee_id: int, day: date, as_of: datetime) -> dict:
    employee = db.get(Employee, employee_id)
    if employee is None:
        raise HTTPException(404, "Сотрудник не найден")
    lower, upper = day_bounds(day)
    cameras = {row.id: row.name for row in db.query(Camera.id, Camera.name)}
    data = _inputs(db, lower, upper, employee_id).get(employee_id, EmployeeInputs())
    result = _day(data, lower, upper, as_of)
    cuts = data.exclusions
    segments = [{"camera_id": m.camera_id, "camera_name": cameras.get(m.camera_id),
                 "started_at": attendance_iso(m.start), "last_seen_at": attendance_iso(m.end),
                 "face_seen": m.face_seen, "excluded": not subtract([m], cuts)}
                for m in sorted(data.segments, key=lambda m: (m.start, m.end))]
    crossings = [{"camera_id": c.camera_id, "camera_name": cameras.get(c.camera_id), "direction": c.direction,
                  "occurred_at": attendance_iso(c.at),
                  "excluded": any(cut.start <= c.at < cut.end for cut in cuts)}
                 for c in sorted(data.crossings, key=lambda c: c.at)]
    corrections = {"manual": [], "exclusion": []}
    for kind, model in _CORRECTION_MODELS.items():
        rows = db.query(model).filter(
            model.employee_id == employee_id, model.started_at < upper, model.ended_at > lower).order_by(
                model.started_at, model.id)
        corrections[kind] = [_correction_view(kind, row, employee.name) for row in rows]
    return {"date": day.isoformat(), "method": result.method, "worked_ms": _ms(result.worked),
            "sessions": [{"started_at": attendance_iso(s.start), "ended_at": attendance_iso(s.end)}
                         for s in result.sessions],
            "segments": segments, "crossings": crossings, "manual": corrections["manual"],
            "exclusions": corrections["exclusion"],
            "door_gaps": [{"started_at": attendance_iso(g.start), "ended_at": attendance_iso(g.end)}
                          for g in result.door_gaps]}


def list_corrections(db: Session, from_date: date | None, to_date: date | None, employee_id: int | None) -> dict:
    if from_date is not None and to_date is not None and to_date < from_date:
        raise HTTPException(422, _RANGE_ERROR)
    items = []
    for kind, model in _CORRECTION_MODELS.items():
        query = db.query(model)
        if employee_id is not None:
            query = query.filter(model.employee_id == employee_id)
        if from_date is not None:
            query = query.filter(model.ended_at > day_bounds(from_date)[0])
        if to_date is not None:
            query = query.filter(model.started_at < day_bounds(to_date)[1])
        items.extend((kind, row) for row in query.order_by(model.started_at.desc(), model.id.desc()).limit(
            CORRECTION_LIMIT))
    names = _names(db, {row.employee_id for _kind, row in items})
    items.sort(key=lambda pair: (pair[1].started_at, pair[1].id), reverse=True)
    return {"items": [_correction_view(kind, row, names.get(row.employee_id))
                      for kind, row in items[:CORRECTION_LIMIT]]}


def create_correction(engine, kind: str, payload: CorrectionInput, now: datetime) -> dict:
    model = _CORRECTION_MODELS[kind]
    values = {"employee_id": payload.employee_id, "started_at": milliseconds(payload.started_at),
              "ended_at": milliseconds(payload.ended_at), "actor": payload.actor, "reason": payload.reason}
    with immediate_session(engine) as writer:
        employee = writer.get(Employee, payload.employee_id)
        if employee is None:
            raise HTTPException(404, "Сотрудник не найден")
        existing = writer.query(model).filter_by(request_id=str(payload.request_id)).one_or_none()
        if existing is not None:
            if any(getattr(existing, name) != value for name, value in values.items()):
                raise HTTPException(409, "request_id уже использован с другими данными")
            return _correction_view(kind, existing, employee.name)
        if values["ended_at"] > now:
            raise HTTPException(422, "Время в будущем недопустимо")
        row = model(request_id=str(payload.request_id), created_at=now, **values)
        writer.add(row)
        writer.flush()
        return _correction_view(kind, row, employee.name)


def revoke_correction(engine, kind: str, correction_id: int, payload: RevokeInput, now: datetime) -> dict:
    model = _CORRECTION_MODELS[kind]
    with immediate_session(engine) as writer:
        row = writer.get(model, correction_id)
        if row is None:
            raise HTTPException(404, "Корректировка не найдена")
        if row.revoked_at is not None:
            raise HTTPException(409, "Корректировка уже отменена")
        row.revoked_at, row.revoked_by, row.revoke_reason = now, payload.actor, payload.reason
        writer.flush()
        employee = writer.get(Employee, row.employee_id)
        return _correction_view(kind, row, employee.name if employee is not None else None)
