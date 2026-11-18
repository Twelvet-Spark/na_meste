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
from openpyxl import Workbook
from openpyxl.cell import WriteOnlyCell
from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE
from openpyxl.styles import Font
from openpyxl.utils import get_column_letter
from sqlalchemy.orm import Session

from app.attendance_types import CorrectionInput, RevokeInput, milliseconds
from app.config import settings
from app.database.database import immediate_session
from app.database.models import (
    Camera, DoorCrossing, Employee, EmployeeWorkplace, ManualPresenceInterval, PresenceExclusion,
    PresenceSegment, Workplace,
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


def _inputs(db: Session, lower: datetime, upper: datetime, employee_id: int | None = None, *,
            employee_ids: list[int] | None = None) -> dict[int, EmployeeInputs]:
    """Everything overlapping [lower, upper), grouped by employee; revoked corrections are ignored."""
    result: dict[int, EmployeeInputs] = defaultdict(EmployeeInputs)

    def scoped(query, model):
        if employee_ids is not None:
            return query.filter(model.employee_id.in_(employee_ids))
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


def build_report(db: Session, from_date: date, to_date: date, employee_id: int | None, as_of: datetime, *,
                 workplace_id: int | None = None, employee_status: str = "ALL") -> dict:
    days = (to_date - from_date).days + 1
    if not 1 <= days <= MAX_REPORT_DAYS:
        raise HTTPException(422, _RANGE_ERROR)
    if employee_id is not None and db.get(Employee, employee_id) is None:
        raise HTTPException(404, "Сотрудник не найден")
    if employee_status not in ("ALL", "ACTIVE", "INACTIVE"):
        raise HTTPException(422, "Недопустимый статус сотрудников")
    workplace = db.get(Workplace, workplace_id) if workplace_id is not None else None
    if workplace_id is not None and workplace is None:
        raise HTTPException(404, "Рабочее место не найдено")
    selected = db.query(Employee)
    if employee_id is not None:
        selected = selected.filter(Employee.id == employee_id)
    if employee_status != "ALL":
        selected = selected.filter(Employee.active.is_(employee_status == "ACTIVE"))
    if workplace_id is not None:
        selected = selected.filter(Employee.id.in_(db.query(EmployeeWorkplace.employee_id).filter(
            EmployeeWorkplace.workplace_id == workplace_id)))
    selected = selected.order_by(Employee.last_name, Employee.first_name, Employee.name, Employee.id).all()
    ids = [employee.id for employee in selected]
    places = defaultdict(list)
    if ids:
        for owner, place_id, name in db.query(EmployeeWorkplace.employee_id, Workplace.id, Workplace.name).join(
                Workplace, Workplace.id == EmployeeWorkplace.workplace_id).filter(
                    EmployeeWorkplace.employee_id.in_(ids)).order_by(Workplace.name, Workplace.id):
            places[owner].append({"id": place_id, "name": name})
    employees = [{"id": employee.id, "name": employee.name, "first_name": employee.first_name,
                  "last_name": employee.last_name, "active": employee.active,
                  "workplaces": places[employee.id]} for employee in selected]
    lower, upper = day_bounds(from_date)[0], day_bounds(to_date)[1]
    inputs = _inputs(db, lower, upper, employee_ids=ids) if ids else {}
    names = {employee.id: employee.name for employee in selected}
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
            "employees": employees,
            "filters": {"employee_id": employee_id, "workplace_id": workplace_id,
                        "workplace_name": workplace.name if workplace is not None else None,
                        "employee_status": employee_status},
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


def report_xlsx(report: dict) -> bytes:
    """A streamed employee × local-day matrix from the same counted report, never new accounting."""
    book = Workbook(write_only=True)
    sheet = book.create_sheet("Рабочее время")
    lower, upper = date.fromisoformat(report["from_date"]), date.fromisoformat(report["to_date"])
    dates = [lower + timedelta(days=offset) for offset in range((upper - lower).days + 1)]
    today = date.fromisoformat(report["as_of"][:10])
    bold = Font(bold=True)

    def cell(target, value, *, header=False):
        if isinstance(value, str):
            value = ILLEGAL_CHARACTERS_RE.sub("", value).replace("\ufffe", "").replace("\uffff", "")
        result = WriteOnlyCell(target, value=value)
        if isinstance(value, str):
            # Do not turn employee/workplace names into formulas or Excel error cells.
            result.data_type = "s"
        elif isinstance(value, timedelta):
            result.number_format = "[h]:mm"
        elif isinstance(value, date):
            result.number_format = "dd.mm.yyyy"
        if header:
            result.font = bold
        return result

    headers = ["Фамилия", "Имя", "Сотрудник", "Рабочие места", *dates, "Итого"]
    sheet.freeze_panes = "E2"
    for index in range(1, len(headers) + 1):
        sheet.column_dimensions[get_column_letter(index)].width = 26 if index <= 4 else 13
    sheet.print_options.horizontalCentered = True
    sheet.sheet_properties.pageSetUpPr.fitToPage = True
    sheet.page_setup.orientation = "landscape"
    sheet.page_setup.fitToWidth, sheet.page_setup.fitToHeight = 1, 0
    sheet.print_title_rows = "1:1"
    sheet.append([cell(sheet, value, header=True) for value in headers])
    counted = {(row["employee_id"], row["date"]): row["worked_ms"] for row in report["rows"]}
    totals = {row["employee_id"]: row["worked_ms"] for row in report["totals"]}
    for employee in report["employees"]:
        values = [employee["last_name"] or "", employee["first_name"] or "", employee["name"],
                  ", ".join(place["name"] for place in employee["workplaces"])]
        values.extend(timedelta(milliseconds=counted.get((employee["id"], day.isoformat()), 0))
                      if day <= today else None for day in dates)
        values.append(timedelta(milliseconds=totals.get(employee["id"], 0)))
        sheet.append([cell(sheet, value) for value in values])
    sheet.auto_filter.ref = f"A1:{get_column_letter(len(headers))}{len(report['employees']) + 1}"

    metadata = book.create_sheet("Параметры")
    metadata.column_dimensions["A"].width = 32
    metadata.column_dimensions["B"].width = 100
    info = [
        ("Период с", report["from_date"]), ("Период по", report["to_date"]), ("На момент", report["as_of"]),
        ("Смещение UTC, мин", report["utc_offset_minutes"]),
        ("Текущий перерыв, мин", report["live_gap_minutes"]), ("Итоговый перерыв, мин", report["day_gap_minutes"]),
        ("Рабочее место", report["filters"]["workplace_name"] or "Все"),
        ("Статус сотрудников", report["filters"]["employee_status"]),
        ("Сотрудник ID", report["filters"]["employee_id"] or "Все"),
        ("Правило отбора", "Рабочее место отбирает сотрудников; учитываются все их посчитанные часы."),
        ("Текущий день", "Предварительный расчёт «День идёт»; будущие даты пустые, отсутствие появлений — 0."),
        ("Суммы", "Числа — интервалы Excel [h]:mm; итог может превышать 24 часа."),
    ]
    for key, value in info:
        metadata.append([cell(metadata, key), cell(metadata, value)])
    output = io.BytesIO()
    book.save(output)
    return output.getvalue()


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


def insert_correction(writer: Session, kind: str, payload: CorrectionInput, now: datetime) -> dict:
    """Insert within the caller's transaction, including grouped journal corrections."""
    model = _CORRECTION_MODELS[kind]
    values = {"employee_id": payload.employee_id, "started_at": milliseconds(payload.started_at),
              "ended_at": milliseconds(payload.ended_at), "actor": payload.actor, "reason": payload.reason}
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


def create_correction(engine, kind: str, payload: CorrectionInput, now: datetime) -> dict:
    with immediate_session(engine) as writer:
        return insert_correction(writer, kind, payload, now)


def revoke_correction_row(writer: Session, kind: str, correction_id: int, payload: RevokeInput,
                          now: datetime, *, allow_revoked=False) -> dict:
    model = _CORRECTION_MODELS[kind]
    row = writer.get(model, correction_id)
    if row is None:
        raise HTTPException(404, "Корректировка не найдена")
    if row.revoked_at is not None:
        if not allow_revoked:
            raise HTTPException(409, "Корректировка уже отменена")
    else:
        row.revoked_at, row.revoked_by, row.revoke_reason = now, payload.actor, payload.reason
        writer.flush()
    employee = writer.get(Employee, row.employee_id)
    return _correction_view(kind, row, employee.name if employee is not None else None)


def revoke_correction(engine, kind: str, correction_id: int, payload: RevokeInput, now: datetime) -> dict:
    with immediate_session(engine) as writer:
        return revoke_correction_row(writer, kind, correction_id, payload, now)
