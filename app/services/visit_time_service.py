"""Previewed, atomic work-time corrections linked to journal observations."""
import hashlib
import json
from dataclasses import asdict, replace
from datetime import datetime, timedelta, timezone
from uuid import UUID, uuid4

from fastapi import HTTPException
from pydantic import Field, StrictInt, StrictStr, field_validator, model_validator
from sqlalchemy import func
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.attendance_types import CorrectionInput, HumanInput, RevokeInput, milliseconds
from app.config import settings
from app.database.database import immediate_session, read_snapshot
from app.database.models import Employee, HumanVisit, OperatorRequest
from app.services import attendance_reports as reports
from app.services.employee_service import committed_request
from app.services.visit_identity_service import effective_identity
from app.services.work_time import Span
from app.time_utils import attendance_iso, day_bounds, local_date, utcnow

APPLY = "visit.time.apply"
UNDO = "visit.time.undo"


class VisitTimeInput(HumanInput):
    expected_identity_revision: StrictInt = Field(ge=0)
    employee_id: StrictInt = Field(gt=0)
    started_at: datetime
    ended_at: datetime
    exclude_employee_id: StrictInt | None = Field(default=None, gt=0)

    @field_validator("started_at", "ended_at")
    @classmethod
    def aware_timestamp(cls, value):
        return CorrectionInput.aware_timestamp(value)

    @model_validator(mode="after")
    def valid_interval(self):
        if self.ended_at <= self.started_at:
            raise ValueError("Конец интервала должен быть позже начала")
        if self.exclude_employee_id == self.employee_id:
            raise ValueError("Нельзя добавить время и исключить его у того же сотрудника")
        # Validate both local day bounds before any report calculations or writes.
        day_bounds(local_date(milliseconds(self.started_at)))
        day_bounds(local_date(milliseconds(self.ended_at) - timedelta(milliseconds=1)))
        return self


class VisitTimeApplyInput(VisitTimeInput):
    preview_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")


class VisitTimeUndoInput(HumanInput):
    operation_id: UUID


def _error(status, code):
    return HTTPException(status, {"code": code})


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _digest(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _selection(payload):
    return dict(employee_id=payload.employee_id, exclude_employee_id=payload.exclude_employee_id,
                expected_identity_revision=payload.expected_identity_revision,
                started_at=attendance_iso(milliseconds(payload.started_at)),
                ended_at=attendance_iso(milliseconds(payload.ended_at)))


def _days(start, end):
    day = local_date(start)
    final_day = local_date(end - timedelta(milliseconds=1))
    while True:
        lower, upper = day_bounds(day)
        yield day, lower, upper
        if day == final_day:
            break
        day += timedelta(days=1)


class VisitTimeService:
    def __init__(self, engine, presence, *, now=utcnow):
        self.engine, self.presence, self.now = engine, presence, now

    def _flush(self):
        if self.presence is None:
            raise _error(503, "PRESENCE_UNAVAILABLE")
        self.presence.flush()
        if self.presence.snapshot().get("last_error"):
            raise _error(503, "PRESENCE_FLUSH_FAILED")

    def _preview(self, db, visit_id, payload, now, *, applying=False):
        visit = db.get(HumanVisit, visit_id)
        if visit is None:
            raise _error(404, "VISIT_NOT_FOUND")
        identity = effective_identity(db, visit)
        if identity["identity_revision"] != payload.expected_identity_revision:
            raise _error(409, "IDENTITY_REVISION_CONFLICT")
        invalid = 409 if applying else 422
        if identity["employee_id"] != payload.employee_id:
            raise _error(invalid, "VISIT_TIME_IDENTITY_CHANGED")
        start, end = milliseconds(payload.started_at), milliseconds(payload.ended_at)
        if start < milliseconds(visit.started_at) or end > milliseconds(visit.last_seen_at):
            raise _error(invalid, "VISIT_TIME_OUTSIDE_OBSERVATION")
        if end > now:
            raise _error(invalid, "VISIT_TIME_FUTURE")
        ids = [payload.employee_id]
        if payload.exclude_employee_id is not None:
            ids.append(payload.exclude_employee_id)
        names = reports._names(db, ids)
        if len(names) != len(ids):
            raise _error(404, "EMPLOYEE_NOT_FOUND")
        days = list(_days(start, end))
        inputs = reports._inputs(db, days[0][1], days[-1][2], employee_ids=ids)
        rows = []
        for day, lower, upper in days:
            for employee_id in ids:
                data = inputs[employee_id]
                before = reports._day(data, lower, upper, now)
                after_data = replace(
                    data, manual=([*data.manual, Span(start, end)] if employee_id == payload.employee_id else data.manual),
                    exclusions=([*data.exclusions, Span(start, end)]
                                if employee_id == payload.exclude_employee_id else data.exclusions))
                after = reports._day(after_data, lower, upper, now)
                before_ms, after_ms = reports._ms(before.worked), reports._ms(after.worked)
                rows.append(dict(employee_id=employee_id, employee_name=names[employee_id], date=day.isoformat(),
                                 method=before.method, before_ms=before_ms, after_ms=after_ms,
                                 delta_ms=after_ms-before_ms))
        evidence = {str(employee_id): {kind: sorted(values, key=_json) for kind, values in asdict(data).items()}
                    for employee_id, data in inputs.items()}
        # Extending a live visit beyond the frozen selection does not change that
        # selection. Shrinking its containing bounds was rejected above.
        fingerprint = dict(visit_id=visit_id, selection=_selection(payload), inputs=evidence,
                           methods=[(row["date"], row["method"]) for row in rows],
                           live_gap_minutes=settings.attendance_live_gap_minutes,
                           day_gap_minutes=settings.attendance_day_gap_minutes,
                           utc_offset_minutes=settings.attendance_utc_offset_minutes)
        warnings = ["MANUAL_INTERVAL_INCLUDES_GAPS", "TOTALS_INCLUDE_BRIDGING"]
        if payload.exclude_employee_id is not None:
            warnings += ["EXCLUSION_ALL_CAMERAS", "EXCLUSION_KEEPS_MANUAL_TIME"]
        return dict(visit_id=visit_id, selection=_selection(payload), preview_sha256=_digest(fingerprint),
                    rows=rows, warnings=warnings)

    def preview(self, visit_id, payload: VisitTimeInput) -> dict:
        try:
            self._flush()
            return read_snapshot(self.engine, lambda db: self._preview(
                db, str(visit_id), payload, milliseconds(self.now())))
        except SQLAlchemyError as exc:
            raise _error(503, "VISIT_TIME_UNAVAILABLE") from exc

    def apply(self, visit_id, payload: VisitTimeApplyInput) -> dict:
        visit_id = str(visit_id)
        digest = _digest({"visit_id": visit_id, **payload.model_dump(mode="json")})
        try:
            with Session(self.engine) as db:
                replay = committed_request(db, payload.request_id, digest, operation=APPLY)
                if replay is not None:
                    return replay
            self._flush()
            with immediate_session(self.engine) as db:
                replay = committed_request(db, payload.request_id, digest, operation=APPLY)
                if replay is not None:
                    return replay
                now = milliseconds(self.now())
                preview = self._preview(db, visit_id, payload, now, applying=True)
                if preview["preview_sha256"] != payload.preview_sha256:
                    raise _error(409, "VISIT_TIME_PREVIEW_CHANGED")
                start, end = milliseconds(payload.started_at), milliseconds(payload.ended_at)
                targets = [("manual", payload.employee_id)]
                if payload.exclude_employee_id is not None:
                    targets.append(("exclusion", payload.exclude_employee_id))
                created = []
                for _, lower, upper in _days(start, end):
                    for kind, employee_id in targets:
                        correction = CorrectionInput(
                            request_id=uuid4(), actor=payload.actor, reason=payload.reason, employee_id=employee_id,
                            started_at=max(start, lower).replace(tzinfo=timezone.utc),
                            ended_at=min(end, upper).replace(tzinfo=timezone.utc))
                        row = reports.insert_correction(db, kind, correction, now)
                        row["correction_request_id"] = str(correction.request_id)
                        created.append(row)
                result = dict(operation_id=str(payload.request_id), visit_id=visit_id,
                              selection=preview["selection"], created=created, actor=payload.actor,
                              reason=payload.reason, created_at=attendance_iso(now))
                db.add(OperatorRequest(id=str(payload.request_id), operation=APPLY, payload_sha256=digest,
                                       created_at=now, result=result))
                return result
        except SQLAlchemyError as exc:
            raise _error(503, "VISIT_TIME_UNAVAILABLE") from exc

    @staticmethod
    def _owned_row(db, item):
        kind = item["kind"].lower()
        row = db.get(reports._CORRECTION_MODELS[kind], item["id"])
        if row is None or row.request_id != item["correction_request_id"]:
            raise _error(409, "VISIT_TIME_CORRECTION_CHANGED")
        return kind, row

    def undo(self, visit_id, payload: VisitTimeUndoInput) -> dict:
        visit_id = str(visit_id)
        digest = _digest({"visit_id": visit_id, **payload.model_dump(mode="json")})
        try:
            with immediate_session(self.engine) as db:
                replay = committed_request(db, payload.request_id, digest, operation=UNDO)
                if replay is not None:
                    return replay
                proof = db.get(OperatorRequest, str(payload.operation_id))
                if proof is None or proof.operation != APPLY or proof.result.get("visit_id") != visit_id:
                    raise _error(404, "VISIT_TIME_OPERATION_NOT_FOUND")
                now = milliseconds(self.now())
                revoked = []
                for item in proof.result["created"]:
                    kind, row = self._owned_row(db, item)
                    result = reports.revoke_correction_row(
                        db, kind, row.id, RevokeInput(actor=payload.actor, reason=payload.reason), now, allow_revoked=True)
                    result["correction_request_id"] = item["correction_request_id"]
                    revoked.append(result)
                result = dict(operation_id=str(payload.operation_id), visit_id=visit_id, created=revoked,
                              actor=payload.actor, reason=payload.reason, revoked_at=attendance_iso(now))
                db.add(OperatorRequest(id=str(payload.request_id), operation=UNDO, payload_sha256=digest,
                                       created_at=now, result=result))
                return result
        except SQLAlchemyError as exc:
            raise _error(503, "VISIT_TIME_UNAVAILABLE") from exc

    def history(self, visit_id) -> dict:
        try:
            def read_history(db):
                if not db.query(HumanVisit.id).filter_by(id=str(visit_id)).first():
                    raise _error(404, "VISIT_NOT_FOUND")
                operations = db.query(OperatorRequest).filter(
                    OperatorRequest.operation == APPLY,
                    func.json_extract(OperatorRequest.result, "$.visit_id") == str(visit_id),
                ).order_by(OperatorRequest.created_at.desc(), OperatorRequest.id.desc()).all()
                current = {}
                for kind, model in reports._CORRECTION_MODELS.items():
                    ids = [item["id"] for operation in operations for item in operation.result["created"]
                           if item["kind"].lower() == kind]
                    if ids:
                        current.update({(kind, row.id): row for row in db.query(model).filter(model.id.in_(ids))})
                items = []
                for operation in operations:
                    result = {**operation.result, "created": []}
                    for item in operation.result["created"]:
                        kind = item["kind"].lower()
                        row = current.get((kind, item["id"]))
                        if row is None or row.request_id != item["correction_request_id"]:
                            result["created"].append({**item, "unavailable": True})
                        else:
                            result["created"].append({**item, **reports._correction_view(kind, row, item["employee_name"])})
                    items.append(result)
                return {"items": items}
            return read_snapshot(self.engine, read_history)
        except SQLAlchemyError as exc:
            raise _error(503, "VISIT_TIME_UNAVAILABLE") from exc
