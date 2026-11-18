"""Append-only operator identities; AI journal fields remain owned by the worker."""
import hashlib
import json
from typing import Literal

from fastapi import HTTPException
from pydantic import Field, StrictInt, model_validator
from sqlalchemy import func
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.attendance_types import HumanInput
from app.database.database import immediate_session
from app.database.models import Employee, HumanVisit, HumanVisitIdentityRevision, OperatorRequest
from app.services.employee_service import committed_request
from app.time_utils import attendance_iso, utcnow


class IdentityInput(HumanInput):
    expected_revision: StrictInt = Field(ge=0)
    mode: Literal["EMPLOYEE", "UNKNOWN", "AI"]
    employee_id: StrictInt | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def employee_for_mode(self):
        if (self.mode == "EMPLOYEE") != (self.employee_id is not None):
            raise ValueError("Выберите сотрудника только для режима EMPLOYEE")
        return self


class RestoreIdentityInput(HumanInput):
    expected_revision: StrictInt = Field(ge=0)
    restore_revision: StrictInt = Field(ge=0)


def effective_identity(db, visit):
    revision = db.query(HumanVisitIdentityRevision).filter_by(visit_id=visit.id).order_by(
        HumanVisitIdentityRevision.revision.desc()).first()
    mode = revision.mode if revision else "AI"
    employee_id = visit.employee_id if mode == "AI" else revision.employee_id
    employee = db.get(Employee, employee_id) if employee_id is not None else None
    return dict(identity_revision=revision.revision if revision else 0,
                identity_source="AI" if mode == "AI" else "OPERATOR",
                identity_state=(visit.identity_state if mode == "AI" else
                                "OPERATOR_CONFIRMED" if mode == "EMPLOYEE" else "OPERATOR_UNKNOWN"),
                employee_id=employee_id, employee_name=employee.name if employee else None,
                employee_active=employee.active if employee else None,
                ai_employee_id=visit.employee_id, ai_identity_state=visit.identity_state)


class VisitIdentityService:
    def __init__(self, engine, *, now=utcnow):
        self.engine, self.now = engine, now

    def set_identity(self, visit_id, payload: IdentityInput) -> dict:
        return self._write(str(visit_id), payload, "visit.identity.set")

    def restore(self, visit_id, payload: RestoreIdentityInput) -> dict:
        return self._write(str(visit_id), payload, "visit.identity.restore")

    def _write(self, visit_id, payload, operation):
        digest = hashlib.sha256(json.dumps(
            {"visit_id": visit_id, **payload.model_dump(mode="json")},
            sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        try:
            with immediate_session(self.engine) as db:
                replay = committed_request(db, payload.request_id, digest, operation=operation)
                if replay is not None:
                    return replay
                visit = db.get(HumanVisit, visit_id)
                if visit is None:
                    raise HTTPException(404, {"code": "VISIT_NOT_FOUND"})
                current = db.query(func.max(HumanVisitIdentityRevision.revision)).filter_by(visit_id=visit_id).scalar() or 0
                if current != payload.expected_revision:
                    raise HTTPException(409, {"code": "IDENTITY_REVISION_CONFLICT", "identity_revision": current})
                if isinstance(payload, RestoreIdentityInput):
                    previous = (db.query(HumanVisitIdentityRevision).filter_by(
                        visit_id=visit_id, revision=payload.restore_revision).first()
                        if payload.restore_revision else None)
                    if payload.restore_revision and previous is None:
                        raise HTTPException(404, {"code": "IDENTITY_REVISION_NOT_FOUND"})
                    mode, employee_id = (previous.mode, previous.employee_id) if previous else ("AI", None)
                else:
                    mode, employee_id = payload.mode, payload.employee_id
                if employee_id is not None and db.get(Employee, employee_id) is None:
                    raise HTTPException(404, {"code": "EMPLOYEE_NOT_FOUND"})
                db.add(HumanVisitIdentityRevision(
                    visit_id=visit_id, revision=current + 1, mode=mode, employee_id=employee_id,
                    actor=payload.actor, reason=payload.reason, created_at=self.now(), request_id=str(payload.request_id)))
                db.flush()
                result = {"visit_id": visit_id, **effective_identity(db, visit)}
                db.add(OperatorRequest(id=str(payload.request_id), operation=operation,
                                       payload_sha256=digest, result=result))
                return result
        except SQLAlchemyError as exc:
            raise HTTPException(503, {"code": "IDENTITY_WRITE_UNAVAILABLE"}) from exc

    def history(self, visit_id) -> dict:
        with Session(self.engine) as db:
            if not db.query(HumanVisit.id).filter_by(id=str(visit_id)).first():
                raise HTTPException(404, {"code": "VISIT_NOT_FOUND"})
            rows = db.query(HumanVisitIdentityRevision, Employee.name).outerjoin(
                Employee, Employee.id == HumanVisitIdentityRevision.employee_id).filter(
                HumanVisitIdentityRevision.visit_id == str(visit_id)).order_by(
                HumanVisitIdentityRevision.revision.desc()).all()
            return {"items": [dict(revision=row.revision, mode=row.mode, employee_id=row.employee_id,
                                   employee_name=name, actor=row.actor, reason=row.reason,
                                   created_at=attendance_iso(row.created_at), request_id=row.request_id)
                              for row, name in rows]}
