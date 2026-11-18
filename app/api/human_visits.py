"""Durable camera-local visits; image retention never removes journal metadata."""
import base64
import binascii
import hashlib
import json
import re
from datetime import datetime, timezone
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr
from sqlalchemy import and_, or_, case, func, select
from sqlalchemy.orm import Session, aliased
from starlette.concurrency import run_in_threadpool

from app.api.operator_security import require_operator_request
from app.database.database import get_db
from app.database.models import Camera, Employee, HumanVisit, HumanVisitEvent, HumanVisitIdentityRevision
from app.services.visit_identity_service import IdentityInput, RestoreIdentityInput, VisitIdentityService
from app.time_utils import utcnow

router = APIRouter(prefix="/visits", tags=["human-visits"])

# Explicit projections keep even the largest list independent of image BLOB size.
_METADATA = (
    "id", "camera_id", "tracking_generation", "track_id", "source_revision",
    "stream_id", "started_at", "last_seen_at", "ended_at", "end_reason",
    "status", "employee_id", "identity_state", "global_id", "image_expires_at",
    "images_purged_at",
)
_FACE_QUALITY = frozenset({
    "min_size", "det_score", "blur", "aligned", "unambiguous", "sha256",
    "frame_seq", "received_at",
})
_EVENT_FIELDS = frozenset({
    "version", "transition_seq", "reason", "employee_id", "previous_employee_id",
    "global_id", "score", "identity_state",
})

_IDENTITY_VERSION = select(
    HumanVisitIdentityRevision.visit_id,
    func.max(HumanVisitIdentityRevision.revision).label("revision"),
).group_by(HumanVisitIdentityRevision.visit_id).subquery()
_IDENTITY = aliased(HumanVisitIdentityRevision)
_AI_EMPLOYEE = aliased(Employee)
_EFFECTIVE_EMPLOYEE = case(
    (_IDENTITY.mode == "EMPLOYEE", _IDENTITY.employee_id),
    (_IDENTITY.mode == "UNKNOWN", None), else_=HumanVisit.employee_id,
)
_EFFECTIVE_STATE = case(
    (_IDENTITY.mode == "EMPLOYEE", "OPERATOR_CONFIRMED"),
    (_IDENTITY.mode == "UNKNOWN", "OPERATOR_UNKNOWN"), else_=HumanVisit.identity_state,
)


class EmployeeSample(BaseModel):
    model_config = ConfigDict(extra="forbid")
    employee_id: Annotated[StrictInt, Field(gt=0)]
    expected_revision: Annotated[StrictInt, Field(ge=0)]
    request_id: UUID
    face_sha256: Annotated[StrictStr, Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")]


def _iso(value):
    if value is None:
        return None
    aware = value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value
    return aware.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _utc_query(value, field):
    if value is None:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError("Timezone is required")
        return parsed.astimezone(timezone.utc).replace(tzinfo=None)
    except (ValueError, TypeError, OverflowError) as exc:
        raise HTTPException(422, {"code": "INVALID_VISIT_TIME", "field": field}) from exc


def _cursor(value):
    if value is None:
        return None
    try:
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,512}", value):
            raise ValueError("Invalid cursor encoding")
        raw = base64.b64decode(value + "=" * (-len(value) % 4), altchars=b"-_", validate=True)
        if base64.urlsafe_b64encode(raw).decode().rstrip("=") != value:
            raise ValueError("Noncanonical cursor")
        pair = json.loads(raw)
        if not isinstance(pair, list) or len(pair) != 2 or not all(isinstance(v, str) for v in pair):
            raise ValueError("Invalid cursor tuple")
        timestamp = _utc_query(pair[0], "cursor")
        identifier = str(UUID(pair[1]))
        if identifier != pair[1]:
            raise ValueError("Invalid cursor UUID")
        return timestamp, identifier
    except (ValueError, TypeError, UnicodeDecodeError, binascii.Error, HTTPException) as exc:
        raise HTTPException(422, {"code": "INVALID_VISIT_CURSOR"}) from exc


def _encode_cursor(row):
    raw = json.dumps([_iso(row.last_seen_at), row.id], separators=(",", ":")).encode()
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def _image_status(present, expires_at, purged_at, now):
    if purged_at is not None:
        return "PURGED"
    if expires_at <= now:
        return "EXPIRED"
    return "AVAILABLE" if present else "MISSING"


def _metadata(row, now):
    values = {key: getattr(row, key) for key in _METADATA}
    for key in ("started_at", "last_seen_at", "ended_at", "image_expires_at", "images_purged_at"):
        values[key] = _iso(values[key])
    values.update(
        camera_name=row.camera_name, employee_name=row.employee_name,
        employee_active=row.employee_active,
        identity_revision=row.identity_revision, identity_source=row.identity_source,
        ai_employee_id=row.ai_employee_id, ai_employee_name=row.ai_employee_name,
        ai_identity_state=row.ai_identity_state,
        duration_seconds=max(0., (row.last_seen_at - row.started_at).total_seconds()),
        photo_status=_image_status(row.has_photo, row.image_expires_at, row.images_purged_at, now),
        face_status=_image_status(row.has_face, row.image_expires_at, row.images_purged_at, now),
    )
    return values


def _query(db):
    return db.query(
        *(getattr(HumanVisit, name) for name in _METADATA if name not in ("employee_id", "identity_state")),
        _EFFECTIVE_EMPLOYEE.label("employee_id"), _EFFECTIVE_STATE.label("identity_state"),
        Camera.name.label("camera_name"), Employee.name.label("employee_name"),
        Employee.active.label("employee_active"),
        func.coalesce(_IDENTITY.revision, 0).label("identity_revision"),
        case((_IDENTITY.mode.in_(("EMPLOYEE", "UNKNOWN")), "OPERATOR"), else_="AI").label("identity_source"),
        HumanVisit.employee_id.label("ai_employee_id"), _AI_EMPLOYEE.name.label("ai_employee_name"),
        HumanVisit.identity_state.label("ai_identity_state"),
        HumanVisit.snapshot.isnot(None).label("has_photo"),
        HumanVisit.face_thumbnail.isnot(None).label("has_face"),
    ).join(Camera, Camera.id == HumanVisit.camera_id).outerjoin(
        _IDENTITY_VERSION, _IDENTITY_VERSION.c.visit_id == HumanVisit.id,
    ).outerjoin(_IDENTITY, and_(_IDENTITY.visit_id == HumanVisit.id,
                               _IDENTITY.revision == _IDENTITY_VERSION.c.revision)).outerjoin(
        Employee, Employee.id == _EFFECTIVE_EMPLOYEE,
    ).outerjoin(_AI_EMPLOYEE, _AI_EMPLOYEE.id == HumanVisit.employee_id)


def _visit_or_404(db, visit_id):
    visit = db.get(HumanVisit, str(visit_id))
    if visit is None:
        raise HTTPException(404, {"code": "VISIT_NOT_FOUND"})
    return visit


def _public_scalars(value, allowed):
    # Never let a future internal vector/crop field become an accidental API field.
    return {key: item for key, item in (value or {}).items()
            if key in allowed and (item is None or isinstance(item, (str, bool, int, float)))}


@router.get("")
def api_list_visits(
    camera_id: Annotated[int | None, Query(gt=0)] = None,
    employee_id: Annotated[int | None, Query(gt=0)] = None,
    unknown_only: bool = False,
    from_time: Annotated[str | None, Query(alias="from", max_length=64)] = None,
    to_time: Annotated[str | None, Query(alias="to", max_length=64)] = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 100,
    cursor: Annotated[str | None, Query(max_length=512)] = None,
    db: Session = Depends(get_db),
):
    since, until = _utc_query(from_time, "from"), _utc_query(to_time, "to")
    if since is not None and until is not None and since > until:
        raise HTTPException(422, {"code": "INVALID_VISIT_TIME_RANGE"})
    position = _cursor(cursor)
    query = _query(db)
    if camera_id is not None:
        query = query.filter(HumanVisit.camera_id == camera_id)
    if employee_id is not None:
        query = query.filter(_EFFECTIVE_EMPLOYEE == employee_id)
    if unknown_only:
        query = query.filter(_EFFECTIVE_STATE.in_(("UNKNOWN", "OPERATOR_UNKNOWN")), _EFFECTIVE_EMPLOYEE.is_(None))
    if since is not None:
        query = query.filter(HumanVisit.last_seen_at >= since)
    if until is not None:
        query = query.filter(HumanVisit.last_seen_at <= until)
    if position is not None:
        timestamp, identifier = position
        query = query.filter(or_(HumanVisit.last_seen_at < timestamp,
                                 and_(HumanVisit.last_seen_at == timestamp, HumanVisit.id < identifier)))
    rows = query.order_by(HumanVisit.last_seen_at.desc(), HumanVisit.id.desc()).limit(limit + 1).all()
    now = utcnow()
    return {"items": [_metadata(row, now) for row in rows[:limit]],
            "next_cursor": _encode_cursor(rows[limit - 1]) if len(rows) > limit else None}


@router.get("/{visit_id}")
def api_get_visit(visit_id: UUID, request: Request, db: Session = Depends(get_db)):
    visit = _visit_or_404(db, visit_id)
    row = _query(db).filter(HumanVisit.id == str(visit_id)).one()
    result = _metadata(row, utcnow())
    service = getattr(request.app.state, "visit_sample_service", None)
    eligible, reason = service.eligibility(visit) if service is not None else (False, "FACE_MODEL_UNAVAILABLE")
    result.update(face_sha256=(hashlib.sha256(visit.face_thumbnail).hexdigest()
                               if result["face_status"] == "AVAILABLE" and visit.face_thumbnail else None),
                  face_quality=_public_scalars(visit.face_quality, _FACE_QUALITY),
                  employee_sample_eligible=eligible, employee_sample_reason=reason)
    return result


def _image(db, visit_id, field):
    row = db.query(getattr(HumanVisit, field), HumanVisit.image_expires_at,
                   HumanVisit.images_purged_at).filter(HumanVisit.id == str(visit_id)).first()
    if row is None:
        raise HTTPException(404, {"code": "VISIT_NOT_FOUND"})
    image, expires_at, purged_at = row
    if purged_at is not None or expires_at <= utcnow():
        raise HTTPException(410, {"code": "VISIT_IMAGE_EXPIRED"}, headers={"Cache-Control": "no-store"})
    if not image:
        raise HTTPException(404, {"code": "VISIT_IMAGE_MISSING"}, headers={"Cache-Control": "no-store"})
    return Response(image, media_type="image/jpeg", headers={
        "Cache-Control": "no-store", "ETag": f'"{hashlib.sha256(image).hexdigest()}"',
    })


@router.get("/{visit_id}/photo")
def api_visit_photo(visit_id: UUID, db: Session = Depends(get_db)):
    return _image(db, visit_id, "snapshot")


@router.get("/{visit_id}/face")
def api_visit_face(visit_id: UUID, db: Session = Depends(get_db)):
    return _image(db, visit_id, "face_thumbnail")


@router.get("/{visit_id}/events")
def api_visit_events(visit_id: UUID, limit: Annotated[int, Query(ge=1, le=1000)] = 200,
                     db: Session = Depends(get_db)):
    if db.query(HumanVisit.id).filter(HumanVisit.id == str(visit_id)).first() is None:
        raise HTTPException(404, {"code": "VISIT_NOT_FOUND"})
    rows = db.query(HumanVisitEvent.id, HumanVisitEvent.type, HumanVisitEvent.occurred_at,
                    HumanVisitEvent.payload).filter(HumanVisitEvent.visit_id == str(visit_id)).order_by(
                        HumanVisitEvent.occurred_at.desc(), HumanVisitEvent.id.desc()).limit(limit).all()
    return [{"id": row.id, "type": row.type, "occurred_at": _iso(row.occurred_at),
             "payload": _public_scalars(row.payload, _EVENT_FIELDS)} for row in rows]


@router.post("/{visit_id}/employee-sample", dependencies=[Depends(require_operator_request)])
async def api_append_employee_sample(visit_id: UUID, payload: EmployeeSample, request: Request):
    service = getattr(request.app.state, "visit_sample_service", None)
    if service is None:
        raise HTTPException(503, {"code": "FACE_MODEL_UNAVAILABLE"})
    return await run_in_threadpool(service.append, str(visit_id), employee_id=payload.employee_id,
                                   expected_revision=payload.expected_revision,
                                   request_id=payload.request_id, face_sha256=payload.face_sha256)


@router.post("/{visit_id}/identity", dependencies=[Depends(require_operator_request)])
def api_set_identity(visit_id: UUID, payload: IdentityInput, db: Session = Depends(get_db)):
    return VisitIdentityService(db.get_bind()).set_identity(visit_id, payload)


@router.post("/{visit_id}/identity/restore", dependencies=[Depends(require_operator_request)])
def api_restore_identity(visit_id: UUID, payload: RestoreIdentityInput, db: Session = Depends(get_db)):
    return VisitIdentityService(db.get_bind()).restore(visit_id, payload)


@router.get("/{visit_id}/identity/history")
def api_identity_history(visit_id: UUID, db: Session = Depends(get_db)):
    return VisitIdentityService(db.get_bind()).history(visit_id)
