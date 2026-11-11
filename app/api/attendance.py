"""Attendance calibration, consistent work views and audited human corrections."""
import base64
import time
from datetime import date, datetime
from typing import Literal

import cv2
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import Response
from pydantic import BaseModel, ConfigDict, Field, model_validator
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session

from app.ai.doorway import DoorwayCalibration
from app.attendance_types import (
    ActiveView, CommissionInput, EventDecisionInput, EventDetail, EventPage, EventView,
    GapDecisionInput, GapDetail, GapView, ManualEventInput, ReviewPage, SessionPage, VisitorBatchInput, milliseconds,
)
from app.config import settings
from app.database.database import get_db, immediate_session
from app.database.models import (
    AttendanceCamera, AttendanceCommission, AttendanceCoverageGap, AttendanceDecision,
    AttendanceEvent, AttendanceEvidence, AttendanceRuntime, Camera, Employee, WorkSession,
)
from app.rtsp.reader import CameraStatus
from app.time_utils import attendance_iso, day_bounds, utcnow
from app.services.attendance_queries import (
    evaluate_session, event_view, gap_view, load_trust_blockers, read_snapshot,
)

router = APIRouter(prefix="/api/attendance", tags=["attendance"])


@router.get("/status")
def attendance_status(request: Request):
    service = getattr(request.app.state, "attendance", None)
    if service is None:
        raise HTTPException(503, "Attendance service unavailable")
    snapshot = service.health.snapshot()

    def read(db):
        runtime = db.get(AttendanceRuntime, 1)
        configurations = [_view(row) for row in db.query(AttendanceCamera).order_by(AttendanceCamera.camera_id)]
        valid = service._commission_valid(db, invalidate=False, providers=snapshot["providers"])
        blocked = service._clock_blocked(db)
        commission = db.get(AttendanceCommission, runtime.commission_id) if runtime.commission_id else None
        return dict(configuration=configurations, commission_id=runtime.commission_id,
                    commissioning_required=not valid, clock_blocked=blocked,
                    commission=None if commission is None else dict(
                        id=commission.id, actor=commission.actor, reason=commission.reason,
                        created_at=attendance_iso(commission.created_at),
                        acceptance_record=commission.acceptance_record),
                    monitoring_started_at=attendance_iso(runtime.monitoring_started_at),
                    as_of=attendance_iso(milliseconds(service._now())),
                    heartbeat_at=attendance_iso(runtime.heartbeat_at), stopped_at=attendance_iso(runtime.stopped_at),
                    unresolved_crossings=db.query(AttendanceEvent).filter_by(status="REVIEW").count(),
                    projection_conflicts=db.query(AttendanceEvent).filter(AttendanceEvent.projection_reasons != []).count(),
                    unreviewed_gaps=db.query(AttendanceCoverageGap).filter_by(reviewed=False).count(),
                    open_sessions=db.query(WorkSession).filter(WorkSession.ended_at.is_(None)).count(),
                    utc_offset_minutes=settings.attendance_utc_offset_minutes)
    try:
        persisted = read_snapshot(service.engine, read)
    except SQLAlchemyError as exc:
        raise HTTPException(503, "Attendance status unavailable") from exc
    snapshot.update(persisted)
    snapshot["automatic_ready"] = (snapshot["technical_ready"] and not persisted["commissioning_required"]
                                   and not persisted["clock_blocked"]
                                   and not snapshot["pending_fault_count"] and not snapshot["pending_write_count"])
    reasons = set(snapshot["reasons"]) - {"COMMISSIONING_REQUIRED", "CLOCK_DISCONTINUITY"}
    if persisted["commissioning_required"]:
        reasons.add("COMMISSIONING_REQUIRED")
    if persisted["clock_blocked"]:
        reasons.add("CLOCK_DISCONTINUITY")
    snapshot["reasons"] = sorted(reasons)
    return snapshot


class CalibrationInput(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    role: Literal["ENTRY", "EXIT"] | None
    enabled: bool
    line_start: tuple[float, float] | None
    line_end: tuple[float, float] | None
    inside_point: tuple[float, float] | None
    reference_width: int | None = Field(gt=0)
    reference_height: int | None = Field(gt=0)
    expected_revision: int = Field(ge=0)
    expected_source_revision: int = Field(ge=1)

    @model_validator(mode="after")
    def validate_geometry(self):
        geometry = (self.line_start, self.line_end, self.inside_point,
                    self.reference_width, self.reference_height)
        complete = all(value is not None for value in geometry)
        if self.enabled and (self.role is None or not complete):
            raise ValueError("Enabled attendance requires a role and complete geometry")
        if any(value is not None for value in geometry):
            if not complete:
                raise ValueError("Supply all calibration coordinates and dimensions together")
            DoorwayCalibration(0, self.role or "ENTRY", self.expected_revision,
                               self.expected_source_revision, self.line_start, self.line_end,
                               self.inside_point, self.reference_width, self.reference_height)
        return self


class CalibrationView(BaseModel):
    camera_id: int
    role: Literal["ENTRY", "EXIT"] | None
    enabled: bool
    revision: int
    source_revision: int
    line_start: tuple[float, float] | None
    line_end: tuple[float, float] | None
    inside_point: tuple[float, float] | None
    reference_width: int | None
    reference_height: int | None
    updated_at: str | None
    camera_enabled: bool | None = None
    ai_enabled: bool | None = None
    ai_included: bool | None = None
    ai_available: bool | None = None
    commissioning_required: bool | None = None


class CalibrationSnapshotView(BaseModel):
    jpeg_base64: str
    source_revision: int
    stream_id: str
    seq: int
    received_at: str
    width: int
    height: int


def _configuration(db: Session, camera_id: int) -> AttendanceCamera:
    try:
        if db.get(Camera, camera_id) is None:
            raise HTTPException(404, "Camera not found")
        config = db.get(AttendanceCamera, camera_id)
        if config is None:
            raise HTTPException(503, "Attendance camera configuration unavailable")
        return config
    except SQLAlchemyError as exc:
        raise HTTPException(503, "Database unavailable") from exc


def _view(config: AttendanceCamera) -> dict:
    return {"camera_id": config.camera_id, "role": config.role, "enabled": config.enabled,
            "revision": config.revision, "source_revision": config.source_revision,
            "line_start": config.line_start, "line_end": config.line_end,
            "inside_point": config.inside_point, "reference_width": config.reference_width,
            "reference_height": config.reference_height, "updated_at": attendance_iso(config.updated_at)}


@router.get("/cameras/{camera_id}", response_model=CalibrationView)
def get_camera_calibration(camera_id: int, request: Request, db: Session = Depends(get_db)):
    try:
        data = _view(_configuration(db, camera_id))
        camera = db.get(Camera, camera_id)
        runtime = db.get(AttendanceRuntime, 1)
        included = settings.ai_camera_ids()
        worker = getattr(request.app.state, "ai_worker", None)
        data.update(camera_enabled=camera.enabled, ai_enabled=settings.ai_enabled,
                    ai_included=included is None or camera_id in included,
                    ai_available=worker is not None and worker.is_alive(),
                    commissioning_required=runtime is None or runtime.commission_id is None)
        return data
    except SQLAlchemyError as exc:
        raise HTTPException(503, "Database unavailable") from exc


@router.put("/cameras/{camera_id}", response_model=CalibrationView)
def put_camera_calibration(camera_id: int, payload: CalibrationInput, db: Session = Depends(get_db)):
    engine = db.get_bind().engine
    db.rollback()
    try:
        with immediate_session(engine) as writer:
            config = _configuration(writer, camera_id)
            if (config.revision != payload.expected_revision
                    or config.source_revision != payload.expected_source_revision):
                raise HTTPException(409, "Calibration or camera source changed; freeze a new snapshot")
            if config.role is not None and payload.role is None:
                raise HTTPException(409, "Assigned attendance role cannot be erased; disable instead")
            if payload.enabled and writer.query(AttendanceCamera.camera_id).filter(
                    AttendanceCamera.camera_id != camera_id,
                    AttendanceCamera.enabled.is_(True), AttendanceCamera.role == payload.role).first():
                raise HTTPException(409, "Another enabled camera already owns this attendance role")
            config.role, config.enabled = payload.role, payload.enabled
            for name in ("line_start", "line_end", "inside_point"):
                value = getattr(payload, name)
                setattr(config, name, list(value) if value is not None else None)
            config.reference_width, config.reference_height = payload.reference_width, payload.reference_height
            config.revision += 1
            config.updated_at = utcnow()
            runtime = writer.get(AttendanceRuntime, 1)
            if runtime is None:
                raise HTTPException(503, "Attendance runtime unavailable")
            runtime.commission_id = None
            writer.flush()
            roles = {row[0] for row in writer.query(AttendanceCamera.role).filter(
                AttendanceCamera.enabled.is_(True)).all()}
            if runtime.monitoring_started_at is None and roles == {"ENTRY", "EXIT"}:
                runtime.monitoring_started_at = utcnow()
            writer.flush()
            result = _view(config)
        db.expire_all()
        return result
    except IntegrityError as exc:
        raise HTTPException(409, "Attendance role conflict; refresh configuration") from exc
    except SQLAlchemyError as exc:
        raise HTTPException(503, "Database unavailable; calibration was not saved") from exc


@router.get("/cameras/{camera_id}/snapshot", response_model=CalibrationSnapshotView)
def get_camera_calibration_snapshot(camera_id: int, request: Request, db: Session = Depends(get_db)):
    config = _configuration(db, camera_id)
    reader = request.app.state.manager.get(camera_id)
    if reader is None or reader.status() is not CameraStatus.ONLINE:
        raise HTTPException(503, "Camera is not online")
    packet = reader.buffer.latest()
    if (packet is None or packet.source_revision != config.source_revision
            or time.monotonic() - packet.received_mono >
            min(settings.frame_stall_timeout, settings.attendance_frame_max_age)):
        raise HTTPException(503, "No fresh frame for this camera source")
    try:
        ok, encoded = cv2.imencode(".jpg", packet.frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    except cv2.error as exc:
        raise HTTPException(503, "Could not encode calibration snapshot") from exc
    if not ok:
        raise HTTPException(503, "Could not encode calibration snapshot")
    height, width = packet.frame.shape[:2]
    return {"jpeg_base64": base64.b64encode(encoded).decode("ascii"),
            "source_revision": packet.source_revision, "stream_id": packet.stream_id,
            "seq": packet.seq, "received_at": attendance_iso(packet.received_at),
            "width": width, "height": height}


def _service(request: Request):
    service = getattr(request.app.state, "attendance", None)
    if service is None:
        raise HTTPException(503, "Attendance service unavailable")
    return service


def _read(request, reader):
    try:
        return read_snapshot(_service(request).engine, reader)
    except SQLAlchemyError as exc:
        raise HTTPException(503, "Attendance storage unavailable; retry the request") from exc


def _validate_filters(db, employee_id, from_date, to_date):
    if from_date is not None and to_date is not None and to_date < from_date:
        raise HTTPException(422, "End date precedes start date")
    if employee_id is not None and db.get(Employee, employee_id) is None:
        raise HTTPException(404, "Employee not found")


def _dates(query, column, from_date, to_date):
    if from_date is not None:
        query = query.filter(column >= day_bounds(from_date)[0])
    if to_date is not None:
        query = query.filter(column < day_bounds(to_date)[1])
    return query


def _event_query(db, employee_id, from_date, to_date, status=None):
    query = db.query(AttendanceEvent)
    if employee_id is not None:
        query = query.filter(AttendanceEvent.employee_id == employee_id)
    if status is not None:
        query = query.filter(AttendanceEvent.status == status)
    return _dates(query, AttendanceEvent.occurred_at, from_date, to_date)


@router.get("/events", response_model=EventPage)
def list_events(request: Request, employee_id: int | None = None,
                from_date: date | None = None, to_date: date | None = None,
                status: Literal["ACCEPTED", "REVIEW", "VISITOR", "VOID"] | None = None,
                limit: int = Query(50, ge=1, le=200), offset: int = Query(0, ge=0)):
    def read(db):
        _validate_filters(db, employee_id, from_date, to_date)
        query = _event_query(db, employee_id, from_date, to_date, status)
        total = query.count()
        rows = query.order_by(AttendanceEvent.occurred_at.desc(), AttendanceEvent.id.desc()).offset(offset).limit(limit)
        return EventPage(items=[event_view(db, row) for row in rows], total=total, limit=limit, offset=offset)
    return _read(request, read)


def _public_effective(snapshot):
    result = dict(snapshot)
    if result.get("occurred_at") is not None:
        result["occurred_at"] = attendance_iso(datetime.fromisoformat(result["occurred_at"]))
    return result


@router.get("/events/{event_id}", response_model=EventDetail)
def event_detail(event_id: str, request: Request):
    def read(db):
        row = db.get(AttendanceEvent, event_id)
        if row is None:
            raise HTTPException(404, "Event not found")
        decisions = [dict(
            id=item.id, request_id=item.request_id, revision=item.revision, actor=item.actor,
            reason=item.reason, before=_public_effective(item.before), after=_public_effective(item.after),
            created_at=attendance_iso(item.created_at),
        ) for item in db.query(AttendanceDecision).filter_by(event_id=event_id).order_by(
            AttendanceDecision.revision, AttendanceDecision.id)]
        return EventDetail(**event_view(db, row).model_dump(), raw=row.raw, decisions=decisions)
    return _read(request, read)


@router.get("/events/{event_id}/evidence/{kind}")
def event_evidence(event_id: str, kind: Literal["before", "after", "face"], request: Request):
    def read(db):
        if db.get(AttendanceEvent, event_id) is None:
            raise HTTPException(404, "Event not found")
        row = db.get(AttendanceEvidence, event_id)
        if row is not None and row.purged_at is not None:
            raise HTTPException(410, "Evidence retention period expired")
        image = getattr(row, f"{kind}_jpeg") if row is not None else None
        if not image:
            raise HTTPException(404, "This evidence image is absent")
        return image
    return Response(_read(request, read), media_type="image/jpeg", headers={"Cache-Control": "no-store"})


def _sessions(db, employee_id, from_date, to_date, status, as_of, *, active=False):
    _validate_filters(db, employee_id, from_date, to_date)
    query = db.query(WorkSession)
    if employee_id is not None:
        query = query.filter(WorkSession.employee_id == employee_id)
    if active:
        query = query.filter(WorkSession.ended_at.is_(None))
    query = _dates(query, WorkSession.started_at, from_date, to_date)
    blockers = load_trust_blockers(db, as_of=as_of)
    rows = [evaluate_session(row, blockers, as_of=as_of) for row in query.order_by(
        WorkSession.started_at.desc(), WorkSession.entry_event_id.desc())]
    return [row for row in rows if status is None or row.status == status]


@router.get("/sessions", response_model=SessionPage)
def list_sessions(request: Request, employee_id: int | None = None,
                  from_date: date | None = None, to_date: date | None = None,
                  status: Literal["OPEN", "CLOSED", "REVIEW"] | None = None,
                  limit: int = Query(50, ge=1, le=200), offset: int = Query(0, ge=0)):
    as_of = milliseconds(_service(request)._now())
    def read(db):
        rows = _sessions(db, employee_id, from_date, to_date, status, as_of)
        return SessionPage(items=rows[offset:offset + limit], total=len(rows), limit=limit, offset=offset)
    return _read(request, read)


@router.get("/active", response_model=ActiveView)
def active_sessions(request: Request, employee_id: int | None = None):
    as_of = milliseconds(_service(request)._now())
    def read(db):
        rows = _sessions(db, employee_id, None, None, None, as_of, active=True)
        return ActiveView(items=rows, total=len(rows), as_of=attendance_iso(as_of),
                          utc_offset_minutes=settings.attendance_utc_offset_minutes)
    return _read(request, read)


@router.get("/review", response_model=ReviewPage)
def list_review(request: Request, employee_id: int | None = None,
                from_date: date | None = None, to_date: date | None = None,
                kind: Literal["event", "gap"] | None = None,
                limit: int = Query(50, ge=1, le=200), offset: int = Query(0, ge=0)):
    def read(db):
        _validate_filters(db, employee_id, from_date, to_date)
        ordered = []
        if kind != "gap":
            events = _event_query(db, employee_id, from_date, to_date).filter(
                (AttendanceEvent.status == "REVIEW") | (AttendanceEvent.projection_reasons != []))
            ordered.extend((row.occurred_at, f"event:{row.id}", "event", row) for row in events)
        if kind != "event":
            gaps = db.query(AttendanceCoverageGap).filter_by(reviewed=False)
            if from_date is not None:
                lower = day_bounds(from_date)[0]
                gaps = gaps.filter((AttendanceCoverageGap.ended_at.is_(None)) |
                                   (AttendanceCoverageGap.ended_at > lower) |
                                   (AttendanceCoverageGap.started_at >= lower))
            if to_date is not None:
                gaps = gaps.filter(AttendanceCoverageGap.started_at < day_bounds(to_date)[1])
            ordered.extend((row.started_at, f"gap:{row.id}", "gap", row) for row in gaps)
        ordered.sort(key=lambda item: (item[0], item[1]), reverse=True)
        items = [dict(kind=kind, **{kind: event_view(db, row) if kind == "event" else gap_view(row)})
                 for _stamp, _key, kind, row in ordered[offset:offset + limit]]
        return ReviewPage(items=items, total=len(ordered), limit=limit, offset=offset)
    return _read(request, read)


@router.post("/events/{event_id}/decision", response_model=EventView)
def decide_event(event_id: str, payload: EventDecisionInput, request: Request):
    return _service(request).decide_event(event_id, payload)


@router.post("/visitors", response_model=list[EventView])
def mark_visitors(payload: VisitorBatchInput, request: Request):
    return _service(request).mark_visitors(payload)


@router.post("/manual", response_model=EventView)
def create_manual_event(payload: ManualEventInput, request: Request):
    return _service(request).create_manual_event(payload)


@router.post("/gaps/{gap_id}/review", response_model=GapView)
def review_gap(gap_id: int, payload: GapDecisionInput, request: Request):
    return _service(request).review_gap(gap_id, payload)


@router.post("/commission")
def commission(payload: CommissionInput, request: Request):
    return _service(request).commission(payload)


@router.get("/gaps/{gap_id}", response_model=GapDetail)
def gap_detail(gap_id: int, request: Request):
    def read(db):
        row = db.get(AttendanceCoverageGap, gap_id)
        if row is None:
            raise HTTPException(404, "Coverage gap not found")
        decisions = [dict(
            id=item.id, request_id=item.request_id, revision=item.revision, actor=item.actor,
            reason=item.reason, before=item.before, after=item.after, created_at=attendance_iso(item.created_at),
        ) for item in db.query(AttendanceDecision).filter_by(gap_id=gap_id).order_by(
            AttendanceDecision.revision, AttendanceDecision.id)]
        return GapDetail(**gap_view(row).model_dump(), decisions=decisions)
    return _read(request, read)
