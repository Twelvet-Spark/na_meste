"""Door-gate calibration and work-time views over presence segments, with audited corrections."""
import base64
import time
from typing import Literal

import cv2
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import Response
from pydantic import BaseModel, ConfigDict, Field, model_validator
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session

from app.ai.doorway import DoorwayCalibration
from app.api.operator_security import require_operator_request
from app.attendance_types import AttendanceDate, CorrectionInput, RevokeInput, milliseconds
from app.config import settings
from app.database.database import get_db, immediate_session, read_snapshot
from app.database.models import AttendanceCamera, Camera, CameraProfile
from app.rtsp.reader import CameraStatus
from app.scene_geometry import validate_door
from app.services import attendance_reports as reports
from app.time_utils import attendance_iso, utcnow

router = APIRouter(prefix="/api/attendance", tags=["attendance"])


class CalibrationInput(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    role: Literal["ENTRY", "EXIT", "BOTH"] | None
    enabled: bool
    line_start: tuple[float, float] | None
    line_end: tuple[float, float] | None
    inside_point: tuple[float, float] | None
    door_kind: Literal["DOOR", "PASSAGE"] | None
    door_top: tuple[tuple[float, float], tuple[float, float]] | None
    reference_width: int | None = Field(gt=0)
    reference_height: int | None = Field(gt=0)
    expected_revision: int = Field(ge=0)
    expected_source_revision: int = Field(ge=1)

    @model_validator(mode="after")
    def validate_geometry(self):
        geometry = (self.line_start, self.line_end, self.inside_point, self.door_kind, self.door_top,
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
            validate_door(self.line_start, self.line_end, self.door_top)
        return self


class CalibrationView(BaseModel):
    camera_id: int
    role: Literal["ENTRY", "EXIT", "BOTH"] | None
    enabled: bool
    revision: int
    source_revision: int
    line_start: tuple[float, float] | None
    line_end: tuple[float, float] | None
    inside_point: tuple[float, float] | None
    door_kind: Literal["DOOR", "PASSAGE"] | None
    door_top: tuple[tuple[float, float], tuple[float, float]] | None
    reference_width: int | None
    reference_height: int | None
    updated_at: str | None
    camera_enabled: bool | None = None
    ai_enabled: bool | None = None
    ai_included: bool | None = None
    ai_available: bool | None = None
    time_tracking: bool | None = None


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
            "inside_point": config.inside_point, "door_kind": config.door_kind, "door_top": config.door_top,
            "reference_width": config.reference_width,
            "reference_height": config.reference_height, "updated_at": attendance_iso(config.updated_at)}


@router.get("/cameras/{camera_id}", response_model=CalibrationView)
def get_camera_calibration(camera_id: int, request: Request, db: Session = Depends(get_db)):
    try:
        data = _view(_configuration(db, camera_id))
        camera = db.get(Camera, camera_id)
        profile = db.get(CameraProfile, camera_id)
        worker = getattr(request.app.state, "ai_worker", None)
        data.update(camera_enabled=camera.enabled, ai_enabled=settings.ai_enabled,
                    ai_included=profile is not None and profile.ai_enabled,
                    ai_available=worker is not None and worker.is_alive(),
                    time_tracking=profile is not None and profile.time_tracking)
        return data
    except SQLAlchemyError as exc:
        raise HTTPException(503, "Database unavailable") from exc


@router.put("/cameras/{camera_id}", response_model=CalibrationView)
def put_camera_calibration(camera_id: int, payload: CalibrationInput, db: Session = Depends(get_db)):
    engine = db.get_bind().engine
    db.rollback()
    try:
        with immediate_session(engine, publish=True) as writer:
            config = _configuration(writer, camera_id)
            if (config.revision != payload.expected_revision
                    or config.source_revision != payload.expected_source_revision):
                raise HTTPException(409, "Calibration or camera source changed; freeze a new snapshot")
            if config.role is not None and payload.role is None:
                raise HTTPException(409, "Assigned attendance role cannot be erased; disable instead")
            config.role, config.enabled = payload.role, payload.enabled
            for name in ("line_start", "line_end", "inside_point"):
                value = getattr(payload, name)
                setattr(config, name, list(value) if value is not None else None)
            config.door_kind = payload.door_kind
            config.door_top = None if payload.door_top is None else [list(point) for point in payload.door_top]
            config.reference_width, config.reference_height = payload.reference_width, payload.reference_height
            config.revision += 1
            config.updated_at = utcnow()
            writer.flush()
            result = _view(config)
        db.expire_all()
        return result
    except IntegrityError as exc:
        raise HTTPException(409, "Attendance configuration conflict; refresh configuration") from exc
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


# ------------------------------------------------------------------ work time

def _presence(request: Request):
    presence = getattr(request.app.state, "presence", None)
    if presence is None:
        raise HTTPException(503, "Work-time recording unavailable")
    return presence


def _read(request: Request, reader):
    try:
        return read_snapshot(_presence(request).engine, reader)
    except SQLAlchemyError as exc:
        raise HTTPException(503, "Work-time storage unavailable; retry the request") from exc


def _now():
    return milliseconds(utcnow())


@router.get("/status")
def attendance_status(request: Request):
    presence = _presence(request)
    state = request.app.state
    manager = getattr(state, "manager", None)
    worker = getattr(state, "ai_worker", None)

    def read(db):
        rows = db.query(Camera, CameraProfile).outerjoin(
            CameraProfile, CameraProfile.camera_id == Camera.id).order_by(Camera.channel, Camera.id).all()
        return [{"camera_id": camera.id, "name": camera.name, "enabled": camera.enabled,
                 "time_tracking": bool(profile is not None and profile.time_tracking),
                 "ai_enabled": bool(profile is not None and profile.ai_enabled),
                 "status": manager.status(camera.id)["status"] if manager is not None else "OFFLINE"}
                for camera, profile in rows]
    cameras = _read(request, read)
    return {"as_of": attendance_iso(_now()), "utc_offset_minutes": settings.attendance_utc_offset_minutes,
            "live_gap_minutes": settings.attendance_live_gap_minutes,
            "day_gap_minutes": settings.attendance_day_gap_minutes,
            "ai_enabled": settings.ai_enabled, "worker_alive": worker is not None and worker.is_alive(),
            "face_available": getattr(state, "embedding_store", None) is not None,
            "appearance_available": worker is not None and getattr(worker, "reid", None) is not None,
            "cameras": cameras, "recorder": presence.snapshot()}


@router.get("/active")
def attendance_active(request: Request):
    as_of = _now()
    return _read(request, lambda db: reports.active_view(db, as_of))


@router.get("/report")
def attendance_report(request: Request, from_date: AttendanceDate, to_date: AttendanceDate,
                      employee_id: int | None = None, workplace_id: int | None = None,
                      employee_status: Literal["ALL", "ACTIVE", "INACTIVE"] = "ALL"):
    as_of = _now()
    return _read(request, lambda db: reports.build_report(
        db, from_date, to_date, employee_id, as_of, workplace_id=workplace_id, employee_status=employee_status))


@router.get("/report.csv")
def attendance_report_csv(request: Request, from_date: AttendanceDate, to_date: AttendanceDate,
                          employee_id: int | None = None, workplace_id: int | None = None,
                          employee_status: Literal["ALL", "ACTIVE", "INACTIVE"] = "ALL"):
    as_of = _now()
    report = _read(request, lambda db: reports.build_report(
        db, from_date, to_date, employee_id, as_of, workplace_id=workplace_id, employee_status=employee_status))
    return Response(reports.report_csv(report), media_type="text/csv; charset=utf-8", headers={
        "Content-Disposition": f'attachment; filename="work-time-{from_date}-{to_date}.csv"',
        "Cache-Control": "no-store",
    })


@router.get("/report.xlsx")
def attendance_report_xlsx(request: Request, from_date: AttendanceDate, to_date: AttendanceDate,
                           employee_id: int | None = None, workplace_id: int | None = None,
                           employee_status: Literal["ALL", "ACTIVE", "INACTIVE"] = "ALL"):
    as_of = _now()
    report = _read(request, lambda db: reports.build_report(
        db, from_date, to_date, employee_id, as_of, workplace_id=workplace_id, employee_status=employee_status))
    return Response(reports.report_xlsx(report),
                    media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", headers={
                        "Content-Disposition": f'attachment; filename="work-time-{from_date}-{to_date}.xlsx"',
                        "Cache-Control": "no-store",
                    })


@router.get("/timeline")
def attendance_timeline(request: Request, employee_id: int, date: AttendanceDate):
    as_of = _now()
    return _read(request, lambda db: reports.timeline(db, employee_id, date, as_of))


@router.get("/corrections")
def list_corrections(request: Request, employee_id: int | None = None,
                     from_date: AttendanceDate | None = None, to_date: AttendanceDate | None = None):
    return _read(request, lambda db: reports.list_corrections(db, from_date, to_date, employee_id))


def _write(request: Request, action):
    try:
        return action(_presence(request).engine)
    except SQLAlchemyError as exc:
        raise HTTPException(503, "Work-time storage unavailable; the correction was not saved") from exc


@router.post("/corrections/{kind}", dependencies=[Depends(require_operator_request)])
def create_correction(kind: Literal["manual", "exclusion"], payload: CorrectionInput, request: Request):
    return _write(request, lambda engine: reports.create_correction(engine, kind, payload, _now()))


@router.post("/corrections/{kind}/{correction_id}/revoke", dependencies=[Depends(require_operator_request)])
def revoke_correction(kind: Literal["manual", "exclusion"], correction_id: int, payload: RevokeInput,
                      request: Request):
    return _write(request, lambda engine: reports.revoke_correction(engine, kind, correction_id, payload, _now()))
