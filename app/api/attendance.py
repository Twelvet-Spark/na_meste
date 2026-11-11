"""Transactional attendance calibration and metadata-bound receipt snapshots."""
import base64
import time
from typing import Literal

import cv2
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, model_validator
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session

from app.ai.doorway import DoorwayCalibration
from app.config import settings
from app.database.database import get_db, immediate_session
from app.database.models import (
    AttendanceCamera, AttendanceCoverageGap, AttendanceEvent, AttendanceRuntime, Camera, WorkSession,
)
from app.rtsp.reader import CameraStatus
from app.time_utils import attendance_iso, utcnow
from app.services.attendance_queries import read_snapshot

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
        return dict(configuration=configurations, commission_id=runtime.commission_id,
                    monitoring_started_at=attendance_iso(runtime.monitoring_started_at),
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
    snapshot["automatic_ready"] = snapshot["automatic_ready"] and persisted["commission_id"] is not None
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
