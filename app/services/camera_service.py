"""CRUD камер и реальная проверка RTSP-подключения (ffprobe)."""
import json
import logging
import subprocess
from datetime import datetime
from typing import Optional

from fastapi import HTTPException
from sqlalchemy.exc import SQLAlchemyError
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.orm import Session

from app.database.database import immediate_session
from app.database.models import AttendanceCamera, AttendanceEvent, AttendanceRuntime, Camera, PresenceSession
from app.rtsp.manager import ReaderManager
from app.rtsp.dahua import build_rtsp_url, mask_url
from app.rtsp.reader import parse_frame_rate

logger = logging.getLogger("app.services.cameras")


# ------------------------------------------------------------------- схемы

class CameraCreate(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    nvr_host: str = Field(min_length=1, max_length=255)
    rtsp_port: int = Field(default=554, ge=1, le=65535)
    username: str = Field(min_length=1, max_length=100)
    password: str = Field(min_length=1, max_length=255)
    channel: int = Field(ge=1, le=64)
    stream_type: str = Field(pattern="^(main|sub)$")
    enabled: bool = True


class CameraUpdate(BaseModel):
    """Частичное обновление. Пароль None/пусто — существующий не меняется."""
    name: Optional[str] = Field(default=None, min_length=1, max_length=100)
    nvr_host: Optional[str] = Field(default=None, min_length=1, max_length=255)
    rtsp_port: Optional[int] = Field(default=None, ge=1, le=65535)
    username: Optional[str] = Field(default=None, min_length=1, max_length=100)
    password: Optional[str] = Field(default=None, max_length=255)
    channel: Optional[int] = Field(default=None, ge=1, le=64)
    stream_type: Optional[str] = Field(default=None, pattern="^(main|sub)$")
    enabled: Optional[bool] = None


class CameraOut(BaseModel):
    """Ответ API. Пароль никогда не возвращается — только факт его наличия."""
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    nvr_host: str
    rtsp_port: int
    username: str
    channel: int
    stream_type: str
    enabled: bool
    has_password: bool
    created_at: datetime
    updated_at: datetime


class ConnectionTestRequest(BaseModel):
    """Параметры для проверки подключения без сохранения камеры.

    password может быть пустым, если передан camera_id существующей камеры —
    тогда пароль берётся из базы (для формы редактирования).
    """
    nvr_host: str = Field(min_length=1, max_length=255)
    rtsp_port: int = Field(default=554, ge=1, le=65535)
    username: str = Field(min_length=1, max_length=100)
    password: Optional[str] = Field(default=None, max_length=255)
    camera_id: Optional[int] = None
    channel: int = Field(ge=1, le=64)
    stream_type: str = Field(pattern="^(main|sub)$")


class ConnectionTestResult(BaseModel):
    success: bool
    error: Optional[str] = None
    codec: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    fps: Optional[float] = None
    resolution: Optional[str] = None


# ------------------------------------------------------- проверка подключения

def test_rtsp_connection(
    *,
    nvr_host: str,
    rtsp_port: int,
    username: str,
    password: str,
    channel: int,
    stream_type: str,
    ffprobe_path: Optional[str],
    timeout: int = 10,
) -> ConnectionTestResult:
    """Реально открывает RTSP-поток через ffprobe.

    Ошибки наружу отдаются generic-текстом: без пароля, без URL, без
    деталей stderr (они могут содержать URL с учётными данными).
    """
    if not ffprobe_path:
        return ConnectionTestResult(
            success=False, error="FFmpeg/ffprobe не найден на сервере"
        )

    url = build_rtsp_url(nvr_host, rtsp_port, username, password, channel, stream_type)
    cmd = [
        ffprobe_path, "-v", "error",
        "-rtsp_transport", "tcp",
        "-select_streams", "v:0",
        "-show_entries", "stream=codec_name,width,height,avg_frame_rate",
        "-of", "json",
        url,
    ]
    logger.info(
        "Проверка подключения: %s:%s канал %d (%s)",
        nvr_host, rtsp_port, channel, stream_type,
    )
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=timeout + 5)
    except subprocess.TimeoutExpired:
        return ConnectionTestResult(
            success=False, error="Unable to connect to RTSP stream (timeout)"
        )

    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", "replace").strip()
        # подробности — только в лог, с маскировкой пароля
        logger.warning(
            "Проверка подключения не удалась (%s:%s ch%d): %s",
            nvr_host, rtsp_port, channel,
            mask_url(stderr.splitlines()[-1] if stderr else "unknown error"),
        )
        return ConnectionTestResult(
            success=False, error="Unable to connect to RTSP stream."
        )

    try:
        stream = json.loads(result.stdout.decode("utf-8", "replace"))["streams"][0]
        width, height = int(stream["width"]), int(stream["height"])
    except (ValueError, KeyError, IndexError, TypeError):
        return ConnectionTestResult(
            success=False, error="Unable to connect to RTSP stream."
        )

    return ConnectionTestResult(
        success=True,
        codec=str(stream.get("codec_name") or "unknown"),
        width=width,
        height=height,
        fps=parse_frame_rate(stream.get("avg_frame_rate")),
        resolution=f"{width}x{height}",
    )


# -------------------------------------------------------------------- CRUD

def get_camera(db: Session, camera_id: int) -> Optional[Camera]:
    return db.get(Camera, camera_id)


def list_cameras(db: Session) -> list[Camera]:
    return db.query(Camera).order_by(Camera.channel, Camera.id).all()


def create_camera(db: Session, data: CameraCreate) -> Camera:
    camera = Camera(**data.model_dump())
    db.add(camera)
    db.flush()
    db.add(AttendanceCamera(camera_id=camera.id))
    db.commit()
    db.refresh(camera)
    logger.info("Создана камера #%d «%s» (%s ch%d)", camera.id, camera.name,
                camera.nvr_host, camera.channel)
    return camera


def update_camera(db: Session, camera: Camera, data: CameraUpdate) -> tuple[Camera, set[str]]:
    """Commit source invalidation before the API can publish a replacement reader."""
    camera_id = camera.id
    engine = db.get_bind().engine
    db.rollback()
    changed: set[str] = set()
    try:
        with immediate_session(engine) as writer:
            current = writer.get(Camera, camera_id)
            if current is None:
                raise HTTPException(404, "Camera not found")
            for field, value in data.model_dump(exclude_unset=True).items():
                if field == "password" and not value:
                    continue
                if getattr(current, field) != value:
                    setattr(current, field, value)
                    changed.add(field)
            if changed & ReaderManager.CONNECTION_FIELDS:
                config = writer.get(AttendanceCamera, camera_id)
                runtime = writer.get(AttendanceRuntime, 1)
                if config is None or runtime is None:
                    raise HTTPException(503, "Attendance configuration unavailable")
                config.source_revision += 1
                config.revision += 1
                config.enabled = False
                runtime.commission_id = None
        db.expire_all()
        return db.get(Camera, camera_id), changed
    except SQLAlchemyError as exc:
        raise HTTPException(503, "Database unavailable; camera was not changed") from exc


def delete_camera(db: Session, camera: Camera) -> None:
    """Retain every historical/ever-assigned camera, even when disabled."""
    camera_id, name = camera.id, camera.name
    bind = db.get_bind().engine
    db.rollback()
    try:
        with immediate_session(bind) as writer:
            current = writer.get(Camera, camera_id)
            if current is None:
                raise HTTPException(status_code=404, detail="Camera not found")
            config = writer.get(AttendanceCamera, camera_id)
            history = (
                writer.query(PresenceSession.id).filter_by(camera_id=camera_id).first()
                or writer.query(AttendanceEvent.id).filter_by(camera_id=camera_id).first()
                or (config is not None and config.role is not None)
            )
            if history:
                raise HTTPException(status_code=409, detail="Camera has history; disable instead")
            if config is not None:
                writer.delete(config)
                writer.flush()
            writer.delete(current)
        db.expire_all()
    except SQLAlchemyError as exc:
        raise HTTPException(status_code=503, detail="Database unavailable") from exc
    logger.info("Удалена камера #%d «%s»", camera_id, name)
