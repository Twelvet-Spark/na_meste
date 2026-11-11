"""CRUD камер и реальная проверка RTSP-подключения (ffprobe)."""
import hashlib
import json
import logging
import subprocess
from datetime import datetime
from itertools import count
from typing import Optional
from uuid import UUID
from typing import Literal

from fastapi import HTTPException
from sqlalchemy import case, func, or_
from sqlalchemy.exc import SQLAlchemyError
from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, field_validator, model_validator
from sqlalchemy.orm import Session

from app.database.database import immediate_session
from app.database.models import (
    AttendanceCamera, AttendanceEvent, AttendanceRuntime, Camera, CameraArchive,
    GlobalEvent, GlobalObservation, GlobalPerson, OperatorRequest, PresenceSession,
)
from app.rtsp.source import CONNECTION_FIELDS, ArchiveSource, CameraSource
from app.rtsp.dahua import build_rtsp_url, mask_url, normalize_host
from app.rtsp.reader import parse_frame_rate

logger = logging.getLogger("app.services.cameras")

# Captured under the DB write lock, so delayed publications retain commit order.
_publication_sequences = count(1)


# ------------------------------------------------------------------- схемы

class ArchiveCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    http_scheme: Literal["http", "https"]
    http_port: StrictInt = Field(ge=1, le=65535)
    utc_offset_minutes: StrictInt = Field(ge=-720, le=840)


class ArchiveOut(ArchiveCreate):
    model_config = ConfigDict(from_attributes=True)
    revision: int
    updated_at: datetime


class HostInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    @field_validator("nvr_host", mode="before", check_fields=False)
    @classmethod
    def bare_host(cls, value):
        return normalize_host(value) if isinstance(value, str) else value

    @field_validator("name", mode="before", check_fields=False)
    @classmethod
    def trimmed_name(cls, value):
        return value.strip() if isinstance(value, str) else value


class CameraCreate(HostInput):
    name: str = Field(min_length=1, max_length=100)
    nvr_host: str = Field(min_length=1, max_length=255)
    rtsp_port: StrictInt = Field(default=554, ge=1, le=65535)
    username: str = Field(min_length=1, max_length=100)
    password: str = Field(min_length=1, max_length=255, repr=False)
    channel: StrictInt = Field(default=1, ge=1, le=64)
    stream_type: Literal["main", "sub"] = "main"
    enabled: StrictBool = True
    archive: ArchiveCreate | None = None


class CameraUpdate(HostInput):
    """Blank password keeps the stored secret; archive configuration has its own CAS."""
    name: str | None = Field(default=None, min_length=1, max_length=100)
    nvr_host: str | None = Field(default=None, min_length=1, max_length=255)
    rtsp_port: StrictInt | None = Field(default=None, ge=1, le=65535)
    username: str | None = Field(default=None, min_length=1, max_length=100)
    password: str | None = Field(default=None, max_length=255, repr=False)
    channel: StrictInt | None = Field(default=None, ge=1, le=64)
    stream_type: Literal["main", "sub"] | None = None
    enabled: StrictBool | None = None
    archive: ArchiveCreate | None = None
    # Explicit null observes absence; integers observe an existing row, including revision 0.
    expected_archive_revision: StrictInt | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_update(self):
        for name in self.model_fields_set - {"password", "archive", "expected_archive_revision"}:
            if getattr(self, name) is None:
                raise ValueError(f"{name} не может быть null")
        if "archive" in self.model_fields_set and "expected_archive_revision" not in self.model_fields_set:
            raise ValueError("Для изменения архива требуется expected_archive_revision")
        if "archive" in self.model_fields_set and self.archive is None:
            raise ValueError("Укажите протокол, порт и смещение архива")
        if "expected_archive_revision" in self.model_fields_set and "archive" not in self.model_fields_set:
            raise ValueError("expected_archive_revision относится только к изменению архива")
        return self


class BatchChannel(HostInput):
    channel: StrictInt = Field(ge=1, le=64)
    name: str = Field(min_length=1, max_length=100)
    stream_type: Literal["main", "sub"] = "main"
    enabled: StrictBool = True


class CameraBatch(HostInput):
    request_id: UUID
    nvr_host: str = Field(min_length=1, max_length=255)
    rtsp_port: StrictInt = Field(default=554, ge=1, le=65535)
    username: str = Field(min_length=1, max_length=100)
    password: str = Field(min_length=1, max_length=255, repr=False)
    archive: ArchiveCreate | None = None
    channels: list[BatchChannel] = Field(min_length=1, max_length=64)

    @field_validator("channels")
    @classmethod
    def unique_channels(cls, value):
        if len({item.channel for item in value}) != len(value):
            raise ValueError("Каналы регистратора должны быть уникальными")
        return value

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
    archive: ArchiveOut | None = None


class ConnectionTestRequest(HostInput):
    """Параметры для проверки подключения без сохранения камеры.

    password может быть пустым, если передан camera_id существующей камеры —
    тогда пароль берётся из базы (для формы редактирования).
    """
    nvr_host: str = Field(min_length=1, max_length=255)
    rtsp_port: int = Field(default=554, ge=1, le=65535)
    username: str = Field(min_length=1, max_length=100)
    password: Optional[str] = Field(default=None, max_length=255, repr=False)
    camera_id: Optional[int] = None
    channel: int = Field(ge=1, le=64)
    stream_type: str = Field(pattern="^(main|sub)$")


class DiscoveryRequest(ConnectionTestRequest):
    channel: StrictInt = Field(default=1, ge=1, le=64)
    stream_type: Literal["main", "sub"] = "main"
    http_scheme: Literal["http", "https"] = "http"
    http_port: StrictInt = Field(default=80, ge=1, le=65535)

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


def camera_out(db: Session, camera: Camera | CameraSource) -> dict:
    data = CameraOut.model_validate(camera).model_dump(mode="json")
    if not isinstance(camera, CameraSource):
        archive = db.get(CameraArchive, camera.id)
        data["archive"] = ArchiveOut.model_validate(archive).model_dump(mode="json") if archive else None
    return data


def camera_snapshot(db: Session, camera_id: int) -> CameraSource | None:
    row = db.query(Camera, AttendanceCamera.source_revision, CameraArchive).join(
        AttendanceCamera, AttendanceCamera.camera_id == Camera.id).outerjoin(
            CameraArchive, CameraArchive.camera_id == Camera.id).filter(Camera.id == camera_id).populate_existing().one_or_none()
    if row is None:
        return None
    camera, source_revision, archive = row
    return CameraSource.from_camera(camera, source_revision=source_revision, archive=archive)


def get_archive_source(db: Session, camera_id: int) -> ArchiveSource | None:
    """One SELECT captures credentials and both revisions, never late ORM loads."""
    row = db.query(
        Camera.id, AttendanceCamera.source_revision, CameraArchive.revision, Camera.nvr_host,
        Camera.rtsp_port, CameraArchive.http_scheme, CameraArchive.http_port, Camera.username,
        Camera.password, Camera.channel, CameraArchive.utc_offset_minutes,
    ).select_from(Camera).join(
        AttendanceCamera, AttendanceCamera.camera_id == Camera.id).join(
            CameraArchive, CameraArchive.camera_id == Camera.id).filter(Camera.id == camera_id).one_or_none()
    return ArchiveSource(*row) if row is not None else None


def _create_camera(writer: Session, data: CameraCreate) -> CameraSource:
    """Shared transactional core; the caller alone commits/publishes."""
    camera = Camera(**data.model_dump(exclude={"archive"}))
    writer.add(camera)
    writer.flush()
    config = AttendanceCamera(camera_id=camera.id)
    archive = CameraArchive(camera_id=camera.id, **data.archive.model_dump()) if data.archive else None
    writer.add(config)
    if archive is not None:
        writer.add(archive)
    writer.flush()
    return CameraSource.from_camera(camera, source_revision=config.source_revision,
                                    publication_sequence=next(_publication_sequences), archive=archive)


def create_camera(db: Session, data: CameraCreate) -> CameraSource:
    engine = db.get_bind().engine
    db.rollback()
    try:
        with immediate_session(engine) as writer:
            snapshot = _create_camera(writer, data)
        db.expire_all()
        return snapshot
    except SQLAlchemyError as exc:
        raise HTTPException(503, "Database unavailable; camera was not created") from exc


def create_camera_batch(db: Session, data: CameraBatch) -> tuple[dict, list[CameraSource], bool]:
    canonical = data.model_dump(mode="json", exclude={"request_id"})
    digest = hashlib.sha256(json.dumps(canonical, sort_keys=True, separators=(",", ":"),
                                       ensure_ascii=False).encode("utf-8")).hexdigest()
    engine = db.get_bind().engine
    db.rollback()
    try:
        with immediate_session(engine) as writer:
            prior = writer.get(OperatorRequest, str(data.request_id))
            if prior is not None:
                if prior.operation != "camera.batch" or prior.payload_sha256 != digest:
                    raise HTTPException(409, "Этот request_id уже использован с другими данными")
                return prior.result, [], True
            # Existing legacy hosts may not yet be normalized in storage.
            candidates = writer.query(Camera).filter_by(rtsp_port=data.rtsp_port, username=data.username).all()
            existing = set()
            for camera in candidates:
                try:
                    host = normalize_host(camera.nvr_host)
                except ValueError:
                    continue
                existing.add((host, camera.channel, camera.stream_type))
            if any((data.nvr_host, item.channel, item.stream_type) in existing for item in data.channels):
                raise HTTPException(409, "Этот канал и поток уже добавлены")
            shared = data.model_dump(exclude={"request_id", "channels"})
            snapshots = [_create_camera(writer, CameraCreate(**shared, **item.model_dump())) for item in data.channels]
            result = {"cameras": [camera_out(writer, source) for source in snapshots]}
            writer.add(OperatorRequest(id=str(data.request_id), operation="camera.batch",
                                       payload_sha256=digest, result=result))
        db.expire_all()
        return result, snapshots, False
    except SQLAlchemyError as exc:
        raise HTTPException(503, "Database unavailable; recorder was not created") from exc


def update_camera(db: Session, camera: Camera, data: CameraUpdate) -> tuple[CameraSource, set[str]]:
    """Capture the source/revision pair under the write lock, then commit it."""
    camera_id = camera.id
    engine = db.get_bind().engine
    db.rollback()
    changed: set[str] = set()
    try:
        with immediate_session(engine) as writer:
            current = writer.get(Camera, camera_id)
            if current is None:
                raise HTTPException(404, "Camera not found")
            config = writer.get(AttendanceCamera, camera_id)
            if config is None:
                raise HTTPException(503, "Attendance configuration unavailable")
            archive = writer.get(CameraArchive, camera_id)
            for field, value in data.model_dump(exclude_unset=True, exclude={"archive", "expected_archive_revision"}).items():
                if field == "password" and not value:
                    continue
                if getattr(current, field) != value:
                    setattr(current, field, value)
                    changed.add(field)
            if changed & CONNECTION_FIELDS:
                runtime = writer.get(AttendanceRuntime, 1)
                if runtime is None:
                    raise HTTPException(503, "Attendance configuration unavailable")
                config.source_revision += 1
                config.revision += 1
                config.enabled = False
                runtime.commission_id = None
            if "archive" in data.model_fields_set:
                actual_revision = archive.revision if archive is not None else None
                if data.expected_archive_revision != actual_revision:
                    raise HTTPException(409, "Настройки архива изменились. Обновите версию, сохранив свой черновик.")
                if archive is None:
                    archive = CameraArchive(camera_id=camera_id, **data.archive.model_dump())
                    writer.add(archive)
                    changed.add("archive")
                elif any(getattr(archive, key) != value for key, value in data.archive.model_dump().items()):
                    for key, value in data.archive.model_dump().items():
                        setattr(archive, key, value)
                    archive.revision += 1
                    changed.add("archive")
            writer.flush()
            snapshot = CameraSource.from_camera(
                current, source_revision=config.source_revision,
                publication_sequence=next(_publication_sequences), archive=archive)
        db.expire_all()
        return snapshot, changed
    except SQLAlchemyError as exc:
        raise HTTPException(503, "Database unavailable; camera was not changed") from exc


def delete_camera(db: Session, camera: Camera | CameraSource) -> None:
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
            # Old global events use a text JSON payload, not normalized FKs.
            # Transition endpoints are still used to name historical cameras.
            payload = case((func.json_valid(GlobalEvent.payload), GlobalEvent.payload), else_="{}")
            history = (
                writer.query(PresenceSession.id).filter_by(camera_id=camera_id).first()
                or writer.query(AttendanceEvent.id).filter_by(camera_id=camera_id).first()
                or writer.query(GlobalPerson.id).filter_by(last_camera_id=camera_id).first()
                or writer.query(GlobalObservation.id).filter_by(camera_id=camera_id).first()
                or writer.query(GlobalEvent.id).filter(or_(
                    GlobalEvent.camera_id == camera_id,
                    func.json_extract(payload, "$.from_camera") == camera_id,
                    func.json_extract(payload, "$.to_camera") == camera_id,
                    func.json_extract(payload, "$.last_camera") == camera_id,
                )).first()
                or (config is not None and config.role is not None)
            )
            if history:
                raise HTTPException(status_code=409, detail="Camera has history; disable instead")
            if config is not None:
                writer.delete(config)
                writer.flush()
            archive = writer.get(CameraArchive, camera_id)
            if archive is not None:
                writer.delete(archive)
                writer.flush()
            writer.delete(current)
        db.expire_all()
    except SQLAlchemyError as exc:
        raise HTTPException(status_code=503, detail="Database unavailable") from exc
    logger.info("Удалена камера #%d «%s»", camera_id, name)
