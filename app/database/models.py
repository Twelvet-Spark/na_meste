"""Модели данных.

RTSP URL в базе НЕ хранится — генерируется из параметров.
AI-таблицы (сотрудники, лица, присутствие) хранят эмбеддинги в BLOB.
"""
from datetime import datetime, timezone

from sqlalchemy import (
    JSON, Boolean, CheckConstraint, DateTime, Float, ForeignKey, Index, Integer,
    LargeBinary, String, UniqueConstraint, text,
)
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.orm import Mapped, mapped_column

from app.database.database import Base
from app.time_utils import utcnow


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Camera(Base):
    __tablename__ = "cameras"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(100))
    nvr_host: Mapped[str] = mapped_column(String(255))
    rtsp_port: Mapped[int] = mapped_column(Integer, default=554)
    username: Mapped[str] = mapped_column(String(100))
    # Пароль хранится в БД (не в .env), но никогда не отдаётся наружу через API
    password: Mapped[str] = mapped_column(String(255))
    channel: Mapped[int] = mapped_column(Integer)
    stream_type: Mapped[str] = mapped_column(String(10), default="main")  # 'main' | 'sub'
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, onupdate=_utcnow)

    @hybrid_property
    def has_password(self) -> bool:
        return bool(self.password)

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Camera id={self.id} name={self.name!r} host={self.nvr_host} ch={self.channel}>"


class CameraArchive(Base):
    __tablename__ = "camera_archives"
    __table_args__ = (
        CheckConstraint("http_scheme IN ('http', 'https')"),
        CheckConstraint("http_port BETWEEN 1 AND 65535"),
        CheckConstraint("utc_offset_minutes BETWEEN -720 AND 840"),
        CheckConstraint("revision >= 0"),
    )

    camera_id: Mapped[int] = mapped_column(ForeignKey("cameras.id", ondelete="RESTRICT"), primary_key=True)
    http_scheme: Mapped[str] = mapped_column(String(5))
    http_port: Mapped[int] = mapped_column(Integer)
    utc_offset_minutes: Mapped[int] = mapped_column(Integer)
    revision: Mapped[int] = mapped_column(Integer, default=0, server_default=text("0"))
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)


class RecordingSession(Base):
    __tablename__ = "recording_sessions"
    __table_args__ = (
        CheckConstraint("status IN ('STARTING','RECORDING','RECONNECTING','STOPPING','STOPPED','COMPLETED','FAILED','INTERRUPTED')"),
        Index("uq_recording_active_camera", "camera_id", unique=True,
              sqlite_where=text("status IN ('STARTING','RECORDING','RECONNECTING','STOPPING')")),
    )
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    camera_id: Mapped[int] = mapped_column(ForeignKey("cameras.id", ondelete="RESTRICT"))
    camera_name: Mapped[str] = mapped_column(String(100))
    source_revision: Mapped[int] = mapped_column(Integer)
    requested_at: Mapped[datetime] = mapped_column(DateTime)
    started_at: Mapped[datetime | None] = mapped_column(DateTime)
    deadline_at: Mapped[datetime] = mapped_column(DateTime)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime)
    status: Mapped[str] = mapped_column(String(20))
    reason_code: Mapped[str | None] = mapped_column(String(64))


class RecordingConnection(Base):
    __tablename__ = "recording_connections"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    session_id: Mapped[str] = mapped_column(ForeignKey("recording_sessions.id", ondelete="RESTRICT"))
    directory: Mapped[str] = mapped_column(String(200), unique=True)
    source_revision: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime)
    anchor_utc: Mapped[datetime | None] = mapped_column(DateTime)
    media_origin_seconds: Mapped[float | None] = mapped_column(Float)
    closed_at: Mapped[datetime | None] = mapped_column(DateTime)


class RecordingSegment(Base):
    __tablename__ = "recording_segments"
    __table_args__ = (
        CheckConstraint("state IN ('READY','EXPIRED','MISSING','CORRUPT','DELETING','UNINDEXED')"),
        CheckConstraint("state != 'READY' OR (started_at IS NOT NULL AND ended_at IS NOT NULL AND duration_ms > 0)"),
        CheckConstraint("duration_ms >= 0"), CheckConstraint("bytes >= 0"),
    )
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    session_id: Mapped[str] = mapped_column(ForeignKey("recording_sessions.id", ondelete="RESTRICT"))
    connection_id: Mapped[str] = mapped_column(ForeignKey("recording_connections.id", ondelete="RESTRICT"))
    camera_id: Mapped[int] = mapped_column(ForeignKey("cameras.id", ondelete="RESTRICT"))
    source_revision: Mapped[int] = mapped_column(Integer)
    storage_key: Mapped[str] = mapped_column(String(250), unique=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime)
    duration_ms: Mapped[int] = mapped_column(Integer, default=0, server_default=text("0"))
    bytes: Mapped[int] = mapped_column(Integer, default=0, server_default=text("0"))
    codec: Mapped[str | None] = mapped_column(String(32))
    state: Mapped[str] = mapped_column(String(20))


class RecordingGap(Base):
    __tablename__ = "recording_gaps"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    session_id: Mapped[str] = mapped_column(ForeignKey("recording_sessions.id", ondelete="RESTRICT"))
    started_at: Mapped[datetime] = mapped_column(DateTime)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime)
    reason_code: Mapped[str] = mapped_column(String(64))


class Employee(Base):
    __tablename__ = "employees"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(150))
    external_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, onupdate=_utcnow)
    first_name: Mapped[str | None] = mapped_column(String(74), nullable=True)
    last_name: Mapped[str | None] = mapped_column(String(74), nullable=True)
    revision: Mapped[int] = mapped_column(Integer, default=0, server_default=text("0"))

    __table_args__ = (CheckConstraint("revision >= 0"),)

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Employee id={self.id} name={self.name!r}>"


class EmployeeFace(Base):
    """Эмбеддинг лица сотрудника (ArcFace, 512-d float32 → BLOB).

    Один сотрудник может иметь несколько эмбеддингов (разные углы, очки,
    освещение). thumbnail — маленький JPEG 112x112 выровненного лица
    для управления в UI; исходные фотографии не хранятся.
    """
    __tablename__ = "employee_faces"

    id: Mapped[int] = mapped_column(primary_key=True)
    employee_id: Mapped[int] = mapped_column(
        ForeignKey("employees.id", ondelete="CASCADE"), index=True
    )
    embedding: Mapped[bytes] = mapped_column(LargeBinary)  # float32 × 512
    thumbnail: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)


class PresenceSession(Base):
    """Сессия присутствия сотрудника на камере.

    Вместо тысяч событий на каждый кадр — одна открытая сессия:
    started_at при первом подтверждении, last_seen_at обновляется,
    ended_at = last_seen_at при исчезновении (по таймауту).
    """
    __tablename__ = "presence_sessions"
    __table_args__ = (
        Index("ix_presence_emp_cam_end", "employee_id", "camera_id", "ended_at"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    employee_id: Mapped[int] = mapped_column(
        ForeignKey("employees.id", ondelete="CASCADE"), index=True
    )
    camera_id: Mapped[int] = mapped_column(
        ForeignKey("cameras.id", ondelete="CASCADE"), index=True
    )
    started_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    last_seen_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)


class GlobalPerson(Base):
    """Глобальная личность человека (global_id) — единая для всех камер.

    Локальные track_id привязываются к global_id через GlobalIdentityManager
    (composite matching: Re-ID + время + топология + лицо).
    """
    __tablename__ = "global_persons"

    id: Mapped[int] = mapped_column(primary_key=True)          # = global_id
    employee_id: Mapped[int | None] = mapped_column(
        ForeignKey("employees.id", ondelete="SET NULL"), nullable=True
    )
    status: Mapped[str] = mapped_column(String(20), default="ACTIVE")  # NEW/ACTIVE/LOST
    last_camera_id: Mapped[int | None] = mapped_column(
        ForeignKey("cameras.id", ondelete="SET NULL"), nullable=True
    )
    last_track_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    last_seen_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)


class GlobalObservation(Base):
    """Наблюдение глобальной личности: появление на камере с новым локальным
    треком. Appearance-эмбеддинг (OSNet, 512-d) и снимок хранятся как архив;
    после перезапуска эти записи не загружаются в живую галерею."""
    __tablename__ = "global_observations"

    id: Mapped[int] = mapped_column(primary_key=True)
    global_id: Mapped[int] = mapped_column(
        ForeignKey("global_persons.id", ondelete="CASCADE"), index=True
    )
    camera_id: Mapped[int] = mapped_column(
        ForeignKey("cameras.id", ondelete="CASCADE"), index=True
    )
    track_id: Mapped[int] = mapped_column(Integer)
    embedding: Mapped[bytes] = mapped_column(LargeBinary)
    snapshot: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)


class GlobalEvent(Base):
    """События глобального трекинга: person_seen / camera_transition / person_lost.
    payload — JSON с деталями матча (score, similarity, from/to)."""
    __tablename__ = "global_events"

    id: Mapped[int] = mapped_column(primary_key=True)
    global_id: Mapped[int] = mapped_column(
        ForeignKey("global_persons.id", ondelete="CASCADE"), index=True
    )
    event_type: Mapped[str] = mapped_column(String(30))
    camera_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    track_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    payload: Mapped[str | None] = mapped_column(String(2000), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)

class OperatorSchema(Base):
    __tablename__ = "operator_schema"
    __table_args__ = (CheckConstraint("id = 1"), CheckConstraint("version >= 0"))

    id: Mapped[int] = mapped_column(primary_key=True)
    version: Mapped[int] = mapped_column(Integer)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class Workplace(Base):
    __tablename__ = "workplaces"
    __table_args__ = (
        CheckConstraint("length(name) BETWEEN 1 AND 150 AND name = trim(name)"),
        CheckConstraint("active IN (0, 1)"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(150))
    name_key: Mapped[str] = mapped_column(String(450), unique=True)
    active: Mapped[bool] = mapped_column(Boolean, default=True, server_default=text("1"))


class EmployeeWorkplace(Base):
    __tablename__ = "employee_workplaces"

    employee_id: Mapped[int] = mapped_column(ForeignKey("employees.id", ondelete="CASCADE"), primary_key=True)
    workplace_id: Mapped[int] = mapped_column(ForeignKey("workplaces.id", ondelete="RESTRICT"), primary_key=True)


class OperatorRequest(Base):
    __tablename__ = "operator_requests"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    operation: Mapped[str] = mapped_column(String(100))
    payload_sha256: Mapped[str] = mapped_column(String(64))
    result: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)



class AttendanceCamera(Base):
    __tablename__ = "attendance_cameras"
    __table_args__ = (
        CheckConstraint("role IS NULL OR role IN ('ENTRY', 'EXIT')", name="ck_attendance_camera_role"),
        CheckConstraint("revision >= 0 AND source_revision >= 1", name="ck_attendance_camera_revisions"),
        CheckConstraint("enabled IN (0, 1)", name="ck_attendance_camera_enabled"),
        CheckConstraint(
            "enabled = 0 OR (role IS NOT NULL AND line_start IS NOT NULL AND "
            "line_end IS NOT NULL AND inside_point IS NOT NULL AND "
            "reference_width IS NOT NULL AND reference_height IS NOT NULL AND "
            "reference_width > 0 AND reference_height > 0)",
            name="ck_attendance_camera_complete",
        ),
        Index("uq_attendance_cameras_enabled_role", "role", unique=True,
              sqlite_where=text("enabled = 1")),
    )

    camera_id: Mapped[int] = mapped_column(ForeignKey("cameras.id", ondelete="RESTRICT"), primary_key=True)
    role: Mapped[str | None] = mapped_column(String(5), nullable=True)
    enabled: Mapped[bool] = mapped_column(Boolean, default=False, server_default=text("0"))
    revision: Mapped[int] = mapped_column(Integer, default=0, server_default=text("0"))
    source_revision: Mapped[int] = mapped_column(Integer, default=1, server_default=text("1"))
    line_start: Mapped[list | None] = mapped_column(JSON(none_as_null=True), nullable=True)
    line_end: Mapped[list | None] = mapped_column(JSON(none_as_null=True), nullable=True)
    inside_point: Mapped[list | None] = mapped_column(JSON(none_as_null=True), nullable=True)
    reference_width: Mapped[int | None] = mapped_column(Integer, nullable=True)
    reference_height: Mapped[int | None] = mapped_column(Integer, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)


class AttendanceEvent(Base):
    __tablename__ = "attendance_events"
    __table_args__ = (
        CheckConstraint("source IN ('CAMERA', 'MANUAL')", name="ck_attendance_event_source"),
        CheckConstraint("direction IS NULL OR direction IN ('ENTRY', 'EXIT')", name="ck_attendance_event_direction"),
        CheckConstraint("status IN ('ACCEPTED', 'REVIEW', 'VISITOR', 'VOID')", name="ck_attendance_event_status"),
        CheckConstraint("revision >= 0", name="ck_attendance_event_revision"),
        CheckConstraint("json_valid(raw) AND json_type(raw, '$.version') IS 'integer' "
                        "AND json_extract(raw, '$.version') = 1", name="ck_attendance_event_raw_version"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    source_key: Mapped[str] = mapped_column(String, unique=True)
    source: Mapped[str] = mapped_column(String(6))
    camera_id: Mapped[int | None] = mapped_column(ForeignKey("cameras.id", ondelete="RESTRICT"), nullable=True)
    camera_name: Mapped[str | None] = mapped_column(String(100), nullable=True)
    stream_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    tracking_generation: Mapped[str | None] = mapped_column(String(36), nullable=True)
    track_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    crossing_index: Mapped[int | None] = mapped_column(Integer, nullable=True)
    recorded_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    raw: Mapped[dict] = mapped_column(JSON)
    employee_id: Mapped[int | None] = mapped_column(ForeignKey("employees.id", ondelete="RESTRICT"), nullable=True)
    employee_name: Mapped[str | None] = mapped_column(String(150), nullable=True)
    direction: Mapped[str | None] = mapped_column(String(5), nullable=True)
    occurred_at: Mapped[datetime] = mapped_column(DateTime)
    status: Mapped[str] = mapped_column(String(8))
    revision: Mapped[int] = mapped_column(Integer, default=0, server_default=text("0"))
    reason_codes: Mapped[list] = mapped_column(JSON, default=list)
    projection_reasons: Mapped[list] = mapped_column(JSON, default=list)


class AttendanceEvidence(Base):
    __tablename__ = "attendance_evidence"

    event_id: Mapped[str] = mapped_column(ForeignKey("attendance_events.id", ondelete="CASCADE"), primary_key=True)
    before_jpeg: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)
    after_jpeg: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)
    face_jpeg: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime, index=True)
    purged_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class AttendanceRequest(Base):
    __tablename__ = "attendance_requests"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    operation: Mapped[str] = mapped_column(String(100))
    payload: Mapped[dict] = mapped_column(JSON)
    result: Mapped[dict | list] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class AttendanceDecision(Base):
    __tablename__ = "attendance_decisions"
    __table_args__ = (
        CheckConstraint("(event_id IS NOT NULL AND gap_id IS NULL) OR "
                        "(event_id IS NULL AND gap_id IS NOT NULL)", name="ck_attendance_decision_target"),
        CheckConstraint("revision >= 0", name="ck_attendance_decision_revision"),
        CheckConstraint("length(trim(actor)) BETWEEN 1 AND 100", name="ck_attendance_decision_actor"),
        CheckConstraint("length(trim(reason)) BETWEEN 1 AND 1000", name="ck_attendance_decision_reason"),
        UniqueConstraint("event_id", "revision", name="uq_attendance_decision_event_revision"),
        UniqueConstraint("gap_id", "revision", name="uq_attendance_decision_gap_revision"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    request_id: Mapped[str] = mapped_column(ForeignKey("attendance_requests.id", ondelete="RESTRICT"))
    event_id: Mapped[str | None] = mapped_column(ForeignKey("attendance_events.id", ondelete="RESTRICT"), nullable=True)
    gap_id: Mapped[int | None] = mapped_column(ForeignKey("attendance_coverage_gaps.id", ondelete="RESTRICT"), nullable=True)
    revision: Mapped[int] = mapped_column(Integer)
    actor: Mapped[str] = mapped_column(String(100))
    reason: Mapped[str] = mapped_column(String(1000))
    before: Mapped[dict] = mapped_column(JSON)
    after: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class WorkSession(Base):
    __tablename__ = "work_sessions"
    __table_args__ = (
        CheckConstraint("status IN ('OPEN', 'CLOSED', 'REVIEW')", name="ck_work_session_status"),
        CheckConstraint("ended_at IS NULL OR ended_at >= started_at", name="ck_work_session_times"),
        Index("uq_work_sessions_open_employee", "employee_id", unique=True,
              sqlite_where=text("ended_at IS NULL")),
    )

    entry_event_id: Mapped[str] = mapped_column(ForeignKey("attendance_events.id", ondelete="RESTRICT"), primary_key=True)
    exit_event_id: Mapped[str | None] = mapped_column(ForeignKey("attendance_events.id", ondelete="RESTRICT"), nullable=True, unique=True)
    employee_id: Mapped[int] = mapped_column(ForeignKey("employees.id", ondelete="RESTRICT"), index=True)
    employee_name: Mapped[str] = mapped_column(String(150))
    started_at: Mapped[datetime] = mapped_column(DateTime)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    status: Mapped[str] = mapped_column(String(6))
    reason_codes: Mapped[list] = mapped_column(JSON, default=list)


class AttendanceCoverageGap(Base):
    __tablename__ = "attendance_coverage_gaps"
    __table_args__ = (
        CheckConstraint("singleton = 1", name="ck_attendance_gap_singleton"),
        CheckConstraint("ended_at IS NULL OR ended_at >= started_at", name="ck_attendance_gap_times"),
        CheckConstraint("revision >= 0", name="ck_attendance_gap_revision"),
        CheckConstraint("reviewed IN (0, 1)", name="ck_attendance_gap_reviewed"),
        Index("uq_attendance_coverage_gaps_open", "singleton", unique=True,
              sqlite_where=text("ended_at IS NULL")),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    singleton: Mapped[int] = mapped_column(Integer, default=1, server_default=text("1"))
    started_at: Mapped[datetime] = mapped_column(DateTime)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    reason_codes: Mapped[list] = mapped_column(JSON, default=list)
    source_fault_ids: Mapped[list] = mapped_column(JSON, default=list)
    reviewed: Mapped[bool] = mapped_column(Boolean, default=False, server_default=text("0"))
    revision: Mapped[int] = mapped_column(Integer, default=0, server_default=text("0"))


class AttendanceEmployeeReference(Base):
    __tablename__ = "attendance_employee_references"

    event_id: Mapped[str] = mapped_column(ForeignKey("attendance_events.id", ondelete="RESTRICT"), primary_key=True)
    employee_id: Mapped[int] = mapped_column(ForeignKey("employees.id", ondelete="RESTRICT"), primary_key=True)
    employee_name: Mapped[str] = mapped_column(String(150))


class AttendanceCommission(Base):
    __tablename__ = "attendance_commissions"
    __table_args__ = (
        CheckConstraint("length(trim(actor)) BETWEEN 1 AND 100", name="ck_attendance_commission_actor"),
        CheckConstraint("length(trim(reason)) BETWEEN 1 AND 1000", name="ck_attendance_commission_reason"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    request_id: Mapped[str] = mapped_column(String(36), unique=True)
    actor: Mapped[str] = mapped_column(String(100))
    reason: Mapped[str] = mapped_column(String(1000))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    configuration_fingerprint: Mapped[dict] = mapped_column(JSON)
    acceptance_record: Mapped[dict] = mapped_column(JSON)


class AttendanceRuntime(Base):
    __tablename__ = "attendance_runtime"
    __table_args__ = (
        CheckConstraint("id = 1", name="ck_attendance_runtime_singleton"),
        CheckConstraint("schema_version = 1", name="ck_attendance_runtime_version"),
        CheckConstraint("gallery_revision >= 0", name="ck_attendance_runtime_gallery"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, default=1)
    schema_version: Mapped[int] = mapped_column(Integer, default=1, server_default=text("1"))
    gallery_revision: Mapped[int] = mapped_column(Integer, default=0, server_default=text("0"))
    commission_id: Mapped[int | None] = mapped_column(ForeignKey("attendance_commissions.id", ondelete="RESTRICT"), nullable=True)
    monitoring_started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    heartbeat_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    stopped_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
