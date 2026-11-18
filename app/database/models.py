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
        CheckConstraint("role IS NULL OR role IN ('ENTRY', 'EXIT', 'BOTH')", name="ck_attendance_camera_role"),
        CheckConstraint("door_kind IS NULL OR door_kind IN ('DOOR', 'PASSAGE')",
                        name="ck_attendance_camera_door_kind"),
        CheckConstraint("revision >= 0 AND source_revision >= 1", name="ck_attendance_camera_revisions"),
        CheckConstraint("enabled IN (0, 1)", name="ck_attendance_camera_enabled"),
        CheckConstraint(
            "enabled = 0 OR (role IS NOT NULL AND line_start IS NOT NULL AND "
            "line_end IS NOT NULL AND inside_point IS NOT NULL AND "
            "door_kind IS NOT NULL AND door_top IS NOT NULL AND "
            "reference_width IS NOT NULL AND reference_height IS NOT NULL AND "
            "reference_width > 0 AND reference_height > 0)",
            name="ck_attendance_camera_complete",
        ),
    )

    camera_id: Mapped[int] = mapped_column(ForeignKey("cameras.id", ondelete="RESTRICT"), primary_key=True)
    role: Mapped[str | None] = mapped_column(String(5), nullable=True)
    enabled: Mapped[bool] = mapped_column(Boolean, default=False, server_default=text("0"))
    revision: Mapped[int] = mapped_column(Integer, default=0, server_default=text("0"))
    source_revision: Mapped[int] = mapped_column(Integer, default=1, server_default=text("1"))
    line_start: Mapped[list | None] = mapped_column(JSON(none_as_null=True), nullable=True)
    line_end: Mapped[list | None] = mapped_column(JSON(none_as_null=True), nullable=True)
    inside_point: Mapped[list | None] = mapped_column(JSON(none_as_null=True), nullable=True)
    door_kind: Mapped[str | None] = mapped_column(String(7), nullable=True)
    door_top: Mapped[list | None] = mapped_column(JSON(none_as_null=True), nullable=True)
    reference_width: Mapped[int | None] = mapped_column(Integer, nullable=True)
    reference_height: Mapped[int | None] = mapped_column(Integer, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)


class AttendanceRuntime(Base):
    __tablename__ = "attendance_runtime"
    __table_args__ = (
        CheckConstraint("id = 1", name="ck_attendance_runtime_singleton"),
        CheckConstraint("schema_version = 3", name="ck_attendance_runtime_version"),
        CheckConstraint("gallery_revision >= 0", name="ck_attendance_runtime_gallery"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, default=1)
    schema_version: Mapped[int] = mapped_column(Integer, default=3, server_default=text("3"))
    gallery_revision: Mapped[int] = mapped_column(Integer, default=0, server_default=text("0"))


class PresenceSegment(Base):
    """One employee seen on one camera: consecutive sightings no more than 60 s apart."""
    __tablename__ = "attendance_presence_segments"
    __table_args__ = (
        CheckConstraint("last_seen_at >= started_at", name="ck_presence_segment_order"),
        CheckConstraint("face_seen IN (0, 1)", name="ck_presence_segment_face"),
        Index("ix_presence_segments_employee_start", "employee_id", "started_at"),
        Index("ix_presence_segments_start", "started_at"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    employee_id: Mapped[int] = mapped_column(ForeignKey("employees.id", ondelete="RESTRICT"))
    camera_id: Mapped[int] = mapped_column(ForeignKey("cameras.id", ondelete="RESTRICT"))
    started_at: Mapped[datetime] = mapped_column(DateTime)
    last_seen_at: Mapped[datetime] = mapped_column(DateTime)
    face_seen: Mapped[bool] = mapped_column(Boolean)


class DoorCrossing(Base):
    __tablename__ = "attendance_door_crossings"
    __table_args__ = (
        CheckConstraint("direction IN ('ENTRY', 'EXIT')", name="ck_door_crossing_direction"),
        Index("ix_door_crossings_employee_time", "employee_id", "occurred_at"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    source_key: Mapped[str] = mapped_column(String(300), unique=True)
    camera_id: Mapped[int] = mapped_column(ForeignKey("cameras.id", ondelete="RESTRICT"))
    employee_id: Mapped[int] = mapped_column(ForeignKey("employees.id", ondelete="RESTRICT"))
    direction: Mapped[str] = mapped_column(String(5))
    occurred_at: Mapped[datetime] = mapped_column(DateTime)
    recorded_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class ManualPresenceInterval(Base):
    """Audited operator interval; revoking keeps the row with the revoker and reason."""
    __tablename__ = "attendance_manual_intervals"
    __table_args__ = (
        CheckConstraint("ended_at > started_at", name="ck_manual_interval_order"),
        CheckConstraint("(revoked_at IS NULL AND revoked_by IS NULL AND revoke_reason IS NULL) OR "
                        "(revoked_at IS NOT NULL AND revoked_by IS NOT NULL AND revoke_reason IS NOT NULL)",
                        name="ck_manual_interval_revoke"),
        Index("ix_manual_intervals_employee_start", "employee_id", "started_at"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    request_id: Mapped[str] = mapped_column(String(36), unique=True)
    employee_id: Mapped[int] = mapped_column(ForeignKey("employees.id", ondelete="RESTRICT"))
    started_at: Mapped[datetime] = mapped_column(DateTime)
    ended_at: Mapped[datetime] = mapped_column(DateTime)
    actor: Mapped[str] = mapped_column(String(100))
    reason: Mapped[str] = mapped_column(String(1000))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    revoked_by: Mapped[str | None] = mapped_column(String(100), nullable=True)
    revoke_reason: Mapped[str | None] = mapped_column(String(1000), nullable=True)


class PresenceExclusion(Base):
    """Audited operator exclusion of a wrong automatic span; same lifecycle as a manual interval."""
    __tablename__ = "attendance_exclusions"
    __table_args__ = (
        CheckConstraint("ended_at > started_at", name="ck_exclusion_order"),
        CheckConstraint("(revoked_at IS NULL AND revoked_by IS NULL AND revoke_reason IS NULL) OR "
                        "(revoked_at IS NOT NULL AND revoked_by IS NOT NULL AND revoke_reason IS NOT NULL)",
                        name="ck_exclusion_revoke"),
        Index("ix_exclusions_employee_start", "employee_id", "started_at"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    request_id: Mapped[str] = mapped_column(String(36), unique=True)
    employee_id: Mapped[int] = mapped_column(ForeignKey("employees.id", ondelete="RESTRICT"))
    started_at: Mapped[datetime] = mapped_column(DateTime)
    ended_at: Mapped[datetime] = mapped_column(DateTime)
    actor: Mapped[str] = mapped_column(String(100))
    reason: Mapped[str] = mapped_column(String(1000))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    revoked_by: Mapped[str | None] = mapped_column(String(100), nullable=True)
    revoke_reason: Mapped[str | None] = mapped_column(String(1000), nullable=True)


class FaceModelMetadata(Base):
    __tablename__ = "face_model_metadata"
    face_id: Mapped[int] = mapped_column(ForeignKey("employee_faces.id", ondelete="CASCADE"), primary_key=True)
    model_id: Mapped[str] = mapped_column(String(64))
    alignment_id: Mapped[str] = mapped_column(String(40))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    __table_args__ = (CheckConstraint("length(model_id) = 64"),
                      CheckConstraint("length(alignment_id) BETWEEN 1 AND 40"))


class CameraScene(Base):
    __tablename__ = "camera_scenes"
    camera_id: Mapped[int] = mapped_column(ForeignKey("cameras.id", ondelete="CASCADE"), primary_key=True)
    revision: Mapped[int] = mapped_column(Integer, default=0)
    source_revision: Mapped[int] = mapped_column(Integer)
    reference_width: Mapped[int] = mapped_column(Integer)
    reference_height: Mapped[int] = mapped_column(Integer)
    geometry: Mapped[dict] = mapped_column(JSON)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)
    __table_args__ = (CheckConstraint("revision >= 0"), CheckConstraint("source_revision >= 0"),
                      CheckConstraint("reference_width > 0"), CheckConstraint("reference_height > 0"),
                      CheckConstraint("json_valid(geometry)"))


class CameraProfile(Base):
    """Operator grouping/AI settings; DDL is frozen in operator_schema (queries only here)."""
    __tablename__ = "camera_profiles"
    camera_id: Mapped[int] = mapped_column(ForeignKey("cameras.id", ondelete="CASCADE"), primary_key=True)
    workplace_id: Mapped[int | None] = mapped_column(ForeignKey("workplaces.id", ondelete="RESTRICT"), nullable=True)
    zone: Mapped[str] = mapped_column(String(8))
    ai_enabled: Mapped[bool] = mapped_column(Boolean)
    ai_fps: Mapped[float | None] = mapped_column(Float, nullable=True)
    time_tracking: Mapped[bool] = mapped_column(Boolean)
    revision: Mapped[int] = mapped_column(Integer, default=0)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)


class HumanVisit(Base):
    __tablename__ = "human_visits"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    camera_id: Mapped[int] = mapped_column(ForeignKey("cameras.id", ondelete="RESTRICT"))
    tracking_generation: Mapped[str] = mapped_column(String(36))
    track_id: Mapped[int] = mapped_column(Integer)
    source_revision: Mapped[int] = mapped_column(Integer)
    stream_id: Mapped[str] = mapped_column(String(36))
    started_at: Mapped[datetime] = mapped_column(DateTime)
    last_seen_at: Mapped[datetime] = mapped_column(DateTime)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime)
    end_reason: Mapped[str | None] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(20), default="PROVISIONAL")
    employee_id: Mapped[int | None] = mapped_column(ForeignKey("employees.id", ondelete="RESTRICT"))
    identity_state: Mapped[str] = mapped_column(String(20), default="UNKNOWN")
    global_id: Mapped[int | None] = mapped_column(ForeignKey("global_persons.id", ondelete="SET NULL"))
    last_bbox: Mapped[list] = mapped_column(JSON)
    max_person_score: Mapped[float] = mapped_column(Float)
    snapshot: Mapped[bytes | None] = mapped_column(LargeBinary)
    face_thumbnail: Mapped[bytes | None] = mapped_column(LargeBinary)
    face_quality: Mapped[dict | None] = mapped_column(JSON)
    image_expires_at: Mapped[datetime] = mapped_column(DateTime)
    images_purged_at: Mapped[datetime | None] = mapped_column(DateTime)
    __table_args__ = (
        UniqueConstraint("camera_id", "tracking_generation", "track_id"),
        CheckConstraint("source_revision >= 1"),
        CheckConstraint("last_seen_at >= started_at"),
        CheckConstraint("ended_at IS NULL OR ended_at >= last_seen_at"),
        CheckConstraint("status IN ('PROVISIONAL','ACTIVE','CLOSED')"),
        CheckConstraint("status != 'CLOSED' OR ended_at IS NOT NULL"),
        CheckConstraint("identity_state IN ('UNKNOWN','FACE_CONFIRMED','CONFLICT')"),
        CheckConstraint("json_valid(last_bbox) AND json_type(last_bbox) = 'array' AND json_array_length(last_bbox) = 4"),
        CheckConstraint("max_person_score BETWEEN 0 AND 1"),
        CheckConstraint("snapshot IS NULL OR length(snapshot) <= 262144"),
        CheckConstraint("face_thumbnail IS NULL OR length(face_thumbnail) <= 262144"),
        Index("ix_human_visits_last_seen_id", "last_seen_at", "id"),
        Index("ix_human_visits_camera_last_seen", "camera_id", "last_seen_at"),
        Index("ix_human_visits_employee_last_seen", "employee_id", "last_seen_at"),
    )


class HumanVisitIdentityRevision(Base):
    __tablename__ = "human_visit_identity_revisions"
    id: Mapped[int] = mapped_column(primary_key=True)
    visit_id: Mapped[str] = mapped_column(ForeignKey("human_visits.id", ondelete="RESTRICT"))
    revision: Mapped[int] = mapped_column(Integer)
    mode: Mapped[str] = mapped_column(String(8))
    employee_id: Mapped[int | None] = mapped_column(ForeignKey("employees.id", ondelete="RESTRICT"))
    actor: Mapped[str] = mapped_column(String(100))
    reason: Mapped[str] = mapped_column(String(1000))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    request_id: Mapped[str] = mapped_column(String(36), unique=True)
    __table_args__ = (
        UniqueConstraint("visit_id", "revision"),
        CheckConstraint("revision >= 1"),
        CheckConstraint("mode IN ('EMPLOYEE','UNKNOWN','AI')"),
        CheckConstraint("(mode = 'EMPLOYEE' AND employee_id IS NOT NULL) OR "
                        "(mode != 'EMPLOYEE' AND employee_id IS NULL)"),
        Index("ix_visit_identity_revision", "visit_id", "revision"),
    )


class HumanVisitEvent(Base):
    __tablename__ = "human_visit_events"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    visit_id: Mapped[str] = mapped_column(ForeignKey("human_visits.id", ondelete="CASCADE"))
    dedupe_key: Mapped[str] = mapped_column(String(200), unique=True)
    type: Mapped[str] = mapped_column(String(24))
    occurred_at: Mapped[datetime] = mapped_column(DateTime)
    payload: Mapped[dict] = mapped_column(JSON)
    __table_args__ = (
        CheckConstraint("length(dedupe_key) BETWEEN 1 AND 200"),
        CheckConstraint("type IN ('FIRST_SEEN','IDENTIFIED','IDENTITY_CONFLICT','LOST','RECOVERED','CLOSED','LINKED')"),
        CheckConstraint("json_valid(payload) AND json_type(payload) = 'object' AND length(CAST(payload AS BLOB)) <= 16384"),
        Index("ix_human_visit_events_visit_occurred", "visit_id", "occurred_at", "id"),
    )


class HumanJournalGap(Base):
    __tablename__ = "human_journal_gaps"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    camera_id: Mapped[int] = mapped_column(ForeignKey("cameras.id", ondelete="RESTRICT"))
    tracking_generation: Mapped[str] = mapped_column(String(36))
    started_at: Mapped[datetime] = mapped_column(DateTime)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime)
    reason: Mapped[str] = mapped_column(String(64))
    __table_args__ = (
        CheckConstraint("ended_at IS NULL OR ended_at >= started_at"),
        Index("ix_human_journal_gaps_camera_started", "camera_id", "started_at"),
    )


def _sha(column: str) -> str:
    return (f"typeof({column}) = 'text' AND length({column}) = 64 AND "
            f"length(CAST({column} AS BLOB)) = 64 AND {column} NOT GLOB '*[^0-9a-f]*'")


def _ms(column: str) -> str:
    """Naive UTC text with at most millisecond precision (SQLAlchemy may pad zeros to six digits)."""
    return (f"typeof({column}) = 'text' AND datetime({column}) IS NOT NULL AND substr({column}, 11, 1) = ' ' "
            f"AND length({column}) IN (19, 23, 26) AND (length({column}) < 26 OR substr({column}, 24, 3) = '000')")


def _rectangle(column: str) -> str:
    kinds = " AND ".join(f"json_type({column}, '$[{i}]') IN ('integer', 'real')" for i in range(4))
    return (f"json_valid({column}) AND json_type({column}) = 'array' AND json_array_length({column}) = 4 "
            f"AND {kinds} AND json_extract({column}, '$[0]') >= 0 AND json_extract({column}, '$[0]') < "
            f"json_extract({column}, '$[2]') AND json_extract({column}, '$[2]') <= 1 "
            f"AND json_extract({column}, '$[1]') >= 0 AND json_extract({column}, '$[1]') < "
            f"json_extract({column}, '$[3]') AND json_extract({column}, '$[3]') <= 1")


def _audit(column: str, limit: int) -> str:
    return (f"typeof({column}) = 'text' AND length({column}) BETWEEN 1 AND {limit} AND "
            f"instr({column}, char(0)) = 0 AND trim({column}, ' ' || char(9, 10, 11, 12, 13)) != ''")


class RecognitionLearningRuntime(Base):
    """Singleton revision of supervised recognition evidence; seeded only by the frozen operator install."""
    __tablename__ = "operator_learning_runtime"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=False)
    revision: Mapped[int] = mapped_column(Integer, default=0, server_default=text("0"))
    __table_args__ = (CheckConstraint("id = 1"), CheckConstraint("typeof(revision) = 'integer' AND revision >= 0"))


class HumanVisitPhotoMetadata(Base):
    """Exact provenance of HumanVisit.snapshot; absent for pre-v8 images."""
    __tablename__ = "human_visit_photo_metadata"
    visit_id: Mapped[str] = mapped_column(ForeignKey("human_visits.id", ondelete="RESTRICT"), primary_key=True)
    sha256: Mapped[str] = mapped_column(String(64))
    captured_at: Mapped[datetime] = mapped_column(DateTime)
    frame_seq: Mapped[int] = mapped_column(Integer)
    native_width: Mapped[int] = mapped_column(Integer)
    native_height: Mapped[int] = mapped_column(Integer)
    __table_args__ = (
        CheckConstraint(_sha("sha256")), CheckConstraint(_ms("captured_at")),
        CheckConstraint("typeof(frame_seq) = 'integer' AND frame_seq >= 0"),
        CheckConstraint("typeof(native_width) = 'integer' AND native_width > 0"),
        CheckConstraint("typeof(native_height) = 'integer' AND native_height > 0"),
    )


class RecognitionFeedback(Base):
    __tablename__ = "recognition_feedback"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    visit_id: Mapped[str] = mapped_column(ForeignKey("human_visits.id", ondelete="RESTRICT"))
    employee_id: Mapped[int] = mapped_column(ForeignKey("employees.id", ondelete="RESTRICT"))
    action: Mapped[str] = mapped_column(String(6))
    source: Mapped[str] = mapped_column(String(10))
    source_sha256: Mapped[str] = mapped_column(String(64))
    image_sha256: Mapped[str] = mapped_column(String(64))
    crop: Mapped[list] = mapped_column(JSON)
    captured_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    image_jpeg: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)
    actor: Mapped[str] = mapped_column(String(100))
    reason: Mapped[str] = mapped_column(String(1000))
    created_at: Mapped[datetime] = mapped_column(DateTime)
    identity_revision: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # Audited external reference, deliberately not a FK: face IDs can be reused, so face_key must also match.
    face_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    face_key: Mapped[str | None] = mapped_column(String(64), nullable=True)
    face_status: Mapped[str] = mapped_column(String(24))
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    revoked_by: Mapped[str | None] = mapped_column(String(100), nullable=True)
    revoke_reason: Mapped[str | None] = mapped_column(String(1000), nullable=True)
    __table_args__ = (
        CheckConstraint("length(id) = 36"),
        CheckConstraint("action IN ('ACCEPT', 'REJECT')"),
        CheckConstraint("source IN ('PHOTO_CROP', 'UPLOAD')"),
        CheckConstraint(_sha("source_sha256")), CheckConstraint(_sha("image_sha256")),
        CheckConstraint(_rectangle("crop")),
        CheckConstraint(f"captured_at IS NULL OR ({_ms('captured_at')})"),
        CheckConstraint("image_jpeg IS NULL OR (typeof(image_jpeg) = 'blob' AND length(image_jpeg) <= 262144)"),
        CheckConstraint(_audit("actor", 100)), CheckConstraint(_audit("reason", 1000)),
        CheckConstraint("identity_revision IS NULL OR identity_revision >= 1"),
        CheckConstraint("face_id IS NULL OR face_id >= 1"),
        CheckConstraint(f"face_key IS NULL OR ({_sha('face_key')})"),
        CheckConstraint("face_status IN ('ADDED', 'NEGATIVE_ADDED', 'INELIGIBLE', 'LIMIT', 'CONFLICT', 'UNAVAILABLE')"),
        CheckConstraint("(face_id IS NULL) = (face_key IS NULL)"),
        CheckConstraint("(revoked_at IS NULL AND revoked_by IS NULL AND revoke_reason IS NULL) OR "
                        "(revoked_at IS NOT NULL AND revoked_by IS NOT NULL AND revoke_reason IS NOT NULL AND "
                        f"{_audit('revoked_by', 100)} AND {_audit('revoke_reason', 1000)})"),
        Index("ix_recognition_feedback_employee_created", "employee_id", "created_at", "id"),
        Index("ix_recognition_feedback_visit_created", "visit_id", "created_at", "id"),
    )


class RecognitionExample(Base):
    """Vector evidence of one feedback row; accepted face positives stay in EmployeeFace."""
    __tablename__ = "recognition_examples"
    id: Mapped[int] = mapped_column(primary_key=True)
    feedback_id: Mapped[str] = mapped_column(ForeignKey("recognition_feedback.id", ondelete="RESTRICT"))
    modality: Mapped[str] = mapped_column(String(4))
    polarity: Mapped[str] = mapped_column(String(8))
    model_id: Mapped[str] = mapped_column(String(64))
    alignment_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    embedding: Mapped[bytes] = mapped_column(LargeBinary)
    __table_args__ = (
        CheckConstraint("modality IN ('BODY', 'FACE')"),
        CheckConstraint("polarity IN ('POSITIVE', 'NEGATIVE')"),
        CheckConstraint(_sha("model_id")),
        CheckConstraint("alignment_id IS NULL OR length(alignment_id) BETWEEN 1 AND 64"),
        CheckConstraint("typeof(embedding) = 'blob' AND length(embedding) = 2048"),
        CheckConstraint("(modality = 'BODY' AND alignment_id IS NULL) OR "
                        "(modality = 'FACE' AND alignment_id IS NOT NULL)"),
        CheckConstraint("polarity = 'NEGATIVE' OR modality = 'BODY'"),
        UniqueConstraint("feedback_id", "modality"),
        Index("ix_recognition_examples_model", "modality", "model_id", "feedback_id"),
    )
