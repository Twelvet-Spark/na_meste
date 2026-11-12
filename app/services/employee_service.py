"""Atomic employee identity/face-set transactions; original images are never stored."""
import io
import warnings
from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError
from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, field_validator, model_validator
from sqlalchemy import update
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.ai.face_quality import measure_face_quality, quality_reasons
from app.ai.face_recognition import EmbeddingStore, normalized_embedding
from app.config import settings
from app.database.database import immediate_session
from app.database.models import (
    AttendanceEmployeeReference, AttendanceEvent, AttendanceRuntime, Employee,
    EmployeeFace, EmployeeWorkplace, GlobalPerson, OperatorRequest, PresenceSession,
    WorkSession, Workplace,
)
from app.services.attendance_evidence import encode_evidence
from app.services.workplace_service import WorkplaceOut, replace_assignments, validate_assignments
from app.time_utils import utcnow


class EmployeeCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    first_name: str = Field(min_length=1, max_length=74)
    last_name: str = Field(min_length=1, max_length=74)
    external_id: str | None = Field(default=None, max_length=64)
    active: StrictBool = True
    workplace_ids: list[StrictInt] = Field(default_factory=list)
    request_id: UUID

    @field_validator("first_name", "last_name", mode="before")
    @classmethod
    def trim_name(cls, value):
        return value.strip() if isinstance(value, str) else value

    @field_validator("workplace_ids")
    @classmethod
    def unique_places(cls, value):
        if len(set(value)) != len(value) or any(item <= 0 for item in value):
            raise ValueError("Рабочие места должны быть уникальными положительными ID")
        return sorted(value)


class EmployeeUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    expected_revision: StrictInt = Field(ge=0)
    first_name: str | None = Field(default=None, min_length=1, max_length=74)
    last_name: str | None = Field(default=None, min_length=1, max_length=74)
    external_id: str | None = Field(default=None, max_length=64)
    active: StrictBool | None = None
    workplace_ids: list[StrictInt] | None = None

    @field_validator("first_name", "last_name", mode="before")
    @classmethod
    def trim_name(cls, value):
        return value.strip() if isinstance(value, str) else value

    @field_validator("workplace_ids")
    @classmethod
    def unique_places(cls, value):
        if value is None:
            raise ValueError("workplace_ids не может быть null")
        return EmployeeCreate.unique_places(value)

    @model_validator(mode="after")
    def complete_fields(self):
        names = self.model_fields_set & {"first_name", "last_name"}
        if names and (len(names) != 2 or self.first_name is None or self.last_name is None):
            raise ValueError("Укажите непустые имя и фамилию вместе")
        if "active" in self.model_fields_set and self.active is None:
            raise ValueError("active не может быть null")
        return self


class FaceSetUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    expected_revision: StrictInt = Field(ge=0)
    keep_face_ids: list[StrictInt]

    @field_validator("keep_face_ids")
    @classmethod
    def unique_faces(cls, value):
        if len(value) != len(set(value)) or any(item <= 0 for item in value):
            raise ValueError("Сохранённые фотографии должны иметь уникальные положительные ID")
        return value


class FaceOut(BaseModel):
    id: int
    created_at: datetime
    thumbnail_url: str


class EmployeeOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    name: str
    first_name: str | None
    last_name: str | None
    external_id: str | None
    active: bool
    revision: int
    created_at: datetime
    updated_at: datetime
    faces_count: int = 0
    faces: list[FaceOut] = Field(default_factory=list)
    workplaces: list[WorkplaceOut] = Field(default_factory=list)
    needs_name_completion: bool = False
    needs_photo_remediation: bool = False


class FaceEnrollResult(BaseModel):
    filename: str
    success: bool
    face_id: int | None = None
    error: str | None = None
    warnings: list[str] = Field(default_factory=list)


@dataclass(frozen=True)
class PreparedFace:
    filename: str
    embedding: bytes
    thumbnail: bytes


def get_employee(db, employee_id):
    return db.get(Employee, employee_id)


def list_employees(db):
    return db.query(Employee).order_by(Employee.name).all()


def employee_out(db, employee):
    data = EmployeeOut.model_validate(employee)
    samples = db.query(EmployeeFace.id, EmployeeFace.created_at).filter(
        EmployeeFace.employee_id == employee.id).order_by(EmployeeFace.id).all()
    data.faces = [FaceOut(id=row.id, created_at=row.created_at,
                          thumbnail_url=f"/api/employees/{employee.id}/faces/{row.id}/thumbnail") for row in samples]
    data.faces_count = len(samples)
    data.workplaces = [WorkplaceOut.model_validate(row) for row in db.query(Workplace).join(
        EmployeeWorkplace, Workplace.id == EmployeeWorkplace.workplace_id).filter(
            EmployeeWorkplace.employee_id == employee.id).order_by(Workplace.name, Workplace.id)]
    data.needs_name_completion = not (employee.first_name and employee.last_name)
    data.needs_photo_remediation = not 1 <= len(samples) <= 15
    return data.model_dump(mode="json")


def photo_failure(errors):
    return HTTPException(422, {"code": "PHOTO_VALIDATION_FAILED", "committed": False, "files": errors})


def prepare_face(image_bytes: bytes, face_engine, filename: str) -> PreparedFace | FaceEnrollResult:
    def failed(message):
        return FaceEnrollResult(filename=filename, success=False, error=message)

    if not image_bytes or len(image_bytes) > 15 * 1024 * 1024:
        return failed("Файл пуст или превышает 15 МиБ")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(image_bytes)) as header:
                if header.format not in {"JPEG", "PNG"}:
                    return failed("Допустимы только JPEG и PNG")
                if header.width * header.height > 20_000_000 or getattr(header, "is_animated", False):
                    return failed("Изображение должно быть неподвижным, не более 20 миллионов пикселей")
                header.verify()
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError,
            Image.DecompressionBombError, Image.DecompressionBombWarning):
        return failed("Не удалось прочитать безопасное изображение JPEG/PNG")
    try:
        image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    except cv2.error:
        return failed("Не удалось прочитать изображение")
    if image is None:
        return failed("Не удалось прочитать изображение")
    observations = face_engine.detect(image)
    if len(observations) != 1:
        return failed("На фотографии должно быть ровно одно лицо")
    observation = observations[0]
    reasons = quality_reasons(measure_face_quality(observation), min_size=max(80, settings.face_min_size),
                              det_score=.80, blur_min=20.)
    embedding = normalized_embedding(observation.embedding)
    errors = {"FACE_TOO_SMALL": "Лицо слишком маленькое — минимум 80 px",
              "BAD_ALIGNMENT": "Не удалось выровнять лицо",
              "LOW_FACE_DETECTION": "Низкая уверенность детекции — минимум 0.80",
              "BLURRY_FACE": "Изображение размыто — загрузите более чёткое фото"}
    if reasons or embedding is None:
        return failed(errors[reasons[0]] if reasons else "Некорректный эмбеддинг лица")
    thumbnail = encode_evidence(before=None, after=None, face=observation.aligned).face_jpeg
    if thumbnail is None:
        return failed("Не удалось сохранить выровненное лицо")
    return PreparedFace(filename, embedding.tobytes(), thumbnail)


def committed_request(db, request_id, digest):
    row = db.get(OperatorRequest, str(request_id))
    if row is not None and (row.operation != "employee.create" or row.payload_sha256 != digest):
        raise HTTPException(409, "Этот request_id уже использован с другими данными")
    return row.result if row is not None else None


def _advance_gallery(db):
    db.execute(update(AttendanceRuntime).where(AttendanceRuntime.id == 1).values(
        gallery_revision=AttendanceRuntime.gallery_revision + 1))


def _check_faces(db, employee_id, prepared_faces):
    gallery = EmbeddingStore()
    gallery.refresh(db)
    errors = []
    for index, sample in enumerate(prepared_faces):
        match = gallery.match(np.frombuffer(sample.embedding, dtype=np.float32), exclude_employee_id=employee_id)
        if match.employee_id is not None and match.score >= .55 and match.margin >= .08:
            errors.append({"index": index, "filename": sample.filename,
                           "error": "Лицо совпадает с другим активным сотрудником"})
    if errors:
        raise photo_failure(errors)


def create_employee_with_faces(db: Session, data: EmployeeCreate, prepared_faces, payload_sha256) -> Employee:
    bind = db.get_bind().engine
    db.rollback()
    try:
        with immediate_session(bind) as writer:
            replay = committed_request(writer, data.request_id, payload_sha256)
            if replay is not None:
                return writer.get(Employee, replay["id"])
            if not 1 <= len(prepared_faces) <= 15:
                raise HTTPException(422, "Нужно от 1 до 15 фотографий")
            validate_assignments(writer, data.workplace_ids)
            _check_faces(writer, None, prepared_faces)
            employee = Employee(first_name=data.first_name, last_name=data.last_name,
                                name=f"{data.last_name} {data.first_name}", external_id=data.external_id,
                                active=data.active, revision=1, created_at=utcnow(), updated_at=utcnow())
            writer.add(employee)
            writer.flush()
            writer.add_all(EmployeeFace(employee_id=employee.id, embedding=row.embedding,
                                        thumbnail=row.thumbnail, created_at=utcnow()) for row in prepared_faces)
            replace_assignments(writer, employee.id, data.workplace_ids)
            _advance_gallery(writer)
            writer.flush()
            writer.add(OperatorRequest(id=str(data.request_id), operation="employee.create",
                                       payload_sha256=payload_sha256, result=employee_out(writer, employee)))
        db.expire_all()
        return employee
    except SQLAlchemyError as exc:
        raise HTTPException(503, "База данных недоступна") from exc


def replace_employee_faces(db: Session, employee_id, expected_revision, keep_face_ids, prepared_faces) -> Employee:
    bind = db.get_bind().engine
    db.rollback()
    try:
        with immediate_session(bind) as writer:
            current = writer.get(Employee, employee_id)
            _check_revision(current, expected_revision)
            owned = {row[0] for row in writer.query(EmployeeFace.id).filter_by(employee_id=employee_id)}
            if (len(keep_face_ids) != len(set(keep_face_ids)) or not set(keep_face_ids) <= owned
                    or not 1 <= len(keep_face_ids) + len(prepared_faces) <= 15):
                raise HTTPException(422, "Итоговый набор должен содержать 1–15 собственных фотографий без повторов")
            _check_faces(writer, employee_id, prepared_faces)
            # Insert before deleting so SQLite cannot recycle an omitted thumbnail ID.
            writer.add_all(EmployeeFace(employee_id=employee_id, embedding=row.embedding,
                                        thumbnail=row.thumbnail, created_at=utcnow()) for row in prepared_faces)
            writer.flush()
            removed = owned - set(keep_face_ids)
            if removed:
                writer.query(EmployeeFace).filter(EmployeeFace.id.in_(removed)).delete(synchronize_session=False)
            current.revision += 1
            current.updated_at = utcnow()
            _advance_gallery(writer)
            writer.flush()
        db.expire_all()
        return current
    except SQLAlchemyError as exc:
        raise HTTPException(503, "База данных недоступна") from exc


def _check_revision(current, revision):
    if current is None:
        raise HTTPException(404, "Сотрудник не найден")
    if current.revision != revision:
        raise HTTPException(409, "Карточка изменена. Загрузите текущую версию и повторите изменения")


def update_employee(db: Session, employee: Employee, data: EmployeeUpdate) -> Employee:
    employee_id, bind = employee.id, db.get_bind().engine
    db.rollback()
    try:
        with immediate_session(bind) as writer:
            current = writer.get(Employee, employee_id)
            _check_revision(current, data.expected_revision)
            values = data.model_dump(exclude_unset=True, exclude={"expected_revision", "workplace_ids"})
            for field, value in values.items():
                setattr(current, field, value)
            if "first_name" in values:
                current.name = f"{current.last_name} {current.first_name}"
            if "workplace_ids" in data.model_fields_set:
                replace_assignments(writer, employee_id, data.workplace_ids)
            current.revision += 1
            current.updated_at = utcnow()
            _advance_gallery(writer)
            writer.flush()
        db.expire_all()
        return current
    except SQLAlchemyError as exc:
        raise HTTPException(503, "База данных недоступна") from exc


def delete_employee(db: Session, employee: Employee):
    employee_id, bind = employee.id, db.get_bind().engine
    db.rollback()
    try:
        with immediate_session(bind) as writer:
            current = writer.get(Employee, employee_id)
            if current is None:
                raise HTTPException(404, "Сотрудник не найден")
            history = (writer.query(PresenceSession.id).filter_by(employee_id=employee_id).first()
                       or writer.query(AttendanceEmployeeReference.event_id).filter_by(employee_id=employee_id).first()
                       or writer.query(WorkSession.entry_event_id).filter_by(employee_id=employee_id).first()
                       or writer.query(AttendanceEvent.id).filter_by(employee_id=employee_id).first()
                       or writer.query(GlobalPerson.id).filter_by(employee_id=employee_id).first())
            if history:
                raise HTTPException(409, "У сотрудника есть история; отключите активность вместо удаления")
            writer.query(EmployeeFace).filter_by(employee_id=employee_id).delete()
            writer.query(EmployeeWorkplace).filter_by(employee_id=employee_id).delete()
            writer.delete(current)
            _advance_gallery(writer)
        db.expire_all()
    except SQLAlchemyError as exc:
        raise HTTPException(503, "База данных недоступна") from exc


def get_face(db, employee_id, face_id):
    return db.query(EmployeeFace).filter_by(id=face_id, employee_id=employee_id).first()
