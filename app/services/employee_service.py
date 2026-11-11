"""CRUD сотрудников и зачисление лиц (фото → эмбеддинг → SQLite)."""
import logging
from datetime import datetime
from typing import Optional

import cv2
import numpy as np
from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import func, update
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.ai.face_detector import FaceEngine
from app.ai.face_quality import measure_face_quality, quality_reasons
from app.ai.face_recognition import EmbeddingStore, normalized_embedding
from app.config import settings
from app.database.database import immediate_session
from app.database.models import (
    AttendanceEmployeeReference, AttendanceEvent, AttendanceRuntime, Employee,
    EmployeeFace, GlobalPerson, PresenceSession, WorkSession,
)
from app.services.attendance_evidence import encode_evidence

logger = logging.getLogger("app.services.employees")


# ------------------------------------------------------------------- схемы

class EmployeeCreate(BaseModel):
    name: str = Field(min_length=1, max_length=150)
    external_id: Optional[str] = Field(default=None, max_length=64)
    active: bool = True


class EmployeeUpdate(BaseModel):
    name: Optional[str] = Field(default=None, min_length=1, max_length=150)
    external_id: Optional[str] = Field(default=None, max_length=64)
    active: Optional[bool] = None


class EmployeeOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    external_id: Optional[str]
    active: bool
    faces_count: int = 0
    created_at: datetime
    updated_at: datetime


class FaceEnrollResult(BaseModel):
    filename: str
    success: bool
    face_id: Optional[int] = None
    error: Optional[str] = None
    warnings: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------- CRUD

def get_employee(db: Session, employee_id: int) -> Optional[Employee]:
    return db.get(Employee, employee_id)


def list_employees(db: Session) -> list[tuple[Employee, int]]:
    """Сотрудники + количество лиц каждого."""
    employees = db.query(Employee).order_by(Employee.name).all()
    counts = dict(
        db.query(EmployeeFace.employee_id, func.count(EmployeeFace.id))
        .group_by(EmployeeFace.employee_id).all()
    )
    return [(e, counts.get(e.id, 0)) for e in employees]


def create_employee(db: Session, data: EmployeeCreate) -> Employee:
    bind = db.get_bind().engine
    db.rollback()
    try:
        with immediate_session(bind) as writer:
            employee = Employee(**data.model_dump())
            writer.add(employee)
            writer.flush()
            _advance_gallery(writer)
        db.expire_all()
    except SQLAlchemyError as exc:
        raise HTTPException(status_code=503, detail="Database unavailable") from exc
    logger.info("Создан сотрудник #%d «%s»", employee.id, employee.name)
    return employee


def update_employee(db: Session, employee: Employee, data: EmployeeUpdate) -> Employee:
    employee_id = employee.id
    bind = db.get_bind().engine
    db.rollback()  # End the caller's read before acquiring a fresh write lock.
    try:
        with immediate_session(bind) as writer:
            current = writer.get(Employee, employee_id)
            if current is None:
                raise HTTPException(status_code=404, detail="Employee not found")
            for field, value in data.model_dump(exclude_unset=True).items():
                setattr(current, field, value)
            _advance_gallery(writer)
            writer.flush()
        db.expire_all()
        return current
    except SQLAlchemyError as exc:
        raise HTTPException(status_code=503, detail="Database unavailable") from exc


def delete_employee(db: Session, employee: Employee, store: Optional[EmbeddingStore]) -> None:
    """Delete only history-free employees, checking under the deletion lock.

    The caller refreshes its embedding store after the durable commit.
    """
    employee_id, name = employee.id, employee.name
    bind = db.get_bind().engine
    db.rollback()
    try:
        with immediate_session(bind) as writer:
            current = writer.get(Employee, employee_id)
            if current is None:
                raise HTTPException(status_code=404, detail="Employee not found")
            history = (
                writer.query(PresenceSession.id).filter_by(employee_id=employee_id).first()
                or writer.query(AttendanceEmployeeReference.event_id).filter_by(employee_id=employee_id).first()
                or writer.query(WorkSession.entry_event_id).filter_by(employee_id=employee_id).first()
                or writer.query(AttendanceEvent.id).filter_by(employee_id=employee_id).first()
                or writer.query(GlobalPerson.id).filter_by(employee_id=employee_id).first()
            )
            if history:
                raise HTTPException(status_code=409, detail="Employee has history; deactivate instead")
            writer.query(EmployeeFace).filter_by(employee_id=employee_id).delete()
            writer.delete(current)
            _advance_gallery(writer)
        db.expire_all()
    except SQLAlchemyError as exc:
        raise HTTPException(status_code=503, detail="Database unavailable") from exc
    logger.info("Удалён сотрудник #%d «%s»", employee_id, name)


def _advance_gallery(db: Session) -> None:
    """Identity data and its durable invalidation token share one transaction."""
    db.execute(update(AttendanceRuntime).where(AttendanceRuntime.id == 1).values(
        gallery_revision=AttendanceRuntime.gallery_revision + 1))


# --------------------------------------------------------- зачисление лица

def enroll_face(
    db: Session,
    employee: Employee,
    image_bytes: bytes,
    face_engine: FaceEngine,
    store: EmbeddingStore,
    filename: str = "photo",
) -> FaceEnrollResult:
    """Persist only usable aligned faces, rejecting another active identity."""
    img = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return FaceEnrollResult(filename=filename, success=False,
                                error="Не удалось прочитать файл изображения")
    faces = face_engine.detect(img)
    if not faces:
        return FaceEnrollResult(filename=filename, success=False,
                                error="Лицо на фотографии не найдено")
    if len(faces) != 1:
        return FaceEnrollResult(filename=filename, success=False,
                                error=f"На фотографии {len(faces)} лиц — загрузите фото с одним лицом")
    face = faces[0]
    quality = measure_face_quality(face)
    reasons = quality_reasons(quality, min_size=max(80, settings.face_min_size),
                              det_score=.80, blur_min=20.)
    embedding = normalized_embedding(face.embedding)
    errors = {
        "FACE_TOO_SMALL": "Лицо слишком маленькое — минимум 80 px",
        "BAD_ALIGNMENT": "Не удалось выровнять лицо — загрузите другое фото",
        "LOW_FACE_DETECTION": "Низкая уверенность детекции — минимум 0.80",
        "BLURRY_FACE": "Изображение размыто — загрузите более чёткое фото",
    }
    if reasons or embedding is None:
        return FaceEnrollResult(filename=filename, success=False,
                                error=errors[reasons[0]] if reasons else "Некорректный эмбеддинг лица")
    thumbnail = encode_evidence(before=None, after=None, face=face.aligned).face_jpeg
    if thumbnail is None:
        return FaceEnrollResult(filename=filename, success=False,
                                error="Не удалось сохранить выровненное лицо")

    employee_id = employee.id
    bind = db.get_bind().engine
    db.rollback()
    try:
        with immediate_session(bind) as writer:
            if writer.get(Employee, employee_id) is None:
                raise HTTPException(status_code=404, detail="Employee not found")
            # Another enrollment may have committed while model inference ran.
            # Match the current gallery under the write lock, not a stale cache.
            current_gallery = EmbeddingStore()
            current_gallery.refresh(writer)
            match = current_gallery.match(embedding, exclude_employee_id=employee_id)
            if match.employee_id is not None and match.score >= .55 and match.margin >= .08:
                return FaceEnrollResult(
                    filename=filename, success=False,
                    error="Лицо совпадает с другим активным сотрудником: "
                          f"{current_gallery.employee_name(match.employee_id)}",
                )
            employee_face = EmployeeFace(employee_id=employee_id,
                                         embedding=embedding.tobytes(), thumbnail=thumbnail)
            writer.add(employee_face)
            writer.flush()
            _advance_gallery(writer)
        db.expire_all()
    except SQLAlchemyError as exc:
        raise HTTPException(status_code=503, detail="Database unavailable") from exc
    store.refresh(db)
    logger.info("Сотруднику #%d добавлено лицо (face_id=%d)", employee_id, employee_face.id)
    return FaceEnrollResult(filename=filename, success=True, face_id=employee_face.id)


def delete_face(db: Session, face: EmployeeFace, store: EmbeddingStore) -> None:
    face_id = face.id
    bind = db.get_bind().engine
    db.rollback()
    try:
        with immediate_session(bind) as writer:
            current = writer.get(EmployeeFace, face_id)
            if current is None:
                raise HTTPException(status_code=404, detail="Face not found")
            writer.delete(current)
            _advance_gallery(writer)
        db.expire_all()
    except SQLAlchemyError as exc:
        raise HTTPException(status_code=503, detail="Database unavailable") from exc
    store.refresh(db)
    logger.info("Удалено лицо #%d", face_id)


def get_face(db: Session, employee_id: int, face_id: int) -> Optional[EmployeeFace]:
    return db.query(EmployeeFace).filter(
        EmployeeFace.id == face_id, EmployeeFace.employee_id == employee_id
    ).first()
