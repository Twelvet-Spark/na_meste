"""Exact-image, operator-verified face append with frozen transactional receipts."""
import hashlib
import json
import logging
import math
from datetime import datetime, timezone
from uuid import UUID
from typing import Annotated, Literal

import numpy as np
import cv2
from pydantic import Field, StrictFloat, StrictInt, StrictStr, model_validator
from sqlalchemy import func
from fastapi import HTTPException
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.ai.face_detector import ALIGNMENT_ID
from app.ai.face_recognition import normalized_embedding
from app.attendance_types import HumanInput
from app.database.database import immediate_session
from app.database.models import Employee, EmployeeFace, HumanVisit, OperatorRequest
from app.services import employee_service as employees
from app.services.gallery_migration import decode_aligned_thumbnail
from app.time_utils import attendance_iso, utcnow

logger = logging.getLogger(__name__)
OPERATION = "employee.face.append_from_visit"
_DOMAIN = b"na_meste/employee.face.append_from_visit/v1\0"
PHOTO_OPERATION = "employee.face.append_visit_photo"
REMOVE_OPERATION = "employee.face.remove_visit_sample"
Coordinate = Annotated[StrictFloat, Field(ge=0, le=1)]


class VisitPhotoInput(HumanInput):
    employee_id: StrictInt = Field(gt=0)
    expected_revision: StrictInt = Field(ge=0)
    source: Literal["PHOTO_CROP", "UPLOAD"]
    photo_sha256: Annotated[StrictStr, Field(pattern=r"^[0-9a-f]{64}$")] | None = None
    crop: tuple[Coordinate, Coordinate, Coordinate, Coordinate] | None = None

    @model_validator(mode="after")
    def source_fields(self):
        if self.source == "PHOTO_CROP":
            if self.photo_sha256 is None or self.crop is None or not (
                    self.crop[0] < self.crop[2] and self.crop[1] < self.crop[3]):
                raise ValueError("Укажите точный сохранённый снимок и непустую область лица")
        elif self.photo_sha256 is not None or self.crop is not None:
            raise ValueError("Для загруженной фотографии координаты снимка не нужны")
        return self


class RemoveVisitSampleInput(HumanInput):
    sample_request_id: UUID
    expected_revision: StrictInt = Field(ge=0)


def _sample_key(face):
    # SQLite may reuse deleted integer IDs. Bind undo to this actual photograph,
    # not a later sample that happens to receive the same id.
    identity = f"{face.id}/{face.employee_id}/{face.created_at.isoformat()}".encode()
    return hashlib.sha256(identity + (face.thumbnail or b"")).hexdigest()


def _request_digest(visit_id, payload, upload=None):
    value = {"visit_id": visit_id, **payload.model_dump(mode="json")}
    if upload is not None:
        value["upload_sha256"] = hashlib.sha256(upload[1]).hexdigest()
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _error(status, code):
    return HTTPException(status, {"code": code})


def _utc(value):
    return value.astimezone(timezone.utc).replace(tzinfo=None) if value.tzinfo is not None else value


class VisitSampleService:
    def __init__(self, engine, face_engine, face_model_id, enrollment_semaphore, *, store=None, now=utcnow):
        self.engine = engine
        self.face_engine = face_engine
        self.face_model_id = face_model_id
        self.semaphore = enrollment_semaphore
        self.store = store
        self.now = now

    def _model_available(self):
        return (self.face_engine is not None and self.face_model_id is not None
                and self.face_engine.model_id == self.face_model_id
                and self.face_engine.alignment_id == ALIGNMENT_ID)

    def _checked_face(self, visit, expected_sha=None, *, decode=False):
        if visit is None:
            raise _error(404, "VISIT_NOT_FOUND")
        if (visit.images_purged_at is not None or visit.image_expires_at is None
                or _utc(visit.image_expires_at) <= _utc(self.now())):
            raise _error(410, "VISIT_FACE_EXPIRED")
        blob = visit.face_thumbnail
        if not isinstance(blob, bytes) or not 0 < len(blob) <= 256 * 1024:
            raise _error(409 if expected_sha is not None else 422,
                         "VISIT_FACE_CHANGED" if expected_sha is not None else "VISIT_FACE_MISSING")
        if expected_sha is not None and hashlib.sha256(blob).hexdigest() != expected_sha:
            raise _error(409, "VISIT_FACE_CHANGED")
        quality = visit.face_quality
        valid = isinstance(quality, dict) and quality.get("aligned") is True and quality.get("unambiguous") is True
        if valid:
            for key, minimum in (("min_size", 80.), ("det_score", .8), ("blur", 20.)):
                value = quality.get(key)
                if (isinstance(value, bool) or not isinstance(value, (int, float))
                        or not math.isfinite(value) or value < minimum):
                    valid = False
                    break
        if not valid:
            raise _error(422, "VISIT_FACE_QUALITY")
        image = decode_aligned_thumbnail(blob) if decode else None
        if decode and image is None:
            raise _error(422, "VISIT_FACE_INVALID")
        return blob, image

    def eligibility(self, visit):
        try:
            self._checked_face(visit, decode=True)
        except HTTPException as exc:
            return False, exc.detail["code"]
        return (True, None) if self._model_available() else (False, "FACE_MODEL_UNAVAILABLE")

    @staticmethod
    def _digest(visit_id, employee_id, expected_revision, face_sha256):
        payload = json.dumps({"visit_id": visit_id, "employee_id": employee_id,
                              "expected_revision": expected_revision, "face_sha256": face_sha256},
                             sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(_DOMAIN + payload).hexdigest()

    def append(self, visit_id, *, employee_id, expected_revision, request_id, face_sha256):
        visit_id, request_id = str(UUID(str(visit_id))), str(UUID(str(request_id)))
        digest = self._digest(visit_id, employee_id, expected_revision, face_sha256)
        try:
            # Receipt lookup precedes expiry, employee/model availability and GPU work.
            with Session(self.engine) as reader:
                replay = employees.committed_request(reader, request_id, digest, operation=OPERATION)
                if replay is not None:
                    return replay
                employee = reader.get(Employee, employee_id)
                employees._check_revision(employee, expected_revision)
                blob, image = self._checked_face(reader.get(HumanVisit, visit_id), face_sha256, decode=True)
                if reader.query(EmployeeFace).filter_by(employee_id=employee_id).count() >= 15:
                    raise _error(409, "FACE_LIMIT_REACHED")
            if not self._model_available():
                raise _error(503, "FACE_MODEL_UNAVAILABLE")
            # No read/write transaction remains open while acquiring or using the engine.
            with employees.enrollment_slot(self.semaphore):
                try:
                    vectors = np.asarray(self.face_engine.embed_aligned([image]))
                    vector = normalized_embedding(vectors[0]) if vectors.shape == (1, 512) else None
                except Exception as exc:
                    logger.exception("Visit face preparation unavailable")
                    raise _error(503, "FACE_MODEL_UNAVAILABLE") from exc
            if vector is None or not self._model_available():
                raise _error(503, "FACE_MODEL_UNAVAILABLE")
            prepared = employees.PreparedFace(f"visit-{visit_id}.jpg", vector.tobytes(), blob,
                                               self.face_model_id, ALIGNMENT_ID)
            return self._commit_sample(
                visit_id, employee_id, expected_revision, request_id, digest, prepared, OPERATION,
                lambda visit: self._checked_face(visit, face_sha256), source="ALIGNED_FACE")
        except SQLAlchemyError as exc:
            raise _error(503, "PERSISTENCE_ERROR") from exc

    def _refresh(self):
        if self.store is not None:
            try:
                with Session(self.engine) as reader:
                    self.store.refresh(reader)
            except SQLAlchemyError:
                logger.exception("Committed journal sample gallery publication unavailable")

    def _commit_sample(self, visit_id, employee_id, revision, request_id, digest, prepared,
                       operation, check_source, *, source, actor=None, reason=None):
        with immediate_session(self.engine, publish=True) as writer:
            replay = employees.committed_request(writer, request_id, digest, operation=operation)
            if replay is not None:
                return replay
            current = writer.get(Employee, employee_id)
            employees._check_revision(current, revision)
            check_source(writer.get(HumanVisit, visit_id))
            if writer.query(EmployeeFace).filter_by(employee_id=employee_id).count() >= 15:
                raise _error(409, "FACE_LIMIT_REACHED")
            employees.check_prepared_faces(writer, employee_id, [prepared], self.face_model_id)
            inserted = employees.insert_prepared_faces(writer, employee_id, [prepared])[0]
            current.revision += 1
            current.updated_at = _utc(self.now())
            employees.advance_gallery(writer)
            writer.flush()
            result = employees.employee_out(writer, current, model_id=self.face_model_id)
            result.update(visit_id=visit_id, face_id=inserted.id, sample_request_id=str(request_id),
                          sample_source=source, sample_key=_sample_key(inserted), actor=actor, reason=reason)
            writer.add(OperatorRequest(id=str(request_id), operation=operation, payload_sha256=digest,
                                       created_at=_utc(self.now()), result=result))
        self._refresh()
        return result

    def _checked_photo(self, visit, digest):
        if visit is None:
            raise _error(404, "VISIT_NOT_FOUND")
        if visit.images_purged_at is not None or _utc(visit.image_expires_at) <= _utc(self.now()):
            raise _error(410, "VISIT_PHOTO_EXPIRED")
        if not visit.snapshot or hashlib.sha256(visit.snapshot).hexdigest() != digest:
            raise _error(409, "VISIT_PHOTO_CHANGED")
        return visit.snapshot

    @staticmethod
    def _visit_exists(visit):
        if visit is None:
            raise _error(404, "VISIT_NOT_FOUND")

    def append_photo(self, visit_id, payload: VisitPhotoInput, upload=None) -> dict:
        visit_id = str(UUID(str(visit_id)))
        if (payload.source == "UPLOAD") != (upload is not None):
            raise _error(422, "VISIT_PHOTO_SOURCE_INVALID")
        digest = _request_digest(visit_id, payload, upload)
        check_source = (lambda visit: self._checked_photo(visit, payload.photo_sha256)
                        if payload.source == "PHOTO_CROP" else self._visit_exists(visit))
        try:
            with Session(self.engine) as reader:
                replay = employees.committed_request(reader, payload.request_id, digest, operation=PHOTO_OPERATION)
                if replay is not None:
                    return replay
                employees._check_revision(reader.get(Employee, payload.employee_id), payload.expected_revision)
                saved = check_source(reader.get(HumanVisit, visit_id))
                if reader.query(EmployeeFace).filter_by(employee_id=payload.employee_id).count() >= 15:
                    raise _error(409, "FACE_LIMIT_REACHED")
            if not self._model_available():
                raise _error(503, "FACE_MODEL_UNAVAILABLE")
            if payload.source == "PHOTO_CROP":
                filename = f"visit-{visit_id}.png"
                image = employees.decode_photo(saved, filename)
                if isinstance(image, employees.FaceEnrollResult):
                    raise employees.photo_failure([{"filename": filename, "error": image.error}])
                height, width = image.shape[:2]
                x1, y1, x2, y2 = payload.crop
                crop = image[math.floor(y1 * height):math.ceil(y2 * height),
                             math.floor(x1 * width):math.ceil(x2 * width)]
                if min(crop.shape[:2]) < 80:
                    raise employees.photo_failure([{"filename": filename, "error": "Область лица меньше 80 px"}])
                ok, encoded = cv2.imencode(".png", crop)
                if not ok:
                    raise _error(422, "VISIT_PHOTO_INVALID")
                content = encoded.tobytes()
            else:
                filename, content = upload
            with employees.enrollment_slot(self.semaphore):
                try:
                    prepared = employees.prepare_face(content, self.face_engine, filename)
                except Exception as exc:
                    logger.exception("Journal photograph preparation unavailable")
                    raise _error(503, "FACE_MODEL_UNAVAILABLE") from exc
            if isinstance(prepared, employees.FaceEnrollResult):
                raise employees.photo_failure([{"filename": filename, "error": prepared.error}])
            if not self._model_available():
                raise _error(503, "FACE_MODEL_UNAVAILABLE")
            return self._commit_sample(
                visit_id, payload.employee_id, payload.expected_revision, payload.request_id, digest,
                prepared, PHOTO_OPERATION, check_source, source=payload.source,
                actor=payload.actor, reason=payload.reason)
        except SQLAlchemyError as exc:
            raise _error(503, "PERSISTENCE_ERROR") from exc

    def remove_sample(self, visit_id, payload: RemoveVisitSampleInput) -> dict:
        visit_id = str(UUID(str(visit_id)))
        digest = _request_digest(visit_id, payload)
        try:
            with immediate_session(self.engine, publish=True) as writer:
                replay = employees.committed_request(writer, payload.request_id, digest, operation=REMOVE_OPERATION)
                if replay is not None:
                    return replay
                proof = writer.get(OperatorRequest, str(payload.sample_request_id))
                if (proof is None or proof.operation not in (OPERATION, PHOTO_OPERATION)
                        or proof.result.get("visit_id") != visit_id):
                    raise _error(404, "VISIT_SAMPLE_NOT_FOUND")
                employee = writer.get(Employee, proof.result["id"])
                employees._check_revision(employee, payload.expected_revision)
                face = writer.get(EmployeeFace, proof.result["face_id"])
                if face is None:
                    raise _error(409, "VISIT_SAMPLE_ALREADY_REMOVED")
                if face.employee_id != employee.id or proof.result.get("sample_key") != _sample_key(face):
                    raise _error(409, "VISIT_SAMPLE_CHANGED")
                writer.delete(face)
                writer.flush()
                employee.revision += 1
                employee.updated_at = _utc(self.now())
                employees.advance_gallery(writer)
                result = employees.employee_out(writer, employee, model_id=self.face_model_id)
                result.update(visit_id=visit_id, sample_request_id=str(payload.sample_request_id),
                              actor=payload.actor, reason=payload.reason)
                writer.add(OperatorRequest(id=str(payload.request_id), operation=REMOVE_OPERATION,
                                           payload_sha256=digest, created_at=_utc(self.now()), result=result))
            self._refresh()
            return result
        except SQLAlchemyError as exc:
            raise _error(503, "PERSISTENCE_ERROR") from exc

    def samples(self, visit_id) -> dict:
        with Session(self.engine) as db:
            self._visit_exists(db.get(HumanVisit, str(visit_id)))
            rows = db.query(OperatorRequest).filter(
                OperatorRequest.operation.in_((OPERATION, PHOTO_OPERATION)),
                func.json_extract(OperatorRequest.result, "$.visit_id") == str(visit_id),
            ).order_by(OperatorRequest.created_at.desc(), OperatorRequest.id.desc()).all()
            faces = {row.id: row for row in db.query(EmployeeFace).filter(
                EmployeeFace.id.in_([r.result["face_id"] for r in rows])).all()} if rows else {}
            return {"items": [dict(
                sample_request_id=row.id, employee_id=row.result["id"], employee_name=row.result["name"],
                face_id=row.result["face_id"], source=row.result.get("sample_source", "ALIGNED_FACE"),
                created_at=attendance_iso(row.created_at), actor=row.result.get("actor"), reason=row.result.get("reason"),
                present=((row.result["face_id"] in faces and
                          row.result["sample_key"] == _sample_key(faces[row.result["face_id"]]))
                         if "sample_key" in row.result else None),
                removable="sample_key" in row.result,
            ) for row in rows]}
