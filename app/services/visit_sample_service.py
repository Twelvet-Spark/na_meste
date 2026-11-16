"""Exact-image, operator-verified face append with frozen transactional receipts."""
import hashlib
import json
import logging
import math
from datetime import datetime, timezone
from uuid import UUID

import numpy as np
from fastapi import HTTPException
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.ai.face_detector import ALIGNMENT_ID
from app.ai.face_recognition import normalized_embedding
from app.database.database import immediate_session
from app.database.models import Employee, EmployeeFace, HumanVisit, OperatorRequest
from app.services import employee_service as employees
from app.services.gallery_migration import decode_aligned_thumbnail
from app.time_utils import utcnow

logger = logging.getLogger(__name__)
OPERATION = "employee.face.append_from_visit"
_DOMAIN = b"na_meste/employee.face.append_from_visit/v1\0"


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
            with immediate_session(self.engine, publish=True) as writer:
                replay = employees.committed_request(writer, request_id, digest, operation=OPERATION)
                if replay is not None:
                    return replay
                current = writer.get(Employee, employee_id)
                employees._check_revision(current, expected_revision)
                self._checked_face(writer.get(HumanVisit, visit_id), face_sha256)
                if writer.query(EmployeeFace).filter_by(employee_id=employee_id).count() >= 15:
                    raise _error(409, "FACE_LIMIT_REACHED")
                employees.check_prepared_faces(writer, employee_id, [prepared], self.face_model_id)
                inserted = employees.insert_prepared_faces(writer, employee_id, [prepared])
                current.revision += 1
                current.updated_at = _utc(self.now())
                employees.advance_gallery(writer)
                writer.flush()
                result = employees.employee_out(writer, current, model_id=self.face_model_id)
                result.update(visit_id=visit_id, face_id=inserted[0].id)
                writer.add(OperatorRequest(id=request_id, operation=OPERATION, payload_sha256=digest,
                                           created_at=_utc(self.now()), result=result))
            if self.store is not None:
                try:
                    with Session(self.engine) as reader:
                        self.store.refresh(reader)
                except SQLAlchemyError:
                    # The receipt is already durable. Worker revision checks fail
                    # closed and retry publication; do not misreport an append rollback.
                    logger.exception("Committed visit sample gallery publication unavailable")
            return result
        except SQLAlchemyError as exc:
            raise _error(503, "PERSISTENCE_ERROR") from exc
