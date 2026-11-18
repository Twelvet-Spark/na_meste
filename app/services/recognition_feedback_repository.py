"""Receipt-free, caller-transactional removal of operator recognition feedback."""
from collections.abc import Sequence
from datetime import datetime, timezone

from fastapi import HTTPException

from app.database.models import (
    Employee, EmployeeFace, RecognitionExample, RecognitionFeedback, RecognitionLearningRuntime,
)
from app.services import employee_service as employees
from app.services.visit_sample_service import _sample_key


def _naive_utc(value: datetime) -> datetime:
    return value.astimezone(timezone.utc).replace(tzinfo=None) if value.tzinfo is not None else value


def advance_learning(db) -> int:
    """Increment the exact singleton learning revision; a missing seed is an error, never a no-op."""
    runtime = db.query(RecognitionLearningRuntime).filter(RecognitionLearningRuntime.id == 1).one_or_none()
    if runtime is None:
        raise HTTPException(503, {"code": "LEARNING_RUNTIME_MISSING"})
    runtime.revision += 1
    db.flush()
    return runtime.revision


def revoke_feedback_rows(db, feedback_ids: Sequence[str], *, actor: str, reason: str, at: datetime) -> dict:
    """Revoke feedback inside the caller's transaction.

    Deletes only the rows' vector examples, clears retained crops and removes an owned face
    only while its ID, owner and content key still match. Writes no receipt and never restores
    an identity. Already-revoked rows are reported as revoked and left untouched.
    """
    ordered = sorted({str(value) for value in feedback_ids})
    result = dict(revoked_feedback_ids=[], removed_face_ids=[], face_removal_skipped_ids=[])
    if not ordered:
        return result
    rows = {row.id: row for row in db.query(RecognitionFeedback).filter(RecognitionFeedback.id.in_(ordered)).all()}
    missing = [value for value in ordered if value not in rows]
    if missing:
        raise HTTPException(404, {"code": "LEARNING_FEEDBACK_NOT_FOUND", "feedback_ids": missing})
    when = _naive_utc(at)
    learning_changed = gallery_changed = False
    for feedback_id in ordered:
        row = rows[feedback_id]
        result["revoked_feedback_ids"].append(feedback_id)
        if row.revoked_at is not None:
            continue
        examples = db.query(RecognitionExample).filter_by(feedback_id=feedback_id).all()
        for example in examples:
            gallery_changed = gallery_changed or example.modality == "FACE"
            db.delete(example)
        row.revoked_at, row.revoked_by, row.revoke_reason = when, actor, reason
        row.image_jpeg = None
        learning_changed = True
        if row.face_id is not None:
            face = db.get(EmployeeFace, row.face_id)
            if face is None or face.employee_id != row.employee_id or _sample_key(face) != row.face_key:
                result["face_removal_skipped_ids"].append(row.face_id)
            else:
                employee = db.get(Employee, face.employee_id)
                db.delete(face)
                db.flush()
                if employee is not None:
                    employee.revision += 1
                    employee.updated_at = when
                result["removed_face_ids"].append(face.id)
                gallery_changed = True
    db.flush()
    result["removed_face_ids"].sort()
    result["face_removal_skipped_ids"].sort()
    if learning_changed:
        advance_learning(db)
    if gallery_changed:
        employees.advance_gallery(db)
    return result
