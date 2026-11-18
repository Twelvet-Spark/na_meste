"""Provenance-aware aligned-thumbnail regeneration before gallery publication."""
import io
import warnings

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError
from sqlalchemy import update
from sqlalchemy.orm import Session

from app.ai.face_detector import ALIGNMENT_ID
from app.ai.face_recognition import normalized_embedding, stored_embedding
from app.ai.model_bundle import ModelBundleError
from app.database.database import immediate_session
from app.database.models import AttendanceRuntime, Employee, EmployeeFace, FaceModelMetadata
from app.time_utils import utcnow


class GalleryChangedError(RuntimeError):
    code = "GALLERY_CHANGED"

    def __init__(self):
        super().__init__(self.code)


def decode_aligned_thumbnail(blob: bytes | None) -> np.ndarray | None:
    if not isinstance(blob, bytes) or not blob:
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(blob)) as image:
                if image.format != "JPEG" or image.size != (112, 112) or getattr(image, "is_animated", False):
                    return None
                image.load()  # Reject truncated images, not just plausible headers.
                return cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError, cv2.error,
            Image.DecompressionBombError, Image.DecompressionBombWarning):
        return None


def reembed_aligned_thumbnails(face_engine, thumbnails: list[bytes | None]) -> list[np.ndarray | None]:
    """Ordered optional vectors; only real aligned 112x112 JPEGs, no redetection.

    Request callers hold the shared enrollment semaphore; startup runs before
    publication. Decode and embed at most four images at once in either case.
    Model execution/contract failures propagate, never masquerading as bad photos.
    """
    result = [None] * len(thumbnails)
    for start in range(0, len(thumbnails), 4):
        indices, images = [], []
        for index in range(start, min(start + 4, len(thumbnails))):
            image = decode_aligned_thumbnail(thumbnails[index])
            if image is not None:
                indices.append(index)
                images.append(image)
        if not images:
            continue
        if face_engine is None:
            raise ModelBundleError("MODEL_MISSING", "face_recognition", "Thumbnail preparation needs a recognizer")
        vectors = np.asarray(face_engine.embed_aligned(images))
        if vectors.shape != (len(images), 512):
            raise ModelBundleError("MODEL_CONTRACT_INVALID", "face_recognition", "Thumbnail embedding batch")
        for index, vector in zip(indices, vectors):
            normalized = normalized_embedding(vector)
            if normalized is None:
                raise ModelBundleError("MODEL_CONTRACT_INVALID", "face_recognition", "Thumbnail embedding vector")
            result[index] = normalized
    return result


def rebuild_gallery(engine, face_engine) -> dict:
    """Prepare outside the writer; atomically publish only an unchanged snapshot.

    Compatible, normalized vectors are retained byte-for-byte even without images.
    Unconvertible samples remain stored and excluded by EmbeddingStore. An
    unchanged gallery never obtains a writer lock.
    """
    model_id, alignment_id = face_engine.model_id, face_engine.alignment_id
    if alignment_id != ALIGNMENT_ID:
        raise ModelBundleError("MODEL_CONTRACT_INVALID", "face_recognition", "Unknown gallery alignment")
    with engine.connect() as connection:
        connection.exec_driver_sql("BEGIN")
        with Session(bind=connection) as reader:
            gallery_revision = reader.query(AttendanceRuntime.gallery_revision).filter_by(id=1).one()[0]
            employee_revisions = dict(reader.query(Employee.id, Employee.revision).all())
            samples = reader.query(EmployeeFace.id, EmployeeFace.employee_id, EmployeeFace.embedding,
                                   EmployeeFace.thumbnail, FaceModelMetadata.model_id,
                                   FaceModelMetadata.alignment_id).outerjoin(
                FaceModelMetadata, FaceModelMetadata.face_id == EmployeeFace.id,
            ).order_by(EmployeeFace.id).all()
        connection.rollback()

    needs_preparation, usable_ids = [], set()
    for row in samples:
        if (row.model_id == model_id and row.alignment_id == alignment_id
                and stored_embedding(row.embedding) is not None):
            usable_ids.add(row.id)
        else:
            needs_preparation.append(row)
    vectors = reembed_aligned_thumbnails(face_engine, [row.thumbnail for row in needs_preparation])
    changed = [(row, vector) for row, vector in zip(needs_preparation, vectors) if vector is not None]
    unusable = [row for row, vector in zip(needs_preparation, vectors) if vector is None]
    usable_ids.update(row.id for row, _vector in changed)
    owners_with_samples = {row.employee_id for row in samples}
    remediation = sorted({row.employee_id for row in unusable} | (employee_revisions.keys() - owners_with_samples))
    report = {"face_model_id": model_id, "alignment_id": alignment_id,
              "stored_faces_count": len(samples), "usable_faces_count": len(usable_ids),
              "faces_reencoded": len(changed), "faces_need_reenrollment": len(unusable),
              "employees_needing_photos": remediation}
    if not changed:
        return report

    with immediate_session(engine, publish=True) as writer:
        runtime = writer.get(AttendanceRuntime, 1)
        current_employees = dict(writer.query(Employee.id, Employee.revision).all())
        if runtime.gallery_revision != gallery_revision or current_employees != employee_revisions:
            raise GalleryChangedError()
        for row, vector in changed:
            face = writer.get(EmployeeFace, row.id)
            # Revisions protect supported mutations; this also prevents overwriting
            # an independently changed/deleted sample or provenance sidecar.
            metadata = writer.get(FaceModelMetadata, row.id)
            current_provenance = (metadata.model_id, metadata.alignment_id) if metadata is not None else (None, None)
            if (face is None or face.employee_id != row.employee_id or face.embedding != row.embedding
                    or face.thumbnail != row.thumbnail or current_provenance != (row.model_id, row.alignment_id)):
                raise GalleryChangedError()
            face.embedding = vector.tobytes()
            if metadata is None:
                writer.add(FaceModelMetadata(face_id=row.id, model_id=model_id,
                                             alignment_id=alignment_id, created_at=utcnow()))
            else:
                metadata.model_id, metadata.alignment_id = model_id, alignment_id
                metadata.created_at = utcnow()
        changed_owners = {row.employee_id for row, _vector in changed}
        writer.execute(update(Employee).where(Employee.id.in_(changed_owners)).values(
            revision=Employee.revision + 1, updated_at=utcnow()))
        writer.execute(update(AttendanceRuntime).where(AttendanceRuntime.id == 1).values(
            gallery_revision=AttendanceRuntime.gallery_revision + 1))
    return report
