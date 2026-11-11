"""Bounded JPEG evidence and image-only retention for the attendance ledger."""
from dataclasses import dataclass
from datetime import datetime, timedelta

import cv2
import numpy as np
from sqlalchemy import Engine, update

from app.config import settings
from app.database.database import immediate_session
from app.database.models import AttendanceEvidence

_CONTEXT_MAX_DIMENSION = 640
_CONTEXT_MAX_BYTES = 96 * 1024
_FACE_MAX_BYTES = 32 * 1024
_JPEG_QUALITY = 70


@dataclass(frozen=True, slots=True)
class EncodedEvidence:
    before_jpeg: bytes | None
    after_jpeg: bytes | None
    face_jpeg: bytes | None
    reason_codes: tuple[str, ...]


def _encode(image: np.ndarray | None, *, face: bool) -> bytes | None:
    if (not isinstance(image, np.ndarray) or image.dtype != np.uint8
            or image.ndim not in (2, 3) or not image.size
            or (image.ndim == 3 and image.shape[2] not in (1, 3, 4))):
        return None
    try:
        height, width = image.shape[:2]
        if face:
            if (height, width) != (112, 112):
                image = cv2.resize(image, (112, 112), interpolation=cv2.INTER_AREA)
        elif max(height, width) > _CONTEXT_MAX_DIMENSION:
            scale = _CONTEXT_MAX_DIMENSION / max(height, width)
            image = cv2.resize(image, (max(1, int(width * scale)), max(1, int(height * scale))),
                               interpolation=cv2.INTER_AREA)
        quality = _JPEG_QUALITY
        bound = _FACE_MAX_BYTES if face else _CONTEXT_MAX_BYTES
        while True:
            ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
            if not ok:
                return None
            if encoded.nbytes <= bound:
                return encoded.tobytes()
            if face:
                # Keep aligned evidence exactly 112x112; lower quality if needed.
                if quality <= 10:
                    return None
                quality -= 10
            else:
                height, width = image.shape[:2]
                if max(height, width) <= 1:
                    return None
                image = cv2.resize(image, (max(1, int(width * 0.75)), max(1, int(height * 0.75))),
                                   interpolation=cv2.INTER_AREA)
    except (cv2.error, ValueError, TypeError, OverflowError):
        return None


def encode_evidence(*, before: np.ndarray | None, after: np.ndarray | None,
                    face: np.ndarray | None = None) -> EncodedEvidence:
    """Encode person crops (never full video) and an optional aligned face.

    Missing contextual crops or a supplied-but-unencodable face disqualify
    automatic admission. The caller carries these reasons into crossing evidence.
    """
    before_jpeg = _encode(before, face=False)
    after_jpeg = _encode(after, face=False)
    face_jpeg = _encode(face, face=True) if face is not None else None
    unavailable = before_jpeg is None or after_jpeg is None or (face is not None and face_jpeg is None)
    return EncodedEvidence(before_jpeg, after_jpeg, face_jpeg,
                           ("EVIDENCE_UNAVAILABLE",) if unavailable else ())


def evidence_expires_at(recorded_at: datetime) -> datetime:
    return recorded_at + timedelta(days=settings.attendance_evidence_keep_days)


def purge_evidence(engine: Engine, now: datetime) -> int:
    """Clear expired image BLOBs, retaining ledger/audit/session rows forever."""
    with immediate_session(engine) as db:
        result = db.execute(update(AttendanceEvidence).where(
            AttendanceEvidence.expires_at <= now,
            AttendanceEvidence.purged_at.is_(None),
        ).values(before_jpeg=None, after_jpeg=None, face_jpeg=None, purged_at=now))
        return result.rowcount
