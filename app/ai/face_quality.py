"""Shared live/enrollment face quality; failed alignment is never synthesized."""
from dataclasses import dataclass

import cv2
import numpy as np

from app.ai.face_detector import FaceInfo


@dataclass(frozen=True, slots=True)
class FaceQuality:
    min_size: float
    det_score: float
    blur: float | None
    aligned: bool


def measure_face_quality(face: FaceInfo) -> FaceQuality:
    bbox = np.asarray(face.bbox)
    min_size = 0.
    if bbox.shape == (4,) and np.isfinite(bbox).all():
        min_size = max(0., float(min(bbox[2] - bbox[0], bbox[3] - bbox[1])))
    det_score = float(face.det_score)
    if not np.isfinite(det_score):
        det_score = 0.
    image = face.aligned
    aligned = (isinstance(image, np.ndarray) and image.dtype == np.uint8
               and image.shape == (112, 112, 3))
    blur = None
    if aligned:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        measured = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if np.isfinite(measured):
            blur = measured
        else:
            aligned = False
    return FaceQuality(min_size, det_score, blur, aligned)


def quality_reasons(quality: FaceQuality, *, min_size: float, det_score: float,
                    blur_min: float) -> tuple[str, ...]:
    reasons = []
    if quality.min_size < min_size:
        reasons.append("FACE_TOO_SMALL")
    if not quality.aligned:
        reasons.append("BAD_ALIGNMENT")
    if quality.det_score < det_score:
        reasons.append("LOW_FACE_DETECTION")
    if quality.blur is not None and quality.blur < blur_min:
        reasons.append("BLURRY_FACE")
    return tuple(reasons)
