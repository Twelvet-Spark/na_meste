"""Operator-reviewed person crops: read-only candidates and audited, atomic learning.

Decoding, cropping, face detection and body embedding run on one owned executor thread under the
shared nonblocking enrollment lease, never inside a database transaction or the publication fence.
Analysis only reads committed state and returns revisions it was computed against. A decision commits
its BODY example, optional same-crop face evidence, identity transition, revisions and frozen receipt
together or not at all. Uploads are employee reference material and never change a visit identity.
"""
import base64
import binascii
import hashlib
import json
import logging
import math
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Annotated, Literal
from uuid import UUID

import cv2
import numpy as np
from fastapi import HTTPException
from pydantic import (
    BaseModel, ConfigDict, Field, StrictFloat, StrictInt, StrictStr, field_validator, model_validator,
)
from sqlalchemy import and_, func, or_
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.ai.face_detector import ALIGNMENT_ID
from app.ai.face_quality import measure_face_quality, quality_reasons
from app.ai.face_recognition import EMBEDDING_SIZE, normalized_embedding, stored_embedding
from app.ai.recognition_learning import RecognitionLearningStore, match_body, rank_body
from app.attendance_types import HumanInput, milliseconds
from app.config import settings
from app.database.database import immediate_session, read_snapshot
from app.database.models import (
    AttendanceRuntime, Employee, EmployeeFace, FaceModelMetadata, HumanVisit, HumanVisitIdentityRevision,
    HumanVisitPhotoMetadata, OperatorRequest, RecognitionExample, RecognitionFeedback,
    RecognitionLearningRuntime,
)
from app.services import employee_service as employees
from app.services.attendance_evidence import encode_person_crop
from app.services.recognition_feedback_repository import advance_learning, revoke_feedback_rows
from app.services.visit_identity_service import append_identity_revision, effective_identity
from app.services.visit_sample_service import _sample_key
from app.time_utils import attendance_iso, day_bounds, local_date, utcnow

logger = logging.getLogger(__name__)
OPERATION = "recognition.feedback"
REVOKE_OPERATION = "recognition.feedback.revoke"
SQLITE_MAX = 2 ** 63 - 1
MIN_BODY_WIDTH, MIN_BODY_HEIGHT, LOW_DETAIL_HEIGHT = 16, 32, 128
EXAMPLE_LIMIT, FACE_LIMIT = 64, 15
# Candidate/negative FACE evidence gate; a positive face sample still needs prepare_face's 80 px gate.
FACE_MIN_SIZE, FACE_DET_SCORE, FACE_BLUR = 48., .8, 20.
_REMOVAL_KEYS = ("revoked_feedback_ids", "removed_face_ids", "face_removal_skipped_ids")
_SHA = re.compile(r"[0-9a-f]{64}")
_CURSOR = re.compile(r"[A-Za-z0-9_-]{1,512}")

Coordinate = Annotated[StrictFloat, Field(ge=0, le=1, allow_inf_nan=False)]
Sha256 = Annotated[StrictStr, Field(pattern=r"^[0-9a-f]{64}$")]
Revision = Annotated[StrictInt, Field(ge=0, le=SQLITE_MAX)]


class LearningEvidenceInput(BaseModel):
    """One operator-selected person rectangle on a saved journal photo or an uploaded image."""
    model_config = ConfigDict(extra="forbid")
    source: Literal["PHOTO_CROP", "UPLOAD"]
    crop: tuple[Coordinate, Coordinate, Coordinate, Coordinate]
    photo_sha256: Sha256 | None = None
    captured_at: datetime | None = None

    @field_validator("captured_at")
    @classmethod
    def aware_capture(cls, value):
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("Время съёмки требует явного часового пояса")
        normalized = milliseconds(value)
        attendance_iso(normalized)  # Accepted input must also render in receipts.
        try:
            local_date(normalized)
        except OverflowError as exc:
            raise ValueError("Время съёмки нельзя представить в часовом поясе учёта") from exc
        return normalized.replace(tzinfo=timezone.utc)

    @model_validator(mode="after")
    def source_fields(self):
        x1, y1, x2, y2 = self.crop
        if not (x1 < x2 and y1 < y2):
            raise ValueError("Выделите непустую область с человеком")
        if self.source == "PHOTO_CROP":
            if self.photo_sha256 is None:
                raise ValueError("Укажите точный сохранённый снимок")
            if self.captured_at is not None:
                raise ValueError("Время сохранённого снимка берётся только из журнала")
        elif self.photo_sha256 is not None:
            raise ValueError("Для загруженной фотографии хеш снимка журнала не нужен")
        return self


class LearningDecisionInput(LearningEvidenceInput, HumanInput):
    action: Literal["ACCEPT", "REJECT"]
    employee_id: Annotated[StrictInt, Field(gt=0, le=SQLITE_MAX)]
    expected_employee_revision: Revision
    expected_identity_revision: Revision
    expected_learning_revision: Revision
    expected_gallery_revision: Revision
    expected_image_sha256: Sha256
    expected_body_model_id: Sha256
    expected_face_model_id: Sha256 | None
    person_confirmed: Literal[True]


class LearningRevokeInput(HumanInput):
    expected_identity_revision: Revision | None


@dataclass(frozen=True, slots=True)
class _Source:
    """Frozen request bytes and their exact provenance; nothing mutable defines the evidence."""
    content: bytes
    filename: str
    sha256: str
    captured_at: datetime | None


@dataclass(frozen=True, slots=True, eq=False)
class _Prepared:
    image_jpeg: bytes
    image_sha256: str
    body: np.ndarray
    width: int
    height: int
    warnings: tuple[str, ...]
    face_status: str
    face: np.ndarray | None
    face_sample: employees.PreparedFace | None


def _error(status: int, code: str, **detail) -> HTTPException:
    return HTTPException(status, {"code": code, **detail})


def _naive(value: datetime | None) -> datetime | None:
    if value is None or value.tzinfo is None:
        return value
    return value.astimezone(timezone.utc).replace(tzinfo=None)


def _is_sha(value) -> bool:
    return isinstance(value, str) and _SHA.fullmatch(value) is not None


def _uuid(value, code: str) -> str:
    try:
        return str(UUID(str(value)))
    except (ValueError, TypeError, AttributeError):
        raise _error(404, code) from None


def _digest(scope: dict, payload: BaseModel, upload_sha256: str | None = None) -> str:
    value = {**scope, **payload.model_dump(mode="json")}
    if upload_sha256 is not None:
        value["upload_sha256"] = upload_sha256
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _identity_revision(db, visit_id: str) -> int:
    return db.query(func.max(HumanVisitIdentityRevision.revision)).filter(
        HumanVisitIdentityRevision.visit_id == visit_id).scalar() or 0


def _learning_revision(db) -> int:
    """Exact singleton read: an absent seed fails closed instead of reading as revision 0."""
    row = db.query(RecognitionLearningRuntime.revision).filter(RecognitionLearningRuntime.id == 1).one_or_none()
    if row is None:
        raise _error(503, "LEARNING_RUNTIME_MISSING")
    return int(row[0])


def _gallery_revision(db) -> int:
    return int(db.query(AttendanceRuntime.gallery_revision).filter(AttendanceRuntime.id == 1).one()[0])


def _anchor(captured_at: datetime | None, now: datetime, *, positive: bool = True) -> dict:
    """Workplace day of the evidence; only positive evidence from today anchors today's automatic naming."""
    day = local_date(captured_at) if captured_at is not None else None
    return {"date": day.isoformat() if day is not None else None,
            "today": positive and day is not None and day == local_date(now)}


def _pixel_bounds(crop, width: int, height: int) -> tuple[int, int, int, int]:
    """CameraCrop.nativeCrop: floor the start and ceil the end in source pixels."""
    x1, y1, x2, y2 = crop
    return (min(width, max(0, math.floor(x1 * width))), min(height, max(0, math.floor(y1 * height))),
            min(width, max(0, math.ceil(x2 * width))), min(height, max(0, math.ceil(y2 * height))))


def _face_gate() -> tuple[float, float]:
    """Face threshold/margin shared by automatic-candidate flags and the negative-face veto."""
    return (max(settings.min_recognition_confidence, settings.attendance_face_threshold),
            max(settings.live_face_margin, settings.attendance_face_margin))


def _candidate_rows(ranked, people, winner: int | None, vetoes: dict[int, list[str]]) -> list[dict]:
    rows = []
    for row in ranked:
        person = people.get(row.employee_id)
        if person is None or not person.active:
            continue
        rows.append({"employee_id": int(row.employee_id), "name": person.name,
                     "employee_revision": int(person.revision), "score": float(row.score),
                     "negative_score": None if row.negative_score is None else float(row.negative_score),
                     "rejected": bool(row.rejected), "automatic": winner is not None and row.employee_id == winner,
                     "rejection_feedback_ids": list(vetoes.get(row.employee_id, ())) if row.rejected else []})
    return rows


def _vetoing_feedback(db, modality: str, model_id: str, vector: np.ndarray, rejected: dict[int, float],
                      threshold: float, margin: float) -> dict[int, list[str]]:
    """Active negative feedback that vetoes this exact evidence per rejected candidate, most similar first.

    Read in the same committed snapshot as the store revisions, so these are the vetoing rows; a score
    exactly at the veto boundary still names the closest negative rather than nothing."""
    if not rejected:
        return {}
    query = db.query(RecognitionFeedback.id, RecognitionFeedback.employee_id, RecognitionExample.embedding).join(
        RecognitionExample, RecognitionExample.feedback_id == RecognitionFeedback.id).filter(
        RecognitionFeedback.employee_id.in_(list(rejected)), RecognitionFeedback.revoked_at.is_(None),
        RecognitionExample.modality == modality, RecognitionExample.polarity == "NEGATIVE",
        RecognitionExample.model_id == model_id)
    if modality == "FACE":
        query = query.filter(RecognitionExample.alignment_id == ALIGNMENT_ID)
    similar: dict[int, list[tuple[float, str]]] = {}
    for feedback_id, owner, blob in query:
        negative = stored_embedding(blob)
        if negative is not None:
            similar.setdefault(owner, []).append((float(negative @ vector), feedback_id))
    vetoes = {}
    for owner, values in similar.items():
        values.sort(key=lambda item: (-item[0], item[1]))
        positive = rejected[owner]
        vetoes[owner] = [feedback_id for score, feedback_id in values
                         if score >= threshold and score >= positive - margin] or [values[0][1]]
    return vetoes


def _learned_examples(row, receipt, stored, body_reference, face_reference) -> list[dict]:
    """What one decision learned, with frozen model provenance, whether or not it is still active.

    The decision receipt names BODY always and FACE when added (accepted faces live in EmployeeFace).
    Feedback without a receipt can only be described by its still-stored example rows."""
    if receipt is not None:
        entries = [("BODY", "POSITIVE" if row.action == "ACCEPT" else "NEGATIVE", receipt.body_model_id, None)]
        if row.face_status in ("ADDED", "NEGATIVE_ADDED"):
            entries.append(("FACE", "POSITIVE" if row.face_status == "ADDED" else "NEGATIVE",
                            receipt.face_model_id, receipt.face_alignment_id))
    else:
        entries = [(example.modality, example.polarity, example.model_id, example.alignment_id)
                   for example in (stored.get((row.id, "BODY")), stored.get((row.id, "FACE")))
                   if example is not None]
    learned = []
    for modality, polarity, model_id, alignment_id in entries:
        if model_id is None:
            compatible = None
        elif modality == "BODY":
            compatible = None if body_reference is None else model_id == body_reference
        else:
            compatible = (None if face_reference is None or alignment_id is None
                          else model_id == face_reference and alignment_id == ALIGNMENT_ID)
        learned.append({"modality": modality, "polarity": polarity, "model_id": model_id, "compatible": compatible})
    return learned


def _merged_removals(first: dict, second: dict) -> dict:
    return {key: sorted(set(first[key]) | set(second[key])) for key in _REMOVAL_KEYS}


def _encode_cursor(created_at: datetime, feedback_id: str) -> str:
    stamp = created_at.replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")
    raw = json.dumps([stamp, feedback_id], separators=(",", ":")).encode()
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def _decode_cursor(value) -> tuple[datetime, str]:
    try:
        if not isinstance(value, str) or not _CURSOR.fullmatch(value):
            raise ValueError("Invalid cursor encoding")
        raw = base64.b64decode(value + "=" * (-len(value) % 4), altchars=b"-_", validate=True)
        if base64.urlsafe_b64encode(raw).decode().rstrip("=") != value:
            raise ValueError("Noncanonical cursor")
        pair = json.loads(raw)
        if not isinstance(pair, list) or len(pair) != 2 or not all(isinstance(item, str) for item in pair):
            raise ValueError("Invalid cursor tuple")
        stamp = datetime.fromisoformat(pair[0].replace("Z", "+00:00"))
        if stamp.tzinfo is None or stamp.utcoffset() is None:
            raise ValueError("Cursor time requires an offset")
        identifier = str(UUID(pair[1]))
        if identifier != pair[1]:
            raise ValueError("Noncanonical cursor id")
        return stamp.astimezone(timezone.utc).replace(tzinfo=None), identifier
    except (ValueError, TypeError, UnicodeDecodeError, binascii.Error, OverflowError) as exc:
        raise _error(422, "INVALID_LEARNING_CURSOR") from exc


def snapshot_captured_at(db, visit_id: str, snapshot_sha256: str) -> datetime | None:
    """Exact capture time of the retained person snapshot; None for legacy or replaced images."""
    metadata = db.get(HumanVisitPhotoMetadata, visit_id)
    return metadata.captured_at if metadata is not None and metadata.sha256 == snapshot_sha256 else None


class RecognitionLearningService:
    def __init__(self, engine, body_engine, face_engine, face_model_id, enrollment_semaphore, *,
                 store=None, learning_store=None, appearance_gallery=None, now=utcnow):
        self.engine = engine
        self.body_engine = body_engine
        self.face_engine = face_engine
        self.face_model_id = face_model_id
        self.semaphore = enrollment_semaphore
        self.store = store
        body_model = getattr(body_engine, "model_id", None)
        if learning_store is None and _is_sha(body_model):
            learning_store = RecognitionLearningStore(engine, body_model)
        self.learning_store = learning_store
        self.appearance_gallery = appearance_gallery
        self.now = now
        # One owned native thread; the nonblocking lease is taken before every submission, so
        # requests never queue behind the GPU.
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="recognition-learning")
        self._lifecycle = threading.Lock()
        self._closed = False

    def model_available(self) -> bool:
        return not self._closed and self._models()[0] is not None

    def close(self) -> None:
        """Refuse new native work and drain the one owned executor."""
        with self._lifecycle:
            self._closed = True
        self._executor.shutdown(wait=True)

    def _now(self) -> datetime:
        return _naive(self.now())

    def _models(self) -> tuple[str | None, str | None]:
        """Model identities usable for new evidence now: (body, face); None when unavailable."""
        body = getattr(self.body_engine, "model_id", None) if self.body_engine is not None else None
        if not _is_sha(body) or self.learning_store is None or self.learning_store.body_model_id != body:
            body = None
        face, engine = self.face_model_id, self.face_engine
        if (engine is None or not _is_sha(face) or getattr(engine, "model_id", None) != face
                or getattr(engine, "alignment_id", None) != ALIGNMENT_ID):
            face = None
        return body, face

    def _check_models(self, body_model: str, face_model: str | None) -> None:
        body, face = self._models()
        if body is None:
            raise _error(503, "BODY_MODEL_UNAVAILABLE")
        if body != body_model:
            raise _error(409, "BODY_MODEL_CHANGED")
        if face != face_model:
            raise _error(409, "FACE_MODEL_CHANGED")

    @staticmethod
    def _upload_sha256(payload: LearningEvidenceInput, upload) -> str | None:
        if payload.source != "UPLOAD":
            if upload is not None:
                raise _error(422, "LEARNING_SOURCE_INVALID")
            return None
        if (not isinstance(upload, tuple) or len(upload) != 2 or not isinstance(upload[0], str)
                or not isinstance(upload[1], bytes)):
            raise _error(422, "LEARNING_SOURCE_INVALID")
        return hashlib.sha256(upload[1]).hexdigest()

    def _source(self, db, visit_id: str, payload: LearningEvidenceInput, upload, upload_sha256, now) -> _Source:
        visit = db.get(HumanVisit, visit_id)
        if visit is None:
            raise _error(404, "VISIT_NOT_FOUND")
        if payload.source == "UPLOAD":
            captured = _naive(payload.captured_at)
            if captured is not None and captured > now:
                raise _error(422, "CAPTURED_AT_IN_FUTURE")
            return _Source(upload[1], upload[0], upload_sha256, captured)
        if (visit.images_purged_at is not None or visit.image_expires_at is None
                or _naive(visit.image_expires_at) <= now):
            raise _error(410, "VISIT_PHOTO_EXPIRED")
        content = visit.snapshot
        if not isinstance(content, bytes) or not content or hashlib.sha256(content).hexdigest() != payload.photo_sha256:
            raise _error(409, "VISIT_PHOTO_CHANGED")
        return _Source(content, f"visit-{visit_id}.jpg", payload.photo_sha256,
                       snapshot_captured_at(db, visit_id, payload.photo_sha256))

    def _same_source(self, db, visit_id, payload, upload, upload_sha256, source: _Source) -> None:
        """Bytes, retention and capture provenance still equal what native preparation used."""
        current = self._source(db, visit_id, payload, upload, upload_sha256, self._now())
        if current.captured_at != source.captured_at:
            raise _error(409, "VISIT_PHOTO_CHANGED")

    def _native(self, source: _Source, crop, body_model: str, face_model: str | None) -> _Prepared:
        body_engine, face_engine = self.body_engine, self.face_engine if face_model is not None else None
        with employees.enrollment_slot(self.semaphore):
            with self._lifecycle:
                if self._closed:
                    raise _error(503, "BODY_MODEL_UNAVAILABLE")
                future = self._executor.submit(self._prepare, source, crop, body_engine, body_model,
                                               face_engine, face_model)
            return future.result()

    def _prepare(self, source: _Source, crop, body_engine, body_model, face_engine, face_model) -> _Prepared:
        image = employees.decode_photo(source.content, source.filename)
        if isinstance(image, employees.FaceEnrollResult):
            raise employees.photo_failure([{"filename": source.filename, "error": image.error}])
        height, width = image.shape[:2]
        left, top, right, bottom = _pixel_bounds(crop, width, height)
        crop_width, crop_height = right - left, bottom - top
        if crop_width < MIN_BODY_WIDTH or crop_height < MIN_BODY_HEIGHT:
            raise _error(422, "PERSON_CROP_TOO_SMALL", width=crop_width, height=crop_height,
                         min_width=MIN_BODY_WIDTH, min_height=MIN_BODY_HEIGHT)
        # Only the selected native pixels reach either extractor; nothing is upscaled.
        selected = np.ascontiguousarray(image[top:bottom, left:right])
        del image
        face_status, face, face_sample = self._face_evidence(selected, face_engine, face_model, source.filename)
        try:
            jpeg = encode_person_crop(selected)
            canonical = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR) if jpeg else None
        except cv2.error:
            jpeg = canonical = None
        if canonical is None:
            raise _error(422, "PERSON_CROP_INVALID")
        # The retained canonical JPEG, not the native crop, is what the body model sees.
        try:
            vectors = np.asarray(body_engine.embed([canonical]))
            body = normalized_embedding(vectors[0]) if vectors.shape == (1, EMBEDDING_SIZE) else None
        except Exception as exc:
            logger.exception("Body learning preparation unavailable")
            raise _error(503, "BODY_MODEL_UNAVAILABLE") from exc
        if body is None or getattr(body_engine, "model_id", None) != body_model:
            raise _error(503, "BODY_MODEL_UNAVAILABLE")
        return _Prepared(jpeg, hashlib.sha256(jpeg).hexdigest(), body, crop_width, crop_height,
                         ("LOW_DETAIL",) if crop_height < LOW_DETAIL_HEIGHT else (),
                         face_status, face, face_sample)

    @staticmethod
    def _face_evidence(selected, face_engine, face_model, filename):
        """Optional face of the selected crop: (status, candidate vector, strict positive sample).

        No face, several faces, low quality or an unavailable engine never fail body learning."""
        if face_engine is None or face_model is None:
            return "MODEL_UNAVAILABLE", None, None
        try:
            observations = list(face_engine.detect(selected))
            if not observations:
                return "NO_FACE", None, None
            if len(observations) > 1:
                return "MULTIPLE_FACES", None, None
            observation = observations[0]
            if (observation.model_id != face_model or getattr(face_engine, "model_id", None) != face_model
                    or getattr(face_engine, "alignment_id", None) != ALIGNMENT_ID):
                return "MODEL_UNAVAILABLE", None, None
            reasons = quality_reasons(measure_face_quality(observation), min_size=FACE_MIN_SIZE,
                                      det_score=FACE_DET_SCORE, blur_min=FACE_BLUR)
            vector = normalized_embedding(observation.embedding) if observation.embedding is not None else None
            if reasons or vector is None:
                return "LOW_QUALITY", None, None
            sample = employees.prepare_detected_face(observations, face_engine, filename)
        except Exception:
            logger.exception("Learning face preparation unavailable")
            return "MODEL_UNAVAILABLE", None, None
        return "ELIGIBLE", vector, sample if isinstance(sample, employees.PreparedFace) else None

    def analyze(self, visit_id: str, payload: LearningEvidenceInput,
                upload: tuple[str, bytes] | None = None) -> dict:
        """Read-only: prepare the selected crop and rank candidates; never labels or learns."""
        visit_id = _uuid(visit_id, "VISIT_NOT_FOUND")
        upload_sha256 = self._upload_sha256(payload, upload)
        try:
            now = self._now()
            source = read_snapshot(self.engine, lambda db: self._source(
                db, visit_id, payload, upload, upload_sha256, now))
            body_model, face_model = self._models()
            if body_model is None:
                raise _error(503, "BODY_MODEL_UNAVAILABLE")
            prepared = self._native(source, payload.crop, body_model, face_model)
            self._check_models(body_model, face_model)
            return self._analysis(visit_id, payload, upload, upload_sha256, source, prepared,
                                  body_model, face_model)
        except SQLAlchemyError as exc:
            raise _error(503, "PERSISTENCE_ERROR") from exc

    def _analysis(self, visit_id, payload, upload, upload_sha256, source, prepared, body_model, face_model):
        """Rank against store snapshots whose revisions equal one committed database snapshot."""
        store = self.store if (prepared.face is not None and self.store is not None
                               and self.store.model_id == face_model) else None
        captured = source.captured_at
        day = local_date(captured) if captured is not None else None
        transient = (self.appearance_gallery.samples_for_day(day)
                     if self.appearance_gallery is not None and day is not None else None)
        face_threshold, face_margin = _face_gate()
        for _attempt in range(3):
            snapshot = self.learning_store.snapshot()
            if not snapshot.loaded:
                snapshot = self.learning_store.refresh()
            now = self._now()
            # Unknown capture time is suggestion-only: no transient samples and no automatic match.
            body_rows = rank_body(snapshot, prepared.body, observed_at=captured or now, automatic=False,
                                  transient_samples=transient)[:3]
            body_winner = (match_body(snapshot, prepared.body, observed_at=captured, transient_samples=transient)
                           if captured is not None else None)
            face_rows, face_winner, face_revision = (), None, None
            if store is not None:
                before = store.revision
                face_rows = store.rank(prepared.face, limit=3)
                match = store.match(prepared.face)
                face_revision = before if before == match.store_revision == store.revision else None
                if (match.employee_id is not None and match.score >= face_threshold
                        and match.margin >= face_margin):
                    face_winner = match.employee_id
            ids = sorted({row.employee_id for row in (*body_rows, *face_rows)})
            body_rejected = {row.employee_id: row.score for row in body_rows if row.rejected}
            face_rejected = {row.employee_id: row.score for row in face_rows if row.rejected}
            identity, learning, gallery, people, body_vetoes, face_vetoes = read_snapshot(
                self.engine, lambda db: self._analysis_state(
                    db, visit_id, payload, upload, upload_sha256, source, ids, prepared, body_model, face_model,
                    body_rejected, face_rejected))
            body_current = snapshot.revision == learning and snapshot.gallery_revision == gallery
            face_current = store is None or face_revision == gallery
            if body_current and face_current:
                return {
                    "visit_id": visit_id,
                    "evidence": {
                        "source": payload.source, "source_sha256": source.sha256,
                        "image_sha256": prepared.image_sha256, "crop": list(payload.crop),
                        "crop_width": prepared.width, "crop_height": prepared.height,
                        "captured_at": attendance_iso(captured), "daily_anchor": _anchor(captured, now),
                        "warnings": list(prepared.warnings),
                    },
                    "identity_revision": identity, "learning_revision": learning, "gallery_revision": gallery,
                    "body_model_id": body_model, "face_model_id": face_model, "face_status": prepared.face_status,
                    "candidates": {
                        "face": _candidate_rows(face_rows, people, face_winner, face_vetoes),
                        "body": _candidate_rows(body_rows, people,
                                                body_winner.employee_id if body_winner is not None else None,
                                                body_vetoes),
                    },
                }
            if not body_current:
                self.learning_store.refresh()
            if not face_current:
                with Session(self.engine) as reader:
                    store.refresh(reader)
        raise _error(409, "LEARNING_REVISION_CHANGED")

    def _analysis_state(self, db, visit_id, payload, upload, upload_sha256, source, ids, prepared, body_model,
                        face_model, body_rejected, face_rejected):
        self._same_source(db, visit_id, payload, upload, upload_sha256, source)
        people = ({row.id: row for row in db.query(Employee.id, Employee.name, Employee.revision, Employee.active)
                   .filter(Employee.id.in_(ids))} if ids else {})
        body_vetoes = _vetoing_feedback(db, "BODY", body_model, prepared.body, body_rejected,
                                        settings.reid_employee_threshold, settings.reid_employee_margin)
        face_vetoes = (_vetoing_feedback(db, "FACE", face_model, prepared.face, face_rejected, *_face_gate())
                       if face_rejected else {})
        return (_identity_revision(db, visit_id), _learning_revision(db), _gallery_revision(db), people,
                body_vetoes, face_vetoes)

    @staticmethod
    def _check_target(db, payload: LearningDecisionInput):
        employee = db.get(Employee, payload.employee_id)
        if employee is None:
            raise _error(404, "EMPLOYEE_NOT_FOUND")
        if employee.revision != payload.expected_employee_revision:
            raise _error(409, "EMPLOYEE_REVISION_CONFLICT", employee_revision=employee.revision)
        return employee

    @staticmethod
    def _check_revisions(db, visit_id: str, payload: LearningDecisionInput) -> None:
        identity = _identity_revision(db, visit_id)
        if identity != payload.expected_identity_revision:
            raise _error(409, "IDENTITY_REVISION_CONFLICT", identity_revision=identity)
        learning = _learning_revision(db)
        if learning != payload.expected_learning_revision:
            raise _error(409, "LEARNING_REVISION_CONFLICT", learning_revision=learning)
        gallery = _gallery_revision(db)
        if gallery != payload.expected_gallery_revision:
            raise _error(409, "GALLERY_REVISION_CONFLICT", gallery_revision=gallery)

    def _decision_source(self, db, visit_id, payload, upload, upload_sha256, now) -> _Source:
        source = self._source(db, visit_id, payload, upload, upload_sha256, now)
        self._check_target(db, payload)
        self._check_revisions(db, visit_id, payload)
        return source

    def decide(self, visit_id: str, payload: LearningDecisionInput,
               upload: tuple[str, bytes] | None = None) -> dict:
        """Commit one operator decision about the selected crop together with its frozen receipt."""
        visit_id = _uuid(visit_id, "VISIT_NOT_FOUND")
        upload_sha256 = self._upload_sha256(payload, upload)
        digest = _digest({"visit_id": visit_id}, payload, upload_sha256)
        try:
            # The receipt lookup precedes expiry, model availability, revisions and native work.
            with Session(self.engine) as reader:
                replay = employees.committed_request(reader, payload.request_id, digest, operation=OPERATION)
            if replay is not None:
                return replay
            now = self._now()
            source = read_snapshot(self.engine, lambda db: self._decision_source(
                db, visit_id, payload, upload, upload_sha256, now))
            body_model, face_model = self._models()
            if body_model is None:
                raise _error(503, "BODY_MODEL_UNAVAILABLE")
            if body_model != payload.expected_body_model_id:
                raise _error(409, "BODY_MODEL_CHANGED")
            if face_model != payload.expected_face_model_id:
                raise _error(409, "FACE_MODEL_CHANGED")
            prepared = self._native(source, payload.crop, body_model, face_model)
            self._check_models(body_model, face_model)
            if prepared.image_sha256 != payload.expected_image_sha256:
                raise _error(409, "LEARNING_EVIDENCE_CHANGED", image_sha256=prepared.image_sha256)
            result, gallery_changed = self._commit_decision(
                visit_id, payload, upload, upload_sha256, digest, source, prepared, body_model, face_model)
        except SQLAlchemyError as exc:
            raise _error(503, "PERSISTENCE_ERROR") from exc
        if gallery_changed is not None:
            self._publish(gallery_changed)
        return result

    def _commit_decision(self, visit_id, payload, upload, upload_sha256, digest, source, prepared,
                         body_model, face_model):
        feedback_id, accept = str(payload.request_id), payload.action == "ACCEPT"
        polarity = "POSITIVE" if accept else "NEGATIVE"
        with immediate_session(self.engine, publish=True) as db:
            replay = employees.committed_request(db, feedback_id, digest, operation=OPERATION)
            if replay is not None:
                return replay, None
            self._check_models(body_model, face_model)  # Model identities are rechecked inside the fence.
            self._same_source(db, visit_id, payload, upload, upload_sha256, source)
            employee = self._check_target(db, payload)
            self._check_revisions(db, visit_id, payload)
            self._check_conflicts(db, payload, prepared.image_sha256, body_model)
            # Caps count every active example of a kind, whatever its model; curated evidence is never evicted.
            limited = [("BODY", polarity)]
            if not accept and face_model is not None and prepared.face is not None:
                limited.append(("FACE", "NEGATIVE"))
            for modality, kind in limited:
                if self._active_examples(db, employee.id, modality, kind) >= EXAMPLE_LIMIT:
                    raise _error(409, "LEARNING_LIMIT_REACHED", employee_id=employee.id, modality=modality,
                                 polarity=kind, limit=EXAMPLE_LIMIT,
                                 examples_url=f"/api/recognition-learning/examples?employee_id={employee.id}")
            now = milliseconds(self._now())
            visit = db.get(HumanVisit, visit_id)
            transition = None
            # Only retained visit evidence identifies the visit; a rejection never infers an alternative.
            if payload.source == "PHOTO_CROP" and (
                    accept or effective_identity(db, visit)["employee_id"] == employee.id):
                transition = append_identity_revision(
                    db, visit, expected_revision=payload.expected_identity_revision,
                    mode="EMPLOYEE" if accept else "UNKNOWN", employee_id=employee.id if accept else None,
                    actor=payload.actor, reason=payload.reason, request_id=feedback_id, created_at=now)
            face_status, face = self._face_outcome(db, employee, accept, prepared, face_model, now)
            db.add(RecognitionFeedback(
                id=feedback_id, visit_id=visit_id, employee_id=employee.id, action=payload.action,
                source=payload.source, source_sha256=source.sha256, image_sha256=prepared.image_sha256,
                crop=list(payload.crop), captured_at=source.captured_at, image_jpeg=prepared.image_jpeg,
                actor=payload.actor, reason=payload.reason, created_at=now,
                identity_revision=transition["identity_revision"] if transition is not None else None,
                face_id=face.id if face is not None else None,
                face_key=_sample_key(face) if face is not None else None, face_status=face_status))
            db.flush()
            db.add(RecognitionExample(feedback_id=feedback_id, modality="BODY", polarity=polarity,
                                      model_id=body_model, alignment_id=None, embedding=prepared.body.tobytes()))
            if face_status == "NEGATIVE_ADDED":
                db.add(RecognitionExample(feedback_id=feedback_id, modality="FACE", polarity="NEGATIVE",
                                          model_id=face_model, alignment_id=ALIGNMENT_ID,
                                          embedding=prepared.face.tobytes()))
            if face_status in ("ADDED", "NEGATIVE_ADDED"):
                employees.advance_gallery(db)
            learning_revision = advance_learning(db)
            gallery_revision = _gallery_revision(db)
            removal = transition if transition is not None else dict.fromkeys(_REMOVAL_KEYS, ())
            result = {
                "feedback_id": feedback_id, "visit_id": visit_id, "action": payload.action,
                "source": payload.source, "employee_id": employee.id, "employee_name": employee.name,
                "employee_active": bool(employee.active), "employee_revision": employee.revision,
                "source_sha256": source.sha256, "image_sha256": prepared.image_sha256,
                "crop": list(payload.crop), "captured_at": attendance_iso(source.captured_at),
                "daily_anchor": _anchor(source.captured_at, now, positive=accept),
                "learned_modalities": ["BODY", "FACE"] if face_status in ("ADDED", "NEGATIVE_ADDED") else ["BODY"],
                "face_status": face_status, "face_evidence": prepared.face_status,
                "face_id": face.id if face is not None else None,
                "identity": {"visit_id": visit_id, **effective_identity(db, visit)} if transition is not None else None,
                **{key: list(removal[key]) for key in _REMOVAL_KEYS},
                "learning_revision": learning_revision, "gallery_revision": gallery_revision,
                "body_model_id": body_model, "face_model_id": face_model,
                "face_alignment_id": ALIGNMENT_ID if face_model is not None else None,
                "historical_hours_changed": False,
                "actor": payload.actor, "reason": payload.reason, "created_at": attendance_iso(now),
            }
            db.add(OperatorRequest(id=feedback_id, operation=OPERATION, payload_sha256=digest,
                                   created_at=now, result=result))
        return result, gallery_revision != payload.expected_gallery_revision

    @staticmethod
    def _check_conflicts(db, payload: LearningDecisionInput, image_sha256: str, body_model: str) -> None:
        """Active evidence for this exact canonical image is never silently duplicated or contradicted."""
        rows = db.query(RecognitionFeedback.id, RecognitionFeedback.employee_id, RecognitionFeedback.action).filter(
            RecognitionFeedback.image_sha256 == image_sha256, RecognitionFeedback.revoked_at.is_(None),
        ).order_by(RecognitionFeedback.id).all()
        same = [row.id for row in rows if row.employee_id == payload.employee_id and row.action == payload.action]
        if same:
            current = {value for (value,) in db.query(RecognitionExample.feedback_id).filter(
                RecognitionExample.feedback_id.in_(same), RecognitionExample.modality == "BODY",
                RecognitionExample.model_id == body_model)}
            duplicate = next((value for value in same if value in current), None)
            if duplicate is not None:
                raise _error(409, "LEARNING_DUPLICATE", feedback_id=duplicate)
        conflicts = [row.id for row in rows if (
            (row.employee_id == payload.employee_id and row.action != payload.action)
            or (payload.action == "ACCEPT" and row.action == "ACCEPT" and row.employee_id != payload.employee_id))]
        if conflicts:
            raise _error(409, "LEARNING_CONFLICT", feedback_ids=conflicts)

    @staticmethod
    def _active_examples(db, employee_id: int, modality: str, polarity: str) -> int:
        """Active examples of one kind for one employee, across every model and alignment."""
        return int(db.query(func.count(RecognitionExample.id)).join(
            RecognitionFeedback, RecognitionFeedback.id == RecognitionExample.feedback_id).filter(
            RecognitionFeedback.employee_id == employee_id, RecognitionFeedback.revoked_at.is_(None),
            RecognitionExample.modality == modality, RecognitionExample.polarity == polarity).scalar() or 0)

    def _face_outcome(self, db, employee, accept: bool, prepared: _Prepared, face_model, now):
        """Same-crop face evidence: (feedback face_status, inserted EmployeeFace or None)."""
        if face_model is None or prepared.face_status == "MODEL_UNAVAILABLE":
            return "UNAVAILABLE", None
        if not accept:
            # The negative FACE cap was enforced before any write.
            return ("NEGATIVE_ADDED" if prepared.face is not None else "INELIGIBLE"), None
        if prepared.face_sample is None:
            return "INELIGIBLE", None
        if db.query(EmployeeFace.id).filter(EmployeeFace.employee_id == employee.id).count() >= FACE_LIMIT:
            return "LIMIT", None
        try:
            # Unchanged cross-owner protection; an operator rejection never relaxes enrollment overlap.
            employees.check_prepared_faces(db, employee.id, [prepared.face_sample], face_model)
        except HTTPException as exc:
            if exc.status_code != 422:
                raise
            return "CONFLICT", None
        face = employees.insert_prepared_faces(db, employee.id, [prepared.face_sample])[0]
        employee.revision += 1
        employee.updated_at = now
        return "ADDED", face

    def _publish(self, gallery_changed: bool) -> None:
        """Publish committed evidence to the shared stores; the receipt is already durable."""
        if self.learning_store is not None:
            try:
                self.learning_store.refresh()
            except SQLAlchemyError:
                logger.exception("Committed learning evidence publication unavailable")
        if gallery_changed and self.store is not None:
            try:
                with Session(self.engine) as reader:
                    self.store.refresh(reader)
            except SQLAlchemyError:
                logger.exception("Committed face gallery publication unavailable")

    def revoke(self, feedback_id: str, payload: LearningRevokeInput) -> dict:
        """Audited, model-independent removal; restores the visit label only while still current."""
        feedback_id = _uuid(feedback_id, "LEARNING_FEEDBACK_NOT_FOUND")
        digest = _digest({"feedback_id": feedback_id}, payload)
        try:
            with immediate_session(self.engine, publish=True) as db:
                replay = employees.committed_request(db, payload.request_id, digest, operation=REVOKE_OPERATION)
                if replay is not None:
                    return replay
                row = db.get(RecognitionFeedback, feedback_id)
                if row is None:
                    raise _error(404, "LEARNING_FEEDBACK_NOT_FOUND")
                if row.revoked_at is not None:
                    raise _error(409, "LEARNING_FEEDBACK_REVOKED", revoked_at=attendance_iso(row.revoked_at),
                                 revoked_by=row.revoked_by, revoke_reason=row.revoke_reason)
                at = milliseconds(self._now())
                gallery_before = _gallery_revision(db)
                removal = revoke_feedback_rows(db, [feedback_id], actor=payload.actor, reason=payload.reason, at=at)
                identity, outcome = None, "SAMPLE_ONLY"
                if row.identity_revision is not None and payload.expected_identity_revision is not None:
                    current = _identity_revision(db, row.visit_id)
                    if current == row.identity_revision == payload.expected_identity_revision:
                        previous = (db.query(HumanVisitIdentityRevision).filter_by(
                            visit_id=row.visit_id, revision=current - 1).one() if current > 1 else None)
                        visit = db.get(HumanVisit, row.visit_id)
                        transition = append_identity_revision(
                            db, visit, expected_revision=current,
                            mode=previous.mode if previous is not None else "AI",
                            employee_id=previous.employee_id if previous is not None else None,
                            actor=payload.actor, reason=payload.reason, request_id=str(payload.request_id),
                            created_at=at)
                        removal = _merged_removals(removal, transition)
                        identity, outcome = {"visit_id": visit.id, **effective_identity(db, visit)}, "RESTORED"
                    else:
                        outcome = "KEPT"
                gallery_revision = _gallery_revision(db)
                result = {
                    "feedback_id": feedback_id, "visit_id": row.visit_id, "employee_id": row.employee_id,
                    "action": row.action, "source": row.source,
                    **{key: list(removal[key]) for key in _REMOVAL_KEYS},
                    "identity": identity, "identity_outcome": outcome,
                    "learning_revision": _learning_revision(db), "gallery_revision": gallery_revision,
                    "historical_hours_changed": False,
                    "actor": payload.actor, "reason": payload.reason, "revoked_at": attendance_iso(at),
                }
                db.add(OperatorRequest(id=str(payload.request_id), operation=REVOKE_OPERATION,
                                       payload_sha256=digest, created_at=at, result=result))
        except SQLAlchemyError as exc:
            raise _error(503, "PERSISTENCE_ERROR") from exc
        self._publish(gallery_revision != gallery_before)
        return result

    def _reference_models(self) -> tuple[str | None, str | None]:
        """Models that decide example compatibility, even while an engine is unavailable."""
        body = self._models()[0]
        if body is None and self.learning_store is not None:
            body = self.learning_store.body_model_id
        return body, self.face_model_id if _is_sha(self.face_model_id) else None

    def history(self, *, visit_id: str | None = None, employee_id: int | None = None,
                cursor: str | None = None, limit: int = 50) -> dict:
        """Newest-first feedback of one visit or one employee; no image or vector bytes are read."""
        if (visit_id is None) == (employee_id is None):
            raise _error(422, "LEARNING_HISTORY_SCOPE")
        if type(limit) is not int or not 1 <= limit <= 200:
            raise _error(422, "INVALID_LEARNING_LIMIT")
        if visit_id is not None:
            visit_id = _uuid(visit_id, "VISIT_NOT_FOUND")
        elif type(employee_id) is not int or not 0 < employee_id <= SQLITE_MAX:
            raise _error(422, "INVALID_EMPLOYEE_ID")
        position = _decode_cursor(cursor) if cursor is not None else None
        try:
            return read_snapshot(self.engine, lambda db: self._history(db, visit_id, employee_id, position, limit))
        except SQLAlchemyError as exc:
            raise _error(503, "PERSISTENCE_ERROR") from exc

    def _history(self, db, visit_id, employee_id, position, limit) -> dict:
        feedback = RecognitionFeedback
        if visit_id is not None:
            if db.query(HumanVisit.id).filter(HumanVisit.id == visit_id).first() is None:
                raise _error(404, "VISIT_NOT_FOUND")
            scope = feedback.visit_id == visit_id
        else:
            if db.query(Employee.id).filter(Employee.id == employee_id).first() is None:
                raise _error(404, "EMPLOYEE_NOT_FOUND")
            scope = feedback.employee_id == employee_id
        query = db.query(
            feedback.id, feedback.visit_id, feedback.employee_id, feedback.action, feedback.source,
            feedback.source_sha256, feedback.image_sha256, feedback.crop, feedback.captured_at, feedback.actor,
            feedback.reason, feedback.created_at, feedback.identity_revision, feedback.face_id,
            feedback.face_status, feedback.revoked_at, feedback.revoked_by, feedback.revoke_reason,
            feedback.image_jpeg.is_not(None).label("has_image"), Employee.name.label("employee_name"),
            Employee.active.label("employee_active"),
        ).join(Employee, Employee.id == feedback.employee_id).filter(scope)
        if position is not None:
            created, identifier = position
            query = query.filter(or_(feedback.created_at < created,
                                     and_(feedback.created_at == created, feedback.id < identifier)))
        rows = query.order_by(feedback.created_at.desc(), feedback.id.desc()).limit(limit + 1).all()
        page = rows[:limit]
        body_model, face_model = self._reference_models()
        receipts, stored = {}, {}
        if page:
            ids = [row.id for row in page]
            # Frozen receipt provenance survives revocation (examples are deleted) and covers accepted faces
            # kept in EmployeeFace; only scalar JSON fields and narrow example columns are read.
            receipts = {receipt.id: receipt for receipt in db.query(
                OperatorRequest.id,
                func.json_extract(OperatorRequest.result, "$.body_model_id").label("body_model_id"),
                func.json_extract(OperatorRequest.result, "$.face_model_id").label("face_model_id"),
                func.json_extract(OperatorRequest.result, "$.face_alignment_id").label("face_alignment_id"),
            ).filter(OperatorRequest.operation == OPERATION, OperatorRequest.id.in_(ids))}
            stored = {(example.feedback_id, example.modality): example for example in db.query(
                RecognitionExample.feedback_id, RecognitionExample.modality, RecognitionExample.polarity,
                RecognitionExample.model_id, RecognitionExample.alignment_id,
            ).filter(RecognitionExample.feedback_id.in_(ids))}
        now = self._now()
        items = []
        for row in page:
            active = row.revoked_at is None
            owned = _learned_examples(row, receipts.get(row.id), stored, body_model, face_model)
            # Revoked or incompatible evidence keeps its date but never claims today's anchor.
            anchors = active and row.action == "ACCEPT" and any(
                item["modality"] == "BODY" and item["compatible"] is True for item in owned)
            items.append({
                "feedback_id": row.id, "visit_id": row.visit_id, "employee_id": row.employee_id,
                "employee_name": row.employee_name, "employee_active": bool(row.employee_active),
                "action": row.action, "source": row.source, "active": active,
                "source_sha256": row.source_sha256,
                "image_sha256": row.image_sha256 if row.has_image else None,
                "photo_url": f"/api/recognition-learning/{row.id}/photo" if row.has_image else None,
                "crop": list(row.crop), "captured_at": attendance_iso(row.captured_at),
                "daily_anchor": _anchor(row.captured_at, now, positive=anchors),
                "identity_revision": row.identity_revision, "face_status": row.face_status,
                "face_id": row.face_id, "examples": owned,
                "actor": row.actor, "reason": row.reason, "created_at": attendance_iso(row.created_at),
                "revocation": None if active else {"revoked_at": attendance_iso(row.revoked_at),
                                                   "revoked_by": row.revoked_by, "reason": row.revoke_reason},
            })
        return {"items": items,
                "next_cursor": _encode_cursor(page[-1].created_at, page[-1].id) if len(rows) > limit else None,
                "counts": self._counts(db, scope, employee_id, body_model, face_model, now)}

    @staticmethod
    def _counts(db, scope, employee_id, body_model, face_model, now) -> dict:
        """Active evidence in scope. Employee scope counts current-model face templates; visit scope
        counts faces added from this visit's feedback. None: the comparison model is unknown."""
        def examples(*conditions) -> int:
            return int(db.query(func.count(RecognitionExample.id)).join(
                RecognitionFeedback, RecognitionFeedback.id == RecognitionExample.feedback_id).filter(
                scope, RecognitionFeedback.revoked_at.is_(None), *conditions).scalar() or 0)

        counts = dict.fromkeys(("body_positive", "body_negative", "face_positive", "face_negative", "today_body"))
        if body_model is not None:
            body = (RecognitionExample.modality == "BODY", RecognitionExample.model_id == body_model)
            start, end = day_bounds(local_date(now))
            counts.update(
                body_positive=examples(*body, RecognitionExample.polarity == "POSITIVE"),
                body_negative=examples(*body, RecognitionExample.polarity == "NEGATIVE"),
                today_body=examples(*body, RecognitionExample.polarity == "POSITIVE",
                                    RecognitionFeedback.captured_at >= start, RecognitionFeedback.captured_at < end))
        if face_model is not None:
            counts["face_negative"] = examples(
                RecognitionExample.modality == "FACE", RecognitionExample.polarity == "NEGATIVE",
                RecognitionExample.model_id == face_model, RecognitionExample.alignment_id == ALIGNMENT_ID)
            if employee_id is not None:
                counts["face_positive"] = int(db.query(func.count(EmployeeFace.id)).join(
                    FaceModelMetadata, FaceModelMetadata.face_id == EmployeeFace.id).filter(
                    EmployeeFace.employee_id == employee_id, FaceModelMetadata.model_id == face_model,
                    FaceModelMetadata.alignment_id == ALIGNMENT_ID).scalar() or 0)
            else:
                counts["face_positive"] = int(db.query(func.count(RecognitionFeedback.id)).filter(
                    scope, RecognitionFeedback.revoked_at.is_(None),
                    RecognitionFeedback.face_status == "ADDED").scalar() or 0)
        return counts

    def photo(self, feedback_id: str) -> tuple[bytes, str]:
        """Retained canonical crop of active feedback as (JPEG, SHA-256): 404 unknown, 410 removed."""
        feedback_id = _uuid(feedback_id, "LEARNING_FEEDBACK_NOT_FOUND")
        try:
            with Session(self.engine) as db:
                row = db.query(RecognitionFeedback.image_jpeg, RecognitionFeedback.image_sha256,
                               RecognitionFeedback.revoked_at).filter(RecognitionFeedback.id == feedback_id).first()
        except SQLAlchemyError as exc:
            raise _error(503, "PERSISTENCE_ERROR") from exc
        if row is None:
            raise _error(404, "LEARNING_FEEDBACK_NOT_FOUND")
        image = row.image_jpeg
        if (row.revoked_at is not None or not isinstance(image, bytes) or not image
                or hashlib.sha256(image).hexdigest() != row.image_sha256):
            raise _error(410, "LEARNING_IMAGE_REMOVED")
        return image, row.image_sha256
