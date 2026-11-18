"""Logical setup snapshots and empty-only imports, independent of media services."""
import base64
import hashlib
import json
import re
import threading
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import UUID

import numpy as np
from sqlalchemy import select, update
from sqlalchemy.orm import Session
from app.ai.face_recognition import is_normalized_embedding

from app.database import attendance_schema, operator_schema
from app.database.database import immediate_session
from app.database.models import (AttendanceCamera, AttendanceRuntime, Camera, CameraArchive, CameraProfile,
                                 CameraScene, Employee, EmployeeFace, EmployeeWorkplace, FaceModelMetadata,
                                 OperatorRequest, Workplace)
from app.scene_geometry import extruded_door_top
from app.services.gallery_migration import reembed_aligned_thumbnails
from app.setup_transfer import (ALIGNMENT_ID, MAX_FILE_BYTES, TransferError, decode_binary,
                                decode_transfer, encode_transfer, validate_document)

# Explicit projection: adding a database column cannot silently expand the export.
PROJECTIONS = {
    "cameras": (Camera, ("id", "name", "nvr_host", "rtsp_port", "username", "password", "channel", "stream_type", "enabled", "created_at", "updated_at"), 10000),
    "archives": (CameraArchive, ("camera_id", "http_scheme", "http_port", "utc_offset_minutes"), 10000),
    "calibrations": (AttendanceCamera, ("camera_id", "role", "line_start", "line_end", "inside_point", "door_kind", "door_top", "reference_width", "reference_height"), 10000),
    "workplaces": (Workplace, ("id", "name", "active"), 10000),
    "employees": (Employee, ("id", "name", "first_name", "last_name", "external_id", "active", "created_at", "updated_at"), 10000),
    "faces": (EmployeeFace, ("id", "employee_id", "embedding", "thumbnail", "created_at", "face_model_id", "alignment_id"), 150000),
    "assignments": (EmployeeWorkplace, ("employee_id", "workplace_id"), 100000),
    "scenes": (CameraScene, ("camera_id", "reference_width", "reference_height", "geometry"), 10000),
    "profiles": (CameraProfile, ("camera_id", "workplace_id", "zone", "ai_enabled", "ai_fps", "time_tracking"), 10000),
}


def _iso(value):
    if isinstance(value, datetime):
        return value.replace(tzinfo=timezone.utc).isoformat() if value.tzinfo is None else value.isoformat()
    if isinstance(value, bytes):
        return base64.b64encode(value).decode("ascii")
    return value


def _dates(record):
    values = record.model_dump()
    for key in ("created_at", "updated_at"):
        if key in values:
            values[key] = datetime.fromisoformat(values[key].replace("Z", "+00:00")).astimezone(timezone.utc).replace(tzinfo=None)
    return values


def _upgraded_calibration(values, camera):
    """A v1/v2 gate becomes a DOOR; with no room above its threshold it must be redrawn."""
    if values.get("line_start") is not None:
        top = extruded_door_top(values["line_start"], values["line_end"],
                                values["reference_width"], values["reference_height"], "DOOR")
        if top is None:
            return {"camera_id": camera.id, "role": values["role"]}
        values = {**values, "door_kind": "DOOR", "door_top": top}
    return values


def _schema(connection):
    try:
        if (attendance_schema._preflight(connection) != attendance_schema.SCHEMA_VERSION
                or operator_schema.preflight(connection) != operator_schema.VERSION):
            raise TransferError("TRANSFER_INCOMPATIBLE")
    except TransferError:
        raise
    except Exception:
        raise TransferError("TRANSFER_INCOMPATIBLE") from None


def _empty(connection):
    # Do not use _catalog's LIKE filter: '_' also matches non-system names.
    for (name,) in connection.exec_driver_sql("SELECT name FROM sqlite_master WHERE type='table'"):
        if name.startswith("sqlite_") or name in {"operator_schema", "attendance_runtime"}:
            continue
        quoted = connection.dialect.identifier_preparer.quote_identifier(name)
        if connection.exec_driver_sql(f"SELECT 1 FROM {quoted} LIMIT 1").first() is not None:
            return False
    return True


def _warnings(document, preparation):
    warnings = ["TRANSFER_NO_HISTORY_OR_VIDEO"]
    counts = {}
    for face in document.faces:
        counts[face.employee_id] = counts.get(face.employee_id, 0) + 1
    if any(not e.first_name or not e.last_name or not 1 <= counts.get(e.id, 0) <= 15 for e in document.employees):
        warnings.append("TRANSFER_LEGACY_REMEDIATION")
    if any(face.thumbnail is None for face in document.faces):
        warnings.append("TRANSFER_MISSING_THUMBNAILS")
    if preparation["faces_need_reenrollment"]:
        warnings.append("TRANSFER_FACES_NEED_REENROLLMENT")
    if preparation["scenes_need_review"]:
        warnings.append("TRANSFER_SCENES_NEED_REVIEW")
    return warnings


@dataclass(frozen=True, slots=True)
class _PreparedTransferFace:
    embedding: bytes
    thumbnail: bytes | None
    model_id: str | None
    alignment_id: str | None




@contextmanager
def _enrollment_slot(semaphore):
    if semaphore is not None and not semaphore.acquire(blocking=False):
        raise TransferError("ENROLLMENT_BUSY", 503)
    try:
        yield
    finally:
        if semaphore is not None:
            semaphore.release()


def _receipt(db, request_id, digest):
    prior = db.get(OperatorRequest, str(request_id))
    if prior is None:
        return None
    if prior.operation != "setup.import" or prior.payload_sha256 != digest:
        raise TransferError("TRANSFER_REQUEST_CONFLICT", 409)
    return dict(prior.result)


class SetupTransferService:
    def __init__(self, engine, *, face_model_id: str, face_engine=None, enrollment_semaphore=None):
        if not isinstance(face_model_id, str) or re.fullmatch(r"[0-9a-f]{64}", face_model_id) is None:
            raise TransferError("TRANSFER_MODEL_UNAVAILABLE", 503)
        self.engine, self.face_model_id = engine, face_model_id
        self.face_engine = face_engine
        self.enrollment_semaphore = enrollment_semaphore
        self.restart_required = False
        self.operation_lock = threading.Lock()

    def _prepare_faces(self, document):
        """One preview/commit path; no writer transaction may surround inference."""
        prepared, needing = [], []
        for index, row in enumerate(document.faces):
            embedding = decode_binary(row.embedding)
            thumbnail = decode_binary(row.thumbnail) if row.thumbnail is not None else None
            proven = (getattr(row, "face_model_id", None) == self.face_model_id
                      and getattr(row, "alignment_id", None) == ALIGNMENT_ID
                      and is_normalized_embedding(np.frombuffer(embedding, dtype="<f4")))
            prepared.append(_PreparedTransferFace(
                embedding, thumbnail, self.face_model_id if proven else None, ALIGNMENT_ID if proven else None))
            if not proven:
                needing.append(index)
        reencoded = 0
        if needing:
            engine = self.face_engine
            if (engine is not None and (engine.model_id != self.face_model_id
                                       or engine.alignment_id != ALIGNMENT_ID)):
                engine = None
            # No model/lease is needed to retain bytes that cannot be embedded.
            # The shared helper rejects missing engines only after finding a
            # valid aligned JPEG, before any model execution.
            lease = (_enrollment_slot(self.enrollment_semaphore) if engine is not None else nullcontext())
            with lease:
                try:
                    vectors = reembed_aligned_thumbnails(
                        engine, [prepared[index].thumbnail for index in needing])
                    if len(vectors) != len(needing):
                        raise ValueError("invalid preparation result")
                    for index, vector in zip(needing, vectors):
                        if vector is None:
                            continue
                        if not is_normalized_embedding(vector):
                            raise ValueError("invalid prepared embedding")
                        prepared[index] = _PreparedTransferFace(
                            np.asarray(vector, dtype="<f4").tobytes(), prepared[index].thumbnail,
                            self.face_model_id, ALIGNMENT_ID)
                        reencoded += 1
                except Exception:
                    raise TransferError("TRANSFER_MODEL_UNAVAILABLE", 503) from None
        return prepared, {"faces_reencoded": reencoded, "faces_need_reenrollment": len(needing) - reencoded,
                          "scenes_need_review": len(getattr(document, "scenes", []))}

    def status(self) -> dict:
        try:
            with self.engine.connect() as connection:
                connection.exec_driver_sql("BEGIN")
                _schema(connection)
                empty = _empty(connection)
            return {"available": True, "destination_empty": empty, "restart_required": self.restart_required,
                    "max_file_bytes": MAX_FILE_BYTES}
        except TransferError:
            raise
        except Exception:
            raise TransferError("TRANSFER_STORAGE_UNAVAILABLE", 503) from None

    def export(self, password: str) -> bytes:
        try:
            payload = {"transfer_version": 4, "exported_at": datetime.now(timezone.utc).isoformat(),
                       "attendance_schema_version": attendance_schema.SCHEMA_VERSION,
                       "operator_schema_version": operator_schema.VERSION,
                       "face_model_id": self.face_model_id, "embedding_encoding": "float32-le-512"}
            budget = 0
            with self.engine.connect() as connection:
                connection.exec_driver_sql("BEGIN")
                _schema(connection)
                for key, (model, names, maximum) in PROJECTIONS.items():
                    rows = []
                    # Bound row count and use streaming iteration, not query.all().
                    if key == "faces":
                        query = select(
                            EmployeeFace.id, EmployeeFace.employee_id, EmployeeFace.embedding,
                            EmployeeFace.thumbnail, EmployeeFace.created_at,
                            FaceModelMetadata.model_id, FaceModelMetadata.alignment_id,
                        ).outerjoin(FaceModelMetadata, FaceModelMetadata.face_id == EmployeeFace.id)
                    else:
                        query = select(*(getattr(model, name) for name in names))
                    query = query.limit(maximum + 1)
                    for row in connection.execute(query):
                        if len(rows) == maximum:
                            raise TransferError("TRANSFER_TOO_LARGE", 413)
                        value = {name: _iso(v) for name, v in zip(names, row)}
                        budget += len(json.dumps(value, ensure_ascii=False).encode("utf-8"))
                        if budget > MAX_FILE_BYTES:
                            raise TransferError("TRANSFER_TOO_LARGE", 413)
                        rows.append(value)
                    payload[key] = rows
            data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
            document = validate_document(data)
            return encode_transfer(document, password)
        except TransferError:
            raise
        except Exception:
            raise TransferError("TRANSFER_STORAGE_UNAVAILABLE", 503) from None

    def preview(self, blob: bytes, password: str) -> dict:
        document = decode_transfer(blob, password, expected_face_model_id=self.face_model_id)
        _, preparation = self._prepare_faces(document)
        return {"sha256": hashlib.sha256(blob).hexdigest(), "counts": document.counts().model_dump(),
                **preparation, "warnings": _warnings(document, preparation),
                "destination_empty": self.status()["destination_empty"]}

    def import_setup(self, blob: bytes, password: str, *, request_id: UUID, expected_sha256: str) -> tuple[dict, bool]:
        document = decode_transfer(blob, password, expected_face_model_id=self.face_model_id)
        digest = hashlib.sha256(blob).hexdigest()
        if digest != expected_sha256:
            raise TransferError("TRANSFER_DIGEST_MISMATCH", 409)
        try:
            # Authenticate before replay, but do not require a model or regenerate
            # samples after a matching receipt has already frozen the result.
            with Session(self.engine) as db:
                _schema(db.connection())
                prior = _receipt(db, request_id, digest)
                if prior is not None:
                    return prior, True
                if not _empty(db.connection()):
                    raise TransferError("TRANSFER_DESTINATION_NOT_EMPTY", 409)
            faces, preparation = self._prepare_faces(document)
            with immediate_session(self.engine, publish=True) as db:
                _schema(db.connection())
                prior = _receipt(db, request_id, digest)
                if prior is not None:
                    return prior, True
                if not _empty(db.connection()):
                    raise TransferError("TRANSFER_DESTINATION_NOT_EMPTY", 409)
                for row in document.cameras:
                    db.add(Camera(**_dates(row)))
                for row in document.workplaces:
                    db.add(Workplace(**row.model_dump(), name_key=row.name.casefold()))
                for row in document.employees:
                    db.add(Employee(**_dates(row), revision=1))
                db.flush()
                by_camera = {row.camera_id: row for row in document.calibrations}
                for row in document.cameras:
                    calibration = by_camera.get(row.id)
                    values = calibration.model_dump() if calibration else {"camera_id": row.id}
                    if document.transfer_version < 3:
                        values = _upgraded_calibration(values, row)
                    db.add(AttendanceCamera(**values, enabled=False, revision=0, source_revision=1))
                profiles = {row.camera_id: row for row in getattr(document, "profiles", [])}
                for row in document.cameras:
                    profile = profiles.get(row.id)
                    if profile is not None:
                        values = profile.model_dump()
                        # Version 3 had no time tracking: every AI camera starts tracking time.
                        values.setdefault("time_tracking", values["ai_enabled"])
                    else:
                        values = dict(
                            camera_id=row.id, workplace_id=None, ai_fps=None, ai_enabled=True, time_tracking=True,
                            zone="ENTRANCE" if by_camera.get(row.id) and by_camera[row.id].role else "INTERIOR")
                    db.add(CameraProfile(**values, revision=0))
                for row in document.archives:
                    db.add(CameraArchive(**row.model_dump(), revision=0))
                for row in getattr(document, "scenes", []):
                    values = row.model_dump()
                    values["geometry"] = {**values["geometry"], "version": 2,
                                          "doors": values["geometry"].get("doors", [])}
                    db.add(CameraScene(**values, source_revision=0, revision=1))
                for row, face in zip(document.faces, faces):
                    values = _dates(row)
                    values.pop("face_model_id", None)
                    values.pop("alignment_id", None)
                    values["embedding"], values["thumbnail"] = face.embedding, face.thumbnail
                    db.add(EmployeeFace(**values))
                db.flush()
                for row, face in zip(document.faces, faces):
                    if face.model_id is not None:
                        db.add(FaceModelMetadata(face_id=row.id, model_id=face.model_id, alignment_id=face.alignment_id))
                for row in document.assignments:
                    db.add(EmployeeWorkplace(**row.model_dump()))
                db.execute(update(AttendanceRuntime).where(AttendanceRuntime.id == 1).values(
                    gallery_revision=AttendanceRuntime.gallery_revision + 1))
                result = {"committed": True, "counts": document.counts().model_dump(), "restart_required": True,
                          **preparation, "warnings": _warnings(document, preparation)}
                db.add(OperatorRequest(id=str(request_id), operation="setup.import", payload_sha256=digest,
                                       result=result))
            self.restart_required = True
            return result, False
        except TransferError:
            raise
        except Exception:
            raise TransferError("TRANSFER_STORAGE_UNAVAILABLE", 503) from None
