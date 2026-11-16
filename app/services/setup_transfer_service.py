"""Logical setup snapshots and empty-only imports, independent of media services."""
import base64
import hashlib
import json
import threading
from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from app.database import attendance_schema, operator_schema
from app.database.database import immediate_session
from app.database.models import (AttendanceCamera, AttendanceRuntime, Camera, CameraArchive,
                                 Employee, EmployeeFace, EmployeeWorkplace, OperatorRequest, Workplace)
from app.setup_transfer import (MAX_FILE_BYTES, SetupDocument, TransferError, decode_binary,
                                decode_transfer, encode_transfer, validate_document)

# Explicit projection: adding a database column cannot silently expand the export.
PROJECTIONS = {
    "cameras": (Camera, ("id", "name", "nvr_host", "rtsp_port", "username", "password", "channel", "stream_type", "enabled", "created_at", "updated_at"), 10000),
    "archives": (CameraArchive, ("camera_id", "http_scheme", "http_port", "utc_offset_minutes"), 10000),
    "calibrations": (AttendanceCamera, ("camera_id", "role", "line_start", "line_end", "inside_point", "reference_width", "reference_height"), 10000),
    "workplaces": (Workplace, ("id", "name", "active"), 10000),
    "employees": (Employee, ("id", "name", "first_name", "last_name", "external_id", "active", "created_at", "updated_at"), 10000),
    "faces": (EmployeeFace, ("id", "employee_id", "embedding", "thumbnail", "created_at"), 150000),
    "assignments": (EmployeeWorkplace, ("employee_id", "workplace_id"), 100000),
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


def _schema(connection):
    try:
        if not attendance_schema._preflight(connection) or operator_schema.preflight(connection) != 3:
            raise TransferError("TRANSFER_INCOMPATIBLE")
    except TransferError:
        raise
    except Exception:
        raise TransferError("TRANSFER_INCOMPATIBLE") from None


def _empty(connection):
    for name, (kind, _, _) in attendance_schema._catalog(connection).items():
        if kind != "table" or name.startswith("sqlite_") or name in {"operator_schema", "attendance_runtime"}:
            continue
        quoted = connection.dialect.identifier_preparer.quote_identifier(name)
        if connection.exec_driver_sql(f"SELECT 1 FROM {quoted} LIMIT 1").first() is not None:
            return False
    return True


def _warnings(document):
    warnings = ["TRANSFER_NO_HISTORY_OR_VIDEO", "TRANSFER_RECOMMISSION_REQUIRED"]
    counts = {}
    for face in document.faces:
        counts[face.employee_id] = counts.get(face.employee_id, 0) + 1
    if any(not e.first_name or not e.last_name or not 1 <= counts.get(e.id, 0) <= 15 for e in document.employees):
        warnings.append("TRANSFER_LEGACY_REMEDIATION")
    if any(face.thumbnail is None for face in document.faces):
        warnings.append("TRANSFER_MISSING_THUMBNAILS")
    return warnings


class SetupTransferService:
    def __init__(self, engine, *, embedding_model: str):
        self.engine, self.embedding_model = engine, embedding_model
        self.restart_required = False
        self.operation_lock = threading.Lock()

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
            payload = {"transfer_version": 1, "exported_at": datetime.now(timezone.utc).isoformat(),
                       "attendance_schema_version": 1, "operator_schema_version": 3,
                       "embedding_model": self.embedding_model, "embedding_encoding": "float32-le-512"}
            budget = 0
            with self.engine.connect() as connection:
                connection.exec_driver_sql("BEGIN")
                _schema(connection)
                for key, (model, names, maximum) in PROJECTIONS.items():
                    rows = []
                    # Bound row count and use streaming iteration, not query.all().
                    query = select(*(getattr(model, name) for name in names)).limit(maximum + 1)
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
            document = validate_document(data, expected_model=self.embedding_model)
            return encode_transfer(document, password)
        except TransferError:
            raise
        except Exception:
            raise TransferError("TRANSFER_STORAGE_UNAVAILABLE", 503) from None

    def preview(self, blob: bytes, password: str) -> dict:
        document = decode_transfer(blob, password, expected_model=self.embedding_model)
        return {"sha256": hashlib.sha256(blob).hexdigest(), "counts": document.counts().model_dump(),
                "warnings": _warnings(document), "destination_empty": self.status()["destination_empty"]}

    def import_setup(self, blob: bytes, password: str, *, request_id: UUID, expected_sha256: str) -> tuple[dict, bool]:
        document = decode_transfer(blob, password, expected_model=self.embedding_model)
        digest = hashlib.sha256(blob).hexdigest()
        if digest != expected_sha256:
            raise TransferError("TRANSFER_DIGEST_MISMATCH", 409)
        try:
            with immediate_session(self.engine) as db:
                _schema(db.connection())
                prior = db.get(OperatorRequest, str(request_id))
                if prior is not None:
                    if prior.operation != "setup.import" or prior.payload_sha256 != digest:
                        raise TransferError("TRANSFER_REQUEST_CONFLICT", 409)
                    return dict(prior.result), True
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
                    db.add(AttendanceCamera(**values, enabled=False, revision=0, source_revision=1))
                for row in document.archives:
                    db.add(CameraArchive(**row.model_dump(), revision=0))
                for row in document.faces:
                    values = _dates(row)
                    values["embedding"] = decode_binary(row.embedding)
                    values["thumbnail"] = decode_binary(row.thumbnail) if row.thumbnail is not None else None
                    db.add(EmployeeFace(**values))
                for row in document.assignments:
                    db.add(EmployeeWorkplace(**row.model_dump()))
                db.execute(update(AttendanceRuntime).where(AttendanceRuntime.id == 1).values(
                    gallery_revision=AttendanceRuntime.gallery_revision + 1))
                result = {"committed": True, "counts": document.counts().model_dump(), "restart_required": True,
                          "warnings": _warnings(document)}
                db.add(OperatorRequest(id=str(request_id), operation="setup.import", payload_sha256=digest,
                                       result=result))
            self.restart_required = True
            return result, False
        except TransferError:
            raise
        except Exception:
            raise TransferError("TRANSFER_STORAGE_UNAVAILABLE", 503) from None
