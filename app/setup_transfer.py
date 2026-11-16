"""Versioned, authenticated setup transfer; never SQL, pickle, archives or video."""
import base64
import binascii
import io
import json
import math
import os
import struct
import warnings
from datetime import datetime
from typing import Annotated, Literal

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt
from PIL import Image
from app.ai.doorway import DoorwayCalibration
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from app.rtsp.dahua import normalize_host

MAX_FILE_BYTES = 64 * 1024 * 1024
MAX_BODY_BYTES = MAX_FILE_BYTES + 16 * 1024
HEADER = b"NMSETUP\x00\x01"
OVERHEAD = 8 + 1 + 16 + 12 + 16
ID = Annotated[int, Field(strict=True, ge=1, le=2**63-1)]
Port = Annotated[int, Field(strict=True, ge=1, le=65535)]


class TransferError(Exception):
    def __init__(self, code: str, status_code: int = 422):
        self.code, self.status_code = code, status_code
        super().__init__(code)


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False, hide_input_in_errors=True)


def _date(value):
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or not 1 <= parsed.year <= 9998:
        raise ValueError("invalid date")
    return value


class DatedRecord(Record):
    created_at: str
    _dates = field_validator("created_at")(_date)


class CameraRecord(DatedRecord):
    id: ID
    name: str = Field(min_length=1, max_length=100)
    nvr_host: str = Field(min_length=1, max_length=255)
    rtsp_port: Port
    username: str = Field(min_length=1, max_length=100)
    password: str = Field(max_length=255, repr=False)
    channel: int = Field(ge=1, le=64)
    stream_type: Literal["main", "sub"]
    enabled: bool
    updated_at: str
    _updated = field_validator("updated_at")(_date)

    @field_validator("nvr_host")
    @classmethod
    def host(cls, value):
        normalize_host(value)  # Validate, never silently rewrite the stored address.
        return value


class ArchiveRecord(Record):
    camera_id: ID
    http_scheme: Literal["http", "https"]
    http_port: Port
    utc_offset_minutes: int = Field(ge=-720, le=840)


class CalibrationRecord(Record):
    camera_id: ID
    role: Literal["ENTRY", "EXIT"] | None
    line_start: list[float] | None
    line_end: list[float] | None
    inside_point: list[float] | None
    reference_width: int | None = Field(gt=0)
    reference_height: int | None = Field(gt=0)

    @model_validator(mode="after")
    def geometry(self):
        points = (self.line_start, self.line_end, self.inside_point)
        values = (*points, self.reference_width, self.reference_height)
        if any(v is not None for v in values):
            if any(v is None for v in values):
                raise ValueError("incomplete geometry")
            DoorwayCalibration(self.camera_id, self.role or "ENTRY", 0, 1,
                               *points, self.reference_width, self.reference_height)
        return self


class WorkplaceRecord(Record):
    id: ID
    name: str = Field(min_length=1, max_length=150)
    active: bool

    @field_validator("name")
    @classmethod
    def name_value(cls, value):
        if value != " ".join(value.split()):
            raise ValueError("invalid workplace name")
        return value


class EmployeeRecord(DatedRecord):
    id: ID
    name: str = Field(min_length=1, max_length=150)
    first_name: str | None = Field(max_length=74)
    last_name: str | None = Field(max_length=74)
    external_id: str | None = Field(max_length=64)
    active: bool
    updated_at: str
    _updated = field_validator("updated_at")(_date)


class FaceRecord(DatedRecord):
    id: ID
    employee_id: ID
    embedding: str = Field(repr=False, min_length=2732, max_length=2732)
    thumbnail: str | None = Field(repr=False, max_length=1398104)

    @field_validator("embedding")
    @classmethod
    def embedding_value(cls, value):
        raw = decode_binary(value)
        if len(raw) != 2048:
            raise ValueError("invalid embedding")
        vector = struct.unpack("<512f", raw)
        if not all(math.isfinite(v) for v in vector) or sum(v*v for v in vector) < 1e-12:
            raise ValueError("invalid embedding")
        return value

    @field_validator("thumbnail")
    @classmethod
    def thumbnail_value(cls, value):
        if value is None:
            return None
        raw = decode_binary(value)
        if len(raw) > 1024*1024:
            raise ValueError("image too large")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(io.BytesIO(raw)) as image:
                    if image.format not in {"JPEG", "PNG"} or image.width*image.height > 1_000_000 or getattr(image, "is_animated", False):
                        raise ValueError("invalid image")
                    image.verify()
                with Image.open(io.BytesIO(raw)) as image:
                    image.load()
        except Exception:
            raise ValueError("invalid image") from None
        return value


class AssignmentRecord(Record):
    employee_id: ID
    workplace_id: ID


class TransferCounts(Record):
    cameras: int
    workplaces: int
    employees: int
    faces: int
    assignments: int


class SetupDocument(Record):
    transfer_version: Literal[1]
    exported_at: str
    attendance_schema_version: Literal[1]
    operator_schema_version: Literal[3]
    embedding_model: str = Field(min_length=1, max_length=100)
    embedding_encoding: Literal["float32-le-512"]
    cameras: list[CameraRecord] = Field(max_length=10000)
    archives: list[ArchiveRecord] = Field(max_length=10000)
    calibrations: list[CalibrationRecord] = Field(max_length=10000)
    workplaces: list[WorkplaceRecord] = Field(max_length=10000)
    employees: list[EmployeeRecord] = Field(max_length=10000)
    faces: list[FaceRecord] = Field(max_length=150000, repr=False)
    assignments: list[AssignmentRecord] = Field(max_length=100000)
    _export_date = field_validator("exported_at")(_date)

    @model_validator(mode="after")
    def references(self):
        def unique(rows, key):
            ids = [getattr(r, key) for r in rows]
            if len(set(ids)) != len(ids):
                raise ValueError("duplicate identifier")
            return set(ids)
        cameras = unique(self.cameras, "id")
        employees = unique(self.employees, "id")
        workplaces = unique(self.workplaces, "id")
        unique(self.faces, "id")
        for rows in (self.archives, self.calibrations):
            if not unique(rows, "camera_id") <= cameras:
                raise ValueError("missing camera")
        if any(r.employee_id not in employees for r in self.faces):
            raise ValueError("missing employee")
        links = {(r.employee_id, r.workplace_id) for r in self.assignments}
        if len(links) != len(self.assignments) or any(e not in employees or w not in workplaces for e, w in links):
            raise ValueError("invalid assignment")
        if len({r.name.casefold() for r in self.workplaces}) != len(self.workplaces):
            raise ValueError("duplicate workplace")
        return self

    def counts(self) -> TransferCounts:
        return TransferCounts(**{key: len(getattr(self, key)) for key in TransferCounts.model_fields})


def decode_binary(value: str) -> bytes:
    try:
        return base64.b64decode(value, validate=True)
    except (ValueError, binascii.Error):
        raise ValueError("invalid binary data") from None


def _unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate key")
        result[key] = value
    return result


def _bounded_json(payload: bytes):
    if len(payload) > MAX_FILE_BYTES - OVERHEAD:
        raise TransferError("TRANSFER_TOO_LARGE", 413)
    # Scan nesting before the JSON decoder can recurse; ignore brackets in strings.
    depth = 0
    quoted = escaped = False
    for char in payload:
        if quoted:
            if escaped:
                escaped = False
            elif char == 92:
                escaped = True
            elif char == 34:
                quoted = False
        elif char == 34:
            quoted = True
        elif char in (91, 123):
            depth += 1
            if depth > 16:
                raise TransferError("TRANSFER_INVALID")
        elif char in (93, 125):
            depth -= 1
    return json.loads(payload, object_pairs_hook=_unique_keys,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite")))


def validate_document(payload: bytes, *, expected_model: str) -> SetupDocument:
    try:
        raw = _bounded_json(payload)
        if isinstance(raw, dict) and any(raw.get(k) != v for k, v in (
            ("transfer_version", 1), ("attendance_schema_version", 1),
            ("operator_schema_version", 3), ("embedding_encoding", "float32-le-512"))):
            raise TransferError("TRANSFER_INCOMPATIBLE")
        result = SetupDocument.model_validate(raw)
        if result.embedding_model != expected_model:
            raise TransferError("TRANSFER_MODEL_MISMATCH")
        return result
    except TransferError:
        raise
    except (ValueError, TypeError, ValidationError, RecursionError, OverflowError):
        raise TransferError("TRANSFER_INVALID") from None


def _key(password: str, salt: bytes) -> bytes:
    raw = password.encode("utf-8")
    if not 12 <= len(raw) <= 1024:
        raise TransferError("TRANSFER_PASSWORD_INVALID")
    return Scrypt(salt=salt, length=32, n=32768, r=8, p=1).derive(raw)


def encode_transfer(document: SetupDocument, password: str) -> bytes:
    # iterencode avoids constructing an over-limit serialized string before checking.
    output = bytearray()
    for chunk in json.JSONEncoder(ensure_ascii=False, separators=(",", ":"), allow_nan=False).iterencode(document.model_dump()):
        encoded = chunk.encode("utf-8")
        if len(output) + len(encoded) > MAX_FILE_BYTES - OVERHEAD:
            raise TransferError("TRANSFER_TOO_LARGE", 413)
        output.extend(encoded)
    # Models may have been constructed unsafely by a caller; the export must still validate.
    validate_document(output, expected_model=document.embedding_model)
    salt, nonce = os.urandom(16), os.urandom(12)
    header = HEADER + salt + nonce
    return header + AESGCM(_key(password, salt)).encrypt(nonce, bytes(output), header)


def decode_transfer(blob: bytes, password: str, *, expected_model: str) -> SetupDocument:
    if len(blob) > MAX_FILE_BYTES:
        raise TransferError("TRANSFER_TOO_LARGE", 413)
    if len(blob) < OVERHEAD or blob[:8] != HEADER[:8]:
        raise TransferError("TRANSFER_INVALID")
    if blob[8] != 1:
        raise TransferError("TRANSFER_INCOMPATIBLE")
    try:
        plaintext = AESGCM(_key(password, blob[9:25])).decrypt(blob[25:37], blob[37:], blob[:37])
    except InvalidTag:
        raise TransferError("TRANSFER_PASSWORD_OR_FILE_INVALID") from None
    return validate_document(plaintext, expected_model=expected_model)
