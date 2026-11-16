"""Versioned, authenticated setup transfer; never SQL, pickle, archives or video."""
import base64
import binascii
import json
import os
from uuid import UUID
from datetime import datetime
from typing import Annotated, Literal

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt
from app.ai.doorway import DoorwayCalibration
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from app.rtsp.dahua import normalize_host
from app.scene_geometry import validate_geometry

MAX_FILE_BYTES = 64 * 1024 * 1024
MAX_BODY_BYTES = MAX_FILE_BYTES + 16 * 1024
HEADER = b"NMSETUP\x00\x01"
OVERHEAD = 8 + 1 + 16 + 12 + 16
ID = Annotated[int, Field(strict=True, ge=1, le=2**53-1)]
Port = Annotated[int, Field(strict=True, ge=1, le=65535)]
FaceModelID = Annotated[str, Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")]
ALIGNMENT_ID = "arcface-5pt-112-v1"
Coordinate = Annotated[float, Field(ge=0, le=1)]
Point = Annotated[list[Coordinate], Field(min_length=2, max_length=2)]
Polygon = Annotated[list[Point], Field(min_length=3, max_length=128)]


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


class LegacyFaceRecord(DatedRecord):
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
        # Shape/encoding is a wire constraint; validity in the destination vector
        # space is decided during preparation, never inferred from legacy bytes.
        return value

    @field_validator("thumbnail")
    @classmethod
    def thumbnail_value(cls, value):
        if value is None:
            return None
        raw = decode_binary(value)
        if len(raw) > 1024*1024:
            raise ValueError("image too large")
        # Unusable legacy thumbnails are retained as bounded opaque bytes. Only the
        # shared aligned-JPEG preparation helper may turn them into usable samples.
        return value


class FaceRecord(LegacyFaceRecord):
    face_model_id: FaceModelID | None
    alignment_id: str | None = Field(min_length=1, max_length=40)

    @model_validator(mode="after")
    def provenance(self):
        if (self.face_model_id is None) != (self.alignment_id is None):
            raise ValueError("incomplete provenance")
        return self


class SceneObjectRecord(Record):
    id: str = Field(min_length=36, max_length=36)
    label: str = Field(min_length=1, max_length=80)
    polygon: Polygon

    @field_validator("id")
    @classmethod
    def object_id(cls, value):
        if str(UUID(value)) != value:
            raise ValueError("invalid object identifier")
        return value


class SceneMetricRecord(Record):
    corners: list[Point] = Field(min_length=4, max_length=4)
    width_m: float = Field(gt=0, le=1000)
    depth_m: float = Field(gt=0, le=1000)

    @field_validator("corners")
    @classmethod
    def distinct_corners(cls, value):
        if len({tuple(point) for point in value}) != 4:
            raise ValueError("duplicate metric corner")
        return value


class TransferSceneGeometry(Record):
    """Transfer's bounded wire contract; geometric authority is app.scene_geometry."""
    version: Annotated[int, Field(strict=True, ge=1, le=1)]
    floor: Polygon | None
    objects: list[SceneObjectRecord] = Field(max_length=32)
    metric: SceneMetricRecord | None

    @model_validator(mode="after")
    def structure(self):
        validate_geometry(self.model_dump())
        return self


class SceneRecord(Record):
    camera_id: ID
    reference_width: int = Field(gt=0)
    reference_height: int = Field(gt=0)
    geometry: TransferSceneGeometry


class AssignmentRecord(Record):
    employee_id: ID
    workplace_id: ID


class TransferCounts(Record):
    cameras: int
    workplaces: int
    employees: int
    faces: int
    assignments: int
    scenes: int


class _SetupRecords(Record):
    exported_at: str
    attendance_schema_version: Literal[1]
    embedding_encoding: Literal["float32-le-512"]
    cameras: list[CameraRecord] = Field(max_length=10000)
    archives: list[ArchiveRecord] = Field(max_length=10000)
    calibrations: list[CalibrationRecord] = Field(max_length=10000)
    workplaces: list[WorkplaceRecord] = Field(max_length=10000)
    employees: list[EmployeeRecord] = Field(max_length=10000)
    faces: list[LegacyFaceRecord] = Field(max_length=150000, repr=False)
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
        for rows in (self.archives, self.calibrations, getattr(self, "scenes", [])):
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
        return TransferCounts(**{key: len(getattr(self, key, [])) for key in TransferCounts.model_fields})


class LegacySetupDocument(_SetupRecords):
    """Authenticated v1 data only; its package name never selects a runtime model."""
    transfer_version: Literal[1]
    operator_schema_version: Literal[3]
    embedding_model: str = Field(min_length=1, max_length=100, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


class SetupDocument(_SetupRecords):
    transfer_version: Literal[2]
    operator_schema_version: Literal[4]
    face_model_id: FaceModelID
    faces: list[FaceRecord] = Field(max_length=150000, repr=False)
    scenes: list[SceneRecord] = Field(max_length=10000)


TransferDocument = SetupDocument | LegacySetupDocument


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
    # Bound containers BEFORE json.loads builds Python objects. A byte limit alone
    # permits millions of tiny dicts. This scanner only budgets; json.loads remains
    # the syntax/UTF-8 authority, including escapes and duplicate keys.
    limits = {"cameras": 10000, "archives": 10000, "calibrations": 10000,
              "workplaces": 10000, "employees": 10000, "faces": 150000,
              "assignments": 100000, "scenes": 10000}
    stack = []  # kind, item/member count, awaiting array item, maximum
    quoted = escaped = False
    start = 0
    root_key = None
    nodes = 0
    for index, char in enumerate(payload):
        if quoted:
            if escaped:
                escaped = False
            elif char == 92:
                escaped = True
            elif char == 34:
                quoted = False
                if len(stack) == 1 and stack[0][0] == 123:
                    token = payload[start:index+1]
                    root_key = token if len(token) <= 128 else None
            continue
        if char in (9, 10, 13, 32):
            continue
        if stack and stack[-1][0] == 91 and stack[-1][2] and char != 93:
            stack[-1][1] += 1
            stack[-1][2] = False
            if stack[-1][1] > stack[-1][3]:
                raise TransferError("TRANSFER_TOO_LARGE", 413)
        if char == 34:
            quoted, start = True, index
        elif char in (91, 123):
            maximum = 150000
            if char == 91 and len(stack) == 1 and root_key is not None:
                maximum = limits.get(json.loads(root_key), maximum)
            stack.append([char, 0, char == 91, maximum])
            nodes += 1
            if len(stack) > 16:
                raise TransferError("TRANSFER_INVALID")
        elif char in (93, 125):
            if stack:
                stack.pop()
        elif char == 44:
            nodes += 1
            if stack and stack[-1][0] == 91:
                stack[-1][2] = True
        elif char == 58 and stack and stack[-1][0] == 123:
            stack[-1][1] += 1
            if stack[-1][1] > 32:
                raise TransferError("TRANSFER_INVALID")
        if nodes > 2_000_000:
            raise TransferError("TRANSFER_TOO_LARGE", 413)
    return json.loads(payload, object_pairs_hook=_unique_keys,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite")))


def validate_document(payload: bytes, *, expected_face_model_id: str | None = None) -> TransferDocument:
    try:
        raw = _bounded_json(payload)
        if not isinstance(raw, dict):
            raise TransferError("TRANSFER_INVALID")
        version = raw.get("transfer_version")
        schema_version = {1: 3, 2: 4}.get(version) if type(version) is int else None
        if schema_version is None or any(type(raw.get(k)) is not type(v) or raw.get(k) != v for k, v in (
            ("attendance_schema_version", 1), ("operator_schema_version", schema_version),
            ("embedding_encoding", "float32-le-512"))):
            raise TransferError("TRANSFER_INCOMPATIBLE")
        model = SetupDocument if version == 2 else LegacySetupDocument
        result = model.model_validate(raw)
        if version == 2 and expected_face_model_id is not None:
            if result.face_model_id != expected_face_model_id or any(
                face.face_model_id is not None and (
                    face.face_model_id != expected_face_model_id or face.alignment_id != ALIGNMENT_ID
                ) for face in result.faces
            ):
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


def encode_transfer(document: TransferDocument, password: str) -> bytes:
    # iterencode avoids constructing an over-limit serialized string before checking.
    output = bytearray()
    for chunk in json.JSONEncoder(ensure_ascii=False, separators=(",", ":"), allow_nan=False).iterencode(document.model_dump()):
        encoded = chunk.encode("utf-8")
        if len(output) + len(encoded) > MAX_FILE_BYTES - OVERHEAD:
            raise TransferError("TRANSFER_TOO_LARGE", 413)
        output.extend(encoded)
    # Models may have been constructed unsafely by a caller; the export must still validate.
    validate_document(output)
    salt, nonce = os.urandom(16), os.urandom(12)
    header = HEADER + salt + nonce
    return header + AESGCM(_key(password, salt)).encrypt(nonce, bytes(output), header)


def decode_transfer(blob: bytes, password: str, *, expected_face_model_id: str | None = None) -> TransferDocument:
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
    return validate_document(plaintext, expected_face_model_id=expected_face_model_id)
