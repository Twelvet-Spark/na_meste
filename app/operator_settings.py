"""Strict operator-only overrides; deployment configuration is never rewritten."""
import hashlib
import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Annotated

from pydantic import AliasChoices, TypeAdapter, ValidationError
from pydantic_settings import EnvSettingsSource, DotEnvSettingsSource

BASE_DIR = Path(__file__).resolve().parent.parent
EDITABLE = (
    "preview_fps", "rtsp_connect_timeout", "rtsp_reconnect_max_delay", "frame_stall_timeout",
    "ai_enabled", "ai_device", "ai_fps", "ai_cameras",
    "attendance_ai_fps", "attendance_frame_max_age", "attendance_max_frame_gap",
    "attendance_face_threshold", "attendance_face_margin", "attendance_face_confirmations",
    "attendance_face_window", "attendance_face_interval", "attendance_face_min_size",
    "attendance_face_det_score", "attendance_face_blur_min", "attendance_evidence_keep_days",
    "telegram_enabled", "telegram_bot_token", "telegram_chat_id", "telegram_notify_presence",
    "telegram_notify_global_transition", "telegram_notify_global_new",
    "recording_retention_days", "recording_max_bytes", "mc_debug",
)
SECRET = "telegram_bot_token"
# One writer across service instances in this single-worker deployment. Read/CAS/
# validate/replace all hold this lock; no separate read-then-write race.
WRITE_LOCK = threading.RLock()
MAX_DOCUMENT_BYTES = 65536


class OverrideError(ValueError):
    """Only static codes cross the startup/API boundary, never validation inputs."""


def settings_path():
    value = Path(os.environ.get("OPERATOR_SETTINGS_PATH") or "data/operator-settings.json")
    return value if value.is_absolute() else BASE_DIR / value


def normalize_camera_ids(value):
    if not isinstance(value, str):
        raise ValueError("AI_CAMERAS_INVALID")
    if not value.strip():
        return ""
    parts = [part.strip() for part in value.replace(";", ",").split(",")]
    if any(not part.isascii() or not part.isdecimal() or int(part) <= 0 for part in parts):
        raise ValueError("AI_CAMERAS_INVALID")
    return ",".join(str(item) for item in sorted({int(part) for part in parts}))


def _names(name, field):
    alias = field.validation_alias
    if isinstance(alias, AliasChoices):
        return [value for value in alias.choices if isinstance(value, str)] + [name]
    return [alias, name] if isinstance(alias, str) else [name]


def source_fields(source, settings_cls):
    # Use Pydantic's actual source (case/alias/empty-value rules), not guessed ENV names.
    values = source()
    return {name for name, field in settings_cls.model_fields.items()
            if any(key in values for key in _names(name, field))}


def environment_fields(settings_cls):
    return source_fields(EnvSettingsSource(settings_cls), settings_cls)


def dotenv_fields(settings_cls, env_file):
    return source_fields(DotEnvSettingsSource(settings_cls, env_file=env_file), settings_cls)


def validate_values(values, settings_cls):
    if type(values) is not dict or any(key not in EDITABLE for key in values):
        raise OverrideError("OPERATOR_SETTINGS_UNKNOWN_FIELDS")
    result = {}
    try:
        for name, value in values.items():
            field = settings_cls.model_fields[name]
            annotation = Annotated[field.annotation, *field.metadata] if field.metadata else field.annotation
            result[name] = TypeAdapter(annotation).validate_python(value, strict=True)
        if "ai_cameras" in result:
            result["ai_cameras"] = normalize_camera_ids(result["ai_cameras"])
        if ("attendance_face_window" in result and "attendance_face_interval" in result
                and result["attendance_face_interval"] > result["attendance_face_window"]):
            raise ValueError("ATTENDANCE_INTERVAL_EXCEEDS_WINDOW")
    except (ValidationError, ValueError, TypeError, OverflowError):
        raise OverrideError("OPERATOR_SETTINGS_INVALID_VALUES") from None
    return result


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate key")
        result[key] = value
    return result


def _reject_nonfinite(_value):
    raise ValueError("nonfinite")


def read_document(path, settings_cls):
    try:
        with Path(path).open("rb") as stream:
            raw = stream.read(MAX_DOCUMENT_BYTES + 1)
    except FileNotFoundError:
        return {}, hashlib.sha256(b"").hexdigest()
    except OSError:
        raise OverrideError("OPERATOR_SETTINGS_READ_FAILED: проверьте доступ к файлу настроек") from None
    try:
        if len(raw) > MAX_DOCUMENT_BYTES:
            raise ValueError("size")
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=_object, parse_constant=_reject_nonfinite)
        if (type(value) is not dict or set(value) != {"version", "values"}
                or type(value["version"]) is not int or value["version"] != 1):
            raise ValueError("shape")
    except (ValueError, UnicodeError, RecursionError):
        raise OverrideError("OPERATOR_SETTINGS_INVALID_DOCUMENT: исправьте JSON версии 1 перед запуском") from None
    return validate_values(value["values"], settings_cls), hashlib.sha256(raw).hexdigest()


def replace_document(path, values, expected_version, settings_cls):
    """Caller holds WRITE_LOCK. Recheck external edits immediately before replace."""
    path = Path(path)
    temporary = None
    try:
        raw = (json.dumps({"version": 1, "values": values}, ensure_ascii=False,
                          sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
        if len(raw) > MAX_DOCUMENT_BYTES:
            raise OverrideError("OPERATOR_SETTINGS_DOCUMENT_TOO_LARGE")
        path.parent.mkdir(parents=True, exist_ok=True)
        # mkstemp uses restrictive permissions where supported; same volume for replace.
        descriptor, temporary = tempfile.mkstemp(prefix=".operator-settings-", suffix=".tmp", dir=path.parent)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        if read_document(path, settings_cls)[1] != expected_version:
            raise OverrideError("OPERATOR_SETTINGS_CONFLICT")
        os.replace(temporary, path)
        temporary = None
        return hashlib.sha256(raw).hexdigest()
    except OSError:
        raise OverrideError("OPERATOR_SETTINGS_WRITE_FAILED") from None
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except OSError:
                pass
