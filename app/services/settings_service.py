"""Immutable active settings and serialized next-start configuration transactions."""
import logging
from pathlib import Path

from fastapi import HTTPException
from sqlalchemy.orm import Session
from pydantic_settings import SecretsSettingsSource

from app.config import BASE_DIR, Settings
from app.database.models import Camera
from app.operator_settings import (
    EDITABLE, SECRET, WRITE_LOCK, OverrideError, dotenv_fields, environment_fields,
    read_document, replace_document, settings_path, source_fields, validate_values,
)

logger = logging.getLogger("app.operator.settings")


def public_values(settings):
    result = {name: getattr(settings, name) for name in EDITABLE if name != SECRET}
    result["has_telegram_bot_token"] = bool(settings.telegram_bot_token)
    return result


class SettingsService:
    def __init__(self, active, engine, *, path=None, env_file=BASE_DIR / ".env"):
        self.active, self.engine = active, engine
        self.path = Path(path) if path is not None else settings_path()
        self.env_file = env_file
        schema = Settings.model_json_schema(by_alias=False)["properties"]
        self.constraints = {key: {name: value for name, value in schema[key].items()
                                 if name in {"type", "minimum", "maximum", "exclusiveMinimum",
                                             "exclusiveMaximum", "enum", "minLength", "maxLength"}}
                            for key in EDITABLE}

    def _effective(self, values):
        try:
            return Settings(_env_file=self.env_file, **values)
        except Exception:
            raise HTTPException(422, "OPERATOR_SETTINGS_INVALID_EFFECTIVE_CONFIG") from None

    def _snapshot(self, values, version, pending):
        try:
            environment = environment_fields(Settings)
            dotenv = dotenv_fields(Settings, self.env_file)
            secrets = source_fields(SecretsSettingsSource(Settings), Settings)
        except Exception:
            raise HTTPException(503, "OPERATOR_SETTINGS_SOURCES_UNAVAILABLE") from None
        changed = [name for name in EDITABLE if getattr(self.active, name) != getattr(pending, name)]
        return {
            "version": version, "active": public_values(self.active), "pending": public_values(pending),
            "fields": {name: {
                "constraints": self.constraints[name], "locked": name in environment,
                "source": ("environment" if name in environment else "operator" if name in values
                           else "dotenv" if name in dotenv else "file_secret" if name in secrets else "default"),
                "overridden": name in values,
            } for name in EDITABLE},
            "read_only": {"attendance_utc_offset_minutes": self.active.attendance_utc_offset_minutes,
                          "recording_max_duration_seconds": 86400,
                          "recording_min_free_bytes": self.active.recording_min_free_bytes},
            "changed_keys": changed, "restart_required": bool(changed),
        }

    def snapshot(self):
        with WRITE_LOCK:
            try:
                values, version = read_document(self.path, Settings)
            except OverrideError:
                raise HTTPException(503, "OPERATOR_SETTINGS_READ_FAILED") from None
            return self._snapshot(values, version, self._effective(values))

    def save(self, expected_version, values, reset_keys):
        # API and direct callers share validation; no token-bearing ValidationError escapes.
        if (not isinstance(expected_version, str) or len(expected_version) != 64
                or any(c not in "0123456789abcdef" for c in expected_version)
                or type(reset_keys) is not list or any(type(key) is not str for key in reset_keys)):
            raise HTTPException(422, "OPERATOR_SETTINGS_INVALID_REQUEST")
        if any(key not in EDITABLE for key in reset_keys):
            raise HTTPException(422, "OPERATOR_SETTINGS_UNKNOWN_FIELDS")
        try:
            changes = validate_values(values, Settings)
        except OverrideError as exc:
            raise HTTPException(422, str(exc)) from None
        if set(changes) & set(reset_keys):
            raise HTTPException(422, "OPERATOR_SETTINGS_SAVE_RESET_OVERLAP")
        with WRITE_LOCK:
            try:
                locked = environment_fields(Settings)
            except Exception:
                raise HTTPException(503, "OPERATOR_SETTINGS_SOURCES_UNAVAILABLE") from None
            if locked & (set(changes) | set(reset_keys)):
                raise HTTPException(409, "OPERATOR_SETTINGS_ENVIRONMENT_LOCKED")
            try:
                stored, version = read_document(self.path, Settings)
            except OverrideError as exc:
                status = 503 if "READ_FAILED" in str(exc) else 409
                raise HTTPException(status, "OPERATOR_SETTINGS_EXTERNAL_CHANGE") from None
            if version != expected_version:
                raise HTTPException(409, "OPERATOR_SETTINGS_CONFLICT")
            merged = {key: value for key, value in stored.items() if key not in reset_keys}
            merged.update(changes)
            pending = self._effective(merged)
            ids = pending.ai_camera_ids()
            if ids:
                try:
                    with Session(self.engine) as db:
                        existing = {row[0] for row in db.query(Camera.id).filter(Camera.id.in_(ids))}
                except Exception:
                    raise HTTPException(503, "OPERATOR_SETTINGS_CAMERAS_UNAVAILABLE") from None
                if ids != existing:
                    raise HTTPException(422, "OPERATOR_SETTINGS_AI_CAMERAS_NOT_FOUND")
            result = self._snapshot(merged, version, pending)
            try:
                result["version"] = replace_document(self.path, merged, version, Settings)
            except OverrideError as exc:
                code = str(exc)
                raise HTTPException(409 if code == "OPERATOR_SETTINGS_CONFLICT" else 503, code) from None
            logger.info("OPERATOR_SETTINGS_SAVED")
            return result
