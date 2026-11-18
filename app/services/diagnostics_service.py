"""Scalar diagnostic projections and a pre-sanitized, bounded support ring.

Never buffer LogRecords, formatted arbitrary messages, exception objects or
tracebacks. Unknown messages become generic event codes before retention.
"""
import logging
import math
import re
import threading
from collections import deque
from datetime import datetime, timezone
from urllib.parse import unquote, urlsplit
from uuid import UUID

from app.services.attendance_health import PROVIDER_KEYS

SAFE_CODES = frozenset({
    "OPERATOR_SETTINGS_SAVED", "OPERATOR_EVENT", "RUNTIME_WARNING", "UNEXPECTED_ERROR",
    "DIAGNOSTICS_UNAVAILABLE", "STREAM_UNAVAILABLE", "FFMPEG_UNAVAILABLE", "LIBX264_UNAVAILABLE",
    "MEDIA_PROCESS_CONTAINMENT_UNAVAILABLE", "MEDIA_ROOT_UNAVAILABLE", "MEDIA_STORAGE_UNAVAILABLE",
    "MEDIA_DATABASE_UNAVAILABLE", "MEDIA_LAUNCH_FAILED", "MEDIA_PROCESS_FAILED", "MEDIA_STOP_PENDING",
    "MEDIA_SHUTTING_DOWN", "MEDIA_STORAGE_CLOSED", "STORAGE_FULL", "SOURCE_CHANGED", "CAMERA_DISABLED",
    "CAMERA_NOT_FOUND", "CAMERA_OFFLINE", "CAMERA_UNCONFIGURED", "AI_UNAVAILABLE", "AI_EXCLUDED",
    "COMMISSIONING_REQUIRED", "CLOCK_DISCONTINUITY", "FRAME_GAP", "PERFORMANCE_DEGRADED",
    "PERFORMANCE_WARMUP", "PERSISTENCE_ERROR", "APPLICATION_SHUTDOWN", "APPLICATION_RESTART",
    "DEVICE_AUTH_FAILED", "DEVICE_UNREACHABLE", "DEVICE_TIMEOUT", "DEVICE_TLS_FAILED",
    "DEVICE_PROTOCOL_ERROR", "ARCHIVE_UNSUPPORTED", "ARCHIVE_NOT_CONFIGURED", "SEARCH_LIMIT",
    "ARCHIVE_SEARCH_OK", "ARCHIVE_DISCOVERY_OK", "ARCHIVE_PLAYBACK_READY",
    "PLAYBACK_UNAVAILABLE", "PLAYBACK_STORAGE_UNAVAILABLE", "PLAYBACK_BUSY", "PLAYBACK_CLOSED",
    "PLAYBACK_FAILED", "PLAYBACK_TIMEOUT", "PLAYBACK_PARTIAL", "PLAYBACK_EXPIRED", "PLAYBACK_CANCELLED",
    "NO_RECORDING_AT_TIME", "RECORDING_UNAVAILABLE", "RECORDING_STOPPED", "RECORDING_ALREADY_ACTIVE",
    "CONNECTING", "CONNECTION_LOST", "NO_PROGRESS", "ENCODER_FAILED", "PARTIAL_OUTPUT",
    "MODEL_MISSING", "MODEL_HASH_MISMATCH", "MODEL_CONTRACT_INVALID",
    "FACE_MODEL_UNAVAILABLE", "GALLERY_UNAVAILABLE", "GALLERY_CHANGED",
    "JOURNAL_DEGRADED", "CAPACITY_DEGRADED", "DECODE_SOFTWARE_FALLBACK",
    "JOURNAL_WRITE_FAILED", "JOURNAL_CAPACITY_EXCEEDED", "JOURNAL_GAP", "JOURNAL_UNFLUSHED",
    "SCENE_BUSY", "SCENE_INFERENCE_FAILED", "SCENE_NATIVE_WORK_UNRESOLVED",
    "DECODE_HARDWARE_INIT_FAILED",
    "SCENE_MODEL_UNAVAILABLE", "ENROLLMENT_BUSY", "LEGACY_MODEL_SETTINGS_IGNORED",
})
PROVIDERS = frozenset({"CPUExecutionProvider", "CUDAExecutionProvider", "TensorrtExecutionProvider",
                      "DmlExecutionProvider", "OpenVINOExecutionProvider", "CoreMLExecutionProvider"})
CAMERA_STATES = frozenset({"ONLINE", "OFFLINE", "CONNECTING", "RECONNECTING", "ERROR"})
SESSION_STATES = frozenset({"STARTING", "RECORDING", "RECONNECTING", "STOPPING", "STOPPED",
                            "COMPLETED", "FAILED", "INTERRUPTED"})
PLAYBACK_STATES = frozenset({"PREPARING", "READY", "ENDED", "FAILED", "CANCELLED", "EXPIRED"})
MODEL_COMPONENTS = ("person", "face_detection", "face_recognition", "body", "scene")
MODEL_STATUSES = frozenset({"READY", "MODEL_MISSING", "MODEL_HASH_MISMATCH", "MODEL_CONTRACT_INVALID"})
LEGACY_MODEL_SETTINGS = ("YOLO_MODEL", "INSIGHTFACE_MODEL", "INSIGHTFACE_ROOT", "REID_MODEL")
_URL = re.compile(r"\b(?:rtsp|rtsps|https?)://[^\s<>\"']+", re.IGNORECASE)
_HEADERS = re.compile(r"\b(?:authorization|proxy-authorization|cookie|set-cookie)[\"']?\s*[:=][^\r\n]*", re.IGNORECASE)
_CREDENTIAL = re.compile(r"\b(?:token|password|passwd|secret|telegram_bot_token)[\"']?\s*[:=][^\r\n]*", re.IGNORECASE)
_BOT_TOKEN = re.compile(r"\b\d{6,}:[A-Za-z0-9_-]{10,}\b")
_BEARER = re.compile(r"\bBearer\s+[^\s,\"']+", re.IGNORECASE)


def redact_text(value):
    """Server-log detail only; the support ring never retains this free text."""
    if not isinstance(value, str):
        return "UNEXPECTED_ERROR"
    value = _URL.sub("[URL REDACTED]", value)
    value = _HEADERS.sub("[HEADER REDACTED]", value)
    value = _CREDENTIAL.sub("[CREDENTIAL REDACTED]", value)
    value = _BEARER.sub("Bearer [REDACTED]", value)
    return _BOT_TOKEN.sub("[TOKEN REDACTED]", value)[:2000]


def safe_code(value, fallback="UNEXPECTED_ERROR"):
    return value if type(value) is str and value in SAFE_CODES else fallback


def number(value, *, integer=False):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        return None
    if integer and type(value) is not int:
        return None
    return value


def timestamp(value):
    try:
        if isinstance(value, str):
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if not isinstance(value, datetime):
            return None
        return value.replace(tzinfo=value.tzinfo or timezone.utc).isoformat()
    except (ValueError, OverflowError):
        return None


def identifier(value):
    try:
        return str(UUID(value)) if type(value) is str else None
    except ValueError:
        return None


def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def provider_summary(value):
    if type(value) is not dict:
        value = {}
    return {key: [item for item in value.get(key, ()) if type(item) is str and item in PROVIDERS]
            if isinstance(value.get(key, ()), (list, tuple)) else [] for key in PROVIDER_KEYS}



def model_identifier(value):
    return value if type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) else None

# Keep logging's standard metadata, not arbitrary request/header/body extras.
# In particular, color_message and cached message are independent rendering paths.
_LOG_RECORD_FIELDS = frozenset({
    "name", "msg", "args", "levelname", "levelno", "pathname", "filename", "module",
    "exc_info", "exc_text", "stack_info", "lineno", "funcName", "created", "msecs",
    "relativeCreated", "thread", "threadName", "processName", "process", "taskName",
})
_HTTP_METHODS = frozenset({"GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "TRACE", "CONNECT"})
_HTTP_VERSIONS = frozenset({"1.0", "1.1", "2", "2.0", "3", "3.0"})


class SafeLogFilter(logging.Filter):
    """Sanitize before formatting while retaining Uvicorn's structured contract."""

    @staticmethod
    def _access_args(args):
        # AccessFormatter consumes these five values even when msg is already
        # formatted. Retain only validated/sanitized scalars, never the raw tuple.
        if not isinstance(args, tuple) or len(args) != 5:
            return ("-", "UNKNOWN", "[REQUEST REDACTED]", "?", 0)
        client, method, target, version, status = args
        client = client if type(client) is str and re.fullmatch(r"[0-9a-fA-F.:\[\]]{1,80}", client) else "-"
        method = method if type(method) is str and method in _HTTP_METHODS else "UNKNOWN"
        version = version if type(version) is str and version in _HTTP_VERSIONS else "?"
        status = status if type(status) is int and 100 <= status <= 599 else 0
        try:
            # Query/fragment data is never useful enough to retain in access logs.
            # Decode the path before redaction so encoded tokens cannot bypass it.
            path = redact_text(unquote(urlsplit(target).path)) if type(target) is str else "[REQUEST REDACTED]"
        except ValueError:
            path = "[REQUEST REDACTED]"
        return client, method, path, version, status

    def filter(self, record):
        args = record.args.values() if isinstance(record.args, dict) else (
            record.args if isinstance(record.args, tuple) else ())
        exceptional = bool(record.exc_info) or any(isinstance(value, BaseException) for value in args)
        if record.name == "uvicorn.access":
            record.args = self._access_args(()) if exceptional else self._access_args(record.args)
            record.msg = '%s - "%s %s HTTP/%s" %d'
        else:
            if exceptional:
                text = safe_code(record.msg, "UNEXPECTED_ERROR")
            else:
                try:
                    text = redact_text(record.getMessage())
                except Exception:
                    text = "UNEXPECTED_ERROR"
            record.msg, record.args = text, ()
        record.exc_info, record.exc_text, record.stack_info = None, None, None
        for key in tuple(record.__dict__):
            if key not in _LOG_RECORD_FIELDS:
                del record.__dict__[key]
        return True


class DiagnosticRing(logging.Handler):
    def __init__(self):
        super().__init__(logging.INFO)
        self._records = deque(maxlen=200)
        self._sequence = 0
        self._guard = threading.Lock()

    def emit(self, record):
        # No getMessage(): even __str__ on arbitrary args is outside this boundary.
        source = ("camera" if record.name.startswith("app.rtsp") else
                  "media" if record.name.startswith(("app.media", "app.recordings", "app.playback", "app.dahua"))
                  else "operator")
        fallback = "UNEXPECTED_ERROR" if record.levelno >= logging.ERROR else (
            "RUNTIME_WARNING" if record.levelno >= logging.WARNING else "OPERATOR_EVENT")
        code = safe_code(record.msg, fallback)
        self.add(source, code, camera_id=getattr(record, "camera_id", None),
                 level="ERROR" if record.levelno >= logging.ERROR else "WARNING" if record.levelno >= logging.WARNING else "INFO")

    def add(self, source, code, *, camera_id=None, level="INFO"):
        entry = {"at": now_iso(), "source": source if source in {"camera", "media", "operator"} else "operator",
                 "code": safe_code(code), "level": level if level in {"INFO", "WARNING", "ERROR"} else "INFO"}
        if type(camera_id) is int and camera_id > 0:
            entry["camera_id"] = camera_id
        with self._guard:
            self._sequence += 1
            entry["sequence"] = self._sequence
            self._records.append(entry)

    def events(self, *, after=0, limit=100):
        with self._guard:
            return {"events": [dict(item) for item in self._records if item["sequence"] > after][:min(100, max(0, limit))],
                    "latest_sequence": self._sequence,
                    "oldest_sequence": self._records[0]["sequence"] if self._records else None}

    def export(self):
        with self._guard:
            return {"events": [dict(item) for item in self._records],
                    "latest_sequence": self._sequence,
                    "oldest_sequence": self._records[0]["sequence"] if self._records else None}


class DiagnosticsService:
    def __init__(self, state, engine, active):
        self.state, self.engine, self.active = state, engine, active
        self.ring = DiagnosticRing()
        self._archive_lock = threading.Lock()
        self._archive = {"updated_at": None, "camera_id": None, "channel_titles": None,
                         "search": None, "error_code": None}

    def archive_result(self, *, camera_id=None, channel_titles=None, search=None, error_code=None):
        with self._archive_lock:
            if camera_id is None or camera_id != self._archive["camera_id"]:
                self._archive.update(channel_titles=None, search=None)
            self._archive.update(updated_at=now_iso(), camera_id=number(camera_id, integer=True),
                                 error_code=safe_code(error_code) if error_code else None)
            if type(channel_titles) is bool:
                self._archive["channel_titles"] = channel_titles
            if type(search) is bool:
                self._archive["search"] = search
        self.ring.add("media", safe_code(error_code) if error_code else (
            "ARCHIVE_SEARCH_OK" if search else "ARCHIVE_DISCOVERY_OK"), camera_id=camera_id,
            level="WARNING" if error_code else "INFO")

    @staticmethod
    def _isolated(fn, fallback):
        try:
            return fn()
        except Exception:
            return {**fallback, "error_code": "DIAGNOSTICS_UNAVAILABLE"}

    def _attendance(self):
        health = self.state.attendance.health.snapshot()
        reasons = [safe_code(item) for item in health.get("reasons", [])]
        cameras = []
        for row in health.get("cameras", []):
            if not isinstance(row, dict) or row.get("role") not in ("ENTRY", "EXIT", "BOTH"):
                continue
            item = {key: number(row.get(key)) for key in (
                "camera_id", "source_revision", "processed_fps", "frame_age_seconds",
                "processing_age_p95_seconds", "completion_age_p95_seconds")}
            item["role"] = row["role"]
            item["reasons"] = [safe_code(reason) for reason in row.get("reasons", [])]
            cameras.append(item)
        return {"technical_ready": health.get("technical_ready") is True,
                "automatic_ready": health.get("automatic_ready") is True,
                "reasons": reasons, "cameras": cameras,
                "pending_fault_count": number(health.get("pending_fault_count"), integer=True),
                "pending_write_count": number(health.get("pending_write_count"), integer=True),
                "providers": provider_summary(health.get("providers")), "error_code": None}

    def _models(self):
        """Project startup state only: no paths, provenance URLs or biometrics."""
        bundle = getattr(self.state, "model_bundle", None)
        from sqlalchemy.orm import Session
        from app.database.models import AttendanceRuntime
        with Session(self.engine) as db:
            runtime = db.get(AttendanceRuntime, 1)
            revision = runtime.gallery_revision if runtime is not None else None
        store = getattr(self.state, "embedding_store", None)
        loaded_revision = getattr(store, "revision", None)
        statuses = getattr(bundle, "component_status", {}) if bundle is not None else {}
        statuses = statuses if type(statuses) is dict else {}
        components = {}
        for component in MODEL_COMPONENTS:
            status = statuses.get(component)
            components[component] = (status if type(status) is str and status in MODEL_STATUSES
                                     else "MODEL_MISSING" if bundle is None else "MODEL_CONTRACT_INVALID")
        raw_migration = getattr(self.state, "gallery_migration", None)
        migration = None
        if type(raw_migration) is dict:
            migration = {
                key: number(raw_migration.get(key), integer=True) for key in (
                    "stored_faces_count", "usable_faces_count", "faces_reencoded", "faces_need_reenrollment",
                )
            }
            remediation = raw_migration.get("employees_needing_photos")
            migration.update(
                face_model_id=model_identifier(raw_migration.get("face_model_id")),
                alignment_id=("arcface-5pt-112-v1"
                              if raw_migration.get("alignment_id") == "arcface-5pt-112-v1" else None),
                employees_needing_photos_count=len(remediation) if type(remediation) in (list, tuple) else None,
                error_code=(safe_code(raw_migration["error_code"], "GALLERY_UNAVAILABLE")
                            if raw_migration.get("error_code") else None),
            )
        ignored = getattr(self.state, "legacy_model_settings_ignored", ())
        ignored = ignored if type(ignored) in (tuple, list) else ()
        ignored = [name for name in LEGACY_MODEL_SETTINGS if name in ignored]
        return {
            "bundle_id": model_identifier(getattr(bundle, "bundle_id", None)),
            "face_model_id": model_identifier(getattr(bundle, "face_model_id", None)),
            "alignment_id": ("arcface-5pt-112-v1"
                             if getattr(bundle, "alignment_id", None) == "arcface-5pt-112-v1" else None),
            "component_status": components,
            "gallery_revision": number(revision, integer=True),
            "loaded_gallery_revision": number(loaded_revision, integer=True),
            "gallery_current": (revision is not None and revision == loaded_revision
                                and model_identifier(getattr(store, "model_id", None)) is not None
                                and getattr(store, "model_id", None) == getattr(bundle, "face_model_id", None)),
            "error_code": next((components[key] for key in MODEL_COMPONENTS[:3]
                                if components[key] != "READY"), None),
            "gallery_migration": migration,
            "legacy_model_settings_ignored": ignored,
            "legacy_model_settings_code": "LEGACY_MODEL_SETTINGS_IGNORED" if ignored else None,
        }

    def _journal(self):
        journal = getattr(self.state, "human_journal", None)
        if journal is None:
            return {"state": "UNAVAILABLE", "error_code": "JOURNAL_DEGRADED"}
        raw = journal.health_snapshot()
        result = {key: number(raw.get(key), integer=True) for key in (
            "pending_commands", "pending_updates", "pending_barriers", "blocked_generations",
            "reserved_generations", "active_visits", "pending_acknowledgements")}
        result.update(state=raw.get("state") if raw.get("state") in ("OK", "JOURNAL_DEGRADED") else "UNAVAILABLE",
                      error_code=safe_code(raw["error_code"], "JOURNAL_DEGRADED") if raw.get("error_code") else None,
                      accepting=raw.get("accepting") is True, writer_alive=raw.get("writer_alive") is True)
        return result

    def _scene(self):
        model = getattr(self.state, "scene_segmenter", None)
        service = getattr(self.state, "scene_assistance", None)
        raw = service.health_snapshot() if service is not None else {}
        return dict(model_available=model is not None,
                    error_code=safe_code(getattr(self.state, "scene_error", None), "SCENE_MODEL_UNAVAILABLE")
                               if getattr(self.state, "scene_error", None) else None,
                    last_inference_seconds=number(getattr(model, "last_inference_seconds", None)),
                    running=number(raw.get("running"), integer=True), queued=number(raw.get("queued"), integer=True),
                    records=number(raw.get("records"), integer=True), accepting=raw.get("accepting") is True,
                    worker_alive=raw.get("worker_alive") is True)

    def _cameras(self):
        from sqlalchemy.orm import Session
        from app.database.models import Camera
        with Session(self.engine) as db:
            cameras = db.query(Camera.id, Camera.enabled).order_by(Camera.id).all()
        manager = getattr(self.state, "manager", None)
        worker = getattr(self.state, "ai_worker", None)
        inference = self._isolated(worker.inference_snapshot, {}) if worker is not None else {}
        result = []
        for camera_id, enabled in cameras:
            try:
                raw = manager.status(camera_id)
                row = {"id": camera_id, "enabled": bool(enabled),
                       "status": raw.get("status") if raw.get("status") in CAMERA_STATES else "ERROR",
                       "error": safe_code(raw["error"], "STREAM_UNAVAILABLE") if raw.get("error") else None}
                for key in ("current_fps", "seconds_since_last_frame", "stream_fps", "reconnect_count"):
                    row[key] = number(raw.get(key))
                row["last_frame_at"] = timestamp(raw.get("last_frame_at"))
                resolution = raw.get("resolution")
                row["resolution"] = resolution if type(resolution) is str and re.fullmatch(r"[0-9]{1,5}x[0-9]{1,5}", resolution) else None
                row["codec"] = raw.get("codec") if raw.get("codec") in {"h264", "hevc", "mjpeg", "mpeg4", "vp8", "vp9", "av1"} else None
                row["decoder_mode"] = raw.get("decoder_mode") if raw.get("decoder_mode") in {"cuda", "software"} else None
                row["decoder_requested"] = raw.get("decoder_requested") if raw.get("decoder_requested") in {"auto", "cuda", "none"} else None
                row["decoder_warning"] = safe_code(raw["decoder_warning"]) if raw.get("decoder_warning") else None
                measurement = inference.get(camera_id)
                if measurement is not None:
                    row["inference"] = {
                        key: number(measurement.get(key)) for key in (
                            "processed_fps_10s", "target_fps", "completion_age_p95_seconds", "pending_age_seconds",
                            "queue_age_seconds", "processing_seconds", "completion_age_seconds",
                            "skipped_slots", "skipped_frames", "face_deferred", "reid_deferred")}
                    row["inference"].update(
                        reason_codes=[safe_code(code) for code in measurement.get("reason_codes", [])],
                        stage_times_ms={key: number(measurement.get("stage_times_ms", {}).get(key))
                                        for key in ("person", "face_detection", "face_recognition", "tracking", "appearance", "journal")},
                        model_ids={key: model_identifier(measurement.get("model_ids", {}).get(key))
                                   for key in ("bundle", "person", "face", "body")},
                        providers=provider_summary(measurement.get("providers")))
            except Exception:
                row = {"id": camera_id, "enabled": bool(enabled), "status": "ERROR", "error": "DIAGNOSTICS_UNAVAILABLE"}
            result.append(row)
        return result

    def _recordings(self):
        manager = getattr(self.state, "recordings", None)
        if manager is None:
            return {"sessions": [], "storage": None,
                    "error_code": safe_code(getattr(self.state, "recording_error", None), "MEDIA_STORAGE_UNAVAILABLE")}
        raw = manager.snapshot()
        storage = {key: number(raw["storage"].get(key)) for key in (
            "bytes_used", "max_bytes", "free_bytes", "min_free_bytes", "reserve_bytes", "retention_days")}
        storage["oldest_available"] = timestamp(raw["storage"].get("oldest_available"))
        storage["quota_shortened_retention"] = raw["storage"].get("quota_shortened_retention") is True
        storage["full"] = (storage["free_bytes"] is not None and storage["min_free_bytes"] is not None
                           and storage["free_bytes"] < storage["min_free_bytes"]) or (
            all(storage[key] is not None for key in ("bytes_used", "reserve_bytes", "max_bytes"))
            and storage["bytes_used"] + storage["reserve_bytes"] > storage["max_bytes"])
        sessions = []
        for item in raw.get("sessions", []):
            row = {"id": identifier(item.get("id")), "camera_id": number(item.get("camera_id"), integer=True),
                   "status": item.get("status") if item.get("status") in SESSION_STATES else "FAILED",
                   "reason_code": safe_code(item["reason_code"]) if item.get("reason_code") else None,
                   "bytes_recorded": number(item.get("bytes_recorded"))}
            for key in ("requested_at", "started_at", "deadline_at", "ended_at", "available_from", "available_to"):
                row[key] = timestamp(item.get(key))
            sessions.append(row)
        return {"sessions": sessions, "storage": storage, "error_code": "STORAGE_FULL" if storage["full"] else None}

    def _processes(self):
        from app.media.process_guard import MediaProcessGuard
        result = {"containment_available": MediaProcessGuard._handle is not None,
                  "recorders": [], "playbacks": []}
        recordings = getattr(self.state, "recordings", None)
        if recordings is not None:
            with recordings.lock:
                for job in recordings.jobs.values():
                    process = job.recorder.process if job.recorder else None
                    if process is not None:
                        result["recorders"].append({"pid": number(process.pid, integer=True),
                                                    "running": process.poll() is None})
        playbacks = getattr(self.state, "playbacks", None)
        if playbacks is not None:
            with playbacks.lock:
                for job in playbacks.jobs.values():
                    if job.state == "EXPIRED":
                        continue
                    result["playbacks"].append({"id": identifier(job.id), "camera_id": number(job.camera_id, integer=True),
                        "state": job.state if job.state in PLAYBACK_STATES else "FAILED",
                        "pid": number(job.process.pid, integer=True) if job.process else None,
                        "running": job.process is not None and job.process.poll() is None,
                        "error_code": safe_code(job.error) if job.error else None})
        return result

    def status(self):
        attendance = self._isolated(self._attendance, {"automatic_ready": False, "technical_ready": False,
            "reasons": ["DIAGNOSTICS_UNAVAILABLE"], "providers": provider_summary({}), "cameras": []})
        def tools():
            manager = self.state.manager
            return {"ffmpeg_available": bool(manager.ffmpeg), "ffprobe_available": bool(manager.ffprobe)}
        try:
            cameras = self._cameras()
            camera_error = None
        except Exception:
            cameras, camera_error = [], "DIAGNOSTICS_UNAVAILABLE"
        with self._archive_lock:
            archive = dict(self._archive)
        return {"updated_at": now_iso(),
                "tools": self._isolated(tools, {"ffmpeg_available": False, "ffprobe_available": False}),
                "ai": {"enabled": self.active.ai_enabled, "providers": attendance["providers"],
                       "error_code": attendance.get("error_code")},
                "models": self._isolated(self._models, {"bundle_id": None, "face_model_id": None,
                    "alignment_id": None, "component_status": {}, "gallery_migration": None,
                    "legacy_model_settings_ignored": [], "legacy_model_settings_code": None}),
                "attendance": attendance, "cameras": cameras, "camera_error": camera_error,
                "journal": self._isolated(self._journal, {"state": "UNAVAILABLE", "error_code": "DIAGNOSTICS_UNAVAILABLE"}),
                "scene": self._isolated(self._scene, {"model_available": False, "error_code": "DIAGNOSTICS_UNAVAILABLE"}),
                "recordings": self._isolated(self._recordings, {"sessions": [], "storage": None}),
                "processes": self._isolated(self._processes, {"recorders": [], "playbacks": []}), "archive": archive}

    def export(self):
        return {"status": self.status(), "events": self.ring.export(),
                "settings": self._isolated(self.state.settings_service.snapshot, {})}
