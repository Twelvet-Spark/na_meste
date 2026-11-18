"""Мониторинг камер Dahua (RTSP) и присутствия сотрудников (AI).

Запуск:
    python main.py

Открыть:
    http://localhost:8000
"""
import logging
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import UUID
from dataclasses import asdict

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.ai.face_recognition import EmbeddingStore
from app.ai.recognition_learning import RecognitionLearningStore
from app.ai.model_bundle import ModelBundle
from app.ai.providers import provider_label, resolve_providers
from app.ai.worker import (
    build_ai_worker, build_body_enroll_engine, build_enroll_engine, loaded_providers)
from app.ai.scene_detector import SceneObjectDetector
from app.ai.scene_segmenter import SceneSegmenter
from app.api.attendance import router as attendance_router
from app.api.cameras import router as cameras_router
from app.api.employees import router as employees_router
from app.api.workplaces import router as workplaces_router
from app.api.global_persons import router as global_persons_router
from app.api.recognition_learning import router as recognition_learning_router
from app.api.recordings import router as recordings_router
from app.api.settings import router as settings_router
from app.api.debug import router as debug_router
from app.api.setup_transfer import router as setup_transfer_router
from app.api.scenes import router as scenes_router
from app.config import BASE_DIR, settings
from app.database.database import SessionLocal, engine, init_db
from app.database.models import AttendanceCamera, Camera
from app.rtsp.manager import ReaderManager
from app.rtsp.source import CameraSource
from app.services.presence_recorder import PresenceFlusher, PresenceRecorder
from app.time_utils import utcnow
from app.services.telegram_service import TelegramNotifier
from app.media.process_guard import MediaProcessGuard
from app.services.recording_service import RecordingManager
from app.media.playback import PlaybackManager
from app.services.settings_service import SettingsService
from app.services.diagnostics_service import DiagnosticsService, SafeLogFilter
from app.services.setup_transfer_service import SetupTransferService
from app.services.gallery_migration import rebuild_gallery
from app.services.human_journal import HumanJournal
from app.services.visit_sample_service import VisitSampleService
from app.services.recognition_learning_service import RecognitionLearningService
from app.services.scene_service import SceneService
from app.services.scene_assistance import SceneAssistanceService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
)
logger = logging.getLogger("app")

# Sanitize before server handlers format messages/tracebacks. The support ring
# separately projects codes and never retains a LogRecord or its exception context.
_safe_log_filter = SafeLogFilter()
for _log in (logging.getLogger(), logging.getLogger("uvicorn"),
             logging.getLogger("uvicorn.error"), logging.getLogger("uvicorn.access")):
    for _handler in _log.handlers:
        _handler.addFilter(_safe_log_filter)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Self-containing job must precede even incumbent live-reader/probe children.
    # Its sole handle intentionally survives lifespan teardown until process death.
    MediaProcessGuard.initialize()
    # Initialize persisted revisions before publishing any camera frames.
    init_db()
    app.state.model_bundle = None
    app.state.model_error = None
    manifest_path = Path(settings.ai_model_manifest)
    if not manifest_path.is_absolute():
        manifest_path = BASE_DIR / manifest_path
    try:
        app.state.model_bundle = ModelBundle.load(manifest_path)
    except Exception as exc:
        app.state.model_error = getattr(exc, "code", "MODEL_CONTRACT_INVALID")
        logger.error("Model bundle unavailable: %s", app.state.model_error)
    bundle = app.state.model_bundle
    app.state.face_model_id = bundle.face_model_id if bundle is not None else None
    app.state.gallery_migration = None
    # Policy names, never deployment values: these selectors cannot choose weights.
    app.state.legacy_model_settings_ignored = (
        "YOLO_MODEL", "INSIGHTFACE_MODEL", "INSIGHTFACE_ROOT", "REID_MODEL")
    app.state.settings_service = SettingsService(settings, engine)
    app.state.setup_transfer = None
    app.state.diagnostics = DiagnosticsService(app.state, engine, settings)
    app.state.manager = ReaderManager()
    app.state.scene_service = SceneService(engine, app.state.manager)
    app.state.scene_segmenter = None
    app.state.scene_error = None
    if bundle is not None:
        try:
            app.state.scene_segmenter = SceneSegmenter(bundle)
        except Exception as exc:
            app.state.scene_error = getattr(exc, "code", "SCENE_MODEL_UNAVAILABLE")
            logger.error("Scene assistance unavailable: %s", app.state.scene_error)
    else:
        app.state.scene_error = "SCENE_MODEL_UNAVAILABLE"
    app.state.scene_detector = None
    app.state.scene_detector_error = None
    if bundle is not None and bundle.component_status.get("scene_detector") == "READY":
        try:
            app.state.scene_detector = SceneObjectDetector(bundle, resolve_providers(settings.ai_device))
        except Exception as exc:
            app.state.scene_detector_error = getattr(exc, "code", "SCENE_DETECTOR_UNAVAILABLE")
            logger.error("Scene detector unavailable: %s", app.state.scene_detector_error)
    app.state.scene_assistance = SceneAssistanceService(
        app.state.scene_service, app.state.scene_segmenter, app.state.scene_detector)
    app.state.recordings = None
    app.state.recording_error = None
    try:
        app.state.recordings = RecordingManager(
            engine, settings.recording_root, app.state.manager.ffmpeg, app.state.manager.ffprobe)
    except HTTPException as exc:
        app.state.recording_error = exc.detail
        logger.warning("Manual recording unavailable: %s", exc.detail)
    except Exception:
        app.state.recording_error = "MEDIA_STORAGE_UNAVAILABLE"
        logger.warning("Manual recording unavailable: MEDIA_STORAGE_UNAVAILABLE")
    app.state.playbacks = None
    app.state.playback_error = None
    if app.state.recordings is not None:
        try:
            app.state.playbacks = PlaybackManager(
                engine, app.state.recordings.storage, app.state.manager.ffmpeg, app.state.manager.ffprobe)
        except Exception:
            app.state.playback_error = "PLAYBACK_STORAGE_UNAVAILABLE"
            logger.warning("Archive playback unavailable: PLAYBACK_STORAGE_UNAVAILABLE")

    # --- Telegram-уведомления ---
    app.state.telegram = None
    notifier = None
    if settings.telegram_enabled and settings.telegram_bot_token:
        notifier = TelegramNotifier(settings.telegram_bot_token, settings.telegram_chat_id)
        notifier.start()
        app.state.telegram = notifier
    else:
        logger.info("Telegram: уведомления отключены (TELEGRAM_BOT_TOKEN не задан)")

    # One separate enrollment owner and one live owner; no per-request engine.
    app.state.face_engine = build_enroll_engine(bundle)
    app.state.body_engine = build_body_enroll_engine(bundle)
    app.state.recognition_learning_service = None
    app.state.learning_store = None
    if app.state.body_engine is not None:
        try:
            learning_store = RecognitionLearningStore(engine, app.state.body_engine.model_id)
            app.state.learning_store = learning_store
            try:
                learning_store.refresh()
            except Exception as exc:
                # The worker retries on its own cadence; learning stays unavailable until then.
                logger.error("Learning examples unavailable: %s", getattr(exc, "code", type(exc).__name__))
        except Exception as exc:
            logger.error("Learning store unavailable: %s", getattr(exc, "code", type(exc).__name__))
    app.state.enrollment_lock = threading.BoundedSemaphore(1)
    app.state.embedding_store = None
    app.state.ai_worker = None
    app.state.detection_state = None
    app.state.presence = PresenceRecorder(engine, notifier=notifier)
    app.state.human_journal = HumanJournal(
        engine, lost_timeout=settings.track_lost_timeout,
        image_keep_days=settings.journal_image_keep_days)
    if app.state.face_engine is not None:
        try:
            app.state.gallery_migration = rebuild_gallery(engine, app.state.face_engine)
            store = EmbeddingStore(bundle.face_model_id)
            with SessionLocal() as session:
                store.refresh(session)
            app.state.embedding_store = store
        except Exception as exc:
            code = getattr(exc, "code", "GALLERY_UNAVAILABLE")
            app.state.gallery_migration = {"error_code": code}
            logger.error("Gallery unavailable; anonymous tracking remains available: %s", code)
        logger.info("Enrollment provider: %s",
                    provider_label(app.state.face_engine.providers()["recognition"]))
    else:
        app.state.gallery_migration = {"error_code": "FACE_MODEL_UNAVAILABLE"}
    app.state.visit_sample_service = VisitSampleService(
        engine, app.state.face_engine, app.state.face_model_id,
        app.state.enrollment_lock, store=app.state.embedding_store)
    if bundle is not None:
        app.state.setup_transfer = SetupTransferService(
            engine, face_model_id=bundle.face_model_id,
            face_engine=app.state.face_engine, enrollment_semaphore=app.state.enrollment_lock)

    app.state.ai_startup_error = None
    if settings.ai_enabled:
        try:
            worker = build_ai_worker(
                app.state.manager, app.state.presence, app.state.embedding_store,
                bundle=bundle, session_factory=SessionLocal, notifier=notifier,
                journal=app.state.human_journal, learning_store=app.state.learning_store)
            if worker is not None and worker.learning_store is None and worker.reid is not None:
                # Enrollment body model is unavailable but the live one loaded: durable
                # matching still needs a store bound to the live model. Assign before start.
                try:
                    app.state.learning_store = worker.learning_store = RecognitionLearningStore(
                        engine, worker.reid.model_id)
                    worker.learning_store.refresh()
                except Exception as exc:
                    logger.error("Learning examples unavailable: %s", getattr(exc, "code", type(exc).__name__))
            if worker is not None:
                worker.start()
                app.state.ai_worker = worker
                app.state.detection_state = worker.state
            else:
                app.state.ai_startup_error = "AI_UNAVAILABLE"
        except Exception:
            logger.exception("Live inference unavailable; work time is not recorded")
            app.state.ai_startup_error = "AI_UNAVAILABLE"
    else:
        logger.info("AI отключён (AI_ENABLED=false)")
    worker = app.state.ai_worker
    try:
        app.state.recognition_learning_service = RecognitionLearningService(
            engine, app.state.body_engine, app.state.face_engine, app.state.face_model_id,
            app.state.enrollment_lock, store=app.state.embedding_store,
            learning_store=app.state.learning_store,
            appearance_gallery=worker.appearance_gallery if worker is not None else None)
    except Exception as exc:
        logger.error("Learning service unavailable: %s", getattr(exc, "code", type(exc).__name__))
    app.state.ai_providers = loaded_providers(
        detector=worker.service.detector if worker is not None else None,
        live=worker.service.faces if worker is not None else None,
        enroll=app.state.face_engine, reid=worker.reid if worker is not None else None,
        scene=app.state.scene_segmenter, scene_detector=app.state.scene_detector)

    with SessionLocal() as session:
        saved_sources = [
            CameraSource.from_camera(camera, source_revision=source_revision)
            for camera, source_revision in session.query(Camera, AttendanceCamera.source_revision).join(
                AttendanceCamera, AttendanceCamera.camera_id == Camera.id,
            ).filter(Camera.enabled.is_(True)).all()
        ]
    for camera in saved_sources:
        app.state.manager.start(camera)
    app.state.presence_flusher = PresenceFlusher(app.state.presence)
    app.state.presence_flusher.start()

    logger.info(
        "Приложение запущено: http://localhost:%d (API docs: /docs)", settings.app_port
    )
    diagnostic_logger = logging.getLogger("app")
    diagnostic_logger.addHandler(app.state.diagnostics.ring)
    try:
        yield
    finally:
        # Independent media stops first; preserve the existing evidence-drain order below.
        if app.state.playbacks is not None:
            try:
                app.state.playbacks.shutdown()
            except Exception:
                logger.warning("Archive playback shutdown incomplete: MEDIA_STOP_PENDING")
        if app.state.recordings is not None:
            try:
                app.state.recordings.shutdown()
            except Exception:
                logger.warning("Manual recording shutdown incomplete: MEDIA_STOP_PENDING")
        # Join the sole producer first, then drain/persist its evidence and durable
        # stop timestamp before stopping readers or notification delivery.
        if app.state.ai_worker is not None:
            app.state.ai_worker.stop()
        if app.state.recognition_learning_service is not None:
            try:
                app.state.recognition_learning_service.close()
            except Exception:
                logger.error("Learning native work unresolved at shutdown")
        app.state.scene_close_result = asdict(app.state.scene_assistance.close())
        if not app.state.scene_close_result["closed"]:
            logger.error("Scene native work unresolved at shutdown: %s", app.state.scene_close_result)
        app.state.journal_close_result = asdict(app.state.human_journal.close())
        if not app.state.journal_close_result["drained"]:
            logger.error("Journal shutdown incomplete: %s", app.state.journal_close_result)
        app.state.presence_flusher.stop()
        if app.state.telegram is not None:
            app.state.telegram.stop()
        app.state.manager.stop_all()
        logger.info("Приложение остановлено")
        diagnostic_logger.removeHandler(app.state.diagnostics.ring)
        app.state.diagnostics.ring.close()


app = FastAPI(title="Мониторинг", lifespan=lifespan)

app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "app" / "templates")

app.include_router(cameras_router)
app.include_router(scenes_router)
app.include_router(employees_router)
app.include_router(workplaces_router)
app.include_router(recognition_learning_router)
app.include_router(global_persons_router)
app.include_router(attendance_router)
app.include_router(recordings_router)
app.include_router(settings_router)
app.include_router(debug_router)
app.include_router(setup_transfer_router)


@app.get("/", include_in_schema=False)
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html", {
        "workplace_utc_offset_minutes": settings.attendance_utc_offset_minutes,
    })


@app.get("/cameras/{camera_id}", include_in_schema=False)
async def camera_page(request: Request, camera_id: int):
    return templates.TemplateResponse(
        request, "camera.html", {"camera_id": camera_id,
                                  "recordings_at": utcnow().isoformat(timespec="milliseconds") + "+00:00",
                                  "workplace_utc_offset_minutes": settings.attendance_utc_offset_minutes}
    )


@app.get("/employees", include_in_schema=False)
async def employees_page(request: Request):
    return templates.TemplateResponse(request, "employees.html", {})


@app.get("/employees/new", include_in_schema=False)
async def employee_new_page(request: Request):
    return templates.TemplateResponse(request, "employee_new.html", {})


@app.get("/employees/{employee_id}", include_in_schema=False)
async def employee_page(request: Request, employee_id: int):
    return templates.TemplateResponse(
        request, "employee_detail.html", {"employee_id": employee_id}
    )


@app.get("/recordings", include_in_schema=False)
async def recordings_page(request: Request):
    return templates.TemplateResponse(request, "recordings.html", {})


@app.get("/settings", include_in_schema=False)
async def settings_page(request: Request):
    return templates.TemplateResponse(request, "settings.html", {},
                                      headers={"Cache-Control": "no-store"})


@app.get("/debug", include_in_schema=False)
async def debug_page(request: Request):
    return templates.TemplateResponse(request, "debug.html", {},
                                      headers={"Cache-Control": "no-store"})


@app.get("/presence", include_in_schema=False)
async def presence_page(request: Request):
    return templates.TemplateResponse(request, "presence.html", {})


@app.get("/global", include_in_schema=False)
async def global_persons_page(request: Request):
    return templates.TemplateResponse(request, "global_persons.html", {})


@app.get("/global/visits/{visit_id}", include_in_schema=False)
async def human_visit_page(request: Request, visit_id: UUID):
    return templates.TemplateResponse(request, "human_visit_detail.html", {
        "visit_id": str(visit_id),
        "workplace_utc_offset_minutes": settings.attendance_utc_offset_minutes,
    }, headers={"Cache-Control": "no-store"})


@app.get("/global/{global_id}", include_in_schema=False)
async def global_person_page(request: Request, global_id: int):
    return templates.TemplateResponse(
        request, "global_person_detail.html", {"global_id": global_id}
    )


if __name__ == "__main__":
    # This module already owns and sanitizes the handlers. Uvicorn's default
    # dictConfig would replace them after import and bypass the filtered root.
    # CLI startup configures Uvicorn before importing this module instead.
    uvicorn.run(app, host=settings.app_host, port=settings.app_port, log_config=None)
