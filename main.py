"""Мониторинг камер Dahua (RTSP) и присутствия сотрудников (AI).

Запуск:
    python main.py

Открыть:
    http://localhost:8000
"""
import logging
import threading
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.ai.face_recognition import EmbeddingStore
from app.ai.providers import provider_label
from app.ai.worker import build_ai_worker, build_enroll_engine, loaded_providers
from app.api.attendance import router as attendance_router
from app.api.cameras import router as cameras_router
from app.api.employees import router as employees_router
from app.api.workplaces import router as workplaces_router
from app.api.global_persons import router as global_persons_router
from app.api.presence import router as presence_router
from app.api.recordings import router as recordings_router
from app.config import BASE_DIR, settings
from app.database.database import SessionLocal, engine, init_db
from app.database.models import AttendanceCamera, Camera
from app.rtsp.manager import ReaderManager
from app.rtsp.source import CameraSource
from app.services.attendance_service import AttendanceService
from app.services.attendance_monitor import AttendanceMonitor
from app.time_utils import utcnow
from app.services.telegram_service import TelegramNotifier
from app.media.process_guard import MediaProcessGuard
from app.services.recording_service import RecordingManager
from app.media.playback import PlaybackManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
)
logger = logging.getLogger("app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Self-containing job must precede even incumbent live-reader/probe children.
    # Its sole handle intentionally survives lifespan teardown until process death.
    MediaProcessGuard.initialize()
    # Initialize persisted revisions before publishing any camera frames.
    init_db()
    app.state.manager = ReaderManager()
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

    # --- AI: сотрудники/лица/присутствие ---
    app.state.face_engine = None        # FaceEngine для загрузки фото (det 640)
    app.state.enrollment_lock = threading.Lock()
    app.state.embedding_store = None
    app.state.ai_worker = None
    app.state.detection_state = None
    app.state.attendance = AttendanceService(engine, notifier=notifier, now=utcnow)

    try:
        store = EmbeddingStore()
        with SessionLocal() as session:
            store.refresh(session)
        app.state.embedding_store = store
        app.state.face_engine = build_enroll_engine()
        if app.state.face_engine is not None:
            logger.info("AI Provider: %s",
                        provider_label(list(app.state.face_engine.app.models.values())[0].session.get_providers()))
    except Exception:
        logger.exception("AI-инициализация для сотрудников не удалась — "
                         "загрузка фото будет недоступна")
    app.state.attendance.health.set_providers(loaded_providers(enroll=app.state.face_engine))

    if settings.ai_enabled:
        try:
            worker = build_ai_worker(
                app.state.manager, app.state.attendance, app.state.embedding_store,
                session_factory=SessionLocal, notifier=notifier,
                enroll_engine=app.state.face_engine,
            )
            if worker is not None:
                worker.start()
                app.state.ai_worker = worker
                app.state.detection_state = worker.state
        except Exception:
            logger.exception("Live inference unavailable; attendance remains in review/manual mode")
            app.state.attendance.health.transition("inference_startup", ["AI_UNAVAILABLE"])
    else:
        logger.info("AI отключён (AI_ENABLED=false)")

    with SessionLocal() as session:
        saved_sources = [
            CameraSource.from_camera(camera, source_revision=source_revision)
            for camera, source_revision in session.query(Camera, AttendanceCamera.source_revision).join(
                AttendanceCamera, AttendanceCamera.camera_id == Camera.id,
            ).filter(Camera.enabled.is_(True)).all()
        ]
    for camera in saved_sources:
        app.state.manager.start(camera)
    app.state.attendance_monitor = AttendanceMonitor(
        app.state.attendance,
        write_queue=app.state.ai_worker.write_queue if app.state.ai_worker is not None else None,
        manager=app.state.manager, worker=app.state.ai_worker,
    )
    try:
        app.state.attendance_monitor.startup()
    except Exception:
        logger.exception("Attendance restart coverage could not be persisted")
        app.state.attendance.health.transition("monitor", ["PERSISTENCE_ERROR"])
    app.state.attendance_monitor.start()

    logger.info(
        "Приложение запущено: http://localhost:%d (API docs: /docs)", settings.app_port
    )
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
        app.state.attendance_monitor.stop()
        if app.state.telegram is not None:
            app.state.telegram.stop()
        app.state.manager.stop_all()
        logger.info("Приложение остановлено")


app = FastAPI(title="Мониторинг", lifespan=lifespan)

app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "app" / "templates")

app.include_router(cameras_router)
app.include_router(employees_router)
app.include_router(workplaces_router)
app.include_router(presence_router)
app.include_router(global_persons_router)
app.include_router(attendance_router)
app.include_router(recordings_router)


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


@app.get("/presence", include_in_schema=False)
async def presence_page(request: Request):
    return templates.TemplateResponse(request, "presence.html", {})


@app.get("/global", include_in_schema=False)
async def global_persons_page(request: Request):
    return templates.TemplateResponse(request, "global_persons.html", {})


@app.get("/global/{global_id}", include_in_schema=False)
async def global_person_page(request: Request, global_id: int):
    return templates.TemplateResponse(
        request, "global_person_detail.html", {"global_id": global_id}
    )


if __name__ == "__main__":
    uvicorn.run(app, host=settings.app_host, port=settings.app_port)
