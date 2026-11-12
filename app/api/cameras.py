"""REST API камер + MJPEG-стрим.

Эндпоинты:

    GET    /api/cameras                  список + live-статусы
    POST   /api/cameras                  создать
    GET    /api/cameras/{id}             карточка + live-статус
    PUT    /api/cameras/{id}             изменить (пустой пароль = не менять)
    DELETE /api/cameras/{id}             удалить
    POST   /api/cameras/test             проверить параметры БЕЗ сохранения
    POST   /api/cameras/{id}/test        проверить сохранённую камеру
    GET    /api/cameras/{id}/status      runtime-статус потока
    GET    /api/cameras/{id}/stream      MJPEG (multipart/x-mixed-replace)
    GET    /api/cameras/{id}/snapshot    один JPEG-кадр (калибровка)
"""
import logging
import time
import threading
from typing import Optional

import cv2
import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import Response, StreamingResponse
from sqlalchemy.orm import Session
from fastapi.exceptions import RequestValidationError
from fastapi.routing import APIRoute

from app.config import settings
from app.database.database import get_db
from app.rtsp.manager import ReaderManager
from app.rtsp.reader import CameraStatus, RTSPReader
from app.services import camera_service as service
from app.services.camera_service import ConnectionTestRequest
from app.api.operator_security import require_operator_request
from app.services.dahua_service import DahuaSetup

logger = logging.getLogger("app.api.cameras")

class _CameraRoute(APIRoute):
    def get_route_handler(self):
        handler = super().get_route_handler()

        async def guarded(request):
            try:
                return await handler(request)
            except RequestValidationError as exc:
                # Model-level validation can include the whole credential-bearing
                # payload. Never echo request inputs or exception contexts.
                errors = [{key: item[key] for key in ("loc", "msg", "type") if key in item}
                          for item in exc.errors()]
                raise HTTPException(422, errors) from None

        return guarded


router = APIRouter(prefix="/api/cameras", tags=["cameras"], route_class=_CameraRoute)
_setup_lock = threading.Lock()


def _setup(request: Request) -> DahuaSetup:
    with _setup_lock:
        setup = getattr(request.app.state, "dahua_setup", None)
        if setup is None:
            setup = DahuaSetup()
            request.app.state.dahua_setup = setup
        return setup

JPEG_QUALITY = 70


def get_manager(request: Request) -> ReaderManager:
    return request.app.state.manager


def _detection_state(request: Request):
    """DetectionState AI-воркера (может отсутствовать, если AI выключен)."""
    return getattr(request.app.state, "detection_state", None)


def _camera_or_404(db: Session, camera_id: int):
    camera = service.get_camera(db, camera_id)
    if camera is None:
        raise HTTPException(status_code=404, detail="Camera not found")
    return camera


def _camera_response(camera, manager: ReaderManager, request: Request, db: Session) -> dict:
    """Камера без пароля + live-статус + счётчики людей (AI)."""
    data = service.camera_out(db, camera)
    data.update(manager.status(camera.id))
    state = _detection_state(request)
    if state is not None:
        people, known, unknown = state.counts(camera.id)
        data["people_count"] = people
        data["known_count"] = known
        data["unknown_count"] = unknown
    else:
        data["people_count"] = data["known_count"] = data["unknown_count"] = 0
    recordings = getattr(request.app.state, "recordings", None)
    sessions = recordings.list_sessions(camera_id=camera.id, limit=1) if recordings is not None else []
    data["recording"] = sessions[0].model_dump(mode="json") if sessions else None
    data["recording_available"] = recordings is not None
    return data


def _publish_recording_source(request: Request, source) -> None:
    recordings = getattr(request.app.state, "recordings", None)
    if recordings is not None:
        recordings.source_changed(source)
    playbacks = getattr(request.app.state, "playbacks", None)
    if playbacks is not None:
        playbacks.source_changed(source)


# -------------------------------------------------------------- проверка

@router.post("/test", dependencies=[Depends(require_operator_request)])
def api_test_unsaved(
    payload: ConnectionTestRequest,
    db: Session = Depends(get_db),
    manager: ReaderManager = Depends(get_manager),
):
    """Проверка параметров подключения без сохранения (форма добавления/правки).

    Если пароль не указан, но передан camera_id — пароль берётся из базы.
    """
    password = payload.password or ""
    if not password and payload.camera_id:
        camera = service.get_camera(db, payload.camera_id)
        if camera:
            password = camera.password
    if not password:
        return {"success": False, "error": "Укажите пароль для проверки подключения"}

    result = service.test_rtsp_connection(
        nvr_host=payload.nvr_host,
        rtsp_port=payload.rtsp_port,
        username=payload.username,
        password=password,
        channel=payload.channel,
        stream_type=payload.stream_type,
        ffprobe_path=manager.ffprobe,
        timeout=settings.rtsp_connect_timeout,
    )
    if result.error == "MEDIA_PROCESS_CONTAINMENT_UNAVAILABLE":
        raise HTTPException(503, result.error)
    return result


@router.post("/discover", dependencies=[Depends(require_operator_request)])
def api_discover(
    payload: service.DiscoveryRequest,
    request: Request,
    db: Session = Depends(get_db),
):
    source = None
    if payload.camera_id is not None:
        source = service.camera_snapshot(db, payload.camera_id)
        if source is None:
            raise HTTPException(404, "Camera not found")
    db.rollback()
    if not payload.password and not source:
        return {"channels": [], "capabilities": {"channel_titles": False}, "error_code": "DEVICE_AUTH_FAILED"}
    return _setup(request).discover(payload, source=source)


@router.post("/batch", status_code=201, dependencies=[Depends(require_operator_request)])
def api_create_batch(
    payload: service.CameraBatch,
    request: Request,
    db: Session = Depends(get_db),
    manager: ReaderManager = Depends(get_manager),
):
    result, sources, _replay = service.create_camera_batch(db, payload)
    for source in sources:
        _setup(request).source_changed(source)
        _publish_recording_source(request, source)
        if source.enabled:
            manager.start(source)
    return result


# ------------------------------------------------------------------- CRUD

@router.get("")
def api_list_cameras(
    request: Request,
    db: Session = Depends(get_db),
    manager: ReaderManager = Depends(get_manager),
):
    return [_camera_response(c, manager, request, db) for c in service.list_cameras(db)]


@router.post("", status_code=201, dependencies=[Depends(require_operator_request)])
def api_create_camera(
    payload: service.CameraCreate,
    request: Request,
    db: Session = Depends(get_db),
    manager: ReaderManager = Depends(get_manager),
):
    camera = service.create_camera(db, payload)
    _setup(request).source_changed(camera)
    _publish_recording_source(request, camera)
    if camera.enabled:
        manager.start(camera)
    return _camera_response(camera, manager, request, db)


@router.get("/{camera_id}")
def api_get_camera(
    camera_id: int,
    request: Request,
    db: Session = Depends(get_db),
    manager: ReaderManager = Depends(get_manager),
):
    camera = _camera_or_404(db, camera_id)
    return _camera_response(camera, manager, request, db)


@router.put("/{camera_id}", dependencies=[Depends(require_operator_request)])
def api_update_camera(
    camera_id: int,
    payload: service.CameraUpdate,
    request: Request,
    db: Session = Depends(get_db),
    manager: ReaderManager = Depends(get_manager),
):
    camera = _camera_or_404(db, camera_id)
    camera, changed = service.update_camera(db, camera, payload)
    _setup(request).source_changed(camera)
    _publish_recording_source(request, camera)
    manager.sync(camera, changed)
    return _camera_response(camera, manager, request, db)


@router.delete("/{camera_id}", status_code=204, dependencies=[Depends(require_operator_request)])
def api_delete_camera(
    camera_id: int,
    db: Session = Depends(get_db),
    manager: ReaderManager = Depends(get_manager),
    request: Request = None,
):
    camera = _camera_or_404(db, camera_id)
    service.delete_camera(db, camera)
    if request is not None:
        _setup(request).deleted(camera_id)
        playbacks = getattr(request.app.state, "playbacks", None)
        if playbacks is not None:
            playbacks.deleted(camera_id)
    manager.stop(camera_id)
    return None


# ---------------------------------------------------------------- статусы

@router.post("/{camera_id}/test", dependencies=[Depends(require_operator_request)])
def api_test_camera(
    camera_id: int,
    db: Session = Depends(get_db),
    manager: ReaderManager = Depends(get_manager),
):
    camera = _camera_or_404(db, camera_id)
    result = service.test_rtsp_connection(
        nvr_host=camera.nvr_host,
        rtsp_port=camera.rtsp_port,
        username=camera.username,
        password=camera.password,
        channel=camera.channel,
        stream_type=camera.stream_type,
        ffprobe_path=manager.ffprobe,
        timeout=settings.rtsp_connect_timeout,
    )
    if result.error == "MEDIA_PROCESS_CONTAINMENT_UNAVAILABLE":
        raise HTTPException(503, result.error)
    return result


@router.get("/{camera_id}/status")
def api_camera_status(
    camera_id: int,
    db: Session = Depends(get_db),
    manager: ReaderManager = Depends(get_manager),
):
    camera = _camera_or_404(db, camera_id)
    data = manager.status(camera.id)
    data["enabled"] = camera.enabled
    return data


@router.get("/{camera_id}/detections")
def api_camera_detections(
    camera_id: int,
    request: Request,
    db: Session = Depends(get_db),
):
    """Текущие люди на камере (для overlay bounding boxes на странице камеры)."""
    _camera_or_404(db, camera_id)
    state = _detection_state(request)
    data = state.camera(camera_id) if state else None
    if data is None:
        return {
            "camera_id": camera_id,
            "updated_at": None,
            "tracks": [],
            "objects": [],
            "people_count": 0,
            "known_count": 0,
            "unknown_count": 0,
        }
    return data


@router.get("/{camera_id}/tracks")
def api_camera_tracks(
    camera_id: int,
    request: Request,
    db: Session = Depends(get_db),
):
    """Активные локальные треки камеры с привязкой к global_id
    (межкамерный трекинг)."""
    _camera_or_404(db, camera_id)
    worker = getattr(request.app.state, "ai_worker", None)
    global_manager = getattr(worker, "global_manager", None) if worker else None
    if global_manager is None:
        return {"camera_id": camera_id, "multi_camera_tracking": False, "tracks": []}
    return {
        "camera_id": camera_id,
        "multi_camera_tracking": True,
        "tracks": global_manager.camera_tracks(camera_id),
    }


# ------------------------------------------------------------ MJPEG-поток

def _encode_jpeg(frame: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
    if not ok:
        raise RuntimeError("Не удалось закодировать кадр в JPEG")
    return buf.tobytes()


def _placeholder_jpeg(text: str) -> bytes:
    img = np.full((360, 640, 3), 26, dtype=np.uint8)
    cv2.putText(
        img, text, (320 - 8 * len(text), 185),
        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (120, 120, 120), 2, cv2.LINE_AA,
    )
    return _encode_jpeg(img)


def _multipart_part(jpeg: bytes) -> bytes:
    return (
        b"--frame\r\n"
        b"Content-Type: image/jpeg\r\n"
        b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n"
    )


def _mjpeg_generator(reader: Optional[RTSPReader], name: str):
    """MJPEG-поток из буфера ридера.

    Синхронный генератор — Starlette выполняет его в threadpool, event loop
    не блокируется. Кадры отдаются с ограничением PREVIEW_FPS; при потере
    потока — заглушка NO SIGNAL; при остановленном ридере — OFFLINE.
    """
    seq = 0
    min_interval = 1.0 / max(1, settings.preview_fps)
    last_sent = 0.0

    while True:
        if reader is None:
            return  # ридер не существует — поток завершается

        if not reader.is_alive():
            status = reader.status().value
            if status in ("OFFLINE", "ERROR"):
                yield _multipart_part(_placeholder_jpeg(f"{status}: {name}"))
                time.sleep(1.0)
                continue
            return  # ридер в процессе остановки — завершаем поток

        result = reader.buffer.wait_for_new(seq, timeout=2.0)
        if result is None:
            yield _multipart_part(_placeholder_jpeg("NO SIGNAL"))
            time.sleep(1.0)
            continue

        seq = result.seq
        now = time.monotonic()
        if now - last_sent < min_interval:
            continue  # кадр пришёл слишком рано — пропускаем, ждём следующий
        last_sent = now
        yield _multipart_part(_encode_jpeg(result.frame))


@router.get("/{camera_id}/stream")
def api_camera_stream(
    camera_id: int,
    db: Session = Depends(get_db),
    manager: ReaderManager = Depends(get_manager),
):
    camera = _camera_or_404(db, camera_id)
    reader = manager.get(camera.id)
    return StreamingResponse(
        _mjpeg_generator(reader, camera.name),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache, no-store",
            "Pragma": "no-cache",
        },
    )


@router.get("/{camera_id}/snapshot")
def api_camera_snapshot(
    camera_id: int,
    db: Session = Depends(get_db),
    manager: ReaderManager = Depends(get_manager),
):
    """Один статичный JPEG-кадр из буфера ридера."""
    camera = _camera_or_404(db, camera_id)
    reader = manager.get(camera.id)
    packet = reader.buffer.latest() if reader is not None else None
    if (reader is None or reader.status() is not CameraStatus.ONLINE or packet is None
            or time.monotonic() - packet.received_mono > settings.frame_stall_timeout):
        raise HTTPException(503, "Кадр недоступен: камера не подключена")
    return Response(content=_encode_jpeg(packet.frame), media_type="image/jpeg",
                    headers={"Cache-Control": "no-cache, no-store"})
