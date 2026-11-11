"""AI-воркер: один поток обрабатывает кадры всех камер.

Архитектура (ТЗ «Несколько камер → один AI-воркер»):

    Camera 1..N ─► FrameBuffer ридеров (queue=2, drop old / keep latest)
                        │
                        ▼
                   AI Worker ──► RecognitionService ──► DetectionState (для UI/API)
                        │
                        ▼
                 PresenceManager (сессии присутствия)

Кадры берутся напрямую из FrameBuffer ридеров: буфер уже реализует
«max queue size = 2, старый кадр отбрасывается» — бесконечных очередей нет.
Не успеваем — камера просто обрабатывается реже, превью не страдает.
Для масштабирования можно запустить несколько воркеров с разбиением по камерам
(service потокобезопасен).
"""
import logging
import threading
import time
from dataclasses import dataclass
from typing import Optional

from sqlalchemy.exc import SQLAlchemyError

from app.ai.detector import PersonDetector
from app.ai.face_detector import FaceEngine
from app.ai.face_recognition import EmbeddingStore
from app.ai.providers import provider_label, resolve_providers, setup_cuda_dlls
from app.ai.recognition_service import RecognitionService
from app.config import BASE_DIR, settings
from app.database.models import AttendanceCamera, Camera
from app.rtsp.manager import ReaderManager
from app.rtsp.reader import CameraStatus, FramePacket
from app.services.presence_service import PresenceManager

logger = logging.getLogger("app.ai.worker")


class DetectionState:
    """Потокобезопасные последние результаты детекции по камерам (для API/UI)."""

    def __init__(self):
        self._lock = threading.Lock()
        self._data: dict[int, dict] = {}

    def update(self, camera_id: int, detections) -> None:
        tracks = [
            {
                "track_id": d.track_id,
                "employee_id": d.employee_id,
                "employee_name": d.employee_name,
                "confidence": d.confidence,
                "bbox": list(d.bbox),
                "state": d.state,
                "global_id": d.global_id,
            }
            for d in detections
        ]
        with self._lock:
            self._data[camera_id] = {
                "camera_id": camera_id,
                "updated_at": time.time(),
                "tracks": tracks,
                "people_count": len(tracks),
                "known_count": sum(1 for t in tracks if t["employee_id"]),
                "unknown_count": sum(1 for t in tracks if not t["employee_id"]),
            }

    def camera(self, camera_id: int) -> Optional[dict]:
        with self._lock:
            return self._data.get(camera_id)

    def counts(self, camera_id: int) -> tuple[int, int, int]:
        with self._lock:
            item = self._data.get(camera_id)
            if not item:
                return 0, 0, 0
            return item["people_count"], item["known_count"], item["unknown_count"]

    def drop_camera(self, camera_id: int) -> None:
        with self._lock:
            self._data.pop(camera_id, None)


@dataclass(frozen=True, slots=True)
class _CameraConfiguration:
    source_revision: int
    calibration_revision: int
    enabled: bool
    attendance: bool


class AIWorker(threading.Thread):
    def __init__(
        self,
        manager: ReaderManager,
        service: RecognitionService,
        presence: PresenceManager,
        detection_state: DetectionState,
        global_manager=None,
        *,
        session_factory,
    ):
        super().__init__(name="ai-worker", daemon=True)
        self.manager = manager
        self.service = service
        self.presence = presence
        self.state = detection_state
        self.global_manager = global_manager
        self._session_factory = session_factory
        self._stop_event = threading.Event()
        self._last_seq: dict[int, int] = {}
        self._last_ts: dict[int, float] = {}
        self._last_received: dict[int, float] = {}
        self._streams: dict[int, tuple[str, int]] = {}
        self._configurations: dict[int, _CameraConfiguration] = {}
        self._last_global_cleanup = 0.0

    def stop(self, timeout: float = 3.0) -> None:
        self._stop_event.set()
        self.join(timeout=timeout)

    def run(self) -> None:
        logger.info("AI worker запущен (AI_FPS=%.1f на камеру)", settings.ai_fps)
        while not self._stop_event.is_set():
            did_work = False
            try:
                packets = self._collect_frames()
            except SQLAlchemyError:
                # Never process a source under an unverified persisted revision.
                logger.exception("AI worker: настройки камер недоступны")
                for camera_id in list(self._configurations):
                    self._reset_camera(camera_id)
                self._configurations.clear()
                self._stop_event.wait(0.05)
                continue

            for camera_id, packet, config in packets:
                if self._stop_event.is_set():
                    break
                started = time.monotonic()
                if (config.attendance
                        and started - packet.received_mono > settings.attendance_frame_max_age):
                    self._reset_camera(camera_id)
                    continue
                fps = settings.attendance_ai_fps if config.attendance else max(0.5, settings.ai_fps)
                interval = 1.0 / fps
                if started - self._last_ts.get(camera_id, float("-inf")) < interval:
                    continue
                self._last_seq[camera_id] = packet.seq
                self._last_ts[camera_id] = started
                self._last_received[camera_id] = packet.received_mono
                try:
                    outcome = self.service.process_frame(
                        packet.frame, camera_id, packet.received_mono,
                        frame_seq=packet.seq, received_at=packet.received_at,
                        attendance=config.attendance,
                    )
                except Exception:
                    logger.exception("Camera %d: ошибка AI-обработки кадра", camera_id)
                    self._reset_camera(camera_id)
                    self._streams[camera_id] = (packet.stream_id, packet.source_revision)
                    self._last_seq[camera_id] = packet.seq
                    continue

                if self.global_manager is not None:
                    try:
                        self.global_manager.process(
                            camera_id, outcome, packet.frame, packet.received_mono,
                        )
                    except Exception:
                        logger.exception(
                            "Camera %d: ошибка глобального трекинга — "
                            "локальный трекинг продолжается", camera_id,
                        )
                self.state.update(camera_id, outcome.detections)
                for _track_id, employee_id, confidence in outcome.new_recognitions:
                    self.presence.on_recognized(camera_id, employee_id, confidence)
                did_work = True

            self.presence.tick()
            now = time.monotonic()
            if (self.global_manager is not None
                    and now - self._last_global_cleanup > 3600.0):
                self._last_global_cleanup = now
                self.global_manager.cleanup()
            if not did_work:
                self._stop_event.wait(0.05)

        for camera_id in list(self._configurations):
            self._reset_camera(camera_id)
        logger.info("AI worker остановлен")

    def _reset_camera(self, camera_id: int) -> None:
        """Worker-owned reset of every generation-bound consumer."""
        self.service.reset_camera(camera_id)
        self.state.drop_camera(camera_id)
        if self.global_manager is not None:
            self.global_manager.drop_camera(camera_id)
        self._streams.pop(camera_id, None)
        self._last_seq.pop(camera_id, None)
        self._last_ts.pop(camera_id, None)
        self._last_received.pop(camera_id, None)

    def _read_configurations(self) -> dict[int, _CameraConfiguration]:
        with self._session_factory() as db:
            rows = db.query(
                AttendanceCamera.camera_id, AttendanceCamera.source_revision,
                AttendanceCamera.revision, AttendanceCamera.role,
                AttendanceCamera.enabled, Camera.enabled,
            ).join(Camera, Camera.id == AttendanceCamera.camera_id).all()
        return {
            camera_id: _CameraConfiguration(
                source, revision, camera_enabled and (role is None or enabled),
                role is not None,
            )
            for camera_id, source, revision, role, enabled, camera_enabled in rows
        }

    def _collect_frames(self) -> list[tuple[int, FramePacket, _CameraConfiguration]]:
        # Reconcile every camera before doing inference on any of their frames.
        camera_ids = self._camera_ids()
        configurations = self._read_configurations()
        included = {
            camera_id for camera_id in camera_ids
            if camera_id in configurations and configurations[camera_id].enabled
        }
        for camera_id in set(self._configurations) - included:
            self._reset_camera(camera_id)
            self._configurations.pop(camera_id, None)

        packets = []
        for camera_id in camera_ids:
            if camera_id not in included:
                continue
            config = configurations[camera_id]
            previous = self._configurations.get(camera_id)
            if previous is not None and previous != config:
                self._reset_camera(camera_id)
            self._configurations[camera_id] = config
            reader = self.manager.get(camera_id)
            if (reader is None or not reader.is_alive()
                    or reader.status() is not CameraStatus.ONLINE
                    or reader.source_revision != config.source_revision):
                if camera_id in self._streams:
                    self._reset_camera(camera_id)
                continue
            packet = reader.buffer.latest()
            if packet is None or packet.source_revision != config.source_revision:
                if camera_id in self._streams:
                    self._reset_camera(camera_id)
                continue
            stream = (packet.stream_id, packet.source_revision)
            if camera_id in self._streams and self._streams[camera_id] != stream:
                self._reset_camera(camera_id)
            self._streams[camera_id] = stream
            age = time.monotonic() - packet.received_mono
            if config.attendance and age > settings.attendance_frame_max_age:
                self._reset_camera(camera_id)
                continue
            if packet.seq <= self._last_seq.get(camera_id, 0):
                if (camera_id in self._last_received
                        and age > settings.attendance_max_frame_gap):
                    self._reset_camera(camera_id)
                    self._streams[camera_id] = stream
                    self._last_seq[camera_id] = packet.seq
                continue
            last_received = self._last_received.get(camera_id)
            if (last_received is not None
                    and packet.received_mono - last_received > settings.attendance_max_frame_gap):
                self._reset_camera(camera_id)
                self._streams[camera_id] = stream
            packets.append((camera_id, packet, config))
        return packets

    def _camera_ids(self) -> list[int]:
        ids = set(self.manager.reader_ids())
        allowed = settings.ai_camera_ids()
        if allowed:
            ids &= allowed
        return sorted(ids)


# ------------------------------------------------------------- инициализация

def build_enroll_engine() -> Optional[FaceEngine]:
    """Движок для загрузки фото сотрудников (det_size 640). None — AI недоступен."""
    setup_cuda_dlls()
    try:
        return FaceEngine(
            model_name=settings.insightface_model,
            root=settings.insightface_root,
            providers=resolve_providers(settings.ai_device),
            det_size=640,
        )
    except Exception as exc:
        logger.error("FaceEngine недоступна (загрузка фото отключена): %s", exc)
        return None


def build_ai_worker(
    manager: ReaderManager, presence: PresenceManager, store: EmbeddingStore,
    *, session_factory, notifier=None,
) -> Optional[AIWorker]:
    """Собирает полный AI-стек. None — модели недоступны (приложение работает
    только как RTSP-превью, не падая)."""
    setup_cuda_dlls()
    providers = resolve_providers(settings.ai_device)

    yolo_path = BASE_DIR / settings.yolo_model
    if not yolo_path.is_file():
        logger.error("YOLO-модель не найдена: %s — AI отключён (см. README)", yolo_path)
        return None

    # межкамерный трекинг (опциональный слой, не ломает остальное)
    global_manager = None
    if settings.multi_camera_tracking_enabled and session_factory is not None:
        reid_path = BASE_DIR / settings.reid_model
        if reid_path.is_file():
            try:
                from app.ai.global_tracker import CameraTopology, GlobalIdentityManager
                from app.ai.reid import PersonReID

                reid = PersonReID(str(reid_path), providers)
                topology = CameraTopology.load(settings.topology_config)
                global_manager = GlobalIdentityManager(
                    session_factory, reid, topology, notifier=notifier,
                )
                logger.info("Межкамерный трекинг включён (Re-ID: %s)",
                            reid_path.name)
            except Exception:
                logger.exception(
                    "Межкамерный трекинг не запустился — система работает без него")
                global_manager = None
        else:
            logger.warning("Re-ID модель не найдена: %s — межкамерный трекинг "
                           "отключён (см. README)", reid_path)

    try:
        detector = PersonDetector(
            str(yolo_path), providers, settings.person_confidence,
        )
        face_engine = FaceEngine(
            model_name=settings.insightface_model,
            root=settings.insightface_root,
            providers=providers,
            det_size=320,
        )
    except Exception as exc:
        logger.error("AI-модели не загрузились (%s) — AI отключён", exc)
        return None

    service = RecognitionService(
        detector,
        face_engine,
        store,
        min_confidence=settings.min_recognition_confidence,
        min_confirmations=settings.min_confirmations,
        retry_interval=settings.face_recognition_retry_interval,
        track_lost_timeout=settings.track_lost_timeout,
        tracker_high_threshold=settings.tracker_new_track_threshold,
    )
    state = DetectionState()
    logger.info("AI Provider: %s", provider_label(detector.session.get_providers()))
    return AIWorker(manager, service, presence, state,
                    global_manager=global_manager, session_factory=session_factory)
