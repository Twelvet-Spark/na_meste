"""Менеджер RTSP-ридеров: по одному RTSPReader на камеру, управление жизненным циклом."""
import logging
import threading
from typing import Optional

from app.config import resolve_ffmpeg, settings
from app.database.models import Camera
from app.rtsp.dahua import build_rtsp_url
from app.rtsp.reader import RTSPReader

logger = logging.getLogger("app.rtsp.manager")


class ReaderManager:
    """Создаёт/останавливает ридеры и отдаёт их runtime-статусы.

    Потокобезопасен: доступен из FastAPI-хендлеров и фоновых потоков.
    """

    # Изменение любого из этих полей камеры требует перезапуска ридера
    CONNECTION_FIELDS = frozenset(
        {"nvr_host", "rtsp_port", "username", "password", "channel", "stream_type"}
    )

    def __init__(self):
        self._readers: dict[int, RTSPReader] = {}
        self._lock = threading.RLock()
        self._ffmpeg, self._ffprobe = resolve_ffmpeg()
        if not self._ffmpeg or not self._ffprobe:
            logger.error(
                "FFmpeg не найден (FFMPEG_PATH в .env, bin/ проекта или PATH). "
                "Камеры будут в статусе ERROR."
            )

    @property
    def ffprobe(self) -> Optional[str]:
        return self._ffprobe

    def start(self, camera: Camera, *, source_revision: int) -> RTSPReader:
        """Запускает ридер камеры (существующий перезапускается)."""
        with self._lock:
            self._stop_locked(camera.id)
            reader = RTSPReader(
                camera_id=camera.id,
                name=camera.name,
                rtsp_url=build_rtsp_url(
                    camera.nvr_host, camera.rtsp_port, camera.username,
                    camera.password, camera.channel, camera.stream_type,
                ),
                source_revision=source_revision,
                ffmpeg_path=self._ffmpeg,
                ffprobe_path=self._ffprobe,
                frame_buffer_size=settings.frame_buffer_size,
                connect_timeout=settings.rtsp_connect_timeout,
                max_reconnect_delay=settings.rtsp_reconnect_max_delay,
                frame_stall_timeout=settings.frame_stall_timeout,
            )
            self._readers[camera.id] = reader
        reader.start()
        logger.info("Ридер камеры #%d «%s» запущен", camera.id, camera.name)
        return reader

    def stop(self, camera_id: int) -> None:
        with self._lock:
            self._stop_locked(camera_id)

    def get(self, camera_id: int) -> Optional[RTSPReader]:
        with self._lock:
            return self._readers.get(camera_id)

    def reader_ids(self) -> list[int]:
        """Идентификаторы камер с активными ридерами (для AI-воркера)."""
        with self._lock:
            return list(self._readers.keys())

    def sync(
        self, camera: Camera, changed_fields: Optional[set[str]] = None,
        *, source_revision: int,
    ) -> None:
        """Приводит ридер в соответствие с настройками камеры после изменения."""
        with self._lock:
            reader = self._readers.get(camera.id)

        if not camera.enabled:
            if reader:
                self.stop(camera.id)
            return
        if reader is None:
            self.start(camera, source_revision=source_revision)
        elif (changed_fields is None or changed_fields & self.CONNECTION_FIELDS
              or reader.source_revision != source_revision):
            self.start(camera, source_revision=source_revision)
        else:
            reader.name = camera.name  # переименование — без переподключения

    def status(self, camera_id: int) -> dict:
        """Runtime-статус потока камеры (без обращения к БД)."""
        reader = self.get(camera_id)
        if reader:
            return reader.snapshot()
        return {
            "status": "OFFLINE",
            "last_frame_at": None,
            "seconds_since_last_frame": None,
            "current_fps": None,
            "resolution": None,
            "codec": None,
            "stream_fps": None,
            "reconnect_count": 0,
            "error": None,
        }

    def stop_all(self) -> None:
        with self._lock:
            readers = list(self._readers.values())
            self._readers.clear()
        for reader in readers:
            reader.stop()
        logger.info("Все ридеры остановлены (%d)", len(readers))

    def _stop_locked(self, camera_id: int) -> None:
        reader = self._readers.pop(camera_id, None)
        if reader:
            reader.stop()
