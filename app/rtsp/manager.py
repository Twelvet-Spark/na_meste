"""Менеджер RTSP-ридеров: по одному RTSPReader на камеру, управление жизненным циклом."""
import logging
import threading
from typing import Optional

from app.config import resolve_ffmpeg, settings
from app.rtsp.dahua import build_rtsp_url
from app.rtsp.reader import RTSPReader
from app.rtsp.source import CONNECTION_FIELDS, CameraSource

logger = logging.getLogger("app.rtsp.manager")


class ReaderManager:
    """Создаёт/останавливает ридеры и отдаёт их runtime-статусы.

    Потокобезопасен: доступен из FastAPI-хендлеров и фоновых потоков.
    """

    def __init__(self):
        self._readers: dict[int, RTSPReader] = {}
        self._published_sources: dict[int, CameraSource] = {}
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

    def start(self, camera: CameraSource) -> RTSPReader | None:
        """Publish one captured source; reject an already superseded revision."""
        with self._lock:
            if not self._admit_source_locked(camera):
                return None
            return self._start_locked(camera)

    def _admit_source_locked(self, camera: CameraSource) -> bool:
        previous = self._published_sources.get(camera.id)
        if previous is not None:
            if ((camera.publication_sequence, camera.source_revision)
                    < (previous.publication_sequence, previous.source_revision)):
                logger.info("Пропущена устаревшая публикация камеры #%d", camera.id)
                return False
            if (camera.created_at == previous.created_at
                    and camera.source_revision == previous.source_revision
                    and any(getattr(camera, name) != getattr(previous, name) for name in CONNECTION_FIELDS)):
                raise ValueError("One camera source revision cannot describe different connections")
        self._published_sources[camera.id] = camera
        return True

    def _start_locked(self, camera: CameraSource) -> RTSPReader | None:
        self._stop_locked(camera.id)
        if not camera.enabled:
            return None
        reader = RTSPReader(
            camera_id=camera.id,
            name=camera.name,
            rtsp_url=build_rtsp_url(
                camera.nvr_host, camera.rtsp_port, camera.username,
                camera.password, camera.channel, camera.stream_type,
            ),
            source_revision=camera.source_revision,
            ffmpeg_path=self._ffmpeg,
            ffprobe_path=self._ffprobe,
            frame_buffer_size=settings.frame_buffer_size,
            connect_timeout=settings.rtsp_connect_timeout,
            max_reconnect_delay=settings.rtsp_reconnect_max_delay,
            frame_stall_timeout=settings.frame_stall_timeout,
        )
        self._readers[camera.id] = reader
        # Starting is inside the publication lock too: an older delayed start
        # must not run after a newer reader has already replaced it.
        reader.start()
        logger.info("Ридер камеры #%d «%s» запущен", camera.id, camera.name)
        return reader

    def stop(self, camera_id: int) -> None:
        with self._lock:
            self._stop_locked(camera_id)
            self._published_sources.pop(camera_id, None)

    def get(self, camera_id: int) -> Optional[RTSPReader]:
        with self._lock:
            return self._readers.get(camera_id)

    def reader_ids(self) -> list[int]:
        """Идентификаторы камер с активными ридерами (для AI-воркера)."""
        with self._lock:
            return list(self._readers.keys())

    def sync(self, camera: CameraSource, changed_fields: Optional[set[str]] = None) -> None:
        """Apply a complete immutable source under the same publication lock."""
        with self._lock:
            if not self._admit_source_locked(camera):
                return
            reader = self._readers.get(camera.id)
            if not camera.enabled:
                # Keep the revision watermark even when no reader remains.
                self._stop_locked(camera.id)
            elif (reader is None or changed_fields is None or changed_fields & CONNECTION_FIELDS
                  or reader.source_revision != camera.source_revision):
                self._start_locked(camera)
            else:
                reader.name = camera.name

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
