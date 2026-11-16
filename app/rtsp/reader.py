"""RTSP-ридер: FFmpeg subprocess → numpy BGR-кадры → FrameBuffer.

Схема:

    RTSP (TCP) → ffmpeg -f rawvideo -pix_fmt bgr24 (pipe) → кадры numpy

Модуль не зависит ни от FastAPI, ни от AI-компонентов. Потребители разделяют
один неизменяемый исходный кадр; единственный latest-слот не хранит историю.
"""
import json
import logging
import re
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Optional
from uuid import uuid4

import numpy as np

from app.services.diagnostics_service import number, safe_code
from app.time_utils import utcnow
from app.media.process_guard import MediaProcessGuard

logger = logging.getLogger("app.rtsp.reader")


class CameraStatus(str, Enum):
    OFFLINE = "OFFLINE"            # ридер остановлен (камера выключена/удалена)
    CONNECTING = "CONNECTING"      # первичное подключение
    ONLINE = "ONLINE"              # кадры идут
    RECONNECTING = "RECONNECTING"  # соединение потеряно, переподключаемся
    ERROR = "ERROR"                # неисправимая ошибка (нет FFmpeg и т.п.)


class RTSPError(Exception):
    """Базовая ошибка RTSP-ридера."""


class ConnectionLost(RTSPError):
    """Соединение потеряно (EOF, таймаут, ошибка ffmpeg)."""



class HardwareInitializationFailed(ConnectionLost):
    """FFmpeg could not initialize the requested CUDA decoder."""

class StopRequested(RTSPError):
    """Запрошена остановка ридера."""


def parse_frame_rate(value: str | None) -> float | None:
    """'25/1' → 25.0, '0/0'/'N/A' → None."""
    if not value or value in ("0/0", "N/A"):
        return None
    try:
        if "/" in value:
            num, den = value.split("/", 1)
            num_f, den_f = float(num), float(den)
            return num_f / den_f if den_f else None
        return float(value)
    except (ValueError, ZeroDivisionError):
        return None


class Backoff:
    """Задержки реконнекта: 1, 2, 4, 8, 16, 30, 30, ... сек.

    Сбрасывается после успешного подключения (получен первый кадр).
    """

    def __init__(self, max_delay: float = 30.0):
        self.max_delay = max_delay
        self._attempt = 0

    def next(self) -> float:
        delay = min(2.0 ** self._attempt, self.max_delay)
        self._attempt += 1
        return delay

    def reset(self) -> None:
        self._attempt = 0


@dataclass(frozen=True, slots=True)
class FramePacket:
    """Server receipt metadata, not camera exposure time; frame is shared read-only."""
    stream_id: str
    source_revision: int
    seq: int
    received_at: datetime
    received_mono: float
    frame: np.ndarray


class FrameBuffer:
    """One shared latest packet; slow consumers never retain an image FIFO."""

    def __init__(self):
        self._packet: FramePacket | None = None
        self._seq = 0
        self._cond = threading.Condition()

    def put(
        self, frame: np.ndarray, *, stream_id: str, source_revision: int,
        received_at: datetime, received_mono: float,
    ) -> int:
        with self._cond:
            frame.setflags(write=False)
            self._seq += 1
            self._packet = FramePacket(
                stream_id, source_revision, self._seq, received_at, received_mono, frame,
            )
            self._cond.notify_all()
            return self._seq

    def latest(self) -> FramePacket | None:
        """Последний опубликованный пакет без копирования изображения."""
        with self._cond:
            return self._packet

    def wait_for_new(
        self, after_seq: int, timeout: float
    ) -> FramePacket | None:
        """Ждёт пакет новее after_seq; последовательность относится к этому буферу."""
        deadline = time.monotonic() + timeout
        with self._cond:
            while True:
                if self._packet is not None and self._packet.seq > after_seq:
                    return self._packet
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._cond.wait(remaining)

    def clear(self) -> None:
        """Discard disconnected frames without reusing sequence numbers."""
        with self._cond:
            self._packet = None
            self._cond.notify_all()


class RTSPReader:
    """Читает один RTSP-поток через FFmpeg в фоновом потоке.

    - RTSP over TCP (меньше потерь, чем UDP);
    - ffprobe перед стартом ffmpeg: метаданные (кодек/разрешение/FPS) и
      размер кадра для разбора rawvideo-пайпа;
    - при разрыве — автоматический реконнект с exponential backoff;
    - watchdog разрывает зависшие соединения без кадров;
    - корректное завершение: stop() гасит ffmpeg и поток чтения.
    """

    def __init__(
        self,
        camera_id: int,
        name: str,
        rtsp_url: str,
        *,
        source_revision: int,
        ffmpeg_path: Optional[str],
        ffprobe_path: Optional[str],
        target_fps: float = 15,
        hwaccel: str = "auto",
        connect_timeout: int = 10,
        max_reconnect_delay: int = 30,
        frame_stall_timeout: int = 10,
    ):
        self.camera_id = camera_id
        self.name = name
        self.rtsp_url = rtsp_url
        self.source_revision = source_revision

        self._ffmpeg = ffmpeg_path
        self._ffprobe = ffprobe_path
        if hwaccel not in {"auto", "cuda", "none"}:
            raise ValueError("hwaccel must be auto, cuda or none")
        if not np.isfinite(target_fps) or target_fps <= 0:
            raise ValueError("target_fps must be finite and positive")
        self._hwaccel = hwaccel
        self._software_fallback = False
        self._received_any_frame = False
        self._attempt_got_frame = False
        self.decoder_mode: str | None = None
        self._target_fps = target_fps
        self._connect_timeout = connect_timeout
        self._max_reconnect_delay = max_reconnect_delay
        self._stall_timeout = frame_stall_timeout

        self.buffer = FrameBuffer()

        self._thread: Optional[threading.Thread] = None
        self._watchdog_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._proc: Optional[subprocess.Popen] = None
        self._proc_lock = threading.Lock()

        self._status = CameraStatus.OFFLINE
        self._status_lock = threading.Lock()

        # Runtime-информация (только в памяти, не в БД)
        self.last_error: Optional[str] = None
        self.last_frame_at: Optional[float] = None      # time.time() последнего кадра
        self.current_fps: Optional[float] = None        # сглаженный FPS
        self.resolution: Optional[str] = None           # "1920x1080"
        self.codec: Optional[str] = None                # "h264" / "hevc"
        self.stream_fps: Optional[float] = None         # FPS из метаданных потока
        self.reconnect_count: int = 0
        self._last_frame_mono: Optional[float] = None
        self._attempt_started_mono: Optional[float] = None

    # ------------------------------------------------------------- публичное

    def start(self) -> None:
        """Запускает фоновое чтение потока (идемпотентно)."""
        if self.is_alive():
            return
        if not self._ffmpeg or not self._ffprobe:
            self._set_status(CameraStatus.ERROR)
            self.last_error = "FFMPEG_UNAVAILABLE"
            logger.error("FFMPEG_UNAVAILABLE", extra={"camera_id": self.camera_id})
            return

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, name=f"rtsp-reader-c{self.camera_id}", daemon=True
        )
        self._watchdog_thread = threading.Thread(
            target=self._watchdog, name=f"rtsp-watchdog-c{self.camera_id}", daemon=True
        )
        self._thread.start()
        self._watchdog_thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        """Корректная остановка: сигнал + гашение ffmpeg + join потоков."""
        self._stop_event.set()
        self.buffer.clear()
        with self._proc_lock:
            self._terminate_process()
        if self._thread:
            self._thread.join(timeout=timeout)
        if self._watchdog_thread:
            self._watchdog_thread.join(timeout=1.0)
        self._set_status(CameraStatus.OFFLINE)
        self.buffer.clear()

    # Требуемый интерфейс ридера
    def connect(self) -> None:
        self.start()

    def disconnect(self) -> None:
        self.stop()

    def reconnect(self) -> None:
        """Принудительный перезапуск чтения."""
        self.stop()
        self.start()

    def is_alive(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def read(self) -> FramePacket | None:
        """Неблокирующе возвращает последний пакет из буфера."""
        return self.buffer.latest()

    def status(self) -> CameraStatus:
        with self._status_lock:
            return self._status

    def snapshot(self) -> dict:
        """Runtime-информация о потоке для API/UI."""
        seconds_since = None
        if self._last_frame_mono is not None:
            seconds_since = round(max(0.0, time.monotonic() - self._last_frame_mono), 1)
        return {
            "status": self.status().value,
            "last_frame_at": (
                time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(self.last_frame_at))
                if self.last_frame_at else None
            ),
            "seconds_since_last_frame": seconds_since,
            "current_fps": number(round(self.current_fps, 1)) if self.current_fps is not None else None,
            "resolution": self.resolution if isinstance(self.resolution, str) and (
                re.fullmatch(r"[0-9]{1,5}x[0-9]{1,5}", self.resolution)) else None,
            "codec": self.codec if self.codec in {"h264", "hevc", "mjpeg", "mpeg4", "vp8", "vp9", "av1"} else None,
            "stream_fps": number(self.stream_fps),
            "reconnect_count": number(self.reconnect_count, integer=True),
            "decoder_requested": self._hwaccel,
            "decoder_mode": self.decoder_mode,
            "decoder_warning": "DECODE_SOFTWARE_FALLBACK" if self._software_fallback else None,
            "error": safe_code(self.last_error, "STREAM_UNAVAILABLE") if self.last_error else None,
        }

    # ----------------------------------------------------------- внутреннее

    def _set_status(self, status: CameraStatus) -> None:
        with self._status_lock:
            self._status = status

    def _run(self) -> None:
        logger.info("Camera #%d reader started", self.camera_id)
        backoff = Backoff(self._max_reconnect_delay)
        first_attempt = True
        while not self._stop_event.is_set():
            self._set_status(
                CameraStatus.CONNECTING if first_attempt else CameraStatus.RECONNECTING
            )
            try:
                self._connect_and_read()
            except StopRequested:
                break
            except Exception as exc:
                if getattr(exc, "detail", None) == "MEDIA_PROCESS_CONTAINMENT_UNAVAILABLE":
                    self.last_error = "MEDIA_PROCESS_CONTAINMENT_UNAVAILABLE"
                elif isinstance(exc, HardwareInitializationFailed):
                    self.last_error = "DECODE_HARDWARE_INIT_FAILED"
                else:
                    self.last_error = "STREAM_UNAVAILABLE"
                logger.warning(self.last_error, extra={"camera_id": self.camera_id})
            if self._stop_event.is_set():
                break

            if self._attempt_got_frame:
                backoff.reset()
            first_attempt = False
            self.reconnect_count += 1
            delay = backoff.next()
            self._set_status(CameraStatus.RECONNECTING)
            self._attempt_started_mono = None  # ждём backoff — watchdog не активен
            logger.warning(
                "Camera #%d reconnect in %.0f seconds (attempt %d)",
                self.camera_id, delay, self.reconnect_count,
            )
            if self._stop_event.wait(delay):
                break

        self._set_status(CameraStatus.OFFLINE)
        logger.info("Camera #%d reader stopped", self.camera_id)

    def _connect_and_read(self) -> None:
        """Read one connection until EOF/stop, with one eligible auto fallback."""
        self.buffer.clear()
        self._attempt_got_frame = False
        self.decoder_mode = None
        width, height, _codec, _stream_fps = self._probe()
        mode = "software" if self._hwaccel == "none" or self._software_fallback else "cuda"
        try:
            return self._read_connection(width, height, mode)
        except HardwareInitializationFailed:
            if (self._hwaccel != "auto" or self._software_fallback
                    or self._received_any_frame or self._stop_event.is_set()):
                raise
            self._software_fallback = True
            logger.warning("DECODE_SOFTWARE_FALLBACK", extra={"camera_id": self.camera_id})
            return self._read_connection(width, height, "software")

    def _read_connection(self, width: int, height: int, mode: str) -> None:
        """Run one contained decoder; modes are verified only by a real frame."""
        self._attempt_started_mono = time.monotonic()
        frame_bytes = width * height * 3
        cmd = [
            self._ffmpeg,
            "-hide_banner", "-loglevel", "error",
            "-rtsp_transport", "tcp",
            "-timeout", str(self._connect_timeout * 1_000_000),  # мкс
            "-fflags", "nobuffer",
            "-flags", "low_delay",
        ]
        if mode == "cuda":
            cmd += ["-hwaccel", "cuda"]
        cmd += [
            "-i", self.rtsp_url,
            "-map", "0:v:0",
            # кадры отдаются как есть, без дублирования до tbr источника
            "-fps_mode", "passthrough",
            "-an", "-sn", "-dn",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
        ]
        expression = (
            "if(isnan(t),1,if(isnan(prev_selected_t),1,"
            "if(lt(t,prev_selected_t),1,"
            f"gte(t-prev_selected_t,1/{self._target_fps:g}-0.000001))))"
        )
        cmd += ["-vf", "select=" + expression.replace(",", r"\,"), "pipe:1"]

        with self._proc_lock:
            if self._stop_event.is_set():
                raise StopRequested()
            logger.debug("Camera #%d launching live decoder", self.camera_id)
            MediaProcessGuard.require()
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                close_fds=True,
            )
            self._proc = proc
        stream_id = str(uuid4())

        # ffmpeg не должен блокироваться на переполненном stderr — сливаем его
        stderr_tail: deque[str] = deque(maxlen=3)
        stderr_thread = threading.Thread(
            target=self._drain_stderr, args=(proc.stderr, stderr_tail), daemon=True
        )
        stderr_thread.start()

        got_frame = False
        try:
            stdout = proc.stdout
            while not self._stop_event.is_set():
                chunk = self._read_exact(stdout, frame_bytes)
                if chunk is None:
                    break
                received_mono = time.monotonic()
                received_at = utcnow()
                if self._stop_event.is_set():
                    raise StopRequested()
                frame = np.frombuffer(chunk, dtype=np.uint8).reshape((height, width, 3))
                if not got_frame:
                    got_frame = True
                    self._attempt_got_frame = True
                    self._received_any_frame = True
                    self.decoder_mode = mode
                    self.last_error = None
                    self._attempt_started_mono = None
                    self._set_status(CameraStatus.ONLINE)
                    if self.reconnect_count:
                        logger.info("[%s] RECONNECTED", self.name)
                self._on_frame(received_at, received_mono)
                self.buffer.put(
                    frame, stream_id=stream_id, source_revision=self.source_revision,
                    received_at=received_at, received_mono=received_mono,
                )
            if self._stop_event.is_set():
                raise StopRequested()
        finally:
            self.buffer.clear()
            self.decoder_mode = None
            self._set_status(
                CameraStatus.OFFLINE if self._stop_event.is_set() else CameraStatus.RECONNECTING
            )
            with self._proc_lock:
                self._terminate_process()
            try:
                proc.stdout.close()
            except Exception:
                pass
            stderr_thread.join(timeout=1.0)
        if not got_frame and mode == "cuda" and "DECODE_HARDWARE_INIT_FAILED" in stderr_tail:
            self.last_error = "DECODE_HARDWARE_INIT_FAILED"
            raise HardwareInitializationFailed(self.last_error)
        self.last_error = "STREAM_UNAVAILABLE"
        raise ConnectionLost(self.last_error)

    def _probe(self) -> tuple[int, int, str, Optional[float]]:
        """Метаданные потока через ffprobe: (width, height, codec, fps)."""
        cmd = [
            self._ffprobe,
            "-v", "error",
            "-rtsp_transport", "tcp",
            "-select_streams", "v:0",
            "-show_entries", "stream=codec_name,width,height,avg_frame_rate",
            "-of", "json",
            self.rtsp_url,
        ]
        try:
            MediaProcessGuard.require()
            result = subprocess.run(
                cmd, capture_output=True, timeout=self._connect_timeout + 5, close_fds=True
            )
        except subprocess.TimeoutExpired as exc:
            raise ConnectionLost(
                f"ffprobe: нет ответа за {self._connect_timeout + 5} сек"
            ) from exc
        if result.returncode != 0:
            raise ConnectionLost("STREAM_UNAVAILABLE")
        try:
            data = json.loads(result.stdout.decode("utf-8", "replace"))
            stream = data["streams"][0]
            width, height = int(stream["width"]), int(stream["height"])
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise ConnectionLost("ffprobe: нет видеопотока в ответе") from exc

        codec = str(stream.get("codec_name") or "unknown")
        fps = parse_frame_rate(stream.get("avg_frame_rate"))
        self.resolution = f"{width}x{height}"
        self.codec = codec
        self.stream_fps = fps
        return width, height, codec, fps

    def _on_frame(self, received_at: datetime, received_mono: float) -> None:
        """Обновляет FPS по тем же отметкам получения полного кадра."""
        now_mono = received_mono
        prev = self._last_frame_mono
        self._last_frame_mono = now_mono
        self.last_frame_at = received_at.replace(tzinfo=timezone.utc).timestamp()
        if prev is not None:
            dt = now_mono - prev
            if dt > 0:
                instant = 1.0 / dt
                if self.current_fps is None:
                    self.current_fps = instant
                else:
                    self.current_fps = 0.9 * self.current_fps + 0.1 * instant

    def _watchdog(self) -> None:
        """Следит, что кадры вообще приходят:

        - ONLINE: нет кадров дольше frame_stall_timeout → разрыв соединения;
        - CONNECTING/RECONNECTING: первый кадр не получен за
          max(30, connect_timeout*3) сек (NVR может отдать «мёртвую» сессию,
          которая держит TCP, но не шлёт медиа) → принудительный реконнект.
        """
        first_frame_timeout = max(30, self._connect_timeout * 3)
        while not self._stop_event.wait(1.0):
            status = self.status()
            if status is CameraStatus.ONLINE:
                last = self._last_frame_mono
                if last is not None and time.monotonic() - last > self._stall_timeout:
                    logger.warning(
                        "Camera #%d frame stalled for %d seconds",
                        self.camera_id, self._stall_timeout,
                    )
                    with self._proc_lock:
                        # чтение разблокируется по EOF → цикл реконнекта
                        self._terminate_process()
            elif status in (CameraStatus.CONNECTING, CameraStatus.RECONNECTING):
                started = self._attempt_started_mono
                if started is not None and time.monotonic() - started > first_frame_timeout:
                    logger.warning(
                        "Camera #%d first frame timeout after %d seconds",
                        self.camera_id, first_frame_timeout,
                    )
                    with self._proc_lock:
                        self._terminate_process()

    def _terminate_process(self) -> None:
        """Гасит ffmpeg, если он запущен. Вызывается под _proc_lock."""
        proc = self._proc
        self._proc = None
        if proc is None:
            return
        try:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    try:
                        proc.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        pass
        except Exception:
            pass

    @staticmethod
    def _drain_stderr(pipe, sink: deque) -> None:
        """Drain external stderr; only a safe code may enter the bounded buffer."""
        try:
            for raw in iter(lambda: pipe.readline(4096), b""):
                line = raw.decode("utf-8", "replace").strip().lower()
                if any(marker in line for marker in (
                    "cannot load nvcuda", "cannot load libcuda", "cannot dynamically load cuda",
                    "cuda_error_no_device", "cuda_error_system_driver_mismatch",
                    "no device available for decoder: device type cuda",
                    "device creation failed", "device setup failed for decoder",
                    "hwaccel initialisation returned error", "hwaccel initialization returned error",
                    "unrecognized hwaccel: cuda", "unsupported hwaccel: cuda",
                )) or ("cuinit" in line and "failed" in line):
                    sink.append("DECODE_HARDWARE_INIT_FAILED")
                elif line and "DECODE_HARDWARE_INIT_FAILED" not in sink:
                    sink.append("STREAM_UNAVAILABLE")
        except Exception:
            pass
        finally:
            try:
                pipe.close()
            except Exception:
                pass

    @staticmethod
    def _read_exact(pipe, size: int) -> Optional[bytes]:
        """Читает ровно size байт из пайпа. None = EOF (обрыв потока)."""
        buf = bytearray()
        while len(buf) < size:
            chunk = pipe.read(size - len(buf))
            if not chunk:
                return None
            buf += chunk
        return bytes(buf)
