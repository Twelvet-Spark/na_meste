"""Конфигурация приложения (читается из .env в корне проекта)."""
import os
import shutil
from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Корень проекта — папка, в которой лежит main.py
BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_host: str = "0.0.0.0"
    app_port: int = 8000

    database_url: str = "sqlite:///./data/app.db"

    rtsp_connect_timeout: int = 10          # сек, таймаут подключения к RTSP
    rtsp_reconnect_max_delay: int = 30      # сек, потолок exponential backoff
    default_rtsp_port: int = 554

    ffmpeg_path: str = ""                   # пусто — автоопределение
    preview_fps: int = 10                   # ограничение FPS MJPEG-превью
    frame_buffer_size: int = 2              # буфер кадров (drop old, keep latest)
    frame_stall_timeout: int = 10           # сек без кадров → принудительный реконнект

    # --- AI (детекция людей, трекинг, распознавание лиц) ---
    ai_enabled: bool = True
    ai_fps: float = 5.0                     # кадров/сек на камеру для AI (превью не ограничивается)
    ai_cameras: str = ""                    # "1,3,5" — пусто = все камеры
    ai_device: str = "auto"                 # auto | cpu | cuda
    person_confidence: float = 0.35         # порог YOLO для человека
    tracker_new_track_threshold: float = 0.5  # score детекции, создающей НОВЫЙ трек
                                              # (ниже — детекция только продлевает существующий)
    track_lost_timeout: float = 3.0         # сек grace period трека (re-identification)
    min_recognition_confidence: float = Field(
        default=0.45,
        validation_alias=AliasChoices("MIN_RECOGNITION_CONFIDENCE", "FACE_RECOGNITION_THRESHOLD"),
    )
    min_confirmations: int = 2              # подряд совпавших распознаваний до подтверждения
    face_recognition_retry_interval: float = 2.0  # сек между попытками распознать неизвестного
    presence_end_timeout: float = 30.0      # сек без наблюдений → сессия закрывается
    face_min_size: int = 80                 # мин. размер лица на фото, px
    yolo_model: str = "models/yolov8n.onnx"
    insightface_model: str = "buffalo_l"
    insightface_root: str = "."             # модели лежат в ./models/<имя_пакета>
    cuda_dll_path: str = ""                 # путь к CUDA/cuDNN DLL (пусто — авто-поиск)

    # --- Учёт рабочего времени: консервативные начальные пороги ---
    attendance_utc_offset_minutes: int = Field(default=300, ge=-720, le=840)
    attendance_ai_fps: float = Field(default=10, gt=0, allow_inf_nan=False)
    attendance_frame_max_age: float = Field(default=1.0, gt=0, allow_inf_nan=False)
    attendance_max_frame_gap: float = Field(default=0.75, gt=0, allow_inf_nan=False)
    attendance_face_threshold: float = Field(default=0.55, ge=0, le=1, allow_inf_nan=False)
    attendance_face_margin: float = Field(default=0.08, ge=0, le=1, allow_inf_nan=False)
    attendance_face_confirmations: int = Field(default=3, ge=1)
    attendance_face_window: float = Field(default=1.5, gt=0, allow_inf_nan=False)
    attendance_face_interval: float = Field(default=0.25, gt=0, allow_inf_nan=False)
    attendance_face_min_size: int = Field(default=80, ge=1)
    attendance_face_det_score: float = Field(default=0.80, ge=0, le=1, allow_inf_nan=False)
    attendance_face_blur_min: float = Field(default=20.0, ge=0, allow_inf_nan=False)
    attendance_evidence_keep_days: int = Field(default=30, ge=1)

    # --- Telegram ---
    telegram_enabled: bool = True
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""              # пусто — авто-определение по /start в группе
    telegram_notify_presence: bool = True   # сотрудник пришёл / ушёл
    telegram_notify_global_transition: bool = True  # человек перешёл между камерами (с фото)
    telegram_notify_global_new: bool = False        # уведомление о каждой новой глобальной личности

    # --- Межкамерный трекинг (Multi-Camera Person Tracking) ---
    multi_camera_tracking_enabled: bool = True
    reid_model: str = "models/osnet_x0_25_msmt17.onnx"
    reid_embedding_interval: float = 2.5    # сек между обновлениями эмбеддинга трека
    reid_similarity_threshold: float = 0.80 # мин. косинус OSNet для кандидата
    global_match_threshold: float = 0.65    # мин. composite score для связывания
    global_match_margin: float = 0.05       # мин. отрыв от второго кандидата (неоднозначность)
    global_gallery_ttl: float = 600.0       # сек жизни identity в памяти без наблюдений
    global_history_len: int = 20            # эмбеддингов в истории identity
    global_keep_days: int = 30              # хранение наблюдений/событий в БД
    topology_config: str = ""               # topology.json — пусто = без ограничений
    mc_debug: bool = False                  # подробные логи [REID]/[MATCH]/[IDENTITY]

    # веса composite score (сумма ≈ 1.0)
    w_reid: float = 0.55
    w_temporal: float = 0.15
    w_topology: float = 0.15
    w_aspect: float = 0.05
    w_face: float = 0.10

    def ai_camera_ids(self) -> set[int] | None:
        """Идентификаторы камер для AI; None = все."""
        raw = self.ai_cameras.strip()
        if not raw:
            return None
        ids = set()
        for part in raw.replace(";", ",").split(","):
            part = part.strip()
            if part.isdigit():
                ids.add(int(part))
        return ids or None


settings = Settings()


def _exe(name: str) -> str:
    return f"{name}.exe" if os.name == "nt" else name


def resolve_ffmpeg() -> tuple[str | None, str | None]:
    """Ищет ffmpeg/ffprobe: FFMPEG_PATH из .env → bin/ проекта → системный PATH.

    Возвращает (путь_к_ffmpeg, путь_к_ffprobe) или (None, None).
    """
    candidates: list[tuple[Path, Path]] = []

    if settings.ffmpeg_path:
        ffmpeg = Path(settings.ffmpeg_path)
        candidates.append((ffmpeg, ffmpeg.parent / _exe("ffprobe")))

    local_bin = BASE_DIR / "bin"
    candidates.append((local_bin / _exe("ffmpeg"), local_bin / _exe("ffprobe")))

    for ffmpeg, ffprobe in candidates:
        if ffmpeg.is_file() and ffprobe.is_file():
            return str(ffmpeg), str(ffprobe)

    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if ffmpeg and ffprobe:
        return ffmpeg, ffprobe
    return None, None
