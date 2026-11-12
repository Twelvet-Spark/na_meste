"""Deployment settings plus validated next-start operator overrides."""
import os
import shutil
from pathlib import Path

from typing import Literal

from pydantic import AliasChoices, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from app.operator_settings import normalize_camera_ids, read_document, settings_path, OverrideError

# Корень проекта — папка, в которой лежит main.py
BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    @classmethod
    def settings_customise_sources(cls, settings_cls, init_settings, env_settings,
                                   dotenv_settings, file_secret_settings):
        return env_settings, init_settings, dotenv_settings, file_secret_settings

    app_host: str = "0.0.0.0"
    app_port: int = 8000

    database_url: str = "sqlite:///./data/app.db"

    rtsp_connect_timeout: int = Field(default=10, ge=1, le=60)
    rtsp_reconnect_max_delay: int = Field(default=30, ge=1, le=300)
    default_rtsp_port: int = 554

    ffmpeg_path: str = ""                   # пусто — автоопределение
    preview_fps: int = Field(default=10, ge=1, le=30)
    frame_buffer_size: int = 2              # буфер кадров (drop old, keep latest)
    frame_stall_timeout: int = Field(default=10, ge=1, le=120)

    # Manual video only. Deployment paths/reserve are not operator-editable.
    recording_root: Path = BASE_DIR / "data" / "recordings"
    recording_retention_days: int = Field(default=14, ge=1, le=365)
    recording_max_bytes: int = Field(default=150000000000, ge=10000000000, le=100000000000000)
    recording_min_free_bytes: int = Field(default=2000000000, ge=0)

    @field_validator("recording_root", mode="after")
    @classmethod
    def resolve_recording_root(cls, value: Path) -> Path:
        return value if value.is_absolute() else BASE_DIR / value

    # --- AI (детекция людей, трекинг, распознавание лиц) ---
    ai_enabled: bool = True
    ai_fps: float = Field(default=5.0, gt=0, le=30, allow_inf_nan=False)
    ai_cameras: str = ""                    # canonical sorted IDs; empty = all cameras
    ai_device: Literal["auto", "cpu", "cuda"] = "auto"
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
    telegram_bot_token: str = Field(default="", repr=False)
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
    topology_config: str = ""               # topology.json — пусто = без ограничений
    mc_debug: bool = False                  # подробные логи [REID]/[MATCH]/[IDENTITY]

    # веса composite score (сумма ≈ 1.0)
    w_reid: float = 0.55
    w_temporal: float = 0.15
    w_topology: float = 0.15
    w_aspect: float = 0.05
    w_face: float = 0.10

    @field_validator("ai_cameras", mode="before")
    @classmethod
    def validate_camera_ids(cls, value):
        return normalize_camera_ids(value)

    @model_validator(mode="after")
    def attendance_window(self):
        if self.attendance_face_interval > self.attendance_face_window:
            raise ValueError("ATTENDANCE_INTERVAL_EXCEEDS_WINDOW")
        return self

    def ai_camera_ids(self) -> set[int] | None:
        """Идентификаторы камер для AI; None = все."""
        return {int(part) for part in self.ai_cameras.split(",")} if self.ai_cameras else None


def load_settings(*, path=None, env_file=BASE_DIR / ".env"):
    values, _version = read_document(settings_path() if path is None else path, Settings)
    try:
        return Settings(_env_file=env_file, **values)
    except Exception:
        # Pydantic errors can contain complete secret-bearing inputs. Startup must
        # fail before service initialization, with no chained validation traceback.
        raise OverrideError(
            "OPERATOR_SETTINGS_INVALID_EFFECTIVE_CONFIG: проверьте окружение, JSON и .env"
        ) from None


settings = load_settings()


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
