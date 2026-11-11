"""RecognitionService — оркестрация кадра: YOLO → ByteTrack → лицо → сотрудник.

Ключевая логика (ТЗ):
- распознавание НЕ выполняется каждый кадр: employee_id сохраняется на треке;
- сотрудник подтверждается после MIN_CONFIRMATIONS подряд совпавших
  распознаваний (защита от false positives);
- для нераспознанных — повторная попытка раз в retry_interval секунд;
- ниже порога уверенности — Unknown (никаких автопрофилей).

Модуль не зависит ни от FastAPI, ни от RTSP: на входе numpy-кадр.
"""
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Optional
from uuid import uuid4

import numpy as np

from app.ai.detector import PersonDetector
from app.ai.face_detector import FaceEngine
from app.ai.face_recognition import EmbeddingStore
from app.ai.tracker import ByteTracker, Track, TrackLifecycle

logger = logging.getLogger("app.ai.service")


@dataclass
class DetectionResult:
    """Результат по одному человеку на кадре (нормализованный bbox 0..1)."""
    track_id: int
    employee_id: Optional[int]
    employee_name: Optional[str]
    confidence: Optional[float]
    bbox: tuple[float, float, float, float]
    state: str                      # recognized | unknown | detecting
    just_confirmed: bool = False    # подтверждён именно на этом кадре
    global_id: Optional[int] = None # проставляет GlobalIdentityManager (межкамерный слой)
    association_uncertain: bool = False


@dataclass
class FrameOutcome:
    detections: list[DetectionResult] = field(default_factory=list)
    # (track_id, employee_id, confidence) — только что подтверждённые
    new_recognitions: list[tuple[int, int, float]] = field(default_factory=list)
    tracking_generation: str = ""
    track_lifecycle: list[TrackLifecycle] = field(default_factory=list)


class RecognitionService:
    def __init__(
        self,
        detector: PersonDetector,
        face_engine: FaceEngine,
        store: EmbeddingStore,
        *,
        min_confidence: float = 0.45,
        min_confirmations: int = 2,
        retry_interval: float = 2.0,
        track_lost_timeout: float = 3.0,
        min_hits: int = 2,
        tracker_high_threshold: float = 0.5,
    ):
        self.detector = detector
        self.faces = face_engine
        self.store = store
        self.min_confidence = min_confidence
        self.min_confirmations = min_confirmations
        self.retry_interval = retry_interval
        self.track_lost_timeout = track_lost_timeout
        self.min_hits = min_hits
        self.tracker_high_threshold = tracker_high_threshold
        self._trackers: dict[int, ByteTracker] = {}
        self._generations: dict[int, str] = {}
        self._lock = threading.Lock()  # на случай нескольких AI-воркеров

    def process_frame(
        self, frame: np.ndarray, camera_id: int, now: Optional[float] = None
    ) -> FrameOutcome:
        now = time.monotonic() if now is None else now
        h, w = frame.shape[:2]

        persons = self.detector.detect(frame)
        with self._lock:
            generation = self._generations.get(camera_id)
            if generation is None:
                generation = str(uuid4())
                self._generations[camera_id] = generation
            tracker = self._trackers.setdefault(
                camera_id,
                ByteTracker(lost_timeout=self.track_lost_timeout,
                            high_threshold=self.tracker_high_threshold),
            )
            tracks = tracker.update(persons, now)

        outcome = FrameOutcome(
            tracking_generation=generation, track_lifecycle=tracker.lifecycle,
        )
        for track in tracks:
            if track.is_new:
                logger.info("Camera %d: person detected track=%d", camera_id, track.track_id)
            if track.hits < self.min_hits:
                continue  # неустойчивый трек — не показываем и не распознаём

            if (
                track.employee_id is None
                and (now - track.last_recog_attempt) >= self.retry_interval
            ):
                track.last_recog_attempt = now
                self._try_recognize(track, frame, camera_id, now, outcome)

            outcome.detections.append(self._to_result(track, w, h))
        return outcome

    def reset_camera(self, camera_id: int) -> str:
        """Discard local identity/continuity, including same-stream tracker resets."""
        with self._lock:
            self._trackers.pop(camera_id, None)
            generation = str(uuid4())
            self._generations[camera_id] = generation
            return generation

    # ------------------------------------------------------------ внутреннее

    def _try_recognize(
        self, track: Track, frame: np.ndarray, camera_id: int,
        now: float, outcome: FrameOutcome,
    ) -> None:
        """Кроп области головы → детекция лица → эмбеддинг → сравнение → подтверждение."""
        x1, y1, x2, y2 = track.bbox
        box_w, box_h = x2 - x1, y2 - y1
        # лицо — в верхней части person-bbox; «низкий» бокс (шире, чем выше)
        # обычно покрывает только голову/плечи — ищем лицо по всему боксу
        head_ratio = 0.55 if box_h > 1.2 * box_w else 1.0
        hx1 = max(0, int(x1 - box_w * 0.05))
        hx2 = min(frame.shape[1], int(x2 + box_w * 0.05))
        hy1 = max(0, int(y1))
        hy2 = min(frame.shape[0], int(y1 + box_h * head_ratio))
        if hy2 - hy1 < 40 or hx2 - hx1 < 40:
            return  # голова слишком мала для распознавания на этом потоке

        crop = frame[hy1:hy2, hx1:hx2]
        faces = self.faces.detect(crop, max_num=1)
        if not faces:
            return  # лицо не видно (повёрнут/спиной) — попробуем позже

        face = faces[0]
        employee_id, similarity = self.store.best_match(face.embedding)
        if employee_id is not None and similarity >= self.min_confidence:
            if track.candidate_id == employee_id:
                track.candidate_count += 1
            else:
                track.candidate_id = employee_id
                track.candidate_count = 1
            if track.candidate_count >= self.min_confirmations:
                track.employee_id = employee_id
                track.employee_conf = similarity
                track.best_unknown_sim = None
                name = self.store.employee_name(employee_id)
                logger.info(
                    "Camera %d: employee recognized employee=%d name=%s confidence=%.2f",
                    camera_id, employee_id, name, similarity,
                )
                outcome.new_recognitions.append((track.track_id, employee_id, similarity))
        else:
            # ниже порога — Unknown; запоминаем лучший скор для отображения
            track.candidate_id = None
            track.candidate_count = 0
            track.best_unknown_sim = similarity

    def _to_result(self, track: Track, frame_w: int, frame_h: int) -> DetectionResult:
        x1, y1, x2, y2 = track.bbox
        if track.employee_id is not None:
            state = "recognized"
            confidence = track.employee_conf
        elif track.last_recog_attempt > 0:
            state = "unknown"
            confidence = track.best_unknown_sim
        else:
            state = "detecting"
            confidence = None
        return DetectionResult(
            track_id=track.track_id,
            employee_id=track.employee_id,
            employee_name=self.store.employee_name(track.employee_id),
            confidence=round(confidence, 3) if confidence is not None else None,
            bbox=(
                max(0.0, x1 / frame_w), max(0.0, y1 / frame_h),
                min(1.0, x2 / frame_w), min(1.0, y2 / frame_h),
            ),
            state=state,
            association_uncertain=track.association_uncertain,
        )
