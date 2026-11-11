"""GlobalIdentityManager — глобальная идентичность человека между камерами.

Слой НАД локальным трекингом (ByteTracker на каждую камеру не меняется):
локальный (camera_id, track_id) привязывается к глобальному global_id по
composite score:

    final = w_reid*reid + w_temporal*temporal + w_topology*topology
          + w_aspect*aspect + w_face*face

Ложное слияние опаснее лишнего global_id (ТЗ §10): при неоднозначности
(два кандидата с близкими оценками) илиscore ниже порога создаётся НОВЫЙ
global_id, а сомнительный матч записывается в payload для анализа.

Состояния identity: NEW → ACTIVE → LOST (по таймауту без наблюдений; в
global_events пишется событие person_lost). Связывание: PROVISIONAL_MATCH
при первом матче, CONFIRMED_MATCH после повторного совпадения эмбеддинга
с той же identity.

Дробление identity безопаснее ложного слияния. При запуске сохранённые
личности становятся неизменяемым архивом: новая галерея пуста, новые ID
строго выше максимального сохранённого. Архив не участвует в матчинге.

Ошибка Re-ID/БД не должна ронять pipeline: воркер вызывает process()
в try/except; здесь дополнительно защищены записи в БД.
"""
import json
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from sqlalchemy import func

from app.ai.reid import PersonReID
from app.config import BASE_DIR, settings
from app.database.models import (
    Camera, Employee, GlobalEvent, GlobalObservation, GlobalPerson,
)
from app.time_utils import utcnow

logger = logging.getLogger("app.ai.global_tracker")

SNAPSHOT_MAX_HEIGHT = 400
MIN_CROP_HEIGHT = 48      # меньше — appearance-эмбеддинг ненадёжен
SWEEP_INTERVAL = 2.0      # сек между подчистками привязок
SAME_CAMERA_OVERLAP = 2.0 # сек: identity «занята» другим треком той же камеры
DB_SYNC_INTERVAL = 10.0   # сек между записями global_persons активной identity


def _dbg(msg: str, *args) -> None:
    if settings.mc_debug:
        logger.info(msg, *args)


def _local_time() -> str:
    from datetime import datetime
    return datetime.now().strftime("%d.%m.%Y %H:%M:%S")


# --------------------------------------------------------------- топология

class CameraTopology:
    """Связи между камерами и допустимое время перехода (topology.json).

    Если переход from→to описан — матч разрешён только при
    min_seconds <= dt <= max_seconds, иначе кандидат отклоняется.
    Переход без правила — нейтральная оценка (topology нестрогая).
    """

    def __init__(self, transitions: dict[tuple[int, int], tuple[float, float]]):
        self.transitions = transitions

    @classmethod
    def load(cls, path: str) -> Optional["CameraTopology"]:
        """Читает topology.json; None — файл не задан/не найден."""
        if not path:
            return None
        file = Path(path)
        if not file.is_absolute():
            file = BASE_DIR / path
        if not file.is_file():
            logger.warning("Топология не найдена: %s — работаем без ограничений", path)
            return None
        try:
            data = json.loads(file.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Топология не читается (%s): %s", path, exc)
            return None

        transitions: dict[tuple[int, int], tuple[float, float]] = {}
        for rule in data.get("transitions", []):
            try:
                transitions[(int(rule["from"]), int(rule["to"]))] = (
                    float(rule.get("min_seconds", 0)),
                    float(rule.get("max_seconds", 3600)),
                )
            except (KeyError, TypeError, ValueError):
                continue
        # adjacency без явных времён → широкое окно по умолчанию
        for src, node in (data.get("camera_topology") or {}).items():
            for dst in node.get("next", []):
                key = (int(src), int(dst))
                if key not in transitions:
                    transitions[key] = (0.0, 600.0)
        logger.info("Топология загружена: %d переходов", len(transitions))
        return cls(transitions)

    def check(self, from_cam: Optional[int], to_cam: int, dt: float) -> tuple[bool, float]:
        """(разрешено, оценка 0..1). dt — секунд с последнего наблюдения."""
        if from_cam is None:
            return True, 0.5          # неизвестно откуда пришёл — нейтрально
        if from_cam == to_cam:
            return True, 0.75         # смена трека на той же камере — норма
        rule = self.transitions.get((from_cam, to_cam))
        if rule is None:
            return True, 0.4          # правило не задано — мягкий штраф
        lo, hi = rule
        if lo <= dt <= hi:
            return True, 1.0
        return False, 0.0             # физически не мог перейти — veto


# ---------------------------------------------------------------- identities

@dataclass
class GlobalIdentity:
    global_id: int
    created_at: float                                  # monotonic
    last_seen: float                                   # monotonic
    last_seen_wall: datetime
    last_camera: Optional[int] = None
    last_track_id: Optional[int] = None
    employee_id: Optional[int] = None                  # из распознавания лиц
    status: str = "ACTIVE"                             # NEW/ACTIVE/LOST
    aspect: float = 0.0                                # EMA пропорций фигуры
    embeddings: deque = field(default_factory=deque)   # [(vec, mono_ts, camera)]
    last_db_sync: float = 0.0                          # monotonic, троттлинг записей


@dataclass
class TrackBinding:
    """Привязка локального (camera, track) к global_id."""
    global_id: int
    created_at: float
    last_seen: float
    last_embed_at: float = 0.0
    confirmations: int = 1
    confirmed: bool = False                            # CONFIRMED_MATCH
    last_crop: Optional[np.ndarray] = None


# ------------------------------------------------------------------ менеджер

class GlobalIdentityManager:
    def __init__(self, session_factory, reid: PersonReID,
                 topology: Optional[CameraTopology] = None, notifier=None):
        self._session_factory = session_factory
        self.reid = reid
        self.topology = topology
        self.notifier = notifier
        self._identities: dict[int, GlobalIdentity] = {}
        self._bindings: dict[tuple[int, int], TrackBinding] = {}
        self._lock = threading.RLock()
        self._last_sweep = 0.0
        # A failed seed must abort this optional manager, never reuse ID 1.
        # build_ai_worker isolates this failure from attendance inference.
        with self._session_factory() as db:
            maximum = db.query(func.max(GlobalPerson.id)).scalar()
        self._next_id = (maximum if maximum is not None else 0) + 1

    # ------------------------------------------------------------ публичное

    def process(self, camera_id: int, outcome, frame: np.ndarray, now: float) -> None:
        """Вызывается воркером после process_frame: привязывает локальные треки
        к глобальным личностям и проставляет detection.global_id."""
        h, w = frame.shape[:2]
        for det in outcome.detections:
            key = (camera_id, det.track_id)
            try:
                with self._lock:
                    binding = self._bindings.get(key)
                    if binding is not None:
                        binding.last_seen = now
                        det.global_id = binding.global_id
                        need_embed = (now - binding.last_embed_at
                                      >= settings.reid_embedding_interval)
                    else:
                        need_embed = True
                if binding is None:
                    self._bind_new_track(camera_id, det, frame, w, h, now)
                elif need_embed:
                    self._refresh_embedding(binding, camera_id, det, frame, w, h, now)
            except Exception:
                logger.exception(
                    "Camera %d: ошибка Re-ID/матчинга track=%d — локальный "
                    "трекинг не затронут", camera_id, det.track_id,
                )
        self.sweep(now)

    def sweep(self, now: float) -> None:
        """Отвязать протухшие треки; identity без наблюдений → LOST."""
        if now - self._last_sweep < SWEEP_INTERVAL:
            return
        self._last_sweep = now
        stale_keys = [
            key for key, b in self._bindings.items()
            if now - b.last_seen > settings.track_lost_timeout + 5.0
        ]
        for key in stale_keys:
            with self._lock:
                self._bindings.pop(key, None)
        ttl = settings.global_gallery_ttl
        for ident in list(self._identities.values()):
            if now - ident.last_seen > ttl:
                with self._lock:
                    self._identities.pop(ident.global_id, None)
                    if ident.status != "LOST":
                        ident.status = "LOST"
                        self._db_update_person(ident)
                        self._db_event(
                            ident.global_id, "person_lost",
                            ident.last_camera, ident.last_track_id,
                            {
                                "last_camera": ident.last_camera,
                                "duration_seconds": round(now - ident.created_at, 1),
                                "gallery_size": len(ident.embeddings),
                                "employee_id": ident.employee_id,
                            },
                        )

    def drop_camera(self, camera_id: int) -> None:
        """Forget generation-bound local bindings, never persisted global history."""
        with self._lock:
            for key in [key for key in self._bindings if key[0] == camera_id]:
                self._bindings.pop(key, None)

    def camera_tracks(self, camera_id: int) -> list[dict]:
        """Активные локальные треки камеры с global_id (для API/UI)."""
        now = time.monotonic()
        with self._lock:
            return [
                {
                    "track_id": track_id,
                    "camera_id": camera_id,
                    "global_id": b.global_id,
                    "confirmed": b.confirmed,
                    "seconds_since_seen": round(now - b.last_seen, 1),
                }
                for (cam, track_id), b in self._bindings.items()
                if cam == camera_id
            ]

    def snapshot(self) -> list[dict]:
        """Живые глобальные личности (для отладки/UI)."""
        now = time.monotonic()
        with self._lock:
            return [
                {
                    "global_id": i.global_id,
                    "status": i.status,
                    "employee_id": i.employee_id,
                    "last_camera": i.last_camera,
                    "last_track_id": i.last_track_id,
                    "seconds_since_seen": round(now - i.last_seen, 1),
                    "embeddings": len(i.embeddings),
                }
                for i in self._identities.values()
            ]

    # ------------------------------------------------------------- матчинг

    def _bind_new_track(self, camera_id, det, frame, w, h, now) -> None:
        crop = self._crop(frame, det.bbox, w, h)
        if crop is None:
            _dbg("[REID] camera=%d track=%d embedding_generated=false (кроп мал)",
                 camera_id, det.track_id)
            return  # ждём более крупный кроп — identity пока не назначаем
        embedding = self.reid.embed([crop])[0]
        _dbg("[REID] camera=%d track=%d embedding_generated=true",
             camera_id, det.track_id)

        identity, scores = self._match_or_create(camera_id, det, embedding, now)
        key = (camera_id, det.track_id)
        with self._lock:
            self._bindings[key] = TrackBinding(
                global_id=identity.global_id, created_at=now, last_seen=now,
                last_embed_at=now,
            )
            # прежняя камера нужна событию camera_transition; саму identity
            # обновляем ДО записи в БД, чтобы global_persons не отставал на шаг
            previous_camera = identity.last_camera
            previous_track = identity.last_track_id
            identity.last_camera = camera_id
            identity.last_track_id = det.track_id
            self._update_aspect(identity, det)
        det.global_id = identity.global_id

        snapshot = self._snapshot_jpeg(crop)
        # создать/обновить global_persons (FK для событий и наблюдений) — уже
        # с актуальной камерой и временем
        self._db_update_person(identity)
        identity.last_db_sync = now
        self._record_event(identity, camera_id, det, scores, snapshot,
                           previous_camera=previous_camera,
                           previous_track=previous_track)
        self._persist_observation(identity, camera_id, det, embedding, snapshot)
        _dbg("[IDENTITY] camera=%d track=%d → global_id=%d (%s)",
             camera_id, det.track_id, identity.global_id,
             "match" if scores.get("matched", False) else "new")

    def _refresh_embedding(self, binding: TrackBinding, camera_id, det,
                           frame, w, h, now) -> None:
        """Периодическое обновление appearance-эмбеддинга активного трека:
        обновляет галерею identity и подтверждает связывание."""
        crop = self._crop(frame, det.bbox, w, h)
        if crop is None:
            return
        embedding = self.reid.embed([crop])[0]
        binding.last_embed_at = now

        employee_changed = False
        with self._lock:
            identity = self._identities.get(binding.global_id)
            if identity is None:
                return
            identity.last_seen = now
            identity.last_seen_wall = utcnow()
            identity.last_camera = camera_id
            identity.last_track_id = det.track_id
            self._update_aspect(identity, det)
            if det.employee_id and identity.employee_id != det.employee_id:
                employee_changed = True       # сотрудник распознан — в БД сразу
                identity.employee_id = det.employee_id
            self._push_embedding(identity, embedding, now, camera_id)
            best_gid, best_sim, _scores = self._best_candidate(
                camera_id, det, embedding, now, exclude=binding.global_id)
            same_sim = self._identity_similarity(identity, embedding)
            # подтверждение: свежий эмбеддинг снова совпал со своей identity
            if same_sim >= settings.reid_similarity_threshold:
                binding.confirmations += 1
                if not binding.confirmed and binding.confirmations >= 2:
                    binding.confirmed = True
                    _dbg("[IDENTITY] camera=%d track=%d global_id=%d CONFIRMED_MATCH",
                         camera_id, det.track_id, binding.global_id)
            elif best_gid is not None:
                # свежий эмбеддинг лучше совпал с ДРУГОЙ identity — не
                # переключаем (ложное слияние опаснее), но логируем для анализа
                logger.warning(
                    "Camera %d: track=%d связан с global_id=%d, но свежий "
                    "эмбеддинг ближе к global_id=%d (%.2f) — событие "
                    "записано для анализа", camera_id, det.track_id,
                    binding.global_id, best_gid, best_sim)
                self._db_event(
                    binding.global_id, "ambiguous_match", camera_id, det.track_id,
                    {"bound_global_id": binding.global_id,
                     "better_global_id": best_gid, "better_similarity": round(best_sim, 3)},
                )
        # троттлинг: не чаще DB_SYNC_INTERVAL (кроме смены сотрудника —
        # иначе global_persons.employee_id в БД отстанет от реальности)
        if employee_changed or now - identity.last_db_sync >= DB_SYNC_INTERVAL:
            identity.last_db_sync = now
            self._db_update_person(identity)

    def _match_or_create(self, camera_id, det, embedding, now
                         ) -> tuple[GlobalIdentity, dict]:
        """Ищет лучшего кандидата среди активных identity; неуверенно — новая."""
        best_gid, best_sim, scores = self._best_candidate(camera_id, det, embedding, now)
        if best_gid is not None:
            with self._lock:
                identity = self._identities[best_gid]
                identity.last_seen = now
                identity.last_seen_wall = utcnow()
                # last_camera/last_track_id обновит вызывающий код ПОСЛЕ
                # записи события camera_transition (нужен from_camera)
                if det.employee_id:
                    identity.employee_id = det.employee_id
                self._push_embedding(identity, embedding, now, camera_id)
                identity.status = "ACTIVE"
            scores["matched"] = True
            return identity, scores

        scores["matched"] = False
        with self._lock:
            identity = GlobalIdentity(
                global_id=self._next_id, created_at=now, last_seen=now,
                last_seen_wall=utcnow(), last_camera=camera_id,
                last_track_id=det.track_id, employee_id=det.employee_id,
                status="NEW",
            )
            self._push_embedding(identity, embedding, now, camera_id)
            self._identities[identity.global_id] = identity
            self._next_id += 1
        logger.info("Global: новый global_id=%d (camera=%d track=%d)",
                    identity.global_id, camera_id, det.track_id)
        return identity, scores

    def _best_candidate(self, camera_id, det, embedding, now, exclude=None
                        ) -> tuple[Optional[int], float, dict]:
        """Composite matching по всем живым identity.

        Возвращает (global_id | None, similarity, детализация оценок).
        Veto: конфликт employee (лицо), невозможный переход по топологии,
        одновременное присутствие на одной камере.
        """
        ranked: list[tuple[float, float, int, dict]] = []
        with self._lock:
            identities = list(self._identities.values())
        for identity in identities:
            if identity.global_id == exclude:
                continue
            dt = now - identity.last_seen
            if dt > settings.global_gallery_ttl:
                continue

            # veto: распознанные лица разных сотрудников
            if (det.employee_id is not None and identity.employee_id is not None
                    and det.employee_id != identity.employee_id):
                _dbg("[MATCH] veto employee: track→gid=%d", identity.global_id)
                continue

            # veto: identity прямо сейчас на другом треке той же камеры
            occupied = any(
                key[0] == camera_id and b.global_id == identity.global_id
                and now - b.last_seen < SAME_CAMERA_OVERLAP
                for key, b in self._bindings.items()
            )
            if occupied:
                continue

            reid_sim = self._identity_similarity(identity, embedding)
            if reid_sim < settings.reid_similarity_threshold:
                continue  # кандидат по внешности не проходит — не рассматриваем

            ok, topology_score = (self.topology.check(identity.last_camera, camera_id, dt)
                                  if self.topology else (True, 0.5))
            if not ok:
                _dbg("[MATCH] veto topology: gid=%d cam %s→%d dt=%.0f",
                     identity.global_id, identity.last_camera, camera_id, dt)
                continue

            temporal_score = max(0.0, 1.0 - dt / settings.global_gallery_ttl)
            aspect_score = self._aspect_score(det, identity)
            face_score = self._face_score(det, identity)

            final = (
                settings.w_reid * reid_sim
                + settings.w_temporal * temporal_score
                + settings.w_topology * topology_score
                + settings.w_aspect * aspect_score
                + settings.w_face * face_score
            )
            scores = {
                "reid": round(float(reid_sim), 3),
                "temporal": round(temporal_score, 3),
                "topology": round(topology_score, 3),
                "aspect": round(aspect_score, 3),
                "face": round(face_score, 3),
                "final": round(final, 3),
                "gap_seconds": round(dt, 1),
            }
            _dbg("[MATCH] track=%d candidate_global_id=%d similarity=%.3f "
                 "temporal_score=%.3f topology_score=%.3f final_score=%.3f",
                 det.track_id, identity.global_id, reid_sim,
                 temporal_score, topology_score, final)
            ranked.append((final, reid_sim, identity.global_id, scores))

        if not ranked:
            return None, 0.0, {}
        ranked.sort(key=lambda r: -r[0])
        final, sim, gid, scores = ranked[0]
        # неоднозначность: два кандидата с близкими оценками — не сливаем (ТЗ §10)
        if len(ranked) > 1 and (final - ranked[1][0]) < settings.global_match_margin:
            logger.info(
                "Global: неоднозначный матч (gid=%d %.3f vs gid=%d %.3f) — "
                "создаётся новая identity", gid, final, ranked[1][2], ranked[1][0])
            return None, sim, {**scores, "ambiguous_with": ranked[1][2]}
        if final < settings.global_match_threshold:
            return None, sim, scores
        return gid, sim, scores

    # ------------------------------------------------------------ внутреннее

    @staticmethod
    def _identity_similarity(identity: GlobalIdentity, embedding: np.ndarray) -> float:
        """Максимальный косинус с историей эмбеддингов identity."""
        if not identity.embeddings:
            return 0.0
        matrix = np.stack([e for e, _ts, _cam in identity.embeddings])
        sims = matrix @ embedding
        return float(np.max(sims))

    def _push_embedding(self, identity: GlobalIdentity, embedding: np.ndarray,
                        now: float, camera_id: int) -> None:
        identity.embeddings.append((embedding, now, camera_id))
        while len(identity.embeddings) > settings.global_history_len:
            identity.embeddings.popleft()

    @staticmethod
    def _face_score(det, identity: GlobalIdentity) -> float:
        """Лицо — дополнительный сигнал: один и тот же сотрудник → 1.0,
        нет данных → нейтрально 0.5 (конфликт отсеивается veto выше)."""
        if det.employee_id is not None and identity.employee_id == det.employee_id:
            return 1.0
        return 0.5

    @staticmethod
    def _aspect_score(det, identity: GlobalIdentity) -> float:
        """Пропорции фигуры (h/w) — слабый сигнал, вес маленький."""
        if identity.aspect <= 0:
            return 0.5
        x1, y1, x2, y2 = det.bbox
        aspect = (y2 - y1) / max(1e-6, x2 - x1)
        return max(0.0, 1.0 - abs(aspect - identity.aspect) / max(aspect, identity.aspect))

    def _update_aspect(self, identity: GlobalIdentity, det) -> None:
        x1, y1, x2, y2 = det.bbox
        aspect = (y2 - y1) / max(1e-6, x2 - x1)
        identity.aspect = aspect if identity.aspect <= 0 else \
            0.8 * identity.aspect + 0.2 * aspect

    @staticmethod
    def _crop(frame: np.ndarray, bbox, frame_w: int, frame_h: int
              ) -> Optional[np.ndarray]:
        """Кроп человека по нормализованному bbox с запасом 10%."""
        x1, y1, x2, y2 = bbox
        x1 = max(0, int(x1 * frame_w)); y1 = max(0, int(y1 * frame_h))
        x2 = min(frame_w, int(x2 * frame_w)); y2 = min(frame_h, int(y2 * frame_h))
        mx = max(1, int((x2 - x1) * 0.1)); my = max(1, int((y2 - y1) * 0.1))
        crop = frame[max(0, y1 - my):y2 + my, max(0, x1 - mx):x2 + mx]
        if crop.shape[0] < MIN_CROP_HEIGHT or crop.shape[1] < 24:
            return None
        return crop

    @staticmethod
    def _snapshot_jpeg(crop: np.ndarray) -> bytes:
        img = crop
        if img.shape[0] > SNAPSHOT_MAX_HEIGHT:
            scale = SNAPSHOT_MAX_HEIGHT / img.shape[0]
            img = cv2.resize(img, (int(img.shape[1] * scale), SNAPSHOT_MAX_HEIGHT))
        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        return buf.tobytes() if ok else b""

    # ------------------------------------------------------------------ БД

    def _persist_observation(self, identity: GlobalIdentity, camera_id, det,
                             embedding: np.ndarray, snapshot: bytes) -> None:
        try:
            with self._session_factory() as db:
                db.add(GlobalObservation(
                    global_id=identity.global_id, camera_id=camera_id,
                    track_id=det.track_id,
                    embedding=np.asarray(embedding, dtype=np.float32).tobytes(),
                    snapshot=snapshot or None,
                ))
                db.commit()
        except Exception:
            logger.exception("Global: не удалось записать наблюдение")

    def _record_event(self, identity: GlobalIdentity, camera_id, det,
                      scores: dict, snapshot: bytes = b"",
                      previous_camera=None, previous_track=None) -> None:
        """person_seen + camera_transition при смене камеры + уведомления TG."""
        payload = dict(scores)
        payload["similarity"] = scores.get("reid")
        payload["confidence"] = scores.get("final")
        self._db_event(identity.global_id, "person_seen", camera_id,
                       det.track_id, payload)
        if not scores.get("matched") and settings.telegram_notify_global_new:
            self._notify_new(identity, camera_id, snapshot)
        if previous_camera is not None and previous_camera != camera_id:
            self._db_event(
                identity.global_id, "camera_transition", camera_id, det.track_id,
                {
                    "from_camera": previous_camera,
                    "to_camera": camera_id,
                    "from_track_id": previous_track
                    if previous_track != det.track_id else None,
                    "to_track_id": det.track_id,
                    "similarity": scores.get("reid"),
                    "confidence": scores.get("final"),
                },
            )
            logger.info(
                "Global: global_id=%d перешёл camera %d → %d (track=%d, sim=%s)",
                identity.global_id, previous_camera, camera_id, det.track_id,
                scores.get("reid"),
            )
            self._notify_transition(identity, previous_camera, camera_id,
                                    scores, snapshot)

    # ------------------------------------------------------------ Telegram

    def _notify_transition(self, identity: GlobalIdentity, from_cam: int,
                           to_cam: int, scores: dict, snapshot: bytes) -> None:
        """Фото + подпись о переходе человека между камерами."""
        if self.notifier is None or not settings.telegram_notify_global_transition:
            return
        try:
            with self._session_factory() as db:
                from_name = self._camera_name(db, from_cam)
                to_name = self._camera_name(db, to_cam)
                who = self._person_title(db, identity)
            caption = (
                f"🔄 {who}: «{from_name}» → «{to_name}»\n"
                f"Время: {_local_time()}"
                + (f"\nСходство: {scores.get('reid')}" if scores.get("reid") else "")
            )
            if snapshot:
                self.notifier.send_photo(snapshot, caption)
            else:
                self.notifier.send_text(caption)
        except Exception:
            logger.exception("Global: не удалось отправить уведомление о переходе")

    def _notify_new(self, identity: GlobalIdentity, camera_id: int,
                    snapshot: bytes) -> None:
        """Уведомление о новой глобальной личности (по умолчанию выключено)."""
        if self.notifier is None:
            return
        try:
            with self._session_factory() as db:
                who = self._person_title(db, identity)
                cam = self._camera_name(db, camera_id)
            caption = f"👤 Новый человек {who} — камера «{cam}»\nВремя: {_local_time()}"
            if snapshot:
                self.notifier.send_photo(snapshot, caption)
            else:
                self.notifier.send_text(caption)
        except Exception:
            logger.exception("Global: не удалось отправить уведомление о новой личности")

    @staticmethod
    def _camera_name(db, camera_id: int) -> str:
        camera = db.get(Camera, camera_id)
        return camera.name if camera else f"камера #{camera_id}"

    @staticmethod
    def _person_title(db, identity: GlobalIdentity) -> str:
        """G#184 или G#184 (Иван Петров), если личность связана со сотрудником."""
        if identity.employee_id:
            employee = db.get(Employee, identity.employee_id)
            if employee:
                return f"G#{identity.global_id} ({employee.name})"
        return f"G#{identity.global_id}"

    def _db_event(self, global_id: int, event_type: str, camera_id, track_id,
                  payload: dict) -> None:
        try:
            with self._session_factory() as db:
                db.add(GlobalEvent(
                    global_id=global_id, event_type=event_type,
                    camera_id=camera_id, track_id=track_id,
                    payload=json.dumps(payload, ensure_ascii=False)[:2000],
                ))
                db.commit()
        except Exception:
            logger.exception("Global: не удалось записать событие %s", event_type)

    def _db_update_person(self, identity: GlobalIdentity) -> None:
        try:
            with self._session_factory() as db:
                person = db.get(GlobalPerson, identity.global_id)
                if person is None:
                    db.add(GlobalPerson(
                        id=identity.global_id, status=identity.status,
                        employee_id=identity.employee_id,
                        last_camera_id=identity.last_camera,
                        last_track_id=identity.last_track_id,
                        created_at=identity.last_seen_wall,
                        last_seen_at=identity.last_seen_wall,
                    ))
                else:
                    person.status = identity.status
                    person.employee_id = identity.employee_id
                    person.last_camera_id = identity.last_camera
                    person.last_track_id = identity.last_track_id
                    person.last_seen_at = identity.last_seen_wall
                db.commit()
        except Exception:
            logger.exception("Global: не удалось обновить global_person")

