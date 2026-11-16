"""Pure appearance proposals, published only after a durable journal acknowledgement.

The worker owns inference and token freshness; HumanJournal owns persistence and
integer ID allocation. Only acknowledged, process-local vectors enter this
gallery. Historical archive embeddings are deliberately never loaded.
"""
import json
import logging
import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from numbers import Real
from pathlib import Path
from typing import Optional
from uuid import NAMESPACE_URL, UUID, uuid5

import numpy as np

from app.config import BASE_DIR, settings

logger = logging.getLogger("app.ai.global_tracker")

SAME_CAMERA_OVERLAP = 2.0
MAX_ACTIVE_VISITS = 4096
MAX_IDENTITIES = 4096
LINK_NAMESPACE = uuid5(NAMESPACE_URL, "na_meste/human-journal/link/v1")



def _local_time() -> str:
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


@dataclass(frozen=True, slots=True)
class EmbeddingItem:
    visit_id: str
    camera_id: int
    tracking_generation: str
    track_id: int
    source_revision: int
    stream_id: str
    packet_seq: int
    observed_at: datetime
    received_mono: float
    bbox: tuple[float, float, float, float]
    aspect: float
    crop_quality: float
    embedding: np.ndarray
    employee_id: int | None


@dataclass(frozen=True, slots=True)
class ProposedLink(EmbeddingItem):
    link_key: str
    global_id: int | None
    score: float


@dataclass(slots=True)
class GlobalIdentity:
    global_id: int
    last_seen: float
    last_camera: int
    last_track_id: int
    employee_id: int | None
    status: str = "NEW"
    aspect: float = 0.
    embeddings: deque = field(default_factory=deque)


@dataclass(slots=True)
class TrackBinding:
    global_id: int
    proposal: ProposedLink
    confirmations: int = 1


class GlobalIdentityManager:
    def __init__(self, topology: Optional[CameraTopology] = None, notifier=None):
        self.topology = topology
        self.notifier = notifier
        self._identities: dict[int, GlobalIdentity] = {}
        self._bindings: dict[str, TrackBinding] = {}
        self._pending: dict[str, ProposedLink] = {}
        self._lock = threading.RLock()

    def process_embeddings(self, items: list[EmbeddingItem]) -> list[ProposedLink]:
        """Retain at most one uncommitted proposal per visit, without publishing it.

        Repeating a packet returns its retained receipt. A newer packet supersedes
        the pending token; acknowledgement of the older token then cannot publish.
        Invalid batches fail before admission rather than partially consuming them.
        """
        for item in items:
            self._validate_item(item)
        if not items:
            return []
        proposals = []
        with self._lock:
            self.sweep(max(item.received_mono for item in items))
            for item in items:
                pending = self._pending.get(item.visit_id)
                binding = self._bindings.get(item.visit_id)
                previous = pending if pending is not None else (
                    binding.proposal if binding is not None else None)
                if previous is not None:
                    if not self._same_visit_token(item, previous):
                        continue
                    if item.packet_seq < previous.packet_seq or item.received_mono < previous.received_mono:
                        continue
                    if item.packet_seq == previous.packet_seq:
                        if pending is not None:
                            proposals.append(pending)
                        continue
                elif len(self._pending) + len(self._bindings) >= MAX_ACTIVE_VISITS:
                    continue

                if binding is not None:
                    identity = self._identities.get(binding.global_id)
                    if identity is None or self._employee_conflict(item, identity):
                        continue
                    similarity = self._identity_similarity(identity, item.embedding)
                    if similarity < settings.reid_similarity_threshold:
                        continue
                    global_id = binding.global_id
                    score = self._composite(item, identity, similarity, .5)
                else:
                    global_id, score = self._best_candidate(item)
                # The caller may reuse its embedding buffer for the next batch.
                # This is the one owned copy retained until commit/expiry.
                embedding = np.array(item.embedding, dtype=np.float32, copy=True)
                embedding.setflags(write=False)
                proposal = ProposedLink(
                    visit_id=item.visit_id, camera_id=item.camera_id,
                    tracking_generation=item.tracking_generation, track_id=item.track_id,
                    source_revision=item.source_revision, stream_id=item.stream_id,
                    packet_seq=item.packet_seq, observed_at=item.observed_at,
                    received_mono=item.received_mono, bbox=tuple(item.bbox),
                    aspect=item.aspect, crop_quality=item.crop_quality,
                    embedding=embedding, employee_id=item.employee_id,
                    link_key=str(uuid5(LINK_NAMESPACE, json.dumps(
                        [item.visit_id, item.source_revision, item.stream_id, item.packet_seq],
                        separators=(",", ":"), ensure_ascii=True))),
                    global_id=global_id, score=score,
                )
                self._pending[item.visit_id] = proposal
                proposals.append(proposal)
        return proposals

    def acknowledge(self, ack) -> None:
        """Accept only an exact LINK_COMMITTED token from the journal writer."""
        caption = None
        with self._lock:
            if getattr(ack, "kind", None) != "LINK_COMMITTED":
                return
            visit_id = getattr(ack, "visit_id", None)
            if not isinstance(visit_id, str):
                return
            proposal = self._pending.get(visit_id)
            if proposal is None:
                return
            for name in ("camera_id", "tracking_generation", "visit_id", "track_id",
                         "source_revision", "stream_id", "packet_seq", "link_key"):
                value = getattr(ack, name, None)
                expected = getattr(proposal, name)
                if type(value) is not type(expected) or value != expected:
                    return
            global_id = getattr(ack, "global_id", None)
            if type(global_id) is not int or global_id <= 0:
                return
            binding = self._bindings.get(visit_id)
            if binding is not None and binding.global_id != global_id:
                return  # An already acknowledged visit is never silently relinked.
            identity = self._identities.get(global_id)
            if identity is not None and self._employee_conflict(proposal, identity):
                return

            self._pending.pop(visit_id)
            previous_camera = identity.last_camera if identity is not None else None
            newest_observation = identity is None or proposal.received_mono >= identity.last_seen
            is_new = identity is None
            if identity is None:
                self._make_identity_space()
                identity = GlobalIdentity(
                    global_id=global_id, last_seen=proposal.received_mono,
                    last_camera=proposal.camera_id, last_track_id=proposal.track_id,
                    employee_id=proposal.employee_id,
                )
                self._identities[global_id] = identity
            else:
                identity.status = "ACTIVE"
            identity.embeddings.append(proposal.embedding)
            while len(identity.embeddings) > max(1, settings.global_history_len):
                identity.embeddings.popleft()
            if newest_observation:
                identity.last_seen = proposal.received_mono
                identity.last_camera = proposal.camera_id
                identity.last_track_id = proposal.track_id
            identity.aspect = proposal.aspect if identity.aspect <= 0 else (
                .8 * identity.aspect + .2 * proposal.aspect)
            if proposal.employee_id is not None:
                identity.employee_id = proposal.employee_id  # Face context, never a body guess.
            self._bindings[visit_id] = TrackBinding(
                global_id, proposal, binding.confirmations + 1 if binding is not None else 1)
            if self.notifier is not None:
                if is_new and settings.telegram_notify_global_new:
                    caption = (f"👤 Новый человек G#{global_id} — камера #{proposal.camera_id}"
                               f"\nВремя: {_local_time()}")
                elif (newest_observation and previous_camera is not None
                      and previous_camera != proposal.camera_id
                      and settings.telegram_notify_global_transition):
                    caption = (f"🔄 G#{global_id}: камера #{previous_camera} → "
                               f"камера #{proposal.camera_id}\nВремя: {_local_time()}"
                               f"\nОценка связывания: {proposal.score:.3f}")
        # No DB/name/photo lookup belongs in this pure matcher. Preserve optional
        # text notifications with acknowledged IDs; a replay cannot notify twice.
        if caption is not None:
            try:
                self.notifier.send_text(caption)
            except Exception:
                logger.exception("Global: не удалось отправить уведомление")

    def global_id_for(self, visit_id: str) -> int | None:
        with self._lock:
            binding = self._bindings.get(visit_id)
            return binding.global_id if binding is not None else None

    def drop_camera(self, camera_id: int) -> None:
        """Forget all local generations, not acknowledged cross-camera history."""
        with self._lock:
            for visit_id in [key for key, p in self._pending.items() if p.camera_id == camera_id]:
                self._pending.pop(visit_id)
            for visit_id in [key for key, b in self._bindings.items()
                             if b.proposal.camera_id == camera_id]:
                self._bindings.pop(visit_id)

    def sweep(self, now: float) -> None:
        """Bound pending/local lifetime and expire only the process-local gallery."""
        with self._lock:
            local_ttl = settings.track_lost_timeout + 5.
            for visit_id in [key for key, p in self._pending.items()
                             if now - p.received_mono > local_ttl]:
                self._pending.pop(visit_id)
            for visit_id in [key for key, b in self._bindings.items()
                             if now - b.proposal.received_mono > local_ttl]:
                self._bindings.pop(visit_id)
            expired = [key for key, identity in self._identities.items()
                       if now - identity.last_seen > settings.global_gallery_ttl]
            for global_id in expired:
                self._remove_identity(global_id)

    def camera_tracks(self, camera_id: int) -> list[dict]:
        now = time.monotonic()
        with self._lock:
            return [
                {"visit_id": visit_id, "track_id": binding.proposal.track_id,
                 "tracking_generation": binding.proposal.tracking_generation,
                 "camera_id": camera_id, "global_id": binding.global_id,
                 "confirmed": binding.confirmations >= 2,
                 "seconds_since_seen": round(max(0., now - binding.proposal.received_mono), 1)}
                for visit_id, binding in self._bindings.items()
                if binding.proposal.camera_id == camera_id
            ]

    def snapshot(self) -> list[dict]:
        now = time.monotonic()
        with self._lock:
            return [
                {"global_id": identity.global_id, "status": identity.status,
                 "employee_id": identity.employee_id, "last_camera": identity.last_camera,
                 "last_track_id": identity.last_track_id,
                 "seconds_since_seen": round(max(0., now - identity.last_seen), 1),
                 "embeddings": len(identity.embeddings)}
                for identity in self._identities.values()
            ]

    def _best_candidate(self, item: EmbeddingItem) -> tuple[int | None, float]:
        ranked = []
        occupied = {
            binding.global_id for visit_id, binding in self._bindings.items()
            if visit_id != item.visit_id and binding.proposal.camera_id == item.camera_id
            and item.received_mono - binding.proposal.received_mono < SAME_CAMERA_OVERLAP
        }
        # Pending vectors are not gallery samples; only their requested target
        # reserves same-camera occupancy until commit, supersession or expiry.
        occupied.update(
            pending.global_id for visit_id, pending in self._pending.items()
            if visit_id != item.visit_id and pending.camera_id == item.camera_id
            and pending.global_id is not None
            and item.received_mono - pending.received_mono < SAME_CAMERA_OVERLAP)
        for identity in self._identities.values():
            dt = item.received_mono - identity.last_seen
            if dt < 0 or dt > settings.global_gallery_ttl or self._employee_conflict(item, identity):
                continue
            if identity.global_id in occupied:
                continue
            similarity = self._identity_similarity(identity, item.embedding)
            if similarity < settings.reid_similarity_threshold:
                continue
            allowed, topology_score = (
                self.topology.check(identity.last_camera, item.camera_id, dt)
                if self.topology is not None else (True, .5))
            if allowed:
                ranked.append((self._composite(item, identity, similarity, topology_score),
                               identity.global_id))
        if not ranked:
            return None, 0.
        ranked.sort(key=lambda candidate: -candidate[0])
        score, global_id = ranked[0]
        if (len(ranked) > 1 and score - ranked[1][0] < settings.global_match_margin
                or score < settings.global_match_threshold):
            return None, score
        return global_id, score

    @staticmethod
    def _identity_similarity(identity: GlobalIdentity, embedding: np.ndarray) -> float:
        return max((float(np.dot(sample, embedding)) for sample in identity.embeddings), default=0.)

    @staticmethod
    def _employee_conflict(item: EmbeddingItem, identity: GlobalIdentity) -> bool:
        return (item.employee_id is not None and identity.employee_id is not None
                and item.employee_id != identity.employee_id)

    @staticmethod
    def _composite(item: EmbeddingItem, identity: GlobalIdentity,
                   similarity: float, topology_score: float) -> float:
        dt = max(0., item.received_mono - identity.last_seen)
        temporal = max(0., 1. - dt / settings.global_gallery_ttl)
        aspect = (.5 if identity.aspect <= 0 else
                  max(0., 1. - abs(item.aspect - identity.aspect) / max(item.aspect, identity.aspect)))
        face = 1. if item.employee_id is not None and item.employee_id == identity.employee_id else .5
        return float(settings.w_reid * similarity + settings.w_temporal * temporal
                     + settings.w_topology * topology_score + settings.w_aspect * aspect
                     + settings.w_face * face)

    @staticmethod
    def _same_visit_token(item: EmbeddingItem, previous: ProposedLink) -> bool:
        return all(getattr(item, field) == getattr(previous, field)
                   for field in ("camera_id", "tracking_generation", "track_id",
                                 "source_revision", "stream_id"))

    def _make_identity_space(self) -> None:
        if len(self._identities) >= MAX_IDENTITIES:
            oldest = min(self._identities, key=lambda key: self._identities[key].last_seen)
            self._remove_identity(oldest)

    def _remove_identity(self, global_id: int) -> None:
        self._identities.pop(global_id, None)
        for visit_id in [key for key, binding in self._bindings.items() if binding.global_id == global_id]:
            self._bindings.pop(visit_id)

    @staticmethod
    def _validate_item(item: EmbeddingItem) -> None:
        if not isinstance(item, EmbeddingItem):
            raise ValueError("Expected EmbeddingItem")
        for name in ("visit_id", "tracking_generation"):
            value = getattr(item, name)
            try:
                valid = isinstance(value, str) and str(UUID(value)) == value
            except ValueError:
                valid = False
            if not valid:
                raise ValueError(f"Invalid {name}")
        for name in ("camera_id", "track_id", "source_revision", "packet_seq", "employee_id"):
            value = getattr(item, name)
            if name == "employee_id" and value is None:
                continue
            if type(value) is not int or value < (0 if name == "packet_seq" else 1):
                raise ValueError(f"Invalid {name}")
        if (not isinstance(item.stream_id, str) or not 1 <= len(item.stream_id) <= 128
                or any(ord(character) < 33 for character in item.stream_id)):
            raise ValueError("Invalid stream_id")
        if (not isinstance(item.observed_at, datetime)
                or item.observed_at.utcoffset() not in (None, timedelta(0))):
            raise ValueError("observed_at must be UTC")
        for name in ("received_mono", "aspect", "crop_quality"):
            value = getattr(item, name)
            if (isinstance(value, bool) or not isinstance(value, Real)
                    or not math.isfinite(value) or value < 0
                    or name == "aspect" and value == 0):
                raise ValueError(f"Invalid {name}")
        if not isinstance(item.bbox, tuple) or len(item.bbox) != 4:
            raise ValueError("Invalid normalized bbox")
        if any(isinstance(value, bool) or not isinstance(value, Real)
               or not math.isfinite(value) or not 0 <= value <= 1 for value in item.bbox):
            raise ValueError("Invalid normalized bbox")
        x1, y1, x2, y2 = item.bbox
        if x1 >= x2 or y1 >= y2:
            raise ValueError("Invalid normalized bbox")
        embedding = item.embedding
        if (not isinstance(embedding, np.ndarray) or embedding.shape != (512,)
                or embedding.dtype.kind != "f" or not np.isfinite(embedding).all()
                or not math.isclose(float(np.linalg.norm(embedding)), 1., rel_tol=0., abs_tol=1e-3)):
            raise ValueError("Expected a finite unit 512-vector")
