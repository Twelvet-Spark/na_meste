"""Durable operator-confirmed BODY evidence: immutable snapshots and conservative ranking.

Only explicit operator decisions create rows here; a predicted identity never teaches. A snapshot is
immutable and built once per refresh (contiguous read-only matrices grouped by owner), so the live
matcher performs a single matrix product per bank and no per-match gallery copies.
"""
import hashlib
import logging
import re
import threading
from dataclasses import dataclass
from datetime import date, datetime, timezone
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np
from sqlalchemy.orm import Session

from app.ai.face_recognition import (
    EMBEDDING_SIZE, RankedEmployee, is_normalized_embedding, normalized_embedding, stored_embedding,
)
from app.config import settings
from app.database.database import read_snapshot
from app.database.models import (
    AttendanceRuntime, Employee, RecognitionExample, RecognitionFeedback, RecognitionLearningRuntime,
)
from app.time_utils import local_date

logger = logging.getLogger("app.ai.recognition_learning")
TOP_K = 3
NO_DAY = -1


def _readonly(array: np.ndarray) -> np.ndarray:
    array.flags.writeable = False
    return array


def _empty_matrix() -> np.ndarray:
    return _readonly(np.empty((0, EMBEDDING_SIZE), np.float32))


def _groups(owners: Sequence[int]) -> tuple[tuple[int, int, int], ...]:
    """Contiguous (employee_id, start, stop) runs of an owner-sorted bank."""
    groups: list[tuple[int, int, int]] = []
    for index, owner in enumerate(owners):
        if groups and groups[-1][0] == owner:
            groups[-1] = (owner, groups[-1][1], index + 1)
        else:
            groups.append((owner, index, index + 1))
    return tuple(groups)


def _key(vector: np.ndarray) -> bytes:
    return hashlib.blake2b(np.ascontiguousarray(vector, dtype=np.float32).tobytes(), digest_size=16).digest()


def _day_ordinal(value: datetime | None) -> int:
    if value is None:
        return NO_DAY
    if value.tzinfo is not None:
        value = value.astimezone(timezone.utc).replace(tzinfo=None)
    return local_date(value).toordinal()


@dataclass(frozen=True, slots=True, eq=False)
class LearningSnapshot:
    """Atomic view of active, model-compatible BODY evidence.

    Access pattern for consumers: hold the object returned by `store.snapshot()` for the whole
    decision; never mix fields of two snapshots. Every array is read-only and shared.

    - `revision`/`gallery_revision`: learning-runtime and face-gallery revisions read in the same
      SQLite snapshot as the rows. `loaded` is False until the first successful refresh (such a
      snapshot is empty and must disable body naming).
    - `names`: active employee id -> display name (inactive employees are absent).
    - positive bank: `positive_matrix` (n x 512 float32, unit rows, grouped by owner),
      `positive_owners[i]`, `positive_day_rows` (workplace-day ordinal -> sorted read-only int array of
      the positive row indices with explicit feedback on that day: one stored int per unique dated
      anchor, no dense per-day masks; a row with only unknown capture times is in none:
      suggestion-only), `positive_keys[i]` (vector digest),
      `positive_groups` (employee_id, start, stop). Exact duplicate vectors per owner are collapsed
      to one row that keeps all of its feedback days.
    - negative bank: the same layout without days/keys.
    - `incompatible_rows` (other model) and `invalid_rows` (malformed bytes) are counted, never used.
    """
    revision: int
    gallery_revision: int
    body_model_id: str
    loaded: bool
    names: Mapping[int, str]
    positive_matrix: np.ndarray
    positive_owners: np.ndarray
    positive_day_rows: Mapping[int, np.ndarray]
    positive_keys: tuple[bytes, ...]
    positive_groups: tuple[tuple[int, int, int], ...]
    negative_matrix: np.ndarray
    negative_owners: np.ndarray
    negative_groups: tuple[tuple[int, int, int], ...]
    incompatible_rows: int = 0
    invalid_rows: int = 0

    @staticmethod
    def empty(body_model_id: str) -> "LearningSnapshot":
        return LearningSnapshot(
            0, 0, body_model_id, False, MappingProxyType({}), _empty_matrix(),
            _readonly(np.empty(0, np.intp)), MappingProxyType({}), (), (),
            _empty_matrix(), _readonly(np.empty(0, np.intp)), (),
        )

    def positive_count(self, employee_id: int) -> int:
        return sum(stop - start for owner, start, stop in self.positive_groups if owner == employee_id)

    def negative_count(self, employee_id: int) -> int:
        return sum(stop - start for owner, start, stop in self.negative_groups if owner == employee_id)


class RecognitionLearningStore:
    def __init__(self, engine, body_model_id: str):
        if not isinstance(body_model_id, str) or re.fullmatch(r"[0-9a-f]{64}", body_model_id) is None:
            raise ValueError("body_model_id must be a declared SHA256")
        self._engine = engine
        self.body_model_id = body_model_id
        self._lock = threading.Lock()
        self._refresh_lock = threading.Lock()
        self._snapshot = LearningSnapshot.empty(body_model_id)

    def snapshot(self) -> LearningSnapshot:
        with self._lock:
            return self._snapshot

    def refresh(self) -> LearningSnapshot:
        """Rebuild from one SQLite read snapshot; refreshes serialize so an older read never wins.

        A failed refresh raises and leaves the previous snapshot published unchanged."""
        with self._refresh_lock:
            rows, employees, revision, gallery_revision = read_snapshot(self._engine, self._read)
            snapshot = self._build(rows, employees, revision, gallery_revision)
            with self._lock:
                self._snapshot = snapshot
        logger.info("Образцы внешности перезагружены: %d положительных, %d отрицательных, revision=%d",
                    len(snapshot.positive_owners), len(snapshot.negative_owners), snapshot.revision)
        return snapshot

    @staticmethod
    def _read(db: Session):
        revision = db.query(RecognitionLearningRuntime.revision).filter_by(id=1).one()[0]
        gallery_revision = db.query(AttendanceRuntime.gallery_revision).filter_by(id=1).one()[0]
        employees = db.query(Employee.id, Employee.name).filter(Employee.active.is_(True)).all()
        rows = db.query(
            Employee.id, RecognitionExample.polarity, RecognitionExample.model_id,
            RecognitionExample.embedding, RecognitionFeedback.captured_at,
        ).select_from(RecognitionExample).join(
            RecognitionFeedback, RecognitionFeedback.id == RecognitionExample.feedback_id,
        ).join(Employee, Employee.id == RecognitionFeedback.employee_id).filter(
            Employee.active.is_(True), RecognitionFeedback.revoked_at.is_(None),
            RecognitionExample.modality == "BODY",
        ).order_by(Employee.id, RecognitionExample.id).all()
        return rows, employees, int(revision), int(gallery_revision)

    def _build(self, rows, employees, revision: int, gallery_revision: int) -> LearningSnapshot:
        positives: dict[tuple[int, bytes], list] = {}
        negatives: dict[tuple[int, bytes], np.ndarray] = {}
        incompatible = invalid = 0
        for owner, polarity, model_id, blob, captured_at in rows:
            if model_id != self.body_model_id:
                incompatible += 1
                continue
            vector = stored_embedding(blob)
            if vector is None:
                invalid += 1
                continue
            key = (owner, _key(vector))
            if polarity == "NEGATIVE":
                negatives.setdefault(key, vector)
                continue
            day = _day_ordinal(captured_at)
            held = positives.setdefault(key, [vector, set()])
            if day != NO_DAY:
                held[1].add(day)

        def bank(items: dict, vectors: list[np.ndarray]):
            owners = [key[0] for key in items]
            matrix = _readonly(np.stack(vectors)) if vectors else _empty_matrix()
            return matrix, _readonly(np.asarray(owners, dtype=np.intp)), _groups(owners)

        pos_matrix, pos_owners, pos_groups = bank(positives, [v[0] for v in positives.values()])
        neg_matrix, neg_owners, neg_groups = bank(negatives, list(negatives.values()))
        by_day: dict[int, list[int]] = {}
        for row, (_vector, days) in enumerate(positives.values()):
            for ordinal in days:
                by_day.setdefault(ordinal, []).append(row)
        day_rows = {ordinal: _readonly(np.asarray(rows, dtype=np.intp)) for ordinal, rows in by_day.items()}
        return LearningSnapshot(
            revision, gallery_revision, self.body_model_id, True,
            MappingProxyType({employee_id: name for employee_id, name in employees}),
            pos_matrix, pos_owners,
            MappingProxyType(day_rows),
            tuple(key[1] for key in positives), pos_groups, neg_matrix, neg_owners, neg_groups,
            incompatible, invalid,
        )


def _query_day(observed_at: datetime) -> date:
    if observed_at.tzinfo is not None:
        observed_at = observed_at.astimezone(timezone.utc).replace(tzinfo=None)
    return local_date(observed_at)


def _top_mean(values: np.ndarray) -> float | None:
    count = len(values)
    if count == 0:
        return None
    if count > TOP_K:
        values = np.partition(values, count - TOP_K)[count - TOP_K:]
    return float(np.mean(values, dtype=np.float64))


def _transient_vectors(samples: Sequence) -> list[np.ndarray]:
    vectors: dict[bytes, np.ndarray] = {}
    for sample in samples:
        array = np.asarray(sample)
        if not (array.dtype == np.float32 and array.shape == (EMBEDDING_SIZE,)
                and is_normalized_embedding(array)):
            array = normalized_embedding(array)
            if array is None:
                continue
        vectors.setdefault(_key(array), array)
    return list(vectors.values())


def _vetoed(positive: float, negative: float | None) -> bool:
    return (negative is not None and negative >= settings.reid_employee_threshold
            and negative >= positive - settings.reid_employee_margin)


@dataclass(frozen=True, slots=True)
class _Evaluation:
    suggestion: dict[int, float]  # all compatible durable positives + transient
    same_day: dict[int, float]  # durable positives captured on the query day + transient
    negative: dict[int, float]


def _evaluate(snapshot: LearningSnapshot, embedding: np.ndarray, observed_at: datetime,
              transient_samples: Mapping[int, Sequence[np.ndarray]] | None,
              *, suggestions: bool) -> _Evaluation | None:
    """One matrix product per bank; suggestion and automatic scores derive from the same dots."""
    vector = normalized_embedding(embedding)
    if vector is None or not snapshot.loaded:
        return None
    day = _query_day(observed_at).toordinal()
    names = snapshot.names
    dots = (snapshot.positive_matrix @ vector) if len(snapshot.positive_owners) else np.empty(0, np.float32)
    transient = {
        owner: _transient_vectors(samples) for owner, samples in (transient_samples or {}).items()
        if owner in names
    }
    day_mask = np.zeros(len(snapshot.positive_owners), bool)
    rows = snapshot.positive_day_rows.get(day)
    if rows is not None:
        day_mask[rows] = True
    durable = {owner: (start, stop) for owner, start, stop in snapshot.positive_groups}
    suggestion: dict[int, float] = {}
    same_day: dict[int, float] = {}
    for owner in sorted(durable.keys() | transient.keys()):
        start, stop = durable.get(owner, (0, 0))
        owner_dots = dots[start:stop]
        mask = day_mask[start:stop]
        day_dots = owner_dots[mask]
        extra_all: list[float] = []
        extra_day: list[float] = []
        if transient.get(owner):
            all_keys = set(snapshot.positive_keys[start:stop])
            day_keys = {key for key, hit in zip(snapshot.positive_keys[start:stop], mask) if hit}
            for sample in transient[owner]:
                value = float(sample @ vector)
                key = _key(sample)
                if key not in all_keys:
                    extra_all.append(value)
                if key not in day_keys:
                    extra_day.append(value)
        day_score = _top_mean(np.concatenate((day_dots.astype(np.float64), extra_day)))
        if day_score is not None:
            same_day[owner] = day_score
        if suggestions:
            score = _top_mean(np.concatenate((owner_dots.astype(np.float64), extra_all)))
            if score is not None:
                suggestion[owner] = score
    negative: dict[int, float] = {}
    if len(snapshot.negative_owners):
        negative_dots = snapshot.negative_matrix @ vector
        for owner, start, stop in snapshot.negative_groups:
            negative[owner] = float(negative_dots[start:stop].max())
    return _Evaluation(suggestion, same_day, negative)


def _rows(scores: Mapping[int, float], negative: Mapping[int, float]) -> tuple[RankedEmployee, ...]:
    rows = []
    for owner, score in scores.items():
        n = negative.get(owner)
        rows.append(RankedEmployee(owner, score, n, _vetoed(score, n)))
    rows.sort(key=lambda row: (-row.score, row.employee_id))
    return tuple(rows)


def rank_body(snapshot: LearningSnapshot, embedding: np.ndarray, *, observed_at: datetime,
              automatic: bool,
              transient_samples: Mapping[int, Sequence[np.ndarray]] | None = None,
              ) -> tuple[RankedEmployee, ...]:
    """Every distinct active owner with usable evidence, best first (ties by employee id).

    automatic=False: all compatible durable positives (any date) plus the supplied transient
    same-day samples. automatic=True: only owners with same-day durable positives or transient
    samples, scored from that same-day evidence alone; all such raw rows are returned, vetoed rows
    flagged rather than dropped, and eligibility (threshold/margin) is left to `match_body`.
    `transient_samples` must already be restricted to the query's local day
    (`EmployeeAppearanceGallery.samples_for_day`)."""
    evaluation = _evaluate(snapshot, embedding, observed_at, transient_samples, suggestions=not automatic)
    if evaluation is None:
        return ()
    return _rows(evaluation.same_day if automatic else evaluation.suggestion, evaluation.negative)


def match_body(snapshot: LearningSnapshot, embedding: np.ndarray, *, observed_at: datetime,
               transient_samples: Mapping[int, Sequence[np.ndarray]] | None = None,
               ) -> RankedEmployee | None:
    """The one employee automatic BODY recognition may name, else None.

    The winner is the best same-day-anchored owner. It must reach reid_employee_threshold, not be
    locally vetoed by a negative example, and beat every other owner's raw score (same-day or
    older suggestion evidence, vetoed or not) by reid_employee_margin. Older or vetoed competitors
    can block a name but never win one; a vetoed winner never promotes the runner-up."""
    evaluation = _evaluate(snapshot, embedding, observed_at, transient_samples, suggestions=True)
    if evaluation is None or not evaluation.same_day:
        return None
    winner, score = min(evaluation.same_day.items(), key=lambda item: (-item[1], item[0]))
    negative = evaluation.negative.get(winner)
    if score < settings.reid_employee_threshold or _vetoed(score, negative):
        return None
    competitors = [
        max(evaluation.suggestion.get(owner, -np.inf), evaluation.same_day.get(owner, -np.inf))
        for owner in evaluation.suggestion.keys() | evaluation.same_day.keys() if owner != winner
    ]
    if competitors and score - max(competitors) < settings.reid_employee_margin:
        return None
    return RankedEmployee(winner, score, negative, False)
