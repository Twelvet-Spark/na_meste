"""Immutable, revision-bound active-employee gallery and distinct-owner matching."""
import logging
import re
import threading
from dataclasses import dataclass

import numpy as np
from sqlalchemy.orm import Session

from app.ai.face_detector import ALIGNMENT_ID
from app.database.models import AttendanceRuntime, Employee, EmployeeFace, FaceModelMetadata

logger = logging.getLogger("app.ai.face_recognition")
EMBEDDING_SIZE = 512


def normalized_embedding(embedding: np.ndarray) -> np.ndarray | None:
    """Reject malformed model/gallery vectors rather than manufacture a match."""
    try:
        vector = np.asarray(embedding, dtype=np.float64)
        if (vector.shape != (EMBEDDING_SIZE,) or not np.isfinite(vector).all()
                or np.max(np.abs(vector)) > np.finfo(np.float32).max):
            return None
        norm = float(np.linalg.norm(vector))
        if not np.isfinite(norm) or norm <= 0:
            return None
        return (vector / norm).astype(np.float32)
    except (ValueError, TypeError, OverflowError):
        return None


def is_normalized_embedding(embedding: np.ndarray) -> bool:
    """One persisted-vector contract for gallery, enrollment and transfer."""
    try:
        vector = np.asarray(embedding)
        if (vector.shape != (EMBEDDING_SIZE,) or vector.dtype.kind not in "fiu"
                or not np.isfinite(vector).all() or np.max(np.abs(vector)) > 1.001):
            return False
        norm = float(np.linalg.norm(vector.astype(np.float64, copy=False)))
        return abs(norm - 1.) <= 1e-3
    except (ValueError, TypeError, OverflowError):
        return False


def stored_embedding(blob: bytes) -> np.ndarray | None:
    """Persisted vectors must already be finite, normalized float32 x512."""
    try:
        vector = np.frombuffer(blob, dtype=np.float32)
        return vector if is_normalized_embedding(vector) else None
    except (ValueError, TypeError, BufferError):
        return None


@dataclass(frozen=True, slots=True)
class FaceMatch:
    employee_id: int | None
    score: float
    runner_up_employee_id: int | None
    runner_up_score: float
    margin: float
    store_revision: int


class EmbeddingStore:
    def __init__(self, model_id: str):
        if not isinstance(model_id, str) or re.fullmatch(r"[0-9a-f]{64}", model_id) is None:
            raise ValueError("model_id must be a declared SHA256")
        self.model_id = model_id
        self._lock = threading.Lock()
        self._refresh_lock = threading.Lock()
        self._matrix: np.ndarray | None = None
        self._employee_ids: tuple[int, ...] = ()
        self._owner_starts = np.array([], dtype=np.intp)
        self._names: dict[int, str] = {}
        self._active_ids: frozenset[int] = frozenset()
        self._revision = 0

    @property
    def revision(self) -> int:
        with self._lock:
            return self._revision

    def is_active(self, employee_id: int) -> bool:
        with self._lock:
            return employee_id in self._active_ids

    def refresh(self, session: Session) -> int:
        """Publish one explicit SQLite snapshot, never mixing revision and faces.

        Refreshes serialize so an older reader cannot overwrite a newer gallery.
        The caller's session is only used to locate its engine; no session crosses
        threads and no caller read transaction is committed as a side effect.
        """
        with self._refresh_lock:
            with session.get_bind().engine.connect() as connection:
                connection.exec_driver_sql("BEGIN")
                with Session(bind=connection) as reader:
                    revision = reader.query(AttendanceRuntime.gallery_revision).filter_by(id=1).one()[0]
                    employees = reader.query(Employee.id, Employee.name, Employee.active).all()
                    rows = reader.query(EmployeeFace.employee_id, EmployeeFace.embedding).join(
                        Employee, EmployeeFace.employee_id == Employee.id,
                    ).join(FaceModelMetadata, FaceModelMetadata.face_id == EmployeeFace.id).filter(
                        Employee.active.is_(True), FaceModelMetadata.model_id == self.model_id,
                        FaceModelMetadata.alignment_id == ALIGNMENT_ID,
                    ).order_by(EmployeeFace.employee_id, EmployeeFace.id).all()
                connection.rollback()

            vectors, owners, starts, employee_ids = [], [], [], []
            for employee_id, blob in rows:
                vector = stored_embedding(blob)
                if vector is None:
                    continue
                if not owners or owners[-1] != employee_id:
                    starts.append(len(owners))
                    employee_ids.append(employee_id)
                vectors.append(vector)
                owners.append(employee_id)
            matrix = np.stack(vectors) if vectors else None
            if matrix is not None:
                matrix.flags.writeable = False
            owner_starts = np.asarray(starts, dtype=np.intp)
            owner_starts.flags.writeable = False
            names = {employee_id: name for employee_id, name, _active in employees}
            active_ids = frozenset(employee_id for employee_id, _name, active in employees if active)
            with self._lock:
                self._matrix = matrix
                self._employee_ids = tuple(employee_ids)
                self._owner_starts = owner_starts
                self._names = names
                self._active_ids = active_ids
                self._revision = revision
        logger.info("Эмбеддинги перезагружены: %d лиц, %d активных сотрудников, revision=%d",
                    len(vectors), len(active_ids), revision)
        return len(vectors)

    def match(self, embedding: np.ndarray, *, exclude_employee_id: int | None = None) -> FaceMatch:
        vector = normalized_embedding(embedding)
        with self._lock:
            matrix, employee_ids, starts, revision = (
                self._matrix, self._employee_ids, self._owner_starts, self._revision,
            )
        empty = FaceMatch(None, 0., None, 0., 0., revision)
        if vector is None or matrix is None:
            return empty
        # Gallery rows are grouped by owner at refresh; no per-call gallery copy.
        scores = np.maximum.reduceat(matrix @ vector, starts)
        if exclude_employee_id is not None:
            try:
                scores[employee_ids.index(exclude_employee_id)] = -np.inf
            except ValueError:
                pass
        winner = int(np.argmax(scores))
        score = float(scores[winner])
        if not np.isfinite(score):
            return empty
        scores[winner] = -np.inf
        runner = int(np.argmax(scores))
        runner_score = float(scores[runner])
        runner_id = employee_ids[runner] if np.isfinite(runner_score) else None
        if runner_id is None:
            runner_score = 0.
        return FaceMatch(employee_ids[winner], score, runner_id, runner_score,
                         score - runner_score, revision)

    def employee_name(self, employee_id: int | None) -> str | None:
        with self._lock:
            return self._names.get(employee_id)
