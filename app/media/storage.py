"""One root owner, validated owned paths, serialized leases and retention.

Lock order: storage lock -> short SQLite transaction. Never wait for this lock
while holding a SQLite read transaction. Filesystem metadata is never video time.
"""
import os
import shutil
import stat
import threading
import time
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path, PurePosixPath
from uuid import UUID

from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.config import settings
from app.database.database import immediate_session
from app.database.models import RecordingConnection, RecordingSegment, RecordingSession
from app.time_utils import utcnow

ACTIVE = ("STARTING", "RECORDING", "RECONNECTING", "STOPPING")


def ordinary(path: Path, *, directory=False):
    value = path.lstat()
    if stat.S_ISLNK(value.st_mode) or getattr(value, "st_file_attributes", 0) & 0x400:
        raise ValueError("MEDIA_PATH_UNSAFE")
    if not (stat.S_ISDIR(value.st_mode) if directory else stat.S_ISREG(value.st_mode)):
        raise ValueError("MEDIA_PATH_UNSAFE")
    return value


class RootLock:
    def __init__(self, root):
        self.file = None
        try:
            path = Path(root).absolute()
            for parent in reversed((path, *path.parents)):
                if parent.exists():
                    ordinary(parent, directory=True)
            path.mkdir(parents=True, exist_ok=True)
            lock = path / ".owner.lock"
            if lock.exists() or lock.is_symlink():
                ordinary(lock)
            self.file = lock.open("a+b")
            os.set_inheritable(self.file.fileno(), False)
            self.file.seek(0)
            if os.name == "nt":
                import msvcrt
                # Locking one byte beyond EOF is supported; never truncate a peer's lock.
                msvcrt.locking(self.file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.root = path.resolve()
        except (OSError, ValueError) as exc:
            if self.file:
                self.file.close()
            self.file = None
            raise HTTPException(503, "MEDIA_ROOT_UNAVAILABLE") from exc

    def close(self):
        if self.file is not None:
            self.file.close()  # OS releases exclusive lock, including after abrupt death.
            self.file = None


class RecordingStorage:
    def __init__(self, engine, root, *, now=utcnow, monotonic=time.monotonic):
        self.owner = RootLock(root)  # Before any SQL recovery, file reconciliation or lease state.
        self.root = self.owner.root
        self.engine, self.now, self.monotonic = engine, now, monotonic
        self.lock = threading.RLock()
        self.retention_days = settings.recording_retention_days
        self.max_bytes = settings.recording_max_bytes
        self.min_free_bytes = settings.recording_min_free_bytes
        self.leases = {}
        self.open_keys = set()
        self.cache = {}  # manager-known key -> expiry; unknown external files are never evicted
        self.closing = False
        self._sizes = None
        self._last_mono = self.monotonic()
        self._rate = 0.
        self.quota_shortened = False

    def path(self, key, *, exists=True, directory=False):
        if not isinstance(key, str) or "\\" in key or ":" in key:
            raise ValueError("MEDIA_PATH_UNSAFE")
        relative = PurePosixPath(key)
        if (relative.is_absolute() or not relative.parts or relative.as_posix() != key
                or any(p in (".", "..") for p in relative.parts)):
            raise ValueError("MEDIA_PATH_UNSAFE")
        candidate = self.root.joinpath(*relative.parts)
        ordinary(self.root, directory=True)
        for parent in relative.parents:
            if parent == PurePosixPath("."):
                continue
            ordinary(self.root.joinpath(*parent.parts), directory=True)
        if exists:
            ordinary(candidate, directory=directory)
        elif candidate.exists() or candidate.is_symlink():
            ordinary(candidate, directory=directory)
        return candidate

    @contextmanager
    def lease(self, segment_id):
        with self.lock:
            if self.closing:
                raise HTTPException(503, "MEDIA_STORAGE_CLOSED")
            with Session(self.engine) as db:
                row = db.get(RecordingSegment, str(segment_id))
                if row is None or row.state != "READY" or row.storage_key in self.open_keys:
                    raise HTTPException(409, "RECORDING_UNAVAILABLE")
                key = row.storage_key
            try:
                path = self.path(key)
            except (OSError, ValueError):
                raise HTTPException(409, "RECORDING_UNAVAILABLE") from None
            self.leases[key] = self.leases.get(key, 0) + 1
        try:
            yield path
        finally:
            self._release(key)

    def _release(self, key):
        with self.lock:
            count = self.leases[key] - 1
            if count:
                self.leases[key] = count
            else:
                del self.leases[key]
            if self.closing and not self.leases:
                self.owner.close()

    def register_cache(self, key, *, expires_at):
        """Register a generated, closed playback output; never browser-supplied paths."""
        with self.lock:
            try:
                parts = PurePosixPath(key).parts
                if len(parts) != 3 or parts[0] != "cache" or str(UUID(parts[1])) != parts[1]:
                    raise ValueError
            except (ValueError, TypeError):
                raise HTTPException(409, "MEDIA_CACHE_KEY_INVALID") from None
            self.path(key)
            if self.closing:
                raise HTTPException(503, "MEDIA_STORAGE_CLOSED")
            self.cache[key] = expires_at

    @contextmanager
    def cache_lease(self, key):
        with self.lock:
            if self.closing or key not in self.cache or key in self.open_keys:
                raise HTTPException(409, "PLAYBACK_UNAVAILABLE")
            path = self.path(key)
            self.leases[key] = self.leases.get(key, 0) + 1
        try:
            yield path
        finally:
            self._release(key)

    def usage(self):
        total = 0
        sizes = {}
        for directory, folders, files in os.walk(self.root, followlinks=False):
            parent = Path(directory)
            safe = []
            for name in folders:
                try:
                    ordinary(parent / name, directory=True)
                    safe.append(name)
                except (OSError, ValueError):
                    continue
            folders[:] = safe
            for name in files:
                try:
                    size = ordinary(parent / name).st_size
                    total += size
                    sizes[str(parent / name)] = size
                except (OSError, ValueError):
                    continue
        free = shutil.disk_usage(self.root).free
        mono = self.monotonic()
        if self._sizes is not None and mono - self._last_mono >= 1:
            # Deletions must not hide simultaneous writes from the two-second reserve.
            written = sum(max(0, size - self._sizes.get(key, 0)) for key, size in sizes.items())
            self._rate = written / (mono - self._last_mono)
        if self._sizes is None or mono - self._last_mono >= 1:
            self._sizes, self._last_mono = sizes, mono
        return total, free, max(256 * 1024 * 1024, int(2 * self._rate))

    def _expire(self, segment_id):
        with immediate_session(self.engine) as db:
            row = db.get(RecordingSegment, segment_id)
            if row is None or row.state == "EXPIRED":
                return
            if row.storage_key in self.leases or row.storage_key in self.open_keys:
                return
            session = db.get(RecordingSession, row.session_id)
            if session.status in ACTIVE and row.state == "UNINDEXED":
                return
            key = row.storage_key
            row.state = "DELETING"
        # Durable intent commits before unlink; restart resumes an interrupted deletion.
        try:
            self.path(key).unlink()
        except FileNotFoundError:
            pass
        except (OSError, ValueError):
            return  # Preserve DELETING; do not chase links or erase metadata on failure.
        with immediate_session(self.engine) as db:
            row = db.get(RecordingSegment, segment_id)
            row.state = "EXPIRED"
            connection_id = row.connection_id
        self.reclaim_connection_index(connection_id)

    def reclaim_connection_index(self, connection_id):
        """A finalized CSV is disposable only after every original has expired."""
        with self.lock:
            with Session(self.engine) as db:
                connection = db.get(RecordingConnection, connection_id)
                if connection is None or connection.closed_at is None:
                    return
                remaining = db.query(RecordingSegment.id).filter(
                    RecordingSegment.connection_id == connection_id, RecordingSegment.state != "EXPIRED").first()
                if remaining:
                    return
                csv_key = f"{connection.directory}/closed.csv"
            if csv_key in self.leases or csv_key in self.open_keys:
                return
            try:
                self.path(csv_key).unlink()
            except (OSError, ValueError):
                pass  # Index metadata survives; never delete unknown neighbouring files.

    def enforce(self):
        """Evict safe unused objects and return whether a new writer can be admitted."""
        with self.lock:
            now = self.now()
            total, free, reserve = self.usage()
            pressure = total + reserve > self.max_bytes or free < self.min_free_bytes
            for key, expires_at in sorted(self.cache.items(), key=lambda item: item[1]):
                if key in self.leases or key in self.open_keys or (not pressure and expires_at > now):
                    continue
                try:
                    self.path(key).unlink()
                except FileNotFoundError:
                    pass
                except (OSError, ValueError):
                    continue
                del self.cache[key]
                total, free, reserve = self.usage()
                pressure = total + reserve > self.max_bytes or free < self.min_free_bytes
            with Session(self.engine) as db:
                candidates = [(row.id, row.state, row.ended_at or connection.created_at)
                              for row, connection in db.query(RecordingSegment, RecordingConnection).join(
                                  RecordingConnection, RecordingConnection.id == RecordingSegment.connection_id
                              ).filter(RecordingSegment.state != "EXPIRED").all()]
            candidates.sort(key=lambda item: item[2])
            cutoff = now - timedelta(days=self.retention_days)
            for segment_id, state, age in candidates:
                if state == "DELETING" or age <= cutoff:
                    self._expire(segment_id)
            total, free, reserve = self.usage()
            for segment_id, _state, age in candidates:
                if total + reserve <= self.max_bytes and free >= self.min_free_bytes:
                    break
                self._expire(segment_id)
                following, free, reserve = self.usage()
                if following < total and age > cutoff:
                    self.quota_shortened = True
                total = following
            return total + reserve <= self.max_bytes and free >= self.min_free_bytes

    def snapshot(self):
        with self.lock:
            total, free, reserve = self.usage()
            with Session(self.engine) as db:
                oldest = db.query(RecordingSegment.started_at).filter_by(state="READY").order_by(
                    RecordingSegment.started_at).first()
            return {"bytes_used": total, "max_bytes": self.max_bytes, "free_bytes": free,
                    "min_free_bytes": self.min_free_bytes, "reserve_bytes": reserve,
                    "retention_days": self.retention_days,
                    "oldest_available": oldest[0] if oldest else None,
                    "quota_shortened_retention": self.quota_shortened}

    def close(self):
        with self.lock:
            self.closing = True
            if not self.leases:
                self.owner.close()
