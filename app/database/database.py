"""SQLAlchemy engine + сессии для SQLite (data/app.db)."""
import logging
from contextlib import contextmanager, nullcontext
from typing import Callable, TypeVar
from pathlib import Path
from threading import Lock, RLock
from weakref import WeakKeyDictionary

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import BASE_DIR, settings

logger = logging.getLogger("app.database")

# Weak keys let temporary/disposed engines disappear without a permanent registry.
# The registry mutex is never held while waiting for an engine's publication lock.
_publication_locks = WeakKeyDictionary()
_publication_locks_mutex = Lock()


class Base(DeclarativeBase):
    pass


def _resolve_url(url: str) -> str:
    """Относительный sqlite-путь приводим к абсолютному от корня проекта,
    чтобы приложение находило базу при запуске из любой директории."""
    prefix = "sqlite:///./"
    if url.startswith(prefix):
        path = (BASE_DIR / url[len(prefix):]).resolve()
        return f"sqlite:///{path.as_posix()}"
    return url


RESOLVED_DB_URL = _resolve_url(settings.database_url)
_IS_SQLITE = RESOLVED_DB_URL.startswith("sqlite")

if _IS_SQLITE:
    db_file = Path(RESOLVED_DB_URL[len("sqlite:///"):])
    db_file.parent.mkdir(parents=True, exist_ok=True)

engine = create_engine(
    RESOLVED_DB_URL,
    connect_args={"check_same_thread": False, "timeout": 30} if _IS_SQLITE else {},
)

if _IS_SQLITE:
    @event.listens_for(engine, "connect")
    def _sqlite_pragma(dbapi_connection, _record):
        # WAL + busy timeout: параллельные записи из AI-потока и API-хендлеров
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def init_db() -> None:
    from app.database import models  # noqa: F401 — регистрация таблиц

    from app.database.attendance_schema import initialize_attendance_schema

    initialize_attendance_schema(engine)
    logger.info("База данных готова: %s", RESOLVED_DB_URL)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def publication_guard(engine: Engine):
    """Fence token-changing commits against post-inference result publication.

    Reentrant for a caller combining a pending-work check with a guarded mutation.
    Never surround native inference or ordinary background database writes.
    """
    with _publication_locks_mutex:
        lock = _publication_locks.get(engine)
        if lock is None:
            lock = RLock()
            _publication_locks[engine] = lock
    with lock:
        yield


@contextmanager
def immediate_session(bind: Engine, *, publish: bool = False):
    """Own a SQLite write transaction, locking before the first read.

    The Session commits only its savepoint; the Connection owns the durable
    commit. Even an inner Session.commit() remains rollbackable on failure.
    Opt-in publication fencing covers connection acquisition through durable
    commit/rollback. Ordinary journal, event and telemetry writers stay independent.
    """
    with (publication_guard(bind) if publish else nullcontext()), bind.connect() as connection:
        try:
            connection.exec_driver_sql("BEGIN IMMEDIATE")
            with Session(bind=connection, join_transaction_mode="create_savepoint",
                         expire_on_commit=False) as db:
                yield db
                db.commit()
            connection.commit()
        except BaseException:
            connection.rollback()
            raise


T = TypeVar("T")


def read_snapshot(engine: Engine, reader: Callable[[Session], T]) -> T:
    """Run `reader` inside one explicit SQLite read snapshot."""
    with engine.connect() as connection:
        connection.exec_driver_sql("BEGIN")
        try:
            with Session(bind=connection, expire_on_commit=False) as db:
                return reader(db)
        finally:
            connection.rollback()
