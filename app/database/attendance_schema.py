"""Versioned, additive SQLite attendance installation.

Preflight never repairs a partial/unknown installation. Explicit BEGIN IMMEDIATE
makes SQLite DDL and the committed version marker one indivisible transaction.
"""
import logging
import re
import sqlite3
from contextlib import closing
from pathlib import Path

from sqlalchemy import Engine
from sqlalchemy.schema import CreateIndex, CreateTable

from app.config import BASE_DIR
from app.database.database import Base
from app.database import models  # register all original and attendance tables
from app.time_utils import utcnow

logger = logging.getLogger("app.database.attendance")
BACKUP_DIR = BASE_DIR / "data" / "backups"
SCHEMA_VERSION = 1
_ATTENDANCE_TABLES = frozenset({
    "attendance_cameras", "attendance_events", "attendance_evidence",
    "attendance_decisions", "work_sessions", "attendance_coverage_gaps",
    "attendance_runtime", "attendance_employee_references",
    "attendance_commissions", "attendance_requests",
})
_ROLE_TRIGGER = """CREATE TRIGGER attendance_camera_role_retained
BEFORE UPDATE OF role ON attendance_cameras
WHEN OLD.role IS NOT NULL AND NEW.role IS NULL
BEGIN
    SELECT RAISE(ABORT, 'Assigned attendance camera role cannot be erased');
END"""


def _refuse(detail: str) -> RuntimeError:
    return RuntimeError(f"Attendance schema refused: {detail}. Restore a pre-attendance backup or repair the installation explicitly.")


def _sql_key(sql: str) -> tuple[str, ...]:
    # Pin generated DDL while ignoring only whitespace between SQL tokens.
    # Quoted literals/identifiers retain their exact spelling and delimiters:
    # 'ENTRY' differs from 'entry', and "enabled = 1" is not enabled = 1.
    # Keeping words separate also avoids equating NOT NULL with NOTNULL.
    return tuple(re.findall(
        r"""'(?:''|[^'])*'|"(?:[^"]|"")*"|`(?:[^`]|``)*`|\[[^\]]*\]|[A-Za-z_]\w*|\d+|[^\s]""",
        sql.strip().removesuffix(";"),
    ))


def _catalog(connection) -> dict[str, tuple[str, str | None, str]]:
    rows = connection.exec_driver_sql(
        "SELECT name, type, sql, tbl_name FROM sqlite_master WHERE name NOT LIKE 'sqlite_%'"
    ).all()
    return {name: (kind, sql, table) for name, kind, sql, table in rows}


def _preflight(connection) -> bool:
    """Return True only for a complete, recognized v1 installation."""
    catalog = _catalog(connection)
    attendance_objects = {
        name for name, (_kind, _sql, table) in catalog.items()
        if name.startswith("attendance_") or name == "work_sessions" or table in _ATTENDANCE_TABLES
    }
    if not attendance_objects:
        return False
    if "attendance_runtime" not in catalog:
        raise _refuse("partial attendance schema without a committed version")
    try:
        versions = connection.exec_driver_sql(
            "SELECT id, schema_version FROM attendance_runtime"
        ).all()
    except Exception as exc:
        raise _refuse("missing or unreadable committed version") from exc
    if versions != [(1, SCHEMA_VERSION)]:
        raise _refuse(f"unknown or uncommitted version {versions!r}")

    expected_objects = set(_ATTENDANCE_TABLES)
    for name in sorted(_ATTENDANCE_TABLES):
        table = Base.metadata.tables[name]
        stored = catalog.get(name)
        expected = str(CreateTable(table).compile(dialect=connection.dialect))
        if stored is None or stored[0] != "table" or _sql_key(stored[1] or "") != _sql_key(expected):
            raise _refuse(f"table {name} differs from version 1 (columns/constraints/foreign keys)")
        for index in table.indexes:
            expected_objects.add(index.name)
            stored_index = catalog.get(index.name)
            expected_index = str(CreateIndex(index).compile(dialect=connection.dialect))
            if (stored_index is None or stored_index[0] != "index"
                    or _sql_key(stored_index[1] or "") != _sql_key(expected_index)):
                raise _refuse(f"index {index.name} differs from version 1")
    expected_objects.add("attendance_camera_role_retained")
    trigger = catalog.get("attendance_camera_role_retained")
    if trigger is None or trigger[0] != "trigger" or _sql_key(trigger[1] or "") != _sql_key(_ROLE_TRIGGER):
        raise _refuse("assigned-camera role protection differs from version 1")
    if attendance_objects != expected_objects:
        raise _refuse(f"unexpected attendance schema objects {sorted(attendance_objects - expected_objects)!r}")
    return True


def _has_rows(connection) -> bool:
    tables = [name for name, (kind, _sql, _table) in _catalog(connection).items() if kind == "table"]
    for name in tables:
        quoted = '"' + name.replace('"', '""') + '"'
        if connection.exec_driver_sql(f"SELECT 1 FROM {quoted} LIMIT 1").first() is not None:
            return True
    return False


def _backup(connection) -> Path:
    """Back up through an independent read connection under the writer lock.

    A RESERVED SQLite lock permits readers but excludes other writers. Using a
    separate source avoids sqlite3.backup waiting on its own write transaction.
    """
    main_file = next((row[2] for row in connection.exec_driver_sql("PRAGMA database_list") if row[1] == "main"), "")
    if not main_file:
        raise _refuse("a populated legacy database must be file-backed for backup")
    backup_path = None
    created = False
    try:
        BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        backup_path = BACKUP_DIR / f"pre-attendance-{utcnow().strftime('%Y%m%dT%H%M%S%fZ')}.db"
        # Never overwrite an existing backup, even on a timestamp collision.
        with backup_path.open("xb"):
            created = True
        source_uri = Path(main_file).resolve().as_uri() + "?mode=ro"
        with closing(sqlite3.connect(source_uri, uri=True, timeout=30)) as source:
            with closing(sqlite3.connect(backup_path, timeout=30)) as target:
                source.backup(target)
                if target.execute("PRAGMA integrity_check").fetchone() != ("ok",):
                    raise RuntimeError("backup integrity check failed")
    except Exception as exc:
        if created and backup_path is not None:
            backup_path.unlink(missing_ok=True)
        raise RuntimeError("Pre-attendance backup failed; no schema changes were made") from exc
    logger.info("Legacy database backup created: %s", backup_path)
    return backup_path


def _seed(connection) -> None:
    now = utcnow()
    connection.execute(models.AttendanceRuntime.__table__.insert().prefix_with("OR IGNORE"),
                       {"id": 1, "schema_version": SCHEMA_VERSION, "gallery_revision": 0})
    connection.exec_driver_sql(
        "INSERT INTO attendance_cameras (camera_id, enabled, revision, source_revision, updated_at) "
        "SELECT id, 0, 0, 1, ? FROM cameras "
        "WHERE NOT EXISTS (SELECT 1 FROM attendance_cameras WHERE camera_id = cameras.id)",
        (now,),
    )


def initialize_attendance_schema(engine: Engine) -> None:
    """Install or validate v1, refusing unsupported dialects before connecting."""
    if engine.dialect.name != "sqlite":
        raise RuntimeError("Attendance startup requires SQLite; other database dialects are unsupported")
    with engine.connect() as connection:
        try:
            connection.exec_driver_sql("BEGIN IMMEDIATE")
            recognized = _preflight(connection)  # Before any DDL, including original tables.
            if not recognized:
                if _has_rows(connection):
                    _backup(connection)
                Base.metadata.create_all(connection)
                connection.exec_driver_sql(_ROLE_TRIGGER)
            _seed(connection)
            # A fresh install is validated too, before its version is committed.
            _preflight(connection)
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
