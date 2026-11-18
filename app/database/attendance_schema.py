"""Versioned SQLite attendance installation, upgraded one frozen version at a time.

Preflight never repairs a partial/unknown installation. Explicit BEGIN IMMEDIATE
makes SQLite DDL and the committed version marker one indivisible transaction.
"""
import json
import logging
import re
import sqlite3
from contextlib import closing
from pathlib import Path

from sqlalchemy import Engine
from sqlalchemy.schema import CreateIndex, CreateTable

from app.config import BASE_DIR
from app.database.database import Base
from app.scene_geometry import extruded_door_top
from app.database import models  # register all original and attendance tables
from app.time_utils import utcnow

logger = logging.getLogger("app.database.attendance")
BACKUP_DIR = BASE_DIR / "data" / "backups"
SCHEMA_VERSION = 2
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
# Version 1 allowed one enabled camera per ENTRY/EXIT role and had no door frame.
# Its runtime row also pinned schema_version to 1, so both tables are rebuilt on upgrade.
_V1_ATTENDANCE_CAMERAS_DDL = """CREATE TABLE attendance_cameras (
    camera_id INTEGER NOT NULL,
    role VARCHAR(5),
    enabled BOOLEAN DEFAULT 0 NOT NULL,
    revision INTEGER DEFAULT 0 NOT NULL,
    source_revision INTEGER DEFAULT 1 NOT NULL,
    line_start JSON,
    line_end JSON,
    inside_point JSON,
    reference_width INTEGER,
    reference_height INTEGER,
    updated_at DATETIME NOT NULL,
    PRIMARY KEY (camera_id),
    CONSTRAINT ck_attendance_camera_role CHECK (role IS NULL OR role IN ('ENTRY', 'EXIT')),
    CONSTRAINT ck_attendance_camera_revisions CHECK (revision >= 0 AND source_revision >= 1),
    CONSTRAINT ck_attendance_camera_enabled CHECK (enabled IN (0, 1)),
    CONSTRAINT ck_attendance_camera_complete CHECK (enabled = 0 OR (role IS NOT NULL AND line_start IS NOT NULL AND line_end IS NOT NULL AND inside_point IS NOT NULL AND reference_width IS NOT NULL AND reference_height IS NOT NULL AND reference_width > 0 AND reference_height > 0)),
    FOREIGN KEY(camera_id) REFERENCES cameras (id) ON DELETE RESTRICT
)"""
_V1_ROLE_INDEX_DDL = ("CREATE UNIQUE INDEX uq_attendance_cameras_enabled_role "
                      "ON attendance_cameras (role) WHERE enabled = 1")
_V1_ATTENDANCE_RUNTIME_DDL = """CREATE TABLE attendance_runtime (
    id INTEGER NOT NULL,
    schema_version INTEGER DEFAULT 1 NOT NULL,
    gallery_revision INTEGER DEFAULT 0 NOT NULL,
    commission_id INTEGER,
    monitoring_started_at DATETIME,
    heartbeat_at DATETIME,
    stopped_at DATETIME,
    PRIMARY KEY (id),
    CONSTRAINT ck_attendance_runtime_singleton CHECK (id = 1),
    CONSTRAINT ck_attendance_runtime_version CHECK (schema_version = 1),
    CONSTRAINT ck_attendance_runtime_gallery CHECK (gallery_revision >= 0),
    FOREIGN KEY(commission_id) REFERENCES attendance_commissions (id) ON DELETE RESTRICT
)"""
_V1_TABLE_DDL = {"attendance_cameras": _V1_ATTENDANCE_CAMERAS_DDL, "attendance_runtime": _V1_ATTENDANCE_RUNTIME_DDL}


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


def _preflight(connection) -> int:
    """Return 0 when absent, else the version (1 or 2) of a complete, recognized installation."""
    catalog = _catalog(connection)
    attendance_objects = {
        name for name, (_kind, _sql, table) in catalog.items()
        if name.startswith("attendance_") or name == "work_sessions" or table in _ATTENDANCE_TABLES
    }
    if not attendance_objects:
        return 0
    if "attendance_runtime" not in catalog:
        raise _refuse("partial attendance schema without a committed version")
    try:
        versions = connection.exec_driver_sql(
            "SELECT id, schema_version FROM attendance_runtime"
        ).all()
    except Exception as exc:
        raise _refuse("missing or unreadable committed version") from exc
    if len(versions) != 1 or versions[0][0] != 1 or versions[0][1] not in (1, SCHEMA_VERSION):
        raise _refuse(f"unknown or uncommitted version {versions!r}")
    version = versions[0][1]

    expected_objects = set(_ATTENDANCE_TABLES)
    for name in sorted(_ATTENDANCE_TABLES):
        table = Base.metadata.tables[name]
        stored = catalog.get(name)
        frozen = _V1_TABLE_DDL.get(name) if version == 1 else None
        expected = frozen or str(CreateTable(table).compile(dialect=connection.dialect))
        if stored is None or stored[0] != "table" or _sql_key(stored[1] or "") != _sql_key(expected):
            raise _refuse(f"table {name} differs from version {version} (columns/constraints/foreign keys)")
        indexes = {}
        if version == 1 and name == "attendance_cameras":
            indexes["uq_attendance_cameras_enabled_role"] = _V1_ROLE_INDEX_DDL
        else:
            indexes.update((index.name, str(CreateIndex(index).compile(dialect=connection.dialect)))
                           for index in table.indexes)
        for index_name, expected_index in indexes.items():
            expected_objects.add(index_name)
            stored_index = catalog.get(index_name)
            if (stored_index is None or stored_index[0] != "index"
                    or _sql_key(stored_index[1] or "") != _sql_key(expected_index)):
                raise _refuse(f"index {index_name} differs from version {version}")
    expected_objects.add("attendance_camera_role_retained")
    trigger = catalog.get("attendance_camera_role_retained")
    if trigger is None or trigger[0] != "trigger" or _sql_key(trigger[1] or "") != _sql_key(_ROLE_TRIGGER):
        raise _refuse(f"assigned-camera role protection differs from version {version}")
    if attendance_objects != expected_objects:
        raise _refuse(f"unexpected attendance schema objects {sorted(attendance_objects - expected_objects)!r}")
    return version


def _has_rows(connection) -> bool:
    tables = [name for name, (kind, _sql, _table) in _catalog(connection).items() if kind == "table"]
    for name in tables:
        quoted = '"' + name.replace('"', '""') + '"'
        if connection.exec_driver_sql(f"SELECT 1 FROM {quoted} LIMIT 1").first() is not None:
            return True
    return False


def _upgrade_v1(connection) -> None:
    """Rebuild attendance_cameras: any number of ENTRY/EXIT/BOTH cameras, each with a door frame."""
    connection.exec_driver_sql("DROP TRIGGER attendance_camera_role_retained")
    connection.exec_driver_sql("DROP INDEX uq_attendance_cameras_enabled_role")
    connection.exec_driver_sql("ALTER TABLE attendance_cameras RENAME TO attendance_cameras_v1")
    models.AttendanceCamera.__table__.create(connection)
    rows = connection.exec_driver_sql(
        "SELECT camera_id, role, enabled, revision, source_revision, line_start, line_end, inside_point, "
        "reference_width, reference_height, updated_at FROM attendance_cameras_v1").all()
    for (camera_id, role, enabled, revision, source_revision, line_start, line_end, inside_point,
         reference_width, reference_height, updated_at) in rows:
        door_kind = door_top = None
        if line_start is not None and line_end is not None and reference_width and reference_height:
            door_top = extruded_door_top(json.loads(line_start), json.loads(line_end),
                                         reference_width, reference_height, "DOOR")
            if door_top is None:
                # No room above the threshold: the operator must redraw this gate.
                enabled, line_start, line_end, inside_point = 0, None, None, None
                reference_width = reference_height = None
            else:
                door_kind, door_top = "DOOR", json.dumps(door_top)
        connection.exec_driver_sql(
            "INSERT INTO attendance_cameras (camera_id, role, enabled, revision, source_revision, line_start, "
            "line_end, inside_point, door_kind, door_top, reference_width, reference_height, updated_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (camera_id, role, enabled, revision, source_revision, line_start, line_end, inside_point,
             door_kind, door_top, reference_width, reference_height, updated_at))
    connection.exec_driver_sql("DROP TABLE attendance_cameras_v1")
    connection.exec_driver_sql(_ROLE_TRIGGER)
    connection.exec_driver_sql("ALTER TABLE attendance_runtime RENAME TO attendance_runtime_v1")
    models.AttendanceRuntime.__table__.create(connection)
    connection.exec_driver_sql(
        "INSERT INTO attendance_runtime (id, schema_version, gallery_revision, commission_id, "
        "monitoring_started_at, heartbeat_at, stopped_at) SELECT id, ?, gallery_revision, commission_id, "
        "monitoring_started_at, heartbeat_at, stopped_at FROM attendance_runtime_v1", (SCHEMA_VERSION,))
    connection.exec_driver_sql("DROP TABLE attendance_runtime_v1")


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
    now = utcnow().isoformat(sep=" ")
    connection.execute(models.AttendanceRuntime.__table__.insert().prefix_with("OR IGNORE"),
                       {"id": 1, "schema_version": SCHEMA_VERSION, "gallery_revision": 0})
    connection.exec_driver_sql(
        "INSERT INTO attendance_cameras (camera_id, enabled, revision, source_revision, updated_at) "
        "SELECT id, 0, 0, 1, ? FROM cameras "
        "WHERE NOT EXISTS (SELECT 1 FROM attendance_cameras WHERE camera_id = cameras.id)",
        (now,),
    )


def initialize_attendance_schema(engine: Engine) -> None:
    """Validate both schemas before installing either on one backed-up transaction."""
    if engine.dialect.name != "sqlite":
        raise RuntimeError("Attendance startup requires SQLite; other database dialects are unsupported")
    from app.database import operator_schema
    with engine.connect() as connection:
        try:
            connection.exec_driver_sql("BEGIN IMMEDIATE")
            attendance = _preflight(connection)
            operator_installed = operator_schema.preflight(connection)
            if attendance != SCHEMA_VERSION or operator_installed != operator_schema.VERSION:
                if _has_rows(connection):
                    _backup(connection)
                existing = _catalog(connection)
                if "employees" not in existing:
                    operator_schema.create_legacy_employees(connection)
                excluded = operator_schema.TABLES | {"employees"}
                for table in Base.metadata.sorted_tables:
                    if table.name not in excluded and table.name not in existing:
                        table.create(connection)
                if attendance == 0:
                    connection.exec_driver_sql(_ROLE_TRIGGER)
                elif attendance == 1:
                    _upgrade_v1(connection)
                if operator_installed != operator_schema.VERSION:
                    operator_schema.install(connection, operator_installed)
            _seed(connection)
            # A fresh install is validated too, before its version is committed.
            if _preflight(connection) != SCHEMA_VERSION:
                raise _refuse("installation did not reach the current version")
            operator_schema.preflight(connection)
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
