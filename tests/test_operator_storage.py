"""Additive operator-v1/v2 storage contracts; controller executes RED/GREEN.

Run: .venv/Scripts/python.exe tests/test_operator_storage.py
All databases and backups are temporary. Legacy employees use pinned old DDL,
never the expanded Employee ORM table. Attendance fixtures retain exact v1 DDL.
"""
import sqlite3
import sys
import tempfile
import traceback
from contextlib import closing, contextmanager
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import create_engine, event, inspect
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from app.database import attendance_schema as schema
from app.database.database import Base
from app.database.models import (
    AttendanceEmployeeReference, AttendanceEvent, AttendanceRuntime, WorkSession,
)

TESTS = []
LEGACY_EMPLOYEE_DDL = """CREATE TABLE employees (
    id INTEGER NOT NULL,
    name VARCHAR(150) NOT NULL,
    external_id VARCHAR(64),
    active BOOLEAN NOT NULL,
    created_at DATETIME NOT NULL,
    updated_at DATETIME NOT NULL,
    PRIMARY KEY (id)
)"""
BASE_TABLES = frozenset({
    "cameras", "employee_faces", "presence_sessions", "global_persons",
    "global_observations", "global_events",
})
ATTENDANCE_TABLES = frozenset({
    "attendance_cameras", "attendance_events", "attendance_evidence",
    "attendance_decisions", "work_sessions", "attendance_coverage_gaps",
    "attendance_runtime", "attendance_employee_references",
    "attendance_commissions", "attendance_requests",
})
OPERATOR_V1_TABLES = {"operator_schema", "workplaces", "employee_workplaces", "operator_requests"}
OPERATOR_TABLES = OPERATOR_V1_TABLES | {"camera_archives"}
FUTURE_TABLES = {"recording_sessions", "recording_segments", "recording_connections", "recording_gaps"}
STAMP = datetime(2026, 11, 11, 4, 5, 6, 123456)

# Genuine committed-v1 fixtures are pinned independently of the implementation
# and latest ORM. An upgrade must validate this predecessor, not latest metadata.
COMMITTED_V1_ADDITIONS = (
    "first_name VARCHAR(74)",
    "last_name VARCHAR(74)",
    "revision INTEGER NOT NULL DEFAULT 0 CHECK (revision >= 0)",
)
COMMITTED_V1_TABLE_DDL = (
    """CREATE TABLE operator_schema (
        id INTEGER NOT NULL PRIMARY KEY CHECK (id = 1),
        version INTEGER NOT NULL CHECK (version >= 0),
        updated_at DATETIME NOT NULL
    )""",
    """CREATE TABLE workplaces (
        id INTEGER NOT NULL PRIMARY KEY,
        name VARCHAR(150) NOT NULL CHECK (length(name) BETWEEN 1 AND 150 AND name = trim(name)),
        name_key VARCHAR(450) NOT NULL,
        active BOOLEAN NOT NULL DEFAULT 1 CHECK (active IN (0, 1)),
        UNIQUE (name_key)
    )""",
    """CREATE TABLE employee_workplaces (
        employee_id INTEGER NOT NULL,
        workplace_id INTEGER NOT NULL,
        PRIMARY KEY (employee_id, workplace_id),
        FOREIGN KEY(employee_id) REFERENCES employees (id) ON DELETE CASCADE,
        FOREIGN KEY(workplace_id) REFERENCES workplaces (id) ON DELETE RESTRICT
    )""",
    """CREATE TABLE operator_requests (
        id VARCHAR(36) NOT NULL PRIMARY KEY,
        operation VARCHAR(100) NOT NULL,
        payload_sha256 VARCHAR(64) NOT NULL,
        result JSON NOT NULL,
        created_at DATETIME NOT NULL
    )""",
)


def test(fn):
    TESTS.append(fn)
    return fn


@contextmanager
def database():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        engine = create_engine(f"sqlite:///{(root / 'operator.db').as_posix()}",
                               connect_args={"timeout": 30, "check_same_thread": False})

        @event.listens_for(engine, "connect")
        def foreign_keys(connection, _record):
            connection.execute("PRAGMA foreign_keys=ON")

        try:
            with patch.object(schema, "BACKUP_DIR", root / "backups"):
                yield engine, root
        finally:
            engine.dispose()


def legacy(engine, *, attendance=False, populated=True):
    with engine.begin() as connection:
        connection.exec_driver_sql(LEGACY_EMPLOYEE_DDL)
        connection.exec_driver_sql("CREATE INDEX ix_employees_external_id ON employees (external_id)")
        names = BASE_TABLES | (ATTENDANCE_TABLES if attendance else frozenset())
        Base.metadata.create_all(connection, tables=[table for table in Base.metadata.sorted_tables
                                                     if table.name in names])
        if attendance:
            connection.exec_driver_sql(schema._ROLE_TRIGGER)
            connection.exec_driver_sql(
                "INSERT INTO attendance_runtime (id, schema_version, gallery_revision) VALUES (1, 1, 7)")
        if populated:
            connection.exec_driver_sql(
                "INSERT INTO employees (id, name, external_id, active, created_at, updated_at) "
                "VALUES (41, ?, 'old-id', 1, ?, ?)", ("Не разбивать старое имя", STAMP, STAMP))
            connection.exec_driver_sql(
                "INSERT INTO cameras (id,name,nvr_host,rtsp_port,username,password,channel,"
                "stream_type,enabled,created_at,updated_at) VALUES "
                "(12,'Old door','unused',554,'user','secret',1,'main',1,?,?)", (STAMP, STAMP))
            connection.exec_driver_sql(
                "INSERT INTO presence_sessions (id,employee_id,camera_id,started_at,last_seen_at) "
                "VALUES (19,41,12,?,?)", (STAMP, STAMP))
    if attendance and populated:
        with Session(engine) as db:
            db.add(AttendanceEvent(id="old-entry", source_key="old-entry", source="MANUAL",
                                   employee_id=41, employee_name="Снимок события", direction="ENTRY",
                                   occurred_at=STAMP, recorded_at=STAMP, status="ACCEPTED",
                                   raw={"version": 1, "candidate_employee_ids": []}))
            db.flush()
            db.add(AttendanceEmployeeReference(event_id="old-entry", employee_id=41,
                                               employee_name="Снимок ссылки"))
            db.add(WorkSession(entry_event_id="old-entry", employee_id=41,
                               employee_name="Снимок интервала", started_at=STAMP, status="OPEN"))
            db.commit()


def committed_v1(engine, *, populated=True):
    legacy(engine, attendance=True, populated=populated)
    with engine.begin() as connection:
        for addition in COMMITTED_V1_ADDITIONS:
            connection.exec_driver_sql(f"ALTER TABLE employees ADD COLUMN {addition}")
        for ddl in COMMITTED_V1_TABLE_DDL:
            connection.exec_driver_sql(ddl)
        connection.exec_driver_sql(
            "INSERT INTO operator_schema (id,version,updated_at) VALUES (1,1,?)", (STAMP,))
        if populated:
            connection.exec_driver_sql(
                "INSERT INTO workplaces (id,name,name_key,active) VALUES (7,'Магазин','магазин',1)")
            connection.exec_driver_sql(
                "INSERT INTO employee_workplaces (employee_id,workplace_id) VALUES (41,7)")
            connection.exec_driver_sql(
                "INSERT INTO operator_requests (id,operation,payload_sha256,result,created_at) "
                "VALUES ('old-request','employee.create',?, '{\"id\":41}',?)", ("a" * 64, STAMP))


def catalog(engine, names=None):
    with engine.connect() as connection:
        rows = connection.exec_driver_sql(
            "SELECT type,name,tbl_name,sql FROM sqlite_master WHERE name NOT LIKE 'sqlite_%' "
            "ORDER BY type,name").all()
        return tuple(tuple(row) for row in rows if names is None or row[2] in names)


def dump(engine):
    with engine.connect() as connection:
        return tuple(connection.connection.driver_connection.iterdump())


def assert_operator_v2(engine):
    tables = set(inspect(engine).get_table_names())
    assert OPERATOR_TABLES <= tables, f"Missing additive operator-v2 tables: {OPERATOR_TABLES - tables}"
    assert not FUTURE_TABLES & tables, "Task 3 must not install recording/media schemas"
    with engine.connect() as connection:
        assert connection.exec_driver_sql("SELECT id,version FROM operator_schema").all() == [(1, 2)]
        assert connection.exec_driver_sql("SELECT schema_version FROM attendance_runtime").scalar_one() == 1
    columns = {column["name"]: column for column in inspect(engine).get_columns("employees")}
    assert {"first_name", "last_name", "revision"} <= columns.keys()
    assert columns["first_name"]["nullable"] and columns["last_name"]["nullable"]
    assert str(columns["first_name"]["type"]) == "VARCHAR(74)"
    assert str(columns["last_name"]["type"]) == "VARCHAR(74)"
    assert not columns["revision"]["nullable"]
    assert str(columns["revision"]["default"]).strip("()'\"") == "0"


def refused_without_mutation(engine):
    before = dump(engine)
    try:
        schema.initialize_attendance_schema(engine)
    except RuntimeError:
        pass
    else:
        raise AssertionError("Unknown/partial operator or employee schema must fail closed")
    assert dump(engine) == before, "Refused preflight must not repair, seed, or migrate anything"


@test
def test_populated_legacy_upgrade_backs_up_once_and_preserves_ids_and_names():
    with database() as (engine, root):
        legacy(engine)
        before = dump(engine)
        schema.initialize_attendance_schema(engine)
        assert_operator_v2(engine)
        backups = list((root / "backups").glob("*.db"))
        assert len(backups) == 1
        with closing(sqlite3.connect(backups[0])) as backup:
            assert backup.execute("PRAGMA integrity_check").fetchone() == ("ok",)
            assert tuple(backup.iterdump()) == before
        with engine.connect() as connection:
            assert connection.exec_driver_sql(
                "SELECT id,name,external_id,first_name,last_name,revision FROM employees").all() == [
                    (41, "Не разбивать старое имя", "old-id", None, None, 0)]
            assert connection.exec_driver_sql(
                "SELECT id,employee_id,camera_id,ended_at FROM presence_sessions").all() == [(19, 41, 12, None)]
            assert connection.exec_driver_sql("SELECT count(*) FROM work_sessions").scalar_one() == 0
        installed = dump(engine)
        schema.initialize_attendance_schema(engine)
        assert dump(engine) == installed
        assert len(list((root / "backups").glob("*.db"))) == 1


@test
def test_attendance_v1_upgrade_preserves_exact_ddl_and_all_historical_snapshots():
    with database() as (engine, root):
        legacy(engine, attendance=True)
        attendance_sql = catalog(engine, ATTENDANCE_TABLES)
        before = dump(engine)
        schema.initialize_attendance_schema(engine)
        assert_operator_v2(engine)
        assert catalog(engine, ATTENDANCE_TABLES) == attendance_sql
        with Session(engine) as db:
            assert db.get(AttendanceEvent, "old-entry").employee_name == "Снимок события"
            assert db.get(AttendanceEmployeeReference, ("old-entry", 41)).employee_name == "Снимок ссылки"
            assert db.get(WorkSession, "old-entry").employee_name == "Снимок интервала"
            assert db.get(AttendanceRuntime, 1).gallery_revision == 7
        backups = list((root / "backups").glob("*.db"))
        assert len(backups) == 1, "Already installed attendance still needs a backup before operator DDL"
        with closing(sqlite3.connect(backups[0])) as backup:
            assert tuple(backup.iterdump()) == before


@test
def test_fresh_and_legacy_installations_converge_without_future_tables():
    with database() as (fresh, root), database() as (upgraded, _other):
        schema.initialize_attendance_schema(fresh)
        assert_operator_v2(fresh)
        assert not list((root / "backups").glob("*.db"))
        legacy(upgraded, attendance=True, populated=False)
        schema.initialize_attendance_schema(upgraded)
        assert_operator_v2(upgraded)
        normalized = lambda engine: tuple((kind, name, table, schema._sql_key(sql or ""))
                                          for kind, name, table, sql in catalog(engine))
        assert normalized(fresh) == normalized(upgraded), "Fresh install must take the same pinned employee ALTER path"
        before = dump(fresh)
        schema.initialize_attendance_schema(fresh)
        assert dump(fresh) == before


@test
def test_unversioned_partial_operator_tables_refused_before_attendance_install():
    for partial in ("CREATE TABLE workplaces (id INTEGER PRIMARY KEY)",
                    "CREATE TABLE operator_schema (id INTEGER, version INTEGER, updated_at DATETIME)",
                    "ALTER TABLE employees ADD COLUMN first_name VARCHAR(74)"):
        with database() as (engine, root):
            legacy(engine)
            with engine.begin() as connection:
                connection.exec_driver_sql(partial)
            refused_without_mutation(engine)
            assert not list((root / "backups").glob("*.db"))


@test
def test_malformed_legacy_employee_shape_refused_before_any_ddl():
    for alteration in ("ALTER TABLE employees ADD COLUMN surprise INTEGER",
                       "DROP INDEX ix_employees_external_id"):
        with database() as (engine, root):
            legacy(engine)
            with engine.begin() as connection:
                connection.exec_driver_sql(alteration)
            refused_without_mutation(engine)
            assert not list((root / "backups").glob("*.db"))


@test
def test_unknown_committed_operator_version_refused_without_fixed_version_check():
    with database() as (engine, _root):
        schema.initialize_attendance_schema(engine)
        assert_operator_v2(engine)
        with engine.begin() as connection:
            # Higher nonnegative versions must be representable, but this binary refuses them.
            connection.exec_driver_sql("UPDATE operator_schema SET version=42 WHERE id=1")
        refused_without_mutation(engine)


@test
def test_committed_v1_missing_index_fk_default_or_table_is_not_repaired():
    mutations = [
        "DROP TABLE operator_requests",
        "UPDATE sqlite_master SET sql=replace(sql,'ON DELETE RESTRICT','ON DELETE CASCADE') "
        "WHERE name='employee_workplaces'",
        "UPDATE sqlite_master SET sql=replace(sql,'DEFAULT 0','DEFAULT 9') WHERE name='employees'",
    ]
    for mutation in mutations:
        with database() as (engine, _root):
            committed_v1(engine)
            with engine.begin() as connection:
                connection.exec_driver_sql("PRAGMA writable_schema=ON")
                connection.exec_driver_sql(mutation)
                connection.exec_driver_sql("PRAGMA writable_schema=OFF")
            refused_without_mutation(engine)
    with database() as (engine, _root):
        committed_v1(engine)
        with engine.begin() as connection:
            indexes = connection.exec_driver_sql(
                "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='workplaces' "
                "AND sql IS NOT NULL").scalars().all()
            if indexes:
                name = indexes[0].replace('"', '""')
                connection.exec_driver_sql(f'DROP INDEX "{name}"')
            else:
                connection.exec_driver_sql("PRAGMA writable_schema=ON")
                connection.exec_driver_sql(
                    "UPDATE sqlite_master SET sql=replace(sql,'UNIQUE (name_key)','UNIQUE (name)') "
                    "WHERE name='workplaces'")
                connection.exec_driver_sql("PRAGMA writable_schema=OFF")
        refused_without_mutation(engine)


@test
def test_operator_ddl_failure_rolls_back_employee_alters_and_version_marker():
    with database() as (engine, root):
        legacy(engine, attendance=True)
        before = dump(engine)
        triggered = False

        def fail_ddl(connection, cursor, statement, parameters, context, executemany):
            nonlocal triggered
            if statement.lstrip().upper().startswith("CREATE TABLE") and "workplaces" in statement.lower():
                triggered = True
                raise OperationalError(statement, parameters, RuntimeError("injected DDL failure"))

        event.listen(engine, "before_cursor_execute", fail_ddl)
        try:
            try:
                schema.initialize_attendance_schema(engine)
            except (RuntimeError, OperationalError):
                pass
        finally:
            event.remove(engine, "before_cursor_execute", fail_ddl)
        assert triggered, "Upgrade must actually attempt additive operator DDL"
        assert dump(engine) == before
        assert len(list((root / "backups").glob("*.db"))) == 1
        schema.initialize_attendance_schema(engine)
        assert_operator_v2(engine)


@test
def test_operator_backup_failure_refuses_even_when_attendance_is_recognized():
    with database() as (engine, root):
        legacy(engine, attendance=True)
        (root / "backups").write_text("not a directory", encoding="utf-8")
        before = dump(engine)
        try:
            schema.initialize_attendance_schema(engine)
        except RuntimeError:
            pass
        else:
            raise AssertionError("Populated attendance-v1 upgrade must fail when backup cannot be written")
        assert dump(engine) == before


@test
def test_bad_attendance_preflight_never_installs_operator_objects():
    with database() as (engine, root):
        legacy(engine, attendance=True)
        with engine.begin() as connection:
            connection.exec_driver_sql("DROP INDEX uq_work_sessions_open_employee")
        refused_without_mutation(engine)
        assert not OPERATOR_TABLES & set(inspect(engine).get_table_names())
        assert not list((root / "backups").glob("*.db"))


@test
def test_genuine_committed_v1_upgrades_only_archive_schema_after_verified_backup():
    with database() as (engine, root):
        committed_v1(engine)
        before, attendance_sql = dump(engine), catalog(engine, ATTENDANCE_TABLES)
        v1_sql = catalog(engine, OPERATOR_V1_TABLES | {"employees"})
        attempted = []

        def observe_archive_ddl(connection, cursor, statement, parameters, context, executemany):
            if statement.lstrip().upper().startswith("CREATE TABLE") and "camera_archives" in statement:
                backups = list((root / "backups").glob("*.db"))
                assert len(backups) == 1, "Backup must precede the first v2 DDL"
                with closing(sqlite3.connect(backups[0])) as backup:
                    assert backup.execute("PRAGMA integrity_check").fetchone() == ("ok",)
                    assert tuple(backup.iterdump()) == before
                attempted.append(statement)

        event.listen(engine, "before_cursor_execute", observe_archive_ddl)
        try:
            schema.initialize_attendance_schema(engine)
        finally:
            event.remove(engine, "before_cursor_execute", observe_archive_ddl)
        assert attempted, "Committed v1 must be explicitly migrated, not silently treated as latest"
        assert_operator_v2(engine)
        assert catalog(engine, ATTENDANCE_TABLES) == attendance_sql
        assert catalog(engine, OPERATOR_V1_TABLES | {"employees"}) == v1_sql
        with engine.connect() as connection:
            assert connection.exec_driver_sql("SELECT * FROM employee_workplaces").all() == [(41, 7)]
            assert connection.exec_driver_sql(
                "SELECT id,result FROM operator_requests").all() == [("old-request", '{"id":41}')]
            assert connection.exec_driver_sql("SELECT count(*) FROM camera_archives").scalar_one() == 0
            assert connection.exec_driver_sql("SELECT id,name FROM cameras").all() == [(12, "Old door")]
        installed = dump(engine)
        schema.initialize_attendance_schema(engine)
        assert dump(engine) == installed and len(list((root / "backups").glob("*.db"))) == 1


@test
def test_committed_v1_archive_ddl_failure_rolls_back_and_retry_converges_with_fresh():
    with database() as (engine, root):
        committed_v1(engine)
        before, attempted = dump(engine), []

        def fail_archive_ddl(connection, cursor, statement, parameters, context, executemany):
            if statement.lstrip().upper().startswith("CREATE TABLE") and "camera_archives" in statement:
                attempted.append(statement)
                raise OperationalError(statement, parameters, RuntimeError("injected v2 DDL failure"))

        event.listen(engine, "before_cursor_execute", fail_archive_ddl)
        try:
            try:
                schema.initialize_attendance_schema(engine)
            except (RuntimeError, OperationalError):
                pass
        finally:
            event.remove(engine, "before_cursor_execute", fail_archive_ddl)
        assert attempted, "v1→v2 must attempt additive archive DDL"
        assert dump(engine) == before and len(list((root / "backups").glob("*.db"))) == 1
        schema.initialize_attendance_schema(engine)
        assert_operator_v2(engine)
        normalized = lambda db: tuple((kind, name, table, schema._sql_key(sql or ""))
                                      for kind, name, table, sql in catalog(db))
        with database() as (fresh, _other):
            schema.initialize_attendance_schema(fresh)
            assert_operator_v2(fresh)
            assert normalized(engine) == normalized(fresh)


@test
def test_committed_v1_failed_backup_leaves_version_and_schema_untouched():
    with database() as (engine, root):
        committed_v1(engine)
        (root / "backups").write_text("not a directory", encoding="utf-8")
        refused_without_mutation(engine)


@test
def test_committed_v1_with_unversioned_archive_or_unknown_objects_is_refused():
    for mutation in (
        "CREATE TABLE camera_archives (camera_id INTEGER PRIMARY KEY)",
        "CREATE INDEX unrecognized_workplace_index ON workplaces (active)",
        "CREATE TABLE recording_sessions (id TEXT PRIMARY KEY)",
    ):
        with database() as (engine, root):
            committed_v1(engine)
            with engine.begin() as connection:
                connection.exec_driver_sql(mutation)
            refused_without_mutation(engine)
            assert not list((root / "backups").glob("*.db"))


@test
def test_committed_v2_archive_default_fk_constraints_and_extra_objects_fail_closed():
    for mutation in (
        "DROP TABLE camera_archives",
        "UPDATE sqlite_master SET sql=replace(sql,'ON DELETE RESTRICT','ON DELETE CASCADE') "
        "WHERE name='camera_archives'",
        "UPDATE sqlite_master SET sql=replace(sql,'DEFAULT 0','DEFAULT 9') WHERE name='camera_archives'",
        "CREATE INDEX unrecognized_archive_index ON camera_archives (http_port)",
    ):
        with database() as (engine, _root):
            schema.initialize_attendance_schema(engine)
            assert_operator_v2(engine)
            with engine.begin() as connection:
                connection.exec_driver_sql("PRAGMA writable_schema=ON")
                connection.exec_driver_sql(mutation)
                connection.exec_driver_sql("PRAGMA writable_schema=OFF")
            refused_without_mutation(engine)


if __name__ == "__main__":
    failures = 0
    for case in TESTS:
        try:
            case()
            print(f"PASS {case.__name__}")
        except Exception:
            failures += 1
            print(f"FAIL {case.__name__}")
            traceback.print_exc()
    print(f"{len(TESTS) - failures}/{len(TESTS)} tests passed")
    sys.exit(bool(failures))
