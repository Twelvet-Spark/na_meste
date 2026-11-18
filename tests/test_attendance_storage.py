"""Attendance storage contracts. Run with .venv/Scripts/python.exe tests/test_attendance_storage.py.

Every database/backup is an isolated temporary file; production data is never opened.
Released v1/v2 installations are built from the frozen DDL, never from the live ORM.
"""
import sqlite3
import sys
import tempfile
import traceback
from contextlib import closing, contextmanager
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
from fastapi import HTTPException
from pydantic import ValidationError
from sqlalchemy import create_engine, event, inspect
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.orm import Session

from app.config import Settings
from app.database import attendance_schema as schema
from app.database import attendance_schema_frozen as frozen
from app.database.database import immediate_session
from app.database.models import (
    AttendanceCamera, AttendanceRuntime, Camera, DoorCrossing, Employee, ManualPresenceInterval,
    PresenceExclusion, PresenceSegment,
)
from app.services import camera_service, employee_service
from app.services.attendance_evidence import encode_evidence
from app.time_utils import attendance_iso, day_bounds, iso_local, utcnow
from test_operator_storage import ATTENDANCE_TABLES, DROPPED_TABLES, STAMP, dump, legacy
from test_attendance_recognition import create_identity

TESTS = []


def test(fn):
    TESTS.append(fn)
    return fn


@contextmanager
def database(initialize=True):
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        engine = create_engine(f"sqlite:///{(root / 'fixture.db').as_posix()}",
                               connect_args={"timeout": 30})

        @event.listens_for(engine, "connect")
        def foreign_keys(connection, _record):
            connection.execute("PRAGMA foreign_keys=ON")

        try:
            with patch.object(schema, "BACKUP_DIR", root / "backups"):
                if initialize:
                    schema.initialize_attendance_schema(engine)
                yield engine, root
        finally:
            engine.dispose()


def camera(**kwargs):
    return Camera(name="Door", nvr_host="unused", username="u", password="p", channel=1,
                  **kwargs)


def rejects(error_type, operation):
    try:
        operation()
    except error_type as exc:
        return exc
    raise AssertionError(f"Expected {error_type.__name__}")


def rejects_integrity(engine, row):
    with Session(engine) as db:
        db.add(row)
        rejects(IntegrityError, db.commit)
        db.rollback()


def tamper(engine, table, old, new):
    """Edit the stored DDL of `table`, the way a hand-edited or foreign install would differ."""
    with engine.begin() as connection:
        connection.exec_driver_sql("PRAGMA writable_schema=ON")
        connection.exec_driver_sql(
            "UPDATE sqlite_master SET sql=replace(sql, ?, ?) WHERE type='table' AND name=?",
            (old, new, table))
        connection.exec_driver_sql("PRAGMA writable_schema=OFF")


@test
def test_fresh_seed_and_idempotence():
    with database() as (engine, root):
        with Session(engine) as db:
            runtime = db.get(AttendanceRuntime, 1)
            assert runtime.schema_version == 3 and runtime.gallery_revision == 0
        assert [column["name"] for column in inspect(engine).get_columns("attendance_runtime")] == [
            "id", "schema_version", "gallery_revision"]
        assert ATTENDANCE_TABLES <= set(inspect(engine).get_table_names())
        assert not (root / "backups").exists()
        schema.initialize_attendance_schema(engine)
        with Session(engine) as db:
            assert db.query(AttendanceRuntime).count() == 1
            saved = camera_service.create_camera(db, camera_service.CameraCreate(
                name="Door", nvr_host="unused", username="u", password="p", channel=1,
                stream_type="main"))
            config = db.get(AttendanceCamera, saved.id)
            assert config.role is None and not config.enabled
            assert config.revision == 0 and config.source_revision == 1


@test
def test_existing_upgrade_backup_preserves_all_legacy_values():
    with database(False) as (engine, root):
        legacy(engine)
        schema.initialize_attendance_schema(engine)
        backups = list((root / "backups").glob("pre-attendance-*.db"))
        assert len(backups) == 1
        with closing(sqlite3.connect(backups[0])) as backup:
            assert backup.execute("PRAGMA integrity_check").fetchone() == ("ok",)
            assert backup.execute("SELECT started_at, ended_at FROM presence_sessions").fetchone() == (
                "2026-11-11 04:05:06.123456", None)
            assert backup.execute("SELECT name FROM employees").fetchone() == ("Не разбивать старое имя",)
            assert not backup.execute("SELECT name FROM sqlite_master WHERE name='attendance_runtime'").fetchall()
        # The pre-attendance presence history is dropped from the live database; the backup is its only copy.
        assert "presence_sessions" not in inspect(engine).get_table_names()
        with Session(engine) as db:
            assert db.query(Employee).one().name == "Не разбивать старое имя"
            assert db.query(AttendanceCamera).one().role is None
        schema.initialize_attendance_schema(engine)
        assert len(list((root / "backups").glob("*.db"))) == 1


@test
def test_backup_failure_aborts_before_ddl():
    with database(False) as (engine, root):
        Camera.__table__.create(engine)
        with Session(engine) as db:
            db.add(camera())
            db.commit()
        with patch.object(schema, "BACKUP_DIR", root / "blocked"):
            (root / "blocked").write_text("not a directory", encoding="utf-8")
            rejects(RuntimeError, lambda: schema.initialize_attendance_schema(engine))
        assert inspect(engine).get_table_names() == ["cameras"]


@test
def test_unknown_partial_and_malformed_schemas_refused_without_repair():
    with database() as (engine, _root):
        with engine.begin() as connection:
            connection.exec_driver_sql("PRAGMA ignore_check_constraints=ON")
            connection.exec_driver_sql("UPDATE attendance_runtime SET schema_version=42")
            connection.exec_driver_sql("PRAGMA ignore_check_constraints=OFF")
        rejects(RuntimeError, lambda: schema.initialize_attendance_schema(engine))
        with engine.connect() as connection:
            assert connection.exec_driver_sql("SELECT schema_version FROM attendance_runtime").scalar() == 42
    for stray in (frozen.V2_TABLES["attendance_requests"], None):
        with database(False) as (engine, _root):
            # A lone attendance table, old or new, is a partial schema without a committed version.
            if stray is None:
                PresenceSegment.__table__.create(engine)
                expected = ["attendance_presence_segments"]
            else:
                with engine.begin() as connection:
                    connection.exec_driver_sql(stray)
                expected = ["attendance_requests"]
            rejects(RuntimeError, lambda: schema.initialize_attendance_schema(engine))
            assert inspect(engine).get_table_names() == expected
    with database() as (engine, _root):
        with engine.begin() as connection:
            connection.exec_driver_sql("DROP INDEX ix_presence_segments_start")
            connection.exec_driver_sql("CREATE INDEX ix_presence_segments_start ON attendance_presence_segments(last_seen_at)")
        rejects(RuntimeError, lambda: schema.initialize_attendance_schema(engine))
    with database() as (engine, _root):
        with engine.begin() as connection:
            connection.exec_driver_sql("DROP INDEX ix_door_crossings_employee_time")
        rejects(RuntimeError, lambda: schema.initialize_attendance_schema(engine))
    with database() as (engine, _root):
        with engine.begin() as connection:
            connection.exec_driver_sql("DROP TABLE attendance_exclusions")
            connection.exec_driver_sql("CREATE TABLE attendance_exclusions (id INTEGER PRIMARY KEY)")
        rejects(RuntimeError, lambda: schema.initialize_attendance_schema(engine))
    with database() as (engine, _root):
        with engine.begin() as connection:
            # Version-3 databases hold none of the removed history tables.
            connection.exec_driver_sql(frozen.V2_TABLES["attendance_requests"])
        exc = rejects(RuntimeError, lambda: schema.initialize_attendance_schema(engine))
        assert "unexpected attendance schema objects ['attendance_requests']" in str(exc)


@test
def test_malformed_checks_and_fk_actions_refused_for_every_released_version():
    cases = [
        (2, "attendance_coverage_gaps", "singleton = 1", "singleton >= 1"),
        (2, "attendance_employee_references", "ON DELETE RESTRICT", "ON DELETE CASCADE"),
        (3, "attendance_manual_intervals", "ended_at > started_at", "ended_at >= started_at"),
        (3, "attendance_door_crossings", "ON DELETE RESTRICT", "ON DELETE CASCADE"),
    ]
    for version, table, old, new in cases:
        with database(version == 3) as (engine, root):
            if version == 2:
                legacy(engine, attendance=True, attendance_version=2, populated=False)
            tamper(engine, table, old, new)
            before = dump(engine)
            exc = rejects(RuntimeError, lambda: schema.initialize_attendance_schema(engine))
            assert f"table {table} differs from version {version}" in str(exc), str(exc)
            assert dump(engine) == before, "A refused install must not be repaired or migrated"
            assert not (root / "backups").exists()


@test
def test_altered_enum_literal_refused_without_mutation():
    with database() as (engine, _root):
        with engine.begin() as connection:
            table_sql = connection.exec_driver_sql(
                "SELECT sql FROM sqlite_master WHERE name='attendance_cameras'").scalar_one()
            trigger_sql = connection.exec_driver_sql(
                "SELECT sql FROM sqlite_master WHERE name='attendance_camera_role_retained'").scalar_one()
            assert "'ENTRY'" in table_sql
            connection.exec_driver_sql("DROP TABLE attendance_cameras")
            connection.exec_driver_sql(table_sql.replace("'ENTRY'", "'entry'"))
            # DROP TABLE also removes its trigger; restore it so only the
            # case-sensitive enum literal differs from the recognized schema.
            connection.exec_driver_sql(trigger_sql)
        before = dump(engine)
        exc = rejects(RuntimeError, lambda: schema.initialize_attendance_schema(engine))
        assert "table attendance_cameras differs from version 3" in str(exc)
        assert dump(engine) == before


@test
def test_version_one_role_index_is_unexpected_in_version_three():
    with database() as (engine, _root):
        with engine.begin() as connection:
            connection.exec_driver_sql(frozen.V1_ROLE_INDEX_DDL)
        before = dump(engine)
        exc = rejects(RuntimeError, lambda: schema.initialize_attendance_schema(engine))
        assert "unexpected attendance schema objects ['uq_attendance_cameras_enabled_role']" in str(exc)
        assert dump(engine) == before


@test
def test_version_one_upgrade_keeps_roles_and_adds_door_frames_after_backup():
    with database(False) as (engine, root):
        legacy(engine, attendance=True, populated=False)
        with engine.begin() as connection:
            connection.exec_driver_sql("UPDATE attendance_runtime SET gallery_revision=9")
            for ident in (1, 2, 3, 4):
                connection.exec_driver_sql(
                    "INSERT INTO cameras (id,name,nvr_host,rtsp_port,username,password,channel,stream_type,"
                    "enabled,created_at,updated_at) VALUES (?,'Door','unused',554,'u','p',?,'main',1,"
                    "'2026-11-11 04:00:00','2026-11-11 04:00:00')", (ident, ident))
            rows = (
                (1, "ENTRY", 1, 3, 2, "[0.2, 0.8]", "[0.6, 0.8]", "[0.4, 0.95]", 1000, 500),
                (2, "EXIT", 0, 0, 1, "[0.2, 0.9]", "[0.3, 0.9]", "[0.25, 0.99]", 1000, 500),
                (3, "ENTRY", 0, 5, 1, None, None, None, None, None),
                (4, "EXIT", 1, 1, 1, "[0.2, 0.0]", "[0.6, 0.5]", "[0.4, 0.9]", 1000, 500),
            )
            for row in rows:
                connection.exec_driver_sql(
                    "INSERT INTO attendance_cameras (camera_id, role, enabled, revision, source_revision, "
                    "line_start, line_end, inside_point, reference_width, reference_height, updated_at) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,'2026-11-11 04:00:00')", row)
        schema.initialize_attendance_schema(engine)
        assert len(list((root / "backups").glob("*.db"))) == 1
        assert not DROPPED_TABLES & set(inspect(engine).get_table_names())
        with Session(engine) as db:
            runtime = db.get(AttendanceRuntime, 1)
            assert runtime.schema_version == 3 and runtime.gallery_revision == 9
            saved = {row.camera_id: row for row in db.query(AttendanceCamera)}
            assert {ident: row.role for ident, row in saved.items()} == {1: "ENTRY", 2: "EXIT", 3: "ENTRY", 4: "EXIT"}
            assert (saved[1].revision, saved[1].source_revision, saved[1].enabled) == (3, 2, True)
            assert saved[1].line_start == [0.2, 0.8] and saved[1].door_kind == "DOOR"
            # 400 px threshold, 1.8 * 400 = 720 px door, clamped to the first pixel row of the frame.
            assert saved[1].door_top == [[0.2, 0.0], [0.6, 0.0]]
            # 100 px threshold at y=450: a 180 px door ends at y=270.
            assert all(abs(point[1] - 0.54) < 1e-9 for point in saved[2].door_top)
            assert saved[3].door_kind is None and saved[3].door_top is None and not saved[3].enabled
            # No room above a threshold on the first row: the gate must be redrawn, never guessed.
            assert saved[4].line_start is None and saved[4].door_top is None and not saved[4].enabled
            assert saved[4].role == "EXIT"
        with engine.connect() as connection:
            names = {row[0] for row in connection.exec_driver_sql("SELECT name FROM sqlite_master")}
            assert "attendance_camera_role_retained" in names
            assert "uq_attendance_cameras_enabled_role" not in names
            assert "attendance_cameras_v1" not in names and "attendance_runtime_old" not in names
        with Session(engine) as db:
            for ident in (1, 2):
                row = db.get(AttendanceCamera, ident)
                row.role = "BOTH"
            db.commit()


@test
def test_version_two_upgrade_keeps_gates_and_gallery_and_drops_history_after_backup():
    with database(False) as (engine, root):
        legacy(engine, attendance=True, attendance_version=2)
        with engine.begin() as connection:
            connection.exec_driver_sql("UPDATE attendance_runtime SET gallery_revision=9")
            connection.exec_driver_sql(
                "INSERT INTO attendance_cameras (camera_id,role,enabled,revision,source_revision,line_start,"
                "line_end,inside_point,door_kind,door_top,reference_width,reference_height,updated_at) VALUES "
                "(12,'BOTH',1,3,2,'[0.2, 0.8]','[0.6, 0.8]','[0.4, 0.95]','DOOR','[[0.2, 0.1], [0.6, 0.1]]',"
                "1000,500,'2026-11-11 04:00:00')")
        before = dump(engine)
        first_ddl = []

        def observe(_conn, _cursor, statement, _parameters, _context, _many):
            if not first_ddl and statement.lstrip().upper().startswith(("CREATE TABLE", "DROP TABLE")):
                first_ddl.append(statement)
                backups = list((root / "backups").glob("pre-attendance-*.db"))
                assert len(backups) == 1, "The backup must exist before the first DDL statement"
                with closing(sqlite3.connect(backups[0])) as backup:
                    assert backup.execute("PRAGMA integrity_check").fetchone() == ("ok",)
                    assert tuple(backup.iterdump()) == before

        event.listen(engine, "before_cursor_execute", observe)
        try:
            schema.initialize_attendance_schema(engine)
        finally:
            event.remove(engine, "before_cursor_execute", observe)
        assert first_ddl
        tables = set(inspect(engine).get_table_names())
        assert ATTENDANCE_TABLES <= tables
        assert not (DROPPED_TABLES | {"attendance_runtime_old"}) & tables
        assert [column["name"] for column in inspect(engine).get_columns("attendance_runtime")] == [
            "id", "schema_version", "gallery_revision"]
        with Session(engine) as db:
            runtime = db.get(AttendanceRuntime, 1)
            assert runtime.schema_version == 3 and runtime.gallery_revision == 9
            gate = db.get(AttendanceCamera, 12)
            assert (gate.role, gate.enabled, gate.revision, gate.source_revision) == ("BOTH", True, 3, 2)
            assert gate.door_kind == "DOOR" and gate.door_top == [[0.2, 0.1], [0.6, 0.1]]
            # The new tables take rows, and from then on they protect the employee and camera.
            db.add(PresenceSegment(employee_id=41, camera_id=12, started_at=STAMP, last_seen_at=STAMP,
                                   face_seen=True))
            db.commit()
            exc = rejects(HTTPException, lambda: employee_service.delete_employee(db, db.get(Employee, 41)))
            assert exc.status_code == 409
        with engine.begin() as connection:
            rejects(IntegrityError, lambda: connection.exec_driver_sql(
                "UPDATE attendance_cameras SET role=NULL WHERE camera_id=12"))
        installed = dump(engine)
        schema.initialize_attendance_schema(engine)
        assert dump(engine) == installed and len(list((root / "backups").glob("*.db"))) == 1


@test
def test_ddl_interruption_rolls_back_entire_install():
    with database(False) as (engine, _root):
        def interrupt(_conn, _cursor, statement, _parameters, _context, _many):
            if statement.lstrip().startswith("CREATE TABLE attendance_door_crossings"):
                raise RuntimeError("injected DDL interruption")
        event.listen(engine, "before_cursor_execute", interrupt)
        try:
            rejects(RuntimeError, lambda: schema.initialize_attendance_schema(engine))
        finally:
            event.remove(engine, "before_cursor_execute", interrupt)
        assert inspect(engine).get_table_names() == []
        schema.initialize_attendance_schema(engine)


@test
def test_non_sqlite_refused_before_connection():
    class NonSQLite:
        class dialect:
            name = "postgresql"
    rejects(RuntimeError, lambda: schema.initialize_attendance_schema(NonSQLite()))


@test
def test_work_time_tables_enforce_ordering_enums_uniqueness_and_foreign_keys():
    start = datetime(2026, 11, 11, 4)
    end = start + timedelta(hours=1)
    with database() as (engine, _root):
        with Session(engine) as db:
            employee, cam = Employee(name="Worker"), camera()
            db.add_all([employee, cam])
            db.commit()
            employee_id, camera_id = employee.id, cam.id

        def segment(**kwargs):
            values = dict(employee_id=employee_id, camera_id=camera_id, started_at=start,
                          last_seen_at=end, face_seen=True)
            return PresenceSegment(**{**values, **kwargs})

        def crossing(key, direction="ENTRY"):
            return DoorCrossing(source_key=key, camera_id=camera_id, employee_id=employee_id,
                                direction=direction, occurred_at=start)

        with Session(engine) as db:
            db.add_all([segment(), crossing("door:1")])
            db.commit()
        rejects_integrity(engine, segment(started_at=end, last_seen_at=start))
        rejects_integrity(engine, segment(employee_id=employee_id + 100))
        rejects_integrity(engine, crossing("door:2", "BOTH"))
        rejects_integrity(engine, crossing("door:1"))
        for model in (ManualPresenceInterval, PresenceExclusion):
            def correction(request_id, **kwargs):
                values = dict(request_id=request_id, employee_id=employee_id, started_at=start,
                              ended_at=end, actor="Operator", reason="Reason")
                return model(**{**values, **kwargs})

            with Session(engine) as db:
                db.add(correction("request-1"))
                db.commit()
            rejects_integrity(engine, correction("request-2", ended_at=start))
            rejects_integrity(engine, correction("request-2", ended_at=start - timedelta(minutes=1)))
            rejects_integrity(engine, correction("request-1"))
            # A revocation is all of time, actor and reason, or none of them.
            rejects_integrity(engine, correction("request-3", revoked_at=end))
            rejects_integrity(engine, correction("request-3", revoked_at=end, revoked_by="Operator"))
            rejects_integrity(engine, correction("request-3", revoked_by="Operator", revoke_reason="Mistake"))
            with Session(engine) as db:
                db.add(correction("request-4", revoked_at=end, revoked_by="Operator", revoke_reason="Mistake"))
                db.commit()
        with engine.begin() as connection:
            rejects(IntegrityError, lambda: connection.exec_driver_sql("DELETE FROM employees WHERE id=?", (employee_id,)))
            rejects(IntegrityError, lambda: connection.exec_driver_sql("DELETE FROM cameras WHERE id=?", (camera_id,)))


@test
def test_enabled_camera_requires_dimensions_and_assigned_role_cannot_be_erased():
    with database() as (engine, _root):
        with Session(engine) as db:
            db.add(camera(id=1))
            db.commit()
        rejects_integrity(engine, AttendanceCamera(camera_id=1, role="ENTRY", enabled=True,
            line_start=[0.1, 0.5], line_end=[0.9, 0.5], inside_point=[0.5, 0.9]))
        with Session(engine) as db:
            db.add(AttendanceCamera(camera_id=1, role="ENTRY", enabled=False))
            db.commit()
        with engine.begin() as connection:
            rejects(IntegrityError, lambda: connection.exec_driver_sql(
                "UPDATE attendance_cameras SET role=NULL WHERE camera_id=1"))


@test
def test_door_gate_configuration_constraints():
    with database() as (engine, _root):
        with Session(engine) as db:
            db.add_all([camera(id=1), camera(id=2)])
            db.commit()
        rejects_integrity(engine, AttendanceCamera(camera_id=1, role="ENTRY", enabled=True))
        door = dict(line_start=[0.1, 0.5], line_end=[0.9, 0.5], inside_point=[0.5, 0.9],
                    reference_width=1280, reference_height=720)
        # An enabled gate needs its door frame too.
        rejects_integrity(engine, AttendanceCamera(camera_id=1, role="ENTRY", enabled=True, **door))
        rejects_integrity(engine, AttendanceCamera(camera_id=1, role="ENTRY", enabled=True, door_kind="GATE",
                                                   door_top=[[0.1, 0.1], [0.9, 0.1]], **door))
        with Session(engine) as db:
            # Any number of cameras may count the same direction, and one camera may count both.
            db.add_all([AttendanceCamera(camera_id=ident, role=role, enabled=True, door_kind="DOOR",
                                         door_top=[[0.1, 0.1], [0.9, 0.1]], **door)
                        for ident, role in ((1, "ENTRY"), (2, "BOTH"))])
            db.commit()


@test
def test_work_history_protects_employee_and_camera_and_deactivation_keeps_it():
    with database() as (engine, _root):
        with Session(engine) as db:
            employee = create_identity(db)
            cam = camera_service.create_camera(db, camera_service.CameraCreate(
                name="Door", nvr_host="unused", username="u", password="p", channel=1, stream_type="main"))
            employee_id, camera_id = employee.id, cam.id
            db.add(PresenceSegment(employee_id=employee_id, camera_id=camera_id, started_at=STAMP,
                                   last_seen_at=STAMP, face_seen=False))
            db.commit()
            exc = rejects(HTTPException, lambda: employee_service.delete_employee(db, employee))
            assert exc.status_code == 409
            exc = rejects(HTTPException, lambda: camera_service.delete_camera(db, cam))
            assert exc.status_code == 409 and exc.detail == "Camera has history; disable instead"
            before = db.get(AttendanceRuntime, 1).gallery_revision
            updated = employee_service.update_employee(db, employee, employee_service.EmployeeUpdate(
                expected_revision=employee.revision, active=False))
            assert not updated.active
            db.expire_all()
            assert db.get(AttendanceRuntime, 1).gallery_revision == before + 1
            assert db.query(PresenceSegment).one().last_seen_at == STAMP


@test
def test_each_work_record_alone_protects_its_employee_and_door_records_protect_the_camera():
    with database() as (engine, _root):
        with Session(engine) as db:
            cam = camera()
            db.add(cam)
            db.commit()
            camera_id = cam.id
            makers = {
                "segment": lambda employee_id: PresenceSegment(
                    employee_id=employee_id, camera_id=camera_id, started_at=STAMP, last_seen_at=STAMP,
                    face_seen=True),
                "crossing": lambda employee_id: DoorCrossing(
                    source_key=f"door:{employee_id}", camera_id=camera_id, employee_id=employee_id,
                    direction="EXIT", occurred_at=STAMP),
                "manual": lambda employee_id: ManualPresenceInterval(
                    request_id=f"manual-{employee_id}", employee_id=employee_id, started_at=STAMP,
                    ended_at=STAMP + timedelta(hours=1), actor="Operator", reason="Reason"),
                "exclusion": lambda employee_id: PresenceExclusion(
                    request_id=f"exclusion-{employee_id}", employee_id=employee_id, started_at=STAMP,
                    ended_at=STAMP + timedelta(hours=1), actor="Operator", reason="Reason"),
            }
            for kind, make in makers.items():
                employee = Employee(name=f"Worker {kind}")
                db.add(employee)
                db.commit()
                db.add(make(employee.id))
                db.commit()
                exc = rejects(HTTPException, lambda: employee_service.delete_employee(db, employee))
                assert exc.status_code == 409, f"A {kind} record must keep its employee"
            # Both automatic records pin the camera, even without an assigned door role.
            exc = rejects(HTTPException, lambda: camera_service.delete_camera(db, cam))
            assert exc.status_code == 409
            db.query(PresenceSegment).delete()
            db.commit()
            exc = rejects(HTTPException, lambda: camera_service.delete_camera(db, cam))
            assert exc.status_code == 409, "A door crossing alone must keep its camera"
            db.query(DoorCrossing).delete()
            db.add(AttendanceCamera(camera_id=camera_id, role="EXIT", enabled=False))
            db.commit()
            exc = rejects(HTTPException, lambda: camera_service.delete_camera(db, cam))
            assert exc.status_code == 409, "An assigned door role is retained"
            unused = camera_service.create_camera(db, camera_service.CameraCreate(
                name="Unused", nvr_host="unused", username="u", password="p", channel=2, stream_type="main"))
            unused_id = unused.id
            camera_service.delete_camera(db, unused)
            assert db.get(Camera, unused_id) is None
            assert db.get(AttendanceCamera, unused_id) is None


@test
def test_camera_history_rejection_does_not_stop_reader():
    from app.api.cameras import api_delete_camera
    with database() as (engine, _root):
        with Session(engine) as db:
            cam, employee = camera(), Employee(name="Worker")
            db.add_all([cam, employee])
            db.flush()
            db.add(DoorCrossing(source_key="door:1", camera_id=cam.id, employee_id=employee.id,
                                direction="ENTRY", occurred_at=STAMP))
            db.commit()
            class RunningReader:
                stopped = False
                def stop(self, _deletion):
                    self.stopped = True
            manager = RunningReader()
            exc = rejects(HTTPException, lambda: api_delete_camera(cam.id, db, manager))
            assert exc.status_code == 409 and not manager.stopped


@test
def test_immediate_transaction_rolls_back_session_commit_on_failure():
    with database() as (engine, _root):
        def write_and_fail():
            with immediate_session(engine) as db:
                db.add(Employee(name="Must roll back"))
                db.commit()
                raise RuntimeError("after savepoint release")
        rejects(RuntimeError, write_and_fail)
        with Session(engine) as db:
            assert db.query(Employee).count() == 0


@test
def test_deletion_history_checks_exclude_concurrent_writers():
    with database() as (engine, root):
        with Session(engine) as db:
            employee, cam = Employee(name="Unused"), camera()
            db.add_all([employee, cam])
            db.commit()
            employee_id, camera_id = employee.id, cam.id
            attempts = []

            def competing_write(_conn, _cursor, statement, _params, _context, _many):
                if "FROM attendance_presence_segments" not in statement:
                    return
                # The deletion's very first history SELECT must already hold a
                # reserved writer lock. No sleeps or timing-sensitive threads.
                with closing(sqlite3.connect(root / "fixture.db", timeout=0)) as other:
                    rejects(sqlite3.OperationalError, lambda: other.execute(
                        "INSERT INTO employees (name, active, created_at, updated_at) "
                        "VALUES ('Race', 1, '2026-11-11', '2026-11-11')"))
                    attempts.append(True)

            event.listen(engine, "before_cursor_execute", competing_write)
            try:
                employee_service.delete_employee(db, employee)
                camera_service.delete_camera(db, cam)
            finally:
                event.remove(engine, "before_cursor_execute", competing_write)
            assert len(attempts) == 2
            assert db.get(Employee, employee_id) is None
            assert db.get(Camera, camera_id) is None
            assert db.query(PresenceSegment).count() == 0


@test
def test_database_failure_returns_503_without_deleting_history():
    with database() as (engine, _root):
        with Session(engine) as db:
            employee = Employee(name="Retained")
            db.add(employee)
            db.commit()
            employee_id = employee.id

            def fail_read(_conn, _cursor, statement, _params, _context, _many):
                if "FROM attendance_presence_segments" in statement:
                    raise OperationalError(statement, (), RuntimeError("injected failure"))

            event.listen(engine, "before_cursor_execute", fail_read)
            try:
                exc = rejects(HTTPException, lambda: employee_service.delete_employee(db, employee))
                assert exc.status_code == 503
            finally:
                event.remove(engine, "before_cursor_execute", fail_read)
            assert db.get(Employee, employee_id).name == "Retained"


@test
def test_context_and_face_encoding_bounds_and_unavailable_evidence():
    rng = np.random.default_rng(7)
    image = rng.integers(0, 256, (1600, 2400, 3), dtype=np.uint8)
    encoded = encode_evidence(before=image, after=image[:, ::-1], face=image[:112, :112])
    assert not encoded.reason_codes
    for jpeg in (encoded.before_jpeg, encoded.after_jpeg):
        assert jpeg and len(jpeg) <= 96 * 1024
        decoded = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        assert max(decoded.shape[:2]) <= 640
    assert len(encoded.face_jpeg) <= 32 * 1024
    assert cv2.imdecode(np.frombuffer(encoded.face_jpeg, np.uint8), cv2.IMREAD_COLOR).shape[:2] == (112, 112)
    missing = encode_evidence(before=np.empty((0, 0, 3), np.uint8), after=None, face=None)
    assert missing.before_jpeg is None and missing.after_jpeg is None
    assert missing.reason_codes == ("EVIDENCE_UNAVAILABLE",)
    with patch("app.services.attendance_evidence.cv2.imencode", side_effect=cv2.error("failed")):
        failed = encode_evidence(before=image, after=image, face=image[:112, :112])
        assert failed.before_jpeg is None and failed.face_jpeg is None
        assert failed.reason_codes == ("EVIDENCE_UNAVAILABLE",)


@test
def test_workplace_time_uses_fixed_offset_and_preserves_microseconds():
    stamp = datetime(2026, 11, 11, 19, 0, 0, 123456)
    assert attendance_iso(stamp) == "2026-11-12T00:00:00.123456+05:00"
    assert attendance_iso(None) is None and iso_local(None) is None
    assert day_bounds(date(2026, 11, 12)) == (datetime(2026, 11, 11, 19), datetime(2026, 11, 12, 19))
    assert utcnow().tzinfo is None
    assert iso_local(stamp) == stamp.replace(tzinfo=timezone.utc).astimezone().isoformat(timespec="seconds")


@test
def test_attendance_setting_ranges_reject_instead_of_clamping():
    for field, invalid in [
        ("attendance_utc_offset_minutes", -721), ("attendance_utc_offset_minutes", 841),
        ("attendance_ai_fps", 0), ("attendance_frame_max_age", 0),
        ("attendance_max_frame_gap", -1), ("attendance_face_threshold", 1.1),
        ("attendance_face_margin", -0.1), ("attendance_face_confirmations", 0),
        ("attendance_face_window", 0), ("attendance_face_interval", 0),
        ("attendance_face_min_size", 0), ("attendance_face_det_score", 1.1),
        ("attendance_face_blur_min", -1),
        ("attendance_live_gap_minutes", 0), ("attendance_live_gap_minutes", 241),
        ("attendance_day_gap_minutes", 0), ("attendance_day_gap_minutes", 721),
        ("attendance_ai_fps", float("nan")), ("attendance_frame_max_age", float("inf")),
    ]:
        rejects(ValidationError, lambda: Settings(_env_file=None, **{field: invalid}))
    # The documented edges are valid, and the defaults bridge 35 minutes while running, 120 for a finished day.
    assert Settings(_env_file=None).attendance_live_gap_minutes == 35
    assert Settings(_env_file=None).attendance_day_gap_minutes == 120
    edge = Settings(_env_file=None, attendance_live_gap_minutes=240, attendance_day_gap_minutes=720)
    assert (edge.attendance_live_gap_minutes, edge.attendance_day_gap_minutes) == (240, 720)
    assert Settings(_env_file=None, attendance_live_gap_minutes=1,
                    attendance_day_gap_minutes=1).attendance_day_gap_minutes == 1


@test
def test_day_gap_shorter_than_live_gap_is_rejected_with_a_named_reason():
    message = "attendance_day_gap_minutes must not be shorter than attendance_live_gap_minutes"
    for live, day in ((60, 59), (240, 120), (36, 35)):
        exc = rejects(ValidationError, lambda: Settings(
            _env_file=None, attendance_live_gap_minutes=live, attendance_day_gap_minutes=day))
        assert message in str(exc)
    # Lowering only the day gap below the default live gap is the same contradiction.
    exc = rejects(ValidationError, lambda: Settings(_env_file=None, attendance_day_gap_minutes=34))
    assert message in str(exc)
    assert Settings(_env_file=None, attendance_live_gap_minutes=60,
                    attendance_day_gap_minutes=60).attendance_day_gap_minutes == 60


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
