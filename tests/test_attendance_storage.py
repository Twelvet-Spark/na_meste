"""Attendance storage contracts. Run with .venv/Scripts/python.exe tests/test_attendance_storage.py.

Every database/backup is an isolated temporary file; production data is never opened.
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
from app.database.database import Base, immediate_session
from app.database.models import (
    AttendanceCamera, AttendanceCoverageGap, AttendanceDecision,
    AttendanceEmployeeReference, AttendanceEvent, AttendanceEvidence, AttendanceRequest,
    AttendanceRuntime, Camera, Employee, PresenceSession, WorkSession,
)
from app.services import camera_service, employee_service
from app.services.attendance_evidence import encode_evidence, evidence_expires_at, purge_evidence
from app.time_utils import attendance_iso, day_bounds, iso_local, utcnow
from test_operator_storage import LEGACY_EMPLOYEE_DDL, BASE_TABLES
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


def event_row(identifier, **kwargs):
    values = dict(id=identifier, source_key=f"camera:{identifier}", source="CAMERA",
                  camera_name="Door", recorded_at=datetime(2026, 11, 11, 4),
                  raw={"version": 1, "candidate_employee_ids": []},
                  occurred_at=datetime(2026, 11, 11, 4), status="REVIEW")
    values.update(kwargs)
    return AttendanceEvent(**values)


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


@test
def test_fresh_seed_and_idempotence():
    with database() as (engine, root):
        with Session(engine) as db:
            runtime = db.get(AttendanceRuntime, 1)
            assert runtime.schema_version == 1 and runtime.gallery_revision == 0
            assert runtime.monitoring_started_at is None and runtime.commission_id is None
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
        with engine.begin() as connection:
            connection.exec_driver_sql(LEGACY_EMPLOYEE_DDL)
            connection.exec_driver_sql("CREATE INDEX ix_employees_external_id ON employees (external_id)")
            for table in Base.metadata.sorted_tables:
                if table.name in BASE_TABLES:
                    table.create(connection)
        start = datetime(2026, 11, 10, 12, 3, 4, 123456)
        with Session(engine) as db:
            cam = camera()
            db.add(cam)
            db.flush()
            db.connection().exec_driver_sql(
                "INSERT INTO employees (id,name,active,created_at,updated_at) VALUES (1,'Historical',1,?,?)",
                (start, start))
            db.add(PresenceSession(employee_id=1, camera_id=cam.id,
                                   started_at=start, last_seen_at=start, ended_at=None))
            db.commit()
        schema.initialize_attendance_schema(engine)
        backups = list((root / "backups").glob("pre-attendance-*.db"))
        assert len(backups) == 1
        with closing(sqlite3.connect(backups[0])) as backup:
            assert backup.execute("PRAGMA integrity_check").fetchone() == ("ok",)
            assert backup.execute("SELECT ended_at FROM presence_sessions").fetchone() == (None,)
            assert backup.execute("SELECT name FROM employees").fetchone() == ("Historical",)
            assert not backup.execute("SELECT name FROM sqlite_master WHERE name='attendance_runtime'").fetchall()
        with Session(engine) as db:
            old = db.query(PresenceSession).one()
            assert old.started_at == start and old.ended_at is None
            assert db.query(WorkSession).count() == 0
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
    with database(False) as (engine, _root):
        AttendanceRequest.__table__.create(engine)
        rejects(RuntimeError, lambda: schema.initialize_attendance_schema(engine))
        assert inspect(engine).get_table_names() == ["attendance_requests"]
    with database() as (engine, _root):
        with engine.begin() as connection:
            connection.exec_driver_sql("DROP INDEX uq_work_sessions_open_employee")
            connection.exec_driver_sql("CREATE UNIQUE INDEX uq_work_sessions_open_employee ON work_sessions(employee_id) WHERE ended_at IS NOT NULL")
        rejects(RuntimeError, lambda: schema.initialize_attendance_schema(engine))
    with database() as (engine, _root):
        with engine.begin() as connection:
            connection.exec_driver_sql("DROP TABLE attendance_decisions")
            connection.exec_driver_sql("CREATE TABLE attendance_decisions (id INTEGER PRIMARY KEY)")
        rejects(RuntimeError, lambda: schema.initialize_attendance_schema(engine))


@test
def test_malformed_checks_and_fk_actions_refused():
    for old, new in [("singleton = 1", "singleton >= 1"),
                     ("ON DELETE RESTRICT", "ON DELETE CASCADE")]:
        with database() as (engine, _root):
            table = "attendance_coverage_gaps" if old.startswith("singleton") else "attendance_employee_references"
            with engine.begin() as connection:
                connection.exec_driver_sql("PRAGMA writable_schema=ON")
                connection.exec_driver_sql(
                    "UPDATE sqlite_master SET sql=replace(sql, ?, ?) WHERE type='table' AND name=?",
                    (old, new, table))
                connection.exec_driver_sql("PRAGMA writable_schema=OFF")
            rejects(RuntimeError, lambda: schema.initialize_attendance_schema(engine))


@test
def test_altered_enum_literal_refused_without_mutation():
    with database() as (engine, _root):
        with engine.begin() as connection:
            table_sql = connection.exec_driver_sql(
                "SELECT sql FROM sqlite_master WHERE name='attendance_cameras'").scalar_one()
            index_sql = connection.exec_driver_sql(
                "SELECT sql FROM sqlite_master WHERE name='uq_attendance_cameras_enabled_role'").scalar_one()
            trigger_sql = connection.exec_driver_sql(
                "SELECT sql FROM sqlite_master WHERE name='attendance_camera_role_retained'").scalar_one()
            assert "'ENTRY'" in table_sql
            connection.exec_driver_sql("DROP TABLE attendance_cameras")
            connection.exec_driver_sql(table_sql.replace("'ENTRY'", "'entry'"))
            connection.exec_driver_sql(index_sql)
            # DROP TABLE also removes its trigger; restore it so only the
            # case-sensitive enum literal differs from the recognized schema.
            connection.exec_driver_sql(trigger_sql)
        with engine.connect() as connection:
            before = tuple(connection.connection.driver_connection.iterdump())
        exc = rejects(RuntimeError, lambda: schema.initialize_attendance_schema(engine))
        assert "table attendance_cameras differs from version 1" in str(exc)
        with engine.connect() as connection:
            assert tuple(connection.connection.driver_connection.iterdump()) == before


@test
def test_quoted_partial_index_predicate_refused_without_mutation():
    with database() as (engine, _root):
        with engine.begin() as connection:
            connection.exec_driver_sql("DROP INDEX uq_attendance_cameras_enabled_role")
            # SQLite treats this unknown double-quoted name as a string literal,
            # not as the enabled-column comparison. Removing quotes hides that.
            connection.exec_driver_sql(
                'CREATE UNIQUE INDEX uq_attendance_cameras_enabled_role '
                'ON attendance_cameras(role) WHERE "enabled = 1"')
        with engine.connect() as connection:
            before = tuple(connection.connection.driver_connection.iterdump())
        exc = rejects(RuntimeError, lambda: schema.initialize_attendance_schema(engine))
        assert "index uq_attendance_cameras_enabled_role differs from version 1" in str(exc)
        with engine.connect() as connection:
            assert tuple(connection.connection.driver_connection.iterdump()) == before


@test
def test_ddl_interruption_rolls_back_entire_install():
    with database(False) as (engine, _root):
        def interrupt(_conn, _cursor, statement, _parameters, _context, _many):
            if statement.lstrip().startswith("CREATE TABLE attendance_events"):
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
def test_source_and_session_uniqueness_and_history_fks():
    with database() as (engine, _root):
        with Session(engine) as db:
            employee, cam = Employee(name="Worker"), camera()
            db.add_all([employee, cam])
            db.flush()
            employee_id, camera_id = employee.id, cam.id
            db.add_all([event_row("entry", employee_id=employee_id, camera_id=camera_id),
                        event_row("entry2", employee_id=employee_id)])
            db.commit()
        rejects_integrity(engine, event_row("duplicate", source_key="camera:entry"))
        with Session(engine) as db:
            db.add(WorkSession(entry_event_id="entry", employee_id=employee_id,
                               employee_name="Worker", started_at=datetime(2026, 11, 11, 4), status="OPEN"))
            db.commit()
        rejects_integrity(engine, WorkSession(entry_event_id="entry2", employee_id=employee_id,
                          employee_name="Worker", started_at=datetime(2026, 11, 11, 5), status="OPEN"))
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
def test_configuration_and_gap_constraints():
    with database() as (engine, _root):
        with Session(engine) as db:
            db.add_all([camera(id=1), camera(id=2)])
            db.commit()
        rejects_integrity(engine, AttendanceCamera(camera_id=1, role="ENTRY", enabled=True))
        with Session(engine) as db:
            db.add(AttendanceCamera(camera_id=1, role="ENTRY", enabled=True,
                   line_start=[0.1, 0.5], line_end=[0.9, 0.5], inside_point=[0.5, 0.9],
                   reference_width=1280, reference_height=720))
            db.add(AttendanceCoverageGap(started_at=datetime(2026, 11, 11), reason_codes=["FRAME_GAP"]))
            db.commit()
        rejects_integrity(engine, AttendanceCamera(camera_id=2, role="ENTRY", enabled=True,
            line_start=[0.1, 0.5], line_end=[0.9, 0.5], inside_point=[0.5, 0.9],
            reference_width=1280, reference_height=720))
        rejects_integrity(engine, AttendanceCoverageGap(started_at=datetime(2026, 11, 11)))
        rejects_integrity(engine, AttendanceCoverageGap(singleton=2, started_at=datetime(2026, 11, 11)))


@test
def test_decision_targets_revision_uniqueness_and_references_survive_mutation():
    with database() as (engine, _root):
        with Session(engine) as db:
            employee = Employee(name="Original")
            db.add(employee)
            db.flush()
            employee_id = employee.id
            db.add(AttendanceRequest(id="request", operation="DECISION", payload={}, result={}))
            db.add(event_row("review", employee_id=employee_id))
            db.flush()
            db.add(AttendanceEmployeeReference(event_id="review", employee_id=employee_id, employee_name="Original"))
            db.add(AttendanceDecision(request_id="request", event_id="review", revision=1,
                                     actor="Operator", reason="Visitor", before={}, after={}))
            db.commit()
            saved = db.get(AttendanceEvent, "review")
            saved.status, saved.employee_id = "VISITOR", None
            db.commit()
            exc = rejects(HTTPException, lambda: employee_service.delete_employee(db, employee))
            assert exc.status_code == 409
        rejects_integrity(engine, AttendanceDecision(request_id="request", revision=2, actor="Op", reason="Reason", before={}, after={}))
        rejects_integrity(engine, AttendanceDecision(request_id="request", event_id="review", revision=1, actor="Op", reason="Reason", before={}, after={}))
        rejects_integrity(engine, AttendanceDecision(request_id="request", event_id="review", revision=2, actor="", reason="Reason", before={}, after={}))


@test
def test_legacy_deletion_protection_and_deactivation_revision():
    with database() as (engine, _root):
        with Session(engine) as db:
            employee = create_identity(db)
            cam = camera_service.create_camera(db, camera_service.CameraCreate(
                name="Door", nvr_host="unused", username="u", password="p", channel=1, stream_type="main"))
            employee_id, camera_id = employee.id, cam.id
            db.add(PresenceSession(employee_id=employee_id, camera_id=camera_id, ended_at=None))
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
            assert db.query(PresenceSession).one().ended_at is None
            assert db.query(AttendanceEvent).count() == 0


@test
def test_work_history_alone_protects_employee_and_assigned_camera_retained():
    with database() as (engine, _root):
        with Session(engine) as db:
            employee, cam = Employee(name="Worker"), camera()
            db.add_all([employee, cam])
            db.commit()
            db.add(AttendanceCamera(camera_id=cam.id, role="EXIT", enabled=False))
            db.add(event_row("entry"))
            db.flush()
            db.add(WorkSession(entry_event_id="entry", employee_id=employee.id, employee_name="Worker",
                               started_at=datetime(2026, 11, 11), status="OPEN"))
            db.commit()
            rejects(HTTPException, lambda: employee_service.delete_employee(db, employee))
            rejects(HTTPException, lambda: camera_service.delete_camera(db, cam))
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
            cam = camera()
            db.add(cam)
            db.flush()
            db.add(event_row("crossing", camera_id=cam.id))
            db.commit()
            class RunningReader:
                stopped = False
                def stop(self, _camera_id):
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
                if "FROM presence_sessions" not in statement:
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
            assert db.query(PresenceSession).count() == 0


@test
def test_database_failure_returns_503_without_deleting_history():
    with database() as (engine, _root):
        with Session(engine) as db:
            employee = Employee(name="Retained")
            db.add(employee)
            db.commit()
            employee_id = employee.id

            def fail_read(_conn, _cursor, statement, _params, _context, _many):
                if "FROM presence_sessions" in statement:
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
def test_expiry_clears_only_images_even_when_review_unresolved():
    now = datetime(2026, 11, 11, 12)
    assert evidence_expires_at(now) == datetime(2026, 12, 11, 12)
    with database() as (engine, _root):
        with Session(engine) as db:
            employee = Employee(name="Retained history")
            db.add(employee)
            db.flush()
            db.add_all([event_row("old"), event_row("new", status="ACCEPTED",
                        employee_id=employee.id, employee_name=employee.name, direction="ENTRY")])
            db.add(AttendanceRequest(id="audit", operation="DECISION", payload={}, result={}))
            db.flush()
            db.add(AttendanceDecision(request_id="audit", event_id="old", revision=1,
                   actor="Operator", reason="Recorded review", before={}, after={}))
            db.add(WorkSession(entry_event_id="new", employee_id=employee.id,
                   employee_name=employee.name, started_at=now, status="OPEN"))
            db.add_all([
                AttendanceEvidence(event_id="old", before_jpeg=b"old", after_jpeg=b"old", face_jpeg=b"old", expires_at=now),
                AttendanceEvidence(event_id="new", before_jpeg=b"new", expires_at=now + timedelta(seconds=1)),
            ])
            db.commit()
        assert purge_evidence(engine, now) == 1
        assert purge_evidence(engine, now) == 0
        with Session(engine) as db:
            old = db.get(AttendanceEvidence, "old")
            assert old.before_jpeg is None and old.after_jpeg is None and old.face_jpeg is None
            assert old.purged_at == now
            assert db.get(AttendanceEvent, "old").status == "REVIEW"
            assert db.get(AttendanceEvidence, "new").before_jpeg == b"new"
            assert db.query(AttendanceEvent).count() == 2
            assert db.query(AttendanceDecision).count() == 1
            assert db.query(WorkSession).count() == 1


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
        ("attendance_face_blur_min", -1), ("attendance_evidence_keep_days", 0),
        ("attendance_ai_fps", float("nan")), ("attendance_frame_max_age", float("inf")),
    ]:
        rejects(ValidationError, lambda: Settings(_env_file=None, **{field: invalid}))


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
