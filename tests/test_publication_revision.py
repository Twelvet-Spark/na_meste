"""Publication fencing and durable gallery admission on isolated SQLite.

Run: .venv/Scripts/python.exe tests/test_publication_revision.py
Only model execution and an injected gallery-read failure are doubled. No live
camera, production database, or model assets are opened.
"""
import sys
import threading
import traceback
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi import FastAPI
from sqlalchemy.orm import Session

from app.ai.recognition_service import RecognitionService
from app.api import employees
from app.database import database as transactions
from app.database.database import get_db, immediate_session, publication_guard
from app.database.models import AttendanceRuntime, Camera, Employee
from app.services import camera_service, employee_service
from operator_test_support import Client
from test_attendance_recognition import database, load_store
from test_batch_recognition import Detector, Faces, confirmed, process

TESTS = []


def test(fn):
    TESTS.append(fn)
    return fn


@contextmanager
def blocked_mutation(engine, mutation):
    """Observe the real acquisition boundary, not a scheduler-dependent sleep."""
    reached, finished = threading.Event(), threading.Event()
    failures = []

    @contextmanager
    def observed_guard(bind):
        if threading.current_thread() is thread:
            reached.set()
        with publication_guard(bind):
            yield

    def run():
        try:
            mutation()
        except BaseException as exc:
            failures.append(exc)
        finally:
            finished.set()

    thread = threading.Thread(target=run, daemon=True)
    try:
        with publication_guard(engine):
            with patch.object(transactions, "publication_guard", observed_guard):
                thread.start()
                assert reached.wait(5), "Mutation did not opt into publication fencing"
                assert not finished.is_set(), "Mutation escaped the held publication guard"
                yield
    finally:
        thread.join(5)
        assert not thread.is_alive(), "Mutation did not resume after guard release"
    if failures:
        raise failures[0]
    assert finished.is_set()


@test
def test_source_mutation_waits_but_ordinary_writer_commits_under_publication_guard():
    with database() as engine:
        with Session(engine) as db:
            source = camera_service.create_camera(db, camera_service.CameraCreate(
                name="Test", nvr_host="127.0.0.1", username="test", password="test"))

        def mutate():
            with Session(engine) as db:
                camera_service.update_camera(db, db.get(Camera, source.id),
                                             camera_service.CameraUpdate(channel=2))

        with blocked_mutation(engine, mutate):
            # No SQL write transaction is held by publication_guard itself.
            with immediate_session(engine) as db:
                assert db.get(Camera, source.id).channel == 1
                db.get(Employee, 2).external_id = "ordinary-writer-committed"
            with Session(engine) as db:
                assert db.get(Employee, 2).external_id == "ordinary-writer-committed"
        with Session(engine) as db:
            current = camera_service.camera_snapshot(db, source.id)
            assert current.channel == 2
            assert current.source_revision == source.source_revision + 1


@test
def test_employee_gallery_commit_waits_for_publication_guard():
    with database() as engine:
        def mutate():
            with Session(engine) as db:
                employee = db.get(Employee, 1)
                employee_service.update_employee(db, employee, employee_service.EmployeeUpdate(
                    expected_revision=employee.revision, active=False))

        with blocked_mutation(engine, mutate):
            with Session(engine) as db:
                assert db.get(Employee, 1).active
                assert db.get(AttendanceRuntime, 1).gallery_revision == 7
        with Session(engine) as db:
            assert not db.get(Employee, 1).active
            assert db.get(AttendanceRuntime, 1).gallery_revision == 8


@test
def test_guard_is_reentrant_per_engine_and_failed_publish_rolls_back():
    with database() as first, database() as second:
        errors, finished = [], threading.Event()

        def other_engine():
            try:
                with immediate_session(second, publish=True) as db:
                    db.get(Employee, 1).external_id = "other-engine"
            except BaseException as exc:
                errors.append(exc)
            finally:
                finished.set()

        with publication_guard(first):
            with publication_guard(first):
                with immediate_session(first, publish=True) as db:
                    db.get(Employee, 1).external_id = "nested"
            thread = threading.Thread(target=other_engine, daemon=True)
            thread.start()
            assert finished.wait(5), "Unrelated engine shared a publication lock"
            thread.join(5)
        assert not errors
        try:
            with immediate_session(first, publish=True) as db:
                db.get(Employee, 1).external_id = "must-rollback"
                db.commit()  # Session savepoint is not the durable commit.
                raise ValueError("injected failure")
        except ValueError:
            pass
        with immediate_session(first, publish=True) as db:
            assert db.get(Employee, 1).external_id == "nested"
        with Session(second) as db:
            assert db.get(Employee, 1).external_id == "other-engine"


@test
def test_background_write_transaction_does_not_hold_publication_guard():
    with database() as engine:
        acquired, release, errors = threading.Event(), threading.Event(), []

        def publish():
            try:
                with publication_guard(engine):
                    acquired.set()
                    assert release.wait(5)
            except BaseException as exc:
                errors.append(exc)

        with immediate_session(engine) as db:
            db.get(Employee, 1).external_id = "background"
            db.flush()
            thread = threading.Thread(target=publish, daemon=True)
            thread.start()
            try:
                assert acquired.wait(5), "Background SQLite writer blocked publication"
            finally:
                release.set()
                thread.join(5)
        assert not errors and not thread.is_alive()


@test
def test_stale_durable_gallery_clears_votes_and_recovers_with_bounded_refresh():
    with database() as engine:
        store, faces = load_store(engine), Faces()
        service = RecognitionService(Detector(), faces, store, min_hits=1)
        service.set_expected_gallery_revision(7)
        confirmed(service)
        with Session(engine) as db:
            employee = db.get(Employee, 1)
            employee_service.update_employee(db, employee, employee_service.EmployeeUpdate(
                expected_revision=employee.revision, first_name="New", last_name="Name"))
        assert store.revision == 7
        service.set_expected_gallery_revision(8)
        with patch.object(store, "refresh", side_effect=RuntimeError("injected read failure")):
            assert not service.refresh_gallery(engine, now=100.)
        result = process(service, 4, 11.5)
        assert len(result.observed_tracks) == 1
        assert result.observed_tracks[0].employee_id is None
        assert result.observed_tracks[0].identity_reason_codes == ("GALLERY_UNAVAILABLE",)
        assert result.observed_tracks[0].fresh_face_evidence is None
        # Restore the real reader: a too-early retry must not publish revision 8.
        assert not service.refresh_gallery(engine, now=100.999)
        assert store.revision == 7
        assert service.refresh_gallery(engine, now=101.)
        assert store.revision == 8
        for seq, mono in ((5, 12.), (6, 12.5)):
            result = process(service, seq, mono).observed_tracks[0]
            assert result.employee_id is None
        recovered = process(service, 7, 13.).observed_tracks[0]
        assert recovered.employee_id == 1 and recovered.employee_name == "Name New"
        assert recovered.fresh_face_evidence.supporting_frame_sequences == (5, 6, 7)
        assert recovered.fresh_face_evidence.store_revision == 8


@test
def test_cached_packet_cannot_keep_identity_after_durable_gallery_change():
    with database() as engine:
        service = RecognitionService(Detector(), Faces(), load_store(engine), min_hits=1)
        service.set_expected_gallery_revision(7)
        confirmed(service)
        service.set_expected_gallery_revision(8)
        replay = process(service, 3, 11.)
        assert replay.observed_tracks == []
        assert replay.detections[0].employee_id is None
        assert replay.detections[0].identity_reason_codes == ("GALLERY_UNAVAILABLE",)


@test
def test_missing_face_or_gallery_preserves_first_hit_and_unknown_people():
    with database() as engine:
        for faces, store, reason in (
                (None, load_store(engine), "FACE_MODEL_UNAVAILABLE"),
                (Faces(), None, "GALLERY_UNAVAILABLE"),
                (None, None, "FACE_MODEL_UNAVAILABLE")):
            service = RecognitionService(Detector(), faces, store, min_hits=2)
            service.set_expected_gallery_revision(7)
            first = process(service, 1, 10.)
            assert len(first.observed_tracks) == 1 and first.detections == []
            second = process(service, 2, 10.5)
            assert len(second.observed_tracks) == len(second.detections) == 1
            result = second.detections[0]
            assert result.employee_id is None and result.identity_reason_codes == (reason,)


@test
def test_fresh_face_evidence_is_available_for_ordinary_and_attendance_only_when_attempted():
    with database() as engine:
        for attendance in (False, True):
            service = RecognitionService(Detector(), Faces(), load_store(engine), min_hits=1)
            service.set_expected_gallery_revision(7)
            first = process(service, 1, 10., attendance=attendance).observed_tracks[0]
            assert first.fresh_face_evidence.frame_seq == 1
            assert first.fresh_face_evidence.aligned_jpeg
            assert first.visit_id is None
            cached = process(service, 2, 10.01, attendance=attendance).observed_tracks[0]
            assert cached.fresh_face_evidence is None
            replay = process(service, 2, 10.01, attendance=attendance)
            assert not replay.observed_tracks and replay.detections[0].fresh_face_evidence is None


@test
def test_employee_http_availability_requires_durable_gallery_revision():
    with database() as engine:
        store, faces = load_store(engine), Faces()
        app = FastAPI()
        app.state.face_model_id = faces.model_id
        app.state.face_engine = faces
        app.state.embedding_store = store
        app.state.ai_worker = SimpleNamespace(service=SimpleNamespace(faces=faces, store=store))
        app.include_router(employees.router)

        def isolated_db():
            with Session(engine) as db:
                yield db

        app.dependency_overrides[get_db] = isolated_db
        client = Client(app)
        assert client.json("GET", "/api/employees/1")["face_runtime_available"]
        with Session(engine) as db:
            employee = db.get(Employee, 1)
            employee_service.update_employee(db, employee, employee_service.EmployeeUpdate(
                expected_revision=employee.revision, active=False))
        assert not client.json("GET", "/api/employees/1")["face_runtime_available"]
        assert all(not row["face_runtime_available"] for row in client.json("GET", "/api/employees"))
        with Session(engine) as db:
            store.refresh(db)
        assert client.json("GET", "/api/employees/1")["face_runtime_available"]


if __name__ == "__main__":
    failures = 0
    for fn in TESTS:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception:
            failures += 1
            print(f"FAIL {fn.__name__}")
            traceback.print_exc()
    print(f"{len(TESTS) - failures}/{len(TESTS)} passed")
    raise SystemExit(bool(failures))
