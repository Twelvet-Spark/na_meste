"""Read-only historical global archive and pure matcher isolation regressions.

Run: .venv/Scripts/python.exe tests/test_global_persons.py
Appearance proposal/acknowledgement policy is covered by test_global_proposals.py.
"""
import sys
import traceback
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from uuid import NAMESPACE_URL, uuid5

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import numpy as np
from fastapi import HTTPException
from sqlalchemy import create_engine, event, select
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.ai.global_tracker import EmbeddingItem, GlobalIdentityManager
from app.api import global_persons as api
from app.database.database import Base
from app.database import models as db_models  # noqa: F401 — регистрация таблиц
from app.database.models import (
    Camera, Employee, GlobalEvent, GlobalObservation, GlobalPerson,
)
from app.time_utils import utcnow


def archive_vector():
    vector = np.zeros(512, dtype=np.float32)
    vector[0] = 1.
    return vector


def appearance_item(*, camera_id=1, generation=1, now=1., embedding=None):
    return EmbeddingItem(
        visit_id=str(uuid5(NAMESPACE_URL, f"archive-test/{camera_id}/{generation}")),
        camera_id=camera_id,
        tracking_generation=str(uuid5(NAMESPACE_URL, f"archive-generation/{camera_id}/{generation}")),
        track_id=1, source_revision=1, stream_id=f"archive-stream-{camera_id}-{generation}",
        packet_seq=1, observed_at=utcnow(), received_mono=now,
        bbox=(.3, .2, .45, .85), aspect=3., crop_quality=.9,
        embedding=archive_vector() if embedding is None else embedding, employee_id=None,
    )


def link_ack(proposal, global_id):
    return SimpleNamespace(
        kind="LINK_COMMITTED", visit_id=proposal.visit_id, camera_id=proposal.camera_id,
        tracking_generation=proposal.tracking_generation, track_id=proposal.track_id,
        source_revision=proposal.source_revision, stream_id=proposal.stream_id,
        packet_seq=proposal.packet_seq, link_key=proposal.link_key, global_id=global_id,
        reason=None,
    )


_engine = create_engine(
    "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool,
)
SessionFactory = sessionmaker(bind=_engine, expire_on_commit=False)


def fresh_archive():
    """Seed legacy read-only history explicitly; the matcher no longer writes it."""
    Base.metadata.drop_all(_engine)
    Base.metadata.create_all(_engine)
    stamp = utcnow()
    ok, encoded = cv2.imencode(".jpg", np.zeros((80, 80, 3), dtype=np.uint8))
    assert ok
    snapshot = encoded.tobytes()
    with SessionFactory() as db:
        db.add_all([
            Camera(id=1, name="Камера 1", nvr_host="h", username="u", password="p", channel=1),
            Camera(id=2, name="Камера 2", nvr_host="h", username="u", password="p", channel=2),
            Employee(id=7, name="Иван Петров"),
        ])
        db.flush()
        db.add_all([
            GlobalPerson(id=1, status="ACTIVE", last_camera_id=2, last_track_id=3,
                         created_at=stamp, last_seen_at=stamp + timedelta(seconds=6)),
            GlobalPerson(id=2, status="NEW", employee_id=7, last_camera_id=2, last_track_id=2,
                         created_at=stamp + timedelta(seconds=5),
                         last_seen_at=stamp + timedelta(seconds=5)),
        ])
        db.flush()
        for identifier, global_id, camera_id, track_id, seconds in (
                (1, 1, 1, 1, 0), (2, 1, 2, 3, 6), (3, 2, 2, 2, 5)):
            db.add(GlobalObservation(
                id=identifier, global_id=global_id, camera_id=camera_id, track_id=track_id,
                embedding=archive_vector().tobytes(), snapshot=snapshot,
                created_at=stamp + timedelta(seconds=seconds)))
        db.add(GlobalEvent(
            global_id=1, camera_id=2, track_id=3, event_type="camera_transition",
            payload='{"from_camera":1,"to_camera":2}', created_at=stamp + timedelta(seconds=6)))
        db.commit()


def expect_http(code: int, fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except HTTPException as exc:
        assert exc.status_code == code, f"ожидали HTTP {code}, получили {exc.status_code}"
        return
    raise AssertionError(f"ожидали HTTP {code}, вызов прошёл без ошибки")


# ------------------------------------------------------------------- тесты

TESTS = []


def test(fn):
    TESTS.append(fn)
    return fn



@test
def test_list_filters_and_fields():
    fresh_archive()

    with SessionFactory() as db:
        rows = api.api_list_global_persons(db=db)
        assert [r["global_id"] for r in rows] == [1, 2], "свежие сверху"
        by_id = {r["global_id"]: r for r in rows}
        assert by_id[1]["has_photo"] is True
        assert by_id[1]["observations_count"] == 2
        assert by_id[2]["employee_name"] == "Иван Петров"
        assert by_id[2]["last_camera_name"] == "Камера 2"

        rows = api.api_list_global_persons(status="ACTIVE", db=db)
        assert {r["global_id"] for r in rows} == {1}
        rows = api.api_list_global_persons(status="NEW", db=db)
        assert {r["global_id"] for r in rows} == {2}
        rows = api.api_list_global_persons(employee_id=7, db=db)
        assert {r["global_id"] for r in rows} == {2}
        rows = api.api_list_global_persons(camera_id=2, db=db)
        assert {r["global_id"] for r in rows} == {1, 2}, "обе личности сейчас на камере 2"
        rows = api.api_list_global_persons(camera_id=1, db=db)
        assert rows == []
        rows = api.api_list_global_persons(search="G#2", db=db)
        assert {r["global_id"] for r in rows} == {2}
        rows = api.api_list_global_persons(search="иван", db=db)
        assert {r["global_id"] for r in rows} == {2}, "поиск по имени без учёта регистра"
        rows = api.api_list_global_persons(search="нет такого", db=db)
        assert rows == []

        single = api.api_get_global_person(1, db=db)
        assert single["global_id"] == 1 and single["has_photo"] is True
        expect_http(404, api.api_get_global_person, 999, db=db)


@test
def test_photo_endpoints():
    fresh_archive()
    with SessionFactory() as db:
        db.add(GlobalPerson(id=99, status="LOST"))
        db.commit()

    with SessionFactory() as db:
        response = api.api_global_person_photo(1, db=db)
        assert response.status_code == 200 and len(response.body) > 100, \
            "аватар — последний снимок личности"
        expect_http(404, api.api_global_person_photo, 99, db=db)


@test
def test_timeline_names_and_trajectory():
    fresh_archive()

    with SessionFactory() as db:
        items = api.api_global_timeline(1, db=db)
        transitions = [i for i in items if i.get("event_type") == "camera_transition"]
        assert len(transitions) == 1
        assert transitions[0]["from_camera_name"] == "Камера 1"
        assert transitions[0]["to_camera_name"] == "Камера 2"
        observations = [i for i in items if i["kind"] == "observation"]
        assert len(observations) == 2
        assert all(o["has_snapshot"] for o in observations)

        trajectory = api.api_global_trajectory(1, db=db)
        assert trajectory["chain"] == ["Камера 1", "Камера 2"]
        assert trajectory["segments"][0]["observations"] == 1

        obs_list = api.api_global_observations(1, db=db)
        assert len(obs_list) == 2


@contextmanager
def archived_database():
    """Real file SQLite/FKs with independent person, observation and event links."""
    from app.database.database import immediate_session
    from app.database.models import AttendanceCamera
    from attendance_test_support import database

    with database() as (engine, service):
        vector = archive_vector()
        recent, old = utcnow(), utcnow() - timedelta(days=999)
        with immediate_session(engine) as db:
            for identifier in (11, 12, 13, 14):
                db.add(Camera(id=identifier, name=f"Archive camera {identifier}",
                              nvr_host="unused", username="u", password="p",
                              channel=identifier, enabled=False))
            db.flush()
            db.add_all([AttendanceCamera(camera_id=identifier) for identifier in (11, 12, 13, 14)])
            db.add_all([
                GlobalPerson(id=41, employee_id=1, last_camera_id=11, last_track_id=8,
                             status="ACTIVE", created_at=recent, last_seen_at=recent),
                GlobalPerson(id=9001, status="LOST", created_at=old, last_seen_at=old),
            ])
            db.flush()
            db.add_all([
                GlobalObservation(id=51, global_id=41, camera_id=12, track_id=8,
                                  embedding=vector.tobytes(), snapshot=b"archived-photo",
                                  created_at=recent),
                GlobalObservation(id=52, global_id=9001, camera_id=12, track_id=9,
                                  embedding=vector.tobytes(), snapshot=b"old-photo",
                                  created_at=old),
                GlobalEvent(id=61, global_id=41, camera_id=13, track_id=8,
                            event_type="person_seen", payload='{"score":0.9}', created_at=recent),
                GlobalEvent(id=62, global_id=9001, camera_id=13, track_id=9,
                            event_type="person_lost", payload='{"duration_seconds":10}',
                            created_at=old),
            ])
        yield engine, service, vector


def archive_rows(engine):
    """All persisted column values, including bytes, links and timestamps."""
    with engine.connect() as connection:
        return tuple(tuple(connection.execute(
            select(model.__table__).order_by(model.id)).all())
            for model in (GlobalPerson, GlobalObservation, GlobalEvent))


def archive_client(engine, service):
    from fastapi import FastAPI
    from app.api import cameras, employees
    from app.database.database import get_db
    from app.rtsp.manager import ReaderManager
    from operator_test_support import Client

    app = FastAPI()
    app.state.presence = service
    app.state.manager = ReaderManager()
    for router in (api.router, cameras.router, employees.router):
        app.include_router(router)

    def isolated_db():
        with Session(engine) as db:
            yield db

    app.dependency_overrides[get_db] = isolated_db
    return Client(app)


def assert_camera_archive_guard(camera_id):
    with archived_database() as (engine, service, _reid):
        client = archive_client(engine, service)
        before = archive_rows(engine)
        # Reader stop is the external side effect: a rejected delete must never
        # reach it, irrespective of which historical relation protected the row.
        with patch.object(client.app.state.manager, "stop") as stop:
            client.json("DELETE", f"/api/cameras/{camera_id}", status=409)
            stop.assert_not_called()
        assert archive_rows(engine) == before
        with Session(engine) as db:
            assert db.get(Camera, camera_id) is not None


@test
def test_camera_with_only_global_person_link_cannot_be_deleted():
    assert_camera_archive_guard(11)


@test
def test_camera_with_only_global_observation_link_cannot_be_deleted():
    assert_camera_archive_guard(12)


@test
def test_camera_with_only_global_event_reference_cannot_be_deleted():
    assert_camera_archive_guard(13)


@test
def test_camera_referenced_only_in_global_event_payload_cannot_be_deleted():
    import json
    from app.database.database import immediate_session

    for key, event_type in (("from_camera", "camera_transition"),
                            ("to_camera", "camera_transition"), ("last_camera", "person_lost")):
        with archived_database() as (engine, service, _reid):
            with immediate_session(engine) as db:
                db.add(GlobalEvent(global_id=41, camera_id=None, event_type=event_type,
                                   payload=json.dumps({key: 14})))
            before = archive_rows(engine)
            client = archive_client(engine, service)
            with patch.object(client.app.state.manager, "stop") as stop:
                client.json("DELETE", "/api/cameras/14", status=409)
                stop.assert_not_called()
            assert archive_rows(engine) == before
            with Session(engine) as db:
                assert db.get(Camera, 14) is not None


@test
def test_employee_with_only_global_assignment_cannot_be_deleted():
    with archived_database() as (engine, service, _reid):
        client = archive_client(engine, service)
        before = archive_rows(engine)
        client.json("DELETE", "/api/employees/1", status=409)
        assert archive_rows(engine) == before
        with Session(engine) as db:
            assert db.get(Employee, 1) is not None


@test
def test_unused_camera_and_employee_remain_deletable_with_global_archive():
    with archived_database() as (engine, service, _reid):
        client = archive_client(engine, service)
        before = archive_rows(engine)
        client.json("DELETE", "/api/cameras/14", status=204)
        client.json("DELETE", "/api/employees/2", status=204)
        assert archive_rows(engine) == before
        with Session(engine) as db:
            assert db.get(Camera, 14) is None and db.get(Employee, 2) is None


@test
def test_global_archive_rejects_mutations_and_preserves_gets_and_photos():
    with archived_database() as (engine, service, _reid):
        client = archive_client(engine, service)
        before = archive_rows(engine)
        for method, path in (("POST", "/41/merge/9001"), ("PUT", "/41/employee/2"),
                             ("DELETE", "/41/employee")):
            status, _headers, _body = client.request(method, "/api/global-persons" + path)
            assert status in (404, 405), (method, path, status)
            assert archive_rows(engine) == before
        rows = client.json("GET", "/api/global-persons?limit=1")
        assert len(rows) == 1 and rows[0]["global_id"] == 41
        detail = client.json("GET", "/api/global-persons/41")
        assert detail["employee_id"] == 1 and detail["last_camera_id"] == 11
        observations = client.json("GET", "/api/global-persons/41/observations?limit=1")
        assert len(observations) == 1 and observations[0]["id"] == 51
        timeline = client.json("GET", "/api/global-persons/41/timeline")
        assert {row["kind"] for row in timeline} == {"observation", "event"}
        for path in ("/41/photo", "/41/observations/51/photo"):
            status, headers, body = client.request("GET", "/api/global-persons" + path)
            assert status == 200 and body == b"archived-photo"
            assert headers[b"content-type"].startswith(b"image/jpeg")
        assert archive_rows(engine) == before


@test
def test_restart_never_loads_archived_vectors_or_allocates_an_integer_id():
    with archived_database() as (engine, _service, vector):
        before = archive_rows(engine)
        manager = GlobalIdentityManager()
        assert manager.snapshot() == [], "Persisted identities are archive, not a live gallery"
        value = appearance_item(embedding=vector)
        proposal = manager.process_embeddings([value])[0]
        assert proposal.global_id is None and manager.global_id_for(value.visit_id) is None
        # Stand in only for the persistence boundary: SQLite, not this manager,
        # allocates the actual parent ID. Full LINKED atomicity has journal tests.
        with Session(engine) as db:
            parent = GlobalPerson(status="ACTIVE")
            db.add(parent)
            db.flush()
            assigned_id = parent.id
            db.commit()
        assert assigned_id == 9002
        manager.acknowledge(link_ack(proposal, assigned_id))
        assert manager.global_id_for(value.visit_id) == assigned_id
        manager.drop_camera(1)
        manager.sweep(100_000.)
        for original, current in zip(before, archive_rows(engine)):
            assert all(row in current for row in original), "Startup archive changed"
        frozen = archive_rows(engine)
        restarted = GlobalIdentityManager()
        new = restarted.process_embeddings([appearance_item(generation=2, now=100_001.)])[0]
        assert new.global_id is None and restarted.snapshot() == []
        assert archive_rows(engine) == frozen


@test
def test_idle_worker_never_retention_purges_persisted_global_archive():
    from app.ai.worker import AIWorker, DetectionState
    from app.rtsp.manager import ReaderManager

    with archived_database() as (engine, attendance, _vector):
        before = archive_rows(engine)
        manager = GlobalIdentityManager()
        # No readers means no model invocation. Run the real worker maintenance
        # path past multiple hourly intervals, not a direct obsolete cleanup call.
        worker = AIWorker(ReaderManager(), SimpleNamespace(store=SimpleNamespace(revision=7)),
                          attendance, DetectionState(), global_manager=manager)
        with patch("app.ai.worker.time.monotonic", return_value=10_000.):
            worker.step()
        with patch("app.ai.worker.time.monotonic", return_value=20_000.):
            worker.step()
        assert archive_rows(engine) == before


@test
def test_matcher_never_queries_or_mutates_sql_or_invokes_a_model():
    with archived_database() as (engine, _service, vector):
        before = archive_rows(engine)

        def forbid_sql(*_args):
            raise AssertionError("Pure appearance matcher attempted SQL")

        event.listen(engine, "before_cursor_execute", forbid_sql)
        try:
            with patch("app.ai.reid.PersonReID.embed",
                       side_effect=AssertionError("Matcher attempted inference")):
                manager = GlobalIdentityManager()
                proposal = manager.process_embeddings([appearance_item(embedding=vector)])[0]
                assert proposal.global_id is None
                manager.acknowledge(link_ack(proposal, 8701))
                assert manager.global_id_for(proposal.visit_id) == 8701
                manager.drop_camera(1)
                manager.sweep(100_000.)
                assert GlobalIdentityManager().snapshot() == []
        finally:
            event.remove(engine, "before_cursor_execute", forbid_sql)
        assert archive_rows(engine) == before


# -------------------------------------------------------------------- раннер

def main() -> int:
    print("Тесты системы глобальных личностей")
    failures = 0
    for fn in TESTS:
        try:
            fn()
            print(f"  ok   {fn.__name__}")
        except Exception:
            traceback.print_exc()
            print(f"  FAIL {fn.__name__}")
            failures += 1
    total = len(TESTS)
    print(f"\n{'FAIL' if failures else 'PASS'}: {total - failures}/{total}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
