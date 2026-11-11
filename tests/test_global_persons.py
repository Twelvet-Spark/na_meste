"""Тесты системы глобальных личностей: GlobalIdentityManager + REST API.

Запуск (без pytest, только зависимости проекта):

    venv/Scripts/python tests/test_global_persons.py      # Windows
    venv/bin/python tests/test_global_persons.py          # macOS / Linux

Проверяется:
- создание identity, межкамерный матч, camera_transition;
- last_seen_at в БД обновляется (не застревает на моменте создания);
- employee_id попадает в БД при распознавании уже привязанного трека;
- событие person_lost при истечении gallery_ttl;
- неизменяемый архив после перезапуска, отсутствие ручных мутаций;
- защита исторических связей при удалении камер/сотрудников;
- API: фильтры списка, аватар, имена камер в переходах, траектория.
"""
import sys
import time
import traceback
from dataclasses import dataclass
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from fastapi import HTTPException
from sqlalchemy import create_engine, event, select
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.ai.global_tracker import GlobalIdentityManager
from app.api import global_persons as api
from app.database.database import Base
from app.database import models as db_models  # noqa: F401 — регистрация таблиц
from app.database.models import (
    Camera, Employee, GlobalEvent, GlobalObservation, GlobalPerson,
)
from app.time_utils import utcnow

# ------------------------------------------------------------------ стенды

class FakeReID:
    """Re-ID по расписанию: embed() возвращает заранее заданный вектор."""

    def __init__(self):
        self.vector = None

    def set_person(self, index: int) -> None:
        vec = np.zeros(512, dtype=np.float32)
        vec[index] = 1.0
        self.vector = vec

    def embed(self, crops):
        return np.stack([self.vector] * len(crops))


@dataclass
class Det:
    track_id: int
    employee_id: int = None
    bbox: tuple = (0.3, 0.2, 0.45, 0.85)
    global_id: int = None


class Outcome:
    def __init__(self, *dets):
        self.detections = list(dets)


FRAME = np.zeros((720, 1280, 3), dtype=np.uint8)

_engine = create_engine(
    "sqlite://",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
SessionFactory = sessionmaker(bind=_engine, expire_on_commit=False)


def fresh():
    """Чистая БД + камеры/сотрудники + менеджер. Возвращает (manager, reid)."""
    Base.metadata.drop_all(_engine)
    Base.metadata.create_all(_engine)
    with SessionFactory() as db:
        db.add_all([
            Camera(id=1, name="Камера 1", nvr_host="h", username="u",
                   password="p", channel=1),
            Camera(id=2, name="Камера 2", nvr_host="h", username="u",
                   password="p", channel=2),
            Employee(id=7, name="Иван Петров"),
        ])
        db.commit()
    reid = FakeReID()
    manager = GlobalIdentityManager(SessionFactory, reid, topology=None)
    return manager, reid


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
def test_new_identity_and_cross_camera_match():
    manager, reid = fresh()
    reid.set_person(0)

    d1 = Det(track_id=1)
    manager.process(1, Outcome(d1), FRAME, now=0.5)
    assert d1.global_id == 1, "первый трек должен создать G#1"

    time.sleep(0.01)  # чтобы last_seen_at строго позже created_at
    d2 = Det(track_id=7)
    manager.process(2, Outcome(d2), FRAME, now=5.0)
    assert d2.global_id == 1, "та же внешность на другой камере — тот же global_id"

    reid.set_person(1)  # другой человек
    d3 = Det(track_id=9)
    manager.process(1, Outcome(d3), FRAME, now=10.0)
    assert d3.global_id == 2, "другая внешность — новая identity"

    with SessionFactory() as db:
        persons = {p.id: p for p in db.query(GlobalPerson).all()}
        assert set(persons) == {1, 2}
        p1 = persons[1]
        assert p1.status == "ACTIVE"          # matched → ACTIVE
        assert p1.last_camera_id == 2
        assert p1.last_seen_at > p1.created_at, \
            "last_seen_at должен обновляться (баг: застревал на создании)"
        events = db.query(GlobalEvent).filter_by(
            global_id=1, event_type="camera_transition").all()
        assert len(events) == 1
        import json
        payload = json.loads(events[0].payload)
        assert payload["from_camera"] == 1 and payload["to_camera"] == 2
        observations = db.query(GlobalObservation).filter_by(global_id=1).count()
        assert observations == 2, "по одному наблюдению на привязку трека"


@test
def test_employee_persisted_mid_track():
    """Сотрудник распознан ПОСЛЕ привязки трека — employee_id должен попасть
    в global_persons сразу, а не при следующей привязке."""
    manager, reid = fresh()
    reid.set_person(2)

    d = Det(track_id=20)
    manager.process(1, Outcome(d), FRAME, now=20.0)
    assert d.global_id == 1
    with SessionFactory() as db:
        assert db.get(GlobalPerson, 1).employee_id is None

    d2 = Det(track_id=20, employee_id=7)   # тот же трек, распознали лицом
    manager.process(1, Outcome(d2), FRAME, now=23.5)   # > reid_embedding_interval
    with SessionFactory() as db:
        person = db.get(GlobalPerson, 1)
        assert person.employee_id == 7, \
            "смена employee_id должна писаться в БД немедленно"
        assert person.last_seen_at > person.created_at


@test
def test_person_lost_event():
    manager, reid = fresh()
    reid.set_person(0)
    manager.process(1, Outcome(Det(track_id=1)), FRAME, now=0.5)

    manager.sweep(now=10_000.0)   # gallery_ttl=600 давно истёк

    with SessionFactory() as db:
        person = db.get(GlobalPerson, 1)
        assert person.status == "LOST"
        lost = db.query(GlobalEvent).filter_by(
            global_id=1, event_type="person_lost").all()
        assert len(lost) == 1, "истечение identity должно писать person_lost"
        import json
        payload = json.loads(lost[0].payload)
        assert payload["last_camera"] == 1
        assert payload["duration_seconds"] > 0
    assert 1 not in manager._identities, "identity уходит из памяти"




@test
def test_list_filters_and_fields():
    manager, reid = fresh()
    reid.set_person(0)
    manager.process(1, Outcome(Det(track_id=1)), FRAME, now=0.5)
    time.sleep(0.01)
    reid.set_person(1)
    manager.process(2, Outcome(Det(track_id=2, employee_id=7)), FRAME, now=5.0)
    time.sleep(0.01)
    reid.set_person(0)   # G#1 снова замечен на камере 2 → ACTIVE
    manager.process(2, Outcome(Det(track_id=3)), FRAME, now=6.0)

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
    manager, reid = fresh()
    reid.set_person(0)
    manager.process(1, Outcome(Det(track_id=1)), FRAME, now=0.5)
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
    manager, reid = fresh()
    reid.set_person(0)
    manager.process(1, Outcome(Det(track_id=1)), FRAME, now=0.5)
    time.sleep(0.01)
    manager.process(2, Outcome(Det(track_id=2)), FRAME, now=5.0)

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
    from test_attendance_service import database

    with database() as (engine, service):
        reid = FakeReID()
        reid.set_person(0)
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
                                  embedding=reid.vector.tobytes(), snapshot=b"archived-photo",
                                  created_at=recent),
                GlobalObservation(id=52, global_id=9001, camera_id=12, track_id=9,
                                  embedding=reid.vector.tobytes(), snapshot=b"old-photo",
                                  created_at=old),
                GlobalEvent(id=61, global_id=41, camera_id=13, track_id=8,
                            event_type="person_seen", payload='{"score":0.9}', created_at=recent),
                GlobalEvent(id=62, global_id=9001, camera_id=13, track_id=9,
                            event_type="person_lost", payload='{"duration_seconds":10}',
                            created_at=old),
            ])
        yield engine, service, reid


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
    from test_attendance_api import Client

    app = FastAPI()
    app.state.attendance = service
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
def test_restart_freezes_existing_gallery_and_allocates_above_absolute_maximum():
    with archived_database() as (engine, _service, reid):
        before = archive_rows(engine)
        factory = sessionmaker(bind=engine, expire_on_commit=False)
        manager = GlobalIdentityManager(factory, reid)
        assert manager.snapshot() == [], "Persisted identities are archive, not a live gallery"
        first = Det(track_id=1, employee_id=2)
        manager.process(1, Outcome(first), FRAME, now=1.)
        assert first.global_id == 9002, "Even an ancient maximum ID must not be reused"
        manager.process(2, Outcome(Det(track_id=2, employee_id=2)), FRAME, now=5.)
        manager.sweep(now=100_000.)
        after = archive_rows(engine)
        for original, current in zip(before, after):
            assert all(row in current for row in original), "Startup archive changed"
        with Session(engine) as db:
            live = db.get(GlobalPerson, 9002)
            assert live.employee_id == 2 and live.last_camera_id == 2
            assert db.query(GlobalObservation).filter_by(global_id=9002).count() == 2
        # A second restart freezes this process's rows too, without loading them.
        frozen = archive_rows(engine)
        restarted = GlobalIdentityManager(factory, reid)
        second = Det(track_id=3)
        restarted.process(1, Outcome(second), FRAME, now=100_001.)
        assert second.global_id == 9003
        for original, current in zip(frozen, archive_rows(engine)):
            assert all(row in current for row in original)


@test
def test_idle_worker_never_retention_purges_persisted_global_archive():
    from app.ai.worker import AIWorker, DetectionState
    from app.rtsp.manager import ReaderManager

    with archived_database() as (engine, attendance, reid):
        before = archive_rows(engine)
        manager = GlobalIdentityManager(sessionmaker(bind=engine), reid)
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
def test_global_id_seed_database_failure_cannot_fall_back_to_one():
    with archived_database() as (engine, _service, reid):
        before = archive_rows(engine)

        def fail_seed(_conn, _cursor, statement, parameters, _context, _many):
            if statement.lstrip().upper().startswith("SELECT") and "global_persons" in statement:
                raise OperationalError(statement, parameters, RuntimeError("seed unavailable"))

        event.listen(engine, "before_cursor_execute", fail_seed)
        try:
            try:
                GlobalIdentityManager(sessionmaker(bind=engine), reid)
            except (OperationalError, RuntimeError):
                pass
            else:
                raise AssertionError("An unseeded optional manager must not be published")
        finally:
            event.remove(engine, "before_cursor_execute", fail_seed)
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
