"""Тесты системы глобальных личностей: GlobalIdentityManager + REST API.

Запуск (без pytest, только зависимости проекта):

    venv/Scripts/python tests/test_global_persons.py      # Windows
    venv/bin/python tests/test_global_persons.py          # macOS / Linux

Проверяется:
- создание identity, межкамерный матч, camera_transition;
- last_seen_at в БД обновляется (не застревает на моменте создания);
- employee_id попадает в БД при распознавании уже привязанного трека;
- событие person_lost при истечении gallery_ttl;
- merge: перенос наблюдений/событий, память менеджера;
- ручная привязка/отвязка сотрудника (БД + память + 404);
- API: фильтры списка, аватар, имена камер в переходах, траектория.
"""
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
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
def test_merge_api_and_memory():
    manager, reid = fresh()
    reid.set_person(0)
    manager.process(1, Outcome(Det(track_id=1)), FRAME, now=0.5)
    time.sleep(0.01)
    reid.set_person(1)
    manager.process(1, Outcome(Det(track_id=9)), FRAME, now=10.0)
    assert set(manager._identities) == {1, 2}

    with SessionFactory() as db:
        result = api.api_merge_global_person(1, 2, db=db, manager=manager)

    assert result["merged_from"] == 2
    assert result["moved"] == {"observations": 1, "events": 1}
    assert not result.get("employee_conflict")
    with SessionFactory() as db:
        assert db.get(GlobalPerson, 2) is None, "source должен быть удалён"
        assert db.query(GlobalObservation).filter_by(global_id=2).count() == 0
        assert db.query(GlobalEvent).filter_by(global_id=2).count() == 0
        assert db.query(GlobalObservation).filter_by(global_id=1).count() == 2
        target = db.get(GlobalPerson, 1)
        assert target.last_camera_id == 1    # source был свежее (t=10 > t=0.5)
    # память: галерея объединена, привязки переключены, source исчез
    assert set(manager._identities) == {1}
    assert len(manager._identities[1].embeddings) == 2
    for binding in manager._bindings.values():
        assert binding.global_id != 2
    assert manager._bindings[(1, 9)].global_id == 1

    with SessionFactory() as db:
        expect_http(400, api.api_merge_global_person, 1, 1, db=db, manager=None)
        expect_http(404, api.api_merge_global_person, 1, 999, db=db, manager=None)


@test
def test_merge_employee_and_conflict_flag():
    manager, reid = fresh()
    reid.set_person(0)
    manager.process(1, Outcome(Det(track_id=1, employee_id=7)), FRAME, now=0.5)
    reid.set_person(1)
    manager.process(2, Outcome(Det(track_id=2, employee_id=8)), FRAME, now=1.0)
    with SessionFactory() as db:
        db.add(Employee(id=8, name="Пётр Иванов"))
        db.commit()

    with SessionFactory() as db:
        result = api.api_merge_global_person(1, 2, db=db, manager=manager)
    assert result.get("employee_conflict") is True, \
        "разные сотрудники у личностей — конфликт должен быть виден"

    # а сюда employee должен перенестись
    manager2, reid2 = fresh()
    reid2.set_person(0)
    manager2.process(1, Outcome(Det(track_id=1)), FRAME, now=0.5)
    reid2.set_person(1)
    manager2.process(2, Outcome(Det(track_id=2, employee_id=7)), FRAME, now=1.0)
    with SessionFactory() as db:
        result = api.api_merge_global_person(1, 2, db=db, manager=manager2)
    assert result["employee_id"] == 7, "employee source переносится в target"


@test
def test_assign_employee_api_and_memory():
    manager, reid = fresh()
    reid.set_person(0)
    manager.process(1, Outcome(Det(track_id=1)), FRAME, now=0.5)
    assert manager._identities[1].employee_id is None

    with SessionFactory() as db:
        result = api.api_assign_employee(1, 7, db=db, manager=manager)
    assert result["employee_id"] == 7
    assert result["employee_name"] == "Иван Петров"

    # БД и память синхронны: периодический _db_update_person активной
    # identity не должен вернуть None (причина ручной привязки)
    assert manager._identities[1].employee_id == 7
    manager._db_update_person(manager._identities[1])
    with SessionFactory() as db:
        assert db.get(GlobalPerson, 1).employee_id == 7

    # отвязка — тоже в БД и памяти
    with SessionFactory() as db:
        result = api.api_unassign_employee(1, db=db, manager=manager)
    assert result["employee_id"] is None
    assert manager._identities[1].employee_id is None
    with SessionFactory() as db:
        assert db.get(GlobalPerson, 1).employee_id is None

    # привязка к несуществующим сущностям
    with SessionFactory() as db:
        expect_http(404, api.api_assign_employee, 1, 999, db=db, manager=None)
        expect_http(404, api.api_assign_employee, 999, 7, db=db, manager=None)

    # менеджера нет (AI выключен) — БД всё равно обновляется
    with SessionFactory() as db:
        result = api.api_assign_employee(1, 7, db=db, manager=None)
    assert result["employee_id"] == 7


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


@test
def test_recovery_from_db():
    manager, reid = fresh()
    reid.set_person(0)
    manager.process(1, Outcome(Det(track_id=1, employee_id=7)), FRAME, now=0.5)

    # fresh() пересоздаёт таблицы — для проверки восстановления нужен второй
    # менеджер над ТОЙ ЖЕ базой, поэтому готовим его вручную
    manager2 = GlobalIdentityManager(SessionFactory, FakeReID(), topology=None)
    assert 1 in manager2._identities, "identity переживает рестарт"
    restored = manager2._identities[1]
    assert restored.employee_id == 7
    assert len(restored.embeddings) == 1
    assert restored.status == "LOST"


@test
def test_cleanup_respects_keep_days():
    manager, reid = fresh()
    reid.set_person(0)
    manager.process(1, Outcome(Det(track_id=1)), FRAME, now=0.5)
    with SessionFactory() as db:
        old = utcnow() - timedelta(days=999)
        db.get(GlobalPerson, 1).last_seen_at = old
        db.query(GlobalObservation).update({"created_at": old})
        db.query(GlobalEvent).update({"created_at": old})
        db.commit()
    manager.cleanup()
    with SessionFactory() as db:
        assert db.get(GlobalPerson, 1) is None
        assert db.query(GlobalObservation).count() == 0
        assert db.query(GlobalEvent).count() == 0


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
