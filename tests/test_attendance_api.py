"""Work-time HTTP contracts over direct ASGI and a real temporary SQLite database.

Run: .venv/Scripts/python.exe tests/test_attendance_api.py
Local workplace offset is +05:00 (the default); 2026-03-10 local = 03-09 19:00 .. 03-10 19:00 UTC.
"""
import sys
import traceback
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi import FastAPI
from sqlalchemy.orm import Session

from app.api import attendance
from app.database.database import get_db
from app.database.models import (
    DoorCrossing, ManualPresenceInterval, PresenceExclusion, PresenceSegment,
)
from app.services.presence_recorder import PresenceRecorder
from operator_test_support import Client
from test_presence_recorder import database

TESTS = []
DAY_START = datetime(2026, 3, 9, 19, 0)         # 2026-03-10 00:00 local
FINISHED = datetime(2026, 3, 11, 10, 0)          # the day is over
EVENING = datetime(2026, 3, 10, 14, 0)           # 19:00 local, the day is running


def test(fn):
    TESTS.append(fn)
    return fn


def local(hours, minutes=0):
    return DAY_START + timedelta(hours=hours, minutes=minutes)


def iso(value):
    return value.strftime("%Y-%m-%dT%H:%M:%S+00:00")


@contextmanager
def api(as_of=FINISHED):
    with database() as engine:
        app = FastAPI()
        app.state.presence = PresenceRecorder(engine)
        app.include_router(attendance.router)

        def isolated_db():
            with Session(engine) as db:
                yield db

        app.dependency_overrides[get_db] = isolated_db
        with patch("app.api.attendance._now", return_value=as_of):
            yield Client(app), engine


def seed(engine, employee=1, camera=1, start=local(9), end=local(12), face=True):
    with Session(engine) as db:
        db.add(PresenceSegment(employee_id=employee, camera_id=camera, started_at=start, last_seen_at=end,
                               face_seen=face))
        db.commit()


def seed_day(engine, employee=1):
    seed(engine, employee, 1, local(9), local(12))
    seed(engine, employee, 2, local(15), local(18))


def get(client, url, status=200):
    code, _headers, data = client.request("GET", url)
    assert code == status, (url, code, data)
    return data


def correction(**values):
    return {"request_id": str(uuid4()), "actor": " Operator ", "reason": " Wrong camera ", "employee_id": 1,
            "started_at": "2026-03-10T09:00:00+05:00", "ended_at": "2026-03-10T10:00:00+05:00", **values}


DAY_URL = "from_date=2026-03-10&to_date=2026-03-10"


@test
def report_gives_final_day_hours_and_live_hours_for_a_running_day():
    with api() as (client, engine):
        seed_day(engine)
        report = get(client, f"/api/attendance/report?{DAY_URL}")
        (row,) = report["rows"]
        assert row["worked_ms"] == 21600000 and row["method"] == "FINAL" and row["sessions"] == 2
        assert (row["employee_name"], row["face_seen"], row["has_manual"]) == ("Анна", True, False)
        assert row["first_seen_at"] == "2026-03-10T09:00:00+05:00" and row["last_seen_at"] == "2026-03-10T18:00:00+05:00"
        assert report["totals"] == [{"employee_id": 1, "employee_name": "Анна", "worked_ms": 21600000, "days": 1}]
        assert (report["live_gap_minutes"], report["day_gap_minutes"], report["utc_offset_minutes"]) == (35, 120, 300)
    with api(EVENING) as (client, engine):
        seed_day(engine)
        (row,) = get(client, f"/api/attendance/report?{DAY_URL}")["rows"]
        assert row["method"] == "LIVE" and row["worked_ms"] == 21600000


@test
def report_filters_by_employee_skips_empty_days_and_future_days():
    with api(EVENING) as (client, engine):
        seed(engine, 1, 1, local(9), local(10))
        seed(engine, 2, 1, local(9), local(11))
        both = get(client, f"/api/attendance/report?from_date=2026-03-09&to_date=2026-03-12")["rows"]
        assert [(r["date"], r["employee_name"]) for r in both] == [("2026-03-10", "Анна"), ("2026-03-10", "Борис")]
        only = get(client, f"/api/attendance/report?{DAY_URL}&employee_id=2")["rows"]
        assert [r["employee_id"] for r in only] == [2]
        get(client, f"/api/attendance/report?{DAY_URL}&employee_id=99", 404)


@test
def report_period_must_be_one_to_366_days():
    with api() as (client, _engine):
        get(client, "/api/attendance/report?from_date=2026-03-10&to_date=2026-03-09", 422)
        get(client, "/api/attendance/report?from_date=2025-01-01&to_date=2026-03-10", 422)
        assert get(client, "/api/attendance/report?from_date=2025-03-10&to_date=2026-03-10")["rows"] == []
        code, _h, data = client.request("GET", "/api/attendance/report?from_date=2025-01-01&to_date=2026-03-10")
        assert data["detail"] == "Укажите период от 1 до 366 дней"


@test
def active_lists_employees_seen_within_the_live_gap_with_their_source():
    with api(datetime(2026, 3, 10, 13, 10)) as (client, engine):
        seed(engine, 1, 1, local(9), local(12), face=True)
        seed(engine, 1, 2, local(15), local(18), face=True)      # ends 13:00 UTC, 10 min before as_of
        seed(engine, 2, 2, local(17), local(18), face=False)
        active = get(client, "/api/attendance/active")
        assert [(i["employee_name"], i["source"], i["last_camera_name"]) for i in active["items"]] == [
            ("Анна", "FACE", "Зал"), ("Борис", "REID", "Зал")]
        anna = active["items"][0]
        assert anna["today_ms"] == 21600000 and anna["since"] == "2026-03-10T15:00:00+05:00"
    with api(EVENING) as (client, engine):
        seed(engine, 1, 1, local(15), local(18), face=True)       # last seen 60 min ago
        assert get(client, "/api/attendance/active")["items"] == []


@test
def csv_export_has_bom_header_and_rows():
    with api() as (client, engine):
        seed_day(engine)
        seed(engine, 2, 1, local(9), local(10), face=False)
        code, headers, body = client.request("GET", f"/api/attendance/report.csv?{DAY_URL}")
        assert code == 200 and headers[b"content-type"].startswith(b"text/csv")
        assert b"work-time-2026-03-10-2026-03-10.csv" in headers[b"content-disposition"]
        assert body.startswith(b"\xef\xbb\xbf")
        lines = body.decode("utf-8-sig").splitlines()
        assert lines[0] == ('"Дата";"Сотрудник";"Отработано (ч:мм)";"Минут";"Первое появление";'
                            '"Последнее появление";"Сессий";"Расчёт";"Примечания"')
        assert '"2026-03-10";"Анна";"6:00";"360";"09:00";"18:00";"2";"Итог дня";""' in lines
        assert '"2026-03-10";"Борис";"1:00";"60";"09:00";"10:00";"1";"Итог дня";"только по внешности"' in lines


@test
def timeline_shows_sessions_door_gaps_and_excluded_rows():
    with api() as (client, engine):
        seed(engine, 1, 1, local(9), local(11, 40))
        seed(engine, 1, 2, local(12), local(15))
        with Session(engine) as db:
            db.add_all([
                DoorCrossing(source_key="a", camera_id=1, employee_id=1, direction="EXIT", occurred_at=local(11, 40)),
                DoorCrossing(source_key="b", camera_id=1, employee_id=1, direction="ENTRY", occurred_at=local(12)),
            ])
            db.commit()
        line = get(client, "/api/attendance/timeline?employee_id=1&date=2026-03-10")
        assert line["worked_ms"] == 21600000 - 1200000 and line["method"] == "FINAL"
        assert len(line["sessions"]) == 2 and len(line["door_gaps"]) == 1
        assert line["door_gaps"][0] == {"started_at": "2026-03-10T11:40:00+05:00",
                                        "ended_at": "2026-03-10T12:00:00+05:00"}
        assert [s["camera_name"] for s in line["segments"]] == ["Вход", "Зал"]
        assert [c["direction"] for c in line["crossings"]] == ["EXIT", "ENTRY"]
        get(client, "/api/attendance/timeline?employee_id=99&date=2026-03-10", 404)


@test
def corrections_create_replay_conflict_validate_and_revoke():
    with api() as (client, engine):
        seed_day(engine)
        payload = correction()
        code, _h, first = client.request("POST", "/api/attendance/corrections/exclusion", payload)
        assert code == 200 and first["kind"] == "EXCLUSION" and first["actor"] == "Operator"
        assert first["reason"] == "Wrong camera" and first["revoked_at"] is None
        assert first["started_at"] == "2026-03-10T09:00:00+05:00"
        # Excluding 09:00-10:00 moves the first sighting to 10:00 and leaves 5 hours.
        (row,) = get(client, f"/api/attendance/report?{DAY_URL}")["rows"]
        assert row["worked_ms"] == 18000000 and row["first_seen_at"] == "2026-03-10T10:00:00+05:00"
        assert client.request("POST", "/api/attendance/corrections/exclusion", payload)[2] == first
        different = dict(payload, ended_at="2026-03-10T10:30:00+05:00")
        code, _h, conflict = client.request("POST", "/api/attendance/corrections/exclusion", different)
        assert code == 409 and conflict["detail"] == "request_id уже использован с другими данными"
        with Session(engine) as db:
            assert db.query(PresenceExclusion).count() == 1
        revoked = client.request("POST", f"/api/attendance/corrections/exclusion/{first['id']}/revoke",
                                 {"actor": "Boss", "reason": "Mistake"})[2]
        assert revoked["revoked_by"] == "Boss" and revoked["revoke_reason"] == "Mistake" and revoked["revoked_at"]
        code, _h, again = client.request("POST", f"/api/attendance/corrections/exclusion/{first['id']}/revoke",
                                         {"actor": "Boss", "reason": "Again"})
        assert code == 409 and again["detail"] == "Корректировка уже отменена"
        (row,) = get(client, f"/api/attendance/report?{DAY_URL}")["rows"]
        assert row["worked_ms"] == 21600000
        listed = get(client, f"/api/attendance/corrections?{DAY_URL}")["items"]
        assert [(i["id"], i["revoked_by"], i["employee_name"]) for i in listed] == [(first["id"], "Boss", "Анна")]
        assert client.request("POST", "/api/attendance/corrections/manual/999/revoke",
                              {"actor": "a", "reason": "r"})[0] == 404


@test
def manual_interval_adds_time_and_is_flagged_in_the_report():
    with api() as (client, engine):
        seed(engine, 1, 1, local(9), local(10))
        code, _h, row = client.request("POST", "/api/attendance/corrections/manual", correction(
            started_at="2026-03-10T10:00:00+05:00", ended_at="2026-03-10T12:00:00+05:00"))
        assert code == 200 and row["kind"] == "MANUAL"
        (day,) = get(client, f"/api/attendance/report?{DAY_URL}")["rows"]
        assert day["worked_ms"] == 3 * 3600000 and day["has_manual"] is True
        with Session(engine) as db:
            assert db.query(ManualPresenceInterval).count() == 1


@test
def corrections_reject_bad_input_and_foreign_requests():
    with api() as (client, _engine):
        post = lambda body, kind="manual": client.request("POST", f"/api/attendance/corrections/{kind}", body)
        assert post(correction(), "manual")[0] == 200
        assert post(correction(employee_id=99))[0] == 404
        future = correction(started_at="2026-03-11T09:00:00+05:00", ended_at="2026-03-11T16:00:00+05:00")
        code, _h, data = post(future)
        assert code == 422 and data["detail"] == "Время в будущем недопустимо"
        too_long = correction(started_at="2026-03-09T09:00:00+05:00", ended_at="2026-03-10T09:00:01+05:00")
        assert post(too_long)[0] == 422
        assert post(correction(ended_at="2026-03-10T08:00:00+05:00"))[0] == 422
        assert post(correction(started_at="2026-03-10T09:00:00"))[0] == 422      # no UTC offset
        assert post(dict(correction(), reason="   "))[0] == 422
        assert post(correction(), "other")[0] == 422
        code, _h, _d = client.request("POST", "/api/attendance/corrections/manual", correction(), operator=False)
        assert code == 403


@test
def status_reports_cameras_workers_and_recorder_state():
    with api() as (client, _engine):
        status = get(client, "/api/attendance/status")
        assert [(c["camera_id"], c["time_tracking"], c["status"]) for c in status["cameras"]] == [
            (1, False, "OFFLINE"), (2, False, "OFFLINE")]
        assert (status["worker_alive"], status["appearance_available"], status["face_available"]) == (False, False, False)
        assert status["recorder"]["open_segments"] == 0 and status["recorder"]["last_error"] is None
        assert (status["live_gap_minutes"], status["day_gap_minutes"]) == (35, 120)
        assert status["as_of"] == "2026-03-11T15:00:00+05:00"


@test
def endpoints_report_503_without_the_recorder():
    with api() as (client, _engine):
        client.app.state.presence = None
        assert get(client, "/api/attendance/status", 503)


@test
def report_and_period_totals_preserve_exact_millisecond_durations():
    with api() as (client, engine):
        previous = local(9) - timedelta(days=1)
        seed(engine, start=previous, end=previous + timedelta(milliseconds=1001))
        seed(engine, start=local(9), end=local(9) + timedelta(milliseconds=58999))
        report = get(client, "/api/attendance/report?from_date=2026-03-09&to_date=2026-03-10")
        assert [row["worked_ms"] for row in report["rows"]] == [1001, 58999]
        assert report["totals"][0]["worked_ms"] == 60000


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
    print(f"{len(TESTS) - failures}/{len(TESTS)} passed")
    raise SystemExit(bool(failures))
