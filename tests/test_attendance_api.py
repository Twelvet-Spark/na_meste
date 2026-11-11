"""Attendance HTTP contracts using direct ASGI and isolated real SQLite services.

Run: .venv/Scripts/python.exe tests/test_attendance_api.py
Controller owns RED/GREEN execution; no lifespan, models, camera or production DB.
"""
import asyncio
import copy
import json
import sys
import traceback
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path
from urllib.parse import urlsplit
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi import FastAPI
from sqlalchemy import event
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from app.api import attendance, employees, presence
from app.attendance_types import VisitorBatchInput
from app.database.database import get_db, immediate_session
from app.database.models import (
    AttendanceCamera, AttendanceCommission, AttendanceCoverageGap, AttendanceDecision,
    AttendanceEmployeeReference, AttendanceEvent, AttendanceEvidence, AttendanceRequest,
    AttendanceRuntime, Employee, EmployeeFace, PresenceSession, WorkSession,
)
from test_attendance_service import AS_OF, BASE, CUDA_PROVIDERS, crossing, database
from test_attendance_lifecycle import Clock, health_for, warm

TESTS = []


def test(fn):
    TESTS.append(fn)
    return fn


class Client:
    def __init__(self, app):
        self.app = app

    def request(self, method, url, payload=None):
        async def run():
            parsed = urlsplit(url)
            body = b"" if payload is None else json.dumps(payload).encode("utf-8")
            sent, messages = False, []

            async def receive():
                nonlocal sent
                if sent:
                    return {"type": "http.disconnect"}
                sent = True
                return {"type": "http.request", "body": body, "more_body": False}

            async def send(message):
                messages.append(message)

            await self.app({
                "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
                "method": method, "scheme": "http", "path": parsed.path,
                "raw_path": parsed.path.encode(), "query_string": parsed.query.encode(),
                "root_path": "", "headers": [(b"content-type", b"application/json")],
                "client": ("127.0.0.1", 1234), "server": ("test", 80),
            }, receive, send)
            start = next(item for item in messages if item["type"] == "http.response.start")
            data = b"".join(item.get("body", b"") for item in messages if item["type"] == "http.response.body")
            headers = dict(start["headers"])
            if headers.get(b"content-type", b"").startswith(b"application/json"):
                data = json.loads(data)
            return start["status"], headers, data
        return asyncio.run(run())

    def json(self, method, url, payload=None, *, status=200):
        actual, _headers, data = self.request(method, url, payload)
        assert actual == status, (method, url, actual, data)
        return data


@contextmanager
def api():
    with database() as (engine, service):
        app = FastAPI()
        app.state.attendance = service
        app.include_router(attendance.router)
        app.include_router(employees.router)
        app.include_router(presence.router)

        def isolated_db():
            with Session(engine) as db:
                yield db

        app.dependency_overrides[get_db] = isolated_db
        yield Client(app), engine, service


def human(**values):
    return dict(request_id=str(uuid4()), actor=" Operator ", reason=" Reconciled doorway ", **values)


def manual_payload(direction="ENTRY", when="2026-11-11T09:00:00+05:00", employee=1):
    return human(employee_id=employee, direction=direction, occurred_at=when)


def manual(client, **values):
    return client.json("POST", "/api/attendance/manual", manual_payload(**values))


def decide(client, identifier, action, revision=0, **values):
    payload = human(expected_revision=revision, action=action, **values)
    return client.json("POST", f"/api/attendance/events/{identifier}/decision", payload)


def review_crossing(service, *, ordinal=1, hours=0):
    return service.record_crossing(crossing(ordinal=ordinal, when=BASE + timedelta(hours=hours)))


def commission_payload(**counts):
    record = dict(entry_trials=30, exit_trials=30, employee_count=1, visitor_trials=20,
                  retreat_trials=10, wrong_employee=0, duplicate_trips=0, missed_unflagged=0,
                  frontal_trials=100, frontal_eligible=95, performance_minutes=15, site_confirmed=True)
    record.update(counts)
    return human(expected_entry_revision=3, expected_exit_revision=3, acceptance_record=record)


def commission_ready(engine, service):
    with immediate_session(engine) as db:
        db.add(EmployeeFace(employee_id=1, embedding=bytes(512 * 4)))
    clock = Clock()
    health = health_for(service, clock)
    warm(health, clock)
    # Persist real immutable notifications before commission; commissioning-only
    # readiness is intentionally not part of technical readiness.
    faults = health.pending_faults()
    service.persist_health([(fault, health.fault_source_ids(fault.id)) for fault in faults],
                           healthy=True, reasons=[], observed_at=clock.utc)
    for fault in faults:
        health.ack_fault(fault.id)
    assert health.snapshot()["technical_ready"]
    return clock, health


@test
def test_manual_json_normalization_audit_and_replay_survives_later_correction():
    with api() as (client, engine, _service):
        payload = manual_payload(when="2026-11-11T09:00:00.123999+05:00")
        original = client.json("POST", "/api/attendance/manual", payload)
        assert original["occurred_at"].startswith("2026-11-11T09:00:00.123")
        assert original["occurred_at"].endswith("+05:00")
        assert original["source"] == "MANUAL" and original["status"] == "ACCEPTED"
        assert original["evidence"] == dict(before=False, after=False, face=False, purged=False)
        decide(client, original["id"], "VOID")
        assert client.json("POST", "/api/attendance/manual", payload) == original
        detail = client.json("GET", f"/api/attendance/events/{original['id']}")
        assert detail["raw"] == {"version": 1, "candidate_employee_ids": []}
        assert [row["revision"] for row in detail["decisions"]] == [0, 1]
        assert detail["decisions"][0]["actor"] == "Operator"
        assert detail["decisions"][0]["before"] == {}
        with Session(engine) as db:
            assert db.query(AttendanceEvent).count() == 1
            assert db.query(AttendanceDecision).count() == 2
            assert db.query(WorkSession).count() == 0
        changed = dict(payload, direction="EXIT")
        client.json("POST", "/api/attendance/manual", changed, status=409)


@test
def test_decision_replay_precedes_revision_and_preserves_original_employee_reference():
    with api() as (client, engine, service):
        identifier = review_crossing(service)
        payload = human(expected_revision=0, action="ACCEPT", employee_id=1, direction="ENTRY",
                        occurred_at="2026-11-11T09:00:00.456789+05:00")
        first = client.json("POST", f"/api/attendance/events/{identifier}/decision", payload)
        assert first["employee_id"] == 1 and first["revision"] == 1
        assert first["occurred_at"].startswith("2026-11-11T09:00:00.456")
        assert first["occurred_at"].endswith("+05:00")
        reassigned = decide(client, identifier, "ACCEPT", 1, employee_id=2)
        assert reassigned["occurred_at"] == first["occurred_at"]
        assert client.json("POST", f"/api/attendance/events/{identifier}/decision", payload) == first
        client.json("POST", f"/api/attendance/events/{identifier}/decision",
                    human(expected_revision=0, action="VOID"), status=409)
        other = review_crossing(service, ordinal=2)
        client.json("POST", f"/api/attendance/events/{other}/decision", payload, status=409)
        decide(client, identifier, "VISITOR", 2)
        client.json("DELETE", "/api/employees/1", status=409)
        with Session(engine) as db:
            assert db.get(Employee, 1) is not None
            assert db.get(AttendanceEmployeeReference, (identifier, 1)) is not None
            assert db.get(AttendanceEmployeeReference, (identifier, 2)) is not None
            assert db.query(AttendanceDecision).filter_by(event_id=identifier).count() == 3
            assert db.query(WorkSession).count() == 0
        detail = client.json("GET", f"/api/attendance/events/{identifier}")
        assert detail["raw"]["would_auto_accept"] is True
        assert detail["status"] == "VISITOR" and detail["employee_id"] is None


@test
def test_visitor_batch_is_atomic_unique_canonical_and_replays_original_rows():
    with api() as (client, engine, service):
        one = review_crossing(service)
        two = review_crossing(service, ordinal=2, hours=1)
        items = [{"event_id": one, "expected_revision": 0}, {"event_id": two, "expected_revision": 1}]
        failed = human(items=items)
        client.json("POST", "/api/attendance/visitors", failed, status=409)
        with Session(engine) as db:
            assert db.get(AttendanceEvent, one).status == "REVIEW"
            assert db.get(AttendanceRequest, failed["request_id"]) is None
            assert db.query(AttendanceDecision).count() == 0
        client.json("POST", "/api/attendance/visitors", human(items=[items[0], items[0]]), status=422)
        items[1]["expected_revision"] = 0
        payload = human(items=items)
        first = client.json("POST", "/api/attendance/visitors", payload)
        assert len(first) == 2 and all(row["status"] == "VISITOR" for row in first)
        decide(client, one, "VOID", 1)
        retry = dict(payload, items=list(reversed(items)))
        assert client.json("POST", "/api/attendance/visitors", retry) == first
        with Session(engine) as db:
            assert db.query(AttendanceDecision).filter_by(request_id=payload["request_id"]).count() == 2


@test
def test_gap_review_closed_only_clock_reconciliation_and_immutable_replay():
    with api() as (client, engine, _service):
        with immediate_session(engine) as db:
            closed = AttendanceCoverageGap(started_at=BASE, ended_at=BASE + timedelta(seconds=1),
                                           reason_codes=["CLOCK_DISCONTINUITY"])
            opened = AttendanceCoverageGap(started_at=BASE + timedelta(hours=1), reason_codes=["FRAME_GAP"])
            db.add_all([closed, opened])
            db.flush()
            closed_id, open_id = closed.id, opened.id
        client.json("POST", f"/api/attendance/gaps/{open_id}/review", human(expected_revision=0), status=409)
        payload = human(expected_revision=0)
        first = client.json("POST", f"/api/attendance/gaps/{closed_id}/review", payload)
        assert first["reviewed"] and first["revision"] == 1
        newer = client.json("POST", f"/api/attendance/gaps/{closed_id}/review", human(expected_revision=1))
        assert newer["revision"] == 2
        assert client.json("POST", f"/api/attendance/gaps/{closed_id}/review", payload) == first
        with Session(engine) as db:
            assert db.query(AttendanceDecision).filter_by(gap_id=closed_id).count() == 2
        client.json("POST", f"/api/attendance/gaps/{open_id}/review", payload, status=409)
        detail = client.json("GET", f"/api/attendance/gaps/{closed_id}")
        assert detail["reviewed"] and detail["revision"] == 2
        assert [item["revision"] for item in detail["decisions"]] == [1, 2]
        assert detail["decisions"][0]["before"]["started_at"].endswith("+05:00")
        client.json("GET", "/api/attendance/gaps/999", status=404)


@test
def test_events_lists_filter_count_local_dates_and_validate_inputs():
    with api() as (client, _engine, _service):
        first = manual(client, when="2026-11-10T23:59:59+05:00")
        second = manual(client, direction="EXIT", when="2026-11-11T00:00:00+05:00")
        third = manual(client, when="2026-11-11T23:59:59+05:00", employee=2)
        page = client.json("GET", "/api/attendance/events?from_date=2026-11-11&to_date=2026-11-11&limit=1")
        assert page["total"] == 2 and len(page["items"]) == 1
        assert page["items"][0]["id"] == third["id"]
        page = client.json("GET", "/api/attendance/events?from_date=2026-11-11&to_date=2026-11-11&limit=1&offset=1")
        assert page["items"][0]["id"] == second["id"] and page["total"] == 2
        page = client.json("GET", "/api/attendance/events?employee_id=1")
        assert page["total"] == 2 and {row["id"] for row in page["items"]} == {first["id"], second["id"]}
        for query, status in (("limit=201", 422), ("offset=-1", 422), ("status=UNKNOWN", 422),
                              ("from_date=bad", 422), ("from_date=2026-11-12&to_date=2026-11-11", 422),
                              ("employee_id=999", 404)):
            client.json("GET", f"/api/attendance/events?{query}", status=status)


@test
def test_sessions_active_and_review_share_quarantine_and_pagination_counts():
    with api() as (client, engine, service):
        manual(client)
        manual(client, direction="EXIT", when="2026-11-11T18:00:00+05:00")
        open_entry = manual(client, when="2026-11-12T09:00:00+05:00", employee=2)
        identifier = review_crossing(service, hours=3)
        with immediate_session(engine) as db:
            db.add(AttendanceCoverageGap(started_at=BASE + timedelta(days=1, hours=1),
                                         ended_at=BASE + timedelta(days=1, hours=2), reason_codes=["FRAME_GAP"]))
        page = client.json("GET", "/api/attendance/sessions?employee_id=1")
        assert page["total"] == 1
        assert page["items"][0]["status"] == "REVIEW" and page["items"][0]["review_ms"] == 32_400_000
        assert page["items"][0]["confirmed_ms"] == 0
        active = client.json("GET", "/api/attendance/active")
        assert active["total"] == 1 and active["utc_offset_minutes"] == 300
        assert active["as_of"].endswith("+05:00")
        assert active["items"][0]["entry_event_id"] == open_entry["id"]
        assert active["items"][0]["ended_at"] is None and active["items"][0]["status"] == "REVIEW"
        review = client.json("GET", "/api/attendance/review?limit=1")
        assert review["total"] == 2 and len(review["items"]) == 1
        both = client.json("GET", "/api/attendance/review?limit=2")
        assert {item["kind"] for item in both["items"]} == {"event", "gap"}
        assert any(item["kind"] == "event" and item["event"]["id"] == identifier for item in both["items"])
        decide(client, identifier, "VISITOR")
        repaired = client.json("GET", "/api/attendance/sessions?employee_id=1")["items"][0]
        assert repaired["confirmed_ms"] == 32_400_000 and repaired["status"] == "CLOSED"
        assert client.json("GET", "/api/attendance/sessions?status=CLOSED")["total"] == 1
        client.json("GET", "/api/attendance/active?employee_id=999", status=404)


@test
def test_evidence_returns_actual_jpeg_absent_and_expired_without_ledger_loss():
    with api() as (client, engine, service):
        identifier = review_crossing(service)
        status, headers, data = client.request("GET", f"/api/attendance/events/{identifier}/evidence/before")
        assert status == 200 and headers[b"content-type"].startswith(b"image/jpeg")
        assert data.startswith(b"\xff\xd8")
        manual_row = manual(client)
        client.json("GET", f"/api/attendance/events/{manual_row['id']}/evidence/face", status=404)
        client.json("GET", f"/api/attendance/events/{identifier}/evidence/wrong", status=422)
        with immediate_session(engine) as db:
            image = db.get(AttendanceEvidence, identifier)
            image.before_jpeg = image.after_jpeg = image.face_jpeg = None
            image.purged_at = AS_OF
        client.json("GET", f"/api/attendance/events/{identifier}/evidence/before", status=410)
        detail = client.json("GET", f"/api/attendance/events/{identifier}")
        assert detail["evidence"]["purged"] and detail["raw"]["version"] == 1


@test
def test_json_validation_and_historical_inactive_employee_manual_assignment():
    with api() as (client, engine, _service):
        for change in (dict(occurred_at="2026-11-11T09:00:00"), dict(occurred_at="2027-01-01T09:00:00+05:00"),
                       dict(direction="SIDEWAYS"), dict(actor=" "), dict(reason=" "), dict(extra="forbidden")):
            client.json("POST", "/api/attendance/manual", dict(manual_payload(), **change), status=422)
        client.json("POST", "/api/attendance/manual", manual_payload(employee=999), status=404)
        with immediate_session(engine) as db:
            db.get(Employee, 1).active = False
        row = manual(client, when="2026-11-01T09:00:00+05:00")
        assert row["employee_id"] == 1 and row["status"] == "ACCEPTED"
        client.json("POST", f"/api/attendance/events/{row['id']}/decision",
                    human(expected_revision=0, action="UNKNOWN"), status=422)
        with Session(engine) as db:
            assert db.query(AttendanceEvent).count() == 1 and db.query(AttendanceDecision).count() == 1


@test
def test_read_count_and_rows_use_one_snapshot_during_concurrent_visitor_commit():
    with api() as (client, engine, service):
        identifier = review_crossing(service)
        changed = False

        def concurrent_write(_connection, _cursor, statement, _parameters, _context, _many):
            nonlocal changed
            if not changed and "count(" in statement.lower() and "attendance_events" in statement.lower():
                changed = True
                service.mark_visitors(VisitorBatchInput(**human(items=[dict(event_id=identifier, expected_revision=0)])))

        event.listen(engine, "after_cursor_execute", concurrent_write)
        try:
            page = client.json("GET", "/api/attendance/events?status=REVIEW")
        finally:
            event.remove(engine, "after_cursor_execute", concurrent_write)
        assert changed and page["total"] == 1 and page["items"][0]["status"] == "REVIEW"
        assert client.json("GET", "/api/attendance/events?status=REVIEW")["total"] == 0


@test
def test_database_failure_returns_503_and_rolls_back_request_and_event():
    with api() as (client, engine, _service):
        payload = manual_payload()

        def fail_insert(_connection, _cursor, statement, parameters, _context, _many):
            if statement.startswith("INSERT INTO attendance_events"):
                raise OperationalError(statement, parameters, RuntimeError("injected write fault"))

        event.listen(engine, "before_cursor_execute", fail_insert)
        try:
            client.json("POST", "/api/attendance/manual", payload, status=503)
        finally:
            event.remove(engine, "before_cursor_execute", fail_insert)
        with Session(engine) as db:
            assert db.get(AttendanceRequest, payload["request_id"]) is None
            assert db.query(AttendanceEvent).count() == db.query(AttendanceDecision).count() == 0
        assert client.json("POST", "/api/attendance/manual", payload)["status"] == "ACCEPTED"


@test
def test_status_keeps_live_clock_fault_before_persistence_and_clears_reviewed_admission():
    with api() as (client, engine, service):
        clock, health = commission_ready(engine, service)
        client.json("POST", "/api/attendance/commission", commission_payload())
        service.persist_health([], healthy=True, reasons=[], observed_at=clock.utc)
        assert client.json("GET", "/api/attendance/status")["automatic_ready"]
        assert health.observe_clock(clock.utc, clock.mono)

        clock.advance(.1, utc_seconds=-5)
        assert not health.observe_clock(clock.utc, clock.mono)
        # No monitor tick has durably recorded this live technical failure yet.
        with Session(engine) as db:
            assert not any("CLOCK_DISCONTINUITY" in gap.reason_codes
                           for gap in db.query(AttendanceCoverageGap))
        live = client.json("GET", "/api/attendance/status")
        assert not live["technical_ready"] and not live["automatic_ready"]
        assert live["pending_fault_count"] > 0
        assert "CLOCK_DISCONTINUITY" in live["reasons"]

        clock.advance(.1)
        assert health.observe_clock(clock.utc, clock.mono)
        faults = health.pending_faults()
        service.persist_health([(fault, health.fault_source_ids(fault.id)) for fault in faults],
                               healthy=True, reasons=[], observed_at=clock.utc)
        for fault in faults:
            health.ack_fault(fault.id)
        durable = client.json("GET", "/api/attendance/status")
        assert durable["technical_ready"] and not durable["automatic_ready"]
        assert "CLOCK_DISCONTINUITY" in durable["reasons"]
        with Session(engine) as db:
            gap = next(gap for gap in db.query(AttendanceCoverageGap).filter_by(reviewed=False)
                       if "CLOCK_DISCONTINUITY" in gap.reason_codes)
            gap_id, revision = gap.id, gap.revision
        client.json("POST", f"/api/attendance/gaps/{gap_id}/review",
                    human(expected_revision=revision))
        # The health monitor has not refreshed its cached admission flag. A
        # completed human reconciliation must still clear that obsolete reason.
        assert "CLOCK_DISCONTINUITY" in health.snapshot()["reasons"]
        reviewed = client.json("GET", "/api/attendance/status")
        assert reviewed["technical_ready"] and reviewed["automatic_ready"]
        assert not reviewed["clock_blocked"]
        assert "CLOCK_DISCONTINUITY" not in reviewed["reasons"]
        health.set_pending_writes(1)
        assert "CLOCK_DISCONTINUITY" in health.snapshot()["reasons"]
        other_fault = client.json("GET", "/api/attendance/status")
        assert not other_fault["technical_ready"] and not other_fault["automatic_ready"]
        assert "PERSISTENCE_ERROR" in other_fault["reasons"]
        assert "CLOCK_DISCONTINUITY" not in other_fault["reasons"]


@test
def test_status_exposes_actual_readiness_providers_metrics_and_unresolved_counts():
    with api() as (client, _engine, service):
        review_crossing(service)
        status = client.json("GET", "/api/attendance/status")
        assert status["commission_id"] is None and not status["automatic_ready"]
        assert not status["technical_ready"] and status["unresolved_crossings"] == 1
        assert status["providers"]["yolo"] == ["CUDAExecutionProvider", "CPUExecutionProvider"]
        assert "pending_write_count" in status and "pending_fault_count" in status
        assert len(status["configuration"]) == 2
        assert "COMMISSIONING_REQUIRED" in status["reasons"]


@test
def test_commission_valid_attestation_is_atomic_replayable_and_controls_future_capture():
    with api() as (client, engine, service):
        trial = review_crossing(service)
        clock, health = commission_ready(engine, service)
        payload = commission_payload()
        first = client.json("POST", "/api/attendance/commission", payload)
        assert first["commission_id"] > 0
        with Session(engine) as db:
            record = db.get(AttendanceCommission, first["commission_id"])
            assert record.actor == "Operator" and record.acceptance_record["entry_trials"] == 30
            assert record.acceptance_record["site_confirmed"] is True
            assert record.configuration_fingerprint["providers"]["yolo"][0] == "CUDAExecutionProvider"
            assert db.get(AttendanceRuntime, 1).commission_id == record.id
        second = client.json("POST", "/api/attendance/commission", commission_payload())
        assert second["commission_id"] != first["commission_id"]
        assert client.json("POST", "/api/attendance/commission", payload) == first
        client.json("POST", "/api/attendance/commission", dict(payload, reason="Changed"), status=409)
        service.persist_health([], healthy=True, reasons=[], observed_at=clock.utc)
        captured = review_crossing(service, ordinal=2, hours=1)
        assert client.json("GET", f"/api/attendance/events/{captured}")["status"] == "ACCEPTED"
        assert client.json("GET", f"/api/attendance/events/{trial}")["status"] == "REVIEW"
        with immediate_session(engine) as db:
            db.get(AttendanceCamera, 1).revision += 1
        # A status read must not advertise stale approval while monitor waits.
        status = client.json("GET", "/api/attendance/status")
        assert not status["automatic_ready"] and status["commissioning_required"]
        with Session(engine) as db:
            assert db.query(AttendanceCommission).count() == 2


@test
def test_commission_rejects_counts_stale_revisions_cpu_and_unhealthy_metrics():
    with api() as (client, engine, service):
        _clock, health = commission_ready(engine, service)
        for changes in (dict(entry_trials=29), dict(exit_trials=29), dict(visitor_trials=19),
                        dict(retreat_trials=9), dict(employee_count=0), dict(wrong_employee=1),
                        dict(duplicate_trips=1), dict(missed_unflagged=1), dict(frontal_trials=0),
                        dict(frontal_eligible=94), dict(frontal_eligible=101), dict(performance_minutes=14),
                        dict(site_confirmed=False)):
            client.json("POST", "/api/attendance/commission", commission_payload(**changes), status=422)
        stale = dict(commission_payload(), expected_exit_revision=2)
        client.json("POST", "/api/attendance/commission", stale, status=409)
        cpu = copy.deepcopy(CUDA_PROVIDERS)
        cpu["live_face_detection"] = ["CPUExecutionProvider"]
        health.set_providers(cpu)
        client.json("POST", "/api/attendance/commission", commission_payload(), status=422)
        health.set_providers(CUDA_PROVIDERS)
        health.set_pending_writes(1)
        client.json("POST", "/api/attendance/commission", commission_payload(), status=422)
        health.set_pending_writes(0)
        with immediate_session(engine) as db:
            db.query(EmployeeFace).delete()
        client.json("POST", "/api/attendance/commission", commission_payload(), status=422)
        with Session(engine) as db:
            assert db.query(AttendanceCommission).count() == db.query(AttendanceRequest).count() == 0


@test
def test_commission_rejects_short_technical_streak_and_slow_completion():
    with api() as (client, engine, service):
        with immediate_session(engine) as db:
            db.add(EmployeeFace(employee_id=1, embedding=bytes(512 * 4)))
        clock = Clock()
        health = health_for(service, clock)
        warm(health, clock, seconds=9)
        assert health.snapshot()["technically_healthy_seconds"] < 10
        client.json("POST", "/api/attendance/commission", commission_payload(), status=422)
        warm(health, clock, seconds=21, completion=1.1)
        assert health.snapshot()["roles"]["ENTRY"]["completion_age_p95_seconds"] > 1
        client.json("POST", "/api/attendance/commission", commission_payload(), status=422)
        with Session(engine) as db:
            assert db.get(AttendanceRuntime, 1).commission_id is None
            assert db.query(AttendanceCommission).count() == 0


@test
def test_commission_employee_floor_counts_distinct_active_enrolled_people():
    with api() as (client, engine, service):
        commission_ready(engine, service)
        with immediate_session(engine) as db:
            db.add_all([EmployeeFace(employee_id=1, embedding=bytes(512 * 4)),
                        EmployeeFace(employee_id=2, embedding=bytes(512 * 4))])
        client.json("POST", "/api/attendance/commission", commission_payload(employee_count=1), status=422)
        approved = client.json("POST", "/api/attendance/commission", commission_payload(employee_count=2))
        assert approved["commission_id"] > 0
        with Session(engine) as db:
            assert db.query(AttendanceCommission).count() == 1


@test
def test_legacy_archive_is_separate_offset_explicit_and_not_trusted_for_hours():
    with api() as (client, engine, _service):
        with immediate_session(engine) as db:
            db.add(PresenceSession(employee_id=1, camera_id=1, started_at=BASE,
                                   last_seen_at=BASE + timedelta(seconds=30), ended_at=None, confidence=.9))
        archive = client.json("GET", "/api/presence/legacy?employee_id=1&from_date=2026-11-11&to_date=2026-11-11")
        assert archive["total"] == 1 and not archive["trusted_for_hours"]
        assert archive["items"][0]["historical_unclosed"]
        assert archive["items"][0]["started_at"].endswith("+05:00")
        assert client.json("GET", "/api/attendance/sessions")["total"] == 0
        client.json("GET", "/api/presence", status=404)


if __name__ == "__main__":
    failed = 0
    for case in TESTS:
        try:
            case()
            print(f"PASS {case.__name__}")
        except Exception:
            failed += 1
            print(f"FAIL {case.__name__}")
            traceback.print_exc()
    print(f"{len(TESTS) - failed}/{len(TESTS)} passed")
    raise SystemExit(1 if failed else 0)
