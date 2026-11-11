"""Date reports exercise real ledger projections, SQLite snapshots and HTTP CSV.

Run: .venv/Scripts/python.exe tests/test_attendance_reports.py
The controller owns RED/GREEN execution. No production database or model startup.
"""
import csv
import io
import sys
import threading
import traceback
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi import HTTPException
from sqlalchemy import event
from sqlalchemy.orm import Session

from app.attendance_types import GapDecisionInput, VisitorBatchInput
from app.database.database import immediate_session
from app.database.models import (
    AttendanceCommission, AttendanceCoverageGap, AttendanceEvent, AttendanceRuntime, Employee, WorkSession,
)
from app.services.attendance_queries import read_snapshot
from test_attendance_api import api as attendance_api
from test_attendance_service import BASE, crossing, database, decision, manual

@contextmanager
def api():
    # Report cases start with approved historical coverage. The API harness still
    # uses real routers/services and never starts production lifespan or models.
    with attendance_api() as (client, engine, service):
        with immediate_session(engine) as db:
            commission = AttendanceCommission(
                request_id=str(uuid4()), actor="Site operator", reason="Historical report fixture",
                created_at=BASE - timedelta(days=2),
                configuration_fingerprint=service.configuration_fingerprint(db),
                acceptance_record={"site_confirmed": True},
            )
            db.add(commission)
            db.flush()
            db.get(AttendanceRuntime, 1).commission_id = commission.id
        yield client, engine, service


TESTS = []
DAY = date(2026, 11, 11)
NEXT = date(2026, 11, 12)
FINISHED = datetime(2026, 11, 13, 0)
HEADER = ["date", "employee_id", "employee_name", "confirmed_hms", "running_hms",
          "review_hms", "visits", "first_entry", "last_exit", "pending", "status", "report_finalizable"]


def test(fn):
    TESTS.append(fn)
    return fn


def build(db, start=DAY, end=DAY, employee=1, *, as_of=FINISHED):
    from app.services.attendance_reports import build_report
    return build_report(db, start, end, employee, as_of=as_of).model_dump(mode="json")


def report(engine, start=DAY, end=DAY, employee=1, *, as_of=FINISHED):
    return read_snapshot(engine, lambda db: build(db, start, end, employee, as_of=as_of))


def row(result, day="2026-11-11", employee=1):
    return next(item for item in result["rows"] if item["date"] == day and item["employee_id"] == employee)


def lunch(service):
    for direction, hours in (("ENTRY", 0), ("EXIT", 3), ("ENTRY", 4), ("EXIT", 9)):
        manual(service, direction, BASE + timedelta(hours=hours))


def visit(service, identifier):
    return service.mark_visitors(VisitorBatchInput(
        request_id=uuid4(), items=[dict(event_id=identifier, expected_revision=0)],
        actor="Operator", reason="Verified visitor",
    ))


def gap(engine, start, end, reason="FRAME_GAP", reviewed=False):
    with immediate_session(engine) as db:
        item = AttendanceCoverageGap(started_at=start, ended_at=end,
                                     reason_codes=[reason], reviewed=reviewed)
        db.add(item)
        db.flush()
        return item.id


@test
def test_lunch_counts_only_confirmed_visits_with_exact_public_shape():
    with database(commissioned=True) as (engine, service):
        lunch(service)
        result = report(engine)
        item = row(result)
        assert set(result) == {"from_date", "to_date", "utc_offset_minutes", "as_of", "finalizable",
                               "unresolved_crossings", "unreviewed_gaps", "rows", "totals"}
        assert set(item) == {"date", "employee_id", "employee_name", "confirmed_ms", "running_ms",
                             "review_ms", "visit_count", "first_entry", "last_exit", "pending_count",
                             "status", "coverage_reasons"}
        assert set(result["totals"][0]) == {"employee_id", "employee_name", "confirmed_ms", "running_ms",
                                           "review_ms", "pending_count"}
        assert (item["confirmed_ms"], item["running_ms"], item["review_ms"]) == (28_800_000, 0, 0)
        assert item["visit_count"] == 2 and item["pending_count"] == 0
        assert item["first_entry"] == "2026-11-11T09:00:00+05:00"
        assert item["last_exit"] == "2026-11-11T18:00:00+05:00"
        assert item["status"] == "CONFIRMED" and item["coverage_reasons"] == []
        assert result["utc_offset_minutes"] == 300 and result["as_of"].endswith("+05:00")
        assert result["finalizable"] and result["totals"][0]["confirmed_ms"] == 28_800_000
        assert result["unresolved_crossings"] == result["unreviewed_gaps"] == 0


@test
def test_overnight_splits_local_midnight_without_inventing_second_visit():
    with database(commissioned=True) as (engine, service):
        manual(service, "ENTRY", datetime(2026, 11, 11, 18, 30))
        manual(service, "EXIT", datetime(2026, 11, 11, 20, 15))
        result = report(engine, DAY, NEXT)
        first, second = row(result), row(result, "2026-11-12")
        assert first["confirmed_ms"] == 1_800_000 and second["confirmed_ms"] == 4_500_000
        assert first["visit_count"] == 1 and second["visit_count"] == 0
        assert first["last_exit"] is None and second["first_entry"] is None
        assert second["last_exit"] == "2026-11-12T01:15:00+05:00"
        assert result["totals"][0]["confirmed_ms"] == 6_300_000 and result["finalizable"]


@test
def test_entry_at_midnight_belongs_only_to_new_local_day():
    with database(commissioned=True) as (engine, service):
        manual(service, "ENTRY", datetime(2026, 11, 11, 19))
        manual(service, "EXIT", datetime(2026, 11, 11, 20))
        result = report(engine, DAY, NEXT)
        assert row(result)["confirmed_ms"] == row(result)["visit_count"] == 0
        assert row(result)["first_entry"] is None
        assert row(result, "2026-11-12")["confirmed_ms"] == 3_600_000
        assert row(result, "2026-11-12")["visit_count"] == 1


@test
def test_open_noon_clock_is_running_not_confirmed_and_never_auto_closed():
    with database(commissioned=True) as (engine, service):
        manual(service, "ENTRY", BASE)
        result = report(engine, as_of=BASE + timedelta(hours=3))
        item = row(result)
        assert (item["confirmed_ms"], item["running_ms"], item["review_ms"]) == (0, 10_800_000, 0)
        assert item["status"] == "RUNNING" and item["last_exit"] is None and not result["finalizable"]
        with Session(engine) as db:
            assert db.query(WorkSession).one().ended_at is None


@test
def test_integer_milliseconds_are_truncated_once_and_aware_as_of_is_normalized():
    with database(commissioned=True) as (engine, service):
        manual(service, "ENTRY", BASE.replace(microsecond=1999))
        manual(service, "EXIT", BASE.replace(microsecond=3001))
        result = report(engine, as_of=FINISHED.replace(tzinfo=timezone.utc, microsecond=999))
        assert row(result)["confirmed_ms"] == 2
        assert result["as_of"] == "2026-11-13T05:00:00+05:00"


@test
def test_missing_entry_and_duplicate_entry_never_add_guessed_hours_and_void_repairs():
    with database(commissioned=True) as (engine, service):
        orphan = manual(service, "EXIT", BASE - timedelta(hours=1))
        manual(service, "ENTRY", BASE)
        duplicate = manual(service, "ENTRY", BASE + timedelta(hours=1))
        manual(service, "EXIT", BASE + timedelta(hours=9))
        disputed = report(engine)
        item = row(disputed)
        assert item["confirmed_ms"] == 0 and item["review_ms"] == 32_400_000
        assert item["visit_count"] == 0 and item["pending_count"] == 3
        assert item["first_entry"] == "2026-11-11T09:00:00+05:00"
        assert "MISSING_ENTRY" in item["coverage_reasons"] and not disputed["finalizable"]
        decision(service, orphan.id, "VOID")
        decision(service, duplicate.id, "VOID")
        repaired = row(report(engine))
        assert repaired["confirmed_ms"] == 32_400_000 and repaired["pending_count"] == 0
        assert repaired["visit_count"] == 1 and repaired["status"] == "CONFIRMED"


@test
def test_exit_without_entry_has_no_duration_but_counts_unresolved_event():
    with database(commissioned=True) as (engine, service):
        manual(service, "EXIT", BASE)
        result = report(engine)
        item = row(result)
        assert item["confirmed_ms"] == item["running_ms"] == item["review_ms"] == 0
        assert item["pending_count"] == 1 and item["visit_count"] == 0
        assert item["first_entry"] is None and item["last_exit"] is None
        assert item["status"] == "REVIEW" and not result["finalizable"]


@test
def test_unknown_lunch_quarantines_whole_span_and_assignment_recomputes_eight_hours():
    with database(commissioned=True) as (engine, service):
        manual(service, "ENTRY", BASE)
        manual(service, "EXIT", BASE + timedelta(hours=9))
        identifiers = [service.record_crossing(crossing(when=BASE + timedelta(hours=hours),
                       ordinal=index, employee=None, direction=direction, eligible=False))
                       for index, (hours, direction) in enumerate(((3, "EXIT"), (4, "ENTRY")), 1)]
        result = report(engine, employee=None)
        item = row(result)
        assert item["confirmed_ms"] == 0 and item["review_ms"] == 32_400_000
        assert result["unresolved_crossings"] == 2 and item["pending_count"] == 3
        assert row(result, employee=2)["status"] == "REVIEW"
        for identifier, direction in zip(identifiers, ("EXIT", "ENTRY")):
            decision(service, identifier, "ACCEPT", employee_id=1, direction=direction)
        repaired = report(engine)
        assert row(repaired)["confirmed_ms"] == 28_800_000
        assert row(repaired)["review_ms"] == 0 and row(repaired)["pending_count"] == 0
        assert repaired["unresolved_crossings"] == 0 and repaired["finalizable"]


@test
def test_unknown_before_requested_range_extends_to_as_of_and_visitor_removes_it():
    with database(commissioned=True) as (engine, service):
        manual(service, "ENTRY", BASE + timedelta(days=1))
        manual(service, "EXIT", BASE + timedelta(days=1, hours=3))
        unknown = service.record_crossing(crossing(when=BASE, employee=None, eligible=False))
        result = report(engine, NEXT, NEXT)
        assert row(result, "2026-11-12")["review_ms"] == 10_800_000
        assert result["unresolved_crossings"] == 1 and not result["finalizable"]
        visit(service, unknown)
        repaired = report(engine, NEXT, NEXT)
        assert row(repaired, "2026-11-12")["confirmed_ms"] == 10_800_000
        assert repaired["unresolved_crossings"] == 0 and repaired["finalizable"]


@test
def test_whole_overnight_span_is_quarantined_before_splitting_days_and_pending_deduplicates():
    with database(commissioned=True) as (engine, service):
        manual(service, "ENTRY", datetime(2026, 11, 11, 18, 30))
        manual(service, "EXIT", datetime(2026, 11, 11, 20, 15))
        service.record_crossing(crossing(when=datetime(2026, 11, 11, 20), employee=None, eligible=False))
        result = report(engine, DAY, NEXT)
        assert row(result)["review_ms"] == 1_800_000 and row(result)["confirmed_ms"] == 0
        assert row(result, "2026-11-12")["review_ms"] == 4_500_000
        assert row(result)["pending_count"] == 1  # disputed session, blocker begins tomorrow
        assert row(result, "2026-11-12")["pending_count"] == 2
        assert result["totals"][0]["pending_count"] == 2 and result["unresolved_crossings"] == 1


@test
def test_half_open_blocker_at_exit_preserves_session_but_at_entry_quarantines():
    for hours, confirmed, review in ((3, 10_800_000, 0), (0, 0, 10_800_000)):
        with database(commissioned=True) as (engine, service):
            manual(service, "ENTRY", BASE)
            manual(service, "EXIT", BASE + timedelta(hours=3))
            service.record_crossing(crossing(when=BASE + timedelta(hours=hours), employee=None, eligible=False))
            item = row(report(engine))
            assert (item["confirmed_ms"], item["review_ms"]) == (confirmed, review)
            assert item["status"] == "REVIEW"  # whole-day completeness still blocked


@test
def test_assigned_review_quarantines_only_that_employee():
    with database(commissioned=True) as (engine, service):
        for employee in (1, 2):
            manual(service, "ENTRY", BASE, employee=employee)
            manual(service, "EXIT", BASE + timedelta(hours=3), employee=employee)
        with immediate_session(engine) as db:
            pending = AttendanceEvent(id=str(uuid4()), source_key=str(uuid4()), source="CAMERA", raw={"version": 1},
                                      employee_id=1, employee_name="Employee 1", direction="EXIT",
                                      occurred_at=BASE + timedelta(hours=1), recorded_at=BASE, status="REVIEW",
                                      reason_codes=["NO_FACE"], projection_reasons=[])
            db.add(pending)
        result = report(engine, employee=None)
        assert row(result)["review_ms"] == 10_800_000
        assert row(result, employee=2)["confirmed_ms"] == 10_800_000
        assert row(result, employee=2)["status"] == "CONFIRMED"


@test
def test_gap_quarantines_entire_session_and_dates_without_sessions_until_reviewed():
    with database(commissioned=True) as (engine, service):
        lunch(service)
        identifier = gap(engine, BASE + timedelta(hours=1), BASE + timedelta(days=1, hours=1))
        result = report(engine, DAY, NEXT)
        assert row(result)["review_ms"] == 28_800_000
        assert row(result, "2026-11-12")["status"] == "REVIEW"
        assert row(result, "2026-11-12")["confirmed_ms"] == 0
        assert result["unreviewed_gaps"] == 1 and result["unresolved_crossings"] == 0
        service.review_gap(identifier, GapDecisionInput(request_id=uuid4(), expected_revision=0,
                           actor="Operator", reason="Reconciled missing evidence"))
        repaired = report(engine, DAY, NEXT)
        assert row(repaired)["confirmed_ms"] == 28_800_000 and repaired["unreviewed_gaps"] == 0
        assert repaired["finalizable"]


@test
def test_open_gap_and_earlier_gap_are_included_only_when_touching_requested_days():
    with database(commissioned=True) as (engine, service):
        gap(engine, BASE - timedelta(days=2), BASE - timedelta(days=1))
        gap(engine, BASE - timedelta(hours=1), None)
        result = report(engine, DAY, NEXT)
        assert result["unreviewed_gaps"] == 1
        assert all(item["status"] == "REVIEW" for item in result["rows"])
        assert all("FRAME_GAP" in item["coverage_reasons"] for item in result["rows"])


@test
def test_unmonitored_prefix_blocks_completeness_not_explicit_manual_hours():
    for began in (None, BASE + timedelta(hours=1)):
        with database(commissioned=True) as (engine, service):
            lunch(service)
            with immediate_session(engine) as db:
                db.get(AttendanceRuntime, 1).monitoring_started_at = began
            result = report(engine)
            assert row(result)["confirmed_ms"] == 28_800_000
            assert row(result)["review_ms"] == 0
            assert "UNMONITORED" in row(result)["coverage_reasons"]
            assert row(result)["status"] == "REVIEW" and not result["finalizable"]


@test
def test_future_and_today_are_nonfinalizable_even_without_employees():
    with database(commissioned=True) as (engine, _service):
        result = report(engine, DAY, NEXT, as_of=BASE)
        assert row(result)["status"] == "RUNNING"
        assert row(result, "2026-11-12")["status"] == "REVIEW"
        assert not result["finalizable"]
        with immediate_session(engine) as db:
            for employee in db.query(Employee):
                employee.active = False
        empty = report(engine, DAY, NEXT, employee=None, as_of=BASE)
        assert empty["rows"] == empty["totals"] == [] and not empty["finalizable"]
        with immediate_session(engine) as db:
            db.get(AttendanceRuntime, 1).monitoring_started_at = None
        assert not report(engine, employee=None)["finalizable"]


@test
def test_inactive_history_is_included_and_historical_snapshot_name_survives_rename():
    with database(commissioned=True) as (engine, service):
        lunch(service)
        with immediate_session(engine) as db:
            employee = db.get(Employee, 1)
            employee.active = False
            employee.name = "Renamed after attendance"
        result = report(engine, employee=None)
        assert {item["employee_id"] for item in result["rows"]} == {1, 2}
        assert row(result)["employee_name"] == "Employee 1"
        assert next(item for item in result["totals"] if item["employee_id"] == 1)["employee_name"] == "Employee 1"
        assert row(result)["confirmed_ms"] == 28_800_000
        explicit = report(engine, NEXT, NEXT, employee=1)
        assert len(explicit["rows"]) == 1 and row(explicit, "2026-11-12")["confirmed_ms"] == 0


@test
def test_active_empty_employee_has_daily_zero_rows_and_no_employee_range_is_empty():
    with database(commissioned=True) as (engine, _service):
        result = report(engine, DAY, NEXT)
        assert len(result["rows"]) == 2 and len(result["totals"]) == 1
        assert all(item["confirmed_ms"] == item["visit_count"] == 0 for item in result["rows"])
        assert result["finalizable"]
        with immediate_session(engine) as db:
            for employee in db.query(Employee):
                employee.active = False
        result = report(engine, employee=None)
        assert result["rows"] == result["totals"] == [] and result["finalizable"]


@test
def test_range_validation_and_missing_employee_return_no_partial_result():
    with database(commissioned=True) as (engine, _service):
        for start, end, employee, status in ((NEXT, DAY, 1, 422),
                                           (DAY, DAY + timedelta(days=366), 1, 422),
                                           (DAY, DAY, 999, 404)):
            try:
                report(engine, start, end, employee)
            except HTTPException as exc:
                assert exc.status_code == status
            else:
                raise AssertionError((start, end, employee, "should reject"))
        result = report(engine, DAY, DAY + timedelta(days=365))
        assert len(result["rows"]) == 366


@test
def test_more_than_page_size_events_are_not_silently_truncated():
    with database(commissioned=True) as (engine, service):
        for index in range(205):
            start = BASE + timedelta(seconds=index * 2)
            manual(service, "ENTRY", start)
            manual(service, "EXIT", start + timedelta(seconds=1))
        item = row(report(engine))
        assert item["confirmed_ms"] == 205_000 and item["visit_count"] == 205
        assert item["pending_count"] == 0


@test
def test_snapshot_cannot_mix_old_blocker_counts_with_visitor_and_manual_commit():
    with database(commissioned=True) as (engine, service):
        manual(service, "ENTRY", BASE)
        manual(service, "EXIT", BASE + timedelta(hours=3))
        unknown = service.record_crossing(crossing(when=BASE + timedelta(hours=1), employee=None, eligible=False))
        errors = []

        def mutate():
            try:
                visit(service, unknown)
                manual(service, "ENTRY", BASE + timedelta(hours=4))
                manual(service, "EXIT", BASE + timedelta(hours=9))
            except Exception as exc:
                errors.append(exc)

        def read(db):
            assert db.query(AttendanceEvent).count() == 3  # establish real SQLite snapshot
            writer = threading.Thread(target=mutate)
            writer.start()
            writer.join(10)
            assert not writer.is_alive() and not errors, errors
            return build(db)

        before = read_snapshot(engine, read)
        assert before["unresolved_crossings"] == 1
        assert row(before)["confirmed_ms"] == 0 and row(before)["review_ms"] == 10_800_000
        after = report(engine)
        assert after["unresolved_crossings"] == 0 and row(after)["confirmed_ms"] == 28_800_000


@test
def test_report_routes_parse_errors_and_csv_agrees_with_json_and_floor_duration():
    with api() as (client, _engine, service):
        lunch(service)
        url = "/api/attendance/report?from_date=2026-11-11&to_date=2026-11-11&employee_id=1"
        result = client.json("GET", url)
        assert row(result)["confirmed_ms"] == 28_800_000 and result["finalizable"]
        status, headers, content = client.request("GET", url.replace("/report?", "/report.csv?"))
        assert status == 200 and headers[b"content-type"].startswith(b"text/csv")
        assert content.startswith(b"\xef\xbb\xbf")
        parsed = list(csv.reader(io.StringIO(content.decode("utf-8-sig"))))
        assert parsed[0] == HEADER and len(parsed) == 2
        assert parsed[1] == ["2026-11-11", "1", "Employee 1", "08:00:00", "00:00:00", "00:00:00", "2",
                             "2026-11-11T09:00:00+05:00", "2026-11-11T18:00:00+05:00", "0", "CONFIRMED", "true"]
        for query in ("from_date=bad&to_date=2026-11-11", "from_date=2026-11-12&to_date=2026-11-11",
                      "from_date=2026-11-11&to_date=2027-11-12", "from_date=2026-11-11&to_date=2026-11-11&employee_id=999"):
            expected = 404 if "999" in query else 422
            for endpoint in ("report", "report.csv"):
                client.json("GET", f"/api/attendance/{endpoint}?{query}", status=expected)


@test
def test_csv_quotes_and_neutralizes_every_formula_prefix_without_changing_json_name():
    for name in ('=HYPERLINK("bad")', "+command", "-command", "@command", "\tcommand", "\rcommand", 'Surname, "Full" Name'):
        with api() as (client, engine, service):
            with immediate_session(engine) as db:
                db.get(Employee, 1).name = name
            lunch(service)
            url = "/api/attendance/report?from_date=2026-11-11&to_date=2026-11-11&employee_id=1"
            assert row(client.json("GET", url))["employee_name"] == name
            status, _headers, content = client.request("GET", url.replace("/report?", "/report.csv?"))
            assert status == 200
            records = list(csv.reader(io.StringIO(content.decode("utf-8-sig"), newline="")))
            assert records[1][2] == ("'" + name if name[0] in "=+-@\t\r" else name)


@test
def test_csv_hours_do_not_wrap_after_twenty_four_or_round_up_subseconds():
    with api() as (client, _engine, service):
        manual(service, "ENTRY", datetime(2026, 11, 10, 19))
        manual(service, "EXIT", datetime(2026, 11, 11, 19) - timedelta(milliseconds=1))
        url = "/api/attendance/report.csv?from_date=2026-11-11&to_date=2026-11-11&employee_id=1"
        status, _headers, content = client.request("GET", url)
        assert status == 200
        records = list(csv.reader(io.StringIO(content.decode("utf-8-sig"))))
        assert records[1][3] == "23:59:59"
        from app.services.attendance_reports import format_hms
        assert format_hms(108_000_999) == "30:00:00"


@test
def test_http_snapshot_remains_consistent_when_visitor_commits_during_report_read():
    with api() as (client, engine, service):
        lunch(service)
        unknown = service.record_crossing(crossing(when=BASE + timedelta(hours=1), employee=None, eligible=False))
        changed = False
        errors = []

        def on_read(connection, cursor, statement, parameters, context, many):
            nonlocal changed
            if changed or not statement.lstrip().upper().startswith("SELECT"):
                return
            changed = True
            try:
                visit(service, unknown)
            except Exception as exc:
                errors.append(exc)

        event.listen(engine, "after_cursor_execute", on_read)
        try:
            result = client.json("GET", "/api/attendance/report?from_date=2026-11-11&to_date=2026-11-11&employee_id=1")
        finally:
            event.remove(engine, "after_cursor_execute", on_read)
        assert changed and not errors, errors
        assert result["unresolved_crossings"] == 1
        assert row(result)["confirmed_ms"] == 0 and row(result)["review_ms"] == 28_800_000
        assert row(report(engine))["confirmed_ms"] == 28_800_000


@test
def test_exit_at_midnight_is_reported_on_new_day_even_with_no_duration_there():
    with database(commissioned=True) as (engine, service):
        manual(service, "ENTRY", datetime(2026, 11, 11, 18))
        manual(service, "EXIT", datetime(2026, 11, 11, 19))
        result = report(engine, DAY, NEXT)
        assert row(result)["confirmed_ms"] == 3_600_000
        following = row(result, "2026-11-12")
        assert following["confirmed_ms"] == following["visit_count"] == 0
        assert following["first_entry"] is None
        assert following["last_exit"] == "2026-11-12T00:00:00+05:00"
        assert row(report(engine, NEXT, NEXT), "2026-11-12")["last_exit"] == following["last_exit"]


@test
def test_simultaneous_events_make_no_visits_or_duration_until_voided():
    with database(commissioned=True) as (engine, service):
        manual(service, "ENTRY", BASE)
        exit_event = manual(service, "EXIT", BASE)
        result = report(engine)
        item = row(result)
        assert item["confirmed_ms"] == item["running_ms"] == item["review_ms"] == 0
        assert item["visit_count"] == 0 and item["pending_count"] == 2
        assert item["last_exit"] is None and "SIMULTANEOUS_EVENTS" in item["coverage_reasons"]
        decision(service, exit_event.id, "VOID")
        repaired = row(report(engine, as_of=BASE + timedelta(hours=3)))
        assert repaired["running_ms"] == 10_800_000 and repaired["pending_count"] == 0


@test
def test_multi_day_running_period_total_exceeds_twenty_four_hours():
    with database(commissioned=True) as (engine, service):
        manual(service, "ENTRY", BASE)
        result = report(engine, DAY, date(2026, 11, 13), as_of=BASE + timedelta(days=2))
        assert [item["running_ms"] for item in result["rows"]] == [54_000_000, 86_400_000, 32_400_000]
        assert result["totals"][0]["running_ms"] == 172_800_000
        assert result["totals"][0]["confirmed_ms"] == 0 and not result["finalizable"]


@test
def test_inactive_earlier_conflict_keeps_empty_dates_reviewable_with_snapshot_name():
    with database(commissioned=True) as (engine, service):
        manual(service, "EXIT", BASE - timedelta(days=1))
        with immediate_session(engine) as db:
            employee = db.get(Employee, 1)
            employee.active = False
            employee.name = "Renamed after unresolved history"
        result = report(engine, employee=None)
        item = row(result)
        assert item["employee_name"] == "Employee 1"
        assert item["status"] == "REVIEW" and item["pending_count"] == 1
        assert item["confirmed_ms"] == item["review_ms"] == 0
        assert "MISSING_ENTRY" in item["coverage_reasons"] and not result["finalizable"]


@test
def test_csv_running_and_review_columns_match_json_without_becoming_confirmed():
    with api() as (client, _engine, service):
        manual(service, "ENTRY", BASE)
        url = "/api/attendance/report?from_date=2026-11-11&to_date=2026-11-12&employee_id=1"
        before = client.json("GET", url)
        assert [item["running_ms"] for item in before["rows"]] == [54_000_000, 72_000_000]
        status, _headers, content = client.request("GET", url.replace("/report?", "/report.csv?"))
        records = list(csv.reader(io.StringIO(content.decode("utf-8-sig"))))
        assert status == 200 and [record[4] for record in records[1:]] == ["15:00:00", "20:00:00"]
        assert all(record[3] == record[5] == "00:00:00" and record[-1] == "false" for record in records[1:])
        service.record_crossing(crossing(when=BASE + timedelta(hours=1), employee=None, eligible=False))
        after = client.json("GET", url)
        assert [item["review_ms"] for item in after["rows"]] == [54_000_000, 72_000_000]
        status, _headers, content = client.request("GET", url.replace("/report?", "/report.csv?"))
        records = list(csv.reader(io.StringIO(content.decode("utf-8-sig"))))
        assert status == 200 and [record[5] for record in records[1:]] == ["15:00:00", "20:00:00"]
        assert all(record[3] == record[4] == "00:00:00" and record[-1] == "false" for record in records[1:])


@test
def test_empty_csv_has_bom_and_header_without_synthetic_employee_rows():
    with api() as (client, engine, _service):
        with immediate_session(engine) as db:
            for employee in db.query(Employee):
                employee.active = False
        url = "/api/attendance/report?from_date=2026-11-11&to_date=2026-11-11"
        result = client.json("GET", url)
        assert result["rows"] == result["totals"] == []
        status, _headers, content = client.request("GET", url.replace("/report?", "/report.csv?"))
        assert status == 200 and content.startswith(b"\xef\xbb\xbf")
        assert list(csv.reader(io.StringIO(content.decode("utf-8-sig")))) == [HEADER]


@test
def test_unknown_at_midnight_blocks_new_date_not_completed_previous_date():
    with database(commissioned=True) as (engine, service):
        service.record_crossing(crossing(when=datetime(2026, 11, 11, 19), employee=None, eligible=False))
        previous = report(engine)
        assert row(previous)["status"] == "CONFIRMED" and previous["finalizable"]
        assert previous["unresolved_crossings"] == 0
        following = report(engine, NEXT, NEXT)
        assert row(following, "2026-11-12")["status"] == "REVIEW"
        assert following["unresolved_crossings"] == 1 and not following["finalizable"]


if __name__ == "__main__":
    failed = 0
    for fn in TESTS:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception:
            failed += 1
            print(f"FAIL {fn.__name__}")
            traceback.print_exc()
    print(f"{len(TESTS) - failed}/{len(TESTS)} passed")
    sys.exit(1 if failed else 0)
