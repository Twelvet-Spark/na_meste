"""Ledger/replay/trust contracts against isolated file SQLite, never production data.

Run: .venv/Scripts/python.exe tests/test_attendance_service.py
The controller owns RED/GREEN execution. Model observations are immutable records;
transactions, reducers, requests, constraints and query snapshots are real.
"""
import copy
import sys
import tempfile
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import UUID, uuid4

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from fastapi import HTTPException
from sqlalchemy import create_engine, event
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from app.attendance_types import EventDecisionInput, GapDecisionInput, ManualEventInput, VisitorBatchInput
from app.database.attendance_schema import initialize_attendance_schema
from app.database.database import immediate_session
from app.database.models import (
    AttendanceCamera, AttendanceCommission, AttendanceCoverageGap, AttendanceDecision,
    AttendanceEmployeeReference, AttendanceEvent, AttendanceEvidence, AttendanceRequest,
    AttendanceRuntime, Camera, Employee, PresenceSession, WorkSession,
)
from app.rtsp.reader import FramePacket
from app.services.attendance_health import AttendanceHealth
from app.services.attendance_monitor import AttendanceMonitor
from app.services.attendance_queries import (
    TrustBlocker, evaluate_session, load_trust_blockers, read_snapshot,
)
from app.services.attendance_service import AttendanceService
from test_attendance_crossings import Trajectory, single_crossing

TESTS = []
BASE = datetime(2026, 11, 11, 4)  # 09:00 UTC+05:00
AS_OF = datetime(2026, 11, 12, 15)
CUDA_PROVIDERS = {
    key: ["CUDAExecutionProvider", "CPUExecutionProvider"] for key in (
        "yolo", "live_face_detection", "live_face_recognition",
        "enroll_face_detection", "enroll_face_recognition",
    )
}
CUDA_PROVIDERS["reid"] = []


def test(fn):
    TESTS.append(fn)
    return fn


def rejects(error_type, operation):
    try:
        operation()
    except error_type as exc:
        return exc
    name = getattr(error_type, "__name__", str(error_type))
    raise AssertionError(f"Expected {name}")


def error_status(status, operation):
    error = rejects(HTTPException, operation)
    assert error.status_code == status, error


@contextmanager
def database(*, commissioned=False):
    with tempfile.TemporaryDirectory() as directory:
        engine = create_engine(
            f"sqlite:///{Path(directory).as_posix()}/attendance.db",
            connect_args={"timeout": 30, "check_same_thread": False},
        )

        @event.listens_for(engine, "connect")
        def pragmas(connection, _record):
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA foreign_keys=ON")

        initialize_attendance_schema(engine)
        try:
            with immediate_session(engine) as db:
                db.add_all([Employee(id=1, name="Employee 1"), Employee(id=2, name="Other")])
                for camera_id, role in ((1, "ENTRY"), (2, "EXIT")):
                    db.add(Camera(id=camera_id, name=role, nvr_host="unused", username="u",
                                  password="p", channel=camera_id, enabled=True))
                db.flush()
                for camera_id, role in ((1, "ENTRY"), (2, "EXIT")):
                    db.add(AttendanceCamera(
                        camera_id=camera_id, role=role, enabled=True, revision=3, source_revision=2,
                        line_start=[.2, .5], line_end=[.8, .5], inside_point=[.5, .8],
                        reference_width=640, reference_height=480,
                    ))
                runtime = db.get(AttendanceRuntime, 1)
                runtime.gallery_revision = 7
                runtime.monitoring_started_at = BASE - timedelta(days=1)
                runtime.heartbeat_at = BASE - timedelta(seconds=1)
            service = AttendanceService(engine, now=lambda: AS_OF)
            service.health.set_providers(CUDA_PROVIDERS)
            if commissioned:
                # This fixture installs an already-approved record; it is NOT an
                # attestation bypass API. The step-6 endpoint validates attestation.
                with immediate_session(engine) as db:
                    commission = AttendanceCommission(
                        request_id=str(uuid4()), actor="Site operator", reason="Approved fixture",
                        configuration_fingerprint=service.configuration_fingerprint(db),
                        acceptance_record={"site_confirmed": True},
                    )
                    db.add(commission)
                    db.flush()
                    db.get(AttendanceRuntime, 1).commission_id = commission.id
            yield engine, service
        finally:
            engine.dispose()


def crossing(*, when=BASE, ordinal=1, employee=1, direction="ENTRY", eligible=True):
    # Obtain real immutable evidence and bounded JPEGs from the actual detector.
    path = Trajectory(role=direction)
    path.walk([.25, .3, .45, .55, .7] if direction == "ENTRY" else [.75, .7, .55, .45, .3])
    evidence = single_crossing(path, direction=direction)
    camera_id = 1 if direction == "ENTRY" else 2
    delta = when - evidence.occurred_at
    faces = tuple(replace(
        face, received_at=face.received_at + delta,
        supporting_received_ats=tuple(value + delta for value in face.supporting_received_ats),
    ) for face in evidence.face_evidence)
    return replace(
        evidence, track_key=(camera_id, "stream-a", "generation-a", 1),
        calibration=replace(evidence.calibration, camera_id=camera_id),
        crossing_index=ordinal, occurred_at=when, confirmed_employee_id=employee,
        confirmed_employee_name=f"Employee {employee}" if employee else None,
        face_evidence=faces, auto_eligible=eligible,
        disqualifying_reasons=() if eligible else ("NO_FACE",),
    )


def manual(service, direction, when, *, employee=1, request_id=None):
    return service.create_manual_event(ManualEventInput(
        request_id=request_id or uuid4(), employee_id=employee, direction=direction,
        occurred_at=when.replace(tzinfo=timezone.utc), actor=" Operator ", reason=" Reconciled doorway ",
    ))


def decision(service, event_id, action, revision=0, **changes):
    payload = dict(request_id=uuid4(), expected_revision=revision, action=action,
                   actor="Operator", reason="Reconciled observation")
    payload.update(changes)
    return service.decide_event(event_id, EventDecisionInput(**payload))


def sessions(engine, *, as_of=AS_OF):
    def read(db):
        blockers = load_trust_blockers(db, as_of=as_of)
        return [evaluate_session(row, blockers, as_of=as_of) for row in
                db.query(WorkSession).order_by(WorkSession.started_at, WorkSession.entry_event_id)]
    return read_snapshot(engine, read)


def projection(engine):
    def read(db):
        return (
            [(row.id, row.status, list(row.projection_reasons)) for row in
             db.query(AttendanceEvent).order_by(AttendanceEvent.id)],
            [(row.entry_event_id, row.exit_event_id, row.started_at, row.ended_at,
              row.status, list(row.reason_codes)) for row in
             db.query(WorkSession).order_by(WorkSession.entry_event_id)],
        )
    return read_snapshot(engine, read)


@test
def test_lunch_departures_stop_time_and_service_restart_never_invents_exit():
    with database() as (engine, service):
        for direction, hours in (("ENTRY", 0), ("EXIT", 3), ("ENTRY", 4), ("EXIT", 9)):
            manual(service, direction, BASE + timedelta(hours=hours))
        rows = sessions(engine)
        assert [row.confirmed_ms for row in rows] == [10_800_000, 18_000_000]
        assert sum(row.confirmed_ms for row in rows) == 28_800_000
        assert [row.status for row in rows] == ["CLOSED", "CLOSED"]
        entry = manual(service, "ENTRY", BASE + timedelta(days=1))
        restarted = AttendanceService(engine, now=lambda: AS_OF)
        rows = sessions(engine, as_of=BASE + timedelta(days=1, hours=3))
        assert rows[-1].entry_event_id == entry.id and rows[-1].ended_at is None
        assert rows[-1].running_ms == 10_800_000 and rows[-1].confirmed_ms == 0
        assert restarted is not service


@test
def test_duplicate_entry_replay_is_stable_and_void_repairs_without_rewriting_acceptance():
    with database() as (engine, service):
        entry = manual(service, "ENTRY", BASE)
        duplicate = manual(service, "ENTRY", BASE + timedelta(hours=1))
        manual(service, "EXIT", BASE + timedelta(hours=9))
        before = projection(engine)
        for _ in range(2):
            with immediate_session(engine) as db:
                service.rebuild_employee(db, 1)
            assert projection(engine) == before
        rows = sessions(engine)
        assert len(rows) == 1 and rows[0].entry_event_id == entry.id
        assert rows[0].status == "REVIEW" and rows[0].review_ms == 32_400_000
        assert rows[0].confirmed_ms == rows[0].running_ms == 0
        with Session(engine) as db:
            assert db.get(AttendanceEvent, duplicate.id).status == "ACCEPTED"
            assert "DUPLICATE_ENTRY" in db.get(AttendanceEvent, duplicate.id).projection_reasons
        decision(service, duplicate.id, "VOID")
        repaired = sessions(engine)[0]
        assert repaired.status == "CLOSED" and repaired.confirmed_ms == 32_400_000
        with Session(engine) as db:
            assert db.get(AttendanceEvent, duplicate.id).projection_reasons == []
            assert db.query(AttendanceDecision).filter_by(event_id=duplicate.id).count() == 2


@test
def test_missing_entry_exit_has_no_duration_and_late_manual_entry_repairs():
    with database() as (engine, service):
        exit_event = manual(service, "EXIT", BASE + timedelta(hours=3))
        assert sessions(engine) == []
        with Session(engine) as db:
            row = db.get(AttendanceEvent, exit_event.id)
            assert row.status == "ACCEPTED" and row.projection_reasons == ["MISSING_ENTRY"]
        manual(service, "ENTRY", BASE)
        assert sessions(engine)[0].confirmed_ms == 10_800_000
        with Session(engine) as db:
            assert db.get(AttendanceEvent, exit_event.id).projection_reasons == []


@test
def test_simultaneous_group_makes_no_transition_and_quarantines_following_session():
    with database() as (engine, service):
        first = manual(service, "ENTRY", BASE)
        second = manual(service, "EXIT", BASE)
        later = manual(service, "ENTRY", BASE + timedelta(hours=1))
        manual(service, "EXIT", BASE + timedelta(hours=3))
        rows = sessions(engine)
        assert len(rows) == 1 and rows[0].entry_event_id == later.id
        assert rows[0].review_ms == 7_200_000 and rows[0].confirmed_ms == 0
        with Session(engine) as db:
            for event_id in (first.id, second.id):
                assert db.get(AttendanceEvent, event_id).projection_reasons == ["SIMULTANEOUS_EVENTS"]
        decision(service, first.id, "VOID")
        decision(service, second.id, "VOID")
        assert sessions(engine)[0].confirmed_ms == 7_200_000


@test
def test_simultaneous_group_inside_preserves_original_open_interval():
    with database() as (engine, service):
        entry = manual(service, "ENTRY", BASE)
        manual(service, "EXIT", BASE + timedelta(hours=2))
        manual(service, "ENTRY", BASE + timedelta(hours=2))
        manual(service, "EXIT", BASE + timedelta(hours=4))
        rows = sessions(engine)
        assert len(rows) == 1 and rows[0].entry_event_id == entry.id
        assert rows[0].review_ms == 14_400_000 and "SIMULTANEOUS_EVENTS" in rows[0].reason_codes


@test
def test_unknown_lunch_quarantines_entire_nine_hours_then_assignment_rebuilds_eight():
    with database() as (engine, service):
        manual(service, "ENTRY", BASE)
        manual(service, "EXIT", BASE + timedelta(hours=9))
        unknown_exit = service.record_crossing(crossing(when=BASE + timedelta(hours=3), direction="EXIT", eligible=False))
        unknown_entry = service.record_crossing(crossing(when=BASE + timedelta(hours=4), ordinal=2, eligible=False))
        before = sessions(engine)
        assert len(before) == 1 and before[0].review_ms == 32_400_000
        assert before[0].confirmed_ms == 0
        for event_id in (unknown_exit, unknown_entry):
            decision(service, event_id, "ACCEPT", employee_id=1)
        after = sessions(engine)
        assert [row.confirmed_ms for row in after] == [10_800_000, 18_000_000]
        assert sum(row.review_ms for row in after) == 0


@test
def test_visitor_reconciliation_removes_global_blocker_without_employee_event():
    with database() as (engine, service):
        for employee in (1, 2):
            manual(service, "ENTRY", BASE, employee=employee)
            manual(service, "EXIT", BASE + timedelta(hours=9), employee=employee)
        unknown = service.record_crossing(crossing(when=BASE + timedelta(hours=3), eligible=False))
        assert [row.review_ms for row in sessions(engine)] == [32_400_000, 32_400_000]
        decision(service, unknown, "VISITOR")
        assert [row.confirmed_ms for row in sessions(engine)] == [32_400_000, 32_400_000]
        with Session(engine) as db:
            row = db.get(AttendanceEvent, unknown)
            assert row.employee_id is None and row.status == "VISITOR"


@test
def test_shared_trust_overlap_is_half_open_and_employee_scoped():
    row = WorkSession(entry_event_id="entry", exit_event_id="exit", employee_id=1,
                      employee_name="Name", started_at=BASE, ended_at=BASE + timedelta(hours=1),
                      status="CLOSED", reason_codes=[])
    at_end = TrustBlocker("at-end", None, row.ended_at, row.ended_at, "UNKNOWN_CROSSING")
    other = TrustBlocker("other", 2, BASE, AS_OF, "MISSING_ENTRY")
    valid = evaluate_session(row, [at_end, other], as_of=AS_OF)
    assert valid.confirmed_ms == 3_600_000 and valid.review_ms == 0
    assert valid.started_at == "2026-11-11T09:00:00+05:00"
    at_start = TrustBlocker("at-start", None, BASE, BASE, "UNKNOWN_CROSSING")
    review = evaluate_session(row, [at_start], as_of=AS_OF)
    assert review.review_ms == 3_600_000 and review.confirmed_ms == 0
    assert review.duration_ms == 3_600_000 and review.status == "REVIEW"
    finite_before = TrustBlocker("before", None, BASE - timedelta(hours=1), BASE, "FRAME_GAP")
    assert evaluate_session(row, [finite_before], as_of=AS_OF).confirmed_ms == 3_600_000


@test
def test_effective_endpoints_truncate_once_to_milliseconds_without_rounding_up():
    with database() as (engine, service):
        manual(service, "ENTRY", BASE.replace(microsecond=999))
        manual(service, "EXIT", (BASE + timedelta(seconds=1)).replace(microsecond=1001))
        row = sessions(engine)[0]
        assert row.duration_ms == row.confirmed_ms == 1001
        with Session(engine) as db:
            saved = db.query(WorkSession).one()
            assert saved.started_at.microsecond == 0 and saved.ended_at.microsecond == 1000


@test
def test_camera_idempotency_precedes_gallery_revalidation_and_conflict_never_replaces_raw():
    with database() as (engine, service):
        evidence = crossing(when=BASE.replace(microsecond=123456))
        identifier = service.record_crossing(evidence)
        assert UUID(identifier).version == 5
        with Session(engine) as db:
            row = db.get(AttendanceEvent, identifier)
            original = copy.deepcopy(row.raw)
            assert row.source_key == "camera:1:stream-a:generation-a:1:1"
            assert row.occurred_at.microsecond == 123000
            assert "123456" in str(row.raw) and row.raw["version"] == 1
            assert 1 in row.raw["candidate_employee_ids"]
            assert row.raw["would_auto_accept"] is True
            assert row.employee_id is None and row.status == "REVIEW"
            assert "COMMISSIONING_REQUIRED" in row.reason_codes
        with immediate_session(engine) as db:
            db.get(AttendanceRuntime, 1).gallery_revision += 1
            db.get(Employee, 1).active = False
        assert service.record_crossing(evidence) == identifier
        error_status(409, lambda: service.record_crossing(replace(evidence, direction="EXIT")))
        with Session(engine) as db:
            assert db.query(AttendanceEvent).count() == 1
            assert db.get(AttendanceEvent, identifier).raw == original


@test
def test_capture_eligibility_and_disqualifying_reasons_independently_override_identity():
    with database(commissioned=True) as (engine, service):
        # Real role metrics and committed coverage isolate the capture gates;
        # a commission row alone must not make a blocked test pass accidentally.
        mono, utc = 1000., BASE
        health = AttendanceHealth(now=lambda: utc, monotonic=lambda: mono)
        health.set_providers(CUDA_PROVIDERS)
        service.health = health
        for role, camera_id in (("ENTRY", 1), ("EXIT", 2)):
            health.configure_role(role, camera_id=camera_id, source_revision=2,
                                  enabled=True, online=True, included=True)
        frame = np.zeros((1, 1, 3), dtype=np.uint8)
        frame.setflags(write=False)
        for seq in range(1, 211):
            mono += .1
            utc += timedelta(seconds=.1)
            for role, camera_id in (("ENTRY", 1), ("EXIT", 2)):
                packet = FramePacket(f"stream-{camera_id}", 2, seq,
                                     utc - timedelta(seconds=.07), mono - .07, frame)
                health.record_frame(role, packet, tracking_generation=f"generation-{camera_id}",
                                    started_mono=mono - .05, completed_mono=mono)
            health.snapshot()
        monitor = AttendanceMonitor(service, now=lambda: utc, monotonic=lambda: mono)
        monitor.tick()
        assert health.snapshot()["technical_ready"] and health.snapshot()["automatic_ready"]
        assert health.pending_faults() == ()
        control = crossing(when=utc)
        assert control.auto_eligible and control.disqualifying_reasons == ()
        control_id = service.record_crossing(control)
        ineligible_id = service.record_crossing(replace(
            control, crossing_index=2, auto_eligible=False, disqualifying_reasons=()))
        disqualified_id = service.record_crossing(replace(
            control, crossing_index=3, auto_eligible=True,
            disqualifying_reasons=("ASSOCIATION_UNCERTAIN",)))
        with Session(engine) as db:
            accepted = db.get(AttendanceEvent, control_id)
            assert accepted.status == "ACCEPTED" and accepted.employee_id == 1
            assert accepted.reason_codes == []
            for identifier, only_reason in (
                    (ineligible_id, "CAPTURE_INELIGIBLE"),
                    (disqualified_id, "ASSOCIATION_UNCERTAIN")):
                row = db.get(AttendanceEvent, identifier)
                assert row.status == "REVIEW" and row.employee_id is None and row.direction == "ENTRY"
                assert row.reason_codes == [only_reason]
            assert db.query(WorkSession).one().entry_event_id == control_id


@test
def test_stale_gallery_never_binds_captured_candidate_to_reused_employee_id():
    with database() as (engine, service):
        evidence = crossing()
        with immediate_session(engine) as db:
            db.get(AttendanceRuntime, 1).gallery_revision = 8
            db.delete(db.get(Employee, 1))
            db.flush()
            db.add(Employee(id=1, name="Different person"))
        identifier = service.record_crossing(evidence)
        with Session(engine) as db:
            row = db.get(AttendanceEvent, identifier)
            assert row.status == "REVIEW" and row.employee_id is None
            assert "GALLERY_CHANGED" in row.reason_codes
            assert db.query(AttendanceEmployeeReference).filter_by(event_id=identifier).count() == 0
            assert "Employee 1" in str(row.raw) and "Different person" not in str(row.raw)


@test
def test_first_write_revalidates_source_calibration_employee_and_gallery():
    mutations = (
        (lambda db: setattr(db.get(AttendanceCamera, 1), "source_revision", 3), "SOURCE_CHANGED"),
        (lambda db: setattr(db.get(AttendanceCamera, 1), "revision", 4), "CALIBRATION_CHANGED"),
        (lambda db: setattr(db.get(Employee, 1), "active", False), "EMPLOYEE_INACTIVE"),
        (lambda db: setattr(db.get(AttendanceRuntime, 1), "gallery_revision", 8), "GALLERY_CHANGED"),
    )
    for mutate, reason in mutations:
        with database(commissioned=True) as (engine, service):
            evidence = crossing()
            with immediate_session(engine) as db:
                mutate(db)
            identifier = service.record_crossing(evidence)
            with Session(engine) as db:
                row = db.get(AttendanceEvent, identifier)
                assert row.status == "REVIEW" and row.employee_id is None
                assert reason in row.reason_codes, row.reason_codes


@test
def test_manual_retry_returns_original_result_after_newer_decision():
    with database() as (engine, service):
        request = uuid4()
        original = manual(service, "ENTRY", BASE, request_id=request)
        decision(service, original.id, "VOID")
        replay = manual(service, "ENTRY", BASE, request_id=request)
        assert replay.model_dump() == original.model_dump()
        error_status(409, lambda: manual(service, "EXIT", BASE, request_id=request))
        with Session(engine) as db:
            assert db.query(AttendanceRequest).count() == 2
            assert db.query(AttendanceDecision).count() == 2
            assert db.get(AttendanceEvent, original.id).status == "VOID"
            creation = db.query(AttendanceDecision).filter_by(revision=0).one()
            assert creation.actor == "Operator" and creation.reason == "Reconciled doorway"


@test
def test_event_retry_precedes_revision_check_and_changed_target_conflicts():
    with database() as (engine, service):
        first = service.record_crossing(crossing(eligible=False))
        second = service.record_crossing(crossing(ordinal=2, eligible=False))
        payload = EventDecisionInput(request_id=uuid4(), expected_revision=0, action="VISITOR",
                                     actor="Operator", reason="Client checked")
        result = service.decide_event(first, payload)
        decision(service, first, "VOID", revision=1)
        assert service.decide_event(first, payload).model_dump() == result.model_dump()
        error_status(409, lambda: service.decide_event(second, payload))
        error_status(409, lambda: decision(service, first, "VISITOR", revision=0))
        with Session(engine) as db:
            assert db.query(AttendanceDecision).filter_by(event_id=first).count() == 2


@test
def test_batch_visitors_all_or_none_and_replay_is_canonical_and_immutable():
    with database() as (engine, service):
        first = service.record_crossing(crossing(eligible=False))
        second = service.record_crossing(crossing(ordinal=2, eligible=False))
        payload = dict(request_id=uuid4(), items=[{"event_id": first, "expected_revision": 0},
                                                {"event_id": second, "expected_revision": 1}],
                       actor="Operator", reason="Clients checked")
        error_status(409, lambda: service.mark_visitors(VisitorBatchInput(**payload)))
        with Session(engine) as db:
            assert db.query(AttendanceDecision).count() == db.query(AttendanceRequest).count() == 0
            assert {row.status for row in db.query(AttendanceEvent)} == {"REVIEW"}
        payload["items"][1]["expected_revision"] = 0
        result = service.mark_visitors(VisitorBatchInput(**payload))
        decision(service, first, "VOID", revision=1)
        payload["items"].reverse()
        replay = service.mark_visitors(VisitorBatchInput(**payload))
        assert [row.model_dump() for row in replay] == [row.model_dump() for row in result]
        with Session(engine) as db:
            assert db.query(AttendanceRequest).count() == 2
            assert db.query(AttendanceDecision).count() == 3


@test
def test_reassignment_rebuilds_both_employees_and_preserves_all_identity_references():
    with database() as (engine, service):
        entry = manual(service, "ENTRY", BASE)
        manual(service, "EXIT", BASE + timedelta(hours=3))
        decision(service, entry.id, "ACCEPT", employee_id=2)
        with Session(engine) as db:
            rows = db.query(WorkSession).all()
            assert len(rows) == 1 and rows[0].employee_id == 2 and rows[0].ended_at is None
            references = db.query(AttendanceEmployeeReference).filter_by(event_id=entry.id).all()
            assert {row.employee_id for row in references} == {1, 2}
        decision(service, entry.id, "VOID", revision=1)
        with Session(engine) as db:
            assert db.query(WorkSession).count() == 0
            assert db.query(AttendanceEmployeeReference).filter_by(event_id=entry.id).count() == 2


@test
def test_concurrent_entries_and_duplicate_requests_use_independent_serialized_transactions():
    with database() as (engine, service):
        barrier = threading.Barrier(2)
        request = uuid4()

        def create(index, shared=False):
            barrier.wait(timeout=10)
            return manual(service, "ENTRY", BASE + timedelta(minutes=0 if shared else index),
                          request_id=request if shared else uuid4()).id

        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(create, index) for index in (0, 1)]
            ids = [future.result(timeout=15) for future in futures]
        assert len(set(ids)) == 2
        with Session(engine) as db:
            assert db.query(WorkSession).filter(WorkSession.ended_at.is_(None)).count() == 1
            assert db.query(WorkSession).one().status == "REVIEW"
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(create, index, True) for index in (0, 1)]
            ids = [future.result(timeout=15) for future in futures]
        assert ids[0] == ids[1]
        with Session(engine) as db:
            assert db.query(AttendanceEvent).count() == 3
            assert db.query(AttendanceDecision).count() == 3


@test
def test_failed_projection_rolls_back_event_request_decision_and_references_together():
    with database() as (engine, service):
        def fail_insert(_conn, _cursor, statement, parameters, _context, _many):
            if statement.lstrip().upper().startswith("INSERT INTO WORK_SESSIONS"):
                raise OperationalError(statement, parameters, RuntimeError("injected disk failure"))
        event.listen(engine, "before_cursor_execute", fail_insert)
        try:
            rejects((OperationalError, HTTPException), lambda: manual(service, "ENTRY", BASE))
        finally:
            event.remove(engine, "before_cursor_execute", fail_insert)
        with Session(engine) as db:
            for model in (AttendanceEvent, AttendanceRequest, AttendanceDecision,
                          AttendanceEmployeeReference, WorkSession):
                assert db.query(model).count() == 0, model


@test
def test_read_snapshot_keeps_blockers_and_sessions_consistent_across_concurrent_write():
    with database() as (engine, service):
        manual(service, "ENTRY", BASE)
        manual(service, "EXIT", BASE + timedelta(hours=9))
        unknown = service.record_crossing(crossing(eligible=False, when=BASE + timedelta(hours=3)))

        def read(db):
            before = db.query(AttendanceEvent).filter_by(status="REVIEW").count()
            with ThreadPoolExecutor(max_workers=1) as pool:
                pool.submit(decision, service, unknown, "VISITOR").result(timeout=10)
            blockers = load_trust_blockers(db, as_of=AS_OF)
            row = evaluate_session(db.query(WorkSession).one(), blockers, as_of=AS_OF)
            after = db.query(AttendanceEvent).filter_by(status="REVIEW").count()
            return before, after, row

        before, after, row = read_snapshot(engine, read)
        assert before == after == 1 and row.review_ms == 32_400_000
        assert sessions(engine)[0].confirmed_ms == 32_400_000


@test
def test_coverage_overlap_quarantines_without_closing_and_review_retry_survives_new_revision():
    with database() as (engine, service):
        manual(service, "ENTRY", BASE)
        service.update_coverage(False, ["FRAME_GAP"], BASE + timedelta(hours=1))
        assert sessions(engine, as_of=BASE + timedelta(hours=2))[0].review_ms == 7_200_000
        service.update_coverage(True, [], BASE + timedelta(hours=2))
        manual(service, "EXIT", BASE + timedelta(hours=3))
        assert sessions(engine)[0].review_ms == 10_800_000
        with Session(engine) as db:
            gap = db.query(AttendanceCoverageGap).one()
            gap_id, revision = gap.id, gap.revision
        payload = GapDecisionInput(request_id=uuid4(), expected_revision=revision,
                                   actor="Operator", reason="Missing crossings reconciled")
        result = service.review_gap(gap_id, payload)
        assert result.reviewed and sessions(engine)[0].confirmed_ms == 10_800_000
        with immediate_session(engine) as db:
            db.get(AttendanceCoverageGap, gap_id).revision += 1
        assert service.review_gap(gap_id, payload).model_dump() == result.model_dump()


@test
def test_open_gap_review_rejected_and_no_coverage_is_inferred_before_monitoring():
    with database() as (engine, service):
        with immediate_session(engine) as db:
            db.get(AttendanceRuntime, 1).monitoring_started_at = None
        service.update_coverage(False, ["AI_UNAVAILABLE"], BASE)
        with Session(engine) as db:
            assert db.query(AttendanceCoverageGap).count() == 0
        with immediate_session(engine) as db:
            db.get(AttendanceRuntime, 1).monitoring_started_at = BASE
        service.update_coverage(False, ["AI_UNAVAILABLE"], BASE - timedelta(days=1))
        with Session(engine) as db:
            gap = db.query(AttendanceCoverageGap).one()
            assert gap.started_at == BASE and gap.ended_at is None
            gap_id = gap.id
        error_status(409, lambda: service.review_gap(gap_id, GapDecisionInput(
            request_id=uuid4(), expected_revision=0, actor="Operator", reason="Cannot certify ongoing loss")))


@test
def test_evidence_purge_and_employee_deactivation_never_close_or_delete_history():
    with database() as (engine, service):
        identifier = service.record_crossing(crossing())
        decision(service, identifier, "ACCEPT", employee_id=1)
        with immediate_session(engine) as db:
            db.get(Employee, 1).active = False
            db.get(AttendanceRuntime, 1).gallery_revision += 1
        assert service.purge_evidence(AS_OF + timedelta(days=31)) == 1
        with Session(engine) as db:
            saved = db.get(AttendanceEvidence, identifier)
            assert saved.purged_at is not None and saved.before_jpeg is None
            assert saved.after_jpeg is None and saved.face_jpeg is None
            assert db.query(AttendanceEvent).count() == db.query(AttendanceDecision).count() == 1
            assert db.query(WorkSession).one().ended_at is None


@test
def test_reassignment_to_lower_employee_id_releases_old_stable_projection_key_first():
    with database() as (engine, service):
        entry = manual(service, "ENTRY", BASE, employee=2)
        decision(service, entry.id, "ACCEPT", employee_id=1)
        with Session(engine) as db:
            row = db.query(WorkSession).one()
            assert row.entry_event_id == entry.id and row.employee_id == 1
            assert {ref.employee_id for ref in db.query(AttendanceEmployeeReference).filter_by(
                event_id=entry.id)} == {1, 2}


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
    print(f"{len(TESTS) - failures}/{len(TESTS)} attendance service tests passed")
    raise SystemExit(bool(failures))
