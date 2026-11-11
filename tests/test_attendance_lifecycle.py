"""Failure/coverage/worker lifecycle contracts with explicit clocks and file SQLite.

Run: .venv/Scripts/python.exe tests/test_attendance_lifecycle.py
No sleeps, production DB, loaded neural models, RTSP subprocesses or network calls.
The real worker, recognition tracker, doorway, ledger, FIFO and monitor own state.
"""
import asyncio
import json
import logging
import sys
import traceback
from contextlib import contextmanager
from dataclasses import FrozenInstanceError, replace
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from fastapi import FastAPI
from sqlalchemy import event
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session, sessionmaker

from app.ai.doorway import DoorwayDetector
from app.ai.face_recognition import EmbeddingStore
from app.ai.recognition_service import RecognitionService
from app.ai.worker import AIWorker, DetectionState
from app.attendance_types import GapDecisionInput
from app.config import settings
from app.database.database import get_db, immediate_session
from app.database.models import (
    AttendanceCamera, AttendanceCoverageGap, AttendanceEvent, AttendanceRuntime,
    Camera, PresenceSession, WorkSession,
)
from app.rtsp.reader import CameraStatus, FrameBuffer, FramePacket
from app.services.attendance_health import AttendanceHealth, CoverageFault
from app.services.attendance_monitor import AttendanceMonitor, AttendanceWriteQueue
from test_attendance_service import (
    AS_OF, BASE, CUDA_PROVIDERS, crossing, database, manual, rejects,
)

TESTS = []
FRAME = np.zeros((480, 640, 3), dtype=np.uint8)
FRAME.setflags(write=False)


def test(fn):
    TESTS.append(fn)
    return fn


@contextmanager
def expected_error_logs(logger_name, exception_text):
    """Capture scoped injected failures without pinning production log wording."""
    with TestCase().assertLogs(logger_name, level=logging.WARNING) as captured:
        yield
    for record in captured.records:
        assert record.exc_info is not None, captured.output
        error = record.exc_info[1]
        cause = error.orig if isinstance(error, OperationalError) else error
        assert isinstance(cause, RuntimeError) and str(cause) == exception_text, captured.output


class Clock:
    def __init__(self):
        self.mono = 1000.
        self.utc = BASE

    def advance(self, seconds, *, utc_seconds=None):
        self.mono += seconds
        self.utc += timedelta(seconds=seconds if utc_seconds is None else utc_seconds)

    def monotonic(self):
        return self.mono

    def utcnow(self):
        return self.utc


def health_for(service, clock):
    health = AttendanceHealth(now=clock.utcnow, monotonic=clock.monotonic)
    health.set_providers(CUDA_PROVIDERS)
    for role, camera_id in (("ENTRY", 1), ("EXIT", 2)):
        health.configure_role(role, camera_id=camera_id, source_revision=2,
                              enabled=True, online=True, included=True)
    service.health = health
    return health


def packet(clock, camera_id=1, seq=1, *, age=0., stream=None):
    return FramePacket(stream_id=stream or f"stream-{camera_id}", source_revision=2, seq=seq,
                       received_at=clock.utc - timedelta(seconds=age),
                       received_mono=clock.mono - age, frame=FRAME)


def process_metrics(health, clock, seq, *, age=.02, completion=.05):
    # The injected clock is observation/completion time, never an instant before
    # a supposedly successful frame finished. Preserve receipt→start and
    # receipt→completion ages while backdating that frame's receipt and start.
    for role, camera_id in (("ENTRY", 1), ("EXIT", 2)):
        health.record_frame(role, packet(clock, camera_id, seq, age=age + completion),
                            tracking_generation=f"generation-{camera_id}",
                            started_mono=clock.mono - completion,
                            completed_mono=clock.mono)


def warm(health, clock, *, seconds=21, fps=10, age=.02, completion=.05):
    for seq in range(1, seconds * fps + 1):
        clock.advance(1 / fps)
        process_metrics(health, clock, seq, age=age, completion=completion)
        health.snapshot()


def fault(clock, reason="FRAME_GAP", *, start=0., end=.2):
    return CoverageFault(id=str(uuid4()), started_at=clock.utc + timedelta(seconds=start),
                         started_mono=clock.mono + start,
                         ended_at=None if end is None else clock.utc + timedelta(seconds=end),
                         reason_codes=(reason,))


@test
def test_subtick_fault_survives_recovery_until_successful_database_commit():
    with database() as (engine, service):
        clock = Clock()
        health = health_for(service, clock)
        monitor = AttendanceMonitor(service, now=clock.utcnow, monotonic=clock.monotonic)
        notification = fault(clock)
        health.publish_fault(notification)
        clock.advance(1)
        assert health.pending_faults() == (notification,)
        monitor.tick()
        assert health.pending_faults() == ()
        with Session(engine) as db:
            gaps = db.query(AttendanceCoverageGap).all()
            saved = [gap for gap in gaps if notification.id in gap.source_fault_ids]
            assert len(saved) == 1 and saved[0].started_at <= BASE
            assert "FRAME_GAP" in saved[0].reason_codes and not saved[0].reviewed
        # Replaying a notification cannot create duplicate gap coverage.
        health.publish_fault(notification)
        monitor.tick()
        with Session(engine) as db:
            assert sum(notification.id in row.source_fault_ids for row in db.query(AttendanceCoverageGap)) == 1


@test
def test_fault_commit_failure_latches_notification_and_recovery_records_closed_span():
    with database() as (engine, service):
        clock = Clock()
        health = health_for(service, clock)
        monitor = AttendanceMonitor(service, now=clock.utcnow, monotonic=clock.monotonic)
        opened = fault(clock, end=None)
        health.publish_fault(opened)

        def fail(_conn, _cursor, statement, parameters, _context, _many):
            if "INSERT INTO attendance_coverage_gaps" in statement:
                raise OperationalError(statement, parameters, RuntimeError("disk unavailable"))

        event.listen(engine, "before_cursor_execute", fail)
        try:
            with expected_error_logs("app.attendance.monitor", "disk unavailable"):
                monitor.tick()
            assert opened in health.pending_faults()
            assert not health.snapshot()["technical_ready"]
        finally:
            event.remove(engine, "before_cursor_execute", fail)
        monitor.tick()
        assert opened not in health.pending_faults()
        closed = replace(opened, id=str(uuid4()), ended_at=BASE + timedelta(seconds=.2))
        health.publish_fault(closed)
        monitor.tick()
        with Session(engine) as db:
            gaps = db.query(AttendanceCoverageGap).all()
            assert any(opened.id in row.source_fault_ids for row in gaps)
            assert any(closed.id in row.source_fault_ids for row in gaps)
            assert min(row.started_at for row in gaps) <= BASE


@test
def test_fault_cap_conservatively_encloses_all_spans_instead_of_dropping_oldest():
    clock = Clock()
    health = AttendanceHealth(now=clock.utcnow, monotonic=clock.monotonic)
    for index in range(101):
        health.publish_fault(fault(clock, reason=f"FAULT_{index}", start=index * 2., end=index * 2. + .1))
    pending = health.pending_faults()
    assert 1 <= len(pending) <= 100
    assert min(item.started_at for item in pending) == BASE
    assert max(item.ended_at for item in pending) == BASE + timedelta(seconds=200.1)
    reasons = {reason for item in pending for reason in item.reason_codes}
    assert reasons == {f"FAULT_{index}" for index in range(101)}
    rejects(FrozenInstanceError, lambda: setattr(pending[0], "ended_at", AS_OF))


@test
def test_simultaneous_faults_merge_but_never_extend_reviewed_coverage():
    with database() as (engine, service):
        clock = Clock()
        health = health_for(service, clock)
        with immediate_session(engine) as db:
            reviewed = AttendanceCoverageGap(started_at=BASE - timedelta(seconds=2), ended_at=BASE,
                                              reason_codes=["FRAME_GAP"], reviewed=True,
                                              source_fault_ids=["already-reviewed"])
            db.add(reviewed)
            db.flush()
            old_id = reviewed.id
        first = fault(clock, start=-.1, end=.2)
        second = fault(clock, "PERSISTENCE_ERROR", start=.1, end=.5)
        health.publish_fault(first)
        health.publish_fault(second)
        monitor = AttendanceMonitor(service, now=clock.utcnow, monotonic=clock.monotonic)
        monitor.tick()
        with Session(engine) as db:
            old = db.get(AttendanceCoverageGap, old_id)
            assert old.reviewed and old.ended_at == BASE and old.source_fault_ids == ["already-reviewed"]
            current = db.query(AttendanceCoverageGap).filter_by(reviewed=False).all()
            assert len(current) == 1
            assert {"FRAME_GAP", "PERSISTENCE_ERROR"} <= set(current[0].reason_codes)
            assert {first.id, second.id} <= set(current[0].source_fault_ids)


@test
def test_fifo_head_failure_preserves_order_and_drain_never_acknowledges_health_faults():
    with database() as (engine, service):
        clock = Clock()
        health = health_for(service, clock)
        queue = AttendanceWriteQueue(service, health)
        one, two = crossing(eligible=False), crossing(ordinal=2, eligible=False)
        independent = fault(clock)
        health.publish_fault(independent)

        def fail(_conn, _cursor, statement, parameters, _context, _many):
            if "INSERT INTO attendance_events" in statement:
                raise OperationalError(statement, parameters, RuntimeError("disk unavailable"))

        event.listen(engine, "before_cursor_execute", fail)
        try:
            with expected_error_logs("app.attendance.monitor", "disk unavailable"):
                assert queue.submit(one) is None
                assert queue.submit(two) is None
                assert queue.pending() == (one, two)
                assert queue.retry_one() is False
                assert queue.pending() == (one, two)
        finally:
            event.remove(engine, "before_cursor_execute", fail)
        assert queue.retry_one() is True and queue.pending() == (two,)
        with Session(engine) as db:
            assert [row.crossing_index for row in db.query(AttendanceEvent)] == [1]
        assert queue.retry_one() is True and queue.pending() == ()
        assert any("FRAME_GAP" in item.reason_codes and item.started_at <= independent.started_at
                   for item in health.pending_faults())
        with Session(engine) as db:
            assert db.query(AttendanceEvent).count() == 2


@test
def test_fifo_full_drops_newest_only_with_latched_evidence_lost():
    with database() as (engine, service):
        clock = Clock()
        health = health_for(service, clock)
        queue = AttendanceWriteQueue(service, health)
        original = crossing(eligible=False)

        def fail(_conn, _cursor, statement, parameters, _context, _many):
            if "INSERT INTO attendance_events" in statement:
                raise OperationalError(statement, parameters, RuntimeError("disk unavailable"))

        event.listen(engine, "before_cursor_execute", fail)
        try:
            with expected_error_logs("app.attendance.monitor", "disk unavailable"):
                for index in range(101):
                    queue.submit(replace(original, crossing_index=index + 1,
                                         occurred_at=BASE + timedelta(milliseconds=index)))
        finally:
            event.remove(engine, "before_cursor_execute", fail)
        assert [item.crossing_index for item in queue.pending()] == list(range(1, 101))
        lost = [item for item in health.pending_faults() if "EVIDENCE_LOST" in item.reason_codes]
        assert lost and any(item.started_at <= BASE + timedelta(milliseconds=100)
                            and (item.ended_at is None or item.ended_at >= BASE + timedelta(milliseconds=100))
                            for item in lost)
        with Session(engine) as db:
            assert db.query(AttendanceEvent).count() == 0


@test
def test_metrics_use_trailing_window_not_warmup_extrapolation_and_include_completion_age():
    with database() as (_engine, service):
        clock = Clock()
        health = health_for(service, clock)
        process_metrics(health, clock, 1, age=.2, completion=.4)
        entry = health.snapshot()["roles"]["ENTRY"]
        assert entry["camera_id"] == 1 and entry["stream_id"] == "stream-1"
        assert entry["source_revision"] == 2 and entry["tracking_generation"] == "generation-1"
        assert entry["processed_fps"] == .1
        assert abs(entry["processing_age_p95_seconds"] - .2) < 1e-9
        assert abs(entry["completion_age_p95_seconds"] - .6) < 1e-9
        clock.advance(10.1)
        assert health.snapshot()["roles"]["ENTRY"]["processed_fps"] == 0
        assert not health.snapshot()["technical_ready"]


@test
def test_performance_fault_requires_ten_healthy_seconds_to_recover():
    with database() as (_engine, service):
        clock = Clock()
        health = health_for(service, clock)
        warm(health, clock)
        assert health.snapshot()["technical_ready"]
        warm(health, clock, seconds=12, completion=1.1)
        assert "PERFORMANCE_DEGRADED" in health.snapshot()["roles"]["ENTRY"]["reasons"]
        assert not health.snapshot()["technical_ready"]
        # P95 samples have a trailing-60-second lifetime; an old slow sample
        # cannot remain forever, nor can one fast frame clear sustained failure.
        warm(health, clock, seconds=60)
        warm(health, clock, seconds=10)
        assert health.snapshot()["technical_ready"]
        assert any("PERFORMANCE_DEGRADED" in item.reason_codes for item in health.pending_faults())


@test
def test_missing_model_or_excluded_role_is_explicit_not_automatic_readiness():
    with database() as (_engine, service):
        clock = Clock()
        health = health_for(service, clock)
        warm(health, clock)
        providers = dict(CUDA_PROVIDERS, live_face_recognition=[])
        health.set_providers(providers)
        snapshot = health.snapshot()
        assert snapshot["providers"]["live_face_recognition"] == []
        assert not snapshot["technical_ready"] and "AI_UNAVAILABLE" in snapshot["reasons"]
        health.set_providers(CUDA_PROVIDERS)
        health.configure_role("EXIT", camera_id=2, source_revision=2,
                              enabled=True, online=True, included=False)
        assert not health.snapshot()["technical_ready"]


@test
def test_technical_readiness_without_commission_keeps_trial_capture_review_only():
    with database() as (engine, service):
        clock = Clock()
        health = health_for(service, clock)
        warm(health, clock)
        monitor = AttendanceMonitor(service, now=clock.utcnow, monotonic=clock.monotonic)
        monitor.tick()
        snapshot = health.snapshot()
        assert snapshot["technical_ready"] and not snapshot["automatic_ready"]
        identifier = service.record_crossing(crossing(when=clock.utc))
        with Session(engine) as db:
            row = db.get(AttendanceEvent, identifier)
            assert row.status == "REVIEW" and row.raw["would_auto_accept"] is True
            assert "COMMISSIONING_REQUIRED" in row.reason_codes
            assert any("COMMISSIONING_REQUIRED" in gap.reason_codes for gap in db.query(AttendanceCoverageGap))


@test
def test_valid_commission_and_live_coverage_admit_capture_and_committed_retry_stays_accepted():
    with database(commissioned=True) as (engine, service):
        clock = Clock()
        health = health_for(service, clock)
        warm(health, clock)
        monitor = AttendanceMonitor(service, now=clock.utcnow, monotonic=clock.monotonic)
        monitor.tick()
        evidence = crossing(when=clock.utc)
        identifier = service.record_crossing(evidence)
        with Session(engine) as db:
            assert db.get(AttendanceEvent, identifier).status == "ACCEPTED"
            assert db.query(WorkSession).one().ended_at is None
        health.set_providers({key: ["CPUExecutionProvider"] for key in CUDA_PROVIDERS})
        monitor.tick()
        assert service.record_crossing(evidence) == identifier
        with Session(engine) as db:
            assert db.get(AttendanceEvent, identifier).status == "ACCEPTED"
            assert db.get(AttendanceRuntime, 1).commission_id is None


@test
def test_commissioning_only_notifications_preserve_the_technical_healthy_streak():
    with database() as (_engine, service):
        clock = Clock()
        health = health_for(service, clock)
        warm(health, clock)
        monitor = AttendanceMonitor(service, now=clock.utcnow, monotonic=clock.monotonic)
        monitor.tick()
        before = health.snapshot()
        assert before["technical_ready"] and before["technically_healthy_seconds"] >= 10
        assert not before["automatic_ready"]
        approval = fault(clock, "COMMISSIONING_REQUIRED", end=None)
        health.publish_fault(approval)
        monitor.tick()
        after_ack = health.snapshot()
        assert after_ack["technical_ready"]
        assert after_ack["technically_healthy_seconds"] == before["technically_healthy_seconds"]
        assert health.pending_faults() == ()
        clock.advance(.2)
        health.publish_fault(replace(approval, id=str(uuid4()), ended_at=clock.utc))
        process_metrics(health, clock, 999)
        monitor.tick()
        recovered = health.snapshot()
        assert recovered["technical_ready"] and not recovered["automatic_ready"]
        assert abs(recovered["technically_healthy_seconds"] -
                   (before["technically_healthy_seconds"] + .2)) < 1e-9
        assert health.pending_faults() == ()


@test
def test_reconnect_resets_warmed_technical_streak_even_after_fault_persistence():
    with database(commissioned=True) as (engine, service):
        clock = Clock()
        health = health_for(service, clock)
        warm(health, clock)
        monitor = AttendanceMonitor(service, now=clock.utcnow, monotonic=clock.monotonic)
        monitor.tick()
        assert health.snapshot()["technically_healthy_seconds"] >= 10
        assert health.pending_faults() == ()
        clock.advance(.1)
        health.record_frame("ENTRY", packet(clock, age=.07, stream="reconnected"),
                            tracking_generation="new-generation",
                            started_mono=clock.mono - .05, completed_mono=clock.mono)
        reset_faults = [item for item in health.pending_faults() if "STREAM_RESET" in item.reason_codes]
        assert len(reset_faults) == 1
        # The new packet is explicit recovery, not ten seconds of healthy
        # operation on the new stream. Polling must not retain the old streak.
        assert health.snapshot()["technical_ready"]
        assert 0 <= health.snapshot()["technically_healthy_seconds"] <= .1
        monitor.tick()
        assert health.pending_faults() == ()
        assert 0 <= health.snapshot()["technically_healthy_seconds"] <= .1
        with Session(engine) as db:
            assert any(reset_faults[0].id in row.source_fault_ids
                       for row in db.query(AttendanceCoverageGap))
        for seq in range(2, 104):
            clock.advance(.1)
            for role, camera_id, stream in (("ENTRY", 1, "reconnected"), ("EXIT", 2, "stream-2")):
                health.record_frame(role, packet(clock, camera_id, seq, age=.07, stream=stream),
                                    tracking_generation="new-generation" if camera_id == 1 else "generation-2",
                                    started_mono=clock.mono - .05, completed_mono=clock.mono)
            health.snapshot()
        assert health.snapshot()["technically_healthy_seconds"] >= 10


@test
def test_published_subtick_fault_breaks_streak_and_acknowledgement_is_not_recovery():
    with database(commissioned=True) as (engine, service):
        clock = Clock()
        health = health_for(service, clock)
        warm(health, clock)
        monitor = AttendanceMonitor(service, now=clock.utcnow, monotonic=clock.monotonic)
        monitor.tick()
        assert health.snapshot()["technically_healthy_seconds"] >= 10
        opened = fault(clock, end=None)
        health.publish_fault(opened)
        assert not health.snapshot()["technical_ready"]
        assert health.snapshot()["technically_healthy_seconds"] == 0
        monitor.tick()
        assert health.pending_faults() == ()
        # Durable acknowledgement only clears the notification backlog; this
        # externally published technical fault is still unsafe until recovery.
        assert not health.snapshot()["technical_ready"]
        assert health.snapshot()["technically_healthy_seconds"] == 0
        clock.advance(.2)
        closed = replace(opened, id=str(uuid4()), ended_at=clock.utc)
        health.publish_fault(closed)
        process_metrics(health, clock, 999)
        monitor.tick()
        assert health.pending_faults() == ()
        assert health.snapshot()["technical_ready"]
        assert 0 <= health.snapshot()["technically_healthy_seconds"] <= .2
        warm(health, clock, seconds=11)
        assert health.snapshot()["technically_healthy_seconds"] >= 10
        # Now hide the entire start/recovery interval between snapshots/ticks:
        # publication itself must invalidate the streak, not polling cadence.
        subtick = fault(clock, end=None)
        health.publish_fault(subtick)
        clock.advance(.2)
        recovered = replace(subtick, id=str(uuid4()), ended_at=clock.utc)
        health.publish_fault(recovered)
        process_metrics(health, clock, 1000)
        monitor.tick()
        assert health.pending_faults() == ()
        assert health.snapshot()["technical_ready"]
        assert 0 <= health.snapshot()["technically_healthy_seconds"] <= .2
        with Session(engine) as db:
            persisted_ids = {identifier for row in db.query(AttendanceCoverageGap)
                             for identifier in row.source_fault_ids}
            assert {opened.id, closed.id, subtick.id, recovered.id} <= persisted_ids


@test
def test_threshold_fingerprint_change_revokes_but_gallery_and_same_source_reconnect_do_not():
    with database(commissioned=True) as (engine, service):
        clock = Clock()
        health = health_for(service, clock)
        warm(health, clock)
        monitor = AttendanceMonitor(service, now=clock.utcnow, monotonic=clock.monotonic)
        monitor.tick()
        with Session(engine) as db:
            commission_id = db.get(AttendanceRuntime, 1).commission_id
            assert commission_id is not None
        with immediate_session(engine) as db:
            db.get(AttendanceRuntime, 1).gallery_revision += 1
        health.record_frame("ENTRY", packet(clock, stream="reconnected"),
                            tracking_generation="new-generation", started_mono=clock.mono,
                            completed_mono=clock.mono + .05)
        monitor.tick()
        with Session(engine) as db:
            assert db.get(AttendanceRuntime, 1).commission_id == commission_id
        with patch.object(settings, "attendance_face_margin", .09):
            monitor.tick()
            with Session(engine) as db:
                assert db.get(AttendanceRuntime, 1).commission_id is None


@test
def test_restart_gap_begins_at_durable_stop_or_heartbeat_and_never_closes_work():
    for clean in (True, False):
        with database() as (engine, service):
            manual(service, "ENTRY", BASE - timedelta(hours=1))
            with immediate_session(engine) as db:
                runtime = db.get(AttendanceRuntime, 1)
                runtime.heartbeat_at = BASE - timedelta(seconds=30)
                runtime.stopped_at = BASE - timedelta(seconds=10) if clean else None
            clock = Clock()
            health_for(service, clock)
            monitor = AttendanceMonitor(service, now=clock.utcnow, monotonic=clock.monotonic)
            monitor.startup()
            with Session(engine) as db:
                gaps = db.query(AttendanceCoverageGap).all()
                assert min(row.started_at for row in gaps) == BASE - timedelta(seconds=10 if clean else 30)
                assert db.query(WorkSession).one().ended_at is None
            clock.advance(5)
            monitor.tick()
            with Session(engine) as db:
                assert db.get(AttendanceRuntime, 1).heartbeat_at == clock.utc


@test
def test_backward_clock_gap_is_nonnegative_and_blocks_admission_until_review():
    with database(commissioned=True) as (engine, service):
        clock = Clock()
        health = health_for(service, clock)
        warm(health, clock)
        monitor = AttendanceMonitor(service, now=clock.utcnow, monotonic=clock.monotonic)
        monitor.tick()
        assert health.snapshot()["automatic_ready"]
        previous = clock.utc
        assert health.observe_clock(clock.utc, clock.mono)
        clock.advance(.1, utc_seconds=-120)
        assert health.observe_clock(clock.utc, clock.mono) is False
        monitor.tick()
        # Explicitly observe a recovered UTC/monotonic pair. Otherwise the
        # in-memory ongoing fault, not the persisted gap, would block admission.
        clock.advance(.1)
        assert health.observe_clock(clock.utc, clock.mono)
        clock.advance(121)
        assert health.observe_clock(clock.utc, clock.mono)
        warm(health, clock)
        assert health.observe_clock(clock.utc, clock.mono)
        monitor.tick()
        snapshot = health.snapshot()
        assert snapshot["technical_ready"] and snapshot["technically_healthy_seconds"] >= 10
        assert health.pending_faults() == () and snapshot["pending_write_count"] == 0
        assert snapshot["reasons"] == ["CLOCK_DISCONTINUITY"]
        assert not snapshot["automatic_ready"]
        with Session(engine) as db:
            assert db.query(AttendanceCoverageGap).filter_by(ended_at=None).count() == 0
            clock_gaps = [row for row in db.query(AttendanceCoverageGap) if "CLOCK_DISCONTINUITY" in row.reason_codes]
            assert clock_gaps and min(row.started_at for row in clock_gaps) <= previous - timedelta(seconds=120)
            for row in clock_gaps:
                assert row.ended_at is not None and row.ended_at >= max(row.started_at, previous)
                assert not row.reviewed
            revisions = [(row.id, row.revision) for row in clock_gaps]
        evidence = crossing(when=clock.utc)
        assert evidence.auto_eligible and evidence.disqualifying_reasons == ()
        identifier = service.record_crossing(evidence)
        with Session(engine) as db:
            blocked = db.get(AttendanceEvent, identifier)
            assert blocked.status == "REVIEW" and blocked.employee_id is None
            assert blocked.reason_codes == ["CLOCK_DISCONTINUITY"]
        for gap_id, revision in revisions:
            service.review_gap(gap_id, GapDecisionInput(
                request_id=uuid4(), expected_revision=revision, actor="Operator",
                reason="Host clock and all crossings in the closed unsafe interval reconciled"))
        monitor.tick()
        assert health.snapshot()["technical_ready"] and health.snapshot()["automatic_ready"]
        assert health.pending_faults() == ()
        accepted_id = service.record_crossing(replace(evidence, crossing_index=2))
        with Session(engine) as db:
            accepted = db.get(AttendanceEvent, accepted_id)
            assert accepted.status == "ACCEPTED" and accepted.employee_id == 1
            assert accepted.reason_codes == []
            assert db.query(WorkSession).one().entry_event_id == accepted_id


class PersonObservations:
    """Only model inference is supplied; tracking/ownership/crossing stay real."""
    def __init__(self):
        self.y = .25

    def detect(self, frame):
        if self.y is None:
            return []
        return [(np.array([220., self.y * 480 - 200., 420., self.y * 480]), .99)]


class NoFaces:
    def detect(self, image, max_num=0):
        return []


class BufferedReader:
    def __init__(self):
        self.buffer = FrameBuffer()
        self.source_revision = 2
        self.online = True

    def is_alive(self):
        return True

    def status(self):
        return CameraStatus.ONLINE if self.online else CameraStatus.OFFLINE


class Readers:
    def __init__(self):
        self.readers = {1: BufferedReader(), 2: BufferedReader()}

    def reader_ids(self):
        return list(self.readers)

    def get(self, camera_id):
        return self.readers.get(camera_id)


def worker_for(engine, attendance, clock):
    health = health_for(attendance, clock)
    store = EmbeddingStore()
    with Session(engine) as db:
        store.refresh(db)
    observations = PersonObservations()
    recognition = RecognitionService(observations, NoFaces(), store, min_hits=1)
    manager = Readers()
    queue = AttendanceWriteQueue(attendance, health)
    worker = AIWorker(manager, recognition, attendance, DetectionState(),
                      doorway=DoorwayDetector(), health=health, write_queue=queue)
    return worker, manager, observations, queue


def step(worker, manager, observations, clock, y=None, *, publish=True, stream="source-a"):
    observations.y = y
    if publish:
        manager.readers[1].buffer.put(FRAME, stream_id=stream, source_revision=2,
                                     received_at=clock.utc, received_mono=clock.mono)
    with patch("app.ai.worker.time.monotonic", side_effect=clock.monotonic), \
            patch("app.ai.worker.utcnow", side_effect=clock.utcnow), \
            patch("app.ai.doorway.time.monotonic", side_effect=clock.monotonic):
        worker.step()
    clock.advance(.25)


@test
def test_real_worker_persists_doorway_crossing_without_any_recognition_or_legacy_write():
    with database() as (engine, service):
        clock = Clock()
        worker, manager, observations, _queue = worker_for(engine, service, clock)
        for y in (.25, .3, .45, .55, .7):
            step(worker, manager, observations, clock, y)
        clock.advance(2.1)
        step(worker, manager, observations, clock, publish=False)
        with Session(engine) as db:
            rows = db.query(AttendanceEvent).all()
            assert len(rows) == 1 and rows[0].status == "REVIEW" and rows[0].direction == "ENTRY"
            assert db.query(PresenceSession).count() == db.query(WorkSession).count() == 0


@test
def test_worker_disable_flushes_partial_before_reset_and_removes_stale_detections():
    with database() as (engine, service):
        clock = Clock()
        worker, manager, observations, _queue = worker_for(engine, service, clock)
        for y in (.25, .3, .49):
            step(worker, manager, observations, clock, y)
        assert worker.state.camera(1) is not None
        with immediate_session(engine) as db:
            db.get(AttendanceCamera, 1).enabled = False
        step(worker, manager, observations, clock, publish=False)
        assert worker.state.camera(1) is None
        with Session(engine) as db:
            row = db.query(AttendanceEvent).one()
            assert row.direction is None and row.status == "REVIEW"
            assert "INCOMPLETE_CROSSING" in row.reason_codes
        step(worker, manager, observations, clock, publish=False)
        with Session(engine) as db:
            assert db.query(AttendanceEvent).count() == 1


@test
def test_same_stream_calibration_reset_changes_source_key_without_reusing_track_identity():
    with database() as (engine, service):
        clock = Clock()
        worker, manager, observations, _queue = worker_for(engine, service, clock)
        for y in (.25, .3, .49):
            step(worker, manager, observations, clock, y)
        with immediate_session(engine) as db:
            db.get(AttendanceCamera, 1).revision += 1
        step(worker, manager, observations, clock, publish=False)
        for y in (.25, .3, .49):
            step(worker, manager, observations, clock, y)
        worker.flush("SHUTDOWN")
        with Session(engine) as db:
            rows = db.query(AttendanceEvent).all()
            assert len(rows) == 2
            assert len({row.source_key for row in rows}) == 2
            assert len({row.tracking_generation for row in rows}) == 2
            assert {row.stream_id for row in rows} == {"source-a"}


@test
def test_orderly_shutdown_flushes_partial_and_persists_stop_before_monitor_stops():
    with database() as (engine, service):
        clock = Clock()
        worker, manager, observations, queue = worker_for(engine, service, clock)
        for y in (.25, .3, .49):
            step(worker, manager, observations, clock, y)
        worker.flush("SHUTDOWN")
        monitor = AttendanceMonitor(service, write_queue=queue, now=clock.utcnow, monotonic=clock.monotonic)
        monitor.shutdown()
        with Session(engine) as db:
            assert db.query(AttendanceEvent).one().status == "REVIEW"
            assert db.get(AttendanceRuntime, 1).stopped_at == clock.utc
            assert db.query(WorkSession).count() == 0
        assert queue.pending() == ()


async def request(app, path):
    messages = []
    sent = False

    async def receive():
        nonlocal sent
        if not sent:
            sent = True
            return {"type": "http.request", "body": b"", "more_body": False}
        return {"type": "http.disconnect"}

    async def send(message):
        messages.append(message)

    route, _, query = path.partition("?")
    await app({"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
               "scheme": "http", "method": "GET", "path": route, "raw_path": route.encode(),
               "query_string": query.encode(), "headers": [], "server": ("test", 80),
               "client": ("test", 1), "root_path": ""}, receive, send)
    status = next(item["status"] for item in messages if item["type"] == "http.response.start")
    body = b"".join(item.get("body", b"") for item in messages if item["type"] == "http.response.body")
    return status, json.loads(body) if body else None


@test
def test_legacy_archive_filters_counts_offsets_and_never_mutates_unclosed_history():
    from app.api.presence import router
    with database() as (engine, service):
        with immediate_session(engine) as db:
            db.add_all([
                PresenceSession(employee_id=1, camera_id=1, started_at=BASE,
                                last_seen_at=BASE + timedelta(seconds=5), ended_at=None, confidence=.9),
                PresenceSession(employee_id=1, camera_id=1, started_at=BASE - timedelta(days=2),
                                last_seen_at=BASE - timedelta(days=2), ended_at=None, confidence=.8),
            ])
        app = FastAPI()
        app.include_router(router)
        app.state.attendance = service

        def isolated_db():
            with Session(engine) as db:
                yield db

        app.dependency_overrides[get_db] = isolated_db
        status, body = asyncio.run(request(app, "/api/presence/legacy?employee_id=1&from_date=2026-11-11&to_date=2026-11-11&limit=1&offset=0"))
        assert status == 200 and body["trusted_for_hours"] is False
        assert body["total"] == 1 and body["limit"] == 1 and body["offset"] == 0
        row = body["items"][0]
        assert row["started_at"] == "2026-11-11T09:00:00+05:00" and row["ended_at"] is None
        for old_path in ("/api/presence", "/api/presence/active", "/api/presence/employee/1"):
            assert asyncio.run(request(app, old_path))[0] == 404
        assert asyncio.run(request(app, "/api/presence/legacy?from_date=invalid"))[0] == 422
        assert asyncio.run(request(app, "/api/presence/legacy?employee_id=999"))[0] == 404
        with Session(engine) as db:
            assert db.query(PresenceSession).filter(PresenceSession.ended_at.is_(None)).count() == 2
            assert db.query(WorkSession).count() == 0


@test
def test_lifespan_restores_enabled_frozen_sources_and_keeps_manual_access_when_ai_disabled():
    import main
    from app.rtsp.reader import RTSPReader
    with database() as (engine, _service):
        with immediate_session(engine) as db:
            db.get(Camera, 2).enabled = False
        factory = sessionmaker(bind=engine)
        app = FastAPI()

        async def exercise():
            async with main.lifespan(app):
                assert app.state.attendance is not None
                restored = app.state.manager.get(1)
                assert restored is not None and restored.source_revision == 2
                assert app.state.manager.get(2) is None
                # A real manual write must remain available without any models.
                app.state.attendance._now = lambda: AS_OF
                saved = manual(app.state.attendance, "ENTRY", BASE)
                with Session(engine) as db:
                    assert db.get(AttendanceEvent, saved.id).status == "ACCEPTED"
            with Session(engine) as db:
                assert db.get(AttendanceRuntime, 1).stopped_at is not None
                assert db.query(WorkSession).one().ended_at is None

        # Replace only lifecycle externals: initialization points to the already
        # initialized fixture, camera process/thread launch and real model loading.
        with patch.object(main, "init_db"), patch.object(main, "SessionLocal", factory), \
                patch.object(main, "engine", engine), \
                patch.object(main, "build_enroll_engine", return_value=None), \
                patch.object(settings, "ai_enabled", False), \
                patch.object(settings, "telegram_enabled", False), \
                patch.object(RTSPReader, "start"), \
                patch.object(AttendanceMonitor, "start"):
            asyncio.run(exercise())


@test
def test_loaded_sessions_not_requested_cuda_determine_commission_validity():
    from app.ai.worker import loaded_providers

    class SessionProviders:
        def __init__(self, providers):
            self.providers = providers

        def get_providers(self):
            return self.providers

    cpu = SimpleNamespace(session=SessionProviders(["CPUExecutionProvider"]))
    cuda = SimpleNamespace(session=SessionProviders(["CUDAExecutionProvider", "CPUExecutionProvider"]))
    faces = SimpleNamespace(app=SimpleNamespace(models={"detection": cuda, "recognition": cpu}))
    with database(commissioned=True) as (engine, service):
        with patch.object(settings, "ai_device", "cuda"):
            service.health.set_providers(loaded_providers(detector=cuda, live=faces, enroll=faces))
            identifier = service.record_crossing(crossing())
        with Session(engine) as db:
            assert db.get(AttendanceRuntime, 1).commission_id is None
            assert db.get(AttendanceEvent, identifier).status == "REVIEW"
        assert service.health.snapshot()["providers"]["live_face_recognition"] == ["CPUExecutionProvider"]


@test
def test_model_file_replacement_revokes_commission_without_loading_any_model():
    import tempfile
    with tempfile.TemporaryDirectory() as directory:
        model = Path(directory) / "model.onnx"
        model.write_bytes(b"isolated fingerprint fixture")
        with patch.object(settings, "yolo_model", str(model)):
            with database(commissioned=True) as (engine, service):
                clock = Clock()
                health = health_for(service, clock)
                warm(health, clock)
                monitor = AttendanceMonitor(service, now=clock.utcnow, monotonic=clock.monotonic)
                monitor.tick()
                with Session(engine) as db:
                    assert db.get(AttendanceRuntime, 1).commission_id is not None
                model.write_bytes(b"replacement model with a different size")
                monitor.tick()
                with Session(engine) as db:
                    assert db.get(AttendanceRuntime, 1).commission_id is None


@test
def test_automatic_notification_failure_cannot_rollback_or_resend_on_source_retry():
    class BrokenNotifier:
        def __init__(self, engine):
            self.engine = engine
            self.committed_counts = []

        def send_text(self, text):
            with Session(self.engine) as db:
                self.committed_counts.append(db.query(WorkSession).count())
            raise RuntimeError("Telegram unavailable")

    with database(commissioned=True) as (engine, service):
        clock = Clock()
        health = health_for(service, clock)
        service._now = clock.utcnow
        warm(health, clock)
        monitor = AttendanceMonitor(service, now=clock.utcnow, monotonic=clock.monotonic)
        monitor.tick()
        notifier = BrokenNotifier(engine)
        service.notifier = notifier
        evidence = crossing(when=clock.utc)
        with patch.object(settings, "telegram_notify_presence", True):
            with expected_error_logs("app.attendance", "Telegram unavailable"):
                identifier = service.record_crossing(evidence)
            assert service.record_crossing(evidence) == identifier
        assert notifier.committed_counts == [1]
        with Session(engine) as db:
            assert db.get(AttendanceEvent, identifier).status == "ACCEPTED"
            assert db.query(WorkSession).one().ended_at is None
        with patch.object(settings, "telegram_notify_presence", False):
            clock.advance(.25)
            process_metrics(health, clock, 999)
            service.record_crossing(crossing(when=clock.utc, direction="EXIT"))
        assert notifier.committed_counts == [1]


@test
def test_monitor_retries_only_one_fifo_head_per_second_and_shutdown_keeps_failed_remainder():
    with database() as (engine, service):
        clock = Clock()
        health = health_for(service, clock)
        queue = AttendanceWriteQueue(service, health)
        monitor = AttendanceMonitor(service, write_queue=queue, now=clock.utcnow, monotonic=clock.monotonic)

        def fail(_conn, _cursor, statement, parameters, _context, _many):
            if "INSERT INTO attendance_events" in statement:
                raise OperationalError(statement, parameters, RuntimeError("disk unavailable"))

        event.listen(engine, "before_cursor_execute", fail)
        try:
            with expected_error_logs("app.attendance.monitor", "disk unavailable"):
                queue.submit(crossing(eligible=False))
                queue.submit(crossing(ordinal=2, eligible=False))
                queue.submit(crossing(ordinal=3, eligible=False))
        finally:
            event.remove(engine, "before_cursor_execute", fail)
        monitor.tick()
        assert len(queue.pending()) == 2
        monitor.tick()
        assert len(queue.pending()) == 2
        clock.advance(1)
        monitor.tick()
        assert len(queue.pending()) == 1
        event.listen(engine, "before_cursor_execute", fail)
        try:
            with expected_error_logs("app.attendance.monitor", "disk unavailable"):
                monitor.shutdown()
        finally:
            event.remove(engine, "before_cursor_execute", fail)
        assert len(queue.pending()) == 1
        with Session(engine) as db:
            assert db.query(AttendanceEvent).count() == 2
            assert any("EVIDENCE_LOST" in row.reason_codes for row in db.query(AttendanceCoverageGap))
            assert db.get(AttendanceRuntime, 1).stopped_at is not None


@test
def test_worker_rotates_two_attendance_roles_ahead_of_optional_nonattendance_camera():
    with database() as (engine, service):
        clock = Clock()
        worker, manager, observations, _queue = worker_for(engine, service, clock)
        with immediate_session(engine) as db:
            db.add(Camera(id=3, name="Preview", nvr_host="unused", username="u", password="p", channel=3))
            db.flush()
            db.add(AttendanceCamera(camera_id=3))
        manager.readers[3] = BufferedReader()
        manager.readers[3].source_revision = 1
        order = []
        original = worker.service.process_frame

        def observed(frame, camera_id, *args, **kwargs):
            result = original(frame, camera_id, *args, **kwargs)
            order.append(camera_id)
            return result

        observations.y = .3
        with patch.object(worker.service, "process_frame", side_effect=observed), \
                patch.object(settings, "ai_cameras", ""), \
                patch("app.ai.worker.time.monotonic", side_effect=clock.monotonic), \
                patch("app.ai.worker.utcnow", side_effect=clock.utcnow):
            for _ in range(2):
                for camera_id, reader in manager.readers.items():
                    reader.buffer.put(FRAME, stream_id=f"stream-{camera_id}",
                                      source_revision=reader.source_revision,
                                      received_at=clock.utc, received_mono=clock.mono)
                worker.step()
                clock.advance(.5)
        assert order == [1, 2, 3, 2, 1, 3]
        assert worker.state.camera(3)["people_count"] == 1


@test
def test_pending_crossing_is_saved_before_optional_global_processing_even_when_it_fails():
    with database() as (engine, service):
        clock = Clock()
        worker, manager, observations, _queue = worker_for(engine, service, clock)
        for y in (.25, .3, .45, .55, .7):
            step(worker, manager, observations, clock, y)
        observed_commits = []

        class OptionalGlobal:
            def drop_camera(self, camera_id):
                return None

            def process(self, *args):
                # An external embedding failure must occur AFTER old crossing
                # state was durably flushed by continuity reconciliation.
                with Session(engine) as db:
                    observed_commits.append(db.query(AttendanceEvent).count())
                raise RuntimeError("Optional model failure")

        worker.global_manager = OptionalGlobal()
        clock.advance(2.1)
        with expected_error_logs("app.ai.worker", "Optional model failure"):
            step(worker, manager, observations, clock, .8)
        with Session(engine) as db:
            assert db.query(AttendanceEvent).one().status == "REVIEW"
        assert worker.state.camera(1) is not None
        assert observed_commits == [1]


@test
def test_recovered_subtick_fault_uses_actual_end_not_later_monitor_sample():
    with database(commissioned=True) as (engine, service):
        clock = Clock()
        health = health_for(service, clock)
        warm(health, clock)
        monitor = AttendanceMonitor(service, now=clock.utcnow, monotonic=clock.monotonic)
        monitor.tick()
        opened = fault(clock, end=None)
        closed = replace(opened, id=str(uuid4()), ended_at=clock.utc + timedelta(seconds=.2))
        health.publish_fault(opened)
        health.publish_fault(closed)
        clock.advance(.5)
        process_metrics(health, clock, 999)
        monitor.tick()
        with Session(engine) as db:
            saved = [row for row in db.query(AttendanceCoverageGap) if opened.id in row.source_fault_ids]
            assert len(saved) == 1 and saved[0].ended_at == closed.ended_at
            assert closed.id in saved[0].source_fault_ids


@test
def test_queued_crossing_revalidates_revisions_without_mutating_captured_record():
    with database() as (engine, service):
        clock = Clock()
        health = health_for(service, clock)
        queue = AttendanceWriteQueue(service, health)
        evidence = crossing()

        def fail(_conn, _cursor, statement, parameters, _context, _many):
            if "INSERT INTO attendance_events" in statement:
                raise OperationalError(statement, parameters, RuntimeError("disk unavailable"))

        event.listen(engine, "before_cursor_execute", fail)
        try:
            with expected_error_logs("app.attendance.monitor", "disk unavailable"):
                queue.submit(evidence)
        finally:
            event.remove(engine, "before_cursor_execute", fail)
        with immediate_session(engine) as db:
            db.get(AttendanceRuntime, 1).gallery_revision += 1
            db.get(AttendanceCamera, 1).source_revision += 1
        assert queue.pending()[0] is evidence
        assert queue.retry_one()
        with Session(engine) as db:
            row = db.query(AttendanceEvent).one()
            assert row.status == "REVIEW" and row.employee_id is None
            assert {"GALLERY_CHANGED", "SOURCE_CHANGED"} <= set(row.reason_codes)
            assert row.raw["gallery_revision"] == 7 and row.raw["source_revision"] == 2
        assert evidence.auto_eligible and evidence.disqualifying_reasons == ()


@test
def test_lifespan_model_initialization_exception_keeps_real_manual_accounting_available():
    import main
    from app.rtsp.reader import RTSPReader
    with database() as (engine, _service):
        app = FastAPI()

        async def exercise():
            async with main.lifespan(app):
                assert app.state.ai_worker is None
                assert not app.state.attendance.health.snapshot()["automatic_ready"]
                app.state.attendance._now = lambda: AS_OF
                saved = manual(app.state.attendance, "ENTRY", BASE)
                with Session(engine) as db:
                    assert db.get(AttendanceEvent, saved.id).status == "ACCEPTED"
                    assert db.query(WorkSession).one().ended_at is None

        with patch.object(main, "init_db"), patch.object(main, "engine", engine), \
                patch.object(main, "SessionLocal", sessionmaker(bind=engine)), \
                patch.object(main, "build_enroll_engine", return_value=None), \
                patch.object(main, "build_ai_worker", side_effect=RuntimeError("ONNX provider load failed")), \
                patch.object(settings, "ai_enabled", True), \
                patch.object(settings, "telegram_enabled", False), \
                patch.object(RTSPReader, "start"), patch.object(AttendanceMonitor, "start"):
            with expected_error_logs("app", "ONNX provider load failed"):
                asyncio.run(exercise())


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
    print(f"{len(TESTS) - failures}/{len(TESTS)} attendance lifecycle tests passed")
    raise SystemExit(bool(failures))
