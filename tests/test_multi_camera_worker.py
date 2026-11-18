"""Deterministic worker admission, deadline and bounded telemetry contracts."""
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.ai.scheduler import DeadlineScheduler
from app.ai.inference_metrics import InferenceMetrics
from app.config import settings
from app.database.database import immediate_session
from app.database.models import AttendanceCamera, AttendanceEvent, Camera, CameraProfile
from sqlalchemy.orm import Session
from test_attendance_lifecycle import Clock, FRAME, BufferedReader, worker_for
from test_attendance_service import database


def test_deadlines_skip_missed_slots_without_catchup_and_respect_limits():
    scheduler = DeadlineScheduler(4)
    scheduler.retain(range(1, 11), 0.)
    first = scheduler.select(range(1, 11), 0.)
    assert first == [1, 2, 3, 4]
    assert scheduler.advance(1, 10., .351) == 3
    assert abs(scheduler.deadlines[1] - .4) < 1e-9
    for camera_id in first[1:]:
        scheduler.advance(camera_id, 5., .01)
    second = scheduler.select(range(1, 11), .01)
    assert second == [5, 6, 7, 8]
    for camera_id in second:
        scheduler.advance(camera_id, 5., .02)
    assert scheduler.select(range(1, 11), .02) == [9, 10]
    assert 1 not in scheduler.select(range(1, 11), .399)
    assert scheduler.delay(.399, .01) == 0.  # cameras9/10 are still due
    scheduler.retain([1], .399)
    assert 0 < scheduler.delay(.399, .01) <= .001000001


def test_rotating_ties_do_not_starve_and_never_duplicate_cameras():
    scheduler = DeadlineScheduler(2)
    scheduler.retain([1, 2, 3, 4, 5], 10.)
    seen = []
    for _ in range(3):
        selected = scheduler.select([1, 2, 2, 3, 4, 5], 10.)
        assert len(selected) <= 2 and len(set(selected)) == len(selected)
        seen.extend(selected)
        for camera_id in selected:
            scheduler.advance(camera_id, 5., 10.)
    assert seen == [1, 2, 3, 4, 5]


def test_early_analysis_fps_is_a_real_ten_second_window_not_extrapolation():
    metrics = InferenceMetrics()
    metrics.configure(1, 5., 0.)
    metrics.complete(1, .05, .1, started_mono=.08)
    view = metrics.snapshot(1, .2)
    assert view["processed_fps_10s"] == .1
    assert abs(view["queue_age_seconds"] - .03) < 1e-9
    assert abs(view["processing_seconds"] - .02) < 1e-9
    assert abs(view["completion_age_seconds"] - .05) < 1e-9


def test_capacity_hysteresis_latency_and_retention_use_real_completions():
    metrics = InferenceMetrics()
    metrics.configure(1, 5., 0.)
    metrics.snapshot(1, 0.)
    assert metrics.snapshot(1, 10.)["reason_codes"] == ["CAPACITY_DEGRADED"]
    for index in range(51, 151):
        now = index / 5.
        metrics.complete(1, now - .05, now, face_deferred=1, stage_times_ms={"person": 5.})
        view = metrics.snapshot(1, now)
    assert view["reason_codes"] == [] and view["processed_fps_10s"] >= 4.5
    assert abs(view["completion_age_p95_seconds"] - .05) < 1e-9
    assert view["face_deferred"] == 100 and view["stage_times_ms"] == {"person": 5.}
    metrics.pending(1, 30.)
    assert metrics.snapshot(1, 30.25)["pending_age_seconds"] == .25
    assert metrics.snapshot(1, 100.)["completion_age_p95_seconds"] is None
    metrics.drop(1)
    assert metrics.snapshots(100.) == {}


def contexts(clock):
    return (patch("app.ai.worker.time.monotonic", side_effect=clock.monotonic),
            patch("app.ai.worker.utcnow", side_effect=clock.utcnow))


def publish(manager, clock, camera_id=1, stream="stream-a"):
    reader = manager.get(camera_id)
    reader.buffer.put(FRAME, stream_id=stream, source_revision=reader.source_revision,
                      received_at=clock.utc, received_mono=clock.mono)


def test_disabled_attendance_camera_still_runs_general_person_tracking():
    with database() as (engine, service):
        clock = Clock()
        worker, manager, observations, queue = worker_for(engine, service, clock)
        with immediate_session(engine) as db:
            db.get(AttendanceCamera, 1).enabled = False
        observations.y = .3
        publish(manager, clock)
        a, b = contexts(clock)
        with a, b:
            assert worker.step()
        assert worker.state.camera(1)["people_count"] == 1
        assert worker._configurations[1].attendance is False
        assert queue.pending() == ()
        with Session(engine) as db:
            assert db.query(AttendanceEvent).count() == 0


def test_replacement_during_native_inference_publishes_no_old_boxes_or_crossings():
    with database() as (engine, service):
        clock = Clock()
        worker, manager, observations, queue = worker_for(engine, service, clock)
        observations.y = .3
        publish(manager, clock)
        original = worker.service.process_batch
        def replace_reader(inputs):
            outcomes = original(inputs)
            replacement = BufferedReader()
            manager.readers[1] = replacement
            publish(manager, clock, stream="replacement")
            return outcomes
        a, b = contexts(clock)
        with a, b, patch.object(worker.service, "process_batch", side_effect=replace_reader):
            assert worker.step()
        assert worker.state.camera(1) is None
        assert queue.pending() == ()
        with Session(engine) as db:
            assert db.query(AttendanceEvent).count() == 0
        assert 1 not in worker.service._trackers


def test_newer_slot_is_refreshed_at_admission_and_batch_failures_are_rate_limited():
    with database() as (engine, service):
        clock = Clock()
        worker, manager, observations, _queue = worker_for(engine, service, clock)
        observations.y = .3
        publish(manager, clock)
        captured = []
        original_read = worker._read_configurations
        calls = 0
        def read():
            nonlocal calls
            calls += 1
            result = original_read()
            if calls == 2:
                publish(manager, clock)
            return result
        def fail(inputs):
            captured.extend(inputs)
            raise RuntimeError("controlled batch failure")
        a, b = contexts(clock)
        with a, b, patch.object(worker, "_read_configurations", side_effect=read), \
                patch.object(worker.service, "process_batch", side_effect=fail) as inference:
            import logging
            from unittest import TestCase
            with TestCase().assertLogs("app.ai.worker", level=logging.ERROR):
                assert worker.step()
            assert captured[0].packet.seq == 2
            publish(manager, clock)
            assert worker.step() is False and inference.call_count == 1
        with patch("app.ai.worker.time.monotonic", side_effect=clock.monotonic):
            assert worker.inference_snapshot()[1]["reason_codes"] == ["AI_UNAVAILABLE"]


def test_ten_cameras_are_batched_once_and_attendance_deadlines_remain_faster():
    with database() as (engine, service):
        clock = Clock()
        with patch.object(settings, "ai_batch_size", 4):
            worker, manager, observations, _queue = worker_for(engine, service, clock)
        with immediate_session(engine) as db:
            for camera_id in range(3, 11):
                db.add(Camera(id=camera_id, name=f"Fixture {camera_id}", nvr_host="unused",
                              username="test", password="test", channel=camera_id))
                db.flush()
                db.add_all([AttendanceCamera(camera_id=camera_id),
                            CameraProfile(camera_id=camera_id, zone="INTERIOR", ai_enabled=True, revision=0)])
                reader = BufferedReader()
                reader.source_revision = 1
                manager.readers[camera_id] = reader
        observations.y = .3
        batches = []
        original = worker.service.process_batch
        def capture(inputs):
            batches.append([item.camera_id for item in inputs])
            return original(inputs)
        a, b = contexts(clock)
        with a, b, patch.object(settings, "ai_fps", 5.), \
                patch.object(settings, "attendance_ai_fps", 10.), \
                patch.object(worker.service, "process_batch", side_effect=capture):
            for camera_id in manager.reader_ids():
                publish(manager, clock, camera_id)
            for _ in range(3):
                assert worker.step()
            admitted = [camera_id for batch in batches for camera_id in batch]
            assert len(admitted) == len(set(admitted)) == 10 and set(admitted) == set(range(1, 11))
            assert all(len(batch) <= 4 for batch in batches)
            assert all(worker.state.camera(camera_id)["people_count"] == 1
                       for camera_id in range(1, 11))
            clock.advance(.11)
            for camera_id in manager.reader_ids():
                publish(manager, clock, camera_id)
            assert worker.step()
            assert set(batches[-1]) == {1, 2} and len(batches[-1]) == 2
            assert worker.step() is False

def interior_camera(engine, manager, camera_id, **profile):
    with immediate_session(engine) as db:
        db.add(Camera(id=camera_id, name=f"Fixture {camera_id}", nvr_host="unused",
                      username="test", password="test", channel=camera_id))
        db.flush()
        db.add_all([AttendanceCamera(camera_id=camera_id),
                    CameraProfile(camera_id=camera_id, zone="INTERIOR", revision=0,
                                  **dict(dict(ai_enabled=True), **profile))])
    reader = BufferedReader()
    reader.source_revision = 1
    manager.readers[camera_id] = reader


def run_rounds(worker, manager, clock, batches, advances):
    original = worker.service.process_batch

    def capture(inputs):
        batches.append([item.camera_id for item in inputs])
        return original(inputs)

    a, b = contexts(clock)
    with a, b, patch.object(settings, "ai_fps", 5.), patch.object(settings, "attendance_ai_fps", 10.), \
            patch.object(worker.service, "process_batch", side_effect=capture):
        for seconds in advances:
            clock.advance(seconds)
            for camera_id in manager.reader_ids():
                publish(manager, clock, camera_id)
            while worker.step():
                pass


def test_camera_with_ai_switched_off_is_never_submitted_but_stays_online():
    with database() as (engine, service):
        clock = Clock()
        worker, manager, observations, _queue = worker_for(engine, service, clock)
        interior_camera(engine, manager, 3, ai_enabled=False)
        interior_camera(engine, manager, 4)
        observations.y = .3
        batches = []
        run_rounds(worker, manager, clock, batches, [.1, .1])
        admitted = {camera_id for batch in batches for camera_id in batch}
        assert admitted == {1, 2, 4}, admitted
        assert worker.state.camera(3) is None


def test_interior_camera_runs_at_its_own_ai_rate():
    with database() as (engine, service):
        clock = Clock()
        worker, manager, observations, _queue = worker_for(engine, service, clock)
        interior_camera(engine, manager, 3)
        interior_camera(engine, manager, 4, ai_fps=2.)
        observations.y = .3
        batches = []
        run_rounds(worker, manager, clock, batches, [.1])
        assert {3, 4} <= {camera_id for batch in batches for camera_id in batch}
        batches.clear()
        # 0.3 s later the 5 FPS camera is due again (0.2 s); the 2 FPS camera (0.5 s) is not.
        run_rounds(worker, manager, clock, batches, [.3])
        second = {camera_id for batch in batches for camera_id in batch}
        assert 3 in second and 4 not in second, second
        batches.clear()
        run_rounds(worker, manager, clock, batches, [.25])
        assert 4 in {camera_id for batch in batches for camera_id in batch}

if __name__ == "__main__":
    import traceback
    cases = [value for name, value in list(globals().items()) if name.startswith("test_") and callable(value)]
    failed = 0
    for case in cases:
        try:
            case()
            print("PASS", case.__name__)
        except Exception:
            failed += 1
            traceback.print_exc()
    print(f"{len(cases)-failed}/{len(cases)} passed")
    raise SystemExit(bool(failed))
