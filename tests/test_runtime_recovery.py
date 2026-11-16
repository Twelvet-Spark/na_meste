"""Failure/reconnect telemetry, idle polling and costed multi-camera scheduling."""
import logging
import sys
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch
from datetime import timedelta

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from app.config import settings
from app.database.database import immediate_session
from app.database.models import AttendanceCamera, Camera
from test_attendance_lifecycle import Clock, FRAME, BufferedReader, worker_for
from test_attendance_service import database
from test_multi_camera_worker import contexts, publish


def add_camera(engine, manager, camera_id):
    with immediate_session(engine) as db:
        db.add(Camera(id=camera_id, name=f"Fixture {camera_id}", nvr_host="unused",
                      username="test", password="test", channel=camera_id))
        db.flush()
        db.add(AttendanceCamera(camera_id=camera_id))
    reader = BufferedReader()
    reader.source_revision = 1
    manager.readers[camera_id] = reader


def test_repeated_failed_batches_cannot_restart_capacity_hysteresis():
    with database() as (engine, service):
        clock = Clock()
        worker, manager, observations, _queue = worker_for(engine, service, clock)
        observations.y = .3
        a, b, c = contexts(clock)
        with a, b, c:
            with patch.object(worker.service, "process_batch", side_effect=RuntimeError("injected inference failure")), \
                    TestCase().assertLogs("app.ai.worker", level=logging.ERROR):
                for _ in range(12):
                    publish(manager, clock)
                    assert worker.step()
                    clock.advance(1.1)
            failed = worker.inference_snapshot()[1]
            assert set(failed["reason_codes"]) == {"CAPACITY_DEGRADED", "AI_UNAVAILABLE"}
            assert failed["processed_fps_10s"] == 0.
            publish(manager, clock)
            assert worker.step()
            recovered_once = worker.inference_snapshot()[1]
            assert recovered_once["reason_codes"] == ["CAPACITY_DEGRADED"]
            assert recovered_once["processed_fps_10s"] == .1


def test_reconnect_keeps_both_windows_until_expiry_but_analysis_removal_retires_them():
    with database() as (engine, service):
        clock = Clock()
        worker, manager, observations, _queue = worker_for(engine, service, clock)
        observations.y = .3
        original = worker.service.process_batch
        slow_once = True
        def inferred(inputs):
            nonlocal slow_once
            result = original(inputs)
            if slow_once:
                clock.advance(.2)
                slow_once = False
            return result
        a, b, c = contexts(clock)
        with a, b, c, patch.object(worker.service, "process_batch", side_effect=inferred):
            publish(manager, clock)
            assert worker.step()
            clock.advance(.2)
            publish(manager, clock)
            assert worker.step()
            clock.advance(.2)
            manager.readers[1] = BufferedReader()
            publish(manager, clock, stream="reconnected")
            assert worker.step()
            view = worker.inference_snapshot()[1]
            assert view["processed_fps_10s"] == .3
            assert .199 < view["completion_age_p95_seconds"] < .201
            clock.advance(11.)
            view = worker.inference_snapshot()[1]
            assert view["processed_fps_10s"] == 0.
            assert .199 < view["completion_age_p95_seconds"] < .201
            clock.advance(50.)
            assert worker.inference_snapshot()[1]["completion_age_p95_seconds"] is None
            with immediate_session(engine) as db:
                db.get(Camera, 1).enabled = False
            worker.step()
            assert 1 not in worker.inference_snapshot()


def test_offline_and_cooldown_do_not_force_tight_polling_or_block_healthy_camera():
    with database() as (engine, service):
        clock = Clock()
        worker, manager, observations, _queue = worker_for(engine, service, clock)
        add_camera(engine, manager, 3)
        observations.y = .3
        batches = []
        original = worker.service.process_batch
        def inferred(inputs):
            batches.append([item.camera_id for item in inputs])
            return original(inputs)
        a, b, c = contexts(clock)
        with a, b, c, patch.object(worker.service, "process_batch", side_effect=inferred):
            publish(manager, clock, 1)
            publish(manager, clock, 3)
            assert worker.step()
            clock.advance(.21)
            assert worker.step() is False
            assert worker._idle_delay(clock.mono) == .01
            worker._retry_after[1] = clock.mono + .4
            publish(manager, clock, 1)
            publish(manager, clock, 3)
            assert worker.step()
            assert batches[-1] == [3]
            assert worker.step() is False
            assert worker._idle_delay(clock.mono) == .01
            clock.advance(.399)
            assert 0 < worker._idle_delay(clock.mono) <= .0010001
            clock.advance(.002)
            publish(manager, clock, 1)
            assert worker.step() and batches[-1] == [1]


def test_source_frame_skips_are_not_missed_deadlines_and_restart_never_inflates_them():
    with database() as (engine, service):
        clock = Clock()
        worker, manager, observations, _queue = worker_for(engine, service, clock)
        with immediate_session(engine) as db:
            db.get(AttendanceCamera, 1).enabled = False
        observations.y = .3
        a, b, c = contexts(clock)
        with a, b, c, patch.object(settings, "ai_fps", 5.):
            publish(manager, clock)
            assert worker.step()
            clock.advance(.2)
            for _ in range(3):
                publish(manager, clock)
            assert worker.step()
            view = worker.inference_snapshot()[1]
            assert view["skipped_frames"] == 2 and view["skipped_slots"] == 0
            clock.advance(.2)
            manager.readers[1] = BufferedReader()
            publish(manager, clock, stream="new-reader-sequence-one")
            assert worker.step()
            view = worker.inference_snapshot()[1]
            assert view["skipped_frames"] == 2 and view["skipped_slots"] == 0


def test_ten_and_thirty_cameras_with_nonzero_batch_cost_keep_due_service_fair():
    for camera_count in (10, 30):
        with database() as (engine, service):
            clock = Clock()
            with patch.object(settings, "ai_batch_size", 4):
                worker, manager, observations, _queue = worker_for(engine, service, clock)
            for camera_id in range(3, camera_count + 1):
                add_camera(engine, manager, camera_id)
            observations.y = .3
            original = worker.service.process_batch
            def costed(inputs):
                assert len(inputs) <= 4 and len({item.camera_id for item in inputs}) == len(inputs)
                result = original(inputs)
                clock.advance(.005 + .002 * len(inputs))
                return result
            started, next_source = clock.mono, clock.mono
            a, b, c = contexts(clock)
            with a, b, c, patch.object(settings, "ai_fps", 5.), \
                    patch.object(settings, "attendance_ai_fps", 10.), \
                    patch.object(worker.service, "process_batch", side_effect=costed):
                while clock.mono - started < 25.:
                    while next_source <= clock.mono + 1e-9:
                        for camera_id, reader in manager.readers.items():
                            reader.buffer.put(FRAME, stream_id=f"source-{camera_id}",
                                              source_revision=reader.source_revision,
                                              received_at=clock.utc - timedelta(seconds=clock.mono - next_source),
                                              received_mono=next_source)
                        next_source += 1. / 15.
                    if not worker.step():
                        clock.advance(max(.0001, worker._idle_delay(clock.mono)))
                measurements = worker.inference_snapshot()
                assert set(measurements) == set(range(1, camera_count + 1))
                for camera_id, view in measurements.items():
                    minimum = 8. if camera_id in (1, 2) else 4.5
                    assert view["processed_fps_10s"] >= minimum, (camera_count, camera_id, view)
                    assert view["completion_age_p95_seconds"] <= 1., (camera_count, camera_id, view)
                print(f"MEASURED synthetic-cost cameras={camera_count}, batch=4, fixed=5ms + 2ms/frame")


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
    print(f"{len(cases)-failed}/{len(cases)} runtime recovery cases passed")
    raise SystemExit(bool(failed))
