"""Deterministic worker admission, deadline and bounded telemetry contracts."""
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.ai.employee_appearance import EmployeeAppearanceGallery
from app.ai.face_quality import FaceQuality
from app.ai.recognition_service import FaceEvidence
from app.ai.scheduler import DeadlineScheduler
from app.ai.inference_metrics import InferenceMetrics
from app.config import settings
from app.database.database import immediate_session
from app.database.models import AttendanceCamera, Camera, CameraProfile, DoorCrossing
from sqlalchemy.orm import Session
from attendance_test_support import Clock, FRAME, BufferedReader, database, step, worker_for


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
        worker, manager, observations = worker_for(engine, service, clock)
        with immediate_session(engine) as db:
            db.get(AttendanceCamera, 1).enabled = False
        observations.y = .3
        publish(manager, clock)
        a, b = contexts(clock)
        with a, b:
            assert worker.step()
        assert worker.state.camera(1)["people_count"] == 1
        assert worker._configurations[1].attendance is False
        assert service.snapshot()["pending_crossings"] == 0
        service.flush()
        with Session(engine) as db:
            assert db.query(DoorCrossing).count() == 0


def test_replacement_during_native_inference_publishes_no_old_boxes_or_crossings():
    with database() as (engine, service):
        clock = Clock()
        worker, manager, observations = worker_for(engine, service, clock)
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
        assert service.snapshot()["pending_crossings"] == 0
        service.flush()
        with Session(engine) as db:
            assert db.query(DoorCrossing).count() == 0
        assert 1 not in worker.service._trackers


def test_newer_slot_is_refreshed_at_admission_and_batch_failures_are_rate_limited():
    with database() as (engine, service):
        clock = Clock()
        worker, manager, observations = worker_for(engine, service, clock)
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
            worker, manager, observations = worker_for(engine, service, clock)
        with immediate_session(engine) as db:
            for camera_id in range(3, 11):
                db.add(Camera(id=camera_id, name=f"Fixture {camera_id}", nvr_host="unused",
                              username="test", password="test", channel=camera_id))
                db.flush()
                db.add_all([AttendanceCamera(camera_id=camera_id),
                            CameraProfile(camera_id=camera_id, zone="INTERIOR", ai_enabled=True,
                                          time_tracking=False, revision=0)])
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
                    CameraProfile(camera_id=camera_id, zone="INTERIOR", revision=0, time_tracking=False,
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
        worker, manager, observations = worker_for(engine, service, clock)
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
        worker, manager, observations = worker_for(engine, service, clock)
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


VECTOR = np.zeros(512, np.float32)
VECTOR[0] = 1.


class RecordingPresence:
    """Presence sink double: records what the worker reports; storage has its own recorder tests."""
    def __init__(self, engine):
        self.engine = engine
        self.observed, self.crossings = [], []

    def observe(self, camera_id, employee_id, source, observed_at):
        self.observed.append((camera_id, employee_id, source))

    def record_crossing(self, **crossing):
        self.crossings.append(crossing)

    def has_employee_work(self, employee_id):
        return False

    def has_camera_work(self, camera_id):
        return False


def confirmed_face(packet, employee_id):
    name = f"Employee {employee_id}"
    return FaceEvidence(
        frame_seq=packet.seq, received_at=packet.received_at, received_mono=packet.received_mono,
        candidate_employee_id=employee_id, candidate_employee_name=name, score=.91,
        runner_up_employee_id=None, runner_up_employee_name=None, runner_up_score=0., margin=.91,
        quality=FaceQuality(100., .99, 90., True), store_revision=7,
        confirmed_employee_id=employee_id, confirmed_employee_name=name,
        supporting_frame_sequences=(packet.seq,), supporting_received_monos=(packet.received_mono,),
        supporting_received_ats=(packet.received_at,), confirmed_at_mono=packet.received_mono,
        face_bbox=None, aligned_jpeg=None, reason_codes=())


class ScriptedFaces:
    """The verdicts a face model would have produced, injected over the real tracker outcome."""
    def __init__(self, worker):
        self.verdicts = {}
        original = worker.service.process_batch

        def process(inputs):
            outcomes = original(inputs)
            for item, outcome in zip(inputs, outcomes):
                people = {id(person): person for person in outcome.detections + outcome.observed_tracks}
                for person in people.values():
                    verdict = self.verdicts.get((item.camera_id, person.track_id),
                                                self.verdicts.get((item.camera_id, None)))
                    if verdict is None:
                        continue
                    employee_id = verdict.get("employee")
                    if employee_id is not None:
                        person.employee_id, person.employee_name = employee_id, f"Employee {employee_id}"
                        if verdict.get("evidence", True):
                            person.fresh_face_evidence = confirmed_face(item.packet, employee_id)
                    person.association_uncertain = verdict.get("uncertain", False)
                    person.identity_reason_codes = tuple(verdict.get("reasons", ()))
            return outcomes

        worker.service.process_batch = process

    def set(self, camera_id, track_id=None, **verdict):
        """`track_id=None` applies to every track of the camera that has no verdict of its own."""
        self.verdicts[(camera_id, track_id)] = verdict

    def clear(self, camera_id, track_id=None):
        self.verdicts.pop((camera_id, track_id), None)


class AdmittingJournal:
    """Journal double that admits every observed track, so appearance work is queued for it."""
    def tick(self, now):
        pass

    def observe(self, camera_id, packet, outcome):
        return [SimpleNamespace(track_id=person.track_id, status="COMMITTED",
                                visit_id=f"{camera_id}-{outcome.tracking_generation}-{person.track_id}")
                for person in outcome.observed_tracks]

    def drain_acknowledgements(self):
        return []

    def reset_camera(self, camera_id, generation, reason):
        pass


class FixedReid:
    def __init__(self, vector):
        self.vector = vector
        self.calls = []

    def embed(self, crops):
        self.calls.append(len(crops))
        return np.tile(self.vector, (len(crops), 1))


def tracking_worker(engine, clock, *, appearance=False):
    worker, manager, observations = worker_for(engine, RecordingPresence(engine), clock)
    faces = ScriptedFaces(worker)
    gallery = None
    if appearance:
        gallery = worker.appearance_gallery = EmployeeAppearanceGallery()
        worker.journal, worker.reid = AdmittingJournal(), FixedReid(VECTOR)
    return worker, manager, observations, faces, gallery


def tick(worker, manager, clock, *camera_ids, advance=.25):
    for camera_id in camera_ids:
        publish(manager, clock, camera_id)
    a, b = contexts(clock)
    with a, b, patch("app.ai.doorway.time.monotonic", side_effect=clock.monotonic):
        while worker.step():
            pass
    clock.advance(advance)


def test_face_identity_is_sticky_cleared_by_uncertainty_or_gallery_loss_and_replaced_only_by_another_face():
    with database() as (engine, _presence):
        clock = Clock()
        worker, manager, _observations, faces, _gallery = tracking_worker(engine, clock)

        def frame(**verdict):
            faces.clear(1)
            if verdict:
                faces.set(1, **verdict)
            before = len(worker.presence.observed)
            tick(worker, manager, clock, 1)
            return worker.presence.observed[before:]

        assert frame() == []
        assert frame(employee=1) == [(1, 1, "FACE")]
        assert frame() == [(1, 1, "FACE")], "A track keeps its face identity after the face disappears"
        assert frame(uncertain=True) == []
        assert frame() == [], "An uncertain association drops the identity for good"
        assert frame(employee=1) == [(1, 1, "FACE")]
        assert frame(reasons=("GALLERY_CHANGED",)) == []
        assert frame() == []
        assert frame(employee=1) == [(1, 1, "FACE")]
        assert frame(reasons=("GALLERY_UNAVAILABLE",)) == []
        assert frame(employee=1) == [(1, 1, "FACE")]
        assert frame(employee=2) == [(1, 2, "FACE")]
        assert frame() == [(1, 2, "FACE")], "Only a different confirmed face replaces the identity"


def test_presence_is_observed_only_on_time_tracking_cameras_and_toggling_never_resets_tracking():
    with database() as (engine, _presence):
        with immediate_session(engine) as db:
            db.get(CameraProfile, 2).time_tracking = False
        clock = Clock()
        worker, manager, _observations, faces, _gallery = tracking_worker(engine, clock)
        faces.set(1, employee=1)
        faces.set(2, employee=1)
        tick(worker, manager, clock, 1, 2)
        assert worker.presence.observed == [(1, 1, "FACE")]
        assert worker.state.camera(2)["tracks"][0]["employee_id"] == 1, "Still shown, just not counted"
        generation = worker.service._generations[2]
        with immediate_session(engine) as db:
            db.get(CameraProfile, 2).time_tracking = True
            db.get(CameraProfile, 1).time_tracking = False
        worker.presence.observed.clear()
        tick(worker, manager, clock, 1, 2)
        assert worker.presence.observed == [(2, 1, "FACE")]
        assert worker.service._generations[2] == generation, "Toggling time tracking must not reset a camera"


def test_faceless_track_is_named_by_appearance_after_two_matches_learned_from_a_face_elsewhere():
    with database() as (engine, _presence):
        clock = Clock()
        worker, manager, _observations, faces, gallery = tracking_worker(engine, clock, appearance=True)
        with patch.object(settings, "reid_embedding_interval", .2):
            assert gallery.match(VECTOR, clock.utc) is None
            faces.set(1, employee=1)
            tick(worker, manager, clock, 1)
            assert gallery.match(VECTOR, clock.utc) == 1, "A face-confirmed track teaches the gallery"
            embedded = lambda: sum(worker.reid.calls) - 1  # crops of camera 2; the first one taught the gallery
            for _ in range(12):
                tick(worker, manager, clock, 2)
                track = worker.state.camera(2)["tracks"][0]
                if embedded() < 2:
                    assert track["employee_id"] is None, "One appearance match is not enough"
                    assert [item for item in worker.presence.observed if item[0] == 2] == []
                if track["employee_id"] is not None:
                    break
            assert embedded() >= 2
            assert (track["employee_id"], track["employee_name"], track["identity_source"]) == (
                1, "Employee 1", "REID")
            tick(worker, manager, clock, 2)
            assert [item for item in worker.presence.observed if item[0] == 2][-1:] == [(2, 1, "REID")]


def test_appearance_cannot_name_a_second_track_while_another_live_track_holds_the_employee():
    two_people = lambda frame: [(np.array([220., -80., 420., 120.]), .99),
                                (np.array([20., 100., 160., 400.]), .98)]
    with database() as (engine, _presence):
        clock = Clock()
        worker, manager, observations, faces, _gallery = tracking_worker(engine, clock, appearance=True)
        observations.detect = two_people
        with patch.object(settings, "reid_embedding_interval", .2):
            tick(worker, manager, clock, 2)
            tracks = sorted(worker.state.camera(2)["tracks"], key=lambda track: track["bbox"][0])
            left, right = tracks[0]["track_id"], tracks[1]["track_id"]
            faces.set(2, left, employee=1)
            for _ in range(8):
                tick(worker, manager, clock, 2)
            assert {item[1:] for item in worker.presence.observed if item[0] == 2} == {(1, "FACE")}
            view = {track["track_id"]: track for track in worker.state.camera(2)["tracks"]}
            assert view[left]["employee_id"] == 1 and view[right]["employee_id"] is None
            votes = next(item for key, item in worker._identities.items() if key[0] == 2 and key[2] == right)
            assert votes.reid_candidate == 1 and votes.reid_votes >= 2, "Blocked by the holder, not unmatched"
            # The holder leaves: once its track is gone the other one can be named.
            observations.detect = lambda frame: [(np.array([220., -80., 420., 120.]), .99)]
            for _ in range(40):
                tick(worker, manager, clock, 2)
                if worker.state.camera(2)["tracks"][0]["employee_id"] is not None:
                    break
            track = worker.state.camera(2)["tracks"][0]
            assert track["track_id"] == right and track["identity_source"] == "REID"
            assert not [key for key in worker._identities if key[0] == 2 and key[2] == left]


def door_crossings(*, employee, time_tracking=True, expire_with_frames=False, evidence=True):
    """One person walks through the ENTRY door of camera 1; returns what reached the presence sink."""
    with database() as (engine, _presence):
        if not time_tracking:
            with immediate_session(engine) as db:
                db.get(CameraProfile, 1).time_tracking = False
        clock = Clock()
        worker, manager, observations, faces, _gallery = tracking_worker(engine, clock)
        seen, record = [], worker._record_crossings

        def spy(crossings, *args):
            crossings = list(crossings)
            seen.extend(crossings)
            return record(crossings, *args)

        worker._record_crossings = spy
        generation = None
        if employee is not None:
            faces.set(1, employee=employee, evidence=evidence)
        for index, y in enumerate((.25, .3, .45, .55, .7)):
            step(worker, manager, observations, clock, y)
            if index == 0:
                faces.clear(1)  # the face is visible on the first frame only
                generation = worker.service._generations[1]
        if expire_with_frames:
            for _ in range(16):
                step(worker, manager, observations, clock, None)
        else:
            clock.advance(2.1)
            step(worker, manager, observations, clock, publish=False)
        return worker.presence.crossings, seen, generation


def test_door_crossing_of_a_known_employee_is_recorded_with_the_documented_source_key():
    for evidence in (True, False):
        for expire_with_frames in (False, True):
            recorded, seen, generation = door_crossings(
                employee=1, evidence=evidence, expire_with_frames=expire_with_frames)
            # With evidence the door itself confirmed the face; without it only the track identity knows.
            assert [(item.direction, item.confirmed_employee_id) for item in seen] == [
                ("ENTRY", 1 if evidence else None)], (evidence, expire_with_frames)
            assert recorded == [dict(
                source_key=f"camera:1:source-a:{generation}:1:1", camera_id=1, employee_id=1,
                direction="ENTRY", occurred_at=seen[0].occurred_at)], (evidence, expire_with_frames)


def test_unidentified_crossing_and_crossing_on_a_camera_without_time_tracking_are_discarded():
    recorded, seen, _generation = door_crossings(employee=None)
    assert len(seen) == 1 and recorded == []
    recorded, seen, _generation = door_crossings(employee=1, time_tracking=False)
    assert len(seen) == 1 and recorded == []


def test_track_identity_ends_with_an_expired_track_and_with_a_camera_reset():
    with database() as (engine, _presence):
        clock = Clock()
        worker, manager, observations, faces, _gallery = tracking_worker(engine, clock)
        faces.set(1, employee=1)
        step(worker, manager, observations, clock, .25)
        faces.clear(1)
        assert [key for key in worker._identities if key[0] == 1]
        for _ in range(16):
            step(worker, manager, observations, clock, None)
        assert not [key for key in worker._identities if key[0] == 1], "An expired track drops its identity"
        faces.set(1, employee=1)
        step(worker, manager, observations, clock, .25)
        faces.clear(1)
        assert any(item.employee_id == 1 for item in worker._identities.values())
        reported = len(worker.presence.observed)
        step(worker, manager, observations, clock, .25, stream="source-b")
        assert not any(item.employee_id for item in worker._identities.values())
        assert len(worker.presence.observed) == reported, "A new stream starts unidentified"

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
