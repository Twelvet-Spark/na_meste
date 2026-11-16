"""Real worker/journal/SQLite seams; synthetic model outputs, never identity accuracy."""
import sys
from contextlib import contextmanager, ExitStack
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from fastapi import FastAPI, HTTPException
from sqlalchemy.orm import Session

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from app.ai.global_tracker import GlobalIdentityManager
from app.api import cameras
from app.database.database import immediate_session
from app.database.models import (
    AttendanceCamera, AttendanceEvent, AttendanceRuntime, Camera, EmployeeFace,
    FaceModelMetadata, GlobalPerson, HumanVisit, HumanVisitEvent, HumanJournalGap, WorkSession,
)
from app.services.attendance_monitor import AttendanceMonitor
from app.services.human_journal import HumanJournal
from test_attendance_lifecycle import Clock, FRAME, worker_for
from test_attendance_service import database
from test_human_journal import idle
from test_multi_camera_worker import contexts, publish
from test_runtime_recovery import add_camera
from test_batch_recognition import Faces
from test_attendance_recognition import MODEL_ID, vector


class Body:
    def __init__(self):
        self.calls = []
        self.mutate = None

    def embed(self, crops):
        assert all(crop.shape == (256, 128, 3) and crop.flags.owndata for crop in crops)
        self.calls.append(len(crops))
        if self.mutate is not None:
            self.mutate()
        result = np.zeros((len(crops), 512), np.float32)
        for index, crop in enumerate(crops):
            result[index, int(crop.mean() >= 127)] = 1.
        return result


@contextmanager
def fixture(*, body=None, journal_options=None):
    with database() as (engine, attendance):
        clock = Clock()
        worker, manager, detector, _queue = worker_for(engine, attendance, clock)
        add_camera(engine, manager, 3)
        journal = HumanJournal(engine, now=clock.utcnow, monotonic=clock.monotonic,
                               **(journal_options or {}))
        worker.journal = journal
        worker.reid = body
        worker.global_manager = GlobalIdentityManager() if body is not None else None
        idle(journal)
        with ExitStack() as stack:
            for context in contexts(clock):
                stack.enter_context(context)
            try:
                yield engine, attendance, clock, worker, manager, detector, journal
            finally:
                worker.flush('SHUTDOWN')
                result = journal.close()
                assert result.drained, result


def request_for(worker, manager, journal):
    app = FastAPI()
    app.state.manager = manager
    app.state.ai_worker = worker
    app.state.detection_state = worker.state
    app.state.human_journal = journal
    return SimpleNamespace(app=app)


def test_first_hit_provisional_without_face_or_body_and_ack_only_live_link():
    with fixture() as (engine, _attendance, clock, worker, manager, _detector, journal):
        worker.service.faces = None
        worker.service.min_hits = 2
        with journal._condition:
            publish(manager, clock, 3)
            assert worker.step()
            assert worker.state.camera(3)['tracks'] == []
        idle(journal)
        with Session(engine) as db:
            row = db.query(HumanVisit).one()
            visit_id = row.id
            assert row.status == 'PROVISIONAL' and row.employee_id is None and row.global_id is None
            assert row.snapshot is not None and row.face_thumbnail is None
        clock.advance(.2)
        with journal._condition:
            publish(manager, clock, 3)
            assert worker.step()
        idle(journal)
        worker.step()
        view = worker.state.camera(3)
        assert view['tracks'][0]['visit_id'] == visit_id
        assert view['tracks'][0]['employee_id'] is None
        assert (view['source_revision'], view['stream_id'], view['frame_seq']) == (1, 'stream-a', 2)
        assert (view['frame_width'], view['frame_height']) == (640, 480)
        observed = datetime.fromisoformat(view['observed_at'])
        assert observed.utcoffset() == timedelta(0) and observed.replace(tzinfo=None) == clock.utc


def test_pending_visit_protects_camera_before_async_writer_can_commit():
    with fixture() as (engine, _attendance, clock, worker, manager, _detector, journal):
        request = request_for(worker, manager, journal)
        with journal._condition:
            publish(manager, clock, 3)
            worker.step()
            with Session(engine) as db:
                assert db.query(HumanVisit).count() == 0
                try:
                    cameras.api_delete_camera(3, db=db, manager=SimpleNamespace(stop=lambda _: None), request=request)
                except HTTPException as exc:
                    assert exc.status_code == 409
                else:
                    raise AssertionError('Admitted but uncommitted visit lost its camera FK')
                assert db.get(Camera, 3) is not None
        idle(journal)
        with Session(engine) as db:
            assert db.query(HumanVisit).one().camera_id == 3


def test_old_stream_ack_never_becomes_a_live_visit_link():
    with fixture() as (engine, _attendance, clock, worker, manager, _detector, journal):
        with journal._condition:
            publish(manager, clock, 3, stream='old')
            worker.step()
            first = worker.state.camera(3)
            assert first['tracks'][0]['visit_id'] is None
            clock.advance(.2)
            publish(manager, clock, 3, stream='new')
        idle(journal)
        request = request_for(worker, manager, journal)
        with Session(engine) as db:
            assert cameras.api_camera_detections(3, request, db)['tracks'] == []
        with journal._condition:
            worker.step()
            second = worker.state.camera(3)
            assert second['tracking_generation'] != first['tracking_generation']
            assert second['tracks'][0]['visit_id'] is None
        idle(journal)
        worker.step()
        with Session(engine) as db:
            rows = db.query(HumanVisit).all()
            assert len(rows) == 2 and rows[0].id != rows[1].id
            assert worker.state.camera(3)['tracks'][0]['visit_id'] in {row.id for row in rows}


def test_committed_source_or_gallery_change_during_native_discards_entire_outcome():
    for mutation in ('source', 'gallery'):
        with fixture() as (engine, _attendance, clock, worker, manager, _detector, journal):
            process = worker.service.process_batch
            def native(inputs):
                outcomes = process(inputs)
                with immediate_session(engine, publish=True) as db:
                    if mutation == 'source':
                        db.get(AttendanceCamera, 3).source_revision += 1
                    else:
                        db.get(AttendanceRuntime, 1).gallery_revision += 1
                return outcomes
            with patch.object(worker.service, 'process_batch', side_effect=native):
                publish(manager, clock, 3)
                assert worker.step()
            idle(journal)
            assert worker.state.camera(3) is None
            with Session(engine) as db:
                assert db.query(HumanVisit).count() == 0


def test_body_parent_and_live_binding_exist_only_after_atomic_link_ack():
    body = Body()
    with fixture(body=body) as (engine, _attendance, clock, worker, manager, _detector, journal):
        with journal._condition:
            publish(manager, clock, 3)
            worker.step()
            track = worker.state.camera(3)['tracks'][0]
            assert track['visit_id'] is None and track['global_id'] is None
            assert worker.global_manager.snapshot() == []
            assert body.calls == [1]
        idle(journal)
        worker.step()
        track = worker.state.camera(3)['tracks'][0]
        with Session(engine) as db:
            row = db.query(HumanVisit).one()
            assert track['visit_id'] == row.id and track['global_id'] == row.global_id
            assert db.get(GlobalPerson, row.global_id) is not None
            assert row.employee_id is None
        first_id, first_generation = track['global_id'], worker.state.camera(3)['tracking_generation']
        clock.advance(.2)
        manager.get(3).buffer.put(np.full_like(FRAME, 255), stream_id='different-person', source_revision=1,
                                  received_at=clock.utc, received_mono=clock.mono)
        with journal._condition:
            worker.step()
            assert worker.state.camera(3)['tracks'][0]['global_id'] is None
        idle(journal)
        worker.step()
        second = worker.state.camera(3)
        assert second['tracking_generation'] != first_generation
        assert second['tracks'][0]['global_id'] != first_id


def test_appearance_mutation_during_native_cannot_create_an_old_source_parent():
    body = Body()
    with fixture(body=body) as (engine, _attendance, clock, worker, manager, _detector, journal):
        def mutate():
            with immediate_session(engine, publish=True) as db:
                db.get(AttendanceCamera, 3).source_revision += 1
        body.mutate = mutate
        publish(manager, clock, 3)
        worker.step()
        idle(journal)
        with Session(engine) as db:
            assert db.query(HumanVisit).count() == 1
            assert db.query(GlobalPerson).count() == 0
        assert worker.global_manager.snapshot() == []


def test_one_due_body_chunk_is_bounded_and_coalesced_across_many_cameras():
    body = Body()
    with fixture(body=body) as (engine, _attendance, clock, worker, manager, _detector, journal):
        for camera_id in range(4, 23):
            add_camera(engine, manager, camera_id)
        for camera_id in range(3, 23):
            publish(manager, clock, camera_id)
        for _ in range(10):
            worker.step()
        idle(journal)
        worker.step()
        assert sum(body.calls) == 20 and max(body.calls) <= 16
        assert len(worker._appearance_pending) == 0
        with Session(engine) as db:
            assert db.query(HumanVisit).count() == 20
            assert all(row.global_id is not None for row in db.query(HumanVisit))


def test_saturated_generation_gap_recovers_once_before_new_tracking_admission():
    with fixture(journal_options={'lifecycle_capacity': 1}) as (
            engine, attendance, clock, worker, manager, detector, journal):
        def two_people(frames):
            return [[(np.array([10., 10., 100., 200.]), .99),
                     (np.array([300., 100., 400., 300.]), .99)] for _ in frames]
        with patch.object(detector, 'detect_batch', side_effect=two_people), journal._condition:
            publish(manager, clock, 3)
            worker.step()
            generation = worker._journal_generations[3]
            assert journal.health_snapshot()['state'] == 'JOURNAL_DEGRADED'
            assert any('JOURNAL_GAP' in fault.reason_codes for fault in attendance.health.pending_faults())
        idle(journal)
        worker.step()
        assert 3 not in worker._journal_generations
        with Session(engine) as db:
            assert db.query(HumanJournalGap).one().tracking_generation == generation
            assert db.query(HumanVisit).one().end_reason == 'JOURNAL_GAP'
        clock.advance(.2)
        publish(manager, clock, 3)
        worker.step()
        assert worker._journal_generations[3] != generation


def test_monitor_purges_expired_visit_images_even_without_live_ai_worker():
    with fixture() as (engine, attendance, clock, worker, manager, _detector, journal):
        publish(manager, clock, 3)
        worker.step()
        idle(journal)
        worker.flush('AI_UNAVAILABLE')
        idle(journal)
        clock.advance(31 * 86400)
        monitor = AttendanceMonitor(attendance, journal=journal, now=clock.utcnow, monotonic=clock.monotonic)
        monitor.tick()
        idle(journal)
        with Session(engine) as db:
            row = db.query(HumanVisit).one()
            assert row.snapshot is None and row.images_purged_at is not None
            assert row.status == 'CLOSED' and row.camera_id == 3


def test_completion_age_includes_journal_capture_and_publication_work():
    with fixture() as (_engine, _attendance, clock, worker, manager, _detector, journal):
        observe = journal.observe
        def slow_capture(*args):
            result = observe(*args)
            clock.advance(.35)
            return result
        with patch.object(journal, 'observe', side_effect=slow_capture):
            publish(manager, clock, 3)
            worker.step()
        metric = worker.inference_snapshot()[3]
        assert abs(metric['completion_age_seconds'] - .35) < 1e-8
        assert metric['stage_times_ms']['journal'] >= 350. - 1e-6


def test_gallery_or_calibration_commit_hides_cached_identities_before_next_worker_step():
    for mutation in ('gallery', 'calibration'):
        with fixture() as (engine, _attendance, clock, worker, manager, _detector, journal):
            publish(manager, clock, 3)
            worker.step()
            request = request_for(worker, manager, journal)
            with Session(engine) as db:
                assert cameras.api_camera_detections(3, request, db)['tracks']
            with immediate_session(engine, publish=True) as db:
                if mutation == 'gallery':
                    db.get(AttendanceRuntime, 1).gallery_revision += 1
                else:
                    db.get(AttendanceCamera, 3).revision += 1
            with Session(engine) as db:
                assert cameras.api_camera_detections(3, request, db)['tracks'] == []


def test_crowded_camera_cannot_starve_another_camera_at_crop_capacity():
    body = Body()
    with fixture(body=body) as (engine, _attendance, clock, worker, manager, detector, journal):
        add_camera(engine, manager, 4)
        def crowded(frames):
            result = []
            for frame in frames:
                boxes = [(np.array([(i%14)*40., (i//14)*35., (i%14)*40.+30., (i//14)*35.+60.]), .99)
                         for i in range(140)] if frame.mean() < 127 else [(np.array([10.,10.,80.,100.]), .99)]
                result.append(boxes)
            return result
        with patch.object(detector, 'detect_batch', side_effect=crowded):
            publish(manager, clock, 3)
            manager.get(4).buffer.put(np.full_like(FRAME, 255), stream_id='quiet-camera', source_revision=1,
                                      received_at=clock.utc, received_mono=clock.mono)
            worker.step()
            assert len(worker._appearance_pending) <= 128 and max(body.calls) <= 16
            idle(journal)
            # At most one more bounded chunk before the quiet camera's first
            # acknowledgement; not 140 crowded-camera embeddings first.
            worker.step()
            idle(journal)
            with Session(engine) as db:
                quiet = db.query(HumanVisit).filter_by(camera_id=4).one()
                assert quiet.global_id is not None
            assert sum(body.calls) <= 32


def test_costed_optional_body_chunk_finishes_before_the_next_live_deadline():
    class CostedBody(Body):
        def embed(self, crops):
            due = min(worker.scheduler.deadlines[key] for key in worker._last_received)
            result = super().embed(crops)
            clock.advance(.02 * len(crops))
            finished.append((clock.mono, due))
            return result
    body, finished = CostedBody(), []
    with fixture(body=body) as (_engine, _attendance, clock, worker, manager, detector, _journal):
        for camera_id in range(4, 21):
            add_camera(_engine, manager, camera_id)
        def crowded(frames):
            return [[(np.array([10.+40*i, 10., 40.+40*i, 100.]), .99) for i in range(8)]
                    for _ in frames]
        with patch.object(detector, 'detect_batch', side_effect=crowded):
            for camera_id in range(1, 21):
                publish(manager, clock, camera_id)
            for _ in range(8):
                worker.step()
        assert finished and all(end <= due + 1e-9 for end, due in finished), finished
        assert len(worker._appearance_pending) <= 128


def test_observed_expensive_body_work_is_deferred_without_losing_the_pending_crop():
    class CostedBody(Body):
        def embed(self, crops):
            result = super().embed(crops)
            clock.advance(.12 * len(crops))
            return result
    body = CostedBody()
    with fixture(body=body) as (_engine, _attendance, clock, worker, manager, detector, _journal):
        def crowded(frames):
            return [[(np.array([10.+40*i, 10., 40.+40*i, 100.]), .99) for i in range(8)]
                    for _ in frames]
        with patch.object(detector, 'detect_batch', side_effect=crowded):
            publish(manager, clock, 3)
            worker.step()
            assert clock.mono <= worker.scheduler.deadlines[3]
            assert body.calls == [1]
            worker.step()
            assert body.calls == [1] and worker._appearance_pending


def test_metadata_stall_rechecks_budget_and_keeps_unattempted_body_crops():
    body = Body()
    with fixture(body=body) as (_engine, _attendance, clock, worker, manager, detector, _journal):
        def crowded(frames):
            return [[(np.array([10.+40*i, 10., 40.+40*i, 100.]), .99) for i in range(8)]
                    for _ in frames]
        with patch.object(detector, 'detect_batch', side_effect=crowded):
            publish(manager, clock, 3)
            worker.step()
        pending = set(worker._appearance_pending)
        before = list(body.calls)
        read = worker._read_configurations
        def delayed_read():
            result = read()
            clock.advance(.2)
            return result
        with patch.object(worker, '_read_configurations', side_effect=delayed_read):
            worker._process_appearance()
        assert body.calls == before and pending and set(worker._appearance_pending) == pending


def test_appearance_budget_learns_matching_and_publication_cost_not_only_embedding():
    body = Body()
    with fixture(body=body) as (_engine, _attendance, clock, worker, manager, detector, _journal):
        process = worker.global_manager.process_embeddings
        def costly_matching(items):
            result = process(items)
            clock.advance(.12 * len(items))
            return result
        def crowded(frames):
            return [[(np.array([10.+40*i, 10., 40.+40*i, 100.]), .99) for i in range(8)]
                    for _ in frames]
        with patch.object(detector, 'detect_batch', side_effect=crowded), patch.object(
                worker.global_manager, 'process_embeddings', side_effect=costly_matching):
            publish(manager, clock, 3)
            worker.step()
            assert body.calls == [1] and clock.mono <= worker.scheduler.deadlines[3]
            worker.step()
            assert body.calls == [1] and worker._appearance_pending


class FailingLiveFaces(Faces):
    """Only native face sessions fail; gallery, tracking and publication stay real."""
    def __init__(self):
        super().__init__()
        self.failure = None
        self.failed_detection_calls = 0
        self.failed_embedding_calls = 0

    def providers(self):
        return {"detection": ["CUDAExecutionProvider"], "recognition": ["CUDAExecutionProvider"]}

    def detect_faces(self, image):
        if self.failure == "detect":
            self.failed_detection_calls += 1
            # Leave one camera's selected face pending before another camera
            # throws. No partial result may retain stale shared-session identity.
            if self.failed_detection_calls >= 2:
                raise RuntimeError("injected YuNet session failure")
        return super().detect_faces(image)

    def embed_aligned(self, crops):
        if self.failure in ("embed", "shape"):
            self.failed_embedding_calls += 1
            if self.failure == "shape":
                return np.empty((0, 512), np.float32)
            raise RuntimeError("injected AuraFace session failure")
        return super().embed_aligned(crops)


def face_failure_preserves_worker_journal(stage, *, warmup):
    with fixture() as (engine, _attendance, clock, worker, manager, detector, journal):
        faces = FailingLiveFaces()
        with immediate_session(engine) as db:
            sample = EmployeeFace(employee_id=1, embedding=vector().tobytes())
            db.add(sample)
            db.flush()
            db.add(FaceModelMetadata(face_id=sample.id, model_id=MODEL_ID,
                                     alignment_id="arcface-5pt-112-v1"))
        with Session(engine) as db:
            worker.service.store.refresh(db)
        worker.service.faces = faces
        detector.y = .45  # A 200px-high crop with a real 100px synthetic face.

        def submit(camera_ids, *, advance=0., y=.45):
            clock.advance(advance)
            detector.y = y
            for camera_id in camera_ids:
                publish(manager, clock, camera_id)
            assert worker.step()
            idle(journal)
            # Consume commit acknowledgements without manufacturing another frame.
            worker.step()

        initial_cameras = (1, 2, 3)
        for index in range(warmup):
            submit(initial_cameras, advance=.5 if index else 0.)
        previous = {}
        if warmup:
            for camera_id in initial_cameras:
                view = worker.state.camera(camera_id)
                assert view["people_count"] == 1
                assert view["tracks"][0]["employee_id"] == (1 if warmup == 3 else None)
                previous[camera_id] = (view["tracking_generation"], view["tracks"][0]["visit_id"])
                assert previous[camera_id][1] is not None
        # New first-hit camera shares the failing batch with old confirmations/
        # candidates. The ordinary confirmed camera is NOT due for its 2s recheck.
        add_camera(engine, manager, 4)
        all_cameras = (1, 2, 3, 4)
        faces.failure = stage
        submit(all_cameras, advance=.3 if warmup else 0.)
        failed_at = clock.utc
        baseline = {}
        for camera_id in all_cameras:
            view = worker.state.camera(camera_id)
            assert view["people_count"] == 1, (
                stage, camera_id, "optional face failure discarded a real person", view)
            track = view["tracks"][0]
            assert track["employee_id"] is None and track["employee_name"] is None
            assert track["state"] == "unknown"
            assert "FACE_MODEL_UNAVAILABLE" in track["identity_reason_codes"]
            assert track["visit_id"] is not None
            baseline[camera_id] = (view["tracking_generation"], track["visit_id"])
            if camera_id in previous:
                assert baseline[camera_id] == previous[camera_id], "Face failure reset person continuity"
        assert "FACE_MODEL_UNAVAILABLE" in worker.health.snapshot()["technical_reasons"]
        measurements = worker.inference_snapshot()
        assert all("AI_UNAVAILABLE" not in measurements[camera_id]["reason_codes"]
                   and measurements[camera_id]["processed_fps_10s"] > 0
                   for camera_id in all_cameras)
        with Session(engine) as db:
            visits = db.query(HumanVisit).all()
            assert len(visits) == 4
            assert all(row.employee_id is None and row.ended_at is None for row in visits)
            first = db.query(HumanVisit).filter_by(camera_id=4).one()
            assert first.status == "PROVISIONAL" and first.last_seen_at == failed_at
            assert db.query(HumanVisitEvent).filter_by(type="CLOSED").count() == 0

        # A fast subsequent analyzed attendance frame must not hammer either
        # failed native stage, nor revive a cached ordinary confirmation.
        attempts = (faces.failed_detection_calls, faces.failed_embedding_calls)
        submit(all_cameras, advance=.1)
        assert (faces.failed_detection_calls, faces.failed_embedding_calls) == attempts
        for camera_id in (1, 2):
            assert worker.state.camera(camera_id)["tracks"][0]["employee_id"] is None

        if warmup == 3:
            # The failure precedes the line while the previous confirmation is
            # still fresh. Merely dropping fresh_face_evidence would let doorway
            # reuse that stale proof. Complete and persist the real crossing.
            submit(all_cameras, advance=.2, y=.55)
            for _ in range(8):
                submit(all_cameras, advance=.3, y=.65)
            with Session(engine) as db:
                crossings = db.query(AttendanceEvent).filter_by(camera_id=1, direction="ENTRY").all()
                assert crossings, "Anonymous person geometry disappeared with the face session"
                assert all(row.status == "REVIEW" and row.raw["would_auto_accept"] is False
                           for row in crossings), "Failed face work credited stale crossing proof"
                assert db.query(WorkSession).count() == 0
                visits = db.query(HumanVisit).all()
                assert len(visits) == 4
                assert all(row.employee_id is None and row.ended_at is None
                           and row.last_seen_at > failed_at for row in visits)

        faces.failure = None
        recovery_y = .65 if warmup == 3 else .45
        # Three NEW votes are needed even when two votes or a confirmation existed
        # before failure. State + durable identity are the assertions, not counters.
        for observation in range(1, 4):
            submit(all_cameras, advance=.5, y=recovery_y)
            for camera_id in all_cameras:
                view = worker.state.camera(camera_id)
                track = view["tracks"][0]
                assert (view["tracking_generation"], track["visit_id"]) == baseline[camera_id]
                assert track["employee_id"] == (1 if observation == 3 else None)
                assert "FACE_MODEL_UNAVAILABLE" not in track["identity_reason_codes"]
            with Session(engine) as db:
                visits = db.query(HumanVisit).all()
                assert len(visits) == 4
                assert all(row.ended_at is None for row in visits)
                # A previously conflicted durable visit stays conservative even
                # when live recognition recovers; do not erase its history here.
                newly_unknown = [row for row in visits if warmup != 3 or row.camera_id == 4]
                assert all(row.employee_id == (1 if observation == 3 else None)
                           for row in newly_unknown)
                if observation < 3:
                    assert all(row.employee_id is None for row in visits)
        assert "FACE_MODEL_UNAVAILABLE" not in worker.health.snapshot()["technical_reasons"]
        assert journal.health_snapshot()["state"] == "OK"


def test_face_detection_failure_preserves_confirmed_batch_visits_and_safe_crossing():
    face_failure_preserves_worker_journal("detect", warmup=3)


def test_face_embedding_failure_preserves_confirmed_batch_visits_and_safe_crossing():
    face_failure_preserves_worker_journal("embed", warmup=3)


def test_face_embedding_contract_failure_preserves_confirmed_batch_visits_and_safe_crossing():
    face_failure_preserves_worker_journal("shape", warmup=3)


def test_face_detection_failure_clears_candidate_votes_before_recovery():
    face_failure_preserves_worker_journal("detect", warmup=2)


def test_face_embedding_failure_clears_candidate_votes_before_recovery():
    face_failure_preserves_worker_journal("embed", warmup=2)


def test_face_detection_failure_journals_first_observations_without_prior_identity():
    face_failure_preserves_worker_journal("detect", warmup=0)


def test_face_embedding_failure_journals_first_observations_without_prior_identity():
    face_failure_preserves_worker_journal("embed", warmup=0)


def test_person_detector_failure_still_closes_continuity_and_backs_off_without_new_visits():
    with fixture() as (engine, _attendance, clock, worker, manager, detector, journal):
        worker.service.faces = None
        publish(manager, clock, 3)
        assert worker.step()
        idle(journal)
        worker.step()
        before = worker.state.camera(3)
        visit_id = before["tracks"][0]["visit_id"]
        assert visit_id is not None
        last_seen = clock.utc
        clock.advance(.2)
        with patch.object(detector, "detect_batch", side_effect=RuntimeError("person model failed")):
            publish(manager, clock, 3)
            assert worker.step()
        idle(journal)
        assert worker.state.camera(3)["people_count"] == 0
        assert "AI_UNAVAILABLE" in worker.inference_snapshot()[3]["reason_codes"]
        with Session(engine) as db:
            row = db.get(HumanVisit, visit_id)
            assert row.status == "CLOSED" and row.end_reason == "AI_UNAVAILABLE"
            assert row.last_seen_at == row.ended_at == last_seen
        clock.advance(.2)
        publish(manager, clock, 3)
        assert not worker.step(), "Whole person-detector failures must retain bounded backoff"
        with Session(engine) as db:
            assert db.query(HumanVisit).count() == 1
        clock.advance(.81)
        publish(manager, clock, 3)
        assert worker.step()
        idle(journal)
        worker.step()
        recovered = worker.state.camera(3)
        assert recovered["people_count"] == 1
        assert recovered["tracking_generation"] != before["tracking_generation"]
        assert recovered["tracks"][0]["visit_id"] != visit_id
        with Session(engine) as db:
            assert db.query(HumanVisit).count() == 2


if __name__ == '__main__':
    import traceback
    cases = [value for name, value in list(globals().items()) if name.startswith('test_') and callable(value)]
    failed = 0
    for case in cases:
        try:
            case(); print('PASS', case.__name__)
        except Exception:
            failed += 1; traceback.print_exc()
    print(f'{len(cases)-failed}/{len(cases)} worker journal cases passed')
    raise SystemExit(bool(failed))
