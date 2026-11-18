"""Body revision fences through the real AI worker, journal, recorder and SQLite."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import timedelta
from threading import Event
from types import SimpleNamespace
from unittest.mock import patch
import traceback

import numpy as np
from sqlalchemy.orm import Session

from test_recognition_learning_gallery import BODY_MODEL, STAMP, add_example, direction, examples_db
from attendance_test_support import BufferedReader, Clock, FRAME, Readers
from test_human_journal import idle
from app.ai.employee_appearance import EmployeeAppearanceGallery
from app.ai.recognition_service import RecognitionService
from app.ai.recognition_learning import RecognitionLearningStore
from app.ai.worker import AIWorker, DetectionState
from app.database.database import immediate_session
from app.database.models import AttendanceCamera, Camera, CameraProfile, DoorCrossing, EmployeeFace, PresenceSegment
from app.services.human_journal import HumanJournal
from app.services.presence_recorder import PresenceRecorder
from app.services.recognition_feedback_repository import advance_learning, revoke_feedback_rows

TESTS = []
BOXES = [np.array([32., 24., 176., 264.]), np.array([320., 44., 480., 284.])]


def test(fn):
    TESTS.append(fn)
    return fn


class Detector:
    def __init__(self):
        self.two = False
        self.inside = None
        self.session = SimpleNamespace(get_providers=lambda: ['CPUExecutionProvider'])

    def detect_batch(self, frames):
        if self.inside:
            self.inside()
        return [[(box, .99) for box in (BOXES if self.two else BOXES[:1])] for _ in frames]


class Body:
    model_id = BODY_MODEL

    def __init__(self):
        self.calls = 0
        self.inside = None
        self.session = SimpleNamespace(get_providers=lambda: ['CPUExecutionProvider'])

    def embed(self, images):
        self.calls += len(images)
        if self.inside:
            self.inside()
        return np.stack([direction() for _ in images])


class Replay:
    def __init__(self, engine, *, two=False):
        self.engine = engine
        self.clock = Clock()
        self.clock.utc = STAMP
        self.readers = Readers()
        self.readers.readers = {15: BufferedReader(), 16: BufferedReader()}
        for reader in self.readers.readers.values():
            reader.source_revision = 1
        self.detector, self.body = Detector(), Body()
        self.detector.two = two
        self.presence = PresenceRecorder(engine, now=self.clock.utcnow)
        self.journal = HumanJournal(engine, min_hits=1, now=self.clock.utcnow, monotonic=self.clock.monotonic)
        idle(self.journal)
        self.gallery = EmployeeAppearanceGallery()
        self.admitted = []
        observe = self.presence.observe
        def observed(camera, employee, source, at):
            self.admitted.append((camera, employee, source, at))
            observe(camera, employee, source, at)
        self.presence.observe = observed
        self.worker = AIWorker(self.readers, RecognitionService(self.detector, None, None, min_hits=1),
            self.presence, DetectionState(), journal=self.journal, reid=self.body,
            appearance_gallery=self.gallery, learning_store=RecognitionLearningStore(engine, BODY_MODEL))

    def frames(self, count, cameras=(15,), *, stream='first'):
        for _ in range(count):
            for camera in cameras:
                self.readers.get(camera).buffer.put(FRAME, stream_id=stream, source_revision=1,
                    received_at=self.clock.utc, received_mono=self.clock.mono)
            with patch('app.ai.worker.time.monotonic', side_effect=self.clock.monotonic), \
                    patch('app.ai.worker.utcnow', side_effect=self.clock.utcnow):
                self.worker.step()
                idle(self.journal)
            self.clock.advance(.5)

    def names(self, camera=15):
        state = self.worker.state.camera(camera)
        return [track for track in state['tracks'] if track['employee_id'] is not None] if state else []

    def close(self):
        assert self.journal.close().drained
        self.presence.flush()


@contextmanager
def fixture():
    with examples_db() as engine:
        with immediate_session(engine) as db:
            db.add(Camera(id=16, name='No time recording', nvr_host='unused', username='', password='', channel=16))
            db.flush()
            db.query(Camera).filter(Camera.id.in_((15, 16))).update({'enabled': True})
            for camera in (15, 16):
                db.add(AttendanceCamera(camera_id=camera, enabled=False, source_revision=1))
                db.add(CameraProfile(camera_id=camera, zone='INTERIOR', ai_enabled=True, time_tracking=camera == 15))
            receipt = add_example(db, 1, direction())
            advance_learning(db)
        yield engine, receipt


def revoke(engine, receipt, at):
    with immediate_session(engine, publish=True) as db:
        revoke_feedback_rows(db, [receipt], actor='Operator', reason='Incorrect attribution', at=at)


def commit_while_inside(inside, change, run):
    entered, committed = Event(), Event()
    def barrier():
        entered.set()
        assert committed.wait(5), 'native inference held the publication lock'
    def writer():
        assert entered.wait(5), 'native inference was not reached'
        try:
            change()
        finally:
            committed.set()
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(writer)
        try:
            inside.inside = barrier
            run()
            future.result(timeout=10)
        finally:
            inside.inside = None


@test
def revocation_during_detection_cannot_publish_a_sticky_name_crossing_or_new_time():
    with fixture() as (engine, receipt):
        rig = Replay(engine)
        try:
            rig.frames(20)
            initial = rig.worker.state.camera(15)
            assert [track['employee_id'] for track in rig.names()] == [1]
            assert rig.names()[0]['identity_source'] == 'REID'
            previous = tuple(rig.admitted)
            assert len(previous) >= 2
            def change():
                revoke(engine, receipt, rig.clock.utc)
                crossing = SimpleNamespace(direction='EXIT', confirmed_employee_id=None, crossing_index=1,
                    track_key=(15, initial['stream_id'], initial['tracking_generation'], rig.names()[0]['track_id']),
                    occurred_at=rig.clock.utc)
                rig.worker._record_crossings([crossing])
            commit_while_inside(rig.detector, change, lambda: rig.frames(1))
            assert not rig.names() and tuple(rig.admitted) == previous
            assert rig.worker.state.camera(15)['tracking_generation'] == initial['tracking_generation']
            rig.presence.flush()
            with Session(engine) as db:
                assert db.query(DoorCrossing).count() == 0
                assert db.query(PresenceSegment).one().last_seen_at == previous[-1][-1]
        finally:
            rig.close()


@test
def revocation_or_rejection_during_body_inference_cannot_complete_stale_votes():
    for action in ('revoke', 'reject', 'accept'):
        with fixture() as (engine, receipt):
            rig = Replay(engine)
            try:
                # First eligible vote, before the default 2.5-second second sample.
                rig.frames(2)
                assert not rig.names()
                def change():
                    if action == 'revoke':
                        revoke(engine, receipt, rig.clock.utc)
                    else:
                        with immediate_session(engine, publish=True) as db:
                            add_example(db, 1 if action == 'reject' else 2,
                                        direction() if action == 'reject' else direction(0.),
                                        negative=action == 'reject')
                            advance_learning(db)
                commit_while_inside(rig.body, change, lambda: rig.frames(5))
                assert not rig.names() and not rig.admitted
                if action == 'accept':
                    rig.frames(20)
                    assert rig.names()[0]['employee_id'] == 1
                else:
                    rig.frames(20)
                    assert not rig.names() and not rig.admitted
            finally:
                rig.close()


@test
def restart_daily_rollover_and_nontracking_camera_keep_anchors_and_admissions_separate():
    with fixture() as (engine, _receipt):
        for _ in range(2):
            rig = Replay(engine)
            try:
                rig.frames(20, (15, 16))
                assert rig.names(15)[0]['employee_id'] == rig.names(16)[0]['employee_id'] == 1
                assert {row[:3] for row in rig.admitted} == {(15, 1, 'REID')}
                assert not rig.gallery.samples_for_day(STAMP.date()), 'predicted identities must not teach'
                prior = list(rig.admitted)
                rig.clock.advance(.5, utc_seconds=86400)
                rig.frames(20, (15, 16))
                assert not rig.names(15) and not rig.names(16) and rig.admitted == prior
                with immediate_session(engine, publish=True) as db:
                    add_example(db, 1, direction(), captured=rig.clock.utc)
                    advance_learning(db)
                rig.frames(20)
                assert rig.names()[0]['employee_id'] == 1
                with Session(engine) as db:
                    assert db.query(EmployeeFace).count() == 0
            finally:
                rig.close()
            # Remove tomorrow's reference so a new iteration still starts with just today's anchor.
            from app.database.models import RecognitionFeedback
            with immediate_session(engine, publish=True) as db:
                tomorrow = db.query(RecognitionFeedback.id).filter(RecognitionFeedback.captured_at > STAMP,
                    RecognitionFeedback.revoked_at.is_(None)).all()
                revoke_feedback_rows(db, [row[0] for row in tomorrow], actor='Operator', reason='Next scenario', at=STAMP)


@test
def distinct_tracks_on_the_same_camera_cannot_hold_the_same_body_name():
    with fixture() as (engine, _):
        rig = Replay(engine, two=True)
        try:
            rig.frames(24)
            assert len(rig.worker.state.camera(15)['tracks']) >= 2
            assert len(rig.names()) == 1
            assert {row[:3] for row in rig.admitted} == {(15, 1, 'REID')}
        finally:
            rig.close()


@test
def local_supervision_works_from_production_builder_with_cross_camera_linking_disabled():
    from app.ai.worker import build_ai_worker
    from app.config import settings

    with fixture() as (engine, _):
        rig = Replay(engine)
        detector, body = rig.detector, rig.body
        bundle = SimpleNamespace(bundle_id='bundle', face_model_id=None, metadata={
            'components': {'person': {'files': [{'sha256': 'person'}]}},
        })
        try:
            with patch.object(settings, 'multi_camera_tracking_enabled', False), \
                    patch('app.ai.worker.setup_cuda_dlls'), \
                    patch('app.ai.worker.resolve_providers', return_value=['CPUExecutionProvider']), \
                    patch('app.ai.worker.FaceEngine', side_effect=RuntimeError('fixture without face model')), \
                    patch('app.ai.detector.PersonDetector', return_value=detector), \
                    patch('app.ai.reid.PersonReID', return_value=body):
                rig.worker = build_ai_worker(
                    rig.readers, rig.presence, None, bundle=bundle, session_factory=lambda: Session(engine),
                    journal=rig.journal, learning_store=RecognitionLearningStore(engine, BODY_MODEL))
            assert rig.worker.reid is body and rig.worker.global_manager is None
            assert rig.worker.service.store is None and rig.worker.service.faces is None
            rig.frames(25)
            assert [(track['employee_id'], track['identity_source']) for track in rig.names()] == [(1, 'REID')]
            assert rig.admitted
        finally:
            rig.close()


@test
def loss_of_the_current_body_model_clears_sticky_names_even_without_a_commit():
    with fixture() as (engine, _):
        rig = Replay(engine)
        try:
            rig.frames(20)
            assert rig.names() and rig.admitted
            presence = tuple(rig.admitted)
            rig.worker.reid = None
            rig.frames(1)
            assert not rig.names() and tuple(rig.admitted) == presence
        finally:
            rig.close()


@test
def supervised_body_can_collect_votes_while_the_face_gallery_is_unavailable():
    with fixture() as (engine, _):
        rig = Replay(engine)
        native = rig.worker.service.process_batch
        def gallery_outage(batch):
            outcomes = native(batch)
            for outcome in outcomes:
                for person in outcome.observed_tracks:
                    person.identity_reason_codes = ('GALLERY_UNAVAILABLE',)
            return outcomes
        try:
            with patch.object(rig.worker.service, 'process_batch', side_effect=gallery_outage):
                rig.frames(20)
            assert [(track['employee_id'], track['identity_source']) for track in rig.names()] == [(1, 'REID')]
            assert rig.admitted
        finally:
            rig.close()


@test
def authoritative_face_supersedes_a_different_sticky_body_track_before_attribution():
    with fixture() as (engine, _):
        rig = Replay(engine, two=True)
        try:
            rig.frames(24)
            (body_track,) = rig.names()
            native = rig.worker.service.process_batch
            def confirmed_face(batch):
                outcomes = native(batch)
                for outcome in outcomes:
                    for person in outcome.observed_tracks:
                        if person.track_id != body_track['track_id']:
                            person.employee_id, person.employee_name = 1, 'First'
                            person.confidence = .99
                return outcomes
            before = len(rig.admitted)
            with patch.object(rig.worker.service, 'process_batch', side_effect=confirmed_face):
                rig.frames(1)
            assert len(rig.names()) == 1 and rig.names()[0]['identity_source'] == 'FACE'
            assert rig.names()[0]['track_id'] != body_track['track_id']
            assert [row[2] for row in rig.admitted[before:]] == ['FACE']
        finally:
            rig.close()


@test
def historical_uniform_similarity_blocks_current_day_body_names_and_worked_time():
    with fixture() as (engine, _):
        with immediate_session(engine, publish=True) as db:
            add_example(db, 2, direction(.98), captured=STAMP - timedelta(days=1))
            advance_learning(db)
        rig = Replay(engine, two=True)
        try:
            rig.frames(24)
            assert len(rig.worker.state.camera(15)['tracks']) >= 2
            assert not rig.names() and not rig.admitted
            with Session(engine) as db:
                assert db.query(PresenceSegment).count() == 0
        finally:
            rig.close()


if __name__ == '__main__':
    failed = 0
    for fn in TESTS:
        try:
            fn()
            print('PASS', fn.__name__)
        except Exception:
            failed += 1
            traceback.print_exc()
    print(f'{len(TESTS)-failed}/{len(TESTS)} passed')
    raise SystemExit(bool(failed))
