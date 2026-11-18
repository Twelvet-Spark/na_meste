"""Model-bound feedback and conservative same-day attendance ranking."""
from contextlib import contextmanager
from datetime import timedelta
from hashlib import sha256
import traceback
from uuid import uuid4

import numpy as np

from test_attendance_recognition import ALIGNMENT_ID, MODEL_ID, STAMP, database, detection, load_store, recognition
from sqlalchemy.orm import Session
from fastapi import HTTPException
from app.ai.employee_appearance import EmployeeAppearanceGallery
from app.database.database import immediate_session
from app.database.models import (
    AttendanceRuntime, Camera, Employee, HumanVisit, RecognitionExample,
    RecognitionFeedback, RecognitionLearningRuntime,
)
from app.time_utils import local_date

TESTS = []
BODY_MODEL = 'b' * 64
VISIT = 'a59700da-94fb-442b-970f-4899bb695033'


def test(fn):
    TESTS.append(fn)
    return fn


def direction(cosine=1., axis=1):
    value = np.zeros(512, np.float32)
    value[0], value[axis] = cosine, np.sqrt(1 - cosine ** 2)
    return value


@contextmanager
def examples_db(faces=()):
    with database(faces) as engine:
        with immediate_session(engine) as db:
            db.add(Camera(id=15, name='Test', nvr_host='unused', username='', password='', channel=1))
            db.add(Employee(id=3, name='Inactive', active=False))
            db.flush()
            db.add(HumanVisit(id=VISIT, camera_id=15, tracking_generation='gen', track_id=1,
                              source_revision=1, stream_id='stream', started_at=STAMP,
                              last_seen_at=STAMP, status='ACTIVE', identity_state='UNKNOWN',
                              last_bbox=[0., 0., 1., 1.], max_person_score=.9,
                              image_expires_at=STAMP + timedelta(days=1)))
        yield engine


def add_example(db, owner, vector, *, captured=STAMP, model=BODY_MODEL, negative=False,
                modality='BODY', alignment=None, revoked=False):
    feedback = str(uuid4())
    digest = sha256(feedback.encode()).hexdigest()
    db.add(RecognitionFeedback(id=feedback, visit_id=VISIT, employee_id=owner,
                               source='UPLOAD', action='REJECT' if negative else 'ACCEPT',
                               source_sha256=digest, image_sha256=digest, crop=[0, 0, 1, 1],
                               captured_at=captured, created_at=STAMP, actor='Operator', reason='Review',
                               face_status='NEGATIVE_ADDED' if modality == 'FACE' else 'INELIGIBLE',
                               revoked_at=STAMP if revoked else None, revoked_by='Operator' if revoked else None,
                               revoke_reason='Undo' if revoked else None))
    db.flush()
    db.add(RecognitionExample(feedback_id=feedback, modality=modality, model_id=model,
                              alignment_id=alignment, polarity='NEGATIVE' if negative else 'POSITIVE',
                              embedding=vector.tobytes()))
    return feedback


def load_body(engine):
    from app.ai.recognition_learning import RecognitionLearningStore
    store = RecognitionLearningStore(engine, BODY_MODEL)
    store.refresh()
    return store


@test
def body_ranking_uses_distinct_vectors_and_deterministic_owner_ties():
    from app.ai.recognition_learning import rank_body
    with examples_db() as engine:
        with immediate_session(engine) as db:
            for value in (1., 1., 1., .7):
                add_example(db, 1, direction(value))
            add_example(db, 2, direction(.87))
        snapshot = load_body(engine).snapshot()
        ranked = rank_body(snapshot, direction(), observed_at=STAMP, automatic=False,
                           transient_samples={1: (direction(),)})
        assert [row.employee_id for row in ranked] == [2, 1]
        assert abs(ranked[0].score - .87) < 1e-6
        assert abs(ranked[1].score - .85) < 1e-6
    with examples_db() as engine:
        with immediate_session(engine) as db:
            add_example(db, 2, direction(.91))
            add_example(db, 1, direction(.91))
        ranked = rank_body(load_body(engine).snapshot(), direction(), observed_at=STAMP, automatic=False)
        assert [row.employee_id for row in ranked] == [1, 2]


@test
def old_lookalikes_block_a_same_day_match_but_cannot_win_it():
    from app.ai.recognition_learning import match_body, rank_body
    with examples_db() as engine:
        with immediate_session(engine) as db:
            add_example(db, 1, direction())
            add_example(db, 2, direction(.98), captured=STAMP - timedelta(days=1))
        store = load_body(engine)
        first = store.snapshot()
        assert match_body(first, direction(), observed_at=STAMP) is None
        assert [row.employee_id for row in rank_body(first, direction(), observed_at=STAMP,
                                                     automatic=True)] == [1]
        assert match_body(first, direction(), observed_at=STAMP + timedelta(days=1)) is None
        with immediate_session(engine) as db:
            db.get(Employee, 2).active = False
            db.get(AttendanceRuntime, 1).gallery_revision += 1
        second = store.refresh()
        assert second.gallery_revision == first.gallery_revision + 1
        assert second.revision == first.revision
        assert match_body(second, direction(), observed_at=STAMP).employee_id == 1
        assert match_body(first, direction(), observed_at=STAMP) is None


@test
def body_veto_is_local_and_the_raw_winner_cannot_promote_a_runner_up():
    from app.ai.recognition_learning import match_body, rank_body
    with examples_db() as engine:
        with immediate_session(engine) as db:
            add_example(db, 1, direction())
            add_example(db, 1, direction(.65))
            add_example(db, 2, direction(.82, 2))
            add_example(db, 1, direction(), negative=True)
        snapshot = load_body(engine).snapshot()
        ranked = rank_body(snapshot, direction(), observed_at=STAMP, automatic=False)
        assert ranked[0].employee_id == 1 and ranked[0].rejected
        assert abs(ranked[0].negative_score - 1.) < 1e-6
        assert ranked[1].employee_id == 2 and not ranked[1].rejected
        assert match_body(snapshot, direction(), observed_at=STAMP) is None
        assert match_body(snapshot, direction(.65), observed_at=STAMP).employee_id == 1


@test
def inactive_revoked_wrong_model_bad_vectors_and_wrong_modality_do_not_rank():
    from app.ai.recognition_learning import rank_body
    with examples_db() as engine:
        with immediate_session(engine) as db:
            db.get(RecognitionLearningRuntime, 1).revision = 3
            add_example(db, 1, direction(.9), captured=None)
            add_example(db, 1, direction(), modality='FACE', negative=True, alignment=ALIGNMENT_ID)
            add_example(db, 2, direction(), model=MODEL_ID)
            add_example(db, 2, direction(), revoked=True)
            add_example(db, 2, np.full(512, np.nan, np.float32))
            add_example(db, 3, direction())
        snapshot = load_body(engine).snapshot()
        assert snapshot.revision == 3
        ranked = rank_body(snapshot, direction(), observed_at=STAMP, automatic=False)
        assert [row.employee_id for row in ranked] == [1]
        assert ranked[0].negative_score is None and not ranked[0].rejected
        assert rank_body(snapshot, direction(), observed_at=STAMP, automatic=True) == ()


@test
def duplicate_vector_preserves_every_explicit_anchor_day():
    from app.ai.recognition_learning import match_body, rank_body
    tomorrow = STAMP + timedelta(days=1)
    with examples_db() as engine:
        with immediate_session(engine) as db:
            add_example(db, 1, direction(), captured=STAMP)
            add_example(db, 1, direction(), captured=tomorrow)
            add_example(db, 1, direction(.70), captured=None)
        snapshot = load_body(engine).snapshot()
        for observed in (STAMP, tomorrow):
            assert match_body(snapshot, direction(), observed_at=observed).employee_id == 1
        ranked = rank_body(snapshot, direction(), observed_at=tomorrow, automatic=False)
        assert abs(ranked[0].score - .85) < 1e-6


@test
def face_feedback_veto_never_promotes_runner_or_weakens_enrollment_overlap_check():
    with examples_db([(1, direction()), (2, direction(.91, 2))]) as engine:
        with immediate_session(engine) as db:
            add_example(db, 1, direction(), model=MODEL_ID, modality='FACE',
                        alignment=ALIGNMENT_ID, negative=True)
            add_example(db, 2, direction(), model=MODEL_ID, negative=True)
        store = load_store(engine)
        ranked = store.rank(direction())
        assert [(r.employee_id, r.rejected) for r in ranked] == [(1, True), (2, False)]
        assert store.match(direction()).employee_id is None
        assert store.match(direction(), use_feedback=False).employee_id == 1
    from app.services.employee_service import PreparedFace, check_prepared_faces
    with examples_db([(1, direction()), (2, direction(.80, 2))]) as engine:
        with immediate_session(engine) as db:
            add_example(db, 1, direction(), model=MODEL_ID, modality='FACE',
                        alignment=ALIGNMENT_ID, negative=True)
        with Session(engine) as db:
            sample = PreparedFace('candidate.jpg', direction().tobytes(), b'', MODEL_ID, ALIGNMENT_ID)
            try:
                check_prepared_faces(db, 3, [sample], MODEL_ID)
            except HTTPException as error:
                assert error.status_code == 422 and error.detail['code'] == 'PHOTO_VALIDATION_FAILED'
            else:
                raise AssertionError('operator rejection hid a cross-owner enrollment conflict')


@test
def published_face_rejection_clears_live_confirmation_until_revoked():
    with examples_db([(1, direction())]) as engine:
        service, _, _, store = recognition(engine)
        for seq, mono in enumerate((10., 10.25, 10.5), 1):
            last = detection(service, seq, mono)
        assert last.employee_id == 1
        with immediate_session(engine, publish=True) as db:
            feedback_id = add_example(db, 1, direction(), model=MODEL_ID, modality='FACE',
                                      alignment=ALIGNMENT_ID, negative=True)
            db.get(AttendanceRuntime, 1).gallery_revision += 1
        with Session(engine) as db:
            store.refresh(db)
        assert detection(service, 4, 10.75).employee_id is None
        assert detection(service, 5, 11.).employee_id is None
        assert detection(service, 6, 11.25).employee_id is None
        with immediate_session(engine, publish=True) as db:
            feedback = db.get(RecognitionFeedback, feedback_id)
            feedback.revoked_at, feedback.revoked_by, feedback.revoke_reason = STAMP, 'Operator', 'Undo'
            db.get(AttendanceRuntime, 1).gallery_revision += 1
        with Session(engine) as db:
            store.refresh(db)
        for seq, mono in enumerate((11.5, 11.75, 12.), 7):
            last = detection(service, seq, mono)
        assert last.employee_id == 1


@test
def looking_up_another_day_does_not_roll_transient_face_confirmed_samples():
    gallery = EmployeeAppearanceGallery()
    gallery.learn(1, direction(), STAMP)
    day = local_date(STAMP)
    view = gallery.samples_for_day(day)
    assert len(view[1]) == 1 and not view[1][0].flags.writeable
    assert not gallery.samples_for_day(day - timedelta(days=1))
    assert gallery.samples_for_day(day)[1][0] is view[1][0]
    gallery.clear()
    assert not gallery.samples_for_day(day)
    assert view[1][0][0] == 1.


if __name__ == '__main__':
    failed = 0
    for fn in TESTS:
        try:
            fn()
            print('PASS', fn.__name__)
        except Exception:
            failed += 1
            traceback.print_exc()
    print(f'{len(TESTS) - failed}/{len(TESTS)} passed')
    raise SystemExit(bool(failed))
