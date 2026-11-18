"""Operator-reviewed crops: exact evidence, real transactions, and isolated failure paths."""
from contextlib import contextmanager
from datetime import timedelta, timezone
from hashlib import sha256
from threading import BoundedSemaphore
import traceback
from uuid import uuid4

import cv2
import numpy as np
from fastapi import HTTPException
from pydantic import ValidationError
from sqlalchemy import event
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from test_attendance_recognition import ALIGNMENT_ID, MODEL_ID, STAMP, face
from test_recognition_learning_gallery import BODY_MODEL, VISIT, add_example, direction, examples_db
from app.database.database import immediate_session
from app.database.models import (
    AttendanceRuntime, Employee, EmployeeFace, HumanVisit, HumanVisitIdentityRevision,
    HumanVisitPhotoMetadata, OperatorRequest, PresenceSegment, RecognitionExample,
    RecognitionFeedback, RecognitionLearningRuntime,
)
from app.services.visit_identity_service import IdentityInput, RestoreIdentityInput, VisitIdentityService

TESTS = []


def test(fn):
    TESTS.append(fn)
    return fn


def make_upload(width=160, height=120):
    image = np.zeros((height, width, 3), np.uint8)
    image[:, :width // 2, 1] = 230
    image[:, width // 2:, 2] = 230
    ok, result = cv2.imencode('.png', image)
    assert ok
    return ('source.png', result.tobytes())


def saved_photo(engine, *, captured=STAMP, provenance=True):
    _, content = make_upload()
    frame = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
    ok, photo = cv2.imencode('.jpg', frame)
    assert ok
    photo = photo.tobytes()
    digest = sha256(photo).hexdigest()
    with immediate_session(engine) as db:
        visit = db.get(HumanVisit, VISIT)
        visit.snapshot, visit.face_thumbnail = photo, None
        if provenance:
            db.add(HumanVisitPhotoMetadata(visit_id=VISIT, sha256=digest, captured_at=captured,
                                           frame_seq=55, native_width=160, native_height=120))
    return dict(source='PHOTO_CROP', crop=(.5, .25, 1., .75), photo_sha256=digest)


class Body:
    model_id = BODY_MODEL

    def __init__(self):
        self.images = []
        self.forward = None
        self.output = direction()

    def embed(self, images):
        self.images.extend(images)
        if self.forward:
            self.forward()
        return np.stack([self.output for _ in images])


class Faces:
    model_id = MODEL_ID
    alignment_id = ALIGNMENT_ID

    def __init__(self, count=0):
        self.count = count
        self.images = []

    def detect(self, crop, max_num=0):
        self.images.append(crop)
        return [face() for _ in range(self.count)]


@contextmanager
def service_db(*, faces=None):
    from app.ai.face_recognition import EmbeddingStore
    from app.ai.recognition_learning import RecognitionLearningStore
    from app.services.recognition_learning_service import RecognitionLearningService
    with examples_db() as engine:
        slot = BoundedSemaphore(1)
        clock = [STAMP]
        body = Body()
        learning = RecognitionLearningStore(engine, BODY_MODEL)
        service = RecognitionLearningService(engine, body, faces, MODEL_ID if faces else None, slot,
                                             store=EmbeddingStore(MODEL_ID) if faces else None,
                                             learning_store=learning, now=lambda: clock[0])
        try:
            yield engine, service, body, clock, slot
        finally:
            service.close()


def analyze(service, source, upload=None):
    from app.services.recognition_learning_service import LearningEvidenceInput
    return service.analyze(VISIT, LearningEvidenceInput(**source), upload)


def decision(service, source, *, upload=None, owner=1, action='ACCEPT', revision=0, analysis=None, **changes):
    from app.services.recognition_learning_service import LearningDecisionInput
    result = analysis if analysis is not None else analyze(service, source, upload)
    values = dict(source, actor='Operator', reason='Direct evidence', request_id=str(uuid4()),
                  action=action, employee_id=owner, person_confirmed=True,
                  expected_employee_revision=revision, expected_identity_revision=result['identity_revision'],
                  expected_learning_revision=result['learning_revision'], expected_gallery_revision=result['gallery_revision'],
                  expected_image_sha256=result['evidence']['image_sha256'], expected_body_model_id=result['body_model_id'],
                  expected_face_model_id=result['face_model_id'])
    return LearningDecisionInput(**(values | changes))


def rejected(fn, status, code):
    try:
        fn()
    except HTTPException as error:
        assert error.status_code == status, error.detail
        assert error.detail['code'] == code, error.detail
    else:
        raise AssertionError(f'Expected {status} {code}')


def feedback_counts(engine):
    with Session(engine) as db:
        return tuple(db.query(table).count() for table in (RecognitionFeedback, RecognitionExample,
                                                          HumanVisitIdentityRevision, EmployeeFace, OperatorRequest,
                                                          PresenceSegment))


@test
def confirmed_no_face_visit_uses_only_selected_native_pixels_and_replays_after_expiry():
    faces = Faces()
    with service_db(faces=faces) as (engine, service, body, clock, _):
        source = saved_photo(engine)
        scan = analyze(service, source)
        assert scan['face_status'] == 'NO_FACE'
        assert scan['candidates'] == {'face': [], 'body': []}
        assert feedback_counts(engine) == (0, 0, 0, 0, 0, 0)
        assert body.images[0].shape == (60, 80, 3) == faces.images[0].shape
        for image in (body.images[0], faces.images[0]):
            assert np.median(image[..., 2]) > 210 and np.median(image[..., 1]) < 20
        payload = decision(service, source, analysis=scan)
        saved = service.decide(VISIT, payload)
        assert saved['identity']['employee_id'] == 1
        assert saved['historical_hours_changed'] is False
        assert saved['face_status'] == 'INELIGIBLE'
        assert saved['gallery_revision'] == scan['gallery_revision']
        assert feedback_counts(engine) == (1, 1, 1, 0, 1, 0)
        with Session(engine) as db:
            receipt = db.get(RecognitionFeedback, str(payload.request_id))
            assert receipt.captured_at == STAMP
            assert receipt.source_sha256 == source['photo_sha256']
            assert receipt.image_sha256 == sha256(receipt.image_jpeg).hexdigest()
            assert db.query(RecognitionExample).one().modality == 'BODY'
        clock[0] += timedelta(days=3)
        service.body_engine = None
        assert service.decide(VISIT, payload) == saved
        conflicting = payload.model_copy(update={'reason': 'Different payload for this receipt'})
        try:
            service.decide(VISIT, conflicting)
        except HTTPException as error:
            assert error.status_code == 409
        else:
            raise AssertionError('the same UUID accepted different evidence')
        assert feedback_counts(engine) == (1, 1, 1, 0, 1, 0)


@test
def uploads_never_change_visit_identity_and_reference_survives_identity_undo():
    from app.ai.recognition_learning import match_body
    from app.services.recognition_learning_service import LearningRevokeInput
    upload = make_upload()
    source = dict(source='UPLOAD', crop=[0., 0., 1., 1.], captured_at=STAMP.replace(tzinfo=timezone.utc))
    with service_db() as (engine, service, _, _, _):
        learning = decision(service, source, upload=upload, owner=2)
        result = service.decide(VISIT, learning, upload)
        assert result['identity'] is None
        assert feedback_counts(engine) == (1, 1, 0, 0, 1, 0)
        identity = VisitIdentityService(engine, now=lambda: STAMP)
        fields = dict(actor='Operator', reason='Independent decision')
        identity.set_identity(VISIT, IdentityInput(**fields, expected_revision=0, mode='EMPLOYEE',
                                                   employee_id=1, request_id=uuid4()))
        rejection = decision(service, source | {'crop': [.5, .0, 1., 1.]}, upload=upload, owner=1, action='REJECT')
        assert service.decide(VISIT, rejection, upload)['identity'] is None
        restored = identity.restore(VISIT, RestoreIdentityInput(**fields, expected_revision=1,
                                                               restore_revision=0, request_id=uuid4()))
        assert restored['employee_id'] is None and restored['identity_revision'] == 2
        from app.ai.recognition_learning import RecognitionLearningStore
        snapshot = RecognitionLearningStore(engine, BODY_MODEL).refresh()
        assert match_body(snapshot, direction(), observed_at=STAMP).employee_id == 2
        body_result = service.revoke(result['feedback_id'], LearningRevokeInput(**fields, request_id=uuid4(),
                                                                                 expected_identity_revision=None))
        assert body_result['identity'] is None
        assert feedback_counts(engine)[2] == 2 and feedback_counts(engine)[-1] == 0


@test
def crop_floor_low_detail_multiple_faces_and_body_failure_are_independent():
    source = dict(source='UPLOAD', crop=[0., 0., 1., 1.])
    faces = Faces(2)
    with service_db(faces=faces) as (engine, service, body, _, slot):
        scan = analyze(service, source, make_upload(16, 32))
        assert scan['face_status'] == 'MULTIPLE_FACES' and scan['evidence']['warnings']
        rejected(lambda: analyze(service, source, make_upload(15, 32)), 422, 'PERSON_CROP_TOO_SMALL')
        body.output = np.full(512, np.nan, np.float32)
        rejected(lambda: analyze(service, source, make_upload()), 503, 'BODY_MODEL_UNAVAILABLE')
        with slot:
            rejected(lambda: analyze(service, source, make_upload()), 503, 'ENROLLMENT_BUSY')
        assert feedback_counts(engine) == (0, 0, 0, 0, 0, 0)
        result = VisitIdentityService(engine).set_identity(VISIT, IdentityInput(actor='Operator', reason='Observed',
            request_id=uuid4(), expected_revision=0, employee_id=1, mode='EMPLOYEE'))
        assert result['employee_id'] == 1


@test
def undated_legacy_photo_and_recorded_upload_dates_never_invent_a_today_anchor():
    from app.ai.recognition_learning import RecognitionLearningStore, match_body, rank_body
    with service_db() as (engine, service, _, _, _):
        source = saved_photo(engine, provenance=False)
        payload = decision(service, source)
        assert analyze(service, source)['evidence']['captured_at'] is None
        assert service.decide(VISIT, payload)['identity']['employee_id'] == 1
        store = RecognitionLearningStore(engine, BODY_MODEL)
        snapshot = store.refresh()
        assert rank_body(snapshot, direction(), observed_at=STAMP, automatic=False)[0].employee_id == 1
        assert match_body(snapshot, direction(), observed_at=STAMP) is None
        upload = make_upload()
        old_date = (STAMP - timedelta(days=1)).replace(tzinfo=timezone.utc)
        fields = dict(source='UPLOAD', crop=[0., 0., .45, 1.], captured_at=old_date)
        result = service.decide(VISIT, decision(service, fields, upload=upload, owner=2), upload)
        assert result['identity'] is None
        assert match_body(store.refresh(), direction(), observed_at=STAMP) is None
        future = fields | {'captured_at': (STAMP + timedelta(seconds=1)).replace(tzinfo=timezone.utc)}
        try:
            analyze(service, future, upload)
        except HTTPException as error:
            assert error.status_code == 422
        else:
            raise AssertionError('future upload date became observation evidence')


@test
def incompatible_face_owner_or_face_cap_does_not_block_body_feedback():
    from app.database.models import FaceModelMetadata
    for conflict in (True, False):
        with service_db(faces=Faces(1)) as (engine, service, _, _, _):
            with immediate_session(engine, publish=True) as db:
                for _ in range(1 if conflict else 15):
                    sample = EmployeeFace(employee_id=2 if conflict else 1, embedding=direction().tobytes())
                    db.add(sample)
                    db.flush()
                    db.add(FaceModelMetadata(face_id=sample.id, model_id=MODEL_ID, alignment_id=ALIGNMENT_ID))
                db.get(AttendanceRuntime, 1).gallery_revision += 1
            upload = make_upload()
            source = dict(source='UPLOAD', crop=[0., 0., 1., 1.])
            payload = decision(service, source, upload=upload)
            result = service.decide(VISIT, payload, upload)
            assert result['face_status'] == ('CONFLICT' if conflict else 'LIMIT')
            assert result['identity'] is None and result['gallery_revision'] == payload.expected_gallery_revision
            assert feedback_counts(engine) == (1, 1, 0, 1 if conflict else 15, 1, 0)


@test
def analysis_becomes_stale_on_replacement_and_write_rechecks_revision():
    with service_db() as (engine, service, body, _, _):
        source = saved_photo(engine)
        data = decision(service, source)
        # Simulate a committed gallery revision arriving during the native call.
        def advance():
            with immediate_session(engine, publish=True) as db:
                db.get(RecognitionLearningRuntime, 1).revision += 1
        body.forward = advance
        try:
            service.decide(VISIT, data)
        except HTTPException as error:
            assert error.status_code == 409
        else:
            raise AssertionError('stale revisions were committed')
        assert feedback_counts(engine) == (0, 0, 0, 0, 0, 0)
        body.forward = None
        data = decision(service, source)
        with immediate_session(engine) as db:
            db.get(HumanVisit, VISIT).snapshot = b'different bytes'
        try:
            service.decide(VISIT, data)
        except HTTPException as error:
            assert error.status_code == 409
        else:
            raise AssertionError('changed source was learned')
        assert feedback_counts(engine) == (0, 0, 0, 0, 0, 0)


@test
def changing_face_model_while_waiting_for_writer_transaction_rejects_old_preparation():
    faces = Faces(1)
    with service_db(faces=faces) as (engine, service, _, _, _):
        source = saved_photo(engine) | {'crop': (0., 0., 1., 1.)}
        payload = decision(service, source)

        def disable_face(_conn, _cursor, sql, _params, _ctx, _many):
            if sql == 'BEGIN IMMEDIATE':
                service.face_engine = None

        with event_listener(engine, disable_face):
            rejected(lambda: service.decide(VISIT, payload), 409, 'FACE_MODEL_CHANGED')
        assert feedback_counts(engine) == (0, 0, 0, 0, 0, 0)


@contextmanager
def event_listener(engine, callback):
    event.listen(engine, 'before_cursor_execute', callback)
    try:
        yield
    finally:
        event.remove(engine, 'before_cursor_execute', callback)


@test
def same_crop_rejection_removes_label_but_a_later_label_survives_undo():
    from app.services.recognition_learning_service import LearningRevokeInput
    with service_db(faces=Faces(1)) as (engine, service, _, _, _):
        source = saved_photo(engine) | {'crop': (0., 0., 1., 1.)}
        identity = VisitIdentityService(engine, now=lambda: STAMP)
        audit = dict(actor='Operator', reason='Reviewed visit')
        identity.set_identity(VISIT, IdentityInput(**audit, expected_revision=0, mode='EMPLOYEE',
                                                   employee_id=1, request_id=uuid4()))
        payload = decision(service, source, action='REJECT')
        result = service.decide(VISIT, payload)
        assert result['identity']['employee_id'] is None and result['identity']['identity_revision'] == 2
        assert result['face_status'] == 'NEGATIVE_ADDED' and result['learned_modalities'] == ['BODY', 'FACE']
        assert result['gallery_revision'] == payload.expected_gallery_revision + 1
        with Session(engine) as db:
            example = db.query(RecognitionExample).filter_by(modality='FACE').one()
            assert (example.polarity, example.model_id, example.alignment_id) == ('NEGATIVE', MODEL_ID, ALIGNMENT_ID)
        identity.set_identity(VISIT, IdentityInput(**audit, expected_revision=2, mode='EMPLOYEE',
                                                   employee_id=2, request_id=uuid4()))
        undone = service.revoke(result['feedback_id'], LearningRevokeInput(**audit,
                                request_id=uuid4(), expected_identity_revision=2))
        assert undone['identity_outcome'] == 'KEPT' and undone['identity'] is None
        assert identity.history(VISIT)['items'][0]['employee_id'] == 2
        assert feedback_counts(engine) == (1, 0, 3, 0, 4, 0)


@test
def duplicate_opposite_decisions_and_example_cap_do_not_overwrite_evidence():
    source, upload = dict(source='UPLOAD', crop=[0., 0., 1., 1.]), make_upload()
    with service_db() as (engine, service, _, _, _):
        first = decision(service, source, upload=upload)
        result = service.decide(VISIT, first, upload)
        assert result['identity'] is None
        rejected(lambda: service.decide(VISIT, decision(service, source, upload=upload), upload),
                 409, 'LEARNING_DUPLICATE')
        rejected(lambda: service.decide(VISIT, decision(service, source, upload=upload, action='REJECT'), upload),
                 409, 'LEARNING_CONFLICT')
        rejected(lambda: service.decide(VISIT, decision(service, source, upload=upload, owner=2), upload),
                 409, 'LEARNING_CONFLICT')
        assert feedback_counts(engine) == (1, 1, 0, 0, 1, 0)
        with immediate_session(engine) as db:
            for _ in range(63):
                add_example(db, 1, direction(), model='c' * 64)
        alternative = source | {'crop': [.5, 0., 1., 1.]}
        rejected(lambda: service.decide(VISIT, decision(service, alternative, upload=upload), upload),
                 409, 'LEARNING_LIMIT_REACHED')
        assert feedback_counts(engine) == (64, 64, 0, 0, 1, 0)
        from app.services.recognition_learning_service import LearningRevokeInput
        service.revoke(result['feedback_id'], LearningRevokeInput(request_id=uuid4(), actor='Operator',
                       reason='Make space', expected_identity_revision=None))
        accepted = service.decide(VISIT, decision(service, alternative, upload=upload), upload)
        assert accepted['identity'] is None
        assert feedback_counts(engine) == (65, 64, 0, 0, 3, 0)


@test
def a_full_negative_face_bank_refuses_the_whole_reject_decision():
    with service_db(faces=Faces(1)) as (engine, service, _, _, _):
        source = saved_photo(engine) | {'crop': (0., 0., 1., 1.)}
        identity = VisitIdentityService(engine, now=lambda: STAMP)
        identity.set_identity(VISIT, IdentityInput(actor='Operator', reason='Independent label',
                                       employee_id=1, mode='EMPLOYEE', expected_revision=0, request_id=uuid4()))
        with immediate_session(engine) as db:
            for _ in range(64):
                add_example(db, 1, direction(), model=MODEL_ID, alignment=ALIGNMENT_ID,
                            modality='FACE', negative=True)
        payload = decision(service, source, action='REJECT')
        rejected(lambda: service.decide(VISIT, payload), 409, 'LEARNING_LIMIT_REACHED')
        assert feedback_counts(engine) == (64, 64, 1, 0, 1, 0)


@test
def face_body_identity_and_receipt_commit_or_rollback_together():
    with service_db(faces=Faces(1)) as (engine, service, _, _, _):
        source = saved_photo(engine) | {'crop': (0., 0., 1., 1.)}
        payload = decision(service, source)

        def fail(_conn, _cursor, sql, _parameters, _context, _many):
            if sql.startswith('INSERT INTO operator_requests'):
                raise SQLAlchemyError('test rollback after staging examples')

        with event_listener(engine, fail):
            try:
                service.decide(VISIT, payload)
            except HTTPException as error:
                assert error.status_code == 503
            else:
                raise AssertionError('receipt failure did not roll back the decision')
        assert feedback_counts(engine) == (0, 0, 0, 0, 0, 0)
        with Session(engine) as db:
            assert db.get(RecognitionLearningRuntime, 1).revision == 0
            assert db.get(AttendanceRuntime, 1).gallery_revision == payload.expected_gallery_revision
            assert db.get(Employee, 1).revision == 0
        saved = service.decide(VISIT, payload)
        assert saved['face_status'] == 'ADDED'
        assert saved['gallery_revision'] == payload.expected_gallery_revision + 1
        assert feedback_counts(engine) == (1, 1, 1, 1, 1, 0)
        from app.services.recognition_learning_service import LearningRevokeInput
        def modalities():
            row, = service.history(visit_id=VISIT)['items']
            return {item['modality']: item['model_id'] for item in row['examples']}
        assert modalities() == {'BODY': BODY_MODEL, 'FACE': MODEL_ID}
        service.revoke(saved['feedback_id'], LearningRevokeInput(actor='Operator', reason='Undo',
                       request_id=uuid4(), expected_identity_revision=1))
        assert modalities() == {'BODY': BODY_MODEL, 'FACE': MODEL_ID}
        assert feedback_counts(engine) == (1, 0, 2, 0, 2, 0)


@test
def evidence_and_decision_validate_strict_nonfinite_and_unrepresentable_fields():
    from app.services.recognition_learning_service import LearningEvidenceInput
    valid = dict(source='UPLOAD', crop=(0., 0., 1., 1.))
    for changes in ({'crop': (0., 0., float('nan'), 1.)}, {'crop': (0., 0., 0., 1.)},
                    {'unknown': 1}, {'captured_at': STAMP}, {'photo_sha256': 'a' * 64},
                    {'source': 'PHOTO_CROP', 'photo_sha256': 'a' * 64,
                     'captured_at': STAMP.replace(tzinfo=timezone.utc)}):
        try:
            LearningEvidenceInput(**(valid | changes))
        except ValidationError:
            pass
        else:
            raise AssertionError(f'Invalid evidence accepted: {changes}')
    with service_db() as (_, service, _, _, _):
        upload = make_upload()
        result = analyze(service, valid, upload)
        for changes in ({'employee_id': 2**63}, {'person_confirmed': False},
                        {'expected_learning_revision': -1}, {'expected_employee_revision': True},
                        {'expected_gallery_revision': '7'}):
            try:
                decision(service, valid, upload=upload, analysis=result, **changes)
            except ValidationError:
                pass
            else:
                raise AssertionError(f'Invalid decision accepted: {changes}')


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
