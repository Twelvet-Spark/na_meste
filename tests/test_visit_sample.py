"""Isolated synthetic-image transaction tests, not biometric identity accuracy."""
import hashlib
import sys
import threading
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
from fastapi import HTTPException
from sqlalchemy import event
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from app.ai.face_detector import ALIGNMENT_ID
from app.ai.face_recognition import EmbeddingStore
from app.database.database import immediate_session
from app.database.models import AttendanceCamera, AttendanceRuntime, Employee, EmployeeFace, FaceModelMetadata, HumanVisit, OperatorRequest
from app.services.visit_sample_service import VisitSampleService, OPERATION
from attendance_test_support import database

MODEL = "a" * 64
NOW = datetime(2026, 11, 16, 12)


def vector(index=7):
    value = np.zeros(512, np.float32)
    value[index] = 1.
    return value


def jpeg(seed=3):
    image = np.random.default_rng(seed).integers(0, 256, (112, 112, 3), np.uint8)
    ok, encoded = cv2.imencode('.jpg', image)
    assert ok
    return encoded.tobytes()


class Engine:
    model_id = MODEL
    alignment_id = ALIGNMENT_ID
    def __init__(self):
        self.calls = 0
        self.callback = None
        self.value = vector()
    def embed_aligned(self, images):
        self.calls += 1
        assert len(images) == 1 and images[0].shape == (112, 112, 3)
        if self.callback:
            self.callback()
        return np.stack([self.value])


@contextmanager
def fixture():
    with database() as (engine, _attendance):
        face = Engine()
        clock = [NOW]
        semaphore = threading.BoundedSemaphore(1)
        service = VisitSampleService(engine, face, MODEL, semaphore, now=lambda: clock[0])
        visit_id, thumbnail = str(uuid4()), jpeg()
        with immediate_session(engine) as db:
            source = db.get(AttendanceCamera, 1).source_revision
            db.add(HumanVisit(id=visit_id, camera_id=1, tracking_generation=str(uuid4()),
                             source_revision=source, stream_id=str(uuid4()), track_id=1,
                             status='PROVISIONAL', started_at=NOW, last_seen_at=NOW,
                             identity_state='UNKNOWN', last_bbox=[.1, .1, .8, .9], max_person_score=.9,
                             face_thumbnail=thumbnail, image_expires_at=NOW + timedelta(days=1),
                             face_quality={'min_size':112., 'det_score':.99, 'blur':50.,
                                           'aligned':True, 'unambiguous':True,
                                           'sha256':hashlib.sha256(thumbnail).hexdigest()}))
            revision = db.get(Employee, 1).revision
        payload = dict(employee_id=1, expected_revision=revision, request_id=uuid4(),
                       face_sha256=hashlib.sha256(thumbnail).hexdigest())
        yield engine, service, face, clock, visit_id, payload


def state(engine):
    with Session(engine) as db:
        return ([(row.id, row.employee_id, row.embedding, row.thumbnail) for row in db.query(EmployeeFace).order_by(EmployeeFace.id)],
                [(row.face_id,row.model_id,row.alignment_id) for row in db.query(FaceModelMetadata).order_by(FaceModelMetadata.face_id)],
                [(row.id,row.revision,row.updated_at) for row in db.query(Employee).order_by(Employee.id)],
                db.get(AttendanceRuntime,1).gallery_revision,
                [(row.id,row.operation,row.payload_sha256,row.result) for row in db.query(OperatorRequest).order_by(OperatorRequest.id)])


def fails(status, code, action):
    try:
        action()
    except HTTPException as exc:
        assert exc.status_code == status, (exc.status_code, exc.detail)
        if code:
            assert exc.detail.get('code') == code, exc.detail
    else:
        raise AssertionError('mutation unexpectedly succeeded')


def test_append_preserves_existing_samples_and_stamps_only_new_target_space():
    with fixture() as (engine, service, face, _clock, visit_id, payload):
        original = jpeg(4)
        with immediate_session(engine) as db:
            old = EmployeeFace(employee_id=1, embedding=vector(11).tobytes(), thumbnail=original)
            db.add(old); db.flush()
            old_id = old.id
            db.add(FaceModelMetadata(face_id=old_id, model_id='b'*64, alignment_id=ALIGNMENT_ID))
        before = state(engine)
        result = service.append(visit_id, **payload)
        with Session(engine) as db:
            old = db.get(EmployeeFace, old_id)
            assert old.thumbnail == original and old.embedding == vector(11).tobytes()
            assert db.get(FaceModelMetadata, old_id).model_id == 'b'*64
            added = db.get(EmployeeFace, result['face_id'])
            assert added.employee_id == 1 and np.array_equal(np.frombuffer(added.embedding,np.float32),vector())
            provenance = db.get(FaceModelMetadata, added.id)
            assert provenance.model_id == MODEL and provenance.alignment_id == ALIGNMENT_ID
            assert result['faces_count'] == 2 and result['usable_faces_count'] == 1
            assert db.get(Employee,1).revision == payload['expected_revision'] + 1
            assert db.get(AttendanceRuntime,1).gallery_revision == before[3] + 1
            assert db.get(OperatorRequest,str(payload['request_id'])).operation == OPERATION
            assert db.get(HumanVisit,visit_id).employee_id is None
        assert face.calls == 1


def test_frozen_replay_precedes_expiry_model_and_current_employee_revision():
    with fixture() as (engine, service, face, clock, visit_id, payload):
        result = service.append(visit_id, **payload)
        with immediate_session(engine) as db:
            visit = db.get(HumanVisit,visit_id)
            visit.face_thumbnail = None; visit.images_purged_at = NOW
            db.get(Employee,1).revision += 1
        clock[0] += timedelta(days=2)
        service.face_engine = None
        assert service.append(visit_id, **payload) == result and face.calls == 1


def test_request_uuid_payload_change_conflicts_before_any_repreparation():
    with fixture() as (_engine, service, face, _clock, visit_id, payload):
        service.append(visit_id, **payload)
        changed = dict(payload, employee_id=2)
        fails(409, None, lambda: service.append(visit_id, **changed))
        assert face.calls == 1


def test_changed_exact_image_during_embedding_cannot_append_an_unseen_face():
    with fixture() as (engine, service, face, _clock, visit_id, payload):
        before = state(engine)
        def changed():
            with immediate_session(engine) as db:
                visit = db.get(HumanVisit,visit_id)
                visit.face_thumbnail = jpeg(8)
                visit.face_quality = dict(visit.face_quality,sha256=hashlib.sha256(visit.face_thumbnail).hexdigest())
        face.callback = changed
        fails(409,'VISIT_FACE_CHANGED',lambda: service.append(visit_id,**payload))
        assert state(engine) == before


def test_expiry_during_embedding_is_rechecked_inside_the_write_transaction():
    with fixture() as (engine, service, face, clock, visit_id, payload):
        before = state(engine)
        face.callback = lambda: clock.__setitem__(0,NOW+timedelta(days=2))
        fails(410,'VISIT_FACE_EXPIRED',lambda: service.append(visit_id,**payload))
        assert state(engine) == before


def test_employee_revision_race_does_not_append_or_undo_the_other_change():
    with fixture() as (engine, service, face, _clock, visit_id, payload):
        def changed():
            with immediate_session(engine) as db:
                row=db.get(Employee,1); row.revision += 1; row.external_id='other-operator'
        face.callback=changed
        fails(409,None,lambda: service.append(visit_id,**payload))
        with Session(engine) as db:
            assert db.get(Employee,1).external_id == 'other-operator'
            assert db.query(EmployeeFace).count() == 0 and db.query(OperatorRequest).count() == 0


def test_other_active_owner_duplicate_uses_current_target_space():
    with fixture() as (engine, service, _face, _clock, visit_id, payload):
        with immediate_session(engine) as db:
            existing=EmployeeFace(employee_id=2,embedding=vector().tobytes()); db.add(existing); db.flush()
            db.add(FaceModelMetadata(face_id=existing.id,model_id=MODEL,alignment_id=ALIGNMENT_ID))
        before=state(engine)
        fails(422,'PHOTO_VALIDATION_FAILED',lambda: service.append(visit_id,**payload))
        assert state(engine) == before


def test_native_quality_and_association_fail_before_embedding():
    for key,value in (('min_size',79.),('det_score',.79),('blur',19.),('aligned',False),('unambiguous',False),('blur',None)):
        with fixture() as (engine, service, face, _clock, visit_id, payload):
            with immediate_session(engine) as db:
                row=db.get(HumanVisit,visit_id); row.face_quality=dict(row.face_quality,**{key:value})
            fails(422,'VISIT_FACE_QUALITY',lambda: service.append(visit_id,**payload))
            assert face.calls == 0 and state(engine)[4] == []


def test_full_fifteen_sample_set_never_spends_the_shared_engine_or_replaces():
    with fixture() as (engine, service, face, _clock, visit_id, payload):
        with immediate_session(engine) as db:
            db.add_all(EmployeeFace(employee_id=1,embedding=vector(20+i).tobytes()) for i in range(15))
        before=state(engine)
        fails(409,'FACE_LIMIT_REACHED',lambda: service.append(visit_id,**payload))
        assert state(engine) == before and face.calls == 0


def test_busy_enrollment_lease_leaves_every_mutation_pending_not_partially_written():
    with fixture() as (engine, service, face, _clock, visit_id, payload):
        before=state(engine)
        service.semaphore.acquire()
        try:
            fails(503,'ENROLLMENT_BUSY',lambda: service.append(visit_id,**payload))
        finally:
            service.semaphore.release()
        assert state(engine) == before and face.calls == 0


def test_nonfinite_model_output_cannot_create_provenance_or_receipt():
    with fixture() as (engine, service, face, _clock, visit_id, payload):
        before=state(engine); face.value=np.full(512,np.nan,np.float32)
        fails(503,'FACE_MODEL_UNAVAILABLE',lambda: service.append(visit_id,**payload))
        assert state(engine) == before


def test_metadata_insert_failure_rolls_back_face_employee_gallery_and_receipt():
    with fixture() as (engine, service, _face, _clock, visit_id, payload):
        before=state(engine)
        def reject(_connection,_cursor,statement,_parameters,_context,_many):
            if statement.lower().startswith('insert into face_model_metadata'):
                raise SQLAlchemyError('injected metadata write failure')
        event.listen(engine,'before_cursor_execute',reject)
        try:
            fails(503,'PERSISTENCE_ERROR',lambda: service.append(visit_id,**payload))
        finally:
            event.remove(engine,'before_cursor_execute',reject)
        assert state(engine) == before


def test_embedding_preparation_holds_no_database_transaction():
    with fixture() as (engine, service, face, _clock, visit_id, payload):
        def external_write():
            with immediate_session(engine) as db:
                db.get(Employee,2).external_id='committed-during-inference'
        face.callback=external_write
        service.append(visit_id,**payload)
        with Session(engine) as db:
            assert db.get(Employee,2).external_id == 'committed-during-inference'


def test_unknown_missing_image_is_not_reported_as_expired():
    with fixture() as (engine, service, _face, _clock, visit_id, _payload):
        with immediate_session(engine) as db:
            row=db.get(HumanVisit,visit_id); row.face_thumbnail=None
        with Session(engine) as db:
            assert service.eligibility(db.get(HumanVisit,visit_id)) == (False,'VISIT_FACE_MISSING')


def test_mismatched_engine_never_stamps_vectors_as_the_declared_model():
    with fixture() as (engine, service, face, _clock, visit_id, payload):
        face.model_id='b'*64; before=state(engine)
        fails(503,'FACE_MODEL_UNAVAILABLE',lambda: service.append(visit_id,**payload))
        assert state(engine) == before and face.calls == 0


def test_committed_append_publishes_the_real_shared_gallery():
    with fixture() as (engine, service, _face, _clock, visit_id, payload):
        store = EmbeddingStore(MODEL)
        with Session(engine) as db:
            store.refresh(db)
        service.store = store
        result = service.append(visit_id, **payload)
        with Session(engine) as db:
            assert store.revision == db.get(AttendanceRuntime, 1).gallery_revision
        assert store.match(vector()).employee_id == result["id"]
        assert service.append(visit_id, **payload) == result


if __name__ == '__main__':
    import traceback
    cases=[value for name,value in list(globals().items()) if name.startswith('test_') and callable(value)]
    failed=0
    for case in cases:
        try:
            case(); print('PASS',case.__name__)
        except Exception:
            failed+=1; traceback.print_exc()
    print(f'{len(cases)-failed}/{len(cases)} visit sample cases passed')
    raise SystemExit(bool(failed))
