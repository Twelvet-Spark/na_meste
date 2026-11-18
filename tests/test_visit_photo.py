"""Real image validation, gallery matching and journal-photo HTTP transactions."""
import hashlib
import sys
import unittest
from pathlib import Path
from uuid import uuid4
from dataclasses import replace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import numpy as np
from fastapi import FastAPI
from sqlalchemy.orm import Session
from app.ai.face_detector import FaceInfo
from app.ai.face_recognition import EmbeddingStore
from app.api.global_persons import router
from app.database.database import get_db, immediate_session
from app.database.models import EmployeeFace, HumanVisit, FaceModelMetadata, Employee
from operator_test_support import Client, multipart
from test_visit_sample import fixture, jpeg, vector, MODEL, ALIGNMENT_ID, state, NOW


class Faces:
    model_id = MODEL
    alignment_id = ALIGNMENT_ID
    def __init__(self):
        aligned = cv2.imdecode(np.frombuffer(jpeg(), np.uint8), cv2.IMREAD_COLOR)
        self.observations = [FaceInfo(np.array([0., 0., 112., 112.]), .99, vector(), aligned, model_id=MODEL)]
        self.callback = None
    def detect(self, image):
        if self.callback:
            self.callback()
        return self.observations


def client_for(engine, service):
    app = FastAPI()
    app.state.visit_sample_service = service
    app.include_router(router)
    def db():
        with Session(engine) as session:
            yield session
    app.dependency_overrides[get_db] = db
    return Client(app)


def metadata(revision, **changes):
    values = dict(employee_id=1, expected_revision=revision, request_id=str(uuid4()),
                  source="UPLOAD", actor="Operator", reason="Verified recognition sample")
    values.update(changes)
    return values


def post_photo(client, visit_id, data, files=(), **kwargs):
    body, content_type = multipart(data, files)
    body = body.replace(b'name="files"', b'name="file"')
    return client.json("POST", f"/api/global-persons/visits/{visit_id}/employee-photo", body=body,
                       headers={"content-type": content_type}, **kwargs)


class VisitPhotoTests(unittest.TestCase):
    def test_enrollment_availability_is_independent_of_retained_face(self):
        with fixture() as (engine, service, _, _, visit_id, _):
            service.face_engine = Faces()
            with immediate_session(engine) as db:
                db.get(HumanVisit, visit_id).face_thumbnail = None
            client = client_for(engine, service)
            self.assertTrue(client.json('GET', f'/api/global-persons/visits/{visit_id}')['enrollment_available'])
            service.face_engine = None
            self.assertFalse(client.json('GET', f'/api/global-persons/visits/{visit_id}')['enrollment_available'])

    def test_upload_matches_real_gallery_and_last_sample_can_be_removed_without_model(self):
        with fixture() as (engine, service, _, clock, visit_id, original):
            service.face_engine = Faces()
            service.store = EmbeddingStore(MODEL)
            client = client_for(engine, service)
            data = metadata(original['expected_revision'])
            files = [("employee.jpg", jpeg(), "image/jpeg")]
            saved = post_photo(client, visit_id, data, files)
            self.assertEqual(service.store.match(vector()).employee_id, 1)
            service.face_engine = None
            clock[0] = NOW.replace(year=2027)
            self.assertEqual(post_photo(client, visit_id, data, files), saved)
            remove = dict(request_id=str(uuid4()), sample_request_id=saved['sample_request_id'],
                          expected_revision=saved['revision'], actor='Operator', reason='Wrong sample')
            url = f"/api/global-persons/visits/{visit_id}/employee-sample/remove"
            removed = client.json('POST', url, remove)
            self.assertEqual(removed['faces_count'], 0)
            self.assertIsNone(service.store.match(vector()).employee_id)
            self.assertEqual(client.json('POST', url, remove), removed)
            items = client.json('GET', f"/api/global-persons/visits/{visit_id}/employee-samples")['items']
            self.assertEqual(len(items), 1)
            self.assertFalse(items[0]['present'])
            self.assertEqual(items[0]['removal']['actor'], 'Operator')
            self.assertEqual(items[0]['removal']['reason'], 'Wrong sample')
            self.assertEqual(items[0]['removal']['request_id'], remove['request_id'])
            self.assertIsNotNone(items[0]['removal']['created_at'])
            with Session(engine) as db:
                self.assertEqual(db.query(EmployeeFace).count(), 0)
                self.assertEqual(db.query(FaceModelMetadata).count(), 0)

    def test_saved_person_crop_is_independent_of_aligned_face_eligibility(self):
        with fixture() as (engine, service, _, _, visit_id, original):
            service.face_engine = Faces()
            raw = jpeg()
            with immediate_session(engine) as db:
                visit = db.get(HumanVisit, visit_id)
                visit.snapshot, visit.face_thumbnail, visit.face_quality = raw, None, None
                old = EmployeeFace(employee_id=1, embedding=vector(10).tobytes(), thumbnail=jpeg(8))
                db.add(old); db.flush()
                old_id = old.id
            client = client_for(engine, service)
            data = metadata(original['expected_revision'], source='PHOTO_CROP',
                            photo_sha256=hashlib.sha256(raw).hexdigest(), crop=[0., 0., 1., 1.])
            saved = post_photo(client, visit_id, data)
            self.assertEqual(saved['faces_count'], 2)
            with Session(engine) as db:
                self.assertEqual(db.get(EmployeeFace, old_id).embedding, vector(10).tobytes())
            data['reason'] = 'Changed request'
            post_photo(client, visit_id, data, status=409)

    def test_changed_crop_during_inference_cannot_enroll_an_unseen_image(self):
        with fixture() as (engine, service, _, _, visit_id, original):
            faces = Faces(); service.face_engine = faces
            raw = jpeg()
            with immediate_session(engine) as db:
                db.get(HumanVisit, visit_id).snapshot = raw
            def changed():
                with immediate_session(engine) as db:
                    db.get(HumanVisit, visit_id).snapshot = jpeg(8)
            faces.callback = changed
            before = state(engine)
            client = client_for(engine, service)
            data = metadata(original['expected_revision'], source='PHOTO_CROP',
                            photo_sha256=hashlib.sha256(raw).hexdigest(), crop=[0.,0.,1.,1.])
            post_photo(client, visit_id, data, status=409)
            self.assertEqual(state(engine), before)

    def test_upload_quality_and_conflicting_owner_never_poison_gallery(self):
        for invalid in ('none', 'multiple', 'small', 'blurred', 'low_score', 'owner'):
            with self.subTest(invalid=invalid), fixture() as (engine, service, _, _, visit_id, original):
                faces = Faces(); service.face_engine = faces
                observation = faces.observations[0]
                if invalid == 'none': faces.observations = []
                if invalid == 'multiple': faces.observations *= 2
                if invalid == 'small': faces.observations = [replace(observation, bbox=np.array([0.,0.,79.,79.]))]
                if invalid == 'blurred': faces.observations = [replace(observation, aligned=np.zeros((112,112,3),np.uint8))]
                if invalid == 'low_score': faces.observations = [replace(observation, det_score=.79)]
                if invalid == 'owner':
                    with immediate_session(engine) as db:
                        face = EmployeeFace(employee_id=2, embedding=vector().tobytes(), thumbnail=jpeg())
                        db.add(face); db.flush()
                        db.add(FaceModelMetadata(face_id=face.id, model_id=MODEL, alignment_id=ALIGNMENT_ID))
                before = state(engine)
                post_photo(client_for(engine, service), visit_id, metadata(original['expected_revision']),
                           [('sample.jpg', jpeg(), 'image/jpeg')], status=422)
                self.assertEqual(state(engine), before)

    def test_source_validation_never_appends(self):
        with fixture() as (engine, service, _, _, visit_id, original):
            service.face_engine = Faces()
            client = client_for(engine, service)
            data = metadata(original['expected_revision'])
            before = state(engine)
            post_photo(client, visit_id, data, status=422)
            post_photo(client, visit_id, data, [('bad.jpg', b'not an image', 'image/jpeg')], status=422)
            post_photo(client, visit_id, data, [('x.jpg', jpeg(), 'image/jpeg')]*2, status=422)
            post_photo(client, visit_id, {**data, 'source':'PHOTO_CROP','photo_sha256':'a'*64,
                                         'crop':[0,0,1.1,1]}, status=422)
            post_photo(client, visit_id, {**data, 'source':'PHOTO_CROP','photo_sha256':'a'*64,
                                         'crop':[0,0,0,1]}, status=422)
            self.assertEqual(state(engine), before)

    def test_sample_removal_rejects_stale_revision_and_foreign_visit(self):
        with fixture() as (engine, service, _, _, visit_id, original):
            saved = service.append(visit_id, **original)
            client = client_for(engine, service)
            data = dict(request_id=str(uuid4()), sample_request_id=str(original['request_id']),
                        expected_revision=saved['revision']-1, actor='Operator', reason='Correction')
            before = state(engine)
            client.json('POST', f'/api/global-persons/visits/{visit_id}/employee-sample/remove', data, status=409)
            data.update(request_id=str(uuid4()), expected_revision=saved['revision'])
            client.json('POST', f'/api/global-persons/visits/{uuid4()}/employee-sample/remove', data, status=404)
            self.assertEqual(state(engine), before)


    def test_expired_saved_image_does_not_disable_new_upload(self):
        with fixture() as (engine, service, _, clock, visit_id, original):
            service.face_engine = Faces()
            clock[0] = NOW.replace(year=2027)
            client = client_for(engine, service)
            crop = metadata(original['expected_revision'], source='PHOTO_CROP',
                            photo_sha256='a'*64, crop=[0, 0, 1, 1])
            post_photo(client, visit_id, crop, status=410)
            saved = post_photo(client, visit_id, metadata(original['expected_revision']),
                               [('sample.jpg', jpeg(), 'image/jpeg')])
            self.assertEqual(saved['faces_count'], 1)

    def test_reused_face_id_and_legacy_receipt_cannot_delete_another_sample(self):
        from app.database.models import OperatorRequest
        with fixture() as (engine, service, _, _, visit_id, original):
            saved = service.append(visit_id, **original)
            client = client_for(engine, service)
            with immediate_session(engine) as db:
                db.delete(db.get(EmployeeFace, saved['face_id']))
                db.flush()
                db.add(EmployeeFace(id=saved['face_id'], employee_id=1,
                                    embedding=vector(10).tobytes(), thumbnail=jpeg(8)))
            remove = dict(request_id=str(uuid4()), sample_request_id=str(original['request_id']),
                          expected_revision=saved['revision'], actor='Operator', reason='Wrong sample')
            before = state(engine)
            client.json('POST', f'/api/global-persons/visits/{visit_id}/employee-sample/remove', remove, status=409)
            self.assertEqual(state(engine), before)
            with immediate_session(engine) as db:
                receipt = db.get(OperatorRequest, str(original['request_id']))
                receipt.result = {k:v for k,v in receipt.result.items() if k != 'sample_key'}
            sample = client.json('GET', f'/api/global-persons/visits/{visit_id}/employee-samples')['items'][0]
            self.assertIsNone(sample['present'])
            self.assertFalse(sample['removable'])

    def test_oversized_journal_bodies_and_foreign_origin_are_rejected(self):
        with fixture() as (engine, service, _, _, visit_id, _):
            client = client_for(engine, service)
            url = f'/api/global-persons/visits/{visit_id}/employee-photo'
            for body, headers in ((b'', {'content-length': str(16*1024*1024)}),
                                  (b'x'*(16*1024*1024), {})):
                status, _, _ = client.request('POST', url, body=body,
                    headers={'content-type':'multipart/form-data; boundary=x', **headers})
                self.assertEqual(status, 413)
            status, _, _ = client.request('POST', url, body=b'',
                                       headers={'origin':'https://outside.example'})
            self.assertEqual(status, 403)

if __name__ == '__main__':
    unittest.main()
