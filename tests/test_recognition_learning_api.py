"""ASGI multipart admissions and real stored recognition feedback over an isolated database."""
from contextlib import contextmanager
from datetime import timedelta, timezone
from hashlib import sha256
import traceback
from unittest.mock import patch
from uuid import uuid4

from fastapi import FastAPI
from sqlalchemy.orm import Session
from starlette import formparsers

from operator_test_support import Client, multipart
from test_recognition_learning import feedback_counts, make_upload, saved_photo, service_db
from test_recognition_learning_gallery import STAMP, VISIT
from app.api import human_visits
from app.api.global_persons import router as global_persons_router
from app.database.database import get_db, immediate_session
from app.database.models import Employee, HumanVisit, RecognitionFeedback

TESTS = []
PATH = f'/api/global-persons/visits/{VISIT}/learning'
EXAMPLES = '/api/recognition-learning/examples'
MAX_PHOTO = 15 * 1024 * 1024


def test(fn):
    TESTS.append(fn)
    return fn


@contextmanager
def api():
    from app.api.recognition_learning import router
    with service_db() as (engine, service, body, clock, _slot):
        app = FastAPI()
        app.state.recognition_learning_service = service
        app.include_router(global_persons_router)
        app.include_router(router)

        def db():
            with Session(engine) as session:
                yield session
        app.dependency_overrides[get_db] = db
        with patch('app.api.human_visits.utcnow', side_effect=lambda: clock[0]):
            yield Client(app), engine, service, body, clock


def form(payload, upload=None, **kwargs):
    files = [(upload[0], upload[1], 'image/png')] if upload else []
    body, content = multipart(payload, files, **kwargs)
    return body.replace(b'name="files"', b'name="photo"'), content


def post(client, suffix, payload, upload=None, *, status=200, **kwargs):
    blob, content = form(payload, upload)
    return client.json('POST', PATH + suffix, status=status, body=blob,
                       headers={'content-type': content, **kwargs.pop('headers', {})}, **kwargs)


def feedback(client, source, *, employee=1, action='ACCEPT', upload=None):
    scanned = post(client, '/analyze', source, upload)
    request = dict(source, action=action, employee_id=employee, person_confirmed=True,
                   expected_employee_revision=0, expected_identity_revision=scanned['identity_revision'],
                   expected_learning_revision=scanned['learning_revision'], expected_gallery_revision=scanned['gallery_revision'],
                   expected_body_model_id=scanned['body_model_id'], expected_face_model_id=scanned['face_model_id'],
                   expected_image_sha256=scanned['evidence']['image_sha256'], request_id=str(uuid4()),
                   actor='Audit operator', reason='Verified exact crop')
    _, _, result = client.request('POST', PATH + '/feedback', **multipart_options(request, upload))
    assert result.get('feedback_id'), result
    return request, result


def multipart_options(payload, upload=None):
    contents, content_type = form(payload, upload)
    return {'body': contents, 'headers': {'content-type': content_type}}


@test
def multipart_marker_limits_spool_cleanup_and_read_only_preparation():
    with api() as (client, engine, _service, body, _clock):
        upload = make_upload()
        source = {'source': 'UPLOAD', 'crop': [0, 0, 1, 1]}
        for suffix in ('/analyze', '/feedback'):
            post(client, suffix, source, upload, status=403, operator=False)
            post(client, suffix, source, upload, status=403, headers={'origin': 'http://not-test'})
            _, content = form(source, upload)
            def must_not_read():
                raise AssertionError('declared oversize body was consumed')
                yield b''
            client.json('POST', PATH + suffix, status=413, chunks=must_not_read(), headers={
                'content-type': content, 'content-length': MAX_PHOTO + 80 * 1024 + 1})
        made = []
        real_spool = formparsers.SpooledTemporaryFile
        def spool(*args, **kwargs):
            f = real_spool(*args, **kwargs)
            made.append(f)
            return f
        with patch.object(formparsers, 'SpooledTemporaryFile', side_effect=spool):
            blob, content = form(source, ('large.png', b'x' * (MAX_PHOTO + 90 * 1024)))
            client.json('POST', PATH + '/analyze', status=413,
                        chunks=(blob[i:i+65536] for i in range(0, len(blob), 65536)),
                        headers={'content-type': content})
            assert made and all(stream.closed for stream in made)
            post(client, '/analyze', source, ('large.png', b'x' * (MAX_PHOTO + 1)), status=413)
            blob, content = form(source, upload, fields=[('metadata', '{}')])
            code, _, _ = client.request('POST', PATH + '/analyze', body=blob, headers={'content-type': content})
            assert code in (400, 422)
            blob, content = multipart(source, [(upload[0], upload[1], 'image/png')] * 2)
            code, _, _ = client.request('POST', PATH + '/analyze', body=blob.replace(b'name="files"', b'name="photo"'),
                                         headers={'content-type': content})
            assert code in (400, 422)
            blob, content = form(source, upload)
            # A graceful end to a truncated HTTP body is not a multipart close boundary.
            delimiter = b'--' + content.split('boundary=', 1)[1].encode()
            truncated = blob[:blob.rindex(delimiter)]
            client.json('POST', PATH + '/analyze', status=422, body=truncated,
                        headers={'content-type': content})
            assert all(stream.closed for stream in made), 'Unfinished file part was left open'
            assert not body.images
            result = post(client, '/analyze', source, upload)
            assert len(body.images) == 1 and all(stream.closed for stream in made)
            assert result['evidence']['captured_at'] is None
        assert feedback_counts(engine) == (0, 0, 0, 0, 0, 0)


@test
def saved_photo_relabels_with_exact_history_media_and_undo():
    with api() as (client, engine, service, _, clock):
        source = saved_photo(engine)
        request_a, accepted = feedback(client, source)
        visit_url = f'/api/global-persons/visits/{VISIT}'
        original = client.json('GET', visit_url)
        assert original['employee_id'] == 1 and original['identity_revision'] == 1
        assert original['learning_available'] and original['photo_captured_at']
        assert original['learning_revision'] == accepted['learning_revision']
        assert original['gallery_revision'] == accepted['gallery_revision']
        l = client.json('GET', '/api/global-persons/visits')['items'][0]
        assert 'snapshot' not in l and 'candidates' not in l and 'photo_captured_at' not in l
        media = f"/api/recognition-learning/{accepted['feedback_id']}/photo"
        status, headers, photo = client.request('GET', media)
        assert status == 200 and headers[b'cache-control'] == b'no-store'
        assert headers[b'etag'].strip(b'"').decode() == sha256(photo).hexdigest() == request_a['expected_image_sha256']
        assert bytes(photo).startswith(b'\xff\xd8')
        with immediate_session(engine) as db:
            db.get(Employee, 2).name = 'Changed display name'
        clock[0] += timedelta(seconds=1)
        request_b, replacement = feedback(client, source | {'crop': [0, 0., .5, 1.]}, employee=2)
        assert replacement['identity']['employee_id'] == 2
        assert client.request('GET', media)[0] == 410
        assert client.request('GET', f'/api/recognition-learning/{uuid4()}/photo')[0] == 404
        history = client.json('GET', PATH + '/history?limit=1')
        assert history['next_cursor'] and history['items'][0]['actor'] == request_b['actor']
        assert history['items'][0]['reason'] == request_b['reason']
        next_page = client.json('GET', PATH + '/history?limit=1&cursor=' + history['next_cursor'])
        assert next_page['items'][0]['employee_id'] == 1
        assert client.request('GET', PATH + '/history?limit=201')[0] == 422
        assert client.request('GET', PATH + '/history?cursor=bad')[0] == 422
        assert client.json('GET', EXAMPLES + '?employee_id=1')['items'][0]['employee_id'] == 1
        target_id = replacement['feedback_id']
        targeted = client.json('GET', f'{EXAMPLES}?employee_id=2&feedback_id={target_id}')
        assert [row['feedback_id'] for row in targeted['items']] == [target_id]
        assert targeted['next_cursor'] is None
        assert client.json('GET', f'{EXAMPLES}?employee_id=1&feedback_id={target_id}')['items'] == []
        duplicate = client.json('POST', PATH + '/feedback', status=409,
                                **multipart_options(request_b | {
                                    'request_id': str(uuid4()), 'expected_identity_revision': 2,
                                    'expected_learning_revision': replacement['learning_revision'],
                                    'expected_gallery_revision': replacement['gallery_revision'],
                                }))
        assert duplicate['detail']['code'] == 'LEARNING_DUPLICATE'
        assert duplicate['detail']['employee_id'] == 2 and duplicate['detail']['feedback_id'] == target_id
        service.body_engine = None
        revoke_url = f"/api/recognition-learning/{replacement['feedback_id']}/revoke"
        payload = dict(request_id=str(uuid4()), actor='Reviewer', reason='Incorrect label', expected_identity_revision=2)
        client.json('POST', revoke_url, payload, operator=False, status=403)
        undone = client.json('POST', revoke_url, payload)
        assert undone['identity']['employee_id'] == 1
        assert undone['historical_hours_changed'] is False
        assert client.json('POST', revoke_url, payload) == undone
        with Session(engine) as db:
            rows = db.query(RecognitionFeedback).all()
            assert all(row.revoked_at and row.image_jpeg is None for row in rows)
            assert {row.revoked_by for row in rows} == {'Audit operator', 'Reviewer'}


@test
def dated_upload_is_only_employee_reference_while_saved_rejection_can_clear_visit():
    with api() as (client, engine, _service, _, _clock):
        source = saved_photo(engine)
        upload = make_upload()
        ref = {'source': 'UPLOAD', 'crop': [.5, .0, 1, 1], 'captured_at': STAMP.replace(tzinfo=timezone.utc).isoformat()}
        _, result = feedback(client, ref, employee=2, upload=upload)
        assert result['identity'] is None
        with Session(engine) as db:
            assert db.get(RecognitionFeedback, result['feedback_id']).captured_at == STAMP
            assert db.get(HumanVisit, VISIT).identity_state == 'UNKNOWN'
        assert client.json('GET', f'/api/global-persons/visits/{VISIT}')['identity_revision'] == 0
        feedback(client, source)
        _, rejected = feedback(client, source | {'crop': [0, .25, .5, .75]}, action='REJECT')
        assert rejected['identity']['employee_id'] is None and rejected['identity']['identity_revision'] == 2
        assert feedback_counts(engine)[-1] == 0
        with Session(engine) as db:
            assert db.get(RecognitionFeedback, result['feedback_id']).revoked_at is None


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
