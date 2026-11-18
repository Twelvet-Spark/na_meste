"""Scene service and HTTP contracts against temporary real schema-v4 SQLite.

Run: .venv/Scripts/python.exe tests/test_scene_service.py
No app lifespan, recorder, network, production database, or model is used.
"""
import base64
import sys
import tempfile
import traceback
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from sqlalchemy import create_engine, event
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from app.api.scenes import router
from app.database.attendance_schema import initialize_attendance_schema
from app.database.database import immediate_session
from app.database.models import AttendanceCamera, Camera, CameraScene
from app.rtsp.reader import CameraStatus, FrameBuffer
from app.services.scene_service import SceneService
from operator_test_support import Client
from test_scene_geometry import geometry

TESTS = []
BASE = datetime(2026, 11, 16, 10, tzinfo=timezone.utc)


def test(fn):
    TESTS.append(fn)
    return fn


class Clock:
    value = 100.0

    def __call__(self):
        return self.value


class Reader:
    def __init__(self, clock, *, source_revision=1, stream_id=None, size=(640, 360)):
        self.buffer = FrameBuffer()
        self.source_revision = source_revision
        self.state = CameraStatus.ONLINE
        self.publish(clock, stream_id=stream_id or str(uuid4()), size=size)

    def publish(self, clock, *, stream_id=None, size=(640, 360)):
        previous = self.buffer.latest()
        self.buffer.put(np.zeros((size[1], size[0], 3), np.uint8),
                        stream_id=stream_id or previous.stream_id,
                        source_revision=self.source_revision, received_at=BASE, received_mono=clock())

    def status(self):
        return self.state


@contextmanager
def api():
    with tempfile.TemporaryDirectory() as directory:
        engine = create_engine(f"sqlite:///{Path(directory, 'scene.db').as_posix()}",
                               connect_args={'check_same_thread': False})

        @event.listens_for(engine, 'connect')
        def foreign_keys(connection, _record):
            connection.execute('PRAGMA foreign_keys=ON')

        try:
            initialize_attendance_schema(engine)
            with immediate_session(engine) as db:
                db.add_all([Camera(id=i, name=f'Camera {i}', nvr_host='unused.invalid',
                                   username='unused', password='unused', channel=i) for i in (1, 2, 3)])
                db.flush()
                # Schema startup supplies rows for existing cameras only.
                db.add_all([AttendanceCamera(camera_id=i, source_revision=1) for i in (1, 2, 3)])
            clock = Clock()
            readers = {i: Reader(clock) for i in (1, 2, 3)}
            manager = SimpleNamespace(get=readers.get)
            service = SceneService(engine, manager, monotonic=clock)
            app = FastAPI()
            app.state.scene_service = service
            app.include_router(router)
            yield Client(app), engine, service, readers, clock
        finally:
            engine.dispose()


def freeze(client, camera=1, **kwargs):
    return client.json('POST', f'/api/cameras/{camera}/scene/snapshot', **kwargs)


def saved_scene(client, camera=1):
    """The persisted scene document; availability flags describe the running app, not the scene."""
    document = client.json('GET', f'/api/cameras/{camera}/scene')
    assert (document.pop('assistance_available'), document.pop('analysis_available')) == (False, False)
    return document


def payload(snapshot, *, revision=0, source=1, value=None):
    return dict(snapshot_id=snapshot['snapshot_id'], expected_revision=revision,
                expected_source_revision=source, geometry=geometry() if value is None else value)


def put(client, value, camera=1, **kwargs):
    return client.json('PUT', f'/api/cameras/{camera}/scene', value, **kwargs)


def rejected_snapshot(service, identifier, code, camera=1):
    try:
        service.get_snapshot(identifier, camera)
    except HTTPException as exc:
        assert exc.status_code == code, exc.detail
    else:
        raise AssertionError('Invalid snapshot was usable')


@test
def test_real_routes_freeze_native_save_reload():
    with api() as (client, engine, service, readers, _clock):
        unset = client.json('GET', '/api/cameras/1/scene')
        assert unset['status'] == 'UNSET' and unset['revision'] == 0 and unset['geometry'] is None
        snap = freeze(client)
        assert snap['width'] == 640 and snap['height'] == 360 and snap['expires_in_seconds'] == 300
        decoded = cv2.imdecode(np.frombuffer(base64.b64decode(snap['jpeg_base64']), np.uint8), cv2.IMREAD_COLOR)
        assert decoded.shape == (360, 640, 3)
        frozen = service.get_snapshot(snap['snapshot_id'], 1)
        assert frozen.packet is readers[1].buffer.latest() and not frozen.packet.frame.flags.writeable
        value = geometry(metric=True)
        value['floor'][0][0] = .000000123456789
        saved = put(client, payload(snap, value=value))
        assert saved['status'] == 'CURRENT' and saved['revision'] == 1
        assert saved['reference_width'] == 640 and saved['reference_height'] == 360
        assert saved['geometry'] == value and saved['source_revision'] == saved['current_source_revision'] == 1
        assert saved['metric_transform']['rectified_width'] == 1280
        assert saved_scene(client) == saved
        with Session(engine) as db:
            assert db.get(AttendanceCamera, 1).revision == 0


@test
def test_mutation_security_and_exact_body_rejection_leave_no_rows():
    with api() as (client, engine, _service, _readers, _clock):
        freeze(client, operator=False, status=403)
        freeze(client, headers={'origin': 'https://evil.invalid'}, status=403)
        snap = freeze(client, headers={'origin': 'http://test'})
        put(client, payload(snap), operator=False, status=403)
        put(client, payload(snap), headers={'origin': 'http://evil.invalid'}, status=403)
        for extra in ({'reference_width': 10}, {'jpeg_base64': 'bad'}, {'url': 'http://unused.invalid'}):
            put(client, dict(payload(snap), **extra), status=422)
            client.json('POST', '/api/cameras/2/scene/snapshot', extra, status=422)
        for key in ('expected_revision', 'expected_source_revision'):
            put(client, dict(payload(snap), **{key: True}), status=422)
        put(client, dict(payload(snap), snapshot_id='not-a-token'), status=422)
        put(client, dict(payload(snap), geometry=dict(geometry(), unknown=1)), status=422)
        with Session(engine) as db:
            assert db.query(CameraScene).count() == 0


@test
def test_global_two_snapshot_limit_ttl_unknown_wrong_camera_and_reuse():
    with api() as (client, _engine, service, readers, clock):
        first = freeze(client)
        freeze(client, camera=2)
        freeze(client, camera=3, status=503)
        rejected_snapshot(service, first['snapshot_id'], 409, camera=2)
        rejected_snapshot(service, str(uuid4()), 410)
        clock.value += 299.999
        assert service.get_snapshot(first['snapshot_id']).camera_id == 1
        clock.value += .001
        rejected_snapshot(service, first['snapshot_id'], 410)
        readers[3].publish(clock)
        assert freeze(client, camera=3)['width'] == 640


@test
def test_freshness_offline_oversize_and_dimension_change_are_not_freezable():
    with api() as (client, _engine, _service, readers, clock):
        readers[1].state = CameraStatus.RECONNECTING
        freeze(client, status=503)
        readers[1].state = CameraStatus.ONLINE
        clock.value += 100
        freeze(client, status=503)
        readers[1].publish(clock, size=(4001, 4000))
        freeze(client, status=503)
        readers[1].publish(clock)
        snap = freeze(client)
        readers[1].publish(clock, size=(320, 180))
        put(client, payload(snap), status=409)


@test
def test_same_source_reader_reconnect_and_stream_change_invalidate_tokens():
    with api() as (client, _engine, service, readers, clock):
        snap = freeze(client)
        original_stream = readers[1].buffer.latest().stream_id
        readers[1] = Reader(clock, stream_id=original_stream)
        rejected_snapshot(service, snap['snapshot_id'], 409)
        new = freeze(client)
        readers[1].publish(clock, stream_id=str(uuid4()))
        rejected_snapshot(service, new['snapshot_id'], 409)


@test
def test_camera_disable_source_replacement_and_deletion_reject_tokens():
    for change in ('disable', 'source', 'delete'):
        with api() as (client, engine, service, _readers, _clock):
            snap = freeze(client)
            with immediate_session(engine) as db:
                if change == 'disable':
                    db.get(Camera, 1).enabled = False
                elif change == 'source':
                    db.get(AttendanceCamera, 1).source_revision += 1
                else:
                    db.delete(db.get(AttendanceCamera, 1))
                    db.delete(db.get(Camera, 1))
            rejected_snapshot(service, snap['snapshot_id'], 409)
            with Session(engine) as db:
                assert db.query(CameraScene).count() == 0


@test
def test_invalidation_callback_and_old_source_scene_stays_stale():
    with api() as (client, engine, service, readers, clock):
        snap = freeze(client)
        put(client, payload(snap, value=geometry(metric=True)))
        notified = []
        service.add_invalidation_callback(notified.append)
        with immediate_session(engine) as db:
            db.get(AttendanceCamera, 1).source_revision = 2
        readers[1] = Reader(clock, source_revision=2)
        service.invalidate(1)
        assert notified == [1]
        rejected_snapshot(service, snap['snapshot_id'], 410)
        changed = put(client, payload(snap, revision=1), status=409)
        assert changed['detail']['code'] == 'CAMERA_SOURCE_CHANGED'
        stale = client.json('GET', '/api/cameras/1/scene')
        assert stale['status'] == 'STALE' and stale['source_revision'] == 1
        assert stale['current_source_revision'] == 2 and stale['metric_transform'] is None
        assert stale['revision'] == 1 and stale['geometry']['metric'] is not None
        assert service.get_metric_transform(1) is None


@test
def test_imported_source_zero_stays_stale_across_restart_until_valid_save():
    with api() as (client, engine, service, _readers, clock):
        with immediate_session(engine) as db:
            db.add(CameraScene(camera_id=1, revision=7, source_revision=0,
                               reference_width=1920, reference_height=1080, geometry=geometry(metric=True)))
        assert client.json('GET', '/api/cameras/1/scene')['status'] == 'STALE'
        restarted = SceneService(engine, service.manager, monotonic=clock)
        assert restarted.get(1)['status'] == 'STALE' and restarted.get_metric_transform(1) is None
        snap = freeze(client)
        saved = put(client, payload(snap, revision=7))
        assert saved['status'] == 'CURRENT' and saved['source_revision'] == 1 and saved['revision'] == 8
        assert saved['reference_width'] == 640


@test
def test_rejected_geometry_and_scene_source_cas_preserve_saved_metadata():
    with api() as (client, engine, _service, _readers, _clock):
        snap = freeze(client)
        original = put(client, payload(snap))
        put(client, payload(snap), status=409)
        put(client, payload(snap, revision=1, source=2), status=409)
        invalid = geometry(metric=True)
        invalid['metric']['corners'] = [[0, 0], [1, 1], [1, 0], [0, 1]]
        put(client, payload(snap, revision=1, value=invalid), status=422)
        horizon = geometry(metric=True)
        horizon['metric']['corners'] = [[.4, .2], [.6, .2], [.9, .5], [.1, .5]]
        horizon['floor'] = [[.5, .05], [.8, .2], [1, .6], [0, .6], [.2, .2]]
        put(client, payload(snap, revision=1, value=horizon), status=422)
        assert saved_scene(client) == original
        with Session(engine) as db:
            row = db.get(CameraScene, 1)
            assert row.revision == 1 and row.geometry == original['geometry']


@test
def test_revalidation_after_encoding_does_not_publish_retired_snapshot():
    with api() as (client, _engine, _service, readers, clock):
        encode = cv2.imencode

        def reconnect(*args, **kwargs):
            result = encode(*args, **kwargs)
            readers[1] = Reader(clock)
            return result

        with patch('app.services.scene_service.cv2.imencode', side_effect=reconnect):
            freeze(client, status=409)
        # Failed work released its reserved slot.
        freeze(client)
        freeze(client, camera=2)


@test
def test_disable_during_encoding_does_not_publish_snapshot():
    with api() as (client, engine, _service, _readers, _clock):
        encode = cv2.imencode

        def disable(*args, **kwargs):
            result = encode(*args, **kwargs)
            with immediate_session(engine) as db:
                db.get(Camera, 1).enabled = False
            return result

        with patch('app.services.scene_service.cv2.imencode', side_effect=disable):
            freeze(client, status=409)


@test
def test_save_rechecks_source_after_validation_and_stream_after_flush():
    with api() as (client, engine, _service, readers, clock):
        from app.services.scene_service import validate_geometry
        snap = freeze(client)

        def changed_source(value):
            result = validate_geometry(value)
            with immediate_session(engine) as db:
                db.get(AttendanceCamera, 1).source_revision = 2
            return result

        with patch('app.services.scene_service.validate_geometry', side_effect=changed_source):
            put(client, payload(snap), status=409)
        with Session(engine) as db:
            assert db.get(CameraScene, 1) is None
        with immediate_session(engine) as db:
            db.get(AttendanceCamera, 1).source_revision = 1
        readers[1] = Reader(clock)
        new = freeze(client)

        def replace_on_write(_connection, _cursor, statement, _parameters, _context, _many):
            if statement.lstrip().upper().startswith('INSERT INTO CAMERA_SCENES'):
                readers[1] = Reader(clock)

        event.listen(engine, 'before_cursor_execute', replace_on_write)
        try:
            put(client, payload(new), status=409)
        finally:
            event.remove(engine, 'before_cursor_execute', replace_on_write)
        with Session(engine) as db:
            assert db.get(CameraScene, 1) is None


@test
def test_homography_cache_is_revision_scoped_and_stale_never_applies():
    with api() as (client, _engine, service, _readers, _clock):
        snap = freeze(client)
        put(client, payload(snap, value=geometry(metric=True)))
        first = service.get_metric_transform(1)
        assert first is service.get_metric_transform(1)
        changed = geometry(metric=True)
        changed['metric']['width_m'] = 8
        put(client, payload(snap, revision=1, value=changed))
        second = service.get_metric_transform(1)
        assert second is not first
        np.testing.assert_allclose(second.to_metres([[.5, .5]]), [[4, 1.5]], atol=1e-5)


@test
def test_failed_encoding_and_database_write_release_slots_without_metadata_changes():
    with api() as (client, engine, _service, _readers, _clock):
        with patch('app.services.scene_service.cv2.imencode', return_value=(False, None)):
            freeze(client, status=503)
        snap = freeze(client)
        original = put(client, payload(snap))

        def fail_write(_connection, _cursor, statement, parameters, _context, _many):
            if statement.lstrip().upper().startswith('UPDATE CAMERA_SCENES'):
                raise OperationalError(statement, parameters, RuntimeError('isolated write failure'))

        event.listen(engine, 'before_cursor_execute', fail_write)
        try:
            put(client, payload(snap, revision=1), status=503)
        finally:
            event.remove(engine, 'before_cursor_execute', fail_write)
        assert saved_scene(client) == original
        freeze(client, camera=2)


@test
def test_invalidation_during_encoding_cannot_free_an_inflight_capture_slot():
    with api() as (client, _engine, service, _readers, _clock):
        freeze(client, camera=2)
        encode = cv2.imencode
        entered = False

        def invalidate_and_try_capture(*args, **kwargs):
            nonlocal entered
            result = encode(*args, **kwargs)
            if not entered:
                entered = True
                service.invalidate(1)
                try:
                    service.snapshot(3)
                except HTTPException as exc:
                    assert exc.status_code == 503
                else:
                    raise AssertionError('Inflight native image lost its bounded capture reservation')
            return result

        with patch('app.services.scene_service.cv2.imencode', side_effect=invalidate_and_try_capture):
            freeze(client, status=410)
        # The original native work finished; its reservation can now be reused.
        freeze(client, camera=3)


@test
def test_scene_revision_changed_during_preparation_is_not_overwritten():
    with api() as (client, engine, _service, _readers, _clock):
        from app.services.scene_service import validate_geometry
        snap = freeze(client)
        put(client, payload(snap))
        concurrent = geometry()
        concurrent['objects'] = [dict(id=str(uuid4()), label='Concurrent edit',
                                      polygon=[[.1, .1], [.3, .1], [.3, .3]])]

        def concurrent_edit(value):
            result = validate_geometry(value)
            with immediate_session(engine) as db:
                row = db.get(CameraScene, 1)
                row.geometry = concurrent
                row.revision = 2
            return result

        with patch('app.services.scene_service.validate_geometry', side_effect=concurrent_edit):
            put(client, payload(snap, revision=1), status=409)
        current = client.json('GET', '/api/cameras/1/scene')
        assert current['revision'] == 2 and current['geometry'] == concurrent


if __name__ == '__main__':
    failures = 0
    for case in TESTS:
        try:
            case()
            print(f'PASS {case.__name__}')
        except Exception:
            failures += 1
            print(f'FAIL {case.__name__}')
            traceback.print_exc()
    print(f'{len(TESTS) - failures}/{len(TESTS)} scene service cases passed')
    raise SystemExit(bool(failures))
