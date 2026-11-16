"""Isolated real ASGI/SQLite queue, cancellation and native lease proofs.
Run: .venv/Scripts/python.exe tests/test_scene_assistance.py
Real approved MobileSAM proof is opt-in: add --manifest PATH --image PRIVATE_PATH.
Add --diagnostics for bounded scalar mask/prompt evidence, even on empty output.
"""
import io
import tempfile
import argparse
import json
import math
import sys
import gc
import threading
import time
import unittest
import weakref
from contextlib import contextmanager, redirect_stdout
from pathlib import Path
from uuid import uuid4
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi import HTTPException
from app.ai.scene_segmenter import PolygonSuggestion
from app.services.scene_assistance import SceneAssistanceService
from test_scene_service import api, freeze, Reader


class ControlledSegmenter:
    """Only native inference is doubled; SceneService, writer and HTTP stay real."""
    def __init__(self, *, blocked=True, fail=False):
        self.entered = threading.Event()
        self.release = threading.Event()
        self.finished = threading.Event()
        self.calls = 0
        self.fail = fail
        if not blocked:
            self.release.set()

    def set_snapshot_validator(self, callback):
        self.validator = callback

    def suggest(self, snapshot, positive, negative, box=None):
        self.calls += 1
        self.entered.set()
        if not self.release.wait(5):
            raise RuntimeError('test failed to release native inference')
        self.finished.set()
        if self.fail:
            raise RuntimeError('native model failed')
        return [PolygonSuggestion([[.1, .1], [.8, .1], [.5, .8]], .9)]

    def invalidate(self, camera):
        pass

    def clear_cache(self):
        pass

    def purge_expired(self):
        pass


@contextmanager
def assistance(*, blocked=True, fail=False, available=True):
    with api() as (client, engine, scene, readers, clock):
        native = ControlledSegmenter(blocked=blocked, fail=fail)
        service = SceneAssistanceService(scene, native if available else None, monotonic=clock)
        client.app.state.scene_assistance = service
        try:
            yield client, engine, scene, readers, clock, service, native
        finally:
            native.release.set()
            result = service.close()
            assert result.closed, result


def body(snapshot, **extra):
    return dict(snapshot_id=snapshot['snapshot_id'], kind='floor', positive_points=[[.5, .5]],
                negative_points=[], box=None, **extra)


def submit(client, snapshot, **kwargs):
    return client.json('POST', '/api/cameras/1/scene/suggestions', body(snapshot), status=202, **kwargs)


def endpoint(job, camera=1):
    return f'/api/cameras/{camera}/scene/suggestions/{job["job_id"]}'


def terminal(client, job, timeout_seconds=5):
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        result = client.json('GET', endpoint(job))
        if result['state'] in ('SUCCEEDED', 'FAILED', 'CANCELLED'):
            return result
        time.sleep(.005 if timeout_seconds <= 5 else .2)
    raise AssertionError('job never reached terminal state')


@contextmanager
def capture_native_diagnostics(enabled):
    """Instrument one opt-in proof; never retain masks, embeddings or frames."""
    if not enabled:
        yield None
        return
    from app.ai import scene_segmenter as adapter
    original_prompts, original_rank = adapter._prompt_inputs, adapter._rank_polygons
    report = dict(mask_threshold=0., prompt_inputs=None, decoder_outputs=None, candidates=[])

    def prompts(height, width, positive, negative, box):
        feeds = original_prompts(height, width, positive, negative, box)
        report['prompt_inputs'] = {name: feeds[name].tolist()
                                   for name in ('point_coords', 'point_labels', 'orig_im_size')}
        return feeds

    def rank(masks, scores, height, width):
        report['decoder_outputs'] = dict(
            masks_shape=list(getattr(masks, 'shape', ())), masks_dtype=str(getattr(masks, 'dtype', None)),
            scores_shape=list(getattr(scores, 'shape', ())), scores_dtype=str(getattr(scores, 'dtype', None)))
        # A proof admits one job; overwrite rather than accumulating native calls.
        report['candidates'] = []
        return original_rank(masks, scores, height, width, diagnostics=report['candidates'])

    with patch.object(adapter, '_prompt_inputs', prompts), patch.object(adapter, '_rank_polygons', rank):
        yield report


class NativeDiagnosticTests(unittest.TestCase):
    def test_capture_reports_actual_prompts_and_rejections_without_retaining_masks(self):
        import numpy as np
        from app.ai import scene_segmenter as adapter
        masks = np.full((1, 4, 40, 80), -1, np.float32)
        scores = np.array([[.2, .9, .5, .7]], np.float32)
        reference = weakref.ref(masks)
        with capture_native_diagnostics(True) as report:
            adapter._prompt_inputs(40, 80, [[.5, .5]], [[.1, .2]], None)
            self.assertEqual(adapter._rank_polygons(masks, scores, 40, 80), [])
        np.testing.assert_allclose(report['prompt_inputs']['point_coords'],
                                   [[[512., 256.], [102.4, 102.4], [0., 0.]]], rtol=1e-6)
        self.assertEqual(report['prompt_inputs']['point_labels'], [[1., 0., -1.]])
        self.assertEqual(report['prompt_inputs']['orig_im_size'], [40., 80.])
        self.assertEqual(report['decoder_outputs']['masks_shape'], [1, 4, 40, 80])
        self.assertEqual(report['decoder_outputs']['masks_dtype'], 'float32')
        self.assertEqual(len(report['candidates']), 4)
        self.assertEqual([item['mask_index'] for item in report['candidates']], [1, 3, 2, 0])
        self.assertTrue(all(item['rejection_reason'] == 'NO_CONTOURS' for item in report['candidates']))
        del masks
        self.assertIsNone(reference())
        self.assertLess(len(json.dumps(report, allow_nan=False)), 8192)

    def test_opt_in_runner_prints_diagnostics_before_rejecting_empty_native_output(self):
        # Real small ONNX sessions and real ASGI jobs cover the diagnostic runner,
        # not selected-weight MobileSAM quality. No image fixture leaves the tempdir.
        import cv2
        import numpy as np
        from test_scene_segmenter import tiny_bundle
        for foreground in (False, True):
            with self.subTest(foreground=foreground), tempfile.TemporaryDirectory() as directory:
                image = Path(directory, 'synthetic.png')
                self.assertTrue(cv2.imwrite(str(image), np.zeros((40, 80, 3), np.uint8)))
                output = io.StringIO()
                with patch('app.ai.model_bundle.ModelBundle.load',
                           return_value=tiny_bundle(directory, foreground=foreground)), redirect_stdout(output):
                    if foreground:
                        real_model_proof('isolated-tiny-graph', image, [[.5, .5]], [], None, 5,
                                         kind='object', diagnostics=True)
                    else:
                        with self.assertRaises(AssertionError):
                            real_model_proof('isolated-tiny-graph', image, [[.5, .5]], [], None, 5,
                                             kind='object', diagnostics=True)
                report = json.loads(output.getvalue())
                self.assertEqual(report['image_size'], [80, 40])
                self.assertEqual(report['request']['kind'], 'object')
                self.assertEqual(report['result']['state'], 'SUCCEEDED')
                self.assertEqual(len(report['result']['polygons']), 3 if foreground else 0)
                self.assertEqual(len(report['diagnostics']['candidates']), 4)
                if not foreground:
                    self.assertTrue(all(item['rejection_reason'] == 'NO_CONTOURS'
                                        for item in report['diagnostics']['candidates']))


class AssistanceTests(unittest.TestCase):
    def test_routes_states_busy_cancellation_and_no_automatic_save(self):
        with assistance() as (client, _engine, _scene, _readers, _clock, _service, native):
            snap = freeze(client)
            first = submit(client, snap)
            self.assertEqual(set(first), {'job_id', 'snapshot_id', 'state'})
            self.assertEqual(first['state'], 'QUEUED')
            self.assertTrue(native.entered.wait(2))
            self.assertEqual(client.json('GET', endpoint(first))['state'], 'RUNNING')
            queued = submit(client, snap)
            self.assertEqual(client.json('GET', endpoint(queued))['state'], 'QUEUED')
            busy = client.json('POST', '/api/cameras/1/scene/suggestions', body(snap), status=503)
            self.assertEqual(busy['detail']['code'], 'SCENE_BUSY')
            self.assertEqual(client.json('DELETE', endpoint(queued))['state'], 'CANCELLED')
            self.assertEqual(client.json('DELETE', endpoint(queued))['state'], 'CANCELLED')
            client.json('DELETE', endpoint(first))
            native.release.set()
            self.assertEqual(terminal(client, first)['polygons'], [])
            self.assertEqual(client.json('GET', '/api/cameras/1/scene')['status'], 'UNSET')
            self.assertEqual(native.calls, 1)

    def test_success_exact_shape_failure_visibility_and_security(self):
        with assistance(blocked=False) as (client, _e, _s, _r, _c, _jobs, _native):
            snap = freeze(client)
            url = '/api/cameras/1/scene/suggestions'
            client.json('POST', url, body(snap), operator=False, status=403)
            client.json('POST', url, body(snap), headers={'origin': 'http://evil.invalid'}, status=403)
            first = submit(client, snap)
            done = terminal(client, first)
            self.assertEqual(set(done), {'job_id', 'snapshot_id', 'state', 'polygons', 'error_code'})
            self.assertEqual(done['state'], 'SUCCEEDED')
            self.assertIsNone(done['error_code'])
            self.assertTrue(done['polygons'][0]['needs_review'])
            client.json('DELETE', endpoint(first), operator=False, status=403)
            client.json('GET', endpoint(first, camera=2), status=409)
            for value in (dict(body(snap), kind='person'), dict(body(snap), upload='bad'),
                          dict(body(snap), positive_points=[[True, .5]]),
                          dict(body(snap), positive_points=[[10**1000, .5]]),
                          dict(body(snap), negative_points=[[0, 0]]*32)):
                client.json('POST', url, value, status=422)
        with self.assertLogs('app.scene.assistance', level='ERROR') as captured:
            with assistance(blocked=False, fail=True) as (client, _e, _s, _r, _c, _jobs, _native):
                done = terminal(client, submit(client, freeze(client)))
                self.assertEqual((done['state'], done['error_code']), ('FAILED', 'SCENE_INFERENCE_FAILED'))
        self.assertEqual(len(captured.records), 1)
        self.assertIs(captured.records[0].exc_info[0], RuntimeError)
        self.assertEqual(str(captured.records[0].exc_info[1]), 'native model failed')
        with assistance(available=False) as (client, _e, _s, _r, _c, _jobs, _native):
            unavailable = client.json('POST', url, body(freeze(client)), status=503)
            self.assertEqual(unavailable['detail']['code'], 'SCENE_MODEL_UNAVAILABLE')

    def test_source_replacement_during_native_never_publishes_and_has_no_lock_inversion(self):
        with assistance() as (client, _engine, scene, readers, clock, jobs, native):
            snap = freeze(client)
            first = submit(client, snap)
            self.assertTrue(native.entered.wait(2))
            readers[1] = Reader(clock, stream_id=readers[1].buffer.latest().stream_id)
            response = client.json('GET', endpoint(first), status=409)
            self.assertEqual(response['detail']['code'], 'CAMERA_SOURCE_CHANGED')
            scene.invalidate(1)
            native.release.set()
            self.assertEqual(jobs.close().unresolved_running, 0)
            self.assertEqual(native.calls, 1)

    def test_record_cap_evicts_oldest_terminal_and_expiry_returns_410(self):
        with assistance(blocked=False) as (client, _e, _s, readers, clock, _jobs, _native):
            snap = freeze(client)
            first = submit(client, snap)
            terminal(client, first)
            for _ in range(15):
                terminal(client, submit(client, snap))
            invalid = dict(body(snap), snapshot_id=str(uuid4()))
            client.json('POST', '/api/cameras/1/scene/suggestions', invalid, status=410)
            self.assertEqual(client.json('GET', endpoint(first))['state'], 'SUCCEEDED')
            terminal(client, submit(client, snap))
            client.json('GET', endpoint(first), status=410)
            clock.value += 300
            client.json('GET', endpoint({'job_id': str(uuid4())}), status=410)
            readers[1].publish(clock)
            self.assertEqual(submit(client, freeze(client))['state'], 'QUEUED')

    def test_close_is_bounded_honest_and_stops_admission(self):
        with assistance() as (client, _e, _scene, _r, _clock, jobs, native):
            snap = freeze(client)
            first = submit(client, snap)
            self.assertTrue(native.entered.wait(2))
            queued = submit(client, snap)
            start = time.monotonic()
            result = jobs.close(timeout_seconds=.01)
            self.assertLess(time.monotonic() - start, .5)
            self.assertFalse(result.closed)
            self.assertEqual(result.unresolved_running, 1)
            self.assertEqual(result.unresolved_queued, 0)
            client.json('POST', '/api/cameras/1/scene/suggestions', body(snap), status=503)
            self.assertEqual(client.json('GET', endpoint(queued))['state'], 'CANCELLED')
            native.release.set()
            self.assertTrue(jobs.close().closed)
            self.assertEqual(client.json('GET', endpoint(first))['state'], 'CANCELLED')

    def test_native_leases_count_unique_tokens_after_expiry_and_invalidation(self):
        with api() as (client, _engine, scene, readers, clock):
            first = freeze(client)
            with scene.lease_snapshot(first['snapshot_id'], 1) as one:
                with scene.lease_snapshot(first['snapshot_id'], 1) as two:
                    self.assertIs(one, two)
                    clock.value += 300
                    readers[2].publish(clock)
                    second = freeze(client, camera=2)
                    readers[3].publish(clock)
                    freeze(client, camera=3, status=503)
                    scene.invalidate(1)
                # One of two leases exiting must not free the unique-token reservation.
                freeze(client, camera=3, status=503)
                with self.assertRaises(HTTPException) as exc:
                    scene.get_snapshot(first['snapshot_id'], 1)
                self.assertEqual(exc.exception.status_code, 410)
            self.assertEqual(freeze(client, camera=3)['camera_id'], 3)
            self.assertEqual(scene.get_snapshot(second['snapshot_id']).camera_id, 2)

    def test_lease_capacity_survives_running_job_expiry_until_native_returns(self):
        with assistance() as (client, _engine, scene, readers, clock, jobs, native):
            first = submit(client, freeze(client))
            self.assertTrue(native.entered.wait(2))
            clock.value += 300
            readers[2].publish(clock)
            freeze(client, camera=2)
            readers[3].publish(clock)
            freeze(client, camera=3, status=503)
            client.json('GET', endpoint(first), status=410)
            native.release.set()
            self.assertTrue(jobs.close().closed)
            self.assertEqual(freeze(client, camera=3)['camera_id'], 3)

    def test_terminal_records_do_not_retain_retired_native_frames(self):
        with assistance() as (client, _engine, scene, readers, clock, jobs, native):
            snap = freeze(client)
            native_frame = weakref.ref(scene.get_snapshot(snap['snapshot_id']).packet.frame)
            job = submit(client, snap)
            self.assertTrue(native.entered.wait(2))
            readers[1].publish(clock)
            scene.invalidate(1)
            self.assertIsNotNone(native_frame())
            native.release.set()
            self.assertTrue(jobs.close().closed)
            gc.collect()
            self.assertIsNone(native_frame())

    def test_snapshot_database_validation_does_not_hold_job_lock(self):
        with assistance(blocked=False) as (client, _engine, scene, _readers, _clock, jobs, _native):
            snap = freeze(client)
            entered, release, invalidated = threading.Event(), threading.Event(), threading.Event()
            original = scene.get_snapshot
            outcomes = []
            def slow_validation(token, camera=None):
                result = original(token, camera)
                entered.set()
                if not release.wait(5):
                    raise RuntimeError('test did not release snapshot validation')
                return result
            scene.get_snapshot = slow_validation
            def admission():
                try:
                    outcomes.append(submit(client, snap))
                except Exception as exc:
                    outcomes.append(exc)
            worker = threading.Thread(target=admission)
            worker.start()
            try:
                self.assertTrue(entered.wait(2))
                invalidation = threading.Thread(target=lambda: (jobs.invalidate(2), invalidated.set()))
                invalidation.start()
                self.assertTrue(invalidated.wait(1), 'Snapshot database work held the job lock')
            finally:
                release.set()
                scene.get_snapshot = original
                worker.join(5)
                if 'invalidation' in locals():
                    invalidation.join(5)
            self.assertFalse(worker.is_alive())
            self.assertEqual(len(outcomes), 1)
            self.assertIsInstance(outcomes[0], dict, outcomes)
            self.assertEqual(terminal(client, outcomes[0])['state'], 'SUCCEEDED')


def real_model_proof(manifest, image, positive, negative, box, timeout_seconds, *, kind='floor', diagnostics=False):
    """Actual inference and strict acceptance; diagnostics never turn empty into a pass."""
    import cv2
    from app.ai.model_bundle import ModelBundle
    from app.ai.scene_segmenter import SceneSegmenter
    with api() as (client, _engine, scene, readers, clock):
        frame = cv2.imread(str(image))
        if frame is None:
            raise AssertionError('Cannot decode private reference image')
        segmenter = SceneSegmenter(ModelBundle.load(Path(manifest)))
        clock.value = time.monotonic()
        scene._monotonic = time.monotonic
        readers[1].buffer.put(frame, stream_id=readers[1].buffer.latest().stream_id,
                              source_revision=1, received_at=readers[1].buffer.latest().received_at,
                              received_mono=clock())
        jobs = SceneAssistanceService(scene, segmenter)
        client.app.state.scene_assistance = jobs
        try:
            with capture_native_diagnostics(diagnostics) as diagnostic_report:
                snap = freeze(client)
                request = dict(snapshot_id=snap['snapshot_id'], kind=kind, positive_points=positive,
                               negative_points=negative, box=box)
                job = client.json('POST', '/api/cameras/1/scene/suggestions', request, status=202)
                result = terminal(client, job, timeout_seconds)
                # Emit evidence BEFORE the assertion, including legitimate empty
                # proposals. No decoder arrays or private image pixels are printed.
                report = dict(providers=segmenter.providers(), model_id=segmenter.model_id,
                              latency_seconds=segmenter.last_inference_seconds,
                              image_size=[snap['width'], snap['height']],
                              request=dict(kind=kind, positive_points=positive, negative_points=negative, box=box),
                              result=result)
                if diagnostic_report is not None:
                    report['diagnostics'] = diagnostic_report
                print(json.dumps(report, allow_nan=False), flush=True)
                assert result['state'] == 'SUCCEEDED' and result['polygons'], result
        finally:
            assert jobs.close().closed


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest')
    parser.add_argument('--image')
    parser.add_argument('--positive', type=float, nargs=2, action='append', default=[])
    parser.add_argument('--negative', type=float, nargs=2, action='append', default=[])
    parser.add_argument('--box', type=float, nargs=4)
    parser.add_argument('--timeout', type=float, default=290)
    parser.add_argument('--kind', choices=('floor', 'object'), default='floor')
    parser.add_argument('--diagnostics', action='store_true',
                        help='Print four native mask scores, topology counts and exact rejection stages; acceptance stays strict')
    args, remaining = parser.parse_known_args()
    if args.manifest or args.image:
        if not (args.manifest and args.image):
            parser.error('--manifest and --image are required together')
        if not math.isfinite(args.timeout) or not 0 < args.timeout <= 290:
            parser.error('--timeout must be finite and within (0,290] seconds')
        if not args.positive and not args.negative and args.box is None:
            parser.error('Supply an explicitly chosen normalized --positive/--negative point or --box')
        real_model_proof(args.manifest, args.image, args.positive, args.negative, args.box, args.timeout,
                         kind=args.kind, diagnostics=args.diagnostics)
    else:
        if args.diagnostics:
            parser.error('--diagnostics requires --manifest and --image')
        unittest.main(argv=[sys.argv[0], *remaining], verbosity=2)
