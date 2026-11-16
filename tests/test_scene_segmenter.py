"""Deterministic adapter tests, not a real MobileSAM accuracy proof.
Run: .venv/Scripts/python.exe tests/test_scene_segmenter.py
"""
import json
import sys
import tempfile
import threading
import unittest
import weakref
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto as T, helper, numpy_helper

from app.ai.scene_segmenter import (SceneSegmenter, SceneSegmentationError,
                                    _preprocess, _prompt_inputs, _mask_polygon,
                                    _rank_polygons, validate_prompts)


def snapshot(clock=lambda: 100., size=(80, 40), camera=1):
    frame = np.zeros((size[1], size[0], 3), np.uint8)
    frame[:, :, 2] = 255
    frame.setflags(write=False)
    return SimpleNamespace(snapshot_id=str(uuid4()), camera_id=camera, width=size[0], height=size[1],
                           expires_mono=clock() + 300,
                           packet=SimpleNamespace(frame=frame, source_revision=1, stream_id=str(uuid4()), seq=1))


def tiny_bundle(directory, size=(80, 40), *, foreground=True):
    """Real ORT graphs exercise named IO/rank contracts without selected weights."""
    def save(name, inputs, outputs, nodes, initializers):
        graph = helper.make_graph(nodes, name, inputs, outputs, initializers)
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 16)])
        model.ir_version = 8
        path = Path(directory, name + '.onnx')
        onnx.save(model, path)
        return path
    def value(name, shape):
        return helper.make_tensor_value_info(name, T.FLOAT, shape)
    def tensor(name, data):
        return numpy_helper.from_array(np.asarray(data), name)
    encoder = save('encoder', [value('images', [1, 3, 1024, 1024])],
                   [value('image_embeddings', [1, 256, 64, 64])],
                   [helper.make_node('ReduceMean', ['images'], ['mean'], keepdims=0),
                    helper.make_node('Expand', ['mean', 'shape'], ['image_embeddings'])],
                   [tensor('shape', np.array([1, 256, 64, 64], np.int64))])
    masks = np.full((1, 4, size[1], size[0]), -1, np.float32)
    if foreground:
        masks[0, :, 8:32, 16:64] = 1
    decoder = save('decoder', [value('image_embeddings', [1, 256, 64, 64]),
                              value('point_coords', [1, 'points', 2]), value('point_labels', [1, 'points']),
                              value('mask_input', [1, 1, 256, 256]), value('has_mask_input', [1]),
                              value('orig_im_size', [2])],
                   [value('masks', [1, 4, size[1], size[0]]), value('iou_predictions', [1, 4]),
                    value('low_res_masks', [1, 4, 256, 256])],
                   [helper.make_node('Identity', ['mask'], ['masks']),
                    helper.make_node('Identity', ['scores'], ['iou_predictions']),
                    helper.make_node('Identity', ['low'], ['low_res_masks'])],
                   [tensor('mask', masks), tensor('scores', np.array([[.2, .9, .5, .7]], np.float32)),
                    tensor('low', np.zeros((1, 4, 256, 256), np.float32))])
    return SimpleNamespace(require=lambda component: encoder,
                           component_paths={'scene_encoder': encoder, 'scene_decoder': decoder},
                           metadata={'components': {'scene': {'preprocessing_id': 'test', 'files': [
                               {'role': 'encoder', 'sha256': 'a' * 64},
                               {'role': 'decoder', 'sha256': 'b' * 64}]}}})


class SegmenterTests(unittest.TestCase):
    def test_rgb_pillow_resize_mean_std_and_bottom_padding(self):
        original = snapshot().packet.frame
        data = _preprocess(original)
        self.assertEqual(data.shape, (1, 3, 1024, 1024))
        self.assertEqual(data.dtype, np.float32)
        np.testing.assert_allclose(data[0, :, 0, 0], [(255-123.675)/58.395, -116.28/57.12, -103.53/57.375], rtol=1e-6)
        self.assertTrue(np.all(data[:, :, 512:, :] == 0))
        self.assertFalse(original.flags.writeable)
        portrait = _preprocess(snapshot(size=(3, 7)).packet.frame)
        self.assertTrue(np.all(portrait[:, :, :, 439:] == 0))

    def test_prompt_mapping_uses_rounded_resize_and_official_padding_labels(self):
        feeds = _prompt_inputs(7, 3, [[.25, .5]], [[1., 0.]], None)
        np.testing.assert_allclose(feeds['point_coords'], [[[109.75, 512], [439, 0], [0, 0]]])
        np.testing.assert_array_equal(feeds['point_labels'], [[1, 0, -1]])
        boxed = _prompt_inputs(7, 3, [], [], [.1, .2, .8, .9])
        np.testing.assert_allclose(boxed['point_coords'], [[[43.9, 204.8], [351.2, 921.6]]], rtol=1e-6)
        np.testing.assert_array_equal(boxed['point_labels'], [[2, 3]])
        np.testing.assert_array_equal(boxed['orig_im_size'], [7, 3])
        self.assertEqual(boxed['mask_input'].shape, (1, 1, 256, 256))
        self.assertTrue(all(value.dtype == np.float32 for value in boxed.values()))

    def test_strict_prompts_reject_nonfinite_boolean_empty_and_overbudget(self):
        bad = [([], [], None), ([[True, .5]], [], None), ([[float('nan'), 0]], [], None),
               ([[1.01, 0]], [], None), ([[0, 0]]*17, [[0, 0]]*16, None),
               ([], [], [0, 0, 0, 1]), ([], [], [0, 1, 1, 0])]
        for positive, negative, box in bad:
            with self.subTest(positive=positive, box=box), self.assertRaises(ValueError):
                validate_prompts(positive, negative, box)
        validate_prompts([[0, 0]]*32, [], [0, 0, 1, 1])

    def test_contours_do_not_invent_rectangles_or_bridge_regions_and_holes(self):
        thin = np.zeros((40, 80), np.uint8)
        thin[20, 5:50] = 1
        self.assertIsNone(_mask_polygon(thin))
        split = np.zeros_like(thin)
        split[3:14, 4:15] = 1
        split[25:36, 60:74] = 1
        self.assertIsNone(_mask_polygon(split))
        holed = np.ones_like(thin)
        holed[10:20, 20:30] = 0
        self.assertIsNone(_mask_polygon(holed))
        touching = np.zeros_like(thin)
        touching[3:14, 3:14] = 1
        touching[13:24, 13:24] = 1
        self.assertIsNone(_mask_polygon(touching))
        diagonal = np.zeros_like(thin)
        diagonal[3:14, 3:14] = 1
        diagonal[14:25, 14:25] = 1
        self.assertIsNone(_mask_polygon(diagonal))
        # Two pixels of shared width provide genuine positive-area continuity;
        # rejecting every narrow or concave outline would hide the topology bug.
        connected = np.zeros_like(thin)
        connected[3:14, 3:14] = 1
        connected[12:23, 12:23] = 1
        self.assertIsNotNone(_mask_polygon(connected))
        concave = np.zeros_like(thin)
        concave[3:30, 3:14] = 1
        concave[20:30, 3:55] = 1
        outline = _mask_polygon(concave)
        self.assertIsNotNone(outline)
        from app.scene_geometry import point_in_polygon
        self.assertTrue(point_in_polygon([.1, .2], outline))
        self.assertTrue(point_in_polygon([.5, .6], outline))
        self.assertFalse(point_in_polygon([.4, .2], outline))
        triangle = np.zeros_like(thin)
        cv2.fillPoly(triangle, [np.array([[5, 5], [65, 8], [25, 33]])], 1)
        points = _mask_polygon(triangle)
        self.assertGreaterEqual(len(points), 3)
        self.assertLessEqual(len(points), 128)
        self.assertTrue(all(0 <= x <= 1 and 0 <= y <= 1 for x, y in points))

    def test_mask_diagnostics_distinguish_empty_split_holes_and_point_contact(self):
        # The report must explain the same strict decisions, never repair masks.
        empty = np.zeros((40, 80), np.uint8)
        split = empty.copy()
        split[3:14, 4:15] = 1
        split[25:36, 60:74] = 1
        holed = empty.copy()
        holed[3:30, 4:35] = 1
        holed[10:20, 15:25] = 0
        touching = empty.copy()
        touching[3:14, 3:14] = 1
        touching[13:24, 13:24] = 1
        thin = empty.copy()
        thin[20, 5:50] = 1
        for mask, pixels, components, holes, reason in (
                (empty, 0, 0, 0, 'NO_CONTOURS'),
                (split, 275, 2, 0, 'MULTIPLE_CONTOURS'),
                (holed, 737, 1, 1, 'MULTIPLE_CONTOURS'),
                (touching, 241, 1, 0, 'DISCONNECTED_POSITIVE_AREA_SUPPORT'),
                (thin, 45, 1, 0, 'ZERO_CONTOUR_AREA')):
            with self.subTest(reason=reason, holes=holes):
                detail = {}
                self.assertIsNone(_mask_polygon(mask, diagnostics=detail))
                self.assertEqual(detail['foreground_pixels'], pixels)
                self.assertEqual(detail['foreground_components_8'], components)
                self.assertEqual(detail['hole_count'], holes)
                self.assertEqual(detail['rejection_reason'], reason)
                self.assertFalse(detail['polygon_valid'])
                json.dumps(detail, allow_nan=False)
        self.assertEqual(detail['contour_count'], 1)

    def test_mask_diagnostics_count_nested_islands_without_filling_holes(self):
        mask = np.zeros((40, 80), np.uint8)
        mask[2:38, 2:78] = 1
        mask[8:32, 8:72] = 0
        mask[14:26, 14:66] = 1
        mask[18:22, 20:60] = 0
        detail = {}
        self.assertIsNone(_mask_polygon(mask, diagnostics=detail))
        self.assertEqual((detail['contour_count'], detail['foreground_components_8'], detail['hole_count']),
                         (4, 2, 2))
        self.assertEqual(detail['rejection_reason'], 'MULTIPLE_CONTOURS')

    def test_rank_diagnostics_cover_all_native_masks_without_changing_proposals(self):
        masks = np.full((1, 4, 40, 80), -1, np.float32)
        masks[:, :, 8:32, 16:64] = 1
        scores = np.array([[.2, .9, .5, .7]], np.float32)
        details = []
        normal = _rank_polygons(masks, scores, 40, 80)
        reported = _rank_polygons(masks, scores, 40, 80, diagnostics=details)
        self.assertEqual(normal, reported)
        self.assertEqual(len(details), 4)
        self.assertEqual([item['mask_index'] for item in details], [1, 3, 2, 0])
        np.testing.assert_allclose([item['native_score'] for item in details], [.9, .7, .5, .2])
        self.assertEqual([item['selected'] for item in details], [True, True, True, False])
        self.assertTrue(all(item['polygon_valid'] and item['rejection_reason'] is None for item in details))
        self.assertTrue(all(item['foreground_pixels'] == 1152 and item['vertex_count'] == 4 for item in details))
        self.assertTrue(all(len(item['simplification_attempts']) <= 6 for item in details))
        json.dumps(details, allow_nan=False)

    def test_rank_diagnostics_expose_rejected_scores_and_do_not_keep_native_arrays(self):
        masks = np.full((1, 4, 40, 80), -1, np.float32)
        masks[0, 1, 8:32, 16:64] = 1
        scores = np.array([[.9, .8, .7, .6]], np.float32)
        reference = weakref.ref(masks)
        details = []
        result = _rank_polygons(masks, scores, 40, 80, diagnostics=details)
        self.assertEqual(len(result), 1)
        self.assertEqual([item['rejection_reason'] for item in details],
                         ['NO_CONTOURS', None, 'NO_CONTOURS', 'NO_CONTOURS'])
        del masks
        self.assertIsNone(reference())
        self.assertLess(len(json.dumps(details, allow_nan=False)), 8192)

    def test_output_rank_shape_finite_checks_and_three_score_order(self):
        masks = np.full((1, 4, 40, 80), -1, np.float32)
        masks[:, :, 8:32, 16:64] = 1
        result = _rank_polygons(masks, np.array([[.2, .9, .5, .7]], np.float32), 40, 80)
        np.testing.assert_allclose([p.score for p in result], [.9, .7, .5])
        self.assertTrue(all(p.needs_review is True for p in result))
        for wrong in (masks[0], masks[:, :, :-1], masks[:, :3]):
            with self.assertRaises(SceneSegmentationError):
                _rank_polygons(wrong, np.ones((1, 4), np.float32), 40, 80)
        for wrong_scores in (np.ones(4, np.float32), np.ones((1, 3), np.float32),
                             np.full((1, 4), np.inf, np.float32)):
            with self.assertRaises(SceneSegmentationError):
                _rank_polygons(masks, wrong_scores, 40, 80)
        masks[0, 0, 0, 0] = np.nan
        with self.assertRaises(SceneSegmentationError):
            _rank_polygons(masks, np.ones((1, 4), np.float32), 40, 80)

    def test_real_tiny_onnx_cpu_sessions_cache_and_invalidation(self):
        now = [100.]
        valid = set()
        calls = []
        original_session = ort.InferenceSession
        def factory(*args, **kwargs):
            self.assertEqual(kwargs['providers'], ['CPUExecutionProvider'])
            self.assertEqual(kwargs['sess_options'].intra_op_num_threads, 2)
            native = original_session(*args, **kwargs)
            return SimpleNamespace(get_inputs=native.get_inputs, get_outputs=native.get_outputs,
                                   get_providers=native.get_providers,
                                   run=lambda names, feeds: (calls.append((Path(args[0]).stem, feeds)), native.run(names, feeds))[1])
        with tempfile.TemporaryDirectory() as directory, patch('app.ai.scene_segmenter.ort.InferenceSession', factory):
            segmenter = SceneSegmenter(tiny_bundle(directory), monotonic=lambda: now[0])
            def validate(token, camera):
                if token not in valid:
                    raise ValueError('retired snapshot')
            segmenter.set_snapshot_validator(validate)
            first = snapshot(lambda: now[0]); valid.add(first.snapshot_id)
            self.assertEqual(segmenter.providers(), {'encoder': ['CPUExecutionProvider'], 'decoder': ['CPUExecutionProvider']})
            result = segmenter.suggest(first, [[.5, .5]], [])
            self.assertEqual(len(result), 3)
            xs, ys = zip(*result[0].points)
            np.testing.assert_allclose([min(xs), max(xs), min(ys), max(ys)], [.2, .7875, .2, .775])
            segmenter.suggest(first, [[.6, .5]], [])
            self.assertEqual([name for name, _ in calls].count('encoder'), 1)
            self.assertEqual(calls[-1][1]['point_coords'][0, 0, 0], np.float32(614.4))
            segmenter.invalidate(1)
            segmenter.suggest(first, [[.5, .5]], [])
            self.assertEqual([name for name, _ in calls].count('encoder'), 2)
            now[0] = 400.
            with self.assertRaises(SceneSegmentationError):
                segmenter.suggest(first, [[.5, .5]], [])
            segmenter.clear_cache()

    def test_cache_ttl_releases_embedding_without_another_inference(self):
        now, references = [100.], []
        original_session = ort.InferenceSession
        def factory(*args, **kwargs):
            native = original_session(*args, **kwargs)
            def run(names, feeds):
                outputs = native.run(names, feeds)
                if Path(args[0]).stem == 'encoder':
                    references.append(weakref.ref(outputs[0]))
                return outputs
            return SimpleNamespace(get_inputs=native.get_inputs, get_outputs=native.get_outputs,
                                   get_providers=native.get_providers, run=run)
        with tempfile.TemporaryDirectory() as directory, patch('app.ai.scene_segmenter.ort.InferenceSession', factory):
            segmenter = SceneSegmenter(tiny_bundle(directory), monotonic=lambda: now[0])
            segmenter.set_snapshot_validator(lambda token, camera: None)
            segmenter.suggest(snapshot(), [[.5, .5]], [])
            self.assertIsNotNone(references[0]())
            now[0] = 400.
            segmenter.purge_expired()
            self.assertIsNone(references[0]())

    def test_cache_retains_only_two_embeddings_and_reencodes_evicted_token(self):
        calls = []
        original_session = ort.InferenceSession
        def factory(*args, **kwargs):
            native = original_session(*args, **kwargs)
            def run(names, feeds):
                calls.append(Path(args[0]).stem)
                return native.run(names, feeds)
            return SimpleNamespace(get_inputs=native.get_inputs, get_outputs=native.get_outputs,
                                   get_providers=native.get_providers, run=run)
        with tempfile.TemporaryDirectory() as directory, patch('app.ai.scene_segmenter.ort.InferenceSession', factory):
            segmenter = SceneSegmenter(tiny_bundle(directory), monotonic=lambda: 100.)
            segmenter.set_snapshot_validator(lambda token, camera: None)
            images = [snapshot() for _ in range(3)]
            for image in images:
                segmenter.suggest(image, [[.5, .5]], [])
            segmenter.suggest(images[0], [[.5, .5]], [])
            self.assertEqual(calls.count('encoder'), 4)

    def test_stale_after_either_native_stage_drops_embedding_before_reuse(self):
        original_session = ort.InferenceSession
        for phase in ('encoder', 'decoder'):
            with self.subTest(phase=phase):
                valid = [True]
                retire = [True]
                calls = []
                def factory(*args, **kwargs):
                    native = original_session(*args, **kwargs)
                    def run(names, feeds):
                        name = Path(args[0]).stem
                        calls.append(name)
                        result = native.run(names, feeds)
                        if name == phase and retire[0]:
                            valid[0] = False
                        return result
                    return SimpleNamespace(get_inputs=native.get_inputs, get_outputs=native.get_outputs,
                                           get_providers=native.get_providers, run=run)
                def validate(token, camera):
                    if not valid[0]:
                        raise ValueError('reader changed during native inference')
                with tempfile.TemporaryDirectory() as directory, patch('app.ai.scene_segmenter.ort.InferenceSession', factory):
                    segmenter = SceneSegmenter(tiny_bundle(directory), monotonic=lambda: 100.)
                    segmenter.set_snapshot_validator(validate)
                    image = snapshot()
                    with self.assertRaises(ValueError):
                        segmenter.suggest(image, [[.5, .5]], [])
                    valid[0], retire[0] = True, False
                    self.assertEqual(len(segmenter.suggest(image, [[.5, .5]], [])), 3)
                    self.assertEqual(calls.count('encoder'), 2)

    def test_invalidation_never_waits_on_native_and_concurrent_suggest_is_busy(self):
        entered, release = threading.Event(), threading.Event()
        original_session = ort.InferenceSession
        errors = []
        def factory(*args, **kwargs):
            native = original_session(*args, **kwargs)
            def run(names, feeds):
                if Path(args[0]).stem == 'encoder':
                    entered.set()
                    if not release.wait(5):
                        raise RuntimeError('test did not release encoder')
                return native.run(names, feeds)
            return SimpleNamespace(get_inputs=native.get_inputs, get_outputs=native.get_outputs,
                                   get_providers=native.get_providers, run=run)
        with tempfile.TemporaryDirectory() as directory, patch('app.ai.scene_segmenter.ort.InferenceSession', factory):
            segmenter = SceneSegmenter(tiny_bundle(directory), monotonic=lambda: 100.)
            segmenter.set_snapshot_validator(lambda token, camera: None)
            image = snapshot()
            def native():
                try:
                    segmenter.suggest(image, [[.5, .5]], [])
                except Exception as exc:
                    errors.append(exc)
            worker = threading.Thread(target=native)
            worker.start()
            try:
                self.assertTrue(entered.wait(2))
                with self.assertRaises(SceneSegmentationError) as busy:
                    segmenter.suggest(image, [[.5, .5]], [])
                self.assertEqual(busy.exception.code, 'SCENE_BUSY')
                segmenter.invalidate(1)
            finally:
                release.set()
                worker.join(5)
            self.assertFalse(worker.is_alive())
            self.assertEqual(len(errors), 1)
            self.assertEqual(errors[0].code, 'CAMERA_SOURCE_CHANGED')


if __name__ == '__main__':
    unittest.main(verbosity=2)
