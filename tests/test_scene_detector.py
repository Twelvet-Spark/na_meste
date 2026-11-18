"""Deterministic OWLv2 adapter contracts, without loading model weights."""
import hashlib
import importlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


class FakeSession:
    def __init__(self):
        self.logits = np.full((1, 3600, 20), -100, dtype=np.float32)
        self.boxes = np.zeros((1, 3600, 4), dtype=np.float32)
        self.metadata = None
        self.input_name = "pixel_values"
        self.feed = None

    def get_inputs(self):
        return [SimpleNamespace(name=self.input_name)]

    def get_outputs(self):
        return [SimpleNamespace(name=name) for name in ("logits", "boxes")]

    def get_modelmeta(self):
        return SimpleNamespace(custom_metadata_map=self.metadata)

    def run(self, names, feeds):
        assert names == ["logits", "boxes"]
        self.feed = feeds["pixel_values"]
        return [self.logits, self.boxes]

    def candidate(self, patch, prompt, score, box):
        self.logits[0, patch, prompt] = np.log(score / (1 - score))
        self.boxes[0, patch] = box


class SceneDetectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.subject = importlib.import_module("app.ai.scene_detector")
        cls.vocabulary = importlib.import_module("app.ai.scene_vocabulary")
        cls.error = importlib.import_module("app.ai.scene_segmenter").SceneSegmentationError

    def detector(self, session=None):
        session = session or FakeSession()
        session.metadata = {"na_meste.scene_vocabulary": self.vocabulary.VOCABULARY_ID}
        bundle = SimpleNamespace(require=lambda name: Path("fake.onnx"))
        with patch.object(self.subject.ort, "InferenceSession", return_value=session):
            detector = self.subject.SceneObjectDetector(bundle, ["CPUExecutionProvider"])
        return detector, session

    def test_vocabulary_order_and_identity(self):
        vocabulary = self.vocabulary.VOCABULARY
        self.assertEqual(len(vocabulary), 20)
        self.assertEqual(vocabulary[:5], (("a door", "DOOR", "Дверь"),
                         ("an open doorway", "DOOR", "Дверной проём"),
                         ("a glass door", "DOOR", "Стеклянная дверь"),
                         ("a staircase", "PASSAGE", "Лестница"),
                         ("a corridor", "PASSAGE", "Коридор")))
        self.assertEqual(vocabulary[-1], ("a person", "PERSON", "Человек"))
        identity = hashlib.sha256(json.dumps([item[0] for item in vocabulary]).encode()).hexdigest()
        self.assertEqual(self.vocabulary.VOCABULARY_ID, identity)

    def test_preprocessing_uses_rgb_square_padding_area_resize_and_clip_normalization(self):
        detector, session = self.detector()
        frame = np.zeros((3, 6, 3), dtype=np.uint8)
        frame[:] = [10, 30, 90]
        detector.detect(frame)
        square = np.full((6, 6, 3), 127, dtype=np.uint8)
        square[:3] = [90, 30, 10]
        expected = cv2.resize(square, (960, 960), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
        expected = (expected - np.array([.48145466, .4578275, .40821073], np.float32)) / np.array([.26862954, .26130258, .27577711], np.float32)
        np.testing.assert_allclose(session.feed, expected.transpose(2, 0, 1)[None], atol=1e-6)
        self.assertEqual(session.feed.dtype, np.float32)
        self.assertTrue(session.feed.flags.c_contiguous)

    def test_boxes_are_clipped_and_normalized_to_native_frame(self):
        detector, session = self.detector()
        session.candidate(0, 0, .8, [.5, .4, .4, .6])
        session.candidate(1, 5, .7, [.9, .5, .5, 1.0])
        detections = detector.detect(np.zeros((100, 200, 3), np.uint8))
        np.testing.assert_allclose(detections[0].box, [.3, .2, .7, 1], atol=1e-6)
        np.testing.assert_allclose(detections[1].box, [.65, 0, 1, 1], atol=1e-6)
        self.assertEqual((detections[0].prompt_index, detections[0].kind, detections[0].label), (0, "DOOR", "Дверь"))

    def test_thresholds_use_best_prompt_per_patch_and_sort_scores(self):
        detector, session = self.detector()
        candidates = [(0, .149), (1, .151), (3, .151), (5, .199), (6, .201), (19, .201)]
        for patch, (prompt, score) in enumerate(candidates):
            session.candidate(patch, prompt, score, [.1 + patch * .15, .4, .08, .1])
        # A patch produces exactly one label, even when two prompts are confident.
        session.logits[0, 4, 5] = np.log(.3 / .7)
        detections = detector.detect(np.zeros((100, 100, 3), np.uint8))
        self.assertEqual([item.prompt_index for item in detections], [5, 19, 1, 3])
        self.assertEqual([item.score for item in detections], sorted((item.score for item in detections), reverse=True))

    def test_nms_merges_door_passage_and_object_synonyms_but_not_other_groups(self):
        detector, session = self.detector()
        for patch, (prompt, score) in enumerate(((0, .9), (3, .8), (5, .7), (6, .6), (19, .5))):
            session.candidate(patch, prompt, score, [.5, .5, .4, .4])
        detections = detector.detect(np.zeros((100, 100, 3), np.uint8))
        self.assertEqual([item.kind for item in detections], ["DOOR", "OBJECT", "PERSON"])
        self.assertEqual([item.prompt_index for item in detections], [0, 5, 19])
    def test_threshold_boundaries_are_inclusive(self):
        detector, session = self.detector()
        session.candidate(0, 0, .15, [.2, .2, .1, .1])
        session.candidate(1, 5, .20, [.5, .2, .1, .1])
        session.candidate(2, 19, .20, [.8, .2, .1, .1])
        detections = detector.detect(np.zeros((100, 100, 3), np.uint8))
        self.assertEqual({item.prompt_index for item in detections}, {0, 5, 19})


    def test_missing_component_bad_io_and_vocabulary_are_unavailable(self):
        bundle = SimpleNamespace(require=lambda name: Path("fake.onnx"))
        for failure in ("missing", "io", "vocabulary"):
            with self.subTest(failure=failure):
                session = FakeSession()
                session.metadata = {"na_meste.scene_vocabulary": "wrong" if failure == "vocabulary" else self.vocabulary.VOCABULARY_ID}
                if failure == "io":
                    session.input_name = "wrong"
                with patch.object(self.subject.ort, "InferenceSession", side_effect=RuntimeError("missing") if failure == "missing" else None, return_value=session):
                    with self.assertRaises(self.error) as raised:
                        self.subject.SceneObjectDetector(bundle, ["CPUExecutionProvider"])
                self.assertEqual(raised.exception.code, "SCENE_DETECTOR_UNAVAILABLE")


if __name__ == "__main__":
    unittest.main(verbosity=2)
