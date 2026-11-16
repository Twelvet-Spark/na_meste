"""Labeled replay scoring: real accounting/quality math, native face inference doubled.
Run: .venv/Scripts/python.exe tests/test_cctv_identity.py
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools import evaluate_cctv as runner


class NativeFaces:
    model_id = "quality-fixture"

    def __init__(self):
        texture = (np.indices((112, 112)).sum(axis=0) % 2 * 255).astype(np.uint8)
        self.aligned = np.repeat(texture[:, :, None], 3, axis=2)
        self.score = .95
        self.calls = 0
        self.ambiguous = False

    def detect_faces(self, crop):
        self.calls += 1
        face = SimpleNamespace(native_bbox=np.array([40., 20., 100., 80.]),
                               aligned=self.aligned, det_score=self.score, model_id=self.model_id,
                               landmarks=np.array([[50., 35.], [85., 35.], [68., 50.], [55., 65.], [82., 65.]]))
        return [face, face] if self.ambiguous else [face]


def person(label="known", **changes):
    return dict(track_label=label, bbox=[.1, .1, .9, .9], face_bbox=[.3, .2, .6, .5],
                face_visible=True, occluded=False, seated=False, **changes)


class IdentityTests(unittest.TestCase):
    def setup_case(self, times=(0., 1., 2.), *, subject="alice", labels=True, people=None, cameras=1):
        rows = people if people is not None else [[person()] for _ in times]
        clip = runner.Clip(Path("private-fixture.mkv"), "view", 0., None,
                           dict(version=1, frames=[dict(time_seconds=t, people=p) for t, p in zip(times, rows)], crossings=[]))
        specs = [runner.ReplaySpec(camera, clip, runner.ClipProbe(200, 200, 15., 20., "fixture"),
                                   0., f"stream-{camera}", False, 5.) for camera in range(1, cameras + 1)]
        self.annotations = runner.AnnotationSamples(specs)
        self.faces = NativeFaces()
        self.scorer = runner.IdentityEvaluator(specs, [dict(source_camera_key="view", track_label="known", subject_id=subject)]
                                               if labels else [], {7: "alice", 8: "bob"}, self.faces,
                                               ordinary_interval=.5, attendance_interval=.25)
        self.frame = np.zeros((200, 200, 3), np.uint8)
        self.frame.flags.writeable = False
        return self.scorer

    def observe(self, stamp, seq, employee=None, *, camera=1, stream=None, prepare=True,
                tracks=True, uncertain=False, generation="generation", extra=False):
        source = runner.ReplayFrame(camera, stream or f"stream-{camera}", seq, stamp, stamp + 100, None, None)
        self.annotations.observe_decoded(source)
        if prepare:
            self.scorer.capture_quality(camera, runner.ReplayFrame(
                camera, source.stream_id, seq, stamp, source.received_mono, None, self.frame))
        prediction = SimpleNamespace(track_id=1, bbox=(.1, .1, .9, .9), employee_id=employee,
                                     association_uncertain=uncertain, fresh_face_evidence=None)
        detections = [prediction] if tracks else []
        if extra:
            detections.append(SimpleNamespace(**{**vars(prediction), "track_id": 2}))
        self.scorer.observe(camera, source, SimpleNamespace(tracking_generation=generation, observed_tracks=detections))
        return source

    def report(self):
        return self.scorer.summary(self.annotations)

    def complete_labeled_case(self, between_employee=None):
        rows = [[person()] for _ in range(3)] + [[person("visitor")] for _ in range(3)]
        self.setup_case((0., 1., 2., 3., 4., 5.), people=rows)
        self.scorer.labels[("view", "visitor")] = None
        for seq, stamp in enumerate((0., .5, 1., 2., 3., 4., 5.), 1):
            employee = between_employee if stamp == .5 else 7 if stamp <= 2 else None
            self.observe(stamp, seq, employee)
        return self.report()

    def deployment(self, recognition):
        # Isolate the recognition input to the real final aggregation; these
        # other gate results are deterministic fixtures, not measured capacity.
        return runner.deployment_acceptance({
            "capacity": {"status": "PASS"}, "journal": {"status": "PASS"},
            "detection_tracking": {"status": "PASS"}, "target_hardware": {"certified": True},
            "recognition": recognition}, gpu_contaminated=False)

    def test_declared_held_out_contract_allows_complete_attributable_validation(self):
        result = self.complete_labeled_case()
        self.assertEqual(result["recall"], 1.)
        self.assertEqual(result["wrong_confirmations"], 0)
        self.assertEqual(result["unknown_false_assignment_visits"], 0)
        self.assertEqual(result["unattributed_confirmations"], 0)
        self.assertTrue(result["certified"])
        self.assertEqual(self.deployment(result)["status"], "PASS")

    def test_between_annotation_confirmation_remains_incomplete_after_correction(self):
        result = self.complete_labeled_case(between_employee=8)
        self.assertEqual(result["recall"], 1.)
        # No interpolated GT identity: the event is unscored, not labelled wrong.
        self.assertEqual(result["wrong_confirmations"], 0)
        self.assertEqual(result["confirmed_observations_outside_annotation_tolerance"], 1)
        self.assertEqual(result["unattributed_confirmations"], 1)
        self.assertIn("UNATTRIBUTED_CONFIRMED_IDENTITIES", result["prerequisites"])
        self.assertEqual(result["status"], "NOT_EVALUATED")
        self.assertFalse(result["certified"])
        self.assertEqual(self.deployment(result)["status"], "NOT_ACCEPTED")
        # Re-delivery/cached correction must not erase the earlier evidence.
        self.observe(.5, 2, 7, prepare=False)
        self.assertEqual(self.report()["unattributed_confirmations"], 1)
        self.assertFalse(self.report()["certified"])

    def test_confirmed_unscored_visit_blocks_otherwise_complete_validation(self):
        self.complete_labeled_case()
        self.observe(.5, 1, 7, stream="unscored-visit", prepare=False)
        result = self.report()
        self.assertEqual(result["eligible_known_visits"], 1)
        self.assertEqual(result["recall"], 1.)
        self.assertEqual(result["unattributed_confirmations"], 1)
        self.assertEqual(result["wrong_confirmations"], 0)
        self.assertEqual(result["status"], "NOT_EVALUATED")
        self.assertEqual(self.deployment(result)["status"], "NOT_ACCEPTED")

    def test_attribution_failures_block_certification_without_guessing_identity(self):
        for mode in ("ambiguous", "uncertain"):
            with self.subTest(mode=mode):
                self.complete_labeled_case()
                self.observe(1., 20, 7, extra=mode == "ambiguous", uncertain=mode == "uncertain")
                result = self.report()
                self.assertEqual(result["recall"], 1.)
                self.assertGreater(result["unattributed_confirmations"], 0)
                self.assertEqual(result["wrong_confirmations"], 0)
                self.assertEqual(self.deployment(result)["status"], "NOT_ACCEPTED")

    def test_offline_native_opportunities_count_even_with_zero_admitted_or_published_results(self):
        self.setup_case()
        for seq, stamp in enumerate((0., 1., 2.), 1):
            native = runner.ReplayFrame(1, "independent-quality-pass", seq, stamp, stamp + 10, None, self.frame)
            self.scorer.capture_quality(1, native)
            self.annotations.observe_decoded(runner.ReplayFrame(1, "stream-1", seq, stamp, stamp + 100, None, None))
        # No worker batch, scorer.observe(), detection or recognition is needed
        # to establish this eligible denominator from independently decoded frames.
        result = self.report()
        self.assertEqual(result["eligible_known_visits"], 1)
        self.assertEqual(result["confirmed_eligible_known_visits"], 0)
        self.assertEqual(result["recall"], 0.)
        self.assertEqual(result["recall_status"], "EVALUATED")
        self.assertEqual(result["status"], "FAIL")

    def test_real_ffmpeg_prepass_preserves_eligibility_with_no_live_inference(self):
        import subprocess
        import tempfile
        from dataclasses import replace
        try:
            ffmpeg, _ffprobe = runner.resolve_media_tools()
        except runner.EvaluationError:
            self.skipTest("Local FFmpeg unavailable")
        self.setup_case()
        with tempfile.TemporaryDirectory(prefix="identity-prepass-test-") as directory:
            clip_path = Path(directory) / "generated.mkv"
            subprocess.run([ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
                            "-f", "lavfi", "-i", "color=c=black:s=200x200:r=15:d=2.2",
                            "-c:v", "ffv1", str(clip_path)], check=True, capture_output=True, timeout=30,
                           **runner._containment())
            original = self.scorer.specs[1]
            spec = replace(original, clip=replace(original.clip, path=clip_path),
                           probe=runner.ClipProbe(200, 200, 15., 2.2, "ffv1"))
            self.annotations = runner.AnnotationSamples([spec])
            self.scorer = runner.IdentityEvaluator(
                [spec], [dict(source_camera_key="view", track_label="known", subject_id="alice")],
                {7: "alice"}, self.faces, ordinary_interval=.5, attendance_interval=.25)
            self.scorer.precompute(ffmpeg)
            for seq, stamp in enumerate((0., 1., 2.), 1):
                self.annotations.observe_decoded(runner.ReplayFrame(1, "stream-1", seq, stamp, stamp + 100, None, None))
            result = self.report()
            self.assertEqual(result["quality_prepass"]["decoded_clips"], 1)
            self.assertGreaterEqual(result["quality_prepass"]["decoded_frames"], 30)
            self.assertEqual(result["eligible_known_visits"], 1)
            self.assertEqual(result["recall"], 0.)
            self.assertEqual(result["status"], "FAIL")

    def test_eligible_correct_and_never_confirmed_real_miss(self):
        for assigned, expected in ((7, 1.), (None, 0.)):
            with self.subTest(assigned=assigned):
                self.setup_case()
                for seq, stamp in enumerate((0., 1., 2.), 1):
                    self.observe(stamp, seq, assigned if seq == 3 else None)
                result = self.report()
                self.assertEqual(result["eligible_known_visits"], 1)
                self.assertEqual(result["recall"], expected)
                self.assertEqual(result["status"], "PASS" if assigned else "FAIL")
                self.assertEqual(result["wrong_confirmations"], 0)
                if assigned:
                    self.assertEqual(result["time_to_first_correct_confirmation_seconds"]["p50"], 2.)

    def test_wrong_known_remains_failure_after_correct_confirmation(self):
        self.setup_case()
        for seq, employee in enumerate((8, None, 7), 1):
            self.observe(float(seq - 1), seq, employee)
        result = self.report()
        self.assertEqual(result["recall"], 1.)
        self.assertEqual(result["wrong_confirmations"], 1)
        self.assertEqual(result["wrong_known_visits"], 1)
        self.assertEqual(result["status"], "FAIL")

    def test_unknowns_never_enter_recall_or_recognition_success(self):
        for employee in (None, 7):
            with self.subTest(employee=employee):
                self.setup_case(subject=None)
                for seq, stamp in enumerate((0., 1., 2.), 1):
                    self.observe(stamp, seq, employee)
                result = self.report()
                self.assertEqual(result["eligible_known_visits"], 0)
                self.assertIsNone(result["recall"])
                self.assertEqual(result["recall_status"], "NOT_EVALUATED")
                self.assertEqual(result["unknown_false_assignment_visits"], int(employee is not None))
                self.assertEqual(result["all_visit_identification_coverage"], 0.)
                self.assertEqual(result["status"], "FAIL" if employee else "NOT_EVALUATED")

    def test_ineligible_known_wrong_assignment_is_still_reported(self):
        self.setup_case()
        self.faces.aligned = np.zeros((112, 112, 3), np.uint8)
        for seq, stamp in enumerate((0., 1., 2.), 1):
            self.observe(stamp, seq, 8)
        result = self.report()
        self.assertEqual(result["ineligible_known_visits"], 1)
        self.assertEqual(result["rejection_counts"]["BLURRY_FACE"], 3)
        self.assertEqual(result["wrong_confirmations"], 1)
        self.assertIsNone(result["recall"])

    def test_quality_eligibility_is_independent_of_failed_person_association(self):
        for mode in ("missing", "uncertain", "ambiguous"):
            with self.subTest(mode=mode):
                self.setup_case()
                for seq, stamp in enumerate((0., 1., 2.), 1):
                    self.observe(stamp, seq, 7, tracks=mode != "missing", uncertain=mode == "uncertain", extra=mode == "ambiguous")
                result = self.report()
                self.assertEqual(result["eligible_known_visits"], 1)
                self.assertEqual(result["recall"], 0.)
                self.assertEqual(result["correct_known_visits"], 0)

    def test_cached_evidence_and_replayed_packets_do_not_create_quality_votes(self):
        self.setup_case()
        self.observe(0., 1, 7)
        self.observe(1., 1, 7)
        self.observe(2., 1, 7)
        self.assertEqual(self.report()["eligible_known_visits"], 0)
        self.assertEqual(self.faces.calls, 1)
        self.setup_case()
        self.observe(0., 1, 7)
        self.observe(1., 2, 7, prepare=False)
        self.observe(2., 3, 7, prepare=False)
        self.assertEqual(self.report()["eligible_known_visits"], 0)
        self.assertEqual(self.report()["eligibility_undetermined_known_visits"], 1)

    def test_distinct_opportunities_respect_window_and_configured_cadence(self):
        for times, expected in (((0., .5, 1.), 1), ((0., 1.1, 2.2), 0), ((0., .1, .2), 0)):
            with self.subTest(times=times):
                self.setup_case(times)
                for seq, stamp in enumerate(times, 1):
                    self.observe(stamp, seq, None)
                self.assertEqual(self.report()["eligible_known_visits"], expected)

    def test_camera_stream_and_gt_absence_separate_validation_visits(self):
        self.setup_case(cameras=2)
        for camera in (1, 2):
            for stream in (f"stream-{camera}", f"reconnected-{camera}"):
                for seq, stamp in enumerate((0., 1., 2.), 1):
                    self.observe(stamp, seq, 7, camera=camera, stream=stream)
        self.assertEqual(self.report()["eligible_known_visits"], 4)
        self.setup_case((0., .5, 1., 1.5, 2.), people=[[person()], [person()], [], [person()], [person()]])
        for seq, stamp in enumerate((0., .5, 1., 1.5, 2.), 1):
            self.observe(stamp, seq, 7, tracks=stamp != 1.)
        self.assertEqual(self.report()["known_visits"], 2)
        self.assertEqual(self.report()["eligible_known_visits"], 0)

    def test_known_only_metrics_do_not_certify_unobserved_unknown_visitors(self):
        self.setup_case()
        for seq, stamp in enumerate((0., 1., 2.), 1):
            self.observe(stamp, seq, 7)
        result = self.report()
        self.assertEqual(result["status"], "PASS")
        self.assertFalse(result["certified"])
        self.assertIsNone(result["unknown_false_assignment_visits"])
        self.assertEqual(result["unknown_false_assignment_status"], "NOT_EVALUATED")
        self.assertEqual(self.deployment(result)["status"], "NOT_ACCEPTED")
        rows = [[person()] for _ in range(3)] + [[person("visitor")] for _ in range(3)]
        self.setup_case((0., 1., 2., 3., 4., 5.), people=rows)
        self.scorer.labels[("view", "visitor")] = None
        for seq, stamp in enumerate((0., 1., 2., 3., 4., 5.), 1):
            self.observe(stamp, seq, 7 if stamp <= 2 else None)
        result = self.report()
        self.assertTrue(result["certified"])
        self.assertEqual(result["unknown_false_assignment_visits"], 0)
        self.assertEqual(result["all_visit_identification_coverage"], .5)

    def test_no_labels_zero_denominator_and_partial_labels_are_not_certified(self):
        self.setup_case(labels=False)
        for seq, stamp in enumerate((0., 1., 2.), 1):
            self.observe(stamp, seq, 7)
        result = self.report()
        self.assertEqual(result["status"], "NOT_EVALUATED")
        self.assertEqual(self.deployment(result)["status"], "NOT_ACCEPTED")
        self.assertIsNone(result["wrong_confirmations"])
        self.assertEqual(self.faces.calls, 0)
        self.observe(.5, 20, 8, prepare=False)
        self.assertEqual(self.report()["status"], "NOT_EVALUATED")
        self.assertIsNone(self.report()["wrong_confirmations"])
        self.setup_case(people=[[person(), {**person("unlabeled"), "bbox": [.91, .1, .99, .9], "face_bbox": None}]
                               for _ in range(3)])
        for seq, stamp in enumerate((0., 1., 2.), 1):
            self.observe(stamp, seq, 7)
        result = self.report()
        self.assertEqual(result["recall"], 1.)
        self.assertEqual(result["unlabeled_visits"], 1)
        self.assertEqual(result["status"], "NOT_EVALUATED")
        self.assertFalse(result["certified"])
        self.assertEqual(self.deployment(result)["status"], "NOT_ACCEPTED")
        source = runner.ReplayFrame(1, "stream-1", 20, 1., 101., None, None)
        self.scorer.observe(1, source, SimpleNamespace(tracking_generation="generation", observed_tracks=[
            SimpleNamespace(track_id=2, bbox=(.91, .1, .99, .9), employee_id=7, association_uncertain=False)]))
        result = self.report()
        self.assertEqual(result["unattributed_confirmations"], 1)
        self.assertEqual(result["wrong_confirmations"], 0)
        self.assertEqual(self.deployment(result)["status"], "NOT_ACCEPTED")

    def test_annotation_size_visibility_and_native_alignment_rejections_remain_separate(self):
        rows = [[{**person(), "face_bbox": [.3, .2, .4, .3]}],
                [{**person(), "face_visible": False, "face_bbox": None, "occluded": True}], [person()]]
        self.setup_case(people=rows)
        self.faces.aligned = None
        for seq, stamp in enumerate((0., 1., 2.), 1):
            self.observe(stamp, seq)
        result = self.report()
        self.assertEqual(result["rejection_counts"]["FACE_TOO_SMALL"], 1)
        self.assertEqual(result["rejection_counts"]["FACE_OCCLUDED_OR_NOT_VISIBLE"], 1)
        self.assertEqual(result["rejection_counts"]["BAD_ALIGNMENT"], 1)
        self.assertIsNone(result["pose_rejections"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
