"""Offline replay contracts; no app settings, employee identities or production DB.

Run: .venv/Scripts/python.exe tests/test_evaluate_cctv.py
The FFmpeg case uses a generated, temporary clip, not any store recording.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import tempfile
import time
import unittest
from contextlib import closing
from types import SimpleNamespace
from pathlib import Path, PureWindowsPath
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import tools.evaluate_cctv as runner

from tools.evaluate_cctv import (
    AnalysisSample, AnnotationSamples, CameraMeasurements, ClipProbe, EvaluationError,
    FaceObservations, FFmpegReplay, LatestSlot, Prediction, QualityEvaluator,
    ReplayFrame, acceptance_gates, build_replay_specs, isolated_environment,
    load_manifest, match_people, percentile, probe_clip, resolve_media_tools,
    validate_output_path, write_report,
)


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / "recordings" / "session").mkdir(parents=True)
        (self.root / "recordings" / "session" / "clip.mkv").write_bytes(b"local fixture")
        self.path = self.root / "manifest.json"
        self.document = {
            "version": 1,
            "clips": [{"path": "recordings/session/clip.mkv", "source_camera_key": "view-01",
                       "offset_seconds": 0, "annotations_path": None}],
            "enrollments": [], "identity_labels": [], "scene_calibrations": [],
        }

    def save(self):
        self.path.write_text(json.dumps(self.document), encoding="utf-8")
        return load_manifest(self.path)

    def annotations(self):
        document = {"version": 1, "frames": [
            {"time_seconds": 0, "people": [
                {"track_label": "clip-person-01", "bbox": [.1, .1, .4, .9],
                 "face_bbox": None, "face_visible": False, "occluded": True, "seated": True},
            ]}], "crossings": []}
        self.document["clips"][0]["annotations_path"] = "annotations.json"
        return document

    def save_annotations(self, document):
        (self.root / "annotations.json").write_text(json.dumps(document), encoding="utf-8")
        return self.save()

    def test_resolves_paths_without_loading_application_database(self):
        manifest = self.save()
        self.assertEqual(manifest.clips[0].path, self.root / "recordings/session/clip.mkv")
        self.assertNotIn("app.config", sys.modules)
        self.assertNotIn("app.database.database", sys.modules)

    def test_rejects_remote_and_ambiguous_windows_paths_before_opening(self):
        for path in ("https://host/video.mkv", "file:///C:/video.mkv", "rtsp://host/video",
                     "\\\\host\\share\\video.mkv", "//host/share/video.mkv",
                     "C:relative.mkv", "C:\\video.mkv:payload", "clip.mkv\x00", "\\rooted.mkv"):
            with self.subTest(path=path):
                self.document["clips"][0]["path"] = path
                with self.assertRaises(EvaluationError):
                    self.save()

    def test_resolved_alias_to_network_or_device_is_rejected_before_file_check(self):
        class ResolvedAlias:
            def __init__(self, target):
                self.target = target

            def __str__(self):
                return self.target

            def is_file(self):
                raise AssertionError("Remote target must not reach file existence checks")

        for target in (r"\\server\share\clip.mkv", "//server/share/clip.mkv",
                       r"\\?\UNC\server\share\clip.mkv", r"\\.\C:\clip.mkv"):
            for must_exist in (True, False):
                with self.subTest(target=target, must_exist=must_exist):
                    with patch.object(Path, "resolve", return_value=ResolvedAlias(target)):
                        with self.assertRaises(EvaluationError):
                            runner.local_path("local-alias.mkv", self.root, must_exist=must_exist)

    def test_network_junction_is_rejected_before_resolving_child_path(self):
        alias = self.root / "junction"
        for target in (r"\\server\share", r"\\?\UNC\server\share"):
            with self.subTest(target=target):
                with patch.object(runner, "_alias_target", side_effect=lambda path: target if path == alias else None), \
                        patch.object(Path, "resolve", side_effect=AssertionError("Must not resolve a remote junction")):
                    with self.assertRaises(EvaluationError):
                        runner.local_path(alias / "clip.mkv", self.root)

    def test_mapped_network_drive_is_rejected_before_alias_inspection(self):
        with patch.object(runner, "_windows_drive_type", return_value=4), \
                patch.object(runner, "_alias_target", side_effect=AssertionError("Must not inspect a remote drive")):
            with self.assertRaises(EvaluationError):
                runner._preflight_local_aliases(PureWindowsPath(r"Z:\clips\clip.mkv"))

    def test_rejects_nonfinite_negative_boolean_offsets_and_missing_files(self):
        for offset in (float("nan"), float("inf"), 10 ** 400, -1, True):
            with self.subTest(offset=offset):
                self.document["clips"][0]["offset_seconds"] = offset
                with self.assertRaises(EvaluationError):
                    self.save()
        self.document["clips"][0]["offset_seconds"] = 0
        self.document["clips"][0]["path"] = "missing.mkv"
        with self.assertRaises(EvaluationError):
            self.save()

    def test_duplicate_json_keys_do_not_silently_replace_manifest_fields(self):
        self.path.write_text('{"version":1,"version":1}', encoding="utf-8")
        with self.assertRaises(EvaluationError):
            load_manifest(self.path)

    def test_scene_data_is_not_silently_ignored_by_step_one(self):
        self.document["scene_calibrations"] = [{"width_m": 4}]
        with self.assertRaises(EvaluationError):
            self.save()

    def test_rejects_degenerate_boxes_duplicate_labels_and_nonboolean_flags(self):
        original = self.annotations()
        for box in ([0, 0, 0, .5], [0, 0, 1.1, 1], [0, 0, math.nan, 1], [0, 1, 1, 0]):
            document = json.loads(json.dumps(original))
            document["frames"][0]["people"][0]["bbox"] = box
            with self.subTest(box=box), self.assertRaises(EvaluationError):
                self.save_annotations(document)
        original["frames"][0]["people"].append(original["frames"][0]["people"][0].copy())
        with self.assertRaises(EvaluationError):
            self.save_annotations(original)
        document = self.annotations()
        document["frames"][0]["people"][0]["seated"] = 1
        with self.assertRaises(EvaluationError):
            self.save_annotations(document)

    def test_rejects_nonmonotonic_times_and_unseen_crossing_labels(self):
        document = self.annotations()
        document["frames"].append({"time_seconds": 0, "people": []})
        with self.assertRaises(EvaluationError):
            self.save_annotations(document)
        document = self.annotations()
        document["crossings"] = [{"track_label": "not-annotated", "time_seconds": 1, "direction": "IN"}]
        with self.assertRaises(EvaluationError):
            self.save_annotations(document)

    def test_identity_ground_truth_cannot_name_unknown_track_or_unenrolled_subject(self):
        self.save_annotations(self.annotations())
        self.document["identity_labels"] = [
            {"source_camera_key": "view-01", "track_label": "clip-person-01", "subject_id": "unknown-owner"},
        ]
        with self.assertRaises(EvaluationError):
            self.save()
        self.document["identity_labels"][0]["subject_id"] = None
        self.save()
        self.document["identity_labels"][0]["track_label"] = "not-annotated"
        with self.assertRaises(EvaluationError):
            self.save()

    def test_output_cannot_overwrite_inputs_or_sibling_recording_tree(self):
        manifest = self.save()
        for output in (self.path, self.root / "recordings" / "other-session" / "report.json",
                       manifest.clips[0].path):
            with self.subTest(output=output), self.assertRaises(EvaluationError):
                validate_output_path(output, manifest)
        self.assertEqual(validate_output_path(self.root / "reports" / "result.json", manifest),
                         self.root / "reports" / "result.json")

    def test_output_symlink_cannot_bypass_recording_guard(self):
        manifest = self.save()
        try:
            (self.root / "alias").symlink_to(self.root / "recordings", target_is_directory=True)
        except OSError as exc:
            self.skipTest(f"Symlink privilege unavailable: {exc.__class__.__name__}")
        with self.assertRaises(EvaluationError):
            validate_output_path(self.root / "alias" / "report.json", manifest)


    def test_decoded_but_unanalyzed_annotations_count_as_misses(self):
        document = self.annotations()
        person = document["frames"][0]["people"][0]
        document["frames"] = [{"time_seconds": stamp, "people": [person.copy()]} for stamp in (0, 1, 2)]
        manifest = self.save_annotations(document)
        specs = build_replay_specs(manifest, {manifest.clips[0].path: ClipProbe(1920, 1080, 15, 3, "fixture")}, 1)
        collector = AnnotationSamples(specs)
        first = ReplayFrame(1, specs[0].stream_id, 1, 0, 0, None, None)
        last = ReplayFrame(1, specs[0].stream_id, 3, 2, 2, None, None)
        collector.observe_decoded(first)
        collector.observe(first, [Prediction("real-track-token", tuple(person["bbox"]))])
        collector.observe_decoded(last)
        result = collector.evaluate()
        self.assertEqual(result["frames"], 3)
        self.assertEqual(result["annotation_opportunities_without_nearby_inference"], 2)
        self.assertAlmostEqual(result["recall"], 1 / 3)

    def test_collector_keeps_transition_visibility_and_misses_out_of_1hz_totals(self):
        document = self.annotations()
        person = document["frames"][0]["people"][0]
        document["frames"] = [{"time_seconds": stamp, "people": [person.copy()]}
                              for stamp in (61.045, 61.925, 62.045)]
        manifest = self.save_annotations(document)
        specs = build_replay_specs(manifest, {manifest.clips[0].path: ClipProbe(1920, 1080, 15, 63, "fixture")}, 1)
        collector = AnnotationSamples(specs)
        for sequence, stamp in enumerate((61.045, 61.85, 62.5), 1):
            packet = ReplayFrame(1, specs[0].stream_id, sequence, stamp, sequence, None, None)
            collector.observe_decoded(packet)
            if stamp == 61.85:
                collector.observe(packet, [Prediction("transition-track", tuple(person["bbox"]))])
        result = collector.evaluate()
        self.assertEqual(result["frames"], 2)
        self.assertEqual(result["recall"], 0)
        self.assertAlmostEqual(result["visible_person_seconds"], 1.455)
        self.assertEqual(result["annotation_opportunities_without_nearby_inference"], 2)
        self.assertEqual(result["transitions"]["frames"], 1)
        self.assertEqual(result["transitions"]["recall"], 1)
        self.assertEqual(result["transitions"]["annotation_opportunities_without_nearby_inference"], 0)

    def test_report_hardlink_alias_cannot_overwrite_input(self):
        manifest = self.save()
        output = self.root / "hardlink.json"
        os.link(manifest.clips[0].path, output)
        with self.assertRaises(EvaluationError):
            validate_output_path(output, manifest)

    def test_invalid_report_metrics_leave_existing_report_and_no_staged_file(self):
        manifest = self.save()
        output = self.root / "reports" / "result"
        output.parent.mkdir()
        output.write_text('{"old": true}', encoding="utf-8")
        with self.assertRaises(ValueError):
            write_report(output, {"invalid_metric": math.nan}, manifest)
        self.assertEqual(json.loads(output.read_text(encoding="utf-8")), {"old": True})
        self.assertEqual(list(output.parent.iterdir()), [output])


class QualityTests(unittest.TestCase):
    def test_matching_is_one_to_one_and_maximizes_threshold_qualified_count(self):
        # Flexible first prediction must not steal the only match for the second.
        boxes = [(0, 0, .6, 1), (.4, 0, 1, 1)]
        predictions = [Prediction("flexible", (.2, 0, .8, 1)), Prediction("left", boxes[0])]
        matches = match_people(boxes, predictions, iou_threshold=.49)
        self.assertEqual(set(matches), {(0, 1), (1, 0)})
        self.assertEqual(match_people([boxes[0]], [predictions[1], predictions[1]]), [(0, 0)])

    def test_counts_unknown_seated_misses_false_positives_and_generation_bound_switches(self):
        evaluator = QualityEvaluator()
        person = {"track_label": "anonymous-1", "bbox": [.1, .1, .4, .9],
                  "face_bbox": None, "face_visible": False, "occluded": True, "seated": True}
        evaluator.add_frame("camera-a", "stream-1", 0, [person],
                            [Prediction("track-1", tuple(person["bbox"]))], visible_seconds=1)
        evaluator.add_frame("camera-a", "stream-1", 1, [person],
                            [Prediction("track-2", tuple(person["bbox"])),
                             Prediction("extra", (.7, .1, .9, .5))], visible_seconds=1)
        evaluator.add_frame("camera-a", "stream-2", 0, [person], [], visible_seconds=1)
        result = evaluator.summary()
        self.assertAlmostEqual(result["recall"], 2 / 3)
        self.assertAlmostEqual(result["false_positive_fraction"], 1 / 3)
        self.assertEqual(result["identity_switches"], 1)
        self.assertEqual(result["subsets"]["seated"]["annotated"], 3)
        self.assertEqual(result["subsets"]["occluded"]["matched"], 2)
        self.assertEqual(result["recognition"]["status"], "NOT_EVALUATED")

    def test_no_annotations_or_predictions_is_not_perfect_accuracy(self):
        summary = QualityEvaluator().summary()
        self.assertEqual(summary["status"], "NOT_EVALUATED")
        self.assertIsNone(summary["recall"])
        self.assertIsNone(summary["false_positive_fraction"])
        self.assertIsNone(percentile([], .95))


    def test_successful_transition_frames_cannot_rescue_failing_fixed_1hz_recall(self):
        person = {"track_label": "anonymous-1", "bbox": [.1, .1, .4, .9],
                  "face_bbox": None, "face_visible": False, "occluded": False, "seated": True}
        baseline = QualityEvaluator()
        with_transitions = QualityEvaluator()
        for second in range(82):
            predictions = [Prediction("track-1", tuple(person["bbox"]))] if second < 73 else []
            for evaluator in (baseline, with_transitions):
                evaluator.add_frame("camera-a", "stream-1", second + .045, [person],
                                    predictions, visible_seconds=1)
        for second in range(11):
            with_transitions.add_frame("camera-a", "stream-1", second + .925, [person],
                                       [Prediction("track-1", tuple(person["bbox"]))], visible_seconds=.075)
        original, result = baseline.summary(), with_transitions.summary()
        self.assertEqual(original["detection_tracking_acceptance"]["status"], "FAIL")
        self.assertEqual(result["detection_tracking_acceptance"]["status"], "FAIL")
        self.assertEqual(result["frames"], 82)
        self.assertEqual(result["annotated_people"], 82)
        self.assertEqual(result["matched_people"], 73)
        self.assertAlmostEqual(result["recall"], 73 / 82)
        self.assertEqual(result["visible_person_seconds"], original["visible_person_seconds"])
        self.assertEqual(result["transitions"]["frames"], 11)
        self.assertEqual(result["transitions"]["recall"], 1)

    def test_earliest_regular_annotation_wins_even_when_transition_is_added_first(self):
        person = {"track_label": "anonymous-1", "bbox": [.1, .1, .4, .9],
                  "face_bbox": None, "face_visible": False, "occluded": False, "seated": False}
        evaluator = QualityEvaluator()
        evaluator.add_frame("camera-a", "stream-1", 61.925, [person],
                            [Prediction("transition-track", tuple(person["bbox"]))], visible_seconds=.075)
        evaluator.add_frame("camera-a", "stream-1", 61.045, [person], [], visible_seconds=1)
        result = evaluator.summary()
        self.assertEqual(result["frames"], 1)
        self.assertEqual(result["recall"], 0)
        self.assertEqual(result["transitions"]["frames"], 1)
        self.assertEqual(result["transitions"]["recall"], 1)

    def test_face_observations_report_native_sizes_and_reasons_not_guessed_accuracy(self):
        quality = SimpleNamespace(min_size=32, blur=12)
        evidence = SimpleNamespace(quality=quality, reason_codes=("FACE_TOO_SMALL", "BLURRY_FACE"))
        outcome = SimpleNamespace(observed_tracks=[SimpleNamespace(bbox=(.1, .1, .4, .9))],
                                  detections=[SimpleNamespace(fresh_face_evidence=evidence),
                                              SimpleNamespace(fresh_face_evidence=None)])
        observations = FaceObservations()
        observations.observe(outcome, (1080, 1920, 3))
        result = observations.summary()
        self.assertEqual(result["native_face_min_dimension_px"]["p50"], 32)
        self.assertAlmostEqual(result["native_person_box_width_px"]["p50"], 576)
        self.assertAlmostEqual(result["native_person_box_height_px"]["p50"], 864)
        self.assertEqual(result["reason_counts"], {"FACE_TOO_SMALL": 1, "BLURRY_FACE": 1})
        self.assertEqual(result["display_observations_without_fresh_evidence"], 1)


class MetricsTests(unittest.TestCase):
    def test_warmup_idle_time_and_zero_camera_are_in_fps_denominator(self):
        measurements = CameraMeasurements(1, target_fps=5, attendance=False)
        for stamp in (1, 29, 30, 31, 40):
            measurements.record_analysis(AnalysisSample(stamp, stamp + .1, stamp + .2))
        summary = measurements.summary(0, 90)
        self.assertAlmostEqual(summary["analyzed_fps"], 3 / 60)
        self.assertAlmostEqual(summary["p95_completion_age_seconds"], .2)
        self.assertAlmostEqual(summary["worst_rolling_60s"]["analyzed_fps"], 3 / 60)
        empty = CameraMeasurements(2, target_fps=5, attendance=False).summary(0, 90)
        self.assertEqual(empty["analyzed_fps"], 0)
        self.assertIsNone(empty["p95_completion_age_seconds"])

    def test_unintegrated_or_short_benchmark_never_passes_capacity(self):
        result = acceptance_gates([], duration_seconds=900, copies=10,
                                  gpu_inventory=[{"name": "NVIDIA GeForce RTX 4070", "memory_total_mib": 12282}],
                                  integration_needs=["WORKER_NOT_EXERCISED"])
        self.assertEqual(result["capacity"]["status"], "NOT_EVALUATED")
        self.assertEqual(result["target_hardware"]["status"], "NOT_TARGET_HARDWARE")
        result = acceptance_gates([], duration_seconds=20, copies=10,
                                  gpu_inventory=[], integration_needs=[])
        self.assertNotEqual(result["capacity"]["status"], "PASS")

    def test_all_cameras_and_worst_window_must_meet_fps_and_age_gates(self):
        good = {"camera_id": 1, "attendance": False, "analyzed_fps": 5,
                "p95_completion_age_seconds": .2,
                "worst_rolling_60s": {"analyzed_fps": 5, "p95_completion_age_seconds": .3}}
        slow = {**good, "camera_id": 2, "analyzed_fps": 4.4}
        result = acceptance_gates([good, slow], duration_seconds=900, copies=2,
                                  gpu_inventory=[], integration_needs=[])
        self.assertEqual(result["capacity"]["status"], "FAIL")
        bad_window = {**good, "worst_rolling_60s": {"analyzed_fps": 3, "p95_completion_age_seconds": 2}}
        result = acceptance_gates([bad_window], duration_seconds=900, copies=1,
                                  gpu_inventory=[], integration_needs=[])
        self.assertEqual(result["capacity"]["status"], "FAIL")

    def test_isolated_environment_ignores_secrets_and_restores_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            old = dict(os.environ)
            os.environ["DATABASE_URL"] = "sqlite:///production-must-not-open.db"
            os.environ["TELEGRAM_BOT_TOKEN"] = "not-a-real-token"
            try:
                with isolated_environment(Path(directory)) as isolated:
                    self.assertTrue(isolated.database_path.is_file())
                    self.assertNotIn("TELEGRAM_BOT_TOKEN", os.environ)
                    self.assertIn(directory.replace("\\", "/"), os.environ["DATABASE_URL"])
                    self.assertEqual(os.environ["TELEGRAM_ENABLED"], "false")
                    self.assertTrue(isolated.media_root.is_dir())
                self.assertEqual(os.environ["TELEGRAM_BOT_TOKEN"], "not-a-real-token")
            finally:
                os.environ.clear()
                os.environ.update(old)


class ProductionAccountingTests(unittest.TestCase):
    def test_unpublished_or_wrong_generation_batch_never_counts_as_analysis(self):
        metrics = {1: CameraMeasurements(1, 5, False)}
        accounting = runner.ProductionAccounting(metrics, None, None)
        packet = SimpleNamespace(stream_id="stream", source_revision=1, seq=7, received_mono=31)
        outcome = SimpleNamespace(tracking_generation="generation", observed_tracks=[])
        item = SimpleNamespace(camera_id=1, packet=packet)
        accounting.batch([item], [outcome], 31.1)
        self.assertEqual(metrics[1].analyses, [])
        accounting.published(1, packet, "retired-generation", 31.2)
        self.assertEqual(metrics[1].analyses, [])
        accounting.published(1, packet, "generation", 31.4)
        accounting.published(1, packet, "generation", 31.5)
        self.assertEqual(metrics[1].analyses, [AnalysisSample(31, 31.1, 31.4)])
        accounting.batch([item], [outcome], 32)
        accounting.batch([], [], 33)
        accounting.published(1, packet, "generation", 33.1)
        self.assertEqual(len(metrics[1].analyses), 1)
        accounting.measurement_end = 34
        accounting.batch([item], [outcome], 33.5)
        accounting.published(1, packet, "generation", 34.1)
        self.assertEqual(len(metrics[1].analyses), 1)

    def test_journal_health_keeps_recovered_degradation_and_growth_history(self):
        accounting = runner.ProductionAccounting({}, None, None)
        for stamp, state, pending in ((0, "OK", 0), (31, "JOURNAL_DEGRADED", 2),
                                      (32, "JOURNAL_DEGRADED", 3), (33, "OK", 0)):
            accounting.health(stamp, dict(state=state, pending_commands=pending,
                                          pending_updates=0, pending_barriers=0))
        report = accounting.health_report(34)
        self.assertEqual(report["degraded_intervals"], [{"started_mono": 31, "ended_mono": 33}])
        self.assertEqual(report["peak_pending"], 3)
        self.assertFalse(report["persistent_growth"])
        for stamp in range(40, 102):
            accounting.health(stamp, dict(state="OK", pending_commands=stamp,
                                          pending_updates=0, pending_barriers=0))
        self.assertTrue(accounting.health_report(102)["persistent_growth"])

    def test_growth_requires_sustained_post_warmup_bucket_increases(self):
        workloads = {
            "stable_positive": lambda second: 2,
            "step_then_plateau": lambda second: 1 if second == 0 else 2,
            "bounded_oscillation": lambda second: 2 + (second % 10),
            "warmup_growth_only": lambda second: min(second + 1, 30),
            "sustained_growth": lambda second: second + 1,
        }
        for name, pending_at in workloads.items():
            with self.subTest(name=name):
                accounting = runner.ProductionAccounting({}, None, None)
                accounting.measurement_start = 30.
                for second in range(151):
                    accounting.health(second, dict(state="OK", pending_commands=pending_at(second),
                                                   pending_updates=0, pending_barriers=0))
                self.assertEqual(accounting.health_report(150)["persistent_growth"], name == "sustained_growth")

    def test_growth_does_not_include_warmup_in_its_minute(self):
        accounting = runner.ProductionAccounting({}, None, None)
        accounting.measurement_start = 30.
        for second in range(90):
            accounting.health(second, dict(state="OK", pending_commands=second + 1,
                                           pending_updates=0, pending_barriers=0))
        self.assertFalse(accounting.health_report(89)["persistent_growth"])
        accounting.health(90, dict(state="OK", pending_commands=91, pending_updates=0, pending_barriers=0))
        self.assertTrue(accounting.health_report(90)["persistent_growth"])

    def test_journal_integrity_checks_missing_duplicate_and_unbounded_events(self):
        import sqlite3
        with closing(sqlite3.connect(":memory:")) as db:
            db.executescript("""
                CREATE TABLE human_visits(id TEXT, camera_id INTEGER, tracking_generation TEXT,
                    track_id INTEGER, source_revision INTEGER, stream_id TEXT, status TEXT);
                CREATE TABLE human_visit_events(id TEXT, visit_id TEXT, dedupe_key TEXT,
                    type TEXT, payload TEXT);
                CREATE TABLE human_journal_gaps(id TEXT);
            """)
            expected = {(1, "generation", 1): dict(source_revision=1, stream_id="stream", event_budget=2)}
            missing = runner.journal_integrity(db, expected)
            self.assertEqual(missing["missing_visits"], 1)
            db.execute("INSERT INTO human_visits VALUES ('visit',1,'generation',1,1,'stream','CLOSED')")
            for index, kind in enumerate(("FIRST_SEEN", "CLOSED"), 1):
                db.execute("INSERT INTO human_visit_events VALUES (?,?,?,?,?)",
                           (str(index), "visit", str(index), kind, json.dumps({"transition_seq": index})))
            clean = runner.journal_integrity(db, expected)
            self.assertEqual(clean["status"], "PASS")
            db.execute("INSERT INTO human_visit_events VALUES ('3','visit','3','LOST','{\"transition_seq\":3}')")
            self.assertEqual(runner.journal_integrity(db, expected)["unbounded_event_visits"], 1)
            db.execute("INSERT INTO human_visits VALUES ('duplicate',1,'generation',1,1,'stream','CLOSED')")
            self.assertEqual(runner.journal_integrity(db, expected)["duplicate_visits"], 1)

    def test_capacity_rejects_loss_degradation_and_unevaluated_concurrency(self):
        good = {"camera_id": 1, "attendance": True, "analyzed_fps": 10,
                "p95_completion_age_seconds": .2,
                "worst_rolling_60s": {"analyzed_fps": 10, "p95_completion_age_seconds": .3}}
        for fault in ("JOURNAL_LOSS", "JOURNAL_DEGRADED", "OUT_OF_MEMORY", "PERSISTENT_QUEUE_GROWTH"):
            result = acceptance_gates([good], duration_seconds=900, copies=1,
                                      gpu_inventory=[], integration_needs=[], runtime_failures=[fault])
            self.assertEqual(result["capacity"]["status"], "FAIL")
            self.assertIn(fault, result["capacity"]["failures"])
        result = acceptance_gates([good], duration_seconds=900, copies=1, gpu_inventory=[],
                                  integration_needs=["ENROLLMENT_CONCURRENT_LOAD_NOT_EXERCISED"])
        self.assertEqual(result["capacity"]["status"], "NOT_EVALUATED")


class ConcurrentEvidenceTests(unittest.TestCase):
    def test_face_stage_uses_each_frame_total_not_sum_of_component_percentiles(self):
        report = runner.stage_timings([
            {"stage_times_ms": {"face_detection": 2., "face_recognition": 8.}},
            {"stage_times_ms": {"face_detection": 8., "face_recognition": 2.}}])
        self.assertEqual(report["face"]["p95_ms"], 10.)
        self.assertIsNone(report["journal"]["p95_ms"])

    def test_profiles_require_executed_heavy_cuda_nodes_not_provider_registration(self):
        self.assertEqual(runner.profile_summary([], ["CUDAExecutionProvider"])["status"], "NOT_EVALUATED")
        events = [dict(cat="Node", args=dict(op_name="Conv", provider="CPUExecutionProvider"))]
        self.assertEqual(runner.profile_summary(events, ["CUDAExecutionProvider"])["status"], "FAIL")
        events[0]["args"]["provider"] = "CUDAExecutionProvider"
        self.assertEqual(runner.profile_summary(events, ["CUDAExecutionProvider"])["status"], "PASS")

    def test_concurrent_impact_excludes_warmup_and_shutdown_work(self):
        metrics = {1: CameraMeasurements(1, 5, False)}
        metrics[1].analyses = [AnalysisSample(9, 10, 11), AnalysisSample(30.9, 31, 31.5),
                               AnalysisSample(33, 33.1, 33.5), AnalysisSample(89, 89.5, 91)]
        report = runner.concurrent_impact(metrics, [
            dict(started_mono=10, completed_mono=12), dict(started_mono=31.2, completed_mono=32),
            dict(started_mono=89, completed_mono=92)], 0, 90)
        self.assertEqual(report["overlapping_analyses"], 1)
        self.assertAlmostEqual(report["cameras"][0]["p95_completion_age_seconds"], .6)

    def test_scene_prompts_are_explicit_clip_bound_and_normalized(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clip = root / "clip.mkv"
            clip.write_bytes(b"local fixture")
            source = runner.Clip(clip, "view", 0, None, None)
            manifest = runner.Manifest(root / "manifest.json", (source,), (), ())
            probes = {clip: ClipProbe(1920, 1080, 15, 60, "fixture")}
            path = root / "prompts.json"
            prompt = dict(clip_path="clip.mkv", source_camera_key="view", time_seconds=20,
                          kind="floor", positive_points=[[.2, .8]], negative_points=[], box=None)
            path.write_text(json.dumps(dict(version=1, prompts=[prompt])), encoding="utf-8")
            result = runner.load_scene_prompts(path, manifest, probes)
            self.assertEqual(result[0]["clip_path"], clip)
            for change in (dict(time_seconds=60), dict(positive_points=[[1.1, .5]]),
                           dict(source_camera_key="unreviewed-view"), dict(kind="automatic-floor")):
                path.write_text(json.dumps(dict(version=1, prompts=[{**prompt, **change}])), encoding="utf-8")
                with self.assertRaises(EvaluationError):
                    runner.load_scene_prompts(path, manifest, probes)


class ReplayTests(unittest.TestCase):
    def test_latest_slot_never_accumulates_history(self):
        slot = LatestSlot()
        first = ReplayFrame(1, "stream-a", 1, 0, 0, None, b"native")
        second = ReplayFrame(1, "stream-a", 2, 1, 1, None, b"native")
        slot.publish(first)
        slot.publish(second)
        self.assertIs(slot.latest(), second)
        self.assertEqual(slot.overwritten, 1)
        self.assertIs(slot.take_after(1), second)
        self.assertIsNone(slot.take_after(2))

    def test_real_independent_decoders_preserve_native_size_count_and_cleanup(self):
        try:
            ffmpeg, ffprobe = resolve_media_tools()
        except EvaluationError:
            self.skipTest("FFmpeg/ffprobe unavailable; actual decode proof remains required")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clip = root / "generated.mkv"
            subprocess.run([ffmpeg, "-nostdin", "-v", "error", "-f", "lavfi", "-i",
                            "testsrc=size=32x24:rate=5", "-t", "0.8", "-c:v", "ffv1",
                            "-threads", "1", str(clip)], check=True, timeout=15,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps({"version": 1, "clips": [
                {"path": "generated.mkv", "source_camera_key": "generated",
                 "offset_seconds": 0, "annotations_path": None}],
                "enrollments": [], "identity_labels": [], "scene_calibrations": []}), encoding="utf-8")
            manifest = load_manifest(manifest_path)
            probe = probe_clip(clip, ffprobe)
            self.assertEqual((probe.width, probe.height), (32, 24))
            specs = build_replay_specs(manifest, {clip: probe}, 2)
            self.assertNotEqual(specs[0].stream_id, specs[1].stream_id)
            self.assertNotEqual(specs[0].offset_seconds, specs[1].offset_seconds)
            # Use offset zero to make the finite native-frame count unambiguous.
            from dataclasses import replace
            replays = [FFmpegReplay(replace(spec, offset_seconds=0), ffmpeg,
                                    realtime=False, repeat=False) for spec in specs]
            try:
                for replay in replays:
                    replay.start()
                deadline = time.monotonic() + 10
                while any(not replay.finished.is_set() for replay in replays) and time.monotonic() < deadline:
                    for replay in replays:
                        replay.check_health()
                    time.sleep(.01)
                for replay in replays:
                    replay.check_health()
                    self.assertTrue(replay.finished.is_set(), "Finite FFmpeg decode did not finish")
                    self.assertEqual(replay.decoded_count, 4)
                    self.assertEqual(replay.slot.latest().frame.shape, (24, 32, 3))
                    self.assertFalse(replay.slot.latest().frame.flags.writeable)
                    self.assertAlmostEqual(replay.slot.latest().source_time_seconds, .6, places=5)
                    self.assertEqual(replay.slot.overwritten, 3)
            finally:
                for replay in replays:
                    replay.stop()
            self.assertTrue(all(not replay.is_alive() for replay in replays))
            self.assertTrue(all(replay.process is None or replay.process.poll() is not None for replay in replays))


if __name__ == "__main__":
    unittest.main(verbosity=2)
