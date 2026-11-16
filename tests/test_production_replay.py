"""Temporary-DB worker/journal replay wiring, with only native inference/decode doubled.
Run: .venv/Scripts/python.exe tests/test_production_replay.py
"""
from __future__ import annotations

import sqlite3
import sys
import tempfile
import time
import unittest
from contextlib import closing, ExitStack
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools import evaluate_cctv as runner


class ProductionReplayTests(unittest.TestCase):
    def test_real_worker_publishes_unknown_provisional_visits_and_reconnect_generation(self):
        import numpy as np
        from app.ai.recognition_service import RecognitionService
        from app.ai.worker import AIWorker, DetectionState
        from app.ai.global_tracker import GlobalIdentityManager
        from app.database.database import engine, init_db, SessionLocal
        from app.database.models import HumanVisit, WorkSession
        from app.services.attendance_service import AttendanceService
        from app.services.human_journal import HumanJournal

        class Detector:
            callback = None

            def detect_batch(self, frames):
                results = [[(np.array([100., 100., 350., 800.]), .95)] for _ in frames]
                if self.callback is not None:
                    self.callback()
                return results

        class Body:
            def embed(self, crops):
                vectors = np.zeros((len(crops), 512), np.float32)
                vectors[:, 0] = 1.
                return vectors

        init_db()
        root = Path(tempfile.gettempdir())
        clip = runner.Clip(root / "generated-fixture-only.mkv", "anonymous", 0., None, None)
        manifest = runner.Manifest(root / "manifest.json", (clip,), (), ())
        specs = runner.build_replay_specs(manifest, {clip.path: runner.ClipProbe(1920, 1080, 15, 60, "fixture")}, 3)
        annotations = runner.AnnotationSamples(specs)
        manager = runner.ReplayManager(specs, "unused-decode-boundary", annotations)
        runner.configure_replay_cameras(engine, specs)
        metrics = {spec.camera_id: runner.CameraMeasurements(spec.camera_id, spec.target_fps, spec.attendance) for spec in specs}
        accounting = runner.ProductionAccounting(metrics, annotations, manager)
        service = RecognitionService(Detector(), None, None)
        state = DetectionState()
        journal = HumanJournal(engine)
        body = Body()
        runner.instrument_pipeline(service, state, journal, body, accounting)
        worker = AIWorker(manager, service, AttendanceService(engine), state, journal=journal,
                          reid=body, global_manager=GlobalIdentityManager())
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        frame.flags.writeable = False
        try:
            with ExitStack() as stack:
                for camera in manager.reader_ids():
                    stack.enter_context(patch.object(manager.get(camera), "is_alive", return_value=True))
                for sequence in range(1, 5):
                    for spec in specs:
                        packet = runner.ReplayFrame(spec.camera_id, spec.stream_id, sequence, sequence / 15,
                                                    time.monotonic(), datetime.now(timezone.utc).replace(tzinfo=None), frame)
                        manager.get(spec.camera_id).buffer.publish(packet)
                    worker.step()
                    time.sleep(.21)
                published = state.camera(3)
                self.assertEqual(published["source_revision"], 1)
                self.assertEqual(published["stream_id"], specs[2].stream_id)
                self.assertEqual(published["tracks"][0]["employee_id"], None)
                old_generation = published["tracking_generation"]
                manager.get(3).buffer.publish(runner.ReplayFrame(3, "reconnected-stream", 5, 0., time.monotonic(),
                                                          datetime.now(timezone.utc).replace(tzinfo=None), frame))
                worker.step()
                self.assertNotEqual(state.camera(3)["tracking_generation"], old_generation)
                self.assertEqual(state.camera(3)["stream_id"], "reconnected-stream")
                preview = manager.preview(3)
                import cv2
                decoded = cv2.imdecode(np.frombuffer(preview[1], np.uint8), cv2.IMREAD_COLOR)
                self.assertEqual(decoded.shape[:2], (720, 1280))
                self.assertIs(manager.preview(3)[1], preview[1])
                # A source reconnect during the real batch must be discarded by
                # the worker, including the accounting and journal admissions.
                time.sleep(.21)
                manager.get(3).buffer.publish(runner.ReplayFrame(3, "reconnected-stream", 6, .1, time.monotonic(),
                                                          datetime.now(timezone.utc).replace(tzinfo=None), frame))
                service.detector.callback = lambda: manager.get(3).buffer.publish(
                    runner.ReplayFrame(3, "superseding-stream", 7, 0., time.monotonic(),
                                       datetime.now(timezone.utc).replace(tzinfo=None), frame))
                worker.step()
                self.assertIsNone(state.camera(3))
                self.assertEqual(len(accounting.expected), 4)
                worker.stop()
            close = journal.close()
            self.assertTrue(close.drained)
            self.assertFalse(close.thread_alive)
            with SessionLocal() as db:
                visits = db.query(HumanVisit).all()
                self.assertEqual(len(visits), 4)
                self.assertTrue(all(visit.employee_id is None and visit.status == "CLOSED" for visit in visits))
                self.assertEqual(db.query(WorkSession).count(), 0)
            with closing(sqlite3.connect(engine.url.database)) as db:
                integrity = runner.journal_integrity(db, accounting.expected)
            self.assertEqual(integrity["status"], "PASS", integrity)
            self.assertEqual(integrity["accepted_track_keys"], 4)
            self.assertEqual(sum(len(metric.analyses) for metric in metrics.values()), 13)
            self.assertGreater(accounting.ack_counts["COMMITTED"], 0)
            appearance = accounting.appearance_snapshot()
            self.assertGreater(appearance["acknowledged_links"], 0)
            self.assertGreaterEqual(appearance["embedded_crops"], appearance["acknowledged_links"])
        finally:
            worker.stop()
            journal.close()
            manager.jpeg_cache.clear()
            engine.dispose()


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="production-replay-test-") as directory:
        with runner.isolated_environment(Path(directory)):
            unittest.main(verbosity=2)
