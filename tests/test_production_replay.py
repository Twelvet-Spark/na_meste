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
    def test_prepared_gallery_ids_and_real_worker_publications_reach_labeled_scorer(self):
        import cv2
        import numpy as np
        import threading
        from sqlalchemy import create_engine, event
        from sqlalchemy.orm import sessionmaker
        from app.ai.face_detector import ALIGNMENT_ID, FaceInfo
        from app.ai.face_recognition import EmbeddingStore
        from app.ai.recognition_service import RecognitionService
        from app.ai.worker import AIWorker, DetectionState
        from app.database.attendance_schema import initialize_attendance_schema
        from app.database.models import Employee, FaceModelMetadata
        from app.services.attendance_service import AttendanceService
        from app.services.human_journal import HumanJournal

        checker = (np.indices((112, 112)).sum(axis=0) % 2 * 255).astype(np.uint8)
        aligned = np.repeat(checker[:, :, None], 3, axis=2)
        vector = np.zeros(512, np.float32)
        vector[0] = 1.

        class Faces:
            model_id = "f" * 64
            alignment_id = ALIGNMENT_ID

            def detect(self, image):
                return [FaceInfo(np.array([20., 20., 100., 100.]), .95, vector, aligned,
                                 model_id=self.model_id)]

            def detect_faces(self, crop):
                # Native detector boundary: the true face is at (60,40)-(120,100).
                # Production live head crops expand sideways; GT quality crops do not.
                shift = (crop.shape[1] - 160) / 2
                box = np.array([40. + shift, 20., 100. + shift, 80.])
                points = np.array([[50. + shift, 35.], [85. + shift, 35.], [68. + shift, 50.],
                                   [55. + shift, 65.], [82. + shift, 65.]])
                return [FaceInfo(box, .95, None, aligned, landmarks=points, model_id=self.model_id)]

            def embed_aligned(self, crops):
                return np.repeat(vector[None, :], len(crops), axis=0)

        class Detector:
            def detect_batch(self, frames):
                return [[(np.array([20., 20., 180., 180.]), .95)] for _ in frames]

        with ExitStack() as cleanup:
            directory = cleanup.enter_context(tempfile.TemporaryDirectory(prefix="labeled-worker-test-"))
            root = Path(directory)
            engine = create_engine(f"sqlite:///{(root / 'isolated.db').as_posix()}",
                                   connect_args={"check_same_thread": False})
            cleanup.callback(engine.dispose)
            @event.listens_for(engine, "connect")
            def foreign_keys(connection, _record):
                connection.execute("PRAGMA foreign_keys=ON")
            initialize_attendance_schema(engine)
            sessions = sessionmaker(bind=engine, expire_on_commit=False)
            photo = root / "synthetic-enrollment.png"
            ok, encoded = cv2.imencode(".png", np.pad(aligned, ((8, 8), (8, 8), (0, 0))))
            self.assertTrue(ok)
            photo.write_bytes(encoded.tobytes())
            person = dict(track_label="verified-track", bbox=[.1, .1, .9, .9], face_bbox=[.3, .2, .6, .5],
                          face_visible=True, occluded=False, seated=False)
            annotation = dict(version=1, frames=[dict(time_seconds=t, people=[person]) for t in (0., .5, 1., 1.5)],
                              crossings=[])
            clip = runner.Clip(root / "generated-clip.mkv", "verified-view", 0., None, annotation)
            manifest = runner.Manifest(root / "manifest.json", (clip,),
                (runner.Enrollment("operator-verified-alice", (photo,)),),
                (dict(source_camera_key="verified-view", track_label="verified-track",
                      subject_id="operator-verified-alice"),))
            prepared = runner.ApprovedInference.__new__(runner.ApprovedInference)
            prepared.enroll_engine = Faces()
            prepared.enrollment_semaphore = threading.BoundedSemaphore(1)
            employee_subjects = prepared._enroll(manifest, sessions)
            with sessions() as db:
                employee = db.query(Employee).one()
                self.assertEqual(employee_subjects, {employee.id: "operator-verified-alice"})
                self.assertEqual(db.query(FaceModelMetadata).one().model_id, Faces.model_id)
            spec = runner.ReplaySpec(1, clip, runner.ClipProbe(200, 200, 15., 20., "fixture"),
                                     0., "labeled-stream", False, 5.)
            annotations = runner.AnnotationSamples([spec])
            manager = runner.ReplayManager([spec], "unused-native-decoder", annotations)
            cleanup.callback(manager.jpeg_cache.clear)
            runner.configure_replay_cameras(engine, [spec])
            store = EmbeddingStore(Faces.model_id)
            with sessions() as db:
                store.refresh(db)
            service = RecognitionService(Detector(), Faces(), store, retry_interval=.5, min_confirmations=3)
            metrics = {1: runner.CameraMeasurements(1, 5., False)}
            accounting = runner.ProductionAccounting(metrics, annotations, manager)
            accounting.identity_evaluator = runner.IdentityEvaluator(
                [spec], manifest.identity_labels, employee_subjects, service.faces,
                ordinary_interval=.5, attendance_interval=.25)
            journal = HumanJournal(engine)
            cleanup.callback(journal.close)
            state = DetectionState()
            runner.instrument_pipeline(service, state, journal, None, accounting)
            worker = AIWorker(manager, service, AttendanceService(engine), state, journal=journal)
            cleanup.callback(worker.stop)
            frame = np.zeros((200, 200, 3), np.uint8)
            frame.flags.writeable = False
            for seq, stamp in enumerate((0., .5, 1., 1.5), 1):
                accounting.identity_evaluator.capture_quality(1, runner.ReplayFrame(
                    1, "offline-native-quality", seq, stamp, float(seq), None, frame))
            with patch.object(manager.get(1), "is_alive", return_value=True):
                for seq, stamp in enumerate((0., .5, 1., 1.5), 1):
                    manager.get(1).buffer.publish(runner.ReplayFrame(
                        1, "labeled-stream", seq, stamp, time.monotonic(),
                        datetime.now(timezone.utc).replace(tzinfo=None), frame))
                    worker.step()
                    if seq < 4:
                        time.sleep(.55)
            self.assertEqual(state.camera(1)["tracks"][0]["employee_id"], employee.id)
            result = accounting.identity_evaluator.summary(annotations)
            self.assertEqual(result["eligible_known_visits"], 1)
            self.assertEqual(result["recall"], 1.)
            self.assertEqual(result["wrong_confirmations"], 0)
            self.assertEqual(result["status"], "PASS")
            acceptance = runner.acceptance_gates([], duration_seconds=1.5, copies=1, gpu_inventory=[],
                                                  integration_needs=[], recognition=result)
            self.assertEqual(acceptance["recognition"]["status"], "PASS")
            self.assertEqual(acceptance["recognition"]["visits"][0]["subject_id"], "operator-verified-alice")

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
