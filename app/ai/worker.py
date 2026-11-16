"""Single inference owner: recognition → doorway → durable attendance → optional Re-ID.

Only this thread mutates local trajectories. The monitor consumes immutable FIFO
records and health notifications, never trackers. Reader publication remains the
transaction-captured CameraSource interface.
"""
import logging
import threading
import time
from dataclasses import dataclass
from typing import Optional
from uuid import uuid4
from pathlib import Path

from sqlalchemy.exc import SQLAlchemyError

from app.ai.doorway import DoorwayCalibration, DoorwayDetector
from app.ai.face_detector import FaceEngine
from app.ai.face_recognition import EmbeddingStore
from app.ai.model_bundle import ModelBundle
from app.ai.providers import provider_label, resolve_providers, setup_cuda_dlls
from app.ai.recognition_service import RecognitionInput, RecognitionService
from app.ai.scheduler import DeadlineScheduler
from app.ai.inference_metrics import InferenceMetrics
from app.config import BASE_DIR, settings
from app.database.models import AttendanceCamera, Camera
from app.rtsp.manager import ReaderManager
from app.rtsp.reader import CameraStatus
from app.services.attendance_health import CoverageFault
from app.services.attendance_monitor import AttendanceWriteQueue
from app.services.attendance_queries import read_snapshot
from app.time_utils import utcnow

logger = logging.getLogger("app.ai.worker")


class DetectionState:
    """Потокобезопасные последние результаты детекции по камерам (для API/UI)."""

    def __init__(self):
        self._lock = threading.Lock()
        self._data: dict[int, dict] = {}

    def update(self, camera_id: int, detections) -> None:
        tracks = [
            {
                "track_id": d.track_id,
                "employee_id": d.employee_id,
                "employee_name": d.employee_name,
                "confidence": d.confidence,
                "bbox": list(d.bbox),
                "state": d.state,
                "global_id": d.global_id,
                "identity_reason_codes": list(d.identity_reason_codes),
            }
            for d in detections
        ]
        with self._lock:
            self._data[camera_id] = {
                "camera_id": camera_id,
                "updated_at": time.time(),
                "tracks": tracks,
                "people_count": len(tracks),
                "known_count": sum(1 for t in tracks if t["employee_id"]),
                "unknown_count": sum(1 for t in tracks if not t["employee_id"]),
            }

    def camera(self, camera_id: int) -> Optional[dict]:
        with self._lock:
            return self._data.get(camera_id)

    def counts(self, camera_id: int) -> tuple[int, int, int]:
        with self._lock:
            item = self._data.get(camera_id)
            if not item:
                return 0, 0, 0
            return item["people_count"], item["known_count"], item["unknown_count"]

    def drop_camera(self, camera_id: int) -> None:
        with self._lock:
            self._data.pop(camera_id, None)


@dataclass(frozen=True, slots=True)
class _CameraConfiguration:
    source_revision: int
    calibration_revision: int
    enabled: bool
    attendance: bool
    role: str | None
    calibration: DoorwayCalibration | None


def loaded_providers(*, detector=None, live=None, enroll=None, reid=None, scene=None):
    """Actual resident sessions only; optional CPU scene work is separate."""
    result = {key: list(model.session.get_providers()) if model is not None else []
              for key, model in (("person_detection", detector), ("reid", reid))}
    for prefix, face in (("live_face", live), ("enroll_face", enroll)):
        providers = face.providers() if face is not None else {}
        for kind in ("detection", "recognition"):
            result[f"{prefix}_{kind}"] = list(providers.get(kind, ()))
    providers = scene.providers() if scene is not None else {}
    for kind in ("encoder", "decoder"):
        result[f"scene_{kind}"] = list(providers.get(kind, ()))
    return result


class AIWorker(threading.Thread):
    def __init__(self, manager, service, attendance, detection_state, global_manager=None,
                 *, doorway=None, health=None, write_queue=None):
        super().__init__(name="ai-worker", daemon=True)
        self.manager, self.service, self.attendance = manager, service, attendance
        self.state, self.global_manager = detection_state, global_manager
        self.doorway = doorway if doorway is not None else DoorwayDetector()
        self.health = health if health is not None else attendance.health
        self.write_queue = write_queue if write_queue is not None else AttendanceWriteQueue(attendance, self.health)
        self._stop_event = threading.Event()
        self._last_seq, self._last_received, self._last_received_at = {}, {}, {}
        self._streams, self._configurations = {}, {}
        self._reset_epochs, self._retry_after = {}, {}
        self._gallery_revision = service.store.revision if service.store is not None else None
        self.scheduler = DeadlineScheduler(settings.ai_batch_size)
        self.metrics = InferenceMetrics()
        self._optional_rotation = 0
        self.model_ids = {}

    def stop(self, timeout=None):
        self._stop_event.set()
        if self.ident is not None:
            # The producer must stop before its durable consumers drain.
            self.join()
        else:
            self.flush("SHUTDOWN")

    def run(self):
        logger.info("AI worker started")
        try:
            while not self._stop_event.is_set():
                if not self.step():
                    maximum = max(.001, settings.ai_batch_wait_ms / 1000.)
                    delay = self.scheduler.delay(time.monotonic(), maximum)
                    self._stop_event.wait(delay if delay > 0 else .001)
        finally:
            self.flush("SHUTDOWN")
            logger.info("AI worker stopped")

    def _persist(self, events):
        for crossing in events:
            self.write_queue.submit(crossing)

    def flush(self, reason):
        for camera_id in list(self._configurations):
            self._reset_camera(camera_id, reason)
        self._persist(self.doorway.expire(time.monotonic()))

    def inference_snapshot(self):
        result = self.metrics.snapshots(time.monotonic())
        for camera_id, metric in result.items():
            reader = self.manager.get(camera_id)
            metric["decoder_mode"] = getattr(reader, "decoder_mode", None)
            metric["model_ids"] = dict(self.model_ids)
            metric["providers"] = loaded_providers(
                detector=self.service.detector, live=self.service.faces,
                reid=self.global_manager.reid if self.global_manager is not None else None)
        return result

    def _is_current(self, camera_id, packet, config, reader, epoch, gallery_revision, configurations):
        current_reader = self.manager.get(camera_id)
        latest = current_reader.buffer.latest() if current_reader is not None else None
        return (
            not self._stop_event.is_set()
            and configurations.get(camera_id) == config and config.enabled
            and self._reset_epochs.get(camera_id, 0) == epoch
            and (self.service.store.revision if self.service.store is not None else None) == gallery_revision
            and current_reader is reader and reader.is_alive()
            and reader.status() is CameraStatus.ONLINE
            and reader.source_revision == config.source_revision
            and latest is not None and latest.stream_id == packet.stream_id
            and latest.source_revision == packet.source_revision == config.source_revision
        )

    def step(self):
        now = time.monotonic()
        if not self.health.observe_clock(utcnow(), now):
            for camera_id in list(self._configurations):
                self._reset_camera(camera_id, "CLOCK_DISCONTINUITY")
        revision = self.service.store.revision if self.service.store is not None else None
        if revision != self._gallery_revision:
            for camera_id in list(self._configurations):
                self._reset_camera(camera_id, "GALLERY_CHANGED")
            self._gallery_revision = revision
        try:
            packets = self._collect_frames()
            self.health.transition("configuration", [])
            selected = self.scheduler.select((item[0] for item in packets), time.monotonic())
            # Configuration is re-read at admission, not inferred from cached readers.
            configurations = self._read_configurations() if selected else {}
        except SQLAlchemyError:
            logger.exception("Attendance camera configuration unavailable")
            self.health.transition("configuration", ["PERSISTENCE_ERROR"])
            for camera_id in list(self._configurations):
                self._reset_camera(camera_id, "CONFIGURATION_UNAVAILABLE")
            self._persist(self.doorway.expire(now))
            return False
        candidates = {camera_id: config for camera_id, _packet, config in packets}
        submissions = []
        for camera_id in selected:
            config = candidates[camera_id]
            reader = self.manager.get(camera_id)
            # Refresh the latest slot immediately before the native batch call.
            packet = reader.buffer.latest() if reader is not None else None
            if packet is None or packet.seq <= self._last_seq.get(camera_id, 0):
                continue
            if self._streams.get(camera_id) != (packet.stream_id, packet.source_revision):
                self._reset_camera(camera_id, "STREAM_RESET")
                self._streams[camera_id] = (packet.stream_id, packet.source_revision)
                self.metrics.configure(camera_id, settings.attendance_ai_fps if config.attendance else settings.ai_fps,
                                       time.monotonic())
            epoch = self._reset_epochs.get(camera_id, 0)
            if not self._is_current(camera_id, packet, config, reader, epoch, revision, configurations):
                self._reset_camera(camera_id, "SOURCE_CHANGED")
                continue
            if config.attendance and time.monotonic() - packet.received_mono > settings.attendance_frame_max_age:
                self._reset_camera(camera_id, "FRAME_GAP")
                continue
            submissions.append((camera_id, packet, config, reader, epoch))
            self.metrics.pending(camera_id, packet.received_mono)
        if not submissions:
            self._persist(self.doorway.expire(time.monotonic()))
            self.metrics.snapshots(time.monotonic())
            self.health.snapshot()
            return False
        started = time.monotonic()
        try:
            outcomes = self.service.process_batch([
                RecognitionInput(camera_id, packet, config.attendance)
                for camera_id, packet, config, _reader, _epoch in submissions])
            if len(outcomes) != len(submissions):
                raise ValueError("Recognition batch omitted a camera")
            completed = time.monotonic()
            configurations = self._read_configurations()
        except Exception:
            logger.exception("Inference batch unavailable")
            completed = time.monotonic()
            for camera_id, packet, config, _reader, _epoch in submissions:
                self._reset_camera(camera_id, "AI_UNAVAILABLE")
                self._retry_after[camera_id] = completed + 1.
                fps = settings.attendance_ai_fps if config.attendance else settings.ai_fps
                skipped = self.scheduler.advance(camera_id, fps, completed)
                self.metrics.configure(camera_id, fps, completed)
                self.metrics.complete(camera_id, packet.received_mono, completed,
                                      skipped=skipped, error_code="AI_UNAVAILABLE", started_mono=started)
                self.state.update(camera_id, [])
                if config.attendance:
                    self.health.transition(f"inference:{camera_id}", ["AI_UNAVAILABLE"],
                                           at=packet.received_at, mono=packet.received_mono)
            self._persist(self.doorway.expire(completed))
            self.metrics.snapshots(completed)
            return True
        published = []
        for submission, outcome in zip(submissions, outcomes):
            camera_id, packet, config, reader, epoch = submission
            fps = settings.attendance_ai_fps if config.attendance else settings.ai_fps
            skipped = self.scheduler.advance(camera_id, fps, completed)
            if not self._is_current(camera_id, packet, config, reader, epoch, revision, configurations):
                # Native inference may have mutated tracker votes. Reset them too;
                # stale output cannot publish boxes, visits, crossings or identities.
                self._reset_camera(camera_id, "SOURCE_CHANGED")
                continue
            self._last_seq[camera_id] = packet.seq
            self._last_received[camera_id] = packet.received_mono
            self._last_received_at[camera_id] = packet.received_at
            self.metrics.complete(camera_id, packet.received_mono, completed, skipped=skipped,
                                  face_deferred=outcome.face_deferred, started_mono=started,
                                  stage_times_ms=outcome.stage_times_ms)
            self.health.transition(f"inference:{camera_id}", [])
            if config.calibration is not None:
                self.health.record_frame(config.role, packet, tracking_generation=outcome.tracking_generation,
                                         started_mono=started, completed_mono=completed)
                self._persist(self.doorway.process(packet, camera_id, outcome, config.calibration))
            self.state.update(camera_id, outcome.detections)
            published.append((submission, outcome))
        # Every completed attendance crossing is enqueued before optional appearance.
        if self.global_manager is not None and published:
            active = set(candidates) | {
                camera_id for camera_id, receipt in self._last_received.items()
                if time.monotonic() - receipt <= settings.attendance_max_frame_gap}
            if self.scheduler.has_slack(time.monotonic(), max(.02, settings.ai_batch_wait_ms / 1000.),
                                        camera_ids=active):
                submission, outcome = published[self._optional_rotation % len(published)]
                self._optional_rotation += 1
                camera_id, packet, config, reader, epoch = submission
                try:
                    self.global_manager.process(camera_id, outcome, packet.frame, packet.received_mono)
                    configurations = self._read_configurations()
                    if self._is_current(camera_id, packet, config, reader, epoch, revision, configurations):
                        self.state.update(camera_id, outcome.detections)
                    else:
                        self._reset_camera(camera_id, "SOURCE_CHANGED")
                except Exception:
                    logger.exception("Optional global tracking failed for camera %d", camera_id)
            else:
                for submission, _outcome in published:
                    self.metrics.defer_reid(submission[0])
        self._persist(self.doorway.expire(time.monotonic()))
        self.metrics.snapshots(time.monotonic())
        self.health.snapshot()
        return True

    def _reset_camera(self, camera_id, reason="STREAM_RESET"):
        self._reset_epochs[camera_id] = self._reset_epochs.get(camera_id, 0) + 1
        if camera_id in self._last_received and reason in (
                "FRAME_GAP", "STREAM_RESET", "SOURCE_CHANGED", "CALIBRATION_CHANGED",
                "GALLERY_CHANGED", "AI_UNAVAILABLE", "CLOCK_DISCONTINUITY"):
            config = self._configurations.get(camera_id)
            if config is not None and config.attendance:
                current = utcnow()
                start = min(self._last_received_at[camera_id], current)
                self.health.publish_fault(CoverageFault(str(uuid4()), start,
                                                        self._last_received[camera_id],
                                                        max(current, self._last_received_at[camera_id]), (reason,)))
        self._persist(self.doorway.reset_camera(camera_id, reason))
        self.service.reset_camera(camera_id)
        self.state.drop_camera(camera_id)
        self.metrics.drop(camera_id)
        if self.global_manager is not None:
            self.global_manager.drop_camera(camera_id)
        for mapping in (self._streams, self._last_seq, self._last_received, self._last_received_at, self._retry_after):
            mapping.pop(camera_id, None)

    def _read_configurations(self):
        def read(db):
            result = {}
            for config, camera_enabled in db.query(AttendanceCamera, Camera.enabled).join(
                    Camera, Camera.id == AttendanceCamera.camera_id):
                attendance = config.enabled and config.role is not None
                calibration = None
                if attendance:
                    calibration = DoorwayCalibration(
                        config.camera_id, config.role, config.revision, config.source_revision,
                        tuple(config.line_start), tuple(config.line_end), tuple(config.inside_point),
                        config.reference_width, config.reference_height)
                result[config.camera_id] = _CameraConfiguration(
                    config.source_revision, config.revision, camera_enabled,
                    attendance, config.role, calibration)
            return result
        return read_snapshot(self.attendance.engine, read)

    def _collect_frames(self):
        now = time.monotonic()
        camera_ids = self._camera_ids()
        configurations = self._read_configurations()
        included = {key for key in camera_ids if key in configurations and configurations[key].enabled}
        self.scheduler.retain(included, now)
        for role in ("ENTRY", "EXIT"):
            assigned = [(key, config) for key, config in configurations.items()
                        if config.role == role and config.enabled and config.attendance]
            if assigned:
                key, config = assigned[0]
                reader = self.manager.get(key)
                online = (reader is not None and reader.is_alive() and reader.status() is CameraStatus.ONLINE
                          and reader.source_revision == config.source_revision)
                self.health.configure_role(role, camera_id=key, source_revision=config.source_revision,
                                           enabled=True, online=online, included=key in included)
            else:
                self.health.configure_role(role, camera_id=None, source_revision=0,
                                           enabled=False, online=False, included=False)
        for camera_id in set(self._configurations) - included:
            self._reset_camera(camera_id, "CAMERA_DISABLED")
            self.health.transition(f"inference:{camera_id}", [])
            self._configurations.pop(camera_id, None)
        packets = []
        for camera_id in camera_ids:
            if camera_id not in included:
                continue
            config = configurations[camera_id]
            previous = self._configurations.get(camera_id)
            if previous is not None and previous != config:
                reason = "SOURCE_CHANGED" if previous.source_revision != config.source_revision else "CALIBRATION_CHANGED"
                self._reset_camera(camera_id, reason)
            self._configurations[camera_id] = config
            self.metrics.configure(camera_id, settings.attendance_ai_fps if config.attendance else settings.ai_fps, now)
            if now < self._retry_after.get(camera_id, float("-inf")):
                continue
            reader = self.manager.get(camera_id)
            if (reader is None or not reader.is_alive() or reader.status() is not CameraStatus.ONLINE
                    or reader.source_revision != config.source_revision):
                if camera_id in self._streams:
                    self._reset_camera(camera_id, "STREAM_RESET")
                continue
            packet = reader.buffer.latest()
            if packet is None or packet.source_revision != config.source_revision:
                if camera_id in self._streams:
                    self._reset_camera(camera_id, "STREAM_RESET")
                continue
            stream = (packet.stream_id, packet.source_revision)
            if camera_id in self._streams and self._streams[camera_id] != stream:
                self._reset_camera(camera_id, "STREAM_RESET")
                self.metrics.configure(camera_id, settings.attendance_ai_fps if config.attendance else settings.ai_fps, now)
            self._streams[camera_id] = stream
            age = now - packet.received_mono
            if config.attendance and age > settings.attendance_frame_max_age:
                if camera_id in self._last_received:
                    self._reset_camera(camera_id, "FRAME_GAP")
                continue
            if packet.seq <= self._last_seq.get(camera_id, 0):
                if camera_id in self._last_received and age > settings.attendance_max_frame_gap:
                    self._reset_camera(camera_id, "FRAME_GAP")
                    self._streams[camera_id] = stream
                    self._last_seq[camera_id] = packet.seq
                continue
            previous_receipt = self._last_received.get(camera_id)
            if previous_receipt is not None and packet.received_mono - previous_receipt > settings.attendance_max_frame_gap:
                self._reset_camera(camera_id, "FRAME_GAP")
                self._streams[camera_id] = stream
                self.metrics.configure(camera_id, settings.attendance_ai_fps if config.attendance else settings.ai_fps, now)
            packets.append((camera_id, packet, config))
            self.metrics.pending(camera_id, packet.received_mono)
        return packets

    def _camera_ids(self):
        ids = set(self.manager.reader_ids())
        allowed = settings.ai_camera_ids()
        return sorted(ids if not allowed else ids & allowed)


# ------------------------------------------------------------- инициализация

def build_enroll_engine(bundle: ModelBundle | None) -> Optional[FaceEngine]:
    """One shared enrollment owner; missing face components never load a fallback."""
    if bundle is None:
        return None
    setup_cuda_dlls()
    try:
        return FaceEngine(bundle, resolve_providers(settings.ai_device), det_size=640)
    except Exception as exc:
        logger.error("Enrollment face model unavailable: %s", getattr(exc, "code", type(exc).__name__))
        return None


def build_ai_worker(
    manager: ReaderManager, attendance, store: EmbeddingStore | None,
    *, bundle: ModelBundle | None, session_factory, notifier=None, enroll_engine=None,
) -> Optional[AIWorker]:
    """Require people and journal storage, not face/gallery/appearance availability."""
    if bundle is None or session_factory is None:
        logger.error("Person model or journal database unavailable")
        return None
    setup_cuda_dlls()
    providers = resolve_providers(settings.ai_device)
    try:
        from app.database.operator_schema import preflight, VERSION
        with session_factory() as db:
            if preflight(db.connection()) != VERSION:
                raise RuntimeError("Journal schema unavailable")
        from app.ai.detector import PersonDetector
        detector = PersonDetector(bundle, providers, settings.person_confidence)
    except Exception as exc:
        logger.error("Person inference unavailable: %s", getattr(exc, "code", type(exc).__name__))
        return None

    face_engine = None
    try:
        face_engine = FaceEngine(bundle, providers, det_size=640)
    except Exception as exc:
        logger.error("Live face model unavailable: %s", getattr(exc, "code", type(exc).__name__))
    global_manager = None
    if settings.multi_camera_tracking_enabled:
        try:
            from app.ai.global_tracker import CameraTopology, GlobalIdentityManager
            from app.ai.reid import PersonReID
            topology_path = Path(settings.topology_config)
            if not topology_path.is_absolute():
                topology_path = BASE_DIR / topology_path
            global_manager = GlobalIdentityManager(
                session_factory, PersonReID(bundle, providers),
                CameraTopology.load(str(topology_path)), notifier=notifier)
        except Exception as exc:
            logger.error("Optional appearance unavailable: %s", getattr(exc, "code", type(exc).__name__))
    attendance.health.set_providers(loaded_providers(
        detector=detector, live=face_engine, enroll=enroll_engine,
        reid=global_manager.reid if global_manager is not None else None))
    service = RecognitionService(
        detector, face_engine, store,
        min_confidence=settings.min_recognition_confidence,
        min_confirmations=settings.min_confirmations,
        retry_interval=settings.face_recognition_retry_interval,
        track_lost_timeout=settings.track_lost_timeout,
        tracker_high_threshold=settings.tracker_new_track_threshold)
    state = DetectionState()
    logger.info("AI Provider: %s", provider_label(detector.session.get_providers()))
    worker = AIWorker(manager, service, attendance, state, global_manager=global_manager)
    worker.model_ids = {
        "bundle": bundle.bundle_id, "face": bundle.face_model_id,
        "person": bundle.metadata["components"]["person"]["files"][0]["sha256"],
        "body": global_manager.reid.model_id if global_manager is not None else None,
    }
    return worker
