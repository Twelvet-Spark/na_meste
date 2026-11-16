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
from app.ai.recognition_service import RecognitionService
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
        self._last_seq, self._last_ts, self._last_received = {}, {}, {}
        self._last_received_at = {}
        self._streams, self._configurations = {}, {}
        self._gallery_revision = service.store.revision if service.store is not None else None
        self._rotation = 0

    def stop(self, timeout=None):
        self._stop_event.set()
        if self.ident is not None:
            # Do not stop the monitor while a timed-out producer can still enqueue.
            self.join()
        else:
            self.flush("SHUTDOWN")

    def run(self):
        logger.info("AI worker started")
        try:
            while not self._stop_event.is_set():
                did_work = self.step()
                if not did_work:
                    self._stop_event.wait(.05)
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
        except SQLAlchemyError:
            logger.exception("Attendance camera configuration unavailable")
            self.health.transition("configuration", ["PERSISTENCE_ERROR"])
            for camera_id in list(self._configurations):
                self._reset_camera(camera_id, "CONFIGURATION_UNAVAILABLE")
            self._persist(self.doorway.expire(now))
            return False
        priority = [item for item in packets if item[2].attendance]
        others = [item for item in packets if not item[2].attendance]
        if priority:
            offset = self._rotation % len(priority)
            priority = priority[offset:] + priority[:offset]
            self._rotation += 1
        did_work = False
        for camera_id, packet, config in priority + others:
            if self._stop_event.is_set():
                break
            started = time.monotonic()
            if config.attendance and started - packet.received_mono > settings.attendance_frame_max_age:
                self._reset_camera(camera_id, "FRAME_GAP")
                continue
            fps = settings.attendance_ai_fps if config.attendance else max(.5, settings.ai_fps)
            if started - self._last_ts.get(camera_id, float("-inf")) < 1 / fps:
                continue
            self._last_seq[camera_id], self._last_ts[camera_id] = packet.seq, started
            self._last_received[camera_id] = packet.received_mono
            self._last_received_at[camera_id] = packet.received_at
            try:
                outcome = self.service.process_frame(
                    packet.frame, camera_id, packet.received_mono, frame_seq=packet.seq,
                    received_at=packet.received_at, attendance=config.attendance)
                completed = time.monotonic()
                self.health.transition(f"inference:{camera_id}", [])
                if config.calibration is not None:
                    self.health.record_frame(config.role, packet, tracking_generation=outcome.tracking_generation,
                                             started_mono=started, completed_mono=completed)
                    self._persist(self.doorway.process(packet, camera_id, outcome, config.calibration))
            except Exception:
                logger.exception("Camera %d inference failed", camera_id)
                if config.attendance:
                    self.health.transition(f"inference:{camera_id}", ["AI_UNAVAILABLE"],
                                           at=packet.received_at, mono=packet.received_mono)
                self._reset_camera(camera_id, "AI_UNAVAILABLE")
                self._streams[camera_id] = (packet.stream_id, packet.source_revision)
                self._last_seq[camera_id] = packet.seq
                continue
            # This optional layer cannot influence a work key or delay saving the
            # completed crossing produced by this frame.
            if self.global_manager is not None:
                try:
                    self.global_manager.process(camera_id, outcome, packet.frame, packet.received_mono)
                except Exception:
                    logger.exception("Optional global tracking failed for camera %d", camera_id)
            self.state.update(camera_id, outcome.detections)
            did_work = True
        self._persist(self.doorway.expire(time.monotonic()))
        self.health.snapshot()
        return did_work

    def _reset_camera(self, camera_id, reason="STREAM_RESET"):
        # Flush BEFORE destroying recognition/generation/global bindings.
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
        if self.global_manager is not None:
            self.global_manager.drop_camera(camera_id)
        for mapping in (self._streams, self._last_seq, self._last_ts, self._last_received, self._last_received_at):
            mapping.pop(camera_id, None)

    def _read_configurations(self):
        def read(db):
            result = {}
            for config, camera_enabled in db.query(AttendanceCamera, Camera.enabled).join(
                    Camera, Camera.id == AttendanceCamera.camera_id):
                calibration = None
                if config.enabled and config.role is not None:
                    calibration = DoorwayCalibration(
                        config.camera_id, config.role, config.revision, config.source_revision,
                        tuple(config.line_start), tuple(config.line_end), tuple(config.inside_point),
                        config.reference_width, config.reference_height)
                result[config.camera_id] = _CameraConfiguration(
                    config.source_revision, config.revision,
                    camera_enabled and (config.role is None or config.enabled),
                    config.role is not None, config.role, calibration)
            return result
        return read_snapshot(self.attendance.engine, read)

    def _collect_frames(self):
        camera_ids = self._camera_ids()
        configurations = self._read_configurations()
        included = {key for key in camera_ids if key in configurations and configurations[key].enabled}
        for role in ("ENTRY", "EXIT"):
            assigned = [(key, config) for key, config in configurations.items() if config.role == role and config.enabled]
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
            self._streams[camera_id] = stream
            age = time.monotonic() - packet.received_mono
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
            packets.append((camera_id, packet, config))
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
    return AIWorker(manager, service, attendance, state, global_manager=global_manager)
