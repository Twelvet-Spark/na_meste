"""Single inference owner: recognition → doorway → durable attendance → optional Re-ID.

Only this thread mutates local trajectories. The monitor consumes immutable FIFO
records and health notifications, never trackers. Reader publication remains the
transaction-captured CameraSource interface.
"""
import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4
from pathlib import Path

import numpy as np
from PIL import Image
from sqlalchemy.exc import SQLAlchemyError

from app.ai.doorway import DoorwayCalibration, DoorwayDetector
from app.ai.face_detector import FaceEngine
from app.ai.face_recognition import EmbeddingStore
from app.ai.model_bundle import ModelBundle
from app.ai.providers import provider_label, resolve_providers, setup_cuda_dlls
from app.ai.recognition_service import RecognitionInput, RecognitionService
from app.ai.global_tracker import EmbeddingItem
from app.ai.scheduler import DeadlineScheduler
from app.ai.inference_metrics import InferenceMetrics
from app.config import BASE_DIR, settings
from app.database.database import publication_guard
from app.database.models import AttendanceCamera, AttendanceRuntime, Camera
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

    def update(self, camera_id: int, detections, *, packet=None, generation=None, configuration=None) -> None:
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
                "visit_id": d.visit_id,
            }
            for d in detections
        ]
        with self._lock:
            self._data[camera_id] = {
                "camera_id": camera_id,
                "updated_at": time.time(),
                "source_revision": packet.source_revision if packet is not None else None,
                "stream_id": packet.stream_id if packet is not None else None,
                "frame_seq": packet.seq if packet is not None else None,
                "tracking_generation": generation,
                "frame_width": int(packet.frame.shape[1]) if packet is not None else None,
                "frame_height": int(packet.frame.shape[0]) if packet is not None else None,
                "gallery_revision": configuration.gallery_revision if configuration is not None else None,
                "calibration_revision": configuration.calibration_revision if configuration is not None else None,
                "observed_at": ((packet.received_at.replace(tzinfo=timezone.utc)
                                 if packet.received_at.tzinfo is None else packet.received_at)
                                .astimezone(timezone.utc).isoformat()) if packet is not None else None,
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

    def acknowledge(self, ack, *, global_id=None) -> None:
        with self._lock:
            current = self._data.get(ack.camera_id)
            if (current is None or current["tracking_generation"] != ack.tracking_generation
                    or current["source_revision"] != ack.source_revision
                    or current["stream_id"] != ack.stream_id):
                return
            tracks = [dict(track) for track in current["tracks"]]
            for track in tracks:
                if track["track_id"] == ack.track_id:
                    track["visit_id"] = ack.visit_id
                    if global_id is not None:
                        track["global_id"] = global_id
            self._data[ack.camera_id] = {**current, "tracks": tracks}

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
    gallery_revision: int


@dataclass(slots=True)
class _VisitBinding:
    visit_id: str
    last_mono: float
    committed: bool = False
    last_appearance: float = float("-inf")


@dataclass(frozen=True, slots=True)
class _AppearanceWork:
    visit_id: str
    camera_id: int
    tracking_generation: str
    track_id: int
    source_revision: int
    stream_id: str
    seq: int
    received_at: datetime
    received_mono: float
    bbox: tuple
    aspect: float
    crop_quality: float
    employee_id: int | None
    config: _CameraConfiguration
    epoch: int
    gallery_revision: int | None
    reader_token: int
    crop: np.ndarray


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
                 *, doorway=None, health=None, write_queue=None, journal=None, reid=None):
        super().__init__(name="ai-worker", daemon=True)
        self.manager, self.service, self.attendance = manager, service, attendance
        self.state, self.global_manager = detection_state, global_manager
        self.journal, self.reid = journal, reid
        self._journal_generations = {}
        self._visit_bindings = OrderedDict()
        self._appearance_pending = OrderedDict()
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
        self._appearance_retry_at = 0.
        self._appearance_cursor = -1
        self._appearance_seconds_per_crop = None
        self.model_ids = {}
        self._runnable_ids = set()

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
                    self._stop_event.wait(self._idle_delay(time.monotonic()))
        finally:
            self.flush("SHUTDOWN")
            logger.info("AI worker stopped")

    def _idle_delay(self, now):
        # Period waiting is not a historical frame queue. Offline/duplicate slots
        # cannot force a tight loop, and a pending retry has its own wake deadline.
        maximum = (settings.ai_batch_wait_ms / 1000. or .01) if self._runnable_ids else .01
        delay = self.scheduler.delay(now, maximum, camera_ids=self._runnable_ids)
        retries = [deadline - now for deadline in self._retry_after.values() if deadline > now]
        if retries:
            delay = min(delay, min(retries))
        return delay if delay > 0 else .001


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
                reid=self.reid)
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
        if self.journal is not None:
            self.journal.tick(utcnow())
            self._journal_health()
        for key, binding in tuple(self._visit_bindings.items()):
            if now - binding.last_mono > settings.track_lost_timeout + 5.:
                self._visit_bindings.pop(key)
                self._appearance_pending.pop(binding.visit_id, None)
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
            selected = self.scheduler.select(
                (item[0] for item in packets), time.monotonic(),
                attendance_ids=(camera_id for camera_id, _packet, config in packets if config.attendance))
            # Configuration is re-read at admission, not inferred from cached readers.
            configurations = self._read_configurations() if selected or self.journal is not None else {}
            if self.journal is not None:
                with publication_guard(self.attendance.engine):
                    configurations = self._read_configurations()
                    self._drain_journal(configurations)
        except SQLAlchemyError:
            logger.exception("Attendance camera configuration unavailable")
            self.health.transition("configuration", ["PERSISTENCE_ERROR"])
            for camera_id in list(self._configurations):
                self._reset_camera(camera_id, "CONFIGURATION_UNAVAILABLE")
            self._persist(self.doorway.expire(now))
            return False
        candidates = {camera_id: config for camera_id, _packet, config in packets}
        if selected and configurations:
            self.service.set_expected_gallery_revision(next(iter(configurations.values())).gallery_revision)
            self.service.refresh_gallery(self.attendance.engine, now=now)
            revision = self.service.store.revision if self.service.store is not None else None
            if revision != self._gallery_revision:
                for camera_id in list(self._configurations):
                    self._reset_camera(camera_id, "GALLERY_CHANGED")
                self._gallery_revision = revision
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
            self.metrics.admit_packet(camera_id, packet, reader_token=id(reader))
        if not submissions:
            self._process_appearance()
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
            self._publish_outcomes(submissions, outcomes, revision, started, completed)
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
                self.state.update(camera_id, [], packet=packet)
                if config.attendance:
                    self.health.transition(f"inference:{camera_id}", ["AI_UNAVAILABLE"],
                                           at=packet.received_at, mono=packet.received_mono)
            self._persist(self.doorway.expire(completed))
            self.metrics.snapshots(completed)
            return True
        # Service coalesced ordinary-camera crops even in attendance-only batches.
        # One due native chunk, only after all completed crossings are enqueued.
        self._process_appearance()
        self._persist(self.doorway.expire(time.monotonic()))
        self.metrics.snapshots(time.monotonic())
        self.health.snapshot()
        return True

    def _publish_outcomes(self, submissions, outcomes, revision, started, completed):
        with publication_guard(self.attendance.engine):
            configurations = self._read_configurations()
            self._drain_journal(configurations)
            for submission, outcome in zip(submissions, outcomes):
                camera_id, packet, config, reader, epoch = submission
                fps = settings.attendance_ai_fps if config.attendance else settings.ai_fps
                skipped = self.scheduler.advance(camera_id, fps, completed)
                if not self._is_current(camera_id, packet, config, reader, epoch, revision, configurations):
                    self._reset_camera(camera_id, "SOURCE_CHANGED")
                    continue
                self._last_seq[camera_id] = packet.seq
                self._last_received[camera_id] = packet.received_mono
                self._last_received_at[camera_id] = packet.received_at
                self.health.transition(
                    f"inference:{camera_id}",
                    ["FACE_MODEL_UNAVAILABLE"] if outcome.face_unavailable else [],
                    at=packet.received_at, mono=packet.received_mono)
                if config.calibration is not None:
                    self._persist(self.doorway.process(packet, camera_id, outcome, config.calibration))
                self._observe_journal(camera_id, packet, outcome)
                self.state.update(camera_id, outcome.detections, packet=packet,
                                  generation=outcome.tracking_generation, configuration=config)
                self._queue_appearance(submission, outcome, revision)
                finished = time.monotonic()
                self.metrics.complete(camera_id, packet.received_mono, finished, skipped=skipped,
                                      face_deferred=outcome.face_deferred, started_mono=started,
                                      stage_times_ms=outcome.stage_times_ms)
                if config.calibration is not None:
                    self.health.record_frame(config.role, packet, tracking_generation=outcome.tracking_generation,
                                             started_mono=started, completed_mono=finished)
            self._journal_health()

    def _observe_journal(self, camera_id, packet, outcome):
        if self.journal is None:
            return
        started = time.monotonic()
        generation = outcome.tracking_generation
        self._journal_generations[camera_id] = generation
        admissions = self.journal.observe(camera_id, packet, outcome)
        for admission in admissions:
            key = (camera_id, generation, admission.track_id)
            if admission.status == "REJECTED":
                previous = self._visit_bindings.pop(key, None)
                if previous is not None:
                    self._appearance_pending.pop(previous.visit_id, None)
                continue
            binding = self._visit_bindings.get(key)
            if binding is None:
                if len(self._visit_bindings) >= 4096:
                    _old_key, old = self._visit_bindings.popitem(last=False)
                    self._appearance_pending.pop(old.visit_id, None)
                binding = _VisitBinding(admission.visit_id, packet.received_mono)
                self._visit_bindings[key] = binding
            binding.last_mono = packet.received_mono
            self._visit_bindings.move_to_end(key)
        for transition in outcome.track_lifecycle:
            if transition.state == "EXPIRED":
                previous = self._visit_bindings.pop((camera_id, generation, transition.track_id), None)
                if previous is not None:
                    self._appearance_pending.pop(previous.visit_id, None)
        for person in outcome.observed_tracks:
            binding = self._visit_bindings.get((camera_id, generation, person.track_id))
            if binding is not None and binding.committed:
                person.visit_id = binding.visit_id
                if self.global_manager is not None:
                    person.global_id = self.global_manager.global_id_for(binding.visit_id)
        self.metrics.record_stage(camera_id, "journal", (time.monotonic() - started) * 1000.)

    def _drain_journal(self, configurations):
        if self.journal is None:
            return
        for ack in self.journal.drain_acknowledgements():
            if ack.kind == "RESET_REQUIRED":
                self.health.transition(f"journal:{ack.camera_id}:{ack.tracking_generation}", [])
                if self._journal_generations.get(ack.camera_id) == ack.tracking_generation:
                    self._reset_camera(ack.camera_id, "JOURNAL_GAP")
                continue
            config = configurations.get(ack.camera_id)
            if (config is None or not config.enabled
                    or config != self._configurations.get(ack.camera_id)
                    or config.source_revision != ack.source_revision
                    or self._journal_generations.get(ack.camera_id) != ack.tracking_generation
                    or self._streams.get(ack.camera_id) != (ack.stream_id, ack.source_revision)):
                continue
            reader = self.manager.get(ack.camera_id)
            latest = reader.buffer.latest() if reader is not None else None
            if (latest is None or not reader.is_alive() or reader.status() is not CameraStatus.ONLINE
                    or latest.stream_id != ack.stream_id or latest.source_revision != ack.source_revision):
                continue
            binding = self._visit_bindings.get((ack.camera_id, ack.tracking_generation, ack.track_id))
            if binding is None or binding.visit_id != ack.visit_id:
                continue
            if ack.kind == "COMMITTED":
                binding.committed = True
                self.state.acknowledge(ack)
            elif ack.kind == "LINK_COMMITTED" and self.global_manager is not None:
                self.global_manager.acknowledge(ack)
                global_id = self.global_manager.global_id_for(ack.visit_id)
                if global_id is not None and binding.committed:
                    self.state.acknowledge(ack, global_id=global_id)

    def _journal_health(self):
        if self.journal is None:
            return
        now, mono = utcnow(), time.monotonic()
        for camera_id, generation, started_at in self.journal.blocked_generations():
            if started_at is not None:
                self.health.transition(f"journal:{camera_id}:{generation}", ["JOURNAL_GAP"],
                                       at=started_at, mono=mono - max(0., (now - started_at).total_seconds()))
        self.health.transition("journal", ["JOURNAL_DEGRADED"]
                               if self.journal.health_snapshot()["state"] != "OK" else [])

    def _queue_appearance(self, submission, outcome, revision):
        if self.reid is None or self.global_manager is None or self.journal is None:
            return
        camera_id, packet, config, reader, epoch = submission
        height, width = packet.frame.shape[:2]
        counts = {}
        for pending in self._appearance_pending.values():
            counts[pending.camera_id] = counts.get(pending.camera_id, 0) + 1
        for person in outcome.observed_tracks:
            binding = self._visit_bindings.get((camera_id, outcome.tracking_generation, person.track_id))
            if (binding is None or person.association_uncertain
                    or packet.received_mono - binding.last_appearance < settings.reid_embedding_interval):
                continue
            x1, y1, x2, y2 = person.bbox
            left, top = max(0, int(x1 * width)), max(0, int(y1 * height))
            right, bottom = min(width, int(x2 * width)), min(height, int(y2 * height))
            if bottom - top < 48 or right <= left:
                continue
            if binding.visit_id not in self._appearance_pending and len(self._appearance_pending) >= 128:
                # Reserve representation for a new camera by displacing only
                # surplus work from a crowded one, never another quiet camera.
                crowded = max(counts, key=counts.get)
                if counts.get(camera_id, 0) or counts[crowded] <= 1:
                    self.metrics.defer_reid(camera_id)
                    continue
                victim = next(key for key in reversed(self._appearance_pending)
                              if self._appearance_pending[key].camera_id == crowded)
                self._appearance_pending.pop(victim)
                counts[crowded] -= 1
                self.metrics.defer_reid(crowded)
            crop = packet.frame[top:bottom, left:right]
            # Own only a 96 KiB crop, never a view retaining a native frame.
            resized = Image.fromarray(crop[:, :, ::-1]).resize((128, 256), Image.Resampling.BILINEAR)
            resized = np.array(np.asarray(resized)[:, :, ::-1], copy=True)
            evidence = person.fresh_face_evidence
            employee_id = (person.employee_id if evidence is not None and not evidence.reason_codes
                           and evidence.frame_seq == packet.seq
                           and evidence.confirmed_employee_id == person.employee_id
                           and evidence.store_revision == config.gallery_revision
                           else None)
            work = _AppearanceWork(
                binding.visit_id, camera_id, outcome.tracking_generation, person.track_id,
                packet.source_revision, packet.stream_id, packet.seq, packet.received_at,
                packet.received_mono, person.bbox, (bottom - top) / (right - left),
                min(1., max(0., float(person.person_score))), employee_id, config, epoch,
                revision, id(reader), resized)
            if work.visit_id not in self._appearance_pending:
                counts[camera_id] = counts.get(camera_id, 0) + 1
            self._appearance_pending[work.visit_id] = work

    def _appearance_current(self, work, configurations):
        reader = self.manager.get(work.camera_id)
        return (reader is not None and id(reader) == work.reader_token
                and self._journal_generations.get(work.camera_id) == work.tracking_generation
                and self._last_seq.get(work.camera_id) == work.seq
                and time.monotonic() - work.received_mono <= settings.track_lost_timeout
                and self._is_current(work.camera_id, work, work.config, reader, work.epoch,
                                     work.gallery_revision, configurations))

    def _appearance_limit(self, now, active):
        # Probe one crop before estimating a chunk. Native calls are not
        # preemptible; reserve time for the next live deadline, not merely 20ms.
        estimate = (self._appearance_seconds_per_crop or .02) * 1.25
        reserve = max(.005, settings.ai_batch_wait_ms / 1000.)
        slack = self.scheduler.delay(now, 16 * estimate + reserve, camera_ids=active) - reserve
        limit = min(16, max(0, int(slack / estimate)))
        return min(limit, 1) if self._appearance_seconds_per_crop is None else limit

    def _process_appearance(self):
        now = time.monotonic()
        if (not self._appearance_pending or now < self._appearance_retry_at
                or self.reid is None or self.global_manager is None):
            return
        active = {camera_id for camera_id, receipt in self._last_received.items()
                  if now - receipt <= settings.attendance_max_frame_gap}
        limit = self._appearance_limit(now, active)
        if not limit:
            for camera_id in {work.camera_id for work in self._appearance_pending.values()}:
                self.metrics.defer_reid(camera_id)
            return
        started = time.monotonic()
        batch = []
        try:
            with publication_guard(self.attendance.engine):
                configurations = self._read_configurations()
                while self._appearance_pending and len(batch) < limit:
                    cameras = sorted({work.camera_id for work in self._appearance_pending.values()})
                    camera_id = next((key for key in cameras if key > self._appearance_cursor), cameras[0])
                    self._appearance_cursor = camera_id
                    visit_id = next(key for key, work in self._appearance_pending.items()
                                    if work.camera_id == camera_id)
                    work = self._appearance_pending.pop(visit_id)
                    if self._appearance_current(work, configurations):
                        batch.append(work)
            # Metadata reads can consume the previously available slack.
            limit = self._appearance_limit(time.monotonic(), active)
            for work in reversed(batch[limit:]):
                self._appearance_pending[work.visit_id] = work
                self._appearance_pending.move_to_end(work.visit_id, last=False)
                self.metrics.defer_reid(work.camera_id)
            del batch[limit:]
            if not batch:
                return
            chunk_started = time.monotonic()
            vectors = self.reid.embed([work.crop for work in batch])
            if len(vectors) != len(batch):
                raise ValueError("Appearance batch omitted an observation")
            # Never hold the publication guard during a model call.
            with publication_guard(self.attendance.engine):
                configurations = self._read_configurations()
                self._drain_journal(configurations)
                items = []
                for work, vector in zip(batch, vectors):
                    if not self._appearance_current(work, configurations):
                        continue
                    binding = self._visit_bindings.get(
                        (work.camera_id, work.tracking_generation, work.track_id))
                    if binding is None:
                        continue
                    binding.last_appearance = work.received_mono
                    items.append(EmbeddingItem(
                        work.visit_id, work.camera_id, work.tracking_generation, work.track_id,
                        work.source_revision, work.stream_id, work.seq, work.received_at,
                        work.received_mono, work.bbox, work.aspect, work.crop_quality,
                        vector, work.employee_id))
                for proposal in self.global_manager.process_embeddings(items):
                    self.journal.submit_link(proposal)
            sample = (time.monotonic() - chunk_started) / len(batch)
            previous = self._appearance_seconds_per_crop or .02
            # Include matching/publication; react up immediately, down gradually.
            self._appearance_seconds_per_crop = max(.002, sample, .8 * previous + .2 * sample)
        except Exception:
            self._appearance_retry_at = time.monotonic() + 1.
            logger.exception("Optional appearance unavailable")
        finally:
            elapsed_ms = (time.monotonic() - started) * 1000.
            for camera_id in {work.camera_id for work in batch}:
                self.metrics.record_stage(camera_id, "appearance", elapsed_ms)

    def _reset_camera(self, camera_id, reason="STREAM_RESET"):
        self._reset_epochs[camera_id] = self._reset_epochs.get(camera_id, 0) + 1
        generation = self._journal_generations.pop(camera_id, None)
        if self.journal is not None and generation is not None:
            self.journal.reset_camera(camera_id, generation, reason)
        for key in [key for key in self._visit_bindings if key[0] == camera_id]:
            binding = self._visit_bindings.pop(key)
            self._appearance_pending.pop(binding.visit_id, None)
        if camera_id in self._last_received and reason in (
                "FRAME_GAP", "STREAM_RESET", "SOURCE_CHANGED", "CALIBRATION_CHANGED",
                "GALLERY_CHANGED", "AI_UNAVAILABLE", "CLOCK_DISCONTINUITY", "JOURNAL_GAP"):
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
        self.metrics.clear_pending(camera_id)
        if self.global_manager is not None:
            self.global_manager.drop_camera(camera_id)
        for mapping in (self._streams, self._last_seq, self._last_received, self._last_received_at, self._retry_after):
            mapping.pop(camera_id, None)

    def _read_configurations(self):
        def read(db):
            result = {}
            runtime = db.get(AttendanceRuntime, 1)
            if runtime is None:
                raise SQLAlchemyError("Attendance runtime unavailable")
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
                    attendance, config.role, calibration, runtime.gallery_revision)
            return result
        return read_snapshot(self.attendance.engine, read)

    def _collect_frames(self):
        self._runnable_ids.clear()
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
            self.metrics.drop(camera_id)
            self.health.transition(f"inference:{camera_id}", [])
            self._configurations.pop(camera_id, None)
        packets = []
        for camera_id in camera_ids:
            if camera_id not in included:
                continue
            config = configurations[camera_id]
            previous = self._configurations.get(camera_id)
            if previous is not None and previous != config:
                reason = ("SOURCE_CHANGED" if previous.source_revision != config.source_revision else
                          "GALLERY_CHANGED" if previous.gallery_revision != config.gallery_revision else
                          "CALIBRATION_CHANGED")
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
        self._runnable_ids = {camera_id for camera_id, _packet, _config in packets}
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
    *, bundle: ModelBundle | None, session_factory, journal, notifier=None, enroll_engine=None,
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
    reid = None
    if settings.multi_camera_tracking_enabled:
        try:
            from app.ai.global_tracker import CameraTopology, GlobalIdentityManager
            from app.ai.reid import PersonReID
            topology_path = Path(settings.topology_config)
            if not topology_path.is_absolute():
                topology_path = BASE_DIR / topology_path
            reid = PersonReID(bundle, providers)
            global_manager = GlobalIdentityManager(CameraTopology.load(str(topology_path)), notifier=notifier)
        except Exception as exc:
            logger.error("Optional appearance unavailable: %s", getattr(exc, "code", type(exc).__name__))
    attendance.health.set_providers(loaded_providers(
        detector=detector, live=face_engine, enroll=enroll_engine,
        reid=reid))
    service = RecognitionService(
        detector, face_engine, store,
        min_confidence=settings.min_recognition_confidence,
        min_confirmations=settings.min_confirmations,
        retry_interval=settings.face_recognition_retry_interval,
        track_lost_timeout=settings.track_lost_timeout,
        tracker_high_threshold=settings.tracker_new_track_threshold)
    state = DetectionState()
    logger.info("AI Provider: %s", provider_label(detector.session.get_providers()))
    worker = AIWorker(manager, service, attendance, state, global_manager=global_manager,
                      journal=journal, reid=reid)
    worker.model_ids = {
        "bundle": bundle.bundle_id, "face": bundle.face_model_id,
        "person": bundle.metadata["components"]["person"]["files"][0]["sha256"],
        "body": reid.model_id if reid is not None else None,
    }
    return worker
