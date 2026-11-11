"""Frame receipt and consumer continuity contracts; no cameras/models/production DB.

Run: .venv/Scripts/python.exe tests/test_attendance_frames.py
"""
import io
import sys
import tempfile
import traceback
from contextlib import contextmanager
from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
from fastapi import HTTPException
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from app.ai.face_recognition import EmbeddingStore
from app.ai.global_tracker import GlobalIdentityManager
from app.ai.recognition_service import RecognitionService
from app.ai.worker import AIWorker, DetectionState
from app.api import cameras as api
from app.config import settings
from app.database.attendance_schema import initialize_attendance_schema
from app.database.models import AttendanceCamera, Camera
from app.rtsp.reader import CameraStatus, ConnectionLost, FrameBuffer, RTSPReader

TESTS = []
STAMP = datetime(2026, 11, 11, 4, 0, 0, 123456)
FRAME = np.zeros((240, 320, 3), dtype=np.uint8)


def test(fn):
    TESTS.append(fn)
    return fn


def put(buffer, *, stream="stream-a", revision=1, mono=100.0, stamp=STAMP, frame=FRAME):
    return buffer.put(frame, stream_id=stream, source_revision=revision,
                      received_at=stamp, received_mono=mono)


def reader():
    return RTSPReader(1, "Door", "rtsp://unused", source_revision=7,
                      ffmpeg_path="unused", ffprobe_path="unused")


@contextmanager
def database():
    with tempfile.TemporaryDirectory() as directory:
        engine = create_engine(f"sqlite:///{Path(directory).as_posix()}/frames.db")

        @event.listens_for(engine, "connect")
        def foreign_keys(connection, _record):
            connection.execute("PRAGMA foreign_keys=ON")

        initialize_attendance_schema(engine)
        try:
            with Session(engine) as db:
                db.add(Camera(id=1, name="Door", nvr_host="unused", username="u",
                              password="p", channel=1))
                db.commit()
                db.add(AttendanceCamera(camera_id=1))
                db.commit()
                yield db
        finally:
            engine.dispose()


class StaticManager:
    def __init__(self, value):
        self.value = value

    def get(self, camera_id):
        assert camera_id == 1
        return self.value


@test
def test_packet_keeps_receipt_metadata_and_frame_without_copying():
    buffer = FrameBuffer()
    assert put(buffer) == 1
    packet = buffer.latest()
    assert packet.seq == 1 and packet.stream_id == "stream-a"
    assert packet.source_revision == 1 and packet.received_at == STAMP
    assert packet.received_at.tzinfo is None and packet.received_mono == 100.0
    assert packet.frame is FRAME and not hasattr(packet, "__dict__")
    try:
        packet.seq = 9
    except FrozenInstanceError:
        pass
    else:
        raise AssertionError("Published receipt metadata must be immutable")
    assert buffer.wait_for_new(0, 0) is packet
    assert buffer.wait_for_new(1, 0) is None


@test
def test_clear_drops_frames_but_keeps_buffer_sequence_monotonic():
    buffer = FrameBuffer()
    put(buffer)
    buffer.clear()
    assert buffer.latest() is None and buffer.wait_for_new(0, 0) is None
    assert put(buffer, stream="stream-b", mono=100.25, stamp=STAMP + timedelta(seconds=.25)) == 2
    packet = buffer.wait_for_new(1, 0)
    assert packet.stream_id == "stream-b" and packet.seq == 2
    assert packet.received_mono == 100.25
    assert packet.received_at == datetime(2026, 11, 11, 4, 0, 0, 373456)


@test
def test_reader_stop_clears_public_packet():
    value = reader()
    put(value.buffer)
    assert value.read().frame is FRAME
    value.stop()
    assert value.read() is None and value.status() is CameraStatus.OFFLINE


class Process:
    def __init__(self):
        self.stdout = io.BytesIO(b"\x00" * 12)
        self.stderr = io.BytesIO()

    def poll(self):
        return 0


@test
def test_complete_receipt_gets_new_connection_uuid_and_disconnect_clears():
    value = reader()
    packets = []
    original_put = value.buffer.put

    def capture(frame, **metadata):
        seq = original_put(frame, **metadata)
        packets.append(value.buffer.latest())
        return seq

    with patch.object(value, "_probe", return_value=(2, 2, "h264", 15.0)), \
            patch("app.rtsp.reader.subprocess.Popen", side_effect=lambda *a, **k: Process()), \
            patch.object(value.buffer, "put", side_effect=capture), \
            patch("app.rtsp.reader.time.monotonic", return_value=123.5), \
            patch("app.rtsp.reader.utcnow", return_value=STAMP):
        for _ in range(2):
            try:
                value._connect_and_read()
            except ConnectionLost:
                pass
            else:
                raise AssertionError("Finite raw pipe must signal disconnect")
            assert value.buffer.latest() is None
    assert len(packets) == 2
    assert packets[0].stream_id != packets[1].stream_id
    assert packets[0].seq == 1 and packets[1].seq == 2
    for packet in packets:
        assert packet.source_revision == 7
        assert packet.received_at == STAMP and packet.received_mono == 123.5
        assert packet.frame.shape == (2, 2, 3)


@test
def test_snapshot_rejects_stale_or_non_online_packets():
    with database() as db:
        value = reader()
        value._set_status(CameraStatus.ONLINE)
        put(value.buffer, mono=100.0)
        with patch("app.api.cameras.time.monotonic", return_value=100.0 + settings.frame_stall_timeout + .001):
            try:
                api.api_camera_snapshot(1, db=db, manager=StaticManager(value))
            except HTTPException as exc:
                assert exc.status_code == 503
            else:
                raise AssertionError("A stale packet must not be returned as a snapshot")
        for status in (CameraStatus.RECONNECTING, CameraStatus.OFFLINE, CameraStatus.ERROR):
            value._set_status(status)
            with patch("app.api.cameras.time.monotonic", return_value=100.0):
                try:
                    api.api_camera_snapshot(1, db=db, manager=StaticManager(value))
                except HTTPException as exc:
                    assert exc.status_code == 503
                else:
                    raise AssertionError(f"{status} is not a usable snapshot")


@test
def test_snapshot_encodes_the_fresh_packet_not_a_tuple():
    with database() as db:
        value = reader()
        value._set_status(CameraStatus.ONLINE)
        put(value.buffer)
        with patch("app.api.cameras.time.monotonic", return_value=100.1):
            response = api.api_camera_snapshot(1, db=db, manager=StaticManager(value))
        assert response.media_type == "image/jpeg"
        image = cv2.imdecode(np.frombuffer(response.body, np.uint8), cv2.IMREAD_COLOR)
        assert image.shape == FRAME.shape


@test
def test_mjpeg_encodes_packet_and_keeps_preview_limiter_independent():
    value = reader()
    put(value.buffer)
    value._set_status(CameraStatus.ONLINE)
    with patch.object(value, "is_alive", return_value=True), \
            patch("app.api.cameras.time.monotonic", return_value=100.1):
        stream = api._mjpeg_generator(value, "Door")
        part = next(stream)
        stream.close()
    assert part.startswith(b"--frame\r\nContent-Type: image/jpeg\r\n")
    assert b"\xff\xd8" in part and b"\xff\xd9" in part


class EndLoop(Exception):
    pass


class PersonDetector:
    def detect(self, frame):
        return [(np.array([30., 20., 160., 230.]), .99)]


class NoFaces:
    def detect(self, image, max_num=0):
        return []


class LoopManager:
    """Deterministic camera source changes at the worker's next loop boundary."""
    def __init__(self, stages, clock, on_stage):
        self.stages = iter(stages)
        self.current = None
        self.clock = clock
        self.index = -1
        self.on_stage = on_stage

    def reader_ids(self):
        try:
            self.current, self.clock[0] = next(self.stages)
        except StopIteration:
            raise EndLoop()
        self.index += 1
        self.on_stage(self.index)
        return [1] if self.current is not None else []

    def get(self, camera_id):
        return self.current


class FrameReader:
    def __init__(self, stream, seq, mono, *, revision=1, frame=FRAME):
        self.buffer = FrameBuffer()
        self.source_revision = revision
        self.buffer._seq = seq - 1
        put(self.buffer, stream=stream, revision=revision, mono=mono, frame=frame)

    def is_alive(self):
        return True

    def status(self):
        return CameraStatus.ONLINE


class Presence:
    def tick(self):
        pass

    def on_recognized(self, camera_id, employee_id, confidence):
        raise AssertionError("No-face fixture cannot recognize an employee")


def run_worker(stages, *, attendance=False, on_stage=None, global_reid=None):
    clock = [100.0]
    service = RecognitionService(PersonDetector(), NoFaces(), EmbeddingStore(), min_hits=1)
    outcomes = []
    process = service.process_frame

    def collect(*args, **kwargs):
        outcome = process(*args, **kwargs)
        outcomes.append(outcome)
        return outcome

    state = DetectionState()
    with database() as db:
        if attendance:
            config = db.get(AttendanceCamera, 1)
            config.role, config.enabled = "ENTRY", True
            config.line_start, config.line_end, config.inside_point = [0., .5], [1., .5], [.5, .8]
            config.reference_width, config.reference_height = 320, 240
            db.commit()

        def advance(index):
            if on_stage is not None:
                on_stage(index, db)

        factory = sessionmaker(bind=db.get_bind())
        global_manager = (
            GlobalIdentityManager(factory, global_reid, topology=None)
            if global_reid is not None else None
        )
        worker = AIWorker(
            LoopManager(stages, clock, advance), service, Presence(), state,
            session_factory=factory, global_manager=global_manager,
        )
        with patch.object(service, "process_frame", side_effect=collect), \
                patch("app.ai.worker.time.monotonic", side_effect=lambda: clock[0]), \
                patch.object(settings, "ai_cameras", ""), patch.object(settings, "ai_fps", 100.0):
            try:
                worker.run()
            except EndLoop:
                pass
    return service, state, outcomes


@test
def test_replacement_seq_one_after_old_seq_hundred_is_processed_with_new_generation():
    _service, _state, outcomes = run_worker([
        (FrameReader("old", 100, 100.0), 100.0),
        (FrameReader("new", 1, 100.1), 100.1),
    ])
    assert len(outcomes) == 2, "Replacement reader sequence must not be compared to old buffer"
    assert outcomes[0].tracking_generation != outcomes[1].tracking_generation
    assert outcomes[0].detections[0].track_id == outcomes[1].detections[0].track_id == 1


@test
def test_tracker_receives_each_packets_receipt_monotonic_not_loop_time():
    service, _state, outcomes = run_worker([(FrameReader("a", 1, 99.6), 100.0)])
    assert len(outcomes) == 1
    assert service._trackers[1].tracks[0].last_seen == 99.6
    evidence = outcomes[0].detections[0].fresh_face_evidence
    assert evidence.frame_seq == 1 and evidence.received_at == STAMP
    assert evidence.received_mono == 99.6


@test
def test_continuity_gap_resets_same_stream_generation_and_detection_state():
    _service, _state, outcomes = run_worker([
        (FrameReader("a", 1, 100.0), 100.0),
        (FrameReader("a", 2, 100.8), 100.8),
    ])
    assert len(outcomes) == 2
    assert outcomes[0].tracking_generation != outcomes[1].tracking_generation
    assert outcomes[1].detections[0].track_id == 1


@test
def test_source_revision_change_resets_even_if_stream_uuid_did_not_change():
    def change(index, db):
        if index == 1:
            db.get(AttendanceCamera, 1).source_revision = 2
            db.commit()

    _service, _state, outcomes = run_worker([
        (FrameReader("a", 1, 100.0, revision=1), 100.0),
        (FrameReader("a", 2, 100.1, revision=2), 100.1),
    ], on_stage=change)
    assert len(outcomes) == 2
    assert outcomes[0].tracking_generation != outcomes[1].tracking_generation


@test
def test_disabled_camera_immediately_drops_cached_detections():
    _service, state, outcomes = run_worker([
        (FrameReader("a", 1, 100.0), 100.0), (None, 100.1),
    ])
    assert len(outcomes) == 1
    assert state.camera(1) is None, "Disable/removal must not wait for ten-second cleanup"


@test
def test_calibration_change_resets_identity_on_same_stream():
    def change(index, db):
        if index == 1:
            db.get(AttendanceCamera, 1).revision += 1
            db.commit()

    _service, _state, outcomes = run_worker([
        (FrameReader("a", 1, 100.0), 100.0),
        (FrameReader("a", 2, 100.1), 100.1),
    ], attendance=True, on_stage=change)
    assert len(outcomes) == 2
    assert outcomes[0].tracking_generation != outcomes[1].tracking_generation


@test
def test_stale_attendance_packet_is_rejected_and_old_detections_are_removed():
    _service, state, outcomes = run_worker([
        (FrameReader("a", 1, 100.0), 100.0),
        (FrameReader("a", 2, 100.1), 101.101),
    ], attendance=True)
    assert len(outcomes) == 1 and state.camera(1) is None


@test
def test_packet_cannot_override_committed_source_revision():
    _service, state, outcomes = run_worker([
        (FrameReader("a", 1, 100.0), 100.0),
        (FrameReader("a", 2, 100.1, revision=2), 100.1),
    ], attendance=True)
    assert len(outcomes) == 1 and state.camera(1) is None


@test
def test_disabling_attendance_calibration_immediately_clears_identity():
    def change(index, db):
        if index == 1:
            db.get(AttendanceCamera, 1).enabled = False
            db.commit()

    _service, state, outcomes = run_worker([
        (FrameReader("a", 1, 100.0), 100.0),
        (FrameReader("a", 2, 100.1), 100.1),
    ], attendance=True, on_stage=change)
    assert len(outcomes) == 1 and state.camera(1) is None


@test
def test_ai_exclusion_immediately_clears_old_detections():
    def change(index, db):
        if index == 1:
            settings.ai_cameras = "2"

    _service, state, outcomes = run_worker([
        (FrameReader("a", 1, 100.0), 100.0),
        (FrameReader("a", 2, 100.1), 100.1),
    ], on_stage=change)
    assert len(outcomes) == 1 and state.camera(1) is None


@test
def test_worker_reconnect_cannot_reuse_previous_global_local_id_binding():
    class DifferentPeopleReID:
        def embed(self, crops):
            result = np.zeros((len(crops), 512), dtype=np.float32)
            for index, crop in enumerate(crops):
                result[index, 0 if crop.mean() < 127 else 1] = 1.
            return result

    _service, _state, outcomes = run_worker([
        (FrameReader("old", 100, 100.0), 100.0),
        (FrameReader("new", 1, 100.1, frame=np.full_like(FRAME, 255)), 100.1),
    ], global_reid=DifferentPeopleReID())
    assert len(outcomes) == 2
    first, second = (item.detections[0] for item in outcomes)
    assert first.global_id is not None and second.global_id is not None
    assert first.global_id != second.global_id, "Local ID 1 must not inherit its previous binding"


if __name__ == "__main__":
    failed = 0
    for case in TESTS:
        try:
            case()
            print(f"PASS {case.__name__}")
        except Exception:
            failed += 1
            print(f"FAIL {case.__name__}")
            traceback.print_exc()
    print(f"{len(TESTS) - failed}/{len(TESTS)} passed")
    sys.exit(bool(failed))
