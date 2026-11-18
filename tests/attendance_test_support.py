"""Shared fixtures for work-time tests: an isolated v3 database, real worker pieces, a fake clock.

No production DB, models, RTSP subprocesses or network calls; model observations are supplied
by tiny doubles while tracking, doorway, identity and the presence recorder stay real.
"""
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session

from app.ai.doorway import DoorwayDetector
from app.ai.face_recognition import EmbeddingStore
from app.ai.recognition_service import RecognitionService
from app.ai.worker import AIWorker, DetectionState
from app.database.attendance_schema import initialize_attendance_schema
from app.database.database import immediate_session
from app.database.models import AttendanceCamera, AttendanceRuntime, Camera, CameraProfile, Employee
from app.rtsp.reader import CameraStatus, FrameBuffer
from app.services.presence_recorder import PresenceRecorder

BASE = datetime(2026, 11, 11, 4)  # 09:00 UTC+05:00
AS_OF = datetime(2026, 11, 12, 15)
FRAME = np.zeros((480, 640, 3), dtype=np.uint8)
FRAME.setflags(write=False)


def model_bundle_fixture():
    """Verified startup metadata double, not evidence of real model execution."""
    preprocessing = {
        "person": "dfine-rgb-letterbox640-black-div255-v1",
        "face_detection": "yunet-bgr-f32-strides8-16-32-v1",
        "face_recognition": "auraface-rgb112-minus127.5-div127.5-v1",
        "body": "osnet-rgb256x128-imagenet-v1",
        "scene": "mobilesam-rgb-longest1024-meanstd-pad-v1",
    }
    components = {
        name: {"preprocessing_id": policy,
               "files": [{"role": role, "sha256": str(index) * 64,
                          "path": f"{name}-{role}.onnx"} for role in
                         (("encoder", "decoder") if name == "scene" else ("model",))]}
        for index, (name, policy) in enumerate(preprocessing.items(), 1)
    }
    return SimpleNamespace(
        bundle_id="a" * 64, face_model_id="b" * 64, alignment_id="arcface-5pt-112-v1",
        metadata={"version": 1, "alignment_id": "arcface-5pt-112-v1", "components": components},
        component_status={name: "READY" for name in components}, component_paths={},
    )


@contextmanager
def database(*, roles=("ENTRY", "EXIT"), time_tracking=True):
    """Temporary v3 database with employees 1-2 and one door camera per role; yields (engine, presence)."""
    with tempfile.TemporaryDirectory() as directory:
        engine = create_engine(
            f"sqlite:///{Path(directory).as_posix()}/attendance.db",
            connect_args={"timeout": 30, "check_same_thread": False},
        )

        @event.listens_for(engine, "connect")
        def pragmas(connection, _record):
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA foreign_keys=ON")

        initialize_attendance_schema(engine)
        try:
            with immediate_session(engine) as db:
                db.add_all([Employee(id=1, name="Employee 1"), Employee(id=2, name="Other")])
                for camera_id, role in enumerate(roles, 1):
                    db.add(Camera(id=camera_id, name=role, nvr_host="unused", username="u",
                                  password="p", channel=camera_id, enabled=True))
                db.flush()
                db.add_all([CameraProfile(camera_id=camera_id, zone="ENTRANCE", ai_enabled=True,
                                          time_tracking=time_tracking, revision=0)
                            for camera_id in range(1, len(roles) + 1)])
                for camera_id, role in enumerate(roles, 1):
                    db.add(AttendanceCamera(
                        camera_id=camera_id, role=role, enabled=True, revision=3, source_revision=2,
                        line_start=[.2, .5], line_end=[.8, .5], inside_point=[.5, .8],
                        door_kind="DOOR", door_top=[[.2, .1], [.8, .1]],
                        reference_width=640, reference_height=480,
                    ))
                db.get(AttendanceRuntime, 1).gallery_revision = 7
            yield engine, PresenceRecorder(engine, now=lambda: AS_OF)
        finally:
            engine.dispose()


class Clock:
    def __init__(self):
        self.mono = 1000.
        self.utc = BASE

    def advance(self, seconds, *, utc_seconds=None):
        self.mono += seconds
        self.utc += timedelta(seconds=seconds if utc_seconds is None else utc_seconds)

    def monotonic(self):
        return self.mono

    def utcnow(self):
        return self.utc


class PersonObservations:
    """Only model inference is supplied; tracking/ownership/crossing stay real."""
    def __init__(self):
        self.session = SimpleNamespace(get_providers=lambda: ["CPUExecutionProvider"])
        self.y = .25

    def detect(self, frame):
        if self.y is None:
            return []
        return [(np.array([220., self.y * 480 - 200., 420., self.y * 480]), .99)]

    def detect_batch(self, frames):
        return [self.detect(frame) for frame in frames]


class NoFaces:
    model_id = "a" * 64
    alignment_id = "arcface-5pt-112-v1"

    def detect_faces(self, image):
        return []

    def providers(self):
        return {"detection": [], "recognition": []}


class BufferedReader:
    def __init__(self):
        self.buffer = FrameBuffer()
        self.source_revision = 2
        self.online = True

    def is_alive(self):
        return True

    def status(self):
        return CameraStatus.ONLINE if self.online else CameraStatus.OFFLINE


class Readers:
    def __init__(self):
        self.readers = {1: BufferedReader(), 2: BufferedReader()}

    def reader_ids(self):
        return list(self.readers)

    def get(self, camera_id):
        return self.readers.get(camera_id)


def worker_for(engine, presence, clock):
    """The real AIWorker over fake readers and detections; returns (worker, manager, observations)."""
    store = EmbeddingStore("a" * 64)
    with Session(engine) as db:
        store.refresh(db)
    observations = PersonObservations()
    recognition = RecognitionService(observations, NoFaces(), store, min_hits=1)
    manager = Readers()
    worker = AIWorker(manager, recognition, presence, DetectionState(), doorway=DoorwayDetector())
    return worker, manager, observations


def step(worker, manager, observations, clock, y=None, *, publish=True, stream="source-a"):
    """Publish one frame on camera 1 with the person at normalized height `y`, then run one worker step."""
    observations.y = y
    if publish:
        manager.readers[1].buffer.put(FRAME, stream_id=stream, source_revision=2,
                                     received_at=clock.utc, received_mono=clock.mono)
    with patch("app.ai.worker.time.monotonic", side_effect=clock.monotonic), \
            patch("app.ai.worker.utcnow", side_effect=clock.utcnow), \
            patch("app.ai.doorway.time.monotonic", side_effect=clock.monotonic):
        worker.step()
    clock.advance(.25)
