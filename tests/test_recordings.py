"""Task 4 contracts: real temporary SQLite/filesystem, fake OS media boundary.

Run: .venv/Scripts/python.exe tests/test_recordings.py
Opt-in owned-parent integration: append --windows-process-integration
No main import/lifespan, production DB, RTSP device or external network access.
The process double writes only fixture bytes; it does NOT prove decodability.
Main owns real FFmpeg/browser proof. No production tick/test-only API is needed.
"""
import asyncio
import importlib
import io
import json
import os
import subprocess
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from urllib.parse import urlencode, urlsplit
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi import FastAPI, HTTPException
from pydantic import ValidationError
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.config import BASE_DIR, Settings, settings
from app.database import attendance_schema
from app.database.database import get_db
from app.database.models import Camera, OperatorRequest
from app.services import camera_service
from operator_test_support import Client
from test_operator_storage import database, catalog, ATTENDANCE_TABLES

TESTS = []
BASE = datetime(2026, 11, 12, 10, 0, 0)
ACTIVE = {"STARTING", "RECORDING", "RECONNECTING", "STOPPING"}
VIEW_KEYS = {"id", "camera_id", "camera_name", "source_revision", "requested_at",
             "started_at", "deadline_at", "ended_at", "status", "reason_code",
             "bytes_recorded", "available_from", "available_to"}
SECRET = "do-not-expose-recording-password"


def test(fn):
    TESTS.append(fn)
    return fn


def feature(name):
    # RED is an explicit missing-feature assertion, never a collection failure.
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as exc:
        if exc.name == name or name.startswith(exc.name + "."):
            raise AssertionError(f"Required media feature is not implemented: {name}") from None
        raise


def eventually(predicate, message="worker did not publish the required state", timeout=6):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = predicate()
        if result:
            return result
        threading.Event().wait(.02)
    raise AssertionError(message)


def rejects(status, operation, code=None):
    try:
        operation()
    except HTTPException as exc:
        assert exc.status_code == status, (exc.status_code, exc.detail)
        if code:
            assert code in str(exc.detail), exc.detail
        assert SECRET not in str(exc.detail)
        return exc
    raise AssertionError(f"Expected HTTP {status}")


def view(value):
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    elif not isinstance(value, dict):
        from dataclasses import asdict
        value = asdict(value)
    assert set(value) == VIEW_KEYS, value
    assert SECRET not in str(value) and "rtsp://" not in str(value)
    return value


def utc(value):
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return value.replace(tzinfo=None) if value.tzinfo is None else value.astimezone(timezone.utc).replace(tzinfo=None)


def rows(engine, table):
    with engine.connect() as connection:
        return [dict(row) for row in connection.exec_driver_sql(f"SELECT * FROM {table}").mappings()]


class Clock:
    def __init__(self):
        self.lock = threading.Lock()
        self.utc = BASE
        self.mono = 1000.

    def advance(self, seconds, *, utc_seconds=None):
        with self.lock:
            self.mono += seconds
            self.utc += timedelta(seconds=seconds if utc_seconds is None else utc_seconds)

    def now(self):
        with self.lock:
            return self.utc

    def monotonic(self):
        with self.lock:
            return self.mono


class InputPipe(io.BytesIO):
    def __init__(self, process):
        super().__init__()
        self.process = process

    def write(self, data):
        if b"q" in (data.encode() if isinstance(data, str) else data):
            self.process.finish()
        return len(data)


class MediaProcess:
    """Popen boundary only; manager still owns CSV, probing, SQL and scheduling."""
    def __init__(self, args, factory, **kwargs):
        assert isinstance(args, list), "No shell command strings"
        assert not kwargs.get("shell", False)
        assert args[0] == "ffmpeg-fixture"
        assert args[args.index("-rtsp_transport") + 1] == "tcp"
        assert args[args.index("-map") + 1] == "0:v:0"
        assert all(flag in args for flag in ("-an", "-sn", "-dn"))
        codec_flag = "-c:v" if "-c:v" in args else "-codec:v"
        assert args[args.index(codec_flag) + 1] == "copy"
        assert args[args.index("-segment_time") + 1] == "60"
        assert args[args.index("-reset_timestamps") + 1] == "1"
        self.pattern = Path(args[-1])
        self.csv = Path(args[args.index("-segment_list") + 1])
        assert self.pattern.suffix == ".mkv"
        assert self.pattern.parent == self.csv.parent
        assert self.pattern.parent.is_dir(), "Owned connection directory must precede launch"
        connections = rows(factory.engine, "recording_connections")
        assert any((factory.root / row["directory"]).resolve() == self.pattern.parent.resolve()
                   for row in connections), "Connection must commit before Popen"
        self.factory = factory
        self.stdout, self.stderr = io.BytesIO(), io.BytesIO()
        self.stdin = InputPipe(self)
        self.returncode = None
        self.pid = 900000 + len(factory.processes)
        self.final_seconds = 0
        self.terminated = False
        self.paths = []
        self.lock = threading.RLock()

    def appear(self, sequence=0, *, duration=59., codec="h264"):
        with self.lock:
            path = Path(str(self.pattern) % sequence)
            path.write_text(json.dumps({"duration": duration, "codec": codec}), encoding="utf-8")
            self.paths.append(path)
            return path

    def close_segment(self, sequence=0, *, start=0., end=60.):
        path = Path(str(self.pattern) % sequence)
        with self.csv.open("a", encoding="utf-8", newline="") as output:
            output.write(f'"{path.name}",{start},{end}\n')

    def finish(self):
        with self.lock:
            if self.returncode is None:
                if self.final_seconds and not self.paths:
                    self.appear(duration=self.final_seconds)
                if self.final_seconds:
                    self.close_segment(end=self.final_seconds)
                self.returncode = 0

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        if self.returncode is None:
            raise subprocess.TimeoutExpired("fixture", timeout)
        return self.returncode

    def communicate(self, input=None, timeout=None):
        if input:
            self.stdin.write(input)
        self.wait(timeout)
        return b"", b""

    def terminate(self):
        self.terminated = True
        self.returncode = 1

    def kill(self):
        self.terminate()


class ProcessFactory:
    def __init__(self, engine, root):
        self.engine, self.root = engine, root
        self.processes = []
        self.failure = False
        self.free = 10**12

    def popen(self, args, **kwargs):
        if args[0] == "ffprobe-fixture":
            assert args[1:-1] == ["-v", "error", "-select_streams", "v:0", "-show_packets",
                                  "-show_entries", "packet=pts_time,duration_time", "-of", "compact=p=0:nk=0"]
            try:
                data = json.loads(Path(args[-1]).read_text(encoding="utf-8"))
                end = float(data.get("packet_end", data["duration"]))
                content = (f"pts_time=0.000000|duration_time=0.066000\n"
                           f"pts_time={end - .066:.6f}|duration_time=0.066000\n").encode()
                result = 0
            except (OSError, ValueError, KeyError):
                content, result = b"", 1
            # The fixture child is already finished; probes never count as recorder
            # children and do not acquire source URLs or create recording files.
            return SimpleNamespace(stdout=io.BytesIO(content), stderr=io.BytesIO(), stdin=None,
                                   returncode=result, pid=990000, poll=lambda: result,
                                   wait=lambda timeout=None: result, kill=lambda: None,
                                   terminate=lambda: None)
        if self.failure:
            raise OSError(f"launch failed rtsp://operator:{SECRET}@fixture.invalid")
        process = MediaProcess(args, self, **kwargs)
        self.processes.append(process)
        return process

    def probe(self, args, **kwargs):
        assert args[0] == "ffprobe-fixture", args
        path = Path(args[-1])
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return subprocess.CompletedProcess(args, 1, b"", b"invalid media")
        payload = {"streams": [{"codec_type": "video", "codec_name": data["codec"],
                                "width": 320, "height": 240, "duration": str(data["duration"])}],
                   "format": {"duration": str(data["duration"]), "size": str(path.stat().st_size)}}
        output = json.dumps(payload)
        return subprocess.CompletedProcess(args, 0, output if kwargs.get("text") else output.encode(), b"")


@contextmanager
def rig(*, start_manager=True, policy=None):
    service = feature("app.services.recording_service")
    guard = feature("app.media.process_guard").MediaProcessGuard
    api = feature("app.api.recordings")
    assert hasattr(service, "RecordingManager"), "RecordingManager public contract is missing"
    assert hasattr(guard, "require"), "Every media launch needs one fail-closed guard admission"
    with database() as (engine, directory), ExitStack() as stack:
        attendance_schema.initialize_attendance_schema(engine)
        media_root = directory / "recordings"
        clock = Clock()
        factory = ProcessFactory(engine, media_root)
        stack.enter_context(patch.object(guard, "initialize", return_value=True))
        stack.enter_context(patch.object(guard, "require", return_value=None))
        stack.enter_context(patch("subprocess.Popen", side_effect=factory.popen))
        stack.enter_context(patch("subprocess.run", side_effect=factory.probe))
        stack.enter_context(patch("shutil.disk_usage", side_effect=lambda _path: SimpleNamespace(
            total=2 * 10**12, used=10**12, free=factory.free)))
        for key, value in (policy or {}).items():
            assert hasattr(settings, key), f"Missing recording policy setting: {key}"
            stack.enter_context(patch.object(settings, key, value))
        with Session(engine) as db:
            source = camera_service.create_camera(db, camera_service.CameraCreate(
                name="Recording door", nvr_host="fixture.invalid", username="operator", password=SECRET))
        app = FastAPI()
        app.include_router(api.router)
        from app.api.cameras import router as cameras_router
        app.include_router(cameras_router)
        app.state.manager = SimpleNamespace(
            ffprobe="ffprobe-fixture", status=lambda _id: {"status": "OFFLINE"},
            start=lambda _source: None, sync=lambda _source, _changed: None, stop=lambda _deletion: None)

        def isolated_db():
            with Session(engine) as db:
                yield db

        app.dependency_overrides[get_db] = isolated_db
        managers = []

        def make_manager():
            manager = service.RecordingManager(engine, media_root, "ffmpeg-fixture", "ffprobe-fixture",
                                               now=clock.now, monotonic=clock.monotonic)
            managers.append(manager)
            app.state.recordings = manager
            return manager

        manager = make_manager() if start_manager else None
        value = SimpleNamespace(engine=engine, root=media_root, directory=directory, clock=clock,
                                factory=factory, manager=manager, make_manager=make_manager,
                                source=source, client=Client(app), app=app)
        try:
            yield value
        finally:
            for manager in reversed(managers):
                manager.shutdown()


def start(r, seconds=86400, request_id=None):
    return view(r.manager.start(r.source.id, request_id or uuid4(), seconds))


def current(r, session_id):
    return r.client.json("GET", f"/api/recordings/sessions/{session_id}")


def update(r, **changes):
    with Session(r.engine) as db:
        return camera_service.update_camera(db, db.get(Camera, r.source.id),
                                             camera_service.CameraUpdate(**changes))[0]


def ready_segment(r, *, duration=60, media_start=17., codec="h264"):
    session = start(r)
    process = eventually(lambda: r.factory.processes and r.factory.processes[-1])
    path = process.appear(duration=duration, codec=codec)
    eventually(lambda: any(row["session_id"] == session["id"]
                           for row in rows(r.engine, "recording_segments")),
               "Open segment start was not durable")
    r.clock.advance(duration)
    process.close_segment(start=media_start, end=media_start + duration)
    segment = eventually(lambda: next((row for row in rows(r.engine, "recording_segments")
                                      if row["state"] == "READY" and row["session_id"] == session["id"]), None))
    r.manager.stop(session["id"])
    return session, segment, path


@test
def test_policy_defaults_and_validation_are_available_before_consumers():
    config = Settings(_env_file=None)
    assert getattr(config, "recording_retention_days", None) == 14, "14-day policy is missing"
    assert config.recording_max_bytes == 150000000000
    assert config.recording_min_free_bytes == 2000000000
    assert Path(config.recording_root) == BASE_DIR / "data" / "recordings"
    for values in ({"recording_retention_days": 0}, {"recording_retention_days": 366},
                   {"recording_max_bytes": 9999999999}, {"recording_max_bytes": 100000000000001},
                   {"recording_min_free_bytes": -1}):
        try:
            Settings(_env_file=None, **values)
        except ValidationError:
            pass
        else:
            raise AssertionError(f"Unsafe storage policy accepted: {values}")


@test
def test_start_is_manual_strict_bounded_and_replay_is_shared_namespace():
    with rig() as r:
        assert not rows(r.engine, "recording_sessions") and not r.factory.processes
        for invalid in (0, 59, 86401, True, 60.5, "60", None):
            r.client.json("POST", "/api/recordings/sessions", {
                "camera_id": r.source.id, "request_id": str(uuid4()), "duration_seconds": invalid}, status=422)
        assert not rows(r.engine, "recording_sessions")
        for seconds in (60, 86400):
            request_id = str(uuid4())
            payload = {"camera_id": r.source.id, "request_id": request_id, "duration_seconds": seconds}
            first = r.client.json("POST", "/api/recordings/sessions", payload, status=201)
            view(first)
            assert first["id"] == request_id
            assert utc(first["deadline_at"]) - utc(first["requested_at"]) == timedelta(seconds=seconds)
            for key in ("requested_at", "deadline_at"):
                assert datetime.fromisoformat(first[key].replace("Z", "+00:00")).utcoffset() is not None
            replay = r.client.json("POST", "/api/recordings/sessions", payload)
            assert replay["id"] == first["id"]
            r.client.json("POST", "/api/recordings/sessions", {**payload, "duration_seconds": 61}, status=409)
            r.client.json("POST", "/api/recordings/sessions", {**payload, "request_id": str(uuid4())}, status=409)
            r.client.json("POST", f'/api/recordings/sessions/{first["id"]}/stop')
        assert len(rows(r.engine, "recording_sessions")) == 2
        assert len(r.factory.processes) == 2
        with Session(r.engine) as db:
            collision = str(uuid4())
            db.add(OperatorRequest(id=collision, operation="camera.batch", payload_sha256="b" * 64,
                                   result={"cameras": []}))
            db.commit()
        rejects(409, lambda: start(r, request_id=collision))
        assert len(r.factory.processes) == 2
        assert SECRET not in str(rows(r.engine, "operator_requests"))


@test
def test_default_duration_missing_disabled_security_and_pagination():
    with rig() as r:
        payload = {"camera_id": r.source.id, "request_id": str(uuid4())}
        r.client.json("POST", "/api/recordings/sessions", payload, operator=False, status=403)
        r.client.json("POST", "/api/recordings/sessions", payload,
                      headers={"origin": "http://attacker.invalid"}, status=403)
        r.client.json("POST", "/api/recordings/sessions", {**payload, "camera_id": 99999}, status=404)
        update(r, enabled=False)
        r.client.json("POST", "/api/recordings/sessions", payload, status=409)
        update(r, enabled=True)
        saved = r.client.json("POST", "/api/recordings/sessions", payload, status=201)
        assert utc(saved["deadline_at"]) == BASE + timedelta(seconds=86400)
        r.client.json("GET", "/api/recordings/sessions?limit=201", status=422)
        r.client.json("GET", "/api/recordings/sessions?offset=-1", status=422)
        assert r.client.json("GET", "/api/recordings/sessions?camera_id=99999") == []
        assert len(r.client.json("GET", f"/api/recordings/sessions?camera_id={r.source.id}&limit=1")) == 1
        assert r.client.json("GET", "/api/recordings/sessions?offset=1") == []
        r.client.json("GET", f"/api/recordings/sessions/{uuid4()}", status=404)


@test
def test_launch_failure_is_committed_failed_safe_and_never_relaunches_uuid():
    with rig() as r:
        r.factory.failure = True
        request_id = str(uuid4())
        rejects(503, lambda: start(r, request_id=request_id))
        saved = current(r, request_id)
        assert saved["status"] == "FAILED" and saved["reason_code"]
        assert saved["ended_at"] is not None
        assert SECRET not in json.dumps(saved)
        r.factory.failure = False
        replay = view(r.manager.start(r.source.id, request_id, 86400))
        assert replay["status"] == "FAILED" and replay["id"] == request_id
        assert not r.factory.processes
        assert len(rows(r.engine, "operator_requests")) == 1


@test
def test_concurrent_starts_admit_one_writer_and_idempotent_stop():
    with rig() as r:
        def attempt():
            try:
                return start(r)
            except HTTPException as exc:
                return exc.status_code
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(lambda _: attempt(), range(2)))
        assert sum(isinstance(result, dict) for result in results) == 1
        assert 409 in results
        saved = next(result for result in results if isinstance(result, dict))
        first = view(r.manager.stop(saved["id"]))
        assert first["status"] == "STOPPED"
        assert view(r.manager.stop(saved["id"])) == first
        assert len(r.factory.processes) == 1


@test
def test_deadline_and_utc_discontinuity_stop_without_relabelling():
    for seconds, utc_seconds, expected in ((60, 60, "COMPLETED"), (59, 61, "COMPLETED"),
                                           (3, 0, "INTERRUPTED"), (3, 6, "INTERRUPTED")):
        with rig() as r:
            saved = start(r, seconds=60)
            r.clock.advance(seconds, utc_seconds=utc_seconds)
            ended = eventually(lambda: (value if (value := current(r, saved["id"]))["status"] not in ACTIVE else None))
            assert ended["status"] == expected, ended
            assert utc(ended["deadline_at"]) == BASE + timedelta(seconds=60)
            assert r.factory.processes[0].poll() is not None


@test
def test_stop_at_59_seconds_finalizes_partial_video_without_attendance_writes():
    with rig() as r:
        before = catalog(r.engine, ATTENDANCE_TABLES)
        saved = start(r)
        process = r.factory.processes[0]
        process.appear(duration=59)
        process.final_seconds = 59
        eventually(lambda: rows(r.engine, "recording_segments"))
        r.clock.advance(59)
        stopped = view(r.manager.stop(saved["id"]))
        assert stopped["status"] == "STOPPED"
        segment = rows(r.engine, "recording_segments")[0]
        assert segment["state"] == "READY" and segment["duration_ms"] == 59000
        assert utc(segment["started_at"]) == BASE
        assert utc(segment["ended_at"]) == BASE + timedelta(seconds=59)
        assert stopped["bytes_recorded"] > 0
        assert utc(stopped["available_to"]) == BASE + timedelta(seconds=59)
        assert not process.terminated, "Normal stop must send q before terminate"
        assert catalog(r.engine, ATTENDANCE_TABLES) == before
        for table in ("attendance_events", "attendance_decisions", "work_sessions", "attendance_coverage_gaps"):
            assert rows(r.engine, table) == []


@test
def test_closed_csv_uses_durable_anchor_and_first_offset_not_mtime():
    with rig() as r:
        saved = start(r)
        process = r.factory.processes[0]
        r.clock.advance(4)
        path = process.appear(duration=60, codec="hevc")
        os.utime(path, (1, 1))
        opened = eventually(lambda: rows(r.engine, "recording_segments"))
        assert all(row["state"] != "READY" for row in opened), "Never publish an open file"
        connection = rows(r.engine, "recording_connections")[0]
        assert utc(connection["anchor_utc"]) == BASE + timedelta(seconds=4)
        r.clock.advance(60)
        process.close_segment(start=23.5, end=83.5)
        segment = eventually(lambda: next((row for row in rows(r.engine, "recording_segments")
                                          if row["state"] == "READY"), None))
        assert utc(segment["started_at"]) == BASE + timedelta(seconds=4)
        assert utc(segment["ended_at"]) == BASE + timedelta(seconds=64)
        assert segment["codec"] == "hevc" and segment["duration_ms"] == 60000
        assert rows(r.engine, "recording_connections")[0]["media_origin_seconds"] == 23.5
        assert segment["connection_id"] == connection["id"]
        assert current(r, saved["id"])["available_from"] is not None


@test
def test_reconnect_has_new_anchor_and_gap_without_extending_deadline():
    with rig(policy={"frame_stall_timeout": 2}) as r:
        saved = start(r, seconds=60)
        first = r.factory.processes[0]
        first.appear(duration=1)
        eventually(lambda: rows(r.engine, "recording_segments"))
        r.clock.advance(3)
        eventually(lambda: current(r, saved["id"])["status"] == "RECONNECTING")
        r.clock.advance(2)
        second = eventually(lambda: r.factory.processes[-1] if len(r.factory.processes) > 1 else None)
        second.appear(duration=1)
        eventually(lambda: len(rows(r.engine, "recording_segments")) == 2)
        connections = rows(r.engine, "recording_connections")
        assert len(connections) == 2 and connections[0]["id"] != connections[1]["id"]
        assert utc(connections[0]["anchor_utc"]) == BASE
        assert utc(connections[1]["anchor_utc"]) == BASE + timedelta(seconds=5)
        assert rows(r.engine, "recording_gaps"), "A reconnect is not continuous coverage"
        assert utc(current(r, saved["id"])["deadline_at"]) == BASE + timedelta(seconds=60)
        r.clock.advance(55)
        eventually(lambda: current(r, saved["id"])["status"] == "COMPLETED")


@test
def test_archive_only_edits_preserve_recording_but_live_change_stops():
    with rig() as r:
        saved = start(r)
        archive = update(r, archive={"http_scheme": "https", "http_port": 443, "utc_offset_minutes": 300},
                         expected_archive_revision=None)
        r.manager.source_changed(archive)
        assert current(r, saved["id"])["status"] in ACTIVE
        changed = update(r, password="new-secret")
        r.manager.source_changed(changed)
        ended = eventually(lambda: (value if (value := current(r, saved["id"]))["status"] not in ACTIVE else None))
        assert ended["reason_code"] == "SOURCE_CHANGED"
        assert len(r.factory.processes) == 1, "Source edits must not resume recording"


@test
def test_delayed_disable_and_old_source_cannot_stop_new_explicit_session():
    with rig() as r:
        old_session = start(r)
        disabled = update(r, enabled=False)
        r.manager.source_changed(disabled)
        eventually(lambda: current(r, old_session["id"])["status"] not in ACTIVE)
        assert current(r, old_session["id"])["reason_code"] == "CAMERA_DISABLED"
        enabled = update(r, enabled=True)
        assert enabled.source_revision == disabled.source_revision
        # Deliberately do NOT publish enable before admission. DB truth must win.
        new_session = start(r)
        r.manager.source_changed(disabled)
        assert current(r, new_session["id"])["status"] in ACTIVE
        changed = update(r, channel=2)
        r.manager.source_changed(changed)
        eventually(lambda: current(r, new_session["id"])["status"] not in ACTIVE)
        newest = start(r)
        r.manager.source_changed(enabled)
        assert current(r, newest["id"])["status"] in ACTIVE


@test
def test_reordered_reenable_does_not_erase_disable_for_predating_session():
    with rig() as r:
        with r.engine.begin() as connection:
            connection.exec_driver_sql("UPDATE cameras SET updated_at=? WHERE id=?", (BASE, r.source.id))
        old_session = start(r)

        def same_time_update(enabled):
            snapshot = update(r, enabled=enabled)
            with r.engine.begin() as connection:
                connection.exec_driver_sql("UPDATE cameras SET updated_at=? WHERE id=?", (BASE, r.source.id))
            return replace(snapshot, updated_at=BASE)

        disabled = same_time_update(False)
        enabled = same_time_update(True)
        assert disabled.source_revision == enabled.source_revision == old_session["source_revision"]
        assert disabled.updated_at == enabled.updated_at == utc(old_session["requested_at"])
        assert disabled.publication_sequence < enabled.publication_sequence
        # A newer enabled notification is delivered first. It must not erase
        # the disable barrier for the session that existed before that commit.
        r.manager.source_changed(enabled)
        r.manager.source_changed(disabled)
        ended = eventually(lambda: (value if (value := current(r, old_session["id"]))["status"] not in ACTIVE else None))
        assert ended["reason_code"] == "CAMERA_DISABLED"
        assert len(r.factory.processes) == 1, "Re-enable cannot restart the old recording"
        # Same revision and clock instant, different admission order: this new
        # explicit request must survive an identical delayed disable delivery.
        new_session = start(r)
        assert new_session["source_revision"] == old_session["source_revision"]
        assert utc(new_session["requested_at"]) == utc(old_session["requested_at"])
        r.manager.source_changed(disabled)
        r.manager.source_changed(enabled)
        assert current(r, new_session["id"])["status"] in ACTIVE
        assert len(r.factory.processes) == 2


@test
def test_camera_history_prevents_delete_after_recording_stops():
    with rig() as r:
        saved = start(r)
        r.manager.stop(saved["id"])
        with Session(r.engine) as db:
            rejects(409, lambda: camera_service.delete_camera(db, db.get(Camera, r.source.id)))
        assert rows(r.engine, "recording_sessions")[0]["id"] == saved["id"]


@test
def test_root_lock_rejects_second_manager_before_recovery_mutates_status():
    with rig() as r:
        saved = start(r)
        before = rows(r.engine, "recording_sessions")
        rejects(503, r.make_manager)
        assert rows(r.engine, "recording_sessions") == before
        assert current(r, saved["id"])["status"] in ACTIVE
        assert r.factory.processes[0].poll() is None


@test
def test_disk_reserve_stops_writes_and_reports_safe_storage_full():
    with rig() as r:
        saved = start(r)
        r.factory.free = 1999999999
        r.clock.advance(1)
        ended = eventually(lambda: (value if (value := current(r, saved["id"]))["status"] not in ACTIVE else None))
        assert ended["reason_code"] == "STORAGE_FULL"
        assert r.factory.processes[0].poll() is not None
        rejects(503, lambda: start(r), "STORAGE_FULL")
        assert SECRET not in json.dumps(r.client.json("GET", "/api/recordings/status"))


@test
def test_expiration_respects_lease_then_keeps_expired_metadata():
    with rig() as r:
        saved, segment, path = ready_segment(r)
        storage = r.manager.storage
        with storage.lease(segment["id"]):
            r.clock.advance(14 * 86400 + 1)
            # Public enforcement is also used by recorder/playback admission.
            storage.enforce()
            assert path.exists()
            assert rows(r.engine, "recording_segments")[0]["state"] == "READY"
        storage.enforce()
        assert not path.exists()
        retained = rows(r.engine, "recording_segments")[0]
        assert retained["state"] == "EXPIRED" and retained["id"] == segment["id"]
        assert retained["started_at"] == segment["started_at"]
        assert current(r, saved["id"])["available_from"] is None
        rejects(409, lambda: storage.lease(segment["id"]).__enter__())


@test
def test_last_segment_expiring_before_connection_closes_reclaims_owned_index():
    with rig(policy={"recording_min_free_bytes": 0}) as r:
        saved = start(r)
        process = eventually(lambda: r.factory.processes and r.factory.processes[-1])
        path = process.appear(duration=60)
        eventually(lambda: rows(r.engine, "recording_segments"))
        process.close_segment(start=17, end=77)
        segment = eventually(lambda: next((row for row in rows(r.engine, "recording_segments")
                                          if row["state"] == "READY"), None))
        csv_path = path.parent / "closed.csv"
        assert csv_path.exists()
        assert rows(r.engine, "recording_connections")[0]["closed_at"] is None
        storage = r.manager.storage
        with storage.lock:
            storage.max_bytes = 256 * 1024 * 1024 + csv_path.stat().st_size + 1
            storage.enforce()
            assert not path.exists()
            assert rows(r.engine, "recording_segments")[0]["state"] == "EXPIRED"
            assert csv_path.exists(), "An active connection still owns its CSV writer"
        r.manager.stop(saved["id"])
        assert not csv_path.exists(), "Finalization must reclaim an already-expired connection index"
        assert rows(r.engine, "recording_segments")[0]["id"] == segment["id"]


@test
def test_retention_marks_deleting_before_unlink_and_restart_completes_it():
    with rig() as r:
        _saved, segment, path = ready_segment(r)
        r.clock.advance(14 * 86400 + 1)
        original = Path.unlink
        observed = []

        def fail_owned_delete(candidate, *args, **kwargs):
            if candidate == path:
                observed.append(rows(r.engine, "recording_segments")[0]["state"])
                raise PermissionError("fixture busy")
            return original(candidate, *args, **kwargs)

        with patch.object(Path, "unlink", fail_owned_delete):
            r.manager.storage.enforce()
        assert observed and set(observed) == {"DELETING"}
        assert path.exists()
        r.manager.shutdown()
        replacement = r.make_manager()
        eventually(lambda: not path.exists())
        assert rows(r.engine, "recording_segments")[0]["state"] == "EXPIRED"
        assert not replacement.snapshot()["sessions"] or not any(
            item["status"] in ACTIVE for item in replacement.snapshot()["sessions"])


@test
def test_unrecognized_and_symlink_files_never_get_deleted_by_retention():
    with rig() as r:
        _saved, segment, path = ready_segment(r)
        external = r.directory / "attendance-evidence.jpg"
        external.write_bytes(b"protected evidence")
        unrelated = r.root / "operator-note.txt"
        unrelated.write_bytes(b"not an owned media file")
        path.unlink()
        try:
            path.symlink_to(external)
        except OSError:
            # Symlink creation requires privilege on some Windows deployments;
            # the independent traversal test below still always executes.
            print("SKIP symlink branch: Windows symlink privilege unavailable")
        r.clock.advance(14 * 86400 + 1)
        r.manager.storage.enforce()
        assert external.read_bytes() == b"protected evidence"
        assert unrelated.read_bytes() == b"not an owned media file"
        with r.engine.begin() as connection:
            connection.exec_driver_sql("UPDATE recording_segments SET storage_key='../attendance-evidence.jpg', "
                                       "state='DELETING' WHERE id=?", (segment["id"],))
        r.manager.storage.enforce()
        assert external.read_bytes() == b"protected evidence"


@test
def test_quota_counts_active_and_unrecognized_files_without_deleting_external_content():
    with rig(policy={"recording_max_bytes": 268435456 + 4096, "recording_min_free_bytes": 0}) as r:
        saved = start(r)
        unrelated = r.root / "unrecognized.bin"
        unrelated.write_bytes(b"x" * 8192)
        r.clock.advance(1)
        ended = eventually(lambda: (value if (value := current(r, saved["id"]))["status"] not in ACTIVE else None))
        assert ended["reason_code"] == "STORAGE_FULL"
        assert unrelated.stat().st_size == 8192


@test
def test_guard_failure_blocks_recording_and_existing_connection_probe():
    with rig() as r:
        guard = feature("app.media.process_guard").MediaProcessGuard
        error = HTTPException(503, "MEDIA_PROCESS_CONTAINMENT_UNAVAILABLE")
        with patch.object(guard, "require", side_effect=error):
            rejects(503, lambda: start(r), "MEDIA_PROCESS_CONTAINMENT_UNAVAILABLE")
            result = camera_service.test_rtsp_connection(nvr_host="fixture.invalid", rtsp_port=554,
                username="operator", password=SECRET, channel=1, stream_type="main", ffprobe_path="ffprobe-fixture")
            assert not result.success
            assert "CONTAINMENT" in str(result.error)
            r.client.json("POST", f"/api/cameras/{r.source.id}/test", status=503)
        assert not r.factory.processes


@test
def test_restart_recovers_durable_start_but_never_dates_unknown_orphan_from_mtime():
    with rig() as r:
        saved = start(r)
        process = r.factory.processes[0]
        path = process.appear(duration=59)
        durable = eventually(lambda: rows(r.engine, "recording_segments"))[0]
        assert utc(durable["started_at"]) == BASE
        r.manager.shutdown()
        # Persist the exact observable crash boundary: start durable, CSV not
        # closed, no media origin, app dead. No production crash/test method.
        process.csv.unlink(missing_ok=True)
        orphan = Path(str(process.pattern) % 1)
        orphan.write_text(json.dumps({"duration": 12, "codec": "h264"}), encoding="utf-8")
        os.utime(orphan, (1, 1))
        with r.engine.begin() as connection:
            connection.exec_driver_sql(
                "UPDATE recording_sessions SET status='RECONNECTING',ended_at=NULL WHERE id=?", (saved["id"],))
            connection.exec_driver_sql(
                "UPDATE recording_connections SET anchor_utc=NULL,media_origin_seconds=NULL,closed_at=NULL")
            connection.exec_driver_sql(
                "UPDATE recording_segments SET state='UNINDEXED',ended_at=NULL,duration_ms=0 WHERE id=?",
                (durable["id"],))
        replacement = r.make_manager()
        recovered = eventually(lambda: rows(r.engine, "recording_segments")
                               if len(rows(r.engine, "recording_segments")) == 2 else None)
        known = next(row for row in recovered if row["id"] == durable["id"])
        unknown = next(row for row in recovered if row["id"] != durable["id"])
        assert known["state"] == "READY" and known["duration_ms"] == 59000
        assert utc(known["started_at"]) == BASE
        assert utc(known["ended_at"]) == BASE + timedelta(seconds=59)
        assert unknown["state"] == "UNINDEXED"
        assert unknown["started_at"] is None and unknown["ended_at"] is None
        assert unknown["bytes"] == orphan.stat().st_size
        assert current(r, saved["id"])["status"] == "INTERRUPTED"
        assert len(r.factory.processes) == 1, "Restart must never resume a recording"
        replacement.storage.enforce()
        assert orphan.exists(), "1970 mtime is not the orphan's age; connection.created_at is"


@test
def test_restart_marks_missing_and_corrupt_media_explicitly():
    for missing in (True, False):
        with rig() as r:
            saved, segment, path = ready_segment(r)
            r.manager.shutdown()
            if missing:
                path.unlink()
            else:
                path.write_bytes(b"truncated not video")
            r.make_manager()
            state = rows(r.engine, "recording_segments")[0]
            assert state["state"] == ("MISSING" if missing else "CORRUPT")
            assert state["id"] == segment["id"]
            assert current(r, saved["id"])["available_from"] is None
            assert len(r.factory.processes) == 1


@test
def test_schema_rejects_second_active_session_invalid_ready_and_cascading_history_loss():
    with rig() as r:
        saved = start(r)
        original = rows(r.engine, "recording_sessions")[0]
        for status in ("STARTING", "RECORDING", "RECONNECTING", "STOPPING"):
            duplicate = {**original, "id": str(uuid4()), "status": status}
            names = ",".join(duplicate)
            placeholders = ",".join("?" for _ in duplicate)
            try:
                with r.engine.begin() as connection:
                    connection.exec_driver_sql(
                        f"INSERT INTO recording_sessions ({names}) VALUES ({placeholders})",
                        tuple(duplicate.values()))
            except IntegrityError:
                pass
            else:
                raise AssertionError(f"Partial active-camera uniqueness omitted {status}")
        r.manager.stop(saved["id"])
        _second, segment, _path = ready_segment(r)
        for statement, parameters in (
            ("UPDATE recording_segments SET started_at=NULL WHERE id=?", (segment["id"],)),
            ("UPDATE recording_segments SET duration_ms=0 WHERE id=?", (segment["id"],)),
            ("DELETE FROM recording_sessions WHERE id=?", (segment["session_id"],)),
            ("DELETE FROM recording_connections WHERE id=?", (segment["connection_id"],)),
            ("DELETE FROM cameras WHERE id=?", (r.source.id,)),
        ):
            try:
                with r.engine.begin() as connection:
                    connection.exec_driver_sql(statement, parameters)
            except IntegrityError:
                pass
            else:
                raise AssertionError(f"Media history constraint did not reject: {statement}")
        assert rows(r.engine, "recording_segments")[0]["state"] == "READY"


@test
def test_stale_ready_selection_cannot_acquire_lease_after_deleting_commit():
    with rig() as r:
        _saved, segment, path = ready_segment(r)
        r.clock.advance(14 * 86400 + 1)
        entered, release = threading.Event(), threading.Event()
        original = Path.unlink

        def hold_unlink(candidate, *args, **kwargs):
            if candidate == path:
                assert rows(r.engine, "recording_segments")[0]["state"] == "DELETING"
                entered.set()
                assert release.wait(5), "Lease/retention lock order deadlocked"
            return original(candidate, *args, **kwargs)

        with patch.object(Path, "unlink", hold_unlink), ThreadPoolExecutor(max_workers=2) as pool:
            eviction = pool.submit(r.manager.storage.enforce)
            assert entered.wait(5)

            def acquire():
                try:
                    with r.manager.storage.lease(segment["id"]):
                        return "leased-deleting-file"
                except HTTPException as exc:
                    return exc.status_code

            acquisition = pool.submit(acquire)
            release.set()
            eviction.result(timeout=5)
            assert acquisition.result(timeout=5) == 409
        assert not path.exists()


@test
def test_quota_evicts_oldest_unleased_before_newer_and_stops_when_all_are_leased():
    with rig(policy={"recording_min_free_bytes": 0}) as r:
        _one, oldest, first_path = ready_segment(r)
        r.clock.advance(5)
        _two, newer, second_path = ready_segment(r)
        r.manager.shutdown()
        # Budget can retain one small segment plus the mandated 256 MiB reserve.
        total = sum(path.stat().st_size for path in r.root.rglob("*") if path.is_file())
        with patch.object(settings, "recording_max_bytes", 268435456 + total - first_path.stat().st_size):
            r.manager = r.make_manager()
            r.manager.storage.enforce()
        assert not first_path.exists() and second_path.exists()
        states = {row["id"]: row["state"] for row in rows(r.engine, "recording_segments")}
        assert states[oldest["id"]] == "EXPIRED" and states[newer["id"]] == "READY"
        with r.manager.storage.lease(newer["id"]):
            (r.root / "external.bin").write_bytes(b"x" * 8192)
            rejects(503, lambda: start(r), "STORAGE_FULL")
            assert second_path.exists()


@test
def test_actual_camera_edit_api_publishes_local_recording_stop_after_commit():
    with rig() as r:
        saved = start(r)
        camera = r.client.json("PUT", f"/api/cameras/{r.source.id}", {
            "archive": {"http_scheme": "https", "http_port": 443, "utc_offset_minutes": 180},
            "expected_archive_revision": None})
        assert camera["recording"]["id"] == saved["id"]
        assert current(r, saved["id"])["status"] in ACTIVE
        r.client.json("PUT", f"/api/cameras/{r.source.id}", {"enabled": False})
        ended = eventually(lambda: (value if (value := current(r, saved["id"]))["status"] not in ACTIVE else None))
        assert ended["reason_code"] == "CAMERA_DISABLED"
        r.client.json("PUT", f"/api/cameras/{r.source.id}", {"enabled": True})
        assert len(r.factory.processes) == 1, "Re-enable API must not resume recording"


@test
def test_source_edit_between_session_commit_and_launch_refuses_stale_credentials():
    with rig() as r:
        entered, release = threading.Event(), threading.Event()
        original = r.manager._launch
        request_id = str(uuid4())

        def delayed_launch(job):
            entered.set()
            assert release.wait(5)
            return original(job)

        with patch.object(r.manager, "_launch", side_effect=delayed_launch), ThreadPoolExecutor(max_workers=1) as pool:
            pending = pool.submit(lambda: rejects(409, lambda: start(r, request_id=request_id), "SOURCE_CHANGED"))
            assert entered.wait(5)
            assert rows(r.engine, "operator_requests")[0]["id"] == request_id
            update(r, password="replacement-secret")
            release.set()
            pending.result(timeout=8)
        assert current(r, request_id)["status"] == "FAILED"
        assert not r.factory.processes
        replay = view(r.manager.start(r.source.id, request_id, 86400))
        assert replay["status"] == "FAILED"


@test
def test_incumbent_reader_probe_and_frame_child_both_require_containment():
    from app.rtsp.reader import RTSPReader
    with rig() as r:
        guard = feature("app.media.process_guard").MediaProcessGuard
        reader = RTSPReader(r.source.id, "fixture", "rtsp://fixture.invalid", source_revision=1,
                            ffmpeg_path="ffmpeg-fixture", ffprobe_path="ffprobe-fixture")
        with patch.object(guard, "require", side_effect=HTTPException(503, "MEDIA_PROCESS_CONTAINMENT_UNAVAILABLE")):
            rejects(503, reader._probe, "MEDIA_PROCESS_CONTAINMENT_UNAVAILABLE")
            with patch.object(reader, "_probe", return_value=(2, 2, "h264", 1.)):
                rejects(503, reader._connect_and_read, "MEDIA_PROCESS_CONTAINMENT_UNAVAILABLE")
        assert not r.factory.processes


@test
def test_unused_cache_is_evicted_before_ready_original_and_leased_cache_is_retained():
    with rig(policy={"recording_min_free_bytes": 0}) as r:
        _saved, segment, original = ready_segment(r)
        r.manager.shutdown()
        total = sum(path.stat().st_size for path in r.root.rglob("*") if path.is_file())
        with patch.object(settings, "recording_max_bytes", 268435456 + total + 256):
            r.manager = r.make_manager()
        rejects(409, lambda: r.manager.storage.register_cache(
            segment["storage_key"], expires_at=r.clock.now() + timedelta(seconds=120)))
        cache = r.root / "cache" / str(uuid4())
        cache.mkdir(parents=True)
        output = cache / "0.ts"
        output.write_bytes(b"x" * 1024)
        key = output.relative_to(r.root).as_posix()
        r.manager.storage.register_cache(key, expires_at=r.clock.now() + timedelta(seconds=120))
        with r.manager.storage.cache_lease(key):
            with r.manager.storage.lease(segment["id"]):
                assert not r.manager.storage.enforce()
                assert output.exists() and original.exists()
        assert r.manager.storage.enforce()
        assert not output.exists() and original.exists()
        assert rows(r.engine, "recording_segments")[0]["state"] == "READY"


@test
def test_graceful_stop_escalates_bounded_terminate_kill_and_keeps_final_partial():
    with rig() as r:
        saved = start(r)
        process = r.factory.processes[0]
        process.appear(duration=59)
        eventually(lambda: rows(r.engine, "recording_segments"))
        r.clock.advance(59)
        actions = []

        def ignore_quit(data):
            actions.append("q")
            return len(data)

        def ignore_terminate():
            actions.append("terminate")

        def killed():
            actions.append("kill")
            process.returncode = 1

        with patch.object(process.stdin, "write", side_effect=ignore_quit), \
                patch.object(process, "terminate", side_effect=ignore_terminate), \
                patch.object(process, "kill", side_effect=killed):
            result = view(r.manager.stop(saved["id"]))
        assert actions == ["q", "terminate", "kill"]
        assert result["status"] == "STOPPED" and process.poll() is not None
        segment = rows(r.engine, "recording_segments")[0]
        assert segment["state"] == "READY" and segment["duration_ms"] == 59000


def owned_parent(root, ffmpeg):
    """Only invoked by the explicit integration test, in a throwaway process."""
    guard = feature("app.media.process_guard").MediaProcessGuard
    guard.initialize()
    guard.require()
    output = Path(root) / "owned.ts"
    child = subprocess.Popen([ffmpeg, "-hide_banner", "-loglevel", "error", "-re", "-f", "lavfi", "-i",
                              "testsrc2=size=64x64:rate=10", "-an", "-c:v", "mpeg2video", "-f", "mpegts",
                              "-flush_packets", "1", str(output)], stdin=subprocess.DEVNULL,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, close_fds=True)
    (Path(root) / "child.json").write_text(json.dumps({"pid": child.pid}), encoding="utf-8")
    while True:
        time.sleep(1)


def test_windows_owned_parent_death_kills_actual_ffmpeg_and_stops_file_growth():
    assert os.name == "nt", "This integration contract targets Windows 10/11 only"
    import ctypes
    import tempfile
    from ctypes import wintypes
    from app.config import resolve_ffmpeg
    ffmpeg, _probe = resolve_ffmpeg()
    assert ffmpeg, "Installed FFmpeg is required for parent-death integration"
    feature("app.media.process_guard")
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel.WaitForSingleObject.restype = wintypes.DWORD
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.restype = wintypes.BOOL
    kernel.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
    kernel.TerminateProcess.restype = wintypes.BOOL
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        parent = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--owned-parent", directory, ffmpeg],
                                  stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        handle = None
        try:
            marker = root / "child.json"
            eventually(marker.exists, "Owned parent failed containment/launch", timeout=15)
            pid = json.loads(marker.read_text(encoding="utf-8"))["pid"]
            # Open only our parent's explicitly reported child; never enumerate/kill by PID.
            handle = kernel.OpenProcess(0x00100001, False, pid)  # SYNCHRONIZE + owned-child failure cleanup
            assert handle, "Cannot observe owned FFmpeg child lifetime"
            output = root / "owned.ts"
            eventually(lambda: output.exists() and output.stat().st_size > 0, timeout=15)
            before = output.stat().st_size
            eventually(lambda: output.stat().st_size > before, "Fixture must genuinely write before parent death")
            parent.kill()  # Only this Popen-owned throwaway parent is terminated.
            parent.wait(timeout=5)
            assert kernel.WaitForSingleObject(handle, 5000) == 0, "FFmpeg outlived its containing parent"
            final = output.stat().st_size
            time.sleep(.5)
            assert output.stat().st_size == final, "Orphaned media output continued growing"
        finally:
            if parent.poll() is None:
                parent.kill()
                parent.wait(timeout=5)
            if handle:
                if kernel.WaitForSingleObject(handle, 0) != 0:
                    # Only on failed containment proof: don't leave the known owned
                    # fixture child behind. This does not turn the failed assertion green.
                    kernel.TerminateProcess(handle, 1)
                    kernel.WaitForSingleObject(handle, 5000)
                kernel.CloseHandle(handle)
            if parent.stderr:
                parent.stderr.close()


def owned_lock_contender(url, root, result_path):
    from sqlalchemy import create_engine
    engine = create_engine(url)
    try:
        try:
            manager = feature("app.services.recording_service").RecordingManager(engine, root, None, None)
        except HTTPException as exc:
            result = {"status": exc.status_code}
        else:
            manager.shutdown()
            result = {"status": 200}
        Path(result_path).write_text(json.dumps(result), encoding="utf-8")
    finally:
        engine.dispose()


def test_windows_root_owner_excludes_other_process_before_recovery():
    assert os.name == "nt"
    from app.database.models import RecordingSession
    service = feature("app.services.recording_service")
    guard = feature("app.media.process_guard").MediaProcessGuard
    with database() as (engine, root):
        attendance_schema.initialize_attendance_schema(engine)
        with Session(engine) as db:
            source = camera_service.create_camera(db, camera_service.CameraCreate(
                name="Lock fixture", nvr_host="unused.invalid", username="u", password="p"))
        with patch.object(guard, "require"):
            manager = service.RecordingManager(engine, root / "recordings", None, None)
        try:
            identifier = str(uuid4())
            with Session(engine) as db:
                db.add(RecordingSession(id=identifier, camera_id=source.id, camera_name=source.name,
                    source_revision=source.source_revision, requested_at=BASE,
                    deadline_at=BASE + timedelta(seconds=60), status="RECORDING"))
                db.commit()
            result_path = root / "contender.json"
            child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--owned-lock-contender",
                                      str(engine.url), str(root / "recordings"), str(result_path)],
                                     stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            try:
                child.wait(timeout=15)
                assert child.returncode == 0
                assert json.loads(result_path.read_text(encoding="utf-8")) == {"status": 503}
                assert rows(engine, "recording_sessions")[0]["status"] == "RECORDING"
            finally:
                if child.poll() is None:
                    child.kill()
                    child.wait(timeout=5)
                child.stderr.close()
        finally:
            manager.shutdown()


def test_windows_local_seek_between_keyframes_preserves_first_scene_and_duration():
    """A 1ms seek must not skip the red first GOP or advertise false partial footage."""
    assert os.name == "nt", "This real-media integration targets Windows 10/11 only"
    from app.config import resolve_ffmpeg
    from app.database.models import RecordingConnection, RecordingSegment, RecordingSession
    from app.media.playback import PlaybackManager
    from app.media.process_guard import MediaProcessGuard
    from app.services.recording_service import RecordingManager

    ffmpeg, ffprobe = resolve_ffmpeg()
    assert ffmpeg and ffprobe, "Installed FFmpeg/ffprobe with libx264 are required"
    MediaProcessGuard.require()  # Real containment, including fixture/probe children.
    with database() as (engine, directory):
        attendance_schema.initialize_attendance_schema(engine)
        with Session(engine) as db:
            source = camera_service.create_camera(db, camera_service.CameraCreate(
                name="Accurate local seek", nvr_host="unused.invalid", username="u", password="p"))
        recordings = RecordingManager(engine, directory / "recordings", ffmpeg, ffprobe)
        playback = None
        try:
            session_id, connection_id = str(uuid4()), str(uuid4())
            relative = f"{session_id}/{connection_id}"
            original_directory = recordings.storage.root / relative
            original_directory.mkdir(parents=True)
            original = original_directory / "000000.mkv"
            generated = subprocess.run([
                ffmpeg, "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
                "color=c=red:size=322x242:rate=15:duration=6",
                "-vf", "format=yuv444p,crop=321:241,drawbox=c=blue:t=fill:enable='gte(t,2)'",
                "-an", "-c:v", "libx264", "-pix_fmt", "yuv444p", "-preset", "veryfast",
                "-threads", "2", "-g", "30", "-keyint_min", "30", "-sc_threshold", "0",
                str(original),
            ], stdin=subprocess.DEVNULL, capture_output=True, timeout=20, close_fds=True)
            assert generated.returncode == 0, generated.stderr.decode("utf-8", "replace")
            probed = subprocess.run([
                ffprobe, "-v", "error", "-show_streams", "-show_format", "-of", "json", str(original),
            ], stdin=subprocess.DEVNULL, capture_output=True, timeout=10, close_fds=True)
            assert probed.returncode == 0, probed.stderr.decode("utf-8", "replace")
            original_info = json.loads(probed.stdout)
            original_video = original_info["streams"][0]
            assert (original_video["codec_name"], original_video["width"], original_video["height"],
                    original_video["pix_fmt"]) == ("h264", 321, 241, "yuv444p"), original_info
            assert abs(float(original_info["format"]["duration"]) - 6) < .01, original_info

            # Keep fixtures recent for real retention clocks, with an exact millisecond
            # origin so the requested point is independently known to be 1ms inside.
            start = datetime.now(timezone.utc).replace(tzinfo=None, microsecond=0) - timedelta(seconds=10)
            end = start + timedelta(seconds=6)
            with Session(engine) as db:
                db.add(RecordingSession(id=session_id, camera_id=source.id, camera_name=source.name,
                    source_revision=source.source_revision, requested_at=start, started_at=start,
                    deadline_at=start + timedelta(seconds=60), ended_at=end, status="STOPPED"))
                db.flush()
                db.add(RecordingConnection(id=connection_id, session_id=session_id, directory=relative,
                    source_revision=source.source_revision, created_at=start, anchor_utc=start,
                    media_origin_seconds=0, closed_at=end))
                db.flush()
                db.add(RecordingSegment(id=str(uuid4()), session_id=session_id, connection_id=connection_id,
                    camera_id=source.id, source_revision=source.source_revision,
                    storage_key=f"{relative}/000000.mkv", started_at=start, ended_at=end,
                    duration_ms=6000, bytes=original.stat().st_size, codec="h264", state="READY"))
                db.commit()
            before = {table: rows(engine, table) for table in ATTENDANCE_TABLES}
            playback = PlaybackManager(engine, recordings.storage, ffmpeg, ffprobe)
            app = FastAPI()
            app.include_router(feature("app.api.recordings").router)
            app.state.recordings, app.state.playbacks = recordings, playback
            client = Client(app)
            requested = start.replace(tzinfo=timezone.utc) + timedelta(milliseconds=1)
            created = client.json("POST", "/api/recordings/playbacks", {
                "camera_id": source.id, "source": "local", "at": requested.isoformat(),
                "request_id": str(uuid4()),
            }, status=202)
            status_url = f"/api/recordings/playbacks/{created['id']}"
            deadline = time.monotonic() + 20

            def observable_state():
                result = client.json("GET", status_url)
                assert result["state"] not in {"FAILED", "CANCELLED", "EXPIRED"}, result
                return result if result["state"] in {"READY", "ENDED"} else None

            available = eventually(observable_state, "Real HLS did not become playable", timeout=20)
            status, headers, manifest = client.request("GET", available["manifest_url"])
            assert status == 200 and headers[b"cache-control"] == b"no-store"
            lines = manifest.decode("utf-8").splitlines()
            first_time = next(line.split(":", 1)[1] for line in lines
                              if line.startswith("#EXT-X-PROGRAM-DATE-TIME:"))
            assert utc(first_time) == requested.replace(tzinfo=None), first_time
            segment_url = next(line for line in lines if line and not line.startswith("#"))
            status, _headers, segment = asyncio.run(media_request(SimpleNamespace(app=app), segment_url))
            assert status == 200 and segment, (status, segment)
            served = directory / "first-served.ts"
            served.write_bytes(segment)
            probed = subprocess.run([
                ffprobe, "-v", "error", "-show_streams", "-of", "json", str(served),
            ], stdin=subprocess.DEVNULL, capture_output=True, timeout=10, close_fds=True)
            assert probed.returncode == 0, probed.stderr.decode("utf-8", "replace")
            streams = json.loads(probed.stdout)["streams"]
            assert len(streams) == 1, streams
            video = streams[0]
            assert (video["codec_name"], video["width"], video["height"], video["pix_fmt"]) == (
                "h264", 322, 242, "yuv420p"), video
            decoded = subprocess.run([
                ffmpeg, "-hide_banner", "-loglevel", "error", "-i", str(served),
                "-map", "0:v:0", "-frames:v", "1", "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1",
            ], stdin=subprocess.DEVNULL, capture_output=True, timeout=10, close_fds=True)
            assert decoded.returncode == 0, decoded.stderr.decode("utf-8", "replace")
            pixels = decoded.stdout
            assert len(pixels) == 322 * 242 * 3, len(pixels)
            mean_red = sum(pixels[0::3]) / (322 * 242)
            mean_blue = sum(pixels[2::3]) / (322 * 242)
            assert mean_red > mean_blue + 100, (
                f"1ms local seek skipped the red first GOP: R={mean_red:.1f}, B={mean_blue:.1f}")

            def ended_state():
                result = observable_state()
                return result if result and result["state"] == "ENDED" else None

            finished = eventually(ended_state, "Real HLS did not finish within 20s",
                                  timeout=max(.01, deadline - time.monotonic()))
            assert finished["error_code"] is None, finished
            assert (utc(finished["available_to"]) - utc(finished["available_from"])).total_seconds() >= 5.8, finished
            assert {table: rows(engine, table) for table in ATTENDANCE_TABLES} == before
            client.json("DELETE", status_url)
        finally:
            try:
                if playback is not None:
                    playback.shutdown()
            finally:
                recordings.shutdown()


class PlaybackProcess:
    """External encoder double. Closed outputs are not evidence of real decoding."""
    def __init__(self, args, factory, **kwargs):
        assert isinstance(args, list) and not kwargs.get("shell")
        self.args, self.factory = args, factory
        assert args[args.index("-c:v") + 1] == "libx264"
        for option, value in (("-pix_fmt", "yuv420p"), ("-preset", "veryfast"), ("-crf", "23"),
                              ("-threads", "2"), ("-readrate", "1"), ("-hls_time", "2"),
                              ("-hls_list_size", "30")):
            assert args[args.index(option) + 1] == value, (option, args)
        assert all(flag in args for flag in ("-an", "-sn", "-dn", "-force_key_frames"))
        assert "delete_segments" not in " ".join(args)
        assert "temp_file" in args[args.index("-hls_flags") + 1]
        self.pattern = Path(args[args.index("-hls_segment_filename") + 1])
        self.manifest = Path(args[-1])
        error = factory.start_errors.pop(0) if factory.start_errors else b""
        self.stdout, self.stderr = io.BytesIO(), io.BytesIO(error)
        self.stdin = InputPipe(self)
        self.returncode = 1 if error else None
        self.pid = 800000 + len(factory.processes)
        self.durations, self.paths = [], []

    def publish(self, count=2, *, duration=2., ended=False):
        while len(self.paths) < count:
            sequence = len(self.paths)
            path = Path(str(self.pattern) % sequence)
            path.write_text(json.dumps({"duration": duration, "codec": "h264"}), encoding="utf-8")
            self.paths.append(path)
            self.durations.append(duration)
        first = max(0, count - 30)
        lines = ["#EXTM3U", "#EXT-X-VERSION:3", "#EXT-X-TARGETDURATION:2",
                 f"#EXT-X-MEDIA-SEQUENCE:{first}"]
        for sequence in range(first, count):
            # Deliberately wrong encoder wall clock: public manifest must rewrite it.
            lines += ["#EXT-X-PROGRAM-DATE-TIME:2099-01-01T00:00:00Z",
                      f"#EXTINF:{self.durations[sequence]:.6f},", self.paths[sequence].name]
        if ended:
            lines.append("#EXT-X-ENDLIST")
        temporary = self.manifest.with_suffix(".m3u8.tmp")
        temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
        temporary.replace(self.manifest)
        if ended:
            self.returncode = 0

    def finish(self):
        self.returncode = 0

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        if self.returncode is None:
            raise subprocess.TimeoutExpired("playback fixture", timeout)
        return self.returncode

    def communicate(self, input=None, timeout=None):
        if input:
            self.stdin.write(input)
        self.wait(timeout)
        return self.stdout.getvalue(), self.stderr.getvalue()

    def terminate(self):
        self.returncode = 1

    def kill(self):
        self.terminate()


class PlaybackProcesses:
    def __init__(self, r):
        self.r = r
        self.processes = []
        self.decode_ok = True
        self.encoder_available = True
        self.decoded = []
        self.start_errors = []

    def popen(self, args, **kwargs):
        process = PlaybackProcess(args, self, **kwargs)
        self.processes.append(process)
        return process

    def run(self, args, **kwargs):
        if args[0] == "ffprobe-fixture":
            return self.r.factory.probe(args, **kwargs)
        assert args[0] == "ffmpeg-fixture", args
        if "-encoders" in args:
            output = " V..... libx264 H.264 encoder\n" if self.encoder_available else ""
            return subprocess.CompletedProcess(args, 0, output if kwargs.get("text") else output.encode(), b"")
        assert "-i" in args and "-f" in args and "null" in args, args
        self.decoded.append(Path(args[args.index("-i") + 1]))
        return subprocess.CompletedProcess(args, 0 if self.decode_ok else 1, b"", b"")


@contextmanager
def playback_rig(*, seed=True, duration=120, policy=None):
    module = feature("app.media.playback")
    assert hasattr(module, "PlaybackManager"), "Task 5 PlaybackManager is missing"
    with rig(policy=policy) as r, ExitStack() as stack:
        if seed:
            r.recording, r.segment, r.input_path = ready_segment(r, duration=duration)
        factory = PlaybackProcesses(r)
        stack.enter_context(patch("subprocess.Popen", side_effect=factory.popen))
        stack.enter_context(patch("subprocess.run", side_effect=factory.run))
        manager = module.PlaybackManager(r.engine, r.manager.storage, "ffmpeg-fixture", "ffprobe-fixture",
                                         now=r.clock.now, monotonic=r.clock.monotonic)
        r.app.state.playbacks = manager
        r.playback, r.encoder = manager, factory
        try:
            yield r
        finally:
            manager.shutdown()


def playback_payload(r, *, at=None, source="local", request_id=None):
    return {"camera_id": r.source.id, "source": source,
            "at": at or "2026-11-12T10:00:10+00:00", "request_id": request_id or str(uuid4())}


def play(r, **kwargs):
    return r.client.json("POST", "/api/recordings/playbacks", playback_payload(r, **kwargs), status=202)


def playing(r, identifier):
    return r.client.json("GET", f"/api/recordings/playbacks/{identifier}")


def await_playback(r, identifier, states):
    return eventually(lambda: result if (result := playing(r, identifier))["state"] in states else None)


def timeline(r, source="local", start="2026-11-12T10:00:00+00:00",
             end="2026-11-12T11:00:00+00:00", status=200):
    query = urlencode({"camera_id": r.source.id, "source": source, "from": start, "to": end})
    return r.client.json("GET", "/api/recordings/timeline?" + query, status=status)


def assert_media_private(value):
    text = json.dumps(value, default=str)
    assert SECRET not in text and "/mnt/" not in text and "rtsp://" not in text
    assert "storage_key" not in text and "ffmpeg-fixture" not in text


async def media_request(r, path, *, range_header=None, on_send=None):
    """ASGI 2.4 peer allows a controlled send failure or held response lease."""
    messages = []

    async def receive():
        await asyncio.Event().wait()

    async def send(message):
        messages.append(message)
        if on_send:
            await on_send(message)

    headers = [(b"host", b"test")]
    if range_header:
        headers.append((b"range", range_header.encode()))
    await r.app({"type": "http", "asgi": {"version": "3.0", "spec_version": "2.4"},
                 "http_version": "1.1", "method": "GET", "scheme": "http", "path": path,
                 "raw_path": path.encode(), "query_string": b"", "root_path": "",
                 "headers": headers, "client": ("127.0.0.1", 1), "server": ("test", 80)},
                receive, send)
    response = next(message for message in messages if message["type"] == "http.response.start")
    return response["status"], dict(response["headers"]), b"".join(
        message.get("body", b"") for message in messages if message["type"] == "http.response.body")


@test
def test_playback_routes_validate_times_security_and_no_browser_supplied_paths():
    with playback_rig() as r:
        for at in ("2026-11-12T10:00:10", "9999-12-31T23:59:59-14:00",
                   "0001-01-01T00:00:00+14:00", "not-a-date"):
            r.client.json("POST", "/api/recordings/playbacks", playback_payload(r, at=at), status=422)
            timeline(r, start=at, status=422)
        for end in ("2026-11-12T10:00:00Z", "2026-11-13T10:00:01Z"):
            timeline(r, end=end, status=422)
        payload = playback_payload(r)
        r.client.json("POST", "/api/recordings/playbacks", payload, operator=False, status=403)
        r.client.json("POST", "/api/recordings/playbacks", payload,
                      headers={"origin": "http://evil.invalid"}, status=403)
        r.client.json("POST", "/api/recordings/playbacks", {**payload, "path": "/mnt/anything"}, status=422)
        r.client.json("POST", "/api/recordings/playbacks", {**payload, "source": "live"}, status=422)
        r.client.json("GET", f"/api/recordings/playbacks/{uuid4()}", status=404)
        assert not r.encoder.processes


@test
def test_local_timeline_gap_conflict_and_persisted_states_never_invent_coverage():
    with playback_rig(duration=60) as r:
        from app.database.models import RecordingGap
        with Session(r.engine) as db:
            db.add(RecordingGap(id=str(uuid4()), session_id=r.recording["id"],
                               started_at=BASE + timedelta(seconds=60),
                               ended_at=BASE + timedelta(seconds=80), reason_code="CONNECTION_LOST"))
            db.commit()
        result = timeline(r)
        assert result["time_basis"] == "approximate_server_recording_time"
        assert len(result["intervals"]) == 1 and result["intervals"][0]["state"] == "READY"
        assert utc(result["intervals"][0]["from"]) == BASE
        assert utc(result["intervals"][0]["to"]) == BASE + timedelta(seconds=60)
        assert result["gaps"][0]["reason"] == "CONNECTION_LOST"
        conflict = r.client.json("POST", "/api/recordings/playbacks",
                                 playback_payload(r, at="2026-11-12T10:01:10Z"), status=409)
        assert "NO_RECORDING_AT_TIME" in str(conflict) and "next_at" in str(conflict)
        assert not r.encoder.processes
        for state in ("EXPIRED", "MISSING", "CORRUPT"):
            with r.engine.begin() as db:
                db.exec_driver_sql("UPDATE recording_segments SET state=? WHERE id=?", (state, r.segment["id"]))
            result = timeline(r)
            assert result["intervals"][0]["state"] == state
            r.client.json("POST", "/api/recordings/playbacks", playback_payload(r), status=409)
        assert_media_private(result)


@test
def test_playback_one_worker_replay_conflict_cancel_and_no_attendance_writes():
    with playback_rig() as r:
        before = {table: rows(r.engine, table) for table in ATTENDANCE_TABLES}
        payload = playback_payload(r)
        first = r.client.json("POST", "/api/recordings/playbacks", payload, status=202)
        assert first["state"] == "PREPARING" and first["error_code"] is None
        assert utc(first["from"]) == BASE + timedelta(seconds=10)
        assert utc(first["to"]) == BASE + timedelta(seconds=120)
        replay = r.client.json("POST", "/api/recordings/playbacks", payload, status=202)
        assert replay["id"] == first["id"]
        r.client.json("POST", "/api/recordings/playbacks",
                      {**payload, "at": "2026-11-12T10:00:11Z"}, status=409)
        r.client.json("POST", "/api/recordings/playbacks",
                      {**payload, "request_id": str(uuid4())}, status=429)
        process = eventually(lambda: r.encoder.processes and r.encoder.processes[0])
        r.client.json("DELETE", f'/api/recordings/playbacks/{first["id"]}', operator=False, status=403)
        r.client.json("DELETE", f'/api/recordings/playbacks/{first["id"]}')
        assert playing(r, first["id"])["state"] == "CANCELLED"
        assert process.poll() is not None
        r.client.json("DELETE", f'/api/recordings/playbacks/{first["id"]}')
        assert len(r.encoder.processes) == 1
        assert {table: rows(r.engine, table) for table in ATTENDANCE_TABLES} == before
        assert_media_private(first)


@test
def test_progressive_ready_requires_validation_rewrites_source_time_and_serves_ranges():
    with playback_rig() as r:
        first = play(r)
        process = eventually(lambda: r.encoder.processes and r.encoder.processes[0])
        process.publish(1)
        assert playing(r, first["id"])["state"] == "PREPARING"
        process.publish(2)
        ready = await_playback(r, first["id"], {"READY"})
        assert process.poll() is None, "READY must not await completed output"
        assert r.encoder.decoded, "ffprobe metadata alone is not decodable-first-segment validation"
        status, headers, body = r.client.request("GET", ready["manifest_url"])
        assert status == 200 and headers[b"cache-control"] == b"no-store"
        manifest = body.decode()
        dates = [line.split(":", 1)[1] for line in manifest.splitlines()
                 if line.startswith("#EXT-X-PROGRAM-DATE-TIME:")]
        assert [utc(date) for date in dates] == [BASE + timedelta(seconds=10), BASE + timedelta(seconds=12)]
        urls = [line for line in manifest.splitlines() if line and not line.startswith("#")]
        assert all(url.startswith(f'/api/recordings/playbacks/{first["id"]}/segments/') for url in urls)
        status, headers, body = asyncio.run(media_request(r, urls[0], range_header="bytes=0-9"))
        assert status == 206 and len(body) == 10 and headers[b"content-type"].startswith(b"video/mp2t")
        status, headers, _body = asyncio.run(media_request(r, urls[0], range_header="bytes=999999-"))
        assert status == 416 and headers[b"content-range"].startswith(b"bytes */")
        r.client.json("GET", f'/api/recordings/playbacks/{first["id"]}/segments/999999.ts', status=404)
        assert_media_private(manifest)


@test
def test_invalid_decodable_segment_fails_instead_of_ready_and_short_end_is_visible():
    with playback_rig() as r:
        r.encoder.decode_ok = False
        first = play(r)
        process = eventually(lambda: r.encoder.processes and r.encoder.processes[0])
        process.publish(2)
        failed = await_playback(r, first["id"], {"FAILED"})
        assert failed["error_code"] and process.poll() is not None
        assert "manifest_url" not in failed or failed["manifest_url"] is None
    with playback_rig(duration=12) as r:
        first = play(r)
        process = eventually(lambda: r.encoder.processes and r.encoder.processes[0])
        process.publish(1, ended=True)
        ended = await_playback(r, first["id"], {"READY", "ENDED"})
        assert ended["manifest_url"] and ended["error_code"] is None
    with playback_rig() as r:
        first = play(r)
        process = eventually(lambda: r.encoder.processes and r.encoder.processes[0])
        process.publish(2, ended=True)
        ended = await_playback(r, first["id"], {"ENDED", "FAILED"})
        assert ended["error_code"], "Unexpected shortened footage must not look like a complete interval"


@test
def test_short_final_manifest_published_between_scan_and_process_exit_remains_playable():
    original_poll = PlaybackProcess.poll

    def publish_before_exit(process):
        if not process.manifest.exists():
            process.publish(1, ended=True)
        return original_poll(process)

    with playback_rig(duration=12) as r, patch.object(PlaybackProcess, "poll", new=publish_before_exit):
        first = play(r)
        ended = await_playback(r, first["id"], {"READY", "ENDED"})
        assert ended["manifest_url"] and ended["error_code"] is None


@test
def test_fractional_local_start_and_next_at_round_trip_to_playable_time():
    from app.database.models import RecordingSegment
    with playback_rig() as r:
        with Session(r.engine) as db:
            row = db.get(RecordingSegment, r.segment["id"])
            row.started_at += timedelta(microseconds=224)
            row.ended_at += timedelta(microseconds=224)
            stored_start = row.started_at
            db.commit()
        advertised = next(item["from"] for item in timeline(r)["intervals"] if item["state"] == "READY")
        gap = r.client.json("POST", "/api/recordings/playbacks",
                           playback_payload(r, at="2026-11-12T09:59:59+00:00"), status=409)
        assert gap["detail"]["next_at"] == advertised
        assert utc(advertised) >= stored_start, "Public start must not precede actual footage"
        created = play(r, at=gap["detail"]["next_at"])
        assert created["from"] == advertised


@test
def test_local_playback_identity_survives_credentials_channel_offset_and_releases_inputs():
    with playback_rig(policy={"recording_min_free_bytes": 0}) as r:
        first = play(r)
        process = eventually(lambda: r.encoder.processes and r.encoder.processes[0])
        r.client.json("PUT", f"/api/cameras/{r.source.id}", {
            "password": "edited-secret", "channel": 2,
            "archive": {"http_scheme": "http", "http_port": 80, "utc_offset_minutes": 240},
            "expected_archive_revision": None})
        assert playing(r, first["id"])["state"] == "PREPARING" and process.poll() is None
        # Retention cannot remove the input while this local worker needs it.
        with patch.object(r.manager.storage, "now", return_value=BASE + timedelta(days=15)):
            r.manager.storage.enforce()
            assert r.input_path.exists()
            r.client.json("DELETE", f'/api/recordings/playbacks/{first["id"]}')
            r.manager.storage.enforce()
        assert not r.input_path.exists(), "Cancellation must release the original input lease"
        assert rows(r.engine, "recording_segments")[0]["state"] == "EXPIRED"


@test
def test_playback_no_progress_abandonment_and_cache_expiry_return_410_not_unknown():
    with playback_rig() as r:
        first = play(r)
        process = eventually(lambda: r.encoder.processes and r.encoder.processes[0])
        r.clock.advance(31)
        failed = await_playback(r, first["id"], {"FAILED"})
        assert failed["error_code"] and process.poll() is not None
        r.clock.advance(121)
        eventually(lambda: r.client.request("GET", f'/api/recordings/playbacks/{first["id"]}')[0] == 410)
    with playback_rig(duration=600) as r:
        first = play(r)
        process = eventually(lambda: r.encoder.processes and r.encoder.processes[0])
        process.publish(2)
        await_playback(r, first["id"], {"READY"})
        r.clock.advance(121)
        eventually(lambda: process.poll() is not None, "Abandoned player still owns an encoder")
        assert await_playback(r, first["id"], {"CANCELLED"})["state"] == "CANCELLED"
        assert len(r.encoder.processes) == 1


@test
def test_response_leases_release_on_416_disconnect_and_survive_eviction_during_send():
    for outcome in ("416", "disconnect", "held"):
        with playback_rig(policy={"recording_min_free_bytes": 0}) as r:
            first = play(r)
            process = eventually(lambda: r.encoder.processes and r.encoder.processes[0])
            process.publish(2)
            await_playback(r, first["id"], {"READY"})
            url = f'/api/recordings/playbacks/{first["id"]}/segments/0.ts'
            path = process.paths[0]
            if outcome == "416":
                assert asyncio.run(media_request(r, url, range_header="bytes=999999-"))[0] == 416
            else:
                async def during_send(message):
                    if message["type"] != "http.response.start":
                        return
                    if outcome == "disconnect":
                        raise OSError("client disconnected")
                    # Manager retirement must defer physical deletion until the response exits.
                    await asyncio.to_thread(r.playback.cancel, first["id"])
                    r.clock.advance(121)
                    await asyncio.to_thread(r.manager.storage.enforce)
                    assert path.exists(), "A response-lifetime lease was released before send completed"
                try:
                    asyncio.run(media_request(r, url, on_send=during_send))
                except OSError:
                    assert outcome == "disconnect"
            if outcome != "held":
                r.playback.cancel(first["id"])
            r.clock.advance(121)
            eventually(lambda: r.client.request("GET", f'/api/recordings/playbacks/{first["id"]}')[0] == 410)
            r.manager.storage.enforce()
            assert not path.exists(), f"{outcome} leaked a response cache lease"


@contextmanager
def remote_playback_rig(*, failures=None):
    from test_dahua_archive import device, FinderPeer, finder_page
    finder = FinderPeer([finder_page(), b"found=0\r\n"], failures=failures)
    with device(finder=finder) as peer, playback_rig(seed=False) as r:
        # The HTTP fixture's digest credentials are shared only at the external boundary.
        from test_dahua_archive import PASSWORD
        r.client.json("PUT", f"/api/cameras/{r.source.id}", {
            "nvr_host": "127.0.0.1", "password": PASSWORD,
            "archive": {"http_scheme": "http", "http_port": peer.port, "utc_offset_minutes": 180},
            "expected_archive_revision": None})
        r.finder, r.peer = finder, peer
        yield r


@test
def test_remote_timeline_caches_both_revisions_and_never_reports_device_errors_as_empty():
    with playback_rig(seed=False) as r:
        result = timeline(r, "dahua", status=409)
        assert "ARCHIVE_NOT_CONFIGURED" in str(result)
    for failures, code, status in (
        ({"findFile": (401, b"bad authentication")}, "DEVICE_AUTH_FAILED", 502),
        ({"findFile": (404, b"unsupported")}, "ARCHIVE_UNSUPPORTED", 422),
        ({"findNextFile": (200, b"malformed")}, "DEVICE_PROTOCOL_ERROR", 502),
    ):
        with remote_playback_rig(failures=failures) as r:
            result = timeline(r, "dahua", status=status)
            assert code in str(result) and "intervals" not in result
            assert_media_private(result)
    with remote_playback_rig() as r:
        first = timeline(r, "dahua")
        assert len(first["intervals"]) == 1
        identifier = first["intervals"][0]["id"]
        assert "/mnt/" not in identifier and identifier
        assert utc(first["intervals"][0]["from"]) == BASE
        assert timeline(r, "dahua") == first and r.finder.page == 2
        r.clock.advance(61)
        r.finder.page = 0
        timeline(r, "dahua")
        assert r.finder.actions.count("factory.create") == 2
        r.finder.page = 0
        r.client.json("PUT", f"/api/cameras/{r.source.id}", {
            "archive": {"http_scheme": "http", "http_port": r.peer.port, "utc_offset_minutes": 240},
            "expected_archive_revision": 0})
        timeline(r, "dahua")
        assert r.finder.actions.count("factory.create") == 3
        r.finder.page = 0
        r.client.json("PUT", f"/api/cameras/{r.source.id}", {"rtsp_port": 8554})
        timeline(r, "dahua")
        assert r.finder.actions.count("factory.create") == 4
        assert_media_private(first)


@test
def test_remote_timeline_overlong_finder_index_returns_safe_502_and_cleans():
    from test_dahua_archive import PASSWORD, finder_page
    with remote_playback_rig() as r:
        r.finder.pages = [finder_page() + b"items[" + b"9" * 5000 + b"].Length=12345\r\n"]
        try:
            result = timeline(r, "dahua", status=502)
            assert result == {"detail": "DEVICE_PROTOCOL_ERROR"}, result
            assert_media_private(result)
            assert PASSWORD not in json.dumps(result)
        finally:
            assert r.finder.actions == [
                "factory.create", "findFile", "findNextFile", "close", "destroy"]


@test
def test_remote_uses_time_range_first_and_only_explicit_unsupported_allows_filepath():
    with remote_playback_rig() as r:
        r.encoder.start_errors = [b"[rtsp] method DESCRIBE failed: 404 Not Found\n"]
        first = play(r, source="dahua")
        eventually(lambda: len(r.encoder.processes) == 2)
        urls = [process.args[process.args.index("-i") + 1] for process in r.encoder.processes]
        initial, fallback = map(urlsplit, urls)
        assert initial.path == "/cam/playback"
        assert "channel=1" in initial.query
        assert "starttime=2026_11_12_13_00_10" in initial.query
        assert "endtime=2026_11_12_13_01_00" in initial.query
        assert fallback.path == "//mnt/dvr/2026/11/12/first%20%5BR%5D.dav"
        process = r.encoder.processes[-1]
        process.publish(2)
        ready = await_playback(r, first["id"], {"READY"})
        assert utc(ready["from"]) == BASE + timedelta(seconds=10)
        assert_media_private(ready)
    for stderr, code in (
        (b"[rtsp] method DESCRIBE failed: 401 Unauthorized\n", "DEVICE_AUTH_FAILED"),
        (b"Connection timed out\n", "DEVICE_TIMEOUT"),
        (b"Connection refused\n", "DEVICE_UNREACHABLE"),
    ):
        with remote_playback_rig() as r:
            r.encoder.start_errors = [stderr]
            first = play(r, source="dahua")
            failed = await_playback(r, first["id"], {"FAILED"})
            assert failed["error_code"] == code
            assert len(r.encoder.processes) == 1, "Auth/network failures must never select another profile"
    with remote_playback_rig() as r:
        r.encoder.start_errors = [b"method DESCRIBE failed: 404 Not Found\n"] * 2
        first = play(r, source="dahua")
        failed = await_playback(r, first["id"], {"FAILED"})
        assert failed["error_code"] == "ARCHIVE_UNSUPPORTED" and len(r.encoder.processes) == 2


@test
def test_remote_camera_publication_cancels_both_revision_bound_jobs():
    for changes in (
        {"rtsp_port": 8554},
        {"archive": {"http_scheme": "http", "http_port": 80, "utc_offset_minutes": 240},
         "expected_archive_revision": 0},
    ):
        with remote_playback_rig() as r:
            first = play(r, source="dahua")
            process = eventually(lambda: r.encoder.processes and r.encoder.processes[0])
            r.client.json("PUT", f"/api/cameras/{r.source.id}", changes)
            ended = await_playback(r, first["id"], {"CANCELLED", "FAILED"})
            assert ended["error_code"] == "SOURCE_CHANGED" and process.poll() is not None


def append_local_segment(r, *, start_seconds, duration=60, same_connection=True):
    """Persist a real indexed historical segment; only bytes/probe are media doubles."""
    from app.database.models import RecordingConnection, RecordingSegment
    identifier = str(uuid4())
    connection_id = r.segment["connection_id"] if same_connection else str(uuid4())
    directory = r.input_path.parent if same_connection else r.input_path.parent.parent / connection_id
    directory.mkdir(exist_ok=True)
    path = directory / f"segment-{identifier}.mkv"
    path.write_text(json.dumps({"duration": duration, "codec": "h264"}), encoding="utf-8")
    with Session(r.engine) as db:
        if not same_connection:
            db.add(RecordingConnection(id=connection_id, session_id=r.recording["id"],
                directory=directory.relative_to(r.root).as_posix(), source_revision=r.source.source_revision,
                created_at=BASE + timedelta(seconds=start_seconds), closed_at=BASE + timedelta(seconds=start_seconds + duration),
                anchor_utc=BASE + timedelta(seconds=start_seconds), media_origin_seconds=0))
            db.flush()
        db.add(RecordingSegment(id=identifier, session_id=r.recording["id"], connection_id=connection_id,
            camera_id=r.source.id, source_revision=r.source.source_revision,
            storage_key=path.relative_to(r.root).as_posix(), started_at=BASE + timedelta(seconds=start_seconds),
            ended_at=BASE + timedelta(seconds=start_seconds + duration), duration_ms=int(duration * 1000),
            bytes=path.stat().st_size, codec="h264", state="READY"))
        db.commit()
    return identifier, path


@test
def test_local_connected_run_accepts_only_csv_precision_not_reconnect_or_real_gap():
    for offset, same_connection, expected_end in ((60.001, True, 120.001),
                                                   (60.1, True, 60), (60, False, 60)):
        with playback_rig(duration=60) as r:
            append_local_segment(r, start_seconds=offset, same_connection=same_connection)
            first = play(r)
            assert utc(first["to"]) == BASE + timedelta(seconds=expected_end)
            if expected_end == 60:
                assert utc(first["next_at"]) == BASE + timedelta(seconds=offset)
    with playback_rig(duration=7200) as r:
        first = play(r)
        assert utc(first["to"]) == BASE + timedelta(seconds=3610)
        assert utc(first["next_at"]) == BASE + timedelta(seconds=3610)
    with playback_rig(duration=60) as r:
        append_local_segment(r, start_seconds=80)
        result = r.client.json("POST", "/api/recordings/playbacks",
                               playback_payload(r, at="2026-11-12T10:01:10Z"), status=409)
        assert "NO_RECORDING_AT_TIME" in str(result)
        assert utc(result["detail"]["next_at"]) == BASE + timedelta(seconds=80)


@test
def test_consumed_inputs_and_rolling_cache_release_without_compressing_program_time():
    with playback_rig(duration=60, policy={"recording_min_free_bytes": 0}) as r:
        _identifier, second_path = append_local_segment(r, start_seconds=60, duration=300)
        first = play(r, at="2026-11-12T10:00:00Z")
        process = eventually(lambda: r.encoder.processes and r.encoder.processes[0])
        process.publish(2)
        ready = await_playback(r, first["id"], {"READY"})
        for count in (12, 22, 32, 42, 52):
            process.publish(count)
            r.clock.advance(20)
            r.client.request("GET", ready["manifest_url"])
            eventually(lambda: utc(playing(r, first["id"])["available_to"]) >= BASE + timedelta(seconds=count * 2))
        status, _headers, body = r.client.request("GET", ready["manifest_url"])
        assert status == 200
        dates = [utc(line.split(":", 1)[1]) for line in body.decode().splitlines()
                 if line.startswith("#EXT-X-PROGRAM-DATE-TIME:")]
        assert dates[0] == BASE + timedelta(seconds=44) and len(dates) == 30
        r.manager.storage.enforce()
        assert not process.paths[0].exists(), "Unlisted cache beyond 30-second safety must be evicted"
        assert all(path.exists() for path in process.paths[-30:])
        with patch.object(r.manager.storage, "now", return_value=BASE + timedelta(days=15)):
            r.manager.storage.enforce()
            assert not r.input_path.exists(), "Already consumed local input lease unnecessarily pins retention"
            assert second_path.exists(), "Future input must stay leased while the encoder needs it"


@test
def test_playback_quota_pressure_and_missing_encoder_fail_closed():
    with playback_rig(policy={"recording_min_free_bytes": 0}) as r:
        first = play(r)
        process = eventually(lambda: r.encoder.processes and r.encoder.processes[0])
        process.publish(2)
        await_playback(r, first["id"], {"READY"})
        total = sum(path.stat().st_size for path in r.root.rglob("*") if path.is_file())
        with patch.object(r.manager.storage, "max_bytes", 268435456 + total - 1):
            failed = await_playback(r, first["id"], {"FAILED"})
        assert failed["error_code"] == "STORAGE_FULL" and process.poll() is not None
        assert r.input_path.exists(), "Cache pressure must not delete actively leased original footage"
    with playback_rig() as r:
        r.encoder.encoder_available = False
        result = r.client.json("POST", "/api/recordings/playbacks", playback_payload(r), status=503)
        assert "LIBX264" in str(result)
        assert not r.encoder.processes


@test
def test_concurrent_playback_admission_reserves_one_worker_without_queued_launch():
    with playback_rig() as r:
        def attempt():
            return r.client.request("POST", "/api/recordings/playbacks", playback_payload(r))[0]
        with ThreadPoolExecutor(max_workers=2) as pool:
            statuses = list(pool.map(lambda _index: attempt(), range(2)))
        assert sorted(statuses) == [202, 429]
        eventually(lambda: len(r.encoder.processes) == 1)
        r.playback.shutdown()
        assert r.encoder.processes[0].poll() is not None
        assert len(r.encoder.processes) == 1, "Rejected work must not be queued for later"


@test
def test_local_unavailable_identity_refuses_launch_without_touching_other_footage():
    for missing in (True, False):
        with playback_rig() as r:
            if missing:
                r.input_path.unlink()
            else:
                with r.engine.begin() as db:
                    db.exec_driver_sql("UPDATE recording_segments SET state='DELETING' WHERE id=?",
                                       (r.segment["id"],))
            result = r.client.json("POST", "/api/recordings/playbacks", playback_payload(r), status=409)
            assert "NO_RECORDING_AT_TIME" in str(result) or "UNAVAILABLE" in str(result)
            assert not r.encoder.processes


@test
def test_ready_no_progress_and_captured_duration_deadline_stop_even_when_polled():
    with playback_rig(duration=600) as r:
        first = play(r)
        process = eventually(lambda: r.encoder.processes and r.encoder.processes[0])
        process.publish(2)
        await_playback(r, first["id"], {"READY"})
        r.clock.advance(31)
        failed = await_playback(r, first["id"], {"FAILED"})
        assert failed["error_code"] and process.poll() is not None
    with playback_rig(duration=12) as r:
        first = play(r)
        process = eventually(lambda: r.encoder.processes and r.encoder.processes[0])
        process.publish(2, duration=1)
        await_playback(r, first["id"], {"READY"})
        # Requested footage is only two seconds; a stuck child cannot outlive duration + 60.
        r.clock.advance(63)
        ended = await_playback(r, first["id"], {"FAILED", "ENDED"})
        assert process.poll() is not None
        assert utc(ended["to"]) == BASE + timedelta(seconds=12)


@test
def test_crash_recovery_uses_observed_packet_end_not_inherited_container_duration():
    with rig() as r:
        saved = start(r)
        process = r.factory.processes[0]
        path = process.appear(duration=140)
        durable = eventually(lambda: rows(r.engine, "recording_segments"))[0]
        r.manager.shutdown()
        process.csv.unlink(missing_ok=True)
        # A copied Matroska header retains the full 140 s source duration, while
        # the abruptly terminated file contains packets only through 3.999 s.
        path.write_text(json.dumps({"duration": 140, "packet_end": 3.999, "codec": "h264"}), encoding="utf-8")
        with r.engine.begin() as db:
            db.exec_driver_sql("UPDATE recording_sessions SET status='RECORDING',ended_at=NULL WHERE id=?",
                               (saved["id"],))
            db.exec_driver_sql("UPDATE recording_connections SET media_origin_seconds=NULL,closed_at=NULL")
            db.exec_driver_sql(
                "UPDATE recording_segments SET state='UNINDEXED',ended_at=NULL,duration_ms=0 WHERE id=?",
                (durable["id"],))
        r.make_manager()
        recovered = eventually(lambda: next((row for row in rows(r.engine, "recording_segments")
                                             if row["id"] == durable["id"] and row["state"] == "READY"), None))
        assert recovered["duration_ms"] == 3999, recovered
        assert utc(recovered["started_at"]) == BASE
        assert utc(recovered["ended_at"]) == BASE + timedelta(milliseconds=3999)
        assert utc(current(r, saved["id"])["available_to"]) == BASE + timedelta(milliseconds=3999)
        assert current(r, saved["id"])["status"] == "INTERRUPTED"
        assert len(r.factory.processes) == 1, "A packet probe is not a recording child or automatic resume"


@test
def test_explicit_reconnect_gap_bounds_playback_even_when_buffered_segment_time_overlaps():
    with playback_rig(duration=60) as r:
        from app.database.models import RecordingGap
        append_local_segment(r, start_seconds=60, same_connection=False)
        with Session(r.engine) as db:
            db.add(RecordingGap(id=str(uuid4()), session_id=r.recording["id"],
                               started_at=BASE + timedelta(seconds=50),
                               ended_at=BASE + timedelta(seconds=70), reason_code="CONNECTION_LOST"))
            db.commit()
        first = play(r)
        assert utc(first["to"]) == BASE + timedelta(seconds=50)
        assert utc(first["next_at"]) == BASE + timedelta(seconds=70)
        r.client.json("DELETE", f'/api/recordings/playbacks/{first["id"]}')
        conflict = r.client.json("POST", "/api/recordings/playbacks",
                                 playback_payload(r, at="2026-11-12T10:00:55Z"), status=409)
        assert "NO_RECORDING_AT_TIME" in str(conflict)
        assert utc(conflict["detail"]["next_at"]) == BASE + timedelta(seconds=70)
        next_playback = play(r, at="2026-11-12T10:01:15Z")
        assert utc(next_playback["from"]) == BASE + timedelta(seconds=75)
        assert utc(next_playback["to"]) == BASE + timedelta(seconds=120)


@test
def test_remote_overlapping_intervals_continue_at_selected_endpoint_without_rewind():
    from test_dahua_archive import finder_page
    for second_start in ("2026-11-12 13:00:30", "2026-11-12 13:00:00"):
        with remote_playback_rig() as r:
            r.finder.pages = [
                finder_page(),
                finder_page(path="/mnt/dvr/longer.mp4", start=second_start,
                            end="2026-11-12 13:02:00", kind="mp4"),
                b"found=0\r\n",
            ]
            first = play(r, source="dahua")
            assert utc(first["to"]) == BASE + timedelta(seconds=60)
            assert first["next_at"] == first["to"], "Continuation must be after this clip, not an overlapping file's earlier start"
            process = eventually(lambda: r.encoder.processes and r.encoder.processes[0])
            r.client.json("DELETE", f'/api/recordings/playbacks/{first["id"]}')
            assert process.poll() is not None
            # A new query has a new range key; the recorder returns its real
            # overlapping files again, rather than pre-filtering for the app.
            r.finder.page = 0
            following = play(r, source="dahua", at=first["next_at"])
            assert following["id"] != first["id"]
            assert following["from"] == first["to"]
            assert utc(following["to"]) == BASE + timedelta(seconds=120)
            assert following["next_at"] is None
            assert_media_private(following)


@test
def test_local_hourly_continuation_remains_available_after_cancel_and_new_request():
    with playback_rig(duration=60) as r:
        append_local_segment(r, start_seconds=60, duration=7140)
        first = play(r)
        assert utc(first["to"]) == BASE + timedelta(seconds=3610)
        assert first["next_at"] == first["to"]
        process = eventually(lambda: r.encoder.processes and r.encoder.processes[0])
        r.client.json("DELETE", f'/api/recordings/playbacks/{first["id"]}')
        assert process.poll() is not None
        following = play(r, at=first["next_at"])
        assert following["id"] != first["id"]
        assert following["from"] == first["to"]
        assert utc(following["to"]) == BASE + timedelta(seconds=7200)
        assert following["next_at"] is None
        assert rows(r.engine, "recording_segments")[0]["state"] == "READY"


@test
def test_playback_cache_only_exhaustion_stops_worker_with_response_leased_outputs_intact():
    with playback_rig(policy={"recording_min_free_bytes": 0, "recording_max_bytes": 10_000_000_000}) as r:
        first = play(r)
        process = eventually(lambda: r.encoder.processes and r.encoder.processes[0])
        process.publish(2)
        await_playback(r, first["id"], {"READY"})
        storage = r.manager.storage
        total, free, reserve = storage.usage()
        assert total + reserve < storage.max_bytes and free > storage.min_free_bytes + reserve
        assert storage.enforce(), "Aggregate storage pressure must not cause this regression"
        paths = tuple(process.paths)
        budget = paths[0].stat().st_size - 1
        enforce_cache = storage.enforce_cache

        async def hold_response(message):
            if message["type"] != "http.response.start":
                return
            assert message["status"] == 206
            # Scale only the existing cache-policy argument; run its actual
            # eviction/lease implementation. Aggregate limits stay healthy.
            with patch.object(storage, "enforce_cache", side_effect=lambda _maximum: enforce_cache(budget)):
                failed = await asyncio.to_thread(await_playback, r, first["id"], {"FAILED"})
                assert failed["error_code"] == "STORAGE_FULL", failed
                assert process.poll() is not None
                assert storage.enforce(), "The aggregate enforcer is not the cause of failure"
                # Sequence 1 had only its listed-output lease during pressure;
                # sequence 0 additionally has this still-in-flight response.
                assert all(path.exists() for path in paths), "Cache pressure deleted a listed TS before worker stop"
                assert r.input_path.exists(), "Cache-only pressure must not consume original footage"
                # Terminal outputs are now unused except for sequence 0. Even a
                # further eviction pass cannot remove the held response's file.
                assert not enforce_cache(budget), "The held response alone exceeds the cache budget"
                assert paths[0].exists(), "Cache pressure removed an in-flight response"

        status, _headers, body = asyncio.run(media_request(
            r, f'/api/recordings/playbacks/{first["id"]}/segments/0.ts',
            range_header="bytes=0-9", on_send=hold_response))
        assert status == 206 and len(body) == 10
        r.clock.advance(121)
        storage.enforce()
        assert all(not path.exists() for path in paths), "Completed responses leaked their cache leases"


if __name__ == "__main__":
    if "--owned-parent" in sys.argv:
        owned_parent(sys.argv[2], sys.argv[3])
    elif "--owned-lock-contender" in sys.argv:
        owned_lock_contender(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        selected = list(TESTS)
        if "--windows-process-integration" in sys.argv:
            selected.append(test_windows_owned_parent_death_kills_actual_ffmpeg_and_stops_file_growth)
            selected.append(test_windows_root_owner_excludes_other_process_before_recovery)
            selected.append(test_windows_local_seek_between_keyframes_preserves_first_scene_and_duration)
        failures = 0
        for case in selected:
            try:
                case()
                print(f"PASS {case.__name__}")
            except Exception:
                failures += 1
                print(f"FAIL {case.__name__}")
                traceback.print_exc()
        print(f"{len(selected) - failures}/{len(selected)} tests passed")
        sys.exit(bool(failures))
