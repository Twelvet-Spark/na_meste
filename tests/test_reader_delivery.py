"""Latest-only decode and bounded native/preview delivery; isolated from cameras/DB.

Run: .venv/Scripts/python.exe tests/test_reader_delivery.py
The FFmpeg filter cases use generated lavfi frames only, never recorder URLs.
"""
import io
import subprocess
import weakref
import sys
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException

from app.api import cameras as api
from app.config import resolve_ffmpeg, settings
from app.media.process_guard import MediaProcessGuard
from app.rtsp.jpeg_cache import JPEGCache
from app.rtsp.manager import ReaderManager
from app.rtsp.reader import CameraStatus, ConnectionLost, FrameBuffer, RTSPReader
from app.rtsp.source import CameraDeletion, CameraSource
from operator_test_support import Client

TESTS = []
STAMP = datetime(2026, 11, 11, 4)


def test(fn):
    TESTS.append(fn)
    return fn


def publish(reader, *, stream="a", mono=100., frame=None):
    if frame is None:
        frame = np.full((1080, 1920, 3), 64, np.uint8)
    reader.buffer.put(frame, stream_id=stream, source_revision=reader.source_revision,
                      received_at=STAMP, received_mono=mono)
    reader._set_status(CameraStatus.ONLINE)
    return reader.read()


def make_reader(**kwargs):
    return RTSPReader(1, "Door", "rtsp://unused", source_revision=7,
                      ffmpeg_path="unused", ffprobe_path="unused", **kwargs)


class Manager:
    def __init__(self, reader):
        self.reader = reader
        self.jpeg_cache = JPEGCache()

    def get(self, camera_id):
        assert camera_id == 1
        return self.reader


class Process:
    def __init__(self, data=b"", error=b""):
        self.stdout = io.BytesIO(data)
        self.stderr = io.BytesIO(error)
        self.terminated = False
        self.alive = True

    def poll(self):
        return None if self.alive else 1

    def terminate(self):
        self.terminated = True
        self.alive = False

    def wait(self, timeout=None):
        self.alive = False
        return 1

    def kill(self):
        self.terminate()


def decode_attempt(reader, processes, on_frame=None):
    commands = []
    pending = iter(processes)
    original = reader.buffer.put

    def launch(command, **kwargs):
        commands.append(command)
        return next(pending)

    def put(frame, **metadata):
        result = original(frame, **metadata)
        if on_frame is not None:
            on_frame(reader.snapshot())
        return result

    with patch.object(reader, "_probe", return_value=(2, 2, "hevc", 2.)), \
            patch.object(MediaProcessGuard, "require"), \
            patch("app.rtsp.reader.subprocess.Popen", side_effect=launch), \
            patch.object(reader.buffer, "put", side_effect=put):
        try:
            reader._connect_and_read()
        except ConnectionLost:
            pass
        else:
            raise AssertionError("A finite decoder must report EOF")
    return commands


@test
def test_slot_keeps_only_latest_without_copy_and_clear_never_reuses_sequence():
    buffer = FrameBuffer()
    frames = [np.full((3, 4, 3), index, np.uint8) for index in range(5)]
    for frame in frames:
        buffer.put(frame, stream_id="a", source_revision=1,
                   received_at=STAMP, received_mono=100.)
    packet = buffer.wait_for_new(0, 0)
    assert packet.seq == 5 and packet.frame is frames[-1]
    assert not packet.frame.flags.writeable
    try:
        packet.frame[0, 0, 0] = 255
    except ValueError:
        pass
    else:
        raise AssertionError("A consumer cannot mutate published pixels")
    buffer.clear()
    assert buffer.latest() is None and buffer.wait_for_new(0, 0) is None
    buffer.put(frames[0], stream_id="b", source_revision=2,
               received_at=STAMP, received_mono=101.)
    assert buffer.latest().seq == 6 and buffer.latest().stream_id == "b"


@test
def test_auto_retries_actual_hardware_initialization_failure_once_before_first_frame():
    reader = make_reader(hwaccel="auto")
    hardware = Process(error=b"Cannot load nvcuda.dll\nDevice creation failed: -1.\n")
    software = Process(data=b"\0" * 12)
    snapshots = []
    commands = decode_attempt(reader, [hardware, software], snapshots.append)
    assert len(commands) == 2
    assert commands[0].index("-hwaccel") < commands[0].index("-i")
    assert commands[0][commands[0].index("-hwaccel") + 1] == "cuda"
    assert "-hwaccel" not in commands[1]
    assert snapshots[0]["decoder_mode"] == "software"
    assert snapshots[0]["decoder_warning"] == "DECODE_SOFTWARE_FALLBACK"
    assert hardware.terminated and software.terminated
    assert reader.read() is None
    # Reconnects of this reader generation remain software; no repeated CUDA probe.
    assert "-hwaccel" not in decode_attempt(reader, [Process()])[0]


@test
def test_auto_does_not_mask_corruption_or_fallback_after_a_real_frame():
    reader = make_reader(hwaccel="auto")
    assert len(decode_attempt(reader, [Process(error=b"Invalid data found when processing input\n")])) == 1
    observed = []
    decode_attempt(reader, [Process(data=b"\0" * 12)], observed.append)
    assert observed[0]["decoder_mode"] == "cuda"
    assert observed[0]["decoder_warning"] is None
    assert len(decode_attempt(reader, [Process(error=b"Cannot load nvcuda.dll\n")])) == 1
    assert reader.snapshot()["error"] == "DECODE_HARDWARE_INIT_FAILED"


@test
def test_explicit_cuda_has_visible_hardware_error_and_never_software_retry():
    reader = make_reader(hwaccel="cuda")
    assert reader.snapshot()["decoder_mode"] is None
    commands = decode_attempt(reader, [Process(error=b"No device available for decoder: device type cuda needed for codec hevc.\n")])
    assert len(commands) == 1 and "-hwaccel" in commands[0]
    assert reader.snapshot()["error"] == "DECODE_HARDWARE_INIT_FAILED"
    assert reader.snapshot()["decoder_warning"] is None
    assert reader.snapshot()["decoder_mode"] is None


@test
def test_failed_software_retry_does_not_hide_the_stream_failure():
    reader = make_reader(hwaccel="auto")
    decode_attempt(reader, [Process(error=b"Cannot dynamically load CUDA\n"),
                            Process(error=b"Invalid data found when processing input\n")])
    assert reader.snapshot()["error"] == "STREAM_UNAVAILABLE"
    assert reader.snapshot()["decoder_warning"] == "DECODE_SOFTWARE_FALLBACK"
    assert reader.snapshot()["decoder_mode"] is None


@test
def test_containment_failure_never_launches_a_decoder_or_attempts_fallback():
    reader = make_reader()
    launches = []
    with patch.object(reader, "_probe", return_value=(2, 2, "hevc", 2.)), \
            patch.object(MediaProcessGuard, "require", side_effect=HTTPException(503, "MEDIA_PROCESS_CONTAINMENT_UNAVAILABLE")), \
            patch("app.rtsp.reader.subprocess.Popen", side_effect=lambda *a, **kw: launches.append(a)):
        try:
            reader._connect_and_read()
        except HTTPException as exc:
            assert exc.detail == "MEDIA_PROCESS_CONTAINMENT_UNAVAILABLE"
        else:
            raise AssertionError("Containment is mandatory")
    assert not launches and reader.read() is None


@test
def test_production_drop_filter_executes_without_synthesizing_source_frames():
    ffmpeg, _ = resolve_ffmpeg()
    assert ffmpeg, "FFmpeg is required for the generated timestamp fixture proof"
    for rate, target, timestamp_filter, expected in (
        (2, 15, None, 4),
        (30, 5, None, 10),
        (2, 15, "setpts=if(eq(N\\,0)\\,0\\,if(eq(N\\,1)\\,5/TB\\,if(eq(N\\,2)\\,1/TB\\,2/TB)))", 4),
        (2, 15, "setpts=NAN", 4),
    ):
        commands = decode_attempt(make_reader(hwaccel="none", target_fps=target), [Process()])
        command = commands[0]
        filter_ = command[command.index("-vf") + 1]
        if timestamp_filter:
            filter_ = timestamp_filter + "," + filter_
        MediaProcessGuard.require()
        result = subprocess.run([
            ffmpeg, "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
            f"testsrc2=size=32x24:rate={rate}:duration=2", "-vf", filter_,
            "-fps_mode", command[command.index("-fps_mode") + 1],
            "-an", "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1",
        ], capture_output=True, timeout=20, close_fds=True)
        assert result.returncode == 0, result.stderr.decode("utf-8", "replace")
        assert len(result.stdout) == expected * 32 * 24 * 3, (rate, target, timestamp_filter, len(result.stdout))


@test
def test_concurrent_cache_encodes_each_key_once_and_preserves_native_dimensions():
    cache = JPEGCache()
    reader = make_reader()
    packet = publish(reader)
    start = threading.Barrier(8)
    calls = []
    encode = cv2.imencode

    def record(*args, **kwargs):
        calls.append(args[1].shape)
        return encode(*args, **kwargs)

    def request(_):
        start.wait(timeout=5)
        return cache.get(1, packet, "thumbnail")

    with patch("app.rtsp.jpeg_cache.cv2.imencode", side_effect=record), ThreadPoolExecutor(8) as pool:
        results = list(pool.map(request, range(8)))
        preview = cache.get(1, packet, "preview")
        assert len(calls) == 2
    assert all(item is results[0] for item in results)
    assert cv2.imdecode(np.frombuffer(results[0], np.uint8), 1).shape == (270, 480, 3)
    assert cv2.imdecode(np.frombuffer(preview, np.uint8), 1).shape == (720, 1280, 3)
    assert packet.frame.shape == (1080, 1920, 3) and not packet.frame.flags.writeable
    # The old image was evicted rather than retained in a historical FIFO.
    new_packet = publish(reader, frame=np.full((1600, 900, 3), 200, np.uint8))
    new_jpeg = cache.get(1, new_packet, "thumbnail")
    assert cv2.imdecode(np.frombuffer(new_jpeg, np.uint8), 1).shape == (480, 270, 3)
    with patch("app.rtsp.jpeg_cache.cv2.imencode", side_effect=record):
        cache.get(1, packet, "thumbnail")
        cache.get(1, new_packet, "thumbnail")
    assert len(calls) == 3, "An older request must not evict the latest cached derivative"
    cache.invalidate(1)
    with patch("app.rtsp.jpeg_cache.cv2.imencode", side_effect=record):
        cache.get(1, new_packet, "thumbnail")
    assert len(calls) == 4


@test
def test_cache_uses_stream_identity_and_does_not_upscale_small_frames():
    cache = JPEGCache()
    first = make_reader()
    a = publish(first, frame=np.zeros((12, 20, 3), np.uint8))
    old = cache.get(1, a, "preview")
    second = make_reader()
    b = publish(second, stream="b", frame=np.full((12, 20, 3), 255, np.uint8))
    assert a.seq == b.seq
    fresh = cache.get(1, b, "preview")
    image = cv2.imdecode(np.frombuffer(fresh, np.uint8), 1)
    assert image.shape == (12, 20, 3) and image.mean() > 250 and fresh != old
    for invalid in ("native", "481", "full"):
        try:
            cache.get(1, b, invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("The derivative cache cannot retain native/arbitrary sizes")


def source(revision, sequence):
    return CameraSource(id=1, name="Door", nvr_host="unused", rtsp_port=554,
                        username="u", password="p", channel=1, stream_type="main",
                        enabled=True, source_revision=revision, publication_sequence=sequence,
                        created_at=STAMP, updated_at=STAMP)


@test
def test_manager_replacement_stop_and_deletion_invalidate_cached_derivatives():
    made = []

    def construct(**kwargs):
        value = RTSPReader(**kwargs)
        made.append(value)
        return value

    with patch("app.rtsp.manager.resolve_ffmpeg", return_value=("unused", "unused")), \
            patch("app.rtsp.manager.RTSPReader", side_effect=construct), \
            patch.object(RTSPReader, "start"):
        manager = ReaderManager()
        first = manager.start(source(1, 1))
        assert first._target_fps == settings.stream_output_fps
        assert first._hwaccel == settings.stream_hwaccel
        packet = publish(first)
        manager.jpeg_cache.get(1, packet, "thumbnail")
        second = manager.start(source(2, 2))
        assert first.read() is None and manager.get(1) is second
        # An identical token is deliberately reused to make invalidation observable.
        cached = cv2.imencode
        calls = []

        def record(*args, **kwargs):
            calls.append(1)
            return cached(*args, **kwargs)

        with patch("app.rtsp.jpeg_cache.cv2.imencode", side_effect=record):
            manager.jpeg_cache.get(1, packet, "thumbnail")
            manager.stop(CameraDeletion(1, 3))
            manager.jpeg_cache.get(1, packet, "thumbnail")
            manager.stop_all()
            manager.jpeg_cache.get(1, packet, "thumbnail")
        assert len(calls) == 3 and manager.get(1) is None
    assert len(made) == 2


@test
def test_snapshot_native_default_and_only_explicit_bounded_sizes_are_accepted():
    reader = make_reader()
    publish(reader)
    manager = Manager(reader)
    app = FastAPI()
    app.include_router(api.router)
    app.dependency_overrides[api.get_db] = lambda: None
    app.dependency_overrides[api.get_manager] = lambda: manager
    client = Client(app)
    with patch.object(api, "_camera_or_404", return_value=SimpleNamespace(id=1, name="Door", enabled=True)), \
            patch("app.api.cameras.time.monotonic", return_value=100.1):
        for query, shape in (("", (1080, 1920, 3)), ("?size=native", (1080, 1920, 3)),
                             ("?size=thumbnail", (270, 480, 3)), ("?size=preview", (720, 1280, 3))):
            status, headers, content = client.request("GET", "/api/cameras/1/snapshot" + query)
            assert status == 200, content
            assert headers[b"content-type"] == b"image/jpeg"
            assert cv2.imdecode(np.frombuffer(content, np.uint8), 1).shape == shape
        assert client.request("GET", "/api/cameras/1/snapshot?size=480")[0] == 422
        assert client.request("GET", "/api/cameras/1/stream?size=480")[0] == 422


@test
def test_snapshot_discards_replaced_reader_stream_source_or_status_during_encoding():
    encode = cv2.imencode
    for change in ("reader", "stream", "source", "status"):
        for size in ("native", "preview"):
            reader = make_reader()
            publish(reader)
            manager = Manager(reader)

            def replace(*args, **kwargs):
                result = encode(*args, **kwargs)
                if change == "reader":
                    manager.reader = make_reader()
                    publish(manager.reader, stream="b")
                elif change == "stream":
                    publish(reader, stream="b")
                elif change == "source":
                    reader.source_revision += 1
                else:
                    reader._set_status(CameraStatus.RECONNECTING)
                return result

            with patch.object(api, "_camera_or_404", return_value=SimpleNamespace(id=1, name="Door", enabled=True)), \
                    patch("app.api.cameras.time.monotonic", return_value=100.1), \
                    patch("app.api.cameras.cv2.imencode", side_effect=replace):
                try:
                    api.api_camera_snapshot(1, db=None, manager=manager, size=size)
                except HTTPException as exc:
                    assert exc.status_code == 503
                else:
                    raise AssertionError(f"A superseded {change} packet cannot be published")


@test
def test_stream_reacquires_replacement_reader_even_when_sequence_restarts():
    old = make_reader()
    old.buffer._seq = 100
    publish(old, frame=np.zeros((16, 24, 3), np.uint8))
    manager = Manager(old)
    with patch("app.api.cameras.time.monotonic", return_value=100.1), \
            patch("app.api.cameras.time.sleep"):
        stream = api._mjpeg_generator(manager, 1, "Door")
        first = next(stream)
        manager.reader = make_reader()
        publish(manager.reader, stream="new", frame=np.full((16, 24, 3), 255, np.uint8))
        with patch("app.api.cameras.time.monotonic", return_value=101.):
            second = next(stream)
        stream.close()
    assert first != second and b"Content-Type: image/jpeg" in second


@test
def test_stream_defaults_to_preview_and_accepts_explicit_native_without_caching_it():
    reader = make_reader()
    publish(reader)
    manager = Manager(reader)
    with patch("app.api.cameras.time.monotonic", return_value=100.1):
        for size, expected in ((None, (720, 1280, 3)), ("thumbnail", (270, 480, 3)),
                               ("native", (1080, 1920, 3))):
            stream = (api._mjpeg_generator(manager, 1, "Door") if size is None
                      else api._mjpeg_generator(manager, 1, "Door", size))
            part = next(stream)
            stream.close()
            jpeg = part.split(b"\r\n\r\n", 1)[1][:-2]
            assert cv2.imdecode(np.frombuffer(jpeg, np.uint8), 1).shape == expected


@test
def test_invalidating_during_encode_cannot_resurrect_a_retired_cache_slot():
    cache = JPEGCache()
    reader = make_reader()
    old = publish(reader, frame=np.zeros((24, 32, 3), np.uint8))
    entered, release = threading.Event(), threading.Event()
    encode = cv2.imencode
    calls = []

    def blocked(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            entered.set()
            assert release.wait(5), "The in-flight encode was not released"
        return encode(*args, **kwargs)

    with patch("app.rtsp.jpeg_cache.cv2.imencode", side_effect=blocked), ThreadPoolExecutor(1) as pool:
        pending = pool.submit(cache.get, 1, old, "thumbnail")
        assert entered.wait(5)
        try:
            cache.invalidate(1)
            fresh = publish(reader, stream="b", frame=np.full((24, 32, 3), 255, np.uint8))
            current = cache.get(1, fresh, "thumbnail")
        finally:
            release.set()
        obsolete = pending.result(timeout=5)
        assert cache.get(1, fresh, "thumbnail") is current
    assert len(calls) == 2 and current != obsolete


@test
def test_successful_receipt_resets_reconnect_backoff_without_catching_up():
    reader = make_reader(hwaccel="none")
    attempts = iter((False, True, False))
    delays = []

    def attempt():
        reader._attempt_got_frame = next(attempts)
        raise ConnectionLost("STREAM_UNAVAILABLE")

    def wait(delay):
        delays.append(delay)
        return len(delays) == 3

    with patch.object(reader, "_connect_and_read", side_effect=attempt), \
            patch.object(reader._stop_event, "wait", side_effect=wait):
        reader._run()
    assert delays == [1., 1., 2.] and reader.reconnect_count == 3
    assert reader.status() is CameraStatus.OFFLINE


@test
def test_next_publication_releases_the_previous_native_image():
    buffer = FrameBuffer()
    discarded = np.empty((24, 32, 3), np.uint8)
    reference = weakref.ref(discarded)
    buffer.put(discarded, stream_id="a", source_revision=1,
               received_at=STAMP, received_mono=100.)
    del discarded
    buffer.put(np.ones((24, 32, 3), np.uint8), stream_id="a", source_revision=1,
               received_at=STAMP, received_mono=101.)
    assert reference() is None, "Even a two-frame queue retains obsolete native pixels"


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
