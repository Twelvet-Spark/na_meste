"""Private, local CCTV evidence replay; never starts the application server.

CLI: python tools/evaluate_cctv.py --manifest PATH --copies N --duration SECONDS
                                 --report PATH [--device cuda|cpu]

Manifest v1: {version:1, clips:[{path,source_camera_key,offset_seconds,
annotations_path:null|path}], enrollments:[{subject_id,images:[path]}],
identity_labels:[{source_camera_key,track_label,subject_id:null|string}],
scene_calibrations:[]}. Paths are relative to the manifest, not the checkout.
Annotations use the approved normalized-coordinate v1 format. Labels are anonymous
unless an operator supplies verified enrollment-to-video labels; clustering is
never identity ground truth. Nonempty scene_calibrations await the scene contract.

The reusable validation, probe, FFmpegReplay, matching and measurement helpers do
not import app.config or any database/model module. The CLI imports only the new
bundle-backed inference APIs after an isolated settings bootstrap. No legacy model
is loaded. Missing bundles/APIs fail visibly. The direct process_batch benchmark
is NOT full worker/load acceptance: worker scheduling, durable journal/attendance,
body enrichment, CUDA node profiling and concurrent scene assistance must be wired
at their integration checkpoints. Reports enumerate these needs and cannot pass
capacity acceptance until they are exercised, not merely imported.
"""
from __future__ import annotations

import argparse
import bisect
import csv
import io
import json
import math
import os
import re
import shutil
import signal
import stat
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath
from typing import Any, Iterable
from uuid import uuid4

ROOT = Path(__file__).resolve().parent.parent
WARMUP_SECONDS = 30.0
WINDOW_SECONDS = 60.0
ANNOTATION_TOLERANCE_SECONDS = .125
MAX_JSON_BYTES = 16 * 1024 * 1024
INTEGRATION_NEEDS = (
    "AI_WORKER_SCHEDULER_NOT_EXERCISED",
    "DURABLE_HUMAN_JOURNAL_NOT_EXERCISED",
    "ATTENDANCE_QUEUE_GATES_AND_CROSSINGS_NOT_EXERCISED",
    "BATCHED_BODY_LINKING_NOT_EXERCISED",
    "ENROLLMENT_CONCURRENT_LOAD_NOT_EXERCISED",
    "SCENE_ASSISTANCE_CONCURRENT_IMPACT_NOT_EVALUATED",
    "CUDA_HEAVY_NODE_PLACEMENT_NOT_PROFILED",
    "DECODER_CPU_LOAD_NOT_MEASURED",
)


class EvaluationError(ValueError):
    """Static failure code; do not expose inherited settings or secret values."""


def _number(value: Any, code: str, *, minimum: float = 0) -> float:
    try:
        if type(value) not in (int, float) or not math.isfinite(value) or value < minimum:
            raise EvaluationError(code)
        return float(value)
    except OverflowError:
        raise EvaluationError(code) from None


def _label(value: Any, code: str) -> str:
    if (not isinstance(value, str) or not value or value != value.strip()
            or len(value) > 128 or any(ord(char) < 32 or ord(char) == 127 for char in value)):
        raise EvaluationError(code)
    return value


def _object(value: Any, required: set[str], optional: set[str] | None = None) -> dict:
    if (not isinstance(value, dict) or not required <= value.keys()
            or value.keys() - required - (optional or set())):
        raise EvaluationError("DOCUMENT_FIELDS_INVALID")
    return value


def _list(value: Any, code: str) -> list:
    if not isinstance(value, list):
        raise EvaluationError(code)
    return value


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise EvaluationError("JSON_DUPLICATE_KEY")
        result[key] = value
    return result


def read_json(path: Path) -> Any:
    try:
        with path.open("rb") as source:
            data = source.read(MAX_JSON_BYTES + 1)
        if len(data) > MAX_JSON_BYTES:
            raise EvaluationError("JSON_TOO_LARGE")
        return json.loads(data.decode("utf-8-sig"), object_pairs_hook=_unique_object,
                          parse_constant=lambda _: (_ for _ in ()).throw(EvaluationError("JSON_NONFINITE")))
    except (OSError, UnicodeError, json.JSONDecodeError, RecursionError):
        raise EvaluationError("JSON_UNREADABLE_OR_INVALID") from None


def _local_path_text(text: str) -> str:
    """Check lexical locality before any filesystem/drive-target traversal."""
    if (not isinstance(text, str) or not text or text != text.strip()
            or any(ord(char) < 32 for char in text) or text.startswith(("\\", "//"))):
        raise EvaluationError("LOCAL_PATH_REQUIRED")
    windows = PureWindowsPath(text)
    drive_absolute = bool(re.match(r"^[A-Za-z]:[\\/]", text))
    if ":" in text and (not drive_absolute or ":" in text[2:]):
        raise EvaluationError("LOCAL_PATH_REQUIRED")
    if windows.drive and not drive_absolute:
        raise EvaluationError("LOCAL_PATH_REQUIRED")
    if drive_absolute and os.name != "nt":
        raise EvaluationError("WINDOWS_PATH_ON_NONWINDOWS_HOST")
    return text


def _windows_drive_type(root: str) -> int | None:
    if os.name != "nt":
        return None
    import ctypes
    from ctypes import wintypes
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetDriveTypeW.argtypes = [wintypes.LPCWSTR]
    kernel.GetDriveTypeW.restype = wintypes.UINT
    return kernel.GetDriveTypeW(root)


def _check_local_drive(path) -> None:
    drive = PureWindowsPath(str(path)).drive
    if drive:
        if not re.fullmatch(r"[A-Za-z]:", drive) or _windows_drive_type(drive + "\\") in (0, 4):
            raise EvaluationError("LOCAL_DRIVE_REQUIRED")


def _alias_target(path: Path) -> str | None:
    try:
        metadata = os.lstat(path)
    except (FileNotFoundError, NotADirectoryError):
        return None
    if stat.S_ISLNK(metadata.st_mode) or getattr(metadata, "st_file_attributes", 0) & 0x400:
        return os.readlink(path)
    return None


def _preflight_local_aliases(path: Path, active: set[str] | None = None) -> None:
    """Inspect ancestors outward, before resolving a child through any alias."""
    _check_local_drive(path)
    active = set() if active is None else active
    for ancestor in (*reversed(path.parents), path):
        target = _alias_target(ancestor)
        if target is None:
            continue
        # Windows readlink returns this prefix even for a normal local junction.
        # Only a drive-letter substitute is normalized; UNC/device targets fail.
        if target.startswith("\\\\?\\") and re.match(r"^[A-Za-z]:[\\/]", target[4:]):
            target = target[4:]
        target = _local_path_text(target)
        destination = Path(target.replace("\\", "/"))
        destination = destination if destination.is_absolute() else ancestor.parent / destination
        key = os.path.normcase(str(ancestor))
        if key in active or len(active) >= 40:
            raise EvaluationError("LOCAL_ALIAS_CYCLE_OR_LIMIT")
        active.add(key)
        try:
            _preflight_local_aliases(destination, active)
        finally:
            active.remove(key)


def local_path(value: str | Path, base: Path, *, must_exist: bool = True) -> Path:
    """Reject remote syntax, aliases and mapped network drives before file IO."""
    text = _local_path_text(str(value) if isinstance(value, Path) else value)
    path = Path(text.replace("\\", "/"))
    try:
        path = path if path.is_absolute() else base / path
        _preflight_local_aliases(path)
        path = path.resolve()
        _local_path_text(str(path))
        _check_local_drive(path)
        if must_exist and not path.is_file():
            raise EvaluationError("INPUT_FILE_MISSING")
    except (OSError, RuntimeError, ValueError):
        raise EvaluationError("LOCAL_PATH_INVALID_OR_MISSING") from None
    return path


def _bbox(value: Any) -> tuple[float, float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise EvaluationError("ANNOTATION_BOX_INVALID")
    box = tuple(_number(item, "ANNOTATION_BOX_INVALID") for item in value)
    if max(box) > 1 or box[0] >= box[2] or box[1] >= box[3]:
        raise EvaluationError("ANNOTATION_BOX_INVALID")
    return box


@dataclass(frozen=True)
class Clip:
    path: Path
    source_camera_key: str
    offset_seconds: float
    annotations_path: Path | None
    annotations: dict | None


@dataclass(frozen=True)
class Enrollment:
    subject_id: str
    images: tuple[Path, ...]


@dataclass(frozen=True)
class Manifest:
    path: Path
    clips: tuple[Clip, ...]
    enrollments: tuple[Enrollment, ...]
    identity_labels: tuple[dict, ...]


def load_annotations(path: Path) -> dict:
    document = _object(read_json(path), {"version", "frames", "crossings"})
    if type(document["version"]) is not int or document["version"] != 1:
        raise EvaluationError("ANNOTATION_VERSION_INVALID")
    previous = -1.0
    known = set()
    for frame in _list(document["frames"], "ANNOTATION_FRAMES_INVALID"):
        _object(frame, {"time_seconds", "people"})
        stamp = _number(frame["time_seconds"], "ANNOTATION_TIME_INVALID")
        if stamp <= previous:
            raise EvaluationError("ANNOTATION_TIMES_NOT_STRICTLY_INCREASING")
        previous = stamp
        labels = set()
        for person in _list(frame["people"], "ANNOTATION_PEOPLE_INVALID"):
            _object(person, {"track_label", "bbox", "face_bbox", "face_visible", "occluded", "seated"})
            label = _label(person["track_label"], "ANNOTATION_LABEL_INVALID")
            if label in labels:
                raise EvaluationError("ANNOTATION_LABEL_DUPLICATE")
            labels.add(label)
            known.add(label)
            person["bbox"] = _bbox(person["bbox"])
            if person["face_bbox"] is not None:
                person["face_bbox"] = _bbox(person["face_bbox"])
            if any(type(person[key]) is not bool for key in ("face_visible", "occluded", "seated")):
                raise EvaluationError("ANNOTATION_FLAG_INVALID")
            if person["face_bbox"] is not None and not person["face_visible"]:
                raise EvaluationError("ANNOTATION_FACE_VISIBILITY_CONFLICT")
    seen_crossings = set()
    last_crossing = -1.0
    for crossing in _list(document["crossings"], "ANNOTATION_CROSSINGS_INVALID"):
        _object(crossing, {"track_label", "time_seconds", "direction"})
        label = _label(crossing["track_label"], "ANNOTATION_LABEL_INVALID")
        stamp = _number(crossing["time_seconds"], "ANNOTATION_TIME_INVALID")
        if crossing["direction"] not in ("IN", "OUT"):
            raise EvaluationError("ANNOTATION_CROSSING_INVALID")
        key = (label, stamp, crossing["direction"])
        if label not in known or stamp < last_crossing or key in seen_crossings:
            raise EvaluationError("ANNOTATION_CROSSING_INVALID")
        seen_crossings.add(key)
        last_crossing = stamp
    return document


def load_manifest(path: str | Path) -> Manifest:
    path = local_path(path, Path.cwd())
    document = _object(read_json(path), {"clips", "enrollments", "identity_labels", "scene_calibrations"}, {"version"})
    version = document.get("version", 1)
    if type(version) is not int or version != 1:
        raise EvaluationError("MANIFEST_VERSION_INVALID")
    if _list(document["scene_calibrations"], "SCENE_CALIBRATIONS_INVALID"):
        raise EvaluationError("SCENE_CALIBRATION_REPLAY_REQUIRES_SCENE_INTEGRATION")
    clips = []
    tracks: dict[str, set[str]] = {}
    clip_keys = set()
    for item in _list(document["clips"], "CLIPS_INVALID"):
        _object(item, {"path", "source_camera_key", "offset_seconds", "annotations_path"})
        source = _label(item["source_camera_key"], "SOURCE_CAMERA_KEY_INVALID")
        clip_path = local_path(item["path"], path.parent)
        offset = _number(item["offset_seconds"], "CLIP_OFFSET_INVALID")
        key = (clip_path, source, offset)
        if key in clip_keys:
            raise EvaluationError("CLIP_DUPLICATE")
        clip_keys.add(key)
        annotation_path = (None if item["annotations_path"] is None else
                           local_path(item["annotations_path"], path.parent))
        annotations = load_annotations(annotation_path) if annotation_path else None
        clips.append(Clip(clip_path, source, offset, annotation_path, annotations))
        tracks.setdefault(source, set()).update(
            person["track_label"] for frame in (annotations or {}).get("frames", []) for person in frame["people"])
    if not clips:
        raise EvaluationError("CLIPS_EMPTY")
    enrollments = []
    subjects = set()
    for item in _list(document["enrollments"], "ENROLLMENTS_INVALID"):
        _object(item, {"subject_id", "images"})
        subject = _label(item["subject_id"], "SUBJECT_ID_INVALID")
        images = tuple(local_path(image, path.parent) for image in _list(item["images"], "ENROLLMENT_IMAGES_INVALID"))
        if subject in subjects or not images or len(set(images)) != len(images):
            raise EvaluationError("ENROLLMENT_DUPLICATE_OR_EMPTY")
        subjects.add(subject)
        enrollments.append(Enrollment(subject, images))
    labels = []
    seen = set()
    for item in _list(document["identity_labels"], "IDENTITY_LABELS_INVALID"):
        _object(item, {"source_camera_key", "track_label", "subject_id"})
        source = _label(item["source_camera_key"], "SOURCE_CAMERA_KEY_INVALID")
        track = _label(item["track_label"], "ANNOTATION_LABEL_INVALID")
        subject = item["subject_id"]
        if subject is not None:
            _label(subject, "SUBJECT_ID_INVALID")
        key = (source, track)
        if key in seen or track not in tracks.get(source, ()) or (subject is not None and subject not in subjects):
            raise EvaluationError("IDENTITY_LABEL_UNVERIFIABLE_OR_DUPLICATE")
        seen.add(key)
        labels.append(dict(item))
    return Manifest(path, tuple(clips), tuple(enrollments), tuple(labels))


def validate_output_path(path: str | Path, manifest: Manifest) -> Path:
    output = local_path(path, Path.cwd(), must_exist=False)
    protected = (ROOT / "models", ROOT / "data" / "recordings")
    reserved = {ROOT / ".env", ROOT / "data" / "operator-settings.json",
                *(ROOT / "data" / name for name in ("app.db", "app.db-wal", "app.db-shm"))}
    if output in reserved or any(output == parent or output.is_relative_to(parent) for parent in protected):
        raise EvaluationError("REPORT_PATH_PROTECTED")
    inputs = {manifest.path, *(clip.path for clip in manifest.clips),
              *(clip.annotations_path for clip in manifest.clips if clip.annotations_path),
              *(image for enrollment in manifest.enrollments for image in enrollment.images)}
    for source in inputs:
        if output == source or (output.exists() and os.path.samefile(output, source)):
            raise EvaluationError("REPORT_OVERWRITES_INPUT")
    for clip in manifest.clips:
        recording_root = next((parent for parent in clip.path.parents if parent.name.casefold() == "recordings"), clip.path.parent)
        if output == recording_root or output.is_relative_to(recording_root):
            raise EvaluationError("REPORT_INSIDE_SOURCE_RECORDINGS")
    if output.exists() and not output.is_file():
        raise EvaluationError("REPORT_PATH_NOT_FILE")
    return output


def resolve_media_tools() -> tuple[str, str]:
    suffix = ".exe" if os.name == "nt" else ""
    local = ROOT / "bin"
    ffmpeg = str(local / f"ffmpeg{suffix}") if (local / f"ffmpeg{suffix}").is_file() else shutil.which("ffmpeg")
    ffprobe = str(local / f"ffprobe{suffix}") if (local / f"ffprobe{suffix}").is_file() else shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        raise EvaluationError("FFMPEG_OR_FFPROBE_UNAVAILABLE")
    return ffmpeg, ffprobe


def _containment() -> dict:
    if os.name == "nt":
        # This module does not read app settings or create an engine.
        from app.media.process_guard import MediaProcessGuard
        try:
            MediaProcessGuard.require()
        except Exception:
            raise EvaluationError("MEDIA_PROCESS_CONTAINMENT_UNAVAILABLE") from None
        return {"creationflags": subprocess.CREATE_NO_WINDOW}
    return {"start_new_session": True}


@dataclass(frozen=True)
class ClipProbe:
    width: int
    height: int
    fps: float | None
    duration_seconds: float
    codec: str


def probe_clip(path: Path, ffprobe: str) -> ClipProbe:
    path = local_path(path, Path.cwd())
    try:
        result = subprocess.run(
            [ffprobe, "-v", "error", "-protocol_whitelist", "file,pipe", "-select_streams", "v:0",
             "-show_entries", "stream=width,height,codec_name,avg_frame_rate,duration:format=duration",
             "-of", "json", str(path)], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            timeout=20, check=True, **_containment())
        metadata = json.loads(result.stdout)
        stream = metadata["streams"][0]
        width, height = int(stream["width"]), int(stream["height"])
        duration = _number(float(stream.get("duration") or metadata["format"]["duration"]), "CLIP_DURATION_INVALID")
        if width <= 0 or height <= 0 or duration <= 0:
            raise EvaluationError("CLIP_DIMENSIONS_OR_DURATION_INVALID")
        numerator, denominator = stream.get("avg_frame_rate", "0/0").split("/")
        fps = float(numerator) / float(denominator) if float(denominator) else None
        if fps is not None and (not math.isfinite(fps) or fps <= 0):
            fps = None
        return ClipProbe(width, height, fps, duration, str(stream["codec_name"]))
    except (OSError, subprocess.SubprocessError, KeyError, IndexError, TypeError, ValueError):
        raise EvaluationError("CLIP_PROBE_FAILED") from None


def probe_manifest(manifest: Manifest, ffprobe: str) -> dict[Path, ClipProbe]:
    probes = {clip.path: probe_clip(clip.path, ffprobe) for clip in manifest.clips}
    for clip in manifest.clips:
        duration = probes[clip.path].duration_seconds
        if clip.offset_seconds >= duration:
            raise EvaluationError("CLIP_OFFSET_OUTSIDE_DURATION")
        if clip.annotations:
            times = [frame["time_seconds"] for frame in clip.annotations["frames"]]
            times.extend(crossing["time_seconds"] for crossing in clip.annotations["crossings"])
            if any(stamp >= duration for stamp in times):
                raise EvaluationError("ANNOTATION_TIME_OUTSIDE_CLIP")
    return probes


@dataclass(frozen=True)
class ReplaySpec:
    camera_id: int
    clip: Clip
    probe: ClipProbe
    offset_seconds: float
    stream_id: str
    attendance: bool
    target_fps: float


def build_replay_specs(manifest: Manifest, probes: dict[Path, ClipProbe], copies: int) -> list[ReplaySpec]:
    if type(copies) is not int or copies < 1:
        raise EvaluationError("COPIES_INVALID")
    specs = []
    for index in range(copies):
        clip = manifest.clips[index % len(manifest.clips)]
        probe = probes[clip.path]
        available = probe.duration_seconds - clip.offset_seconds
        if available <= 0:
            raise EvaluationError("CLIP_OFFSET_OUTSIDE_DURATION")
        # Independent readers, not concatenated segments; keep at least one source frame.
        span = max(0., available - (1 / probe.fps if probe.fps else .1))
        offset = clip.offset_seconds + ((index * .6180339887498949) % 1) * span
        specs.append(ReplaySpec(index + 1, clip, probe, offset, str(uuid4()), index < 2,
                                10. if index < 2 else 5.))
    return specs


@dataclass(frozen=True, slots=True)
class ReplayFrame:
    camera_id: int
    stream_id: str
    seq: int
    source_time_seconds: float | None
    received_mono: float
    received_at: datetime | None
    frame: Any


class LatestSlot:
    """One immutable native frame; consumers never make a historical-image queue."""
    def __init__(self):
        self._lock = threading.Lock()
        self._value = None
        self._taken = 0
        self.overwritten = 0

    def publish(self, packet: ReplayFrame):
        with self._lock:
            if self._value is not None and self._value.seq > self._taken:
                self.overwritten += 1
            self._value = packet

    def latest(self) -> ReplayFrame | None:
        with self._lock:
            return self._value

    def take_after(self, seq: int) -> ReplayFrame | None:
        with self._lock:
            if self._value is None or self._value.seq <= seq:
                return None
            self._taken = self._value.seq
            return self._value


_SHOWINFO = re.compile(r"\bn:\s*\d+.*?\bpts_time:\s*([^\s]+)")


class FFmpegReplay(threading.Thread):
    """One supervised FFmpeg child per camera, drop-only rawvideo, no resizing.

    showinfo supplies actual selected frame PTS (including VFR); -fps_mode
    passthrough never creates confirmation opportunities. EOF starts a new stream
    generation on repeat, not a concatenation of overlapping recording timestamps.
    """
    def __init__(self, spec: ReplaySpec, ffmpeg: str, *, realtime: bool = True,
                 repeat: bool = True, stall_timeout: float = 10, on_frame=None):
        super().__init__(name=f"replay-{spec.camera_id}", daemon=True)
        self.spec, self.ffmpeg = spec, ffmpeg
        self.realtime, self.repeat, self.stall_timeout = realtime, repeat, stall_timeout
        self.slot = LatestSlot()
        self.finished = threading.Event()
        self._stop_event = threading.Event()
        self._process_lock = threading.Lock()
        self.process = None
        self.error = None
        self.decoded_count = 0
        self.decoded_times: list[float] = []
        self.started_mono = None
        self.last_frame_mono = None
        self.loops = 0
        self.decoder_mode = "SOFTWARE"
        self._on_frame = on_frame

    def check_health(self):
        if self.error:
            raise EvaluationError(self.error)
        last = self.last_frame_mono if self.last_frame_mono is not None else self.started_mono
        if not self.finished.is_set() and last is not None and time.monotonic() - last > self.stall_timeout:
            self._terminate()
            raise EvaluationError("FFMPEG_REPLAY_STALLED")

    def _terminate(self):
        with self._process_lock:
            process = self.process
        if process is not None and process.poll() is None:
            try:
                if os.name == "nt":
                    process.kill()
                else:
                    os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=5)

    def stop(self):
        self._stop_event.set()
        self._terminate()
        if self.ident is not None:
            self.join(timeout=6)
            if self.is_alive():
                raise EvaluationError("FFMPEG_REPLAY_STOP_PENDING")

    def run(self):
        self.started_mono = time.monotonic()
        try:
            while not self._stop_event.is_set():
                self._decode_once()
                self.loops += 1
                if not self.repeat:
                    break
        except Exception as exc:
            if not self._stop_event.is_set():
                self.error = str(exc) if isinstance(exc, EvaluationError) else "FFMPEG_REPLAY_FAILED"
        finally:
            try:
                self._terminate()
            finally:
                self.finished.set()

    def _decode_once(self):
        import numpy as np

        stream_id = self.spec.stream_id if self.loops == 0 else str(uuid4())
        offset = self.spec.offset_seconds if self.loops == 0 else 0.
        command = [self.ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "info", "-threads", "1"]
        if self.realtime:
            command.append("-re")
        command += ["-ss", str(offset), "-noautorotate", "-protocol_whitelist", "file,pipe",
                    "-i", str(self.spec.clip.path), "-map", "0:v:0", "-an", "-sn", "-dn",
                    "-vf", "showinfo", "-fps_mode", "passthrough", "-pix_fmt", "bgr24",
                    "-threads", "1", "-f", "rawvideo", "pipe:1"]
        stamps: deque[float | None] = deque()
        condition = threading.Condition()
        stderr_done = threading.Event()
        stderr_error = []
        with self._process_lock:
            if self._stop_event.is_set():
                return
            self.process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                            bufsize=0, **_containment())
            process = self.process

        def drain_stderr():
            try:
                while line := process.stderr.readline(8192):
                    match = _SHOWINFO.search(line.decode("utf-8", errors="replace"))
                    if match:
                        try:
                            value = float(match[1])
                            value = value if math.isfinite(value) else None
                        except ValueError:
                            value = None
                        with condition:
                            if len(stamps) >= 64:
                                stderr_error.append("FFMPEG_PTS_QUEUE_OVERFLOW")
                                return
                            stamps.append(value)
                            condition.notify()
            finally:
                stderr_done.set()
                with condition:
                    condition.notify_all()

        stderr_thread = threading.Thread(target=drain_stderr, name=f"pts-{self.spec.camera_id}", daemon=True)
        stderr_thread.start()
        frame_bytes = self.spec.probe.width * self.spec.probe.height * 3
        count_before = self.decoded_count
        try:
            while not self._stop_event.is_set():
                data = bytearray(frame_bytes)
                view = memoryview(data)
                filled = 0
                while filled < frame_bytes:
                    size = process.stdout.readinto(view[filled:])
                    if not size:
                        break
                    filled += size
                if not filled:
                    break
                if filled != frame_bytes:
                    if self._stop_event.is_set():
                        break
                    raise EvaluationError("FFMPEG_PARTIAL_NATIVE_FRAME")
                received_mono = time.monotonic()
                received_at = datetime.now(timezone.utc).replace(tzinfo=None)
                with condition:
                    condition.wait_for(lambda: stamps or stderr_done.is_set() or stderr_error, timeout=2)
                    if stderr_error or not stamps:
                        raise EvaluationError(stderr_error[0] if stderr_error else "FFMPEG_FRAME_PTS_MISSING")
                    pts = stamps.popleft()
                frame = np.frombuffer(data, dtype=np.uint8).reshape(self.spec.probe.height, self.spec.probe.width, 3)
                frame.flags.writeable = False
                self.decoded_count += 1
                self.decoded_times.append(received_mono)
                self.last_frame_mono = received_mono
                packet = ReplayFrame(self.spec.camera_id, stream_id, self.decoded_count,
                                     None if pts is None else offset + pts,
                                     received_mono, received_at, frame)
                if self._on_frame is not None:
                    self._on_frame(packet)
                self.slot.publish(packet)
            if self._stop_event.is_set():
                return
            status = process.wait(timeout=5)
            if status != 0:
                raise EvaluationError("FFMPEG_REPLAY_EXIT_FAILED")
            if self.decoded_count == count_before:
                raise EvaluationError("FFMPEG_REPLAY_NO_FRAMES")
        finally:
            self._terminate()
            process.stdout.close()
            stderr_thread.join(timeout=5)
            process.stderr.close()
            if stderr_thread.is_alive():
                raise EvaluationError("FFMPEG_STDERR_STOP_PENDING")


@dataclass(frozen=True)
class Prediction:
    track_id: str
    bbox: tuple[float, float, float, float]
    employee_id: int | None = None


def box_iou(left, right) -> float:
    intersection = max(0., min(left[2], right[2]) - max(left[0], right[0])) * max(0., min(left[3], right[3]) - max(left[1], right[1]))
    union = (left[2] - left[0]) * (left[3] - left[1]) + (right[2] - right[0]) * (right[3] - right[1]) - intersection
    return intersection / union if union > 0 else 0.


def match_people(boxes: list, predictions: list[Prediction], *, iou_threshold: float = .5) -> list[tuple[int, int]]:
    """Maximum-cardinality matching, deterministic IoU-ordered augmenting paths."""
    choices = []
    for box in boxes:
        scores = [(index, box_iou(box, prediction.bbox)) for index, prediction in enumerate(predictions)]
        choices.append([index for index, score in sorted(scores, key=lambda item: (-item[1], item[0])) if score >= iou_threshold])
    owners = {}

    def augment(person, visited):
        for prediction in choices[person]:
            if prediction in visited:
                continue
            visited.add(prediction)
            if prediction not in owners or augment(owners[prediction], visited):
                owners[prediction] = person
                return True
        return False

    # Constrained people go first; an earlier identical prediction wins ties.
    for person in sorted(range(len(boxes)), key=lambda index: (len(choices[index]), index)):
        augment(person, set())
    return sorted((person, prediction) for prediction, person in owners.items())


class _QualityTotals:
    def __init__(self):
        self.annotated = self.matched = self.predicted = self.false_positives = self.switches = self.frames = 0
        self.person_seconds = 0.
        self._bindings = {}
        self.subsets = {name: {"annotated": 0, "matched": 0} for name in ("seated", "occluded", "distant")}

    def add_frame(self, camera_key: str, stream_id: str, time_seconds: float, people: list,
                  predictions: list[Prediction], *, visible_seconds: float = 0.):
        matches = match_people([person["bbox"] for person in people], predictions)
        matched_people = dict(matches)
        self.frames += 1
        self.annotated += len(people)
        self.matched += len(matches)
        self.predicted += len(predictions)
        self.false_positives += len(predictions) - len(matches)
        self.person_seconds += len(people) * visible_seconds
        for index, person in enumerate(people):
            for name, applies in (("seated", person["seated"]), ("occluded", person["occluded"]),
                                  ("distant", person["bbox"][3] - person["bbox"][1] < .15)):
                if applies:
                    self.subsets[name]["annotated"] += 1
                    self.subsets[name]["matched"] += int(index in matched_people)
            if index in matched_people:
                key = (camera_key, stream_id, person["track_label"])
                track = predictions[matched_people[index]].track_id
                previous = self._bindings.get(key)
                if previous is not None and previous != track:
                    self.switches += 1
                self._bindings[key] = track

    def summary(self) -> dict:
        recall = self.matched / self.annotated if self.annotated else None
        fp = self.false_positives / self.predicted if self.predicted else None
        switches = self.switches / (self.person_seconds / 60) if self.person_seconds else None
        failures = []
        if recall is not None and recall < .9:
            failures.append("PERSON_RECALL_BELOW_0.90")
        if fp is not None and fp > .05:
            failures.append("FALSE_POSITIVE_FRACTION_ABOVE_0.05")
        if switches is not None and switches > 2:
            failures.append("IDENTITY_SWITCHES_ABOVE_2_PER_VISIBLE_PERSON_MINUTE")
        return {"status": "EVALUATED" if self.frames and self.annotated else "NOT_EVALUATED",
                "frames": self.frames, "annotated_people": self.annotated, "matched_people": self.matched,
                "predictions": self.predicted, "false_positives": self.false_positives,
                "recall": recall, "false_positive_fraction": fp, "identity_switches": self.switches,
                "visible_person_seconds": self.person_seconds, "switches_per_visible_person_minute": switches,
                "matching": "MAX_CARDINALITY_IOU_0.5_IOU_ORDERED_AUGMENTATION",
                "distant_definition": "normalized_person_box_height_below_0.15_not_measured_distance",
                "visibility_duration_method": "forward_hold_at_most_1_second_clipped_to_played_interval",
                "subsets": {name: {**counts, "recall": counts["matched"] / counts["annotated"] if counts["annotated"] else None}
                            for name, counts in self.subsets.items()},
                "detection_tracking_acceptance": {
                    "status": "FAIL" if failures else "NOT_EVALUATED" if any(value is None for value in (recall, fp, switches)) else "PASS",
                    "failures": failures, "durable_visit_coverage": "NOT_EVALUATED"},
                "recognition": {"status": "NOT_EVALUATED", "reason": "NO_VERIFIED_ELIGIBLE_KNOWN_VISIT_DENOMINATOR",
                                "eligible_known_visits": None, "recall": None, "wrong_confirmations": None}}


class QualityEvaluator:
    """Fixed source-1Hz acceptance; selected transition annotations are diagnostic."""
    def __init__(self):
        self._frames = {}

    def add_frame(self, camera_key: str, stream_id: str, time_seconds: float, people: list,
                  predictions: list[Prediction], *, visible_seconds: float = 0.):
        stamp = _number(time_seconds, "ANNOTATION_TIME_INVALID")
        visible = _number(visible_seconds, "ANNOTATION_VISIBILITY_DURATION_INVALID")
        self._frames[(camera_key, stream_id, stamp)] = (people, predictions, visible)

    def summary(self) -> dict:
        regular, transitions = [], []
        seconds = set()
        for key, values in sorted(self._frames.items()):
            camera, stream, stamp = key
            second = (camera, stream, math.floor(stamp))
            cohort = transitions if second in seconds else regular
            cohort.append((key, values))
            seconds.add(second)

        def summarize(records):
            totals = _QualityTotals()
            for index, (key, values) in enumerate(records):
                camera, stream, stamp = key
                people, predictions, visible = values
                visible = min(1., visible)
                if index + 1 < len(records):
                    following = records[index + 1][0]
                    if following[:2] == key[:2]:
                        visible = min(visible, following[2] - stamp)
                totals.add_frame(camera, stream, stamp, people, predictions, visible_seconds=visible)
            return totals.summary()

        result = summarize(regular)
        result["sampling"] = {
            "method": "EARLIEST_ANNOTATED_TIMESTAMP_PER_SOURCE_INTEGER_SECOND",
            "interval_seconds": 1, "transition_frames_excluded_from_acceptance": len(transitions)}
        result["transitions"] = summarize(transitions)
        result["transitions"].pop("detection_tracking_acceptance")
        result["transitions"]["scope"] = "DIAGNOSTIC_ONLY_EXCLUDED_FROM_ACCEPTANCE"
        return result


def percentile(values: Iterable[float], fraction: float) -> float | None:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)] if ordered else None


@dataclass(frozen=True)
class AnalysisSample:
    received_mono: float
    started_mono: float
    completed_mono: float


@dataclass
class CameraMeasurements:
    camera_id: int
    target_fps: float
    attendance: bool
    decoded_times: list[float] = field(default_factory=list)
    display_times: list[float] = field(default_factory=list)
    analyses: list[AnalysisSample] = field(default_factory=list)

    def record_analysis(self, sample: AnalysisSample):
        self.analyses.append(sample)

    def summary(self, started_mono: float, ended_mono: float) -> dict:
        lower = started_mono + WARMUP_SECONDS
        duration = max(0., ended_mono - lower)
        analyses = [sample for sample in self.analyses if lower <= sample.started_mono < ended_mono and sample.completed_mono <= ended_mono]
        completions = [sample.completed_mono for sample in analyses]
        def fps(values):
            return sum(lower <= stamp < ended_mono for stamp in values) / duration if duration else None

        windows = []
        if duration >= WINDOW_SECONDS:
            edges = [lower + WINDOW_SECONDS + index for index in range(int(duration - WINDOW_SECONDS) + 1)]
            if not edges or edges[-1] < ended_mono:
                edges.append(ended_mono)
            for end in edges:
                first, last = bisect.bisect_left(completions, end - WINDOW_SECONDS), bisect.bisect_left(completions, end)
                samples = analyses[first:last]
                windows.append((len(samples) / WINDOW_SECONDS,
                                percentile((sample.completed_mono - sample.received_mono for sample in samples), .95)))
        return {"camera_id": self.camera_id, "target_fps": self.target_fps, "attendance": self.attendance,
                "measurement_seconds": duration, "input_decoded_fps": fps(self.decoded_times),
                "analyzed_fps": len(analyses) / duration if duration else None,
                "display_fps": fps(self.display_times),
                "display_definition": "actual_JPEG_encodes_for_one_preview_not_browser_paints",
                "p95_queue_age_seconds": percentile((sample.started_mono - sample.received_mono for sample in analyses), .95),
                "p95_completion_age_seconds": percentile((sample.completed_mono - sample.received_mono for sample in analyses), .95),
                "worst_rolling_60s": {"analyzed_fps": min((window[0] for window in windows), default=None),
                                      "p95_completion_age_seconds": max((window[1] for window in windows if window[1] is not None), default=None),
                                      "empty_window": any(window[0] == 0 for window in windows),
                                      "stride_seconds": 1, "windows": len(windows)},
                "processed_frames": len(analyses),
                "inference_batch_seconds_p95": percentile((sample.completed_mono - sample.started_mono for sample in analyses), .95),
                "oldest_pending_age_seconds": None,
                "deferred_face_tasks": None, "deferred_reid_tasks": None,
                "stage_times": {"person": None, "face": None, "appearance": None, "journal": None}}


def acceptance_gates(cameras: list[dict], *, duration_seconds: float, copies: int,
                     gpu_inventory: list[dict], integration_needs: list[str]) -> dict:
    failures = []
    if len(cameras) != copies:
        failures.append("CAMERA_RESULTS_MISSING")
    for camera in cameras:
        floor = 8. if camera["attendance"] else 4.5
        window = camera["worst_rolling_60s"]
        if camera["analyzed_fps"] is None or camera["analyzed_fps"] < floor:
            failures.append(f"CAMERA_{camera['camera_id']}_FPS")
        if camera["p95_completion_age_seconds"] is None or camera["p95_completion_age_seconds"] > 1:
            failures.append(f"CAMERA_{camera['camera_id']}_AGE")
        if window["analyzed_fps"] is None or window["analyzed_fps"] < floor:
            failures.append(f"CAMERA_{camera['camera_id']}_WORST_WINDOW_FPS")
        if window["p95_completion_age_seconds"] is None or window["p95_completion_age_seconds"] > 1 or window.get("empty_window"):
            failures.append(f"CAMERA_{camera['camera_id']}_WORST_WINDOW_AGE")
    target = any("5060 ti" in gpu.get("name", "").lower() and gpu.get("memory_total_mib", 0) >= 15_360 for gpu in gpu_inventory)
    prerequisites = list(integration_needs)
    if duration_seconds < 900:
        prerequisites.append("FULL_900_SECOND_RUN_NOT_COMPLETED")
    return {"capacity": {"status": "NOT_EVALUATED" if prerequisites else "FAIL" if failures else "PASS",
                         "failures": failures, "prerequisites": prerequisites,
                         "minimum_camera_target_met": not prerequisites and not failures and copies >= 10,
                         "supported_camera_count": copies if not prerequisites and not failures else None,
                         "required_total_seconds": 900, "warmup_seconds": WARMUP_SECONDS},
            "target_hardware": {"status": "TARGET_HARDWARE_PRESENT" if target else "NOT_TARGET_HARDWARE",
                                "certified": target and not prerequisites and not failures and copies >= 10},
            "recognition": {"status": "NOT_EVALUATED", "reason": "VERIFIED_LABELS_AND_INDEPENDENT_QUALITY_ELIGIBILITY_REQUIRED"}}


@dataclass(frozen=True)
class IsolatedPaths:
    root: Path
    database_path: Path
    media_root: Path


@contextmanager
def isolated_environment(root: Path, device: str = "cpu"):
    """CLI-only isolation; app.config's real offline bootstrap must honor the marker."""
    if "app.config" in sys.modules or "app.database.database" in sys.modules:
        raise EvaluationError("APPLICATION_ALREADY_IMPORTED_ISOLATION_TOO_LATE")
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    media = root / "media"
    media.mkdir()
    database = root / "evaluation.db"
    sqlite3.connect(database).close()
    previous = dict(os.environ)
    # Child tools need OS/DLL discovery, never application credentials/settings.
    allow = {"PATH", "PATHEXT", "SYSTEMROOT", "WINDIR", "COMSPEC", "TEMP", "TMP", "TMPDIR",
             "HOME", "USERPROFILE", "LOCALAPPDATA", "APPDATA", "LD_LIBRARY_PATH", "CUDA_PATH", "CUDA_HOME"}
    safe = {key: value for key, value in previous.items() if key.upper() in allow}
    safe.update(NA_MESTE_EVALUATION_ROOT=str(root), NA_MESTE_EVALUATION_DEVICE=device,
                DATABASE_URL=f"sqlite:///{database.as_posix()}", TELEGRAM_ENABLED="false",
                OPERATOR_SETTINGS_PATH=str(root / "operator-settings.json"))
    os.environ.clear()
    os.environ.update(safe)
    try:
        yield IsolatedPaths(root, database, media)
    finally:
        os.environ.clear()
        os.environ.update(previous)


class ApprovedInference:
    """Real new-model sessions; missing interfaces never fall back to old models."""
    def __init__(self, manifest: Manifest, isolated: IsolatedPaths, device: str):
        self.engine = None
        bundle_path = ROOT / "models" / "store-v1" / "manifest.json"
        if not bundle_path.is_file():
            raise EvaluationError("MODEL_BUNDLE_UNAVAILABLE_RUN_PREPARE_MODELS_FIRST")
        try:
            from app.config import settings
            expected = f"sqlite:///{isolated.database_path.as_posix()}"
            if settings.database_url != expected or settings.telegram_enabled:
                raise EvaluationError("OFFLINE_SETTINGS_BOOTSTRAP_NOT_ACTIVE")
            from app.ai.model_bundle import ModelBundle
            from app.ai.detector import PersonDetector
            from app.ai.face_detector import FaceEngine
            from app.ai.face_recognition import EmbeddingStore
            from app.ai.recognition_service import RecognitionInput, RecognitionService
            from app.ai.providers import setup_cuda_dlls
            from app.rtsp.reader import FramePacket
            if not callable(getattr(PersonDetector, "detect_batch", None)) or not callable(getattr(FaceEngine, "embed_aligned", None)):
                raise EvaluationError("APPROVED_INFERENCE_API_NOT_READY")
            if not callable(getattr(RecognitionService, "process_batch", None)):
                raise EvaluationError("BATCH_RECOGNITION_API_NOT_READY")
            self.bundle = ModelBundle.load(bundle_path)
            self.bundle.require("person")
            setup_cuda_dlls()
            import onnxruntime as ort
            if device == "cuda" and "CUDAExecutionProvider" not in ort.get_available_providers():
                raise EvaluationError("CUDA_EXECUTION_PROVIDER_UNAVAILABLE")
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]
            self.detector = PersonDetector(self.bundle, providers, confidence=.1)
            self.face_engine = self.enroll_engine = None
            self.component_errors = []
            face_ready = True
            for component in ("face_detection", "face_recognition"):
                try:
                    self.bundle.require(component)
                except Exception:
                    face_ready = False
                    self.component_errors.append(f"{component.upper()}_UNAVAILABLE")
            if face_ready:
                try:
                    self.face_engine = FaceEngine(self.bundle, providers, det_size=640)
                    self.enroll_engine = FaceEngine(self.bundle, providers, det_size=640)
                except Exception:
                    self.face_engine = self.enroll_engine = None
                    self.component_errors.append("FACE_ENGINE_STARTUP_FAILED")
            if self.face_engine is None and manifest.enrollments:
                raise EvaluationError("FACE_MODEL_UNAVAILABLE_FOR_ENROLLMENTS")
            self.providers = {"person_detection": list(self.detector.session.get_providers()),
                              "live_face": self.face_engine.providers() if self.face_engine else {},
                              "enroll_face": self.enroll_engine.providers() if self.enroll_engine else {}}
            if device == "cuda":
                observed = [self.providers["person_detection"]]
                observed += list(self.providers["live_face"].values()) + list(self.providers["enroll_face"].values())
                if any(not value or value[0] != "CUDAExecutionProvider" for value in observed):
                    raise EvaluationError("REQUESTED_CUDA_SESSION_FELL_BACK")
            from app.database.database import engine, init_db, SessionLocal
            self.engine = engine
            init_db()
            self._enroll(manifest, SessionLocal)
            self.store = EmbeddingStore(self.bundle.face_model_id)
            with SessionLocal() as session:
                self.store.refresh(session)
            self.service = RecognitionService(self.detector, self.face_engine, self.store,
                                              min_confidence=.60, min_confirmations=3, retry_interval=.5)
            self._input, self._packet = RecognitionInput, FramePacket
            self._streams: dict[int, str] = {}
        except EvaluationError:
            self.close()
            raise
        except (ImportError, AttributeError):
            self.close()
            raise EvaluationError("APPROVED_MODEL_OR_BATCH_API_NOT_READY") from None
        except Exception:
            self.close()
            raise EvaluationError("APPROVED_INFERENCE_STARTUP_FAILED") from None

    def _enroll(self, manifest: Manifest, session_factory):
        if not manifest.enrollments:
            return
        import cv2
        import numpy as np
        from app.ai.face_quality import measure_face_quality, quality_reasons
        from app.database.models import Employee, EmployeeFace, FaceModelMetadata, AttendanceRuntime
        prepared = []
        for enrollment in manifest.enrollments:
            samples = []
            if len(enrollment.images) > 15:
                raise EvaluationError("ENROLLMENT_SAMPLE_LIMIT_EXCEEDED")
            for path in enrollment.images:
                image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
                if image is None:
                    raise EvaluationError("ENROLLMENT_IMAGE_INVALID")
                # Single-photo calls keep enrollment batches <=4, never unbounded.
                faces = self.enroll_engine.detect_faces(image)
                if len(faces) != 1 or faces[0].aligned is None:
                    raise EvaluationError("ENROLLMENT_REQUIRES_ONE_ALIGNED_FACE")
                face = faces[0]
                quality = measure_face_quality(face)
                if quality_reasons(quality, min_size=80, det_score=.8, blur_min=20):
                    raise EvaluationError("ENROLLMENT_FACE_QUALITY_INSUFFICIENT")
                vector = np.asarray(self.enroll_engine.embed_aligned([face.aligned])[0], dtype=np.float32)
                if vector.shape != (512,) or not np.isfinite(vector).all() or abs(float(np.linalg.norm(vector)) - 1) > .001:
                    raise EvaluationError("ENROLLMENT_VECTOR_INVALID")
                ok, encoded = cv2.imencode(".jpg", face.aligned)
                if not ok:
                    raise EvaluationError("ENROLLMENT_ENCODING_FAILED")
                samples.append((vector.tobytes(), encoded.tobytes()))
            prepared.append((enrollment.subject_id, samples))
        with session_factory.begin() as session:
            for subject, samples in prepared:
                employee = Employee(name=subject, active=True)
                session.add(employee)
                session.flush()
                for embedding, thumbnail in samples:
                    face = EmployeeFace(employee_id=employee.id, embedding=embedding, thumbnail=thumbnail)
                    session.add(face)
                    session.flush()
                    session.add(FaceModelMetadata(face_id=face.id, model_id=self.bundle.face_model_id,
                                                  alignment_id="arcface-5pt-112-v1"))
            session.get(AttendanceRuntime, 1).gallery_revision += 1

    def process(self, items: list[tuple[ReplaySpec, ReplayFrame]]) -> list:
        for spec, packet in items:
            previous = self._streams.get(spec.camera_id)
            if previous is not None and previous != packet.stream_id:
                self.service.reset_camera(spec.camera_id)
            self._streams[spec.camera_id] = packet.stream_id
        inputs = [self._input(spec.camera_id,
                              self._packet(packet.stream_id, 1, packet.seq, packet.received_at,
                                           packet.received_mono, packet.frame), spec.attendance)
                  for spec, packet in items]
        outcomes = self.service.process_batch(inputs)
        if len(outcomes) != len(inputs) or any(not hasattr(outcome, "observed_tracks") for outcome in outcomes):
            raise EvaluationError("BATCH_OUTCOME_CONTRACT_INVALID")
        return outcomes

    def close(self):
        if self.engine is not None:
            self.engine.dispose()
            self.engine = None


def gpu_inventory() -> list[dict]:
    executable = shutil.which("nvidia-smi")
    if not executable:
        return []
    try:
        result = subprocess.run([executable, "--query-gpu=name,memory.total,memory.used,utilization.gpu",
                                 "--format=csv,noheader,nounits"], check=True, timeout=3,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, **_containment())
        return [{"name": row[0].strip(), "memory_total_mib": int(row[1]),
                 "memory_used_mib": int(row[2]), "utilization_percent": int(row[3])}
                for row in csv.reader(io.StringIO(result.stdout)) if len(row) == 4]
    except (OSError, subprocess.SubprocessError, ValueError):
        return []


def gpu_processes() -> dict:
    executable = shutil.which("nvidia-smi")
    if not executable:
        return {"status": "NOT_EVALUATED", "processes": []}
    try:
        result = subprocess.run([executable, "--query-compute-apps=pid,process_name,used_gpu_memory",
                                 "--format=csv,noheader,nounits"], check=True, timeout=3,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, **_containment())
        processes = [{"pid": int(row[0]), "process_name": Path(row[1].strip()).name,
                      "memory_mib": int(row[2]) if row[2].strip().isdigit() else None}
                     for row in csv.reader(io.StringIO(result.stdout)) if len(row) == 3 and int(row[0]) != os.getpid()]
        return {"status": "EVALUATED", "processes": processes,
                "contaminated": bool(processes), "scope": "NVIDIA_COMPUTE_PROCESSES_DISPLAY_LOAD_MAY_BE_UNLISTED"}
    except (OSError, subprocess.SubprocessError, ValueError):
        return {"status": "NOT_EVALUATED", "processes": []}


def process_peak_memory_bytes() -> int | None:
    if os.name != "nt":
        import resource
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(value if sys.platform == "darwin" else value * 1024)
    import ctypes
    from ctypes import wintypes

    class Counters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
            (name, ctypes.c_size_t) for name in ("PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                                                "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage",
                                                "PagefileUsage", "PeakPagefileUsage")]
    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]
    return int(counters.PeakWorkingSetSize) if psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb) else None


class FaceObservations:
    """Measured new-stack evidence, without retaining biometric vectors/images."""
    def __init__(self):
        self.evidence_count = 0
        self.without_fresh_evidence = 0
        self.reason_counts: dict[str, int] = {}
        self.native_face_sizes = []
        self.blur_scores = []
        self.person_widths = []
        self.person_heights = []

    def observe(self, outcome, frame_shape):
        height, width = frame_shape[:2]
        for track in outcome.observed_tracks:
            box = _bbox(list(track.bbox))
            self.person_widths.append((box[2] - box[0]) * width)
            self.person_heights.append((box[3] - box[1]) * height)
        for detection in outcome.detections:
            evidence = detection.fresh_face_evidence
            if evidence is None:
                self.without_fresh_evidence += 1
                continue
            self.evidence_count += 1
            for reason in evidence.reason_codes:
                self.reason_counts[reason] = self.reason_counts.get(reason, 0) + 1
            self.native_face_sizes.append(_number(evidence.quality.min_size, "FACE_QUALITY_CONTRACT_INVALID"))
            if evidence.quality.blur is not None:
                self.blur_scores.append(_number(evidence.quality.blur, "FACE_QUALITY_CONTRACT_INVALID"))

    def summary(self):
        def distribution(values):
            return {"samples": len(values), "p05": percentile(values, .05),
                    "p50": percentile(values, .5), "p95": percentile(values, .95)}
        return {"status": "MEASURED" if self.evidence_count else "NO_FRESH_FACE_EVIDENCE",
                "fresh_evidence_observations": self.evidence_count,
                "display_observations_without_fresh_evidence": self.without_fresh_evidence,
                "reason_counts": self.reason_counts,
                "native_face_min_dimension_px": distribution(self.native_face_sizes),
                "aligned_face_blur": distribution(self.blur_scores),
                "native_person_box_width_px": distribution(self.person_widths),
                "native_person_box_height_px": distribution(self.person_heights),
                "scope": "APPROVED_STACK_OBSERVATIONS_NOT_OLD_NEW_ACCURACY_COMPARISON"}


class AnnotationSamples:
    """Score every decoded annotation opportunity, including never-analyzed frames."""
    def __init__(self, specs: list[ReplaySpec]):
        self.specs = {spec.camera_id: spec for spec in specs}
        self._times = {spec.camera_id: [frame["time_seconds"] for frame in (spec.clip.annotations or {}).get("frames", [])]
                       for spec in specs}
        self._streams = {}
        self._lock = threading.Lock()
        self.unknown_pts = 0

    def observe_decoded(self, packet: ReplayFrame):
        with self._lock:
            if packet.source_time_seconds is None:
                self.unknown_pts += 1
                return
            stamp = packet.source_time_seconds
            state = self._streams.setdefault((packet.camera_id, packet.stream_id),
                                             {"first": stamp, "last": stamp, "samples": {}})
            state["last"] = max(state["last"], stamp)

    def observe(self, packet: ReplayFrame, predictions: list[Prediction]):
        if packet.source_time_seconds is None:
            return
        with self._lock:
            state = self._streams[(packet.camera_id, packet.stream_id)]
            stamp = packet.source_time_seconds
            times = self._times[packet.camera_id]
            insertion = bisect.bisect_left(times, stamp)
            for index in (insertion - 1, insertion):
                if 0 <= index < len(times):
                    delta = abs(times[index] - stamp)
                    old = state["samples"].get(index)
                    if delta <= ANNOTATION_TOLERANCE_SECONDS and (old is None or delta < old[0]):
                        state["samples"][index] = (delta, predictions)

    def evaluate(self) -> dict:
        # Called after all decoder threads are joined; metadata only, no images.
        evaluator = QualityEvaluator()
        missing = transition_missing = 0
        for (camera, stream), state in self._streams.items():
            spec = self.specs[camera]
            frames = (spec.clip.annotations or {}).get("frames", [])
            seconds = set()
            for index, frame in enumerate(frames):
                stamp = frame["time_seconds"]
                if not state["first"] <= stamp <= state["last"]:
                    continue
                sample = state["samples"].get(index)
                second = math.floor(stamp)
                if second in seconds:
                    transition_missing += int(sample is None)
                else:
                    missing += int(sample is None)
                seconds.add(second)
                # Coverage bound is independent of inserted transition annotations;
                # each scoring cohort clips it to its own following sample.
                visible = max(0., min(stamp + 1, state["last"]) - stamp)
                evaluator.add_frame(str(camera), stream, stamp, frame["people"], sample[1] if sample else [], visible_seconds=visible)
        result = evaluator.summary()
        result["annotation_opportunities_without_nearby_inference"] = missing
        result["transitions"]["annotation_opportunities_without_nearby_inference"] = transition_missing
        result["unknown_source_timestamp_observations"] = self.unknown_pts
        result["timestamp_tolerance_seconds"] = ANNOTATION_TOLERANCE_SECONDS
        result["scope"] = "ANONYMOUS_TRACKING_NOT_EMPLOYEE_RECOGNITION"
        return result


def encode_preview(frame) -> bytes:
    import cv2
    height, width = frame.shape[:2]
    scale = min(1., 480 / max(height, width))
    preview = cv2.resize(frame, (max(1, round(width * scale)), max(1, round(height * scale))), interpolation=cv2.INTER_AREA) if scale < 1 else frame
    ok, jpeg = cv2.imencode(".jpg", preview, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ok:
        raise EvaluationError("PREVIEW_ENCODING_FAILED")
    return jpeg.tobytes()


def evaluate(manifest: Manifest, *, copies: int, duration: float, device: str,
             ffmpeg: str, probes: dict[Path, ClipProbe], isolated: IsolatedPaths) -> dict:
    specs = build_replay_specs(manifest, probes, copies)
    inference = ApprovedInference(manifest, isolated, device)
    annotations = AnnotationSamples(specs)
    face_observations = FaceObservations()
    replays = [FFmpegReplay(spec, ffmpeg, on_frame=annotations.observe_decoded) for spec in specs]
    metrics = {spec.camera_id: CameraMeasurements(spec.camera_id, spec.target_fps, spec.attendance) for spec in specs}
    errors = []
    inventory = gpu_inventory()
    other_processes = gpu_processes()
    peaks = {gpu["name"]: gpu["memory_used_mib"] for gpu in inventory}
    started = time.monotonic()
    cpu_started = time.process_time()
    cpu_ended = cpu_started
    ended = started
    next_due = {spec.camera_id: started for spec in specs}
    consumed = {spec.camera_id: 0 for spec in specs}
    preview_seq = 0
    preview = None
    last_resource_sample = started
    try:
        for replay in replays:
            replay.start()
        while time.monotonic() - started < duration:
            now = time.monotonic()
            for replay in replays:
                replay.check_health()
            # Earliest-due distinct cameras, at most four, refresh slots at admission.
            due = sorted((replay for replay in replays if next_due[replay.spec.camera_id] <= now),
                         key=lambda replay: (next_due[replay.spec.camera_id], replay.spec.camera_id))
            items = []
            for replay in due:
                packet = replay.slot.take_after(consumed[replay.spec.camera_id])
                if packet is not None:
                    items.append((replay.spec, packet))
                if len(items) == 4:
                    break
            if items:
                batch_started = time.monotonic()
                outcomes = inference.process(items)
                batch_completed = time.monotonic()
                for (spec, packet), outcome in zip(items, outcomes):
                    consumed[spec.camera_id] = packet.seq
                    metrics[spec.camera_id].record_analysis(AnalysisSample(packet.received_mono, batch_started, batch_completed))
                    period = 1 / spec.target_fps
                    next_due[spec.camera_id] += max(1, math.floor((batch_completed - next_due[spec.camera_id]) / period) + 1) * period
                    predictions = [Prediction(f"{outcome.tracking_generation}:{track.track_id}", _bbox(list(track.bbox)), track.employee_id)
                                   for track in outcome.observed_tracks]
                    annotations.observe(packet, predictions)
                    face_observations.observe(outcome, packet.frame.shape)
            else:
                time.sleep(.002)
            packet = replays[0].slot.latest()
            if packet is not None and packet.seq != preview_seq:
                # One active 480px preview; keep only its latest encoded JPEG.
                preview = encode_preview(packet.frame)
                preview_seq = packet.seq
                metrics[1].display_times.append(time.monotonic())
            if now - last_resource_sample >= 1:
                for gpu in gpu_inventory():
                    peaks[gpu["name"]] = max(peaks.get(gpu["name"], 0), gpu["memory_used_mib"])
                last_resource_sample = now
        ended = time.monotonic()
    except Exception as exc:
        ended = time.monotonic()
        errors.append(str(exc) if isinstance(exc, EvaluationError) else "INFERENCE_RUNTIME_FAILED")
        errors.append(f"FAILURE_TYPE_{type(exc).__name__}")
    finally:
        cpu_ended = time.process_time()
        for replay in replays:
            try:
                replay.stop()
            except EvaluationError as exc:
                errors.append(str(exc))
        inference.close()
    camera_reports = []
    for replay in replays:
        measurements = metrics[replay.spec.camera_id]
        measurements.decoded_times = replay.decoded_times
        report = measurements.summary(started, ended)
        latest = replay.slot.latest()
        report.update(source_camera_key=replay.spec.clip.source_camera_key,
                      stream_id=replay.spec.stream_id, stream_generations=replay.loops,
                      source_width=replay.spec.probe.width, source_height=replay.spec.probe.height,
                      source_fps=replay.spec.probe.fps, staggered_offset_seconds=replay.spec.offset_seconds,
                      decoded_frames=replay.decoded_count, latest_only_overwrites=replay.slot.overwritten,
                      decoder_mode=replay.decoder_mode,
                      oldest_pending_age_seconds=(max(0., ended - latest.received_mono)
                                                  if latest and latest.seq > consumed[replay.spec.camera_id] else 0.))
        camera_reports.append(report)
    integration_needs = list(INTEGRATION_NEEDS) + inference.component_errors
    if errors:
        integration_needs.append("RUN_ERRORS_OR_SHUTDOWN_INCOMPLETE")
    elapsed = ended - started
    quality = annotations.evaluate()
    quality["verified_identity_labels_supplied"] = len(manifest.identity_labels)
    if manifest.identity_labels:
        quality["recognition"]["reason"] = "INDEPENDENT_POSE_BLUR_ASSOCIATION_ELIGIBILITY_AND_VISIT_SCORING_NOT_INTEGRATED"
    return {"status": "ERROR" if errors else "PARTIAL_PIPELINE_MEASURED",
            "workload": "REPEATED_SINGLE_VIEW" if len({clip.source_camera_key for clip in manifest.clips}) == 1 and copies > 1 else "LOCAL_CLIP_REPLAY",
            "scope": "DIRECT_APPROVED_RECOGNITION_BATCH_NOT_FULL_APPLICATION_ACCEPTANCE",
            "duration_seconds": elapsed, "requested_duration_seconds": duration, "warmup_seconds": WARMUP_SECONDS,
            "batch_size": 4, "cameras": camera_reports, "quality": quality,
            "gpu_inventory": inventory, "other_gpu_processes": other_processes,
            "peak_gpu_device_used_memory_mib": peaks,
            "peak_process_memory_bytes": process_peak_memory_bytes(),
            "process_cpu_percent_one_core_basis": 100 * (cpu_ended - cpu_started) / max(elapsed, .001),
            "resource_scope": "INFERENCE_PROCESS_ONLY_NOT_FFMPEG_CHILD_CPU_OR_MEMORY",
            "model_bundle_id": inference.bundle.bundle_id, "face_model_id": inference.bundle.face_model_id,
            "component_status": inference.bundle.component_status, "actual_session_providers": inference.providers,
            "preview": {"jpeg_encodes": len(metrics[1].display_times), "latest_jpeg_bytes": len(preview) if preview else 0,
                        "long_edge_limit": 480, "consumer": "LOCAL_ENCODER_NO_BROWSER"},
            "face_observations": face_observations.summary(),
            "journal": {"status": "NOT_EVALUATED", "loss_or_degradation": None},
            "errors": errors, "integration_needs": integration_needs,
            "acceptance": acceptance_gates(camera_reports, duration_seconds=elapsed, copies=copies,
                                           gpu_inventory=inventory, integration_needs=integration_needs)}


def _positive_duration(text: str) -> float:
    try:
        value = float(text)
        if not math.isfinite(value) or value <= 0:
            raise ValueError
        return value
    except ValueError:
        raise argparse.ArgumentTypeError("duration must be positive and finite") from None


def _positive_copies(text: str) -> int:
    try:
        value = int(text)
        if value < 1:
            raise ValueError
        return value
    except ValueError:
        raise argparse.ArgumentTypeError("copies must be a positive integer") from None


def write_report(path: Path, report: dict, manifest: Manifest):
    # Re-resolve immediately before publication, including changed symlink parents.
    path = validate_output_path(path, manifest)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=".evaluation-", suffix=".json", delete=False) as temporary:
        staged = Path(temporary.name)
        try:
            json.dump(report, temporary, ensure_ascii=False, allow_nan=False, indent=2)
            temporary.write("\n")
        except BaseException:
            temporary.close()
            staged.unlink(missing_ok=True)
            raise
    try:
        validate_output_path(path, manifest)
        os.replace(staged, path)
    finally:
        staged.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--copies", required=True, type=_positive_copies)
    parser.add_argument("--duration", required=True, type=_positive_duration)
    parser.add_argument("--report", required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args(argv)
    manifest = None
    output = None
    report = {"version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
              "recognition": {"status": "NOT_EVALUATED"},
              "baseline": {"old_model_execution": "NOT_RUN_LICENSE_POLICY",
                           "old_new_accuracy_comparison": "NOT_EVALUATED",
                           "reported_observations": ["SEATED_AND_PARTIALLY_OCCLUDED_PEOPLE_MISSED",
                                                     "DISTANT_FACES_MAY_HAVE_INSUFFICIENT_NATIVE_DETAIL"],
                           "source": "APPROVED_PLAN_AND_USER_REPORT_NOT_A_MEASURED_BASELINE"},
              "receipt_age_scope": "SERVER_RECEIPT_TO_COMPLETION_NOT_CAMERA_EXPOSURE_TO_SERVER",
              "integration_needs": list(INTEGRATION_NEEDS)}
    try:
        manifest = load_manifest(args.manifest)
        output = validate_output_path(args.report, manifest)
        ffmpeg, ffprobe = resolve_media_tools()
        # No server/ReaderManager/recording/export/Telegram service is constructed.
        with tempfile.TemporaryDirectory(prefix="na-meste-evaluation-") as directory:
            with isolated_environment(Path(directory), args.device) as isolated:
                report["gpu_inventory"] = gpu_inventory()
                probes = probe_manifest(manifest, ffprobe)
                report["clips"] = [{"path": str(clip.path), "source_camera_key": clip.source_camera_key,
                                     "width": probes[clip.path].width, "height": probes[clip.path].height,
                                     "fps": probes[clip.path].fps, "duration_seconds": probes[clip.path].duration_seconds,
                                     "codec": probes[clip.path].codec} for clip in manifest.clips]
                report.update(evaluate(manifest, copies=args.copies, duration=args.duration, device=args.device,
                                       ffmpeg=ffmpeg, probes=probes, isolated=isolated))
    except (EvaluationError, OSError) as exc:
        report.update(status="ERROR", error_code=str(exc) if isinstance(exc, EvaluationError) else "EVALUATION_IO_FAILED")
        report["acceptance"] = {"capacity": {"status": "NOT_EVALUATED"}, "recognition": {"status": "NOT_EVALUATED"}}
    if output is not None and manifest is not None:
        try:
            write_report(output, report, manifest)
        except (EvaluationError, OSError):
            print("REPORT_PUBLICATION_FAILED", file=sys.stderr)
            return 2
    print(json.dumps({"status": report["status"], "error_code": report.get("error_code"),
                      "report": str(output) if output else None}, ensure_ascii=False))
    return 2 if report["status"] == "ERROR" else 0


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    raise SystemExit(main())
