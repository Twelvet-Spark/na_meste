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
Optional --scene-prompts reads private, explicitly reviewed visible-image prompts.

The CLI bootstraps temporary settings before importing application modules, then
runs the production AIWorker, attendance monitor, journal, appearance linker and
bounded preview cache against independent local FFmpeg readers. Instrumentation
observes real publication; it never bypasses source/gallery/reset guards. Required
stages without real work remain NOT_EVALUATED, including unavailable concurrency.
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
from collections import Counter, OrderedDict, deque
from contextlib import closing, contextmanager
from dataclasses import asdict, dataclass, field
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
    "DERIVED_PREVIEW_NOT_EXERCISED",
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
                 repeat: bool = True, stall_timeout: float = 10, on_frame=None, slot=None):
        super().__init__(name=f"replay-{spec.camera_id}", daemon=True)
        self.spec, self.ffmpeg = spec, ffmpeg
        self.realtime, self.repeat, self.stall_timeout = realtime, repeat, stall_timeout
        self.slot = slot if slot is not None else LatestSlot()
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
        self.decoder_cpu_seconds = 0.
        self.decoder_peak_rss_kib = 0
        self.decoder_resource_samples = 0
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
        command = [self.ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "info", "-benchmark", "-threads", "1"]
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
                    resource = re.search(rb"bench: utime=([\d.]+)s stime=([\d.]+)s", line)
                    if resource:
                        self.decoder_cpu_seconds += float(resource[1]) + float(resource[2])
                        self.decoder_resource_samples += 1
                    memory = re.search(rb"bench: maxrss=(\d+)KiB", line)
                    if memory:
                        self.decoder_peak_rss_kib = max(self.decoder_peak_rss_kib, int(memory[1]))
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
                "display_definition": "production_cache_preview_responses_not_browser_paints",
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


def annotation_face_rejections(person, width, height):
    """Only facts present in the annotation; never invent pose or hidden faces."""
    if not person["face_visible"]:
        return ("FACE_OCCLUDED_OR_NOT_VISIBLE" if person["occluded"] else "FACE_NOT_VISIBLE",)
    box = person["face_bbox"]
    if box is None:
        return ("FACE_BOX_UNANNOTATED",)
    if min((box[2] - box[0]) * width, (box[3] - box[1]) * height) < 48:
        return ("FACE_TOO_SMALL",)
    return ()


def native_annotation_quality(frame, person, people, engine):
    """Identity-free quality on the annotated native person crop, before matching.

    No person prediction, embedding, match score, confirmation or enrollment
    identity is consulted. Alignment is measured, not advertised as a pose test.
    """
    import cv2
    import numpy as np
    height, width = frame.shape[:2]
    reasons = annotation_face_rejections(person, width, height)
    if reasons:
        return reasons
    if engine is None:
        return ("FACE_MODEL_UNAVAILABLE",)
    target = person["face_bbox"]
    if any(other is not person and other["face_bbox"] is not None
           and box_iou(target, other["face_bbox"]) > .1 for other in people):
        return ("AMBIGUOUS_FACE_ASSOCIATION",)
    x1, y1, x2, y2 = person["bbox"]
    left, top = max(0, int(x1 * width)), max(0, int(y1 * height))
    right, bottom = min(width, math.ceil(x2 * width)), min(height, math.ceil(y2 * height))
    faces = engine.detect_faces(frame[top:bottom, left:right])
    if len(faces) != 1:
        return ("NO_NATIVE_FACE" if not faces else "AMBIGUOUS_FACE_ASSOCIATION",)
    face = faces[0]
    box = np.asarray(face.native_bbox)
    if (box.shape != (4,) or not np.isfinite(box).all() or box[2] <= box[0] or box[3] <= box[1]
            or box[0] < 0 or box[1] < 0 or box[2] > right - left or box[3] > bottom - top):
        return ("NATIVE_FACE_ASSOCIATION_UNCERTAIN",)
    native = ((box[0] + left) / width, (box[1] + top) / height,
              (box[2] + left) / width, (box[3] + top) / height)
    if box_iou(native, target) < .5:
        return ("NATIVE_FACE_ASSOCIATION_UNCERTAIN",)
    reasons = []
    if min(box[2] - box[0], box[3] - box[1]) < 48:
        reasons.append("FACE_TOO_SMALL")
    if not math.isfinite(face.det_score) or face.det_score < .8:
        reasons.append("LOW_FACE_DETECTION")
    points = np.asarray(face.landmarks)
    aligned = face.aligned
    if (not isinstance(aligned, np.ndarray) or aligned.dtype != np.uint8 or aligned.shape != (112, 112, 3)
            or points.shape != (5, 2) or not np.isfinite(points).all()
            or np.linalg.matrix_rank(points - points.mean(axis=0)) < 2
            or np.linalg.norm(points[1] - points[0]) < 1
            or (points < 0).any() or (points[:, 0] > right - left).any() or (points[:, 1] > bottom - top).any()):
        reasons.append("BAD_ALIGNMENT")
    else:
        blur = float(cv2.Laplacian(cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
        if not math.isfinite(blur) or blur < 20:
            reasons.append("BLURRY_FACE")
    return tuple(reasons)


class IdentityEvaluator:
    """Explicit operator labels, annotation-only visits, identity-free eligibility."""
    def __init__(self, specs, labels, employee_subjects, face_engine, *,
                 ordinary_interval, attendance_interval):
        self.specs = {spec.camera_id: spec for spec in specs}
        self.labels = {(item["source_camera_key"], item["track_label"]): item["subject_id"] for item in labels}
        self.employee_subjects = dict(employee_subjects)
        self.face_engine = face_engine
        self.intervals = {spec.camera_id: attendance_interval if spec.attendance else ordinary_interval for spec in specs}
        if any(not math.isfinite(value) or value <= 0 for value in self.intervals.values()):
            raise EvaluationError("IDENTITY_EVALUATION_CADENCE_INVALID")
        self.frames = {camera: (spec.clip.annotations or {}).get("frames", []) for camera, spec in self.specs.items()}
        self.times = {camera: [frame["time_seconds"] for frame in frames] for camera, frames in self.frames.items()}
        self.clip_keys = {camera: id(spec.clip) for camera, spec in self.specs.items()}
        self.qualities, self.confirmations = {}, {}
        self.prepared_packets, self.observed_packets = set(), set()
        self.association_rejections = Counter()
        self.unattributed_confirmations = set()
        self.outside_annotation_confirmations = 0
        self.quality_seconds = 0.
        self.prepass = dict(status="NOT_RUN", decoded_frames=0, decoded_clips=0, elapsed_seconds=0.,
                            excluded_from_capacity_timing=True, independent_of_worker_admission=True)

    def _nearest(self, camera, source):
        if source.source_time_seconds is None:
            return None
        times = self.times[camera]
        insertion = bisect.bisect_left(times, source.source_time_seconds)
        candidates = [index for index in (insertion - 1, insertion) if 0 <= index < len(times)]
        if not candidates:
            return None
        index = min(candidates, key=lambda item: (abs(times[item] - source.source_time_seconds), item))
        return index if abs(times[index] - source.source_time_seconds) <= ANNOTATION_TOLERANCE_SECONDS else None

    def capture_quality(self, camera, source):
        """Offline decode callback: retain native quality metadata, never pixels."""
        if not self.labels:
            return
        index = self._nearest(camera, source)
        token = (self.clip_keys[camera], source.stream_id, source.seq)
        if index is None or token in self.prepared_packets:
            return
        self.prepared_packets.add(token)
        if self.prepass["status"] == "NOT_RUN":
            self.prepass["status"] = "CAPTURED_NATIVE_METADATA"
        people = self.frames[camera][index]["people"]
        delta = abs(source.source_time_seconds - self.times[camera][index])
        for person in people:
            if (self.specs[camera].clip.source_camera_key, person["track_label"]) not in self.labels:
                continue
            key = (self.clip_keys[camera], index, person["track_label"])
            old = self.qualities.get(key)
            if old is not None and delta >= old["delta"]:
                continue
            started = time.monotonic()
            try:
                reasons = native_annotation_quality(source.frame, person, people, self.face_engine)
            except Exception as exc:
                if isinstance(exc, MemoryError) or any(text in str(exc).lower()
                        for text in ("out of memory", "cudaerrormemoryallocation")):
                    raise
                reasons = ("NATIVE_QUALITY_UNAVAILABLE",)
            self.quality_seconds += time.monotonic() - started
            self.qualities[key] = dict(delta=delta, time=source.source_time_seconds,
                                       packet_seq=source.seq, reasons=reasons)

    def precompute(self, ffmpeg):
        """Bounded native-frame prepass, before measured replay or matching starts.

        Each distinct authorized annotated clip is decoded independently of the
        live scheduler. Replicas reuse only its measured quality metadata; their
        detector/recognizer/journal work still runs for real during measured replay.
        """
        if not self.labels:
            self.prepass["status"] = "NOT_REQUIRED_NO_LABELS"
            return
        if self.face_engine is None:
            self.prepass["status"] = "FACE_MODEL_UNAVAILABLE"
            return
        self.prepass["status"] = "RUNNING"
        started = time.monotonic()
        seen = set()
        for camera, spec in self.specs.items():
            key = self.clip_keys[camera]
            if key in seen:
                continue
            seen.add(key)
            if not any((spec.clip.source_camera_key, person["track_label"]) in self.labels
                       and not annotation_face_rejections(person, spec.probe.width, spec.probe.height)
                       for frame in self.frames[camera] for person in frame["people"]):
                continue
            native_spec = ReplaySpec(camera, spec.clip, spec.probe, 0.,
                                     f"quality-prepass-{camera}", False, spec.probe.fps)
            reader = FFmpegReplay(native_spec, ffmpeg, realtime=False, repeat=False,
                                  on_frame=lambda packet, owner=camera: self.capture_quality(owner, packet))
            deadline = time.monotonic() + max(60., spec.probe.duration_seconds * 20.)
            try:
                reader.start()
                while reader.is_alive():
                    reader.join(.25)
                    reader.check_health()
                    if time.monotonic() > deadline:
                        raise EvaluationError("IDENTITY_NATIVE_QUALITY_PREPASS_TIMEOUT")
                reader.check_health()
                if not reader.decoded_count:
                    raise EvaluationError("IDENTITY_NATIVE_QUALITY_PREPASS_EMPTY")
                self.prepass["decoded_frames"] += reader.decoded_count
                self.prepass["decoded_clips"] += 1
            except Exception:
                self.prepass["status"] = "FAILED"
                self.prepass["elapsed_seconds"] = time.monotonic() - started
                raise
            finally:
                reader.stop()
        self.prepass["status"] = "COMPLETED"
        self.prepass["elapsed_seconds"] = time.monotonic() - started

    def observe(self, camera, source, outcome):
        """Only worker-published outcomes; candidates/fresh-evidence votes are not confirmations."""
        if not self.labels:
            return
        token = (camera, source.stream_id, source.seq)
        if token in self.observed_packets:
            return
        self.observed_packets.add(token)
        index = self._nearest(camera, source)
        tracks = outcome.observed_tracks
        if index is None:
            self.outside_annotation_confirmations += sum(track.employee_id is not None for track in tracks)
            return
        people = self.frames[camera][index]["people"]
        choices = [[j for j, track in enumerate(tracks) if box_iou(person["bbox"], track.bbox) >= .5]
                   for person in people]
        attributed = set()
        for person, candidates in zip(people, choices):
            if (len(candidates) != 1 or sum(candidates[0] in others for others in choices) != 1):
                self.association_rejections["PERSON_NOT_DETECTED" if not candidates else "AMBIGUOUS_PERSON_ASSOCIATION"] += 1
                continue
            track = tracks[candidates[0]]
            if track.association_uncertain:
                self.association_rejections["PREDICTION_ASSOCIATION_UNCERTAIN"] += 1
                continue
            attributed.add(candidates[0])
            if track.employee_id is None:
                continue
            key = (camera, source.stream_id, index, person["track_label"])
            self.confirmations.setdefault(key, []).append(dict(
                time=source.source_time_seconds, employee_id=track.employee_id,
                subject=self.employee_subjects.get(track.employee_id),
                track_token=(outcome.tracking_generation, track.track_id, track.employee_id)))
        for index, track in enumerate(tracks):
            if index not in attributed and track.employee_id is not None:
                self.unattributed_confirmations.add((*token, outcome.tracking_generation, track.track_id, track.employee_id))

    @staticmethod
    def _eligible(opportunities, cadence):
        times = sorted(set(opportunities))
        for index, first in enumerate(times):
            selected = [first]
            for stamp in times[index + 1:]:
                if stamp - first > 2. + 1e-9:
                    break
                if stamp - selected[-1] + 1e-9 >= cadence:
                    selected.append(stamp)
                    if len(selected) == 3:
                        return True
        return False

    def summary(self, annotations):
        visits = {}
        rejections = Counter()
        occluded = 0
        for (camera, stream), coverage in annotations.coverage().items():
            previous, last_time = {}, None
            for index, frame in enumerate(self.frames[camera]):
                stamp = frame["time_seconds"]
                if not coverage["first"] <= stamp <= coverage["last"]:
                    continue
                current = {}
                for person in frame["people"]:
                    label = person["track_label"]
                    key = previous.get(label) if last_time is not None and stamp - last_time <= 2. else None
                    if key is None:
                        key = (camera, stream, label, stamp)
                        identity_key = (self.specs[camera].clip.source_camera_key, label)
                        visits[key] = dict(camera_id=camera, stream_id=stream, track_label=label, started_at_seconds=stamp,
                            labeled=identity_key in self.labels, subject_id=self.labels.get(identity_key),
                            opportunities=[], unknown_quality=0, confirmations=[], rejection_counts=Counter())
                    current[label] = key
                    visit = visits[key]
                    if not visit["labeled"]:
                        continue
                    sample_key = (camera, stream, index, label)
                    sample = self.qualities.get((self.clip_keys[camera], index, label))
                    reasons = (sample["reasons"] if sample is not None else
                               annotation_face_rejections(person, self.specs[camera].probe.width,
                                                          self.specs[camera].probe.height) or
                               (("FACE_MODEL_UNAVAILABLE",) if self.face_engine is None else ("NO_NATIVE_QUALITY_SAMPLE",)))
                    rejections.update(reasons)
                    visit["rejection_counts"].update(reasons)
                    occluded += int(person["occluded"])
                    if any(reason in reasons for reason in ("NO_NATIVE_QUALITY_SAMPLE", "NATIVE_QUALITY_UNAVAILABLE", "FACE_MODEL_UNAVAILABLE")):
                        visit["unknown_quality"] += 1
                    if not reasons:
                        visit["opportunities"].append(sample["time"])
                    visit["confirmations"].extend(self.confirmations.get(sample_key, ()))
                previous, last_time = current, stamp
        rows = []
        wrong_confirmations = wrong_known = unknown_assigned = eligible = correct_eligible = correct_known = undetermined = 0
        correct_times = []
        ineligibility = Counter()
        for visit in visits.values():
            known = visit["labeled"] and visit["subject_id"] is not None
            qualifies = known and self._eligible(visit["opportunities"], self.intervals[visit["camera_id"]])
            matches = [item for item in visit["confirmations"] if known and item["subject"] == visit["subject_id"]]
            wrong = [item for item in visit["confirmations"] if visit["labeled"]
                     and (not known or item["subject"] != visit["subject_id"])]
            wrong_count = len({item["track_token"] for item in wrong})
            eligible += int(qualifies)
            correct_eligible += int(qualifies and bool(matches))
            correct_known += int(bool(matches))
            wrong_confirmations += wrong_count
            wrong_known += int(known and bool(wrong))
            unknown_assigned += int(visit["labeled"] and not known and bool(wrong))
            undetermined += int(known and not qualifies and visit["unknown_quality"] > 0)
            if known and not qualifies:
                ineligibility["QUALITY_COVERAGE_UNDETERMINED" if visit["unknown_quality"] else
                              "NO_THREE_CADENCED_QUALITY_OPPORTUNITIES_IN_TWO_SECONDS"] += 1
            delay = max(0., min(item["time"] for item in matches) - visit["started_at_seconds"]) if matches else None
            if delay is not None:
                correct_times.append(delay)
            rows.append(dict(camera_id=visit["camera_id"], stream_id=visit["stream_id"], track_label=visit["track_label"],
                             started_at_seconds=visit["started_at_seconds"], labeled=visit["labeled"], subject_id=visit["subject_id"],
                             eligible_known=qualifies, quality_opportunities=len(visit["opportunities"]),
                             correct_confirmed=bool(matches), wrong_confirmations=wrong_count,
                             time_to_first_correct_confirmation_seconds=delay,
                             time_to_first_assignment_seconds=(max(0., min(item["time"] for item in visit["confirmations"])
                                 - visit["started_at_seconds"]) if visit["confirmations"] else None),
                             rejection_counts=dict(visit["rejection_counts"])))
        labeled = sum(row["labeled"] for row in rows)
        known = sum(row["labeled"] and row["subject_id"] is not None for row in rows)
        unlabeled = len(rows) - labeled
        recall = correct_eligible / eligible if eligible else None
        failures = []
        if wrong_confirmations:
            failures.append("WRONG_EMPLOYEE_CONFIRMATION")
        if recall is not None and recall < .9:
            failures.append("ELIGIBLE_KNOWN_RECALL_BELOW_0.90")
        missing = []
        if not self.labels:
            missing.append("NO_VERIFIED_IDENTITY_LABELS")
        if not eligible:
            missing.append("NO_ELIGIBLE_KNOWN_VISITS")
        if unlabeled:
            missing.append("PARTIAL_LABEL_COVERAGE")
        if undetermined:
            missing.append("NATIVE_QUALITY_COVERAGE_INCOMPLETE")
        if self.unattributed_confirmations:
            missing.append("UNATTRIBUTED_CONFIRMED_IDENTITIES")
        status = "FAIL" if failures else "NOT_EVALUATED" if missing else "PASS"
        return dict(status=status, reason=missing[0] if missing else None, failures=failures, prerequisites=missing,
                    certified=status == "PASS" and labeled > known,
                    scope="OPERATOR_LABELED_ANNOTATED_VIDEO_VISITS_ONLY",
                    recall_status="EVALUATED" if eligible else "NOT_EVALUATED", recall=recall,
                    eligible_known_visits=eligible, confirmed_eligible_known_visits=correct_eligible,
                    known_visits=known, ineligible_known_visits=known - eligible - undetermined,
                    not_established_eligible_known_visits=known - eligible, ineligibility_counts=dict(ineligibility),
                    eligibility_undetermined_known_visits=undetermined, unknown_visits=labeled - known,
                    labeled_visits=labeled, unlabeled_visits=unlabeled, correct_known_visits=correct_known,
                    wrong_confirmations=wrong_confirmations if self.labels else None,
                    wrong_known_visits=wrong_known if self.labels else None,
                    unknown_false_assignment_visits=unknown_assigned if labeled > known else None,
                    unknown_false_assignment_status="EVALUATED" if labeled > known else "NOT_EVALUATED",
                    all_visit_identification_coverage=correct_known / labeled if labeled else None,
                    all_visit_identification_coverage_definition="CORRECT_KNOWN_VISITS_OVER_ALL_LABELED_VISITS_UNKNOWN_IS_NEVER_SUCCESS",
                    time_to_first_correct_confirmation_seconds=dict(samples=len(correct_times),
                        p50=percentile(correct_times, .5), p95=percentile(correct_times, .95),
                        origin="FIRST_PLAYED_ANNOTATED_GT_VISIT_OBSERVATION_NOT_TRUE_ARRIVAL"),
                    rejection_counts=dict(rejections), annotated_body_occlusion_opportunities=occluded,
                    prediction_association_rejections=dict(self.association_rejections),
                    unattributed_confirmations=len(self.unattributed_confirmations),
                    confirmed_observations_outside_annotation_tolerance=self.outside_annotation_confirmations,
                    pose_rejections=None, pose_status="NOT_EVALUATED_NO_POSE_LABELS_OR_ESTIMATOR_ALIGNMENT_IS_NOT_POSE",
                    native_quality_seconds=self.quality_seconds, visits=rows,
                    quality_prepass={**self.prepass, "metadata_opportunities": len(self.qualities)},
                    eligibility_policy=dict(native_min_px=48, det_score=.8, blur_min=20, distinct_opportunities=3,
                        window_seconds=2., attempt_intervals=self.intervals,
                        sampling="ONE_CLOSEST_INDEPENDENT_NATIVE_PREPASS_PACKET_PER_ANNOTATION_NO_INTERPOLATION"),
                    association_policy="RECIPROCAL_UNIQUE_PERSON_IOU_GE_0.5_WITHIN_0.125S_NO_UNCERTAIN_TRACKS",
                    visit_policy="CAMERA_STREAM_GT_LABEL_CONTIGUOUS_ANNOTATIONS_SPLIT_ON_ABSENCE_OR_GAP_GT_2S",
                    holdout=dict(validation_source="VIDEO_ONLY_ENROLLMENT_PHOTOS_NEVER_SCORED",
                        photo_time_and_visit_provenance="NOT_PROVIDED_BY_MANIFEST_V1",
                        independent_holdout_verified=False, labels="OPERATOR_SUPPLIED_NOT_INFERRED",
                        repeated_copies="CORRELATED_REPLAYS_NOT_INDEPENDENT_IDENTITY_TRIALS"))


def acceptance_gates(cameras: list[dict], *, duration_seconds: float, copies: int,
                     gpu_inventory: list[dict], integration_needs: list[str],
                     runtime_failures: Iterable[str] = (), recognition: dict | None = None) -> dict:
    failures = list(runtime_failures)
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
            "recognition": recognition if recognition is not None else {
                "status": "NOT_EVALUATED", "reason": "NO_LABELED_RECOGNITION_MEASUREMENT"}}


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
             "HOME", "USERPROFILE", "LOCALAPPDATA", "APPDATA", "LD_LIBRARY_PATH", "CUDA_PATH", "CUDA_HOME",
             "PROGRAMFILES", "PROGRAMW6432"}
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


class ReplayManager:
    """Production reader protocol over independent, local-only replay children."""
    def __init__(self, specs, ffmpeg, annotations):
        from app.rtsp.reader import FrameBuffer, CameraStatus
        from app.rtsp.jpeg_cache import JPEGCache
        self.jpeg_cache = JPEGCache()
        self._metadata = {}
        self._lock = threading.Lock()
        self.measurement_end = float("inf")
        manager = self

        class Buffer(FrameBuffer):
            def __init__(self, camera_id):
                super().__init__()
                self.camera_id = camera_id
                self._stream = None

            def publish(self, packet):
                if packet.received_mono <= manager.measurement_end:
                    annotations.observe_decoded(packet)
                if self._stream != packet.stream_id:
                    manager.jpeg_cache.invalidate(self.camera_id)
                    self._stream = packet.stream_id
                metadata = ReplayFrame(packet.camera_id, packet.stream_id, packet.seq,
                                       packet.source_time_seconds, packet.received_mono,
                                       packet.received_at, None)
                with manager._lock:
                    history = manager._metadata.setdefault(packet.camera_id, OrderedDict())
                    history[(packet.stream_id, packet.seq)] = metadata
                    while len(history) > 256:
                        history.popitem(last=False)
                self.put(packet.frame, stream_id=packet.stream_id, source_revision=1,
                         received_at=packet.received_at, received_mono=packet.received_mono)

        class Reader(FFmpegReplay):
            source_revision = 1

            def __init__(self, spec):
                self.buffer = Buffer(spec.camera_id)
                super().__init__(spec, ffmpeg, slot=self.buffer)

            def status(self):
                return (CameraStatus.ONLINE if self.is_alive() and not self.error
                        and self.buffer.latest() is not None else CameraStatus.OFFLINE)

        self._readers = {spec.camera_id: Reader(spec) for spec in specs}

    def get(self, camera_id):
        return self._readers.get(camera_id)

    def reader_ids(self):
        return list(self._readers)

    def source_packet(self, camera_id, packet):
        with self._lock:
            return self._metadata.get(camera_id, {}).get((packet.stream_id, packet.seq))

    def preview(self, camera_id):
        reader = self.get(camera_id)
        packet = reader.buffer.latest() if reader is not None else None
        if packet is None:
            return None
        image = self.jpeg_cache.get(camera_id, packet, "preview")
        current = self.get(camera_id)
        latest = current.buffer.latest() if current is not None else None
        if current is not reader or latest is None or latest.stream_id != packet.stream_id:
            return None
        return (packet.stream_id, packet.seq), image


class ProductionAccounting:
    """Observe only production-published outcomes, not every attempted inference."""
    def __init__(self, metrics, annotations, manager, face_observations=None):
        self.metrics, self.annotations, self.manager = metrics, annotations, manager
        self.faces = face_observations
        self.pending = {}
        self.expected = {}
        self.admission_counts = Counter()
        self.ack_counts = Counter()
        self.link_keys = set()
        self.reid_crops = 0
        self.batch_calls = 0
        self.source_metadata_missing = 0
        self.errors = Counter()
        self.enrollment = None
        self.identity_evaluator = None
        self.measurement_end = float("inf")
        self.measurement_start = None
        self._growth_checked_bucket = -1
        self._health = deque()
        self._health_lock = threading.Lock()
        self._degraded_start = None
        self.degraded_intervals = []
        self.peak_pending = 0
        self.persistent_growth = False


    def batch(self, inputs, outcomes, started):
        self.batch_calls += 1
        self.pending = {
            (item.camera_id, item.packet.stream_id, item.packet.seq, outcome.tracking_generation):
                (outcome, started)
            for item, outcome in zip(inputs, outcomes)
        }

    def published(self, camera_id, packet, generation, completed):
        measured = self.pending.pop((camera_id, packet.stream_id, packet.seq, generation), None)
        if measured is None:
            return
        outcome, started = measured
        if completed > self.measurement_end:
            return
        self.metrics[camera_id].record_analysis(AnalysisSample(packet.received_mono, started, completed))
        if self.annotations is not None:
            source = self.manager.source_packet(camera_id, packet)
            if source is None:
                self.source_metadata_missing += 1
            else:
                self.annotations.observe(source, [
                    Prediction(f"{generation}:{track.track_id}", _bbox(list(track.bbox)), track.employee_id)
                    for track in outcome.observed_tracks])
                if self.identity_evaluator is not None:
                    self.identity_evaluator.observe(camera_id, source, outcome)
        if self.faces is not None:
            self.faces.observe(outcome, packet.frame.shape)
        if self.enrollment is not None:
            self.enrollment.offer(packet, outcome)

    def admitted(self, camera_id, packet, outcome, admissions):
        for track in outcome.observed_tracks:
            key = (camera_id, outcome.tracking_generation, track.track_id)
            state = self.expected.setdefault(key, dict(
                source_revision=packet.source_revision, stream_id=packet.stream_id,
                event_budget=2, identity=None, lifecycle=None))
            identity = (track.employee_id, track.association_uncertain, tuple(track.identity_reason_codes))
            if identity != state["identity"]:
                state["event_budget"] += 2
                state["identity"] = identity
        for item in outcome.track_lifecycle:
            state = self.expected.get((camera_id, outcome.tracking_generation, item.track_id))
            if state is not None and item.state != state["lifecycle"]:
                state["lifecycle"] = item.state
                state["event_budget"] += 2
        self.admission_counts.update(item.status for item in admissions)

    def health(self, now, snapshot):
        with self._health_lock:
            self._record_health(now, snapshot)

    def _record_health(self, now, snapshot):
        if self._health:
            now = max(now, self._health[-1][0])
        pending = sum(snapshot[key] for key in ("pending_commands", "pending_updates", "pending_barriers"))
        self.peak_pending = max(self.peak_pending, pending)
        if self.measurement_start is None:
            self.measurement_start = now + WARMUP_SECONDS
        if now >= self.measurement_start:
            self._health.append((now, pending))
        while self._health and self._health[0][0] < now - WINDOW_SECONDS:
            self._health.popleft()
        bucket = math.floor((now - self.measurement_start) / 10.)
        if bucket >= 6 and bucket != self._growth_checked_bucket:
            self._growth_checked_bucket = bucket
            # Six completed ten-second buckets, each with >=8 distinct sampled
            # seconds. A one-off step, positive plateau or bounded oscillation
            # is not sustained growth. Do not weight busy sampling bursts more.
            medians = []
            for index in range(bucket - 6, bucket):
                lower = self.measurement_start + index * 10.
                seconds = {math.floor(stamp - lower): value for stamp, value in self._health
                           if lower <= stamp < lower + 10.}
                if len(seconds) < 8:
                    break
                medians.append(percentile(seconds.values(), .5))
            if len(medians) == 6 and medians[0] > 0 and all(
                    right > left for left, right in zip(medians, medians[1:])):
                self.persistent_growth = True
        if snapshot["state"] != "OK" and self._degraded_start is None:
            self._degraded_start = now
        elif snapshot["state"] == "OK" and self._degraded_start is not None:
            self.degraded_intervals.append(dict(started_mono=self._degraded_start, ended_mono=now))
            self._degraded_start = None

    def health_report(self, ended):
        intervals = list(self.degraded_intervals)
        if self._degraded_start is not None:
            intervals.append(dict(started_mono=self._degraded_start, ended_mono=None))
        return dict(degraded_intervals=intervals, peak_pending=self.peak_pending,
                    persistent_growth=self.persistent_growth)

    def appearance_snapshot(self):
        return dict(embedded_crops=self.reid_crops, acknowledged_links=self.ack_counts["LINK_COMMITTED"])


def instrument_pipeline(service, state, journal, reid, accounting):
    """Transparent measurement of real methods; never manufacture their results."""
    process = service.process_batch
    def measured_batch(inputs):
        accounting.pending.clear()
        started = time.monotonic()
        try:
            outcomes = process(inputs)
        except Exception as exc:
            code = "OUT_OF_MEMORY" if isinstance(exc, MemoryError) or any(
                text in str(exc).lower() for text in ("out of memory", "cudaerrormemoryallocation")) else "INFERENCE_FAILED"
            accounting.errors[code] += 1
            raise
        accounting.batch(inputs, outcomes, started)
        return outcomes
    service.process_batch = measured_batch
    update = state.update
    def published(camera_id, detections, *, packet=None, generation=None, configuration=None):
        update(camera_id, detections, packet=packet, generation=generation, configuration=configuration)
        if packet is not None:
            accounting.published(camera_id, packet, generation, time.monotonic())
    state.update = published
    observe = journal.observe
    def admitted(camera_id, packet, outcome):
        admissions = observe(camera_id, packet, outcome)
        accounting.admitted(camera_id, packet, outcome, admissions)
        accounting.health(time.monotonic(), journal.health_snapshot())
        return admissions
    journal.observe = admitted
    drain = journal.drain_acknowledgements
    def acknowledgements():
        result = drain()
        accounting.ack_counts.update(ack.kind for ack in result)
        return result
    journal.drain_acknowledgements = acknowledgements
    submit = journal.submit_link
    def link(proposal):
        accepted = submit(proposal)
        if accepted:
            key = (proposal.camera_id, proposal.tracking_generation, proposal.track_id)
            token = proposal.link_key
            if token not in accounting.link_keys:
                accounting.link_keys.add(token)
                if key in accounting.expected:
                    accounting.expected[key]["event_budget"] += 1
        return accepted
    journal.submit_link = link
    if reid is not None:
        embed = reid.embed
        def measured_embedding(crops):
            try:
                vectors = embed(crops)
            except Exception as exc:
                accounting.errors["OUT_OF_MEMORY" if isinstance(exc, MemoryError) or
                                  "out of memory" in str(exc).lower() else "APPEARANCE_FAILED"] += 1
                raise
            accounting.reid_crops += len(vectors)
            return vectors
        reid.embed = measured_embedding


def journal_integrity(db, expected):
    rows = db.execute("SELECT id,camera_id,tracking_generation,track_id,source_revision,stream_id,status FROM human_visits").fetchall()
    found = Counter((row[1], row[2], row[3]) for row in rows)
    missing = sum(key not in found for key in expected)
    duplicate = sum(max(0, count - 1) for count in found.values())
    mismatch = unbounded = malformed = open_visits = 0
    event_total = 0
    for identifier, camera, generation, track, source, stream, status in rows:
        key = (camera, generation, track)
        wanted = expected.get(key)
        mismatch += int(wanted is None or (source, stream) != (wanted["source_revision"], wanted["stream_id"]))
        open_visits += int(status != "CLOSED")
        events = db.execute("SELECT id,dedupe_key,type,payload FROM human_visit_events WHERE visit_id=?", (identifier,)).fetchall()
        event_total += len(events)
        kinds = Counter(event[2] for event in events)
        sequences = []
        for event in events:
            try:
                sequences.append(json.loads(event[3])["transition_seq"])
            except (ValueError, KeyError, TypeError):
                malformed += 1
        malformed += int(kinds["FIRST_SEEN"] != 1 or kinds["CLOSED"] != 1
                         or len({e[0] for e in events}) != len(events)
                         or len({e[1] for e in events}) != len(events)
                         or len(set(sequences)) != len(sequences))
        unbounded += int(wanted is not None and len(events) > wanted["event_budget"])
    gaps = db.execute("SELECT count(*) FROM human_journal_gaps").fetchone()[0]
    orphan = db.execute("SELECT count(*) FROM human_visit_events e LEFT JOIN human_visits v ON v.id=e.visit_id WHERE v.id IS NULL").fetchone()[0]
    failures = missing + duplicate + mismatch + unbounded + malformed + open_visits + orphan
    return dict(status="FAIL" if failures else "PASS" if expected else "NOT_EVALUATED",
                accepted_track_keys=len(expected), durable_visits=len(rows), missing_visits=missing,
                duplicate_visits=duplicate, token_mismatches=mismatch, unbounded_event_visits=unbounded,
                malformed_event_visits=malformed, open_visits=open_visits, orphan_events=orphan,
                lifecycle_events=event_total, persisted_gap_count=gaps)


def profile_summary(events, providers):
    heavy = [event.get("args", {}) for event in events if event.get("cat") == "Node"
             and event.get("args", {}).get("provider")
             and any(op in event.get("args", {}).get("op_name", "").lower()
                     for op in ("conv", "gemm", "matmul", "attention"))]
    cuda = sum(item["provider"] == "CUDAExecutionProvider" for item in heavy)
    return dict(status=("NOT_EVALUATED" if not heavy or "CUDAExecutionProvider" not in providers
                        else "PASS" if cuda == len(heavy) else "FAIL"),
                heavy_nodes=len(heavy), heavy_cuda_nodes=cuda, providers=list(providers))


class NativeProfiles:
    """Profile the first two real calls per resident session, not a 900s trace."""
    def __init__(self, root):
        self.root, self.records = root, []

    @contextmanager
    def capture(self):
        import onnxruntime as ort
        original = ort.InferenceSession
        def create(path, sess_options=None, *args, **kwargs):
            options = sess_options if sess_options is not None else ort.SessionOptions()
            options.enable_profiling = True
            options.profile_file_prefix = str(self.root / f"ort-{len(self.records)}")
            session = original(path, options, *args, **kwargs)
            record = dict(session=session, model=Path(path).name, calls=0, report=None)
            self.records.append(record)
            run = session.run
            def measured_run(*inputs, **named):
                result = run(*inputs, **named)
                record["calls"] += 1
                if record["calls"] == 2:
                    self._finish(record)
                return result
            session.run = measured_run
            return session
        ort.InferenceSession = create
        try:
            yield
        finally:
            ort.InferenceSession = original

    def _finish(self, record):
        if record["report"] is not None:
            return
        try:
            path = Path(record["session"].end_profiling())
            events = json.loads(path.read_text(encoding="utf-8"))
            record["report"] = profile_summary(events, record["session"].get_providers())
            path.unlink()
        except Exception:
            record["report"] = dict(status="NOT_EVALUATED", error_code="NATIVE_PROFILE_UNAVAILABLE",
                                    providers=list(record["session"].get_providers()))

    def finish(self):
        for record in self.records:
            self._finish(record)

    def summary(self):
        return [dict(model=item["model"], actual_calls=item["calls"], **(
            item["report"] or dict(status="NOT_EVALUATED"))) for item in self.records]


class EnrollmentLoad(threading.Thread):
    """One nonblocking enrollment owner; no employee writes or inferred names."""
    def __init__(self, engine, semaphore, enrollments):
        super().__init__(name="evaluation-enrollment", daemon=True)
        self.engine, self.semaphore = engine, semaphore
        self.images = tuple(path for enrollment in enrollments for path in enrollment.images)
        self._condition = threading.Condition()
        self._crop = None
        self._closed = False
        self._busy = False
        self._next_due = float("inf")
        self.completed = []
        self.errors = Counter()
        self.rejected = 0
        self.started_mono = None

    def start(self):
        self.started_mono = time.monotonic()
        self._next_due = self.started_mono + WARMUP_SECONDS
        super().start()

    def offer(self, packet, outcome):
        if self.engine is None or self.images:
            return
        with self._condition:
            if self._closed or self._busy or self._crop is not None or time.monotonic() < self._next_due:
                return
            for track in outcome.observed_tracks:
                evidence = track.fresh_face_evidence
                if evidence is None or track.association_uncertain:
                    continue
                quality = evidence.quality
                if (evidence.frame_seq != packet.seq or not quality.aligned or quality.min_size < 80
                        or quality.det_score < .8 or quality.blur is None or quality.blur < 20
                        or any(reason in evidence.reason_codes for reason in (
                            "AMBIGUOUS_FACE", "ASSOCIATION_UNCERTAIN", "BAD_ALIGNMENT", "GALLERY_CHANGED"))):
                    continue
                height, width = packet.frame.shape[:2]
                x1, y1, x2, y2 = track.bbox
                # Authorized, quality-qualified real observation; preparation
                # redetects one face and may reject it. Never enroll this identity.
                self._crop = packet.frame[max(0, int(y1 * height)):min(height, math.ceil(y2 * height)),
                                          max(0, int(x1 * width)):min(width, math.ceil(x2 * width))].copy()
                self._condition.notify()
                break

    def run(self):
        import cv2
        from app.services.employee_service import enrollment_slot, prepare_face, PreparedFace
        index = 0
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._closed or (
                    self.engine is not None and time.monotonic() >= self._next_due
                    and (self.images or self._crop is not None)), timeout=.25)
                if self._closed:
                    return
                if self.engine is None or time.monotonic() < self._next_due or not (self.images or self._crop is not None):
                    continue
                crop, self._crop = self._crop, None
                self._busy = True
            started = time.monotonic()
            try:
                with enrollment_slot(self.semaphore):
                    if self.images:
                        image = self.images[index % len(self.images)].read_bytes()
                        index += 1
                    else:
                        ok, encoded = cv2.imencode(".png", crop)
                        if not ok:
                            raise EvaluationError("ENROLLMENT_CROP_ENCODING_FAILED")
                        image = encoded.tobytes()
                    prepared = prepare_face(image, self.engine, "authorized-replay-sample.png")
                    if isinstance(prepared, PreparedFace):
                        self.completed.append((started, time.monotonic()))
                    else:
                        self.rejected += 1
            except Exception as exc:
                self.errors["OUT_OF_MEMORY" if isinstance(exc, MemoryError) or
                            "out of memory" in str(exc).lower() else "ENROLLMENT_PREPARATION_FAILED"] += 1
            finally:
                crop = None
                with self._condition:
                    self._busy = False
                    self._next_due = time.monotonic() + 10.

    def close(self):
        with self._condition:
            self._closed = True
            self._crop = None
            self._condition.notify_all()
        if self.ident is not None:
            self.join(5)
        return not self.is_alive()

    def summary(self):
        return dict(status="MEASURED" if self.completed else "NOT_EVALUATED",
                    reason=None if self.completed else ("FACE_MODEL_UNAVAILABLE" if self.engine is None
                                                        else "NO_ELIGIBLE_AUTHORIZED_FACE_PREPARED"),
                    successful_preparations=len(self.completed), rejected_preparations=self.rejected,
                    errors=dict(self.errors), p95_seconds=percentile((end - start for start, end in self.completed), .95),
                    intervals=[dict(started_mono=start, completed_mono=end) for start, end in self.completed],
                    writes_employees=False, embedding_batch_limit=4, concurrent_photo_requests=1,
                    source="MANIFEST_PHOTOS" if self.images else "QUALITY_QUALIFIED_NATIVE_PERSON_CROP")


def load_scene_prompts(path, manifest, probes):
    if path is None:
        return []
    path = local_path(path, Path.cwd())
    document = _object(read_json(path), {"version", "prompts"})
    if type(document["version"]) is not int or document["version"] != 1:
        raise EvaluationError("SCENE_PROMPT_VERSION_INVALID")
    result = []
    for item in _list(document["prompts"], "SCENE_PROMPTS_INVALID"):
        _object(item, {"clip_path", "source_camera_key", "time_seconds", "kind",
                       "positive_points", "negative_points", "box"})
        clip_path = local_path(item["clip_path"], path.parent)
        if not any(clip.path == clip_path and clip.source_camera_key == item["source_camera_key"] for clip in manifest.clips):
            raise EvaluationError("SCENE_PROMPT_SOURCE_MISMATCH")
        stamp = _number(item["time_seconds"], "SCENE_PROMPT_TIME_INVALID")
        if stamp >= probes[clip_path].duration_seconds or item["kind"] not in ("floor", "object"):
            raise EvaluationError("SCENE_PROMPT_INVALID")
        points = []
        for name in ("positive_points", "negative_points"):
            values = _list(item[name], "SCENE_PROMPT_POINTS_INVALID")
            for point in values:
                if not isinstance(point, list) or len(point) != 2 or any(
                        _number(value, "SCENE_PROMPT_POINTS_INVALID") > 1 for value in point):
                    raise EvaluationError("SCENE_PROMPT_POINTS_INVALID")
            points.extend(values)
        if len(points) > 32 or (not item["positive_points"] and item["box"] is None):
            raise EvaluationError("SCENE_PROMPT_POINTS_INVALID")
        if item["box"] is not None:
            _bbox(item["box"])
        result.append({**item, "clip_path": clip_path, "time_seconds": stamp})
    return result


class SceneLoad:
    """Submit reviewed prompts only at their reviewed clip timestamp."""
    def __init__(self, pipeline, manager, specs, prompts):
        self.pipeline, self.manager, self.specs, self.prompts = pipeline, manager, specs, prompts
        self.active, self.completed, self.seen = {}, [], set()
        self.errors = Counter()
        self.next_retry = 0.

    def tick(self, now):
        for identifier, record in list(self.active.items()):
            try:
                result = self.pipeline.assistance.get(record["camera_id"], identifier)
                if result["state"] not in ("SUCCEEDED", "FAILED", "CANCELLED"):
                    continue
                record.update(state=result["state"], polygons=len(result["polygons"]),
                              error_code=result["error_code"], completed_mono=now)
            except Exception as exc:
                record.update(state="FAILED", polygons=0, error_code=self._code(exc), completed_mono=now)
            self.completed.append(record)
            self.active.pop(identifier)
            if not any(item["camera_id"] == record["camera_id"] for item in self.active.values()):
                self.pipeline.scene.invalidate(record["camera_id"])
        if len(self.active) >= 2 or now < self.next_retry:
            return
        for spec in self.specs:
            reader = self.manager.get(spec.camera_id)
            packet = reader.buffer.latest()
            source = self.manager.source_packet(spec.camera_id, packet) if packet is not None else None
            if source is None or source.source_time_seconds is None:
                continue
            for index, prompt in enumerate(self.prompts):
                key = (spec.camera_id, packet.stream_id, index)
                if (key in self.seen or prompt["clip_path"] != spec.clip.path
                        or prompt["source_camera_key"] != spec.clip.source_camera_key
                        or abs(source.source_time_seconds - prompt["time_seconds"]) > ANNOTATION_TOLERANCE_SECONDS
                        or any(item["camera_id"] == spec.camera_id for item in self.active.values())):
                    continue
                try:
                    snapshot = self.pipeline.scene.snapshot(spec.camera_id)
                    frozen = self.pipeline.scene.get_snapshot(snapshot["snapshot_id"], spec.camera_id)
                    actual = self.manager.source_packet(spec.camera_id, frozen.packet)
                    if actual is None or actual.source_time_seconds is None or abs(
                            actual.source_time_seconds - prompt["time_seconds"]) > ANNOTATION_TOLERANCE_SECONDS:
                        self.pipeline.scene.invalidate(spec.camera_id)
                        continue
                    payload = {name: prompt[name] for name in ("kind", "positive_points", "negative_points", "box")}
                    payload["snapshot_id"] = snapshot["snapshot_id"]
                    job = self.pipeline.assistance.submit(spec.camera_id, payload)
                    self.active[job["job_id"]] = dict(camera_id=spec.camera_id, kind=prompt["kind"],
                        source_time_seconds=actual.source_time_seconds, started_mono=now, state="QUEUED")
                    self.seen.add(key)
                except Exception as exc:
                    self.errors[self._code(exc)] += 1
                    self.next_retry = now + 1.
                    if not any(item["camera_id"] == spec.camera_id for item in self.active.values()):
                        self.pipeline.scene.invalidate(spec.camera_id)
                return

    @staticmethod
    def _code(exc):
        detail = getattr(exc, "detail", None)
        return detail.get("code", "SCENE_INFERENCE_FAILED") if isinstance(detail, dict) else "SCENE_INFERENCE_FAILED"

    def close(self):
        for identifier, record in list(self.active.items()):
            try:
                result = self.pipeline.assistance.cancel(record["camera_id"], identifier)
                record.update(state=result["state"], polygons=len(result["polygons"]),
                              error_code=result["error_code"], completed_mono=time.monotonic())
            except Exception as exc:
                record.update(state="CANCELLED", polygons=0, error_code=self._code(exc),
                              completed_mono=time.monotonic())
            self.completed.append(record)
            self.active.pop(identifier)

    def summary(self):
        successful = [item for item in self.completed if item["state"] == "SUCCEEDED" and item["polygons"]]
        kinds = {item["kind"] for item in successful}
        return dict(status="MEASURED" if kinds == {"floor", "object"} else "NOT_EVALUATED",
                    reason=None if kinds == {"floor", "object"} else (
                        "EXPLICIT_REVIEWED_CLIP_PROMPTS_REQUIRED" if not self.prompts else "FLOOR_AND_OBJECT_SUCCESS_REQUIRED"),
                    successful_jobs=len(successful), successful_kinds=sorted(kinds),
                    errors=dict(self.errors), jobs=self.completed, unresolved_jobs=list(self.active.values()),
                    p95_seconds=percentile((item["completed_mono"] - item["started_mono"] for item in self.completed), .95))


class ApprovedInference:
    """Own the actual production pipeline, with no server or recorder services."""
    def __init__(self, manifest, isolated, device, specs, manager, accounting):
        self.engine = self.worker = self.monitor = self.journal = None
        self.scene = self.assistance = self.segmenter = self.enrollment = None
        self.enrollment_semaphore = threading.BoundedSemaphore(1)
        self.component_errors, self.shutdown_errors = [], []
        self.close_result = None
        self.profiles = NativeProfiles(isolated.root)
        bundle_path = ROOT / "models" / "store-v1" / "manifest.json"
        if not bundle_path.is_file():
            raise EvaluationError("MODEL_BUNDLE_UNAVAILABLE_RUN_PREPARE_MODELS_FIRST")
        try:
            from app.config import settings
            expected = f"sqlite:///{isolated.database_path.as_posix()}"
            if settings.database_url != expected or settings.telegram_enabled:
                raise EvaluationError("OFFLINE_SETTINGS_BOOTSTRAP_NOT_ACTIVE")
            if (settings.ai_fps != 5 or settings.attendance_ai_fps != 10
                    or settings.ai_batch_size != 4 or settings.ai_batch_wait_ms != 10):
                raise EvaluationError("ACCEPTANCE_SCHEDULER_POLICY_MISMATCH")
            from app.ai.model_bundle import ModelBundle
            from app.ai.detector import PersonDetector
            from app.ai.face_detector import FaceEngine
            from app.ai.face_recognition import EmbeddingStore
            from app.ai.recognition_service import RecognitionService
            from app.ai.providers import setup_cuda_dlls
            from app.ai.worker import AIWorker, DetectionState, loaded_providers
            from app.ai.global_tracker import GlobalIdentityManager
            from app.ai.reid import PersonReID
            from app.ai.scene_segmenter import SceneSegmenter
            from app.services.human_journal import HumanJournal
            from app.services.attendance_service import AttendanceService
            from app.services.attendance_monitor import AttendanceMonitor
            from app.services.scene_service import SceneService
            from app.services.scene_assistance import SceneAssistanceService
            from app.database.database import engine, init_db, SessionLocal
            self.bundle = ModelBundle.load(bundle_path)
            self.bundle.require("person")
            setup_cuda_dlls()
            import onnxruntime as ort
            if device == "cuda" and "CUDAExecutionProvider" not in ort.get_available_providers():
                raise EvaluationError("CUDA_EXECUTION_PROVIDER_UNAVAILABLE")
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]
            self.face_engine = self.enroll_engine = self.reid = None
            with self.profiles.capture():
                self.detector = PersonDetector(self.bundle, providers, confidence=.1)
                for component, build in (
                    ("face", lambda: (FaceEngine(self.bundle, providers, det_size=640),
                                      FaceEngine(self.bundle, providers, det_size=640))),
                    ("body", lambda: PersonReID(self.bundle, providers)),
                    ("scene", lambda: SceneSegmenter(self.bundle)),
                ):
                    try:
                        result = build()
                        if component == "face":
                            self.face_engine, self.enroll_engine = result
                        elif component == "body":
                            self.reid = result
                        else:
                            self.segmenter = result
                    except Exception:
                        self.component_errors.append(f"{component.upper()}_MODEL_UNAVAILABLE")
            self.providers = loaded_providers(detector=self.detector, live=self.face_engine,
                                              enroll=self.enroll_engine, reid=self.reid, scene=self.segmenter)
            if device == "cuda" and any(value and value[0] != "CUDAExecutionProvider"
                                       for key, value in self.providers.items() if not key.startswith("scene_")):
                raise EvaluationError("REQUESTED_CUDA_SESSION_FELL_BACK")
            self.engine = engine
            init_db()
            configure_replay_cameras(engine, specs)
            self.employee_subjects = self._enroll(manifest, SessionLocal)
            self.store = EmbeddingStore(self.bundle.face_model_id)
            with SessionLocal() as session:
                self.store.refresh(session)
            self.service = RecognitionService(
                self.detector, self.face_engine, self.store,
                min_confidence=settings.min_recognition_confidence, min_confirmations=settings.min_confirmations,
                retry_interval=settings.face_recognition_retry_interval, track_lost_timeout=settings.track_lost_timeout,
                tracker_high_threshold=settings.tracker_new_track_threshold)
            accounting.identity_evaluator = IdentityEvaluator(
                specs, manifest.identity_labels, self.employee_subjects, self.face_engine,
                ordinary_interval=self.service.retry_interval,
                attendance_interval=settings.attendance_face_attempt_interval)
            self.attendance = AttendanceService(engine, model_bundle=self.bundle)
            self.attendance.health.set_providers(self.providers)
            self.journal = HumanJournal(engine, min_hits=self.service.min_hits,
                                        lost_timeout=settings.track_lost_timeout,
                                        image_keep_days=settings.journal_image_keep_days)
            self.state = DetectionState()
            instrument_pipeline(self.service, self.state, self.journal, self.reid, accounting)
            self.worker = AIWorker(manager, self.service, self.attendance, self.state,
                                   GlobalIdentityManager() if self.reid is not None else None,
                                   journal=self.journal, reid=self.reid)
            self.worker.model_ids = dict(bundle_id=self.bundle.bundle_id, face_model_id=self.bundle.face_model_id)
            self.monitor = AttendanceMonitor(self.attendance, write_queue=self.worker.write_queue,
                                             manager=manager, worker=self.worker)
            self.scene = SceneService(engine, manager)
            self.assistance = SceneAssistanceService(self.scene, self.segmenter)
            self.enrollment = EnrollmentLoad(self.enroll_engine, self.enrollment_semaphore, manifest.enrollments)
            accounting.enrollment = self.enrollment
        except Exception as exc:
            self.close()
            self.dispose()
            if isinstance(exc, EvaluationError):
                raise
            raise EvaluationError("PRODUCTION_PIPELINE_STARTUP_FAILED") from exc

    def _enroll(self, manifest, session_factory):
        if not manifest.enrollments:
            return {}
        if self.enroll_engine is None:
            raise EvaluationError("FACE_MODEL_UNAVAILABLE_FOR_ENROLLMENTS")
        from app.services.employee_service import enrollment_slot, prepare_face, PreparedFace
        from app.database.models import Employee, EmployeeFace, FaceModelMetadata, AttendanceRuntime
        prepared = []
        with enrollment_slot(self.enrollment_semaphore):
            for enrollment in manifest.enrollments:
                if len(enrollment.images) > 15:
                    raise EvaluationError("ENROLLMENT_SAMPLE_LIMIT_EXCEEDED")
                samples = []
                for path in enrollment.images:
                    sample = prepare_face(path.read_bytes(), self.enroll_engine, path.name)
                    if not isinstance(sample, PreparedFace):
                        raise EvaluationError("ENROLLMENT_FACE_QUALITY_INSUFFICIENT")
                    samples.append(sample)
                prepared.append((enrollment.subject_id, samples))
        employee_subjects = {}
        with session_factory.begin() as session:
            for subject, samples in prepared:
                employee = Employee(name=subject, active=True)
                session.add(employee)
                session.flush()
                employee_subjects[employee.id] = subject
                for sample in samples:
                    face = EmployeeFace(employee_id=employee.id, embedding=sample.embedding, thumbnail=sample.thumbnail)
                    session.add(face)
                    session.flush()
                    session.add(FaceModelMetadata(face_id=face.id, model_id=sample.model_id,
                                                  alignment_id=sample.alignment_id))
            session.get(AttendanceRuntime, 1).gallery_revision += 1
        return employee_subjects

    def start(self):
        self.monitor.startup()
        self.worker.start()
        self.monitor.start()
        self.enrollment.start()

    def close(self):
        # Stop producers before draining persistence; never steal worker-owned acks.
        for name in ("enrollment", "worker", "monitor"):
            owner = getattr(self, name, None)
            if owner is not None:
                try:
                    if name == "enrollment":
                        if not owner.close():
                            self.shutdown_errors.append("ENROLLMENT_NATIVE_WORK_UNRESOLVED")
                    else:
                        owner.stop()
                except Exception:
                    self.shutdown_errors.append(f"{name.upper()}_SHUTDOWN_FAILED")
        if self.assistance is not None:
            result = self.assistance.close()
            if not result.closed:
                self.shutdown_errors.append("SCENE_NATIVE_WORK_UNRESOLVED")
        if self.journal is not None:
            self.close_result = self.journal.close(timeout_seconds=5)
            if not self.close_result.drained or self.close_result.thread_alive:
                self.shutdown_errors.append("JOURNAL_UNFLUSHED")
        if not self.shutdown_errors:
            self.profiles.finish()

    def dispose(self):
        if self.engine is not None and not self.shutdown_errors:
            self.engine.dispose()
            self.engine = None


def configure_replay_cameras(engine, specs):
    from app.database.database import immediate_session
    from app.database.models import Camera, AttendanceCamera
    with immediate_session(engine) as db:
        for spec in specs:
            db.add(Camera(id=spec.camera_id, name=f"Replay {spec.camera_id}", nvr_host="invalid.local",
                          username="", password="", channel=spec.camera_id, enabled=True))
        db.flush()
        for spec in specs:
            db.add(AttendanceCamera(
                camera_id=spec.camera_id, source_revision=1, revision=1, enabled=spec.attendance,
                role=("ENTRY" if spec.camera_id == 1 else "EXIT") if spec.attendance else None,
                line_start=[.2, .5] if spec.attendance else None,
                line_end=[.8, .5] if spec.attendance else None,
                inside_point=[.5, .8] if spec.attendance else None,
                reference_width=spec.probe.width, reference_height=spec.probe.height))




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

    def coverage(self):
        with self._lock:
            return {key: {"first": state["first"], "last": state["last"]}
                    for key, state in self._streams.items()}

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




def stage_timings(runtime):
    components = ("person", "face_detection", "face_recognition", "tracking", "appearance", "journal")
    values = {name: [] for name in (*components, "face")}
    for sample in runtime:
        measured = sample.get("stage_times_ms", {})
        for name in components:
            if name in measured:
                values[name].append(measured[name])
        if "face_detection" in measured or "face_recognition" in measured:
            values["face"].append(measured.get("face_detection", 0.) + measured.get("face_recognition", 0.))
    return {name: dict(p95_ms=percentile(observations, .95)) for name, observations in values.items()}


def evaluate(manifest: Manifest, *, copies: int, duration: float, device: str,
             ffmpeg: str, probes: dict[Path, ClipProbe], isolated: IsolatedPaths,
             scene_prompts: list[dict] | None = None) -> dict:
    specs = build_replay_specs(manifest, probes, copies)
    if any((spec.probe.width, spec.probe.height) != (1920, 1080) or spec.probe.fps is None
           or abs(spec.probe.fps - 15.) > .01 for spec in specs):
        raise EvaluationError("ACCEPTANCE_REQUIRES_NATIVE_1920X1080_15FPS")
    annotations = AnnotationSamples(specs)
    face_observations = FaceObservations()
    manager = ReplayManager(specs, ffmpeg, annotations)
    replays = [manager.get(spec.camera_id) for spec in specs]
    metrics = {spec.camera_id: CameraMeasurements(spec.camera_id, spec.target_fps, spec.attendance) for spec in specs}
    accounting = ProductionAccounting(metrics, annotations, manager, face_observations)
    inference = ApprovedInference(manifest, isolated, device, specs, manager, accounting)
    try:
        accounting.identity_evaluator.precompute(ffmpeg)
    except Exception:
        inference.close()
        inference.dispose()
        raise
    scene_load = SceneLoad(inference, manager, specs, scene_prompts or [])
    errors = []
    inventory, other_processes = gpu_inventory(), gpu_processes()
    peaks = {gpu["name"]: gpu["memory_used_mib"] for gpu in inventory}
    started = time.monotonic()
    manager.measurement_end = accounting.measurement_end = started + duration
    accounting.measurement_start = started + WARMUP_SECONDS
    cpu_started = time.process_time()
    ended = started
    preview_token, preview = None, None
    last_resource_sample = started
    runtime_samples = []
    attendance_reasons = Counter()
    peak_attendance_pending = 0
    try:
        for replay in replays:
            replay.start()
        inference.start()
        while time.monotonic() - started < duration:
            now = time.monotonic()
            for replay in replays:
                replay.check_health()
            if not inference.worker.is_alive() or not inference.monitor.is_alive():
                raise EvaluationError("PRODUCTION_OWNER_STOPPED")
            accounting.health(now, inference.journal.health_snapshot())
            result = manager.preview(specs[0].camera_id)
            if result is not None and result[0] != preview_token:
                preview_token, preview = result
                metrics[specs[0].camera_id].display_times.append(time.monotonic())
            if now >= started + WARMUP_SECONDS:
                scene_load.tick(now)
            if now - last_resource_sample >= 1:
                for gpu in gpu_inventory():
                    peaks[gpu["name"]] = max(peaks.get(gpu["name"], 0), gpu["memory_used_mib"])
                snapshot = inference.worker.inference_snapshot()
                runtime_samples.append((now, snapshot))
                attendance_snapshot = inference.attendance.health.snapshot()
                attendance_reasons.update(attendance_snapshot["reasons"])
                peak_attendance_pending = max(peak_attendance_pending, attendance_snapshot["pending_write_count"])
                last_resource_sample = now
            time.sleep(.05)
        ended = min(time.monotonic(), started + duration)
    except Exception as exc:
        ended = min(time.monotonic(), started + duration)
        errors.append(str(exc) if isinstance(exc, EvaluationError) else "PIPELINE_RUNTIME_FAILED")
        errors.append(f"FAILURE_TYPE_{type(exc).__name__}")
    finally:
        cpu_ended = time.process_time()
        manager.measurement_end = accounting.measurement_end = ended
        scene_load.close()
        inference.close()
        # Completed FFmpeg loops supply benchmark CPU/RSS; a forced final stop
        # may omit that process's last resource line and is labelled below.
        for replay in replays:
            try:
                replay.stop()
            except EvaluationError as exc:
                errors.append(str(exc))
    errors.extend(inference.shutdown_errors)
    errors.extend(accounting.errors)
    camera_reports = []
    for replay in replays:
        camera = replay.spec.camera_id
        measurements = metrics[camera]
        measurements.decoded_times = replay.decoded_times
        report = measurements.summary(started, ended)
        runtime = [snapshot[camera] for stamp, snapshot in runtime_samples
                   if stamp >= started + WARMUP_SECONDS and camera in snapshot]
        last = runtime[-1] if runtime else {}
        report.update(source_camera_key=replay.spec.clip.source_camera_key,
                      stream_id=replay.spec.stream_id, stream_generations=replay.loops,
                      source_width=replay.spec.probe.width, source_height=replay.spec.probe.height,
                      source_fps=replay.spec.probe.fps, staggered_offset_seconds=replay.spec.offset_seconds,
                      decoded_frames=replay.decoded_count, decoder_mode=replay.decoder_mode,
                      skipped_frames=last.get("skipped_frames"), skipped_slots=last.get("skipped_slots"),
                      oldest_pending_age_seconds=max((item.get("pending_age_seconds") or 0 for item in runtime), default=None),
                      deferred_face_tasks=last.get("face_deferred"), deferred_reid_tasks=last.get("reid_deferred"),
                      stage_times=stage_timings(runtime),
                      analysis_state_samples=dict(Counter(reason for item in runtime for reason in item["reason_codes"])),
                      decoder_cpu_seconds=replay.decoder_cpu_seconds,
                      decoder_peak_rss_kib=replay.decoder_peak_rss_kib,
                      decoder_completed_resource_samples=replay.decoder_resource_samples)
        camera_reports.append(report)
    with closing(sqlite3.connect(isolated.database_path)) as db:
        integrity = journal_integrity(db, accounting.expected)
        attendance_events = db.execute("SELECT status,count(*) FROM attendance_events GROUP BY status").fetchall()
        payroll_sessions = db.execute("SELECT count(*) FROM work_sessions").fetchone()[0]
    journal_health = accounting.health_report(ended)
    journal_report = dict(
        integrity=integrity, **journal_health, admissions=dict(accounting.admission_counts),
        acknowledgements=dict(accounting.ack_counts),
        shutdown=asdict(inference.close_result) if inference.close_result is not None else None,
        final_health=inference.journal.health_snapshot(),
        coverage_scope="MEMORY_OBSERVED_INTERVALS_AND_DURABLE_GAPS_NOT_A_COMPLETENESS_CLAIM")
    scene_report, enrollment_report = scene_load.summary(), inference.enrollment.summary()
    scene_report["impact"] = concurrent_impact(metrics, scene_report["jobs"], started, ended)
    enrollment_report["impact"] = concurrent_impact(metrics, enrollment_report["intervals"], started, ended)
    profile_report = inference.profiles.summary()
    gpu_profiles = [item for item in profile_report if "CUDAExecutionProvider" in item.get("providers", ())]
    appearance_report = accounting.appearance_snapshot()
    exercised = {
        "AI_WORKER_SCHEDULER_NOT_EXERCISED": bool(accounting.batch_calls and all(metric.analyses for metric in metrics.values())),
        "DURABLE_HUMAN_JOURNAL_NOT_EXERCISED": bool(integrity["durable_visits"] and accounting.ack_counts["COMMITTED"]),
        "ATTENDANCE_QUEUE_GATES_AND_CROSSINGS_NOT_EXERCISED": bool(attendance_events),
        "BATCHED_BODY_LINKING_NOT_EXERCISED": bool(appearance_report["embedded_crops"] and appearance_report["acknowledged_links"]),
        "ENROLLMENT_CONCURRENT_LOAD_NOT_EXERCISED": enrollment_report["status"] == "MEASURED" and enrollment_report["impact"]["overlapping_analyses"] > 0,
        "SCENE_ASSISTANCE_CONCURRENT_IMPACT_NOT_EVALUATED": scene_report["status"] == "MEASURED" and scene_report["impact"]["overlapping_analyses"] > 0,
        "CUDA_HEAVY_NODE_PLACEMENT_NOT_PROFILED": bool(gpu_profiles) and all(item["status"] == "PASS" for item in gpu_profiles),
        "DECODER_CPU_LOAD_NOT_MEASURED": all(replay.decoder_resource_samples for replay in replays),
        "DERIVED_PREVIEW_NOT_EXERCISED": bool(preview),
    }
    integration_needs = [need for need in INTEGRATION_NEEDS if not exercised[need]] + inference.component_errors
    failures = list(errors)
    if integrity["status"] == "FAIL" or accounting.admission_counts["REJECTED"]:
        failures.append("JOURNAL_LOSS")
    if journal_health["degraded_intervals"] or integrity["persisted_gap_count"]:
        failures.append("JOURNAL_DEGRADED")
    if journal_health["persistent_growth"]:
        failures.append("PERSISTENT_QUEUE_GROWTH")
    if payroll_sessions:
        failures.append("UNCOMMISSIONED_REPLAY_CREATED_PAYROLL")
    if accounting.source_metadata_missing:
        failures.append("SOURCE_TIMESTAMP_ACCOUNTING_LOST")
    if enrollment_report["errors"]:
        failures.extend(enrollment_report["errors"])
    if any(item["state"] == "FAILED" for item in scene_report["jobs"]) or scene_report["errors"]:
        failures.append("SCENE_CONCURRENT_WORKLOAD_ERRORS")
    if any(item["status"] == "FAIL" for item in gpu_profiles):
        failures.append("CUDA_HEAVY_NODE_FALLBACK")
    if inference.worker.write_queue.pending():
        failures.append("ATTENDANCE_WRITES_UNFLUSHED")
    if any(reason in attendance_reasons for reason in ("EVIDENCE_LOST", "PERSISTENCE_ERROR")):
        failures.append("ATTENDANCE_PERSISTENCE_DEGRADED")
    elapsed = ended - started
    quality = annotations.evaluate()
    quality["recognition"] = accounting.identity_evaluator.summary(annotations)
    quality["verified_identity_labels_supplied"] = len(manifest.identity_labels)
    durable_status = ("FAIL" if any(reason in failures for reason in
                      ("JOURNAL_LOSS", "JOURNAL_DEGRADED", "JOURNAL_UNFLUSHED")) else integrity["status"])
    quality["detection_tracking_acceptance"]["durable_visit_coverage"] = durable_status
    acceptance = acceptance_gates(camera_reports, duration_seconds=elapsed, copies=copies,
                                  gpu_inventory=inventory, integration_needs=integration_needs,
                                  runtime_failures=failures, recognition=quality["recognition"])
    acceptance["journal"] = dict(status=durable_status)
    acceptance["detection_tracking"] = quality["detection_tracking_acceptance"]
    acceptance["deployment"] = dict(status="PASS" if all((
        acceptance["capacity"]["status"] == "PASS", integrity["status"] == "PASS",
        quality["detection_tracking_acceptance"]["status"] == "PASS",
        acceptance["target_hardware"]["certified"], not other_processes.get("contaminated", True),
        acceptance["recognition"]["status"] == "PASS", acceptance["recognition"]["certified"],
        acceptance["recognition"]["holdout"]["independent_holdout_verified"])) else "NOT_ACCEPTED",
        recognition_holdout="OPERATOR_PHOTO_TIME_VISIT_HOLDOUT_ATTESTATION_NOT_REPRESENTED_IN_MANIFEST_V1")
    result = {
        "status": "ERROR" if errors else "PRODUCTION_PIPELINE_MEASURED",
        "workload": "REPEATED_SINGLE_VIEW" if len({clip.source_camera_key for clip in manifest.clips}) == 1 and copies > 1 else "LOCAL_CLIP_REPLAY",
        "scope": "PRODUCTION_WORKER_WITH_LOCAL_REPLAY_READERS",
        "hardware_scope": "TARGET_HARDWARE" if acceptance["target_hardware"]["certified"] else "PROVISIONAL_NOT_5060TI_CERTIFICATION",
        "duration_seconds": elapsed, "requested_duration_seconds": duration, "warmup_seconds": WARMUP_SECONDS,
        "batch_size": 4, "cameras": camera_reports, "quality": quality,
        "gpu_inventory": inventory, "other_gpu_processes": other_processes,
        "peak_gpu_device_used_memory_mib": peaks, "peak_process_memory_bytes": process_peak_memory_bytes(),
        "process_cpu_percent_one_core_basis": 100 * (cpu_ended - cpu_started) / max(elapsed, .001),
        "decoder_cpu_percent_one_core_basis": 100 * sum(replay.decoder_cpu_seconds for replay in replays) / max(elapsed, .001),
        "decoder_resource_scope": "FFMPEG_COMPLETED_PROCESS_BENCHMARKS_INCLUDING_WARMUP_FINAL_INTERRUPTED_PROCESS_MAY_BE_UNREPORTED",
        "model_bundle_id": inference.bundle.bundle_id, "face_model_id": inference.bundle.face_model_id,
        "component_status": inference.bundle.component_status, "actual_session_providers": inference.providers,
        "native_profiles": profile_report,
        "preview": dict(responses=len(metrics[1].display_times), latest_jpeg_bytes=len(preview) if preview else 0,
                        long_edge_limit=1280, consumer="PRODUCTION_JPEG_CACHE_ONE_LOCAL_PREVIEW"),
        "face_observations": face_observations.summary(), "journal": journal_report,
        "appearance": appearance_report,
        "attendance": dict(events_by_status=dict(attendance_events), payroll_sessions=payroll_sessions,
                           synthetic_gate=True, commissioned=False, verified_crossing_accuracy="NOT_EVALUATED",
                           pending_writes=len(inference.worker.write_queue.pending()),
                           peak_pending_writes=peak_attendance_pending,
                           observed_health_reasons=dict(attendance_reasons)),
        "scene_assistance": scene_report, "enrollment": enrollment_report,
        "errors": errors, "runtime_failures": sorted(set(failures)), "integration_needs": integration_needs,
        "acceptance": acceptance}
    inference.dispose()
    return result


def concurrent_impact(metrics, intervals, started, ended):
    spans = sorted((max(started + WARMUP_SECONDS, item["started_mono"]),
                    min(ended, item["completed_mono"])) for item in intervals
                   if item.get("completed_mono") is not None)
    spans = [(lower, upper) for lower, upper in spans if upper > lower]
    result = []
    for camera, measurements in metrics.items():
        overlapping = [sample for sample in measurements.analyses
                       if sample.started_mono >= started + WARMUP_SECONDS and sample.completed_mono <= ended
                       and any(sample.started_mono < upper and sample.completed_mono > lower for lower, upper in spans)]
        result.append(dict(camera_id=camera, analyzed_frames=len(overlapping),
                           p95_completion_age_seconds=percentile(
                               (sample.completed_mono - sample.received_mono for sample in overlapping), .95)))
    return dict(overlapping_analyses=sum(item["analyzed_frames"] for item in result), cameras=result,
                scope="ACTUAL_OVERLAPPING_INFERENCE_NOT_CAUSAL_SLOWDOWN_ESTIMATE")


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
    parser.add_argument("--scene-prompts", help="Private JSON of explicitly reviewed, clip-bound visible prompts")
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
        prompt_path = local_path(args.scene_prompts, Path.cwd()) if args.scene_prompts else None
        if prompt_path is not None and (output == prompt_path or (output.exists() and os.path.samefile(output, prompt_path))):
            output = None
            raise EvaluationError("REPORT_OVERWRITES_INPUT")
        ffmpeg, ffprobe = resolve_media_tools()
        # No server/ReaderManager/recording/export/Telegram service is constructed.
        with tempfile.TemporaryDirectory(prefix="na-meste-evaluation-") as directory:
            with isolated_environment(Path(directory), args.device) as isolated:
                report["gpu_inventory"] = gpu_inventory()
                probes = probe_manifest(manifest, ffprobe)
                prompts = load_scene_prompts(prompt_path, manifest, probes)
                report["clips"] = [{"path": str(clip.path), "source_camera_key": clip.source_camera_key,
                                     "width": probes[clip.path].width, "height": probes[clip.path].height,
                                     "fps": probes[clip.path].fps, "duration_seconds": probes[clip.path].duration_seconds,
                                     "codec": probes[clip.path].codec} for clip in manifest.clips]
                report.update(evaluate(manifest, copies=args.copies, duration=args.duration, device=args.device,
                                       ffmpeg=ffmpeg, probes=probes, isolated=isolated, scene_prompts=prompts))
    except (EvaluationError, OSError) as exc:
        report.update(status="ERROR", error_code=str(exc) if isinstance(exc, EvaluationError) else "EVALUATION_IO_FAILED")
        report["acceptance"] = {"capacity": {"status": "NOT_EVALUATED"}, "recognition": {"status": "NOT_EVALUATED"}}
    if output is not None and manifest is not None:
        try:
            if args.scene_prompts:
                protected = local_path(args.scene_prompts, Path.cwd())
                if output == protected or (output.exists() and os.path.samefile(output, protected)):
                    raise EvaluationError("REPORT_OVERWRITES_INPUT")
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
