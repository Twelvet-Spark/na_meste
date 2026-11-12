"""Process-local, one-worker archive playback. No attendance writes or device paths in views.

Lock order: manager -> storage -> short SQL snapshot. Never join a worker while
holding the manager lock. Inputs bind persisted local identity, remote jobs bind
both captured revisions. A response owns a separate lease until ASGI send exits.
"""
import json
import math
import re
import subprocess
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from urllib.parse import quote, urlencode
from uuid import UUID, uuid4

from fastapi import HTTPException
from starlette.responses import FileResponse
from sqlalchemy.orm import Session

from app.attendance_types import milliseconds
from app.database.models import Camera, RecordingGap, RecordingSegment, RecordingSession
from app.media.process_guard import MediaProcessGuard
from app.media.recorder import probe_video, stop_process
from app.media.storage import ordinary
from app.rtsp.dahua import url_host
from app.services.camera_service import get_archive_source
from app.services.dahua_service import DahuaArchiveClient, DahuaError, archive_path
from app.time_utils import attendance_iso, utcnow

CACHE_MAX_BYTES = 2_000_000_000
ACTIVE = {"PREPARING", "READY"}
TS_NAME = re.compile(r"([0-9]+)\.ts\Z")
ERROR_STATUS = {"ARCHIVE_NOT_CONFIGURED": 409, "DEVICE_AUTH_FAILED": 502,
                "DEVICE_UNREACHABLE": 502, "DEVICE_TIMEOUT": 504, "ARCHIVE_UNSUPPORTED": 422,
                "DEVICE_PROTOCOL_ERROR": 502, "DEVICE_TLS_FAILED": 502, "SEARCH_LIMIT": 422}


def timestamp(value):
    try:
        if isinstance(value, str):
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if not isinstance(value, datetime) or value.utcoffset() is None:
            raise ValueError
        value = milliseconds(value)
        attendance_iso(value)
        # Both device offsets and the largest allowed query/playback bound must fit.
        value - timedelta(days=1)
        value + timedelta(days=2)
        return value
    except (ValueError, OverflowError, TypeError):
        raise HTTPException(422, "TIMESTAMP_REQUIRES_REPRESENTABLE_EXPLICIT_OFFSET") from None


def iso(value):
    return value.replace(tzinfo=timezone.utc).isoformat(timespec="milliseconds") if value is not None else None


def iso_start(value):
    """Advertise a representable millisecond at or after the footage boundary."""
    return iso(value + timedelta(microseconds=(-value.microsecond) % 1000)) if value is not None else None


def bounds(start, end):
    start, end = timestamp(start), timestamp(end)
    if not timedelta(0) < end - start <= timedelta(days=1):
        raise HTTPException(422, "RANGE_MUST_BE_POSITIVE_AT_MOST_24_HOURS")
    return start, end


@dataclass(frozen=True)
class LocalInput:
    id: str
    key: str
    start: datetime
    end: datetime
    connection: str
    duration: float


@dataclass
class Output:
    sequence: int
    key: str
    start: datetime
    duration: float
    lease: object = None
    retired: float | None = None


@dataclass
class Playback:
    id: str
    camera_id: int
    source: str
    start: datetime
    end: datetime
    next_at: datetime | None
    touched: float
    directory: str
    captured: object = field(default=None, repr=False)
    interval: object = field(default=None, repr=False)
    inputs: list = field(default_factory=list, repr=False)
    input_leases: list = field(default_factory=list, repr=False)
    outputs: dict = field(default_factory=dict, repr=False)
    listed: list = field(default_factory=list)
    state: str = "PREPARING"
    error: str | None = None
    ended: float | None = None
    process: object = field(default=None, repr=False)
    thread: object = field(default=None, repr=False)
    cancel_event: object = field(default_factory=threading.Event, repr=False)
    drainers: list = field(default_factory=list, repr=False)
    process_error: str | None = None
    validated: bool = False
    total_duration: float = 0.
    last_progress: float = 0.
    launched: float = 0.


class LeasedFileResponse(FileResponse):
    def __init__(self, lease):
        self._lease = lease
        try:
            path = lease.__enter__()
            super().__init__(path, media_type="video/mp2t", headers={"Cache-Control": "private, no-store"})
        except BaseException:
            lease.__exit__(None, None, None)
            raise

    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            # FileResponse's background hook is bypassed on 416 and send errors.
            self._lease.__exit__(None, None, None)


class PlaybackManager:
    def __init__(self, engine, storage, ffmpeg_path, ffprobe_path, *, now=utcnow, monotonic=time.monotonic):
        self.engine, self.storage = engine, storage
        self.ffmpeg, self.ffprobe = ffmpeg_path, ffprobe_path
        self.now, self.monotonic = now, monotonic
        self.lock = threading.RLock()
        self.jobs, self.requests, self.searches = {}, {}, {}
        self.search_clients = {}
        self.admission = None
        self.closed = False
        self.stop_event = threading.Event()
        self._recover_cache()
        self.sweeper = threading.Thread(target=self._sweep, daemon=True, name="playback-expiry")
        self.sweeper.start()

    def _recover_cache(self):
        root = self.storage.root / "cache"
        if not root.exists():
            root.mkdir()
        ordinary(root, directory=True)
        for directory in root.iterdir():
            try:
                identifier = str(UUID(directory.name))
                if identifier != directory.name:
                    continue
                ordinary(directory, directory=True)
                marker = self.storage.path(f"cache/{identifier}/owner.json")
                if marker.stat().st_size > 256 or json.loads(marker.read_text("utf-8")) != {"version": 1, "id": identifier}:
                    continue
                for path in directory.iterdir():
                    if self._owned_name(path.name):
                        self.storage.register_cache(f"cache/{identifier}/{path.name}", expires_at=self.now())
            except (OSError, ValueError, HTTPException):
                continue
        self.storage.enforce()

    @staticmethod
    def _owned_name(name):
        return name in {"owner.json", "input.ffconcat", "index.m3u8", "index.m3u8.tmp"} or re.fullmatch(r"[0-9]+\.ts(?:\.tmp)?", name)

    def _camera(self, camera_id, remote=False):
        with Session(self.engine) as db:
            if db.get(Camera, camera_id) is None:
                raise HTTPException(404, "CAMERA_NOT_FOUND")
            source = get_archive_source(db, camera_id) if remote else None
        if remote and source is None:
            raise HTTPException(409, "ARCHIVE_NOT_CONFIGURED")
        return source

    @staticmethod
    def _revision(source):
        return (source.source_revision, source.archive_revision) if source else None

    def _remote_intervals(self, source, start, end):
        key = (source.camera_id, *self._revision(source), start, end)
        with self.lock:
            cached = self.searches.get(key)
            if cached and self.monotonic() - cached[0] < 60:
                return cached[1]
        client = DahuaArchiveClient(source)
        with self.lock:
            if self.closed:
                raise HTTPException(503, "PLAYBACK_CLOSED")
            self.search_clients[client] = source
        try:
            entries = client.list_recordings(start.replace(tzinfo=timezone.utc), end.replace(tzinfo=timezone.utc))
        except DahuaError as exc:
            raise HTTPException(409 if exc.code == "SOURCE_CHANGED" else ERROR_STATUS.get(exc.code, 502), exc.code) from None
        finally:
            with self.lock:
                self.search_clients.pop(client, None)
        current = self._camera(source.camera_id, remote=True)
        if self._revision(current) != self._revision(source):
            raise HTTPException(409, "SOURCE_CHANGED")
        result = [(str(uuid4()), interval) for interval in entries]
        with self.lock:
            self.searches[key] = self.monotonic(), result
        return result

    def _local(self, camera_id, start, end):
        with Session(self.engine) as db:
            segments = db.query(RecordingSegment).filter(
                RecordingSegment.camera_id == camera_id, RecordingSegment.started_at < end,
                RecordingSegment.ended_at > start).order_by(RecordingSegment.started_at, RecordingSegment.id).all()
            rows = [(LocalInput(row.id, row.storage_key, row.started_at, row.ended_at,
                                row.connection_id, row.duration_ms / 1000), row.state) for row in segments]
            gaps = [(row.started_at, row.ended_at, row.reason_code) for row in db.query(RecordingGap).join(
                RecordingSession, RecordingSession.id == RecordingGap.session_id).filter(
                RecordingSession.camera_id == camera_id, RecordingGap.started_at < end).order_by(
                RecordingGap.started_at).all() if row.ended_at is None or row.ended_at > start]
        return rows, gaps

    def timeline(self, camera_id, source, start, end):
        start, end = bounds(start, end)
        if source not in ("local", "dahua"):
            raise HTTPException(422, "INVALID_RECORDING_SOURCE")
        captured = self._camera(camera_id, remote=source == "dahua")
        result = {"camera_id": camera_id, "source": source, "from": iso(start), "to": iso(end),
                  "time_basis": "approximate_server_recording_time" if source == "local" else "configured_device_offset",
                  "intervals": [], "gaps": [], "warnings": []}
        if source == "local":
            rows, gaps = self._local(camera_id, start, end)
            result["intervals"] = [{"id": row.id, "from": iso_start(max(start, row.start)),
                                    "to": iso(min(end, row.end)), "state": state} for row, state in rows]
            result["gaps"] = [{"from": iso_start(max(start, lower)), "to": iso(min(end, upper or end)),
                               "reason": reason} for lower, upper, reason in gaps]
        else:
            result["warnings"] = ["DEVICE_OFFSET_MUST_MATCH_FOOTAGE"]
            result["intervals"] = [{"id": identifier, "from": iso(max(start, item.start_utc.replace(tzinfo=None))),
                                    "to": iso(min(end, item.end_utc.replace(tzinfo=None))), "state": "READY"}
                                   for identifier, item in self._remote_intervals(captured, start, end)
                                   if item.start_utc.replace(tzinfo=None) < end and item.end_utc.replace(tzinfo=None) > start]
        return result

    def _local_selection(self, camera_id, at):
        rows, gaps = self._local(camera_id, at - timedelta(days=1), at + timedelta(days=1))
        pieces = []
        for item, state in rows:
            if state != "READY":
                continue
            fragments = [(item.start, item.end)]
            for lower, upper, _reason in gaps:
                upper = upper or at + timedelta(days=1)
                split = []
                for left, right in fragments:
                    if upper <= left or lower >= right:
                        split.append((left, right))
                    else:
                        if left < lower:
                            split.append((left, lower))
                        if upper < right:
                            split.append((upper, right))
                fragments = split
            pieces.extend((left, right, item) for left, right in fragments)
        pieces.sort(key=lambda item: (item[0], item[1]))
        runs = []
        for left, right, item in pieces:
            prior = runs[-1] if runs else None
            precision_join = prior and timedelta(0) <= left - prior[1] <= timedelta(milliseconds=1)
            if (precision_join and prior[2][-1].connection == item.connection
                    and not any(lower < left and (upper is None or upper > prior[1]) for lower, upper, _ in gaps)):
                prior[1] = right
                prior[2].append(item)
            else:
                runs.append([left, right, [item]])
        for index, (left, right, items) in enumerate(runs):
            if left <= at < right:
                end = min(right, at + timedelta(hours=1))
                next_at = end if end < right else runs[index + 1][0] if index + 1 < len(runs) else None
                return end, next_at, [item for item in items if item.end > at and item.start < end]
        next_at = next((left for left, _right, _items in runs if left > at), None)
        raise HTTPException(409, {"code": "NO_RECORDING_AT_TIME", "next_at": iso_start(next_at)})

    def _encoder(self):
        MediaProcessGuard.require()
        if not self.ffmpeg or not self.ffprobe:
            raise HTTPException(503, "FFMPEG_UNAVAILABLE")
        try:
            result = subprocess.run([self.ffmpeg, "-hide_banner", "-encoders"], capture_output=True,
                                    timeout=10, close_fds=True)
            output = result.stdout.decode("utf-8", "replace") if isinstance(result.stdout, bytes) else result.stdout
            if result.returncode or not re.search(r"\blibx264\b", output):
                raise HTTPException(503, "LIBX264_UNAVAILABLE")
        except (OSError, subprocess.SubprocessError):
            raise HTTPException(503, "FFMPEG_UNAVAILABLE") from None

    def create(self, camera_id, source, at, request_id):
        at = timestamp(at)
        if source not in ("local", "dahua"):
            raise HTTPException(422, "INVALID_RECORDING_SOURCE")
        identifier = str(UUID(str(request_id)))
        payload = (camera_id, source, at)
        waiter = None
        with self.lock:
            if self.closed:
                raise HTTPException(503, "PLAYBACK_CLOSED")
            prior = self.requests.get(identifier)
            if prior is not None:
                if prior != payload:
                    raise HTTPException(409, "REQUEST_ID_CONFLICT")
                return self._view(self._get(identifier))
            if self.admission is not None:
                if self.admission[0] != identifier:
                    raise HTTPException(429, "PLAYBACK_BUSY")
                if self.admission[1] != payload:
                    raise HTTPException(409, "REQUEST_ID_CONFLICT")
                waiter = self.admission
            else:
                if any(job.state in ACTIVE or job.thread and job.thread.is_alive()
                       or job.process is not None and job.process.poll() is None for job in self.jobs.values()):
                    raise HTTPException(429, "PLAYBACK_BUSY")
                self.admission = (identifier, payload, threading.Event(), [])
            admission = self.admission
        if waiter is not None:
            waiter[2].wait()
            if waiter[3]:
                raise waiter[3][0]
            return self.get(identifier)
        job = None
        try:
            captured = self._camera(camera_id, remote=source == "dahua")
            interval, inputs = None, []
            if source == "local":
                end, next_at, inputs = self._local_selection(camera_id, at)
            else:
                entries = self._remote_intervals(captured, at, at + timedelta(days=1))
                interval = next((item for _id, item in entries if item.start_utc.replace(tzinfo=None) <= at < item.end_utc.replace(tzinfo=None)), None)
                next_at = min((item.start_utc.replace(tzinfo=None) for _id, item in entries if item.start_utc.replace(tzinfo=None) > at), default=None)
                if interval is None:
                    raise HTTPException(409, {"code": "NO_RECORDING_AT_TIME", "next_at": iso(next_at)})
                end = min(interval.end_utc.replace(tzinfo=None), at + timedelta(hours=1))
                if end < interval.end_utc.replace(tzinfo=None):
                    next_at = end
            self._encoder()
            if not self.storage.enforce():
                raise HTTPException(503, "STORAGE_FULL")
            directory = f"cache/{identifier}"
            path = self.storage.path(directory, exists=False)
            path.mkdir(exist_ok=True)
            if any(path.iterdir()):
                raise HTTPException(409, "PLAYBACK_ID_UNAVAILABLE")
            (path / "owner.json").write_text(json.dumps({"version": 1, "id": identifier}), encoding="utf-8")
            job = Playback(identifier, camera_id, source, at, end, next_at, self.monotonic(), directory,
                           captured=captured, interval=interval, inputs=inputs)
            for item in inputs:
                lease, leased_path = self._lease_input(item, camera_id)
                job.input_leases.append((item, lease, leased_path))
            with self.lock:
                if self.closed:
                    raise HTTPException(503, "PLAYBACK_CLOSED")
                if captured and self._revision(self._camera(camera_id, True)) != self._revision(captured):
                    raise HTTPException(409, "SOURCE_CHANGED")
                self.jobs[identifier] = job
                self.requests[identifier] = payload
                initial = self._view(job)
                job.thread = threading.Thread(target=self._work, args=(job,), daemon=True, name="archive-playback")
                job.thread.start()
                return initial
        except Exception as exc:
            failure = exc if isinstance(exc, HTTPException) else HTTPException(503, "PLAYBACK_UNAVAILABLE")
            admission[3].append(failure)
            if job is not None:
                self._release_inputs(job)
                job.state, job.error, job.ended = "FAILED", "PLAYBACK_UNAVAILABLE", self.monotonic()
                self._retire_files(job)
            raise failure from None
        finally:
            with self.lock:
                self.admission = None
                admission[2].set()

    def _lease_input(self, item, camera_id):
        with self.storage.lock:
            with Session(self.engine) as db:
                row = db.get(RecordingSegment, item.id)
                if (row is None or row.state != "READY" or row.camera_id != camera_id
                        or (row.storage_key, row.started_at, row.ended_at, row.connection_id, row.duration_ms)
                        != (item.key, item.start, item.end, item.connection, round(item.duration * 1000))):
                    raise HTTPException(409, "RECORDING_UNAVAILABLE")
            # Close the SQL snapshot before entering the storage lease.
            lease = self.storage.lease(item.id)
            return lease, lease.__enter__()

    def _view(self, job):
        result = {"id": job.id, "state": job.state, "from": iso(job.start), "to": iso(job.end),
                  "next_at": iso_start(job.next_at), "error_code": job.error}
        if job.validated and job.listed and job.state in {"READY", "ENDED", "CANCELLED"}:
            result.update(manifest_url=f"/api/recordings/playbacks/{job.id}/manifest.m3u8",
                          available_from=iso(job.outputs[job.listed[0]].start),
                          available_to=iso(min(job.end, job.outputs[job.listed[-1]].start + timedelta(seconds=job.outputs[job.listed[-1]].duration))))
        return result

    def _get(self, identifier, *, touch=True):
        job = self.jobs.get(str(identifier))
        if job is None:
            raise HTTPException(404, "PLAYBACK_NOT_FOUND")
        if job.state == "EXPIRED" or job.ended is not None and self.monotonic() - job.ended >= 120:
            self._expire(job)
            raise HTTPException(410, "PLAYBACK_EXPIRED")
        if touch:
            job.touched = self.monotonic()
        return job

    def get(self, identifier):
        with self.lock:
            return self._view(self._get(identifier))

    def cancel(self, identifier):
        with self.lock:
            job = self._get(identifier, touch=False)
            job.cancel_event.set()
            thread = job.thread
        if thread and thread is not threading.current_thread():
            thread.join()
        with self.lock:
            return self._view(job)

    def source_changed(self, snapshot):
        # Compare the current committed source, not a potentially stale notification.
        with self.lock:
            candidates = [job for job in self.jobs.values() if job.camera_id == snapshot.id and job.source == "dahua" and job.state in ACTIVE]
            try:
                current = self._camera(snapshot.id, remote=True)
            except HTTPException:
                current = None
            for job in candidates:
                if self._revision(current) != self._revision(job.captured):
                    job.error = "SOURCE_CHANGED"
                    job.cancel_event.set()
            clients = [client for client, source in self.search_clients.items()
                       if source.camera_id == snapshot.id and self._revision(current) != self._revision(source)]
        for client in clients:
            client.cancel()

    def deleted(self, camera_id):
        with self.lock:
            for job in self.jobs.values():
                if job.camera_id == camera_id and job.source == "dahua" and job.state in ACTIVE:
                    job.error = "SOURCE_CHANGED"
                    job.cancel_event.set()
            clients = [client for client, source in self.search_clients.items() if source.camera_id == camera_id]
        for client in clients:
            client.cancel()

    def _url(self, job, fallback):
        source = job.captured
        authority = f"{quote(source.username, safe='')}:{quote(source.password, safe='')}@{url_host(source.host)}:{source.rtsp_port}"
        if fallback:
            path = archive_path(job.interval.path)
            return f"rtsp://{authority}/" + "/".join(quote(part, safe="") for part in path.split("/"))
        zone = timezone(timedelta(minutes=source.utc_offset_minutes))
        def device(value):
            return value.replace(tzinfo=timezone.utc).astimezone(zone).strftime("%Y_%m_%d_%H_%M_%S")
        # Device protocol accepts whole seconds; output trim handles subsecond positions.
        return f"rtsp://{authority}/cam/playback?" + urlencode({"channel": source.channel,
            "starttime": device(job.start), "endtime": device(job.end + timedelta(microseconds=999999))})

    def _arguments(self, job, fallback):
        directory = self.storage.path(job.directory, directory=True)
        args = [self.ffmpeg, "-hide_banner", "-loglevel", "error"]
        filters = "setpts=PTS-STARTPTS,pad=ceil(iw/2)*2:ceil(ih/2)*2"
        if job.source == "local":
            concat = directory / "input.ffconcat"
            lines = ["ffconcat version 1.0"]
            for index, (item, _lease, path) in enumerate(job.input_leases):
                escaped = str(path).replace("\\", "/").replace("'", "'\\''")
                lines.append(f"file '{escaped}'")
                seek = (job.start - item.start).total_seconds() if index == 0 else 0.
                if index == 0:
                    lines.append(f"inpoint {seek:.6f}")
                # Explicit duration includes the inpoint unless we subtract it;
                # later files must begin immediately after the retained first span.
                lines.append(f"duration {item.duration - seek:.6f}")
            concat.write_text("\n".join(lines) + "\n", encoding="utf-8")
            # Preserve negative preroll timestamps until concat metadata selects
            # decoded frames, instead of a global demuxer seek skipping a GOP.
            args += ["-copyts", "-readrate", "1", "-segment_time_metadata", "1",
                     "-f", "concat", "-safe", "0", "-i", str(concat), "-fps_mode", "passthrough"]
            filters = "select=concatdec_select," + filters
        else:
            seek = ((job.start - job.interval.start_utc.replace(tzinfo=None)).total_seconds() if fallback
                    else job.start.microsecond / 1_000_000)
            args += ["-rtsp_transport", "tcp", "-timeout", "10000000", "-readrate", "1"]
            if fallback:
                args += ["-ss", f"{seek:.6f}"]
            args += ["-i", self._url(job, fallback)]
            if not fallback:
                args += ["-ss", f"{seek:.6f}"]
        args += ["-map", "0:v:0", "-an", "-sn", "-dn", "-t", f"{(job.end - job.start).total_seconds():.6f}",
                 "-vf", filters, "-c:v", "libx264",
                 "-pix_fmt", "yuv420p", "-preset", "veryfast", "-crf", "23", "-threads", "2",
                 "-force_key_frames", "expr:gte(t,n_forced*2)", "-f", "hls", "-hls_segment_type", "mpegts",
                 "-hls_time", "2", "-hls_list_size", "30", "-hls_flags", "temp_file+independent_segments",
                 "-hls_segment_filename", str(directory / "%06d.ts"), str(directory / "index.m3u8")]
        return args

    @staticmethod
    def _error_code(chunk):
        text = chunk.decode("utf-8", "replace").lower()
        statuses = re.findall(r"(?:method (?:describe|setup|play) failed:|server returned)\s*(\d{3})\b", text)
        if any(status in ("401", "403") for status in statuses):
            return "DEVICE_AUTH_FAILED"
        if re.search(r"\b(?:connection|operation|read|write) timed out\b|\bconnection timeout\b", text):
            return "DEVICE_TIMEOUT"
        if "connection refused" in text or "network is unreachable" in text or "no route to host" in text:
            return "DEVICE_UNREACHABLE"
        if re.search(r"(?:method (?:describe|setup|play) failed:|server returned)\s*(?:404|405|501)\b", text):
            return "ARCHIVE_UNSUPPORTED"
        if "unknown encoder" in text and "libx264" in text:
            return "LIBX264_UNAVAILABLE"
        return None

    def _drain(self, pipe, job, errors=False):
        carry = b""
        try:
            while chunk := pipe.read(8192):
                if errors:
                    carry = (carry + chunk)[-16384:]
                    code = self._error_code(carry)
                    if code and (job.process_error is None or code != "ARCHIVE_UNSUPPORTED"):
                        job.process_error = code
        except (OSError, ValueError):
            pass

    def _launch(self, job, fallback=False):
        if job.cancel_event.is_set():
            raise HTTPException(409, job.error or "PLAYBACK_CANCELLED")
        MediaProcessGuard.require()
        if job.captured and self._revision(self._camera(job.camera_id, True)) != self._revision(job.captured):
            raise HTTPException(409, "SOURCE_CHANGED")
        args = self._arguments(job, fallback)
        with self.storage.lock:
            self.storage.open_keys.update(f"{job.directory}/{name}" for name in ("index.m3u8", "index.m3u8.tmp", "input.ffconcat", "owner.json"))
        job.process_error = None
        job.process = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, close_fds=True)
        job.drainers = []
        for pipe, errors in ((job.process.stdout, False), (job.process.stderr, True)):
            thread = threading.Thread(target=self._drain, args=(pipe, job, errors), daemon=True, name="playback-pipe")
            thread.start()
            job.drainers.append(thread)

    def _stop_child(self, job):
        process = job.process
        if process is None:
            return
        # Drainers run concurrently throughout q/wait/terminate/kill.
        stop_process(process)
        for thread in job.drainers:
            thread.join(timeout=2)
        job.drainers.clear()

    def _decode(self, path):
        MediaProcessGuard.require()
        result = subprocess.run([self.ffmpeg, "-hide_banner", "-v", "error", "-i", str(path),
                                 "-map", "0:v:0", "-an", "-sn", "-dn", "-f", "null", "-"],
                                capture_output=True, timeout=10, close_fds=True)
        return result.returncode == 0 and not result.stderr

    def _publish(self, job):
        try:
            manifest = self.storage.path(f"{job.directory}/index.m3u8")
        except FileNotFoundError:
            return
        if manifest.stat().st_size > 65536:
            raise HTTPException(502, "PLAYBACK_INVALID_MANIFEST")
        lines = manifest.read_text("utf-8").splitlines()
        sequence, duration, entries = None, None, []
        for line in lines:
            if line.startswith("#EXT-X-MEDIA-SEQUENCE:"):
                sequence = int(line.split(":", 1)[1])
            elif line.startswith("#EXTINF:"):
                duration = float(line.split(":", 1)[1].split(",", 1)[0])
            elif line and not line.startswith("#"):
                match = TS_NAME.fullmatch(line)
                if (match is None or sequence is None or int(match[1]) != sequence
                        or duration is None or not math.isfinite(duration) or not 0 < duration <= 60):
                    raise HTTPException(502, "PLAYBACK_INVALID_MANIFEST")
                entries.append((sequence, duration, line))
                sequence += 1
                duration = None
        if not entries:
            return
        for sequence, duration, name in entries:
            if sequence in job.outputs:
                if abs(job.outputs[sequence].duration - duration) > .001:
                    raise HTTPException(502, "PLAYBACK_INVALID_MANIFEST")
                continue
            if sequence != len(job.outputs):
                raise HTTPException(502, "PLAYBACK_OUTPUT_GAP")
            key = f"{job.directory}/{name}"
            path = self.storage.path(key)
            if probe_video(path, self.ffprobe) is None:
                raise HTTPException(502, "PLAYBACK_INVALID_SEGMENT")
            if not job.outputs and not self._decode(path):
                raise HTTPException(502, "PLAYBACK_INVALID_SEGMENT")
            if not job.outputs and self.monotonic() - job.launched >= 30:
                raise HTTPException(504, "PLAYBACK_NO_PROGRESS")
            self.storage.register_cache(key, expires_at=self.now() + timedelta(seconds=120))
            lease = self.storage.cache_lease(key)
            lease.__enter__()
            output = Output(sequence, key, job.start + timedelta(seconds=job.total_duration), duration, lease)
            with self.lock:
                job.outputs[sequence] = output
                job.total_duration += duration
                job.last_progress = self.monotonic()
        listed = [sequence for sequence, _duration, _name in entries]
        now = self.monotonic()
        with self.lock:
            job.listed = listed
            for output in job.outputs.values():
                if output.sequence not in listed and output.retired is None:
                    output.retired = now
                if output.retired is not None and now - output.retired >= 30 and output.lease is not None:
                    output.lease.__exit__(None, None, None)
                    output.lease = None
                    self.storage.register_cache(output.key, expires_at=self.now())
            ended = job.process.poll() is not None
            if len(job.outputs) >= 2 or ended and job.total_duration >= (job.end - job.start).total_seconds() - .1:
                job.validated = True
                job.state = "READY"
        consumed = job.start + timedelta(seconds=job.total_duration)
        for item, lease, path in list(job.input_leases):
            if item.end <= consumed and len(job.input_leases) > 1:
                lease.__exit__(None, None, None)
                job.input_leases.remove((item, lease, path))

    def _cache_pressure(self):
        cache_available = self.storage.enforce_cache(CACHE_MAX_BYTES)
        total_available = self.storage.enforce()
        return not cache_available or not total_available

    def _work(self, job):
        job.launched = self.monotonic()
        job.last_progress = job.launched
        fallback = False
        end_state, end_error = "FAILED", None
        try:
            self._launch(job)
            while True:
                if job.cancel_event.is_set():
                    end_state = "CANCELLED"
                    break
                if self.monotonic() - job.touched >= 120:
                    end_state = "CANCELLED"
                    break
                self._publish(job)
                if self._cache_pressure():
                    raise HTTPException(503, "STORAGE_FULL")
                if job.process.poll() is not None:
                    for thread in job.drainers:
                        thread.join(timeout=2)
                    # Exit can follow an atomic final-manifest publication after
                    # the preceding scan. Re-read only once the child is reaped.
                    self._publish(job)
                    if self._cache_pressure():
                        raise HTTPException(503, "STORAGE_FULL")
                    if (job.source == "dahua" and not fallback and not job.outputs
                            and job.process_error == "ARCHIVE_UNSUPPORTED"):
                        self._stop_child(job)
                        fallback = True
                        self._launch(job, fallback=True)
                        continue
                    if not job.outputs:
                        raise HTTPException(502, job.process_error or "PLAYBACK_NO_VIDEO")
                    end_state = "ENDED" if job.validated else "FAILED"
                    if job.total_duration < (job.end - job.start).total_seconds() - .25 or job.process.returncode:
                        end_error = job.process_error or "PLAYBACK_PARTIAL"
                    break
                if self.monotonic() - job.launched > (job.end - job.start).total_seconds() + 60:
                    raise HTTPException(504, "PLAYBACK_DEADLINE")
                if self.monotonic() - job.last_progress >= 30:
                    raise HTTPException(504, "PLAYBACK_NO_PROGRESS")
                job.cancel_event.wait(.25)
        except HTTPException as exc:
            end_state, end_error = "FAILED", exc.detail if isinstance(exc.detail, str) else "PLAYBACK_FAILED"
        except Exception:
            end_state, end_error = "FAILED", "PLAYBACK_FAILED"
        finally:
            try:
                self._stop_child(job)
            finally:
                stopped = job.process is None or job.process.poll() is not None
                if stopped:
                    self._release_inputs(job)
                with self.lock:
                    if stopped:
                        job.error = job.error or end_error
                        timed_out = job.error in {"PLAYBACK_NO_PROGRESS", "PLAYBACK_DEADLINE"}
                        job.state = "FAILED" if timed_out else "CANCELLED" if job.cancel_event.is_set() else end_state
                        job.ended = self.monotonic()
                        self._retire_files(job)
                    else:
                        # Keep input/output leases and the root owner if an OS kill
                        # failed. Parent-death containment remains in force.
                        job.state, job.error = "FAILED", "MEDIA_STOP_PENDING"

    @staticmethod
    def _release_inputs(job):
        for _item, lease, _path in job.input_leases:
            lease.__exit__(None, None, None)
        job.input_leases.clear()

    def _retire_files(self, job):
        for output in job.outputs.values():
            if output.lease is not None:
                output.lease.__exit__(None, None, None)
                output.lease = None
        with self.storage.lock:
            self.storage.open_keys.difference_update(key for key in list(self.storage.open_keys) if key.startswith(job.directory + "/"))
            directory = self.storage.path(job.directory, directory=True)
            for path in directory.iterdir():
                if self._owned_name(path.name):
                    self.storage.register_cache(f"{job.directory}/{path.name}", expires_at=self.now() + timedelta(seconds=120))

    def manifest(self, identifier):
        with self.lock:
            job = self._get(identifier)
            if not job.validated or not job.listed:
                raise HTTPException(409, "PLAYBACK_NOT_READY")
            outputs = [job.outputs[sequence] for sequence in job.listed]
            lines = ["#EXTM3U", "#EXT-X-VERSION:3", "#EXT-X-INDEPENDENT-SEGMENTS",
                     f"#EXT-X-TARGETDURATION:{math.ceil(max(item.duration for item in outputs))}",
                     f"#EXT-X-MEDIA-SEQUENCE:{outputs[0].sequence}"]
            for item in outputs:
                try:
                    self.storage.path(item.key)
                except (OSError, ValueError):
                    raise HTTPException(410, "PLAYBACK_CACHE_EXPIRED") from None
                lines += [f"#EXT-X-PROGRAM-DATE-TIME:{iso(item.start)}", f"#EXTINF:{item.duration:.6f},",
                          f"/api/recordings/playbacks/{job.id}/segments/{item.sequence}.ts"]
            if job.state not in ACTIVE:
                lines.append("#EXT-X-ENDLIST")
            return "\n".join(lines) + "\n"

    def segment(self, identifier, sequence):
        with self.lock:
            job = self._get(identifier)
            output = job.outputs.get(sequence)
            if output is None or not job.validated:
                raise HTTPException(404, "PLAYBACK_SEGMENT_NOT_FOUND")
            try:
                return LeasedFileResponse(self.storage.cache_lease(output.key))
            except (OSError, ValueError, HTTPException):
                raise HTTPException(410, "PLAYBACK_CACHE_EXPIRED") from None

    def _expire(self, job):
        if job.state == "EXPIRED":
            return
        job.state = "EXPIRED"
        with self.storage.lock:
            for key in self.storage.cache:
                if key.startswith(job.directory + "/"):
                    self.storage.cache[key] = self.now()
        job.outputs.clear()
        job.listed.clear()
        job.inputs.clear()
        job.captured = job.interval = job.process = job.thread = None

    def _sweep(self):
        while not self.stop_event.wait(.5):
            overdue = []
            with self.lock:
                for job in self.jobs.values():
                    elapsed = self.monotonic()
                    if job.state in ACTIVE:
                        if elapsed - job.touched >= 120:
                            job.cancel_event.set()
                        elif job.process is not None and not job.cancel_event.is_set():
                            code = ("PLAYBACK_DEADLINE" if elapsed - job.launched > (job.end - job.start).total_seconds() + 60
                                    else "PLAYBACK_NO_PROGRESS" if elapsed - job.last_progress >= 30 else None)
                            if code:
                                job.error = code
                                job.cancel_event.set()
                                overdue.append(job.process)
                    elif job.ended is not None and elapsed - job.ended >= 120:
                        self._expire(job)
                self.searches = {key: value for key, value in self.searches.items() if self.monotonic() - value[0] < 60}
            # This watchdog is independent of ffprobe/decode waits. The worker
            # still drains, reaps and releases resources before publishing failure.
            for process in overdue:
                if process.poll() is None:
                    try:
                        process.kill()
                    except OSError:
                        pass
            self.storage.enforce()

    def shutdown(self):
        with self.lock:
            self.closed = True
            clients = list(self.search_clients)
            admission = self.admission
        for client in clients:
            client.cancel()
        if admission is not None:
            admission[2].wait()
        with self.lock:
            jobs = list(self.jobs.values())
            for job in jobs:
                job.cancel_event.set()
        for job in jobs:
            if job.thread and job.thread is not threading.current_thread():
                job.thread.join()
        self.stop_event.set()
        self.sweeper.join()
