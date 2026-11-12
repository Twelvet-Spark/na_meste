"""Dedicated video-only stream-copy recorder and durable segment indexer.

Wall time is approximate SERVER recording time, never camera exposure time.
Only a closed/probed file may become READY. Probe work never blocks the 250ms
recording clock/stop loop; active files stay protected until finalization.
"""
import csv
import io
import json
import math
import re
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path
from uuid import uuid4

from sqlalchemy.orm import Session

from app.config import settings
from app.database.database import immediate_session
from app.database.models import RecordingConnection, RecordingSegment, RecordingSession
from app.media.process_guard import MediaProcessGuard
from app.rtsp.dahua import build_rtsp_url

SEGMENT_NAME = re.compile(r"segment-[0-9]{6}\.mkv\Z")


def probe_video(path, ffprobe):
    MediaProcessGuard.require()
    try:
        result = subprocess.run([
            ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries",
            "stream=codec_type,codec_name,width,height,duration:format=duration,size", "-of", "json", str(path),
        ], capture_output=True, timeout=10, close_fds=True)
        if result.returncode or result.stderr:
            return None
        data = json.loads(result.stdout)
        stream = data["streams"][0]
        duration = float(data.get("format", {}).get("duration", stream.get("duration", 0)))
        if (stream.get("codec_name") not in ("h264", "hevc") or not math.isfinite(duration)
                or duration <= 0 or duration > 86400 or int(stream.get("width", 0)) <= 0
                or int(stream.get("height", 0)) <= 0):
            return None
        return max(1, round(duration * 1000)), stream["codec_name"]
    except (OSError, subprocess.SubprocessError, ValueError, KeyError, TypeError, IndexError):
        return None


def stop_process(process):
    """No recovered PIDs: only the Popen object created by this process is owned."""
    if process is None:
        return
    if process.poll() is None:
        try:
            process.stdin.write(b"q\n")
            process.stdin.flush()
        except (OSError, ValueError, AttributeError):
            pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)
    for pipe in (process.stdin, process.stdout, process.stderr):
        if pipe:
            try:
                pipe.close()
            except (OSError, ValueError):
                pass


class SegmentRecorder:
    def __init__(self, engine, storage, connection_id, ffmpeg, ffprobe, *, now, monotonic):
        self.engine, self.storage = engine, storage
        self.connection_id, self.ffmpeg, self.ffprobe = connection_id, ffmpeg, ffprobe
        self.now, self.monotonic = now, monotonic
        with Session(engine) as db:
            connection = db.get(RecordingConnection, connection_id)
            self.directory = connection.directory
            self.session_id = connection.session_id
            self.last_progress_utc = connection.created_at
        self.process = None
        self.drainers = []
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="media-probe")
        self.pending = {}
        self.indexed = set()
        self.observed_sizes = {}
        self.last_progress = monotonic()
        self.first_output_mono = None
        self.closed = False

    def start(self, source, remaining_seconds):
        directory = self.storage.path(self.directory, directory=True)
        MediaProcessGuard.require()
        url = build_rtsp_url(source.nvr_host, source.rtsp_port, source.username, source.password,
                             source.channel, source.stream_type)
        args = [self.ffmpeg, "-hide_banner", "-loglevel", "error", "-rtsp_transport", "tcp",
                "-timeout", str(settings.rtsp_connect_timeout * 1000000), "-i", url,
                "-map", "0:v:0", "-an", "-sn", "-dn", "-c:v", "copy",
                "-t", str(max(0.001, remaining_seconds)), "-f", "segment", "-segment_format", "matroska",
                "-segment_time", "60", "-reset_timestamps", "1", "-segment_list_type", "csv",
                "-segment_format_options", "flush_packets=1",
                "-segment_list", str(directory / "closed.csv"), str(directory / "segment-%06d.mkv")]
        self.process = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE, close_fds=True)
        for pipe in (self.process.stdout, self.process.stderr):
            thread = threading.Thread(target=self._drain, args=(pipe,), daemon=True, name="media-pipe")
            thread.start()
            self.drainers.append(thread)

    @staticmethod
    def _drain(pipe):
        try:
            while pipe.read(8192):
                pass  # Never buffer/log credential-bearing FFmpeg stderr.
        except (OSError, ValueError):
            pass

    def _csv(self, directory):
        try:
            path = self.storage.path(f"{self.directory}/closed.csv")
            with path.open("rb") as handle:
                data = handle.read(1024 * 1024 + 1)
            if len(data) > 1024 * 1024:
                return {}
            # An incomplete final line is still being written, not a closed segment.
            data = data[:data.rfind(b"\n") + 1]
            result = {}
            for row in csv.reader(io.StringIO(data.decode("utf-8"))):
                if len(row) != 3:
                    continue
                name = Path(row[0]).name
                if not SEGMENT_NAME.fullmatch(name):
                    continue
                if row[0] != name and Path(row[0]).absolute() != directory / name:
                    continue
                start, end = float(row[1]), float(row[2])
                if not math.isfinite(start) or not math.isfinite(end) or end <= start or end - start > 86400:
                    continue
                result[name] = (start, end)
            return result
        except (OSError, ValueError, UnicodeError, csv.Error):
            return {}

    def scan(self, *, final=False, recovering=False):
        directory = self.storage.path(self.directory, directory=True)
        closed = self._csv(directory)
        paths = []
        for candidate in sorted(directory.iterdir()):
            if SEGMENT_NAME.fullmatch(candidate.name):
                key = f"{self.directory}/{candidate.name}"
                try:
                    safe = self.storage.path(key)
                    paths.append((key, safe, safe.stat().st_size))
                except (OSError, ValueError):
                    continue
        sizes = {key: size for key, _path, size in paths}
        if any(size > self.observed_sizes.get(key, 0) for key, size in sizes.items()):
            self.last_progress = self.monotonic()
            self.last_progress_utc = self.now()
        self.observed_sizes = sizes
        observed = self.now()
        with self.storage.lock:
            with immediate_session(self.engine) as db:
                connection = db.get(RecordingConnection, self.connection_id)
                session = db.get(RecordingSession, self.session_id)
                if paths and connection.anchor_utc is None and not recovering:
                    connection.anchor_utc = observed
                    self.first_output_mono = self.monotonic()
                if closed and connection.media_origin_seconds is None:
                    connection.media_origin_seconds = next(iter(closed.values()))[0]
                for key, path, size in paths:
                    if key in self.indexed:
                        continue
                    row = db.query(RecordingSegment).filter_by(storage_key=key).one_or_none()
                    if row is None:
                        row = RecordingSegment(id=str(uuid4()), session_id=session.id,
                            connection_id=connection.id, camera_id=session.camera_id,
                            source_revision=connection.source_revision, storage_key=key,
                            started_at=None if recovering else observed, state="UNINDEXED", bytes=size)
                        db.add(row)
                    elif row.state in ("EXPIRED", "DELETING"):
                        continue
                    row.bytes = size
                    if not recovering and key not in self.indexed:
                        self.storage.open_keys.add(key)
                    if session.started_at is None and not recovering:
                        session.started_at = observed
                    if session.status in ("STARTING", "RECONNECTING") and not recovering:
                        session.status = "RECORDING"
                    if key not in self.indexed and key not in self.pending and (path.name in closed or final):
                        # Snapshot the trusted mapping before publishing. Never use mtime.
                        start = row.started_at
                        if (path.name in closed and connection.anchor_utc is not None
                                and connection.media_origin_seconds is not None):
                            start = connection.anchor_utc + timedelta(
                                seconds=closed[path.name][0] - connection.media_origin_seconds)
                        end = None
                        if (path.name in closed and connection.anchor_utc is not None
                                and connection.media_origin_seconds is not None):
                            end = connection.anchor_utc + timedelta(
                                seconds=closed[path.name][1] - connection.media_origin_seconds)
                        self.pending[key] = (self.executor.submit(probe_video, path, self.ffprobe), start, end)
        self._publish(wait=final)
        if recovering:
            present = {key for key, _path, _size in paths}
            with immediate_session(self.engine) as db:
                for row in db.query(RecordingSegment).filter_by(connection_id=self.connection_id).all():
                    if row.storage_key not in present and row.state not in ("EXPIRED", "DELETING"):
                        row.state = "MISSING"

    def _publish(self, *, wait=False):
        for key, (future, start, end) in list(self.pending.items()):
            if not wait and not future.done():
                continue
            try:
                result = future.result(timeout=15)
            except Exception:
                result = None
            with self.storage.lock:
                with immediate_session(self.engine) as db:
                    row = db.query(RecordingSegment).filter_by(storage_key=key).one()
                    if row.state not in ("EXPIRED", "DELETING"):
                        if result is None:
                            row.state = "CORRUPT"
                        else:
                            duration, codec = result
                            if start is not None and end is not None:
                                duration = max(1, round((end - start).total_seconds() * 1000))
                            row.duration_ms, row.codec = duration, codec
                            row.started_at = start
                            row.ended_at = end or (start + timedelta(milliseconds=duration) if start else None)
                            row.state = "READY" if start else "UNINDEXED"
                self.storage.open_keys.discard(key)
            self.indexed.add(key)
            del self.pending[key]

    def finish(self, *, recovering=False):
        if self.closed:
            return
        try:
            stop_process(self.process)
            for thread in self.drainers:
                thread.join(timeout=2)
            self.scan(final=True, recovering=recovering)
        finally:
            if self.process is None or self.process.poll() is not None:
                self.executor.shutdown(wait=True, cancel_futures=True)
                with self.storage.lock:
                    self.storage.open_keys.difference_update(
                        key for key in tuple(self.storage.open_keys) if key.startswith(self.directory + "/"))
                self.closed = True
        with immediate_session(self.engine) as db:
            connection = db.get(RecordingConnection, self.connection_id)
            if connection.closed_at is None:
                connection.closed_at = self.now()
