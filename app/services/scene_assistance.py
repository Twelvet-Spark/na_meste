"""One CPU inference owner, one queued job, and bounded metadata-only history.

No database, reader validation or native operation runs with the job lock held.
Camera invalidation callbacks may therefore safely arrive during native work.
"""
from __future__ import annotations

import logging
import math
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from uuid import UUID, uuid4

from fastapi import HTTPException

from app.ai.scene_segmenter import SceneSegmentationError, validate_prompts

logger = logging.getLogger('app.scene.assistance')
JOB_TTL = 300
MAX_RECORDS = 16
TERMINAL = frozenset(('SUCCEEDED', 'FAILED', 'CANCELLED'))


def _error(status, code, message):
    return HTTPException(status, detail=dict(code=code, message=message))


def _identifier(value):
    try:
        if not isinstance(value, str):
            raise ValueError('Expected UUID string')
        return str(UUID(value))
    except ValueError as exc:
        raise _error(422, 'SCENE_INPUT_INVALID', 'Expected a UUID token') from exc


@dataclass(slots=True)
class _Job:
    job_id: str
    camera_id: int
    snapshot_id: str
    epoch: int
    expires_mono: float
    kind: str
    positive: list
    negative: list
    box: list | None
    state: str = 'QUEUED'
    polygons: list = field(default_factory=list)
    error_code: str | None = None


@dataclass(frozen=True, slots=True)
class SceneAssistanceCloseResult:
    closed: bool
    unresolved_running: int
    unresolved_queued: int
    error: str | None = None


class SceneAssistanceService:
    def __init__(self, scene_service, segmenter=None, *, monotonic=time.monotonic):
        self.scene_service = scene_service
        self.segmenter = segmenter
        self._monotonic = monotonic
        self._condition = threading.Condition()
        self._jobs = OrderedDict()
        self._epochs = {}
        self._running = None
        self._queued = None
        self._closed = False
        if segmenter is not None:
            segmenter.set_snapshot_validator(scene_service.get_snapshot)
        scene_service.add_invalidation_callback(self.invalidate)
        self._thread = threading.Thread(target=self._work, name='scene-assistance', daemon=True)
        self._thread.start()

    @staticmethod
    def _view(job):
        # Never expose the mutable record (or a saved native image) to a caller.
        return dict(job_id=job.job_id, snapshot_id=job.snapshot_id, state=job.state,
                    polygons=[dict(points=[list(point) for point in polygon['points']],
                                   score=polygon['score'], needs_review=True) for polygon in job.polygons],
                    error_code=job.error_code)

    def _purge_locked(self):
        now = self._monotonic()
        expired = [key for key, job in self._jobs.items() if now >= job.expires_mono]
        for key in expired:
            job = self._jobs[key]
            job.state = 'CANCELLED'
            job.polygons = []
            if self._queued == key:
                self._queued = None
            if key != self._running:
                del self._jobs[key]
        # A retired running record remains counted (but invisible) until its
        # native work exits, so cancellation cannot turn 16 records into 17.

    def _lookup_locked(self, camera_id, job_id):
        self._purge_locked()
        job = self._jobs.get(job_id)
        if job is None or self._monotonic() >= job.expires_mono:
            raise _error(410, 'SCENE_JOB_EXPIRED', 'Job is unknown or expired')
        if job.camera_id != camera_id:
            raise _error(409, 'SCENE_SNAPSHOT_MISMATCH', 'Job belongs to another camera')
        return job

    def _capacity_locked(self, *, evict=False):
        if self._closed:
            raise _error(503, 'SCENE_BUSY', 'Scene assistance is shutting down')
        if self._queued is not None:
            raise _error(503, 'SCENE_BUSY', 'One native request and one queued request are allowed')
        if not evict:
            return
        while len(self._jobs) >= MAX_RECORDS:
            oldest = next((key for key, item in self._jobs.items()
                           if key != self._running and item.state in TERMINAL), None)
            if oldest is None:
                raise _error(503, 'SCENE_BUSY', 'Scene job history is full')
            del self._jobs[oldest]

    def submit(self, camera_id, payload):
        if self.segmenter is None:
            raise _error(503, 'SCENE_MODEL_UNAVAILABLE', 'Selected MobileSAM model is unavailable; manual editing remains available')
        if not isinstance(payload, dict) or set(payload) != {
                'snapshot_id', 'kind', 'positive_points', 'negative_points', 'box'}:
            raise _error(422, 'SCENE_INPUT_INVALID', 'Invalid suggestion request fields')
        if payload['kind'] not in ('floor', 'object'):
            raise _error(422, 'SCENE_INPUT_INVALID', 'Choose floor or object explicitly')
        identifier = _identifier(payload['snapshot_id'])
        try:
            positive, negative, box = validate_prompts(payload['positive_points'], payload['negative_points'], payload['box'])
        except ValueError as exc:
            raise _error(422, 'SCENE_INPUT_INVALID', str(exc)) from exc
        with self._condition:
            self._purge_locked()
            self._capacity_locked()
            epoch = self._epochs.get(camera_id, 0)
        # This is validation only. A native reference is acquired by the worker,
        # never stored in admission/history records or in the pending queue.
        self.scene_service.get_snapshot(identifier, camera_id)
        with self._condition:
            self._purge_locked()
            if self._epochs.get(camera_id, 0) != epoch:
                raise _error(409, 'CAMERA_SOURCE_CHANGED', 'Camera changed during suggestion admission')
            self._capacity_locked(evict=True)
            job_id = str(uuid4())
            job = _Job(job_id, camera_id, identifier, epoch, self._monotonic() + JOB_TTL,
                       payload['kind'], positive, negative, box)
            self._jobs[job_id] = job
            self._queued = job_id
            self._condition.notify()
            return dict(job_id=job_id, snapshot_id=identifier, state='QUEUED')

    def _validated_job(self, camera_id, job_id):
        identifier = _identifier(job_id)
        with self._condition:
            job = self._lookup_locked(camera_id, identifier)
            token, epoch = job.snapshot_id, job.epoch
        # Source, reader identity, TTL and camera are checked even on terminal
        # results and DELETE replay; old outlines never survive a source change.
        try:
            self.scene_service.get_snapshot(token, camera_id)
        except HTTPException:
            with self._condition:
                if job.state not in TERMINAL:
                    job.state = 'CANCELLED'
                    job.polygons = []
                    if self._queued == identifier:
                        self._queued = None
            raise
        with self._condition:
            job = self._lookup_locked(camera_id, identifier)
            if self._epochs.get(camera_id, 0) != epoch:
                raise _error(409, 'CAMERA_SOURCE_CHANGED', 'Camera changed during job validation')
        return job

    def get(self, camera_id, job_id):
        job = self._validated_job(camera_id, job_id)
        with self._condition:
            if self._epochs.get(camera_id, 0) != job.epoch:
                raise _error(409, 'CAMERA_SOURCE_CHANGED', 'Camera changed before result publication')
            return self._view(self._lookup_locked(camera_id, job.job_id))

    def cancel(self, camera_id, job_id):
        job = self._validated_job(camera_id, job_id)
        with self._condition:
            job = self._lookup_locked(camera_id, job.job_id)
            if self._epochs.get(camera_id, 0) != job.epoch:
                raise _error(409, 'CAMERA_SOURCE_CHANGED', 'Camera changed before cancellation')
            if job.state not in TERMINAL:
                job.state = 'CANCELLED'
                job.polygons = []
                if self._queued == job.job_id:
                    self._queued = None
                self._condition.notify_all()
            return self._view(job)

    def invalidate(self, camera_id):
        with self._condition:
            self._epochs[camera_id] = self._epochs.get(camera_id, 0) + 1
            for job in self._jobs.values():
                if job.camera_id == camera_id:
                    job.polygons = []
                    if job.state not in TERMINAL:
                        job.state = 'CANCELLED'
                        if self._queued == job.job_id:
                            self._queued = None
            self._condition.notify_all()
        if self.segmenter is not None:
            self.segmenter.invalidate(camera_id)

    def _admissible_locked(self, job):
        return (not self._closed and self._jobs.get(job.job_id) is job
                and job.state == 'RUNNING' and self._monotonic() < job.expires_mono
                and self._epochs.get(job.camera_id, 0) == job.epoch)

    def _execute(self, job):
        # The context reservation includes retired/expired native references.
        # It exits on every exception/cancellation, after native work has ended.
        with self.scene_service.lease_snapshot(job.snapshot_id, job.camera_id) as snapshot:
            with self._condition:
                if not self._admissible_locked(job):
                    return
            self.scene_service.get_snapshot(job.snapshot_id, job.camera_id)
            polygons = self.segmenter.suggest(snapshot, job.positive, job.negative, job.box)
            self.scene_service.get_snapshot(job.snapshot_id, job.camera_id)
            serialized = [dict(points=[list(point) for point in item.points],
                               score=item.score, needs_review=True) for item in polygons]
            with self._condition:
                if self._admissible_locked(job):
                    job.polygons = serialized
                    job.state = 'SUCCEEDED'

    def _work(self):
        while True:
            if self.segmenter is not None:
                self.segmenter.purge_expired()
            with self._condition:
                self._purge_locked()
                if self._queued is None and not self._closed:
                    # Wake periodically so an idle service also retires TTL records.
                    self._condition.wait(timeout=1.)
                    self._purge_locked()
                if self._closed:
                    return
                if self._queued is None:
                    continue
                identifier, self._queued = self._queued, None
                job = self._jobs.get(identifier)
                if job is None or job.state != 'QUEUED':
                    continue
                self._running = identifier
                job.state = 'RUNNING'
            try:
                self._execute(job)
            except Exception as exc:
                if isinstance(exc, HTTPException):
                    code = exc.detail.get('code', 'SCENE_INFERENCE_FAILED') if isinstance(exc.detail, dict) else 'SCENE_INFERENCE_FAILED'
                elif isinstance(exc, SceneSegmentationError):
                    code = exc.code
                else:
                    code = 'SCENE_INFERENCE_FAILED'
                    logger.exception('Scene model inference failed')
                with self._condition:
                    if self._admissible_locked(job):
                        job.state = 'FAILED'
                        job.error_code = code
                        job.polygons = []
            finally:
                # Drop the local job reference before waiting; records contain
                # only bounded prompts/results, never FrozenSceneSnapshot frames.
                with self._condition:
                    self._running = None
                    self._purge_locked()
                    self._condition.notify_all()
                    closed = self._closed
                if closed and self.segmenter is not None:
                    # Close may clear before a just-admitted native call starts.
                    # Clear again once it finishes, without holding the job lock.
                    self.segmenter.clear_cache()
                del job

    def health_snapshot(self):
        with self._condition:
            return dict(running=int(self._running is not None),
                        queued=int(self._queued is not None), records=len(self._jobs),
                        accepting=not self._closed, worker_alive=self._thread.is_alive())

    def close(self, timeout_seconds=5):
        if (type(timeout_seconds) not in (int, float) or not math.isfinite(timeout_seconds)
                or timeout_seconds < 0):
            raise ValueError('close timeout must be finite and nonnegative')
        with self._condition:
            self._closed = True
            self._queued = None
            for job in self._jobs.values():
                if job.state not in TERMINAL:
                    job.state = 'CANCELLED'
                    job.polygons = []
            self._condition.notify_all()
        if self.segmenter is not None:
            self.segmenter.clear_cache()
        self._thread.join(min(float(timeout_seconds), 5.))
        with self._condition:
            unresolved = int(self._running is not None)
            stopped = not self._thread.is_alive()
        return SceneAssistanceCloseResult(stopped, unresolved, 0,
                                           None if stopped else 'SCENE_NATIVE_WORK_UNRESOLVED')
