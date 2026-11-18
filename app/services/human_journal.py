"""Durable local human visits, independent of face/body inference.

One writer owns all transactions. Producers never wait for SQLite: a lifecycle
FIFO, latest-per-visit checkpoints and reserved generation barriers share one
monotonic admission order. A barrier snapshots the *accepted* cutoff, not the
failed observation. It cannot overtake any accepted command or checkpoint.

Queue limits include in-flight commands. Visit/ack memory is released only after
closure commits and the consumer drains its acknowledgements. No acknowledged-ID
history grows with uptime; tracker IDs are monotonic within each generation.
"""
from __future__ import annotations

from collections import OrderedDict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import threading
import time
from uuid import NAMESPACE_URL, UUID, uuid5

import cv2
import numpy as np
from sqlalchemy import func, select, update

from app.database.database import immediate_session
from app.database.models import (
    AttendanceCamera, Camera, GlobalPerson, HumanJournalGap, HumanVisit, HumanVisitEvent,
)
from app.time_utils import utcnow

_NAMESPACE = uuid5(NAMESPACE_URL, 'na_meste/human-journal/v1')
_IMAGE_BYTES = 256 * 1024
_CLEAR_REASONS = frozenset({
    'GALLERY_CHANGED', 'GALLERY_UNAVAILABLE', 'FACE_MODEL_UNAVAILABLE',
    'ASSOCIATION_UNCERTAIN', 'IDENTITY_CONFLICT',
})


def _id(parts) -> str:
    return str(uuid5(_NAMESPACE, json.dumps(parts, separators=(',', ':'), ensure_ascii=True)))


def _utc(value: datetime) -> datetime:
    return value if value.tzinfo is None else value.astimezone(timezone.utc).replace(tzinfo=None)


def _reason(value: str) -> str:
    # Static codes only; never persist an exception message, credentials or names.
    if not isinstance(value, str) or not value or len(value) > 64:
        raise ValueError('Invalid journal reason')
    if any(char not in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_' for char in value):
        raise ValueError('Invalid journal reason')
    return value


@dataclass(frozen=True, slots=True)
class JournalAdmission:
    visit_id: str
    camera_id: int
    tracking_generation: str
    track_id: int
    status: str
    source_revision: int
    stream_id: str
    packet_seq: int


@dataclass(frozen=True, slots=True)
class JournalAck:
    kind: str
    camera_id: int
    tracking_generation: str
    visit_id: str | None = None
    track_id: int | None = None
    source_revision: int | None = None
    stream_id: str | None = None
    packet_seq: int | None = None
    global_id: int | None = None
    link_key: str | None = None
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class JournalCloseResult:
    drained: bool
    unresolved_commands: int
    unresolved_updates: int
    unresolved_barriers: int
    error_code: str | None
    thread_alive: bool = False


@dataclass(slots=True)
class _Visit:
    values: dict
    packet_seq: int
    last_mono: float
    checkpoint_mono: float
    image_mono: float
    hits: int = 1
    transition_seq: int = 0
    lost: bool = False
    closed: bool = False
    committed: bool = False
    closed_durable: bool = False
    person_quality: float = -1.
    face_rank: tuple = ()
    link_pending: str | None = None


@dataclass(slots=True)
class _Generation:
    camera_id: int
    generation: str
    source_revision: int
    stream_id: str
    visits: set = field(default_factory=set)
    max_track_id: int = -1
    last_packet_seq: int = -1
    retiring: bool = False
    blocked: bool = False
    barrier_done: bool = False
    gap_started_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class _Command:
    order: int
    kind: str
    generation: tuple | None = None
    visit_id: str | None = None
    values: dict | None = None
    events: tuple = ()
    packet_seq: int | None = None
    payload: object = None


class HumanJournal:
    def __init__(self, engine, *, min_hits=2, lost_timeout=3., image_keep_days=30,
                 now=utcnow, monotonic=time.monotonic, lifecycle_capacity=4096,
                 active_capacity=4096, control_capacity=4096):
        if (isinstance(min_hits, bool) or min_hits < 1 or not math.isfinite(lost_timeout)
                or lost_timeout <= 0 or not 1 <= image_keep_days <= 365):
            raise ValueError('Invalid journal policy')
        if any(isinstance(n, bool) or not isinstance(n, int) or not 1 <= n <= 4096
               for n in (lifecycle_capacity, active_capacity, control_capacity)):
            raise ValueError('Invalid journal capacity')
        self.engine = engine
        self.min_hits = min_hits
        self.lost_timeout = lost_timeout
        self.image_keep_days = image_keep_days
        self._now, self._monotonic = now, monotonic
        self._lifecycle_capacity = lifecycle_capacity
        self._active_capacity = active_capacity
        self._control_capacity = control_capacity
        self._condition = threading.Condition(threading.RLock())
        self._commands = deque()
        self._updates = OrderedDict()
        self._barriers = OrderedDict()
        self._visits: dict[str, _Visit] = {}
        self._generations: dict[tuple, _Generation] = {}
        self._current: dict[int, tuple] = {}
        self._retired_generations = OrderedDict()
        self._blocked: dict[int, tuple] = {}
        self._acks: dict[tuple, JournalAck] = {}
        self._controls: dict[tuple, JournalAck] = {}
        self._order = 0
        self._startup = True
        self._purge_pending = False
        self._last_purge = float('-inf')
        self._next_tick_at = self._monotonic() + 1.
        self._inflight: _Command | None = None
        self._error: str | None = None
        self._capacity_degraded = False
        self._retry_at = 0.
        self._accepting = True
        self._closing = False
        self._stop_deadline: float | None = None
        self._thread = threading.Thread(target=self._run, name='human-journal-writer', daemon=True)
        self._thread.start()

    def _next_order(self):
        self._order += 1
        return self._order

    def _lifecycle_full(self):
        return (len(self._commands) + int(self._inflight is not None and
                self._inflight.kind in ('VISIT', 'LINK'))) >= self._lifecycle_capacity

    @staticmethod
    def _event(visit, kind, occurred_at, payload=None, *, dedupe_key=None):
        visit.transition_seq += 1
        identifier = _id([visit.values['id'], kind, visit.transition_seq])
        body = {'version': 1, 'transition_seq': visit.transition_seq, **(payload or {})}
        # Internal payloads are scalar and bounded; keep the persisted invariant
        # explicit rather than relying on backend-specific CHECK constraints.
        if len(json.dumps(body, allow_nan=False).encode()) > 16384:
            raise ValueError('Journal event payload exceeds bound')
        return dict(id=identifier, visit_id=visit.values['id'], type=kind,
                    occurred_at=occurred_at, dedupe_key=dedupe_key or identifier, payload=body)

    def _admission(self, camera_id, generation, track_id, packet, status):
        return JournalAdmission(_id([camera_id, generation, track_id]), camera_id,
                                generation, track_id, status, packet.source_revision,
                                packet.stream_id, packet.seq)

    def _generation(self, camera_id, generation, packet):
        key = (camera_id, generation)
        existing = self._generations.get(key)
        if existing is not None:
            if existing.retiring:
                return existing
            if (existing.source_revision != packet.source_revision or
                    existing.stream_id != packet.stream_id):
                self._block(existing)
                return None
            return existing
        if camera_id in self._blocked or self._error is not None:
            return None
        if self._retired_generations.get(camera_id) == generation:
            return None
        previous_key = self._current.get(camera_id)
        previous = self._generations.get(previous_key)
        if previous is not None and not previous.retiring:
            self._barrier(previous, 'GENERATION_CHANGED', blocked=False)
        if len(self._generations) >= self._control_capacity:
            self._capacity_degraded = True
            return None
        result = _Generation(camera_id, generation, packet.source_revision, packet.stream_id)
        self._generations[key] = result
        self._current[camera_id] = key
        self._capacity_degraded = False
        return result

    def observe(self, camera_id, packet, outcome) -> list[JournalAdmission]:
        generation = str(outcome.tracking_generation)
        UUID(generation)
        if packet.source_revision < 1 or not math.isfinite(packet.received_mono):
            raise ValueError('Invalid observation token')
        observed_at = _utc(packet.received_at)
        people = outcome.observed_tracks
        with self._condition:
            gen = self._generation(camera_id, generation, packet) if self._accepting else None
            if gen is None or gen.retiring or camera_id in self._blocked:
                return [self._admission(camera_id, generation, p.track_id, packet, 'REJECTED')
                        for p in people]
            if packet.seq <= gen.last_packet_seq:
                return [self._admission(camera_id, generation, p.track_id, packet,
                        'COMMITTED' if (v := self._visits.get(_id([camera_id, generation, p.track_id])))
                        is not None and v.committed else 'REJECTED') for p in people]
            gen.last_packet_seq = packet.seq
            admissions = []
            # Lifecycle invalidation precedes fresh observations in the same frame.
            for transition in outcome.track_lifecycle:
                visit = self._visits.get(_id([camera_id, generation, transition.track_id]))
                if visit is None or visit.closed or gen.retiring:
                    continue
                if transition.state == 'EXPIRED':
                    self._close_visit(gen, visit, _reason(transition.reason or 'TRACK_EXPIRED'))
                elif transition.state in ('LOST', 'AMBIGUOUS'):
                    self._lose(gen, visit, observed_at, ambiguous=transition.state == 'AMBIGUOUS')
            seen = set()
            previous_max_track_id = gen.max_track_id
            for person in people:
                if person.track_id in seen:
                    continue
                seen.add(person.track_id)
                if gen.retiring:
                    admissions.append(self._admission(camera_id, generation, person.track_id, packet, 'REJECTED'))
                    continue
                identifier = _id([camera_id, generation, person.track_id])
                visit = self._visits.get(identifier)
                if visit is None:
                    if person.track_id <= previous_max_track_id:
                        admissions.append(self._admission(camera_id, generation, person.track_id, packet, 'REJECTED'))
                        continue
                    if len(self._visits) >= self._active_capacity or self._lifecycle_full():
                        self._block(gen)
                        admissions.append(self._admission(camera_id, generation, person.track_id, packet, 'REJECTED'))
                        continue
                    values = dict(id=identifier, camera_id=camera_id, tracking_generation=generation,
                                  track_id=person.track_id, source_revision=packet.source_revision,
                                  stream_id=packet.stream_id, started_at=observed_at,
                                  last_seen_at=observed_at, ended_at=None, end_reason=None,
                                  status='ACTIVE' if self.min_hits == 1 else 'PROVISIONAL',
                                  employee_id=None, identity_state='UNKNOWN', global_id=None,
                                  last_bbox=self._bbox(person.bbox), max_person_score=self._score(person.person_score),
                                  snapshot=None, face_thumbnail=None, face_quality=None,
                                  image_expires_at=observed_at + timedelta(days=self.image_keep_days),
                                  images_purged_at=None)
                    visit = _Visit(values, packet.seq, packet.received_mono,
                                   packet.received_mono, float('-inf'))
                    self._visits[identifier] = visit
                    gen.visits.add(identifier)
                    gen.max_track_id = max(gen.max_track_id, person.track_id)
                    events = [self._event(visit, 'FIRST_SEEN', observed_at)]
                    events.extend(self._identity(visit, person, packet))
                    self._capture(visit, person, packet)
                    self._enqueue_visit(gen, visit, events)
                else:
                    if (visit.closed or packet.received_mono <= visit.last_mono
                            or observed_at < visit.values['last_seen_at']):
                        admissions.append(self._admission(camera_id, generation, person.track_id, packet, 'REJECTED'))
                        continue
                    # Work on a candidate until all necessary lifecycle capacity
                    # is reserved. Failed transitions must not advance the cutoff.
                    candidate = _Visit(dict(visit.values), packet.seq, packet.received_mono,
                                       visit.checkpoint_mono, visit.image_mono, visit.hits + 1,
                                       visit.transition_seq, False, False, visit.committed,
                                       False, visit.person_quality, visit.face_rank, visit.link_pending)
                    candidate.lost = bool(person.association_uncertain)
                    candidate.values.update(last_seen_at=observed_at, last_bbox=self._bbox(person.bbox),
                        max_person_score=max(visit.values['max_person_score'], self._score(person.person_score)))
                    events = []
                    if visit.lost:
                        candidate.hits = 1
                        if not candidate.lost:
                            events.append(self._event(candidate, 'RECOVERED', observed_at))
                    if candidate.hits >= self.min_hits:
                        candidate.values['status'] = 'ACTIVE'
                    events.extend(self._identity(candidate, person, packet))
                    transition = bool(events) or candidate.values['status'] != visit.values['status']
                    if transition and self._lifecycle_full():
                        self._block(gen)
                        admissions.append(self._admission(camera_id, generation, person.track_id, packet, 'REJECTED'))
                        continue
                    image_changed = self._capture(candidate, person, packet)
                    self._visits[identifier] = candidate
                    visit = candidate
                    if transition:
                        self._enqueue_visit(gen, visit, events)
                    elif image_changed or packet.received_mono - visit.checkpoint_mono >= 2.:
                        visit.checkpoint_mono = packet.received_mono
                        self._updates.pop(identifier, None)
                        self._updates[identifier] = _Command(self._next_order(), 'UPDATE',
                            (camera_id, generation), identifier, dict(visit.values), packet_seq=packet.seq)
                admissions.append(self._admission(camera_id, generation, person.track_id, packet,
                                  'COMMITTED' if visit.committed else 'PENDING'))
            self._condition.notify_all()
            return admissions

    @staticmethod
    def _bbox(value):
        result = [float(n) for n in value]
        if (len(result) != 4 or not all(math.isfinite(n) and 0 <= n <= 1 for n in result)
                or result[0] >= result[2] or result[1] >= result[3]):
            raise ValueError('Invalid observed person box')
        return result

    @staticmethod
    def _score(value):
        value = float(value)
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError('Invalid observed person score')
        return value

    @staticmethod
    def _fresh_face(person, packet):
        evidence = person.fresh_face_evidence
        if evidence is None or person.association_uncertain:
            return None
        quality = evidence.quality
        if (evidence.frame_seq != packet.seq or evidence.received_mono != packet.received_mono
                or _utc(evidence.received_at) != _utc(packet.received_at)
                or not quality.aligned or quality.blur is None
                or not all(math.isfinite(float(v)) for v in
                           (quality.min_size, quality.det_score, quality.blur))
                or quality.min_size < 48 or quality.det_score < .8 or quality.blur < 20
                or any(reason in ('AMBIGUOUS_FACE', 'ASSOCIATION_UNCERTAIN', 'BAD_ALIGNMENT',
                                  'FACE_MODEL_MISMATCH', 'GALLERY_CHANGED')
                       for reason in evidence.reason_codes)):
            return None
        return evidence

    def _identity(self, visit, person, packet):
        evidence = self._fresh_face(person, packet)
        events = []
        clear = person.association_uncertain or bool(_CLEAR_REASONS.intersection(person.identity_reason_codes))
        old = visit.values['employee_id']
        confirmed = (evidence.confirmed_employee_id if evidence is not None
                     and not evidence.reason_codes and not clear else None)
        if confirmed is not None and person.employee_id == confirmed:
            if old is not None and old != confirmed:
                visit.values.update(employee_id=None, identity_state='CONFLICT')
                events.append(self._event(visit, 'IDENTITY_CONFLICT', _utc(packet.received_at)))
            elif visit.values['identity_state'] != 'CONFLICT' and old != confirmed:
                visit.values.update(employee_id=confirmed, identity_state='FACE_CONFIRMED')
                events.append(self._event(visit, 'IDENTIFIED', _utc(packet.received_at),
                                          {'employee_id': confirmed}))
        elif clear or (old is not None and person.employee_id != old):
            if old is not None or visit.values['identity_state'] == 'FACE_CONFIRMED':
                visit.values.update(employee_id=None, identity_state='CONFLICT' if clear else 'UNKNOWN')
                events.append(self._event(visit, 'IDENTITY_CONFLICT', _utc(packet.received_at)))
        return events

    @staticmethod
    def _jpeg(image):
        if (not isinstance(image, np.ndarray) or image.dtype != np.uint8
                or image.ndim != 3 or image.shape[2] != 3 or not image.size):
            return None
        height, width = image.shape[:2]
        if max(height, width) > 640:
            scale = 640 / max(height, width)
            image = cv2.resize(image, (max(1, int(width * scale)), max(1, int(height * scale))),
                               interpolation=cv2.INTER_AREA)
        for quality in (85, 65, 45, 25):
            ok, encoded = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, quality])
            if ok and encoded.nbytes <= _IMAGE_BYTES:
                return encoded.tobytes()
        return None

    def _capture(self, visit, person, packet):
        if packet.received_mono - visit.image_mono < 5.:
            return False
        changed = False
        # Encoding failures are deliberately image-only. Database failures are
        # never caught here and always go through the writer's degraded barrier.
        try:
            frame = packet.frame
            if isinstance(frame, np.ndarray) and frame.ndim == 3 and frame.size:
                height, width = frame.shape[:2]
                x1, y1, x2, y2 = visit.values['last_bbox']
                rank = float(person.person_score) * (x2 - x1) * width * (y2 - y1) * height
                if rank > visit.person_quality:
                    image = self._jpeg(frame[max(0, int(y1 * height)):min(height, math.ceil(y2 * height)),
                                             max(0, int(x1 * width)):min(width, math.ceil(x2 * width))])
                    if image is not None:
                        visit.values['snapshot'] = image
                        visit.person_quality = rank
                        changed = True
        except (cv2.error, ValueError, TypeError, OverflowError, MemoryError):
            pass
        evidence = self._fresh_face(person, packet)
        if evidence is not None:
            try:
                quality = evidence.quality
                rank = (float(quality.min_size), float(quality.det_score), float(quality.blur))
                encoded = evidence.aligned_jpeg
                if (rank > visit.face_rank and isinstance(encoded, bytes)
                        and 0 < len(encoded) <= _IMAGE_BYTES):
                    image = cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_COLOR)
                    if image is not None and image.shape == (112, 112, 3):
                        visit.values['face_thumbnail'] = encoded
                        visit.values['face_quality'] = dict(
                            min_size=quality.min_size, det_score=quality.det_score,
                            blur=quality.blur, aligned=True, unambiguous=True,
                            sha256=hashlib.sha256(encoded).hexdigest(),
                            frame_seq=packet.seq, received_at=_utc(packet.received_at).isoformat())
                        visit.face_rank = rank
                        changed = True
            except (cv2.error, ValueError, TypeError, OverflowError, MemoryError):
                pass
        # Bound attempts as well as successful captures; a broken encoder must
        # not consume work on every frame. Better images are reconsidered in 5s.
        visit.image_mono = packet.received_mono
        if changed:
            visit.values['image_expires_at'] = _utc(packet.received_at) + timedelta(days=self.image_keep_days)
            visit.values['images_purged_at'] = None
        return changed

    def _enqueue_visit(self, gen, visit, events):
        identifier = visit.values['id']
        # Superseding a checkpoint is safe only AFTER lifecycle admission: this
        # snapshot includes that accepted checkpoint and all newer real liveness.
        self._updates.pop(identifier, None)
        visit.checkpoint_mono = visit.last_mono
        self._commands.append(_Command(self._next_order(), 'VISIT',
            (gen.camera_id, gen.generation), identifier, dict(visit.values), tuple(events), visit.packet_seq))

    def _lose(self, gen, visit, at, *, ambiguous=False):
        if visit.lost:
            return
        if self._lifecycle_full():
            self._block(gen)
            return
        visit.lost = True
        visit.hits = 0
        visit.values.update(employee_id=None, identity_state='UNKNOWN')
        event = self._event(visit, 'LOST', at,
                            {'reason': 'ASSOCIATION_UNCERTAIN' if ambiguous else 'TRACK_LOST'})
        self._enqueue_visit(gen, visit, [event])

    def _close_visit(self, gen, visit, reason):
        if visit.closed:
            return
        if self._lifecycle_full():
            self._block(gen)
            return
        visit.closed = True
        visit.values.update(status='CLOSED', ended_at=visit.values['last_seen_at'], end_reason=reason)
        event = self._event(visit, 'CLOSED', visit.values['last_seen_at'], {'reason': reason})
        self._enqueue_visit(gen, visit, [event])

    def _barrier(self, gen, reason, *, blocked):
        key = (gen.camera_id, gen.generation)
        if gen.retiring:
            return
        gen.retiring = True
        gen.blocked = blocked
        if blocked:
            gen.gap_started_at = _utc(self._now())
            self._blocked[gen.camera_id] = key
        closures = []
        for identifier in sorted(gen.visits):
            visit = self._visits[identifier]
            if visit.closed:
                continue
            visit.closed = True
            visit.values.update(status='CLOSED', ended_at=visit.values['last_seen_at'], end_reason=reason)
            event = self._event(visit, 'CLOSED', visit.values['last_seen_at'], {'reason': reason})
            closures.append((dict(visit.values), event))
        self._barriers[key] = _Command(self._next_order(), 'BARRIER', key,
                                      payload=(tuple(closures), reason, gen.gap_started_at))
        self._condition.notify_all()

    def _block(self, gen):
        key = (gen.camera_id, gen.generation)
        if gen.blocked or gen.barrier_done:
            return
        if gen.retiring:
            # Upgrade a failed reset barrier in its already reserved slot,
            # retaining admission order and event IDs across transaction retry.
            if gen.camera_id in self._blocked:
                return
            command = self._barriers.get(key)
            inflight = self._inflight
            if command is None and inflight is not None and inflight.kind == 'BARRIER' and inflight.generation == key:
                command = inflight
            if command is None:
                return
            gen.blocked = True
            gen.gap_started_at = _utc(self._now())
            self._blocked[gen.camera_id] = key
            closures = []
            for values, event in command.payload[0]:
                values = {**values, 'end_reason': 'JOURNAL_GAP'}
                event = {**event, 'payload': {**event['payload'], 'reason': 'JOURNAL_GAP'}}
                closures.append((values, event))
                visit = self._visits.get(values['id'])
                if visit is not None:
                    visit.values['end_reason'] = 'JOURNAL_GAP'
            replacement = _Command(command.order, 'BARRIER', key,
                payload=(tuple(closures), 'JOURNAL_GAP', gen.gap_started_at))
            if command is inflight:
                self._inflight = replacement
            else:
                self._barriers[key] = replacement
            return
        self._barrier(gen, 'JOURNAL_GAP', blocked=True)

    def reset_camera(self, camera_id, generation, reason):
        reason = _reason(reason)
        with self._condition:
            gen = self._generations.get((camera_id, str(generation)))
            if gen is not None:
                self._barrier(gen, reason, blocked=False)
            self._condition.notify_all()

    def tick(self, now):
        at = _utc(now)
        with self._condition:
            if self._accepting:
                mono = self._monotonic()
                for gen in tuple(self._generations.values()):
                    if gen.retiring:
                        continue
                    for identifier in tuple(gen.visits):
                        visit = self._visits[identifier]
                        if not visit.closed and mono - visit.last_mono >= self.lost_timeout:
                            self._close_visit(gen, visit, 'TRACK_EXPIRED')
                            if gen.retiring:
                                break
                if not self._purge_pending and mono - self._last_purge >= 3600.:
                    self._purge_pending = True
                    self._last_purge = mono
                    # Purge has one separately bounded maintenance slot.
                    self._purge_at = at
            self._condition.notify_all()

    def submit_link(self, proposal) -> bool:
        """Admit a frozen matching proposal; vectors never become payroll identity."""
        try:
            UUID(proposal.link_key)
            vector = np.asarray(proposal.embedding)
            if (vector.shape != (512,) or not np.isfinite(vector).all()
                    or abs(float(np.linalg.norm(vector)) - 1.) > .01
                    or not math.isfinite(proposal.score) or not math.isfinite(proposal.received_mono)):
                return False
            self._bbox(proposal.bbox)
        except (ValueError, TypeError, AttributeError):
            return False
        with self._condition:
            key = (proposal.camera_id, str(proposal.tracking_generation))
            gen = self._generations.get(key)
            visit = self._visits.get(proposal.visit_id)
            if (not self._accepting or gen is None or gen.retiring or visit is None or visit.closed
                    or self._current.get(proposal.camera_id) != key
                    or proposal.track_id != visit.values['track_id']
                    or proposal.source_revision != gen.source_revision or proposal.stream_id != gen.stream_id
                    or proposal.packet_seq != visit.packet_seq or proposal.received_mono != visit.last_mono
                    or _utc(proposal.observed_at) != visit.values['last_seen_at']):
                return False
            if visit.link_pending is not None:
                return visit.link_pending == proposal.link_key
            pending_ack = self._acks.get(('LINK_COMMITTED', proposal.visit_id))
            if pending_ack is not None:
                return pending_ack.link_key == proposal.link_key
            if self._lifecycle_full():
                self._block(gen)
                return False
            # Copy only immutable scalar attributes needed for the transaction.
            # Matching has already consumed the vector; no inference/archive write
            # or retained vector belongs in the journal writer.
            payload = dict(link_key=proposal.link_key, global_id=proposal.global_id,
                           employee_id=proposal.employee_id, score=float(proposal.score),
                           observed_at=_utc(proposal.observed_at), received_mono=proposal.received_mono,
                           source_revision=proposal.source_revision, stream_id=proposal.stream_id,
                           track_id=proposal.track_id)
            event = self._event(visit, 'LINKED', payload['observed_at'], dedupe_key=proposal.link_key)
            visit.link_pending = proposal.link_key
            self._commands.append(_Command(self._next_order(), 'LINK', key, proposal.visit_id,
                                           events=(event,), packet_seq=proposal.packet_seq, payload=payload))
            self._condition.notify_all()
            return True

    def _next_command(self):
        if self._startup:
            return _Command(0, 'STARTUP')
        candidates = []
        if self._commands:
            candidates.append(self._commands[0])
        if self._updates:
            candidates.append(next(iter(self._updates.values())))
        if self._barriers:
            candidates.append(next(iter(self._barriers.values())))
        if candidates:
            command = min(candidates, key=lambda item: item.order)
            if command.kind in ('VISIT', 'LINK'):
                self._commands.popleft()
            elif command.kind == 'UPDATE':
                self._updates.pop(command.visit_id)
            else:
                self._barriers.pop(command.generation)
            return command
        if self._purge_pending:
            return _Command(self._next_order(), 'PURGE', payload=self._purge_at)
        return None

    def _run(self):
        while True:
            with self._condition:
                # The journal owns its maintenance even when no AI worker exists.
                mono = self._monotonic()
                if not self._closing and mono >= self._next_tick_at:
                    self.tick(self._now())
                    self._next_tick_at = mono + 1.
                if (self._closing and self._stop_deadline is not None
                        and time.monotonic() >= self._stop_deadline):
                    self._condition.notify_all()
                    return
                if self._inflight is None:
                    self._inflight = self._next_command()
                command = self._inflight
                if command is None:
                    self._condition.notify_all()
                    if self._closing:
                        return
                    self._condition.wait(max(0., self._next_tick_at - self._monotonic()))
                    continue
                delay = self._retry_at - self._monotonic() if self._error else 0.
                if delay > 0:
                    if self._closing:
                        delay = min(delay, max(0., self._stop_deadline - time.monotonic()))
                    self._condition.wait(delay)
                    continue
            try:
                acknowledgement = self._write(command)
            except Exception:
                with self._condition:
                    self._error = 'JOURNAL_WRITE_FAILED'
                    self._retry_at = self._monotonic() + 1.
                    # A DB outage affects the entire writer, not only the command
                    # that happened to discover it. Freeze every admitted cutoff.
                    generations = tuple(self._generations.values())
                    for gen in generations:
                        if not gen.retiring:
                            self._block(gen)
                    for gen in generations:
                        if gen.retiring:
                            self._block(gen)
                    self._condition.notify_all()
                continue
            with self._condition:
                self._error = None
                self._retry_at = 0.
                self._complete(command, acknowledgement)
                self._inflight = None
                self._condition.notify_all()

    @staticmethod
    def _put_event(db, values):
        existing = db.scalar(select(HumanVisitEvent).where(
            HumanVisitEvent.dedupe_key == values['dedupe_key']))
        if existing is None:
            db.add(HumanVisitEvent(**values))
        elif existing.visit_id != values['visit_id'] or existing.type != values['type']:
            raise ValueError('Journal receipt collision')

    @staticmethod
    def _put_visit(db, values, at):
        row = db.get(HumanVisit, values['id'])
        data = dict(values)
        # A delayed command cannot resurrect expired evidence after a purge.
        if data['image_expires_at'] <= at:
            data.update(snapshot=None, face_thumbnail=None,
                        images_purged_at=data['images_purged_at'] or at)
        if row is None:
            row = HumanVisit(**data)
            db.add(row)
        else:
            # Link transactions own global_id. A queued metadata snapshot from
            # before LINKED must never wipe a subsequently committed binding.
            data.pop('global_id', None)
            for name, value in data.items():
                setattr(row, name, value)
        db.flush()
        return row

    def _write(self, command):
        at = _utc(self._now())
        with immediate_session(self.engine) as db:
            if command.kind == 'STARTUP':
                for visit in db.scalars(select(HumanVisit).where(
                        HumanVisit.status != 'CLOSED').execution_options(yield_per=256)):
                    seq = db.scalar(select(func.max(func.json_extract(HumanVisitEvent.payload,
                                            '$.transition_seq'))).where(HumanVisitEvent.visit_id == visit.id)) or 0
                    identifier = _id([visit.id, 'CLOSED', int(seq) + 1])
                    visit.status, visit.ended_at, visit.end_reason = 'CLOSED', visit.last_seen_at, 'PROCESS_RESTART'
                    self._put_event(db, dict(id=identifier, visit_id=visit.id, dedupe_key=identifier,
                        type='CLOSED', occurred_at=visit.last_seen_at,
                        payload={'version': 1, 'transition_seq': int(seq) + 1, 'reason': 'PROCESS_RESTART'}))
                return None
            if command.kind in ('VISIT', 'UPDATE'):
                self._put_visit(db, command.values, at)
                for event in command.events:
                    self._put_event(db, event)
                return None
            if command.kind == 'BARRIER':
                closures, reason, gap_start = command.payload
                for values, event in closures:
                    self._put_visit(db, values, at)
                    self._put_event(db, event)
                if gap_start is not None:
                    camera_id, generation = command.generation
                    identifier = _id([camera_id, generation, 'JOURNAL_GAP'])
                    if db.get(HumanJournalGap, identifier) is None:
                        db.add(HumanJournalGap(id=identifier, camera_id=camera_id,
                            tracking_generation=generation, started_at=gap_start,
                            ended_at=max(at, gap_start), reason='JOURNAL_GAP'))
                return None
            if command.kind == 'PURGE':
                db.execute(update(HumanVisit).where(
                    HumanVisit.image_expires_at <= command.payload,
                    HumanVisit.images_purged_at.is_(None)).values(
                        snapshot=None, face_thumbnail=None, images_purged_at=command.payload))
                return None
            if command.kind == 'LINK':
                return self._write_link(db, command)
            raise ValueError('Unknown journal command')

    def _write_link(self, db, command):
        payload = command.payload
        receipt = db.scalar(select(HumanVisitEvent).where(
            HumanVisitEvent.dedupe_key == payload['link_key']))
        if receipt is not None:
            if receipt.visit_id != command.visit_id or receipt.type != 'LINKED':
                raise ValueError('Journal link receipt collision')
            return receipt.payload['global_id']
        row = db.get(HumanVisit, command.visit_id)
        if row is None:
            raise ValueError('Journal link parent visit missing')
        camera = db.get(Camera, row.camera_id)
        source = db.get(AttendanceCamera, row.camera_id)
        with self._condition:
            gen = self._generations.get(command.generation)
            visit = self._visits.get(command.visit_id)
            # A frozen cutoff may drain its already accepted link. A source/reset
            # retirement, unlike a journal gap, invalidates outstanding proposals.
            valid = (gen is not None and (not gen.retiring or gen.blocked)
                and self._current.get(row.camera_id) == command.generation
                and visit is not None and visit.packet_seq == command.packet_seq
                and visit.last_mono == payload['received_mono'])
        if (not valid or row.status == 'CLOSED' or camera is None or not camera.enabled
                or source is None or source.source_revision != payload['source_revision']
                or row.source_revision != payload['source_revision']
                or row.stream_id != payload['stream_id'] or row.track_id != payload['track_id']):
            return None
        if row.global_id is not None:
            # A second proposal cannot relink an acknowledged visit. Record its
            # receipt with the already committed binding for retry-stable replay.
            global_id = row.global_id
        else:
            parent = db.get(GlobalPerson, payload['global_id']) if payload['global_id'] is not None else None
            if payload['global_id'] is not None and parent is None:
                return None
            if parent is not None and parent.employee_id is not None:
                confirmed_ids = (payload['employee_id'], row.employee_id)
                if any(employee is not None and employee != parent.employee_id for employee in confirmed_ids):
                    return None
            if parent is None:
                parent = GlobalPerson(employee_id=None, status='ACTIVE', last_camera_id=row.camera_id,
                                      last_track_id=row.track_id, created_at=payload['observed_at'],
                                      last_seen_at=payload['observed_at'])
                db.add(parent)
                db.flush()
            elif parent.last_seen_at <= payload['observed_at']:
                parent.last_camera_id, parent.last_track_id = row.camera_id, row.track_id
                parent.last_seen_at, parent.status = payload['observed_at'], 'ACTIVE'
            global_id = parent.id
            row.global_id = global_id
        event = dict(command.events[0])
        event['payload'] = {**event['payload'], 'global_id': global_id, 'score': payload['score']}
        self._put_event(db, event)
        return global_id

    def _ack(self, kind, visit, *, packet_seq=None, global_id=None, link_key=None):
        values = visit.values
        return JournalAck(kind, values['camera_id'], values['tracking_generation'], values['id'],
                          values['track_id'], values['source_revision'], values['stream_id'],
                          visit.packet_seq if packet_seq is None else packet_seq, global_id, link_key)

    def _complete(self, command, result):
        if command.kind == 'STARTUP':
            self._startup = False
        elif command.kind in ('VISIT', 'UPDATE'):
            visit = self._visits.get(command.visit_id)
            if visit is not None:
                if not visit.committed:
                    visit.committed = True
                    self._acks[('COMMITTED', command.visit_id)] = self._ack(
                        'COMMITTED', visit, packet_seq=command.packet_seq)
                if command.values['status'] == 'CLOSED':
                    visit.closed_durable = True
        elif command.kind == 'LINK':
            visit = self._visits.get(command.visit_id)
            if visit is not None:
                visit.link_pending = None
                if result is not None:
                    visit.values['global_id'] = result
                    self._acks[('LINK_COMMITTED', command.visit_id)] = self._ack(
                        'LINK_COMMITTED', visit, packet_seq=command.packet_seq,
                        global_id=result, link_key=command.payload['link_key'])
        elif command.kind == 'BARRIER':
            gen = self._generations[command.generation]
            gen.barrier_done = True
            for identifier in gen.visits:
                self._visits[identifier].closed_durable = True
            if gen.blocked:
                self._controls[command.generation] = JournalAck('RESET_REQUIRED', gen.camera_id,
                    gen.generation, source_revision=gen.source_revision, stream_id=gen.stream_id,
                    reason='JOURNAL_GAP')
        elif command.kind == 'PURGE':
            self._purge_pending = False
            for visit in self._visits.values():
                if visit.values['image_expires_at'] <= command.payload:
                    visit.values.update(snapshot=None, face_thumbnail=None, images_purged_at=command.payload)
        self._retire()

    def _retire(self):
        for identifier, visit in tuple(self._visits.items()):
            if (visit.closed_durable and ('COMMITTED', identifier) not in self._acks
                    and ('LINK_COMMITTED', identifier) not in self._acks):
                key = (visit.values['camera_id'], visit.values['tracking_generation'])
                self._generations[key].visits.discard(identifier)
                del self._visits[identifier]
        for key, gen in tuple(self._generations.items()):
            if gen.barrier_done and not gen.visits and key not in self._controls:
                if self._current.get(gen.camera_id) == key:
                    self._current.pop(gen.camera_id)
                if self._blocked.get(gen.camera_id) == key:
                    self._blocked.pop(gen.camera_id)
                self._retired_generations.pop(gen.camera_id, None)
                self._retired_generations[gen.camera_id] = gen.generation
                if len(self._retired_generations) > self._control_capacity:
                    self._retired_generations.popitem(last=False)
                del self._generations[key]

    def drain_acknowledgements(self) -> list[JournalAck]:
        with self._condition:
            result = [*self._acks.values(), *self._controls.values()]
            self._acks.clear()
            self._controls.clear()
            self._retire()
            return result

    def _counts(self):
        command = self._inflight
        return (
            len(self._commands) + int(self._startup) + int(self._purge_pending)
            + int(command is not None and command.kind in ('VISIT', 'LINK')),
            len(self._updates) + int(command is not None and command.kind == 'UPDATE'),
            len(self._barriers) + int(command is not None and command.kind == 'BARRIER'),
        )

    def has_camera_work(self, camera_id: int) -> bool:
        """Include reserved generations: a future gap barrier still needs its FK."""
        with self._condition:
            return any(gen.camera_id == camera_id for gen in self._generations.values())

    @staticmethod
    def _command_references_employee(command, employee_id):
        if command.values is not None and command.values.get('employee_id') == employee_id:
            return True
        if any(event['payload'].get('employee_id') == employee_id for event in command.events):
            return True
        if command.kind == 'LINK':
            return command.payload.get('employee_id') == employee_id
        if command.kind == 'BARRIER':
            return any(values.get('employee_id') == employee_id
                       or event['payload'].get('employee_id') == employee_id
                       for values, event in command.payload[0])
        return False

    def has_employee_work(self, employee_id: int) -> bool:
        """Called inside the deletion publication transaction, never around it.

        Scan only bounded owned state. Frozen commands/events can retain an owner
        after LOST/conflict has cleared the current visit. The in-flight command
        remains visible through commit and _complete; afterward durable history
        protects it. The writer never waits for the publication lock.
        """
        with self._condition:
            if any(visit.values['employee_id'] == employee_id for visit in self._visits.values()):
                return True
            if (self._inflight is not None
                    and self._command_references_employee(self._inflight, employee_id)):
                return True
            return any(self._command_references_employee(command, employee_id)
                       for commands in (self._commands, self._updates.values(), self._barriers.values())
                       for command in commands)

    def blocked_generations(self) -> tuple:
        with self._condition:
            return tuple((gen.camera_id, gen.generation, gen.gap_started_at)
                         for gen in self._generations.values() if gen.blocked)

    def health_snapshot(self) -> dict:
        with self._condition:
            commands, updates, barriers = self._counts()
            degraded = bool(self._error or self._blocked or self._capacity_degraded)
            error = self._error or ('JOURNAL_CAPACITY_EXCEEDED' if self._capacity_degraded else None)
            return dict(state='JOURNAL_DEGRADED' if degraded else 'OK', error_code=error,
                        pending_commands=commands, pending_updates=updates, pending_barriers=barriers,
                        blocked_generations=len(self._blocked), reserved_generations=len(self._generations),
                        active_visits=len(self._visits), pending_acknowledgements=len(self._acks) + len(self._controls),
                        accepting=self._accepting, writer_alive=self._thread.is_alive())

    def close(self, timeout_seconds=5) -> JournalCloseResult:
        if not math.isfinite(timeout_seconds) or timeout_seconds < 0:
            raise ValueError('Invalid journal close timeout')
        timeout_seconds = min(float(timeout_seconds), 5.)
        deadline = time.monotonic() + timeout_seconds
        with self._condition:
            self._accepting = False
            self._closing = True
            self._stop_deadline = deadline if self._stop_deadline is None else min(self._stop_deadline, deadline)
            for gen in tuple(self._generations.values()):
                self._barrier(gen, 'SHUTDOWN', blocked=False)
            self._condition.notify_all()
        self._thread.join(max(0., deadline - time.monotonic()))
        with self._condition:
            commands, updates, barriers = self._counts()
            drained = commands + updates + barriers == 0
            return JournalCloseResult(drained, commands, updates, barriers,
                                      self._error or (None if drained else 'JOURNAL_UNFLUSHED'),
                                      self._thread.is_alive())
