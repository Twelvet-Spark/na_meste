"""Isolated journal behavior; run .venv/Scripts/python.exe tests/test_human_journal.py.

Synchronization uses Events/Conditions, never sleeps or polling. No production DB,
readers, models, HTTP clients or recording directories are accessed.
"""
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from unittest.mock import patch
from uuid import NAMESPACE_URL, uuid4, uuid5
import json
import sys
import tempfile
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
from sqlalchemy import create_engine, event, select
from sqlalchemy.orm import Session

from app.database.database import Base, immediate_session
from app.database.models import (AttendanceCamera, Camera, Employee, EmployeeFace,
                                 GlobalObservation, GlobalPerson, HumanJournalGap,
                                 HumanVisit, HumanVisitEvent)
from app.services import human_journal as module

TESTS = []
STAMP = datetime(2026, 11, 16, 12)
GEN = str(uuid4())
STREAM = str(uuid4())


def test(fn):
    TESTS.append(fn)
    return fn


class Clock:
    value = 0.

    def mono(self):
        return self.value

    def now(self):
        return STAMP + timedelta(seconds=self.value)

    def advance(self, seconds):
        self.value += seconds


@contextmanager
def database():
    with tempfile.TemporaryDirectory() as root:
        engine = create_engine(f"sqlite:///{Path(root, 'journal.db').as_posix()}",
                               connect_args={"check_same_thread": False, "timeout": .1})

        @event.listens_for(engine, "connect")
        def foreign_keys(connection, _):
            connection.execute("PRAGMA foreign_keys=ON")

        Base.metadata.create_all(engine)
        with Session(engine) as db:
            for key in (1, 2, 3):
                db.add(Camera(id=key, name=f"Fixture {key}", nvr_host="unused",
                              username="", password="", channel=key, enabled=True))
                db.add(Employee(id=key, name=f"Anonymous fixture {key}"))
            db.commit()
            for key in (1, 2, 3):
                db.add(AttendanceCamera(camera_id=key, source_revision=1, enabled=False))
            db.commit()
        try:
            yield engine
        finally:
            engine.dispose()


def packet(clock, seq, *, source=1, stream=STREAM, frame=None):
    return SimpleNamespace(seq=seq, received_at=clock.now(), received_mono=clock.mono(),
                           source_revision=source, stream_id=stream,
                           frame=np.zeros((160, 160, 3), np.uint8) if frame is None else frame)


def person(track=1, *, employee=None, evidence=None, uncertain=False, score=.8):
    return SimpleNamespace(track_id=track, bbox=(.1, .1, .8, .9), person_score=score,
                           employee_id=employee, fresh_face_evidence=evidence,
                           association_uncertain=uncertain, identity_reason_codes=())


def outcome(*people, generation=GEN, lifecycle=()):
    return SimpleNamespace(tracking_generation=generation, observed_tracks=list(people),
                           track_lifecycle=list(lifecycle), detections=[])


def face(clock, seq, employee=1):
    image = np.random.default_rng(9).integers(0, 256, (112, 112, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode('.jpg', image)
    assert ok
    return SimpleNamespace(frame_seq=seq, received_at=clock.now(), received_mono=clock.mono(),
                           confirmed_employee_id=employee, candidate_employee_id=employee,
                           quality=SimpleNamespace(min_size=90., det_score=.95, blur=100., aligned=True),
                           aligned_jpeg=encoded.tobytes(), reason_codes=(), score=.8,
                           margin=.2, face_bbox=(10., 10., 100., 100.))


def idle(journal):
    with journal._condition:
        assert journal._condition.wait_for(
            lambda: journal._inflight is None and not journal._commands
            and not journal._updates and not journal._barriers and not journal._startup
            and not journal._purge_pending,
            timeout=5), journal.health_snapshot()


def rows(engine, cls):
    with Session(engine) as db:
        return list(db.scalars(select(cls)))


def start(engine, clock, **kwargs):
    journal = module.HumanJournal(engine, now=clock.now, monotonic=clock.mono, **kwargs)
    idle(journal)
    return journal


def finish(journal):
    result = journal.close()
    assert result.drained, result


class TransactionGate:
    """Block one writer transaction, or roll it back once after real SQL."""
    def __init__(self, *, fail=False):
        self.entered = Event()
        self.release = Event()
        self.fail = fail
        self.used = False

    @contextmanager
    def __call__(self, engine):
        gated = not self.used
        self.used = True
        if gated:
            self.entered.set()
            assert self.release.wait(5), "test did not release writer"
        with immediate_session(engine) as db:
            yield db
            if gated and self.fail:
                raise RuntimeError("injected rollback")


def link(admission, clock, **changes):
    values = dict(link_key=str(uuid4()), visit_id=admission.visit_id,
                  camera_id=admission.camera_id, tracking_generation=admission.tracking_generation,
                  track_id=admission.track_id, source_revision=admission.source_revision,
                  stream_id=admission.stream_id, packet_seq=admission.packet_seq,
                  observed_at=clock.now(), received_mono=clock.mono(), bbox=(.1, .1, .8, .9),
                  global_id=None, score=.9, employee_id=None,
                  embedding=np.ones(512, np.float32) / np.sqrt(512))
    values.update(changes)
    return SimpleNamespace(**values)


@test
def first_hit_unknown_is_atomic_deterministic_and_survives_restart():
    with database() as engine:
        clock = Clock()
        journal = start(engine, clock)
        admission = journal.observe(1, packet(clock, 1, frame=np.empty((0, 0, 3), np.uint8)),
                                    outcome(person()))[0]
        idle(journal)
        expected = str(uuid5(uuid5(NAMESPACE_URL, 'na_meste/human-journal/v1'),
                             json.dumps([1, GEN, 1], separators=(',', ':'))))
        assert admission.visit_id == expected
        visit = rows(engine, HumanVisit)[0]
        assert (visit.status, visit.employee_id, visit.snapshot) == ('PROVISIONAL', None, None)
        assert [e.type for e in rows(engine, HumanVisitEvent)] == ['FIRST_SEEN']
        assert journal.drain_acknowledgements()[0].kind == 'COMMITTED'
        # Simulate process loss without generating a normal closure: startup is the
        # same durable recovery used after a killed writer, on a preexisting row.
        finish(journal)
        with immediate_session(engine) as db:
            visit = db.get(HumanVisit, expected)
            visit.status, visit.ended_at, visit.end_reason = 'PROVISIONAL', None, None
        clock.advance(100)
        restarted = start(engine, clock)
        visit = rows(engine, HumanVisit)[0]
        assert visit.ended_at == STAMP == visit.last_seen_at
        assert visit.end_reason == 'PROCESS_RESTART'
        finish(restarted)


@test
def actual_hits_replay_lost_recovery_and_expiry_never_invent_departure():
    with database() as engine:
        clock = Clock()
        journal = start(engine, clock)
        journal.observe(1, packet(clock, 1), outcome(person()))
        journal.observe(1, packet(clock, 1), outcome(person()))
        idle(journal)
        assert rows(engine, HumanVisit)[0].status == 'PROVISIONAL'
        clock.advance(.2)
        journal.observe(1, packet(clock, 2), outcome(person()))
        idle(journal)
        assert rows(engine, HumanVisit)[0].status == 'ACTIVE'
        lost = SimpleNamespace(track_id=1, state='LOST', reason='TRACK_LOST')
        clock.advance(.2)
        journal.observe(1, packet(clock, 3), outcome(lifecycle=[lost]))
        journal.observe(1, packet(clock, 3), outcome(lifecycle=[lost]))
        clock.advance(.2)
        journal.observe(1, packet(clock, 4), outcome(person()))
        clock.advance(4)
        journal.tick(clock.now())
        idle(journal)
        visit = rows(engine, HumanVisit)[0]
        assert visit.last_seen_at == visit.ended_at == STAMP + timedelta(seconds=.6)
        types = [e.type for e in rows(engine, HumanVisitEvent)]
        assert types.count('LOST') == types.count('RECOVERED') == types.count('CLOSED') == 1
        finish(journal)


@test
def only_fresh_qualified_face_can_identify_and_transitions_are_deduped():
    with database() as engine:
        clock = Clock()
        journal = start(engine, clock)
        journal.observe(1, packet(clock, 1), outcome(person(employee=1)))
        clock.advance(.2)
        journal.observe(1, packet(clock, 2), outcome(person(employee=1, evidence=face(clock, 1))))
        idle(journal)
        assert rows(engine, HumanVisit)[0].employee_id is None
        for seq in (3, 4):
            clock.advance(.2)
            journal.observe(1, packet(clock, seq), outcome(person(employee=1, evidence=face(clock, seq))))
        clock.advance(.2)
        journal.observe(1, packet(clock, 5), outcome(person(employee=2, evidence=face(clock, 5, 2))))
        idle(journal)
        visit = rows(engine, HumanVisit)[0]
        assert visit.employee_id is None and visit.identity_state == 'CONFLICT'
        types = [e.type for e in rows(engine, HumanVisitEvent)]
        assert types.count('IDENTIFIED') == types.count('IDENTITY_CONFLICT') == 1
        finish(journal)


@test
def liveness_checkpoint_and_final_reset_flush_latest_real_observation():
    with database() as engine:
        clock = Clock()
        journal = start(engine, clock, min_hits=1)
        journal.observe(1, packet(clock, 1), outcome(person()))
        idle(journal)
        clock.advance(1)
        journal.observe(1, packet(clock, 2), outcome(person()))
        idle(journal)
        assert rows(engine, HumanVisit)[0].last_seen_at == STAMP
        clock.advance(1)
        journal.observe(1, packet(clock, 3), outcome(person()))
        idle(journal)
        assert rows(engine, HumanVisit)[0].last_seen_at == clock.now()
        clock.advance(.1)
        journal.observe(1, packet(clock, 4), outcome(person()))
        journal.reset_camera(1, GEN, 'SOURCE_CHANGED')
        idle(journal)
        visit = rows(engine, HumanVisit)[0]
        assert visit.ended_at == visit.last_seen_at == clock.now()
        assert visit.end_reason == 'SOURCE_CHANGED'
        finish(journal)


@test
def image_failure_does_not_rollback_metadata_and_retention_is_visit_only():
    with database() as engine:
        clock = Clock()
        with immediate_session(engine) as db:
            db.add(EmployeeFace(employee_id=1, embedding=b'unchanged', thumbnail=b'enrollment'))
            parent = GlobalPerson(status='ACTIVE')
            db.add(parent)
            db.flush()
            db.add(GlobalObservation(global_id=parent.id, camera_id=1, track_id=1,
                                     embedding=b'archive', snapshot=b'historical'))
        journal = start(engine, clock, image_keep_days=1)
        with patch.object(module.cv2, 'imencode', side_effect=cv2.error('encoding failed')):
            journal.observe(1, packet(clock, 1), outcome(person()))
        idle(journal)
        assert rows(engine, HumanVisit)[0].snapshot is None
        clock.advance(5)
        journal.observe(1, packet(clock, 2), outcome(person(score=.9, evidence=face(clock, 2))))
        idle(journal)
        visit = rows(engine, HumanVisit)[0]
        assert visit.snapshot and visit.face_thumbnail
        assert visit.face_quality['sha256']
        clock.advance(86401)
        journal.tick(clock.now())
        idle(journal)
        visit = rows(engine, HumanVisit)[0]
        assert visit.snapshot is None and visit.face_thumbnail is None and visit.images_purged_at
        assert rows(engine, EmployeeFace)[0].thumbnail == b'enrollment'
        assert rows(engine, GlobalObservation)[0].snapshot == b'historical'
        assert rows(engine, HumanVisitEvent)
        finish(journal)


@test
def writer_expires_visits_and_purges_images_without_an_ai_tick():
    with database() as engine:
        clock = Clock()
        journal = start(engine, clock, image_keep_days=1)
        try:
            journal.observe(1, packet(clock, 1), outcome(person(evidence=face(clock, 1))))
            idle(journal)
            assert rows(engine, HumanVisit)[0].snapshot
            clock.advance(86401)
            # Deterministic timer wakeup: no worker exists and no caller invokes tick().
            with journal._condition:
                journal._condition.notify_all()
                assert journal._condition.wait_for(
                    lambda: journal._last_purge >= clock.mono(), timeout=5)
            idle(journal)
            visit = rows(engine, HumanVisit)[0]
            assert visit.status == 'CLOSED' and visit.end_reason == 'TRACK_EXPIRED'
            assert visit.snapshot is None and visit.face_thumbnail is None and visit.images_purged_at
        finally:
            journal.close()


@test
def saturation_freezes_existing_identity_transition_and_reserved_barrier_flushes_cutoff():
    with database() as engine:
        clock = Clock()
        journal = start(engine, clock, lifecycle_capacity=1)
        gate = TransactionGate()
        with patch.object(module, 'immediate_session', gate):
            journal.observe(1, packet(clock, 1), outcome(person()))
            assert gate.entered.wait(5)
            clock.advance(.1)
            rejected = journal.observe(1, packet(clock, 2),
                outcome(person(employee=1, evidence=face(clock, 2))))[0]
            assert rejected.status == 'REJECTED'
            clock.advance(5)
            assert journal.observe(1, packet(clock, 3), outcome(person()))[0].status == 'REJECTED'
            gate.release.set()
            idle(journal)
        visit = rows(engine, HumanVisit)[0]
        assert visit.last_seen_at == visit.ended_at == STAMP
        assert visit.end_reason == 'JOURNAL_GAP' and visit.employee_id is None
        assert len(rows(engine, HumanJournalGap)) == 1
        acknowledgements = journal.drain_acknowledgements()
        assert any(a.kind == 'RESET_REQUIRED' for a in acknowledgements)
        new = journal.observe(1, packet(clock, 4), outcome(person(), generation=str(uuid4())))[0]
        assert new.visit_id != visit.id and new.status == 'PENDING'
        finish(journal)


@test
def failed_transaction_retries_same_event_keys_then_closes_gap_without_orphans():
    with database() as engine:
        clock = Clock()
        journal = start(engine, clock)
        gate = TransactionGate(fail=True)
        with patch.object(module, 'immediate_session', gate):
            admission = journal.observe(1, packet(clock, 1), outcome(person()))[0]
            assert gate.entered.wait(5)
            gate.release.set()
            with journal._condition:
                assert journal._condition.wait_for(lambda: journal._error is not None, timeout=5)
            assert rows(engine, HumanVisit) == rows(engine, HumanVisitEvent) == []
            assert not journal.drain_acknowledgements()
            clock.advance(1)
            journal.tick(clock.now())
            idle(journal)
        visits, events = rows(engine, HumanVisit), rows(engine, HumanVisitEvent)
        assert len(visits) == 1 and visits[0].id == admission.visit_id
        assert {e.type for e in events} == {'FIRST_SEEN', 'CLOSED'}
        namespace = uuid5(NAMESPACE_URL, 'na_meste/human-journal/v1')
        first = next(e for e in events if e.type == 'FIRST_SEEN')
        assert first.id == str(uuid5(namespace, json.dumps([admission.visit_id, 'FIRST_SEEN', 1], separators=(',', ':'))))
        assert len({e.dedupe_key for e in events}) == 2
        finish(journal)


@test
def active_and_control_capacity_reject_without_unbounded_ack_history():
    with database() as engine:
        clock = Clock()
        journal = start(engine, clock, active_capacity=1, control_capacity=1)
        journal.observe(1, packet(clock, 1), outcome(person()))
        idle(journal)
        assert journal.observe(2, packet(clock, 1), outcome(person()))[0].status == 'REJECTED'
        assert journal.observe(1, packet(clock, 2), outcome(person(2)))[0].status == 'REJECTED'
        idle(journal)
        assert journal.health_snapshot()['blocked_generations'] == 1
        journal.drain_acknowledgements()
        assert journal.health_snapshot()['active_visits'] == 0
        finish(journal)


@test
def close_is_bounded_and_truthful_while_database_call_is_blocked():
    with database() as engine:
        clock = Clock()
        journal = start(engine, clock)
        gate = TransactionGate()
        with patch.object(module, 'immediate_session', gate):
            journal.observe(1, packet(clock, 1), outcome(person()))
            assert gate.entered.wait(5)
            result = journal.close(timeout_seconds=0)
            assert not result.drained and result.unresolved_commands == 1
            assert result.unresolved_barriers == 1 and result.thread_alive
            gate.release.set()
            journal._thread.join(5)
            assert not journal._thread.is_alive()


@test
def linked_parent_visit_and_receipt_are_atomic_replay_returns_actual_id():
    with database() as engine:
        clock = Clock()
        journal = start(engine, clock)
        admission = journal.observe(1, packet(clock, 1), outcome(person()))[0]
        idle(journal)
        proposal = link(admission, clock, employee_id=1)
        assert journal.submit_link(proposal)
        idle(journal)
        ack = next(a for a in journal.drain_acknowledgements() if a.kind == 'LINK_COMMITTED')
        assert rows(engine, HumanVisit)[0].global_id == ack.global_id
        assert rows(engine, HumanVisit)[0].employee_id is None
        assert rows(engine, GlobalPerson)[0].employee_id is None
        assert journal.submit_link(proposal)
        idle(journal)
        replay = journal.drain_acknowledgements()[0]
        assert replay.global_id == ack.global_id and replay.link_key == proposal.link_key
        assert len(rows(engine, GlobalPerson)) == 1
        assert len([e for e in rows(engine, HumanVisitEvent) if e.type == 'LINKED']) == 1
        assert not journal.submit_link(link(admission, clock, packet_seq=999))
        assert not journal.submit_link(link(admission, clock, stream_id=str(uuid4())))
        assert rows(engine, GlobalObservation) == []
        finish(journal)


@test
def failed_link_parent_creation_publishes_nothing_and_rolls_back_all_rows():
    with database() as engine:
        clock = Clock()
        journal = start(engine, clock)
        admission = journal.observe(1, packet(clock, 1), outcome(person()))[0]
        idle(journal)
        journal.drain_acknowledgements()
        gate = TransactionGate(fail=True)
        with patch.object(module, 'immediate_session', gate):
            assert journal.submit_link(link(admission, clock))
            assert gate.entered.wait(5)
            gate.release.set()
            with journal._condition:
                assert journal._condition.wait_for(lambda: journal._error is not None, timeout=5)
            assert rows(engine, GlobalPerson) == []
            assert rows(engine, HumanVisit)[0].global_id is None
            assert not journal.drain_acknowledgements()
            clock.advance(1)
            journal.tick(clock.now())
            idle(journal)
        assert len(rows(engine, GlobalPerson)) == 1
        assert rows(engine, HumanVisit)[0].employee_id is None
        finish(journal)


@test
def image_representative_improves_only_after_five_seconds_and_is_bounded():
    with database() as engine:
        clock = Clock()
        journal = start(engine, clock, min_hits=1, lost_timeout=30)
        first_frame = np.zeros((900, 1400, 3), np.uint8)
        journal.observe(1, packet(clock, 1, frame=first_frame), outcome(person(score=.5)))
        idle(journal)
        initial = rows(engine, HumanVisit)[0].snapshot
        assert initial and len(initial) <= 262144
        assert max(cv2.imdecode(np.frombuffer(initial, np.uint8), cv2.IMREAD_COLOR).shape[:2]) <= 640
        brighter = np.full_like(first_frame, 190)
        clock.advance(4.9)
        journal.observe(1, packet(clock, 2, frame=brighter), outcome(person(score=.9)))
        idle(journal)
        assert rows(engine, HumanVisit)[0].snapshot == initial
        clock.advance(.1)
        journal.observe(1, packet(clock, 3, frame=brighter), outcome(person(score=.9)))
        idle(journal)
        improved = rows(engine, HumanVisit)[0].snapshot
        assert improved != initial
        clock.advance(5)
        journal.observe(1, packet(clock, 4, frame=first_frame), outcome(person(score=.7)))
        idle(journal)
        assert rows(engine, HumanVisit)[0].snapshot == improved
        finish(journal)


@test
def unqualified_or_malformed_faces_never_assign_identity_or_lose_visit_metadata():
    for invalid in ('small', 'blur', 'unaligned', 'ambiguous', 'stale', 'malformed_image'):
        with database() as engine:
            clock = Clock()
            journal = start(engine, clock)
            evidence = face(clock, 1)
            if invalid == 'small':
                evidence.quality.min_size = 47.
            elif invalid == 'blur':
                evidence.quality.blur = 19.
            elif invalid == 'unaligned':
                evidence.quality.aligned = False
            elif invalid == 'ambiguous':
                evidence.reason_codes = ('AMBIGUOUS_FACE',)
            elif invalid == 'stale':
                evidence.frame_seq = 0
            else:
                evidence.aligned_jpeg = b'not a JPEG'
            journal.observe(1, packet(clock, 1), outcome(person(employee=1, evidence=evidence)))
            idle(journal)
            visit = rows(engine, HumanVisit)[0]
            assert visit.snapshot and visit.face_thumbnail is None
            # An image encoding failure does not invalidate otherwise qualified
            # fresh recognition; the saved sample is still unavailable to append.
            assert visit.employee_id == (1 if invalid == 'malformed_image' else None)
            finish(journal)


@test
def ambiguous_real_observations_do_not_emit_recovered_until_association_recovers():
    with database() as engine:
        clock = Clock()
        journal = start(engine, clock)
        journal.observe(1, packet(clock, 1), outcome(person(employee=1, evidence=face(clock, 1))))
        for seq in (2, 3, 4):
            clock.advance(.1)
            transition = SimpleNamespace(track_id=1, state='AMBIGUOUS', reason='ASSOCIATION_UNCERTAIN')
            journal.observe(1, packet(clock, seq),
                            outcome(person(uncertain=True), lifecycle=[transition]))
        idle(journal)
        types = [e.type for e in rows(engine, HumanVisitEvent)]
        assert types.count('LOST') == 1 and types.count('RECOVERED') == 0
        assert rows(engine, HumanVisit)[0].employee_id is None
        clock.advance(.1)
        journal.observe(1, packet(clock, 5), outcome(person()))
        idle(journal)
        assert [e.type for e in rows(engine, HumanVisitEvent)].count('RECOVERED') == 1
        finish(journal)


@test
def accepted_coalesced_update_precedes_saturated_closure_barrier():
    with database() as engine:
        clock = Clock()
        journal = start(engine, clock, min_hits=1, lifecycle_capacity=1, lost_timeout=30)
        journal.observe(1, packet(clock, 1), outcome(person()))
        idle(journal)
        gate = TransactionGate()
        with patch.object(module, 'immediate_session', gate):
            # Occupy the lifecycle slot on another camera while camera 1 has a
            # latest accepted checkpoint waiting behind that command.
            journal.observe(2, packet(clock, 1), outcome(person()))
            assert gate.entered.wait(5)
            clock.advance(2)
            journal.observe(1, packet(clock, 2), outcome(person(score=.9)))
            assert journal.health_snapshot()['pending_updates'] == 1
            expired = SimpleNamespace(track_id=1, state='EXPIRED', reason='TRACK_EXPIRED')
            clock.advance(.1)
            journal.observe(1, packet(clock, 3), outcome(lifecycle=[expired]))
            clock.advance(2)
            assert journal.observe(1, packet(clock, 4), outcome(person()))[0].status == 'REJECTED'
            gate.release.set()
            idle(journal)
        visit = next(v for v in rows(engine, HumanVisit) if v.camera_id == 1)
        assert visit.end_reason == 'JOURNAL_GAP'
        assert visit.last_seen_at == visit.ended_at == STAMP + timedelta(seconds=2)
        assert visit.max_person_score == .9
        assert journal.observe(1, packet(clock, 5), outcome(person(), generation=str(uuid4())))[0].status == 'REJECTED'
        journal.drain_acknowledgements()
        assert journal.observe(1, packet(clock, 6), outcome(person()))[0].status == 'REJECTED'
        accepted = journal.observe(1, packet(clock, 7), outcome(person(), generation=str(uuid4())))[0]
        assert accepted.status == 'PENDING'
        finish(journal)


@test
def coalesced_updates_remain_bounded_and_active_saturation_freezes_last_accepted_state():
    with database() as engine:
        clock = Clock()
        journal = start(engine, clock, min_hits=1, active_capacity=2, lost_timeout=30)
        journal.observe(1, packet(clock, 1), outcome(person(1), person(2)))
        idle(journal)
        gate = TransactionGate()
        with patch.object(module, 'immediate_session', gate):
            clock.advance(2)
            journal.observe(1, packet(clock, 2), outcome(person(1), person(2)))
            assert gate.entered.wait(5)
            for seq in (3, 4, 5):
                clock.advance(2)
                journal.observe(1, packet(clock, seq), outcome(person(1), person(2)))
            assert len(journal._updates) <= 2
            cutoff = clock.now()
            clock.advance(.1)
            assert journal.observe(1, packet(clock, 6), outcome(person(3)))[0].status == 'REJECTED'
            gate.release.set()
            idle(journal)
        visits = rows(engine, HumanVisit)
        assert len(visits) == 2
        assert all(v.last_seen_at == v.ended_at == cutoff and v.end_reason == 'JOURNAL_GAP' for v in visits)
        assert len(rows(engine, HumanJournalGap)) == 1
        finish(journal)


@test
def failing_reset_barrier_uses_its_reserved_slot_and_same_closure_event_id():
    with database() as engine:
        clock = Clock()
        journal = start(engine, clock)
        admission = journal.observe(1, packet(clock, 1), outcome(person()))[0]
        idle(journal)
        gate = TransactionGate(fail=True)
        with patch.object(module, 'immediate_session', gate):
            journal.reset_camera(1, GEN, 'SOURCE_CHANGED')
            assert gate.entered.wait(5)
            gate.release.set()
            with journal._condition:
                assert journal._condition.wait_for(lambda: journal._error is not None, timeout=5)
            assert rows(engine, HumanVisit)[0].status == 'PROVISIONAL'
            clock.advance(1)
            journal.tick(clock.now())
            idle(journal)
        visit = rows(engine, HumanVisit)[0]
        assert visit.end_reason == 'JOURNAL_GAP' and visit.ended_at == STAMP
        events = [e for e in rows(engine, HumanVisitEvent) if e.type == 'CLOSED']
        assert len(events) == 1
        namespace = uuid5(NAMESPACE_URL, 'na_meste/human-journal/v1')
        assert events[0].id == str(uuid5(namespace, json.dumps(
            [admission.visit_id, 'CLOSED', 2], separators=(',', ':'))))
        assert any(a.kind == 'RESET_REQUIRED' for a in journal.drain_acknowledgements())
        finish(journal)


@test
def source_reconnect_closes_old_ids_and_cannot_resume_retired_generation():
    with database() as engine:
        clock = Clock()
        journal = start(engine, clock)
        old = journal.observe(1, packet(clock, 1), outcome(person()))[0]
        clock.advance(.5)
        new_generation = str(uuid4())
        new = journal.observe(1, packet(clock, 1, stream=str(uuid4())),
                              outcome(person(), generation=new_generation))[0]
        idle(journal)
        assert old.visit_id != new.visit_id
        visits = rows(engine, HumanVisit)
        first = next(v for v in visits if v.id == old.visit_id)
        assert first.ended_at == STAMP and first.end_reason == 'GENERATION_CHANGED'
        journal.drain_acknowledgements()
        assert journal.observe(1, packet(clock, 2), outcome(person()))[0].status == 'REJECTED'
        finish(journal)


@test
def link_rechecks_database_source_and_current_packet_before_creating_parent():
    for invalidation in ('source', 'disabled', 'packet', 'reset'):
        with database() as engine:
            clock = Clock()
            journal = start(engine, clock, min_hits=1)
            admission = journal.observe(1, packet(clock, 1), outcome(person()))[0]
            idle(journal)
            journal.drain_acknowledgements()
            gate = TransactionGate()
            with patch.object(module, 'immediate_session', gate):
                assert journal.submit_link(link(admission, clock))
                assert gate.entered.wait(5)
                if invalidation == 'source':
                    with immediate_session(engine) as db:
                        db.get(AttendanceCamera, 1).source_revision = 2
                elif invalidation == 'disabled':
                    with immediate_session(engine) as db:
                        db.get(Camera, 1).enabled = False
                elif invalidation == 'reset':
                    journal.reset_camera(1, GEN, 'SOURCE_CHANGED')
                else:
                    clock.advance(.1)
                    journal.observe(1, packet(clock, 2), outcome(person()))
                gate.release.set()
                idle(journal)
            assert rows(engine, GlobalPerson) == []
            assert rows(engine, HumanVisit)[0].global_id is None
            assert not any(a.kind == 'LINK_COMMITTED' for a in journal.drain_acknowledgements())
            finish(journal)


@test
def existing_global_identity_conflict_veto_and_second_link_preserve_committed_parent():
    with database() as engine:
        clock = Clock()
        with immediate_session(engine) as db:
            db.add(GlobalPerson(id=80, employee_id=2, status='ACTIVE'))
        journal = start(engine, clock)
        admission = journal.observe(1, packet(clock, 1), outcome(person()))[0]
        idle(journal)
        journal.drain_acknowledgements()
        assert journal.submit_link(link(admission, clock, global_id=80, employee_id=1))
        idle(journal)
        assert rows(engine, HumanVisit)[0].global_id is None
        assert journal.submit_link(link(admission, clock, global_id=80, employee_id=2))
        idle(journal)
        assert journal.drain_acknowledgements()[0].global_id == 80
        assert journal.submit_link(link(admission, clock))
        idle(journal)
        assert journal.drain_acknowledgements()[0].global_id == 80
        assert len(rows(engine, GlobalPerson)) == 1
        assert rows(engine, HumanVisit)[0].employee_id is None
        finish(journal)


@test
def startup_failure_never_blocks_constructor_and_close_reports_pending_recovery():
    with database() as engine:
        clock = Clock()
        gate = TransactionGate(fail=True)
        with patch.object(module, 'immediate_session', gate):
            journal = module.HumanJournal(engine, now=clock.now, monotonic=clock.mono)
            assert gate.entered.wait(5)
            gate.release.set()
            with journal._condition:
                assert journal._condition.wait_for(lambda: journal._error is not None, timeout=5)
            assert journal.observe(1, packet(clock, 1), outcome(person()))[0].status == 'REJECTED'
            result = journal.close(0)
            assert not result.drained and result.error_code == 'JOURNAL_WRITE_FAILED'
            assert result.unresolved_commands == 1
            journal._thread.join(5)
            assert not journal._thread.is_alive()


@test
def unsorted_first_observations_all_create_visits_and_small_crops_are_not_excluded():
    with database() as engine:
        clock = Clock()
        journal = start(engine, clock)
        tiny = person(3)
        tiny.bbox = (.1, .1, .101, .102)
        admissions = journal.observe(1, packet(clock, 1), outcome(person(7), tiny))
        assert [a.status for a in admissions] == ['PENDING', 'PENDING']
        idle(journal)
        assert {v.track_id for v in rows(engine, HumanVisit)} == {3, 7}
        assert all(v.identity_state == 'UNKNOWN' for v in rows(engine, HumanVisit))
        finish(journal)


@test
def confirmed_visit_face_vetoes_conflicting_global_even_without_proposal_identity_hint():
    with database() as engine:
        clock = Clock()
        with immediate_session(engine) as db:
            db.add(GlobalPerson(id=80, employee_id=2, status='ACTIVE'))
        journal = start(engine, clock)
        admission = journal.observe(1, packet(clock, 1),
                                    outcome(person(employee=1, evidence=face(clock, 1))))[0]
        idle(journal)
        assert journal.submit_link(link(admission, clock, global_id=80))
        idle(journal)
        visit = rows(engine, HumanVisit)[0]
        assert visit.global_id is None and visit.employee_id == 1
        finish(journal)


@test
def drained_closed_visits_release_bounded_ack_slots_without_losing_history():
    with database() as engine:
        clock = Clock()
        journal = start(engine, clock, active_capacity=1, control_capacity=1, min_hits=1)
        for track_id in range(1, 9):
            admission = journal.observe(1, packet(clock, track_id), outcome(person(track_id)))[0]
            assert admission.status == 'PENDING'
            clock.advance(4)
            journal.tick(clock.now())
            idle(journal)
            assert journal.health_snapshot()['pending_acknowledgements'] == 1
            journal.drain_acknowledgements()
            health = journal.health_snapshot()
            assert health['active_visits'] == health['pending_acknowledgements'] == 0
            assert health['reserved_generations'] == 1
        assert len(rows(engine, HumanVisit)) == 8
        assert len(rows(engine, HumanVisitEvent)) == 16
        finish(journal)


if __name__ == '__main__':
    failed = 0
    for case in TESTS:
        try:
            case()
            print(f'PASS {case.__name__}')
        except Exception:
            failed += 1
            print(f'FAIL {case.__name__}')
            traceback.print_exc()
    print(f'{len(TESTS) - failed}/{len(TESTS)} passed')
    raise SystemExit(bool(failed))
