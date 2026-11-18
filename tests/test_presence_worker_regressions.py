"""Review regressions for employee-appearance lifetime and publication-time admission.

Real worker/configuration/SQLite; model observations are deterministic doubles.
"""
import sys
import traceback
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace as N
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from sqlalchemy.orm import Session

from app.ai.employee_appearance import EmployeeAppearanceGallery
from app.ai.recognition_service import DetectionResult, FrameOutcome
from app.ai.worker import _TrackIdentity, _VisitBinding
from app.database.database import immediate_session
from app.database.models import AttendanceRuntime, Employee, CameraProfile
from app.time_utils import day_bounds, local_date
from attendance_test_support import Clock, FRAME, database, worker_for

TESTS = []
VECTOR = np.array([1., 0., 0., 0.], dtype=np.float32)


def test(fn):
    TESTS.append(fn)
    return fn


class Presence:
    def __init__(self, engine):
        self.engine, self.observations, self.crossings = engine, [], []
    def observe(self, *args):
        self.observations.append(args)
    def record_crossing(self, **kwargs):
        self.crossings.append(kwargs)


def outcome(packet, employee=1):
    person = DetectionResult(track_id=7, employee_id=employee,
                             employee_name=f'Employee {employee}' if employee else None,
                             confidence=.99, bbox=(.25, .1, .65, .9), state='identified' if employee else 'unknown')
    person.person_score = .99
    if employee is not None:
        person.fresh_face_evidence = N(reason_codes=(), frame_seq=packet.seq,
                                       confirmed_employee_id=employee, store_revision=7)
    return FrameOutcome(detections=[person], observed_tracks=[person], tracking_generation='g')


def publish(worker, readers, clock, employee=1, during=None):
    reader = readers.get(1)
    reader.buffer.put(FRAME, stream_id='source-a', source_revision=2,
                      received_at=clock.utc, received_mono=clock.mono)
    def infer(inputs):
        if during is not None:
            during()
        return [outcome(item.packet, employee) for item in inputs]
    with patch('app.ai.worker.time.monotonic', side_effect=clock.monotonic), \
            patch('app.ai.worker.utcnow', side_effect=clock.utcnow), \
            patch.object(worker.service, 'process_batch', side_effect=infer):
        worker.step()
    clock.advance(.25)


@test
def gallery_revision_change_discards_previously_learned_appearance():
    for deactivate in (False, True):
        with database() as (engine, recorder):
            clock = Clock()
            worker, readers, _ = worker_for(engine, recorder, clock)
            worker.appearance_gallery = EmployeeAppearanceGallery()
            publish(worker, readers, clock)
            worker.appearance_gallery.learn(1, VECTOR, clock.utc)
            with immediate_session(engine, publish=True) as db:
                db.get(AttendanceRuntime, 1).gallery_revision += 1
                if deactivate:
                    db.get(Employee, 1).active = False
            with Session(engine) as db:
                worker.service.store.refresh(db)
            publish(worker, readers, clock, employee=None)
            assert worker.appearance_gallery.match(VECTOR, clock.utc) is None


@test
def stale_or_inactive_face_gallery_cannot_assign_an_appearance_identity():
    with database() as (engine, recorder):
        clock = Clock()
        worker, readers, _ = worker_for(engine, recorder, clock)
        worker.appearance_gallery = EmployeeAppearanceGallery()
        worker.appearance_gallery.learn(1, VECTOR, clock.utc)
        config = worker._read_configurations()[1]
        identity = _TrackIdentity()
        worker._identities[(1, 'g', 7)] = identity
        stale = N(camera_id=1, received_at=clock.utc, config=replace(config, gallery_revision=8))
        for _ in range(2):
            worker._match_appearance(stale, identity, VECTOR)
        assert identity.employee_id is None, 'stale face gallery must block appearance attribution'
        with immediate_session(engine) as db:
            db.get(Employee, 1).active = False
        with Session(engine) as db:
            worker.service.store.refresh(db)
        current = N(camera_id=1, received_at=clock.utc, config=config)
        for _ in range(2):
            worker._match_appearance(current, identity, VECTOR)
        assert identity.employee_id is None, 'inactive employee must not be assigned'


@test
def an_uninterrupted_reid_track_needs_new_day_face_evidence():
    with database() as (engine, recorder):
        clock = Clock()
        clock.utc = day_bounds(local_date(clock.utc))[1] - timedelta(seconds=.1)
        worker, readers, _ = worker_for(engine, recorder, clock)
        presence = worker.presence = Presence(engine)
        worker.appearance_gallery = EmployeeAppearanceGallery()
        publish(worker, readers, clock, employee=None)
        config = worker._configurations[1]
        observed_at = readers.get(1).buffer.latest().received_at
        worker.appearance_gallery.learn(1, VECTOR, observed_at)
        identity = worker._identities[(1, 'g', 7)]
        work = N(camera_id=1, received_at=observed_at, config=config)
        for _ in range(2):
            worker._match_appearance(work, identity, VECTOR)
        assert identity.source == 'REID'
        publish(worker, readers, clock, employee=None)
        assert not presence.observations, 'yesterday appearance identity must not credit today'
        assert identity.employee_id is None and identity.reid_votes == 0


@test
def votes_do_not_accumulate_across_local_midnight():
    with database() as (engine, recorder):
        clock = Clock()
        midnight = day_bounds(local_date(clock.utc))[1]
        worker, readers, _ = worker_for(engine, recorder, clock)
        worker.appearance_gallery = EmployeeAppearanceGallery()
        identity = _TrackIdentity()
        worker._identities[(1, 'g', 7)] = identity
        config = worker._read_configurations()[1]
        before = N(camera_id=1, received_at=midnight-timedelta(seconds=1), config=config)
        worker.appearance_gallery.learn(1, VECTOR, before.received_at)
        worker._match_appearance(before, identity, VECTOR)
        after = N(camera_id=1, received_at=midnight+timedelta(seconds=1), config=config)
        worker.appearance_gallery.learn(1, VECTOR, after.received_at)  # fresh face on another track today
        worker._match_appearance(after, identity, VECTOR)
        assert identity.employee_id is None, 'one vote on each day is not two votes today'


@test
def yesterday_face_does_not_teach_the_new_day_gallery():
    with database() as (engine, recorder):
        clock = Clock()
        clock.utc = day_bounds(local_date(clock.utc))[1] - timedelta(seconds=.1)
        worker, readers, _ = worker_for(engine, recorder, clock)
        worker.presence = Presence(engine)
        publish(worker, readers, clock)
        publish(worker, readers, clock, employee=None)
        worker.reid = object()
        worker.journal = object()
        worker.appearance_gallery = EmployeeAppearanceGallery()
        worker._visit_bindings[(1, 'g', 7)] = _VisitBinding('visit', clock.mono)
        packet = readers.get(1).buffer.latest()
        config = worker._configurations[1]
        worker._queue_appearance((1, packet, config, readers.get(1), worker._reset_epochs.get(1, 0)),
                                 outcome(packet, employee=None), worker.service.store.revision)
        queued = next(iter(worker._appearance_pending.values()))
        assert queued.learn_employee_id is None


@test
def time_off_committed_during_inference_blocks_both_sightings_and_crossings():
    with database() as (engine, recorder):
        clock = Clock()
        worker, readers, _ = worker_for(engine, recorder, clock)
        presence = worker.presence = Presence(engine)
        crossing = N(direction='ENTRY', confirmed_employee_id=1,
                     track_key=(1, 'source-a', 'g', 7), crossing_index=0, occurred_at=clock.utc)
        def turn_off():
            with immediate_session(engine, publish=True) as db:
                db.get(CameraProfile, 1).time_tracking = False
        with patch.object(worker.doorway, 'process', return_value=[crossing]):
            publish(worker, readers, clock, during=turn_off)
        assert presence.observations == [] and presence.crossings == []
        assert worker.state.camera(1)['tracks'], 'TIME_OFF must not discard regular detection output'


if __name__ == '__main__':
    failures = 0
    for case in TESTS:
        try:
            case()
            print(f'PASS {case.__name__}')
        except Exception:
            failures += 1
            print(f'FAIL {case.__name__}')
            traceback.print_exc()
    print(f'{len(TESTS)-failures}/{len(TESTS)} passed')
    raise SystemExit(bool(failures))
