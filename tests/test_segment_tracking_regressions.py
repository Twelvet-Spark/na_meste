"""Consumer regressions exposed by the segment-01 tracing investigation."""
import sys
import traceback
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from app.ai.tracker import ByteTracker
from app.database.database import immediate_session
from app.database.models import AttendanceCamera, CameraProfile
from attendance_test_support import Clock, database
from test_multi_camera_worker import tracking_worker, tick, contexts


def box(left, right):
    return np.array([left, 20., right, 220.], dtype=np.float32)


def ordinary(engine):
    with immediate_session(engine) as db:
        db.get(AttendanceCamera, 1).enabled = False
        db.get(CameraProfile, 1).zone = 'INTERIOR'
        db.get(CameraProfile, 1).ai_fps = 1.


def test_ordinary_one_fps_reaches_display_hits_and_preserves_identity_between_frames():
    with database() as (engine, _):
        ordinary(engine)
        clock = Clock()
        worker, readers, _, faces, _ = tracking_worker(engine, clock)
        worker.service.min_hits = 2
        tick(worker, readers, clock, 1, advance=1.)
        faces.set(1, employee=1)
        tick(worker, readers, clock, 1, advance=0.)
        assert worker.state.camera(1)['tracks'][0]['employee_id'] == 1, '1 FPS must reach its second hit'
        faces.clear(1)
        clock.advance(.8)
        a,b = contexts(clock)
        with a,b:
            worker.step()  # no newer frame: ordinary identities must not get a door-gap reset
        clock.advance(.2)
        tick(worker, readers, clock, 1)
        assert worker.state.camera(1)['tracks'][0]['employee_id'] == 1


def test_real_ambiguous_lifecycle_revokes_sticky_employee_before_recovery():
    with database() as (engine, _):
        ordinary(engine)
        with immediate_session(engine) as db: db.get(CameraProfile, 1).ai_fps = 5.
        clock = Clock()
        worker, readers, detector, faces, _ = tracking_worker(engine, clock)
        persons = [(box(0.,100.),.99),(box(10.,110.),.99)]
        detector.detect_batch = lambda frames: [persons for _ in frames]
        faces.set(1, employee=1)
        tick(worker, readers, clock, 1)
        assert any(t['employee_id'] == 1 for t in worker.state.camera(1)['tracks'])
        faces.clear(1)
        persons = [(box(5.,105.),.99)]
        tick(worker, readers, clock, 1)
        assert worker.state.camera(1)['tracks'], 'current high-confidence people must remain visible without trusted association'
        assert all(t['employee_id'] is None and t['track_id'] is None for t in worker.state.camera(1)['tracks'])
        assert all(i.employee_id is None for i in worker._identities.values()), 'real ambiguity must revoke sticky identity and face-learning permission'
        persons = [(box(0.,100.),.99),(box(300.,400.),.99)]
        tick(worker, readers, clock, 1)
        assert all(t['employee_id'] is None for t in worker.state.camera(1)['tracks'])


def test_clear_best_match_is_not_lost_to_two_weak_competitors():
    tracker = ByteTracker()
    first = tracker.update([(box(0.,100.),.99)],now=10.)[0]
    first.employee_id = 7
    result = tracker.update([(box(0.,100.),.99),(box(-46.,54.),.99),(box(46.,146.),.99)],now=10.2)
    original = next((t for t in result if t.track_id == first.track_id),None)
    assert original is not None, 'two equally weak side alternatives must not quarantine an exact match'
    assert original.employee_id == 7 and original.hits == 2
    assert not original.association_uncertain


def test_tentative_person_box_is_visible_without_identity_or_counted_time():
    with database() as (engine, _):
        ordinary(engine)
        clock = Clock()
        worker, readers, _, _, _ = tracking_worker(engine, clock)
        worker.service.min_hits = 2
        tick(worker, readers, clock, 1)
        people = worker.state.camera(1)['tracks']
        assert len(people) == 1, 'a high-confidence human must have a box before track confirmation'
        assert people[0]['employee_id'] is None and people[0]['track_id'] is None
        assert people[0]['visit_id'] is None and people[0]['state'] == 'detecting'
        assert not worker._identities or all(i.employee_id is None for i in worker._identities.values())


if __name__ == '__main__':
    cases = [v for k,v in list(globals().items()) if k.startswith('test_') and callable(v)]
    failures=0
    for case in cases:
        try: case(); print('PASS',case.__name__)
        except Exception: failures+=1; print('FAIL',case.__name__); traceback.print_exc()
    print(f'{len(cases)-failures}/{len(cases)} passed')
    raise SystemExit(bool(failures))
