"""Live overlay expiry uses server monotonic age and the configured AI cadence."""
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.ai.worker import DetectionState
from app.api.cameras import api_camera_detections, api_list_cameras
from app.config import settings
from app.database.models import AttendanceCamera, AttendanceRuntime, CameraProfile, RecognitionLearningRuntime
from test_attendance_frames import database, FrameReader


def detection():
    return SimpleNamespace(track_id=1, employee_id=None, employee_name=None, confidence=None,
                           person_score=.91, bbox=(.1, .1, .4, .8), state='detecting',
                           global_id=None, identity_reason_codes=(), visit_id=None)


def response(db, state, mono, wall):
    reader = FrameReader('a', 1, mono)
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(
        detection_state=state, manager=SimpleNamespace(get=lambda _id: reader))))
    with patch('app.ai.worker.time.monotonic', return_value=mono), patch('app.ai.worker.time.time', return_value=wall):
        return api_camera_detections(1, request, db)


def publish(db):
    state = DetectionState()
    config = SimpleNamespace(gallery_revision=db.get(AttendanceRuntime, 1).gallery_revision,
                             calibration_revision=db.get(AttendanceCamera, 1).revision)
    reader = FrameReader('a', 1, 100.)
    with patch('app.ai.worker.time.monotonic', return_value=100.), patch('app.ai.worker.time.time', return_value=1000.):
        state.update(1, [detection()], packet=reader.buffer.latest(), configuration=config)
    return state


def test_slow_camera_keeps_boxes_for_three_intervals_despite_wall_clock_jump():
    with database() as db:
        db.get(CameraProfile, 1).ai_fps = .5
        db.commit()
        state = publish(db)
        data = response(db, state, 104., -50000.)
        assert data['tracks'][0]['state'] == 'detecting'
        assert data['age_ms'] == 4000. and data['max_age_ms'] == 6000.
        assert response(db, state, 106.001, 200000.)['tracks'] == []


def test_global_rate_is_used_when_camera_has_no_override():
    with database() as db, patch.object(settings, 'ai_fps', .25):
        state = publish(db)
        assert response(db, state, 111., 200000.)['tracks']
        assert response(db, state, 112.001, -50000.)['tracks'] == []


def test_fast_camera_stale_state_expires_even_while_frames_are_fresh():
    with database() as db:
        state = publish(db)
        assert response(db, state, 100.1, -50000.)['tracks']
        assert response(db, state, 102.001, 1000.)['tracks'] == []


def test_stale_learning_revision_hides_only_body_names_in_cached_overlay():
    with database() as db:
        state = DetectionState()
        config = SimpleNamespace(gallery_revision=db.get(AttendanceRuntime, 1).gallery_revision,
                                 calibration_revision=db.get(AttendanceCamera, 1).revision,
                                 learning_revision=db.get(RecognitionLearningRuntime, 1).revision)
        reid, face = detection(), detection()
        face.track_id, face.bbox = 2, (.6, .15, .8, .8)
        with patch('app.ai.worker.time.monotonic', return_value=100.):
            state.update(1, [reid, face], packet=FrameReader('a', 1, 100.).buffer.latest(), configuration=config,
                         identities={1: (77, 'REID', 'Body-only'), 2: (88, 'FACE', 'Face-only')},
                         learning_revision=config.learning_revision)
        assert response(db, state, 100.2, 1000.)['known_count'] == 2
        db.get(RecognitionLearningRuntime, 1).revision += 1
        db.commit()
        out = response(db, state, 100.3, 1000.)
        assert out['people_count'] == 2 and out['known_count'] == 1 and out['unknown_count'] == 1
        assert [(t['track_id'], t['employee_id'], t['identity_source'], t['bbox']) for t in out['tracks']] == [
            (1, None, None, list(reid.bbox)), (2, 88, 'FACE', list(face.bbox))]
        assert state.camera(1)['tracks'][0]['employee_id'] == 77, 'HTTP projection must not alter cached worker state'
        request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(detection_state=state)))
        manager = SimpleNamespace(status=lambda _id: {})
        summary, = api_list_cameras(request, db, manager)
        assert (summary['people_count'], summary['known_count'], summary['unknown_count']) == (2, 1, 1)
        db.delete(db.get(RecognitionLearningRuntime, 1))
        db.commit()
        assert response(db, state, 100.4, 1000.)['tracks'][0]['employee_id'] is None


if __name__ == '__main__':
    cases = [value for name, value in list(globals().items()) if name.startswith('test_') and callable(value)]
    failures = 0
    for case in cases:
        try:
            case(); print('PASS', case.__name__)
        except Exception:
            failures += 1; print('FAIL', case.__name__); traceback.print_exc()
    print(f'{len(cases)-failures}/{len(cases)} passed')
    raise SystemExit(bool(failures))
