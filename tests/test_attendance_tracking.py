"""Deterministic local identity and lifecycle tests, without live inference.

Run: .venv/Scripts/python.exe tests/test_attendance_tracking.py
"""
import sys
import tempfile
import traceback
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from app.ai.face_recognition import EmbeddingStore
from app.ai.global_tracker import GlobalIdentityManager
from app.ai.recognition_service import DetectionResult, FrameOutcome, RecognitionService
from app.ai.tracker import ByteTracker
from app.database.attendance_schema import initialize_attendance_schema
from app.database.models import Camera, GlobalEvent, GlobalObservation, GlobalPerson

TESTS = []
STAMP = datetime(2026, 11, 11, 4)
FRAME = np.zeros((240, 320, 3), dtype=np.uint8)


def test(fn):
    TESTS.append(fn)
    return fn


def box(left=0., right=100.):
    return np.array([left, 0., right, 200.], dtype=np.float32)


def identified(track):
    track.employee_id = 7
    track.employee_conf = .93
    track.candidate_id = 7
    track.candidate_count = 3
    return track


@test
def test_high_score_reacquisition_within_grace_keeps_local_id_but_restarts_hits():
    tracker = ByteTracker(lost_timeout=3.0)
    first = tracker.update([(box(), .99)], now=10.)[0]
    tracker.update([(box(1., 101.), .99)], now=10.1)
    assert first.hits == 2
    assert tracker.update([], now=10.2) == []
    recovered = tracker.update([(box(2., 102.), .99)], now=10.3)
    assert len(recovered) == 1 and recovered[0].track_id == first.track_id
    assert recovered[0].hits == 1, "Old hits must not establish consecutive visibility"


@test
def test_expired_track_cannot_revive_on_high_or_low_detection():
    for score in (.99, .2):
        tracker = ByteTracker(lost_timeout=.5)
        first = identified(tracker.update([(box(), .99)], now=10.)[0])
        tracker.update([], now=10.1)
        result = tracker.update([(box(), score)], now=10.6)
        assert all(item.track_id != first.track_id for item in result)
        assert all(item.employee_id is None for item in result)
        assert all(item.track_id != first.track_id for item in tracker.tracks)
        expired = [item for item in tracker.lifecycle if item.state == "EXPIRED"]
        assert len(expired) == 1 and expired[0].track_id == first.track_id
        assert expired[0].last_seen_mono == 10.


@test
def test_expiration_precedes_matching_even_when_no_empty_frame_arrived():
    tracker = ByteTracker(lost_timeout=.5)
    first = tracker.update([(box(), .99)], now=10.)[0]
    result = tracker.update([(box(), .99)], now=10.6)
    assert result[0].track_id != first.track_id
    assert any(item.track_id == first.track_id and item.state == "EXPIRED"
               for item in tracker.lifecycle)


@test
def test_low_score_reacquisition_matches_remaining_lost_track():
    tracker = ByteTracker()
    first = tracker.update([(box(), .99)], now=10.)[0]
    tracker.update([], now=10.1)
    result = tracker.update([(box(1., 101.), .2)], now=10.2)
    assert len(result) == 1 and result[0].track_id == first.track_id
    assert result[0].hits == 1


@test
def test_one_detection_near_two_tracks_invalidates_both_identities_and_candidates():
    tracker = ByteTracker()
    first = tracker.update([(box(0., 100.), .99), (box(10., 110.), .99)], now=10.)
    for track in first:
        identified(track)
    result = tracker.update([(box(5., 105.), .99)], now=10.1)
    assert all(track.employee_id is None for track in tracker.tracks)
    assert all(track.employee_conf is None and track.candidate_id is None
               and track.candidate_count == 0 for track in tracker.tracks)
    assert all(track.association_uncertain for track in first)
    assert not result or all(track.association_uncertain for track in result)
    ambiguous = {item.track_id for item in tracker.lifecycle if item.state == "AMBIGUOUS"}
    assert ambiguous == {1, 2}, "Unmatched competing track also invalidates pending evidence"


@test
def test_one_track_near_two_detections_cannot_create_trusted_continuation():
    tracker = ByteTracker()
    first = identified(tracker.update([(box(), .99)], now=10.)[0])
    result = tracker.update([(box(-2., 98.), .99), (box(2., 102.), .99)], now=10.1)
    assert first.employee_id is None and first.candidate_count == 0
    assert first.association_uncertain
    assert not result or all(track.association_uncertain and track.employee_id is None
                             for track in result)
    assert any(item.track_id == first.track_id and item.state == "AMBIGUOUS"
               for item in tracker.lifecycle)


@test
def test_ambiguity_cannot_be_hidden_by_high_low_detection_partition():
    tracker = ByteTracker()
    first = identified(tracker.update([(box(), .99)], now=10.)[0])
    tracker.update([(box(-2., 98.), .99), (box(2., 102.), .2)], now=10.1)
    assert first.association_uncertain and first.employee_id is None
    assert first.candidate_id is None and first.candidate_count == 0


@test
def test_clear_pairing_keeps_greedy_high_score_matching_and_consecutive_hits():
    tracker = ByteTracker()
    first = tracker.update([(box(0., 100.), .99), (box(200., 300.), .99)], now=10.)
    identified(first[0])
    result = tracker.update([(box(201., 301.), .99), (box(1., 101.), .99)], now=10.1)
    by_id = {track.track_id: track for track in result}
    assert by_id[1].bbox[0] == 1. and by_id[2].bbox[0] == 201.
    assert by_id[1].employee_id == 7 and by_id[1].hits == 2
    assert all(not track.association_uncertain for track in result)


@test
def test_lifecycle_reports_lost_without_displayable_tracks_and_expires_once():
    tracker = ByteTracker(lost_timeout=.5)
    tracker.update([(box(), .99)], now=10.)
    assert tracker.update([], now=10.1) == []
    assert [(item.track_id, item.state) for item in tracker.lifecycle] == [(1, "LOST")]
    assert tracker.update([], now=10.6) == []
    assert [(item.track_id, item.state) for item in tracker.lifecycle] == [(1, "EXPIRED")]
    tracker.update([], now=10.7)
    assert tracker.lifecycle == []


class Detector:
    def __init__(self):
        self.persons = [(box(), .99)]

    def detect(self, frame):
        return self.persons


class NoFaces:
    def detect(self, image, max_num=0):
        return []


@test
def test_recognition_reset_always_changes_generation_and_clears_employee_cache():
    detector = Detector()
    service = RecognitionService(detector, NoFaces(), EmbeddingStore(), min_hits=1)
    first = service.process_frame(FRAME, 1, 10., frame_seq=1, received_at=STAMP)
    identified(service._trackers[1].tracks[0])
    generation = service.reset_camera(1)
    second = service.process_frame(FRAME, 1, 10.1, frame_seq=2,
                                   received_at=STAMP + timedelta(seconds=.1))
    assert first.tracking_generation != generation == second.tracking_generation
    assert second.detections[0].track_id == 1 and second.detections[0].employee_id is None
    third = service.reset_camera(1)
    assert third != generation


@test
def test_recognition_never_displays_cached_employee_after_ambiguous_overlap():
    detector = Detector()
    service = RecognitionService(detector, NoFaces(), EmbeddingStore(), min_hits=1)
    service.process_frame(FRAME, 1, 10., frame_seq=1, received_at=STAMP)
    identified(service._trackers[1].tracks[0])
    detector.persons = [(box(-2., 98.), .99), (box(2., 102.), .99)]
    outcome = service.process_frame(FRAME, 1, 10.1, frame_seq=2,
                                    received_at=STAMP + timedelta(seconds=.1))
    assert all(det.employee_id is None for det in outcome.detections)
    assert any(item.track_id == 1 and item.state == "AMBIGUOUS"
               for item in outcome.track_lifecycle)


class ReID:
    def __init__(self):
        self.index = 0

    def embed(self, crops):
        vector = np.zeros(512, dtype=np.float32)
        vector[self.index] = 1.
        return np.stack([vector] * len(crops))


def outcome(track_id=1):
    return FrameOutcome(detections=[DetectionResult(
        track_id=track_id, employee_id=None, employee_name=None, confidence=None,
        bbox=(.1, .1, .6, .95), state="unknown")])


@test
def test_camera_reset_drops_global_binding_not_persisted_global_history():
    with tempfile.TemporaryDirectory() as directory:
        engine = create_engine(f"sqlite:///{Path(directory).as_posix()}/tracking.db")

        @event.listens_for(engine, "connect")
        def foreign_keys(connection, _record):
            connection.execute("PRAGMA foreign_keys=ON")

        initialize_attendance_schema(engine)
        factory = sessionmaker(bind=engine)
        try:
            with factory() as db:
                db.add(Camera(id=1, name="Door", nvr_host="unused", username="u",
                              password="p", channel=1))
                db.commit()
            reid = ReID()
            manager = GlobalIdentityManager(factory, reid, topology=None)
            first = outcome()
            manager.process(1, first, FRAME, now=10.)
            first_id = first.detections[0].global_id
            assert first_id is not None
            with factory() as db:
                counts = (db.query(GlobalPerson).count(), db.query(GlobalObservation).count(),
                          db.query(GlobalEvent).count())
            manager.drop_camera(1)
            assert manager.camera_tracks(1) == []
            with factory() as db:
                assert counts == (db.query(GlobalPerson).count(), db.query(GlobalObservation).count(),
                                  db.query(GlobalEvent).count())
            reid.index = 1
            second = outcome()
            manager.process(1, second, FRAME, now=10.1)
            assert second.detections[0].global_id != first_id
        finally:
            engine.dispose()


if __name__ == "__main__":
    failed = 0
    for case in TESTS:
        try:
            case()
            print(f"PASS {case.__name__}")
        except Exception:
            failed += 1
            print(f"FAIL {case.__name__}")
            traceback.print_exc()
    print(f"{len(TESTS) - failed}/{len(TESTS)} passed")
    sys.exit(bool(failed))
