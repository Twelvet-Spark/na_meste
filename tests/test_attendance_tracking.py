"""Deterministic local identity and lifecycle tests, without live inference.

Run: .venv/Scripts/python.exe tests/test_attendance_tracking.py
"""
import sys
import tempfile
import traceback
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from uuid import NAMESPACE_URL, uuid5

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from sqlalchemy import create_engine, event, select
from sqlalchemy.orm import sessionmaker

from app.ai.face_recognition import EmbeddingStore
from app.ai.global_tracker import EmbeddingItem, GlobalIdentityManager
from app.ai.recognition_service import RecognitionInput, RecognitionService
from app.ai.tracker import ByteTracker
from app.database.attendance_schema import initialize_attendance_schema
from app.database.models import Camera, GlobalEvent, GlobalObservation, GlobalPerson
from app.rtsp.reader import FramePacket

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

    def detect_batch(self, frames):
        return [self.persons for _frame in frames]


class NoFaces:
    model_id = "a" * 64
    alignment_id = "arcface-5pt-112-v1"

    def detect_faces(self, image):
        return []

    def embed_aligned(self, crops):
        raise AssertionError("No detected faces may be embedded")


def process(service, seq, mono):
    packet = FramePacket("tracking-test", 1, seq,
                         STAMP + timedelta(seconds=mono - 10.), mono, FRAME)
    return service.process_batch([RecognitionInput(1, packet)])[0]


@test
def test_recognition_reset_always_changes_generation_and_clears_employee_cache():
    detector = Detector()
    service = RecognitionService(detector, NoFaces(), EmbeddingStore("a" * 64), min_hits=1)
    first = process(service, 1, 10.)
    identified(service._trackers[1].tracks[0])
    generation = service.reset_camera(1)
    second = process(service, 2, 10.1)
    assert first.tracking_generation != generation == second.tracking_generation
    assert second.detections[0].track_id == 1 and second.detections[0].employee_id is None
    third = service.reset_camera(1)
    assert third != generation


@test
def test_recognition_never_displays_cached_employee_after_ambiguous_overlap():
    detector = Detector()
    service = RecognitionService(detector, NoFaces(), EmbeddingStore("a" * 64), min_hits=1)
    process(service, 1, 10.)
    identified(service._trackers[1].tracks[0])
    detector.persons = [(box(-2., 98.), .99), (box(2., 102.), .99)]
    outcome = process(service, 2, 10.1)
    assert all(det.employee_id is None for det in outcome.detections)
    assert any(item.track_id == 1 and item.state == "AMBIGUOUS"
               for item in outcome.track_lifecycle)


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
                db.add(GlobalPerson(id=41, last_camera_id=1, last_track_id=8, status="LOST"))
                db.flush()
                db.add(GlobalObservation(global_id=41, camera_id=1, track_id=8,
                                         embedding=b"archived-vector", snapshot=b"archived-image"))
                db.add(GlobalEvent(global_id=41, camera_id=1, track_id=8,
                                   event_type="person_lost", payload='{"duration_seconds":5}'))
                db.commit()

            def archive():
                with engine.connect() as connection:
                    return tuple(tuple(connection.execute(select(model.__table__)).all())
                                 for model in (GlobalPerson, GlobalObservation, GlobalEvent))

            before = archive()
            vector = np.zeros(512, dtype=np.float32)
            vector[0] = 1.
            value = EmbeddingItem(
                visit_id=str(uuid5(NAMESPACE_URL, "tracking-visit/1")),
                camera_id=1, tracking_generation=str(uuid5(NAMESPACE_URL, "tracking-generation/1")),
                track_id=1, source_revision=1, stream_id="tracking-stream-1", packet_seq=1,
                observed_at=STAMP, received_mono=10., bbox=(.1, .1, .6, .95),
                aspect=2., crop_quality=.9, embedding=vector, employee_id=None,
            )
            manager = GlobalIdentityManager()
            first = manager.process_embeddings([value])[0]

            def acknowledgement(proposal):
                return SimpleNamespace(
                    kind="LINK_COMMITTED", visit_id=proposal.visit_id, camera_id=1,
                    tracking_generation=proposal.tracking_generation, track_id=1,
                    source_revision=1, stream_id=proposal.stream_id, packet_seq=proposal.packet_seq,
                    link_key=proposal.link_key, global_id=41, reason=None,
                )

            manager.acknowledge(acknowledgement(first))
            assert manager.global_id_for(value.visit_id) == 41
            old_pending = manager.process_embeddings([
                replace(value, packet_seq=2, received_mono=10.1)])[0]
            manager.drop_camera(1)
            assert manager.camera_tracks(1) == [] and manager.global_id_for(value.visit_id) is None
            manager.acknowledge(acknowledgement(old_pending))
            assert manager.global_id_for(value.visit_id) is None
            vector = np.zeros(512, dtype=np.float32)
            vector[1] = 1.
            new_generation = replace(
                value, visit_id=str(uuid5(NAMESPACE_URL, "tracking-visit/2")),
                tracking_generation=str(uuid5(NAMESPACE_URL, "tracking-generation/2")),
                stream_id="tracking-stream-2", received_mono=10.2, embedding=vector)
            second = manager.process_embeddings([new_generation])[0]
            assert second.global_id is None, "A recycled local track ID does not inherit a binding"
            assert manager.global_id_for(new_generation.visit_id) is None
            assert archive() == before, "Dropping local generations never mutates persisted history"
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
