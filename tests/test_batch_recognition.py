"""Deterministic live batching, native face selection and identity isolation.

Run: .venv/Scripts/python.exe tests/test_batch_recognition.py
Only the external detector/face model observations are doubled. Tracking, gallery
matching, quality checks, receipt metadata and recognition voting are production code.
"""
import sys
import traceback
from dataclasses import FrozenInstanceError, replace
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from app.ai.recognition_service import RecognitionInput, RecognitionService
from app.rtsp.reader import FramePacket
from test_attendance_recognition import (
    CHECKER, MODEL_ID, STAMP, database, face, load_store, vector,
)

TESTS = []
PERSON = np.array([100., 0., 300., 500.], dtype=np.float32)


def test(fn):
    TESTS.append(fn)
    return fn


def packet(seq=1, mono=10., *, color=0, stream="one", revision=1, frame=None):
    if frame is None:
        frame = np.full((600, 640, 3), color, dtype=np.uint8)
    frame.flags.writeable = False
    return FramePacket(stream, revision, seq, STAMP + timedelta(seconds=mono - 10.), mono, frame)


class Detector:
    def __init__(self, persons=None):
        self.persons = persons if persons is not None else [(PERSON, .99)]
        self.calls = []

    def detect_batch(self, frames):
        self.calls.append(tuple(frames))
        return [self.persons for _frame in frames]


class Faces:
    model_id = MODEL_ID
    alignment_id = "arcface-5pt-112-v1"

    def __init__(self, observations=None):
        self.observations = [face()] if observations is None else observations
        self.crops = []
        self.chunks = []
        self.after_embed = None
        self.output_vectors = None

    def detect_faces(self, image):
        self.crops.append(image)
        return [replace(item, embedding=None) for item in self.observations]

    def embed_aligned(self, crops):
        self.chunks.append(tuple(crops))
        if self.after_embed is not None:
            self.after_embed()
        if self.output_vectors is not None:
            return np.stack(self.output_vectors[:len(crops)])
        return np.stack([vector() for _crop in crops])


def process(service, seq, mono, *, camera=1, attendance=False, **kwargs):
    return service.process_batch([
        RecognitionInput(camera, packet(seq, mono, **kwargs), attendance),
    ])[0]


def confirmed(service):
    process(service, 1, 10.)
    process(service, 2, 10.5)
    result = process(service, 3, 11.)
    assert result.detections[0].employee_id == 1
    return result


@test
def test_distinct_cameras_share_detector_and_embedding_batch_without_cross_assigning():
    with database() as engine:
        detector, faces = Detector(), Faces()
        faces.output_vectors = [vector(), vector(0., 1.)]
        service = RecognitionService(detector, faces, load_store(engine), min_hits=1)
        inputs = [RecognitionInput(9, packet(31, 12.3, color=9, stream="nine")),
                  RecognitionInput(2, packet(7, 10.1, color=2, stream="two"))]
        outcomes = service.process_batch(inputs)
        assert len(detector.calls) == 1 and len(detector.calls[0]) == 2
        assert detector.calls[0][0] is inputs[0].packet.frame
        assert detector.calls[0][1] is inputs[1].packet.frame
        assert [len(chunk) for chunk in faces.chunks] == [2]
        # Fair selection is camera-key ordered, not the caller's return ordering.
        assert [item.observed_tracks[0].fresh_face_evidence.candidate_employee_id
                for item in outcomes] == [2, 1]
        assert [item.observed_tracks[0].fresh_face_evidence.frame_seq
                for item in outcomes] == [31, 7]
        assert [item.observed_tracks[0].fresh_face_evidence.received_mono
                for item in outcomes] == [12.3, 10.1]
        assert outcomes[0].tracking_generation != outcomes[1].tracking_generation
        assert all(item.face_deferred == 0 and item.stage_times_ms["person"] >= 0.
                   for item in outcomes)
        assert all(set(item.stage_times_ms) == {
            "person", "tracking", "face_detection", "face_recognition",
        } for item in outcomes)
        try:
            inputs[0].camera_id = 3
        except (FrozenInstanceError, AttributeError):
            pass
        else:
            raise AssertionError("Admission metadata must be immutable")


@test
def test_duplicate_camera_is_rejected_before_models_and_empty_batch_is_empty():
    detector, faces = Detector(), Faces()
    service = RecognitionService(detector, faces, None)
    assert service.process_batch([]) == []
    try:
        service.process_batch([RecognitionInput(1, packet()), RecognitionInput(1, packet(2))])
    except ValueError:
        pass
    else:
        raise AssertionError("A detector batch must contain distinct cameras")
    assert detector.calls == [] and faces.crops == []


@test
def test_duplicate_packet_has_no_observation_vote_or_detector_work_and_first_hit_survives():
    with database() as engine:
        detector, faces = Detector(), Faces()
        service = RecognitionService(detector, faces, load_store(engine))
        first = process(service, 1, 10.)
        assert first.detections == [] and len(first.observed_tracks) == 1
        replay = process(service, 1, 10.5)
        assert replay.observed_tracks == [] and replay.detections == []
        assert len(detector.calls) == 1
        second = process(service, 2, 11.)
        assert len(second.observed_tracks) == len(second.detections) == 1
        assert service._trackers[1].tracks[0].hits == 2
        assert second.detections[0].fresh_face_evidence.supporting_frame_sequences == (2,)


@test
def test_mixed_replay_batch_detects_only_new_camera_packet():
    detector = Detector()
    service = RecognitionService(detector, None, None, min_hits=1)
    first = RecognitionInput(1, packet())
    service.process_batch([first])
    outcomes = service.process_batch([first, RecognitionInput(2, packet(stream="two"))])
    assert [len(call) for call in detector.calls] == [1, 1]
    assert not outcomes[0].observed_tracks and len(outcomes[1].observed_tracks) == 1
    assert outcomes[1].observed_tracks[0].identity_reason_codes == ("FACE_MODEL_UNAVAILABLE",)


@test
def test_head_crops_use_native_pixels_ten_percent_sides_and_seated_full_height():
    with database() as engine:
        detector, faces = Detector(), Faces()
        service = RecognitionService(detector, faces, load_store(engine), min_hits=1)
        native = np.arange(600 * 640 * 3, dtype=np.uint32).astype(np.uint8).reshape(600, 640, 3)
        process(service, 1, 10., frame=native)
        assert faces.crops[0].shape == (275, 240, 3)
        assert np.array_equal(faces.crops[0], native[0:275, 80:320])
        detector.persons = [(np.array([100., 100., 300., 400.]), .99)]
        process(service, 1, 10., camera=2, frame=native, stream="two")
        assert faces.crops[1].shape == (300, 240, 3)
        assert np.array_equal(faces.crops[1], native[100:400, 80:320])
        assert not native.flags.writeable


@test
def test_rejected_quality_alignment_ownership_and_models_never_reach_embedding():
    bad_faces = [
        (face(det_score=.79), "LOW_FACE_DETECTION"),
        (face(bbox=np.array([30., 30., 77., 130.])), "FACE_TOO_SMALL"),
        (face(native_bbox=np.array([30., 30., 77., 130.])), "FACE_TOO_SMALL"),
        (face(aligned=np.zeros((112, 112, 3), dtype=np.uint8)), "BLURRY_FACE"),
        (face(aligned=None), "BAD_ALIGNMENT"),
        (face(landmarks=np.full((5, 2), np.nan)), "BAD_ALIGNMENT"),
        (face(model_id="b" * 64), "FACE_MODEL_MISMATCH"),
    ]
    with database() as engine:
        for observation, reason in bad_faces:
            detector, faces = Detector(), Faces([observation])
            service = RecognitionService(detector, faces, load_store(engine), min_hits=1)
            result = process(service, 1, 10.).detections[0]
            assert result.employee_id is None and reason in result.identity_reason_codes
            assert faces.chunks == []
        faces = Faces([face(), face(bbox=np.array([120., 30., 220., 130.]))])
        service = RecognitionService(Detector(), faces, load_store(engine), min_hits=1)
        result = process(service, 1, 10.).detections[0]
        assert "AMBIGUOUS_FACE" in result.identity_reason_codes and faces.chunks == []
        # The second, undisplayed raw person head shares the face centre.
        detector = Detector([(PERSON, .99), (np.array([200., 0., 400., 500.]), .2)])
        faces = Faces([face(bbox=np.array([80., 30., 180., 130.]))])
        service = RecognitionService(detector, faces, load_store(engine), min_hits=1)
        result = process(service, 1, 10.).detections[0]
        assert "AMBIGUOUS_FACE" in result.identity_reason_codes and faces.chunks == []


@test
def test_engine_gallery_model_mismatch_never_detects_faces_or_confirms():
    with database() as engine:
        faces = Faces()
        faces.model_id = "b" * 64
        service = RecognitionService(Detector(), faces, load_store(engine), min_hits=1)
        for seq in range(1, 5):
            result = process(service, seq, 10. + seq * .5).detections[0]
            assert result.employee_id is None
            assert result.identity_reason_codes == ("FACE_MODEL_MISMATCH",)
        assert faces.crops == [] and faces.chunks == []


@test
def test_ordinary_margin_window_and_retry_cannot_borrow_attendance_policy():
    with database([(1, vector()), (2, vector(.95, .31225))]) as engine:
        service = RecognitionService(Detector(), Faces(), load_store(engine), min_hits=1)
        for seq in range(1, 5):
            result = process(service, seq, 10. + seq * .5).detections[0]
            assert result.employee_id is None and "AMBIGUOUS_FACE" in result.identity_reason_codes
    with database() as engine:
        faces = Faces()
        service = RecognitionService(Detector(), faces, load_store(engine), min_hits=1)
        process(service, 1, 10.)
        fast = process(service, 2, 10.25).detections[0]
        assert fast.fresh_face_evidence is None and len(faces.chunks) == 1
        process(service, 3, 11.1)
        late = process(service, 4, 12.2).detections[0]
        assert late.employee_id is None
        assert late.fresh_face_evidence.supporting_frame_sequences == (3, 4)
        final = process(service, 5, 12.7).detections[0]
        assert final.employee_id == 1
        assert final.fresh_face_evidence.supporting_frame_sequences == (3, 4, 5)


@test
def test_confirmed_ordinary_identity_rechecks_and_conflicting_observation_clears_it():
    with database() as engine:
        faces = Faces()
        service = RecognitionService(Detector(), faces, load_store(engine), min_hits=1)
        confirmed(service)
        faces.output_vectors = [vector(0., 1.)]
        cached = process(service, 4, 12.9).detections[0]
        assert cached.employee_id == 1 and cached.fresh_face_evidence is None
        conflicting = process(service, 5, 13.).detections[0]
        assert conflicting.employee_id is None
        assert conflicting.fresh_face_evidence.candidate_employee_id == 2
        assert conflicting.fresh_face_evidence.supporting_frame_sequences == (5,)


@test
def test_gallery_refresh_during_embedding_invalidates_every_camera_before_publication():
    with database() as engine:
        from sqlalchemy.orm import Session
        from app.database.models import AttendanceRuntime

        faces, store = Faces(), load_store(engine)
        service = RecognitionService(Detector(), faces, store, min_hits=1)
        for seq, mono in ((1, 10.), (2, 10.5)):
            service.process_batch([RecognitionInput(camera, packet(seq, mono, stream=str(camera)))
                                   for camera in (1, 2)])

        def refresh():
            with Session(engine) as db:
                db.get(AttendanceRuntime, 1).gallery_revision = 8
                db.commit()
                store.refresh(db)

        faces.after_embed = refresh
        outcomes = service.process_batch([
            RecognitionInput(camera, packet(3, 11., stream=str(camera))) for camera in (1, 2)
        ])
        assert all(item.detections[0].employee_id is None for item in outcomes)
        assert all("GALLERY_CHANGED" in item.detections[0].identity_reason_codes for item in outcomes)
        assert all(item.detections[0].fresh_face_evidence.store_revision == 7 for item in outcomes)
        assert all(not item.detections[0].fresh_face_evidence.supporting_frame_sequences
                   for item in outcomes)


@test
def test_embedding_budget_rotates_tracks_and_deferral_is_not_negative_evidence():
    with database() as engine:
        # Wide native image keeps 17 nonoverlapping head regions, each valid.
        people = [(np.array([i * 300. + 20., 0., i * 300. + 220., 500.]), .99)
                  for i in range(17)]
        detector, faces = Detector(people), Faces()
        service = RecognitionService(detector, faces, load_store(engine), min_hits=1)
        image = np.zeros((600, 5200, 3), dtype=np.uint8)
        first = process(service, 1, 10., frame=image)
        assert len(first.observed_tracks) == 17 and first.face_deferred == 1
        deferred = [item for item in first.observed_tracks if "FACE_DEFERRED" in item.identity_reason_codes]
        assert len(deferred) == 1 and deferred[0].fresh_face_evidence is None
        second = process(service, 2, 10.5, frame=image)
        deferred_track = next(item for item in second.observed_tracks
                              if item.track_id == deferred[0].track_id)
        assert deferred_track.fresh_face_evidence.supporting_frame_sequences == (2,)
        assert max(map(len, faces.chunks)) <= 16
        # At the next attempt a previously sampled track is deferred, retaining
        # its first real vote instead of recording an artificial negative match.
        newly_deferred = next(item for item in second.observed_tracks
                              if "FACE_DEFERRED" in item.identity_reason_codes)
        run = service._face_runs[(1, newly_deferred.track_id)]
        assert [vote[0] for vote in run.votes] == [1]


@test
def test_attendance_attempts_precede_ordinary_budget_and_keep_independent_native_minimum():
    with database() as engine:
        people = [(np.array([i * 300. + 20., 0., i * 300. + 220., 500.]), .99)
                  for i in range(16)]
        faces = Faces([face(bbox=np.array([30., 30., 90., 90.]))])
        detector = Detector(people)
        service = RecognitionService(detector, faces, load_store(engine), min_hits=1)
        image = np.zeros((600, 4900, 3), dtype=np.uint8)
        outcomes = service.process_batch([
            RecognitionInput(1, packet(frame=image)),
            RecognitionInput(2, packet(frame=image, stream="attendance"), True),
        ])
        assert len(outcomes[0].observed_tracks) == len(outcomes[1].observed_tracks) == 16
        assert outcomes[0].face_deferred == 0
        assert all("FACE_TOO_SMALL" in item.identity_reason_codes for item in outcomes[1].observed_tracks)
        assert [len(chunk) for chunk in faces.chunks] == [16]
        # With eligible attendance faces they take all 16 slots, irrespective of
        # camera input ordering; ordinary work is explicitly deferred.
        faces.observations = [face()]
        outcomes = service.process_batch([
            RecognitionInput(1, packet(2, 10.5, frame=image)),
            RecognitionInput(2, packet(2, 10.5, frame=image, stream="attendance"), True),
        ])
        assert outcomes[0].face_deferred == 16 and outcomes[1].face_deferred == 0
        assert all(item.fresh_face_evidence is not None for item in outcomes[1].observed_tracks)


@test
def test_ordinary_low_similarity_does_not_confirm_with_an_unambiguous_owner():
    with database([(1, vector(.59, np.sqrt(1. - .59 ** 2)))]) as engine:
        service = RecognitionService(Detector(), Faces(), load_store(engine), min_hits=1)
        for seq in range(1, 5):
            result = process(service, seq, 10. + seq * .5).detections[0]
            assert result.employee_id is None and "LOW_FACE_SCORE" in result.identity_reason_codes


@test
def test_positive_ordinary_recheck_retains_confirmation_without_fabricating_votes():
    with database() as engine:
        service = RecognitionService(Detector(), Faces(), load_store(engine), min_hits=1)
        confirmed(service)
        result = process(service, 4, 13.1).detections[0]
        assert result.employee_id == 1
        assert result.fresh_face_evidence.supporting_frame_sequences == (4,)
        assert result.fresh_face_evidence.confirmed_at_mono == 11.


@test
def test_attendance_switch_cannot_borrow_ordinary_identity_even_on_replayed_pixels():
    with database() as engine:
        service = RecognitionService(Detector(), Faces(), load_store(engine), min_hits=1)
        confirmed(service)
        replay = process(service, 3, 11., attendance=True)
        assert not replay.observed_tracks and replay.detections[0].employee_id is None
        assert process(service, 4, 11.25, attendance=True).detections[0].employee_id is None
        assert process(service, 5, 11.5, attendance=True).detections[0].employee_id is None
        assert process(service, 6, 11.75, attendance=True).detections[0].employee_id == 1


@test
def test_gallery_publication_inside_matching_does_not_mislabel_new_revision_scores():
    with database() as engine:
        from unittest.mock import patch
        from sqlalchemy.orm import Session
        from app.database.models import AttendanceRuntime

        store = load_store(engine)
        original_match = store.match

        def publish_before_match(embedding):
            with Session(engine) as db:
                db.get(AttendanceRuntime, 1).gallery_revision = 8
                db.commit()
                store.refresh(db)
            return original_match(embedding)

        service = RecognitionService(Detector(), Faces(), store, min_hits=1)
        with patch.object(store, "match", side_effect=publish_before_match):
            result = process(service, 1, 10.).detections[0]
        evidence = result.fresh_face_evidence
        assert result.employee_id is None and evidence.store_revision == 7
        assert evidence.candidate_employee_id is None and evidence.candidate_employee_name is None
        assert evidence.supporting_frame_sequences == ()
        assert "GALLERY_CHANGED" in evidence.reason_codes


@test
def test_busy_camera_cannot_take_every_ordinary_embedding_slot():
    with database() as engine:
        people = [(np.array([i * 300. + 20., 0., i * 300. + 220., 500.]), .99)
                  for i in range(17)]

        class PerCameraDetector(Detector):
            def detect_batch(self, frames):
                self.calls.append(tuple(frames))
                return [people if frame.shape[1] > 1000 else [(PERSON, .99)] for frame in frames]

        faces = Faces()
        service = RecognitionService(PerCameraDetector(), faces, load_store(engine), min_hits=1)
        outcomes = service.process_batch([
            RecognitionInput(1, packet(frame=np.zeros((600, 5200, 3), dtype=np.uint8))),
            RecognitionInput(2, packet(stream="two")),
        ])
        assert outcomes[0].face_deferred == 2 and outcomes[1].face_deferred == 0
        assert outcomes[1].observed_tracks[0].fresh_face_evidence.supporting_frame_sequences == (1,)
        assert [len(chunk) for chunk in faces.chunks] == [16]


@test
def test_unsampled_unknown_keeps_last_quality_reason_without_fresh_evidence():
    with database() as engine:
        faces = Faces([face(det_score=.79)])
        service = RecognitionService(Detector(), faces, load_store(engine), min_hits=1)
        first = process(service, 1, 10.).detections[0]
        assert "LOW_FACE_DETECTION" in first.identity_reason_codes
        unsampled = process(service, 2, 10.1).detections[0]
        assert unsampled.fresh_face_evidence is None
        assert unsampled.identity_reason_codes == first.identity_reason_codes
        faces.observations = [face()]
        eligible = process(service, 3, 10.5).detections[0]
        assert eligible.identity_reason_codes == ()
        assert eligible.fresh_face_evidence.supporting_frame_sequences == (3,)


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
