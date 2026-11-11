"""Deterministic employee identity, enrollment and gallery transaction contracts.

Run: .venv/Scripts/python.exe tests/test_attendance_recognition.py
Only model observations are synthetic; matching, tracking, voting, SQL and enrollment
are the real production implementations. Every database is an isolated temporary file.
"""
import sys
import tempfile
import traceback
from contextlib import contextmanager
from dataclasses import FrozenInstanceError, replace
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4
import hashlib

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
from fastapi import HTTPException
from sqlalchemy import create_engine, event
from sqlalchemy.exc import OperationalError, SQLAlchemyError
from sqlalchemy.orm import Session

from app.ai.face_detector import FaceEngine, FaceInfo
from app.ai.face_recognition import EmbeddingStore
from app.ai.recognition_service import RecognitionService
from app.config import settings
from app.database.attendance_schema import initialize_attendance_schema
from app.database.models import AttendanceRuntime, Employee, EmployeeFace
from app.services import employee_service

TESTS = []
STAMP = datetime(2026, 11, 11, 4)
FRAME = np.zeros((600, 640, 3), dtype=np.uint8)
CHECKER = np.repeat(((np.indices((112, 112)).sum(axis=0) % 2) * 255)
                    .astype(np.uint8)[:, :, None], 3, axis=2)
PERSON = np.array([100., 0., 300., 500.], dtype=np.float32)


def test(fn):
    TESTS.append(fn)
    return fn


def vector(first=1., second=0.):
    result = np.zeros(512, dtype=np.float32)
    result[:2] = first, second
    norm = np.linalg.norm(result)
    return result / norm if norm else result


def face(embedding=None, **changes):
    observation = FaceInfo(
        bbox=np.array([30., 30., 130., 130.], dtype=np.float32),
        det_score=.99, embedding=vector() if embedding is None else embedding,
        aligned=CHECKER,
    )
    return replace(observation, **changes)


@contextmanager
def database(samples=None, revision=7):
    with tempfile.TemporaryDirectory() as directory:
        engine = create_engine(f"sqlite:///{(Path(directory) / 'identity.db').as_posix()}",
                               connect_args={"timeout": 30})

        @event.listens_for(engine, "connect")
        def foreign_keys(connection, _record):
            connection.execute("PRAGMA foreign_keys=ON")

        try:
            initialize_attendance_schema(engine)
            with Session(engine) as db:
                db.add_all([Employee(id=1, name="First"), Employee(id=2, name="Second")])
                db.flush()
                for owner, embedding in (samples if samples is not None else
                                         [(1, vector()), (2, vector(0., 1.))]):
                    db.add(EmployeeFace(employee_id=owner, embedding=embedding.tobytes()))
                db.get(AttendanceRuntime, 1).gallery_revision = revision
                db.commit()
            yield engine
        finally:
            engine.dispose()


def load_store(engine):
    store = EmbeddingStore()
    with Session(engine) as db:
        store.refresh(db)
    return store


class Persons:
    def __init__(self):
        self.observations = [(PERSON, .99)]

    def detect(self, frame):
        return self.observations


class Faces:
    """Synthetic model output; honor max_num so truncation cannot hide ambiguity."""
    def __init__(self):
        self.observations = [face()]

    def detect(self, image, max_num=0):
        return self.observations[:max_num] if max_num else self.observations


def recognition(engine, **kwargs):
    people, faces, store = Persons(), Faces(), load_store(engine)
    return RecognitionService(people, faces, store, min_hits=1, **kwargs), people, faces, store


def process(service, seq, mono, *, attendance=True):
    return service.process_frame(
        FRAME, 1, mono, frame_seq=seq,
        received_at=STAMP + timedelta(seconds=mono - 10.), attendance=attendance,
    )


def detection(service, seq, mono, **kwargs):
    outcome = process(service, seq, mono, **kwargs)
    assert len(outcome.detections) == 1
    return outcome.detections[0]


def confirm(service):
    assert detection(service, 1, 10.).employee_id is None
    assert detection(service, 2, 10.25).employee_id is None
    result = detection(service, 3, 10.5)
    assert result.employee_id == 1
    return result


@test
def test_distinct_employee_runner_up_ignores_multiple_photos_of_winner():
    with database([(1, vector()), (1, vector(.999, .001)),
                   (2, vector(.8, .6))]) as engine:
        match = load_store(engine).match(vector())
        assert match.employee_id == 1 and abs(match.score - 1.) < 1e-6
        assert match.runner_up_employee_id == 2
        assert abs(match.runner_up_score - .8) < 1e-6
        assert abs(match.margin - .2) < 1e-6 and match.store_revision == 7


@test
def test_excluding_self_ranks_other_employee_and_absent_runner_has_zero_score():
    with database([(1, vector()), (1, vector()), (2, vector(.8, .6))]) as engine:
        match = load_store(engine).match(vector(), exclude_employee_id=1)
        assert match.employee_id == 2 and abs(match.score - .8) < 1e-6
        assert match.runner_up_employee_id is None and match.runner_up_score == 0.
        assert abs(match.margin - .8) < 1e-6


@test
def test_empty_invalid_nonfinite_zero_and_wrong_dimension_queries_do_not_match():
    with database() as engine:
        store = load_store(engine)
        for query in (np.array([], dtype=np.float32), np.zeros(512, dtype=np.float32),
                      np.full(512, np.nan, dtype=np.float32),
                      np.full(512, np.inf, dtype=np.float32), np.ones(3, dtype=np.float32)):
            match = store.match(query)
            assert match.employee_id is None and match.score == 0.
            assert match.runner_up_employee_id is None and match.store_revision == 7
        with Session(engine) as db:
            db.query(EmployeeFace).delete()
            db.commit()
            store.refresh(db)
        assert store.match(vector()).employee_id is None


@test
def test_invalid_gallery_samples_cannot_poison_valid_employee_match():
    with database([(1, np.full(512, np.nan, dtype=np.float32)),
                   (1, np.zeros(512, dtype=np.float32)), (1, vector()),
                   (2, np.ones(3, dtype=np.float32))]) as engine:
        match = load_store(engine).match(vector())
        assert match.employee_id == 1 and abs(match.score - 1.) < 1e-6
        assert match.runner_up_employee_id is None


@test
def test_attendance_near_tie_never_confirms_even_after_many_frames():
    with database([(1, vector()), (2, vector(.98, .199))]) as engine:
        service, _people, _faces, _store = recognition(engine)
        for seq in range(1, 8):
            result = detection(service, seq, 10. + (seq - 1) * .25)
            assert result.employee_id is None
            evidence = result.fresh_face_evidence
            assert evidence.candidate_employee_id == 1
            assert evidence.runner_up_employee_id == 2 and evidence.margin < .08


@test
def test_low_similarity_cannot_confirm_even_with_clear_distinct_employee_margin():
    with database([(1, vector(.54, .841665))]) as engine:
        service, _people, _faces, _store = recognition(engine)
        for seq in (1, 2, 3, 4):
            assert detection(service, seq, 10. + (seq - 1) * .25).employee_id is None


@test
def test_three_distinct_fresh_frames_confirm_and_publish_immutable_evidence():
    with database() as engine:
        service, _people, _faces, _store = recognition(engine)
        result = confirm(service)
        assert result.person_score == .99 and not result.association_uncertain
        evidence = result.fresh_face_evidence
        assert evidence.frame_seq == 3 and evidence.received_mono == 10.5
        assert evidence.received_at == STAMP + timedelta(seconds=.5)
        assert evidence.confirmed_employee_id == 1 and evidence.store_revision == 7
        assert evidence.supporting_frame_sequences == (1, 2, 3)
        assert evidence.quality.aligned and evidence.quality.blur >= 20.
        assert evidence.aligned_jpeg and len(evidence.aligned_jpeg) <= 32 * 1024
        assert evidence.face_bbox == (120., 30., 220., 130.)
        try:
            evidence.confirmed_employee_id = 2
        except (FrozenInstanceError, AttributeError):
            pass
        else:
            raise AssertionError("Capture evidence must be immutable")


@test
def test_repeated_frame_sequence_cannot_supply_multiple_confirmation_votes():
    with database() as engine:
        service, _people, _faces, _store = recognition(engine)
        for mono in (10., 10.25, 10.5):
            assert detection(service, 1, mono).employee_id is None
        assert detection(service, 2, 10.75).employee_id is None
        assert detection(service, 3, 11.).employee_id == 1


@test
def test_replayed_older_sequence_cannot_supply_a_new_vote():
    with database() as engine:
        service, _people, _faces, _store = recognition(engine)
        assert detection(service, 10, 10.).employee_id is None
        assert detection(service, 11, 10.25).employee_id is None
        assert detection(service, 9, 10.5).employee_id is None
        result = detection(service, 12, 10.75)
        assert result.employee_id == 1
        assert result.fresh_face_evidence.supporting_frame_sequences == (10, 11, 12)


@test
def test_sampling_interval_cannot_accelerate_confirmation_with_fast_frames():
    with database() as engine:
        service, _people, _faces, _store = recognition(engine)
        assert detection(service, 1, 10.).employee_id is None
        assert detection(service, 2, 10.1).employee_id is None
        assert detection(service, 3, 10.2).employee_id is None
        assert detection(service, 4, 10.25).employee_id is None
        assert detection(service, 5, 10.5).employee_id == 1


@test
def test_votes_older_than_window_cannot_confirm_a_new_crossing():
    with database() as engine:
        service, _people, _faces, _store = recognition(engine)
        assert detection(service, 1, 10.).employee_id is None
        assert detection(service, 2, 10.75).employee_id is None
        assert detection(service, 3, 11.51).employee_id is None
        result = detection(service, 4, 11.8)
        assert result.employee_id == 1
        assert result.fresh_face_evidence.supporting_frame_sequences == (2, 3, 4)


@test
def test_no_face_breaks_candidate_run_and_clears_existing_confirmation():
    with database() as engine:
        service, _people, faces, _store = recognition(engine)
        detection(service, 1, 10.)
        detection(service, 2, 10.25)
        faces.observations = []
        assert detection(service, 3, 10.5).employee_id is None
        faces.observations = [face()]
        assert detection(service, 4, 10.75).employee_id is None
        assert detection(service, 5, 11.).employee_id is None
        assert detection(service, 6, 11.25).employee_id == 1
        faces.observations = []
        assert detection(service, 7, 11.5).employee_id is None


@test
def test_conflicting_employee_immediately_invalidates_old_confirmation():
    with database() as engine:
        service, _people, faces, _store = recognition(engine)
        confirm(service)
        faces.observations = [face(vector(0., 1.))]
        assert detection(service, 4, 10.75).employee_id is None
        assert detection(service, 5, 11.).employee_id is None
        assert detection(service, 6, 11.25).employee_id == 2


@test
@patch.object(settings, "attendance_max_frame_gap", 2.0)
def test_confirmation_expires_without_a_new_distinct_supporting_observation():
    with database() as engine:
        service, _people, _faces, _store = recognition(engine)
        confirm(service)
        assert detection(service, 3, 11.).employee_id == 1
        assert detection(service, 3, 11.26).employee_id is None
        assert detection(service, 4, 11.5).employee_id is None


@test
def test_lost_and_ambiguous_lifecycle_clear_votes_before_reacquisition():
    with database() as engine:
        service, people, _faces, _store = recognition(engine)
        confirm(service)
        people.observations = []
        lost = process(service, 4, 10.6)
        assert not lost.detections
        assert any(item.state == "LOST" for item in lost.track_lifecycle)
        people.observations = [(PERSON, .99)]
        assert detection(service, 5, 10.75).employee_id is None
        assert detection(service, 6, 11.).employee_id is None
        assert detection(service, 7, 11.25).employee_id == 1
        people.observations = [(PERSON + np.array([-2., 0., -2., 0.]), .99),
                               (PERSON + np.array([2., 0., 2., 0.]), .99)]
        ambiguous = process(service, 8, 11.5)
        assert any(item.state == "AMBIGUOUS" for item in ambiguous.track_lifecycle)
        assert all(item.employee_id is None for item in ambiguous.detections)
        people.observations = [(PERSON, .99)]
        assert detection(service, 9, 11.75).employee_id is None


@test
def test_gallery_refresh_during_third_face_observation_requires_three_new_revision_votes():
    with database() as engine:
        service, _people, faces, store = recognition(engine)
        assert detection(service, 1, 10.).employee_id is None
        assert detection(service, 2, 10.25).employee_id is None
        original_detect = faces.detect

        def refresh_then_detect(image, max_num=0):
            with Session(engine) as db:
                employee_service.update_employee(
                    db, db.get(Employee, 1),
                    employee_service.EmployeeUpdate(expected_revision=db.get(Employee, 1).revision,
                                                    last_name="First", first_name="revised"),
                )
                store.refresh(db)
            return original_detect(image, max_num=max_num)

        with patch.object(faces, "detect", side_effect=refresh_then_detect):
            result = detection(service, 3, 10.5)
        assert result.employee_id is None, "Two old-revision votes cannot confirm a new gallery"
        assert result.fresh_face_evidence.store_revision == 8
        assert result.fresh_face_evidence.confirmed_employee_id is None
        assert result.fresh_face_evidence.supporting_frame_sequences == (3,)
        assert detection(service, 4, 10.75).employee_id is None
        result = detection(service, 5, 11.)
        assert result.employee_id == 1 and result.employee_name == "First revised"
        assert result.fresh_face_evidence.store_revision == 8
        assert result.fresh_face_evidence.supporting_frame_sequences == (3, 4, 5)


@test
def test_in_frame_refresh_invalidates_other_track_result_already_materialized():
    with database() as engine:
        service, people, faces, store = recognition(engine)
        people.observations = [
            (PERSON, .99), (np.array([400., 0., 600., 500.]), .99),
        ]
        for seq, mono in ((1, 10.), (2, 10.25), (3, 10.5)):
            prior = process(service, seq, mono)
        assert [item.employee_id for item in prior.detections] == [1, 1]
        original_detect = faces.detect
        calls = 0

        def refresh_on_second_head(image, max_num=0):
            nonlocal calls
            calls += 1
            if calls == 2:
                with Session(engine) as db:
                    employee_service.update_employee(
                        db, db.get(Employee, 1),
                        employee_service.EmployeeUpdate(expected_revision=db.get(Employee, 1).revision,
                                                        last_name="First", first_name="revised"),
                    )
                    store.refresh(db)
            return original_detect(image, max_num=max_num)

        with patch.object(faces, "detect", side_effect=refresh_on_second_head):
            outcome = process(service, 4, 10.75)
        assert store.match(vector()).store_revision == 8
        assert len(outcome.detections) == 2
        assert all(item.employee_id is None for item in outcome.detections)
        old_evidence = outcome.detections[0].fresh_face_evidence
        assert old_evidence.store_revision == 7
        assert old_evidence.confirmed_employee_id == 1
        assert old_evidence.candidate_employee_name == "First"
        assert old_evidence.supporting_frame_sequences == (2, 3, 4)
        assert old_evidence.confirmed_at_mono == 10.5
        assert "GALLERY_CHANGED" in old_evidence.reason_codes
        new_evidence = outcome.detections[1].fresh_face_evidence
        assert new_evidence.store_revision == 8
        assert new_evidence.confirmed_employee_id is None
        assert new_evidence.supporting_frame_sequences == (4,)


@test
def test_in_frame_refresh_clears_other_tracks_cached_confirmation_without_resampling():
    with database() as engine:
        service, people, faces, store = recognition(engine)
        process(service, 1, 10.)
        people.observations = [
            (PERSON, .99), (np.array([400., 0., 600., 500.]), .99),
        ]
        for seq, mono in ((2, 10.1), (3, 10.25), (4, 10.36), (5, 10.5), (6, 10.62)):
            prior = process(service, seq, mono)
        assert [item.employee_id for item in prior.detections] == [1, 1]
        original_detect = faces.detect
        refreshed = False

        def refresh_then_detect(image, max_num=0):
            nonlocal refreshed
            if not refreshed:
                refreshed = True
                with Session(engine) as db:
                    employee_service.update_employee(
                        db, db.get(Employee, 1),
                        employee_service.EmployeeUpdate(expected_revision=db.get(Employee, 1).revision,
                                                        last_name="First", first_name="revised"),
                    )
                    store.refresh(db)
            return original_detect(image, max_num=max_num)

        with patch.object(faces, "detect", side_effect=refresh_then_detect):
            outcome = process(service, 7, 10.75)
        assert store.match(vector()).store_revision == 8
        assert len(outcome.detections) == 2
        assert all(item.employee_id is None for item in outcome.detections)


@test
def test_gallery_deactivation_invalidates_confirmed_identity_before_next_sampling():
    with database() as engine:
        service, _people, _faces, store = recognition(engine)
        confirm(service)
        with Session(engine) as db:
            employee_service.update_employee(db, db.get(Employee, 1),
                                             employee_service.EmployeeUpdate(expected_revision=0, active=False))
            store.refresh(db)
        result = detection(service, 4, 10.6)
        assert result.employee_id is None
        for seq in (5, 6, 7):
            assert detection(service, seq, 10.75 + (seq - 5) * .25).employee_id is None
        assert store.match(vector()).employee_id != 1


@test
def test_face_replacement_invalidates_confirmation_and_uses_committed_gallery_revision():
    with database() as engine:
        service, _people, _faces, store = recognition(engine)
        confirm(service)
        with Session(engine) as db:
            sample = db.query(EmployeeFace).filter_by(employee_id=1).one()
            replace_sample(db, store, sample)
        assert detection(service, 4, 10.6).employee_id is None
        assert store.match(vector()).store_revision == 8


@test
def test_multiple_faces_cannot_be_hidden_by_single_face_detector_limit():
    with database() as engine:
        service, _people, faces, _store = recognition(engine)
        faces.observations = [face(), face(vector(0., 1.),
                                          bbox=np.array([110., 40., 210., 140.]))]
        for seq in (1, 2, 3, 4):
            assert detection(service, seq, 10. + (seq - 1) * .25).employee_id is None


@test
def test_shared_face_center_is_rejected_even_when_other_person_is_not_display_eligible():
    with database() as engine:
        service, people, faces, _store = recognition(engine)
        service.min_hits = 3
        # Face center x=270 after crop offset x=90; both head regions own it.
        faces.observations = [face(bbox=np.array([135., 30., 225., 130.]))]
        process(service, 1, 10.)
        process(service, 2, 10.25)
        other = np.array([260., 0., 460., 500.], dtype=np.float32)
        people.observations = [(PERSON, .99), (other, .99)]
        outcome = process(service, 3, 10.5)
        assert len(outcome.person_regions) == 2
        assert len(outcome.detections) == 1 and outcome.detections[0].employee_id is None
        assert outcome.detections[0].fresh_face_evidence.confirmed_employee_id is None
        for seq in (4, 5, 6):
            assert all(item.employee_id is None for item in process(
                service, seq, 10. + (seq - 1) * .25).detections)


@test
def test_bad_quality_and_nonfinite_embedding_break_live_confirmation():
    bad = [face(aligned=None), face(aligned=np.zeros((112, 112, 3), dtype=np.uint8)),
           face(det_score=.79), face(bbox=np.array([30., 30., 109., 130.])),
           face(embedding=np.full(512, np.nan, dtype=np.float32)),
           face(embedding=np.zeros(512, dtype=np.float32))]
    for observation in bad:
        with database() as engine:
            service, _people, faces, _store = recognition(engine)
            confirm(service)
            faces.observations = [observation]
            assert detection(service, 4, 10.75).employee_id is None
            faces.observations = [face()]
            assert detection(service, 5, 11.).employee_id is None


@test
def test_nonattendance_camera_retains_its_existing_threshold_and_confirmation_count():
    with database([(1, vector(.5, .8660254))]) as engine:
        service, _people, _faces, _store = recognition(
            engine, min_confidence=.45, min_confirmations=2, retry_interval=.25)
        assert detection(service, 1, 10., attendance=False).employee_id is None
        assert detection(service, 2, 10.25, attendance=False).employee_id == 1


@test
def test_same_stream_reset_changes_generation_and_discards_face_confirmation():
    with database() as engine:
        service, _people, _faces, _store = recognition(engine)
        confirm(service)
        old = process(service, 4, 10.75).tracking_generation
        generation = service.reset_camera(1)
        outcome = process(service, 5, 11.)
        assert old != generation == outcome.tracking_generation
        assert outcome.detections[0].track_id == 1 and outcome.detections[0].employee_id is None
        assert service.reset_camera(1) != generation


@test
def test_quality_measurement_exposes_alignment_failure_without_inventing_blur():
    from app.ai.face_quality import measure_face_quality

    quality = measure_face_quality(face())
    assert quality.min_size == 100. and quality.det_score == .99
    assert quality.aligned and quality.blur > 20.
    missing = measure_face_quality(face(aligned=None))
    assert not missing.aligned and missing.blur is None
    assert measure_face_quality(face(aligned=np.zeros((112, 112, 3), dtype=np.uint8))).blur == 0.


@test
def test_face_engine_alignment_failure_keeps_aligned_none():
    model_face = SimpleNamespace(bbox=np.array([10., 10., 110., 110.]), det_score=.99,
                                 normed_embedding=vector(), kps=np.zeros((5, 2)))
    engine = FaceEngine.__new__(FaceEngine)
    engine.app = SimpleNamespace(get=lambda image, max_num=0: [model_face])
    with patch("insightface.utils.face_align.norm_crop", side_effect=ValueError("bad landmarks")):
        observed = engine.detect(FRAME)
    assert len(observed) == 1 and observed[0].aligned is None


def photo_bytes():
    ok, data = cv2.imencode(".png", CHECKER)
    assert ok
    return data.tobytes()


def prepared(observation):
    faces = Faces()
    faces.observations = [observation]
    return employee_service.prepare_face(photo_bytes(), faces, "photo.png")


def third_face():
    embedding = np.zeros(512, dtype=np.float32)
    embedding[2] = 1.
    return face(embedding)


def create_identity(db):
    data = employee_service.EmployeeCreate(first_name="Person", last_name="Third", request_id=uuid4())
    digest = hashlib.sha256(str(data.request_id).encode()).hexdigest()
    return employee_service.create_employee_with_faces(db, data, [prepared(third_face())], digest)


def replace_sample(db, store, sample):
    employee_id, omitted_id = sample.employee_id, sample.id
    keep = [row.id for row in db.query(EmployeeFace).filter_by(employee_id=employee_id) if row.id != omitted_id]
    additions = [] if keep else [prepared(third_face())]
    employee_service.replace_employee_faces(db, employee_id, db.get(Employee, employee_id).revision, keep, additions)
    store.refresh(db)


def enroll(db, store, observation, employee_id=1):
    result = prepared(observation)
    if isinstance(result, employee_service.FaceEnrollResult):
        return result
    keep = [row.id for row in db.query(EmployeeFace).filter_by(employee_id=employee_id)]
    try:
        employee_service.replace_employee_faces(db, employee_id, db.get(Employee, employee_id).revision, keep, [result])
    except HTTPException as exc:
        if exc.status_code != 422:
            raise
        return employee_service.FaceEnrollResult(filename="photo.png", success=False,
                                                error=str(exc.detail))
    store.refresh(db)
    added = db.query(EmployeeFace).filter_by(employee_id=employee_id).order_by(EmployeeFace.id.desc()).first()
    return employee_service.FaceEnrollResult(filename="photo.png", success=True, face_id=added.id)


@test
def test_enrollment_rejects_bad_alignment_detection_blur_size_and_nonfinite_embeddings():
    bad = [face(aligned=None), face(det_score=.79),
           face(aligned=np.zeros((112, 112, 3), dtype=np.uint8)),
           face(bbox=np.array([0., 0., 79., 100.])),
           face(embedding=np.full(512, np.nan, dtype=np.float32)),
           face(embedding=np.zeros(512, dtype=np.float32))]
    for observation in bad:
        with database() as engine:
            store = load_store(engine)
            with Session(engine) as db:
                result = enroll(db, store, observation)
                assert not result.success and result.error
                assert db.query(EmployeeFace).count() == 2
                assert db.get(AttendanceRuntime, 1).gallery_revision == 7


@test
def test_enrollment_rejects_another_active_employee_but_allows_own_additional_photo():
    with database() as engine:
        store = load_store(engine)
        with Session(engine) as db:
            rejected = enroll(db, store, face(), employee_id=2)
            assert not rejected.success and db.query(EmployeeFace).count() == 2
            assert db.get(AttendanceRuntime, 1).gallery_revision == 7
            accepted = enroll(db, store, face(), employee_id=1)
            assert accepted.success and accepted.face_id is not None
            assert db.query(EmployeeFace).count() == 3
            assert db.get(AttendanceRuntime, 1).gallery_revision == 8
            assert store.match(vector()).store_revision == 8


@test
def test_every_employee_and_face_mutation_advances_durable_gallery_revision():
    with database() as engine:
        store = load_store(engine)
        with Session(engine) as db:
            added = create_identity(db)
            assert db.get(AttendanceRuntime, 1).gallery_revision == 8
            added = employee_service.update_employee(db, added,
                                                     employee_service.EmployeeUpdate(expected_revision=added.revision,
                                                                                     last_name="Renamed", first_name="Person"))
            db.expire_all()
            assert db.get(AttendanceRuntime, 1).gallery_revision == 9
            result = enroll(db, store, face(), employee_id=1)
            assert result.success
            db.expire_all()
            assert db.get(AttendanceRuntime, 1).gallery_revision == 10
            replace_sample(db, store, db.get(EmployeeFace, result.face_id))
            db.expire_all()
            assert db.get(AttendanceRuntime, 1).gallery_revision == 11
            employee_service.delete_employee(db, added)
            db.expire_all()
            assert db.get(AttendanceRuntime, 1).gallery_revision == 12
            store.refresh(db)
            assert store.match(vector()).store_revision == 12


def assert_revision_failure_rolls_back(operation):
    with database() as engine:
        store = load_store(engine)

        def reject_revision(connection, cursor, statement, parameters, context, executemany):
            if statement.lstrip().upper().startswith("UPDATE ATTENDANCE_RUNTIME"):
                raise OperationalError(statement, parameters, RuntimeError("injected revision failure"))

        with Session(engine) as db:
            event.listen(engine, "before_cursor_execute", reject_revision)
            failed = False
            try:
                operation(db, store)
            except (SQLAlchemyError, HTTPException) as exc:
                if isinstance(exc, HTTPException):
                    assert exc.status_code == 503
                failed = True
            finally:
                event.remove(engine, "before_cursor_execute", reject_revision)
                db.rollback()
            assert failed, "Mutation must attempt its gallery revision in the same transaction"
        with Session(engine) as db:
            assert db.get(AttendanceRuntime, 1).gallery_revision == 7
            assert [(row.id, row.name, row.active) for row in db.query(Employee).order_by(Employee.id)] == [
                (1, "First", True), (2, "Second", True)]
            assert db.query(EmployeeFace).count() == 2
        assert store.match(vector()).store_revision == 7


@test
def test_employee_create_is_rolled_back_when_gallery_revision_write_fails():
    assert_revision_failure_rolls_back(lambda db, store: create_identity(db))


@test
def test_employee_update_is_rolled_back_when_gallery_revision_write_fails():
    assert_revision_failure_rolls_back(lambda db, store: employee_service.update_employee(
        db, db.get(Employee, 1), employee_service.EmployeeUpdate(expected_revision=0, active=False)))


@test
def test_employee_delete_is_rolled_back_when_gallery_revision_write_fails():
    assert_revision_failure_rolls_back(lambda db, store: employee_service.delete_employee(
        db, db.get(Employee, 1)))


@test
def test_face_enrollment_is_rolled_back_when_gallery_revision_write_fails():
    assert_revision_failure_rolls_back(lambda db, store: enroll(db, store, face()))


@test
def test_face_replacement_is_rolled_back_when_gallery_revision_write_fails():
    assert_revision_failure_rolls_back(lambda db, store: replace_sample(
        db, store, db.query(EmployeeFace).filter_by(employee_id=1).one()))


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
