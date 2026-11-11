"""Directed doorway geometry, evidence lifecycle and saved calibration contracts.

Run: .venv/Scripts/python.exe tests/test_attendance_crossings.py
The detector, immutable frame/face records, HTTP router and temporary SQLite
transactions are real. Only model observations, the receipt clock and camera I/O
are supplied deterministically; no production lifespan, DB or camera is opened.
"""
import asyncio
import base64
import json
import sys
import threading
import tempfile
import traceback
from contextlib import contextmanager
from dataclasses import FrozenInstanceError, replace
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from urllib.parse import urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
from fastapi import FastAPI
from sqlalchemy import create_engine, event
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from app.ai.doorway import DoorwayCalibration, DoorwayDetector
from app.ai.face_quality import FaceQuality
from app.ai.recognition_service import DetectionResult, FaceEvidence, FrameOutcome
from app.ai.tracker import TrackLifecycle
from app.config import settings
from app.database.attendance_schema import initialize_attendance_schema
from app.database.database import get_db
from app.database.models import AttendanceCamera, AttendanceCommission, AttendanceRuntime
from app.rtsp.manager import ReaderManager
from app.rtsp.reader import CameraStatus, FramePacket, RTSPReader
from app.services import camera_service

TESTS = []
STAMP = datetime(2026, 11, 11, 4, 0, 0, 123456)
MONO = 1000.
AUTO_FACE = object()
FRAME = np.zeros((480, 640, 3), dtype=np.uint8)
FRAME[:, :, 0] = np.arange(640, dtype=np.uint16)[None, :] % 256
FRAME[:, :, 1] = np.arange(480, dtype=np.uint16)[:, None] % 256
FRAME.setflags(write=False)
FACE_JPEG = cv2.imencode(".jpg", np.full((112, 112, 3), 120, np.uint8))[1].tobytes()


def test(fn):
    TESTS.append(fn)
    return fn


def calibration(**changes):
    values = dict(camera_id=1, role="ENTRY", revision=3, source_revision=2,
                  line_start=(.2, .5), line_end=(.8, .5), inside_point=(.5, .8),
                  reference_width=640, reference_height=480)
    values.update(changes)
    return DoorwayCalibration(**values)


def face(seq, seconds, *, employee=1, revision=7, confirmed=AUTO_FACE,
         reasons=(), confirmed_at=None, samples=None):
    """A complete immutable recognition observation, not a detector substitute."""
    if confirmed is AUTO_FACE:
        confirmed = employee if seq >= 3 else None
    sequences = tuple(range(max(1, seq - 2), seq + 1))
    offsets = tuple(seconds - .25 * (seq - item) for item in sequences)
    if samples is not None:
        sequences, offsets = samples
    return FaceEvidence(
        frame_seq=seq, received_at=STAMP + timedelta(seconds=seconds),
        received_mono=MONO + seconds, candidate_employee_id=employee,
        candidate_employee_name=f"Employee {employee}" if employee else None,
        score=.91 if employee else 0., runner_up_employee_id=2 if employee == 1 else 1,
        runner_up_employee_name="Other", runner_up_score=.2, margin=.71,
        quality=FaceQuality(100., .99, 90., True), store_revision=revision,
        confirmed_employee_id=confirmed,
        confirmed_employee_name=f"Employee {confirmed}" if confirmed else None,
        supporting_frame_sequences=sequences if confirmed else (),
        supporting_received_monos=tuple(MONO + value for value in offsets) if confirmed else (),
        supporting_received_ats=tuple(STAMP + timedelta(seconds=value) for value in offsets)
        if confirmed else (),
        confirmed_at_mono=(MONO + seconds if confirmed_at is None else MONO + confirmed_at)
        if confirmed else None,
        face_bbox=(.4, .05, .6, .25), aligned_jpeg=FACE_JPEG, reason_codes=tuple(reasons),
    )


def person(y, *, x=.5, track=1, observation=None, uncertain=False):
    return DetectionResult(
        track_id=track,
        employee_id=observation.confirmed_employee_id if observation and not observation.reason_codes else None,
        employee_name=observation.confirmed_employee_name if observation and not observation.reason_codes else None,
        confidence=observation.score if observation else None,
        bbox=(x - .09, max(0., y - .35), x + .09, y), state="known" if observation else "unknown",
        association_uncertain=uncertain, person_score=.98,
        fresh_face_evidence=observation,
    )


class Trajectory:
    def __init__(self, **config):
        self.detector = DoorwayDetector()
        self.calibration = calibration(**config)
        self.seq = 0
        self.seconds = -.25
        self.events = []

    def send(self, y=None, *, seconds=None, x=.5, track=1, observation=AUTO_FACE,
             lifecycle=None, uncertain=False, stream="stream-a", generation="generation-a",
             revision=None, seq=None, frame=FRAME, age=0., detections=None):
        self.seq = self.seq + 1 if seq is None else seq
        self.seconds = self.seconds + .25 if seconds is None else seconds
        if observation is AUTO_FACE:
            observation = face(self.seq, self.seconds)
        if detections is None:
            detections = [] if y is None else [person(y, x=x, track=track,
                                                    observation=observation, uncertain=uncertain)]
        if lifecycle is None:
            lifecycle = [TrackLifecycle(item.track_id, "AMBIGUOUS" if item.association_uncertain else "TRACKED",
                                        MONO + self.seconds,
                                        "ASSOCIATION_UNCERTAIN" if item.association_uncertain else None)
                         for item in detections]
        packet = FramePacket(stream_id=stream,
                             source_revision=self.calibration.source_revision if revision is None else revision,
                             seq=self.seq, received_at=STAMP + timedelta(seconds=self.seconds),
                             received_mono=MONO + self.seconds, frame=frame)
        outcome = FrameOutcome(detections=detections, tracking_generation=generation,
                               track_lifecycle=lifecycle,
                               person_regions=[item.bbox for item in detections])
        with patch("app.ai.doorway.time.monotonic", return_value=MONO + self.seconds + age):
            result = self.detector.process(packet, 1, outcome, self.calibration)
        self.events.extend(result)
        return result

    def walk(self, ys, **kwargs):
        for y in ys:
            self.send(y, **kwargs)
        return self.events

    def finish(self, seconds=None):
        seconds = self.seconds + 2.01 if seconds is None else seconds
        result = self.detector.expire(MONO + seconds)
        self.events.extend(result)
        return self.events


def single_crossing(path, *, direction="ENTRY", eligible=True):
    events = path.finish()
    assert len(events) == 1, events
    crossing = events[0]
    assert crossing.direction == direction
    assert crossing.auto_eligible is eligible
    if eligible:
        assert crossing.confirmed_employee_id == 1
        assert crossing.disqualifying_reasons == ()
    else:
        assert crossing.disqualifying_reasons
    return crossing


@test
def test_entry_uses_footpoint_finite_intersection_and_first_receipt_interpolation():
    path = Trajectory()
    path.walk([.25, .3, .45, .55, .7])
    crossing = single_crossing(path)
    assert crossing.track_key == (1, "stream-a", "generation-a", 1)
    assert crossing.crossing_index == 1
    assert crossing.source_revision == 2 and crossing.calibration_revision == 3
    assert crossing.gallery_revision == 7
    assert crossing.occurred_at == datetime(2026, 11, 11, 4, 0, 0, 748456)
    assert crossing.occurred_at.tzinfo is None
    assert abs(crossing.occurred_mono - 1000.625) < 1e-9
    assert 3 in crossing.frame_sequences and 4 in crossing.frame_sequences
    try:
        crossing.auto_eligible = False
    except FrozenInstanceError:
        pass
    else:
        raise AssertionError("Queued crossing eligibility must be immutable")


@test
def test_exit_camera_emits_inside_to_outside_only():
    path = Trajectory(role="EXIT")
    path.walk([.75, .7, .55, .45, .3])
    crossing = single_crossing(path, direction="EXIT")
    assert abs(crossing.occurred_mono - 1000.625) < 1e-9


@test
def test_reversing_gate_endpoints_does_not_reverse_marked_interior():
    path = Trajectory(line_start=(.8, .5), line_end=(.2, .5))
    path.walk([.25, .3, .45, .55, .7])
    single_crossing(path)


@test
def test_exact_minimum_calibration_distances_remain_usable_after_float_arithmetic():
    path = Trajectory(line_start=(.2, .5), line_end=(.25, .5), inside_point=(.225, .55))
    path.walk([.25, .3, .45, .55, .7], x=.225)
    single_crossing(path)


@test
def test_vertical_and_diagonal_gates_use_oriented_perpendicular_distance():
    vertical = Trajectory(line_start=(.5, .2), line_end=(.5, .8), inside_point=(.8, .5))
    for x in [.25, .3, .45, .55, .7]:
        vertical.send(.6, x=x)
    single_crossing(vertical)
    diagonal = Trajectory(line_start=(.2, .2), line_end=(.8, .8), inside_point=(.2, .8))
    diagonal.walk([.2, .25, .4, .6, .75])
    crossing = single_crossing(diagonal)
    assert abs(crossing.occurred_mono - 1000.625) < 1e-9


@test
def test_opposite_camera_direction_never_creates_a_duplicate_attendance_event():
    for role, points in (("ENTRY", [.75, .7, .55, .45, .3]),
                         ("EXIT", [.25, .3, .45, .55, .7])):
        path = Trajectory(role=role)
        path.walk(points)
        assert path.finish() == [], role


@test
def test_jitter_retreat_spawn_on_line_and_endpoint_miss_produce_no_crossing():
    cases = [
        ([.49, .51, .495, .505, .49, .51], .5),
        ([.25, .3, .47, .49, .46, .3, .25], .5),
        ([.5, .5, .53, .65, .7], .5),
        ([.25, .3, .45, .55, .7], .95),
        # A single observation outside the neutral band is not a stable side.
        ([.25, .49, .55, .7, .75], .5),
    ]
    for points, x in cases:
        path = Trajectory()
        path.walk(points, x=x)
        assert path.finish() == [], (points, x)


@test
def test_neutral_hysteresis_preserves_first_bracketing_time_until_stable_opposite_side():
    path = Trajectory()
    path.walk([.25, .3, .49, .51, .495, .515, .55, .65])
    crossing = single_crossing(path)
    assert abs(crossing.occurred_mono - 1000.625) < 1e-9


@test
def test_gate_intersection_is_not_inferred_from_distant_endpoints():
    path = Trajectory()
    path.send(.25, x=.95)
    path.send(.3, x=.95)
    path.send(.45, x=.95)
    path.send(.55, x=.95)
    path.send(.7, x=.95)
    # Later walking along the interior cannot retroactively cross the finite gate.
    path.send(.75, x=.5)
    path.send(.8, x=.5)
    assert path.finish() == []


@test
def test_one_continuous_track_can_emit_later_trips_with_distinct_ordinals():
    path = Trajectory()
    path.walk([.25, .3, .45, .55, .7, .75, .55, .45, .3, .25,
               .3, .45, .55, .7])
    events = path.finish()
    assert len(events) == 2
    assert [item.direction for item in events] == ["ENTRY", "ENTRY"]
    assert [item.crossing_index for item in events] == [1, 2]
    assert events[0].track_key == events[1].track_key
    assert all(item.auto_eligible for item in events)
    assert events[0].occurred_at < events[1].occurred_at


@test
def test_pending_crossing_waits_for_same_track_late_face_confirmation():
    path = Trajectory()
    path.walk([.25, .3, .45, .55, .7], observation=None)
    assert path.events == [], "Unknown crossing must retain its two-second evidence window"
    path.send(.75, observation=face(6, 1.25))
    crossing = single_crossing(path)
    assert crossing.occurred_at == datetime(2026, 11, 11, 4, 0, 0, 748456)


@test
def test_pending_window_expires_without_another_camera_frame():
    path = Trajectory()
    path.walk([.25, .3, .45, .55, .7], observation=None)
    assert path.detector.expire(1002.624) == []
    path.events.extend(path.detector.expire(1002.626))
    assert len(path.events) == 1
    crossing = path.events[0]
    assert crossing.direction == "ENTRY" and not crossing.auto_eligible
    assert "NO_FACE" in crossing.disqualifying_reasons
    assert path.detector.expire(1100.) == [], "Expired pending evidence must emit exactly once"


@test
def test_late_face_after_plus_one_point_five_seconds_cannot_qualify():
    path = Trajectory()
    path.walk([.25, .3, .45, .55, .7], observation=None)
    path.send(.75, seconds=1.5, observation=None)
    path.send(.8, seconds=2.13, observation=face(7, 2.13))
    single_crossing(path, eligible=False)


@test
def test_expired_pre_crossing_confirmation_is_not_resurrected_by_wide_sample_window():
    path = Trajectory()
    old = face(3, .5, samples=((1, 2, 3), (0., .25, .5)))
    path.send(.25, seconds=0., observation=None)
    path.send(.25, seconds=.25, observation=None)
    path.send(.25, seconds=.5, observation=old)
    path.send(.3, seconds=.75, observation=None)
    path.send(.45, seconds=1.25, observation=None)
    path.send(.55, seconds=1.5, observation=None)
    path.send(.7, seconds=1.75, observation=None)
    # All samples are within1.5s of crossing1.375, but the confirmation expired at1.25.
    single_crossing(path, eligible=False)


@test
def test_new_confirmation_requires_three_distinct_supporting_observations():
    for sequences, offsets in (((4, 4, 5), (.75, .75, 1.)), ((4, 5), (.75, 1.))):
        path = Trajectory()
        path.walk([.25, .3, .45, .55], observation=None)
        path.send(.7, observation=face(5, 1., samples=(sequences, offsets)))
        single_crossing(path, eligible=False)


@test
def test_competing_confirmed_employee_invalidates_this_crossing():
    path = Trajectory()
    path.send(.25, observation=None)
    path.send(.3, observation=None)
    path.send(.45, observation=face(3, .5, employee=1))
    path.send(.55, observation=face(4, .75, employee=2))
    path.send(.7, observation=face(5, 1., employee=2))
    crossing = single_crossing(path, eligible=False)
    assert "AMBIGUOUS_FACE" in crossing.disqualifying_reasons


@test
def test_face_reason_codes_override_retained_raw_confirmed_identity():
    for reason in ("GALLERY_CHANGED", "ASSOCIATION_UNCERTAIN", "BAD_ALIGNMENT",
                   "LOW_FACE_SCORE", "AMBIGUOUS_FACE"):
        path = Trajectory()
        for index, y in enumerate([.25, .3, .45, .55, .7], 1):
            path.send(y, observation=face(index, (index - 1) * .25, reasons=(reason,)))
        crossing = single_crossing(path, eligible=False)
        assert reason in crossing.disqualifying_reasons


@test
def test_gallery_revision_change_during_pending_window_cannot_accept_stale_identity():
    path = Trajectory()
    path.walk([.25, .3, .45, .55, .7], observation=None)
    path.send(.75, observation=face(6, 1.25, revision=7, confirmed=None))
    path.send(.8, observation=face(7, 1.5, revision=8))
    crossing = single_crossing(path, eligible=False)
    assert "GALLERY_CHANGED" in crossing.disqualifying_reasons


@test
def test_other_tracks_face_confirmation_never_resolves_an_unknown_crossing():
    path = Trajectory()
    path.walk([.25, .3, .45, .55, .7], observation=None)
    path.send(.75, track=2, observation=face(6, 1.25))
    crossing = single_crossing(path, eligible=False)
    assert crossing.track_key[-1] == 1


@test
def test_ambiguous_lifecycle_is_authoritative_even_without_a_detection_flag():
    path = Trajectory()
    path.send(.25, lifecycle=[TrackLifecycle(1, "AMBIGUOUS", MONO, "ASSOCIATION_UNCERTAIN")])
    path.walk([.3, .49, .55, .7])
    assert path.finish() == [], "An ambiguous initial point cannot arm a trusted trajectory"


@test
def test_unmatched_ambiguous_lifecycle_invalidates_pending_crossing_immediately():
    path = Trajectory()
    path.walk([.25, .3, .45, .55, .7], observation=None)
    path.send(None, lifecycle=[TrackLifecycle(1, "AMBIGUOUS", 1001., "ASSOCIATION_UNCERTAIN")])
    path.send(.75, observation=face(7, 1.5))
    crossing = single_crossing(path, eligible=False)
    assert "ASSOCIATION_UNCERTAIN" in crossing.disqualifying_reasons


@test
def test_overlapping_people_cannot_carry_one_identity_into_another_trajectory():
    path = Trajectory()
    path.send(.25)
    path.send(.3)
    path.send(.45, observation=None)
    path.send(.55, uncertain=True, observation=face(4, .75,
              reasons=("ASSOCIATION_UNCERTAIN",)))
    path.send(.7, observation=face(5, 1.))
    crossing = single_crossing(path, direction=None, eligible=False)
    assert "ASSOCIATION_UNCERTAIN" in crossing.disqualifying_reasons


@test
def test_neutral_loss_emits_one_unknown_direction_review_but_away_loss_emits_none():
    for points, expected in (([.25, .3, .49], 1), ([.75, .7, .51], 1), ([.25, .3], 0)):
        path = Trajectory()
        path.walk(points, observation=None)
        path.send(None, lifecycle=[TrackLifecycle(1, "LOST", MONO + path.seconds, "TRACK_LOST")])
        path.send(None, lifecycle=[TrackLifecycle(1, "EXPIRED", MONO + path.seconds, "TRACK_EXPIRED")])
        events = path.finish()
        assert len(events) == expected, points
        if expected:
            assert events[0].direction is None and not events[0].auto_eligible
            assert "INCOMPLETE_CROSSING" in events[0].disqualifying_reasons


@test
def test_loss_in_neutral_extension_beyond_finite_gate_emits_no_partial():
    path = Trajectory()
    path.walk([.25, .3, .49], x=.95, observation=None)
    path.send(None, lifecycle=[TrackLifecycle(1, "EXPIRED", 1000.5, "TRACK_EXPIRED")])
    assert path.finish() == []


@test
def test_lost_grace_never_interpolates_across_an_unobserved_gate_segment():
    path = Trajectory()
    path.walk([.25, .3, .49], observation=None)
    path.send(None, lifecycle=[TrackLifecycle(1, "LOST", 1000.5, "TRACK_LOST")])
    path.send(.55, observation=face(5, 1.))
    path.send(.7, observation=face(6, 1.25))
    events = path.finish()
    assert len(events) == 1
    assert events[0].direction is None and not events[0].auto_eligible
    assert "INCOMPLETE_CROSSING" in events[0].disqualifying_reasons


@test
def test_completed_unknown_crossing_survives_temporary_lost_until_window_deadline():
    path = Trajectory()
    path.walk([.25, .3, .45, .55, .7], observation=None)
    result = path.send(None, lifecycle=[TrackLifecycle(1, "LOST", 1001., "TRACK_LOST")])
    assert result == [], "LOST is not irreversible EXPIRED"
    crossing = single_crossing(path, eligible=False)
    assert crossing.direction == "ENTRY"


def lost_pending_with_late_confirmation(*, idle_expiry):
    path = Trajectory()
    for y, seconds in ((.25, 0.), (.3, .25), (.45, .5), (.55, .75), (.7, .8)):
        path.send(y, seconds=seconds, observation=None)
    assert path.events == []
    path.send(None, seconds=.9, observation=None,
              lifecycle=[TrackLifecycle(1, "LOST", MONO + .8, "TRACK_LOST")])
    if idle_expiry:
        path.events.extend(path.detector.expire(MONO + 1.551))
    # New supporting observations are real distinct frames after reacquisition.
    # Their final confirmation is still within +1.5s of crossing at .625.
    path.send(.75, seconds=1.56, observation=face(7, 1.56, confirmed=None))
    path.send(.78, seconds=1.81, observation=face(8, 1.81, confirmed=None))
    path.send(.8, seconds=2.06, observation=face(
        9, 2.06, samples=((7, 8, 9), (1.56, 1.81, 2.06))))
    return path


@test
def test_completed_unknown_lost_gap_cannot_gain_late_face_eligibility():
    crossing = single_crossing(lost_pending_with_late_confirmation(idle_expiry=False), eligible=False)
    assert "FRAME_GAP" in crossing.disqualifying_reasons


@test
def test_idle_expiry_latches_pending_gap_after_lost_clears_geometric_predecessor():
    crossing = single_crossing(lost_pending_with_late_confirmation(idle_expiry=True), eligible=False)
    assert "FRAME_GAP" in crossing.disqualifying_reasons


@test
def test_expired_or_disabled_track_flushes_known_completed_direction_without_guessing_face():
    for reason in ("EXPIRED", "DISABLED"):
        path = Trajectory()
        path.walk([.25, .3, .45, .55, .7], observation=None)
        if reason == "EXPIRED":
            result = path.send(None, lifecycle=[TrackLifecycle(1, "EXPIRED", 1001., "TRACK_EXPIRED")])
        else:
            result = path.detector.reset_camera(1, "DISABLED")
            path.events.extend(result)
        assert len(result) == 1
        crossing = single_crossing(path, eligible=False)
        assert crossing.direction == "ENTRY"


@test
def test_irreversible_reset_does_not_revoke_already_eligible_completed_crossing():
    path = Trajectory()
    path.walk([.25, .3, .45, .55, .7])
    path.events.extend(path.detector.reset_camera(1, "STREAM_RESET"))
    single_crossing(path)


@test
def test_pending_completed_crossing_cannot_gain_eligibility_across_a_frame_gap():
    path = Trajectory()
    path.walk([.25, .3, .45, .55, .7], observation=None)
    path.send(.75, seconds=1.751, observation=face(6, 1.751))
    crossing = single_crossing(path, eligible=False)
    assert "FRAME_GAP" in crossing.disqualifying_reasons


@test
def test_disable_or_shutdown_flushes_identified_neutral_partial_as_unknown_direction():
    for reason in ("DISABLED", "SHUTDOWN"):
        path = Trajectory()
        path.walk([.25, .3, .49])
        path.events.extend(path.detector.reset_camera(1, reason))
        crossing = single_crossing(path, direction=None, eligible=False)
        assert "INCOMPLETE_CROSSING" in crossing.disqualifying_reasons
        assert path.detector.reset_camera(1, reason) == []


@test
def test_frame_gap_does_not_interpolate_or_bridge_a_partial_crossing():
    path = Trajectory()
    path.walk([.25, .3, .49], observation=None)
    path.send(.55, seconds=1.251, observation=face(4, 1.251))
    path.send(.7, seconds=1.501, observation=face(5, 1.501))
    crossing = single_crossing(path, direction=None, eligible=False)
    assert "INCOMPLETE_CROSSING" in crossing.disqualifying_reasons
    assert "FRAME_GAP" in crossing.disqualifying_reasons


@test
def test_frame_free_expiry_flushes_neutral_partial_crossing_after_continuity_limit():
    path = Trajectory()
    path.walk([.25, .3, .49], observation=None)
    events = path.finish(seconds=1.251)
    assert len(events) == 1
    assert events[0].direction is None and not events[0].auto_eligible
    assert "FRAME_GAP" in events[0].disqualifying_reasons


@test
def test_sequence_replay_cannot_supply_second_stable_side_or_duplicate_event():
    path = Trajectory()
    path.send(.25, seq=1)
    path.send(.3, seq=2)
    path.send(.45, seq=3)
    path.send(.55, seq=4)
    path.send(.7, seq=4)
    assert path.events == []
    path.send(.7, seq=5)
    path.send(.75, seq=5)
    single_crossing(path)


@test
def test_stream_generation_source_and_calibration_changes_flush_old_partial_evidence():
    changes = [dict(stream="stream-b", seq=1), dict(generation="generation-b"),
               dict(source_revision=3), dict(calibration_revision=4)]
    for change in changes:
        path = Trajectory()
        path.walk([.25, .3, .49], observation=None)
        kwargs = dict(change)
        if "source_revision" in kwargs:
            path.calibration = replace(path.calibration, source_revision=kwargs.pop("source_revision"))
        if "calibration_revision" in kwargs:
            path.calibration = replace(path.calibration, revision=kwargs.pop("calibration_revision"))
        path.send(.55, **kwargs)
        path.send(.7, **({key: value for key, value in kwargs.items() if key != "seq"}))
        events = path.finish()
        assert len(events) == 1, change
        assert events[0].track_key == (1, "stream-a", "generation-a", 1)
        assert events[0].direction is None and not events[0].auto_eligible
        assert "INCOMPLETE_CROSSING" in events[0].disqualifying_reasons


@test
def test_same_stream_new_generation_has_distinct_crossing_source_identity():
    path = Trajectory()
    path.walk([.25, .3, .45, .55, .7])
    path.events.extend(path.detector.reset_camera(1, "CALIBRATION_CHANGED"))
    for y in [.25, .3, .45, .55, .7]:
        path.send(y, generation="generation-b")
    events = path.finish()
    assert len(events) == 2
    assert events[0].track_key != events[1].track_key
    assert events[0].crossing_index == events[1].crossing_index == 1
    assert all(item.auto_eligible for item in events)


@test
def test_stale_packets_and_wrong_source_packets_cannot_arm_a_crossing():
    for kwargs in (dict(age=settings.attendance_frame_max_age + .001), dict(revision=1)):
        path = Trajectory()
        path.walk([.25, .3, .45, .55, .7], **kwargs)
        assert path.finish() == [], kwargs


@test
def test_same_aspect_scaling_preserves_eligibility_but_aspect_change_blocks_it():
    scaled = np.zeros((960, 1280, 3), np.uint8)
    path = Trajectory()
    path.walk([.25, .3, .45, .55, .7], frame=scaled)
    single_crossing(path)
    changed = np.zeros((480, 800, 3), np.uint8)
    path = Trajectory()
    path.walk([.25, .3, .45, .55, .7], frame=changed)
    single_crossing(path, eligible=False)


@test
def test_before_after_evidence_is_bounded_and_keeps_crop_coordinate_mapping():
    path = Trajectory()
    path.walk([.25, .3, .45, .55, .7])
    crossing = single_crossing(path)
    for kind in ("before", "after"):
        jpeg = getattr(crossing, f"{kind}_jpeg")
        crop = getattr(crossing, f"{kind}_crop")
        assert jpeg and len(jpeg) <= 96 * 1024
        image = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        assert max(image.shape[:2]) <= 640
        assert len(crop) == 4 and 0 <= crop[0] < crop[2] <= 1
        assert 0 <= crop[1] < crop[3] <= 1
        # The saved rectangle maps normalized gate coordinates into this crop,
        # not into the full source image or the review panel's letterboxing.
        assert crop[2] - crop[0] < 1 or crop[3] - crop[1] < 1
        assert abs(image.shape[1] / image.shape[0]
                   - (crop[2] - crop[0]) * 640 / ((crop[3] - crop[1]) * 480)) < .06
    assert crossing.face_jpeg and len(crossing.face_jpeg) <= 32 * 1024
    assert cv2.imdecode(np.frombuffer(crossing.face_jpeg, np.uint8), cv2.IMREAD_COLOR).shape[:2] == (112, 112)
    assert crossing.calibration.line_start == (.2, .5)
    assert crossing.calibration.line_end == (.8, .5)
    assert crossing.calibration.inside_point == (.5, .8)
    assert not FRAME.flags.writeable, "Evidence rendering must not mutate a published frame"


@test
def test_disqualified_face_retains_original_candidate_and_quality_as_raw_evidence():
    path = Trajectory()
    for index, y in enumerate([.25, .3, .45, .55, .7], 1):
        path.send(y, observation=face(index, (index - 1) * .25,
                                     reasons=("GALLERY_CHANGED",)))
    crossing = single_crossing(path, eligible=False)
    observations = crossing.face_evidence
    assert observations
    assert any(item.candidate_employee_id == 1 and item.candidate_employee_name == "Employee 1"
               and item.score == .91 and item.runner_up_employee_id == 2
               and item.margin == .71 and item.quality == FaceQuality(100., .99, 90., True)
               and item.store_revision == 7 and "GALLERY_CHANGED" in item.reason_codes
               for item in observations)


@test
def test_missing_contextual_jpeg_forces_review_even_with_identity_and_direction():
    path = Trajectory()
    with patch("app.services.attendance_evidence.cv2.imencode", return_value=(False, None)):
        path.walk([.25, .3, .45, .55, .7])
        crossing = single_crossing(path, eligible=False)
    assert "EVIDENCE_UNAVAILABLE" in crossing.disqualifying_reasons


@test
def test_calibration_rejects_degenerate_nonfinite_and_out_of_bounds_geometry():
    invalid = [
        dict(line_end=(.21, .5)), dict(inside_point=(.5, .52)),
        dict(line_start=(-.1, .5)), dict(line_end=(1.1, .5)),
        dict(inside_point=(.5, float("nan"))), dict(line_start=(float("inf"), .5)),
        dict(reference_width=0), dict(reference_height=-1), dict(role="BOTH"),
    ]
    for fields in invalid:
        try:
            calibration(**fields)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Invalid calibration must not be accepted: {fields}")


@contextmanager
def calibration_app():
    # Import the actual router without importing main or starting any model.
    from app.api import attendance as attendance_api
    from app.api import cameras as cameras_api

    with tempfile.TemporaryDirectory() as directory:
        engine = create_engine(f"sqlite:///{(Path(directory) / 'calibration.db').as_posix()}",
                               connect_args={"check_same_thread": False, "timeout": 30})

        @event.listens_for(engine, "connect")
        def foreign_keys(connection, _record):
            connection.execute("PRAGMA foreign_keys=ON")

        initialize_attendance_schema(engine)
        with Session(engine) as db:
            for ident in (1, 2, 3):
                camera_service.create_camera(db, camera_service.CameraCreate(
                    name=f"Door {ident}", nvr_host="unused", username="u", password="p",
                    channel=ident, stream_type="main"))
        app = FastAPI()
        app.include_router(attendance_api.router)
        app.include_router(cameras_api.router)
        manager = SnapshotManager(engine)
        app.state.manager = manager
        app.state.detection_state = None
        app.state.ai_worker = None
        app.state.attendance_health = None

        def isolated_db():
            with Session(engine) as db:
                yield db

        app.dependency_overrides[get_db] = isolated_db
        try:
            yield SimpleNamespace(app=app, engine=engine, manager=manager,
                                  api=attendance_api)
        finally:
            engine.dispose()


class SnapshotManager:
    """No network; use the actual reader buffer and record reader publish boundary."""
    def __init__(self, engine):
        self.engine = engine
        self.readers = {}
        self.published = []

    def get(self, camera_id):
        return self.readers.get(camera_id)

    def status(self, camera_id):
        reader = self.get(camera_id)
        return {"status": reader.status().value if reader else "OFFLINE"}

    def sync(self, camera, changed_fields=None):
        with Session(self.engine) as db:
            config = db.get(AttendanceCamera, camera.id)
            self.published.append((camera.source_revision, config.source_revision, config.revision,
                                   config.enabled, db.get(AttendanceRuntime, 1).commission_id))

    def snapshot(self, camera_id=1, *, revision=1, stream="snapshot-stream", seq=9):
        reader = RTSPReader(camera_id, "Door", "rtsp://unused", source_revision=revision,
                            ffmpeg_path="unused", ffprobe_path="unused")
        reader._set_status(CameraStatus.ONLINE)
        reader.buffer._seq = seq - 1
        reader.buffer.put(FRAME, stream_id=stream, source_revision=revision,
                          received_at=STAMP, received_mono=MONO)
        self.readers[camera_id] = reader
        return reader


async def asgi_request(app, method, path, payload=None):
    body = b"" if payload is None else json.dumps(payload).encode("utf-8")
    messages = []
    delivered = False

    async def receive():
        nonlocal delivered
        if delivered:
            return {"type": "http.disconnect"}
        delivered = True
        return {"type": "http.request", "body": body, "more_body": False}

    async def send(message):
        messages.append(message)

    await app({"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
               "method": method, "scheme": "http", "path": path, "raw_path": path.encode(),
               "query_string": b"", "headers": [(b"content-type", b"application/json"),
                                                (b"x-operator-request", b"1")],
               "client": ("127.0.0.1", 12345), "server": ("test", 80), "root_path": ""},
              receive, send)
    start = next(message for message in messages if message["type"] == "http.response.start")
    content = b"".join(message.get("body", b"") for message in messages
                       if message["type"] == "http.response.body")
    return start["status"], json.loads(content) if content else None


def request(fixture, method, path, payload=None):
    return asyncio.run(asgi_request(fixture.app, method, path, payload))


def save_payload(**changes):
    payload = dict(role="ENTRY", enabled=True, line_start=[.2, .5], line_end=[.8, .5],
                   inside_point=[.5, .8], reference_width=640, reference_height=480,
                   expected_revision=0, expected_source_revision=1)
    payload.update(changes)
    return payload


def seed_commission(engine):
    with Session(engine) as db:
        row = AttendanceCommission(request_id="526dbfd5-e296-4ccc-9762-4d3e397404ab",
                                   actor="Operator", reason="Measured site trial",
                                   configuration_fingerprint={"version": 1}, acceptance_record={})
        db.add(row)
        db.flush()
        db.get(AttendanceRuntime, 1).commission_id = row.id
        db.commit()
        return row.id


@test
def test_calibration_get_returns_seeded_unassigned_row_and_missing_camera_404():
    with calibration_app() as fixture:
        status, body = request(fixture, "GET", "/api/attendance/cameras/1")
        assert status == 200, body
        assert body["camera_id"] == 1 and body["role"] is None
        assert body["enabled"] is False and body["revision"] == 0
        assert body["source_revision"] == 1 and body["line_start"] is None
        assert request(fixture, "GET", "/api/attendance/cameras/999")[0] == 404
        assert request(fixture, "PUT", "/api/attendance/cameras/999", save_payload())[0] == 404


@test
def test_calibration_put_commits_geometry_and_increments_revision():
    with calibration_app() as fixture:
        status, body = request(fixture, "PUT", "/api/attendance/cameras/1", save_payload())
        assert status == 200, body
        assert body["revision"] == 1 and body["source_revision"] == 1
        assert body["enabled"] is True and body["role"] == "ENTRY"
        with Session(fixture.engine) as db:
            saved = db.get(AttendanceCamera, 1)
            assert saved.line_start == [.2, .5] and saved.line_end == [.8, .5]
            assert saved.inside_point == [.5, .8]
            assert saved.reference_width == 640 and saved.reference_height == 480
            assert db.get(AttendanceRuntime, 1).monitoring_started_at is None


@test
def test_calibration_put_rejects_invalid_geometry_without_changing_saved_configuration():
    invalid = [dict(line_end=[.21, .5]), dict(inside_point=[.5, .52]),
               dict(line_start=[-.1, .5]), dict(line_end=[1.1, .5]),
               dict(reference_width=0), dict(reference_height=-1),
               dict(line_start=None), dict(role=None), dict(role="BOTH")]
    with calibration_app() as fixture:
        for changes in invalid:
            status, body = request(fixture, "PUT", "/api/attendance/cameras/1", save_payload(**changes))
            assert status == 422, (changes, body)
        with Session(fixture.engine) as db:
            row = db.get(AttendanceCamera, 1)
            assert row.revision == 0 and row.role is None and not row.enabled


@test
def test_calibration_rejects_stale_revision_source_and_enabled_role_conflicts():
    with calibration_app() as fixture:
        assert request(fixture, "PUT", "/api/attendance/cameras/1", save_payload())[0] == 200
        for path, payload in (
                ("/api/attendance/cameras/1", save_payload()),
                ("/api/attendance/cameras/1", save_payload(expected_revision=1, expected_source_revision=2)),
                ("/api/attendance/cameras/2", save_payload())):
            status, body = request(fixture, "PUT", path, payload)
            assert status == 409, body
        with Session(fixture.engine) as db:
            assert db.get(AttendanceCamera, 1).revision == 1
            assert db.get(AttendanceCamera, 2).revision == 0
            assert not db.get(AttendanceCamera, 2).enabled


@test
def test_assigned_role_can_change_or_disable_but_never_be_erased():
    with calibration_app() as fixture:
        assert request(fixture, "PUT", "/api/attendance/cameras/1", save_payload())[0] == 200
        status, body = request(fixture, "PUT", "/api/attendance/cameras/1",
                               save_payload(role=None, enabled=False, expected_revision=1))
        assert status in (409, 422), body
        status, body = request(fixture, "PUT", "/api/attendance/cameras/1",
                               save_payload(role="EXIT", enabled=False, expected_revision=1))
        assert status == 200, body
        with Session(fixture.engine) as db:
            row = db.get(AttendanceCamera, 1)
            assert row.role == "EXIT" and not row.enabled and row.revision == 2


@test
def test_enabling_second_role_atomically_starts_monitoring_and_revokes_commission():
    with calibration_app() as fixture:
        seed_commission(fixture.engine)
        assert request(fixture, "PUT", "/api/attendance/cameras/1", save_payload())[0] == 200
        with Session(fixture.engine) as db:
            runtime = db.get(AttendanceRuntime, 1)
            assert runtime.commission_id is None and runtime.monitoring_started_at is None
        status, body = request(fixture, "PUT", "/api/attendance/cameras/2", save_payload(role="EXIT"))
        assert status == 200, body
        with Session(fixture.engine) as db:
            started = db.get(AttendanceRuntime, 1).monitoring_started_at
            assert started is not None
        assert request(fixture, "PUT", "/api/attendance/cameras/2",
                       save_payload(role="EXIT", enabled=False, expected_revision=1))[0] == 200
        with Session(fixture.engine) as db:
            assert db.get(AttendanceRuntime, 1).monitoring_started_at == started


@test
def test_monitoring_start_failure_rolls_back_second_role_enablement():
    with calibration_app() as fixture:
        assert request(fixture, "PUT", "/api/attendance/cameras/1", save_payload())[0] == 200

        def fail_runtime_update(_conn, _cursor, statement, _parameters, _context, _many):
            if statement.lstrip().upper().startswith("UPDATE ATTENDANCE_RUNTIME"):
                raise OperationalError(statement, {}, RuntimeError("Injected disk failure"))

        event.listen(fixture.engine, "before_cursor_execute", fail_runtime_update)
        try:
            status, body = request(fixture, "PUT", "/api/attendance/cameras/2", save_payload(role="EXIT"))
        finally:
            event.remove(fixture.engine, "before_cursor_execute", fail_runtime_update)
        assert status == 503, body
        with Session(fixture.engine) as db:
            row = db.get(AttendanceCamera, 2)
            assert row.revision == 0 and row.role is None and not row.enabled
            assert db.get(AttendanceRuntime, 1).monitoring_started_at is None


@test
def test_metadata_snapshot_uses_one_fresh_packet_and_explicit_receipt_offset():
    with calibration_app() as fixture:
        fixture.manager.snapshot()
        with patch("app.api.attendance.time.monotonic", return_value=1000.1):
            status, body = request(fixture, "GET", "/api/attendance/cameras/1/snapshot")
        assert status == 200, body
        assert body["source_revision"] == 1 and body["stream_id"] == "snapshot-stream"
        assert body["seq"] == 9 and body["width"] == 640 and body["height"] == 480
        assert body["received_at"] == "2026-11-11T09:00:00.123456+05:00"
        jpeg = base64.b64decode(body["jpeg_base64"], validate=True)
        image = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        assert image.shape == FRAME.shape


@test
def test_metadata_snapshot_rejects_offline_stale_and_wrong_source_packets():
    with calibration_app() as fixture:
        path = "/api/attendance/cameras/1/snapshot"
        assert request(fixture, "GET", path)[0] == 503
        reader = fixture.manager.snapshot()
        for status in (CameraStatus.OFFLINE, CameraStatus.RECONNECTING, CameraStatus.ERROR):
            reader._set_status(status)
            with patch("app.api.attendance.time.monotonic", return_value=1000.1):
                assert request(fixture, "GET", path)[0] == 503
        reader._set_status(CameraStatus.ONLINE)
        with patch("app.api.attendance.time.monotonic", return_value=1000. + settings.attendance_frame_max_age + .001):
            assert request(fixture, "GET", path)[0] == 503
        with patch("app.api.attendance.time.monotonic", return_value=1000. + settings.frame_stall_timeout + .001):
            assert request(fixture, "GET", path)[0] == 503
        fixture.manager.snapshot(revision=2)
        with patch("app.api.attendance.time.monotonic", return_value=1000.1):
            assert request(fixture, "GET", path)[0] == 503


@test
def test_source_update_commits_invalidation_before_publishing_replacement_reader():
    with calibration_app() as fixture:
        assert request(fixture, "PUT", "/api/attendance/cameras/1", save_payload())[0] == 200
        seed_commission(fixture.engine)
        status, body = request(fixture, "PUT", "/api/cameras/1", {"nvr_host": "replacement"})
        assert status == 200, body
        assert fixture.manager.published == [(2, 2, 2, False, None)]
        with Session(fixture.engine) as db:
            camera = camera_service.get_camera(db, 1)
            config = db.get(AttendanceCamera, 1)
            assert camera.nvr_host == "replacement"
            assert config.source_revision == 2 and config.revision == 2
            assert config.role == "ENTRY" and not config.enabled
        # A frozen snapshot from source1 cannot re-enable the replacement source.
        status, body = request(fixture, "PUT", "/api/attendance/cameras/1",
                               save_payload(expected_revision=2, expected_source_revision=1))
        assert status == 409, body


@test
def test_every_connection_field_invalidates_source_but_name_and_empty_password_do_not():
    changes = [("nvr_host", "replacement"), ("rtsp_port", 8554), ("username", "other"),
               ("password", "new-secret"), ("channel", 9), ("stream_type", "sub")]
    for field, value in changes:
        with calibration_app() as fixture:
            assert request(fixture, "PUT", "/api/attendance/cameras/1", save_payload())[0] == 200
            seed_commission(fixture.engine)
            assert request(fixture, "PUT", "/api/cameras/1", {field: value})[0] == 200
            with Session(fixture.engine) as db:
                config = db.get(AttendanceCamera, 1)
                assert (config.source_revision, config.revision, config.enabled) == (2, 2, False), field
                assert db.get(AttendanceRuntime, 1).commission_id is None
    with calibration_app() as fixture:
        assert request(fixture, "PUT", "/api/attendance/cameras/1", save_payload())[0] == 200
        approved = seed_commission(fixture.engine)
        assert request(fixture, "PUT", "/api/cameras/1", {"name": "Renamed", "password": ""})[0] == 200
        with Session(fixture.engine) as db:
            config = db.get(AttendanceCamera, 1)
            assert (config.source_revision, config.revision, config.enabled) == (1, 1, True)
            assert db.get(AttendanceRuntime, 1).commission_id == approved


@test
def test_source_invalidation_failure_rolls_back_camera_and_attendance_together():
    with calibration_app() as fixture:
        assert request(fixture, "PUT", "/api/attendance/cameras/1", save_payload())[0] == 200
        approved = seed_commission(fixture.engine)

        def fail_config_update(_conn, _cursor, statement, _parameters, _context, _many):
            if statement.lstrip().upper().startswith("UPDATE ATTENDANCE_CAMERAS"):
                raise OperationalError(statement, {}, RuntimeError("Injected disk failure"))

        event.listen(fixture.engine, "before_cursor_execute", fail_config_update)
        try:
            status, body = request(fixture, "PUT", "/api/cameras/1", {"nvr_host": "replacement"})
        finally:
            event.remove(fixture.engine, "before_cursor_execute", fail_config_update)
        assert status == 503, body
        assert fixture.manager.published == []
        with Session(fixture.engine) as db:
            assert camera_service.get_camera(db, 1).nvr_host == "unused"
            config = db.get(AttendanceCamera, 1)
            assert (config.source_revision, config.revision, config.enabled) == (1, 1, True)
            assert db.get(AttendanceRuntime, 1).commission_id == approved


@test
def test_committed_camera_source_publication_is_an_immutable_transaction_snapshot():
    with calibration_app() as fixture:
        with Session(fixture.engine) as db:
            camera = camera_service.get_camera(db, 1)
            snapshot, changed = camera_service.update_camera(
                db, camera, camera_service.CameraUpdate(
                    name="Door A", nvr_host="source-a", rtsp_port=8554, username="operator-a",
                    password="secret-a", channel=7, stream_type="sub"))
            assert snapshot.nvr_host == "source-a"
            assert changed == {"name", "nvr_host", "rtsp_port", "username", "password", "channel", "stream_type"}
            try:
                snapshot.nvr_host = "not-the-committed-source"
            except (FrozenInstanceError, AttributeError):
                pass
            else:
                raise AssertionError("Committed camera/source publication must be immutable")
        with Session(fixture.engine) as db:
            newer, _changed = camera_service.update_camera(
                db, camera_service.get_camera(db, 1),
                camera_service.CameraUpdate(
                    name="Door B", nvr_host="source-b", rtsp_port=9554, username="operator-b",
                    password="secret-b", channel=8, stream_type="main"))
        assert (snapshot.id, snapshot.name, snapshot.nvr_host, snapshot.rtsp_port, snapshot.username,
                snapshot.password, snapshot.channel, snapshot.stream_type, snapshot.enabled,
                snapshot.source_revision, snapshot.has_password) == (
                    1, "Door A", "source-a", 8554, "operator-a", "secret-a", 7, "sub", True, 2, True)
        assert snapshot.created_at is not None and snapshot.updated_at is not None
        assert (newer.nvr_host, newer.rtsp_port, newer.username, newer.password, newer.channel,
                newer.stream_type, newer.source_revision) == (
                    "source-b", 9554, "operator-b", "secret-b", 8, "main", 3)


@test
def test_concurrent_updates_publish_matching_sources_and_cannot_roll_reader_back():
    with calibration_app() as fixture:
        with patch("app.rtsp.manager.resolve_ffmpeg", return_value=("unused", "unused")):
            manager = ReaderManager()
        fixture.app.state.manager = manager
        first_captured, release_first = threading.Event(), threading.Event()
        original_update = camera_service.update_camera
        published, outcomes, errors = [], {}, []

        def capture_reader_start(reader):
            published.append((urlsplit(reader.rtsp_url).hostname, reader.source_revision))

        def delay_first_publication(db, camera, payload):
            result = original_update(db, camera, payload)
            if payload.nvr_host == "source-a":
                first_captured.set()
                if not release_first.wait(10):
                    raise AssertionError("Concurrent update coordination did not release source A")
            return result

        def first_request():
            try:
                outcomes["a"] = request(fixture, "PUT", "/api/cameras/1", {"nvr_host": "source-a"})
            except BaseException as exc:
                errors.append(exc)

        with patch.object(camera_service, "update_camera", side_effect=delay_first_publication), \
                patch.object(RTSPReader, "start", autospec=True, side_effect=capture_reader_start):
            first = threading.Thread(target=first_request, name="source-update-a", daemon=True)
            first.start()
            try:
                assert first_captured.wait(10), "Source A did not finish its write before source B"
                outcomes["b"] = request(fixture, "PUT", "/api/cameras/1", {"nvr_host": "source-b"})
            finally:
                release_first.set()
                first.join(10)
            try:
                assert not first.is_alive() and not errors, errors
                assert outcomes["a"][0] == outcomes["b"][0] == 200, outcomes
                with Session(fixture.engine) as db:
                    assert camera_service.get_camera(db, 1).nvr_host == "source-b"
                    assert db.get(AttendanceCamera, 1).source_revision == 3
                assert published and all(item in (("source-a", 2), ("source-b", 3))
                                         for item in published), published
                reader = manager.get(1)
                assert (urlsplit(reader.rtsp_url).hostname, reader.source_revision) == ("source-b", 3), (
                    "A superseded publication must not replace the latest committed source", published)
            finally:
                manager.stop_all()


@test
def test_superseded_start_and_sync_cannot_restart_a_newer_disabled_source():
    for replace_connection in (False, True):
        with calibration_app() as fixture:
            with Session(fixture.engine) as db:
                old, _changed = camera_service.update_camera(
                    db, camera_service.get_camera(db, 1),
                    camera_service.CameraUpdate(nvr_host="source-a"))
                updates = {"enabled": False}
                if replace_connection:
                    updates["nvr_host"] = "source-b"
                retired, changed = camera_service.update_camera(
                    db, camera_service.get_camera(db, 1), camera_service.CameraUpdate(**updates))
            with patch("app.rtsp.manager.resolve_ffmpeg", return_value=("unused", "unused")):
                manager = ReaderManager()
            published = []
            with patch.object(RTSPReader, "start", autospec=True, side_effect=lambda reader:
                              published.append((urlsplit(reader.rtsp_url).hostname, reader.source_revision))):
                try:
                    manager.start(old)
                    manager.sync(retired, changed)
                    assert manager.get(1) is None
                    assert manager.start(old) is None
                    manager.sync(old, {"nvr_host"})
                    assert manager.get(1) is None
                    assert published == [("source-a", 2)]
                finally:
                    manager.stop_all()


@test
def test_calibration_and_snapshot_database_read_failure_returns_503():
    with calibration_app() as fixture:
        fixture.manager.snapshot()

        def fail_read(_conn, _cursor, statement, _parameters, _context, _many):
            if "FROM attendance_cameras" in statement:
                raise OperationalError(statement, {}, RuntimeError("Injected read failure"))

        event.listen(fixture.engine, "before_cursor_execute", fail_read)
        try:
            for path in ("/api/attendance/cameras/1", "/api/attendance/cameras/1/snapshot"):
                status, body = request(fixture, "GET", path)
                assert status == 503, body
        finally:
            event.remove(fixture.engine, "before_cursor_execute", fail_read)


if __name__ == "__main__":
    failed = 0
    for fn in TESTS:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception:
            failed += 1
            print(f"FAIL {fn.__name__}")
            traceback.print_exc()
    print(f"\n{len(TESTS) - failed}/{len(TESTS)} passed")
    raise SystemExit(1 if failed else 0)
