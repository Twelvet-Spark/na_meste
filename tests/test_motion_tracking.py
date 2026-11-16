"""Consumer-visible motion and association regressions, without live inference.

Run: .venv/Scripts/python.exe tests/test_motion_tracking.py
"""
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from app.ai.tracker import ByteTracker

TESTS = []


def test(fn):
    TESTS.append(fn)
    return fn


def box(left=0., width=100., height=200.):
    return np.array([left, 0., left + width, height], dtype=np.float32)


def identify(track):
    track.employee_id = 7
    track.employee_conf = .9
    track.candidate_id = 7
    track.candidate_count = 3
    track.last_recog_attempt = track.last_seen
    track.best_unknown_sim = .4


def assert_unknown(track):
    assert track.employee_id is None and track.employee_conf is None
    assert track.candidate_id is None and track.candidate_count == 0
    assert track.last_recog_attempt == 0. and track.best_unknown_sim is None


@test
def test_elapsed_motion_recovers_after_irregular_receipts_and_a_gap():
    for gap in (.8, 2.4):
        for score in (.91, .2):
            tracker = ByteTracker()
            first = None
            for now, left in ((10., 0.), (10.1, 10.), (10.25, 25.),
                              (10.4, 40.), (10.6, 60.)):
                observed = tracker.update([(box(left, 60., 120.), .99)], now=now)
                assert len(observed) == 1
                first = observed[0] if first is None else first
                assert observed[0].track_id == first.track_id
            assert first.hits == 5
            assert tracker.update([], now=10.8) == []
            assert first.last_seen == 10.6 and first.hits == 0
            receipt = 10.6 + gap
            # More than one measured box width from the last observation. The
            # small offset also proves that publication uses the real detection.
            measured = box(60. + 100. * gap + 3., 60., 120.)
            observed = tracker.update([(measured, score)], now=receipt)
            assert len(observed) == 1 and observed[0].track_id == first.track_id
            np.testing.assert_array_equal(observed[0].bbox, measured)
            assert observed[0].score == score and observed[0].last_seen == receipt
            assert observed[0].hits == 1 and not observed[0].is_new
            assert observed[0].first_seen == 10.
            assert not observed[0].association_uncertain


@test
def test_changing_person_scale_can_recover_without_old_box_overlap():
    def scaled_box(elapsed):
        height = 100. + 100. * elapsed
        width = height * (.5 + .2 * elapsed)
        return np.array([400. - width / 2., 400. - height / 2.,
                         400. + width / 2., 400. + height / 2.], dtype=np.float32)

    tracker = ByteTracker()
    first_id = None
    for elapsed in (0., .1, .2, .3, .4, .5):
        observed = tracker.update([(scaled_box(elapsed), .99)], now=10. + elapsed)
        assert len(observed) == 1
        first_id = observed[0].track_id if first_id is None else first_id
        assert observed[0].track_id == first_id
    assert tracker.update([], now=10.6) == []
    measured = scaled_box(2.5)
    observed = tracker.update([(measured, .9)], now=12.5)
    assert len(observed) == 1 and observed[0].track_id == first_id
    np.testing.assert_array_equal(observed[0].bbox, measured)
    assert observed[0].hits == 1


@test
def test_global_assignment_avoids_greedy_trap_without_close_costs():
    for threshold in (.1, .3):
        tracker = ByteTracker(match_threshold=threshold)
        first = tracker.update([(box(0.), .99), (box(50.), .99)], now=10.)
        for track in first:
            identify(track)
        # IoUs by original track: [.818, .538], [.429, .111].
        # No competing pair is within .10. Greedy takes .818 and strands track 2
        # at the default gate; the global pairing also has the larger total IoU.
        observed = tracker.update([(box(10.), .95), (box(-30.), .96)], now=10.1)
        by_id = {track.track_id: track for track in observed}
        assert set(by_id) == {1, 2}
        np.testing.assert_array_equal(by_id[1].bbox, box(-30.))
        np.testing.assert_array_equal(by_id[2].bbox, box(10.))
        assert by_id[1].score == .96 and by_id[2].score == .95
        assert all(track.hits == 2 and track.employee_id == 7 for track in observed)
        assert all(not track.association_uncertain for track in observed)


@test
def test_high_score_stage_precedes_a_better_low_score_overlap():
    tracker = ByteTracker()
    first = tracker.update([(box(), .99)], now=10.)[0]
    observed = tracker.update([(box(), .2), (box(20.), .9)], now=10.1)
    assert len(observed) == 1 and observed[0].track_id == first.track_id
    np.testing.assert_array_equal(observed[0].bbox, box(20.))
    assert observed[0].score == .9 and observed[0].hits == 2


@test
def test_low_scores_recover_existing_tracks_but_only_high_scores_start_tracks():
    tracker = ByteTracker()
    tracker.update([(box(), .99), (box(300.), .99)], now=10.)
    assert tracker.update([(box(600.), .2)], now=10.1) == []
    observed = tracker.update([(box(1.), .9), (box(301.), .2),
                               (box(600.), .2), (box(900.), .95)], now=10.2)
    by_id = {track.track_id: track for track in observed}
    assert set(by_id) == {1, 2, 3}
    assert by_id[1].bbox[0] == 1. and by_id[2].bbox[0] == 301.
    assert by_id[2].score == .2 and by_id[2].hits == 1
    assert by_id[3].bbox[0] == 900. and by_id[3].is_new and by_id[3].hits == 1
    assert by_id[3].last_seen == by_id[3].first_seen == 10.2


@test
def test_ambiguity_quarantines_the_whole_high_low_connected_component():
    tracker = ByteTracker()
    first = tracker.update([(box(left), .99) for left in (0., 60., 120., 500.)],
                           now=10.)
    for track in first:
        identify(track)
    # The first detection ties tracks 1/2. Track 2 also reaches the low-score
    # detection, which reaches track 3 with a clearly better individual cost.
    observed = tracker.update([(box(30.), .9), (box(100.), .2), (box(501.), .9)],
                              now=10.1)
    assert [track.track_id for track in observed] == [4]
    assert observed[0].employee_id == 7 and observed[0].hits == 2
    ambiguous = {item.track_id for item in tracker.lifecycle if item.state == "AMBIGUOUS"}
    assert ambiguous == {1, 2, 3}
    assert len(tracker.tracks) == 4, "Quarantined detections must not spawn new IDs"
    for track in first[:3]:
        assert_unknown(track)
        assert track.association_uncertain and track.state == "lost"
        assert track.hits == 0 and track.last_seen == 10.
    observed = tracker.update([(box(), .9)], now=10.2)
    assert len(observed) == 1 and observed[0].track_id == 1
    assert_unknown(observed[0])
    assert observed[0].hits == 1 and not observed[0].association_uncertain


@test
def test_loss_clears_confirmed_and_pending_identity_before_recovery():
    for score in (.9, .2):
        tracker = ByteTracker()
        first = tracker.update([(box(), .99)], now=10.)[0]
        identify(first)
        assert tracker.update([], now=10.1) == []
        assert_unknown(first)
        assert not first.association_uncertain and first.state == "lost"
        assert [(item.track_id, item.state, item.last_seen_mono)
                for item in tracker.lifecycle] == [(1, "LOST", 10.)]
        recovered = tracker.update([(box(1.), score)], now=10.2)
        assert len(recovered) == 1 and recovered[0].track_id == first.track_id
        assert recovered[0].hits == 1
        assert_unknown(recovered[0])


@test
def test_prediction_only_updates_preserve_measurements_and_cannot_extend_lifetime():
    for score in (.9, .2):
        tracker = ByteTracker()
        first = tracker.update([(box(), .99)], now=0.)[0]
        tracker.update([(box(20.), .84)], now=1.)
        identify(first)
        for now in (2., 3., 4.):
            assert tracker.update([], now=now) == []
            np.testing.assert_array_equal(first.bbox, box(20.))
            assert first.score == .84 and first.last_seen == 1. and first.hits == 0
            assert_unknown(first)
        observed = tracker.update([(box(20.), score)], now=4.01)
        assert all(track.track_id != first.track_id for track in observed)
        assert all(track.track_id != first.track_id for track in tracker.tracks)
        expired = [item for item in tracker.lifecycle if item.state == "EXPIRED"]
        assert len(expired) == 1 and expired[0].track_id == first.track_id
        assert expired[0].last_seen_mono == 1.
        assert len(observed) == (1 if score >= .5 else 0)
        tracker.update([], now=4.02)
        assert all(item.state != "EXPIRED" for item in tracker.lifecycle)


@test
def test_first_hit_is_immediately_returned_as_the_measured_observation():
    tracker = ByteTracker()
    measured = np.array([12.5, 7.25, 19.5, 11.75], dtype=np.float32)
    observed = tracker.update([(measured, .67)], now=10.)
    assert len(observed) == 1
    track = observed[0]
    np.testing.assert_array_equal(track.bbox, measured)
    assert track.score == .67 and track.hits == 1 and track.is_new
    assert track.first_seen == track.last_seen == 10. and track.state == "tracked"
    assert [(item.track_id, item.state, item.last_seen_mono)
            for item in tracker.lifecycle] == [(track.track_id, "TRACKED", 10.)]


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
