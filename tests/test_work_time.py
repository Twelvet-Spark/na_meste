"""Pure per-day work-time computation: bridging, door gaps, exclusions, day clipping.

Run: .venv/Scripts/python.exe tests/test_work_time.py
"""
import sys
import traceback
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.work_time import Crossing, Mark, Span, compute_day, subtract

TESTS = []
LOWER = datetime(2026, 3, 10)
UPPER = LOWER + timedelta(days=1)
H = timedelta(hours=1)
M = timedelta(minutes=1)


def test(fn):
    TESTS.append(fn)
    return fn


def at(hours, minutes=0):
    return LOWER + timedelta(hours=hours, minutes=minutes)


def seg(start, end, *, face=True, camera=1):
    return Mark(start, end, face, "SEGMENT", camera)


def ms(delta):
    return int(delta.total_seconds() * 1000)


def day(*, segments=(), manual=(), crossings=(), exclusions=(), as_of=None):
    return compute_day(segments=list(segments), manual=list(manual), crossings=list(crossings),
                       exclusions=list(exclusions), lower=LOWER, upper=UPPER, as_of=as_of or UPPER)


MORNING_AND_AFTERNOON = [seg(at(9), at(12)), seg(at(15), at(18))]


@test
def finished_day_counts_two_blocks_across_a_long_absence():
    result = day(segments=MORNING_AND_AFTERNOON)
    assert result.method == "FINAL" and ms(result.worked) == 21600000
    assert (result.first, result.last) == (at(9), at(18)) and len(result.sessions) == 2


@test
def running_day_gives_the_same_six_hours_with_live_method():
    result = day(segments=MORNING_AND_AFTERNOON, as_of=at(19))
    assert result.method == "LIVE" and ms(result.worked) == 21600000


@test
def ninety_minute_gap_is_counted_in_the_final_day_but_not_while_running():
    segments = [seg(at(9), at(12)), seg(at(13, 30), at(17))]
    final = day(segments=segments)
    live = day(segments=segments, as_of=at(18))
    assert final.method == "FINAL" and ms(final.worked) == ms(8 * H) and len(final.sessions) == 1
    assert live.method == "LIVE" and ms(live.worked) == ms(6.5 * H) and len(live.sessions) == 2


@test
def gap_exactly_at_the_bridge_is_bridged():
    segments = [seg(at(9), at(10)), seg(at(10, 35), at(11))]
    assert len(day(segments=segments, as_of=at(12)).sessions) == 1
    segments = [seg(at(9), at(10)), seg(at(10, 36), at(11))]
    assert len(day(segments=segments, as_of=at(12)).sessions) == 2


@test
def exit_entry_pair_removes_its_gap_in_live_and_final():
    # 09:00-15:00 span with a 20 minute door gap inside: 6 h minus 20 min.
    segments = [seg(at(9), at(11, 40)), seg(at(12), at(15))]
    crossings = [Crossing(at(11, 40), "EXIT", 7), Crossing(at(12), "ENTRY", 7)]
    for as_of in (UPPER, at(16)):
        result = day(segments=segments, crossings=crossings, as_of=as_of)
        assert ms(result.worked) == 21600000 - 1200000, result
        assert result.door_gaps == (Span(at(11, 40), at(12)),)
        assert len(result.sessions) == 2


@test
def door_crossing_within_snap_window_of_a_sighting_still_closes_the_gap():
    segments = [seg(at(9), at(11, 40)), seg(at(12), at(15))]
    crossings = [Crossing(at(11, 40) + timedelta(seconds=20), "EXIT", 7),
                 Crossing(at(12) - timedelta(seconds=20), "ENTRY", 7)]
    result = day(segments=segments, crossings=crossings)
    assert len(result.door_gaps) == 1 and len(result.sessions) == 2


@test
def exit_followed_by_a_sighting_inside_without_entry_is_bridged():
    segments = [seg(at(9), at(10)), seg(at(10, 10), at(11))]
    result = day(segments=segments, crossings=[Crossing(at(10), "EXIT", 7)], as_of=at(12))
    assert len(result.sessions) == 1 and ms(result.worked) == ms(2 * H) and not result.door_gaps


@test
def exclusion_removes_segment_time_and_a_crossing_inside_it_but_not_manual_time():
    segments = [seg(at(9), at(12))]
    cut = Span(at(10), at(11))
    result = day(segments=segments, exclusions=[cut], as_of=at(13))
    assert ms(result.worked) == ms(2 * H) and len(result.sessions) == 2
    # A finished day bridges the one-hour hole (absences up to 120 minutes still count).
    assert ms(day(segments=segments, exclusions=[cut]).worked) == ms(3 * H)
    # A crossing inside the exclusion is dropped, so it cannot close a door gap.
    crossings = [Crossing(at(10, 30), "EXIT", 7), Crossing(at(10, 50), "ENTRY", 7)]
    plain = day(segments=segments, exclusions=[cut], crossings=crossings, as_of=at(13))
    assert not plain.door_gaps and ms(plain.worked) == ms(2 * H)
    kept = day(segments=segments, exclusions=[cut], manual=[Span(at(10, 15), at(10, 45))], as_of=at(13))
    assert ms(kept.worked) == ms(3 * H)


@test
def local_midnight_splits_a_segment_between_days():
    crossing_midnight = [seg(at(22), at(26))]
    today = day(segments=crossing_midnight)
    assert ms(today.worked) == ms(2 * H) and today.last == UPPER
    next_day = compute_day(segments=crossing_midnight, manual=[], crossings=[], exclusions=[],
                           lower=UPPER, upper=UPPER + timedelta(days=1), as_of=UPPER + timedelta(days=2))
    assert ms(next_day.worked) == ms(2 * H) and next_day.first == UPPER


@test
def last_sighting_exactly_at_midnight_belongs_to_the_new_day():
    segments = [seg(LOWER - 10 * M, LOWER), seg(LOWER + 30 * M, LOWER + 30 * M)]
    result = day(segments=segments, as_of=LOWER + H)
    assert result.first == LOWER and result.last == LOWER + 30 * M
    assert result.worked == 30 * M, result
    # A manual interval's exclusive end is not a real sighting at midnight.
    manual_only = day(manual=[Span(LOWER - H, LOWER)], as_of=LOWER + H)
    assert manual_only.first is None and not manual_only.sessions


@test
def empty_input_is_an_empty_result():
    result = day()
    assert result.worked == timedelta(0) and result.first is None and result.sessions == ()
    assert result.last_mark is None


@test
def single_sighting_is_presence_without_duration():
    result = day(segments=[seg(at(9), at(9))], as_of=at(9, 5))
    assert result.worked == timedelta(0) and result.sessions == (Span(at(9), at(9)),)
    assert result.last_mark.source == "SEGMENT"


@test
def last_mark_prefers_a_sighting_over_a_door_mark_at_the_same_instant():
    result = day(segments=[seg(at(9), at(10), face=False)], crossings=[Crossing(at(10), "EXIT", 7)],
                 as_of=at(10, 1))
    assert result.last_mark.source == "SEGMENT" and result.last_mark.face_seen is False


@test
def subtract_keeps_both_outer_pieces_and_the_fields():
    pieces = subtract([seg(at(9), at(12), camera=5)], [Span(at(10), at(11))])
    assert [(p.start, p.end, p.camera_id) for p in pieces] == [(at(9), at(10), 5), (at(11), at(12), 5)]
    assert subtract([seg(at(9), at(9))], [Span(at(9), at(10))]) == []
    assert len(subtract([seg(at(9), at(10))], [Span(at(10), at(11))])) == 1


if __name__ == "__main__":
    failures = 0
    for case in TESTS:
        try:
            case()
            print(f"PASS {case.__name__}")
        except Exception:
            failures += 1
            print(f"FAIL {case.__name__}")
            traceback.print_exc()
    print(f"{len(TESTS) - failures}/{len(TESTS)} passed")
    raise SystemExit(bool(failures))
