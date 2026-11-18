"""Pure work-time computation for one employee and one workplace calendar day.

Inputs are naive-UTC instants. A day is the half-open interval [lower, upper).
No I/O and no clocks: callers pass `as_of`, so results are deterministic.
"""
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from typing import Literal

from app.config import settings

DOOR_SNAP = timedelta(seconds=30)
_PRIORITY = {"SEGMENT": 2, "MANUAL": 1, "DOOR": 0}


@dataclass(frozen=True, slots=True)
class Span:
    start: datetime
    end: datetime


@dataclass(frozen=True, slots=True)
class Mark(Span):
    face_seen: bool
    source: Literal["SEGMENT", "MANUAL", "DOOR"]
    camera_id: int | None


@dataclass(frozen=True, slots=True)
class Crossing:
    at: datetime
    direction: Literal["ENTRY", "EXIT"]
    camera_id: int


@dataclass(frozen=True, slots=True)
class DayResult:
    method: Literal["LIVE", "FINAL"]
    sessions: tuple[Span, ...]
    worked: timedelta
    first: datetime | None
    last: datetime | None
    door_gaps: tuple[Span, ...]
    last_mark: Mark | None


def subtract(spans: list[Span], cuts: list[Span]) -> list[Span]:
    """Remove half-open cuts [start, end) from spans, keeping each span's own type and fields.

    A zero-length span disappears when it lies inside a cut; a longer span keeps the parts outside.
    """
    result = list(spans)
    for cut in cuts:
        remaining = []
        for span in result:
            if span.start == span.end:
                if not cut.start <= span.start < cut.end:
                    remaining.append(span)
                continue
            if span.end <= cut.start or span.start >= cut.end:
                remaining.append(span)
                continue
            if span.start < cut.start:
                remaining.append(replace(span, end=cut.start))
            if cut.end < span.end:
                remaining.append(replace(span, start=cut.end))
        result = remaining
    return result


def _clip(span: Span, lower: datetime, upper: datetime) -> Span | None:
    if span.start == span.end:
        return span if lower <= span.start < upper else None
    if isinstance(span, Mark) and span.source == "SEGMENT" and span.end == lower:
        # last_seen_at is an inclusive observation, unlike a manual interval's end.
        return replace(span, start=lower)
    clipped = replace(span, start=max(span.start, lower), end=min(span.end, upper))
    return clipped if clipped.start < clipped.end else None


def compute_day(*, segments: list[Mark], manual: list[Span], crossings: list[Crossing],
                exclusions: list[Span], lower: datetime, upper: datetime, as_of: datetime) -> DayResult:
    final = as_of >= upper
    bridge = timedelta(minutes=settings.attendance_day_gap_minutes if final
                       else settings.attendance_live_gap_minutes)
    cuts = [clipped for clipped in (_clip(span, lower, upper) for span in exclusions) if clipped is not None]

    clipped_segments = [c for c in (_clip(mark, lower, upper) for mark in segments) if c is not None]
    marks: list[Mark] = subtract(clipped_segments, cuts)
    for span in manual:  # Manual intervals are never subtracted.
        clipped = _clip(span, lower, upper)
        if clipped is not None:
            marks.append(Mark(clipped.start, clipped.end, False, "MANUAL", None))
    kept = [c for c in crossings
            if lower <= c.at < upper and not any(cut.start <= c.at < cut.end for cut in cuts)]
    for crossing in kept:
        marks.append(Mark(crossing.at, crossing.at, False, "DOOR", crossing.camera_id))

    method = "FINAL" if final else "LIVE"
    if not marks:
        return DayResult(method, (), timedelta(0), None, None, (), None)

    last_mark = max(marks, key=lambda m: (m.end, _PRIORITY[m.source]))
    marks.sort(key=lambda m: (m.start, m.end))
    pieces: list[Span] = []
    for mark in marks:
        if pieces and mark.start <= pieces[-1].end:
            if mark.end > pieces[-1].end:
                pieces[-1] = Span(pieces[-1].start, mark.end)
        else:
            pieces.append(Span(mark.start, mark.end))

    exits = [c.at for c in kept if c.direction == "EXIT"]
    entries = [c.at for c in kept if c.direction == "ENTRY"]
    sessions: list[Span] = []
    door_gaps: list[Span] = []
    current = pieces[0]
    for previous, following in zip(pieces, pieces[1:]):
        door_closed = (any(previous.end - DOOR_SNAP <= at <= previous.end for at in exits)
                       and any(following.start <= at <= following.start + DOOR_SNAP for at in entries))
        if door_closed:
            door_gaps.append(Span(previous.end, following.start))
        if following.start - previous.end <= bridge and not door_closed:
            current = Span(current.start, following.end)
        else:
            sessions.append(current)
            current = following
    sessions.append(current)
    worked = sum((s.end - s.start for s in sessions), timedelta(0))
    return DayResult(method, tuple(sessions), worked, pieces[0].start, pieces[-1].end,
                     tuple(door_gaps), last_mark)
