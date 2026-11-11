"""Shared attendance HTTP payloads and materialized read views."""
from datetime import date, datetime, timezone
from typing import Annotated, Literal
from uuid import UUID

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, field_validator, model_validator

from app.time_utils import attendance_iso, day_bounds


def milliseconds(value: datetime) -> datetime:
    try:
        if value.tzinfo is not None:
            value = value.astimezone(timezone.utc).replace(tzinfo=None)
        return value.replace(microsecond=value.microsecond // 1000 * 1000)
    except OverflowError as exc:
        raise ValueError("Timestamp cannot be represented in UTC") from exc


def _attendance_date(value: date) -> date:
    """Validate both bounds once at every public date-input boundary."""
    day_bounds(value)
    return value


AttendanceDate = Annotated[date, AfterValidator(_attendance_date)]


class HumanInput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    request_id: UUID
    actor: str = Field(min_length=1, max_length=100)
    reason: str = Field(min_length=1, max_length=1000)

    @field_validator("actor", "reason")
    @classmethod
    def not_blank(cls, value):
        if not value.strip():
            raise ValueError("An operator and reason are required")
        return value.strip()


class TimedInput(HumanInput):
    occurred_at: datetime | None = None

    @field_validator("occurred_at")
    @classmethod
    def aware_timestamp(cls, value):
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("Timestamp requires an explicit UTC offset")
        normalized = milliseconds(value)
        attendance_iso(normalized)  # Accepted input must also render after commit.
        return normalized.replace(tzinfo=timezone.utc)


class EventDecisionInput(TimedInput):
    expected_revision: int = Field(ge=0)
    action: Literal["ACCEPT", "VISITOR", "VOID"]
    employee_id: int | None = Field(default=None, gt=0)
    direction: Literal["ENTRY", "EXIT"] | None = None


class VisitorItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    event_id: str
    expected_revision: int = Field(ge=0)


class VisitorBatchInput(HumanInput):
    items: list[VisitorItem] = Field(min_length=1, max_length=100)

    @model_validator(mode="after")
    def unique_items(self):
        if len({item.event_id for item in self.items}) != len(self.items):
            raise ValueError("Duplicate event IDs")
        self.items.sort(key=lambda item: item.event_id)
        return self


class ManualEventInput(TimedInput):
    employee_id: int = Field(gt=0)
    direction: Literal["ENTRY", "EXIT"]
    occurred_at: datetime


class GapDecisionInput(HumanInput):
    expected_revision: int = Field(ge=0)


class EvidenceView(BaseModel):
    before: bool
    after: bool
    face: bool
    purged: bool


class EventView(BaseModel):
    id: str
    source: str
    camera_id: int | None
    camera_name: str | None
    employee_id: int | None
    employee_name: str | None
    direction: str | None
    occurred_at: str
    recorded_at: str
    status: str
    revision: int
    reason_codes: list[str]
    projection_reasons: list[str]
    evidence: EvidenceView
    needs_review: bool


class GapView(BaseModel):
    id: int
    started_at: str
    ended_at: str | None
    reason_codes: list[str]
    reviewed: bool
    revision: int


class SessionView(BaseModel):
    entry_event_id: str
    exit_event_id: str | None
    employee_id: int
    employee_name: str
    started_at: str
    ended_at: str | None
    status: str
    reason_codes: list[str]
    duration_ms: int
    confirmed_ms: int
    running_ms: int
    review_ms: int


class EventPage(BaseModel):
    items: list[EventView]
    total: int
    limit: int
    offset: int


class SessionPage(BaseModel):
    items: list[SessionView]
    total: int
    limit: int
    offset: int


class CommissionAcceptance(BaseModel):
    model_config = ConfigDict(extra="forbid")
    entry_trials: int = Field(ge=0)
    exit_trials: int = Field(ge=0)
    employee_count: int = Field(ge=0)
    visitor_trials: int = Field(ge=0)
    retreat_trials: int = Field(ge=0)
    wrong_employee: int = Field(ge=0)
    duplicate_trips: int = Field(ge=0)
    missed_unflagged: int = Field(ge=0)
    frontal_trials: int = Field(ge=0)
    frontal_eligible: int = Field(ge=0)
    performance_minutes: int = Field(ge=0)
    site_confirmed: bool


class CommissionInput(HumanInput):
    expected_entry_revision: int = Field(ge=0)
    expected_exit_revision: int = Field(ge=0)
    acceptance_record: CommissionAcceptance


class DecisionView(BaseModel):
    id: int
    request_id: str
    revision: int
    actor: str
    reason: str
    before: dict
    after: dict
    created_at: str


class EventDetail(EventView):
    raw: dict
    decisions: list[DecisionView]


class ActiveView(BaseModel):
    items: list[SessionView]
    total: int
    as_of: str
    utc_offset_minutes: int


class ReviewEvent(BaseModel):
    kind: Literal["event"] = "event"
    event: EventView


class ReviewGap(BaseModel):
    kind: Literal["gap"] = "gap"
    gap: GapView


class ReviewPage(BaseModel):
    items: list[ReviewEvent | ReviewGap]
    total: int
    limit: int
    offset: int


class GapDetail(GapView):
    decisions: list[DecisionView]
