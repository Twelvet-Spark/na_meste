"""Shared attendance HTTP payloads and materialized read views."""
from datetime import datetime, timezone
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def milliseconds(value: datetime) -> datetime:
    if value.tzinfo is not None:
        value = value.astimezone(timezone.utc).replace(tzinfo=None)
    return value.replace(microsecond=value.microsecond // 1000 * 1000)


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
        return milliseconds(value).replace(tzinfo=timezone.utc)


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
