"""Shared work-time input types and UTC/date helpers."""
from datetime import date, datetime, timedelta, timezone
from typing import Annotated
from uuid import UUID

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, field_validator, model_validator

from app.time_utils import attendance_iso, day_bounds

MAX_CORRECTION = timedelta(hours=24)


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


class _Audited(BaseModel):
    """Every operator action names who did it and why."""
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    actor: str = Field(min_length=1, max_length=100)
    reason: str = Field(min_length=1, max_length=1000)

    @field_validator("actor", "reason")
    @classmethod
    def not_blank(cls, value):
        if not value.strip():
            raise ValueError("An operator and reason are required")
        if "\0" in value:
            raise ValueError("Audit text must not contain NUL characters")
        return value.strip()


class HumanInput(_Audited):
    request_id: UUID


class CorrectionInput(HumanInput):
    """A manual presence interval or an exclusion of a wrong automatic span."""
    employee_id: int = Field(gt=0)
    started_at: datetime
    ended_at: datetime

    @field_validator("started_at", "ended_at")
    @classmethod
    def aware_timestamp(cls, value):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("Timestamp requires an explicit UTC offset")
        normalized = milliseconds(value)
        attendance_iso(normalized)  # Accepted input must also render after commit.
        return normalized.replace(tzinfo=timezone.utc)

    @model_validator(mode="after")
    def interval(self):
        if self.ended_at <= self.started_at:
            raise ValueError("The interval must end after it starts")
        if self.ended_at - self.started_at > MAX_CORRECTION:
            raise ValueError("The interval cannot be longer than 24 hours")
        return self


class RevokeInput(_Audited):
    pass
