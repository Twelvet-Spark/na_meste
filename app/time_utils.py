"""UTC storage and explicit workplace-time rendering.

Attendance uses a configured fixed offset, never the server/browser timezone.
The legacy global-history renderer deliberately retains its local-time behavior.
"""
from datetime import date, datetime, time, timedelta, timezone

from app.config import settings


def utcnow() -> datetime:
    """Return naive UTC, matching SQLite timestamp storage."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


def iso_local(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    return dt.replace(tzinfo=timezone.utc).astimezone().isoformat(timespec="seconds")


def attendance_iso(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    zone = timezone(timedelta(minutes=settings.attendance_utc_offset_minutes))
    aware = dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt
    return aware.astimezone(zone).isoformat()


def day_bounds(day: date) -> tuple[datetime, datetime]:
    """Return half-open naive-UTC bounds of a workplace calendar date."""
    start = datetime.combine(day, time.min) - timedelta(minutes=settings.attendance_utc_offset_minutes)
    return start, start + timedelta(days=1)
