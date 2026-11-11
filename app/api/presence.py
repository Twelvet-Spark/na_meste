"""Read-only pre-attendance observations: never evidence of worked hours."""
from datetime import date

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.database.database import get_db
from app.database.models import Camera, Employee, PresenceSession
from app.services.attendance_queries import read_snapshot
from app.time_utils import attendance_iso, day_bounds

router = APIRouter(prefix="/api/presence", tags=["legacy observations"])


@router.get("/legacy")
def legacy_observations(employee_id: int | None = None, from_date: date | None = None,
                        to_date: date | None = None, limit: int = Query(50, ge=1, le=200),
                        offset: int = Query(0, ge=0), db: Session = Depends(get_db)):
    if from_date is not None and to_date is not None and to_date < from_date:
        raise HTTPException(422, "End date precedes start date")

    def read(snapshot):
        if employee_id is not None and snapshot.get(Employee, employee_id) is None:
            raise HTTPException(404, "Employee not found")
        query = snapshot.query(PresenceSession)
        if employee_id is not None:
            query = query.filter(PresenceSession.employee_id == employee_id)
        if from_date is not None:
            query = query.filter(PresenceSession.started_at >= day_bounds(from_date)[0])
        if to_date is not None:
            query = query.filter(PresenceSession.started_at < day_bounds(to_date)[1])
        total = query.count()
        rows = query.outerjoin(Employee, Employee.id == PresenceSession.employee_id).outerjoin(
            Camera, Camera.id == PresenceSession.camera_id).add_columns(Employee.name, Camera.name).order_by(
                PresenceSession.started_at.desc(), PresenceSession.id.desc()).offset(offset).limit(limit).all()
        items = [dict(id=row.id, employee_id=row.employee_id, employee_name=employee_name,
                      camera_id=row.camera_id, camera_name=camera_name,
                      started_at=attendance_iso(row.started_at), last_seen_at=attendance_iso(row.last_seen_at),
                      ended_at=attendance_iso(row.ended_at), confidence=row.confidence,
                      historical_unclosed=row.ended_at is None)
                 for row, employee_name, camera_name in rows]
        return dict(items=items, total=total, limit=limit, offset=offset, trusted_for_hours=False)

    try:
        return read_snapshot(db.get_bind().engine, read)
    except SQLAlchemyError as exc:
        raise HTTPException(503, "Observation archive unavailable") from exc
