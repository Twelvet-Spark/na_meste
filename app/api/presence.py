"""REST API присутствия: активные сессии и история."""
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database.database import get_db
from app.database.models import Camera, Employee, PresenceSession
from app.time_utils import iso_local

router = APIRouter(prefix="/api/presence", tags=["presence"])


def _session_row(session: PresenceSession, employee_name: str | None,
                 camera_name: str | None) -> dict:
    return {
        "id": session.id,
        "employee_id": session.employee_id,
        "employee_name": employee_name,
        "camera_id": session.camera_id,
        "camera_name": camera_name,
        "started_at": iso_local(session.started_at),
        "last_seen_at": iso_local(session.last_seen_at),
        "ended_at": iso_local(session.ended_at),
        "confidence": session.confidence,
    }


def _query(db: Session, q):
    """JOIN с именами сотрудника и камеры."""
    return q.join(Employee, PresenceSession.employee_id == Employee.id, isouter=True)\
            .join(Camera, PresenceSession.camera_id == Camera.id, isouter=True)\
            .add_columns(Employee.name, Camera.name)


@router.get("")
def api_presence_history(limit: int = 100, db: Session = Depends(get_db)):
    """История сессий (последние N), свежие сверху."""
    limit = max(1, min(limit, 1000))
    rows = _query(
        db,
        db.query(PresenceSession),
    ).order_by(PresenceSession.last_seen_at.desc()).limit(limit).all()
    return [_session_row(s, emp, cam) for s, emp, cam in rows]


@router.get("/active")
def api_presence_active(db: Session = Depends(get_db)):
    """Открытые сессии (человек сейчас на камере)."""
    rows = _query(
        db,
        db.query(PresenceSession).filter(PresenceSession.ended_at.is_(None)),
    ).order_by(PresenceSession.last_seen_at.desc()).all()
    return [_session_row(s, emp, cam) for s, emp, cam in rows]


@router.get("/employee/{employee_id}")
def api_presence_employee(employee_id: int, db: Session = Depends(get_db)):
    if db.get(Employee, employee_id) is None:
        raise HTTPException(status_code=404, detail="Employee not found")
    rows = _query(
        db,
        db.query(PresenceSession).filter(PresenceSession.employee_id == employee_id),
    ).order_by(PresenceSession.last_seen_at.desc()).limit(200).all()
    return [_session_row(s, emp, cam) for s, emp, cam in rows]
