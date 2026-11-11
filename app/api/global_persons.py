"""REST API глобальных персон (межкамерный трекинг).

Список считается батч-запросами (имена/счётчики/наличие снимков — по одному
GROUP BY на всю страницу, без N+1). Таймлайн/траектория/наблюдения читают
только нужные колонки — эмбеддинги и снимки (BLOB) не тянутся из БД.
"""
import json
import logging

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from sqlalchemy import String, cast, func, or_
from sqlalchemy.orm import Session

from app.database.database import get_db
from app.database.models import (
    Camera, Employee, GlobalEvent, GlobalObservation, GlobalPerson,
)
from app.time_utils import iso_local

logger = logging.getLogger("app.api.global_persons")

router = APIRouter(prefix="/api/global-persons", tags=["global-persons"])

LIST_CAP = 500          # максимум строк списка
TRAJECTORY_CAP = 5000   # максимум наблюдений в траектории


def _person_or_404(db: Session, global_id: int) -> GlobalPerson:
    person = db.get(GlobalPerson, global_id)
    if person is None:
        raise HTTPException(status_code=404, detail="Global person not found")
    return person


def get_global_manager(request: Request):
    """GlobalIdentityManager из работающего AI-воркера (None — AI выключен)."""
    worker = getattr(request.app.state, "ai_worker", None)
    return getattr(worker, "global_manager", None) if worker else None


def _rows(db: Session, persons: list[GlobalPerson]) -> list[dict]:
    """Строки для списка/карточки: имена, счётчик наблюдений, наличие
    снимка — батч-запросами по id страницы."""
    if not persons:
        return []
    ids = [p.id for p in persons]
    cameras = {c.id: c.name for c in db.query(Camera).all()}
    employees = {e.id: e.name for e in db.query(Employee).all()}
    obs_counts = dict(db.query(GlobalObservation.global_id, func.count(GlobalObservation.id))
                      .filter(GlobalObservation.global_id.in_(ids))
                      .group_by(GlobalObservation.global_id).all())
    has_photo = {row[0] for row in db.query(GlobalObservation.global_id)
                 .filter(GlobalObservation.global_id.in_(ids),
                         GlobalObservation.snapshot.isnot(None))
                 .group_by(GlobalObservation.global_id).all()}
    return [
        {
            "global_id": p.id,
            "status": p.status,
            "employee_id": p.employee_id,
            "employee_name": employees.get(p.employee_id),
            "last_camera_id": p.last_camera_id,
            "last_camera_name": cameras.get(p.last_camera_id),
            "last_track_id": p.last_track_id,
            "observations_count": obs_counts.get(p.id, 0),
            "has_photo": p.id in has_photo,
            "created_at": iso_local(p.created_at),
            "last_seen_at": iso_local(p.last_seen_at),
        }
        for p in persons
    ]


@router.get("")
def api_list_global_persons(
    limit: int = 100,
    status: str | None = None,
    camera_id: int | None = None,
    employee_id: int | None = None,
    search: str = "",
    db: Session = Depends(get_db),
):
    """Глобальные личности, свежие сверху.

    Фильтры: status (NEW/ACTIVE/LOST), camera_id, employee_id,
    search — подстрока global_id («184», «G#184») или имени сотрудника.
    """
    limit = max(1, min(limit, LIST_CAP))
    employees = {e.id: e.name for e in db.query(Employee).all()}

    query = db.query(GlobalPerson)
    if status:
        query = query.filter(GlobalPerson.status == status.upper())
    if camera_id is not None:
        query = query.filter(GlobalPerson.last_camera_id == camera_id)
    if employee_id is not None:
        query = query.filter(GlobalPerson.employee_id == employee_id)
    search = search.strip()
    if search:
        conds = []
        digits = "".join(ch for ch in search if ch.isdigit())
        if digits:
            conds.append(cast(GlobalPerson.id, String).like(f"%{digits}%"))
        needle = search.lower()
        by_name = [eid for eid, name in employees.items() if needle in name.lower()]
        if by_name:
            conds.append(GlobalPerson.employee_id.in_(by_name))
        if not conds:
            return []
        query = query.filter(or_(*conds))
    persons = query.order_by(GlobalPerson.last_seen_at.desc()).limit(limit).all()
    return _rows(db, persons)


@router.get("/{global_id}")
def api_get_global_person(global_id: int, db: Session = Depends(get_db)):
    rows = _rows(db, [_person_or_404(db, global_id)])
    return rows[0]


@router.get("/{global_id}/photo")
def api_global_person_photo(global_id: int, db: Session = Depends(get_db)):
    """Последний снимок личности (для аватарки в списке)."""
    _person_or_404(db, global_id)
    row = db.query(GlobalObservation.snapshot).filter(
        GlobalObservation.global_id == global_id,
        GlobalObservation.snapshot.isnot(None),
    ).order_by(GlobalObservation.id.desc()).first()
    if row is None or not row[0]:
        raise HTTPException(status_code=404, detail="Snapshot not found")
    return Response(content=row[0], media_type="image/jpeg")


@router.post("/{global_id}/merge/{source_id}")
def api_merge_global_person(
    global_id: int, source_id: int,
    db: Session = Depends(get_db), manager=Depends(get_global_manager),
):
    """Объединить личности: G#source вливается в G#global_id (target).

    Дробление identity — осознанная стратегия матчинга (ложное слияние
    опаснее лишнего ID), поэтому фрагменты устраняются вручную: наблюдения
    и события переносятся на target, source удаляется.
    """
    if global_id == source_id:
        raise HTTPException(status_code=400,
                            detail="Нельзя объединить личность с самой собой")
    target = _person_or_404(db, global_id)
    source = _person_or_404(db, source_id)

    employee_conflict = (target.employee_id is not None
                         and source.employee_id is not None
                         and target.employee_id != source.employee_id)
    moved = {
        "observations": db.query(GlobalObservation)
            .filter(GlobalObservation.global_id == source_id)
            .update({"global_id": global_id}, synchronize_session=False),
        "events": db.query(GlobalEvent)
            .filter(GlobalEvent.global_id == source_id)
            .update({"global_id": global_id}, synchronize_session=False),
    }
    if target.employee_id is None and source.employee_id is not None:
        target.employee_id = source.employee_id
    if target.created_at is None or (source.created_at and source.created_at < target.created_at):
        target.created_at = source.created_at
    if source.last_seen_at and source.last_seen_at > target.last_seen_at:
        target.last_seen_at = source.last_seen_at
        target.last_camera_id = source.last_camera_id
        target.last_track_id = source.last_track_id
    if target.status == "LOST" and source.status != "LOST":
        target.status = source.status
    db.delete(source)
    db.commit()
    logger.info("Global: merge G#%d → G#%d (%s)", source_id, global_id, moved)

    if manager is not None:
        try:
            manager.merge(global_id, source_id)   # память: галерея + привязки
        except Exception:
            logger.exception("Global: merge в памяти не выполнен (БД уже слита)")

    row = _rows(db, [target])[0]
    row["merged_from"] = source_id
    row["moved"] = moved
    if employee_conflict:
        row["employee_conflict"] = True
    return row


def _set_employee(db: Session, global_id: int, employee_id: int | None,
                  manager) -> dict:
    """Общая часть привязки/отвязки: правка БД + синхронизация identity
    в памяти воркера, чтобы периодический sync не вернул None."""
    person = _person_or_404(db, global_id)
    if employee_id is not None and db.get(Employee, employee_id) is None:
        raise HTTPException(status_code=404, detail="Employee not found")
    person.employee_id = employee_id
    db.commit()
    if manager is not None:
        try:
            manager.assign_employee(global_id, employee_id)
        except Exception:
            logger.exception("Global: assign в памяти не выполнен (БД уже обновлена)")
    return _rows(db, [person])[0]


@router.put("/{global_id}/employee/{employee_id}")
def api_assign_employee(
    global_id: int, employee_id: int,
    db: Session = Depends(get_db), manager=Depends(get_global_manager),
):
    """Связать личность с сотрудником: ручная коррекция, когда распознавание
    лиц не сработало, но оператор знает, кто это (например, «G#184 — Иван»)."""
    return _set_employee(db, global_id, employee_id, manager)


@router.delete("/{global_id}/employee")
def api_unassign_employee(
    global_id: int,
    db: Session = Depends(get_db), manager=Depends(get_global_manager),
):
    """Снять привязку личности к сотруднику (сотрудник больше не она)."""
    return _set_employee(db, global_id, None, manager)


@router.get("/{global_id}/timeline")
def api_global_timeline(global_id: int, limit: int = 200, db: Session = Depends(get_db)):
    """Хронология: наблюдения + события (переходы и т.п.), свежие сверху."""
    _person_or_404(db, global_id)
    limit = max(1, min(limit, 1000))
    cameras = {c.id: c.name for c in db.query(Camera).all()}
    items = []
    # только нужные колонки: snapshot-BLOB не тянем, проверяем его NULL-ность
    observations = db.query(
        GlobalObservation.id,
        GlobalObservation.camera_id,
        GlobalObservation.track_id,
        GlobalObservation.snapshot.isnot(None).label("has_snapshot"),
        GlobalObservation.created_at,
    ).filter(
        GlobalObservation.global_id == global_id
    ).order_by(GlobalObservation.created_at.desc()).limit(limit).all()
    for o in observations:
        items.append({
            "kind": "observation",
            "id": o.id,
            "camera_id": o.camera_id,
            "camera_name": cameras.get(o.camera_id),
            "track_id": o.track_id,
            "has_snapshot": bool(o.has_snapshot),
            "created_at": iso_local(o.created_at),
        })
    events = db.query(GlobalEvent).filter(
        GlobalEvent.global_id == global_id
    ).order_by(GlobalEvent.created_at.desc()).limit(limit).all()
    for e in events:
        payload = json.loads(e.payload) if e.payload else {}
        item = {
            "kind": "event",
            "id": e.id,
            "event_type": e.event_type,
            "camera_id": e.camera_id,
            "camera_name": cameras.get(e.camera_id),
            "track_id": e.track_id,
            "payload": payload,
            "created_at": iso_local(e.created_at),
        }
        if e.event_type == "camera_transition":
            item["from_camera_name"] = cameras.get(payload.get("from_camera"))
            item["to_camera_name"] = cameras.get(payload.get("to_camera"))
        items.append(item)
    items.sort(key=lambda x: x["created_at"], reverse=True)
    return items


@router.get("/{global_id}/observations")
def api_global_observations(global_id: int, limit: int = 200,
                             db: Session = Depends(get_db)):
    _person_or_404(db, global_id)
    limit = max(1, min(limit, 1000))
    cameras = {c.id: c.name for c in db.query(Camera).all()}
    observations = db.query(
        GlobalObservation.id,
        GlobalObservation.camera_id,
        GlobalObservation.track_id,
        GlobalObservation.snapshot.isnot(None).label("has_snapshot"),
        GlobalObservation.created_at,
    ).filter(
        GlobalObservation.global_id == global_id
    ).order_by(GlobalObservation.created_at.desc()).limit(limit).all()
    return [
        {
            "id": o.id,
            "camera_id": o.camera_id,
            "camera_name": cameras.get(o.camera_id),
            "track_id": o.track_id,
            "has_snapshot": bool(o.has_snapshot),
            "created_at": iso_local(o.created_at),
        }
        for o in observations
    ]


@router.get("/{global_id}/observations/{observation_id}/photo")
def api_observation_photo(global_id: int, observation_id: int,
                          db: Session = Depends(get_db)):
    observation = db.get(GlobalObservation, observation_id)
    if observation is None or observation.global_id != global_id:
        raise HTTPException(status_code=404, detail="Observation not found")
    if not observation.snapshot:
        raise HTTPException(status_code=404, detail="Snapshot not found")
    return Response(content=observation.snapshot, media_type="image/jpeg")


@router.get("/{global_id}/trajectory")
def api_global_trajectory(global_id: int, db: Session = Depends(get_db)):
    """Траектория: последовательность камер по времени + переходы между ними."""
    _person_or_404(db, global_id)
    cameras = {c.id: c.name for c in db.query(Camera).all()}
    # только колонки камеры/времени — без embedding- и snapshot-BLOB
    rows = db.query(
        GlobalObservation.camera_id, GlobalObservation.created_at,
    ).filter(
        GlobalObservation.global_id == global_id
    ).order_by(GlobalObservation.created_at.asc()).limit(TRAJECTORY_CAP).all()

    segments: list[dict] = []
    for camera_id, created_at in rows:
        name = cameras.get(camera_id, f"#{camera_id}")
        if segments and segments[-1]["camera_id"] == camera_id:
            segments[-1]["to"] = iso_local(created_at)
            segments[-1]["observations"] += 1
        else:
            segments.append({
                "camera_id": camera_id,
                "camera_name": name,
                "from": iso_local(created_at),
                "to": iso_local(created_at),
                "observations": 1,
            })
    return {
        "global_id": global_id,
        "segments": segments,
        "chain": [s["camera_name"] for s in segments],
    }
