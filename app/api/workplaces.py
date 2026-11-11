"""Optional workplace labels; archiving never erases assigned history."""
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.operator_security import require_operator_request
from app.database.database import get_db
from app.services import workplace_service as service

router = APIRouter(prefix="/api/workplaces", tags=["workplaces"])


@router.get("", response_model=list[service.WorkplaceOut])
def list_workplaces(db: Session = Depends(get_db)):
    return service.list_workplaces(db)


@router.post("", status_code=201, response_model=service.WorkplaceOut,
             dependencies=[Depends(require_operator_request)])
def create_workplace(payload: service.WorkplaceCreate, db: Session = Depends(get_db)):
    return service.save_workplace(db, payload)


@router.put("/{workplace_id}", response_model=service.WorkplaceOut,
            dependencies=[Depends(require_operator_request)])
def update_workplace(workplace_id: int, payload: service.WorkplaceUpdate, db: Session = Depends(get_db)):
    return service.save_workplace(db, payload, workplace_id)
