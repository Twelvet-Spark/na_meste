"""Optional workplace labels, independent of attendance authority."""
from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict, StrictBool, field_validator, model_validator
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session

from app.database.database import immediate_session
from app.database.models import EmployeeWorkplace, Workplace


def normalize_name(value):
    if not isinstance(value, str):
        raise ValueError("Укажите название")
    value = " ".join(value.split())
    if not 1 <= len(value) <= 150:
        raise ValueError("Название должно содержать от 1 до 150 символов")
    return value


class WorkplaceCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    _name = field_validator("name", mode="before")(normalize_name)


class WorkplaceUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str | None = None
    active: StrictBool | None = None

    @field_validator("name", mode="before")
    @classmethod
    def name_value(cls, value):
        return normalize_name(value)

    @model_validator(mode="after")
    def not_null(self):
        if "active" in self.model_fields_set and self.active is None:
            raise ValueError("active не может быть null")
        return self


class WorkplaceOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    name: str
    active: bool


def list_workplaces(db: Session):
    return db.query(Workplace).order_by(Workplace.name, Workplace.id).all()


def save_workplace(db: Session, data, workplace_id=None):
    bind = db.get_bind().engine
    db.rollback()
    try:
        with immediate_session(bind) as writer:
            if workplace_id is None:
                row = Workplace(name=data.name, name_key=data.name.casefold())
                writer.add(row)
            else:
                row = writer.get(Workplace, workplace_id)
                if row is None:
                    raise HTTPException(404, "Рабочее место не найдено")
                values = data.model_dump(exclude_unset=True)
                if "name" in values:
                    row.name, row.name_key = data.name, data.name.casefold()
                if "active" in values:
                    row.active = data.active
            writer.flush()
        db.expire_all()
        return row
    except IntegrityError as exc:
        raise HTTPException(409, "Рабочее место с таким названием уже существует") from exc
    except SQLAlchemyError as exc:
        raise HTTPException(503, "База данных недоступна") from exc


def assignment_ids(db, employee_id):
    return {row[0] for row in db.query(EmployeeWorkplace.workplace_id).filter_by(employee_id=employee_id)}


def validate_assignments(db, ids, retained=()):
    rows = db.query(Workplace).filter(Workplace.id.in_(ids)).all() if ids else []
    if len(rows) != len(ids) or any(not row.active and row.id not in retained for row in rows):
        raise HTTPException(422, "Выберите существующие активные рабочие места; архивные можно только сохранить или снять")


def replace_assignments(db, employee_id, ids):
    current = assignment_ids(db, employee_id)
    validate_assignments(db, ids, current)
    removed = current - set(ids)
    if removed:
        db.query(EmployeeWorkplace).filter(EmployeeWorkplace.employee_id == employee_id,
                                            EmployeeWorkplace.workplace_id.in_(removed)).delete(synchronize_session=False)
    db.add_all(EmployeeWorkplace(employee_id=employee_id, workplace_id=value) for value in set(ids) - current)
