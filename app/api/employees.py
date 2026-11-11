"""Atomic multipart employee onboarding and final-photo-set replacement."""
import hashlib
import json
import threading

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import Response
from fastapi.routing import APIRoute
from pydantic import ValidationError
from sqlalchemy.orm import Session
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import UploadFile
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.formparsers import MultiPartException

from app.api.operator_security import require_operator_request
from app.database.database import get_db
from app.services import employee_service as service

MAX_PHOTO_BYTES = 15 * 1024 * 1024
MAX_BODY_BYTES = 226 * 1024 * 1024


class EmployeeRoute(APIRoute):
    """Bound receive before multipart spooling; parser owns upload cleanup."""
    def get_route_handler(self):
        handler = super().get_route_handler()

        async def bounded(request):
            upload = ((request.method == "POST" and self.path == "/api/employees")
                      or (request.method == "PUT" and self.path.endswith("/faces")))
            if not upload:
                return await handler(request)
            require_operator_request(request)
            length = request.headers.get("content-length")
            if length is not None:
                try:
                    declared = int(length)
                    if declared < 0:
                        raise ValueError
                except ValueError as exc:
                    raise HTTPException(400, "Некорректный размер запроса") from exc
                if declared > MAX_BODY_BYTES:
                    raise HTTPException(413, "Запрос превышает 226 МиБ")
            received, exceeded = 0, False

            async def receive():
                nonlocal received, exceeded
                message = await request.receive()
                received += len(message.get("body", b""))
                if received > MAX_BODY_BYTES:
                    exceeded = True
                    # Starlette closes all spooled uploads on MultiPartException.
                    raise MultiPartException("Request exceeds 226 MiB")
                return message

            limited = Request(request.scope, receive=receive)
            try:
                return await handler(limited)
            except StarletteHTTPException as exc:
                if exceeded:
                    raise HTTPException(413, "Запрос превышает 226 МиБ") from exc
                raise
        return bounded


router = APIRouter(prefix="/api/employees", tags=["employees"], route_class=EmployeeRoute)


def _employee_or_404(db, employee_id):
    employee = service.get_employee(db, employee_id)
    if employee is None:
        raise HTTPException(404, "Сотрудник не найден")
    return employee


def _refresh(request, db):
    store = getattr(request.app.state, "embedding_store", None)
    if store is not None:
        db.rollback()
        store.refresh(db)


def _enrollment_lock(request):
    # Called on the ASGI loop before offloading; one lock per application/model.
    lock = getattr(request.app.state, "enrollment_lock", None)
    if lock is None:
        lock = threading.Lock()
        request.app.state.enrollment_lock = lock
    return lock


async def _read_form(request, model, *, creating):
    if not request.headers.get("content-type", "").lower().startswith("multipart/form-data"):
        raise HTTPException(415, "Отправьте metadata и фотографии как multipart/form-data")
    async with request.form(max_files=15, max_fields=1, max_part_size=64 * 1024) as form:
        values = form.getlist("metadata")
        if len(values) != 1 or not isinstance(values[0], str) or len(values[0].encode("utf-8")) > 64 * 1024:
            raise HTTPException(422, "metadata должен быть JSON размером не более 64 КиБ")
        if any(key not in {"metadata", "files"} for key in form):
            raise HTTPException(422, "Неизвестное поле формы")
        try:
            metadata = model.model_validate_json(values[0])
        except ValidationError as exc:
            # Never echo raw multipart bytes or untrusted photos in validation JSON.
            errors = [{"loc": list(item["loc"]), "msg": item["msg"], "type": item["type"]}
                      for item in exc.errors(include_input=False, include_context=False)]
            raise HTTPException(422, errors) from exc
        uploads = form.getlist("files")
        if any(not isinstance(item, UploadFile) for item in uploads):
            raise HTTPException(422, "files должны быть файлами")
        count = len(uploads) if creating else len(uploads) + len(metadata.keep_face_ids)
        if not 1 <= count <= 15:
            raise HTTPException(422, "Итоговый набор должен содержать 1–15 фотографий")
        contents = []
        for upload in uploads:
            contents.append((upload.filename or "photo", await upload.read(MAX_PHOTO_BYTES + 1)))
        return metadata, contents


def _prepare(contents, request, lock):
    if not contents:
        return []
    engine = getattr(request.app.state, "face_engine", None)
    if engine is None:
        raise HTTPException(503, "AI-модели недоступны — загрузка фотографий невозможна")
    prepared, errors = [], []
    for index, (filename, image) in enumerate(contents):
        with lock:
            result = service.prepare_face(image, engine, filename)
        if isinstance(result, service.FaceEnrollResult):
            errors.append({"index": index, "filename": filename, "error": result.error})
        else:
            prepared.append(result)
    if errors:
        raise service.photo_failure(errors)
    return prepared


def _create(request, db, metadata, contents, lock):
    canonical = metadata.model_dump(mode="json", exclude={"request_id"})
    payload = {"metadata": canonical, "files": [hashlib.sha256(content).hexdigest() for _name, content in contents]}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False,
                                       separators=(",", ":")).encode("utf-8")).hexdigest()
    replay = service.committed_request(db, metadata.request_id, digest)
    db.rollback()
    if replay is not None:
        return replay
    prepared = _prepare(contents, request, lock)
    service.create_employee_with_faces(db, metadata, prepared, digest)
    # Always return the transaction's frozen result, including a racing replay.
    result = service.committed_request(db, metadata.request_id, digest)
    _refresh(request, db)
    return result


@router.get("")
def api_list_employees(db: Session = Depends(get_db)):
    return [service.employee_out(db, employee) for employee, _count in service.list_employees(db)]


@router.post("", status_code=201)
async def api_create_employee(request: Request, db: Session = Depends(get_db)):
    metadata, contents = await _read_form(request, service.EmployeeCreate, creating=True)
    return await run_in_threadpool(_create, request, db, metadata, contents, _enrollment_lock(request))


@router.get("/{employee_id}")
def api_get_employee(employee_id: int, faces: bool = False, db: Session = Depends(get_db)):
    return service.employee_out(db, _employee_or_404(db, employee_id))


@router.put("/{employee_id}", dependencies=[Depends(require_operator_request)])
def api_update_employee(employee_id: int, payload: service.EmployeeUpdate, request: Request,
                        db: Session = Depends(get_db)):
    employee = service.update_employee(db, _employee_or_404(db, employee_id), payload)
    result = service.employee_out(db, employee)
    _refresh(request, db)
    return result


@router.delete("/{employee_id}", status_code=204, dependencies=[Depends(require_operator_request)])
def api_delete_employee(employee_id: int, request: Request, db: Session = Depends(get_db)):
    service.delete_employee(db, _employee_or_404(db, employee_id))
    _refresh(request, db)
    return None


def _replace(request, db, employee_id, metadata, contents, lock):
    prepared = _prepare(contents, request, lock)
    employee = service.replace_employee_faces(db, employee_id, metadata.expected_revision,
                                              metadata.keep_face_ids, prepared)
    result = service.employee_out(db, employee)
    _refresh(request, db)
    return result


@router.put("/{employee_id}/faces")
async def api_replace_faces(employee_id: int, request: Request, db: Session = Depends(get_db)):
    metadata, contents = await _read_form(request, service.FaceSetUpdate, creating=False)
    return await run_in_threadpool(_replace, request, db, employee_id, metadata, contents, _enrollment_lock(request))


@router.get("/{employee_id}/faces/{face_id}/thumbnail")
def api_face_thumbnail(employee_id: int, face_id: int, db: Session = Depends(get_db)):
    face = service.get_face(db, employee_id, face_id)
    if face is None or not face.thumbnail:
        raise HTTPException(404, "Миниатюра не найдена")
    return Response(content=face.thumbnail, media_type="image/jpeg")
