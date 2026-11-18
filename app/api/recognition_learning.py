"""Operator HTTP surface for person-appearance learning; models never gate history or undo."""
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import UploadFile
from starlette.formparsers import MultiPartException, MultiPartParser

from app.api.operator_security import require_operator_request
from app.api.photo_upload import MAX_PHOTO_BYTES, PhotoUploadRoute
from app.services.recognition_learning_service import (
    LearningDecisionInput, LearningEvidenceInput, LearningRevokeInput)

router = APIRouter(prefix="/api", tags=["recognition-learning"], route_class=PhotoUploadRoute)

_METADATA_BYTES = 64 * 1024
_NO_STORE = {"Cache-Control": "no-store"}


def _service(request: Request):
    service = getattr(request.app.state, "recognition_learning_service", None)
    if service is None:
        raise HTTPException(503, {"code": "LEARNING_SERVICE_UNAVAILABLE"})
    return service


class _CompleteMultipart(MultiPartParser):
    """Reject truncated forms, including unfinished files Starlette cannot return in FormData."""
    _complete = False

    def on_end(self) -> None:
        self._complete = True

    async def parse(self):
        form = await super().parse()
        if not self._complete:
            for spool in self._files_to_close_on_error:
                spool.close()
            raise MultiPartException("Incomplete multipart form")
        return form


async def _multipart(request: Request, model):
    """Read exactly one metadata field and at most one bounded photo into frozen bytes."""
    media_type = request.headers.get("content-type", "").partition(";")[0].strip().lower()
    if media_type != "multipart/form-data":
        raise HTTPException(415, "Отправьте metadata и фотографию как multipart/form-data")
    try:
        form = await _CompleteMultipart(request.headers, request.stream(), max_files=1, max_fields=1,
                                        max_part_size=_METADATA_BYTES).parse()
    except MultiPartException as exc:
        raise HTTPException(422, "Некорректная форма фотографии") from exc
    try:
        values = form.getlist("metadata")
        if (len(values) != 1 or not isinstance(values[0], str)
                or len(values[0].encode()) > _METADATA_BYTES
                or any(key not in {"metadata", "photo"} for key in form)):
            raise HTTPException(422, "Ожидаются metadata и не более одного photo")
        try:
            payload = model.model_validate_json(values[0])
        except ValidationError as exc:
            errors = [{"loc": list(item["loc"]), "msg": item["msg"], "type": item["type"]}
                      for item in exc.errors(include_input=False, include_context=False)]
            raise HTTPException(422, errors) from exc
        files = form.getlist("photo")
        if len(files) != (1 if payload.source == "UPLOAD" else 0) or any(
                not isinstance(item, UploadFile) for item in files):
            raise HTTPException(422, "Загрузите ровно один файл или выберите сохранённый снимок")
        if not files:
            return payload, None
        data = await files[0].read(MAX_PHOTO_BYTES + 1)
        if len(data) > MAX_PHOTO_BYTES:
            raise HTTPException(413, "Фотография слишком большая")
        return payload, (files[0].filename or "photo", data)
    finally:
        # At most one spool. Synchronous close cannot be interrupted by cancellation.
        for _, item in form.multi_items():
            if isinstance(item, UploadFile):
                item.file.close()


@router.post("/global-persons/visits/{visit_id}/learning/analyze",
             dependencies=[Depends(require_operator_request)])
async def api_analyze_learning(visit_id: UUID, request: Request):
    service = _service(request)
    payload, upload = await _multipart(request, LearningEvidenceInput)
    return await run_in_threadpool(service.analyze, str(visit_id), payload, upload)


@router.post("/global-persons/visits/{visit_id}/learning/feedback",
             dependencies=[Depends(require_operator_request)])
async def api_decide_learning(visit_id: UUID, request: Request):
    service = _service(request)
    payload, upload = await _multipart(request, LearningDecisionInput)
    return await run_in_threadpool(service.decide, str(visit_id), payload, upload)


@router.get("/global-persons/visits/{visit_id}/learning/history")
async def api_visit_learning_history(
    visit_id: UUID, request: Request,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    cursor: Annotated[str | None, Query(max_length=512)] = None,
):
    return await run_in_threadpool(lambda: _service(request).history(
        visit_id=str(visit_id), cursor=cursor, limit=limit))


@router.get("/recognition-learning/examples")
async def api_employee_learning_examples(
    request: Request, employee_id: Annotated[int, Query(gt=0)],
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    cursor: Annotated[str | None, Query(max_length=512)] = None,
    feedback_id: UUID | None = None,
):
    return await run_in_threadpool(lambda: _service(request).history(
        employee_id=employee_id, cursor=cursor, limit=limit,
        feedback_id=str(feedback_id) if feedback_id is not None else None))


@router.get("/recognition-learning/{feedback_id}/photo")
async def api_learning_photo(feedback_id: str, request: Request):
    service = _service(request)
    try:
        image, digest = await run_in_threadpool(service.photo, feedback_id)
    except HTTPException as exc:
        raise HTTPException(exc.status_code, exc.detail, headers=_NO_STORE) from exc
    return Response(image, media_type="image/jpeg",
                    headers={**_NO_STORE, "ETag": f'"{digest}"'})


@router.post("/recognition-learning/{feedback_id}/revoke",
             dependencies=[Depends(require_operator_request)])
async def api_revoke_learning(feedback_id: str, payload: LearningRevokeInput, request: Request):
    return await run_in_threadpool(_service(request).revoke, feedback_id, payload)
