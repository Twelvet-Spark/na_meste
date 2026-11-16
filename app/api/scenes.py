"""Independent descriptive scene geometry; entrance calibration stays separate."""
from typing import Any

from fastapi import APIRouter, Body, Depends, HTTPException, Request

from app.api.operator_security import require_operator_request
from app.services.scene_service import SceneService
from app.services.scene_assistance import SceneAssistanceService

router = APIRouter(prefix='/api/cameras', tags=['scenes'])


def _service(request: Request) -> SceneService:
    service = getattr(request.app.state, 'scene_service', None)
    if service is None:
        raise HTTPException(503, detail=dict(code='SCENE_SERVICE_UNAVAILABLE', message='Scene service is unavailable'))
    return service


@router.get('/{camera_id}/scene')
def get_scene(camera_id: int, request: Request):
    return _service(request).get(camera_id)


@router.put('/{camera_id}/scene', dependencies=[Depends(require_operator_request)])
def put_scene(camera_id: int, request: Request, payload: Any = Body(...)):
    return _service(request).save(camera_id, payload)


@router.post('/{camera_id}/scene/snapshot', dependencies=[Depends(require_operator_request)])
def freeze_scene(camera_id: int, request: Request, payload: Any = Body(None)):
    if payload is not None and (not isinstance(payload, dict) or payload):
        raise HTTPException(422, detail=dict(code='SCENE_INPUT_INVALID',
                                             message='Snapshot capture accepts no image, URL or client metadata'))
    return _service(request).snapshot(camera_id)


def _assistance(request: Request) -> SceneAssistanceService:
    service = getattr(request.app.state, 'scene_assistance', None)
    if service is None:
        raise HTTPException(503, detail=dict(code='SCENE_MODEL_UNAVAILABLE',
                                             message='Scene assistance is unavailable; manual editing remains available'))
    return service


@router.post('/{camera_id}/scene/suggestions', status_code=202,
             dependencies=[Depends(require_operator_request)])
def suggest_scene(camera_id: int, request: Request, payload: Any = Body(...)):
    return _assistance(request).submit(camera_id, payload)


@router.get('/{camera_id}/scene/suggestions/{job_id}')
def get_scene_suggestion(camera_id: int, job_id: str, request: Request):
    return _assistance(request).get(camera_id, job_id)


@router.delete('/{camera_id}/scene/suggestions/{job_id}',
               dependencies=[Depends(require_operator_request)])
def cancel_scene_suggestion(camera_id: int, job_id: str, request: Request):
    return _assistance(request).cancel(camera_id, job_id)
