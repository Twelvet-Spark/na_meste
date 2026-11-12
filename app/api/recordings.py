"""Manual local recording API. No automatic/scheduled recording or raw paths."""
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from pydantic import BaseModel, ConfigDict, Field, StrictInt

from app.api.operator_security import require_operator_request
from app.services.recording_service import RecordingManager, SessionView

router = APIRouter(prefix="/api/recordings", tags=["recordings"])


class RecordingStart(BaseModel):
    model_config = ConfigDict(extra="forbid")
    camera_id: StrictInt = Field(gt=0)
    request_id: UUID
    duration_seconds: StrictInt = Field(default=86400, ge=60, le=86400)


def get_manager(request: Request) -> RecordingManager:
    manager = getattr(request.app.state, "recordings", None)
    if manager is None:
        raise HTTPException(503, getattr(request.app.state, "recording_error", "MEDIA_STORAGE_UNAVAILABLE"))
    return manager


@router.post("/sessions", response_model=SessionView, status_code=201,
             dependencies=[Depends(require_operator_request)])
def start_session(payload: RecordingStart, response: Response, manager=Depends(get_manager)):
    result = manager.start(payload.camera_id, payload.request_id, payload.duration_seconds)
    if result._replayed:
        response.status_code = 200
    return result


@router.get("/sessions", response_model=list[SessionView])
def list_sessions(camera_id: int | None = Query(default=None, gt=0),
                  limit: int = Query(default=50, ge=1, le=200), offset: int = Query(default=0, ge=0),
                  manager=Depends(get_manager)):
    return manager.list_sessions(camera_id=camera_id, limit=limit, offset=offset)


@router.get("/status")
def status(manager=Depends(get_manager)):
    return manager.snapshot()


@router.get("/sessions/{session_id}", response_model=SessionView)
def get_session(session_id: UUID, manager=Depends(get_manager)):
    return manager.get(session_id)


@router.post("/sessions/{session_id}/stop", response_model=SessionView,
             dependencies=[Depends(require_operator_request)])
def stop_session(session_id: UUID, manager=Depends(get_manager)):
    return manager.stop(session_id)
