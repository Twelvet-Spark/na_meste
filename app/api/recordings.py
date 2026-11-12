"""Manual recording, bounded archive timelines and process-local browser playback."""
from datetime import datetime, timezone
from typing import Literal
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator

from app.api.operator_security import require_operator_request
from app.services.recording_service import RecordingManager, SessionView
from app.media.playback import timestamp

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


class PlaybackStart(BaseModel):
    model_config = ConfigDict(extra="forbid")
    camera_id: StrictInt = Field(gt=0)
    source: Literal["local", "dahua"]
    at: datetime
    request_id: UUID

    @field_validator("at", mode="before")
    @classmethod
    def explicit_timestamp(cls, value):
        return timestamp(value).replace(tzinfo=timezone.utc)


def get_playback_manager(request: Request):
    manager = getattr(request.app.state, "playbacks", None)
    if manager is None:
        raise HTTPException(503, getattr(request.app.state, "playback_error", None) or "PLAYBACK_UNAVAILABLE")
    return manager


@router.get("/timeline")
def timeline(request: Request, camera_id: int = Query(gt=0), source: Literal["local", "dahua"] = Query(),
             start: str = Query(alias="from"), end: str = Query(alias="to"),
             manager=Depends(get_playback_manager)):
    diagnostics = getattr(request.app.state, "diagnostics", None)
    try:
        result = manager.timeline(camera_id, source, start, end)
    except HTTPException as exc:
        if source == "dahua" and diagnostics is not None:
            diagnostics.archive_result(camera_id=camera_id, search=False, error_code=exc.detail)
        raise
    if source == "dahua" and diagnostics is not None:
        diagnostics.archive_result(camera_id=camera_id, search=True)
    return result


@router.post("/playbacks", status_code=202, dependencies=[Depends(require_operator_request)])
def create_playback(payload: PlaybackStart, manager=Depends(get_playback_manager)):
    return manager.create(payload.camera_id, payload.source, payload.at, payload.request_id)


@router.get("/playbacks/{playback_id}")
def get_playback(playback_id: UUID, manager=Depends(get_playback_manager)):
    return manager.get(playback_id)


@router.delete("/playbacks/{playback_id}", dependencies=[Depends(require_operator_request)])
def cancel_playback(playback_id: UUID, manager=Depends(get_playback_manager)):
    return manager.cancel(playback_id)


@router.get("/playbacks/{playback_id}/manifest.m3u8")
def playback_manifest(playback_id: UUID, manager=Depends(get_playback_manager)):
    return Response(manager.manifest(playback_id), media_type="application/vnd.apple.mpegurl",
                    headers={"Cache-Control": "no-store"})


@router.get("/playbacks/{playback_id}/segments/{sequence}.ts")
def playback_segment(playback_id: UUID, sequence: int, manager=Depends(get_playback_manager)):
    if sequence < 0:
        raise HTTPException(404, "PLAYBACK_SEGMENT_NOT_FOUND")
    return manager.segment(playback_id, sequence)
