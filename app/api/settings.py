"""Operator settings HTTP boundary: no validation input or secret reflection."""
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.routing import APIRoute
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, StrictStr

from app.api.operator_security import require_operator_request


class SafeSettingsRoute(APIRoute):
    def get_route_handler(self):
        handler = super().get_route_handler()

        async def guarded(request):
            try:
                response = await handler(request)
            except RequestValidationError:
                response = JSONResponse({"detail": "OPERATOR_SETTINGS_INVALID_REQUEST"}, status_code=422)
            response.headers["Cache-Control"] = "no-store"
            return response

        return guarded


router = APIRouter(prefix="/api/settings", tags=["settings"], route_class=SafeSettingsRoute)


class SettingsUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    expected_version: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    values: dict = Field(default_factory=dict)
    reset_keys: list[StrictStr] = Field(default_factory=list)


def get_service(request: Request):
    service = getattr(request.app.state, "settings_service", None)
    if service is None:
        raise HTTPException(503, "OPERATOR_SETTINGS_UNAVAILABLE")
    return service


@router.get("")
def get_settings(service=Depends(get_service)):
    return service.snapshot()


@router.put("", dependencies=[Depends(require_operator_request)])
def update_settings(payload: SettingsUpdate, service=Depends(get_service)):
    return service.save(payload.expected_version, payload.values, payload.reset_keys)
