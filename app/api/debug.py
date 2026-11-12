"""Read-only sanitized diagnostics; no file viewer, console or reset operations."""
from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute


class DiagnosticRoute(APIRoute):
    def get_route_handler(self):
        handler = super().get_route_handler()

        async def guarded(request):
            try:
                response = await handler(request)
            except RequestValidationError:
                response = JSONResponse({"detail": "DIAGNOSTICS_INVALID_QUERY"}, status_code=422)
            response.headers["Cache-Control"] = "no-store"
            return response

        return guarded


router = APIRouter(prefix="/api/debug", tags=["diagnostics"], route_class=DiagnosticRoute)


def service(request: Request):
    result = getattr(request.app.state, "diagnostics", None)
    if result is None:
        raise HTTPException(503, "DIAGNOSTICS_UNAVAILABLE")
    return result


@router.get("/status")
def status(diagnostics=Depends(service)):
    return diagnostics.status()


@router.get("/events")
def events(after: int = Query(default=0, ge=0), limit: int = Query(default=100, ge=1, le=100),
           diagnostics=Depends(service)):
    return diagnostics.ring.events(after=after, limit=limit)


@router.get("/export")
def export(response: Response, diagnostics=Depends(service)):
    response.headers["Content-Disposition"] = 'attachment; filename="operator-support.json"'
    return diagnostics.export()
