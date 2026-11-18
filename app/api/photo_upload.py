"""Bound multipart bodies before Starlette spools employee photographs."""
from fastapi import HTTPException, Request
from fastapi.routing import APIRoute
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.formparsers import MultiPartException

from app.api.operator_security import require_operator_request

MAX_PHOTO_BYTES = 15 * 1024 * 1024


class PhotoUploadRoute(APIRoute):
    def get_route_handler(self):
        handler = super().get_route_handler()

        async def bounded(request):
            single = request.method == "POST" and self.path.endswith(
                ("/employee-photo", "/learning/analyze", "/learning/feedback"))
            upload = single or (request.method == "POST" and self.path == "/api/employees") or (
                request.method == "PUT" and self.path.endswith("/faces"))
            if not upload:
                return await handler(request)
            require_operator_request(request)
            maximum = MAX_PHOTO_BYTES + 80 * 1024 if single else 226 * 1024 * 1024
            length = request.headers.get("content-length")
            if length is not None:
                try:
                    declared = int(length)
                    if declared < 0:
                        raise ValueError("negative body length")
                except ValueError as exc:
                    raise HTTPException(400, "Некорректный размер запроса") from exc
                if declared > maximum:
                    raise HTTPException(413, "Запрос с фотографиями слишком большой")
            received, exceeded = 0, False

            async def receive():
                nonlocal received, exceeded
                message = await request.receive()
                received += len(message.get("body", b""))
                if received > maximum:
                    exceeded = True
                    # Starlette closes all spooled uploads on this exception.
                    raise MultiPartException("Photo request exceeds body limit")
                return message

            try:
                return await handler(Request(request.scope, receive=receive))
            except StarletteHTTPException as exc:
                if exceeded:
                    raise HTTPException(413, "Запрос с фотографиями слишком большой") from exc
                raise
        return bounded
