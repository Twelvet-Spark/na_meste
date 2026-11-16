"""Local-administrator setup transfer: guard before body parsing, no secret echoes."""
import asyncio
import ipaddress
import json
import re
from datetime import datetime, timezone
from urllib.parse import urlsplit
from uuid import UUID

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from fastapi.routing import APIRoute
from sqlalchemy.orm import Session
from starlette.datastructures import UploadFile
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.formparsers import MultiPartException

from app.api.operator_security import require_operator_request
from app.setup_transfer import MAX_BODY_BYTES, MAX_FILE_BYTES, TransferError


def require_local_transfer(request: Request) -> None:
    try:
        if request.client is None or not ipaddress.ip_address(request.client.host).is_loopback:
            raise ValueError
        if any(name.lower() in {"forwarded", "x-real-ip"} or name.lower().startswith("x-forwarded-") for name in request.headers):
            raise ValueError
        hosts = request.headers.getlist("host")
        if len(hosts) != 1:
            raise ValueError
        host = hosts[0]
        # Exact authority syntax: no userinfo, path, DNS trailing dot or rebinding host.
        if not re.fullmatch(r"(?:localhost|127\.0\.0\.1|\[::1\])(?::[0-9]{1,5})?", host, re.I):
            raise ValueError
        port = urlsplit("http://" + host).port
        if port is not None and not 1 <= port <= 65535:
            raise ValueError
    except ValueError:
        raise TransferError("TRANSFER_LOCAL_ONLY", 403) from None
    try:
        require_operator_request(request)
    except HTTPException:
        raise TransferError("TRANSFER_OPERATOR_REQUIRED", 403) from None


async def _thread(function, *args, **kwargs):
    """Cancellation cannot outlive the operation lock or a plaintext-owning worker."""
    task = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        # Repeated cancellation must not release the transfer slot early either.
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                continue
            except Exception:
                break
        if not task.cancelled():
            task.exception()  # Retrieve a completed worker exception without logging inputs.
        raise


class TransferRoute(APIRoute):
    def get_route_handler(self):
        handler = super().get_route_handler()

        async def guarded(request):
            lock = None
            try:
                require_local_transfer(request)
                service = getattr(request.app.state, "setup_transfer", None)
                if service is None:
                    raise TransferError("TRANSFER_UNAVAILABLE", 503)
                lock = service.operation_lock
                if not lock.acquire(blocking=False):
                    lock = None
                    raise TransferError("TRANSFER_BUSY", 429)
                maximum = 8192 if self.path.endswith("/export") else MAX_BODY_BYTES
                length = request.headers.get("content-length")
                if length is not None:
                    if not re.fullmatch(r"[0-9]{1,12}", length):
                        raise TransferError("TRANSFER_INVALID")
                    if int(length) > maximum:
                        raise TransferError("TRANSFER_TOO_LARGE", 413)
                received = 0
                exceeded = False

                async def receive():
                    nonlocal received, exceeded
                    message = await request.receive()
                    received += len(message.get("body", b""))
                    if received > maximum:
                        exceeded = True
                        # This exception makes Starlette close partially spooled uploads.
                        raise MultiPartException("TRANSFER_TOO_LARGE")
                    return message

                limited = Request(request.scope, receive=receive)
                try:
                    response = await handler(limited)
                except (MultiPartException, StarletteHTTPException):
                    raise TransferError("TRANSFER_TOO_LARGE" if exceeded else "TRANSFER_INVALID", 413 if exceeded else 422) from None
            except TransferError as exc:
                response = JSONResponse({"detail": exc.code}, status_code=exc.status_code)
            except Exception:
                response = JSONResponse({"detail": "TRANSFER_UNAVAILABLE"}, status_code=503)
            finally:
                if lock is not None:
                    lock.release()
            response.headers["Cache-Control"] = "no-store"
            response.headers["X-Content-Type-Options"] = "nosniff"
            return response
        return guarded


router = APIRouter(prefix="/api/setup-transfer", tags=["setup-transfer"], route_class=TransferRoute)


def _service(request):
    return request.app.state.setup_transfer


@router.get("/status")
async def status(request: Request):
    return await _thread(_service(request).status)


@router.post("/export")
async def export(request: Request):
    if request.headers.get("content-type", "").split(";")[0].strip().lower() != "application/json":
        raise TransferError("TRANSFER_CONTENT_TYPE", 415)
    try:
        payload = json.loads(await request.body())
        if not isinstance(payload, dict) or set(payload) != {"password"} or not isinstance(payload["password"], str):
            raise ValueError
    except (ValueError, UnicodeError):
        raise TransferError("TRANSFER_INVALID") from None
    blob = await _thread(_service(request).export, payload["password"])
    filename = datetime.now(timezone.utc).strftime("setup-%Y%m%d-%H%M%S.nmsetup")
    return Response(blob, media_type="application/octet-stream",
                    headers={"Content-Disposition": f'attachment; filename="{filename}"'})


async def _upload(request, *, importing):
    if request.headers.get("content-type", "").split(";")[0].strip().lower() != "multipart/form-data":
        raise TransferError("TRANSFER_CONTENT_TYPE", 415)
    required = {"file", "password", "request_id", "expected_sha256"} if importing else {"file", "password"}
    async with request.form(max_files=1, max_fields=3 if importing else 1, max_part_size=4096) as form:
        if set(form) != required or any(len(form.getlist(key)) != 1 for key in required):
            raise TransferError("TRANSFER_INVALID")
        upload, password = form["file"], form["password"]
        if not isinstance(upload, UploadFile) or not isinstance(password, str) or len(password.encode("utf-8")) > 1024:
            raise TransferError("TRANSFER_INVALID")
        if upload.size is not None and upload.size > MAX_FILE_BYTES:
            raise TransferError("TRANSFER_TOO_LARGE", 413)
        blob = await upload.read(MAX_FILE_BYTES + 1)
        if len(blob) > MAX_FILE_BYTES:
            raise TransferError("TRANSFER_TOO_LARGE", 413)
        values = {}
        if importing:
            try:
                if not isinstance(form["request_id"], str) or len(form["request_id"]) != 36:
                    raise ValueError
                values["request_id"] = UUID(form["request_id"])
                digest = form["expected_sha256"]
                if not isinstance(digest, str) or not re.fullmatch("[0-9a-f]{64}", digest):
                    raise ValueError
                values["expected_sha256"] = digest
            except (ValueError, TypeError, AttributeError):
                raise TransferError("TRANSFER_INVALID") from None
        return blob, password, values


@router.post("/preview")
async def preview(request: Request):
    blob, password, _ = await _upload(request, importing=False)
    return await _thread(_service(request).preview, blob, password)


def _import_and_publish(request, blob, password, values):
    service = _service(request)
    result, replay = service.import_setup(blob, password, **values)
    store = getattr(request.app.state, "embedding_store", None)
    if store is not None:
        try:
            with Session(service.engine) as db:
                store.refresh(db)
        except Exception:
            result = {**result, "warnings": [*result["warnings"], "TRANSFER_RESTART_FOR_GALLERY"]}
    return result, replay


@router.post("/import")
async def import_setup(request: Request):
    blob, password, values = await _upload(request, importing=True)
    result, replay = await _thread(_import_and_publish, request, blob, password, values)
    return JSONResponse(result, status_code=200 if replay else 201)
