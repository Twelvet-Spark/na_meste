"""Trusted-network mutation guard, not application authentication."""
from urllib.parse import urlsplit

from fastapi import HTTPException, Request


def require_operator_request(request: Request):
    if request.headers.get("x-operator-request") != "1":
        raise HTTPException(403, "Требуется запрос из интерфейса оператора")
    origin = request.headers.get("origin")
    if origin is not None:
        try:
            source = urlsplit(origin)
            target = urlsplit(str(request.base_url))
            def identity(url):
                return (url.scheme.lower(), (url.hostname or "").lower(),
                        url.port if url.port is not None else (443 if url.scheme.lower() == "https" else 80))
            if (source.username is not None or source.password is not None or source.path
                    or source.query or source.fragment or identity(source) != identity(target)):
                raise ValueError("foreign origin")
        except ValueError as exc:
            raise HTTPException(403, "Запрос с другого сайта запрещён") from exc
