"""Direct ASGI operator client; no lifespan, network, devices, or production DB."""
import asyncio
import json
from urllib.parse import urlsplit
from uuid import uuid4


class Client:
    def __init__(self, app):
        self.app = app

    async def arequest(self, method, url, payload=None, *, body=None, chunks=None,
                       headers=None, operator=True):
        parsed = urlsplit(url)
        request_headers = {"host": "test"}
        if operator:
            request_headers["x-operator-request"] = "1"
        if payload is not None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            request_headers["content-type"] = "application/json"
        if headers:
            request_headers.update({key.lower(): str(value) for key, value in headers.items()})
        messages = []
        stream = iter(chunks if chunks is not None else [body or b""])
        exhausted = False

        async def receive():
            nonlocal exhausted
            if exhausted:
                return {"type": "http.disconnect"}
            try:
                chunk = next(stream)
            except StopIteration:
                exhausted = True
                return {"type": "http.request", "body": b"", "more_body": False}
            return {"type": "http.request", "body": chunk, "more_body": True}

        async def send(message):
            messages.append(message)

        await self.app({
            "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
            "method": method, "scheme": "http", "path": parsed.path,
            "raw_path": parsed.path.encode(), "query_string": parsed.query.encode(),
            "root_path": "", "headers": [(key.encode("ascii"), value.encode("latin-1"))
                                           for key, value in request_headers.items()],
            "client": ("127.0.0.1", 1234), "server": ("test", 80),
        }, receive, send)
        start = next(item for item in messages if item["type"] == "http.response.start")
        data = b"".join(item.get("body", b"") for item in messages
                        if item["type"] == "http.response.body")
        response_headers = dict(start["headers"])
        if not data:
            data = None
        elif response_headers.get(b"content-type", b"").startswith(b"application/json"):
            data = json.loads(data)
        return start["status"], response_headers, data

    def request(self, method, url, payload=None, **kwargs):
        return asyncio.run(self.arequest(method, url, payload, **kwargs))

    def json(self, method, url, payload=None, *, status=200, **kwargs):
        actual, _headers, data = self.request(method, url, payload, **kwargs)
        assert actual == status, (method, url, actual, data)
        return data

    def multipart(self, method, url, metadata, files=(), *, status=201, **kwargs):
        body, content_type = multipart(metadata, files)
        headers = {"content-type": content_type, **kwargs.pop("headers", {})}
        return self.json(method, url, body=body, headers=headers, status=status, **kwargs)


def multipart(metadata, files=(), *, fields=()):
    """Files are (filename, bytes, MIME); metadata may be deliberately malformed."""
    boundary = "operator-test-" + uuid4().hex
    parts = []
    value = metadata if isinstance(metadata, str) else json.dumps(metadata, ensure_ascii=False)
    for name, text in [("metadata", value), *fields]:
        parts.append((f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n'
                      f'\r\n{text}\r\n').encode("utf-8"))
    for filename, content, mime in files:
        parts.extend([
            (f'--{boundary}\r\nContent-Disposition: form-data; name="files"; '
             f'filename="{filename}"\r\nContent-Type: {mime}\r\n\r\n').encode("utf-8"),
            content, b"\r\n",
        ])
    parts.append(f"--{boundary}--\r\n".encode("ascii"))
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"
