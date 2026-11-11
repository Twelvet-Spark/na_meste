"""Bounded Dahua CGI transport and channel discovery, independent of live RTSP.

Only internal CGI operations are exposed. No browser-supplied path, proxy,
redirect, unverified TLS, preemptive Basic credentials, or raw device error is
returned. Search/playback is intentionally not implemented here.
"""
import re
import socket
import ssl
import threading
import time
from contextlib import contextmanager
from http.client import HTTPException as HTTPProtocolError
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import (
    HTTPBasicAuthHandler, HTTPDigestAuthHandler, HTTPPasswordMgrWithDefaultRealm,
    HTTPRedirectHandler, HTTPSHandler, ProxyHandler, Request, build_opener,
)

from app.rtsp.dahua import url_host

TIMEOUT_SECONDS = 10
MAX_RESPONSE_BYTES = 1024 * 1024
_TITLE = re.compile(r"(?:table\.)?ChannelTitle\[(\d+)\]\.Name=(.*)")


class DahuaError(Exception):
    """Safe public code only; the originating URL/headers/body never escape."""
    def __init__(self, code):
        self.code = code
        super().__init__(code)


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        fp.close()
        raise DahuaError("DEVICE_PROTOCOL_ERROR")


class DahuaClient:
    def __init__(self, *, host, http_scheme, http_port, username, password):
        self._base = f"{http_scheme}://{url_host(host)}:{http_port}"
        credentials = HTTPPasswordMgrWithDefaultRealm()
        credentials.add_password(None, self._base + "/", username, password)
        handlers = [ProxyHandler({}), _NoRedirect(), HTTPDigestAuthHandler(credentials),
                    HTTPBasicAuthHandler(credentials)]
        if http_scheme == "https":
            handlers.append(HTTPSHandler(context=ssl.create_default_context()))
        self._opener = build_opener(*handlers)
        self._cancelled = threading.Event()
        self._response = None
        self._lock = threading.Lock()

    def cancel(self):
        self._cancelled.set()
        # Interrupt a body read without waiting for HTTPResponse's buffered-read
        # lock. Connecting/authenticating requests remain bounded by their timeout.
        with self._lock:
            response = self._response
        if response is not None:
            try:
                response.fp.raw._sock.shutdown(socket.SHUT_RDWR)
            except (AttributeError, OSError):
                pass

    def _check(self, deadline):
        if self._cancelled.is_set():
            raise DahuaError("SOURCE_CHANGED")
        if time.monotonic() >= deadline:
            raise DahuaError("DEVICE_TIMEOUT")

    def _channel_config(self):
        # The operation/path is internal, not an arbitrary URL transport API.
        query = urlencode({"action": "getConfig", "name": "ChannelTitle"})
        request = Request(self._base + "/cgi-bin/configManager.cgi?" + query,
                          headers={"Accept": "text/plain", "Accept-Encoding": "identity"})
        deadline = time.monotonic() + TIMEOUT_SECONDS
        response = None
        try:
            self._check(deadline)
            response = self._opener.open(request, timeout=TIMEOUT_SECONDS)
            with self._lock:
                self._response = response
            self._check(deadline)
            if response.status != 200 or response.headers.get("Content-Encoding", "identity").lower() != "identity":
                raise DahuaError("DEVICE_PROTOCOL_ERROR")
            size = response.headers.get("Content-Length")
            if size is not None and (not size.isdecimal() or int(size) > MAX_RESPONSE_BYTES):
                raise DahuaError("DEVICE_PROTOCOL_ERROR")
            chunks, total = [], 0
            while True:
                self._check(deadline)
                chunk = response.read1(min(64 * 1024, MAX_RESPONSE_BYTES + 1 - total))
                self._check(deadline)
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_RESPONSE_BYTES:
                    raise DahuaError("DEVICE_PROTOCOL_ERROR")
                chunks.append(chunk)
            if size is not None and total != int(size):
                raise DahuaError("DEVICE_PROTOCOL_ERROR")
            return b"".join(chunks).decode("utf-8-sig", "strict")
        except HTTPError as exc:
            code = ("DEVICE_AUTH_FAILED" if exc.code in (401, 403) else
                    "ARCHIVE_UNSUPPORTED" if exc.code in (404, 405, 501) else "DEVICE_PROTOCOL_ERROR")
            exc.close()
            raise DahuaError(code) from None
        except (TimeoutError, socket.timeout):
            raise DahuaError("SOURCE_CHANGED" if self._cancelled.is_set() else "DEVICE_TIMEOUT") from None
        except URLError as exc:
            code = ("DEVICE_TLS_FAILED" if isinstance(exc.reason, ssl.SSLCertVerificationError) else
                    "DEVICE_TIMEOUT" if isinstance(exc.reason, (TimeoutError, socket.timeout)) else "DEVICE_UNREACHABLE")
            raise DahuaError("SOURCE_CHANGED" if self._cancelled.is_set() else code) from None
        except ssl.SSLCertVerificationError:
            raise DahuaError("DEVICE_TLS_FAILED") from None
        except (OSError, HTTPProtocolError, UnicodeError, ValueError):
            raise DahuaError("SOURCE_CHANGED" if self._cancelled.is_set() else "DEVICE_PROTOCOL_ERROR") from None
        finally:
            with self._lock:
                self._response = None
            if response is not None:
                response.close()

    def channel_titles(self):
        text = self._channel_config()
        channels = {}
        for line in text.splitlines():
            if not line.strip():
                continue
            match = _TITLE.fullmatch(line)
            if match is None or len(match[1]) > 2:
                raise DahuaError("DEVICE_PROTOCOL_ERROR")
            index, name = int(match[1]), match[2].strip()
            if (not 0 <= index < 64 or index in channels or not name or len(name) > 100
                    or any(ord(character) < 32 or ord(character) == 127 for character in name)):
                raise DahuaError("DEVICE_PROTOCOL_ERROR")
            channels[index] = name
        if not channels:
            raise DahuaError("DEVICE_PROTOCOL_ERROR")
        return [{"channel": index + 1, "name": channels[index]} for index in sorted(channels)]


class DahuaSetup:
    """App-local discovery lifetime; committed source publications cancel old work."""
    def __init__(self):
        self._lock = threading.Lock()
        self._current = {}
        self._clients = {}

    @staticmethod
    def _key(source):
        return source.source_revision, source.archive, source.enabled

    @contextmanager
    def _track(self, client, source):
        if source is not None:
            with self._lock:
                latest = self._current.get(source.id)
                if latest is not None and latest[1] != self._key(source):
                    client.cancel()
                self._clients[client] = source
        try:
            yield
        finally:
            with self._lock:
                self._clients.pop(client, None)

    def source_changed(self, source):
        with self._lock:
            latest = self._current.get(source.id)
            if latest is not None and source.publication_sequence < latest[0]:
                return
            self._current[source.id] = source.publication_sequence, self._key(source)
            clients = [client for client, prior in self._clients.items()
                       if prior.id == source.id and self._key(prior) != self._key(source)]
        for client in clients:
            client.cancel()

    def deleted(self, camera_id):
        with self._lock:
            self._current[camera_id] = (0, None)
            clients = [client for client, source in self._clients.items() if source.id == camera_id]
        for client in clients:
            client.cancel()

    def discover(self, payload, *, source=None):
        client = DahuaClient(host=payload.nvr_host, http_scheme=payload.http_scheme,
                             http_port=payload.http_port, username=payload.username,
                             password=payload.password or (source.password if source else ""))
        try:
            with self._track(client, source):
                channels = client.channel_titles()
                if client._cancelled.is_set():
                    raise DahuaError("SOURCE_CHANGED")
            return {"channels": channels, "capabilities": {"channel_titles": True}, "error_code": None}
        except DahuaError as exc:
            return {"channels": [], "capabilities": {"channel_titles": False}, "error_code": exc.code}
