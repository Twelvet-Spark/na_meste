"""Bounded Dahua CGI transport and channel discovery, independent of live RTSP.

Only internal CGI operations are exposed. No browser-supplied path, proxy,
redirect, unverified TLS, preemptive Basic credentials, or raw device error is
returned. Archive finder paths remain private to the server-side playback adapter.
"""
import re
import socket
import ssl
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from http.client import HTTPException as HTTPProtocolError
from urllib.error import HTTPError, URLError
from urllib.parse import quote, unquote, urlencode
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
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise DahuaError("DEVICE_TIMEOUT")
        return remaining

    def _channel_config(self):
        return self._request("/cgi-bin/configManager.cgi?" + urlencode(
            {"action": "getConfig", "name": "ChannelTitle"}))

    def _request(self, operation, *, deadline=None):
        request = Request(self._base + operation,
                          headers={"Accept": "text/plain", "Accept-Encoding": "identity"})
        deadline = min(deadline, time.monotonic() + TIMEOUT_SECONDS) if deadline is not None else time.monotonic() + TIMEOUT_SECONDS
        response = None
        try:
            remaining = self._check(deadline)
            response = self._opener.open(request, timeout=remaining)
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
                remaining = self._check(deadline)
                # read1 avoids buffered read(n)'s repeated data receives;
                # headers and earlier chunks must not restart the socket timeout.
                # The stdlib clears fp when a length-delimited body reaches EOF.
                if response.fp is not None:
                    response.fp.raw._sock.settimeout(remaining)
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


@dataclass(frozen=True)
class ArchiveInterval:
    start_utc: datetime
    end_utc: datetime
    path: str = field(repr=False)
    channel: int


def archive_path(value):
    """Only manufacturer-documented absolute media paths, never URLs or traversal."""
    decoded = unquote(value)
    if (not value.startswith("/mnt/") or not decoded.startswith("/mnt/")
            or "\\" in decoded or "%" in decoded
            or any(ord(char) < 32 or ord(char) == 127 for char in decoded)
            or any(part in ("", ".", "..") for part in decoded.split("/")[1:])):
        raise DahuaError("DEVICE_PROTOCOL_ERROR")
    return decoded


class DahuaArchiveClient(DahuaClient):
    def __init__(self, source):
        super().__init__(host=source.host, http_scheme=source.http_scheme, http_port=source.http_port,
                         username=source.username, password=source.password)
        self.source = source
        self.zone = timezone(timedelta(minutes=source.utc_offset_minutes))

    def _finder(self, action, *, deadline=None, **fields):
        query = urlencode({"action": action, **fields}, quote_via=quote)
        return self._request("/cgi-bin/mediaFileFind.cgi?" + query, deadline=deadline)

    def _page(self, text):
        fields = {}
        for line in text.splitlines():
            if not line:
                continue
            if "=" not in line:
                raise DahuaError("DEVICE_PROTOCOL_ERROR")
            key, value = line.split("=", 1)
            if key in fields:
                raise DahuaError("DEVICE_PROTOCOL_ERROR")
            fields[key] = value
        result = []
        try:
            found = fields.pop("found", "")
            if not found.isdecimal() or len(found) > 3:
                raise ValueError
            count = int(found)
            if count > 100:
                raise ValueError
            for key in fields:
                match = re.fullmatch(r"items\[(\d+)\]\.[A-Za-z][A-Za-z0-9]*", key)
                if match is None or len(match[1]) > 2 or int(match[1]) >= count:
                    raise ValueError
            for index in range(count):
                prefix = f"items[{index}]."
                channel = int(fields[prefix + "Channel"])
                if channel != self.source.channel or fields[prefix + "Type"].lower() not in ("dav", "mp4"):
                    raise ValueError
                start = datetime.strptime(fields[prefix + "StartTime"], "%Y-%m-%d %H:%M:%S").replace(
                    tzinfo=self.zone).astimezone(timezone.utc)
                end = datetime.strptime(fields[prefix + "EndTime"], "%Y-%m-%d %H:%M:%S").replace(
                    tzinfo=self.zone).astimezone(timezone.utc)
                if end <= start:
                    raise ValueError
                result.append(ArchiveInterval(start, end, archive_path(fields[prefix + "FilePath"]), channel))
        except (ValueError, KeyError, OverflowError):
            raise DahuaError("DEVICE_PROTOCOL_ERROR") from None
        return result

    def list_recordings(self, start_utc, end_utc):
        try:
            if (start_utc.utcoffset() is None or end_utc.utcoffset() is None
                    or not timedelta(0) < end_utc - start_utc <= timedelta(days=1)):
                raise ValueError
            start = start_utc.astimezone(self.zone).strftime("%Y-%m-%d %H:%M:%S")
            end = end_utc.astimezone(self.zone).strftime("%Y-%m-%d %H:%M:%S")
        except (ValueError, OverflowError, AttributeError):
            raise DahuaError("DEVICE_PROTOCOL_ERROR") from None
        overall_deadline = time.monotonic() + 60
        # The complete finder lifecycle includes two cancellation-insensitive
        # cleanup attempts: 40 s work, up to 10 s close, up to 10 s destroy.
        deadline = overall_deadline - 20
        handle = None
        try:
            text = self._finder("factory.create", deadline=deadline).strip()
            match = re.fullmatch(r"result=([0-9]{1,32})", text)
            if match is None:
                raise DahuaError("DEVICE_PROTOCOL_ERROR")
            handle = match[1]
            response = self._finder("findFile", deadline=deadline, object=handle, **{
                "condition.Channel": self.source.channel, "condition.StartTime": start,
                "condition.EndTime": end, "condition.Types[0]": "dav", "condition.Types[1]": "mp4"})
            if response.strip() != "OK":
                raise DahuaError("DEVICE_PROTOCOL_ERROR")
            intervals = {}
            for _page in range(100):
                if time.monotonic() >= deadline:
                    raise DahuaError("SEARCH_LIMIT")
                entries = self._page(self._finder("findNextFile", deadline=deadline, object=handle, count=100))
                if time.monotonic() >= deadline:
                    raise DahuaError("SEARCH_LIMIT")
                if not entries:
                    return sorted(intervals.values(), key=lambda row: (row.start_utc, row.end_utc))
                for interval in entries:
                    intervals[(interval.path, interval.start_utc, interval.end_utc)] = interval
            raise DahuaError("SEARCH_LIMIT")
        except DahuaError as exc:
            if time.monotonic() >= deadline:
                raise DahuaError("SEARCH_LIMIT") from None
            raise exc
        finally:
            if handle is not None:
                # Cancellation must not suppress either server-side cleanup operation.
                cleanup = DahuaClient(host=self.source.host, http_scheme=self.source.http_scheme,
                                      http_port=self.source.http_port, username=self.source.username,
                                      password=self.source.password)
                for action, reserved_after in (("close", 10), ("destroy", 0)):
                    try:
                        cleanup._request("/cgi-bin/mediaFileFind.cgi?" + urlencode(
                            {"action": action, "object": handle}, quote_via=quote),
                            deadline=overall_deadline - reserved_after)
                    except DahuaError:
                        pass
