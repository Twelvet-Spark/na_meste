"""Dahua setup/CGI and camera-v2 regressions; controller owns RED/GREEN.

Run: .venv/Scripts/python.exe tests/test_dahua_archive.py
Only loopback HTTP fixtures and temporary SQLite are used. No real device,
application lifespan, inference model, production database or media child starts.
Task 5 will extend this module with finder/search/playback protocol coverage.
"""
import hashlib
import json
import socket
import ssl
import sys
import threading
import time
import traceback
from contextlib import contextmanager
from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from urllib.error import URLError
from urllib.parse import parse_qs, urlsplit
from urllib.request import parse_http_list, parse_keqv_list
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi import FastAPI
from sqlalchemy import event, inspect
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.orm import Session

from app.api import cameras
from app.database.database import get_db, immediate_session
from app.database.models import (
    AttendanceCamera, AttendanceCommission, AttendanceEvent, AttendanceRuntime, Camera,
)
from app.rtsp.dahua import build_rtsp_url, mask_url
from app.services import camera_service
from operator_test_support import Client
from test_operator_storage import database, dump, schema, STAMP

TESTS = []
PASSWORD = "Dahua-secret:@/#%"
CGI_PATH = "/cgi-bin/configManager.cgi?action=getConfig&name=ChannelTitle"


def test(fn):
    TESTS.append(fn)
    return fn


class Publications:
    """Replace only device reader startup; verify the actual committed DB boundary."""
    ffprobe = "fixture-ffprobe"

    def __init__(self, engine):
        self.engine = engine
        self.started = []
        self.synced = []
        self.stopped = []
        self.committed_counts = []

    def status(self, camera_id):
        return {"status": "OFFLINE", "fps": 0}

    def start(self, source):
        with Session(self.engine) as db:
            camera = db.get(Camera, source.id)
            calibration = db.get(AttendanceCamera, source.id)
            assert camera is not None and calibration is not None, "Readers must start only after commit"
            assert (camera.password, camera.channel, calibration.source_revision) == (
                source.password, source.channel, source.source_revision)
            self.committed_counts.append(db.query(Camera).count())
        self.started.append(source)

    def sync(self, source, changed_fields):
        self.synced.append((source, frozenset(changed_fields)))

    def stop(self, deletion):
        self.stopped.append(deletion.id)


@contextmanager
def api():
    with database() as (engine, root):
        schema.initialize_attendance_schema(engine)
        app = FastAPI()
        app.include_router(cameras.router)
        manager = Publications(engine)
        app.state.manager = manager
        app.state.detection_state = None
        app.state.ai_worker = None

        def isolated_db():
            with Session(engine) as db:
                yield db

        app.dependency_overrides[get_db] = isolated_db
        yield Client(app), engine, manager, root


def connection(**changes):
    value = dict(nvr_host="nvr.example", rtsp_port=554, username="operator", password=PASSWORD,
                 channel=1, stream_type="main")
    value.update(changes)
    return value


def archive(**changes):
    value = dict(http_scheme="http", http_port=80, utc_offset_minutes=180)
    value.update(changes)
    return value


def create(client, **changes):
    value = dict(name="Вход", enabled=True, **connection())
    value.update(changes)
    return client.json("POST", "/api/cameras", value, status=201)


def batch(**changes):
    value = connection()
    value.pop("channel")
    value.pop("stream_type")
    value.update(request_id=str(uuid4()), archive=archive(), channels=[
        {"channel": 1, "name": "Вход", "stream_type": "main", "enabled": True},
        {"channel": 2, "name": "Склад", "stream_type": "sub", "enabled": True},
    ])
    value.update(changes)
    return value


def assert_private(value):
    text = json.dumps(value, ensure_ascii=False, default=str)
    assert PASSWORD not in text and "rtsp://" not in text
    if isinstance(value, dict):
        assert "password" not in value
        for nested in value.values():
            if isinstance(nested, (dict, list)):
                assert_private(nested)
    elif isinstance(value, list):
        for nested in value:
            assert_private(nested)


@contextmanager
def device(*, authentication="digest", mode="titles", body=None, redirect=None, finder=None):
    """Local CGI peer validates the actual RFC digest response, not just its prefix."""
    state = SimpleNamespace(requests=[], authorized=[], mode=mode, authentication=authentication,
                            body=body, redirect=redirect, receiving=threading.Event(), release=threading.Event())
    realm, nonce = "Dahua fixture", "fixed-fixture-nonce"

    def md5(value):
        return hashlib.md5(value.encode("utf-8")).hexdigest()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def reply(self, status, content=b"", headers=()):
            self.send_response(status)
            for name, value in headers:
                self.send_header(name, value)
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            try:
                self.wfile.write(content)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass

        def do_GET(self):
            authorization = self.headers.get("Authorization", "")
            state.requests.append((self.path, authorization))
            valid = state.authentication == "none"
            if authorization.startswith("Digest "):
                fields = parse_keqv_list(parse_http_list(authorization[7:]))
                ha1 = md5(f"operator:{realm}:{PASSWORD}")
                ha2 = md5(f"GET:{self.path}")
                expected = md5(f"{ha1}:{nonce}:{fields.get('nc')}:{fields.get('cnonce')}:auth:{ha2}")
                valid = (fields.get("username") == "operator" and fields.get("realm") == realm
                         and fields.get("nonce") == nonce and fields.get("uri") == self.path
                         and fields.get("qop") == "auth" and fields.get("response") == expected)
            elif authorization.startswith("Basic ") and state.authentication == "basic":
                import base64
                valid = authorization[6:] == base64.b64encode(f"operator:{PASSWORD}".encode()).decode()
            if not valid:
                challenge = ('Basic realm="Dahua fixture"' if state.authentication == "basic" else
                             f'Digest realm="{realm}", nonce="{nonce}", algorithm=MD5, qop="auth"')
                self.reply(401, b"unauthorized", [("WWW-Authenticate", challenge)])
                return
            state.authorized.append(self.path)
            if finder is not None and urlsplit(self.path).path == "/cgi-bin/mediaFileFind.cgi":
                status, content = finder(parse_qs(urlsplit(self.path).query))
                self.reply(status, content)
            elif self.path != CGI_PATH:
                self.reply(404)
            elif state.mode == "redirect":
                self.reply(302, headers=[("Location", state.redirect)])
            elif state.mode == "unsupported":
                self.reply(404, b"unsupported")
            elif state.mode == "timeout":
                time.sleep(.25)
                self.reply(200, b"table.ChannelTitle[0].Name=Late\r\n")
            elif state.mode == "hold":
                state.receiving.set()
                if state.release.wait(5):
                    self.reply(200, b"table.ChannelTitle[0].Name=Held camera\r\n")
                else:
                    self.reply(504)
            else:
                content = state.body if state.body is not None else (
                    "table.ChannelTitle[0].Name=Вход\r\n"
                    "table.ChannelTitle[1].Name=Склад = 2\r\n"
                    "table.ChannelTitle[63].Name=Последний\r\n").encode("utf-8")
                self.reply(200, content)

    class Server(ThreadingHTTPServer):
        def finish_request(self, request, client_address):
            if state.mode == "tls_failure":
                request.recv(8192)
                request.sendall(b"HTTP/1.1 400 Bad Request\r\nContent-Length: 0\r\n\r\n")
                return
            super().finish_request(request, client_address)

    server = Server(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    state.port = server.server_address[1]
    try:
        yield state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(2)


def discover(client, peer, **changes):
    value = connection(nvr_host="127.0.0.1")
    value.update(http_scheme="http", http_port=peer.port)
    value.update(changes)
    return client.json("POST", "/api/cameras/discover", value)


def assert_discovery_failure(value, code):
    assert value == {"channels": [], "capabilities": {"channel_titles": False}, "error_code": code}, value
    assert_private(value)


@test
def test_digest_discovery_maps_zero_based_table_titles_to_one_based_channels():
    with api() as (client, engine, _manager, _root), device() as peer:
        before = dump(engine)
        result = discover(client, peer)
        assert result == {"channels": [
            {"channel": 1, "name": "Вход"}, {"channel": 2, "name": "Склад = 2"},
            {"channel": 64, "name": "Последний"}],
            "capabilities": {"channel_titles": True}, "error_code": None}
        assert peer.requests[0] == (CGI_PATH, ""), "Do not send credentials preemptively"
        assert peer.authorized == [CGI_PATH]
        assert all(not auth.startswith("Basic ") for _path, auth in peer.requests)
        assert dump(engine) == before
        assert_private(result)


@test
def test_basic_auth_is_used_only_after_device_advertises_it():
    with api() as (client, _engine, _manager, _root), device(authentication="basic") as peer:
        result = discover(client, peer)
        assert result["capabilities"]["channel_titles"] is True
        assert peer.requests[0][1] == "" and peer.requests[-1][1].startswith("Basic ")
        assert peer.authorized == [CGI_PATH]


@test
def test_bad_auth_and_unsupported_discovery_are_not_empty_successes():
    with api() as (client, _engine, _manager, _root):
        with device() as peer:
            assert_discovery_failure(discover(client, peer, password="wrong"), "DEVICE_AUTH_FAILED")
            assert peer.authorized == []
        with device(mode="unsupported") as peer:
            assert_discovery_failure(discover(client, peer), "ARCHIVE_UNSUPPORTED")


@test
def test_discovery_malformed_out_of_range_duplicate_error_and_oversized_bodies_fail_closed():
    bodies = [b"Error\r\nBad Request\r\n", b"not channel titles", b"table.ChannelTitle[64].Name=65\r\n",
              b"table.ChannelTitle[-1].Name=zero\r\n", b"table.ChannelTitle[0].Name=\r\n",
              b"table.ChannelTitle[0].Name=A\r\ntable.ChannelTitle[0].Name=B\r\n",
              b"table.ChannelTitle[0].Name=bad\x00name\r\n", b"x" * (1024 * 1024 + 1)]
    bodies.append(b"table.ChannelTitle[" + b"9" * 5000 + b"].Name=Malformed\r\n")
    with api() as (client, _engine, _manager, _root):
        for body in bodies:
            with device(body=body) as peer:
                assert_discovery_failure(discover(client, peer), "DEVICE_PROTOCOL_ERROR")


@test
def test_discovery_timeout_is_bounded_and_distinct_from_empty_titles():
    import urllib.request
    original = urllib.request.OpenerDirector.open

    def short_open(opener, fullurl, data=None, timeout=socket._GLOBAL_DEFAULT_TIMEOUT):
        return original(opener, fullurl, data=data, timeout=.05)

    with api() as (client, _engine, _manager, _root), device(authentication="none", mode="timeout") as peer:
        with patch.object(urllib.request.OpenerDirector, "open", new=short_open):
            assert_discovery_failure(discover(client, peer), "DEVICE_TIMEOUT")


@test
def test_cgi_does_not_inherit_proxies_or_follow_redirects_with_credentials():
    with api() as (client, _engine, _manager, _root), device(authentication="none") as trap:
        proxy = f"http://127.0.0.1:{trap.port}"
        with device() as peer, patch.dict("os.environ", {
            "http_proxy": proxy, "https_proxy": proxy, "HTTP_PROXY": proxy,
            "HTTPS_PROXY": proxy, "no_proxy": "", "NO_PROXY": "",
        }):
            assert discover(client, peer)["error_code"] is None
        assert trap.requests == [], "LAN CGI must never inherit deployment proxy settings"
        with device(mode="redirect", redirect=proxy + "/credential-trap") as peer:
            assert_discovery_failure(discover(client, peer), "DEVICE_PROTOCOL_ERROR")
        assert trap.requests == [], "Even a same-network redirect must not receive a follow-up request"


@test
def test_https_transport_uses_verifying_context_and_returns_only_safe_failure():
    contexts = []
    original = ssl.create_default_context

    def capture_context(*args, **kwargs):
        context = original(*args, **kwargs)
        contexts.append(context)
        return context

    # Speaking TLS to this plaintext fixture fails the real handshake. Observing
    # the context also catches accidental CERT_NONE/check_hostname=False fallback.
    with api() as (client, _engine, _manager, _root), device(mode="tls_failure") as peer:
        with patch("ssl.create_default_context", side_effect=capture_context):
            result = discover(client, peer, http_scheme="https")
        assert contexts and all(context.check_hostname and context.verify_mode == ssl.CERT_REQUIRED
                                for context in contexts)
        assert result["channels"] == [] and result["capabilities"] == {"channel_titles": False}
        assert result["error_code"] in {"DEVICE_UNREACHABLE", "DEVICE_PROTOCOL_ERROR"}
        assert_private(result)
        import urllib.request
        failure = URLError(ssl.SSLCertVerificationError("fixture untrusted certificate"))
        with patch.object(urllib.request.OpenerDirector, "open", side_effect=failure):
            assert_discovery_failure(discover(client, peer, http_scheme="https"), "DEVICE_TLS_FAILED")


@test
def test_committed_source_or_archive_change_cancels_old_discovery_but_rename_does_not():
    for update, expected in (
        ({"name": "Переименована"}, None),
        ({"archive": archive(utc_offset_minutes=240), "expected_archive_revision": 0}, "SOURCE_CHANGED"),
        ({"password": "new-source", "channel": 2}, "SOURCE_CHANGED"),
    ):
        with api() as (client, _engine, _manager, _root), device(mode="hold") as peer:
            saved = create(client, nvr_host="127.0.0.1", archive=archive(http_port=peer.port))
            responses, failures = [], []

            def request_discovery():
                try:
                    responses.append(discover(client, peer, camera_id=saved["id"], password=""))
                except BaseException as exc:
                    failures.append(exc)

            worker = threading.Thread(target=request_discovery, daemon=True)
            worker.start()
            try:
                assert peer.receiving.wait(3), "Saved discovery must reach the real CGI fixture"
                client.json("PUT", f"/api/cameras/{saved['id']}", update)
            finally:
                peer.release.set()
                worker.join(3)
            assert not worker.is_alive() and not failures, failures
            assert len(responses) == 1 and responses[0]["error_code"] == expected, responses
            if expected:
                assert_discovery_failure(responses[0], expected)
            else:
                assert responses[0]["channels"] == [{"channel": 1, "name": "Held camera"}]


@test
def test_cgi_permission_failure_does_not_prevent_independent_live_probe_or_save():
    completed = SimpleNamespace(returncode=0, stdout=json.dumps({"streams": [
        {"codec_name": "h264", "width": 1920, "height": 1080, "avg_frame_rate": "25/1"}
    ]}).encode(), stderr=b"")
    with api() as (client, _engine, _manager, _root), device() as peer:
        assert_discovery_failure(discover(client, peer, password="wrong"), "DEVICE_AUTH_FAILED")
        with patch("app.services.camera_service.subprocess.run", return_value=completed), \
                patch("app.media.process_guard.MediaProcessGuard.require"):
            live = client.json("POST", "/api/cameras/test", connection())
        assert live["success"] and live["codec"] == "h264" and live["resolution"] == "1920x1080"
        saved = create(client)
        assert saved["archive"] is None, "Live-only setup must remain useful without CGI permissions"


@test
def test_saved_camera_password_reuse_for_discovery_and_blank_edit_never_exposes_secret():
    with api() as (client, engine, _manager, _root), device() as peer:
        saved = create(client, nvr_host="127.0.0.1", archive=archive(http_port=peer.port))
        assert saved.get("archive") is not None, "Saved archive connection must be returned without its credentials"
        result = discover(client, peer, camera_id=saved["id"], password="")
        assert result["error_code"] is None
        changed = client.json("PUT", f"/api/cameras/{saved['id']}", {"password": "", "name": "Дверь"})
        assert changed["has_password"] is True and changed["name"] == "Дверь"
        assert_private(changed)
        assert_private(client.json("GET", "/api/cameras"))
        with Session(engine) as db:
            assert db.get(Camera, saved["id"]).password == PASSWORD
        invalid = client.json("PUT", f"/api/cameras/{saved['id']}",
                              {"password": PASSWORD, "archive": archive()}, status=422)
        assert_private(invalid)
        invalid = client.json("POST", "/api/cameras",
                              dict(name="Invalid", **connection(password=PASSWORD * 20)), status=422)
        assert_private(invalid)


@test
def test_bare_host_validation_rejects_url_path_userinfo_port_and_control_characters():
    invalid = ["rtsp://nvr.example", "http://nvr.example", "nvr.example/path", "user@nvr.example",
               "nvr.example:554", "nvr.example?x=1", "nvr.example#x", "nvr\\share", "bad\nname"]
    with api() as (client, engine, manager, _root):
        before = dump(engine)
        for host in invalid:
            client.json("POST", "/api/cameras", dict(name="Bad", **connection(nvr_host=host)), status=422)
            client.json("POST", "/api/cameras/discover", dict(connection(nvr_host=host),
                        http_scheme="http", http_port=80), status=422)
        assert dump(engine) == before and manager.started == []


@test
def test_ipv6_brackets_and_special_credentials_form_valid_masked_rtsp_urls():
    for host in ("2001:db8::1", "[2001:db8::1]"):
        value = build_rtsp_url(host, 554, "u@ser", "p:/@#%", 64, "sub")
        assert value == "rtsp://u%40ser:p%3A%2F%40%23%25@[2001:db8::1]:554/cam/realmonitor?channel=64&subtype=1"
        assert urlsplit(value).hostname == "2001:db8::1"
        assert mask_url(value) == "rtsp://u%40ser:***@[2001:db8::1]:554/cam/realmonitor?channel=64&subtype=1"


@test
def test_batch_creates_calibrations_and_archives_atomically_before_ordered_publication():
    with api() as (client, engine, manager, _root):
        value = batch()
        result = client.json("POST", "/api/cameras/batch", value, status=201)
        saved = client.json("GET", "/api/cameras")
        assert [(row["channel"], row["name"]) for row in saved] == [(1, "Вход"), (2, "Склад")]
        assert all(row["archive"]["revision"] == 0 and row["archive"]["utc_offset_minutes"] == 180
                   for row in saved)
        assert manager.committed_counts == [2, 2], "No channel reader may see a partially saved recorder"
        assert [source.channel for source in manager.started] == [1, 2]
        assert 0 < manager.started[0].publication_sequence < manager.started[1].publication_sequence
        assert all(source.source_revision == 1 for source in manager.started)
        with engine.connect() as db:
            assert db.exec_driver_sql("SELECT count(*) FROM attendance_cameras").scalar_one() == 2
            assert db.exec_driver_sql("SELECT count(*) FROM camera_archives").scalar_one() == 2
            requests = db.exec_driver_sql("SELECT id,operation,payload_sha256,result FROM operator_requests").all()
            assert len(requests) == 1 and requests[0][0] == value["request_id"]
            assert len(requests[0][2]) == 64 and PASSWORD not in str(requests[0])
        assert_private(result)
        assert_private(saved)


@test
def test_batch_retry_replays_result_without_duplicate_rows_readers_or_persisted_credentials():
    with api() as (client, engine, manager, _root):
        value = batch()
        result = client.json("POST", "/api/cameras/batch", value, status=201)
        before = dump(engine)
        status, _headers, repeated = client.request("POST", "/api/cameras/batch", value)
        assert status in (200, 201) and repeated == result
        assert dump(engine) == before and len(manager.started) == 2
        conflict = dict(value, password="changed-secret")
        client.json("POST", "/api/cameras/batch", conflict, status=409)
        conflict = dict(value, channels=[dict(value["channels"][0], name="Changed")])
        client.json("POST", "/api/cameras/batch", conflict, status=409)
        with engine.begin() as db:
            db.exec_driver_sql("UPDATE operator_requests SET operation='employee.create' WHERE id=?",
                               (value["request_id"],))
        client.json("POST", "/api/cameras/batch", value, status=409)
        assert len(manager.started) == 2


@test
def test_batch_rejects_duplicate_channels_existing_normalized_tuple_and_rolls_back_every_channel():
    with api() as (client, engine, manager, _root):
        create(client, nvr_host="NVR.Example", channel=2, stream_type="sub")
        before, publications = dump(engine), len(manager.started)
        client.json("POST", "/api/cameras/batch", batch(nvr_host="nvr.example"), status=409)
        assert dump(engine) == before and len(manager.started) == publications
        duplicate = [{"channel": 1, "name": "A", "stream_type": "main", "enabled": True},
                     {"channel": 1, "name": "B", "stream_type": "sub", "enabled": True}]
        actual, _headers, _body = client.request("POST", "/api/cameras/batch", batch(channels=duplicate))
        assert actual in (409, 422)
        assert dump(engine) == before and len(manager.started) == publications
        # A different username or stream is an independent logical source.
        client.json("POST", "/api/cameras/batch", batch(username="other"), status=201)
        assert len(client.json("GET", "/api/cameras")) == 3


@test
def test_full_recorder_batch_accepts_64_disabled_channels_without_starting_readers():
    with api() as (client, engine, manager, _root):
        channels = [{"channel": index, "name": f"Канал {index}", "stream_type": "main", "enabled": False}
                    for index in range(1, 65)]
        result = client.json("POST", "/api/cameras/batch", batch(channels=channels), status=201)
        saved = client.json("GET", "/api/cameras")
        assert [row["channel"] for row in saved] == list(range(1, 65))
        assert all(row["enabled"] is False and row["archive"]["revision"] == 0 for row in saved)
        assert manager.started == []
        with engine.connect() as db:
            assert db.exec_driver_sql("SELECT count(*) FROM attendance_cameras").scalar_one() == 64
            assert db.exec_driver_sql("SELECT count(*) FROM camera_archives").scalar_one() == 64
        assert_private(result)


@test
def test_batch_bounds_channels_and_injected_second_insert_failure_has_no_publication_or_rows():
    with api() as (client, engine, manager, _root):
        before = dump(engine)
        for channels in ([], [{"channel": 0, "name": "Bad", "stream_type": "main", "enabled": True}],
                         [{"channel": 65, "name": "Bad", "stream_type": "main", "enabled": True}],
                         [{"channel": index, "name": "Too many", "stream_type": "main", "enabled": True}
                          for index in range(1, 66)]):
            client.json("POST", "/api/cameras/batch", batch(channels=channels), status=422)
        assert dump(engine) == before
        insertions = []

        def fail_second_camera(conn, cursor, statement, parameters, context, executemany):
            if statement.lstrip().upper().startswith("INSERT INTO CAMERAS"):
                insertions.append(statement)
                if len(insertions) == 2 or executemany:
                    raise OperationalError(statement, parameters, RuntimeError("injected batch disk failure"))

        event.listen(engine, "before_cursor_execute", fail_second_camera)
        try:
            client.json("POST", "/api/cameras/batch", batch(), status=503)
        finally:
            event.remove(engine, "before_cursor_execute", fail_second_camera)
        assert insertions and dump(engine) == before
        assert manager.started == [], "A failed batch must never publish even its first channel"


@test
def test_archive_sql_enforces_scheme_port_offset_revision_and_restrict_camera_fk():
    with api() as (client, engine, _manager, _root):
        assert "camera_archives" in inspect(engine).get_table_names(), "v2 archive table is required"
        saved = create(client)
        for field, invalid in (("http_scheme", "ftp"), ("http_port", 0), ("http_port", 65536),
                               ("utc_offset_minutes", -721), ("utc_offset_minutes", 841),
                               ("revision", -1), ("camera_id", 999)):
            value = dict(camera_id=saved["id"], http_scheme="http", http_port=80,
                         utc_offset_minutes=180, revision=0, updated_at=STAMP)
            value[field] = invalid
            try:
                with engine.begin() as db:
                    db.exec_driver_sql("INSERT INTO camera_archives (camera_id,http_scheme,http_port,"
                                       "utc_offset_minutes,revision,updated_at) VALUES (?,?,?,?,?,?)",
                                       tuple(value.values()))
            except IntegrityError:
                pass
            else:
                raise AssertionError(f"Archive DB must reject {field}={invalid!r}")
        for offset in (-720, 840):
            created = create(client, channel=1 if offset == -720 else 64,
                             username=str(offset), archive=archive(utc_offset_minutes=offset))
            assert created["archive"]["utc_offset_minutes"] == offset
        try:
            with engine.begin() as db:
                db.exec_driver_sql("DELETE FROM attendance_cameras WHERE camera_id=?", (created["id"],))
                db.exec_driver_sql("DELETE FROM cameras WHERE id=?", (created["id"],))
        except IntegrityError:
            pass
        else:
            raise AssertionError("Archive camera FK must RESTRICT direct deletion")


@test
def test_two_absent_archive_drafts_use_null_and_stale_save_rolls_back_live_changes():
    with api() as (client, engine, manager, _root):
        saved = create(client)
        path = f"/api/cameras/{saved['id']}"
        first_draft = client.json("GET", path)
        second_draft = client.json("GET", path)
        assert first_draft["archive"] is None and second_draft["archive"] is None
        with immediate_session(engine) as db:
            config = db.get(AttendanceCamera, saved["id"])
            config.role, config.enabled = "ENTRY", True
            config.line_start, config.line_end, config.inside_point = [0., .5], [1., .5], [.5, .8]
            config.reference_width, config.reference_height = 1920, 1080
            commission = AttendanceCommission(request_id=str(uuid4()), actor="Operator", reason="Verified",
                                             configuration_fingerprint={"version": 1}, acceptance_record={})
            db.add(commission)
            db.flush()
            commission_id = commission.id
            db.get(AttendanceRuntime, 1).commission_id = commission_id
        first = client.json("PUT", path, {
            "archive": archive(), "expected_archive_revision": None})
        assert (first["archive"]["http_scheme"], first["archive"]["http_port"],
                first["archive"]["utc_offset_minutes"], first["archive"]["revision"]) == ("http", 80, 180, 0)
        with Session(engine) as db:
            config = db.get(AttendanceCamera, saved["id"])
            assert (config.source_revision, config.revision, config.enabled) == (1, 0, True)
            assert db.get(AttendanceRuntime, 1).commission_id == commission_id
        before = dump(engine)
        publications = (list(manager.started), list(manager.synced), list(manager.stopped))
        client.json("PUT", path, {
            "name": "Stale draft", "password": "stale-secret", "channel": 2,
            "archive": archive(http_scheme="https", http_port=443, utc_offset_minutes=240),
            "expected_archive_revision": None}, status=409)
        assert dump(engine) == before, "Stale absence must roll back camera, source, calibration and commission"
        assert (manager.started, manager.synced, manager.stopped) == publications
        assert client.json("GET", path) == first, "The first archive configuration must survive the stale draft"


@test
def test_absent_archive_rejects_integer_zero_token_without_creating_configuration():
    with api() as (client, engine, manager, _root):
        saved = create(client)
        path = f"/api/cameras/{saved['id']}"
        assert client.json("GET", path)["archive"] is None
        before = dump(engine)
        client.json("PUT", path, {
            "password": "stale-secret", "channel": 2,
            "archive": archive(), "expected_archive_revision": 0}, status=409)
        assert dump(engine) == before, "Revision zero must refer to an existing row, never absence"
        assert manager.synced == []
        assert client.json("GET", path)["archive"] is None


@test
def test_archive_edit_requires_explicit_token_for_both_absent_and_existing_rows():
    for configured in (False, True):
        with api() as (client, engine, manager, _root):
            saved = create(client, **({"archive": archive()} if configured else {}))
            before = dump(engine)
            client.json("PUT", f"/api/cameras/{saved['id']}", {
                "password": "stale-secret", "channel": 2,
                "archive": archive(utc_offset_minutes=240)}, status=422)
            assert dump(engine) == before, "An omitted token is not an explicit null absence observation"
            assert manager.synced == []


@test
def test_existing_archive_revision_zero_rejects_null_without_mutation():
    with api() as (client, engine, manager, _root):
        saved = create(client, archive=archive())
        assert saved["archive"]["revision"] == 0
        before = dump(engine)
        client.json("PUT", f"/api/cameras/{saved['id']}", {
            "password": "stale-secret", "channel": 2,
            "archive": archive(utc_offset_minutes=240), "expected_archive_revision": None}, status=409)
        assert dump(engine) == before
        assert manager.synced == []


@test
def test_existing_archive_revision_zero_accepts_zero_and_advances_to_one():
    with api() as (client, engine, _manager, _root):
        saved = create(client, archive=archive())
        assert saved["archive"]["revision"] == 0
        path = f"/api/cameras/{saved['id']}"
        changed = client.json("PUT", path, {
            "archive": archive(utc_offset_minutes=240), "expected_archive_revision": 0})
        assert (changed["archive"]["http_scheme"], changed["archive"]["http_port"],
                changed["archive"]["utc_offset_minutes"], changed["archive"]["revision"]) == ("http", 80, 240, 1)
        assert client.json("GET", path)["archive"] == changed["archive"]
        with Session(engine) as db:
            config = db.get(AttendanceCamera, saved["id"])
            assert (config.source_revision, config.revision) == (1, 0)


@test
def test_archive_edit_cas_is_separate_from_live_source_and_commission_invalidation():
    with api() as (client, engine, manager, _root):
        saved = create(client, archive=archive())
        assert saved.get("archive") is not None, "Single-camera creation must persist/output explicit archive settings"
        assert saved["archive"]["revision"] == 0
        camera_id = saved["id"]
        with immediate_session(engine) as db:
            config = db.get(AttendanceCamera, camera_id)
            config.role, config.enabled = "ENTRY", True
            config.line_start, config.line_end, config.inside_point = [0., .5], [1., .5], [.5, .8]
            config.reference_width, config.reference_height = 1920, 1080
            commission = AttendanceCommission(request_id=str(uuid4()), actor="Operator", reason="Verified",
                                             configuration_fingerprint={"version": 1}, acceptance_record={})
            db.add(commission)
            db.flush()
            commission_id = commission.id
            db.get(AttendanceRuntime, 1).commission_id = commission_id
        changed = client.json("PUT", f"/api/cameras/{camera_id}", {
            "archive": archive(http_scheme="https", http_port=443, utc_offset_minutes=240),
            "expected_archive_revision": 0})
        assert changed["archive"]["revision"] == 1
        with Session(engine) as db:
            config = db.get(AttendanceCamera, camera_id)
            assert (config.source_revision, config.revision, config.enabled) == (1, 0, True)
            assert db.get(AttendanceRuntime, 1).commission_id == commission_id
        assert not manager.synced[-1][1] & {"nvr_host", "rtsp_port", "username", "password", "channel", "stream_type"}
        before = dump(engine)
        client.json("PUT", f"/api/cameras/{camera_id}", {
            "archive": archive(utc_offset_minutes=300), "expected_archive_revision": 0}, status=409)
        client.json("PUT", f"/api/cameras/{camera_id}", {"archive": archive()}, status=422)
        assert dump(engine) == before
        client.json("PUT", f"/api/cameras/{camera_id}", {"password": "new-secret", "channel": 2})
        with Session(engine) as db:
            config = db.get(AttendanceCamera, camera_id)
            assert (config.source_revision, config.revision, config.enabled) == (2, 1, False)
            assert db.get(AttendanceRuntime, 1).commission_id is None
        assert manager.synced[-1][0].source_revision == 2


@test
def test_archive_source_is_immutable_private_and_survives_later_source_edits():
    with api() as (client, engine, _manager, _root):
        saved = create(client, archive=archive())
        capture = getattr(camera_service, "get_archive_source", None)
        assert callable(capture), "Task 5 needs a transaction-captured get_archive_source(db, camera_id) seam"
        with Session(engine) as db:
            source = capture(db, saved["id"])
        assert (source.camera_id, source.source_revision, source.archive_revision, source.host,
                source.rtsp_port, source.http_scheme, source.http_port, source.username,
                source.password, source.channel, source.utc_offset_minutes) == (
                    saved["id"], 1, 0, "nvr.example", 554, "http", 80, "operator", PASSWORD, 1, 180)
        assert PASSWORD not in repr(source)
        try:
            source.channel = 2
        except (FrozenInstanceError, AttributeError):
            pass
        else:
            raise AssertionError("ArchiveSource must be immutable")
        client.json("PUT", f"/api/cameras/{saved['id']}", {"password": "later-secret", "channel": 2})
        assert (source.password, source.channel, source.source_revision) == (PASSWORD, 1, 1)
        with Session(engine) as db:
            stale_camera = db.get(Camera, saved["id"])
            assert stale_camera.password == "later-secret"
            client.json("PUT", f"/api/cameras/{saved['id']}", {"password": "third-secret", "channel": 3})
            refreshed = capture(db, saved["id"])
            assert (refreshed.password, refreshed.channel, refreshed.source_revision,
                    refreshed.archive_revision) == ("third-secret", 3, 3, 0), (
                        "A preloaded ORM identity must not contaminate a new archive snapshot")


@test
def test_archive_capture_cannot_mix_credentials_and_revisions_across_database_snapshots():
    with api() as (client, engine, _manager, _root):
        saved = create(client, archive=archive())
        capture = getattr(camera_service, "get_archive_source", None)
        assert callable(capture), "Archive capture seam is required"
        with engine.connect() as db:
            db.exec_driver_sql("PRAGMA journal_mode=WAL")
        updated = []

        def competing_update(conn, cursor, statement, parameters, context, executemany):
            if updated or not statement.lstrip().upper().startswith("SELECT") or "cameras" not in statement.lower():
                return
            updated.append(True)
            with engine.begin() as other:
                other.exec_driver_sql("UPDATE cameras SET password='new-source',channel=2 WHERE id=?", (saved["id"],))
                other.exec_driver_sql("UPDATE attendance_cameras SET source_revision=2 WHERE camera_id=?", (saved["id"],))
                other.exec_driver_sql("UPDATE camera_archives SET revision=1,utc_offset_minutes=240 WHERE camera_id=?",
                                      (saved["id"],))

        event.listen(engine, "after_cursor_execute", competing_update)
        try:
            with Session(engine) as db:
                source = capture(db, saved["id"])
        finally:
            event.remove(engine, "after_cursor_execute", competing_update)
        assert updated, "Race must occur at the real database read boundary"
        assert (source.password, source.channel, source.source_revision, source.archive_revision,
                source.utc_offset_minutes) in {
                    (PASSWORD, 1, 1, 0, 180), ("new-source", 2, 2, 1, 240)}, "Mixed archive snapshots are unsafe"


@test
def test_camera_archive_settings_are_deleted_but_historical_camera_still_returns_409():
    with api() as (client, engine, manager, _root):
        unused = create(client, archive=archive())
        assert unused.get("archive") is not None, "Archive connection settings must be persisted"
        client.json("DELETE", f"/api/cameras/{unused['id']}", status=204)
        with engine.connect() as db:
            assert db.exec_driver_sql("SELECT count(*) FROM camera_archives").scalar_one() == 0
            assert db.exec_driver_sql("SELECT count(*) FROM cameras").scalar_one() == 0
        historical = create(client, archive=archive())
        with immediate_session(engine) as db:
            db.add(AttendanceEvent(id="camera-history", source_key="camera-history", source="CAMERA",
                camera_id=historical["id"], camera_name="Вход", direction="ENTRY", occurred_at=STAMP,
                recorded_at=STAMP, status="VISITOR", raw={"version": 1, "candidate_employee_ids": []}))
        before, stopped = dump(engine), list(manager.stopped)
        client.json("DELETE", f"/api/cameras/{historical['id']}", status=409)
        assert dump(engine) == before and manager.stopped == stopped


@test
def test_new_and_changed_camera_posts_require_operator_header_and_same_origin():
    with api() as (client, engine, manager, _root):
        before = dump(engine)
        for path, payload in (("/api/cameras", dict(name="Blocked", **connection())),
                              ("/api/cameras/batch", batch()),
                              ("/api/cameras/discover", dict(connection(), http_scheme="http", http_port=80))):
            client.json("POST", path, payload, status=403, operator=False)
            client.json("POST", path, payload, status=403, headers={"origin": "http://untrusted.example"})
        assert dump(engine) == before and manager.started == []
        saved = create(client)
        before = dump(engine)
        client.json("PUT", f"/api/cameras/{saved['id']}", {"name": "Blocked"}, status=403, operator=False)
        client.json("DELETE", f"/api/cameras/{saved['id']}", status=403, operator=False)
        assert dump(engine) == before


def archive_feature():
    from app.services import dahua_service
    assert hasattr(dahua_service, "DahuaArchiveClient"), "Task 5 Dahua archive finder is missing"
    return dahua_service


def finder_page(*, path="/mnt/dvr/2026/11/12/first [R].dav", channel=1,
                start="2026-11-12 13:00:00", end="2026-11-12 13:01:00", kind="dav"):
    return (f"found=1\r\nitems[0].Channel={channel}\r\nitems[0].StartTime={start}\r\n"
            f"items[0].EndTime={end}\r\nitems[0].Type={kind}\r\n"
            f"items[0].FilePath={path}\r\nitems[0].Length=12345\r\n"
            f"items[0].Duration=60\r\n").encode()


class FinderPeer:
    """Protocol fixture; transport, digest, parser and cleanup stay production code."""
    def __init__(self, pages=(), failures=None, on_page=None):
        self.pages = list(pages)
        self.failures = failures or {}
        self.on_page = on_page
        self.calls = []
        self.page = 0

    def __call__(self, query):
        self.calls.append(query)
        action = query["action"][0]
        if action in self.failures:
            return self.failures[action]
        if action == "factory.create":
            return 200, b"result=08137\r\n"
        assert query.get("object") == ["08137"], query
        if action == "findNextFile":
            assert query.get("count") == ["100"], query
            if self.on_page:
                self.on_page(self.page)
            page = self.pages[min(self.page, len(self.pages) - 1)] if self.pages else b"found=0\r\n"
            self.page += 1
            return 200, page
        assert action in {"findFile", "close", "destroy"}, query
        return 200, b"OK\r\n"

    @property
    def actions(self):
        return [query["action"][0] for query in self.calls]


@contextmanager
def finder_rig(finder, *, offset=180, password=PASSWORD):
    module = archive_feature()
    with api() as (client, engine, _manager, _root), device(finder=finder) as peer:
        saved = create(client, nvr_host="127.0.0.1", password=password,
                       archive=archive(http_port=peer.port, utc_offset_minutes=offset))
        with Session(engine) as db:
            source = camera_service.get_archive_source(db, saved["id"])
        yield module.DahuaArchiveClient(source), peer, module


def search_error(module, code, operation):
    try:
        operation()
    except module.DahuaError as exc:
        assert exc.code == code, exc.code
        assert PASSWORD not in str(exc) and "/mnt/" not in str(exc)
    else:
        raise AssertionError(f"Search must fail visibly with {code}, not return an empty archive")


SEARCH_FROM = datetime(2026, 11, 12, 10, tzinfo=timezone.utc)
SEARCH_TO = SEARCH_FROM + timedelta(hours=1)


@test
def test_finder_digest_multipage_dedup_offset_types_and_final_cleanup():
    finder = FinderPeer([finder_page(), finder_page(),
                         finder_page(path="/mnt/dvr/second.mp4", start="2026-11-12 13:01:00",
                                     end="2026-11-12 13:02:00", kind="mp4"), b"found=0\r\n"])
    with finder_rig(finder) as (client, peer, _module):
        intervals = client.list_recordings(SEARCH_FROM, SEARCH_TO)
        assert len(intervals) == 2, "Duplicate finder rows must not duplicate timeline coverage"
        assert intervals[0].start_utc == SEARCH_FROM
        assert intervals[0].end_utc == SEARCH_FROM + timedelta(minutes=1)
        assert intervals[1].start_utc == SEARCH_FROM + timedelta(minutes=1)
        query = next(call for call in finder.calls if call["action"] == ["findFile"])
        assert query["condition.Channel"] == ["1"]
        assert query["condition.StartTime"] == ["2026-11-12 13:00:00"]
        assert query["condition.EndTime"] == ["2026-11-12 14:00:00"]
        assert query["condition.Types[0]"] == ["dav"] and query["condition.Types[1]"] == ["mp4"]
        assert not any("Dir" in key or "Event" in key for key in query)
        assert any("%20" in path for path in peer.authorized)
        assert all("+" not in path for path in peer.authorized)
        assert finder.actions == ["factory.create", "findFile", "findNextFile", "findNextFile",
                                  "findNextFile", "findNextFile", "close", "destroy"]


@test
def test_finder_rejects_malformed_channel_times_type_and_unsafe_paths_and_cleans():
    invalid = [
        b"found=1\r\n", b"found=101\r\n", b"found=-1\r\n", b"found=no\r\n",
        finder_page(channel=0), finder_page(channel=2), finder_page(kind="jpg"),
        finder_page(start="2026-11-12 13:02:00"), finder_page(end="not-a-time"),
        finder_page(path="/mnt/../secret"), finder_page(path="/mnt/%2e%2e/secret"),
        finder_page(path="/mnt/a\x00b"), finder_page(path="rtsp://attacker.invalid/file"),
        finder_page(path="/etc/passwd"), finder_page(path="/mnt/a\\..\\secret"),
        b"x" * (1024 * 1024 + 1),
    ]
    for body in invalid:
        finder = FinderPeer([body])
        with finder_rig(finder) as (client, _peer, module):
            search_error(module, "DEVICE_PROTOCOL_ERROR", lambda: client.list_recordings(SEARCH_FROM, SEARCH_TO))
            assert finder.actions[-2:] == ["close", "destroy"], body[:80]


@test
def test_finder_rejects_overlong_item_indices_and_cleans_digest_search():
    # A padded zero would otherwise alias a valid row and be silently ignored.
    for index in (b"9" * 5000, b"000"):
        body = finder_page() + b"items[" + index + b"].Length=12345\r\n"
        finder = FinderPeer([body])
        with finder_rig(finder) as (client, _peer, module):
            try:
                search_error(module, "DEVICE_PROTOCOL_ERROR",
                             lambda: client.list_recordings(SEARCH_FROM, SEARCH_TO))
            finally:
                assert finder.actions == [
                    "factory.create", "findFile", "findNextFile", "close", "destroy"]


@test
def test_finder_accepts_complete_hundred_record_page_including_index_99():
    body = b"found=100\r\n" + b"".join(
        finder_page(path=f"/mnt/dvr/{index}.dav").partition(b"\r\n")[2].replace(
            b"items[0].", f"items[{index}].".encode())
        for index in range(100))
    finder = FinderPeer([body, b"found=0\r\n"])
    with finder_rig(finder) as (client, _peer, _module):
        intervals = client.list_recordings(SEARCH_FROM, SEARCH_TO)
        assert len(intervals) == 100, "All legitimate finder indices 0 through 99 must remain usable"
        assert finder.actions == [
            "factory.create", "findFile", "findNextFile", "findNextFile", "close", "destroy"]


@test
def test_finder_find_failure_and_close_failure_still_destroy_and_never_empty_success():
    for failures, code in (
        ({"findFile": (200, b"Error\r\n")}, "DEVICE_PROTOCOL_ERROR"),
        ({"findNextFile": (401, b"private credentials")}, "DEVICE_AUTH_FAILED"),
        ({"findNextFile": (404, b"unsupported")}, "ARCHIVE_UNSUPPORTED"),
        ({"findNextFile": (200, b"broken"), "close": (500, b"broken")}, "DEVICE_PROTOCOL_ERROR"),
    ):
        finder = FinderPeer(failures=failures)
        with finder_rig(finder) as (client, _peer, module):
            search_error(module, code, lambda: client.list_recordings(SEARCH_FROM, SEARCH_TO))
            assert finder.actions[-2:] == ["close", "destroy"]
    finder = FinderPeer()
    with finder_rig(finder, password="incorrect") as (client, _peer, module):
        search_error(module, "DEVICE_AUTH_FAILED", lambda: client.list_recordings(SEARCH_FROM, SEARCH_TO))
        assert finder.actions == [], "No finder object exists to clean up after failed authentication"


@test
def test_finder_page_and_total_time_limits_are_errors_not_truncated_results():
    finder = FinderPeer([finder_page()])
    with finder_rig(finder) as (client, _peer, module):
        search_error(module, "SEARCH_LIMIT", lambda: client.list_recordings(SEARCH_FROM, SEARCH_TO))
        assert finder.page == 100
        assert finder.actions[-2:] == ["close", "destroy"]
    clock = [100.]
    finder = FinderPeer([finder_page()], on_page=lambda _page: clock.__setitem__(0, clock[0] + 7))
    with finder_rig(finder) as (client, _peer, module):
        with patch.object(module.time, "monotonic", side_effect=lambda: clock[0]):
            search_error(module, "SEARCH_LIMIT", lambda: client.list_recordings(SEARCH_FROM, SEARCH_TO))
        assert finder.page <= 9
        assert finder.actions[-2:] == ["close", "destroy"]


@test
def test_finder_empty_is_success_only_after_find_and_timezone_crosses_date_boundary():
    finder = FinderPeer()
    with finder_rig(finder, offset=-720) as (client, _peer, _module):
        assert client.list_recordings(SEARCH_FROM, SEARCH_TO) == []
        query = next(call for call in finder.calls if call["action"] == ["findFile"])
        assert query["condition.StartTime"] == ["2026-11-11 22:00:00"]
        assert query["condition.EndTime"] == ["2026-11-11 23:00:00"]
        assert finder.actions[-2:] == ["close", "destroy"]


class FinderBudgetTransport:
    """Socket-timeout model; real request/parser/finally code owns the lifecycle."""
    def __init__(self, clock, mode, client):
        self.clock, self.mode, self.client = clock, mode, client
        self.actions, self.responses = [], []
        self.pages = 0

    def open(self, request, timeout):
        action = parse_qs(urlsplit(request.full_url).query)["action"][0]
        self.actions.append(action)
        cleanup = action in ("close", "destroy")
        # Headers consume part of the socket budget. A cleanup body then stalls;
        # reusing the original socket timeout would exceed the request deadline.
        latency = 3. if cleanup else 7. if action == "findNextFile" else 0.
        self.clock[0] += min(latency, timeout)
        if latency >= timeout:
            raise TimeoutError("fixture network deadline")
        if action == "factory.create":
            body = b"result=08137\r\n"
        elif action == "findNextFile":
            self.pages += 1
            body = finder_page()
            if self.mode == "failure" and self.pages == 3:
                body = b"invalid finder response"
            elif self.mode == "cancelled" and self.pages == 3:
                self.client.cancel()
            elif self.mode == "success" and self.pages == 5:
                body = b"found=0\r\n"
        else:
            body = b"OK\r\n"
        response = self.Response(self.clock, body, timeout, cleanup)
        self.responses.append(response)
        return response

    class Response:
        status = 200

        def __init__(self, clock, body, timeout, stalled):
            self.clock, self.body, self.timeout, self.stalled = clock, body, timeout, stalled
            self.headers = {"Content-Length": str(len(body))}
            self.closed = False
            self.fp = SimpleNamespace(raw=SimpleNamespace(_sock=self))

        def settimeout(self, value):
            self.timeout = value

        def read1(self, count):
            if self.stalled:
                self.clock[0] += self.timeout
                raise TimeoutError("fixture stalled response body")
            result, self.body = self.body[:count], self.body[count:]
            return result

        def close(self):
            self.closed = True


@test
def test_finder_whole_lifecycle_reserves_both_stalled_cleanup_attempts_within_sixty_seconds():
    for mode, error_code in (("limit", "SEARCH_LIMIT"), ("success", None),
                             ("failure", "DEVICE_PROTOCOL_ERROR"), ("cancelled", "SOURCE_CHANGED")):
        with finder_rig(FinderPeer()) as (client, _peer, module):
            clock = [100.]
            transport = FinderBudgetTransport(clock, mode, client)
            # Substitute only network opening/socket waits and the monotonic
            # clock. The cleanup client, request loop and error mapping are real.
            with patch.object(client, "_opener", transport), \
                    patch.object(module, "build_opener", return_value=transport), \
                    patch.object(module, "time", SimpleNamespace(monotonic=lambda: clock[0])):
                if error_code is None:
                    result = client.list_recordings(SEARCH_FROM, SEARCH_TO)
                    assert len(result) == 1 and result[0].start_utc == SEARCH_FROM
                else:
                    search_error(module, error_code, lambda: client.list_recordings(SEARCH_FROM, SEARCH_TO))
            assert transport.actions[-2:] == ["close", "destroy"], (mode, transport.actions)
            assert all(response.closed for response in transport.responses)
            assert clock[0] - 100. <= 60., (mode, "finder plus cleanup exceeded 60 seconds", clock[0] - 100.)


if __name__ == "__main__":
    failures = 0
    for case in TESTS:
        try:
            case()
            print(f"PASS {case.__name__}")
        except Exception:
            failures += 1
            print(f"FAIL {case.__name__}")
            traceback.print_exc()
    print(f"{len(TESTS) - failures}/{len(TESTS)} tests passed")
    sys.exit(bool(failures))
