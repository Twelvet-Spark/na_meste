"""Task 6 consumer contracts; Main owns RED/GREEN execution and browser proof.

Run: .venv/Scripts/python.exe tests/test_operator_settings.py
Real Settings, SQLite, ASGI routes, persistence and diagnostic services. No app
lifespan, production database, camera connection, Telegram or media subprocess.
Missing features fail explicit assertions, not import/collection errors.
"""
import gc
import hashlib
import importlib
import importlib.util
import json
import logging
import os
import sys
import tempfile
import threading
import traceback
import weakref
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi import FastAPI
from sqlalchemy.orm import Session

# Importing configuration must never select an operator's override document.
with tempfile.TemporaryDirectory() as bootstrap, patch.dict(os.environ, {
    "OPERATOR_SETTINGS_PATH": str(Path(bootstrap) / "absent.json"),
    "DATABASE_URL": "sqlite://", "AI_ENABLED": "false", "TELEGRAM_ENABLED": "false",
}):
    from app import config
    from app.database.database import get_db
    from app.database.models import Camera
    from operator_test_support import Client
    from test_operator_storage import database, schema

TESTS = []
TOKEN = "123456789:TEST_ONLY_telegram_token_sentinel"
OTHER_TOKEN = "987654321:TEST_ONLY_replacement_token_sentinel"
PASSWORD = "TEST_ONLY_camera_password_sentinel"
USERNAME = "TEST_ONLY_camera_username_sentinel"
AUTH = "TEST_ONLY_authorization_sentinel"
COOKIE = "TEST_ONLY_cookie_sentinel"
URL = f"rtsp://{USERNAME}:{PASSWORD}@127.0.0.1/private?token={TOKEN}"
SECRETS = (TOKEN, OTHER_TOKEN, PASSWORD, USERNAME, AUTH, COOKIE)
EDITABLE = {
    "preview_fps", "rtsp_connect_timeout", "rtsp_reconnect_max_delay", "frame_stall_timeout",
    "stream_output_fps", "stream_hwaccel",
    "ai_enabled", "ai_device", "ai_fps",
    "ai_batch_size", "ai_batch_wait_ms",
    "min_recognition_confidence", "min_confirmations", "face_recognition_retry_interval",
    "live_face_min_size", "live_face_margin", "live_face_window", "live_face_recheck_interval",
    "journal_image_keep_days",
    "attendance_ai_fps", "attendance_frame_max_age", "attendance_max_frame_gap",
    "attendance_face_threshold", "attendance_face_margin", "attendance_face_confirmations",
    "attendance_face_window", "attendance_face_interval", "attendance_face_min_size",
    "attendance_face_det_score", "attendance_face_blur_min",
    "attendance_live_gap_minutes", "attendance_day_gap_minutes",
    "telegram_enabled", "telegram_bot_token", "telegram_chat_id", "telegram_notify_presence",
    "telegram_notify_global_transition", "telegram_notify_global_new",
    "recording_retention_days", "recording_max_bytes", "mc_debug",
}


def test(fn):
    TESTS.append(fn)
    return fn


def feature(module):
    assert importlib.util.find_spec(module) is not None, f"Task6 feature missing: {module}"
    return importlib.import_module(module)


def no_secrets(value):
    text = json.dumps(value, ensure_ascii=False, default=str)
    for sentinel in SECRETS:
        assert sentinel not in text, "Credential leaked into consumer-visible output"


def document(path, values):
    path.write_text(json.dumps({"version": 1, "values": values}), encoding="utf-8")


def load(path, dotenv):
    assert hasattr(config, "load_settings"), "Task6 validated startup loader missing"
    return config.load_settings(path=path, env_file=dotenv)


@contextmanager
def api(*, overrides=None, dotenv="", env=None):
    settings_module = feature("app.services.settings_service")
    settings_api = feature("app.api.settings")
    with database() as (engine, root), patch.dict(os.environ, env or {}, clear=True):
        schema.initialize_attendance_schema(engine)
        path, env_file = root / "operator-settings.json", root / "fixture.env"
        env_file.write_text(dotenv, encoding="utf-8")
        if overrides is not None:
            document(path, overrides)
        active = load(path, env_file)
        service = settings_module.SettingsService(active, engine, path=path, env_file=env_file)
        app = FastAPI()
        app.state.settings_service = service
        app.include_router(settings_api.router)

        def isolated_db():
            with Session(engine) as db:
                yield db

        app.dependency_overrides[get_db] = isolated_db
        yield SimpleNamespace(client=Client(app), app=app, engine=engine, root=root,
                              path=path, env_file=env_file, active=active, service=service)


def get(f):
    return f.client.json("GET", "/api/settings")


def save(f, values=None, reset_keys=(), *, version=None, status=200, **kwargs):
    if version is None:
        version = get(f)["version"]
    return f.client.json("PUT", "/api/settings", {
        "expected_version": version, "values": values or {}, "reset_keys": list(reset_keys),
    }, status=status, **kwargs)


@test
def test_real_settings_environment_overrides_canonical_init_and_dotenv():
    with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
        dotenv = Path(directory) / "fixture.env"
        dotenv.write_text("PREVIEW_FPS=4\nFACE_RECOGNITION_THRESHOLD=0.37\n", encoding="utf-8")
        actual = config.Settings(_env_file=dotenv, preview_fps=8, min_recognition_confidence=.61)
        assert actual.preview_fps == 8 and actual.min_recognition_confidence == .61
        with patch.dict(os.environ, {"PREVIEW_FPS": "17", "FACE_RECOGNITION_THRESHOLD": "0.72"}):
            actual = config.Settings(_env_file=dotenv, preview_fps=8, min_recognition_confidence=.61)
            assert actual.preview_fps == 17 and actual.min_recognition_confidence == .72
        with patch.dict(os.environ, {"MIN_RECOGNITION_CONFIDENCE": ".81",
                                    "FACE_RECOGNITION_THRESHOLD": ".72"}):
            assert config.Settings(_env_file=dotenv).min_recognition_confidence == .81
        assert config.Settings(_env_file=None).preview_fps == 10


@test
def test_environment_lock_detection_handles_uppercase_and_aliaschoices():
    module = feature("app.operator_settings")
    with patch.dict(os.environ, {"PREVIEW_FPS": "19", "FACE_RECOGNITION_THRESHOLD": ".72"}, clear=True):
        locked = module.environment_fields(config.Settings)
        assert "preview_fps" in locked and "min_recognition_confidence" in locked
        assert "ai_fps" not in locked
    with api(overrides={"preview_fps": 8}, env={"PREVIEW_FPS": "19"}) as f:
        view = get(f)
        assert view["active"]["preview_fps"] == view["pending"]["preview_fps"] == 19
        assert view["fields"]["preview_fps"]["locked"] is True
        assert view["fields"]["preview_fps"]["source"] == "environment"
        before = f.path.read_bytes()
        save(f, {"preview_fps": 20}, status=409)
        save(f, reset_keys=["preview_fps"], status=409)
        assert f.path.read_bytes() == before


@test
def test_missing_file_defaults_and_base_relative_startup_path():
    module = feature("app.operator_settings")
    with patch.dict(os.environ, {}, clear=True):
        assert module.settings_path() == config.BASE_DIR / "data" / "operator-settings.json"
        with patch.dict(os.environ, {"OPERATOR_SETTINGS_PATH": "isolated/operator-settings.json"}):
            assert module.settings_path() == config.BASE_DIR / "isolated" / "operator-settings.json"
    with api() as f:
        view = get(f)
        assert not f.path.exists() and view["restart_required"] is False
        assert view["active"]["recording_retention_days"] == 14
        assert view["active"]["recording_max_bytes"] == 150000000000
        assert view["version"] == hashlib.sha256(b"").hexdigest()


@test
def test_json_v1_save_is_durable_but_active_changes_only_on_next_startup():
    with api(dotenv="PREVIEW_FPS=5\n") as f:
        before = f.active.model_dump()
        view = save(f, {"preview_fps": 12})
        assert view["active"]["preview_fps"] == 5 and view["pending"]["preview_fps"] == 12
        assert view["restart_required"] is True and "preview_fps" in view["changed_keys"]
        assert f.active.model_dump() == before
        persisted = json.loads(f.path.read_text(encoding="utf-8"))
        assert persisted == {"version": 1, "values": {"preview_fps": 12}}
        assert view["version"] == hashlib.sha256(f.path.read_bytes()).hexdigest()
        assert get(f)["restart_required"] is True
        restarted = load(f.path, f.env_file)
        assert restarted.preview_fps == 12 and f.active.preview_fps == 5
        service = type(f.service)(restarted, f.engine, path=f.path, env_file=f.env_file)
        assert service.snapshot()["restart_required"] is False
        assert f.env_file.read_text(encoding="utf-8") == "PREVIEW_FPS=5\n"


@test
def test_startup_rejects_unknown_malformed_and_invalid_documents_without_echo():
    feature("app.operator_settings")
    bad = [
        b"{not json", b"[]", b"null", b'{"version":2,"values":{}}',
        b'{"version":true,"values":{}}', b'{"version":1}',
        b'{"version":1,"values":[],"extra":1}',
        json.dumps({"version": 1, "values": {"database_url": URL}}).encode(),
        json.dumps({"version": 1, "values": {"preview_fps": TOKEN}}).encode(),
        json.dumps({"version": 1, "values": {"telegram_bot_token": TOKEN}, "extra": AUTH}).encode(),
        b'{"version":1,"values":{"ai_fps":NaN}}',
        b'{"version":1,"values":{"attendance_face_window":1,"attendance_face_interval":2}}',
    ]
    with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
        path = Path(directory) / "operator-settings.json"
        for raw in bad:
            path.write_bytes(raw)
            try:
                load(path, None)
            except Exception as exc:
                no_secrets(str(exc))
                assert "OPERATOR_SETTINGS" in str(exc), "Startup failure needs a safe actionable code"
            else:
                raise AssertionError("Invalid override document silently admitted at startup")
            assert path.read_bytes() == raw


@test
def test_environment_cannot_hide_invalid_override_document_at_startup():
    feature("app.operator_settings")
    with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"PREVIEW_FPS": "10"}, clear=True):
        path = Path(directory) / "operator-settings.json"
        document(path, {"preview_fps": TOKEN})
        try:
            load(path, None)
        except Exception as exc:
            no_secrets(str(exc))
        else:
            raise AssertionError("Environment masked an invalid stored override")


@test
def test_public_allowlist_constraints_and_deployment_fields_cannot_be_written():
    with api() as f:
        view = get(f)
        assert set(view["fields"]) == EDITABLE
        for key in EDITABLE:
            assert {"constraints", "source", "locked"} <= view["fields"][key].keys()
        assert view["read_only"]["recording_max_duration_seconds"] == 86400
        assert view["read_only"]["attendance_utc_offset_minutes"] == 300
        for key in ("database_url", "app_host", "app_port", "recording_root", "recording_min_free_bytes",
                    "recording_max_duration_seconds", "attendance_utc_offset_minutes", "ai_model_manifest",
                    "cuda_dll_path", "topology_config", "reid_similarity_threshold",
                    "person_confidence", "unknown"):
            no_secrets(save(f, {key: TOKEN}, status=422))
            save(f, reset_keys=[key], status=422)
        assert not f.path.exists()


@test
def test_each_editable_field_accepts_valid_value_and_remains_pending():
    values = {
        "preview_fps": 30, "rtsp_connect_timeout": 60, "rtsp_reconnect_max_delay": 300,
        "frame_stall_timeout": 120, "ai_enabled": False, "ai_device": "cpu", "ai_fps": 30,
        "stream_output_fps": 30, "stream_hwaccel": "none", "ai_batch_size": 8, "ai_batch_wait_ms": 50,
        "attendance_ai_fps": 12.5, "attendance_frame_max_age": 2.,
        "min_recognition_confidence": .72, "min_confirmations": 4,
        "face_recognition_retry_interval": .4, "live_face_min_size": 64,
        "live_face_margin": .12, "live_face_window": 3., "live_face_recheck_interval": 3.,
        "journal_image_keep_days": 45,
        "attendance_max_frame_gap": 1., "attendance_face_threshold": .65,
        "attendance_face_margin": .12, "attendance_face_confirmations": 4,
        "attendance_face_window": 2., "attendance_face_interval": .5,
        "attendance_face_min_size": 90, "attendance_face_det_score": .85,
        "attendance_face_blur_min": 25., "attendance_live_gap_minutes": 30, "attendance_day_gap_minutes": 90,
        "telegram_enabled": False, "telegram_bot_token": TOKEN, "telegram_chat_id": "-100123456",
        "telegram_notify_presence": False, "telegram_notify_global_transition": False,
        "telegram_notify_global_new": True, "recording_retention_days": 365,
        "recording_max_bytes": 100000000000000, "mc_debug": True,
    }
    with api() as f:
        before = f.active.model_dump()
        view = save(f, values)
        for key, value in values.items():
            if key != "telegram_bot_token":
                assert view["pending"][key] == value, key
        assert view["pending"]["has_telegram_bot_token"] is True
        no_secrets(view)
        assert f.active.model_dump() == before


@test
def test_all_numeric_bounds_nonfinite_values_and_crossfield_window_are_enforced():
    invalid = {
        "preview_fps": [0, 31, 1.5, True], "rtsp_connect_timeout": [0, 61],
        "rtsp_reconnect_max_delay": [0, 301], "frame_stall_timeout": [0, 121],
        "ai_fps": [0, -1, 30.1, float("nan"), float("inf"), float("-inf")],
        "ai_device": ["gpu", "", TOKEN], "attendance_ai_fps": [0, float("inf")],
        "stream_output_fps": [4, 31, True], "stream_hwaccel": ["gpu", "", "software"],
        "ai_batch_size": [0, 9, 1.5, True], "ai_batch_wait_ms": [-1, 51, 1.5, True],
        "min_recognition_confidence": [-.01, 1.01, float("nan")],
        "min_confirmations": [0, 1.5, True], "face_recognition_retry_interval": [0, float("inf")],
        "live_face_min_size": [0, 1.5, 4097], "live_face_margin": [-.01, 1.01, float("inf")],
        "live_face_window": [0, 60.1, float("nan")],
        "live_face_recheck_interval": [0, 60.1, float("inf")],
        "journal_image_keep_days": [0, 366, 1.5, True],
        "attendance_frame_max_age": [0, float("nan")], "attendance_max_frame_gap": [0, float("inf")],
        "attendance_face_threshold": [-.01, 1.01, float("nan")],
        "attendance_face_margin": [-.01, 1.01, float("inf")],
        "attendance_face_confirmations": [0, 1.5], "attendance_face_window": [0, float("inf")],
        "attendance_face_interval": [0, float("nan")], "attendance_face_min_size": [0, 1.5],
        "attendance_face_det_score": [-.01, 1.01, float("inf")],
        "attendance_face_blur_min": [-1, float("nan")],
        "attendance_live_gap_minutes": [0, 241, 1.5, True], "attendance_day_gap_minutes": [0, 721, 1.5, True],
        "recording_retention_days": [0, 366], "recording_max_bytes": [9999999999, 100000000000001],
        "ai_enabled": [None, "not-bool"], "telegram_enabled": [None], "mc_debug": [None],
    }
    with api(overrides={"preview_fps": 9}) as f:
        before = f.path.read_bytes()
        for key, values in invalid.items():
            for value in values:
                no_secrets(save(f, {key: value}, status=422))
                assert f.path.read_bytes() == before, key
        save(f, {"attendance_face_interval": 2}, status=422)
        save(f, {"attendance_face_window": .1}, status=422)
        assert f.path.read_bytes() == before
        save(f, {"attendance_face_window": 2, "attendance_face_interval": 2})
        # A finished day may never forgive less than a running day does.
        save(f, {"attendance_live_gap_minutes": 60, "attendance_day_gap_minutes": 30}, status=422)
        save(f, {"attendance_day_gap_minutes": 20}, status=422)  # below the default live gap of 35
        save(f, {"attendance_live_gap_minutes": 20, "attendance_day_gap_minutes": 20})


@test
def test_retired_ai_cameras_key_is_dropped_on_read_and_absent_after_save():
    with api(overrides={"preview_fps": 9, "ai_cameras": "1,3"}) as f:
        assert get(f)["pending"].get("ai_cameras") is None
        assert load(f.path, f.env_file).preview_fps == 9
        save(f, {"ai_fps": 4})
        assert json.loads(f.path.read_text(encoding="utf-8")) == {
            "version": 1, "values": {"preview_fps": 9, "ai_fps": 4}}
        # The key is no longer editable either.
        save(f, {"ai_cameras": "1"}, status=422)


@test
def test_retired_evidence_keep_days_key_is_dropped_on_read_and_absent_after_save():
    with api(overrides={"preview_fps": 9, "attendance_evidence_keep_days": 45}) as f:
        assert get(f)["pending"].get("attendance_evidence_keep_days") is None
        assert load(f.path, f.env_file).preview_fps == 9
        save(f, {"ai_fps": 4})
        assert json.loads(f.path.read_text(encoding="utf-8")) == {
            "version": 1, "values": {"preview_fps": 9, "ai_fps": 4}}
        save(f, {"attendance_evidence_keep_days": 30}, status=422)


@test
def test_legacy_ai_camera_selection_precedence_is_environment_then_document_then_dotenv():
    from app.operator_settings import legacy_ai_camera_ids
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        dotenv, path = root / "fixture.env", root / "operator-settings.json"
        dotenv.write_text("AI_CAMERAS=5\n", encoding="utf-8")
        with patch.dict(os.environ, {"OPERATOR_SETTINGS_PATH": str(path)}, clear=True):
            assert legacy_ai_camera_ids(dotenv) == {5}
            document(path, {"ai_cameras": " 3 ; 1, 3 "})
            assert legacy_ai_camera_ids(dotenv) == {1, 3}
            with patch.dict(os.environ, {"AI_CAMERAS": "2"}):
                assert legacy_ai_camera_ids(dotenv) == {2}
            # An empty value selects every camera.
            with patch.dict(os.environ, {"AI_CAMERAS": ""}):
                assert legacy_ai_camera_ids(dotenv) is None
            with patch.dict(os.environ, {"AI_CAMERAS": "1,garbage"}):
                assert legacy_ai_camera_ids(dotenv) is None


@test
def test_partial_save_and_reset_preserve_unrelated_overrides_and_reveal_dotenv():
    with api(overrides={"preview_fps": 9, "ai_fps": 7, "mc_debug": True},
             dotenv="PREVIEW_FPS=4\nAI_FPS=6\n") as f:
        save(f, {"frame_stall_timeout": 20})
        view = save(f, reset_keys=["preview_fps", "mc_debug"])
        assert view["pending"]["preview_fps"] == 4 and view["pending"]["mc_debug"] is False
        assert view["active"]["preview_fps"] == 9 and view["active"]["mc_debug"] is True
        assert json.loads(f.path.read_text())["values"] == {"ai_fps": 7, "frame_stall_timeout": 20}
        assert view["fields"]["preview_fps"]["source"] == "dotenv"
        assert view["fields"]["mc_debug"]["source"] == "default"


@test
def test_reset_validates_next_effective_settings_not_old_active_override():
    with api(overrides={"attendance_face_window": 3, "attendance_face_interval": 2},
             dotenv="ATTENDANCE_FACE_WINDOW=1\nATTENDANCE_FACE_INTERVAL=0.25\n") as f:
        before = f.path.read_bytes()
        save(f, reset_keys=["attendance_face_window"], status=422)
        assert f.path.read_bytes() == before
        view = save(f, reset_keys=["attendance_face_window", "attendance_face_interval"])
        assert view["pending"]["attendance_face_window"] == 1
        assert view["pending"]["attendance_face_interval"] == .25


@test
def test_stale_cas_and_external_byte_only_edit_cannot_overwrite_file():
    with api(overrides={"preview_fps": 9}) as f:
        old = get(f)["version"]
        save(f, {"preview_fps": 12})
        before = f.path.read_bytes()
        save(f, {"ai_fps": 6}, version=old, status=409)
        assert f.path.read_bytes() == before
        baseline = get(f)["version"]
        f.path.write_bytes(before + b"\n")
        save(f, {"ai_fps": 6}, version=baseline, status=409)
        assert f.path.read_bytes() == before + b"\n"
        assert get(f)["version"] == hashlib.sha256(before + b"\n").hexdigest()


@test
def test_concurrent_services_with_same_version_have_exactly_one_winner():
    with api(overrides={"preview_fps": 9}) as f:
        peer = type(f.service)(f.active, f.engine, path=f.path, env_file=f.env_file)
        other = FastAPI()
        other.state.settings_service = peer
        other.include_router(feature("app.api.settings").router)
        version = get(f)["version"]
        barrier = threading.Barrier(2)

        def writer(client, fps):
            barrier.wait(timeout=10)
            return client.request("PUT", "/api/settings", {
                "expected_version": version, "values": {"preview_fps": fps}, "reset_keys": [],
            })[0]

        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(writer, client, value) for client, value in
                       ((f.client, 11), (Client(other), 12))]
            assert sorted(future.result(timeout=20) for future in futures) == [200, 409]
        assert json.loads(f.path.read_text())["values"]["preview_fps"] in (11, 12)


@test
def test_atomic_replace_failure_preserves_old_document_and_active_state():
    with api(overrides={"preview_fps": 9}) as f:
        before, active = f.path.read_bytes(), f.active.model_dump()
        with patch("os.replace", side_effect=OSError(TOKEN + URL)):
            no_secrets(save(f, {"preview_fps": 11}, status=503))
        assert f.path.read_bytes() == before and f.active.model_dump() == active
        assert get(f)["pending"]["preview_fps"] == 9
        assert not list(f.root.glob("*.tmp")), "Failed write left an override temporary file"


@test
def test_atomic_fsync_failure_preserves_old_document():
    with api(overrides={"preview_fps": 9}) as f:
        before = f.path.read_bytes()
        with patch("os.fsync", side_effect=OSError(AUTH)):
            no_secrets(save(f, {"preview_fps": 11}, status=503))
        assert f.path.read_bytes() == before


@test
def test_secret_omission_replacement_clear_and_reset_never_expose_token():
    with api(overrides={"telegram_bot_token": TOKEN},
             dotenv=f"TELEGRAM_BOT_TOKEN={OTHER_TOKEN}\n") as f:
        initial = get(f)
        no_secrets(initial)
        assert "telegram_bot_token" not in initial["active"]
        assert "telegram_bot_token" not in initial["pending"]
        assert initial["active"]["has_telegram_bot_token"] is True
        save(f, {"preview_fps": 13})
        assert json.loads(f.path.read_text())["values"]["telegram_bot_token"] == TOKEN
        replaced = save(f, {"telegram_bot_token": OTHER_TOKEN})
        assert replaced["active"]["has_telegram_bot_token"] is True
        assert replaced["pending"]["has_telegram_bot_token"] is True
        assert replaced["restart_required"] is True and "telegram_bot_token" in replaced["changed_keys"]
        no_secrets(replaced)
        cleared = save(f, {"telegram_bot_token": ""})
        assert cleared["pending"]["has_telegram_bot_token"] is False
        assert f.active.telegram_bot_token == TOKEN
        reset = save(f, reset_keys=["telegram_bot_token"])
        assert reset["pending"]["has_telegram_bot_token"] is True
        assert "telegram_bot_token" not in json.loads(f.path.read_text())["values"]
        assert load(f.path, f.env_file).telegram_bot_token == OTHER_TOKEN
        no_secrets(reset)


@test
def test_nonempty_token_replacement_alone_requires_restart_despite_same_presence():
    with api(overrides={"telegram_bot_token": TOKEN}) as f:
        changed = save(f, {"telegram_bot_token": OTHER_TOKEN})
        assert changed["restart_required"] is True
        assert changed["changed_keys"] == ["telegram_bot_token"]
        assert get(f)["restart_required"] is True
        no_secrets(changed)


@test
def test_api_validation_never_echoes_credentials_from_nested_or_malformed_payloads():
    with api() as f:
        version = get(f)["version"]
        payloads = [
            {"expected_version": version, "values": {"telegram_bot_token": {"token": TOKEN}}},
            {"expected_version": version, "values": {"preview_fps": URL, "telegram_bot_token": TOKEN}},
            {"expected_version": version, "values": {"telegram_bot_token": TOKEN}, "extra": AUTH},
            {"expected_version": {"password": PASSWORD}, "values": {}},
            {"expected_version": version, "reset_keys": [{"cookie": COOKIE}], "values": {}},
        ]
        for payload in payloads:
            no_secrets(f.client.json("PUT", "/api/settings", payload, status=422))
        no_secrets(f.client.json("PUT", "/api/settings", body=b'{"values":"' + TOKEN.encode(),
                                headers={"content-type": "application/json"}, status=422))
        assert not f.path.exists()


@test
def test_operator_header_origin_and_no_http_restart_or_history_reset():
    with api() as f:
        save(f, {"preview_fps": 12}, operator=False, status=403)
        save(f, {"preview_fps": 12}, headers={"origin": "http://evil.test"}, status=403)
        save(f, {"preview_fps": 12}, headers={"origin": "http://test"})
        for url in ("/api/settings/restart", "/api/debug/reset", "/api/debug/exec"):
            status, _headers, _body = f.client.request("POST", url, {"command": "ignored"})
            assert status in (404, 405)
        assert f.active.preview_fps == 10


@contextmanager
def diagnostics():
    module = feature("app.services.diagnostics_service")
    debug_api = feature("app.api.debug")
    with api(env={"AI_ENABLED": "false", "TELEGRAM_ENABLED": "false"}) as f, ExitStack() as stack:
        from app.rtsp.manager import ReaderManager
        from app.services.presence_recorder import PresenceRecorder
        from app.services.recording_service import RecordingManager
        # Tool resolution is an OS boundary; no child process is ever started.
        with patch("app.rtsp.manager.resolve_ffmpeg", return_value=(None, None)):
            manager = ReaderManager()
        with Session(f.engine) as db:
            db.add(Camera(id=1, name="Отключённая камера", nvr_host="127.0.0.1", username=USERNAME,
                          password=PASSWORD, channel=1, enabled=False))
            db.commit()
        state = f.app.state
        state.manager, state.ai_worker, state.face_engine = manager, None, None
        state.presence = PresenceRecorder(f.engine)
        state.model_bundle, state.gallery_migration = None, None
        state.legacy_model_settings_ignored = ()
        state.recordings = RecordingManager(f.engine, f.root / "recordings", None, None)
        stack.callback(state.recordings.shutdown)
        state.recording_error, state.playbacks, state.playback_error = None, None, None
        state.diagnostics = module.DiagnosticsService(state, f.engine, f.active)
        f.app.include_router(debug_api.router)
        yield f


@test
def test_journal_scene_diagnostics_never_export_tokens_images_or_raw_failure_text():
    with diagnostics() as f:
        raw = dict(state="JOURNAL_DEGRADED", error_code=URL, pending_commands=4,
                   pending_updates=float("nan"), pending_barriers=True,
                   blocked_generations=2, reserved_generations=3, active_visits=5,
                   pending_acknowledgements=1, accepting=False, writer_alive=True,
                   thumbnail=TOKEN, snapshot=COOKIE, visit_id=AUTH)
        f.app.state.human_journal = SimpleNamespace(health_snapshot=lambda: raw)
        f.app.state.scene_segmenter = SimpleNamespace(last_inference_seconds=float("nan"))
        f.app.state.scene_assistance = SimpleNamespace(health_snapshot=lambda: {
            "running": 1, "queued": 1, "records": 2, "accepting": True, "worker_alive": True,
            "snapshot": TOKEN, "job_id": COOKIE})
        f.app.state.scene_error = URL
        view = f.client.json("GET", "/api/debug/status")
        no_secrets(view)
        assert view["journal"]["state"] == "JOURNAL_DEGRADED"
        assert view["journal"]["error_code"] == "JOURNAL_DEGRADED"
        assert view["journal"]["pending_commands"] == 4
        assert view["journal"]["pending_updates"] is None and view["journal"]["pending_barriers"] is None
        assert view["scene"]["error_code"] == "SCENE_MODEL_UNAVAILABLE"
        assert view["scene"]["last_inference_seconds"] is None
        assert view["scene"]["running"] == 1 and view["scene"]["queued"] == 1
        assert not {"snapshot", "job_id", "thumbnail", "visit_id"} & view["journal"].keys()
        assert not {"snapshot", "job_id"} & view["scene"].keys()


@test
def test_diagnostics_reports_actual_ai_off_camera_off_and_time_tracking_state():
    with diagnostics() as f:
        view = f.client.json("GET", "/api/debug/status")
        assert view["tools"]["ffmpeg_available"] is False
        assert view["tools"]["ffprobe_available"] is False
        assert view["ai"]["enabled"] is False
        assert not any(view["ai"]["providers"].values())
        assert view["time_tracking"]["cameras"] == 1
        assert view["time_tracking"]["time_tracking_cameras"] == 0
        assert view["time_tracking"]["time_tracking_online"] == 0
        assert view["time_tracking"]["recorder"]["pending_crossings"] == 0
        assert view["time_tracking"]["recorder"]["last_error"] is None
        camera = next(row for row in view["cameras"] if row["id"] == 1)
        assert camera["status"] == "OFFLINE" and camera["reconnect_count"] == 0
        assert {"current_fps", "seconds_since_last_frame", "error"} <= camera.keys()
        assert view["recordings"]["sessions"] == []
        assert view["recordings"]["storage"]["oldest_available"] is None
        assert view["updated_at"]
        no_secrets(view)


@test
def test_diagnostics_reports_missing_models_without_constructing_sessions():
    with diagnostics() as f:
        view = f.client.json("GET", "/api/debug/status")
        models = view["models"]
        assert models["bundle_id"] is None and models["face_model_id"] is None
        assert models["error_code"] == "MODEL_MISSING"
        assert models["component_status"] == {
            key: "MODEL_MISSING" for key in ("person", "face_detection", "face_recognition", "body", "scene")
        }
        assert models["gallery_migration"] is None
        assert models["legacy_model_settings_ignored"] == []
        no_secrets(view)


@test
def test_model_diagnostics_projects_only_safe_ids_codes_counts_and_ignored_setting_names():
    from attendance_test_support import model_bundle_fixture
    with diagnostics() as f:
        bundle = model_bundle_fixture()
        bundle.metadata["private_path"] = URL
        bundle.component_paths["person"] = Path(USERNAME) / PASSWORD
        bundle.component_status.update(body=URL, scene="MODEL_HASH_MISMATCH", **{AUTH: COOKIE})
        f.app.state.model_bundle = bundle
        f.app.state.gallery_migration = {
            "face_model_id": "b" * 64, "alignment_id": "arcface-5pt-112-v1",
            "stored_faces_count": 6, "usable_faces_count": 4, "faces_reencoded": 3,
            "faces_need_reenrollment": 2, "employees_needing_photos": [1, 2, TOKEN],
            "employee_names": [USERNAME], "source_url": URL, "embedding": PASSWORD,
            "error_code": "GALLERY_CHANGED",
        }
        f.app.state.legacy_model_settings_ignored = (
            "YOLO_MODEL", "INSIGHTFACE_MODEL", "INSIGHTFACE_ROOT", "REID_MODEL",
            "TELEGRAM_BOT_TOKEN", TOKEN, URL, "insightface_model",
        )
        view = f.client.json("GET", "/api/debug/status")
        models = view["models"]
        assert models["bundle_id"] == "a" * 64 and models["face_model_id"] == "b" * 64
        assert models["alignment_id"] == "arcface-5pt-112-v1"
        assert models["component_status"]["body"] == "MODEL_CONTRACT_INVALID"
        assert models["component_status"]["scene"] == "MODEL_HASH_MISMATCH"
        assert models["error_code"] is None
        assert models["gallery_migration"] == {
            "face_model_id": "b" * 64, "alignment_id": "arcface-5pt-112-v1",
            "stored_faces_count": 6, "usable_faces_count": 4, "faces_reencoded": 3,
            "faces_need_reenrollment": 2, "employees_needing_photos_count": 3,
            "error_code": "GALLERY_CHANGED",
        }
        assert models["legacy_model_settings_ignored"] == [
            "YOLO_MODEL", "INSIGHTFACE_MODEL", "INSIGHTFACE_ROOT", "REID_MODEL",
        ]
        assert models["legacy_model_settings_code"] == "LEGACY_MODEL_SETTINGS_IGNORED"
        no_secrets(view)
        assert "employees_needing_photos" not in models["gallery_migration"]
        assert "metadata" not in models and "component_paths" not in models
        bundle.bundle_id, bundle.face_model_id, bundle.alignment_id = URL, TOKEN, PASSWORD
        f.app.state.gallery_migration = {
            "face_model_id": AUTH, "alignment_id": COOKIE, "usable_faces_count": float("inf"),
            "employees_needing_photos": URL, "error_code": URL,
        }
        invalid = f.client.json("GET", "/api/debug/export")
        no_secrets(invalid)
        assert invalid["status"]["models"]["bundle_id"] is None
        assert invalid["status"]["models"]["face_model_id"] is None
        assert invalid["status"]["models"]["alignment_id"] is None
        assert invalid["status"]["models"]["gallery_migration"]["usable_faces_count"] is None
        assert invalid["status"]["models"]["gallery_migration"]["employees_needing_photos_count"] is None


@test
def test_model_failure_codes_remain_static_in_support_events():
    module = feature("app.services.diagnostics_service")
    ring = module.DiagnosticRing()
    codes = ("MODEL_MISSING", "MODEL_HASH_MISMATCH", "MODEL_CONTRACT_INVALID",
             "FACE_MODEL_UNAVAILABLE", "GALLERY_UNAVAILABLE", "GALLERY_CHANGED",
             "JOURNAL_DEGRADED", "CAPACITY_DEGRADED", "DECODE_SOFTWARE_FALLBACK",
             "SCENE_MODEL_UNAVAILABLE", "ENROLLMENT_BUSY")
    for code in codes:
        ring.emit(logging.LogRecord("app.ai", logging.WARNING, __file__, 1, code, (), None))
    assert [item["code"] for item in ring.export()["events"]] == list(codes)


@test
def test_diagnostics_isolates_individual_failures_and_full_disk_remains_visible():
    with diagnostics() as f:
        state = f.app.state
        with patch.object(state.presence, "snapshot", side_effect=RuntimeError(URL + AUTH)), \
                patch("app.media.storage.shutil.disk_usage", return_value=SimpleNamespace(free=0)):
            view = f.client.json("GET", "/api/debug/status")
        assert view["time_tracking"]["error_code"] == "DIAGNOSTICS_UNAVAILABLE"
        assert view["time_tracking"]["time_tracking_cameras"] == 0
        assert view["recordings"]["storage"]["free_bytes"] == 0
        assert view["cameras"][0]["status"] == "OFFLINE"
        no_secrets(view)
        with patch.object(state.recordings, "snapshot", side_effect=OSError(PASSWORD)):
            view = f.client.json("GET", "/api/debug/status")
        assert view["recordings"]["error_code"] == "DIAGNOSTICS_UNAVAILABLE"
        assert view["time_tracking"]["time_tracking_cameras"] == 0
        no_secrets(view)


@test
def test_diagnostic_ring_bounds_sequence_pagination_and_rejects_invalid_queries():
    with diagnostics() as f:
        ring = f.app.state.diagnostics.ring
        for index in range(205):
            ring.emit(logging.LogRecord("app.rtsp.reader", logging.WARNING, __file__, 1,
                                        "STREAM_UNAVAILABLE %s", (index,), None))
        first = f.client.json("GET", "/api/debug/events?after=0&limit=100")
        assert len(first["events"]) == 100
        seq = [item["sequence"] for item in first["events"]]
        assert seq == list(range(6, 106))
        second = f.client.json("GET", "/api/debug/events?after=105&limit=100")
        assert [item["sequence"] for item in second["events"]] == list(range(106, 206))
        assert f.client.json("GET", "/api/debug/events?after=205")["events"] == []
        for query in ("limit=101", "limit=0", "after=-1", "limit=not-a-number"):
            f.client.json("GET", "/api/debug/events?" + query, status=422)


@test
def test_log_ring_sanitizes_before_retention_and_never_keeps_raw_record_or_exception():
    module = feature("app.services.diagnostics_service")
    ring = module.DiagnosticRing()

    class Sensitive:
        def __str__(self):
            return URL + " Authorization: Bearer " + AUTH + " Cookie: session=" + COOKIE

    value = Sensitive()
    retained = weakref.ref(value)
    try:
        raise RuntimeError(PASSWORD)
    except RuntimeError:
        record = logging.LogRecord("app.media", logging.ERROR, __file__, 1,
                                   "unexpected %s", (value,), sys.exc_info())
        record.authorization = AUTH
        record.cookie = COOKIE
        record.body = {"telegram_bot_token": TOKEN}
        ring.emit(record)
    record_ref = weakref.ref(record)
    del value, record
    gc.collect()
    assert retained() is None and record_ref() is None, "Ring retained a raw logging payload"
    result = ring.events(after=0, limit=100)
    assert len(result["events"]) == 1
    assert result["events"][0]["code"] == "UNEXPECTED_ERROR"
    no_secrets(result)
    # Inspect the retained sanitized buffer too, not just its export projection.
    no_secrets(vars(ring))


@test
def test_export_allowlists_status_events_and_settings_without_secret_values():
    with diagnostics() as f:
        save(f, {"telegram_bot_token": TOKEN})
        ring = f.app.state.diagnostics.ring
        ring.emit(logging.LogRecord("app.media", logging.ERROR, __file__, 1,
                                    "Authorization: %s Cookie: %s %s", (AUTH, COOKIE, URL), None))
        bundle = f.client.json("GET", "/api/debug/export")
        assert {"status", "events", "settings"} <= bundle.keys()
        assert bundle["settings"]["pending"]["has_telegram_bot_token"] is True
        assert bundle["status"]["time_tracking"]["recorder"]["open_segments"] == 0
        no_secrets(bundle)
        assert f.active.database_url not in json.dumps(bundle)


@test
def test_reader_unexpected_network_error_and_server_logs_hide_complete_credentials():
    from app.rtsp.reader import RTSPReader
    reader = RTSPReader(1, "Camera", URL, source_revision=1, ffmpeg_path="not-started",
                        ffprobe_path="not-started")
    records = []

    class Capture(logging.Handler):
        def emit(self, record):
            records.append(self.format(record))

    def external_failure():
        reader._stop_event.set()
        raise RuntimeError(URL + " Authorization: " + AUTH)

    logger = logging.getLogger("app.rtsp.reader")
    handler, old_level = Capture(), logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    try:
        with patch.object(reader, "_connect_and_read", side_effect=external_failure):
            reader._run()
    finally:
        logger.removeHandler(handler)
        logger.setLevel(old_level)
    assert reader.snapshot()["error"] == "STREAM_UNAVAILABLE"
    no_secrets(reader.snapshot())
    no_secrets(records)


@test
def test_diagnostics_projects_scalar_allowlists_not_runtime_objects_or_raw_errors():
    from app.rtsp.reader import CameraStatus, RTSPReader
    with diagnostics() as f:
        reader = RTSPReader(1, "Camera", URL, source_revision=1, ffmpeg_path=None, ffprobe_path=None)
        reader.last_error = URL + " Authorization: " + AUTH
        reader.resolution, reader.codec = URL, COOKIE
        reader.current_fps = float("nan")
        reader._set_status(CameraStatus.ERROR)
        reader.command_args = ["ffmpeg", "-i", URL]
        f.app.state.manager._readers[1] = reader
        f.app.state.ai_providers = {
            "person_detection": ["CPUExecutionProvider", TOKEN],
            "scene_encoder": ["CPUExecutionProvider"], "scene_decoder": ["CPUExecutionProvider"],
            "reid": ["CPUExecutionProvider"],
        }
        view = f.client.json("GET", "/api/debug/status")
        no_secrets(view)
        assert view["ai"]["providers"]["person_detection"] == ["CPUExecutionProvider"]
        assert view["ai"]["providers"]["scene_encoder"] == ["CPUExecutionProvider"]
        assert view["ai"]["providers"]["scene_decoder"] == ["CPUExecutionProvider"]
        camera = next(row for row in view["cameras"] if row["id"] == 1)
        assert camera["error"] == "STREAM_UNAVAILABLE"
        assert camera["current_fps"] is None
        assert "command_args" not in camera and "rtsp_url" not in camera


@test
def test_real_discovery_outcome_is_visible_in_diagnostics_without_credentials():
    from app.api import cameras
    from app.services.dahua_service import DahuaClient, DahuaError
    with diagnostics() as f:
        f.app.include_router(cameras.router)
        payload = {"nvr_host": "127.0.0.1", "username": USERNAME, "password": PASSWORD,
                   "rtsp_port": 554, "http_scheme": "http", "http_port": 80}
        # Replace only the external HTTP boundary: real parser, API and diagnostics.
        with patch.object(DahuaClient, "_channel_config", return_value="table.ChannelTitle[0].Name=Door"):
            result = f.client.json("POST", "/api/cameras/discover", payload)
        assert result["channels"] == [{"channel": 1, "name": "Door"}]
        view = f.client.json("GET", "/api/debug/status")
        assert view["archive"]["channel_titles"] is True and view["archive"]["error_code"] is None
        with patch.object(DahuaClient, "_channel_config", side_effect=DahuaError("DEVICE_AUTH_FAILED")):
            f.client.json("POST", "/api/cameras/discover", payload)
        view = f.client.json("GET", "/api/debug/status")
        assert view["archive"]["error_code"] == "DEVICE_AUTH_FAILED"
        no_secrets(view)


def formatted_server_record(formatter, record):
    """Exercise the installed formatter via real handler/filter error handling."""
    import io
    from contextlib import redirect_stderr
    from app.services.diagnostics_service import SafeLogFilter
    output, errors = io.StringIO(), io.StringIO()
    handler = logging.StreamHandler(output)
    handler.setFormatter(formatter)
    handler.addFilter(SafeLogFilter())
    try:
        with redirect_stderr(errors):
            handler.handle(record)
    finally:
        handler.close()
    assert not errors.getvalue(), "Sanitized server record caused a logging formatter error"
    return output.getvalue()


@test
def test_real_uvicorn_access_formatter_preserves_http_status_without_request_secrets():
    from uvicorn.logging import AccessFormatter
    from urllib.parse import urlencode
    query = urlencode({"token": TOKEN, "authorization": AUTH, "cookie": COOKIE, "url": URL})
    for colored in (False, True):
        record = logging.LogRecord("uvicorn.access", logging.INFO, __file__, 1,
            '%s - "%s %s HTTP/%s" %d',
            ("127.0.0.1:54321", "GET", "/api/debug/status?" + query, "1.1", 206), None)
        record.headers = {"Authorization": "Bearer " + AUTH, "Cookie": COOKIE}
        record.body = {"telegram_bot_token": TOKEN}
        result = formatted_server_record(AccessFormatter(
            '%(levelprefix)s %(client_addr)s - "%(request_line)s" %(status_code)s',
            use_colors=colored), record)
        assert "GET" in result and "/api/debug/status" in result
        assert "206" in result and "HTTP/1.1" in result
        no_secrets(result)
        # Query encoding must not turn concealment into a literal-only test.
        assert query not in result and "TEST_ONLY" not in result
        no_secrets(vars(record))


@test
def test_real_uvicorn_colored_default_formatter_keeps_numeric_pid_and_redacts_color_message():
    from uvicorn.logging import DefaultFormatter
    record = logging.LogRecord("uvicorn.error", logging.INFO, __file__, 1,
                               "Started server process [%d]", (73129,), None)
    record.color_message = "Started server process [\033[36m%d\033[0m]"
    result = formatted_server_record(DefaultFormatter("%(levelprefix)s %(message)s", use_colors=True), record)
    assert "73129" in result and "%d" not in result
    # The alternate color template is another rendering path, not trusted content.
    record = logging.LogRecord("uvicorn.error", logging.INFO, __file__, 1,
        "Server listening on %s port %d", (URL, 8765), None)
    record.color_message = "Server %s port %d Authorization: " + AUTH + " Cookie: " + COOKIE
    result = formatted_server_record(DefaultFormatter("%(levelprefix)s %(message)s", use_colors=True), record)
    assert "8765" in result and "%s" not in result and "%d" not in result
    no_secrets(result)
    no_secrets(vars(record))


@test
def test_real_uvicorn_error_formatter_discards_exception_and_cached_rendering_context():
    from uvicorn.logging import DefaultFormatter
    try:
        raise RuntimeError(URL + " Authorization: " + AUTH)
    except RuntimeError as exc:
        record = logging.LogRecord("uvicorn.error", logging.ERROR, __file__, 1,
                                   "ASGI failure: %s", (exc,), sys.exc_info())
        record.color_message = "ASGI failure %s " + TOKEN
        record.exc_text = "cached traceback " + PASSWORD
        record.stack_info = "cached stack " + COOKIE
        record.headers = {"authorization": AUTH}
        result = formatted_server_record(DefaultFormatter("%(levelprefix)s %(message)s", use_colors=True), record)
    assert "UNEXPECTED_ERROR" in result
    no_secrets(result)
    no_secrets(vars(record))


def startup_logging_output(mode):
    """Run real entrypoint/Config ordering, replacing only server socket serving."""
    import io
    import runpy
    import uvicorn
    from contextlib import redirect_stdout, redirect_stderr
    from app.database import database as database_module
    output, errors = io.StringIO(), io.StringIO()
    observed = []

    def emit_requests(server=None, *args, **kwargs):
        if server is not None:
            server.started = True  # The network boundary has been replaced, not Config.
        logging.getLogger("uvicorn.error").info(
            "Started server process [%d]", 73129,
            extra={"color_message": "Started server process [\033[36m%d\033[0m]"})
        access = logging.LogRecord("uvicorn.access", logging.INFO, __file__, 1,
            '%s - "%s %s HTTP/%s" %d',
            ("127.0.0.1:54321", "GET",
             "/api/debug/status?token=" + TOKEN + "&authorization=" + AUTH + "&cookie=" + COOKIE,
             "1.1", 206), None)
        access.headers = {"Authorization": "Bearer " + AUTH, "Cookie": COOKIE}
        access.body = {"password": PASSWORD}
        logging.getLogger("uvicorn.access").handle(access)
        observed.append(access)
        try:
            raise RuntimeError(URL + " Authorization: " + AUTH)
        except RuntimeError as exc:
            failure = logging.LogRecord("uvicorn.error", logging.ERROR, __file__, 1,
                "ASGI failure: %s", (exc,), sys.exc_info())
            failure.color_message = "Failure %s " + TOKEN
            failure.headers = {"Cookie": COOKIE}
            logging.getLogger("uvicorn.error").handle(failure)
            observed.append(failure)

    with database() as (engine, root), ExitStack() as stack:
        stack.enter_context(patch.dict(os.environ, {
            "DATABASE_URL": f"sqlite:///{(root / 'startup.db').as_posix()}",
            "OPERATOR_SETTINGS_PATH": str(root / "operator-settings.json"),
            "RECORDING_ROOT": str(root / "recordings"),
            "AI_ENABLED": "false", "TELEGRAM_ENABLED": "false",
        }, clear=True))
        active = config.Settings(_env_file=None, app_host="127.0.0.1", app_port=0)
        stack.enter_context(patch.object(config, "settings", active))
        stack.enter_context(patch.object(database_module, "engine", engine))
        # Fresh process-equivalent handler state; restore caller's logging afterwards.
        for name in ("", "uvicorn", "uvicorn.error", "uvicorn.access", "app"):
            logger = logging.getLogger(name)
            for key, value in (("handlers", []), ("filters", []), ("level", logging.NOTSET),
                               ("propagate", True), ("disabled", False)):
                stack.enter_context(patch.object(logger, key, value))
        stack.enter_context(redirect_stdout(output))
        stack.enter_context(redirect_stderr(errors))
        if mode == "programmatic":
            stack.enter_context(patch.object(uvicorn.Server, "run", emit_requests))
            runpy.run_path(str(config.BASE_DIR / "main.py"), run_name="__main__")
        else:
            # Config precedes application import in the supported CLI path.
            stack.enter_context(patch.dict(sys.modules))
            sys.modules.pop("main", None)
            server_config = uvicorn.Config("main:app", host="127.0.0.1", port=0,
                                           lifespan="off", use_colors=True)
            server_config.load_app()
            emit_requests()
    rendered = output.getvalue() + errors.getvalue()
    assert "--- Logging error ---" not in rendered
    assert "GET" in rendered and "/api/debug/status" in rendered and "206" in rendered
    assert "73129" in rendered and "%d" not in rendered
    no_secrets(rendered)
    assert "UNEXPECTED_ERROR" in rendered and "Traceback (most recent call last)" not in rendered
    for record in observed:
        no_secrets(vars(record))


@test
def test_programmatic_main_startup_preserves_sanitized_logging_after_real_uvicorn_config():
    startup_logging_output("programmatic")


@test
def test_uvicorn_import_startup_preserves_sanitized_logging_after_real_config_load_app():
    startup_logging_output("import")


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
