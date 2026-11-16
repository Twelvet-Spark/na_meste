"""Portable setup format contracts; no app startup, devices or production DB."""
import base64
import copy
import io
import json
import struct
import sys
import traceback
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from PIL import Image
from app import setup_transfer as transfer

TESTS = []
PASSWORD = " перенос-secret-2026 "


def test(fn):
    TESTS.append(fn)
    return fn


def document():
    image = io.BytesIO()
    Image.new("RGB", (112, 112), "navy").save(image, "JPEG")
    return {
        "transfer_version": 1, "exported_at": "2026-11-16T10:00:00+00:00",
        "attendance_schema_version": 1, "operator_schema_version": 3,
        "embedding_model": "buffalo_l", "embedding_encoding": "float32-le-512",
        "cameras": [{"id": 7, "name": "Магазин", "nvr_host": "127.0.0.1", "rtsp_port": 554,
                     "username": "viewer", "password": "sentinel-device-password", "channel": 2,
                     "stream_type": "sub", "enabled": True, "created_at": "2026-01-01T00:00:00Z",
                     "updated_at": "2026-01-02T00:00:00Z"}],
        "archives": [{"camera_id": 7, "http_scheme": "http", "http_port": 80, "utc_offset_minutes": 180}],
        "calibrations": [{"camera_id": 7, "role": "ENTRY", "line_start": [0.2, 0.3],
                          "line_end": [0.8, 0.3], "inside_point": [0.5, 0.9],
                          "reference_width": 640, "reference_height": 480}],
        "workplaces": [{"id": 9, "name": "Москва", "active": False}],
        "employees": [{"id": 4, "name": "Legacy Имя", "first_name": None, "last_name": None,
                       "external_id": "employee-4", "active": True,
                       "created_at": "2026-01-01T00:00:00Z", "updated_at": "2026-01-02T00:00:00Z"}],
        "faces": [{"id": 12, "employee_id": 4,
                   "embedding": base64.b64encode(struct.pack("<512f", 1., *([0.] * 511))).decode(),
                   "thumbnail": base64.b64encode(image.getvalue()).decode(), "created_at": "2026-01-01T00:00:00Z"}],
        "assignments": [{"employee_id": 4, "workplace_id": 9}],
    }


def parse(data):
    return transfer.validate_document(json.dumps(data, ensure_ascii=False).encode(), expected_model="buffalo_l")


def refused(fn, code=None):
    try:
        fn()
    except transfer.TransferError as exc:
        assert "sentinel-device-password" not in str(exc)
        assert PASSWORD not in str(exc)
        if code:
            assert exc.code == code, exc.code
        return
    raise AssertionError("unsafe transfer accepted")


@test
def roundtrip_preserves_setup_and_hides_credentials():
    source = parse(document())
    first = transfer.encode_transfer(source, PASSWORD)
    second = transfer.encode_transfer(source, PASSWORD)
    assert first != second
    assert b"sentinel-device-password" not in first
    assert "sentinel-device-password" not in repr(source)
    result = transfer.decode_transfer(first, PASSWORD, expected_model="buffalo_l")
    assert result.model_dump(mode="json") == source.model_dump(mode="json")
    assert result.cameras[0].password == "sentinel-device-password"
    assert result.workplaces[0].active is False
    assert result.employees[0].first_name is None
    assert result.counts().model_dump() == {"cameras": 1, "workplaces": 1, "employees": 1, "faces": 1, "assignments": 1}


@test
def password_and_corruption_are_authenticated():
    blob = transfer.encode_transfer(parse(document()), PASSWORD)
    refused(lambda: transfer.decode_transfer(blob, "wrong-password-2026", expected_model="buffalo_l"))
    for index in (0, 9, 25, 40, len(blob)-1):
        damaged = bytearray(blob)
        damaged[index] ^= 1
        refused(lambda: transfer.decode_transfer(bytes(damaged), PASSWORD, expected_model="buffalo_l"))
    refused(lambda: transfer.decode_transfer(blob[:-10], PASSWORD, expected_model="buffalo_l"))
    refused(lambda: transfer.encode_transfer(parse(document()), "short"), "TRANSFER_PASSWORD_INVALID")
    refused(lambda: transfer.decode_transfer(blob, PASSWORD.strip(), expected_model="buffalo_l"))


@test
def schema_model_and_unknown_fields_fail_closed():
    for key, value in (("transfer_version", 2), ("operator_schema_version", 999),
                       ("attendance_schema_version", 999), ("embedding_model", "other"),
                       ("embedding_encoding", "pickle"), ("videos", [])):
        data = document()
        data[key] = value
        refused(lambda: parse(data))
    refused(lambda: transfer.validate_document(b'{"a":1,"a":2}', expected_model="buffalo_l"))
    refused(lambda: transfer.validate_document(b'[' * 17 + b'0' + b']' * 17, expected_model="buffalo_l"))


@test
def malformed_links_geometry_and_fields_are_rejected():
    mutations = [
        lambda d: d["employees"].append(copy.deepcopy(d["employees"][0])),
        lambda d: d["faces"][0].update(employee_id=999),
        lambda d: d["assignments"].append(copy.deepcopy(d["assignments"][0])),
        lambda d: d["workplaces"].append({"id": 10, "name": "МОСКВА", "active": True}),
        lambda d: d["cameras"][0].update(id=True),
        lambda d: d["cameras"][0].update(nvr_host="http://bad/path"),
        lambda d: d["calibrations"][0].update(line_end=[0.201, 0.3]),
        lambda d: d["calibrations"][0].update(inside_point=[0.5, 0.301]),
        lambda d: d["cameras"][0].update(channel=65),
        lambda d: d["archives"][0].update(utc_offset_minutes=841),
        lambda d: d["calibrations"][0].update(line_start=[float("nan"), 0]),
        lambda d: d["calibrations"][0].update(reference_width=0),
        lambda d: d["employees"][0].update(created_at="not-a-date"),
    ]
    for mutation in mutations:
        data = document()
        mutation(data)
        refused(lambda: parse(data))


@test
def invalid_biometrics_cannot_enter_gallery():
    for raw in (b"bad", struct.pack("<512f", *([0.] * 512)), struct.pack("<512f", float("nan"), *([0.] * 511))):
        data = document()
        data["faces"][0]["embedding"] = base64.b64encode(raw).decode()
        refused(lambda: parse(data))
    for value in ("not base64!", base64.b64encode(b"not an image").decode()):
        data = document()
        data["faces"][0]["thumbnail"] = value
        refused(lambda: parse(data))


@test
def legacy_missing_samples_and_limits_are_not_silent_truncation():
    data = document()
    data["faces"][0]["thumbnail"] = None
    parse(data)
    data["faces"] = []
    assert parse(data).counts().faces == 0
    with patch.object(transfer, "MAX_FILE_BYTES", 100):
        refused(lambda: parse(document()), "TRANSFER_TOO_LARGE")
        refused(lambda: transfer.decode_transfer(b"x" * 101, PASSWORD, expected_model="buffalo_l"), "TRANSFER_TOO_LARGE")
    data = document()
    data["cameras"] *= 10001
    refused(lambda: parse(data))


if __name__ == "__main__":
    failed = 0
    for fn in TESTS:
        try:
            fn()
            print("PASS", fn.__name__)
        except Exception:
            failed += 1
            traceback.print_exc()
    print(f"{len(TESTS)-failed}/{len(TESTS)} passed")
    raise SystemExit(bool(failed))
