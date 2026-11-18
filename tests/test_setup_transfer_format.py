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
MODEL_ID = "1" * 64
ALIGNMENT_ID = "arcface-5pt-112-v1"


def test(fn):
    TESTS.append(fn)
    return fn


def document():
    image = io.BytesIO()
    Image.new("RGB", (112, 112), "navy").save(image, "JPEG")
    return {
        "transfer_version": 4, "exported_at": "2026-11-16T10:00:00+00:00",
        "attendance_schema_version": 3, "operator_schema_version": 6,
        "face_model_id": MODEL_ID, "embedding_encoding": "float32-le-512",
        "cameras": [{"id": 7, "name": "Магазин", "nvr_host": "127.0.0.1", "rtsp_port": 554,
                     "username": "viewer", "password": "sentinel-device-password", "channel": 2,
                     "stream_type": "sub", "enabled": True, "created_at": "2026-01-01T00:00:00Z",
                     "updated_at": "2026-01-02T00:00:00Z"}],
        "archives": [{"camera_id": 7, "http_scheme": "http", "http_port": 80, "utc_offset_minutes": 180}],
        "calibrations": [{"camera_id": 7, "role": "BOTH", "line_start": [0.2, 0.3],
                          "line_end": [0.8, 0.3], "inside_point": [0.5, 0.9],
                          "door_kind": "DOOR", "door_top": [[0.2, 0.1], [0.8, 0.1]],
                          "reference_width": 640, "reference_height": 480}],
        "workplaces": [{"id": 9, "name": "Москва", "active": False}],
        "employees": [{"id": 4, "name": "Legacy Имя", "first_name": None, "last_name": None,
                       "external_id": "employee-4", "active": True,
                       "created_at": "2026-01-01T00:00:00Z", "updated_at": "2026-01-02T00:00:00Z"}],
        "faces": [{"id": 12, "employee_id": 4,
                   "embedding": base64.b64encode(struct.pack("<512f", 1., *([0.] * 511))).decode(),
                   "thumbnail": base64.b64encode(image.getvalue()).decode(), "created_at": "2026-01-01T00:00:00Z",
                   "face_model_id": MODEL_ID, "alignment_id": ALIGNMENT_ID}],
        "assignments": [{"employee_id": 4, "workplace_id": 9}],
        "scenes": [{"camera_id": 7, "reference_width": 1920, "reference_height": 1080,
                    "geometry": {"version": 2, "floor": [[0., 0.], [1., 0.], [1., 1.], [0., 1.]],
                                 "objects": [],
                                 "doors": [{"id": "00000000-0000-4000-8000-0000000000d1", "label": "Вход",
                                            "kind": "DOOR",
                                            "quad": [[0.4, 0.8], [0.6, 0.8], [0.6, 0.2], [0.4, 0.2]]}],
                                 "metric": None}}],
        "profiles": [{"camera_id": 7, "workplace_id": 9, "zone": "ENTRANCE", "ai_enabled": True,
                      "ai_fps": 2.5, "time_tracking": True}],
    }


def v3_document():
    """The previous contract: profiles without time tracking."""
    data = document()
    data.update(transfer_version=3, attendance_schema_version=2, operator_schema_version=5)
    del data["profiles"][0]["time_tracking"]
    return data


def v2_document():
    """The previous contract: one-direction gates and scenes without doors."""
    data = document()
    data.update(transfer_version=2, attendance_schema_version=1, operator_schema_version=4)
    del data["profiles"]
    calibration = data["calibrations"][0]
    calibration.update(role="ENTRY")
    del calibration["door_kind"], calibration["door_top"]
    geometry = data["scenes"][0]["geometry"]
    geometry.update(version=1)
    del geometry["doors"]
    return data


def legacy_document():
    data = v2_document()
    data.update(transfer_version=1, operator_schema_version=3, embedding_model="buffalo_l")
    del data["face_model_id"]
    del data["scenes"]
    for face in data["faces"]:
        del face["face_model_id"]
        del face["alignment_id"]
    return data


def parse(data):
    return transfer.validate_document(json.dumps(data, ensure_ascii=False).encode(),
                                      expected_face_model_id=MODEL_ID)


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
    result = transfer.decode_transfer(first, PASSWORD, expected_face_model_id=MODEL_ID)
    assert result.model_dump(mode="json") == source.model_dump(mode="json")
    assert result.cameras[0].password == "sentinel-device-password"
    assert result.workplaces[0].active is False
    assert result.employees[0].first_name is None
    assert result.counts().model_dump() == {"cameras": 1, "workplaces": 1, "employees": 1, "faces": 1,
                                            "assignments": 1, "scenes": 1, "profiles": 1}


@test
def password_and_corruption_are_authenticated():
    blob = transfer.encode_transfer(parse(document()), PASSWORD)
    refused(lambda: transfer.decode_transfer(blob, "wrong-password-2026", expected_face_model_id=MODEL_ID))
    for index in (0, 9, 25, 40, len(blob)-1):
        damaged = bytearray(blob)
        damaged[index] ^= 1
        refused(lambda: transfer.decode_transfer(bytes(damaged), PASSWORD, expected_face_model_id=MODEL_ID))
    refused(lambda: transfer.decode_transfer(blob[:-10], PASSWORD, expected_face_model_id=MODEL_ID))
    refused(lambda: transfer.encode_transfer(parse(document()), "short"), "TRANSFER_PASSWORD_INVALID")
    refused(lambda: transfer.decode_transfer(blob, PASSWORD.strip(), expected_face_model_id=MODEL_ID))


@test
def schema_model_and_unknown_fields_fail_closed():
    for key, value in (("transfer_version", 7), ("operator_schema_version", 999),
                       ("attendance_schema_version", 999), ("face_model_id", "2" * 64),
                       ("embedding_encoding", "pickle"), ("videos", [])):
        data = document()
        data[key] = value
        refused(lambda: parse(data))
    refused(lambda: transfer.validate_document(b'{"a":1,"a":2}', expected_face_model_id=MODEL_ID))
    refused(lambda: transfer.validate_document(b'[' * 17 + b'0' + b']' * 17, expected_face_model_id=MODEL_ID))


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
def invalid_binary_encoding_is_rejected_before_preparation():
    for raw in (b"bad", b"x" * 2049):
        data = document()
        data["faces"][0]["embedding"] = base64.b64encode(raw).decode()
        refused(lambda: parse(data))
    data = document()
    data["faces"][0]["thumbnail"] = "not base64!"
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
        refused(lambda: transfer.decode_transfer(b"x" * 101, PASSWORD, expected_face_model_id=MODEL_ID), "TRANSFER_TOO_LARGE")
    data = document()
    data["cameras"] *= 10001
    refused(lambda: parse(data))


@test
def ids_must_be_lossless_for_browser_links():
    data = document()
    unsafe = 9007199254740993
    data["employees"][0]["id"] = unsafe
    data["faces"][0]["employee_id"] = unsafe
    data["assignments"][0]["employee_id"] = unsafe
    refused(lambda: parse(data))
    safe = 9007199254740991
    data["employees"][0]["id"] = safe
    data["faces"][0]["employee_id"] = safe
    data["assignments"][0]["employee_id"] = safe
    assert int(float(parse(data).employees[0].id)) == safe


@test
def excessive_json_records_are_rejected_without_materialization_spike():
    import tracemalloc
    # 200,001 empty records occupy <1 MiB on disk but >14 MiB as Python dicts.
    payload = b'{"faces":[' + b'{},' * 200000 + b'{}]}'
    tracemalloc.start()
    try:
        refused(lambda: transfer.validate_document(payload, expected_face_model_id=MODEL_ID))
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert peak < 8 * 1024 * 1024, f"unbounded decoder allocation: {peak}"


@test
def legacy_decoder_accepts_package_name_without_comparing_target_space():
    data = legacy_document()
    data["embedding_model"] = "another_legacy-package.v1"
    restored = transfer.decode_transfer(transfer.encode_transfer(parse(data), PASSWORD), PASSWORD,
                                        expected_face_model_id=MODEL_ID)
    assert restored.transfer_version == 1 and restored.faces[0].id == 12
    for value in ("../buffalo_l", "https://example.test/model", "", "bad name", "a" * 101):
        data["embedding_model"] = value
        refused(lambda: parse(data), "TRANSFER_INVALID")


@test
def provenance_is_paired_and_every_populated_sample_must_match():
    for values in ({"face_model_id": None}, {"alignment_id": None}):
        data = document()
        data["faces"][0].update(values)
        refused(lambda: parse(data), "TRANSFER_INVALID")
    for values in ({"face_model_id": "2" * 64}, {"alignment_id": "other-alignment"}):
        data = document()
        data["faces"].append({**data["faces"][0], "id": 13, **values})
        # Mixed actual provenance can be exported, but cannot be imported into this space.
        source = transfer.validate_document(json.dumps(data).encode())
        blob = transfer.encode_transfer(source, PASSWORD)
        refused(lambda: transfer.decode_transfer(blob, PASSWORD, expected_face_model_id=MODEL_ID),
                "TRANSFER_MODEL_MISMATCH")
    data = document()
    data["faces"][0].update(face_model_id=None, alignment_id=None,
                             thumbnail=base64.b64encode(b"malformed aligned JPEG").decode())
    assert parse(data).faces[0].face_model_id is None
    for value in ("not-a-hash", "A" * 64):
        data["face_model_id"] = value
        refused(lambda: parse(data), "TRANSFER_INVALID")


@test
def scenes_accept_only_bounded_normalized_geometry_without_runtime_state():
    mutations = [
        lambda g: g.update(version=1),
        lambda g: g.update(version=True),
        lambda g: g.update(snapshot_id="secret"),
        lambda g: g.update(floor=[[0., 0.], [1., 0.]]),
        lambda g: g.update(floor=[[0., 0.], [1.1, 0.], [0., 1.]]),
        lambda g: g.update(floor=[[0., 0.], [float("inf"), 0.], [0., 1.]]),
        lambda g: g.update(floor=[[0., 0.], [1., 0.], [0., 1.]] * 43),
        lambda g: g.update(objects=[{"id": "bad", "label": "Стол", "polygon": [[0., 0.], [1., 0.], [0., 1.]]}]),
        lambda g: g.update(objects=[{"id": "00000000-0000-4000-8000-000000000001", "label": "x" * 81,
                                    "polygon": [[0., 0.], [1., 0.], [0., 1.]]}]),
        lambda g: g.update(objects=[{"id": f"00000000-0000-4000-8000-{i:012d}", "label": "Стол",
                                    "polygon": [[0., 0.], [1., 0.], [0., 1.]]} for i in range(33)]),
        lambda g: g.update(metric={"corners": [[0., 0.], [1., 0.], [1., 1.], [0., 1.]],
                                   "width_m": 0, "depth_m": 3.}),
        lambda g: g.update(metric={"corners": [[0., 0.], [1., 0.], [1., 1.], [0., 1.]],
                                   "width_m": 1000.01, "depth_m": 3.}),
        lambda g: g.update(floor=None, metric={"corners": [[0., 0.], [1., 0.], [1., 1.], [0., 1.]],
                                               "width_m": 4., "depth_m": 3.}),
    ]
    for mutation in mutations:
        data = document()
        mutation(data["scenes"][0]["geometry"])
        refused(lambda: parse(data), "TRANSFER_INVALID")
    data = document()
    obj = {"id": "00000000-0000-4000-8000-000000000001", "label": "Стол",
           "polygon": [[0., 0.], [1., 0.], [0., 1.]]}
    data["scenes"][0]["geometry"]["objects"] = [obj, obj]
    refused(lambda: parse(data), "TRANSFER_INVALID")
    data = document()
    data["scenes"][0]["source_revision"] = 42
    refused(lambda: parse(data), "TRANSFER_INVALID")
    data = document()
    data["scenes"][0]["camera_id"] = 999
    refused(lambda: parse(data), "TRANSFER_INVALID")
    data = document()
    data["scenes"].append(copy.deepcopy(data["scenes"][0]))
    refused(lambda: parse(data), "TRANSFER_INVALID")


@test
def v2_documents_keep_their_contract_and_have_no_profiles():
    data = v2_document()
    restored = parse(data)
    assert restored.transfer_version == 2 and restored.counts().profiles == 0
    assert restored.scenes[0].geometry.version == 1
    data["scenes"][0]["geometry"]["version"] = 2
    refused(lambda: parse(data), "TRANSFER_INVALID")
    data = v2_document()
    data["calibrations"][0]["role"] = "BOTH"
    refused(lambda: parse(data), "TRANSFER_INVALID")
    data = v2_document()
    data["profiles"] = []
    refused(lambda: parse(data), "TRANSFER_INVALID")


def refuses(data, code):
    """Unlike refused(), accepting the document is a failure."""
    try:
        parse(data)
    except transfer.TransferError as exc:
        assert exc.code == code, exc.code
        return
    raise AssertionError("document was accepted")


@test
def v3_documents_keep_their_contract_and_time_tracking_is_required_from_v4():
    restored = parse(v3_document())
    assert restored.transfer_version == 3 and restored.counts().profiles == 1
    assert not hasattr(restored.profiles[0], "time_tracking")
    data = v3_document()
    data["profiles"][0]["time_tracking"] = True
    refuses(data, "TRANSFER_INVALID")
    data = v3_document()
    data.update(operator_schema_version=6)
    refuses(data, "TRANSFER_INCOMPATIBLE")
    data = document()
    data["profiles"][0].pop("time_tracking")
    refuses(data, "TRANSFER_INVALID")
    data = document()
    data["profiles"][0].update(ai_enabled=False)  # time tracking without AI is impossible
    refuses(data, "TRANSFER_INVALID")
    data["profiles"][0].update(time_tracking=False)
    assert parse(data).profiles[0].time_tracking is False


@test
def v5_setup_roundtrip_keeps_v4_import_and_excludes_correction_history():
    old = document()
    current = {**old, "transfer_version": 5, "operator_schema_version": 7}
    restored = transfer.decode_transfer(transfer.encode_transfer(parse(current), PASSWORD), PASSWORD)
    assert restored.transfer_version == 5
    assert restored.profiles == parse(old).profiles
    assert restored.faces == parse(old).faces
    current["human_visit_identity_revisions"] = []
    refuses(current, "TRANSFER_INVALID")


@test
def v6_is_current_and_historical_v1_to_v5_still_validate():
    old = document()
    current = {**old, "transfer_version": 6, "operator_schema_version": 8}
    restored = transfer.decode_transfer(transfer.encode_transfer(parse(current), PASSWORD), PASSWORD)
    assert isinstance(restored, transfer.SetupDocument) and restored.transfer_version == 6
    assert restored.operator_schema_version == 8 and restored.attendance_schema_version == 3
    assert restored.faces == parse(old).faces
    for version, operator in ((6, 7), (5, 8), (6, 6)):
        wrong = {**current, "transfer_version": version, "operator_schema_version": operator}
        refuses(wrong, "TRANSFER_INCOMPATIBLE")
    current["human_visit_identity_revisions"] = []
    refuses(current, "TRANSFER_INVALID")
    v5 = {**old, "transfer_version": 5, "operator_schema_version": 7}
    assert type(parse(v5)) is transfer.SetupDocumentV5
    for data in (document(), v3_document(), v2_document(), legacy_document()):
        assert parse(data).transfer_version == data["transfer_version"]


@test
def doors_profiles_and_gate_frames_are_validated():
    mutations = [
        lambda d: d["calibrations"][0].update(door_top=[[0.2, 0.5], [0.8, 0.1]]),
        lambda d: d["calibrations"][0].update(door_top=[[0.2, 0.1]]),
        lambda d: d["calibrations"][0].update(door_kind="GATE"),
        lambda d: d["calibrations"][0].update(door_kind=None),
        lambda d: d["calibrations"][0].update(role="NEITHER"),
        lambda d: d["scenes"][0]["geometry"]["doors"][0].update(quad=[[0.4, 0.8], [0.6, 0.8], [0.6, 0.9], [0.4, 0.2]]),
        lambda d: d["scenes"][0]["geometry"]["doors"][0].update(kind="GATE"),
        lambda d: d["scenes"][0]["geometry"]["doors"].extend(
            [dict(d["scenes"][0]["geometry"]["doors"][0], id=f"00000000-0000-4000-8000-{i:012d}") for i in range(8)]),
        lambda d: d["scenes"][0]["geometry"].pop("doors"),
        lambda d: d["profiles"][0].update(workplace_id=999),
        lambda d: d["profiles"][0].update(zone="OUTSIDE"),
        lambda d: d["profiles"][0].update(ai_fps=30.5),
        lambda d: d["profiles"][0].update(ai_fps=0),
        lambda d: d["profiles"][0].update(camera_id=999),
        lambda d: d["profiles"].append(copy.deepcopy(d["profiles"][0])),
        lambda d: d.pop("profiles"),
    ]
    for mutation in mutations:
        data = document()
        mutation(data)
        refused(lambda: parse(data), "TRANSFER_INVALID")
    data = document()
    data["profiles"][0].update(workplace_id=None, ai_fps=None)
    assert parse(data).profiles[0].ai_fps is None


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
