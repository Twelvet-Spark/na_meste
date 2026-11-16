"""Read-only visit HTTP contracts on isolated SQLite; no models/devices/lifespan.

Run: .venv/Scripts/python.exe tests/test_human_visits_api.py
Controller owns execution during the shared-checkout integration wave.
"""
import asyncio
import base64
import hashlib
import json
import sys
import traceback
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import urlencode
from uuid import UUID, uuid4

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi import FastAPI, HTTPException
from sqlalchemy.orm import Session

from app.api.global_persons import router
from app.database.database import get_db
from app.database.models import Camera, Employee, GlobalPerson, HumanVisit, HumanVisitEvent
from app.time_utils import utcnow
from operator_test_support import Client
from test_operator_storage import database, schema

TESTS = []
PHOTO = b"\xff\xd8saved-person\xff\xd9"
FACE = b"\xff\xd8saved-aligned-face\xff\xd9"
STAMP = datetime(2026, 11, 12, 12, 0, 0, 123456)


def test(fn):
    TESTS.append(fn)
    return fn


class SampleBoundary:
    """The separately-owned preparation service boundary, not an inference model."""
    def __init__(self, error=None):
        self.error = error

    def eligibility(self, visit):
        return bool(visit.face_thumbnail), None if visit.face_thumbnail else "VISIT_FACE_MISSING"

    def append(self, visit_id, *, employee_id, expected_revision, request_id, face_sha256):
        # Native preparation must never block FastAPI's event loop.
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            raise AssertionError("Append preparation ran on the ASGI event loop")
        assert isinstance(request_id, UUID)
        if self.error:
            raise HTTPException(*self.error)
        return dict(visit_id=visit_id, employee_id=employee_id, expected_revision=expected_revision,
                    request_id=str(request_id), face_sha256=face_sha256)


@contextmanager
def api(service=None):
    with database() as (engine, _root):
        schema.initialize_attendance_schema(engine)
        with Session(engine) as db:
            db.add_all([
                Camera(id=1, name="Camera <one>", nvr_host="unused", username="unused", password="secret", channel=1),
                Camera(id=2, name="Camera two", nvr_host="unused", username="unused", password="secret", channel=2),
                Employee(id=7, name="Verified employee", revision=4),
                GlobalPerson(id=99, status="LOST", created_at=STAMP, last_seen_at=STAMP),
            ])
            db.commit()
        app = FastAPI()
        app.state.visit_sample_service = service
        app.include_router(router)

        def isolated_db():
            with Session(engine) as db:
                yield db

        app.dependency_overrides[get_db] = isolated_db
        yield Client(app), engine


def visit(engine, number=1, **changes):
    identifier = str(UUID(int=number))
    values = dict(id=identifier, camera_id=1, tracking_generation=str(UUID(int=100)),
                  track_id=number, source_revision=1, stream_id=str(UUID(int=200)),
                  started_at=STAMP - timedelta(seconds=7), last_seen_at=STAMP,
                  ended_at=None, end_reason=None, status="PROVISIONAL", employee_id=None,
                  identity_state="UNKNOWN", global_id=None, last_bbox=[.1, .2, .4, .9],
                  max_person_score=.85, snapshot=PHOTO, face_thumbnail=FACE,
                  face_quality={"min_size": 90., "det_score": .92, "blur": 45., "aligned": True,
                                "unambiguous": True, "embedding": [1.] * 512},
                  image_expires_at=utcnow() + timedelta(days=2), images_purged_at=None)
    values.update(changes)
    with Session(engine) as db:
        db.add(HumanVisit(**values))
        db.commit()
    return identifier


def payload(**changes):
    values = dict(employee_id=7, expected_revision=4, request_id=str(uuid4()),
                  face_sha256=hashlib.sha256(FACE).hexdigest())
    values.update(changes)
    return values


@test
def test_first_hit_unknown_visit_is_listed_without_images_or_vectors():
    with api() as (client, engine):
        identifier = visit(engine, snapshot=None, face_thumbnail=None)
        page = client.json("GET", "/api/global-persons/visits")
        assert page["next_cursor"] is None
        assert len(page["items"]) == 1
        item = page["items"][0]
        assert item["id"] == identifier and item["status"] == "PROVISIONAL"
        assert item["identity_state"] == "UNKNOWN" and item["employee_id"] is None
        assert item["duration_seconds"] == 7
        assert item["camera_name"] == "Camera <one>"
        assert item["photo_status"] == item["face_status"] == "MISSING"
        assert not ({"snapshot", "face_thumbnail", "embedding", "face_quality"} & item.keys())
        assert item["last_seen_at"] == "2026-11-12T12:00:00.123456Z"
        # The static route must not be swallowed by the historical integer route.
        old = client.json("GET", "/api/global-persons/99")
        assert old["global_id"] == 99


@test
def test_cursor_orders_equal_timestamps_by_uuid_and_preserves_microseconds():
    with api() as (client, engine):
        identifiers = [visit(engine, number) for number in (1, 2, 3)]
        older = visit(engine, 4, last_seen_at=STAMP - timedelta(microseconds=1))
        page = client.json("GET", "/api/global-persons/visits?limit=2")
        assert [item["id"] for item in page["items"]] == [identifiers[2], identifiers[1]]
        assert page["next_cursor"]
        second = client.json("GET", "/api/global-persons/visits?" + urlencode({"limit": 2, "cursor": page["next_cursor"]}))
        assert [item["id"] for item in second["items"]] == [identifiers[0], older]
        assert second["next_cursor"] is None


@test
def test_camera_employee_unknown_and_offset_normalized_time_filters():
    with api() as (client, engine):
        unknown = visit(engine)
        known = visit(engine, 2, camera_id=2, employee_id=7, identity_state="FACE_CONFIRMED")
        for filters, expected in [
            ({"camera_id": 1}, [unknown]), ({"employee_id": 7}, [known]),
            ({"unknown_only": "true"}, [unknown]),
            ({"from": "2026-11-12T15:00:00+03:00", "to": "2026-11-12T12:00:01Z"}, [known, unknown]),
            ({"from": "2026-11-12T12:00:01Z"}, []),
            ({"to": "2026-11-12T11:59:59Z"}, []),
        ]:
            page = client.json("GET", "/api/global-persons/visits?" + urlencode(filters))
            assert [item["id"] for item in page["items"]] == expected


@test
def test_invalid_filters_uuid_ranges_and_opaque_cursors_are_rejected():
    invalid = [
        {"camera_id": 0}, {"employee_id": -1}, {"limit": 0}, {"limit": 201},
        {"from": "2026-11-12T12:00:00"}, {"to": "not-a-date"},
        {"from": "2026-11-12T13:00:00Z", "to": "2026-11-12T12:00:00Z"},
        {"cursor": "!"}, {"cursor": "a" * 513}, {"cursor": "bnVsbA"},
    ]
    for value in [["2026-11-12T12:00:00", str(UUID(int=1))],
                  ["2026-11-12T12:00:00Z", "not-a-uuid"], {"offset": 10}]:
        invalid.append({"cursor": base64.urlsafe_b64encode(json.dumps(value).encode()).decode().rstrip("=")})
    with api() as (client, _engine):
        for filters in invalid:
            client.json("GET", "/api/global-persons/visits?" + urlencode(filters), status=422)
        client.json("GET", "/api/global-persons/visits/not-a-uuid", status=422)
        client.json("GET", f"/api/global-persons/visits/{uuid4()}", status=404)


@test
def test_detail_reports_face_digest_quality_eligibility_and_no_vectors():
    with api(SampleBoundary()) as (client, engine):
        identifier = visit(engine, ended_at=STAMP, status="CLOSED", end_reason="PROCESS_RESTART")
        result = client.json("GET", f"/api/global-persons/visits/{identifier}")
        assert result["face_sha256"] == hashlib.sha256(FACE).hexdigest()
        assert result["face_quality"]["min_size"] == 90.
        assert "embedding" not in result["face_quality"]
        assert not ({"snapshot", "face_thumbnail", "embedding"} & result.keys())
        assert result["employee_sample_eligible"] and result["employee_sample_reason"] is None
        assert result["ended_at"] == result["last_seen_at"] and result["end_reason"] == "PROCESS_RESTART"
        assert result["photo_status"] == result["face_status"] == "AVAILABLE"


@test
def test_image_endpoints_return_exact_saved_bytes_and_bind_face_digest():
    with api() as (client, engine):
        identifier = visit(engine)
        for suffix, saved in [("photo", PHOTO), ("face", FACE)]:
            status, headers, content = client.request("GET", f"/api/global-persons/visits/{identifier}/{suffix}")
            assert status == 200 and content == saved
            assert headers[b"content-type"] == b"image/jpeg"
            assert headers[b"cache-control"] == b"no-store"
            assert headers[b"etag"].decode() == '"' + hashlib.sha256(saved).hexdigest() + '"'


@test
def test_expired_and_purged_images_never_leak_even_before_blob_purge():
    with api() as (client, engine):
        expired = visit(engine, image_expires_at=utcnow() - timedelta(seconds=1))
        purged = visit(engine, 2, images_purged_at=utcnow())
        missing = visit(engine, 3, snapshot=None, face_thumbnail=None)
        for identifier, expected in [(expired, 410), (purged, 410), (missing, 404)]:
            for suffix in ["photo", "face"]:
                status, headers, data = client.request("GET", f"/api/global-persons/visits/{identifier}/{suffix}")
                assert status == expected and data != PHOTO and data != FACE
                assert headers[b"cache-control"] == b"no-store"
        page = client.json("GET", "/api/global-persons/visits")
        statuses = {item["id"]: item["face_status"] for item in page["items"]}
        assert statuses == {expired: "EXPIRED", purged: "PURGED", missing: "MISSING"}
        result = client.json("GET", f"/api/global-persons/visits/{expired}")
        assert result["face_sha256"] is None and result["id"] == expired


@test
def test_events_stay_visit_scoped_and_omit_internal_vectors():
    with api() as (client, engine):
        first, second = visit(engine), visit(engine, 2)
        with Session(engine) as db:
            for visit_id, event_type, at in [(first, "FIRST_SEEN", STAMP), (first, "CLOSED", STAMP + timedelta(seconds=2)),
                                              (second, "FIRST_SEEN", STAMP)]:
                identifier = str(uuid4())
                db.add(HumanVisitEvent(id=identifier, visit_id=visit_id, type=event_type, dedupe_key=identifier,
                                      occurred_at=at, payload={"version": 1, "reason": "PROCESS_RESTART", "embedding": [1.] * 512}))
            db.commit()
        events = client.json("GET", f"/api/global-persons/visits/{first}/events")
        assert [event["type"] for event in events] == ["CLOSED", "FIRST_SEEN"]
        assert all("embedding" not in event["payload"] for event in events)
        assert events[0]["occurred_at"] == "2026-11-12T12:00:02.123456Z"
        client.json("GET", f"/api/global-persons/visits/{uuid4()}/events", status=404)
        client.json("GET", f"/api/global-persons/visits/{first}/events?limit=0", status=422)


@test
def test_append_is_operator_same_origin_only_and_prepares_off_event_loop():
    with api(SampleBoundary()) as (client, engine):
        identifier = visit(engine)
        url = f"/api/global-persons/visits/{identifier}/employee-sample"
        body = payload()
        client.json("POST", url, body, operator=False, status=403)
        client.json("POST", url, body, headers={"origin": "http://foreign"}, status=403)
        result = client.json("POST", url, body, headers={"origin": "http://test"})
        assert result == {"visit_id": identifier, **body}


@test
def test_append_strict_payload_rejects_coercion_extra_fields_and_bad_digests():
    bad = [dict(employee_id=0), dict(employee_id=True), dict(employee_id="7"),
           dict(expected_revision=-1), dict(expected_revision=False), dict(expected_revision=1.5),
           dict(face_sha256="A" * 64), dict(face_sha256="f" * 63), dict(request_id="not-a-uuid"),
           dict(unexpected="discard me")]
    with api(SampleBoundary()) as (client, engine):
        identifier = visit(engine)
        for changes in bad:
            client.json("POST", f"/api/global-persons/visits/{identifier}/employee-sample", payload(**changes), status=422)


@test
def test_unavailable_model_keeps_read_surface_and_append_errors_visible():
    with api() as (client, engine):
        identifier = visit(engine)
        detail = client.json("GET", f"/api/global-persons/visits/{identifier}")
        assert detail["employee_sample_eligible"] is False and detail["employee_sample_reason"] == "FACE_MODEL_UNAVAILABLE"
        error = client.json("POST", f"/api/global-persons/visits/{identifier}/employee-sample", payload(), status=503)
        assert error["detail"]["code"] == "FACE_MODEL_UNAVAILABLE"
    for code, status in [("VISIT_FACE_CHANGED", 409), ("VISIT_FACE_EXPIRED", 410), ("FACE_LIMIT_REACHED", 409), ("ENROLLMENT_BUSY", 503)]:
        with api(SampleBoundary((status, {"code": code}))) as (client, engine):
            identifier = visit(engine)
            error = client.json("POST", f"/api/global-persons/visits/{identifier}/employee-sample", payload(), status=status)
            assert error["detail"]["code"] == code


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
    print(f"{len(TESTS) - failures}/{len(TESTS)} passed")
    raise SystemExit(bool(failures))
