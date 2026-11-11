"""Atomic employee/workplace HTTP regressions on isolated real SQLite.

Run: .venv/Scripts/python.exe tests/test_operator_employees.py
Controller owns execution. Only the external face model is synthetic; parsing,
image decoding/quality, matching, transactions, gallery publication and APIs are real.
"""
import asyncio
import base64
import importlib.util
import io
import json
import struct
import sys
import threading
import time
import traceback
import zlib
from contextlib import contextmanager
from unittest.mock import patch
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import numpy as np
from fastapi import FastAPI
from PIL import Image
from sqlalchemy import event
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session
from starlette.datastructures import UploadFile

from app.api import employees
from app.ai.face_recognition import EmbeddingStore
from app.database.database import get_db, immediate_session
from app.database.models import AttendanceRuntime, Employee, EmployeeFace
from operator_test_support import Client, multipart
from test_attendance_recognition import CHECKER, face, photo_bytes, vector
from test_operator_storage import database, dump, legacy, schema, STAMP

TESTS = []
MAX_PHOTO = 15 * 1024 * 1024
BODY_LIMIT = 226 * 1024 * 1024


def test(fn):
    TESTS.append(fn)
    return fn


class Faces:
    """External-model boundary recording concurrency/threading without changing SQL."""
    def __init__(self):
        self.observations = [face()]
        self.calls = 0
        self.running = 0
        self.maximum_running = 0
        self.event_loop_calls = 0
        self.hook = None
        self.delay = 0
        self.lock = threading.Lock()

    def detect(self, image, max_num=0):
        with self.lock:
            self.calls += 1
            self.running += 1
            self.maximum_running = max(self.maximum_running, self.running)
        try:
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                pass
            else:
                self.event_loop_calls += 1
            if self.hook:
                self.hook()
            if self.delay:
                time.sleep(self.delay)
            return self.observations[:max_num] if max_num else self.observations
        finally:
            with self.lock:
                self.running -= 1


@contextmanager
def api(*, historical=False):
    with database() as (engine, root):
        if historical:
            legacy(engine, attendance=True)
        schema.initialize_attendance_schema(engine)
        app = FastAPI()
        model, store = Faces(), EmbeddingStore()
        with Session(engine) as db:
            store.refresh(db)
        app.state.face_engine = model
        app.state.embedding_store = store
        app.include_router(employees.router)
        # Missing new routes must fail HTTP behavior assertions, not import collection.
        if importlib.util.find_spec("app.api.workplaces") is not None:
            from app.api import workplaces
            app.include_router(workplaces.router)

        def isolated_db():
            with Session(engine) as db:
                yield db

        app.dependency_overrides[get_db] = isolated_db
        yield Client(app), engine, model, store, root


def metadata(**changes):
    return dict(first_name="Иван", last_name="Петров", request_id=str(uuid4()), **changes)


def files(count=1, content=None):
    image = photo_bytes() if content is None else content
    return [(f"portrait-{index}.png", image, "image/png") for index in range(count)]


def create(client, data=None, photos=None, *, status=201, **kwargs):
    return client.multipart("POST", "/api/employees", metadata() if data is None else data,
                            files() if photos is None else photos, status=status, **kwargs)


def detail(client, employee_id):
    return client.json("GET", f"/api/employees/{employee_id}?faces=true")


def save_faces(client, employee, keep=(), photos=(), *, status=200, revision=None):
    return client.multipart("PUT", f"/api/employees/{employee['id']}/faces",
                            {"expected_revision": employee["revision"] if revision is None else revision,
                             "keep_face_ids": list(keep)}, photos, status=status)


def gallery_revision(engine):
    with Session(engine) as db:
        return db.get(AttendanceRuntime, 1).gallery_revision


def workplace(client, name="Магазин №1"):
    return client.json("POST", "/api/workplaces", {"name": name}, status=201)


def assert_photo_error(response, index):
    error = response["detail"]
    assert isinstance(error, dict), ("Expected atomic per-file failure details", response)
    assert error["code"] == "PHOTO_VALIDATION_FAILED" and error["committed"] is False
    matching = [item for item in error["files"] if item["index"] == index]
    assert len(matching) == 1 and matching[0]["filename"] and matching[0]["error"]


@test
def test_one_and_fifteen_photos_commit_one_employee_and_one_gallery_revision():
    for count in (1, 15):
        with api() as (client, engine, model, store, _root):
            result = create(client, photos=files(count))
            assert result["name"] == "Петров Иван"
            assert result["first_name"] == "Иван" and result["last_name"] == "Петров"
            assert result["revision"] == 1 and result["faces_count"] == count
            assert result["workplaces"] == []
            assert result["needs_name_completion"] is False and result["needs_photo_remediation"] is False
            assert len(result["faces"]) == count
            assert len({row["id"] for row in result["faces"]}) == count
            for row in result["faces"]:
                status, headers, image = client.request("GET", row["thumbnail_url"])
                assert status == 200 and headers[b"content-type"].startswith(b"image/jpeg")
                assert cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR).shape[:2] == (112, 112)
            assert model.calls == count and model.event_loop_calls == 0
            assert gallery_revision(engine) == 1
            assert store.match(vector()).employee_id == result["id"]
            assert store.match(vector()).store_revision == 1
            with Session(engine) as db:
                assert db.query(Employee).count() == 1 and db.query(EmployeeFace).count() == count
                sample = db.query(EmployeeFace).first()
                assert len(sample.embedding) == 512 * 4
                assert sample.thumbnail != photo_bytes(), "Only aligned JPEG, not original upload, is retained"


@test
def test_zero_and_sixteen_files_rejected_before_model_and_without_writes():
    for count in (0, 16):
        with api() as (client, engine, model, _store, _root):
            before = dump(engine)
            body, mime = multipart(metadata(), files(count))
            status, _headers, _data = client.request("POST", "/api/employees", body=body,
                                                    headers={"content-type": mime})
            assert status in (400, 422), status
            assert model.calls == 0 and dump(engine) == before


@test
def test_bad_second_image_rolls_back_entire_creation_and_reports_file_index():
    with api() as (client, engine, _model, store, _root):
        before = dump(engine)
        response = create(client, photos=[*files(), ("broken.png", b"not an image", "image/png")], status=422)
        assert_photo_error(response, 1)
        assert dump(engine) == before
        assert store.match(vector()).employee_id is None and store.match(vector()).store_revision == 0


@test
def test_all_existing_quality_and_ambiguity_guards_apply_to_atomic_creation():
    observations = [[], [face(), face()], [face(aligned=None)], [face(det_score=.79)],
                    [face(aligned=np.zeros_like(CHECKER))],
                    [face(bbox=np.array([0., 0., 79., 100.]))],
                    [face(embedding=np.full(512, np.nan, dtype=np.float32))],
                    [face(embedding=np.zeros(512, dtype=np.float32))]]
    for observed in observations:
        with api() as (client, engine, model, _store, _root):
            model.observations = observed
            before = dump(engine)
            assert_photo_error(create(client, status=422), 0)
            assert dump(engine) == before


@test
def test_other_active_identity_conflict_is_atomic_and_inactive_identity_is_not_a_match():
    with api() as (client, engine, _model, _store, _root):
        first = create(client)
        before = dump(engine)
        response = create(client, metadata(external_id="another"), status=422)
        assert_photo_error(response, 0)
        assert dump(engine) == before
        client.json("PUT", f"/api/employees/{first['id']}",
                    {"expected_revision": first["revision"], "active": False})
        second = create(client)
        assert second["id"] != first["id"]


@test
def test_committed_uuid_replays_original_result_after_model_loss_and_metadata_edit():
    with api() as (client, engine, model, _store, _root):
        data = metadata()
        original = create(client, data)
        client.json("PUT", f"/api/employees/{original['id']}",
                    {"expected_revision": original["revision"], "first_name": "Павел", "last_name": "Сидоров"})
        before = dump(engine)
        calls = model.calls
        client.app.state.face_engine = None
        replay = create(client, data)
        assert replay == original and dump(engine) == before and model.calls == calls
        create(client, dict(data, first_name="Другой"), status=409)
        create(client, data, files(content=photo_bytes() + b"changed"), status=409)
        assert dump(engine) == before


@test
def test_uuid_digest_uses_normalized_names_and_ordered_photo_hashes():
    with api() as (client, engine, _model, _store, _root):
        data = metadata()
        data.update(first_name="  Иван  ", last_name="  Петров  ")
        original = create(client, data, files(2))
        canonical = dict(data, first_name="Иван", last_name="Петров")
        assert create(client, canonical, files(2)) == original
        distinct = [*files(), ("second.png", photo_bytes() + b"different", "image/png")]
        # An independent identity allows a new request without active-gallery conflict.
        client.app.state.face_engine.observations = [face(vector(0., 1.))]
        ordered = metadata(external_id="ordered")
        create(client, ordered, distinct)
        before = dump(engine)
        create(client, ordered, list(reversed(distinct)), status=409)
        assert dump(engine) == before


@test
def test_operator_ledger_contains_digests_and_results_but_no_biometric_payloads():
    with api() as (client, engine, _model, _store, _root):
        data = metadata()
        created = create(client, data)
        with engine.connect() as connection:
            rows = connection.exec_driver_sql("SELECT * FROM operator_requests").mappings().all()
        assert len(rows) == 1
        record = dict(rows[0])
        assert record["id"] == data["request_id"]
        assert len(record["payload_sha256"]) == 64
        int(record["payload_sha256"], 16)
        result = json.loads(record["result"]) if isinstance(record["result"], str) else record["result"]
        assert result["id"] == created["id"]
        encoded = json.dumps(record, default=str).lower()
        for forbidden in ("embedding", '"password"', '"credentials"', base64.b64encode(photo_bytes()).decode().lower(),
                          base64.b64encode(vector().tobytes()).decode().lower()):
            assert forbidden not in encoded
        with engine.begin() as connection:
            connection.exec_driver_sql("UPDATE operator_requests SET operation='different-operation'")
        create(client, data, status=409)


@test
def test_replacing_sole_photo_and_at_capacity_is_one_atomic_revision():
    for count in (1, 15):
        with api() as (client, engine, _model, store, _root):
            original = create(client, photos=files(count))
            old_ids = [row["id"] for row in original["faces"]]
            saved = save_faces(client, original, old_ids[1:], files())
            new_ids = [row["id"] for row in saved["faces"]]
            assert saved["faces_count"] == count and saved["revision"] == original["revision"] + 1
            assert old_ids[0] not in new_ids and set(old_ids[1:]) <= set(new_ids)
            assert gallery_revision(engine) == 2 and store.match(vector()).store_revision == 2
            assert detail(client, original["id"])["faces_count"] == count
            client.json("GET", f"/api/employees/{original['id']}/faces/{old_ids[0]}/thumbnail", status=404)


@test
def test_invalid_final_set_stale_revision_and_foreign_or_duplicate_keeps_preserve_original():
    with api() as (client, engine, model, _store, _root):
        original = create(client)
        own = original["faces"][0]["id"]
        model.observations = [face(vector(0., 1.))]
        other = create(client)
        foreign = other["faces"][0]["id"]
        model.observations = [face()]
        for keep, new, revision, expected in [
            ([], [], original["revision"], 422),
            ([own, own], [], original["revision"], 422),
            ([foreign], [], original["revision"], 422),
            ([999999], [], original["revision"], 422),
            ([own], files(15), original["revision"], 422),
            ([], files(), original["revision"] - 1, 409),
        ]:
            before = dump(engine)
            save_faces(client, original, keep, new, revision=revision, status=expected)
            assert dump(engine) == before
        before = dump(engine)
        response = save_faces(client, original, [], files(content=b"corrupt"), status=422)
        assert_photo_error(response, 0)
        assert dump(engine) == before


@test
def test_concurrent_face_saves_have_exactly_one_winner_and_preserve_model_serialization():
    with api() as (client, engine, model, _store, _root):
        original = create(client)
        model.delay = .03
        body, mime = multipart({"expected_revision": original["revision"], "keep_face_ids": []}, files())

        async def race():
            return await asyncio.gather(*[
                client.arequest("PUT", f"/api/employees/{original['id']}/faces", body=body,
                                headers={"content-type": mime}) for _ in range(2)
            ])

        results = asyncio.run(race())
        assert sorted(row[0] for row in results) == [200, 409], results
        assert gallery_revision(engine) == 2
        assert detail(client, original["id"])["revision"] == original["revision"] + 1
        assert detail(client, original["id"])["faces_count"] == 1
        assert model.maximum_running == 1 and model.event_loop_calls == 0


@test
def test_concurrent_same_uuid_creates_one_employee_and_replays_the_winner():
    with api() as (client, engine, model, _store, _root):
        model.delay = .03
        body, mime = multipart(metadata(), files())

        async def race():
            return await asyncio.gather(*[
                client.arequest("POST", "/api/employees", body=body,
                                headers={"content-type": mime}) for _ in range(2)
            ])

        results = asyncio.run(race())
        assert [row[0] for row in results] == [201, 201], results
        assert results[0][2] == results[1][2]
        assert gallery_revision(engine) == 1
        with Session(engine) as db:
            assert db.query(Employee).count() == 1 and db.query(EmployeeFace).count() == 1
        assert model.maximum_running == 1 and model.event_loop_calls == 0


@test
def test_conflict_committed_during_preparation_is_rechecked_under_writer_lock():
    with api() as (client, engine, model, store, _root):
        with immediate_session(engine) as db:
            db.add(Employee(id=90, name="Existing identity"))
        inserted = False

        def concurrent_enrollment():
            nonlocal inserted
            if inserted:
                return
            inserted = True
            # Preparation must not hold the writer lock. This independent commit
            # also proves conflict detection cannot use the earlier cached gallery.
            with engine.connect() as connection:
                connection.exec_driver_sql("PRAGMA busy_timeout=250")
                connection.exec_driver_sql("BEGIN IMMEDIATE")
                connection.exec_driver_sql(
                    "INSERT INTO employee_faces (employee_id,embedding,created_at) VALUES (90,?,?)",
                    (vector().tobytes(), STAMP))
                connection.exec_driver_sql("UPDATE attendance_runtime SET gallery_revision=gallery_revision+1")
                connection.commit()

        model.hook = concurrent_enrollment
        assert_photo_error(create(client, status=422), 0)
        assert inserted
        with Session(engine) as db:
            assert db.query(Employee).count() == 1 and db.query(EmployeeFace).count() == 1
        assert gallery_revision(engine) == 1
        # Failed request must not publish an uncommitted candidate to the app gallery.
        assert store.match(vector()).employee_id is None


@test
def test_failed_gallery_revision_write_rolls_back_creation_and_replacement():
    for replace_existing in (False, True):
        with api() as (client, engine, _model, store, _root):
            original = create(client) if replace_existing else None
            before = dump(engine)
            old_revision = store.match(vector()).store_revision

            def fail_revision(connection, cursor, statement, parameters, context, executemany):
                if statement.lstrip().upper().startswith("UPDATE ATTENDANCE_RUNTIME"):
                    raise OperationalError(statement, parameters, RuntimeError("injected gallery failure"))

            event.listen(engine, "before_cursor_execute", fail_revision)
            try:
                if original:
                    save_faces(client, original, [], files(), status=503)
                else:
                    create(client, status=503)
            finally:
                event.remove(engine, "before_cursor_execute", fail_revision)
            assert dump(engine) == before
            assert store.match(vector()).store_revision == old_revision


@test
def test_names_are_trimmed_bounded_and_unknown_or_null_metadata_is_rejected():
    with api() as (client, engine, _model, _store, _root):
        data = metadata()
        data.update(first_name="  Иван  ", last_name="  Петров  ", external_id="x" * 64)
        created = create(client, data)
        assert created["name"] == "Петров Иван" and created["external_id"] == "x" * 64
        bad_values = [dict(first_name=" "), dict(last_name=""), dict(first_name="x" * 75),
                      dict(last_name="x" * 75), dict(external_id="x" * 65),
                      dict(first_name=None), dict(active=None), dict(name="old one-field API"),
                      dict(request_id="not-a-uuid"), dict(unexpected=True)]
        for values in bad_values:
            invalid = metadata()
            invalid.update(values)
            before = dump(engine)
            create(client, invalid, status=422)
            assert dump(engine) == before
        for values in ({"first_name": "Only"}, {"first_name": None, "last_name": None},
                       {"active": None}, {"name": "Old"}, {"unknown": True}):
            before = dump(engine)
            client.json("PUT", f"/api/employees/{created['id']}",
                        {"expected_revision": created["revision"], **values}, status=422)
            assert dump(engine) == before
        client.json("PUT", f"/api/employees/{created['id']}", {"active": False}, status=422)


@test
def test_maximum_names_and_metadata_revision_conflict_are_not_silently_overwritten():
    with api() as (client, engine, _model, _store, _root):
        data = metadata()
        data.update(first_name="И" * 74, last_name="П" * 74)
        created = create(client, data)
        assert len(created["name"]) == 149
        changed = client.json("PUT", f"/api/employees/{created['id']}",
                              {"expected_revision": created["revision"], "first_name": "  Иван ",
                               "last_name": " Петров "})
        assert changed["name"] == "Петров Иван" and changed["revision"] == created["revision"] + 1
        before = dump(engine)
        client.json("PUT", f"/api/employees/{created['id']}",
                    {"expected_revision": created["revision"], "active": False}, status=409)
        assert dump(engine) == before


@test
def test_workplace_normalization_rename_archive_and_multiple_optional_tags():
    with api() as (client, engine, _model, _store, _root):
        one = workplace(client, "  Магазин   №1  ")
        assert one["name"] == "Магазин №1" and one["active"] is True
        client.json("POST", "/api/workplaces", {"name": "МАГАЗИН №1"}, status=409)
        sharp = workplace(client, "Straße")
        client.json("POST", "/api/workplaces", {"name": "STRASSE"}, status=409)
        two = workplace(client, "Офис")
        created = create(client, metadata(workplace_ids=[one["id"], two["id"]]))
        assert {row["id"] for row in created["workplaces"]} == {one["id"], two["id"]}
        renamed = client.json("PUT", f"/api/workplaces/{two['id']}", {"name": "  Новый офис "})
        assert renamed["name"] == "Новый офис"
        client.json("PUT", f"/api/workplaces/{two['id']}", {"name": "strasse"}, status=409)
        archived = client.json("PUT", f"/api/workplaces/{one['id']}", {"active": False})
        assert archived["active"] is False
        listed = client.json("GET", "/api/workplaces")
        assert {row["id"] for row in listed} == {one["id"], two["id"], sharp["id"]}
        current = detail(client, created["id"])
        assert any(not row["active"] for row in current["workplaces"])
        retained = client.json("PUT", f"/api/employees/{created['id']}",
                               {"expected_revision": current["revision"],
                                "workplace_ids": [one["id"], two["id"]]})
        cleared = client.json("PUT", f"/api/employees/{created['id']}",
                              {"expected_revision": retained["revision"], "workplace_ids": []})
        assert cleared["workplaces"] == []
        before = dump(engine)
        client.json("PUT", f"/api/employees/{created['id']}",
                    {"expected_revision": cleared["revision"], "workplace_ids": [one["id"]]}, status=422)
        create(client, metadata(workplace_ids=[one["id"]]), status=422)
        create(client, metadata(workplace_ids=[999999]), status=422)
        assert dump(engine) == before
        client.json("PUT", "/api/workplaces/999999", {"name": "Absent"}, status=404)
        client.json("DELETE", f"/api/workplaces/{one['id']}", status=405)
        for name in (" ", "x" * 151):
            client.json("POST", "/api/workplaces", {"name": name}, status=422)
        assert workplace(client, "x" * 150)["name"] == "x" * 150


@test
def test_workplace_archived_during_preparation_cannot_be_newly_assigned():
    with api() as (client, engine, model, _store, _root):
        place = workplace(client)

        def archive():
            with engine.begin() as connection:
                connection.exec_driver_sql("PRAGMA busy_timeout=250")
                connection.exec_driver_sql("UPDATE workplaces SET active=0 WHERE id=?", (place["id"],))

        model.hook = archive
        create(client, metadata(workplace_ids=[place["id"]]), status=422)
        with Session(engine) as db:
            assert db.query(Employee).count() == 0 and db.query(EmployeeFace).count() == 0
        assert gallery_revision(engine) == 0


@test
def test_legacy_remediation_preserves_history_and_allows_deactivation_before_photo_fix():
    with api(historical=True) as (client, engine, _model, _store, _root):
        old = detail(client, 41)
        assert {"first_name", "last_name", "revision", "needs_name_completion",
                "needs_photo_remediation"} <= old.keys(), old
        assert old["first_name"] is None and old["last_name"] is None
        assert old["name"] == "Не разбивать старое имя"
        assert old["needs_name_completion"] and old["needs_photo_remediation"]
        changed = client.json("PUT", "/api/employees/41", {"expected_revision": 0, "active": False})
        assert changed["needs_name_completion"] and changed["needs_photo_remediation"]
        named = client.json("PUT", "/api/employees/41", {"expected_revision": changed["revision"],
                                                          "first_name": "Иван", "last_name": "Петров"})
        assert named["needs_name_completion"] is False and named["needs_photo_remediation"]
        corrected = save_faces(client, named, [], files())
        assert corrected["needs_photo_remediation"] is False
        client.json("DELETE", "/api/employees/41", status=409)
        with engine.connect() as connection:
            assert connection.exec_driver_sql("SELECT employee_name FROM attendance_events").scalar_one() == "Снимок события"
            assert connection.exec_driver_sql("SELECT employee_name FROM attendance_employee_references").scalar_one() == "Снимок ссылки"
            assert connection.exec_driver_sql("SELECT employee_name FROM work_sessions").scalar_one() == "Снимок интервала"


@test
def test_legacy_over_capacity_is_untouched_until_final_set_is_explicitly_corrected():
    with api(historical=True) as (client, engine, _model, _store, _root):
        with immediate_session(engine) as db:
            db.add_all([EmployeeFace(employee_id=41, embedding=vector().tobytes()) for _ in range(16)])
        old = detail(client, 41)
        assert old["faces_count"] == 16 and old.get("needs_photo_remediation") is True, old
        changed = client.json("PUT", "/api/employees/41", {"expected_revision": old["revision"], "active": False})
        assert changed["faces_count"] == 16
        ids = [row["id"] for row in old["faces"]]
        before = dump(engine)
        save_faces(client, changed, ids, status=422)
        assert dump(engine) == before
        fixed = save_faces(client, changed, ids[:15])
        assert fixed["faces_count"] == 15 and fixed["needs_photo_remediation"] is False


@test
def test_changed_mutations_require_operator_header_and_reject_cross_origin():
    with api() as (client, engine, model, _store, _root):
        before = dump(engine)
        create(client, operator=False, status=403)
        create(client, headers={"Origin": "http://evil.example"}, status=403)
        create(client, headers={"Origin": "http://test:0"}, status=403)
        create(client, headers={"X-Operator-Request": "0"}, status=403)
        assert model.calls == 0 and dump(engine) == before
        created = create(client, headers={"Origin": "http://test"})
        place = workplace(client)
        before = dump(engine)
        for headers, operator in (({}, False), ({"Origin": "http://test.evil"}, True)):
            for method, url, payload in (
                ("PUT", f"/api/employees/{created['id']}", {"expected_revision": created["revision"], "active": False}),
                ("DELETE", f"/api/employees/{created['id']}", None),
                ("POST", "/api/workplaces", {"name": "Forged"}),
                ("PUT", f"/api/workplaces/{place['id']}", {"active": False}),
            ):
                client.json(method, url, payload, status=403, headers=headers, operator=operator)
            client.multipart("PUT", f"/api/employees/{created['id']}/faces",
                             {"expected_revision": created["revision"], "keep_face_ids": []}, files(),
                             status=403, headers=headers, operator=operator)
        assert dump(engine) == before
        _status, headers, _body = client.request("OPTIONS", "/api/employees",
                                                headers={"Origin": "http://evil.example",
                                                         "Access-Control-Request-Method": "POST"})
        assert headers.get(b"access-control-allow-origin") != b"*"


@test
def test_incremental_face_endpoints_and_zero_photo_json_create_are_not_available():
    with api() as (client, engine, _model, _store, _root):
        created = create(client)
        before = dump(engine)
        client.multipart("POST", f"/api/employees/{created['id']}/faces", {}, files(), status=405)
        actual, _headers, _body = client.request(
            "DELETE", f"/api/employees/{created['id']}/faces/{created['faces'][0]['id']}")
        assert actual in (404, 405), actual
        status, _headers, _body = client.request("POST", "/api/employees", {"name": "No photographs"})
        assert status in (400, 415, 422), status
        assert dump(engine) == before


@test
def test_history_free_employee_deletion_removes_faces_tags_and_advances_gallery():
    with api() as (client, engine, _model, store, _root):
        place = workplace(client)
        created = create(client, metadata(workplace_ids=[place["id"]]))
        client.json("DELETE", f"/api/employees/{created['id']}", status=204)
        client.json("GET", f"/api/employees/{created['id']}", status=404)
        with Session(engine) as db:
            assert db.query(EmployeeFace).count() == 0
        with engine.connect() as connection:
            assert connection.exec_driver_sql("SELECT count(*) FROM employee_workplaces").scalar_one() == 0
            assert connection.exec_driver_sql("SELECT count(*) FROM workplaces").scalar_one() == 1
        assert gallery_revision(engine) == 2 and store.match(vector()).employee_id is None


def png_header(width, height):
    payload = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    kind = b"IHDR"
    return b"\x89PNG\r\n\x1a\n" + struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload))


def solid_rgb_png(width, height):
    """Complete PNG without allocating a width*height RGB fixture array."""
    def chunk(kind, payload):
        return (struct.pack(">I", len(payload)) + kind + payload
                + struct.pack(">I", zlib.crc32(kind + payload)))

    compressor = zlib.compressobj()
    row = b"\x00" + b"\x80\x40\x20" * width  # PNG filter None, RGB pixels.
    compressed = []
    for _ in range(height):
        block = compressor.compress(row)
        if block:
            compressed.append(block)
    compressed.append(compressor.flush())
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header)
            + chunk(b"IDAT", b"".join(compressed)) + chunk(b"IEND", b""))


@test
def test_valid_png_above_twenty_million_pixels_is_rejected_before_opencv():
    image = solid_rgb_png(5000, 4001)  # 20,005,000 pixels; below Pillow's bomb threshold.
    assert len(image) < MAX_PHOTO
    assert Image.MAX_IMAGE_PIXELS is not None and 20_005_000 < Image.MAX_IMAGE_PIXELS
    with Image.open(io.BytesIO(image)) as valid:
        assert valid.format == "PNG" and valid.size == (5000, 4001)
        assert not valid.is_animated
        valid.verify()  # Valid CRCs and complete IDAT/IEND, not a truncated header.
    with api() as (client, engine, model, _store, _root):
        before = dump(engine)
        # The real service catches cv2.error. Therefore a mutation reaching this
        # boundary still returns 422: the independent call assertion must fail.
        with patch.object(cv2, "imdecode", side_effect=cv2.error("Forbidden pixel allocation")) as decoder:
            assert_photo_error(create(client, photos=files(content=image), status=422), 0)
        assert decoder.call_count == 0, "Over-limit PNG reached OpenCV's allocation boundary"
        assert model.calls == 0 and dump(engine) == before


@test
def test_valid_png_exactly_twenty_million_pixels_is_accepted():
    image = solid_rgb_png(5000, 4000)
    assert len(image) < MAX_PHOTO
    with api() as (client, engine, model, store, _root):
        detect = model.detect
        decoded_shapes = []

        def observe_real_frame(decoded, max_num=0):
            decoded_shapes.append(decoded.shape)
            return detect(decoded, max_num=max_num)

        with patch.object(model, "detect", side_effect=observe_real_frame):
            created = create(client, photos=files(content=image))
        assert decoded_shapes == [(4000, 5000, 3)], "Boundary image must pass real decoding"
        assert created["faces_count"] == 1 and created["revision"] == 1
        assert model.calls == 1 and gallery_revision(engine) == 1
        assert store.match(vector()).employee_id == created["id"]


@test
def test_header_bombs_animated_and_non_jpeg_png_inputs_are_rejected_before_inference():
    animated = io.BytesIO()
    Image.new("RGB", (112, 112), "red").save(animated, format="PNG", save_all=True,
                                            append_images=[Image.new("RGB", (112, 112), "blue")], duration=100)
    bmp = io.BytesIO()
    Image.fromarray(CHECKER).save(bmp, format="BMP")
    for image in (png_header(100000, 100000),
                  animated.getvalue(), bmp.getvalue(), photo_bytes()[:24]):
        with api() as (client, engine, model, _store, _root):
            before = dump(engine)
            with patch.object(cv2, "imdecode", side_effect=AssertionError("Unsafe image reached OpenCV allocation")):
                assert_photo_error(create(client, photos=files(content=image), status=422), 0)
            assert model.calls == 0 and dump(engine) == before


@test
def test_jpeg_is_accepted_and_exact_byte_limit_is_not_off_by_one():
    for extension, content in ((".jpg", cv2.imencode(".jpg", CHECKER)[1].tobytes()),
                               (".png", photo_bytes() + bytes(MAX_PHOTO - len(photo_bytes())))):
        with api() as (client, _engine, model, _store, _root):
            created = create(client, photos=[("portrait" + extension, content,
                                             "image/jpeg" if extension == ".jpg" else "image/png")])
            assert created["faces_count"] == 1 and model.calls == 1


@test
def test_oversize_upload_reads_are_bounded_and_every_parsed_file_is_closed():
    with api() as (client, engine, model, _store, _root):
        read_sizes, closed, uploaded = [], [], {}
        original_read, original_close = UploadFile.read, UploadFile.close

        async def bounded_read(upload, size=-1):
            read_sizes.append(size)
            assert 0 <= size <= MAX_PHOTO + 1, "Unbounded read of a client-controlled upload"
            result = await original_read(upload, size)
            uploaded[id(upload)] = uploaded.get(id(upload), 0) + len(result)
            assert uploaded[id(upload)] <= MAX_PHOTO + 1
            return result

        async def track_close(upload):
            await original_close(upload)
            closed.append(upload)

        before = dump(engine)
        with patch.object(UploadFile, "read", bounded_read), patch.object(UploadFile, "close", track_close):
            response = create(client, photos=[("large.png", b"x" * (MAX_PHOTO + 2), "image/png"), *files()], status=422)
        assert_photo_error(response, 0)
        assert read_sizes and len({id(upload) for upload in closed}) == 2
        assert all(upload.file.closed for upload in closed)
        assert model.calls <= 1 and dump(engine) == before


@test
def test_metadata_limit_extra_fields_and_malformed_json_do_not_reach_model():
    with api() as (client, engine, model, _store, _root):
        before = dump(engine)
        for bad_metadata, extra_fields in (("{", ()), ("x" * (64 * 1024 + 1), ()),
                                            (metadata(), (("unexpected", "field"),)),
                                            (metadata(), (("metadata", "{}"),))):
            body, mime = multipart(bad_metadata, files(), fields=extra_fields)
            status, _headers, _data = client.request("POST", "/api/employees", body=body,
                                                    headers={"content-type": mime})
            assert status in (400, 413, 422), status
            assert model.calls == 0 and dump(engine) == before


@test
def test_receive_cap_rejects_content_length_before_consuming_body_and_streamed_overflow():
    with api() as (client, engine, model, _store, _root):
        consumed = 0

        def forbidden_body():
            nonlocal consumed
            consumed += 1
            yield b"must not consume an oversized declared body"

        client.json("POST", "/api/employees", status=413, chunks=forbidden_body(),
                    headers={"content-type": "multipart/form-data; boundary=cap",
                             "content-length": str(BODY_LIMIT + 1)})
        assert consumed == 0 and model.calls == 0
        before = dump(engine)
        block = b"x" * (1024 * 1024)

        def streamed_body():
            nonlocal consumed
            yield (b'--cap\r\nContent-Disposition: form-data; name="files"; filename="large.png"\r\n'
                   b'Content-Type: image/png\r\n\r\n')
            for _ in range(228):
                consumed += len(block)
                yield block
            yield b"\r\n--cap--\r\n"

        client.json("POST", "/api/employees", status=413, chunks=streamed_body(),
                    headers={"content-type": "multipart/form-data; boundary=cap"})
        assert consumed <= BODY_LIMIT + len(block), "Server must stop receiving at the cap"
        assert model.calls == 0 and dump(engine) == before
        # Read-only routes remain usable after a rejected oversized request.
        assert client.json("GET", "/api/employees") == []


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
