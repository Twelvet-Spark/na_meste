"""Real SQLite/ASGI transfer tests. Never opens the deployment database."""
import asyncio
import base64
import hashlib
import json
import sys
import tempfile
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fastapi import FastAPI
import numpy as np
from sqlalchemy import create_engine, event, select
from sqlalchemy.orm import Session
from app.database.attendance_schema import initialize_attendance_schema
from app.database.database import Base
from app.database.models import *
from app.ai.face_recognition import EmbeddingStore
from app.setup_transfer import TransferError, decode_binary, decode_transfer, encode_transfer, validate_document
from app.services.setup_transfer_service import SetupTransferService
from app.api.setup_transfer import router
from test_setup_transfer_format import document, legacy_document, parse, PASSWORD, MODEL_ID, ALIGNMENT_ID
from operator_test_support import Client

TESTS = []
STAMP = datetime(2026, 11, 16, 10)


def test(fn):
    TESTS.append(fn)
    return fn


@contextmanager
def database():
    with tempfile.TemporaryDirectory() as directory:
        engine = create_engine("sqlite:///" + str(Path(directory)/"test.db"), connect_args={"check_same_thread": False, "timeout": 10})
        @event.listens_for(engine, "connect")
        def foreign_keys(conn, _):
            conn.execute("PRAGMA foreign_keys=ON")
        initialize_attendance_schema(engine)
        try:
            yield engine
        finally:
            engine.dispose()


def service(engine, **kwargs):
    return SetupTransferService(engine, face_model_id=MODEL_ID, **kwargs)


class AlignedEngine:
    model_id = MODEL_ID
    alignment_id = ALIGNMENT_ID

    def __init__(self, callback=None):
        self.callback = callback
        self.batches = []

    def embed_aligned(self, crops):
        assert 1 <= len(crops) <= 4
        assert all(crop.shape == (112, 112, 3) for crop in crops)
        self.batches.append(len(crops))
        if self.callback is not None:
            self.callback()
        result = np.zeros((len(crops), 512), dtype=np.float32)
        result[:, 1] = 1.
        return result


def transfer_blob(data):
    return encode_transfer(validate_document(json.dumps(data).encode()), PASSWORD)


def seed(engine):
    data = parse(document())
    with Session(engine) as db, db.begin():
        db.add(Camera(**{**data.cameras[0].model_dump(exclude={"created_at", "updated_at"}), "created_at": STAMP, "updated_at": STAMP}))
        db.add(Employee(**{**data.employees[0].model_dump(exclude={"created_at", "updated_at"}), "created_at": STAMP, "updated_at": STAMP, "revision": 9}))
        db.add(Workplace(id=9, name="Москва", name_key="москва", active=False))
        db.flush()
        db.add(CameraArchive(**data.archives[0].model_dump(), revision=8))
        db.add(AttendanceCamera(**data.calibrations[0].model_dump(), enabled=True, revision=9, source_revision=8))
        db.add(EmployeeFace(id=12, employee_id=4, embedding=decode_binary(data.faces[0].embedding), thumbnail=decode_binary(data.faces[0].thumbnail), created_at=STAMP))
        db.flush()
        db.add(FaceModelMetadata(face_id=12, model_id=MODEL_ID, alignment_id=ALIGNMENT_ID))
        db.add(CameraScene(**data.scenes[0].model_dump(), revision=8, source_revision=8))
        db.add(EmployeeWorkplace(employee_id=4, workplace_id=9))
        db.add(GlobalPerson(id=22, employee_id=4))
        db.add(HumanVisit(id=str(uuid4()), camera_id=7, tracking_generation=str(uuid4()), track_id=1,
                          source_revision=8, stream_id=str(uuid4()), started_at=STAMP, last_seen_at=STAMP,
                          last_bbox=[0.1, 0.2, 0.3, 0.8], max_person_score=.9, image_expires_at=STAMP,
                          snapshot=b"private-visit-image-not-for-export"))
        db.add(OperatorRequest(id=str(uuid4()), operation="unrelated", payload_sha256="0"*64, result={}))


def refused(fn, code):
    try:
        fn()
    except TransferError as exc:
        assert exc.code == code, exc
    else:
        raise AssertionError("must refuse " + code)


def app_for(engine):
    app = FastAPI()
    app.state.setup_transfer = service(engine)
    app.state.embedding_store = EmbeddingStore(MODEL_ID)
    app.include_router(router)
    return app


def form(blob, *, password=PASSWORD, request_id=None, digest=None):
    fields = {"password": password}
    if request_id:
        fields.update(request_id=str(request_id), expected_sha256=digest or hashlib.sha256(blob).hexdigest())
    boundary = "transfer-boundary"
    body = b""
    for key, value in fields.items():
        body += f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"\r\n\r\n{value}\r\n'.encode()
    body += f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="setup.nmsetup"\r\nContent-Type: application/octet-stream\r\n\r\n'.encode() + blob + f'\r\n--{boundary}--\r\n'.encode()
    return body, {"content-type": "multipart/form-data; boundary="+boundary, "host": "localhost"}


@test
def roundtrip_preserves_setup_not_history_and_gallery_works():
    with database() as source, database() as dest:
        seed(source)
        blob = service(source).export(PASSWORD)
        target = service(dest)
        request_id = uuid4()
        preview = target.preview(blob, PASSWORD)
        assert preview["faces_reencoded"] == 0 and preview["faces_need_reenrollment"] == 0
        assert preview["scenes_need_review"] == 1
        result, replay = target.import_setup(blob, PASSWORD, request_id=request_id, expected_sha256=hashlib.sha256(blob).hexdigest())
        assert result["committed"] and result["restart_required"] and not replay
        with Session(dest) as db:
            assert db.get(Camera, 7).password == "sentinel-device-password"
            assert db.get(Employee, 4).revision == 1
            assert db.get(CameraArchive, 7).revision == 0
            assert not db.get(AttendanceCamera, 7).enabled
            assert db.get(AttendanceCamera, 7).source_revision == 1
            assert not db.get(Workplace, 9).active
            assert db.get(EmployeeWorkplace, (4,9)) is not None
            assert db.get(AttendanceRuntime, 1).commission_id is None
            assert db.query(GlobalPerson).count() == 0
            assert db.query(RecordingSession).count() == 0
            assert db.query(AttendanceEvent).count() == 0
            assert db.get(EmployeeFace, 12).embedding == decode_binary(document()["faces"][0]["embedding"])
            metadata = db.get(FaceModelMetadata, 12)
            assert metadata.model_id == MODEL_ID and metadata.alignment_id == ALIGNMENT_ID
            scene = db.get(CameraScene, 7)
            assert scene.source_revision == 0 and scene.revision == 1
            assert scene.reference_width == 1920 and scene.reference_height == 1080
            assert scene.geometry == document()["scenes"][0]["geometry"]
            assert db.query(HumanVisit).count() == 0 and db.query(HumanJournalGap).count() == 0
            gallery = EmbeddingStore(MODEL_ID)
            gallery.refresh(db)
            import numpy as np
            assert gallery.match(np.frombuffer(db.get(EmployeeFace, 12).embedding, dtype=np.float32)).employee_id == 4
        retry, replay = service(dest).import_setup(blob, PASSWORD, request_id=request_id, expected_sha256=hashlib.sha256(blob).hexdigest())
        assert replay and retry["committed"]
        refused(lambda: target.import_setup(blob, "wrong-password-2026", request_id=request_id, expected_sha256=hashlib.sha256(blob).hexdigest()), "TRANSFER_PASSWORD_OR_FILE_INVALID")
        refused(lambda: target.import_setup(blob, PASSWORD, request_id=uuid4(), expected_sha256=hashlib.sha256(blob).hexdigest()), "TRANSFER_DESTINATION_NOT_EMPTY")
        refused(lambda: target.import_setup(blob, PASSWORD, request_id=request_id, expected_sha256="0"*64), "TRANSFER_DIGEST_MISMATCH")
        with Session(source) as db:
            assert db.get(Employee, 4).revision == 9 and db.query(GlobalPerson).count() == 1


@test
def emptiness_ignores_only_runtime_and_markers():
    with database() as db:
        with Session(db) as session, session.begin():
            session.get(AttendanceRuntime, 1).heartbeat_at = STAMP
        assert service(db).status()["destination_empty"]
        with db.begin() as conn:
            conn.exec_driver_sql("CREATE TABLE user_custom (x INTEGER)")
            conn.exec_driver_sql("INSERT INTO user_custom VALUES (1)")
        assert not service(db).status()["destination_empty"]
    for cls, values in [(Workplace, {"name":"Place","name_key":"place"}),
                        (OperatorRequest, {"id":str(uuid4()),"operation":"old","payload_sha256":"0"*64,"result":{}})]:
        with database() as db:
            with Session(db) as session, session.begin():
                session.add(cls(**values))
            assert not service(db).status()["destination_empty"]


@test
def stale_preview_concurrency_and_rollback_cannot_overwrite():
    with database() as source, database() as dest:
        seed(source)
        blob = service(source).export(PASSWORD)
        digest = hashlib.sha256(blob).hexdigest()
        target = service(dest)
        target.preview(blob, PASSWORD)
        with Session(dest) as db, db.begin():
            db.add(Workplace(name="New",name_key="new"))
        refused(lambda: target.import_setup(blob, PASSWORD, request_id=uuid4(), expected_sha256=digest), "TRANSFER_DESTINATION_NOT_EMPTY")
    with database() as source, database() as dest:
        seed(source)
        blob = service(source).export(PASSWORD)
        digest = hashlib.sha256(blob).hexdigest()
        def broken(conn, cursor, statement, params, context, many):
            if statement.startswith("INSERT INTO employee_faces"):
                raise RuntimeError("sentinel-device-password")
        event.listen(dest, "before_cursor_execute", broken)
        refused(lambda: service(dest).import_setup(blob, PASSWORD, request_id=uuid4(), expected_sha256=digest), "TRANSFER_STORAGE_UNAVAILABLE")
        event.remove(dest, "before_cursor_execute", broken)
        assert service(dest).status()["destination_empty"]
        def run(_):
            try:
                return service(dest).import_setup(blob, PASSWORD, request_id=uuid4(), expected_sha256=digest)[0]["committed"]
            except TransferError as exc:
                return exc.code
        with ThreadPoolExecutor(2) as pool:
            assert sorted(map(str, pool.map(run, range(2)))) == ["TRANSFER_DESTINATION_NOT_EMPTY", "True"]


@test
def local_guard_denies_before_body_and_success_is_sanitized():
    with database() as source, database() as dest:
        seed(source)
        app = app_for(dest)
        client = Client(app)
        client.json("GET", "/api/setup-transfer/status", headers={"host":"localhost"})
        for headers in ({"host":"evil.example"}, {"host":"localhost","forwarded":"for=127.0.0.1"},
                        {"host":"localhost","x-forwarded-for":"127.0.0.1"},
                        {"host":"localhost","origin":"https://evil.example"}):
            actual, _, data = client.request("POST", "/api/setup-transfer/export", body=b"not JSON", headers=headers)
            assert actual == 403 and "sentinel" not in str(data)
        assert client.request("GET", "/api/setup-transfer/status", operator=False, headers={"host":"localhost"})[0] == 403
        blob = service(source).export(PASSWORD)
        body, headers = form(blob)
        preview = client.json("POST", "/api/setup-transfer/preview", body=body, headers=headers)
        assert preview["destination_empty"] and "sentinel-device-password" not in json.dumps(preview)
        request_id = uuid4()
        body, headers = form(blob, request_id=request_id)
        result = client.json("POST", "/api/setup-transfer/import", body=body, headers=headers, status=201)
        assert result["committed"] and result["restart_required"]
        client.json("POST", "/api/setup-transfer/import", body=body, headers=headers, status=200)
        body, headers = form(blob, password="wrong-password-2026")
        client.json("POST", "/api/setup-transfer/preview", body=body, headers=headers, status=422)
        assert client.request("POST", "/api/setup-transfer/export", payload={"password":"short"}, headers={"host":"localhost"})[0] == 422
        assert client.request("POST", "/api/setup-transfer/preview", body=b"x", headers={"host":"localhost","content-type":"text/plain"})[0] == 415
        assert client.request("POST", "/api/setup-transfer/export", body=b"x", headers={"host":"localhost","content-length":"10000","content-type":"application/json"})[0] == 413
        class BrokenGallery:
            def refresh(self, db):
                raise RuntimeError("credential sentinel")
        # Authenticated committed replay remains success even if gallery publication is unavailable.
        app.state.embedding_store = BrokenGallery()
        body, headers = form(blob, request_id=request_id)
        result = client.json("POST", "/api/setup-transfer/import", body=body, headers=headers, status=200)
        assert result["committed"] and "TRANSFER_RESTART_FOR_GALLERY" in result["warnings"]


@test
def cancelled_operation_holds_slot_until_worker_finishes():
    async def scenario():
        with database() as db:
            app = app_for(db)
            entered, release = threading.Event(), threading.Event()
            original = app.state.setup_transfer.export
            def blocking(password):
                entered.set()
                release.wait(5)
                return original(password)
            app.state.setup_transfer.export = blocking
            async def request():
                return await Client(app).arequest("POST", "/api/setup-transfer/export", payload={"password":PASSWORD}, headers={"host":"localhost"})
            task = asyncio.create_task(request())
            await asyncio.to_thread(entered.wait, 5)
            task.cancel()
            try:
                await asyncio.sleep(.01)
                assert (await request())[0] == 429
            finally:
                release.set()
            try:
                await task
            except asyncio.CancelledError:
                pass
            assert (await request())[0] == 200
    asyncio.run(scenario())


@test
def transport_limits_locality_and_empty_receipts():
    import app.api.setup_transfer as api
    async def scenario():
        with database() as db:
            app = app_for(db)
            async def scope_request(remote, host, *, body=b"", headers=None, method="GET", path="/api/setup-transfer/status"):
                async def wrapped(scope, receive, send):
                    scope["client"] = (remote, 1234)
                    await app(scope, receive, send)
                return await Client(wrapped).arequest(method, path, body=body, headers={"host":host, **(headers or {})})
            assert (await scope_request("::1", "[::1]:8000"))[0] == 200
            for remote, host in (("10.0.0.2", "localhost"), ("127.0.0.1", "localhost.evil"), ("127.0.0.1", "localhost:0")):
                assert (await scope_request(remote, host))[0] == 403
            result = await scope_request("127.0.0.1", "localhost", method="POST", path="/api/setup-transfer/export",
                                         body=b" " * 8193, headers={"content-type":"application/json"})
            assert result[0] == 413
            empty_blob = app.state.setup_transfer.export(PASSWORD)
            request_id = uuid4()
            body, headers = form(empty_blob, request_id=request_id)
            result = await Client(app).arequest("POST", "/api/setup-transfer/import", body=body, headers=headers)
            assert result[0] == 201 and result[2]["counts"]["employees"] == 0
            assert not app.state.setup_transfer.status()["destination_empty"]
            new_blob = app.state.setup_transfer.export(PASSWORD)
            body, headers = form(new_blob, request_id=request_id)
            result = await Client(app).arequest("POST", "/api/setup-transfer/import", body=body, headers=headers)
            assert result[0] == 409 and result[2]["detail"] == "TRANSFER_REQUEST_CONFLICT"
            with patch.object(api, "MAX_BODY_BYTES", 200):
                body, headers = form(empty_blob)
                result = await Client(app).arequest("POST", "/api/setup-transfer/preview", chunks=[body[:150], body[150:]], headers=headers)
                assert result[0] == 413
            body, headers = form(empty_blob)
            body = body.replace(b"--transfer-boundary--", b'--transfer-boundary\r\nContent-Disposition: form-data; name="password"\r\n\r\nextra\r\n--transfer-boundary--')
            result = await Client(app).arequest("POST", "/api/setup-transfer/preview", body=body, headers=headers)
            assert result[0] == 422
    asyncio.run(scenario())


@test
def interrupted_multipart_closes_encrypted_spool():
    import starlette.formparsers as parsers
    async def scenario():
        with database() as db:
            app = app_for(db)
            started = asyncio.Event()
            spools = []
            original = parsers.SpooledTemporaryFile
            def track(*args, **kwargs):
                spool = original(*args, **kwargs)
                spools.append(spool)
                return spool
            chunk = (b'--boundary\r\nContent-Disposition: form-data; name="file"; filename="x"\r\n\r\n' + b"x"*1000)
            async def wrapped(scope, receive, send):
                delivered = False
                async def stream():
                    nonlocal delivered
                    if not delivered:
                        delivered = True
                        return {"type":"http.request","body":chunk,"more_body":True}
                    started.set()
                    await asyncio.Event().wait()
                await app(scope, stream, send)
            with patch.object(parsers, "SpooledTemporaryFile", track):
                task = asyncio.create_task(Client(wrapped).arequest("POST", "/api/setup-transfer/preview",
                    headers={"host":"localhost","content-type":"multipart/form-data; boundary=boundary"}))
                await asyncio.wait_for(started.wait(), 3)
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            assert spools and all(spool.closed for spool in spools), "cancelled upload leaked spool"
    asyncio.run(scenario())


@test
def populated_sqlite_like_user_table_blocks_import():
    with database() as db:
        with db.begin() as connection:
            connection.exec_driver_sql("CREATE TABLE sqliteXhistory (value INTEGER)")
            connection.exec_driver_sql("INSERT INTO sqliteXhistory VALUES (1)")
        assert not service(db).status()["destination_empty"]


@test
def application_entrypoint_registers_transfer_service_and_routes():
    import os
    import subprocess
    root = Path(__file__).resolve().parent.parent
    with tempfile.TemporaryDirectory() as directory:
        env = {**os.environ, "DATABASE_URL": "sqlite:///" + str(Path(directory)/"app.db"),
               "OPERATOR_SETTINGS_PATH": str(Path(directory)/"settings.json"),
               "RECORDING_ROOT": str(Path(directory)/"recordings"), "AI_ENABLED": "false",
               "TELEGRAM_ENABLED": "false", "AI_MODEL_MANIFEST": str(Path(directory)/"missing-model-manifest.json"),
               "PYTHONPATH": str(root) + os.pathsep + str(root/"tests")}
        code = """
import asyncio
import main
from operator_test_support import Client
main.build_enroll_engine = lambda *args, **kwargs: None
async def check():
    async with main.lifespan(main.app):
        status, _, data = await Client(main.app).arequest('GET', '/api/setup-transfer/status',
                                                         headers={'host':'localhost'})
        assert status == 503, (status, data)
        assert data == {'detail': 'TRANSFER_MODEL_UNAVAILABLE'}
asyncio.run(check())
"""
        result = subprocess.run([sys.executable, "-c", code],
            cwd=os.environ.get("TRANSFER_TEST_ENTRYPOINT_CWD", str(root)), env=env,
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30)
        assert result.returncode == 0, result.stdout + result.stderr


@test
def legacy_and_null_faces_share_target_space_preparation_and_keep_bad_thumbnails():
    for data in (legacy_document(), document()):
        if data["transfer_version"] == 2:
            data["faces"][0].update(face_model_id=None, alignment_id=None)
        data["faces"].extend([
            {**data["faces"][0], "id": 13, "thumbnail": None},
            {**data["faces"][0], "id": 14, "thumbnail": base64.b64encode(b"broken JPEG").decode()},
        ])
        blob = transfer_blob(data)
        with database() as dest:
            engine = AlignedEngine()
            target = service(dest, face_engine=engine, enrollment_semaphore=threading.BoundedSemaphore(1))
            preview = target.preview(blob, PASSWORD)
            assert preview["faces_reencoded"] == 1 and preview["faces_need_reenrollment"] == 2
            assert preview["scenes_need_review"] == (1 if data["transfer_version"] == 2 else 0)
            assert target.status()["destination_empty"]
            result, replay = target.import_setup(blob, PASSWORD, request_id=uuid4(),
                                                 expected_sha256=hashlib.sha256(blob).hexdigest())
            assert not replay and result["faces_reencoded"] == 1 and result["faces_need_reenrollment"] == 2
            with Session(dest) as db:
                face = db.get(EmployeeFace, 12)
                vector = np.frombuffer(face.embedding, dtype="<f4")
                assert vector[1] == 1. and vector[0] == 0.
                assert db.get(FaceModelMetadata, 12).model_id == MODEL_ID
                assert db.get(FaceModelMetadata, 13) is None and db.get(FaceModelMetadata, 14) is None
                assert db.get(EmployeeFace, 13).embedding == decode_binary(data["faces"][0]["embedding"])
                assert db.get(EmployeeFace, 14).thumbnail == b"broken JPEG"
                assert db.get(AttendanceRuntime, 1).gallery_revision == 1


@test
def incompatible_samples_require_model_before_writes_and_share_enrollment_slot():
    data = document()
    data["faces"][0].update(face_model_id=None, alignment_id=None)
    blob = transfer_blob(data)
    digest = hashlib.sha256(blob).hexdigest()
    with database() as dest:
        target = service(dest)
        refused(lambda: target.preview(blob, PASSWORD), "TRANSFER_MODEL_UNAVAILABLE")
        refused(lambda: target.import_setup(blob, PASSWORD, request_id=uuid4(), expected_sha256=digest),
                "TRANSFER_MODEL_UNAVAILABLE")
        assert target.status()["destination_empty"]
        semaphore = threading.BoundedSemaphore(1)
        target = service(dest, face_engine=AlignedEngine(), enrollment_semaphore=semaphore)
        semaphore.acquire()
        try:
            refused(lambda: target.preview(blob, PASSWORD), "ENROLLMENT_BUSY")
            refused(lambda: target.import_setup(blob, PASSWORD, request_id=uuid4(), expected_sha256=digest),
                    "ENROLLMENT_BUSY")
        finally:
            semaphore.release()
        assert target.preview(blob, PASSWORD)["faces_reencoded"] == 1
        assert target.status()["destination_empty"]


@test
def compatible_vectors_need_no_recognizer_even_without_thumbnails():
    data = document()
    data["faces"][0]["thumbnail"] = None
    blob = transfer_blob(data)
    with database() as dest:
        target = service(dest)
        assert target.preview(blob, PASSWORD)["faces_reencoded"] == 0
        target.import_setup(blob, PASSWORD, request_id=uuid4(), expected_sha256=hashlib.sha256(blob).hexdigest())
        with Session(dest) as db:
            assert db.get(EmployeeFace, 12).embedding == decode_binary(data["faces"][0]["embedding"])
            assert db.get(FaceModelMetadata, 12).model_id == MODEL_ID


@test
def receipt_replay_returns_frozen_preparation_result_without_model_or_enrollment_slot():
    data = legacy_document()
    blob = transfer_blob(data)
    request_id, digest = uuid4(), hashlib.sha256(blob).hexdigest()
    with database() as dest:
        target = service(dest, face_engine=AlignedEngine())
        committed, replay = target.import_setup(blob, PASSWORD, request_id=request_id, expected_sha256=digest)
        assert not replay and committed["faces_reencoded"] == 1
        unavailable = service(dest)
        restored, replay = unavailable.import_setup(blob, PASSWORD, request_id=request_id, expected_sha256=digest)
        assert replay and restored == committed
        data["employees"][0]["name"] = "Changed payload"
        altered = transfer_blob(data)
        refused(lambda: unavailable.import_setup(altered, PASSWORD, request_id=request_id,
                                                 expected_sha256=hashlib.sha256(altered).hexdigest()),
                "TRANSFER_REQUEST_CONFLICT")


@test
def preparation_outside_writer_rechecks_empty_destination_and_inference_failure_rolls_back():
    data = legacy_document()
    blob = transfer_blob(data)
    digest = hashlib.sha256(blob).hexdigest()
    with database() as dest:
        def operator_edit():
            with Session(dest) as db, db.begin():
                db.add(Workplace(name="Operator edit", name_key="operator edit"))
        target = service(dest, face_engine=AlignedEngine(operator_edit))
        refused(lambda: target.import_setup(blob, PASSWORD, request_id=uuid4(), expected_sha256=digest),
                "TRANSFER_DESTINATION_NOT_EMPTY")
        with Session(dest) as db:
            assert db.query(Employee).count() == 0 and db.query(OperatorRequest).count() == 0
            assert db.query(Workplace).one().name == "Operator edit"
    with database() as dest:
        def broken_model():
            raise RuntimeError("private inference error")
        semaphore = threading.BoundedSemaphore(1)
        target = service(dest, face_engine=AlignedEngine(broken_model), enrollment_semaphore=semaphore)
        refused(lambda: target.import_setup(blob, PASSWORD, request_id=uuid4(), expected_sha256=digest),
                "TRANSFER_MODEL_UNAVAILABLE")
        assert target.status()["destination_empty"]
        assert semaphore.acquire(blocking=False)
        semaphore.release()


@test
def preparation_handles_all_samples_in_enrollment_sized_chunks():
    data = legacy_document()
    data["faces"] = [{**data["faces"][0], "id": i} for i in range(1, 18)]
    blob = transfer_blob(data)
    with database() as dest:
        target = service(dest, face_engine=AlignedEngine())
        assert target.preview(blob, PASSWORD)["faces_reencoded"] == 17
        target.import_setup(blob, PASSWORD, request_id=uuid4(), expected_sha256=hashlib.sha256(blob).hexdigest())
        with Session(dest) as db:
            assert db.query(EmployeeFace).count() == 17 and db.query(FaceModelMetadata).count() == 17


@test
def export_projects_actual_mixed_provenance_instead_of_runtime_stamping():
    with database() as source:
        seed(source)
        with Session(source) as db, db.begin():
            original = db.get(EmployeeFace, 12)
            db.add_all([
                EmployeeFace(id=13, employee_id=4, embedding=original.embedding, thumbnail=None),
                EmployeeFace(id=14, employee_id=4, embedding=original.embedding, thumbnail=None),
            ])
            db.flush()
            db.add(FaceModelMetadata(face_id=14, model_id="2" * 64, alignment_id=ALIGNMENT_ID))
        blob = service(source).export(PASSWORD)
        data = decode_transfer(blob, PASSWORD)
        assert data.face_model_id == MODEL_ID
        faces = {face.id: face for face in data.faces}
        assert faces[12].face_model_id == MODEL_ID
        assert faces[13].face_model_id is None and faces[13].alignment_id is None
        assert faces[14].face_model_id == "2" * 64
        with database() as dest:
            refused(lambda: service(dest).preview(blob, PASSWORD), "TRANSFER_MODEL_MISMATCH")
            refused(lambda: service(dest).import_setup(blob, PASSWORD, request_id=uuid4(),
                                                       expected_sha256=hashlib.sha256(blob).hexdigest()),
                    "TRANSFER_MODEL_MISMATCH")
            assert service(dest).status()["destination_empty"]


@test
def unavailable_declared_model_returns_static_error_after_local_security_guard():
    with database() as dest:
        app = app_for(dest)
        app.state.setup_transfer = None
        client = Client(app)
        for method, path in (("GET", "/status"), ("POST", "/export"), ("POST", "/preview"), ("POST", "/import")):
            status, headers, body = client.request(method, "/api/setup-transfer" + path,
                                                   body=b"not parsed", headers={"host": "localhost"})
            assert status == 503 and body == {"detail": "TRANSFER_MODEL_UNAVAILABLE"}
            assert headers[b"cache-control"] == b"no-store"
        assert client.request("POST", "/api/setup-transfer/import", body=b"invalid",
                              headers={"host": "evil.example"})[0] == 403


@test
def invalid_or_unnormalized_vectors_are_reencoded_or_retained_without_provenance():
    for first in (0., 2., float("nan"), float("inf"), 1e20):
        vector = np.zeros(512, dtype="<f4")
        vector[0] = first
        raw = vector.tobytes()
        for thumbnail in (document()["faces"][0]["thumbnail"], None):
            data = document()
            data["faces"][0].update(embedding=base64.b64encode(raw).decode(), thumbnail=thumbnail)
            blob = transfer_blob(data)
            with database() as dest:
                target = service(dest, face_engine=AlignedEngine())
                preview = target.preview(blob, PASSWORD)
                assert preview["faces_reencoded"] == (1 if thumbnail else 0)
                assert preview["faces_need_reenrollment"] == (0 if thumbnail else 1)
                target.import_setup(blob, PASSWORD, request_id=uuid4(),
                                    expected_sha256=hashlib.sha256(blob).hexdigest())
                with Session(dest) as db:
                    face = db.get(EmployeeFace, 12)
                    if thumbnail:
                        assert np.frombuffer(face.embedding, dtype="<f4")[1] == 1.
                        assert db.get(FaceModelMetadata, 12).model_id == MODEL_ID
                    else:
                        assert face.embedding == raw and db.get(FaceModelMetadata, 12) is None


@test
def import_prepares_again_after_preview_and_does_not_reuse_stale_vectors():
    data = legacy_document()
    blob = transfer_blob(data)
    with database() as dest:
        target = service(dest, face_engine=AlignedEngine())
        assert target.preview(blob, PASSWORD)["faces_reencoded"] == 1
        target.face_engine = None
        refused(lambda: target.import_setup(blob, PASSWORD, request_id=uuid4(),
                                            expected_sha256=hashlib.sha256(blob).hexdigest()),
                "TRANSFER_MODEL_UNAVAILABLE")
        assert target.status()["destination_empty"]


@test
def imported_scene_review_marker_survives_schema_restart():
    blob = transfer_blob(document())
    with database() as dest:
        service(dest).import_setup(blob, PASSWORD, request_id=uuid4(),
                                   expected_sha256=hashlib.sha256(blob).hexdigest())
        initialize_attendance_schema(dest)
        with Session(dest) as db:
            scene = db.get(CameraScene, 7)
            assert scene.source_revision == 0 and scene.revision == 1
            assert scene.geometry == document()["scenes"][0]["geometry"]
            assert db.get(AttendanceCamera, 7).enabled is False
            assert db.get(AttendanceRuntime, 1).commission_id is None


@test
def unusable_legacy_thumbnails_need_no_recognizer_or_gpu_lease():
    for data in (legacy_document(), document()):
        if data["transfer_version"] == 2:
            data["faces"][0].update(face_model_id=None, alignment_id=None)
        data["faces"][0]["thumbnail"] = None
        data["faces"].append({**data["faces"][0], "id": 13,
                              "thumbnail": base64.b64encode(b"broken JPEG").decode()})
        blob = transfer_blob(data)
        with database() as dest:
            target = service(dest)
            preview = target.preview(blob, PASSWORD)
            assert preview["faces_reencoded"] == 0 and preview["faces_need_reenrollment"] == 2
            assert target.status()["destination_empty"]
            result, replay = target.import_setup(
                blob, PASSWORD, request_id=uuid4(), expected_sha256=hashlib.sha256(blob).hexdigest())
            assert not replay and result["faces_need_reenrollment"] == 2
            with Session(dest) as db:
                assert db.query(FaceModelMetadata).count() == 0
                assert db.get(EmployeeFace, 12).embedding == decode_binary(data["faces"][0]["embedding"])
                assert db.get(EmployeeFace, 13).thumbnail == b"broken JPEG"


@test
def compatible_gallery_norm_tolerance_does_not_erase_provenance_without_a_thumbnail():
    data = document()
    vector = np.zeros(512, dtype="<f4")
    vector[0] = .9995  # Valid under the shared persisted-gallery unit-norm tolerance.
    original = vector.tobytes()
    data["faces"][0].update(embedding=base64.b64encode(original).decode(), thumbnail=None)
    blob = transfer_blob(data)
    with database() as dest:
        target = service(dest)
        preview = target.preview(blob, PASSWORD)
        assert preview["faces_reencoded"] == preview["faces_need_reenrollment"] == 0
        target.import_setup(blob, PASSWORD, request_id=uuid4(),
                            expected_sha256=hashlib.sha256(blob).hexdigest())
        with Session(dest) as db:
            assert db.get(EmployeeFace, 12).embedding == original
            assert db.get(FaceModelMetadata, 12).model_id == MODEL_ID


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
