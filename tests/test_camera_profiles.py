"""Camera profile, reconnect and bulk-action HTTP contracts on isolated real SQLite.

Run: .venv/Scripts/python.exe tests/test_camera_profiles.py
The router, services and transactions are real; only the RTSP reader manager is a
recording fake, so no camera, FFmpeg or model is touched.
"""
import sys
import traceback
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi import FastAPI
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.api import cameras as cameras_api
from app.database.database import get_db
from app.database.models import AttendanceCamera, CameraProfile, Workplace
from app.services import camera_service
from operator_test_support import Client
from test_operator_storage import database, schema

TESTS = []


def test(fn):
    TESTS.append(fn)
    return fn


class FakeManager:
    """Records reader lifecycle calls; a camera is 'running' once started or synced enabled."""

    def __init__(self):
        self.running, self.restarted, self.synced = set(), [], []

    def status(self, camera_id):
        return {"status": "ONLINE" if camera_id in self.running else "OFFLINE"}

    def start(self, camera):
        self.running.add(camera.id)

    def sync(self, camera, changed_fields=None):
        self.synced.append((camera.id, camera.enabled, set(changed_fields or ())))
        (self.running.add if camera.enabled else self.running.discard)(camera.id)

    def restart(self, camera_id):
        if camera_id not in self.running:
            return False
        self.restarted.append(camera_id)
        return True

    def stop(self, deletion):
        self.running.discard(deletion.id)


@contextmanager
def fixture(cameras=3):
    with database() as (engine, _root):
        schema.initialize_attendance_schema(engine)
        with Session(engine) as db:
            for ident in range(1, cameras + 1):
                camera_service.create_camera(db, camera_service.CameraCreate(
                    name=f"Camera {ident}", nvr_host="unused", username="u", password="p",
                    channel=ident, stream_type="main"))
            db.add_all([Workplace(id=1, name="Касса", name_key="касса", active=True),
                        Workplace(id=2, name="Склад", name_key="склад", active=True),
                        Workplace(id=3, name="Архив", name_key="архив", active=False)])
            db.commit()
        manager = FakeManager()
        manager.running.update({1, 2})
        app = FastAPI()
        app.state.manager = manager
        app.include_router(cameras_api.router)

        def isolated_db():
            with Session(engine) as db:
                yield db

        app.dependency_overrides[get_db] = isolated_db
        yield SimpleNamespace(client=Client(app), engine=engine, manager=manager)


def profile(f, camera_id):
    return f.client.json("GET", f"/api/cameras/{camera_id}")["profile"]


@test
def camera_responses_carry_profile_and_attendance_state():
    with fixture() as f:
        cameras = f.client.json("GET", "/api/cameras")
        assert [c["id"] for c in cameras] == [1, 2, 3]
        assert cameras[0]["profile"] == dict(
            workplace_id=None, workplace_name=None, workplace_active=None, zone="INTERIOR",
            ai_enabled=True, ai_fps=None, time_tracking=False, revision=0)
        assert cameras[0]["attendance"] == dict(enabled=False, role=None, source_revision=1)
        saved = f.client.json("PUT", "/api/cameras/1/profile", dict(
            expected_revision=0, workplace_id=1, zone="ENTRANCE", ai_fps=2.5))
        assert saved["profile"] == dict(workplace_id=1, workplace_name="Касса", workplace_active=True,
                                        zone="ENTRANCE", ai_enabled=True, ai_fps=2.5, time_tracking=False,
                                        revision=1)


@test
def profile_save_is_compare_and_set_and_validates_every_field():
    with fixture() as f:
        f.client.json("PUT", "/api/cameras/1/profile", dict(expected_revision=0, ai_enabled=False))
        assert profile(f, 1)["revision"] == 1 and profile(f, 1)["ai_enabled"] is False
        f.client.json("PUT", "/api/cameras/1/profile", dict(expected_revision=0, zone="ENTRANCE"), status=409)
        assert profile(f, 1)["zone"] == "INTERIOR"
        # Saving nothing new is not a change and must not burn a revision.
        f.client.json("PUT", "/api/cameras/1/profile", dict(expected_revision=1, ai_enabled=False))
        assert profile(f, 1)["revision"] == 1
        for bad in (dict(ai_fps=0), dict(ai_fps=30.5), dict(zone="OUTSIDE"), dict(ai_enabled="yes"),
                    dict(time_tracking="yes"), dict(time_tracking=None), dict(workplace_id=0), dict(extra=1)):
            f.client.json("PUT", "/api/cameras/1/profile", dict(expected_revision=1, **bad), status=422)
        f.client.json("PUT", "/api/cameras/1/profile", dict(zone="ENTRANCE"), status=422)
        f.client.json("PUT", "/api/cameras/999/profile", dict(expected_revision=0, zone="ENTRANCE"), status=404)


@test
def workplace_must_exist_and_be_active_unless_it_is_already_assigned():
    with fixture() as f:
        f.client.json("PUT", "/api/cameras/1/profile", dict(expected_revision=0, workplace_id=99), status=422)
        f.client.json("PUT", "/api/cameras/1/profile", dict(expected_revision=0, workplace_id=3), status=422)
        assert profile(f, 1)["workplace_id"] is None
        with Session(f.engine) as db:
            db.get(CameraProfile, 2).workplace_id = 2
            db.query(Workplace).filter_by(id=2).update({"active": False})
            db.commit()
        # Keeping an archived workplace is allowed; unassigning is always allowed.
        assert f.client.json("PUT", "/api/cameras/2/profile", dict(
            expected_revision=0, workplace_id=2, ai_enabled=False))["profile"]["workplace_active"] is False
        assert f.client.json("PUT", "/api/cameras/2/profile", dict(
            expected_revision=1, workplace_id=None))["profile"]["workplace_id"] is None


@test
def a_door_gate_never_locks_the_zone_or_ai_of_its_camera():
    with fixture() as f:
        f.client.json("PUT", "/api/cameras/1/profile", dict(expected_revision=0, zone="ENTRANCE"))
        with Session(f.engine) as db:
            row = db.get(AttendanceCamera, 1)
            row.role, row.enabled, row.door_kind = "BOTH", True, "DOOR"
            row.line_start, row.line_end, row.inside_point = [.2, .5], [.8, .5], [.5, .8]
            row.door_top, row.reference_width, row.reference_height = [[.2, .1], [.8, .1]], 640, 480
            db.commit()
        saved = f.client.json("PUT", "/api/cameras/1/profile", dict(
            expected_revision=1, zone="INTERIOR", ai_enabled=False))["profile"]
        assert (saved["zone"], saved["ai_enabled"], saved["revision"]) == ("INTERIOR", False, 2)
        assert f.client.json("GET", "/api/cameras/1")["attendance"] == dict(enabled=True, role="BOTH", source_revision=1)


@test
def time_tracking_forces_ai_on_and_ai_off_forces_time_tracking_off():
    with fixture() as f:
        assert profile(f, 1)["time_tracking"] is False, "A new camera must be switched on explicitly"
        f.client.json("PUT", "/api/cameras/1/profile", dict(expected_revision=0, ai_enabled=False))
        on = f.client.json("PUT", "/api/cameras/1/profile", dict(expected_revision=1, time_tracking=True))["profile"]
        assert (on["ai_enabled"], on["time_tracking"], on["revision"]) == (True, True, 2)
        # Repeating the same switch is not a change.
        f.client.json("PUT", "/api/cameras/1/profile", dict(expected_revision=2, time_tracking=True))
        assert profile(f, 1)["revision"] == 2
        # One save, one revision, both switches.
        off = f.client.json("PUT", "/api/cameras/1/profile", dict(expected_revision=2, ai_enabled=False))["profile"]
        assert (off["ai_enabled"], off["time_tracking"], off["revision"]) == (False, False, 3)
        # Turning only time tracking off leaves AI running.
        f.client.json("PUT", "/api/cameras/1/profile", dict(expected_revision=3, time_tracking=True))
        kept = f.client.json("PUT", "/api/cameras/1/profile", dict(expected_revision=4, time_tracking=False))["profile"]
        assert (kept["ai_enabled"], kept["time_tracking"], kept["revision"]) == (True, False, 5)
        # Unrelated edits keep the switch.
        f.client.json("PUT", "/api/cameras/1/profile", dict(expected_revision=5, time_tracking=True))
        other = f.client.json("PUT", "/api/cameras/1/profile", dict(
            expected_revision=6, workplace_id=1, ai_fps=4))["profile"]
        assert (other["time_tracking"], other["ai_enabled"], other["workplace_id"]) == (True, True, 1)
        assert profile(f, 2)["time_tracking"] is False, "Only the saved camera changes"


@test
def the_database_refuses_time_tracking_without_ai():
    with fixture() as f:
        with f.engine.begin() as connection:
            connection.exec_driver_sql("UPDATE camera_profiles SET time_tracking=1 WHERE camera_id=1")
            connection.exec_driver_sql("UPDATE camera_profiles SET ai_enabled=0 WHERE camera_id=2")
        for statement in ("UPDATE camera_profiles SET ai_enabled=0 WHERE camera_id=1",
                          "UPDATE camera_profiles SET time_tracking=1 WHERE camera_id=2",
                          "UPDATE camera_profiles SET time_tracking=2 WHERE camera_id=3"):
            try:
                with f.engine.begin() as connection:
                    connection.exec_driver_sql(statement)
            except IntegrityError:
                pass
            else:
                raise AssertionError(f"Accepted: {statement}")
        assert [(p["ai_enabled"], p["time_tracking"]) for p in map(lambda i: profile(f, i), (1, 2, 3))] == [
            (True, True), (False, False), (True, False)]


@test
def reconnect_restarts_only_a_running_enabled_camera():
    with fixture() as f:
        status = f.client.json("POST", "/api/cameras/1/reconnect")
        assert status["status"] == "ONLINE" and f.manager.restarted == [1]
        f.client.json("POST", "/api/cameras/3/reconnect", status=409)
        f.client.json("POST", "/api/cameras/999/reconnect", status=404)
        assert f.manager.restarted == [1]
        assert f.client.request("POST", "/api/cameras/1/reconnect", operator=False)[0] == 403


@test
def bulk_actions_report_each_camera_and_never_stop_at_the_first_failure():
    with fixture() as f:
        result = f.client.json("POST", "/api/cameras/bulk", dict(
            camera_ids=[1, 999, 2], action="ASSIGN_WORKPLACE", workplace_id=1))
        assert [(r["camera_id"], r["ok"]) for r in result["results"]] == [(1, True), (999, False), (2, True)]
        assert result["results"][1]["detail"] and result["results"][0]["detail"] is None
        assert profile(f, 1)["workplace_id"] == 1 and profile(f, 2)["workplace_id"] == 1
        archived = f.client.json("POST", "/api/cameras/bulk", dict(
            camera_ids=[3], action="ASSIGN_WORKPLACE", workplace_id=3))
        assert archived["results"][0]["ok"] is False
        f.client.json("POST", "/api/cameras/bulk", dict(camera_ids=[1, 2], action="ASSIGN_WORKPLACE", workplace_id=None))
        assert profile(f, 1)["workplace_id"] is None
        f.client.json("POST", "/api/cameras/bulk", dict(camera_ids=[1, 3], action="SET_ENTRANCE"))
        f.client.json("POST", "/api/cameras/bulk", dict(camera_ids=[1], action="AI_OFF"))
        assert (profile(f, 1)["zone"], profile(f, 1)["ai_enabled"], profile(f, 3)["zone"]) == ("ENTRANCE", False, "ENTRANCE")
        f.client.json("POST", "/api/cameras/bulk", dict(camera_ids=[1, 3], action="AI_ON"))
        f.client.json("POST", "/api/cameras/bulk", dict(camera_ids=[3], action="SET_INTERIOR"))
        assert (profile(f, 1)["ai_enabled"], profile(f, 3)["zone"]) == (True, "INTERIOR")


@test
def bulk_enable_and_disable_use_the_camera_update_path():
    with fixture() as f:
        result = f.client.json("POST", "/api/cameras/bulk", dict(camera_ids=[1, 3], action="DISABLE"))
        assert all(r["ok"] for r in result["results"])
        assert (1, False, {"enabled"}) in f.manager.synced
        assert not any(c["enabled"] for c in f.client.json("GET", "/api/cameras") if c["id"] in (1, 3))
        f.client.json("POST", "/api/cameras/bulk", dict(camera_ids=[3], action="ENABLE"))
        assert (3, True, {"enabled"}) in f.manager.synced and 3 in f.manager.running


@test
def bulk_time_actions_follow_the_profile_rules_for_every_camera():
    with fixture() as f:
        f.client.json("POST", "/api/cameras/bulk", dict(camera_ids=[1], action="AI_OFF"))
        result = f.client.json("POST", "/api/cameras/bulk", dict(camera_ids=[1, 2, 999], action="TIME_ON"))
        assert [(r["camera_id"], r["ok"]) for r in result["results"]] == [(1, True), (2, True), (999, False)]
        assert [(profile(f, i)["time_tracking"], profile(f, i)["ai_enabled"]) for i in (1, 2)] == [
            (True, True), (True, True)]
        f.client.json("POST", "/api/cameras/bulk", dict(camera_ids=[2], action="AI_OFF"))
        assert (profile(f, 2)["ai_enabled"], profile(f, 2)["time_tracking"]) == (False, False)
        f.client.json("POST", "/api/cameras/bulk", dict(camera_ids=[1, 3], action="TIME_OFF"))
        assert (profile(f, 1)["time_tracking"], profile(f, 1)["ai_enabled"]) == (False, True)
        assert profile(f, 3)["revision"] == 0, "Switching an already-off camera off is not a change"
        listed = {c["id"]: c["profile"]["time_tracking"] for c in f.client.json("GET", "/api/cameras")}
        assert listed == {1: False, 2: False, 3: False}


@test
def bulk_request_is_bounded_and_strictly_validated():
    with fixture() as f:
        for bad in (dict(camera_ids=[], action="AI_ON"), dict(camera_ids=[1, 1], action="AI_ON"),
                    dict(camera_ids=[0], action="AI_ON"), dict(camera_ids=list(range(1, 202)), action="AI_ON"),
                    dict(camera_ids=[1], action="DELETE"), dict(camera_ids=[1], action="ASSIGN_WORKPLACE"),
                    dict(camera_ids=[1], action="AI_ON", workplace_id=1), dict(camera_ids=[1], action="AI_ON", extra=1),
                    dict(camera_ids=[1], action="TIME_ON", workplace_id=1), dict(camera_ids=[1], action="TIME_OFF", extra=1)):
            f.client.json("POST", "/api/cameras/bulk", bad, status=422)


@test
def deleting_a_camera_removes_its_profile():
    with fixture() as f:
        f.client.json("DELETE", "/api/cameras/3", status=204)
        with Session(f.engine) as db:
            assert db.get(CameraProfile, 3) is None and db.get(CameraProfile, 1) is not None


if __name__ == "__main__":
    failed = 0
    for case in TESTS:
        try:
            case()
            print(f"PASS {case.__name__}")
        except Exception:
            failed += 1
            print(f"FAIL {case.__name__}")
            traceback.print_exc()
    print(f"{len(TESTS) - failed}/{len(TESTS)} passed")
    raise SystemExit(bool(failed))
