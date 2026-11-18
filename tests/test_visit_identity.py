"""Operator identity is independent of face availability and queued AI metadata."""
import sys
import unittest
from pathlib import Path
from uuid import uuid4
from datetime import timedelta
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi import HTTPException
from sqlalchemy.orm import Session
from app.database.models import Employee, HumanVisit, OperatorRequest
from app.services.employee_service import delete_employee
from app.time_utils import utcnow
from test_human_visits_api import api, visit


def identity_body(**changes):
    result = dict(request_id=str(uuid4()), actor="Operator", reason="Recognized from saved image",
                  expected_revision=0, mode="EMPLOYEE", employee_id=7)
    result.update(changes)
    return result


class IdentityTests(unittest.TestCase):
    def test_concurrent_operators_cannot_overwrite_each_other(self):
        with api() as (client, engine):
            identifier = visit(engine)
            barrier = Barrier(2)
            def submit(mode):
                barrier.wait()
                return client.request("POST", f"/api/global-persons/visits/{identifier}/identity",
                                      identity_body(mode=mode, employee_id=7 if mode == "EMPLOYEE" else None))[0]
            with ThreadPoolExecutor(max_workers=2) as pool:
                results = list(pool.map(submit, ("EMPLOYEE", "UNKNOWN")))
            self.assertEqual(sorted(results), [200, 409])
            history = client.json("GET", f"/api/global-persons/visits/{identifier}/identity/history")["items"]
            self.assertEqual(len(history), 1)

    def test_v6_migration_preserves_visits_images_and_attendance_with_backup(self):
        import sqlite3
        from contextlib import closing
        from app.database import operator_schema as op
        from app.database.database import Base
        from app.database.models import Camera, PresenceSegment
        from test_operator_storage import database, schema, BASE_TABLES, dump
        with database() as (engine, root):
            with engine.begin() as connection:
                op.create_legacy_employees(connection)
                for addition in op.EMPLOYEE_ADDITIONS:
                    connection.exec_driver_sql(f"ALTER TABLE employees ADD COLUMN {addition}")
                Base.metadata.create_all(connection, tables=[t for t in Base.metadata.sorted_tables
                                                             if t.name in BASE_TABLES])
                for ddl in (*op.SCHEMAS[6].values(), *op.INDEXES[6].values()):
                    connection.exec_driver_sql(ddl)
                tables, indexes = schema._expected(3, engine.dialect)
                for ddl in (*tables.values(), *indexes.values(), schema._ROLE_TRIGGER):
                    connection.exec_driver_sql(ddl)
                connection.exec_driver_sql("INSERT INTO operator_schema VALUES (1,6,?)", (utcnow().isoformat(sep=" "),))
                schema._seed(connection)
            with Session(engine) as db:
                db.add(Employee(id=7, name="Existing employee"))
                db.add(Camera(id=1, name="Existing camera", nvr_host="unused",
                              username="unused", password="unused", channel=1))
                db.flush()
                db.add(PresenceSegment(employee_id=7, camera_id=1, started_at=utcnow()-timedelta(hours=1),
                                       last_seen_at=utcnow(), face_seen=True))
                db.commit()
            identifier = visit(engine, employee_id=7, identity_state="FACE_CONFIRMED")
            before = dump(engine)
            schema.initialize_attendance_schema(engine)
            with Session(engine) as db:
                self.assertEqual(db.get(HumanVisit, identifier).snapshot, b"\xff\xd8saved-person\xff\xd9")
                self.assertEqual(db.get(HumanVisit, identifier).employee_id, 7)
                self.assertEqual(db.query(PresenceSegment).count(), 1)
            backups = list((root/"backups").glob("*.db"))
            self.assertEqual(len(backups), 1)
            with closing(sqlite3.connect(backups[0])) as backup:
                self.assertEqual(tuple(backup.iterdump()), before)
            installed = dump(engine)
            schema.initialize_attendance_schema(engine)
            self.assertEqual(dump(engine), installed)

    def test_missing_face_can_be_assigned_and_survives_ai_update(self):
        with api() as (client, engine):
            identifier = visit(engine, snapshot=None, face_thumbnail=None, face_quality=None)
            url = f"/api/global-persons/visits/{identifier}"
            saved = client.json("POST", url + "/identity", identity_body())
            self.assertEqual(saved["identity_revision"], 1)
            with Session(engine) as db:
                db.add(Employee(id=8, name="AI's different employee"))
                db.flush()
                row = db.get(HumanVisit, identifier)
                row.employee_id, row.identity_state = 8, "FACE_CONFIRMED"
                db.commit()
            # Fresh request/session reads the manual label and the newer AI evidence.
            detail = client.json("GET", url)
            self.assertEqual((detail["employee_id"], detail["identity_source"]), (7, "OPERATOR"))
            self.assertEqual((detail["ai_employee_id"], detail["ai_identity_state"]), (8, "FACE_CONFIRMED"))
            self.assertFalse(detail["employee_sample_eligible"])
            selected = client.json("GET", "/api/global-persons/visits?employee_id=7")["items"]
            self.assertEqual([r["id"] for r in selected], [identifier])
            self.assertEqual(client.json("GET", "/api/global-persons/visits?employee_id=8")["items"], [])
            self.assertNotIn("snapshot", selected[0])

    def test_unknown_restore_and_idempotency_are_audited(self):
        with api() as (client, engine):
            identifier = visit(engine)
            url = f"/api/global-persons/visits/{identifier}"
            body = identity_body()
            original = client.json("POST", url + "/identity", body)
            self.assertEqual(client.json("POST", url + "/identity", body), original)
            client.json("POST", url + "/identity", {**body, "reason": "Changed"}, status=409)
            client.json("POST", url + "/identity", identity_body(), status=409)
            unknown = identity_body(expected_revision=1, mode="UNKNOWN", employee_id=None)
            client.json("POST", url + "/identity", unknown)
            detail = client.json("GET", url)
            self.assertIsNone(detail["employee_id"])
            self.assertEqual(detail["identity_state"], "OPERATOR_UNKNOWN")
            self.assertEqual(len(client.json("GET", "/api/global-persons/visits?unknown_only=true")["items"]), 1)
            restore = dict(request_id=str(uuid4()), actor="Operator", reason="Undo mistaken unknown",
                           expected_revision=2, restore_revision=1)
            client.json("POST", url + "/identity/restore", restore)
            self.assertEqual(client.json("GET", url)["employee_id"], 7)
            history = client.json("GET", url + "/identity/history")["items"]
            self.assertEqual([r["revision"] for r in history], [3, 2, 1])
            self.assertEqual([r["mode"] for r in history], ["EMPLOYEE", "UNKNOWN", "EMPLOYEE"])
            restore.update(request_id=str(uuid4()), expected_revision=3, restore_revision=0)
            client.json("POST", url + "/identity/restore", restore)
            detail = client.json("GET", url)
            self.assertEqual((detail["identity_source"], detail["identity_revision"]), ("AI", 4))
            with Session(engine) as db:
                self.assertEqual(db.query(OperatorRequest).count(), 4)
                self.assertIsNone(db.get(HumanVisit, identifier).employee_id)

    def test_expired_images_and_inactive_employee_do_not_block_history(self):
        with api() as (client, engine):
            identifier = visit(engine, image_expires_at=utcnow() - timedelta(days=1))
            with Session(engine) as db:
                db.get(Employee, 7).active = False
                db.commit()
            url = f"/api/global-persons/visits/{identifier}"
            client.json("POST", url + "/identity", identity_body())
            detail = client.json("GET", url)
            self.assertEqual(detail["employee_id"], 7)
            self.assertFalse(detail["employee_active"])
            self.assertEqual(detail["photo_status"], "EXPIRED")
            with Session(engine) as db:
                with self.assertRaises(HTTPException) as exc:
                    delete_employee(db, db.get(Employee, 7))
                self.assertEqual(exc.exception.status_code, 409)

    def test_invalid_identity_and_missing_entities_do_not_create_receipts(self):
        with api() as (client, engine):
            identifier = visit(engine)
            url = f"/api/global-persons/visits/{identifier}"
            for changes in (dict(mode="EMPLOYEE", employee_id=None), dict(mode="AI"),
                            dict(employee_id=True), dict(expected_revision=-1), dict(actor=" "),
                            dict(unrecognized=True)):
                client.json("POST", url + "/identity", identity_body(**changes), status=422)
            client.json("POST", url + "/identity", identity_body(employee_id=999), status=404)
            client.json("POST", f"/api/global-persons/visits/{uuid4()}/identity", identity_body(), status=404)
            with Session(engine) as db:
                self.assertEqual(db.query(OperatorRequest).count(), 0)


if __name__ == "__main__":
    unittest.main()
