"""Journal time changes are opt-in, previewed against real accounting, and atomic."""
import sys
import unittest
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Barrier
from unittest.mock import patch
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi import FastAPI
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session
from app.api import attendance, human_visits
from app.api.global_persons import router
from app.config import settings
from app.database.database import get_db, immediate_session
from app.database.models import HumanVisit, ManualPresenceInterval, PresenceExclusion, PresenceSegment, OperatorRequest
from app.services import attendance_reports as reports
from app.services.presence_recorder import PresenceRecorder
from operator_test_support import Client
from test_presence_recorder import database
from test_human_visits_api import visit

START = datetime(2026, 3, 9, 19)  # local 2026-03-10 00:00

def local(hour, minute=0):
    return START + timedelta(hours=hour, minutes=minute)

def iso(dt):
    return dt.replace(tzinfo=timezone.utc).isoformat()


@contextmanager
def fixture(start=local(8), end=local(18), now=local(36)):
    with database() as engine:
        clock = [now]
        presence = PresenceRecorder(engine, now=lambda: clock[0])
        app = FastAPI(); app.state.presence = presence
        app.include_router(router); app.include_router(attendance.router)
        def isolated_db():
            with Session(engine) as db:
                yield db
        app.dependency_overrides[get_db] = isolated_db
        client = Client(app)
        identifier = visit(engine, started_at=start, last_seen_at=end)
        with patch.object(settings, 'attendance_utc_offset_minutes', 300), \
             patch.object(settings, 'attendance_live_gap_minutes', 35), \
             patch.object(settings, 'attendance_day_gap_minutes', 120), \
             patch.object(human_visits, 'utcnow', side_effect=lambda: clock[0]), \
             patch.object(attendance, '_now', side_effect=lambda: clock[0]):
            client.json('POST', f'/api/global-persons/visits/{identifier}/identity',
                        dict(request_id=str(uuid4()), actor='Operator', reason='Recognized employee',
                             expected_revision=0, mode='EMPLOYEE', employee_id=1))
            yield client, engine, identifier, clock, presence


def selection(**changes):
    return dict(request_id=str(uuid4()), actor='Operator', reason='Confirmed observed interval',
                employee_id=1, expected_identity_revision=1,
                started_at=iso(local(9)), ended_at=iso(local(10)), **changes)


def url(identifier, action):
    return f'/api/global-persons/visits/{identifier}/time/{action}'


def prepared(client, identifier, data):
    result = client.json('POST', url(identifier, 'preview'), data)
    return result, {**data, 'preview_sha256': result['preview_sha256']}


def seed(engine, employee, start, end):
    with immediate_session(engine) as db:
        db.add(PresenceSegment(employee_id=employee, camera_id=1, started_at=start,
                               last_seen_at=end, face_seen=True))


def counts(engine):
    with Session(engine) as db:
        return tuple(db.query(model).count() for model in (ManualPresenceInterval, PresenceExclusion, OperatorRequest))


class VisitTimeTests(unittest.TestCase):
    def test_identity_alone_does_not_change_hours_and_confirmed_preview_does(self):
        with fixture() as (client, engine, identifier, _, _):
            self.assertEqual(counts(engine), (0, 0, 1))
            preview, payload = prepared(client, identifier, selection())
            self.assertEqual([(r['before_ms'], r['after_ms'], r['delta_ms']) for r in preview['rows']],
                             [(0, 3600000, 3600000)])
            self.assertEqual(counts(engine), (0, 0, 1))
            applied = client.json('POST', url(identifier, 'apply'), payload)
            self.assertEqual(client.json('POST', url(identifier, 'apply'), payload), applied)
            self.assertEqual(counts(engine), (1, 0, 2))
            report = client.json('GET', '/api/attendance/report?from_date=2026-03-10&to_date=2026-03-10&employee_id=1')
            self.assertEqual(report['totals'][0]['worked_ms'], 3600000)
            # Restoring the AI label must not silently revoke separately confirmed time.
            client.json('POST', f'/api/global-persons/visits/{identifier}/identity/restore',
                        dict(request_id=str(uuid4()), actor='Operator', reason='Restore AI',
                             expected_revision=1, restore_revision=0))
            with Session(engine) as db:
                self.assertIsNone(db.query(ManualPresenceInterval).one().revoked_at)

    def test_exclusion_keeps_manual_time_and_preview_includes_bridging(self):
        with fixture() as (client, engine, identifier, _, _):
            seed(engine, 2, local(9), local(10))
            seed(engine, 1, local(8), local(8, 10))
            with immediate_session(engine) as db:
                db.add(ManualPresenceInterval(request_id=str(uuid4()), employee_id=2,
                    started_at=local(9, 20), ended_at=local(9, 30), actor='Other', reason='Independent', created_at=local(20)))
            preview, payload = prepared(client, identifier, selection(exclude_employee_id=2))
            changes = {r['employee_id']: (r['before_ms'], r['after_ms'], r['delta_ms']) for r in preview['rows']}
            self.assertEqual(changes[1], (600000, 7200000, 6600000))
            self.assertEqual(changes[2], (3600000, 600000, -3000000))
            applied = client.json('POST', url(identifier, 'apply'), payload)
            exclusion = next(r for r in applied['created'] if r['kind'] == 'EXCLUSION')
            client.json('POST', f"/api/attendance/corrections/exclusion/{exclusion['id']}/revoke",
                        dict(actor='Other', reason='Already revoked elsewhere'))
            undo = dict(request_id=str(uuid4()), operation_id=applied['operation_id'], actor='Operator', reason='Undo group')
            undone = client.json('POST', url(identifier, 'undo'), undo)
            self.assertEqual(client.json('POST', url(identifier, 'undo'), undo), undone)
            with Session(engine) as db:
                remaining = db.query(ManualPresenceInterval).filter_by(revoked_at=None).all()
                self.assertEqual([(r.employee_id, r.actor) for r in remaining], [(2, 'Other')])
            history = client.json('GET', url(identifier, 'history'))['items']
            self.assertEqual(len(history), 1)
            self.assertTrue(all(r['revoked_at'] for r in history[0]['created']))

    def test_midnight_split_and_millisecond_visit_bounds(self):
        with fixture(start=local(23,30).replace(microsecond=123456), end=local(24,30).replace(microsecond=123456)) as (client, engine, identifier, _, _):
            data = selection(); data.update(started_at=iso(local(23,30).replace(microsecond=123000)),
                                            ended_at=iso(local(24,30).replace(microsecond=123000)))
            preview, payload = prepared(client, identifier, data)
            self.assertEqual([r['delta_ms'] for r in preview['rows']], [1799877, 1800123])
            applied = client.json('POST', url(identifier, 'apply'), payload)
            self.assertEqual(len(applied['created']), 2)
            self.assertEqual(sum(r['delta_ms'] for r in preview['rows']), 3600000)

    def test_stale_inputs_settings_identity_and_midnight_reject_without_writes(self):
        for mutation in ('presence', 'settings', 'identity', 'boundary', 'midnight'):
            with self.subTest(mutation=mutation), fixture(now=local(23)) as (client, engine, identifier, clock, _):
                _, payload = prepared(client, identifier, selection())
                if mutation == 'presence': seed(engine, 1, local(9,15), local(9,45))
                if mutation == 'settings': settings.attendance_day_gap_minutes = 130
                if mutation == 'identity':
                    client.json('POST', f'/api/global-persons/visits/{identifier}/identity',
                                dict(request_id=str(uuid4()), actor='Other', reason='Different identity',
                                     expected_revision=1, mode='UNKNOWN'))
                if mutation == 'boundary':
                    with immediate_session(engine) as db:
                        db.get(HumanVisit, identifier).last_seen_at = local(9,30)
                if mutation == 'midnight': clock[0] = local(24)
                before = counts(engine)
                client.json('POST', url(identifier, 'apply'), payload, status=409)
                self.assertEqual(counts(engine), before)

    def test_future_out_of_visit_unknown_and_zero_intervals_are_not_previewed(self):
        with fixture(now=local(12)) as (client, engine, identifier, _, _):
            for changes in (dict(started_at=iso(local(7))), dict(ended_at=iso(local(19))),
                            dict(ended_at=iso(local(13))), dict(ended_at=iso(local(9))),
                            dict(employee_id=2), dict(exclude_employee_id=1)):
                data = selection(); data.update(changes)
                client.json('POST', url(identifier, 'preview'), data, status=422)
            self.assertEqual(counts(engine), (0, 0, 1))

    def test_failed_group_rolls_back_all_rows_and_receipt(self):
        with fixture() as (client, engine, identifier, _, _):
            _, payload = prepared(client, identifier, selection(exclude_employee_id=2))
            original = reports.insert_correction
            calls = [0]
            def fail_second(*args, **kwargs):
                calls[0] += 1
                if calls[0] == 2: raise SQLAlchemyError('injected second-row failure')
                return original(*args, **kwargs)
            with patch.object(reports, 'insert_correction', side_effect=fail_second):
                client.json('POST', url(identifier, 'apply'), payload, status=503)
            self.assertEqual(counts(engine), (0, 0, 1))

    def test_concurrent_apply_with_same_preview_has_one_effect(self):
        with fixture() as (client, engine, identifier, _, _):
            _, payload = prepared(client, identifier, selection())
            barrier = Barrier(2)
            def submit(request_id):
                barrier.wait()
                return client.request('POST', url(identifier, 'apply'), {**payload, 'request_id': request_id})[0]
            with ThreadPoolExecutor(max_workers=2) as pool:
                results = list(pool.map(submit, [str(uuid4()), str(uuid4())]))
            self.assertEqual(sorted(results), [200, 409])
            self.assertEqual(counts(engine), (1, 0, 2))

    def test_failed_recorder_flush_blocks_fresh_preview_but_not_committed_replay(self):
        with fixture() as (client, engine, identifier, _, presence):
            _, payload = prepared(client, identifier, selection())
            applied = client.json('POST', url(identifier, 'apply'), payload)
            presence.observe(1, 2, 'FACE', local(9)); presence.observe(1, 2, 'FACE', local(9,1))
            with patch('app.services.presence_recorder.immediate_session', side_effect=SQLAlchemyError('injected recorder failure')):
                client.json('POST', url(identifier, 'preview'), selection(), status=503)
                self.assertEqual(client.json('POST', url(identifier, 'apply'), payload), applied)
            self.assertEqual(counts(engine), (1, 0, 2))


    def test_live_visit_extension_does_not_expand_or_invalidate_fixed_interval(self):
        with fixture(now=local(20)) as (client, engine, identifier, _, _):
            _, payload = prepared(client, identifier, selection())
            with immediate_session(engine) as db:
                db.get(HumanVisit, identifier).last_seen_at = local(19)
            client.json('POST', url(identifier, 'apply'), payload)
            with Session(engine) as db:
                row = db.query(ManualPresenceInterval).one()
                self.assertEqual((row.started_at, row.ended_at), (local(9), local(10)))

    def test_live_presence_growth_keeps_the_confirmed_delta_applicable(self):
        for now, first, last, extended, growth in (
            (local(20), local(12), local(13), local(14), 3600000),
            (local(36), local(23), local(25), local(26), 0),
        ):
            with self.subTest(now=now), fixture(now=now) as (client, engine, identifier, _, _):
                seed(engine, 1, first, last)
                seed(engine, 2, local(8), last)
                preview, payload = prepared(client, identifier, selection(exclude_employee_id=2))
                with immediate_session(engine) as db:
                    for segment in db.query(PresenceSegment):
                        segment.last_seen_at = extended
                applied = client.json('POST', url(identifier, 'apply'), payload)
                self.assertEqual([row['delta_ms'] for row in applied['rows']],
                                 [row['delta_ms'] for row in preview['rows']])
                report = client.json('GET', '/api/attendance/report?from_date=2026-03-10&to_date=2026-03-10')
                totals = {row['employee_id']: row['worked_ms'] for row in report['totals']}
                for row in preview['rows']:
                    self.assertEqual(totals[row['employee_id']], row['after_ms'] + growth)
                self.assertEqual(counts(engine), (1, 1, 2))

    def test_changed_manual_or_exclusion_requires_new_preview_even_with_same_delta(self):
        for kind in ('manual', 'exclusion'):
            with self.subTest(kind=kind), fixture(now=local(20)) as (client, engine, identifier, _, _):
                seed(engine, 1, local(12), local(13))
                preview, payload = prepared(client, identifier, selection())
                data = dict(request_id=str(uuid4()), actor='Other', reason='Independent correction',
                            employee_id=1, started_at=iso(local(12,10)), ended_at=iso(local(12,20)))
                client.json('POST', f'/api/attendance/corrections/{kind}', data)
                updated, _ = prepared(client, identifier, selection())
                self.assertEqual(updated['rows'][0]['delta_ms'], preview['rows'][0]['delta_ms'])
                before = counts(engine)
                client.json('POST', url(identifier, 'apply'), payload, status=409)
                self.assertEqual(counts(engine), before)

if __name__ == '__main__':
    unittest.main()
