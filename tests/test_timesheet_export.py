"""Filtered timesheet/export behavior over real SQLite and native Excel round trips."""
import io
import sys
import traceback
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from openpyxl import load_workbook
from sqlalchemy.orm import Session
from app.database.models import Employee, EmployeeWorkplace, Workplace
from test_attendance_api import api, get, seed, local, DAY_URL

TESTS = []


def test(fn):
    TESTS.append(fn)
    return fn


def people(engine):
    with Session(engine) as db:
        anna = db.get(Employee, 1)
        anna.first_name, anna.last_name = 'Анна', 'Смирнова'
        db.get(Employee, 2).active = False
        db.add(Employee(id=3, name='=HYPERLINK("https://example.invalid")', active=True,
                        first_name='=USER()', last_name='=NAME()'))
        db.add_all([Workplace(id=1, name='Цех', name_key='цех', active=True),
                    Workplace(id=2, name='=WORKPLACE()', name_key='=workplace()', active=True)])
        db.flush()
        db.add_all([EmployeeWorkplace(employee_id=1, workplace_id=1),
                    EmployeeWorkplace(employee_id=1, workplace_id=2),
                    EmployeeWorkplace(employee_id=2, workplace_id=2),
                    EmployeeWorkplace(employee_id=3, workplace_id=1)])
        db.commit()


def six_hours(engine, days_ago=0):
    start = local(9) - timedelta(days=days_ago)
    seed(engine, 1, 1, start, start + timedelta(hours=3))
    seed(engine, 1, 2, start + timedelta(hours=6), start + timedelta(hours=9))


def excel(client, query):
    code, headers, body = client.request('GET', '/api/attendance/report.xlsx?' + query)
    assert code == 200, (code, body)
    assert headers[b'content-type'].startswith(b'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    assert b'.xlsx' in headers[b'content-disposition']
    return load_workbook(io.BytesIO(body), data_only=False)


@test
def workplace_selects_employees_without_losing_their_other_camera_hours_or_duplicating_memberships():
    with api() as (client, engine):
        people(engine)
        six_hours(engine)
        seed(engine, 2, 1, local(9), local(10))
        report = get(client, f'/api/attendance/report?{DAY_URL}&workplace_id=1')
        assert {e['id'] for e in report['employees']} == {1, 3}
        assert [(row['employee_id'], row['worked_ms']) for row in report['rows']] == [(1, 21600000)]
        assert report['totals'][0]['worked_ms'] == 21600000
        second = get(client, f'/api/attendance/report?{DAY_URL}&workplace_id=2&employee_status=ACTIVE')
        assert [e['id'] for e in second['employees']] == [1]
        assert second['totals'][0]['worked_ms'] == 21600000
        empty = get(client, f'/api/attendance/report?{DAY_URL}&workplace_id=1&employee_id=2')
        assert not empty['employees'] and not empty['rows']


@test
def employee_status_and_workplace_filters_apply_to_csv_and_reject_bad_values():
    with api() as (client, engine):
        people(engine)
        six_hours(engine)
        seed(engine, 2, 1, local(9), local(10))
        report = get(client, f'/api/attendance/report?{DAY_URL}&employee_status=INACTIVE')
        assert [e['id'] for e in report['employees']] == [2]
        assert [row['employee_id'] for row in report['rows']] == [2]
        code, _h, body = client.request('GET', f'/api/attendance/report.csv?{DAY_URL}&employee_status=INACTIVE')
        assert code == 200
        text = body.decode('utf-8-sig')
        assert 'Борис' in text and 'Анна' not in text
        get(client, f'/api/attendance/report?{DAY_URL}&employee_status=invalid', 422)
        get(client, f'/api/attendance/report?{DAY_URL}&workplace_id=999', 404)


@test
def workbook_has_numeric_daily_hours_thirty_hour_total_and_zero_hour_selected_employee():
    with api() as (client, engine):
        people(engine)
        for days_ago in range(5):
            six_hours(engine, days_ago)
        book = excel(client, 'from_date=2026-03-06&to_date=2026-03-12&workplace_id=1')
        sheet = book['Рабочее время']
        headers = [cell.value for cell in sheet[1]]
        assert headers[:4] == ['Фамилия', 'Имя', 'Сотрудник', 'Рабочие места']
        assert len(headers) == 12 and headers[-1] == 'Итого'
        anna = next(row for row in sheet.iter_rows(min_row=2) if row[0].value == 'Смирнова')
        zero = next(row for row in sheet.iter_rows(min_row=2) if row[0].value == '=NAME()')
        assert anna[1].value == 'Анна'
        assert anna[4].value == timedelta(hours=6)
        assert anna[-1].value == timedelta(hours=30) and anna[-1].number_format == '[h]:mm'
        assert zero[-1].value == timedelta(0)
        assert anna[-2].value is None  # March 12 is in the future, not a worked zero
        assert anna[-3].value == timedelta(0)  # March 11 is today with no sightings
        assert all(cell.data_type == 's' for cell in zero[:3])
        assert anna[3].data_type == 's' and '=WORKPLACE()' in anna[3].value
        assert all(cell.data_type != 'f' for row in sheet for cell in row)
        assert 'Параметры' in book.sheetnames
        book.close()


@test
def empty_selection_still_produces_a_valid_workbook_and_full_name_survives_legacy_names():
    with api() as (client, engine):
        people(engine)
        book = excel(client, DAY_URL + '&employee_id=2')
        sheet = book['Рабочее время']
        assert sheet['C2'].value == 'Борис'
        assert sheet['A2'].value is None and sheet['B2'].value is None
        assert sheet['F2'].value == timedelta(0)
        book.close()
        book = excel(client, DAY_URL + '&workplace_id=1&employee_id=2')
        assert book['Рабочее время'].max_row == 1
        book.close()


@test
def supervised_body_attribution_changes_new_observations_not_existing_export_values():
    from datetime import timezone
    from threading import BoundedSemaphore
    from unittest.mock import patch
    from uuid import uuid4

    import numpy as np

    from app.ai.employee_appearance import EmployeeAppearanceGallery
    from app.ai.recognition_learning import RecognitionLearningStore
    from app.ai.recognition_service import RecognitionService
    from app.ai.worker import AIWorker, DetectionState
    from app.database.database import immediate_session
    from app.database.models import AttendanceCamera, Camera, CameraProfile, HumanVisit, PresenceSegment
    from app.services.human_journal import HumanJournal
    from app.services.presence_recorder import PresenceRecorder
    from app.services.recognition_learning_service import (
        LearningDecisionInput, LearningEvidenceInput, LearningRevokeInput, RecognitionLearningService,
    )
    from attendance_test_support import BufferedReader, Clock, FRAME, PersonObservations, Readers
    from test_human_journal import idle
    from test_recognition_learning import Body, make_upload
    from test_recognition_learning_gallery import BODY_MODEL, direction

    class NativeBodyDouble:
        model_id = BODY_MODEL

        def embed(self, crops):
            return np.stack([direction(0.) if crop[0, 0, 0] > 100 else direction() for crop in crops])

    def report_values(client):
        query = DAY_URL + '&workplace_id=1'
        report = get(client, f'/api/attendance/report?{query}')
        with_workbook = excel(client, query)
        try:
            values = {row[2]: (row[4], row[-1]) for row in
                      with_workbook['Рабочее время'].iter_rows(min_row=2, values_only=True)}
            assert len(values) == 2 and with_workbook['Рабочее время'].max_row == 3
        finally:
            with_workbook.close()
        return ([(row['employee_id'], row['worked_ms'], row['face_seen']) for row in report['rows']],
                [(total['employee_id'], total['worked_ms']) for total in report['totals']], values)

    with api() as (client, engine):
        clock, training = Clock(), str(uuid4())
        now = local(9)
        clock.utc = now - timedelta(seconds=10)
        with immediate_session(engine) as db:
            db.get(Employee, 1).first_name, db.get(Employee, 1).last_name = 'Анна', 'Первая'
            db.get(Employee, 2).first_name, db.get(Employee, 2).last_name = 'Борис', 'Другой'
            db.add_all([Workplace(id=1, name='Цех', name_key='цех'), Workplace(id=2, name='Офис', name_key='офис')])
            db.add(Camera(id=3, name='Камера без учёта', nvr_host='unused', username='', password='', channel=3))
            db.flush()
            db.add_all([EmployeeWorkplace(employee_id=1, workplace_id=1), EmployeeWorkplace(employee_id=1, workplace_id=2),
                        EmployeeWorkplace(employee_id=2, workplace_id=1)])
            for camera_id in (1, 2, 3):
                db.add(AttendanceCamera(camera_id=camera_id, source_revision=1, enabled=False))
                db.add(CameraProfile(camera_id=camera_id, zone='INTERIOR', ai_enabled=True, time_tracking=False,
                                     workplace_id=1 if camera_id != 2 else 2))
            db.add(HumanVisit(id=training, camera_id=1, tracking_generation='review', track_id=99,
                              source_revision=1, stream_id='historical', started_at=clock.utc, last_seen_at=clock.utc,
                              status='CLOSED', ended_at=clock.utc, end_reason='TRACK_EXPIRED', identity_state='UNKNOWN',
                              last_bbox=[0., 0., 1., 1.], max_person_score=.9, image_expires_at=now + timedelta(days=1)))

        recorder = PresenceRecorder(engine, now=clock.utcnow)
        store = RecognitionLearningStore(engine, BODY_MODEL)
        gallery = EmployeeAppearanceGallery()
        learning = RecognitionLearningService(engine, Body(), None, None, BoundedSemaphore(1),
                                              learning_store=store, appearance_gallery=gallery, now=clock.utcnow)
        journal = HumanJournal(engine, now=clock.utcnow, monotonic=clock.monotonic, min_hits=1)
        idle(journal)
        detector = PersonObservations()
        detector.y = .8
        readers = Readers()
        readers.readers[3] = BufferedReader()
        for reader in readers.readers.values():
            reader.source_revision = 1
        worker = AIWorker(readers, RecognitionService(detector, None, None, min_hits=1), recorder,
                          DetectionState(), journal=journal, reid=NativeBodyDouble(),
                          appearance_gallery=gallery, learning_store=store)
        upload = make_upload()
        source = LearningEvidenceInput(source='UPLOAD', crop=(0., 0., 1., 1.),
                                      captured_at=clock.utc.replace(tzinfo=timezone.utc))

        def decide(action, crop, request_id):
            scan = learning.analyze(training, crop, upload)
            payload = LearningDecisionInput(**crop.model_dump(), actor='Operator', reason='Reviewed capture',
                request_id=request_id, action=action, person_confirmed=True, employee_id=1,
                expected_identity_revision=scan['identity_revision'], expected_employee_revision=0,
                expected_learning_revision=scan['learning_revision'], expected_gallery_revision=scan['gallery_revision'],
                expected_body_model_id=scan['body_model_id'], expected_face_model_id=scan['face_model_id'],
                expected_image_sha256=scan['evidence']['image_sha256'])
            return learning.decide(training, payload, upload)

        def frame(seconds, cameras, image=FRAME, stream='fresh'):
            clock.utc = now + timedelta(seconds=seconds)
            clock.mono = 2000. + seconds
            for camera in cameras:
                readers.get(camera).buffer.put(image, stream_id=stream, source_revision=1,
                                                received_at=clock.utc, received_mono=clock.mono)
            worker.step()
            idle(journal)

        try:
            before = report_values(client)
            assert before[0] == [] and before[1] == []
            accepted = decide('ACCEPT', source, uuid4())
            assert report_values(client) == before, 'Operator evidence must not create observed time'
            with patch('app.ai.worker.time.monotonic', side_effect=clock.monotonic), \
                    patch('app.ai.worker.utcnow', side_effect=clock.utcnow):
                # Warm the real identities while time tracking is off. Turning time tracking on
                # must not reset these trackers, and this isolates the exact credited intervals.
                for seconds in range(-10, 0):
                    frame(seconds, (1, 2))
                for camera in (1, 2):
                    tracks = worker.state.camera(camera)['tracks']
                    assert tracks and tracks[0]['employee_id'] == 1
                    assert tracks[0]['identity_source'] == 'REID'
                for seconds in range(19):
                    if seconds in (0, 6):
                        with immediate_session(engine, publish=True) as db:
                            db.get(CameraProfile, 1 if seconds == 0 else 2).time_tracking = True
                    frame(seconds, (1, 2) if seconds <= 12 else (2,))
                for seconds in range(3 * 3600, 3 * 3600 + 12):
                    frame(seconds, (3,))
                assert worker.state.camera(3)['tracks'][0]['employee_id'] == 1
                ambiguous = np.full_like(FRAME, 200)
                for seconds in range(6 * 3600, 6 * 3600 + 12):
                    frame(seconds, (1,), ambiguous, 'different-visit')
                assert worker.state.camera(1)['tracks'][0]['employee_id'] is None
            recorder.flush()
            with Session(engine) as db:
                segments = db.query(PresenceSegment).order_by(PresenceSegment.camera_id).all()
                assert [(r.camera_id, r.employee_id, r.started_at, r.last_seen_at) for r in segments] == [
                    (1, 1, now, now + timedelta(seconds=12)),
                    (2, 1, now + timedelta(seconds=6), now + timedelta(seconds=18)),
                ]
            union = timedelta(seconds=12 + (18 - 6) - (12 - 6))
            expected_ms = union // timedelta(milliseconds=1)
            learned = report_values(client)
            assert learned[0] == [(1, expected_ms, False)] and learned[1] == [(1, expected_ms)]
            assert sorted(learned[2].values()) == [(timedelta(0), timedelta(0)), (union, union)]
            learning.revoke(accepted['feedback_id'], LearningRevokeInput(request_id=uuid4(), actor='Operator',
                            reason='Remove reference', expected_identity_revision=None))
            decide('REJECT', source, uuid4())
            assert report_values(client) == learned, 'Revocation and rejection never rewrite recorded values'
        finally:
            learning.close()
            assert journal.close().drained


if __name__ == '__main__':
    failures = 0
    for case in TESTS:
        try:
            case()
            print(f'PASS {case.__name__}')
        except Exception:
            failures += 1
            print(f'FAIL {case.__name__}')
            traceback.print_exc()
    print(f'{len(TESTS)-failures}/{len(TESTS)} passed')
    raise SystemExit(bool(failures))
