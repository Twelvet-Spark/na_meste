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
