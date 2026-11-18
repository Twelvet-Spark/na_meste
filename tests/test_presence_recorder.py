"""PresenceRecorder: segment joining, flushing, crossings, deleted rows, notifications.

Run: .venv/Scripts/python.exe tests/test_presence_recorder.py
All databases are temporary files.
"""
import sys
import tempfile
import traceback
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from threading import Event, Thread
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import create_engine, event
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from app.database import attendance_schema as schema
from app.database.models import Camera, DoorCrossing, Employee, PresenceSegment
from app.services.presence_recorder import PresenceRecorder

TESTS = []
T0 = datetime(2026, 3, 10, 9, 0, 0)  # naive UTC; local day starts 2026-03-10 00:00 +05:00 = 03-09 19:00 UTC


def test(fn):
    TESTS.append(fn)
    return fn


class Notifier:
    def __init__(self):
        self.sent = []

    def send_text(self, text):
        self.sent.append(text)


@contextmanager
def database():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        engine = create_engine(f"sqlite:///{(root / 'presence.db').as_posix()}",
                               connect_args={"timeout": 30, "check_same_thread": False})

        @event.listens_for(engine, "connect")
        def foreign_keys(connection, _record):
            connection.execute("PRAGMA foreign_keys=ON")

        try:
            with patch.object(schema, "BACKUP_DIR", root / "backups"):
                schema.initialize_attendance_schema(engine)
                with Session(engine) as db:
                    db.add_all([Employee(id=1, name="Анна", active=True, revision=0),
                                Employee(id=2, name="Борис", active=True, revision=0)])
                    db.add_all([Camera(id=1, name="Вход", nvr_host="h", username="u", password="p", channel=1),
                                Camera(id=2, name="Зал", nvr_host="h", username="u", password="p", channel=2)])
                    db.commit()
                yield engine
        finally:
            engine.dispose()


def rows(engine, model, *order):
    with Session(engine) as db:
        return db.query(model).order_by(*order).all()


def at(seconds):
    return T0 + timedelta(seconds=seconds)


@test
def sightings_up_to_sixty_seconds_apart_join_and_longer_gaps_split():
    with database() as engine:
        recorder = PresenceRecorder(engine, now=lambda: at(500))
        recorder.observe(1, 1, "REID", at(0))
        recorder.observe(1, 1, "REID", at(60))
        recorder.observe(1, 1, "FACE", at(121))  # 61 s after the last sighting: new segment
        recorder.flush()
        stored = rows(engine, PresenceSegment, PresenceSegment.id)
        assert [(s.started_at, s.last_seen_at, s.face_seen) for s in stored] == [
            (at(0), at(60), False), (at(121), at(121), True)]


@test
def flush_persists_then_updates_last_seen_and_face_flag():
    with database() as engine:
        clock = [at(10)]
        recorder = PresenceRecorder(engine, now=lambda: clock[0])
        recorder.observe(1, 1, "REID", at(0))
        recorder.flush()
        recorder.observe(1, 1, "FACE", at(30))
        recorder.observe(1, 1, "REID", at(20))  # older than last_seen: ignored
        recorder.flush()
        (stored,) = rows(engine, PresenceSegment)
        assert (stored.started_at, stored.last_seen_at, stored.face_seen) == (at(0), at(30), True)
        assert recorder.snapshot()["open_segments"] == 1 and recorder.snapshot()["last_error"] is None
        clock[0] = at(200)
        recorder.flush()
        assert recorder.snapshot()["open_segments"] == 0  # stale and persisted: forgotten


@test
def two_cameras_make_two_segments_for_one_employee():
    with database() as engine:
        recorder = PresenceRecorder(engine, now=lambda: at(5))
        recorder.observe(1, 1, "FACE", at(0))
        recorder.observe(2, 1, "REID", at(1))
        recorder.flush()
        assert {(s.camera_id, s.face_seen) for s in rows(engine, PresenceSegment)} == {(1, True), (2, False)}


@test
def crossing_is_idempotent_by_source_key_and_survives_replay():
    with database() as engine:
        recorder = PresenceRecorder(engine, now=lambda: at(50))
        for _ in range(2):
            recorder.record_crossing(source_key="camera:1:s:g:7:0", camera_id=1, employee_id=1,
                                     direction="EXIT", occurred_at=at(3))
            recorder.flush()
        (stored,) = rows(engine, DoorCrossing)
        assert (stored.direction, stored.occurred_at, stored.recorded_at) == ("EXIT", at(3), at(50))
        assert recorder.snapshot()["pending_crossings"] == 0


@test
def deleted_employee_or_camera_is_dropped_without_a_stuck_error():
    with database() as engine:
        recorder = PresenceRecorder(engine, now=lambda: at(5))
        recorder.observe(1, 99, "FACE", at(0))
        recorder.observe(77, 1, "FACE", at(0))
        recorder.record_crossing(source_key="k", camera_id=1, employee_id=99, direction="ENTRY", occurred_at=at(0))
        recorder.observe(1, 2, "FACE", at(1))
        recorder.flush()
        assert [(s.employee_id, s.camera_id) for s in rows(engine, PresenceSegment)] == [(2, 1)]
        assert not rows(engine, DoorCrossing)
        snapshot = recorder.snapshot()
        assert snapshot["last_error"] is None and snapshot["pending_crossings"] == 0
        assert not recorder.has_employee_work(99) and not recorder.has_camera_work(77)


@test
def first_sighting_of_the_day_notifies_once_even_after_a_recorder_restart():
    with database() as engine:
        notifier = Notifier()
        first = PresenceRecorder(engine, notifier, now=lambda: at(500))
        first.observe(1, 1, "FACE", at(0))
        first.observe(2, 1, "REID", at(2))  # same employee, other camera, same flush
        first.observe(1, 2, "FACE", at(1))
        first.flush()
        assert len(notifier.sent) == 2 and notifier.sent[0].startswith("На месте: Анна, 2026-03-10T14:00:00")
        first.observe(1, 1, "FACE", at(20))
        first.flush()
        restarted = PresenceRecorder(engine, notifier, now=lambda: at(900))
        restarted.observe(1, 1, "FACE", at(700))
        restarted.flush()
        assert len(notifier.sent) == 2, notifier.sent
        next_day = PresenceRecorder(engine, notifier, now=lambda: at(90000))
        next_day.observe(1, 1, "FACE", at(86400 + 60))
        next_day.flush()
        assert len(notifier.sent) == 3


@test
def earlier_replayed_sighting_after_restart_does_not_notify_again():
    with database() as engine:
        notifier = Notifier()
        recorder = PresenceRecorder(engine, notifier, now=lambda: at(900))
        recorder.observe(1, 1, "FACE", at(700))
        recorder.flush()
        restarted = PresenceRecorder(engine, notifier, now=lambda: at(900))
        restarted.observe(2, 1, "FACE", at(0))
        restarted.flush()
        assert len(notifier.sent) == 1, notifier.sent


@test
def notifications_respect_the_setting_and_notifier_failures_are_contained():
    class Broken:
        def send_text(self, _text):
            raise RuntimeError("telegram down")

    with database() as engine:
        recorder = PresenceRecorder(engine, Broken(), now=lambda: at(5))
        recorder.observe(1, 1, "FACE", at(0))
        recorder.flush()  # must not raise
        assert len(rows(engine, PresenceSegment)) == 1
        notifier = Notifier()
        with patch("app.services.presence_recorder.settings.telegram_notify_presence", False):
            quiet = PresenceRecorder(engine, notifier, now=lambda: at(5))
            quiet.observe(1, 2, "FACE", at(0))
            quiet.flush()
        assert notifier.sent == []


@test
def pending_work_is_visible_per_employee_and_camera():
    with database() as engine:
        recorder = PresenceRecorder(engine, now=lambda: at(5))
        assert not recorder.has_employee_work(1) and not recorder.has_camera_work(1)
        recorder.observe(1, 1, "FACE", at(0))
        recorder.record_crossing(source_key="k", camera_id=2, employee_id=2, direction="ENTRY", occurred_at=at(0))
        assert recorder.has_employee_work(1) and recorder.has_camera_work(1)
        assert recorder.has_employee_work(2) and recorder.has_camera_work(2)
        assert not recorder.has_employee_work(3)


@test
def crossing_queue_is_bounded_and_counts_drops():
    with database() as engine:
        recorder = PresenceRecorder(engine, now=lambda: at(5))
        for index in range(1005):
            recorder.record_crossing(source_key=f"k{index}", camera_id=1, employee_id=1,
                                     direction="ENTRY", occurred_at=at(0))
        snapshot = recorder.snapshot()
        assert snapshot["pending_crossings"] == 1000 and snapshot["dropped_crossings"] == 5
        recorder.flush()
        assert len(rows(engine, DoorCrossing)) == 1000 and not rows(engine, DoorCrossing)[0].source_key == "k0"


@test
def database_error_keeps_data_sets_last_error_and_recovers():
    with database() as engine:
        recorder = PresenceRecorder(engine, now=lambda: at(5))
        recorder.observe(1, 1, "FACE", at(0))
        recorder.record_crossing(source_key="k", camera_id=1, employee_id=1, direction="EXIT", occurred_at=at(1))
        with patch("app.services.presence_recorder.immediate_session",
                   side_effect=OperationalError("x", {}, Exception("locked"))):
            recorder.flush()
            recorder.flush()
        snapshot = recorder.snapshot()
        assert snapshot["last_error"] == "PERSISTENCE_ERROR" and snapshot["pending_crossings"] == 1
        assert snapshot["open_segments"] == 1 and not rows(engine, PresenceSegment)
        recorder.flush()
        assert recorder.snapshot()["last_error"] is None
        assert len(rows(engine, PresenceSegment)) == 1 and len(rows(engine, DoorCrossing)) == 1


@test
def observations_extended_and_split_during_flush_survive_the_next_flush():
    with database() as engine:
        notifier = Notifier()
        recorder = PresenceRecorder(engine, notifier, now=lambda: at(200))
        recorder.observe(1, 1, "REID", at(0))
        captured, resume = Event(), Event()
        errors = []

        def pause_insert(_connection, _cursor, statement, _parameters, _context, _many):
            if statement.startswith("INSERT INTO attendance_presence_segments") and not captured.is_set():
                captured.set()
                assert resume.wait(5), "test did not release the captured flush"

        def flushing():
            try:
                recorder.flush()
            except BaseException as exc:
                errors.append(exc)

        event.listen(engine, "before_cursor_execute", pause_insert)
        thread = Thread(target=flushing)
        thread.start()
        try:
            assert captured.wait(5), "flush never reached the first segment insert"
            recorder.observe(1, 1, "FACE", at(20))  # dirty version changes after capture
            recorder.observe(1, 1, "REID", at(100))  # captured segment moves into closing
        finally:
            resume.set()
            thread.join(5)
            event.remove(engine, "before_cursor_execute", pause_insert)
        assert not thread.is_alive() and not errors, errors
        recorder.flush()
        stored = rows(engine, PresenceSegment, PresenceSegment.started_at)
        assert [(s.started_at, s.last_seen_at, s.face_seen) for s in stored] == [
            (at(0), at(20), True), (at(100), at(100), False)]
        assert len(notifier.sent) == 1


@test
def failure_after_insert_or_at_commit_rolls_back_and_retries_without_lost_rows_or_notifications():
    for failure_point in ("after_cursor_execute", "commit"):
        with database() as engine:
            notifier = Notifier()
            recorder = PresenceRecorder(engine, notifier, now=lambda: at(5))
            recorder.observe(1, 1, "FACE", at(0))
            recorder.record_crossing(source_key="rollback", camera_id=1, employee_id=1,
                                     direction="ENTRY", occurred_at=at(1))

            def fail_after_insert(_connection, _cursor, statement, _parameters, _context, _many):
                if statement.startswith("INSERT INTO attendance_presence_segments"):
                    raise OperationalError(statement, {}, Exception("injected after insert"))

            def fail_at_commit(_connection):
                raise OperationalError("COMMIT", {}, Exception("injected at commit"))

            listener = fail_after_insert if failure_point == "after_cursor_execute" else fail_at_commit
            event.listen(engine, failure_point, listener)
            try:
                recorder.flush()
            finally:
                event.remove(engine, failure_point, listener)
            assert not rows(engine, PresenceSegment) and not rows(engine, DoorCrossing)
            assert notifier.sent == [] and recorder.snapshot()["last_error"] == "PERSISTENCE_ERROR"
            recorder.flush()
            (segment,) = rows(engine, PresenceSegment)
            (crossing,) = rows(engine, DoorCrossing)
            assert (segment.started_at, segment.last_seen_at, segment.face_seen) == (at(0), at(0), True)
            assert crossing.source_key == "rollback" and crossing.occurred_at == at(1)
            assert len(notifier.sent) == 1 and recorder.snapshot()["last_error"] is None


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
