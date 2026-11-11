"""PresenceManager — жизненный цикл сессий присутствия (ТЗ №7-8).

Вместо событий на каждый кадр — одна открытая сессия на (сотрудник, камера):
- подтверждено распознавание → сессия открывается или продлевается (last_seen_at);
- человек не наблюдается дольше PRESENCE_END_TIMEOUT → сессия закрывается
  (ended_at = last_seen_at);
- запись last_seen троттлится (не чаще раза в 5 сек на сессию).
"""
import logging
import threading
import time
from datetime import datetime, timedelta, timezone

from sqlalchemy.orm import sessionmaker

from app import time_utils
from app.database.models import Camera, Employee, PresenceSession

logger = logging.getLogger("app.services.presence")

WRITE_THROTTLE = 5.0     # сек между записями last_seen одной сессии
TICK_INTERVAL = 5.0      # сек между проверками истёкших сессий


class PresenceManager:
    def __init__(self, session_factory: sessionmaker, end_timeout: float = 30.0,
                 notifier=None):
        self._session_factory = session_factory
        self.end_timeout = end_timeout
        self.notifier = notifier
        self._last_write: dict[tuple[int, int], datetime] = {}
        self._last_tick = 0.0
        self._lock = threading.Lock()

    def on_recognized(self, camera_id: int, employee_id: int, confidence: float) -> None:
        """Подтверждённое распознавание: открыть/продлить сессию."""
        now = time_utils.utcnow()
        key = (employee_id, camera_id)
        with self._lock:
            if now - self._last_write.get(key, datetime.min) < timedelta(seconds=WRITE_THROTTLE):
                # слишком часто — сессия точно открыта, обновим позже
                return
            self._last_write[key] = now

        try:
            with self._session_factory() as db:
                session = db.query(PresenceSession).filter(
                    PresenceSession.employee_id == employee_id,
                    PresenceSession.camera_id == camera_id,
                    PresenceSession.ended_at.is_(None),
                ).first()
                if session is None:
                    session = PresenceSession(
                        employee_id=employee_id,
                        camera_id=camera_id,
                        started_at=now,
                        last_seen_at=now,
                        confidence=confidence,
                    )
                    db.add(session)
                    db.commit()
                    logger.info(
                        "Presence started employee=%d camera=%d", employee_id, camera_id
                    )
                    self._notify_start(db, session)
                else:
                    session.last_seen_at = now
                    session.confidence = confidence
                    db.commit()
        except Exception:
            logger.exception("Presence: ошибка записи сессии (emp=%d cam=%d)",
                             employee_id, camera_id)

    def _notify_start(self, db, session: PresenceSession) -> None:
        """Уведомление в Telegram о начале присутствия."""
        if self.notifier is None:
            return
        try:
            employee = db.get(Employee, session.employee_id)
            camera = db.get(Camera, session.camera_id)
            name = employee.name if employee else f"#{session.employee_id}"
            cam = camera.name if camera else f"#{session.camera_id}"
            started = session.started_at.replace(tzinfo=timezone.utc).astimezone()
            self.notifier.send_text(
                f"✅ {name} — камера «{cam}»\n"
                f"Присутствие началось в {started.strftime('%H:%M:%S')}"
            )
        except Exception:
            logger.exception("Presence: не удалось отправить уведомление о старте")

    def tick(self) -> None:
        """Закрыть сессии, по которым давно не было наблюдений."""
        now_mono = time.monotonic()
        with self._lock:
            if now_mono - self._last_tick < TICK_INTERVAL:
                return
            self._last_tick = now_mono

        cutoff = time_utils.utcnow() - timedelta(seconds=self.end_timeout)
        try:
            with self._session_factory() as db:
                stale = db.query(PresenceSession).filter(
                    PresenceSession.ended_at.is_(None),
                    PresenceSession.last_seen_at < cutoff,
                ).all()
                for session in stale:
                    session.ended_at = session.last_seen_at
                    logger.info(
                        "Presence ended employee=%d camera=%d",
                        session.employee_id, session.camera_id,
                    )
                if stale:
                    db.commit()
                    for session in stale:
                        self._notify_end(db, session)
        except Exception:
            logger.exception("Presence: ошибка закрытия сессий")

    def _notify_end(self, db, session: PresenceSession) -> None:
        """Уведомление в Telegram о завершении сессии."""
        if self.notifier is None:
            return
        try:
            employee = db.get(Employee, session.employee_id)
            camera = db.get(Camera, session.camera_id)
            name = employee.name if employee else f"#{session.employee_id}"
            cam = camera.name if camera else f"#{session.camera_id}"
            start = session.started_at.replace(tzinfo=timezone.utc).astimezone()
            end = session.ended_at.replace(tzinfo=timezone.utc).astimezone()
            minutes = max(1, round((session.ended_at - session.started_at).total_seconds() / 60))
            self.notifier.send_text(
                f"🏁 {name} — камера «{cam}»\n"
                f"Сессия закрыта: {start.strftime('%H:%M')}–{end.strftime('%H:%M')} "
                f"({minutes} мин)"
            )
        except Exception:
            logger.exception("Presence: не удалось отправить уведомление о закрытии")
