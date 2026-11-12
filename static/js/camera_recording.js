// Explicit manual recording; polling never replaces duration/confirmation drafts.
(() => {
  'use strict';
  const ui = window.OperatorUI;
  const byId = (id) => document.getElementById(id);
  const form = byId('recording-form');
  const startButton = byId('recording-start');
  const stopButton = byId('recording-stop');
  const retry = byId('recording-retry');
  const duration = byId('recording-duration');
  const message = byId('recording-message');
  const activeStates = new Set(['STARTING', 'RECORDING', 'RECONNECTING', 'STOPPING']);
  const labels = { STARTING: 'Запуск записи', RECORDING: 'Идёт запись', RECONNECTING: 'Переподключение — в записи будет разрыв', STOPPING: 'Завершение последнего фрагмента', STOPPED: 'Остановлена оператором', COMPLETED: 'Длительность истекла', FAILED: 'Запись не выполнена', INTERRUPTED: 'Запись прервана' };
  const reasons = {
    STORAGE_FULL: 'Недостаточно свободного места. Освободите место или дождитесь очистки; запись сама не возобновится.',
    MEDIA_ROOT_UNAVAILABLE: 'Хранилище занято другим процессом или недоступно. Проверьте один рабочий процесс сервера и права на каталог.',
    MEDIA_STORAGE_UNAVAILABLE: 'Хранилище недоступно. Проверьте диск и права сервера.',
    MEDIA_PROCESS_CONTAINMENT_UNAVAILABLE: 'Безопасный запуск видео недоступен. Проверьте Windows Job Object и перезапустите сервер.',
    FFMPEG_UNAVAILABLE: 'FFmpeg или ffprobe не найден. Укажите установленный FFmpeg и перезапустите сервер.',
    MEDIA_LAUNCH_FAILED: 'FFmpeg не удалось запустить. Проверьте установку и права сервера.',
    SOURCE_CHANGED: 'Подключение камеры изменилось. При необходимости начните новую запись.',
    CAMERA_DISABLED: 'Камера отключена. Существующую запись можно остановить независимо от живого видео.',
    RECORDING_ALREADY_ACTIVE: 'Для камеры уже выполняется запись. Обновите состояние.',
    CLOCK_DISCONTINUITY: 'Часы сервера изменились. Запись прервана, чтобы не присваивать кадрам неверное время.',
    APPLICATION_RESTART: 'Сервер был перезапущен. Автоматического возобновления нет.',
    APPLICATION_SHUTDOWN: 'Сервер завершил работу.',
    MEDIA_PROCESS_FAILED: 'Процесс записи завершился с ошибкой. Проверьте подключение и повторите запрос.',
    MEDIA_FINALIZATION_FAILED: 'Последний фрагмент не удалось завершить. Проверьте диск и состояние сервера.',
    MEDIA_STOP_PENDING: 'Завершение записи ещё выполняется. Обновите состояние.',
  };
  let session = null;
  let cameraEnabled = false;
  let available = false;
  let busy = false;
  let sequence = 0;
  let pendingRequest = null;
  const formatTime = (value) => value ? new Date(value).toLocaleString('ru-RU') : '—';
  function explain(error) {
    return reasons[error.detail] || (error.status === 409 ? 'Состояние изменилось. Обновите его; введённая длительность сохранена.' : 'Не удалось подтвердить операцию. Проверьте сеть и повторите запрос.');
  }
  function render() {
    const active = session && activeStates.has(session.status);
    startButton.disabled = busy || !available || !cameraEnabled || active;
    stopButton.disabled = busy || !active || session.status === 'STOPPING';
    byId('recording-state').textContent = session ? `${labels[session.status] || session.status}${session.reason_code ? '. ' + (reasons[session.reason_code] || 'Причина: ' + session.reason_code) : ''}` : 'Запись не ведётся';
    byId('recording-started').textContent = formatTime(session?.started_at);
    byId('recording-deadline').textContent = formatTime(session?.deadline_at);
    byId('recording-bytes').textContent = session ? `${(session.bytes_recorded / 1000000).toLocaleString('ru-RU', { maximumFractionDigits: 1 })} МБ` : '—';
    elapsed();
  }
  function elapsed() {
    const from = session?.started_at;
    const seconds = from ? Math.max(0, Math.floor(((session.ended_at ? Date.parse(session.ended_at) : Date.now()) - Date.parse(from)) / 1000)) : null;
    byId('recording-elapsed').textContent = seconds === null ? '—' : `${Math.floor(seconds / 3600)} ч ${Math.floor(seconds % 3600 / 60)} мин ${seconds % 60} с`;
  }
  async function refresh() {
    if (busy) return;
    const ticket = ++sequence;
    try {
      const [sessions, status] = await Promise.all([
        ui.api(`/api/recordings/sessions?camera_id=${window.CAMERA_ID}&limit=1`),
        ui.api('/api/recordings/status'),
      ]);
      if (ticket !== sequence || busy) return;
      session = sessions[0] || null;
      available = true;
      const storage = status.storage;
      byId('recording-oldest').textContent = formatTime(storage.oldest_available);
      byId('recording-policy').textContent = `Хранение: ${storage.retention_days} дней; общий лимит ${(storage.max_bytes / 1000000000).toLocaleString('ru-RU')} ГБ. Использовано ${(storage.bytes_used / 1000000000).toLocaleString('ru-RU', { maximumFractionDigits: 2 })} ГБ; свободно на диске ${(storage.free_bytes / 1000000000).toLocaleString('ru-RU', { maximumFractionDigits: 1 })} ГБ. При достижении лимита старые незанятые записи удаляются раньше срока.${storage.quota_shortened_retention ? ' Срок хранения уже сокращён из-за лимита.' : ''}`;
      retry.hidden = true;
      render();
    } catch (error) {
      if (ticket !== sequence) return;
      available = false;
      render();
      byId('recording-state').textContent = session
        ? byId('recording-state').textContent + ' · Состояние не обновлено' : 'Состояние записи недоступно';
      ui.message(message, explain(error), 'error');
      retry.hidden = false;
    }
  }
  form.addEventListener('submit', async (event) => {
    event.preventDefault();
    if (busy || !form.reportValidity()) return;
    const seconds = Number(duration.value) * 60;
    if (!Number.isInteger(seconds) || seconds < 60 || seconds > 86400) return;
    if (!pendingRequest || pendingRequest.duration_seconds !== seconds) pendingRequest = {
      camera_id: window.CAMERA_ID, request_id: ui.newRequestId(), duration_seconds: seconds,
    };
    busy = true; ++sequence; ui.setBusy(form, true); render();
    try {
      session = await ui.api('/api/recordings/sessions', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(pendingRequest) });
      pendingRequest = null;
      ui.message(message, activeStates.has(session.status)
        ? 'Запрос принят. Запись ограничена указанным сроком; при разрывах соединения будут сохранены интервалы отсутствия видео.'
        : 'Этот запрос уже завершён. Его состояние показано выше; для новой записи нажмите «Начать запись» ещё раз.');
    } catch (error) {
      ui.message(message, explain(error), 'error'); retry.hidden = false;
      // Preserve UUID on an unknown transport outcome: retry cannot launch twice.
    } finally {
      busy = false; ui.setBusy(form, false); render();
    }
  });
  stopButton.addEventListener('click', async () => {
    if (busy || !session) return;
    busy = true; ++sequence; ui.setBusy(form, true); render();
    try {
      session = await ui.api(`/api/recordings/sessions/${session.id}/stop`, { method: 'POST' });
      pendingRequest = null;
      ui.message(message, 'Запись остановлена. Проверенные завершённые фрагменты сохранены.');
    } catch (error) { ui.message(message, explain(error), 'error'); retry.hidden = false; }
    finally { busy = false; ui.setBusy(form, false); render(); }
  });
  window.addEventListener('camera-state', (event) => { cameraEnabled = event.detail.enabled; render(); });
  retry.addEventListener('click', refresh);
  document.addEventListener('visibilitychange', () => { if (!document.hidden) refresh(); });
  const poll = setInterval(() => { if (!document.hidden) refresh(); }, 5000);
  const timer = setInterval(() => { if (!document.hidden) elapsed(); }, 1000);
  window.addEventListener('pagehide', () => { ++sequence; clearInterval(poll); clearInterval(timer); });
  // Leaving this page does not cancel a deliberately requested recording.
  refresh();
})();
