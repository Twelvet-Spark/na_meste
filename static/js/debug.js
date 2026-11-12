(() => {
  'use strict';
  const { api, message } = window.OperatorUI;
  const $ = (id) => document.getElementById(id);
  const reasons = {
    AI_UNAVAILABLE: 'Модели AI не готовы. Проверьте загруженные провайдеры и настройки развёртывания.',
    AI_EXCLUDED: 'Камера исключена из обработки AI. Проверьте список камер в настройках.',
    CAMERA_UNCONFIGURED: 'Не настроены обе роли входа и выхода. Откройте калибровку камер.',
    CAMERA_OFFLINE: 'Камера недоступна. Проверьте питание, сеть и подключение.',
    COMMISSIONING_REQUIRED: 'Нужен ввод системы в эксплуатацию. Проверьте калибровку и критерии на странице рабочего времени.',
    CLOCK_DISCONTINUITY: 'Обнаружено изменение часов. Требуется проверка временных разрывов; история не сбрасывается.',
    FRAME_GAP: 'Недостаточно свежих кадров. Проверьте поток камеры.',
    PERFORMANCE_WARMUP: 'Измеряется производительность. Дождитесь устойчивой обработки кадров.',
    PERFORMANCE_DEGRADED: 'Производительность ниже требований. Проверьте загрузку и провайдеры AI.',
    PERSISTENCE_ERROR: 'Есть ошибка сохранения учёта. Проверьте доступ к базе и свободное место.',
    DIAGNOSTICS_UNAVAILABLE: 'Состояние модуля временно недоступно. Остальные данные показаны независимо.',
    FFMPEG_UNAVAILABLE: 'FFmpeg / ffprobe недоступны. Администратору нужно проверить установку и путь.',
    LIBX264_UNAVAILABLE: 'В FFmpeg отсутствует libx264. Нужна сборка с этим видеокодером.',
    MEDIA_PROCESS_CONTAINMENT_UNAVAILABLE: 'Защита времени жизни медиапроцессов недоступна. Запуск видео заблокирован; проверьте Windows Job Object и один рабочий процесс сервера.',
    MEDIA_ROOT_UNAVAILABLE: 'Каталог видео недоступен или занят другим процессом. Проверьте единственный экземпляр приложения и права доступа.',
    MEDIA_STORAGE_UNAVAILABLE: 'Хранилище видео недоступно. Проверьте права доступа и свободное место.',
    PLAYBACK_STORAGE_UNAVAILABLE: 'Хранилище просмотра недоступно. Проверьте каталог видео.',
    STORAGE_FULL: 'Недостаточно места или исчерпана квота. Остановите ненужный просмотр и проверьте диск; используемые файлы не удаляются.',
    MEDIA_STOP_PENDING: 'Ожидается завершение медиапроцесса. Не запускайте второй экземпляр приложения.',
    MEDIA_PROCESS_FAILED: 'Медиапроцесс завершился с ошибкой. Проверьте инструменты, поток и место на диске.',
    STREAM_UNAVAILABLE: 'Поток недоступен. Откройте камеру и проверьте подключение.',
    SOURCE_CHANGED: 'Источник изменён. Старый запрос отменён; повторите действие с актуальными настройками.',
    CAMERA_DISABLED: 'Камера выключена. Историческое локальное видео остаётся доступным.',
    DEVICE_AUTH_FAILED: 'Регистратор отклонил авторизацию. Проверьте пароль и права пользователя CGI/архива.',
    DEVICE_UNREACHABLE: 'Регистратор недоступен. Проверьте адрес, порт и сеть.',
    DEVICE_TIMEOUT: 'Регистратор не ответил вовремя. Проверьте сеть и повторите поиск.',
    DEVICE_TLS_FAILED: 'Не удалось проверить TLS-сертификат. Исправьте сертификат устройства; проверка не отключается.',
    DEVICE_PROTOCOL_ERROR: 'Регистратор вернул неподдерживаемый ответ. Уточните модель, прошивку и профиль API.',
    ARCHIVE_UNSUPPORTED: 'Архивный профиль не поддерживается устройством. Проверьте прошивку и права доступа; live и локальные записи независимы.',
    ARCHIVE_NOT_CONFIGURED: 'Доступ к архиву не настроен. Откройте настройки выбранной камеры.',
    SEARCH_LIMIT: 'Ответ архива превысил безопасный лимит. Уменьшите период поиска.',
    ARCHIVE_SEARCH_OK: 'Поиск архива выполнен успешно.', ARCHIVE_DISCOVERY_OK: 'Проверка каналов регистратора выполнена успешно.',
    OPERATOR_SETTINGS_SAVED: 'Настройки следующего запуска сохранены.',
    UNEXPECTED_ERROR: 'Произошла ошибка. Подробности исключены из отчёта для защиты секретов; проверьте состояние модуля.',
    RUNTIME_WARNING: 'Предупреждение текущего запуска. Проверьте состояние соответствующего модуля.',
    OPERATOR_EVENT: 'Событие текущего запуска.',
  };
  const stateNames = { ONLINE: 'В сети', OFFLINE: 'Не в сети', CONNECTING: 'Подключение', RECONNECTING: 'Переподключение', ERROR: 'Ошибка',
    STARTING: 'Запускается', RECORDING: 'Записывается', STOPPING: 'Останавливается', STOPPED: 'Остановлено', COMPLETED: 'Завершено', FAILED: 'Ошибка', INTERRUPTED: 'Прервано',
    PREPARING: 'Подготовка', READY: 'Готово', ENDED: 'Завершено', CANCELLED: 'Отменено', EXPIRED: 'Истекло' };
  const providerNames = { yolo: 'Детектор людей', live_face_detection: 'Детектор лиц в потоке', live_face_recognition: 'Распознавание лиц в потоке',
    enroll_face_detection: 'Детектор лиц при добавлении фото', enroll_face_recognition: 'Распознавание при добавлении фото', reid: 'Межкамерное сопоставление' };
  let loading = false, sequence = 0, lastUpdate = null, generation = 0;
  const events = [];
  const text = (value) => value === null || value === undefined ? 'Нет данных' : String(value);
  const time = (value) => value ? new Date(value).toLocaleString('ru-RU') : 'Нет данных';
  const gb = (value) => value === null || value === undefined ? 'Нет данных' : `${(value / 1e9).toLocaleString('ru-RU', { maximumFractionDigits: 3 })} ГБ`;
  const why = (code) => reasons[code] || `Код ${code}. Проверьте состояние модуля и передайте отчёт администратору.`;
  function facts(target, entries) {
    target.replaceChildren();
    for (const [label, value] of entries) {
      const dt = document.createElement('dt'), dd = document.createElement('dd');
      dt.textContent = label; dd.textContent = text(value); target.append(dt, dd);
    }
  }
  function row(target, heading, description, links = []) {
    const article = document.createElement('article'), title = document.createElement('h3'), p = document.createElement('p');
    title.textContent = heading; p.textContent = description; article.append(title, p);
    if (links.length) {
      const actions = document.createElement('div'); actions.className = 'form-actions';
      for (const [label, url] of links) { const a = document.createElement('a'); a.className = 'btn'; a.textContent = label; a.href = url; actions.append(a); }
      article.append(actions);
    }
    target.append(article);
  }
  function render(view) {
    const attendance = view.attendance;
    $('debug-readiness').textContent = attendance.automatic_ready ? 'Автоматический учёт готов по текущему состоянию системы.' : 'Автоматический учёт не готов. Не считайте отсутствие автоматических отметок подтверждением отсутствия сотрудника.';
    const reasonList = $('debug-reasons'); reasonList.replaceChildren();
    for (const code of attendance.reasons || []) { const li = document.createElement('li'); li.textContent = why(code); reasonList.append(li); }
    facts($('debug-attendance'), [['Техническая готовность', attendance.technical_ready ? 'Готово' : 'Не готово'],
      ['Ожидающие сообщения о разрывах', attendance.pending_fault_count], ['Ожидающие записи', attendance.pending_write_count]]);
    const roles = $('debug-roles'); roles.replaceChildren();
    for (const [name, role] of Object.entries(attendance.roles || {})) {
      row(roles, `${name === 'ENTRY' ? 'Вход' : 'Выход'} — ${role.camera_id == null ? 'камера не настроена' : `камера №${role.camera_id}`}`,
        `Обработка: ${text(role.processed_fps)} кадров/с. Возраст кадра: ${text(role.frame_age_seconds)} с. Задержка p95: ${text(role.processing_age_p95_seconds)} с; завершение p95: ${text(role.completion_age_p95_seconds)} с.`);
    }
    facts($('debug-tools'), [['FFmpeg', view.tools.error_code ? 'Нет данных' : view.tools.ffmpeg_available ? 'Найден' : 'Не найден'], ['ffprobe', view.tools.error_code ? 'Нет данных' : view.tools.ffprobe_available ? 'Найден' : 'Не найден'],
      ['AI в настройках текущего запуска', view.ai.enabled ? 'Включён' : 'Отключён'],
      ['Защита медиапроцессов', view.processes.containment_available ? 'Доступна' : 'Недоступна / нет данных']]);
    const providers = $('debug-providers'); providers.replaceChildren();
    if (view.ai.error_code) row(providers, 'Провайдеры AI', why(view.ai.error_code));
    else for (const [key, names] of Object.entries(view.ai.providers || {})) row(providers, providerNames[key] || key, names.length ? names.join(', ') : 'Не загружен');
    const cameras = $('debug-cameras'); cameras.replaceChildren();
    $('debug-cameras-message').textContent = view.camera_error ? why(view.camera_error) : view.cameras.length ? `${view.cameras.length} камер` : 'Камер пока нет. Добавьте камеру для проверки потоков.';
    for (const camera of view.cameras) {
      row(cameras, `Камера №${camera.id} — ${stateNames[camera.status] || camera.status}${camera.enabled ? '' : ' (выключена)'}`,
        `Кадров/с: ${text(camera.current_fps)}. Возраст кадра: ${text(camera.seconds_since_last_frame)} с. Переподключений: ${text(camera.reconnect_count)}. ${camera.resolution || ''} ${camera.codec || ''}. ${camera.error ? why(camera.error) : ''}`,
        [['Подключение и тест', `/cameras/${camera.id}`], ['Калибровка', `/cameras/${camera.id}#calibration-heading`]]);
    }
    const media = view.recordings, storage = media.storage;
    $('debug-media-message').textContent = media.error_code ? why(media.error_code) : 'Хранилище доступно. Запись запускается только вручную.';
    facts($('debug-storage'), storage ? [['Использовано', gb(storage.bytes_used)], ['Лимит', gb(storage.max_bytes)], ['Свободно на диске', gb(storage.free_bytes)],
      ['Резерв свободного места', gb(storage.min_free_bytes)], ['Резерв завершения записей', gb(storage.reserve_bytes)], ['Политика хранения', `${storage.retention_days} дней`],
      ['Самое старое доступное видео', time(storage.oldest_available)], ['Срок сокращён квотой', storage.quota_shortened_retention ? 'Да' : 'Нет']] : [['Хранилище', 'Нет данных']]);
    const sessions = $('debug-sessions'); sessions.replaceChildren();
    for (const session of media.sessions || []) row(sessions, `Камера №${session.camera_id} — ${stateNames[session.status] || session.status}`,
      `Начало: ${time(session.started_at)}. Срок завершения: ${time(session.deadline_at)}. Объём: ${gb(session.bytes_recorded)}. ${session.reason_code ? why(session.reason_code) : ''}`);
    if (!media.sessions?.length) row(sessions, 'Активные записи', media.error_code === 'DIAGNOSTICS_UNAVAILABLE' ? 'Нет данных' : 'Нет активных сессий записи.');
    const processes = $('debug-processes'); processes.replaceChildren();
    if (view.processes.error_code) row(processes, 'Процессы', why(view.processes.error_code));
    for (const process of view.processes.recorders || []) row(processes, `Процесс записи PID ${text(process.pid)}`, process.running ? 'Работает' : 'Завершён');
    for (const process of view.processes.playbacks || []) row(processes, `Просмотр камеры №${process.camera_id} — ${stateNames[process.state] || process.state}`,
      `PID ${text(process.pid)}. ${process.running ? 'Процесс работает.' : 'Процесс не работает.'} ${process.error_code ? why(process.error_code) : ''}`);
    const archive = view.archive;
    facts($('debug-archive'), [['Проверено', time(archive.updated_at)], ['Камера', archive.camera_id ?? 'Новая камера / не выбрана'],
      ['Названия каналов CGI', archive.channel_titles === null ? 'Ещё не проверялись' : archive.channel_titles ? 'Получены' : 'Недоступны'],
      ['Поиск записей', archive.search === null ? 'Ещё не выполнялся' : archive.search ? 'Выполнен успешно' : 'Не выполнен'],
      ['Последний результат', archive.error_code ? why(archive.error_code) : archive.updated_at ? 'Успешно' : 'Проверок ещё не было']]);
  }
  function renderEvents(result) {
    let skipped = false;
    if (sequence && result.oldest_sequence > sequence + 1) skipped = true;
    for (const event of result.events) {
      if (event.sequence > sequence) { events.push(event); sequence = event.sequence; }
    }
    if (events.length > 200) events.splice(0, events.length - 200);
    const list = $('debug-events'); list.replaceChildren();
    for (const event of [...events].reverse()) {
      const li = document.createElement('li');
      li.textContent = `№${event.sequence} · ${time(event.at)} · ${event.level} · ${event.source}${event.camera_id ? ` · камера №${event.camera_id}` : ''}: ${why(event.code)}`;
      list.append(li);
    }
    $('debug-events-status').textContent = skipped ? 'Часть старых событий уже вышла из буфера. Показаны доступные записи.' : events.length ? `Показано ${events.length} событий` : 'Событий текущего запуска пока нет.';
  }
  async function refresh() {
    if (loading) return;
    loading = true; $('debug-refresh').disabled = true; const request = ++generation;
    try {
      const [status, eventResult] = await Promise.allSettled([api('/api/debug/status'), api(`/api/debug/events?after=${sequence}&limit=100`)]);
      if (request !== generation) return;
      if (status.status === 'fulfilled') {
        render(status.value); lastUpdate = new Date();
        $('debug-updated').textContent = `Обновлено ${lastUpdate.toLocaleTimeString('ru-RU')}. Проверка каждые 5 секунд, пока вкладка видима.`;
        message($('debug-error'), '');
      } else {
        $('debug-updated').textContent = lastUpdate ? `Данные устарели. Последнее обновление ${lastUpdate.toLocaleTimeString('ru-RU')}.` : 'Состояние ещё не получено.';
        message($('debug-error'), 'Нет связи с диагностикой. Последние данные оставлены на экране. Повторите обновление.', 'error');
      }
      if (eventResult.status === 'fulfilled') {
        // A normal server restart starts a new ring, not a continuation of old IDs.
        if (eventResult.value.latest_sequence < sequence) { sequence = 0; events.length = 0; }
        renderEvents(eventResult.value);
      } else message($('debug-events-status'), 'Журнал временно недоступен. Остальные данные продолжают обновляться.', 'error');
    } catch {
      message($('debug-error'), 'Не удалось отобразить состояние. Повторите обновление.', 'error');
    } finally { loading = false; $('debug-refresh').disabled = false; }
  }
  $('debug-refresh').addEventListener('click', refresh);
  document.addEventListener('visibilitychange', () => { if (!document.hidden) refresh(); });
  setInterval(() => { if (!document.hidden) refresh(); }, 5000);
  refresh();
})();
