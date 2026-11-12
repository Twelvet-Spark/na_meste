(() => {
  'use strict';
  const { api, setBusy, message, dialog } = window.OperatorUI;
  const $ = (id) => document.getElementById(id);
  const form = $('settings-form');
  const definitions = [
    ['streams', 'preview_fps', 'Частота превью, кадров/с'],
    ['streams', 'rtsp_connect_timeout', 'Таймаут подключения RTSP, с'],
    ['streams', 'rtsp_reconnect_max_delay', 'Максимальная задержка переподключения, с'],
    ['streams', 'frame_stall_timeout', 'Ожидание кадров до переподключения, с'],
    ['ai', 'ai_enabled', 'Включить автоматическое распознавание'],
    ['ai', 'ai_device', 'Вычислительное устройство AI'],
    ['ai', 'ai_fps', 'Частота AI на камеру, кадров/с'],
    ['ai', 'ai_cameras', 'Номера камер для AI'],
    ['attendance', 'attendance_ai_fps', 'Целевая частота учёта, кадров/с'],
    ['attendance', 'attendance_frame_max_age', 'Максимальный возраст кадра, с'],
    ['attendance', 'attendance_max_frame_gap', 'Максимальный разрыв кадров, с'],
    ['attendance', 'attendance_face_threshold', 'Порог сходства лица'],
    ['attendance', 'attendance_face_margin', 'Отрыв от другого кандидата'],
    ['attendance', 'attendance_face_confirmations', 'Число подтверждений лица'],
    ['attendance', 'attendance_face_window', 'Окно подтверждения лица, с'],
    ['attendance', 'attendance_face_interval', 'Интервал проверки лица, с'],
    ['attendance', 'attendance_face_min_size', 'Минимальный размер лица, пикселей'],
    ['attendance', 'attendance_face_det_score', 'Порог уверенности детектора лица'],
    ['attendance', 'attendance_face_blur_min', 'Минимальная резкость лица'],
    ['attendance', 'attendance_evidence_keep_days', 'Хранение снимков-доказательств, дней'],
    ['notifications', 'telegram_enabled', 'Включить уведомления Telegram'],
    ['notifications', 'telegram_bot_token', 'Новый токен бота Telegram'],
    ['notifications', 'telegram_chat_id', 'Идентификатор чата Telegram'],
    ['notifications', 'telegram_notify_presence', 'Уведомлять о приходе и уходе сотрудника'],
    ['notifications', 'telegram_notify_global_transition', 'Уведомлять о переходе человека между камерами'],
    ['notifications', 'telegram_notify_global_new', 'Уведомлять о новом человеке в архиве наблюдений'],
    ['video', 'recording_retention_days', 'Срок хранения локального видео, дней'],
    ['video', 'recording_max_bytes', 'Общий лимит видео и кеша, ГБ (1 ГБ = 10⁹ байт)'],
    ['diagnostics', 'mc_debug', 'Подробные сообщения межкамерного трекинга'],
  ];
  const labels = Object.fromEntries(definitions.map(([, key, label]) => [key, label]));
  const sources = { environment: 'Окружение — изменение только администратором', operator: 'Операторский файл', dotenv: 'Настройка развёртывания (.env)', default: 'По умолчанию', file_secret: 'Секрет развёртывания' };
  const errors = {
    OPERATOR_SETTINGS_CONFLICT: 'Версия настроек изменилась. Сверьте черновик с сервером.',
    OPERATOR_SETTINGS_EXTERNAL_CHANGE: 'Файл настроек изменён вне интерфейса. Сверьте версию или обратитесь к администратору.',
    OPERATOR_SETTINGS_ENVIRONMENT_LOCKED: 'Поле задано окружением и недоступно для изменения.',
    OPERATOR_SETTINGS_INVALID_VALUES: 'Проверьте типы и допустимые границы значений.',
    OPERATOR_SETTINGS_INVALID_EFFECTIVE_CONFIG: 'Проверьте настройки: интервал проверки лица не должен превышать окно подтверждения; все числа должны быть конечными.',
    OPERATOR_SETTINGS_AI_CAMERAS_NOT_FOUND: 'Укажите только существующие номера камер.',
    OPERATOR_SETTINGS_WRITE_FAILED: 'Не удалось записать настройки. Предыдущий файл сохранён. Проверьте доступ к диску и повторите.',
    OPERATOR_SETTINGS_READ_FAILED: 'Файл настроек недоступен или повреждён. Обратитесь к администратору; черновик остаётся в этой вкладке.',
  };
  let baseline = null, latest = null, reviewed = null, busy = false, loading = false;
  let generation = 0, clearToken = false, lastUpdated = null;
  const touched = new Set(), resets = new Set(), controls = new Map();
  const dirty = () => touched.size > 0 || resets.size > 0 || clearToken;
  const errorText = (error) => errors[error.detail] || error.message || 'Не удалось получить данные. Повторите запрос.';
  const display = (key, value) => key === 'telegram_bot_token' ? 'Секрет изменён (значение скрыто)' :
    typeof value === 'boolean' ? (value ? 'Включено' : 'Выключено') :
    key === 'recording_max_bytes' ? `${Number(value) / 1e9} ГБ` : String(value === '' ? 'Не задано' : value);

  function buildFields(view) {
    for (const [group, key, labelText] of definitions) {
      const info = view.fields[key], c = info.constraints;
      const wrap = document.createElement('div'); wrap.className = 'settings-field';
      const label = document.createElement('label');
      const input = document.createElement(c.enum ? 'select' : 'input');
      input.id = `setting-${key}`; input.name = key;
      label.htmlFor = input.id;
      if (c.type === 'boolean') {
        label.className = 'setup-check'; input.type = 'checkbox';
        const text = document.createElement('span'); text.textContent = labelText;
        label.append(input, text);
      } else {
        label.className = 'field'; label.append(document.createTextNode(labelText));
        if (c.enum) {
          for (const value of c.enum) {
            const option = document.createElement('option'); option.value = value;
            option.textContent = { auto: 'Автоматически', cpu: 'CPU', cuda: 'CUDA' }[value] || value;
            input.append(option);
          }
        } else if (key === 'telegram_bot_token') {
          input.type = 'password'; input.autocomplete = 'new-password';
          input.placeholder = 'Оставить текущий токен';
        } else if (c.type === 'number' || c.type === 'integer') {
          input.type = 'number'; input.required = true;
          input.step = c.type === 'integer' ? '1' : 'any';
          const scale = key === 'recording_max_bytes' ? 1e9 : 1;
          if (key === 'recording_max_bytes') input.step = 'any';
          if (c.minimum !== undefined) input.min = String(c.minimum / scale);
          if (c.maximum !== undefined) input.max = String(c.maximum / scale);
          if (c.exclusiveMinimum !== undefined) input.min = String(c.exclusiveMinimum / scale);
        } else { input.type = 'text'; }
        label.append(input);
      }
      const note = document.createElement('p'); note.className = 'muted small'; note.id = `${input.id}-source`;
      input.setAttribute('aria-describedby', note.id);
      const resetLabel = document.createElement('label'); resetLabel.className = 'setup-check settings-reset';
      const reset = document.createElement('input'); reset.type = 'checkbox';
      const resetText = document.createElement('span'); resetText.textContent = 'Сбросить переопределение';
      resetLabel.append(reset, resetText);
      wrap.append(label, note, resetLabel); $(`settings-${group}`).append(wrap);
      controls.set(key, { input, note, reset });
      input.addEventListener('input', () => {
        if (key === 'telegram_bot_token') {
          clearToken = false;
          if (input.value) touched.add(key); else touched.delete(key);
        } else touched.add(key);
        message($('settings-message'), 'Черновик ещё не сохранён. Текущие настройки не изменены.');
      });
      reset.addEventListener('change', () => {
        if (reset.checked) resets.add(key); else resets.delete(key);
        syncDisabled();
        message($('settings-message'), 'Сброс будет применён только после сохранения и обычного перезапуска.');
      });
    }
  }
  function syncDisabled() {
    if (!baseline) return;
    $('settings-controls').disabled = busy;
    for (const [key, row] of controls) {
      const info = baseline.fields[key];
      row.input.disabled = busy || info.locked || resets.has(key);
      row.reset.disabled = busy || info.locked || (!info.overridden && !resets.has(key));
    }
    $('settings-clear-token').disabled = busy || baseline.fields.telegram_bot_token.locked || resets.has('telegram_bot_token');
    $('settings-save').disabled = busy;
  }
  function fill(view, preserve = false) {
    if (!controls.size) buildFields(view);
    for (const [key, row] of controls) {
      if (!preserve || (!touched.has(key) && !resets.has(key) && !(key === 'telegram_bot_token' && clearToken))) {
        if (key === 'telegram_bot_token') row.input.value = '';
        else if (row.input.type === 'checkbox') row.input.checked = view.pending[key];
        else row.input.value = key === 'recording_max_bytes' ? view.pending[key] / 1e9 : view.pending[key];
      }
      row.note.textContent = sources[view.fields[key].source] || 'Источник недоступен';
      row.reset.checked = resets.has(key);
    }
    baseline = view; latest = view; syncDisabled();
    $('settings-token-state').textContent = `Сейчас: ${view.active.has_telegram_bot_token ? 'токен задан' : 'токен не задан'}. После перезапуска: ${view.pending.has_telegram_bot_token ? 'токен задан' : 'токен не задан'}.`;
    const offset = view.read_only.attendance_utc_offset_minutes;
    $('settings-offset').textContent = `Общее смещение рабочего дня: UTC ${offset >= 0 ? '+' : '−'}${Math.abs(offset) / 60} ч. Только чтение: изменение через интерфейс запрещено для сохранения исторических дней.`;
    $('settings-reserve').textContent = `Каталог записей и резерв свободного места (${view.read_only.recording_min_free_bytes / 1e9} ГБ) управляются только настройками развёртывания.`;
  }
  function serverDiff(view) {
    $('settings-restart').hidden = !view.restart_required;
    $('settings-diff-panel').hidden = !view.restart_required;
    const list = $('settings-diff'); list.replaceChildren();
    for (const key of view.changed_keys) {
      const li = document.createElement('li');
      li.textContent = key === 'telegram_bot_token' ? `${labels[key]}: секрет изменён; значение не показывается.` :
        `${labels[key]}: сейчас ${display(key, view.active[key])}; после перезапуска ${display(key, view.pending[key])}.`;
      list.append(li);
    }
  }
  function connectivity(offline = false) {
    $('settings-connection').textContent = offline ? `Нет связи с сервером. ${lastUpdated ? `Последнее обновление: ${lastUpdated.toLocaleTimeString('ru-RU')}. Данные устарели.` : 'Настройки ещё не загружены.'}` :
      `Обновлено ${lastUpdated.toLocaleTimeString('ru-RU')}. Проверка каждые 5 секунд; черновик не заменяется.`;
  }
  async function refresh() {
    if (busy || loading) return;
    loading = true; const request = ++generation;
    try {
      const view = await api('/api/settings');
      if (request !== generation) return;
      latest = view; lastUpdated = new Date(); connectivity(); serverDiff(view);
      if (!baseline || (!dirty() && !form.contains(document.activeElement))) fill(view);
      else if (baseline.version !== view.version) $('settings-conflict').hidden = false;
      $('settings-retry').hidden = true;
    } catch (error) {
      if (request !== generation) return;
      connectivity(true); $('settings-retry').hidden = false;
      message($('settings-message'), errorText(error), 'error');
    } finally { loading = false; }
  }
  function values() {
    const result = {};
    for (const key of touched) {
      if (resets.has(key)) continue;
      const input = controls.get(key).input;
      if (key === 'telegram_bot_token' && !input.value) continue;
      result[key] = input.type === 'checkbox' ? input.checked : input.type === 'number' ? Number(input.value) : input.value;
      if (key === 'recording_max_bytes') result[key] = Math.round(result[key] * 1e9);
    }
    if (clearToken && !resets.has('telegram_bot_token')) result.telegram_bot_token = '';
    return result;
  }
  form.addEventListener('submit', async (event) => {
    event.preventDefault(); if (busy || !baseline) return;
    if (!dirty()) { message($('settings-message'), 'Нет изменений для сохранения.'); return; }
    busy = true; ++generation; setBusy(form, true);
    try {
      const view = await api('/api/settings', { method: 'PUT', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ expected_version: baseline.version, values: values(), reset_keys: [...resets] }) });
      touched.clear(); resets.clear(); clearToken = false; reviewed = null;
      fill(view); serverDiff(view); $('settings-conflict').hidden = true; $('settings-review-panel').hidden = true;
      lastUpdated = new Date(); connectivity();
      message($('settings-message'), view.restart_required ? 'Сохранено. Требуется перезапуск.' : 'Сохранено. Настройки соответствуют текущему запуску.');
    } catch (error) {
      if (error.status === 409) $('settings-conflict').hidden = false;
      message($('settings-message'), errorText(error), 'error');
    } finally { busy = false; setBusy(form, false); syncDisabled(); }
  });
  $('settings-review').addEventListener('click', async () => {
    if (busy || !baseline) return;
    const original = baseline;
    try {
      const view = await api('/api/settings');
      if (busy || baseline !== original) return;
      reviewed = view; latest = reviewed;
      const list = $('settings-review-list'); list.replaceChildren();
      const keys = new Set([...Object.keys(labels).filter((key) => key !== 'telegram_bot_token' && baseline.pending[key] !== reviewed.pending[key]), ...touched, ...resets]);
      if (baseline.version !== reviewed.version && (baseline.pending.has_telegram_bot_token || reviewed.pending.has_telegram_bot_token)) keys.add('telegram_bot_token');
      const draft = values();
      for (const key of keys) {
        const li = document.createElement('li');
        li.textContent = key === 'telegram_bot_token' ? `${labels[key]}: значение скрыто; ${resets.has(key) ? 'вы выбрали сброс переопределения' : Object.hasOwn(draft, key) ? 'ваш секретный черновик сохранён' : 'токен не будет перезаписан вашим черновиком'}.` :
          `${labels[key]}: на сервере ${display(key, reviewed.pending[key])}; ${resets.has(key) ? 'вы выбрали сброс переопределения' : Object.hasOwn(draft, key) ? `ваш черновик ${display(key, draft[key])}` : 'принять серверное значение'}.`;
        list.append(li);
      }
      if (!keys.size) { const li = document.createElement('li'); li.textContent = 'Значения совпадают; изменилась версия файла.'; list.append(li); }
      $('settings-review-panel').hidden = false;
    } catch (error) { message($('settings-message'), errorText(error), 'error'); }
  });
  $('settings-rebase').addEventListener('click', () => {
    if (!reviewed || busy) return;
    fill(reviewed, true); serverDiff(reviewed); reviewed = null;
    $('settings-conflict').hidden = true; $('settings-review-panel').hidden = true;
    message($('settings-message'), 'Черновик перенесён на просмотренную версию. Проверьте поля и сохраните; ничего ещё не записано.');
  });
  $('settings-cancel').addEventListener('click', () => {
    if (busy || !latest) return;
    touched.clear(); resets.clear(); clearToken = false; reviewed = null;
    fill(latest); serverDiff(latest); $('settings-conflict').hidden = true; $('settings-review-panel').hidden = true;
    message($('settings-message'), 'Черновик отменён. Показаны сохранённые настройки следующего запуска.');
  });
  const clearDialog = dialog($('token-clear-dialog'), $('settings-clear-token'));
  $('settings-clear-token').addEventListener('click', () => clearDialog.open());
  $('token-clear-confirm').addEventListener('click', () => {
    clearToken = true; touched.delete('telegram_bot_token'); controls.get('telegram_bot_token').input.value = '';
    clearDialog.close(); message($('settings-message'), 'Очистка токена добавлена в черновик. Нажмите «Сохранить для следующего запуска».');
  });
  $('settings-retry').addEventListener('click', refresh);

  let workplaceId = null, workplaceBusy = false;
  const workplaceDialog = dialog($('workplace-dialog'), $('workplace-add'), { isBusy: () => workplaceBusy });
  function editWorkplace(row, trigger) {
    workplaceId = row?.id || null; $('workplace-name').value = row?.name || '';
    $('workplace-active').checked = row?.active ?? true;
    $('workplace-active').disabled = !row;
    $('workplace-dialog-heading').textContent = row ? 'Изменить рабочее место' : 'Добавить рабочее место';
    message($('workplace-dialog-status'), ''); workplaceDialog.open(trigger);
  }
  async function workplaces() {
    try {
      const rows = await api('/api/workplaces');
      const list = $('settings-workplaces'); list.replaceChildren();
      for (const row of rows) {
        const li = document.createElement('li'), text = document.createElement('span'), button = document.createElement('button');
        text.textContent = `${row.name} — ${row.active ? 'Активное' : 'В архиве'}`;
        button.type = 'button'; button.className = 'btn'; button.textContent = 'Переименовать / архивировать';
        button.addEventListener('click', () => editWorkplace(row, button)); li.append(text, button); list.append(li);
      }
      message($('workplace-status'), rows.length ? `${rows.length} рабочих мест` : 'Рабочих мест пока нет. Добавьте магазин или офис при необходимости.');
      $('workplace-retry').hidden = true;
    } catch (error) { message($('workplace-status'), errorText(error), 'error'); $('workplace-retry').hidden = false; }
  }
  $('workplace-add').addEventListener('click', (event) => editWorkplace(null, event.currentTarget));
  $('workplace-retry').addEventListener('click', workplaces);
  $('workplace-form').addEventListener('submit', async (event) => {
    event.preventDefault(); if (workplaceBusy) return;
    workplaceBusy = true; setBusy($('workplace-form'), true);
    try {
      const payload = { name: $('workplace-name').value };
      if (workplaceId) payload.active = $('workplace-active').checked;
      await api(`/api/workplaces${workplaceId ? `/${workplaceId}` : ''}`, { method: workplaceId ? 'PUT' : 'POST',
        headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
      workplaceBusy = false; workplaceDialog.close(); await workplaces();
    } catch (error) { message($('workplace-dialog-status'), errorText(error), 'error'); }
    finally { workplaceBusy = false; setBusy($('workplace-form'), false); }
  });
  document.addEventListener('visibilitychange', () => { if (!document.hidden) refresh(); });
  setInterval(() => { if (!document.hidden) refresh(); }, 5000);
  refresh(); workplaces();
})();
