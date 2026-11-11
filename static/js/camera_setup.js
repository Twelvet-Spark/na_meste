// Guided camera/NVR setup shared by the directory and camera profile.
(() => {
  'use strict';
  const ui = window.OperatorUI;
  const dialog = document.getElementById('camera-setup');
  const form = document.getElementById('camera-form');
  const $ = (id) => document.getElementById(id);
  const field = (name) => form.elements.namedItem(name);
  const panels = [...form.querySelectorAll('[data-setup-step]')];
  const message = $('setup-message');
  const live = $('setup-live-result');
  const discovery = $('setup-discovery-result');
  let busy = false, step = 0, editing = null, rows = [], channelDirty = false;
  let opening = 0, requestId = null, requestPayload = null;
  const controller = ui.dialog(dialog, null, { isBusy: () => busy });
  const errors = {
    DEVICE_AUTH_FAILED: 'Устройство отклонило доступ CGI. Проверьте пароль и разрешение чтения конфигурации; каналы можно ввести вручную.',
    DEVICE_TIMEOUT: 'Устройство не ответило за отведённое время. Проверьте HTTP-порт и сеть или введите каналы вручную.',
    DEVICE_UNREACHABLE: 'HTTP-подключение недоступно. Проверьте адрес, порт и сертификат HTTPS.',
    DEVICE_TLS_FAILED: 'Сертификат HTTPS не прошёл проверку. Проверьте доверие к сертификату и имя устройства. Небезопасный обход проверки не выполняется.',
    ARCHIVE_UNSUPPORTED: 'Обнаружение CGI не поддерживается. Добавьте номера каналов вручную.',
    DEVICE_PROTOCOL_ERROR: 'Устройство вернуло неподдерживаемый ответ CGI. Проверьте HTTP-порт или добавьте каналы вручную.',
    SOURCE_CHANGED: 'Подключение изменилось в другом окне. Откройте камеру заново перед проверкой.',
  };
  const json = (url, payload, method = 'POST') => ui.api(url, {
    method, headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload),
  });
  const isRecorder = () => !editing && field('device_kind').value === 'recorder';

  function setBusy(value) {
    busy = value; ui.setBusy(form, value);
    if (!value) showStep(step, false);
  }
  function archiveFields() {
    $('setup-offset-fields').disabled = !field('archive_enabled').checked;
  }
  function showStep(value, focus = true) {
    step = value;
    panels.forEach((panel, index) => { panel.hidden = index !== value; panel.disabled = index !== value; });
    dialog.querySelectorAll('[data-progress]').forEach((item, index) => {
      if (index === value) item.setAttribute('aria-current', 'step'); else item.removeAttribute('aria-current');
    });
    $('setup-back').hidden = value === (editing ? 1 : 0);
    $('setup-next').hidden = value === 2;
    $('setup-save').hidden = value !== 2;
    $('setup-save').textContent = editing ? 'Сохранить камеру' : 'Сохранить камеры';
    $('setup-recorder-tools').hidden = !isRecorder();
    archiveFields();
    if (focus) panels[value].querySelector('input:not(:disabled), button, select')?.focus();
  }
  function checkStep() {
    if (field('archive_enabled').checked && step === 2) $('setup-archive-details').open = true;
    return form.reportValidity();
  }
  function rowField(label, control) {
    const wrapper = document.createElement('label'); wrapper.className = 'field';
    const title = document.createElement('span'); title.textContent = label;
    wrapper.append(title, control); return wrapper;
  }
  function renderRows() {
    const list = $('setup-channel-list'); list.replaceChildren();
    rows.forEach((row) => {
      const node = document.createElement('fieldset'); node.className = 'setup-channel';
      const legend = document.createElement('legend'); legend.textContent = `Канал ${row.channel}`;
      const selected = document.createElement('input'); selected.type = 'checkbox'; selected.checked = row.selected;
      const choice = document.createElement('label'); choice.className = 'setup-check';
      choice.append(selected, document.createTextNode('Добавить этот канал')); choice.hidden = !isRecorder();
      selected.addEventListener('change', () => { row.selected = selected.checked; channelDirty = true; });
      const number = document.createElement('input'); number.type = 'number'; number.min = '1'; number.max = '64';
      number.step = '1'; number.value = row.channel; number.required = true;
      number.addEventListener('input', () => { row.channel = Number(number.value); channelDirty = true; legend.textContent = `Канал ${number.value || '—'}`; });
      const name = document.createElement('input'); name.value = row.name; name.maxLength = 100; name.required = true;
      name.addEventListener('input', () => { row.name = name.value; channelDirty = true; });
      const stream = document.createElement('select');
      for (const [value, label] of [['main', 'Основной (main)'], ['sub', 'Дополнительный (sub)']]) {
        const option = document.createElement('option'); option.value = value; option.textContent = label; stream.append(option);
      }
      stream.value = row.stream_type; stream.addEventListener('change', () => { row.stream_type = stream.value; channelDirty = true; });
      const enabled = document.createElement('input'); enabled.type = 'checkbox'; enabled.checked = row.enabled;
      const enabledLabel = document.createElement('label'); enabledLabel.className = 'setup-check';
      enabledLabel.append(enabled, document.createTextNode('Включить живое видео'));
      enabled.addEventListener('change', () => { row.enabled = enabled.checked; channelDirty = true; });
      const fields = document.createElement('div'); fields.className = 'setup-channel-fields';
      fields.append(rowField('Номер', number), rowField('Название', name), rowField('Поток', stream));
      selected.addEventListener('change', () => {
        for (const control of [number, name, stream, enabled]) control.disabled = !selected.checked;
      });
      for (const control of [number, name, stream, enabled]) control.disabled = !row.selected;
      node.append(legend, choice, fields, enabledLabel); list.append(node);
    });
  }
  function sharedConnection() {
    return { nvr_host: field('nvr_host').value.trim(), rtsp_port: Number(field('rtsp_port').value),
      username: field('username').value, password: field('password').value };
  }
  function testPayload() {
    return { ...sharedConnection(), channel: Number(field('probe_channel').value), stream_type: field('probe_stream').value,
      ...(editing ? { camera_id: editing.id } : {}) };
  }
  function archivePayload() {
    return field('archive_enabled').checked ? { http_scheme: field('http_scheme').value,
      http_port: Number(field('http_port').value), utc_offset_minutes: Number(field('utc_offset_minutes').value) } : null;
  }
  function savedArchive(camera) {
    return camera?.archive ? { http_scheme: camera.archive.http_scheme, http_port: camera.archive.http_port,
      utc_offset_minutes: camera.archive.utc_offset_minutes } : null;
  }
  function reset() {
    form.reset(); rows = []; channelDirty = false; editing = null; requestId = null; requestPayload = null;
    field('password').required = true; field('password').placeholder = '';
    field('archive_enabled').disabled = false;
    $('setup-archive-details').open = false;
    $('setup-conflict').hidden = true; $('setup-current-archive').textContent = '';
    ui.message(message, ''); ui.message(discovery, ''); ui.message(live, 'Подключение ещё не проверено.');
    $('setup-password-help').textContent = 'Пароль хранится только на сервере и не возвращается в ответах.';
  }
  function openAdd(trigger) {
    if (busy) return;
    opening += 1; reset(); $('setup-title').textContent = 'Добавить устройство Dahua';
    showStep(0, false); controller.open(trigger);
  }
  async function openEdit(id, trigger) {
    if (busy) return;
    const ticket = ++opening;
    try {
      const camera = await ui.api(`/api/cameras/${id}`);
      if (ticket !== opening) return;
      reset(); editing = camera;
      ['nvr_host', 'rtsp_port', 'username'].forEach((name) => { field(name).value = camera[name]; });
      field('password').required = false; field('password').placeholder = 'Пусто — оставить сохранённый пароль';
      $('setup-password-help').textContent = 'Пустой пароль не меняет сохранённый. Изменение RTSP-параметров отключит разметку и потребует повторной наладки.';
      field('probe_channel').value = camera.channel; field('probe_stream').value = camera.stream_type;
      field('archive_enabled').checked = Boolean(camera.archive);
      // Configured rows retain their revision; the only hard removal is guarded camera deletion.
      field('archive_enabled').disabled = Boolean(camera.archive);
      if (camera.archive) {
        ['http_scheme', 'http_port', 'utc_offset_minutes'].forEach((name) => { field(name).value = camera.archive[name]; });
        field('offset_confirmed').checked = true;
      }
      rows = [{ channel: camera.channel, name: camera.name, stream_type: camera.stream_type, enabled: camera.enabled, selected: true }];
      renderRows(); $('setup-title').textContent = `Изменить камеру: ${camera.name}`;
      showStep(1, false); controller.open(trigger);
    } catch (error) {
      if (ticket === opening) window.alert(`Не удалось открыть камеру. ${error.message}`);
    }
  }
  function next() {
    if (busy || !checkStep()) return;
    ui.message(message, '');
    if (step === 1 && !rows.length) {
      const channel = Number(field('probe_channel').value);
      rows = [{ channel, name: `Камера ${channel}`, stream_type: field('probe_stream').value, enabled: true, selected: true }];
      renderRows();
    }
    showStep(Math.min(2, step + 1));
  }
  $('setup-next').addEventListener('click', next);
  $('setup-back').addEventListener('click', () => { if (!busy) showStep(Math.max(editing ? 1 : 0, step - 1)); });
  $('setup-test').addEventListener('click', async () => {
    if (busy || !checkStep()) return;
    const payload = testPayload(); setBusy(true); ui.message(live, 'Проверка живого видео…');
    try {
      const result = await json('/api/cameras/test', payload);
      ui.message(live, result.success
        ? `Живое видео доступно: ${result.resolution} · ${result.codec} · ${Math.round(result.fps || 0)} кадров/с. Архив не проверен.`
        : `Живое видео недоступно. ${result.error || 'Проверьте параметры подключения.'} Можно сохранить и проверить позже.`, result.success ? '' : 'error');
    } catch (error) { ui.message(live, `Проверка не выполнена. ${error.message}`, 'error'); }
    finally { setBusy(false); }
  });
  $('setup-add-channel').addEventListener('click', () => {
    const input = $('setup-manual-channel'); const channel = Number(input.value);
    if (!input.reportValidity() || !Number.isInteger(channel) || channel < 1 || channel > 64) return;
    if (rows.some((row) => row.channel === channel)) { ui.message(discovery, 'Этот номер уже есть в списке.', 'error'); return; }
    rows.push({ channel, name: `Камера ${channel}`, stream_type: 'main', enabled: true, selected: true });
    channelDirty = true; renderRows(); ui.message(discovery, `Добавлен канал ${channel}.`);
  });
  $('setup-discover').addEventListener('click', async () => {
    if (busy) return;
    $('setup-archive-details').open = true;
    if (!field('http_port').reportValidity()) return;
    if (channelDirty && !window.confirm('Заменить текущий список каналов результатом обнаружения? Введённые названия будут заменены.')) return;
    const payload = { ...testPayload(), http_scheme: field('http_scheme').value, http_port: Number(field('http_port').value) };
    setBusy(true); ui.message(discovery, 'Обнаружение каналов через CGI…');
    try {
      const result = await json('/api/cameras/discover', payload);
      if (result.error_code) { ui.message(discovery, errors[result.error_code] || 'Обнаружение недоступно. Добавьте каналы вручную.', 'error'); return; }
      rows = result.channels.map((row) => ({ ...row, stream_type: 'main', enabled: true, selected: true }));
      renderRows(); channelDirty = false;
      ui.message(discovery, `Найдено каналов: ${rows.length}. Выберите нужные. Доступ к записям архива ещё не проверен.`);
    } catch (error) { ui.message(discovery, `Обнаружение не выполнено. ${error.message} Введите каналы вручную.`, 'error'); }
    finally { setBusy(false); }
  });
  field('archive_enabled').addEventListener('change', () => { archiveFields(); field('offset_confirmed').checked = false; });
  field('utc_offset_minutes').addEventListener('input', () => { field('offset_confirmed').checked = false; });
  field('http_scheme').addEventListener('change', () => {
    if (['80', '443'].includes(field('http_port').value)) field('http_port').value = field('http_scheme').value === 'https' ? '443' : '80';
  });
  const workplaceOffset = Number(dialog.dataset.workplaceOffset);
  function offsetText(value) {
    return `UTC${value < 0 ? '−' : '+'}${String(Math.floor(Math.abs(value) / 60)).padStart(2, '0')}:${String(Math.abs(value) % 60).padStart(2, '0')}`;
  }
  $('setup-use-offset').textContent = `Подставить смещение рабочего места (${offsetText(workplaceOffset)})`;
  $('setup-use-offset').addEventListener('click', () => {
    field('utc_offset_minutes').value = workplaceOffset; field('offset_confirmed').checked = false; field('offset_confirmed').focus();
  });
  $('setup-rebase').addEventListener('click', async () => {
    if (busy || !editing) return;
    setBusy(true);
    try {
      const current = await ui.api(`/api/cameras/${editing.id}`);
      editing = current;
      $('setup-current-archive').textContent = current.archive
        ? `Текущая версия ${current.archive.revision}: ${current.archive.http_scheme.toUpperCase()}, порт ${current.archive.http_port}, ${offsetText(current.archive.utc_offset_minutes)}. Проверьте свой черновик перед сохранением.`
        : 'Сейчас архив не настроен. Проверьте свой черновик перед сохранением.';
      ui.message(message, 'Версия обновлена; поля черновика не менялись. Проверьте и сохраните повторно.');
    } catch (error) { ui.message(message, error.message, 'error'); }
    finally { setBusy(false); }
  });
  form.addEventListener('submit', async (event) => {
    event.preventDefault();
    if (step !== 2) { next(); return; }
    if (busy || !checkStep()) return;
    const selected = rows.filter((row) => row.selected);
    if (!selected.length) { ui.message(message, 'Выберите хотя бы один канал.', 'error'); return; }
    if (new Set(selected.map((row) => row.channel)).size !== selected.length) { ui.message(message, 'Номера каналов не должны повторяться.', 'error'); return; }
    const channels = selected.map(({ channel, name, stream_type, enabled }) => ({ channel, name: name.trim(), stream_type, enabled }));
    const configuration = archivePayload();
    let payload;
    if (editing) {
      payload = { ...sharedConnection(), ...channels[0] };
      if (JSON.stringify(configuration) !== JSON.stringify(savedArchive(editing))) {
        payload.archive = configuration; payload.expected_archive_revision = editing.archive?.revision ?? 0;
      }
    } else {
      payload = { ...sharedConnection(), archive: configuration, channels };
      const signature = JSON.stringify(payload);
      if (signature !== requestPayload) { requestPayload = signature; requestId = ui.newRequestId(); }
      payload.request_id = requestId;
    }
    setBusy(true); ui.message(message, editing ? 'Сохранение камеры…' : 'Все выбранные каналы сохраняются одной операцией…');
    try {
      const result = await json(editing ? `/api/cameras/${editing.id}` : '/api/cameras/batch', payload, editing ? 'PUT' : 'POST');
      busy = false; ui.setBusy(form, false); controller.close();
      window.dispatchEvent(new CustomEvent('cameras-changed', { detail: result }));
    } catch (error) {
      ui.message(message, `Сохранение не подтверждено. ${error.message} Черновик сохранён в этом окне.`, 'error');
      if (error.status === 409 && editing) $('setup-conflict').hidden = false;
    } finally { setBusy(false); }
  });
  dialog.addEventListener('close', () => { opening += 1; field('password').value = ''; });
  window.CameraSetup = { openAdd, openEdit };
})();
