// Atomic employee profile; history below remains read-only and independent.
const employeeId = window.EMPLOYEE_ID;
const $ = (id) => document.getElementById(id);
const form = $('emp-form');
let current = null;
let kept = new Set();
let staged = [];
let busy = false;
let loadSequence = 0;

function errorText(detail) {
  if (typeof detail === 'string') return detail;
  if (detail?.code === 'PHOTO_VALIDATION_FAILED') {
    return 'Ничего не сохранено. ' + detail.files.map((file) => `${file.filename}: ${file.error}`).join('; ');
  }
  if (Array.isArray(detail)) return detail.map((error) => `${error.loc?.join('.') || ''}: ${error.msg}`).join('; ');
  return 'Не удалось выполнить запрос. Повторите попытку.';
}
async function api(url, options = {}) {
  const headers = { ...options.headers };
  if (options.method && options.method !== 'GET') headers['X-Operator-Request'] = '1';
  const response = await fetch(url, { ...options, headers });
  const data = response.status === 204 ? null : await response.json();
  if (!response.ok) {
    const error = new Error(errorText(data?.detail)); error.status = response.status; throw error;
  }
  return data;
}
function status(message, error = null, photo = false) {
  $(photo ? 'photo-status' : 'employee-status').textContent = message;
  if (error?.status === 409) $('reload-conflict').hidden = false;
}
function setBusy(value) {
  busy = value;
  for (const element of form.elements) element.disabled = value;
  for (const id of ['file-input', 'save-photos', 'cancel-photos']) $(id).disabled = value;
  for (const button of $('faces-grid').querySelectorAll('button')) button.disabled = value;
  form.setAttribute('aria-busy', String(value));
}
function updateSummary() {
  document.title = current.name; $('emp-title').textContent = current.name;
  $('employee-remediation').textContent = [
    current.needs_name_completion ? 'Старое имя сохранено. Укажите имя и фамилию, когда сможете их уточнить.' : '',
    current.needs_photo_remediation ? 'Нужно исправить набор фотографий до 1–15. До этого можно менять данные и отключать активность.' : '',
  ].filter(Boolean).join(' ');
}
async function fillPlaces(selected) {
  const places = await api('/api/workplaces');
  const existing = new Set((current?.workplaces || []).map((place) => place.id));
  const selectedSet = new Set(selected);
  form.elements.workplace_ids.replaceChildren(...places.filter((place) => place.active || existing.has(place.id)).map((place) => {
    const option = new Option(place.name + (place.active ? '' : ' (архив)'), place.id);
    option.selected = selectedSet.has(place.id); return option;
  }));
}
async function loadEmployee(preserveDraft = false) {
  const sequence = ++loadSequence;
  try {
    const employee = await api(`/api/employees/${employeeId}?faces=true`);
    if (sequence !== loadSequence) return;
    const omitted = new Set((current?.faces || []).filter((face) => !kept.has(face.id)).map((face) => face.id));
    const selectedPlaces = preserveDraft
      ? Array.from(form.elements.workplace_ids.selectedOptions, (option) => Number(option.value))
      : employee.workplaces.map((place) => place.id);
    current = employee;
    kept = new Set(employee.faces.filter((face) => !preserveDraft || !omitted.has(face.id)).map((face) => face.id));
    if (!preserveDraft) {
      form.elements.first_name.value = employee.first_name || '';
      form.elements.last_name.value = employee.last_name || '';
      form.elements.external_id.value = employee.external_id || '';
      form.elements.active.checked = employee.active;
    }
    updateSummary(); renderFaces();
    await fillPlaces(selectedPlaces);
    $('reload-conflict').hidden = true;
    status(preserveDraft ? 'Текущая версия загружена. Черновик и новые файлы сохранены; проверьте данные и нажмите сохранение ещё раз.' : '');
  } catch (error) {
    status('Не удалось загрузить карточку: ' + error.message, error);
    $('reload-conflict').hidden = false;
  }
}
form.onsubmit = async (event) => {
  event.preventDefault(); if (busy || !current) return;
  const first = form.elements.first_name.value.trim(), last = form.elements.last_name.value.trim();
  if ((!first || !last) && !(current.needs_name_completion && !first && !last)) {
    status('Укажите непустые имя и фамилию вместе.'); return;
  }
  const payload = {
    expected_revision: current.revision, external_id: form.elements.external_id.value.trim() || null,
    active: form.elements.active.checked,
    workplace_ids: Array.from(form.elements.workplace_ids.selectedOptions, (option) => Number(option.value)),
  };
  if (first && last) Object.assign(payload, { first_name: first, last_name: last });
  setBusy(true); status('Сохраняем данные…');
  try {
    current = await api(`/api/employees/${employeeId}`, { method: 'PUT',
      headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
    updateSummary(); status('Данные сохранены.'); $('reload-conflict').hidden = true;
  } catch (error) { status(error.message + ' Черновик оставлен без изменений.', error); }
  finally { setBusy(false); }
};
function renderFaces() {
  if (!current) return;
  $('faces-count').textContent = `· принято ${current.faces_count}/15 · в черновике ${kept.size} сохранённых + ${staged.length} новых`;
  const entries = [
    ...current.faces.filter((face) => kept.has(face.id)).map((face) => ({ face, url: face.thumbnail_url, label: `Фотография ${face.id}` })),
    ...staged.map((entry) => ({ ...entry, label: entry.file.name })),
  ];
  $('faces-grid').replaceChildren(...entries.map((entry) => {
    const cell = document.createElement('div'); cell.className = 'face-cell';
    const image = document.createElement('img'); image.src = entry.url; image.alt = entry.label;
    const remove = document.createElement('button'); remove.type = 'button'; remove.className = 'btn';
    remove.textContent = 'Убрать'; remove.setAttribute('aria-label', `Убрать: ${entry.label}`); remove.disabled = busy;
    remove.onclick = () => {
      if (entry.face) kept.delete(entry.face.id);
      else { URL.revokeObjectURL(entry.url); staged = staged.filter((item) => item.url !== entry.url); }
      renderFaces(); $('file-input').focus();
    };
    cell.append(image, remove); return cell;
  }));
}
$('file-input').onchange = () => {
  const incoming = Array.from($('file-input').files);
  if (kept.size + staged.length + incoming.length > 15) {
    status('Максимум 15 фото в итоговом наборе. Для замены сначала уберите прежнее фото из черновика.', null, true);
  } else {
    staged.push(...incoming.map((file) => ({ file, url: URL.createObjectURL(file) })));
    renderFaces(); status('Файлы выбраны, но ещё не сохранены.', null, true);
  }
  $('file-input').value = '';
};
$('save-photos').onclick = async () => {
  if (busy || !current) return;
  const count = kept.size + staged.length;
  if (count < 1 || count > 15) { status('В итоговом наборе должно быть 1–15 фотографий.', null, true); return; }
  const body = new FormData();
  body.append('metadata', JSON.stringify({ expected_revision: current.revision, keep_face_ids: [...kept] }));
  staged.forEach((entry) => body.append('files', entry.file));
  setBusy(true); status('Проверяем весь набор фотографий…', null, true);
  try {
    current = await api(`/api/employees/${employeeId}/faces`, { method: 'PUT', body });
    staged.forEach((entry) => URL.revokeObjectURL(entry.url)); staged = [];
    kept = new Set(current.faces.map((face) => face.id));
    renderFaces(); updateSummary(); status('Фотографии сохранены одним набором.', null, true);
    $('reload-conflict').hidden = true;
  } catch (error) { status(error.message + ' Прежние фотографии и черновик не изменены.', error, true); }
  finally { setBusy(false); }
};
$('cancel-photos').onclick = () => {
  if (!current || busy) return;
  staged.forEach((entry) => URL.revokeObjectURL(entry.url)); staged = [];
  kept = new Set(current.faces.map((face) => face.id)); renderFaces();
  status('Изменения фотографий отменены.', null, true);
};
$('reload-conflict').onclick = () => { if (!busy) loadEmployee(Boolean(current)); };
$('btn-workplace').onclick = () => { $('workplace-error').textContent = ''; $('workplace-dialog').showModal(); };
$('workplace-cancel').onclick = () => $('workplace-dialog').close();
$('workplace-dialog').addEventListener('close', () => $('btn-workplace').focus());
$('workplace-form').onsubmit = async (event) => {
  event.preventDefault(); const placeForm = event.currentTarget;
  const button = placeForm.querySelector('[type="submit"]'); button.disabled = true;
  try {
    const place = await api('/api/workplaces', { method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name: placeForm.elements.name.value.trim() }) });
    const selected = Array.from(form.elements.workplace_ids.selectedOptions, (option) => Number(option.value));
    await fillPlaces([...selected, place.id]); placeForm.reset(); $('workplace-dialog').close();
  } catch (error) { $('workplace-error').textContent = error.message; }
  finally { button.disabled = false; }
};
window.addEventListener('pagehide', () => staged.forEach((entry) => URL.revokeObjectURL(entry.url)));

// Work views use the server's explicit workplace offset, never browser-local time.
const workOffsets = { work: 0, presence: 0 };
function fmtTime(iso) { return iso ? iso.replace('T', ' ') : '—'; }
function workDuration(ms) {
  const seconds = Math.floor(Math.max(0, ms || 0) / 1000);
  return `${String(Math.floor(seconds / 3600)).padStart(2, '0')}:${String(Math.floor(seconds / 60) % 60).padStart(2, '0')}:${String(seconds % 60).padStart(2, '0')}`;
}
function historyPager(name, page) {
  const target = $(`${name}-pager`);
  target.replaceChildren();
  for (const [title, value, disabled] of [
    ['Назад', Math.max(0, page.offset - page.limit), page.offset === 0],
    ['Далее', page.offset + page.limit, page.offset + page.items.length >= page.total],
  ]) {
    const button = document.createElement('button');
    button.type = 'button'; button.className = 'btn'; button.textContent = title;
    button.disabled = disabled;
    button.onclick = () => { workOffsets[name] = value; loadHistory(name); };
    target.append(button);
  }
  const count = document.createElement('span');
  count.textContent = `Всего: ${page.total}`;
  target.append(count);
}
async function loadHistory(name) {
  try {
    const path = name === 'work' ? '/api/attendance/sessions' : '/api/presence/legacy';
    const page = await api(`${path}?employee_id=${employeeId}&limit=50&offset=${workOffsets[name]}`);
    const target = $(`${name}-table`).querySelector('tbody');
    target.replaceChildren();
    $(`${name}-empty`).hidden = page.items.length > 0;
    $(`${name}-error`).hidden = true;
    for (const item of page.items) {
      const values = name === 'work'
        ? [fmtTime(item.started_at), item.ended_at ? fmtTime(item.ended_at) : 'Нет подтверждённого выхода',
           workDuration(item.confirmed_ms), workDuration(item.running_ms), workDuration(item.review_ms),
           `${{ OPEN: 'Открыт — предварительно', CLOSED: 'Закрыт', REVIEW: 'На проверке' }[item.status] || item.status}: ${item.reason_codes.join('; ') || 'без конфликтов'}`]
        : [item.camera_name || '—', fmtTime(item.started_at), fmtTime(item.last_seen_at),
           item.ended_at ? fmtTime(item.ended_at) : 'Историческое незакрытое наблюдение',
           item.confidence != null ? item.confidence.toFixed(3) : '—'];
      const tr = document.createElement('tr');
      for (const value of values) { const td = document.createElement('td'); td.textContent = value; tr.append(td); }
      target.append(tr);
    }
    historyPager(name, page);
  } catch (error) {
    const message = $(`${name}-error`);
    message.textContent = `Не удалось загрузить историю. Повторите: ${error.message}`;
    message.hidden = false;
    const retry = document.createElement('button');
    retry.type = 'button'; retry.className = 'btn'; retry.textContent = 'Повторить';
    retry.onclick = () => loadHistory(name); message.append(retry);
  }
}
$('employee-report').href = `/presence?employee_id=${employeeId}#report`;
$('employee-events').href = `/presence?employee_id=${employeeId}#events`;
loadEmployee();
// Employee forms are never replaced by polling; only explicit conflict reloads rebase drafts.
loadHistory('work');
loadHistory('presence');
