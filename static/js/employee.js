// Atomic employee profile. Shared helpers own drafts; history stays independent.
const employeeId = window.EMPLOYEE_ID;
const $ = (id) => document.getElementById(id);
const UI = window.OperatorUI;
const api = UI.api;
const form = $('emp-form');
let current = null;
let busy = false;
let placesReady = false;
let loadSequence = 0;
const photos = new EmployeeForm.PhotoDraft({
  input: $('file-input'), grid: $('faces-grid'), count: $('faces-count'),
  drop: $('photo-drop'), status: $('photo-status'),
});
const picker = new EmployeeForm.WorkplacePicker({
  select: form.elements.workplace_ids, trigger: $('btn-workplace'),
  dialog: $('workplace-dialog'), form: $('workplace-form'), error: $('workplace-error'),
  onChange: ({ created } = {}) => {
    if (created) {
      placesReady = true; form.elements.workplace_ids.disabled = false;
      UI.message($('places-status'), 'Новое рабочее место выбрано. Сохраните данные, чтобы назначить его сотруднику.');
    }
  },
});
const revisionConflict = UI.conflict($('revision-conflict'), $('reload-conflict'), () => loadEmployee(true));
photos.setBusy(true);

function status(message, error = null, photo = false) {
  UI.message($(photo ? 'photo-status' : 'employee-status'), message, error ? 'error' : '');
  revisionConflict.show(error);
}
function setBusy(value) {
  busy = value;
  UI.setBusy($('profile-editor'), value);
  photos.setBusy(value || !current);
  if (!value) {
    $('metadata-fields').disabled = !current;
    $('photo-fields').disabled = !current;
    form.elements.workplace_ids.disabled = !placesReady;
  }
}
function updateSummary() {
  document.title = current.name; $('emp-title').textContent = current.name;
  $('employee-remediation').textContent = [
    current.active ? 'Активен.' : 'Неактивен — распознавание отключено.',
    current.needs_name_completion ? 'Старое имя сохранено. Уточните имя и фамилию вместе.' : '',
    current.needs_photo_remediation ? 'Исправьте набор до 1–15 фотографий. Данные и активность можно менять уже сейчас.' : '',
  ].filter(Boolean).join(' ');
}
async function loadPlaces() {
  try {
    await picker.load(null, current.workplaces);
    placesReady = true; UI.message($('places-status'), ''); $('places-retry').hidden = true;
  } catch (error) {
    placesReady = false;
    UI.message($('places-status'), 'Рабочие места не загружены. При сохранении их назначения не изменятся: ' + error.message, 'error');
    $('places-retry').hidden = false;
  }
}
async function loadEmployee(preserveDraft = false) {
  if (busy) return false;
  const sequence = ++loadSequence;
  setBusy(true);
  try {
    const employee = await api(`/api/employees/${employeeId}?faces=true`);
    if (sequence !== loadSequence) return false;
    const selected = preserveDraft ? picker.selected : employee.workplaces.map((place) => place.id);
    current = employee;
    photos.setAccepted(employee.faces, preserveDraft);
    if (!preserveDraft) {
      form.elements.first_name.value = employee.first_name || '';
      form.elements.last_name.value = employee.last_name || '';
      form.elements.external_id.value = employee.external_id || '';
      form.elements.active.checked = employee.active;
    }
    // Keep existing archived tags visible even when the workplace request fails.
    picker.existing = new Set(employee.workplaces.map((place) => place.id));
    picker.places = Array.from(new Map([...picker.places, ...employee.workplaces].map((place) => [place.id, place])).values());
    picker.render(selected);
    await loadPlaces();
    if (sequence !== loadSequence) return false;
    updateSummary(); revisionConflict.clear(); $('profile-load-error').hidden = true;
    status(preserveDraft ? 'Текущая версия загружена. Ваш черновик сохранён. Проверьте данные и повторите сохранение.' : '');
    if (preserveDraft) status('Новые файлы и ваши удаления сохранены в черновике. Проверьте итоговый набор перед повторным сохранением.', null, true);
    return true;
  } catch (error) {
    $('profile-load-error').hidden = false;
    $('profile-load-message').textContent = 'Не удалось загрузить карточку: ' + error.message + (current ? ' Черновик оставлен без изменений.' : '');
    status('Не удалось загрузить карточку.', error);
    return false;
  } finally { setBusy(false); }
}
form.addEventListener('submit', async (event) => {
  event.preventDefault(); if (busy || !current) return;
  const first = form.elements.first_name.value.trim(), last = form.elements.last_name.value.trim();
  if ((!first || !last) && !(current.needs_name_completion && !first && !last)) {
    status('Укажите непустые имя и фамилию вместе.', new Error()); return;
  }
  const payload = {
    expected_revision: current.revision, external_id: form.elements.external_id.value.trim() || null,
    active: form.elements.active.checked,
  };
  if (placesReady) payload.workplace_ids = picker.selected;
  if (first && last) Object.assign(payload, { first_name: first, last_name: last });
  setBusy(true); status('Сохраняем данные…');
  try {
    current = await api(`/api/employees/${employeeId}`, { method: 'PUT',
      headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
    // A metadata save must not discard a separately staged photo edit.
    photos.setAccepted(current.faces, true);
    picker.existing = new Set(current.workplaces.map((place) => place.id));
    picker.render(picker.selected);
    updateSummary(); status('Данные сохранены.'); revisionConflict.clear();
  } catch (error) { status(error.message + ' Черновик оставлен без изменений.', error); }
  finally { setBusy(false); }
});
$('save-photos').addEventListener('click', async () => {
  if (busy || !current) return;
  if (photos.total < 1 || photos.total > 15) {
    status('В итоговом наборе должно быть 1–15 фотографий.', new Error(), true); return;
  }
  const body = new FormData();
  body.append('metadata', JSON.stringify({ expected_revision: current.revision, keep_face_ids: [...photos.kept] }));
  photos.files.forEach((file) => body.append('files', file));
  setBusy(true); status('Проверяем весь набор фотографий…', null, true);
  try {
    current = await api(`/api/employees/${employeeId}/faces`, { method: 'PUT', body });
    photos.setAccepted(current.faces);
    updateSummary(); status('Фотографии сохранены одним набором.', null, true); revisionConflict.clear();
  } catch (error) {
    photos.setErrors(error.detail);
    status(error.message + ' Прежние фотографии и черновик не изменены.', error, true);
  } finally { setBusy(false); }
});
$('cancel-photos').addEventListener('click', () => {
  if (!current || busy) return;
  photos.reset(); status('Изменения фотографий отменены.', null, true); $('file-input').focus();
});
$('profile-retry').addEventListener('click', () => loadEmployee(Boolean(current)));
$('places-retry').addEventListener('click', async () => {
  if (busy || !current) return;
  setBusy(true);
  try { await loadPlaces(); }
  finally { setBusy(false); }
});

// Work views use the server's explicit workplace offset, never browser-local time.
const workOffsets = { work: 0, presence: 0 };
const historySequence = { work: 0, presence: 0 };
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
  const sequence = ++historySequence[name];
  try {
    const path = name === 'work' ? '/api/attendance/sessions' : '/api/presence/legacy';
    const page = await api(`${path}?employee_id=${employeeId}&limit=50&offset=${workOffsets[name]}`);
    if (sequence !== historySequence[name]) return;
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
    if (sequence !== historySequence[name]) return;
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
