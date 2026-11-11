// Employee list and atomic onboarding. Files stay in this page's memory only.
const listEl = document.getElementById('employee-list');
const emptyEl = document.getElementById('empty');
const summaryEl = document.getElementById('summary');
const modal = document.getElementById('modal');
const form = document.getElementById('employee-form');
const $ = (id) => document.getElementById(id);
let staged = [];
let requestId = null;
let busy = false;
let listSequence = 0;

function newRequestId() {
  // getRandomValues also works on the deployment's trusted-LAN HTTP origin.
  const bytes = crypto.getRandomValues(new Uint8Array(16));
  bytes[6] = (bytes[6] & 15) | 64; bytes[8] = (bytes[8] & 63) | 128;
  const hex = Array.from(bytes, (byte) => byte.toString(16).padStart(2, '0')).join('');
  return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20)}`;
}

function errorText(detail) {
  if (typeof detail === 'string') return detail;
  if (detail?.code === 'PHOTO_VALIDATION_FAILED') {
    return 'Ничего не сохранено. ' + detail.files.map((f) => `${f.filename}: ${f.error}`).join('; ');
  }
  if (Array.isArray(detail)) return detail.map((e) => `${e.loc?.join('.') || ''}: ${e.msg}`).join('; ');
  return 'Не удалось выполнить запрос. Повторите попытку.';
}
async function api(url, options = {}) {
  const headers = { ...options.headers };
  if (options.method && options.method !== 'GET') headers['X-Operator-Request'] = '1';
  const response = await fetch(url, { ...options, headers });
  const data = response.status === 204 ? null : await response.json();
  if (!response.ok) {
    const error = new Error(errorText(data?.detail));
    error.status = response.status;
    throw error;
  }
  return data;
}
function setBusy(value) {
  busy = value;
  for (const element of form.elements) element.disabled = value;
  form.setAttribute('aria-busy', String(value));
}
async function loadEmployees() {
  const sequence = ++listSequence;
  try {
    const employees = await api('/api/employees');
    if (sequence !== listSequence) return;
    listEl.replaceChildren(...employees.map(renderEmployee));
    emptyEl.hidden = employees.length > 0;
    summaryEl.textContent = `${employees.length} сотрудников · ${employees.filter((e) => e.active).length} активных`;
    $('btn-retry').hidden = true;
  } catch (error) {
    if (sequence !== listSequence) return;
    summaryEl.textContent = 'Не удалось обновить список: ' + error.message;
    $('btn-retry').hidden = false;
  }
}
function renderEmployee(employee) {
  const item = document.createElement('li'); item.className = 'employee-row';
  const info = document.createElement('div'); info.className = 'camera-info';
  const name = document.createElement('a');
  name.href = `/employees/${employee.id}`; name.textContent = employee.name;
  if (!employee.active) name.classList.add('inactive');
  const meta = document.createElement('div'); meta.className = 'meta muted';
  const bits = [`${employee.faces_count}/15 фото`, employee.active ? 'активен' : 'неактивен'];
  if (employee.external_id) bits.push(`ID: ${employee.external_id}`);
  bits.push(...employee.workplaces.map((p) => p.name + (p.active ? '' : ' (архив)')));
  if (employee.needs_name_completion) bits.push('Уточните имя и фамилию');
  if (employee.needs_photo_remediation) bits.push('Исправьте набор фотографий');
  meta.textContent = bits.join(' · '); info.append(name, meta);
  const actions = document.createElement('div'); actions.className = 'actions';
  const open = document.createElement('a'); open.className = 'btn';
  open.href = name.href; open.textContent = 'Открыть';
  const remove = document.createElement('button'); remove.className = 'btn ghost danger';
  remove.textContent = 'Удалить'; remove.type = 'button';
  remove.onclick = async () => {
    if (!confirm(`Удалить «${employee.name}» и фотографии? При наличии истории удаление запрещено; отключите активность в карточке.`)) return;
    remove.disabled = true;
    try { await api(`/api/employees/${employee.id}`, { method: 'DELETE' }); await loadEmployees(); }
    catch (error) { summaryEl.textContent = error.message; }
    finally { remove.disabled = false; }
  };
  actions.append(open, remove); item.append(info, actions); return item;
}
function renderPhotos() {
  $('draft-count').textContent = `Выбрано: ${staged.length}/15`;
  $('draft-photos').replaceChildren(...staged.map((entry) => {
    const cell = document.createElement('div'); cell.className = 'face-cell';
    const image = document.createElement('img'); image.src = entry.url; image.alt = entry.file.name;
    const remove = document.createElement('button'); remove.type = 'button'; remove.className = 'btn';
    remove.textContent = 'Убрать'; remove.setAttribute('aria-label', `Убрать ${entry.file.name}`);
    remove.onclick = () => {
      URL.revokeObjectURL(entry.url); staged = staged.filter((item) => item !== entry);
      requestId = null; renderPhotos(); $('create-files').focus();
    };
    cell.append(image, remove); return cell;
  }));
}
$('create-files').onchange = () => {
  const incoming = Array.from($('create-files').files);
  if (staged.length + incoming.length > 15) {
    $('create-status').textContent = 'Можно выбрать максимум 15 фотографий. Уберите лишние и повторите выбор.';
  } else {
    staged.push(...incoming.map((file) => ({ file, url: URL.createObjectURL(file) })));
    requestId = null; renderPhotos(); $('create-status').textContent = '';
  }
  $('create-files').value = '';
};
form.addEventListener('input', () => { if (!busy) requestId = null; });
form.onsubmit = async (event) => {
  event.preventDefault();
  if (busy) return;
  if (!staged.length) { $('create-status').textContent = 'Выберите хотя бы одну фотографию.'; return; }
  requestId ||= newRequestId();
  const body = new FormData();
  body.append('metadata', JSON.stringify({
    first_name: form.elements.first_name.value.trim(), last_name: form.elements.last_name.value.trim(),
    external_id: form.elements.external_id.value.trim() || null, request_id: requestId,
    workplace_ids: Array.from(form.elements.workplace_ids.selectedOptions, (option) => Number(option.value)),
  }));
  staged.forEach((entry) => body.append('files', entry.file));
  setBusy(true); $('create-status').textContent = 'Проверяем фотографии. Сотрудник будет создан только при успехе всех проверок…';
  try {
    const employee = await api('/api/employees', { method: 'POST', body });
    location.assign(`/employees/${employee.id}`);
  } catch (error) {
    $('create-status').textContent = error.message + ' Черновик сохранён на этой странице.';
  } finally { setBusy(false); }
};
async function loadPlaces(selected = null) {
  const picker = form.elements.workplace_ids;
  const keep = selected || Array.from(picker.selectedOptions, (option) => Number(option.value));
  const places = await api('/api/workplaces');
  picker.replaceChildren(...places.filter((place) => place.active).map((place) => {
    const option = new Option(place.name, place.id); option.selected = keep.includes(place.id); return option;
  }));
}
$('btn-add').onclick = async () => {
  modal.showModal(); form.elements.last_name.focus();
  try { await loadPlaces(); } catch (error) { $('create-status').textContent = 'Рабочие места недоступны: ' + error.message; }
};
$('btn-cancel').onclick = () => { if (!busy) modal.close(); };
modal.addEventListener('cancel', (event) => { if (busy) event.preventDefault(); });
modal.addEventListener('close', () => $('btn-add').focus());
$('btn-retry').onclick = loadEmployees;
$('btn-workplace').onclick = () => { $('workplace-error').textContent = ''; $('workplace-dialog').showModal(); };
$('workplace-cancel').onclick = () => $('workplace-dialog').close();
$('workplace-dialog').addEventListener('close', () => $('btn-workplace').focus());
$('workplace-form').onsubmit = async (event) => {
  event.preventDefault(); const placeForm = event.currentTarget;
  const button = placeForm.querySelector('[type="submit"]'); button.disabled = true;
  try {
    const place = await api('/api/workplaces', { method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name: placeForm.elements.name.value.trim() }) });
    const keep = Array.from(form.elements.workplace_ids.selectedOptions, (option) => Number(option.value));
    await loadPlaces([...keep, place.id]); requestId = null; placeForm.reset(); $('workplace-dialog').close();
  } catch (error) { $('workplace-error').textContent = error.message; }
  finally { button.disabled = false; }
};
window.addEventListener('pagehide', () => staged.forEach((entry) => URL.revokeObjectURL(entry.url)));
loadEmployees();
