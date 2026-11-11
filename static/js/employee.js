// Employee profile, enrollment, work intervals and separate legacy observations.

const employeeId = window.EMPLOYEE_ID;
const $ = (id) => document.getElementById(id);

async function api(url, options = {}) {
  const res = await fetch(url, { ...options });
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail || detail; } catch (e) { /* ignore */ }
    throw new Error(detail);
  }
  return res.status === 204 ? null : res.json();
}

// ---------- данные сотрудника ----------

async function loadEmployee() {
  let emp;
  try {
    emp = await api(`/api/employees/${employeeId}`);
  } catch (e) {
    alert('Сотрудник не найден');
    location.href = '/employees';
    return;
  }
  document.title = `${emp.name}`;
  $('emp-title').textContent = emp.name;
  const f = $('emp-form');
  f.name.value = emp.name;
  f.external_id.value = emp.external_id || '';
  f.active.checked = emp.active;
  $('faces-count').textContent = emp.faces_count ? `· ${emp.faces_count}` : '';
}

$('emp-form').onsubmit = async (e) => {
  e.preventDefault();
  const f = $('emp-form');
  try {
    await api(`/api/employees/${employeeId}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        name: f.name.value.trim(),
        external_id: f.external_id.value.trim() || null,
        active: f.active.checked,
      }),
    });
    loadEmployee();
  } catch (err) {
    alert('Ошибка сохранения: ' + err.message);
  }
};

// ---------- загрузка фото ----------

$('btn-upload').onclick = () => $('file-input').click();
$('file-input').onchange = () => uploadFiles($('file-input').files);

async function uploadFiles(fileList) {
  const files = Array.from(fileList);
  if (!files.length) return;

  for (const file of files) {
    const li = addStatusItem(file.name);
    setStep(li, 'pending', 'Поиск лица…');
    const fd = new FormData();
    fd.append('files', file);
    try {
      const res = await fetch(`/api/employees/${employeeId}/faces`, {
        method: 'POST',
        body: fd,
      });
      const data = await res.json();
      const r = (data.results || [])[0];
      if (!res.ok || !r || !r.success) {
        const msg = (r && r.error) || data.detail || 'Ошибка обработки';
        setStep(li, 'err', `✕ ${msg}`);
        continue;
      }
      let text = '✓ Лицо найдено · Эмбеддинг создан · Сохранено';
      if (r.warnings && r.warnings.length) {
        text += ` Предупреждения: ${r.warnings.join('; ')}`;
      }
      setStep(li, 'ok', text);
    } catch (e) {
      setStep(li, 'err', '✕ Ошибка загрузки: ' + e.message);
    }
  }
  $('file-input').value = '';
  loadFaces();
  loadEmployee();
}

function addStatusItem(filename) {
  const li = document.createElement('li');
  li.className = 'upload-item';
  for (const name of ['upload-name', 'upload-step']) {
    const span = document.createElement('span');
    span.className = name;
    li.append(span);
  }
  li.querySelector('.upload-name').textContent = filename;
  $('upload-status').prepend(li);
  return li;
}

function setStep(li, kind, text) {
  li.className = `upload-item ${kind}`;
  li.querySelector('.upload-step').textContent = text;
}

// ---------- список лиц ----------

async function loadFaces() {
  let emp;
  try {
    emp = await api(`/api/employees/${employeeId}?faces=1`);
  } catch (e) { return; }
  const grid = $('faces-grid');
  grid.replaceChildren();
  const faces = emp.faces || [];
  $('faces-count').textContent = faces.length ? `· ${faces.length}` : '';
  for (const face of faces) {
    const cell = document.createElement('div');
    cell.className = 'face-cell';
    const img = document.createElement('img');
    img.src = `/api/employees/${employeeId}/faces/${face.id}/thumbnail`;
    img.alt = 'лицо';
    const del = document.createElement('button');
    del.className = 'btn ghost danger face-del';
    del.textContent = '✕';
    del.title = 'Удалить лицо';
    del.onclick = async () => {
      if (!confirm('Удалить это лицо?')) return;
      try {
        await api(`/api/employees/${employeeId}/faces/${face.id}`, { method: 'DELETE' });
        loadFaces();
        loadEmployee();
      } catch (e) { alert('Ошибка удаления: ' + e.message); }
    };
    cell.append(img, del);
    grid.appendChild(cell);
  }
}

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
loadFaces();
loadHistory('work');
loadHistory('presence');
