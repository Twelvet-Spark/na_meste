// сотрудники: список + добавление.

const listEl = document.getElementById('employee-list');
const emptyEl = document.getElementById('empty');
const summaryEl = document.getElementById('summary');
const modal = document.getElementById('modal');
const form = document.getElementById('employee-form');

async function api(url, options = {}) {
  const res = await fetch(url, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  });
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail || detail; } catch (e) { /* ignore */ }
    const error = new Error(detail);
    error.status = res.status;
    throw error;
  }
  return res.status === 204 ? null : res.json();
}

async function loadEmployees() {
  let employees;
  try {
    employees = await api('/api/employees');
  } catch (e) {
    summaryEl.textContent = 'Ошибка загрузки: ' + e.message;
    return;
  }

  listEl.innerHTML = '';
  emptyEl.hidden = employees.length > 0;
  const active = employees.filter((e) => e.active).length;
  summaryEl.textContent = employees.length
    ? `${employees.length} сотрудников · ${active} активных` : '';

  for (const emp of employees) {
    listEl.appendChild(renderEmployee(emp));
  }
}

function renderEmployee(emp) {
  const li = document.createElement('li');
  li.className = 'employee-row';

  const info = document.createElement('div');
  info.className = 'camera-info';

  const name = document.createElement('a');
  name.href = `/employees/${emp.id}`;
  name.textContent = emp.name;
  if (!emp.active) name.classList.add('inactive');

  const meta = document.createElement('div');
  meta.className = 'meta muted';
  const bits = [];
  bits.push(emp.faces_count === 1 ? '1 лицо' : `${emp.faces_count} лиц`);
  if (emp.external_id) bits.push(`ID: ${emp.external_id}`);
  bits.push(emp.active ? 'активен' : 'неактивен');
  meta.textContent = bits.join(' · ');
  info.append(name, meta);

  const actions = document.createElement('div');
  actions.className = 'actions';
  const open = document.createElement('a');
  open.className = 'btn';
  open.href = `/employees/${emp.id}`;
  open.textContent = 'Открыть';
  const del = document.createElement('button');
  del.className = 'btn ghost danger';
  del.textContent = 'Удалить';
  del.onclick = async () => {
    if (!confirm(`Удалить сотрудника «${emp.name}» и его фотографии? Если есть история, удаление запрещено — отключите флаг «Активен» в карточке. История рабочего времени и наблюдений не удаляется.`)) return;
    try { await api(`/api/employees/${emp.id}`, { method: 'DELETE' }); loadEmployees(); }
    catch (e) { alert(e.status === 409 ? 'У сотрудника есть история: удаление запрещено. Откройте карточку и отключите флаг «Активен». Часы и наблюдения останутся в журнале.' : 'Ошибка удаления: ' + e.message); }
  };
  actions.append(open, del);

  li.append(info, actions);
  return li;
}

// ---------- добавление ----------

form.onsubmit = async (e) => {
  e.preventDefault();
  try {
    await api('/api/employees', {
      method: 'POST',
      body: JSON.stringify({
        name: form.name.value.trim(),
        external_id: form.external_id.value.trim() || null,
      }),
    });
    modal.hidden = true;
    form.reset();
    loadEmployees();
  } catch (e) {
    alert('Ошибка сохранения: ' + e.message);
  }
};

document.getElementById('btn-add').onclick = () => {
  form.reset();
  modal.hidden = false;
  form.name.focus();
};
document.getElementById('btn-cancel').onclick = () => { modal.hidden = true; };
modal.addEventListener('click', (e) => { if (e.target === modal) modal.hidden = true; });

loadEmployees();
