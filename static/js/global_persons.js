// Архив глобальных наблюдений: только чтение, отдельно от рабочих часов.

const $ = (id) => document.getElementById(id);

function fmtDateTime(iso) {
  if (!iso) return '—';
  return new Date(iso).toLocaleString('ru-RU', {
    day: '2-digit', month: '2-digit', hour: '2-digit', minute: '2-digit', second: '2-digit',
  });
}

const state = { search: '', status: '', camera: '' };
let searchTimer = null;

async function loadCameras() {
  try {
    const res = await fetch('/api/cameras');
    if (!res.ok) return;
    const cameras = await res.json();
    const select = $('f-camera');
    for (const cam of cameras) {
      const opt = document.createElement('option');
      opt.value = cam.id;
      opt.textContent = cam.name || `камера #${cam.id}`;
      select.appendChild(opt);
    }
  } catch (e) { /* без списка камер фильтр просто не заполнится */ }
}

function apiUrl() {
  const params = new URLSearchParams({ limit: '200' });
  if (state.search) params.set('search', state.search);
  if (state.status) params.set('status', state.status);
  if (state.camera) params.set('camera_id', state.camera);
  return `/api/global-persons?${params}`;
}

async function load() {
  let persons;
  try {
    const res = await fetch(apiUrl());
    if (!res.ok) throw new Error();
    persons = await res.json();
  } catch (e) {
    $('summary').textContent = 'Ошибка загрузки';
    return;
  }

  const tbody = $('persons-table').querySelector('tbody');
  tbody.innerHTML = '';
  $('empty').hidden = persons.length > 0;
  $('summary').textContent = persons.length
    ? `Показано: ${persons.length} · только чтение` : 'Только чтение';

  for (const p of persons) {
    const tr = document.createElement('tr');

    const photo = document.createElement('td');
    if (p.has_photo) {
      const img = document.createElement('img');
      img.src = `/api/global-persons/${p.global_id}/photo`;
      img.className = 'gp-avatar';
      img.loading = 'lazy';
      img.alt = '';
      img.onerror = () => { img.remove(); };
      photo.appendChild(img);
    } else {
      photo.textContent = '—';
    }

    const idTd = document.createElement('td');
    const link = document.createElement('a');
    link.href = `/global/${p.global_id}`;
    link.textContent = `G#${p.global_id}`;
    idTd.appendChild(link);

    const status = document.createElement('td');
    const badge = document.createElement('span');
    badge.className = 'status ' + (p.status === 'ACTIVE' ? 'online'
      : p.status === 'NEW' ? 'connecting' : 'offline');
    badge.textContent = p.status;
    status.appendChild(badge);

    const emp = document.createElement('td');
    if (p.employee_id) {
      const empLink = document.createElement('a');
      empLink.href = `/employees/${p.employee_id}`;
      empLink.textContent = p.employee_name || `#${p.employee_id}`;
      emp.appendChild(empLink);
    } else {
      emp.textContent = '—';
    }

    const cam = document.createElement('td');
    cam.textContent = p.last_camera_name || (p.last_camera_id ? `#${p.last_camera_id}` : '—');

    const count = document.createElement('td');
    count.textContent = String(p.observations_count);

    const seen = document.createElement('td');
    seen.textContent = fmtDateTime(p.last_seen_at);

    const open = document.createElement('td');
    const openLink = document.createElement('a');
    openLink.className = 'btn';
    openLink.href = `/global/${p.global_id}`;
    openLink.textContent = 'Открыть';
    open.appendChild(openLink);

    tr.append(photo, idTd, status, emp, cam, count, seen, open);
    tbody.appendChild(tr);
  }
}

$('f-search').addEventListener('input', (e) => {
  clearTimeout(searchTimer);
  searchTimer = setTimeout(() => { state.search = e.target.value.trim(); load(); }, 300);
});
$('f-status').addEventListener('change', (e) => { state.status = e.target.value; load(); });
$('f-camera').addEventListener('change', (e) => { state.camera = e.target.value; load(); });

loadCameras();
load();
setInterval(load, 5000);
