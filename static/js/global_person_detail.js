// Архив глобальной личности: только чтение, траектория и таймлайн.

const globalId = window.GLOBAL_ID;
const $ = (id) => document.getElementById(id);

function fmtDateTime(iso) {
  if (!iso) return '—';
  return new Date(iso).toLocaleString('ru-RU', {
    day: '2-digit', month: '2-digit', hour: '2-digit', minute: '2-digit', second: '2-digit',
  });
}
function fmtTime(iso) {
  if (!iso) return '—';
  return new Date(iso).toLocaleTimeString('ru-RU', {
    hour: '2-digit', minute: '2-digit', second: '2-digit',
  });
}

async function loadPerson() {
  let p;
  try {
    const res = await fetch(`/api/global-persons/${globalId}`);
    if (!res.ok) throw new Error();
    p = await res.json();
  } catch (e) {
    $('gp-name').textContent = 'Запись недоступна';
    $('gp-title').textContent = 'Не удалось загрузить · обновление через 5 секунд';
    return;
  }
  const title = p.employee_name
    ? `G#${p.global_id} · ${p.employee_name}`
    : `G#${p.global_id}`;
  document.title = `${title}`;
  $('gp-name').textContent = title;
  $('gp-title').textContent = `Только чтение · ${p.status}`;

  const meta = $('gp-meta');
  meta.replaceChildren();
  meta.append(`Сохранённый статус: ${p.status} · `);
  if (p.employee_id) {
    const employee = document.createElement('a');
    employee.href = `/employees/${p.employee_id}`;
    employee.textContent = p.employee_name || `#${p.employee_id}`;
    meta.append('Сотрудник: ', employee, ' · ');
  }
  meta.append(`Наблюдений: ${p.observations_count} · `,
    `Последняя камера: ${p.last_camera_name || (p.last_camera_id ? '#' + p.last_camera_id : '—')} · `,
    `Последнее наблюдение: ${fmtDateTime(p.last_seen_at)} · `,
    `Создана: ${fmtDateTime(p.created_at)}`);

  if (p.has_photo) {
    const img = $('gp-avatar');
    img.src = `/api/global-persons/${globalId}/photo`;
    img.hidden = false;
    img.onerror = () => { img.hidden = true; };
  }
}

async function loadTrajectory() {
  let data;
  try {
    const res = await fetch(`/api/global-persons/${globalId}/trajectory`);
    if (!res.ok) throw new Error();
    data = await res.json();
  } catch (e) {
    $('trajectory').textContent = 'Траектория недоступна. Повторная загрузка через 5 секунд.';
    return;
  }

  const box = $('trajectory');
  box.replaceChildren();
  if (!data.segments.length) {
    box.textContent = 'Траектории пока нет.';
    return;
  }
  data.segments.forEach((seg, i) => {
    if (i > 0) {
      const arrow = document.createElement('span');
      arrow.className = 'traj-arrow';
      arrow.textContent = '→';
      box.appendChild(arrow);
    }
    const node = document.createElement('span');
    node.className = 'traj-node';
    node.textContent = seg.camera_name;
    const times = document.createElement('span');
    times.className = 'muted small';
    times.textContent = `${fmtTime(seg.from)}–${fmtTime(seg.to)} · ${seg.observations}`;
    node.appendChild(times);
    box.appendChild(node);
  });
}

async function loadTimeline() {
  let items;
  try {
    const res = await fetch(`/api/global-persons/${globalId}/timeline?limit=300`);
    if (!res.ok) throw new Error();
    items = await res.json();
  } catch (e) {
    $('timeline-empty').textContent = 'Таймлайн недоступен. Повторная загрузка через 5 секунд.';
    $('timeline-empty').hidden = false;
    return;
  }

  const tbody = $('timeline-table').querySelector('tbody');
  tbody.innerHTML = '';
  $('timeline-empty').textContent = 'Наблюдений пока нет.';
  $('timeline-empty').hidden = items.length > 0;

  for (const item of items) {
    const tr = document.createElement('tr');

    const time = document.createElement('td');
    time.textContent = fmtDateTime(item.created_at);

    const kind = document.createElement('td');
    if (item.kind === 'observation') {
      const badge = document.createElement('span');
      badge.className = 'status online';
      badge.textContent = 'наблюдение';
      kind.appendChild(badge);
    } else if (item.event_type === 'camera_transition') {
      const badge = document.createElement('span');
      badge.className = 'status reconnecting';
      badge.textContent = 'переход';
      kind.appendChild(badge);
    } else if (item.event_type === 'person_lost') {
      const badge = document.createElement('span');
      badge.className = 'status offline';
      badge.textContent = 'потерян';
      kind.appendChild(badge);
    } else {
      const badge = document.createElement('span');
      badge.className = 'status offline';
      badge.textContent = item.event_type;
      kind.appendChild(badge);
    }

    const cam = document.createElement('td');
    if (item.kind === 'event' && item.event_type === 'camera_transition') {
      const from = item.from_camera_name || `#${item.payload.from_camera}`;
      const to = item.to_camera_name || `#${item.payload.to_camera}`;
      cam.textContent = `${from} → ${to}`;
    } else {
      cam.textContent = item.camera_name || '—';
    }

    const track = document.createElement('td');
    track.textContent = item.track_id != null ? `track ${item.track_id}` : '—';

    const details = document.createElement('td');
    details.className = 'muted small';
    if (item.kind === 'event') {
      const p = item.payload || {};
      if (item.event_type === 'camera_transition') {
        details.textContent =
          `similarity=${p.similarity ?? '—'} confidence=${p.confidence ?? '—'}`;
      } else if (item.event_type === 'ambiguous_match') {
        details.textContent = `неоднозначный матч (ближе G#${p.better_global_id})`;
      } else if (item.event_type === 'person_seen') {
        details.textContent = `final=${p.final ?? '—'} reid=${p.reid ?? '—'}`
          + (p.matched === false ? ' · новая identity' : '');
      } else if (item.event_type === 'person_lost') {
        details.textContent = `длительность: ${Math.round((p.duration_seconds ?? 0) / 60)} мин`
          + (p.gallery_size ? ` · эмбеддингов: ${p.gallery_size}` : '');
      } else {
        details.textContent = JSON.stringify(p).slice(0, 80);
      }
    } else {
      details.textContent = '';
    }

    const shot = document.createElement('td');
    if (item.kind === 'observation' && item.has_snapshot) {
      const img = document.createElement('img');
      img.src = `/api/global-persons/${globalId}/observations/${item.id}/photo`;
      img.className = 'obs-thumb';
      img.loading = 'lazy';
      img.alt = 'снимок';
      shot.appendChild(img);
    } else {
      shot.textContent = '—';
    }

    tr.append(time, kind, cam, track, details, shot);
    tbody.appendChild(tr);
  }
}

loadPerson();
loadTrajectory();
loadTimeline();
setInterval(() => { loadPerson(); loadTrajectory(); loadTimeline(); }, 5000);
