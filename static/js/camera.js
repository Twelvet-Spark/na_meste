// страница камеры: MJPEG + overlay bounding boxes + статус.

const cameraId = window.CAMERA_ID;
const $ = (id) => document.getElementById(id);
const boxesEl = $('boxes');
let refreshSequence = 0;

async function refresh() {
  const ticket = ++refreshSequence;
  let cam;
  try {
    cam = await window.OperatorUI.api(`/api/cameras/${cameraId}`);
    if (ticket !== refreshSequence) return;
  } catch (e) {
    if (ticket !== refreshSequence) return;
    $('video-overlay').hidden = false;
    $('video-overlay').textContent = 'Не удалось обновить камеру. Проверьте сеть; последние данные сохранены.';
    return;
  }

  document.title = `${cam.name}`;
  $('camera-title').textContent = cam.name;
  $('camera-connection').textContent = `${cam.nvr_host} · канал ${cam.channel} · ${cam.stream_type === 'sub' ? 'дополнительный' : 'основной'} поток`;
  $('camera-archive-state').textContent = cam.archive
    ? `Подключение настроено: ${cam.archive.http_scheme.toUpperCase()}, порт ${cam.archive.http_port}, смещение UTC ${cam.archive.utc_offset_minutes >= 0 ? '+' : ''}${cam.archive.utc_offset_minutes} мин. Версия ${cam.archive.revision}. Доступ к записям не проверен.`
    : 'Архив не настроен. Живое видео работает независимо от архива.';

  const status = cam.status || 'OFFLINE';
  const badge = $('status-badge');
  badge.textContent = status;
  badge.className = 'status ' + status.toLowerCase();

  $('st-status').textContent = status;
  $('st-people').textContent = String(cam.people_count ?? 0);
  $('st-known').textContent = String(cam.known_count ?? 0);
  $('st-unknown').textContent = String(cam.unknown_count ?? 0);
  $('st-fps').textContent = cam.current_fps != null ? cam.current_fps.toFixed(1) : '—';
  $('st-res').textContent = cam.resolution || '—';
  $('st-codec').textContent = cam.codec ? cam.codec.toUpperCase() : '—';
  $('st-reconnects').textContent = cam.reconnect_count != null ? cam.reconnect_count : '—';
  $('st-lastframe').textContent = cam.seconds_since_last_frame != null
    ? `${cam.seconds_since_last_frame} сек назад` : '—';
  $('st-error').textContent = cam.error || '—';
  $('video-overlay').hidden = true;
  window.dispatchEvent(new CustomEvent('camera-state', { detail: cam }));
}

// ---------- bounding boxes ----------

function labelFor(track) {
  let base;
  if (track.state === 'recognized' && track.employee_name) {
    const conf = track.confidence != null ? track.confidence.toFixed(2) : '';
    base = `${track.employee_name}${conf ? ' ' + conf : ''}`;
  } else if (track.state === 'unknown') {
    const conf = track.confidence != null ? track.confidence.toFixed(2) : '';
    base = `Неизвестный${conf ? ' ' + conf : ''}`;
  } else {
    base = '…';
  }
  // глобальная личность (межкамерный трекинг): локальный трек + global_id
  const tag = `T${track.track_id}` + (track.global_id ? `·G${track.global_id}` : '');
  return `${base} [${tag}]`;
}

async function refreshBoxes() {
  let data;
  try {
    const res = await fetch(`/api/cameras/${cameraId}/detections`);
    if (!res.ok) throw new Error();
    data = await res.json();
  } catch (e) {
    return; // тихо — попробуем в следующий раз
  }

  boxesEl.replaceChildren();
  for (const track of data.tracks || []) {
    const [x1, y1, x2, y2] = track.bbox;
    const box = document.createElement('div');
    box.className = 'bbox ' + track.state;
    box.style.left = (x1 * 100) + '%';
    box.style.top = (y1 * 100) + '%';
    box.style.width = ((x2 - x1) * 100) + '%';
    box.style.height = ((y2 - y1) * 100) + '%';

    const acknowledgedVisit = typeof track.visit_id === 'string'
      && /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(track.visit_id);
    const label = document.createElement(acknowledgedVisit ? 'a' : 'span');
    if (acknowledgedVisit) {
      label.href = `/global/visits/${encodeURIComponent(track.visit_id)}`;
      label.style.pointerEvents = 'auto';
      label.title = 'Открыть сохранённый визит — не запись рабочего времени';
    }
    label.className = 'bbox-label';
    label.textContent = labelFor(track);
    const reasons = Array.isArray(track.identity_reason_codes) ? track.identity_reason_codes.join(', ') : '';
    if (reasons) {
      label.title = `${label.title ? label.title + '. ' : ''}Причина состояния личности: ${reasons}`;
      label.setAttribute('aria-label', `${label.textContent}. Причина состояния личности: ${reasons}`);
    }
    box.appendChild(label);
    const footpoint = document.createElement('span');
    footpoint.className = 'bbox-footpoint';
    footpoint.title = 'Точка пересечения: низ рамки по центру';
    box.appendChild(footpoint);
    const metric = document.createElement('span');
    metric.className = 'bbox-metric';
    metric.dataset.trackId = String(track.track_id);
    box.appendChild(metric);
    boxesEl.appendChild(box);
  }
  window.dispatchEvent(new CustomEvent('camera-detections', { detail: data }));
}

// Если <img> не смог открыть MJPEG-поток (например, камера удалена)
$('stream').onerror = () => { $('video-overlay').hidden = false; };

for (const id of ['camera-edit', 'camera-edit-archive']) {
  const button = $(id);
  button.addEventListener('click', () => window.CameraSetup.openEdit(cameraId, button));
}
window.addEventListener('cameras-changed', refresh);
refresh();
refreshBoxes();
setInterval(refresh, 2000);
setInterval(refreshBoxes, 500);

// Align live boxes to the displayed image, not the stretched workspace row.
function alignBoxes() {
  const stream = $('stream');
  boxesEl.style.inset = 'auto';
  boxesEl.style.left = `${stream.offsetLeft}px`;
  boxesEl.style.top = `${stream.offsetTop}px`;
  boxesEl.style.width = `${stream.clientWidth}px`;
  boxesEl.style.height = `${stream.clientHeight}px`;
}
new ResizeObserver(alignBoxes).observe($('stream'));
$('stream').addEventListener('load', alignBoxes);
alignBoxes();
