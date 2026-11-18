// Camera overview: live stream, stable status polling and explicit operator controls.
const cameraId = window.CAMERA_ID;
const $ = id => document.getElementById(id);
const boxesEl = $('boxes');
const controls = $('camera-controls');
let refreshSequence = 0, camera = null, workplaces = [], profileDirty = false, profileRevision = null, controlsBusy = false;

function normalizeCameraHash() {
  const legacy = {'#camera-live': 'overview', '#calibration-heading': 'mapping', '#camera-recording': 'recording', '#camera-archive': 'recording'};
  const name = legacy[location.hash];
  if (name) history.replaceState(history.state, '', `#${name}`);
}
normalizeCameraHash();
window.addEventListener('hashchange', normalizeCameraHash);
const streamUrl = $('stream').getAttribute('src');
const cameraTabs = window.OperatorUI.tabs($('camera-tabs'), {onChange(name) {
  $('stream').setAttribute('src', name === 'overview' ? streamUrl : '');
  if (name !== 'overview') boxesEl.replaceChildren();
  if (name === 'overview') requestAnimationFrame(alignBoxes);
}});

function renderControls() {
  if (!camera?.profile || controlsBusy) return;
  const profile = camera.profile;
  $('camera-enabled').checked = camera.enabled;
  $('camera-enabled').disabled = false;
  $('camera-reconnect').disabled = !camera.enabled;
  controls.querySelectorAll('[name="camera-stream"]').forEach(input => {
    input.checked = input.value === camera.stream_type; input.disabled = false;
  });
  if (!profileDirty) {
    const choices = workplaces.filter(place => place.active || place.id === profile.workplace_id);
    const options = [new Option('Без рабочего места', '')];
    for (const place of choices) options.push(new Option(`${place.name}${place.active ? '' : ' (архив)'}`, String(place.id)));
    if (profile.workplace_id && !choices.some(place => place.id === profile.workplace_id)) {
      options.push(new Option(`${profile.workplace_name || 'Рабочее место'}${profile.workplace_active === false ? ' (архив)' : ''}`, String(profile.workplace_id)));
    }
    $('camera-workplace').replaceChildren(...options);
    $('camera-workplace').value = profile.workplace_id ?? '';
    controls.querySelectorAll('[name="camera-zone"]').forEach(input => { input.checked = input.value === profile.zone; });
    $('camera-ai-enabled').checked = profile.ai_enabled;
    $('camera-time-tracking').checked = profile.time_tracking;
    $('camera-ai-fps').value = profile.ai_fps ?? '';
    profileRevision = profile.revision;
  }
  $('camera-workplace').disabled = false;
  $('camera-ai-enabled').disabled = false;
  $('camera-time-tracking').disabled = false;
  $('camera-ai-fps').disabled = false;
  $('camera-profile-save').disabled = false;
  controls.querySelectorAll('[name="camera-zone"]').forEach(input => { input.disabled = false; });
}
function showCamera(cam) {
  camera = cam;
  document.title = cam.name;
  $('camera-title').textContent = cam.name;
  $('camera-connection').textContent = `${cam.nvr_host} · канал ${cam.channel} · ${cam.stream_type === 'sub' ? 'дополнительный' : 'основной'} поток`;
  $('camera-archive-state').textContent = cam.archive
    ? `Подключение настроено: ${cam.archive.http_scheme.toUpperCase()}, порт ${cam.archive.http_port}, смещение UTC ${cam.archive.utc_offset_minutes >= 0 ? '+' : ''}${cam.archive.utc_offset_minutes} мин. Версия ${cam.archive.revision}. Доступ к записям не проверен.`
    : 'Архив не настроен. Живое видео работает независимо от архива.';
  const status = cam.status || 'OFFLINE', badge = $('status-badge');
  badge.textContent = status; badge.className = 'status ' + status.toLowerCase();
  $('st-status').textContent = status;
  $('st-people').textContent = String(cam.people_count ?? 0);
  $('st-known').textContent = String(cam.known_count ?? 0);
  $('st-unknown').textContent = String(cam.unknown_count ?? 0);
  $('st-fps').textContent = cam.current_fps != null ? cam.current_fps.toFixed(1) : '—';
  $('st-res').textContent = cam.resolution || '—';
  $('st-codec').textContent = cam.codec ? cam.codec.toUpperCase() : '—';
  $('st-reconnects').textContent = cam.reconnect_count ?? '—';
  $('st-lastframe').textContent = cam.seconds_since_last_frame != null ? `${cam.seconds_since_last_frame} сек назад` : '—';
  $('st-error').textContent = cam.error || '—';
  $('video-overlay').hidden = true;
  renderControls();
  window.dispatchEvent(new CustomEvent('camera-state', {detail: cam}));
}
async function refresh() {
  const ticket = ++refreshSequence;
  try {
    const cam = await window.OperatorUI.api(`/api/cameras/${cameraId}`);
    if (ticket === refreshSequence) showCamera(cam);
  } catch (_) {
    if (ticket !== refreshSequence) return;
    $('video-overlay').hidden = false;
    $('video-overlay').textContent = 'Не удалось обновить камеру. Проверьте сеть; последние данные сохранены.';
  }
}
async function loadWorkplaces() {
  try { workplaces = await window.OperatorUI.api('/api/workplaces'); renderControls(); }
  catch (e) { window.OperatorUI.message($('camera-controls-message'), `Рабочие места не загружены. ${e.message}`, 'err'); }
}
async function controlAction(path, method, payload, success) {
  if (controlsBusy) return;
  controlsBusy = true; ++refreshSequence;
  window.OperatorUI.setBusy(controls, true);
  window.OperatorUI.message($('camera-controls-message'), 'Сохранение…');
  try {
    const result = await window.OperatorUI.api(path, {
      method, ...(payload === undefined ? {} : {headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload)}),
    });
    ++refreshSequence;
    if (result.profile) showCamera(result);
    window.OperatorUI.message($('camera-controls-message'), success, 'ok');
    return result;
  } catch (e) {
    window.OperatorUI.message($('camera-controls-message'), e.message, 'err');
    return null;
  } finally {
    controlsBusy = false; window.OperatorUI.setBusy(controls, false); renderControls();
  }
}
$('camera-enabled').addEventListener('change', async () => {
  const result = await controlAction(`/api/cameras/${cameraId}`, 'PUT', {enabled: $('camera-enabled').checked}, 'Состояние камеры сохранено.');
  if (result) refresh();
});
$('camera-reconnect').addEventListener('click', async () => {
  if (await controlAction(`/api/cameras/${cameraId}/reconnect`, 'POST', undefined, 'Переподключение запрошено.')) refresh();
});
controls.querySelectorAll('[name="camera-stream"]').forEach(input => input.addEventListener('change', async () => {
  if (!input.checked || input.value === camera?.stream_type) return;
  if (!confirm('Смена потока отключит учёт на этой камере и потребует новой разметки прохода')) { renderControls(); return; }
  if (await controlAction(`/api/cameras/${cameraId}`, 'PUT', {stream_type: input.value}, 'Поток изменён. Разметьте проход учёта заново.')) {
    window.dispatchEvent(new CustomEvent('cameras-changed'));
    if (cameraTabs.current() === 'overview') $('stream').setAttribute('src', streamUrl);
  }
}));
$('camera-profile-form').addEventListener('input', () => { profileDirty = true; });
$('camera-time-tracking').addEventListener('change', () => { if ($('camera-time-tracking').checked) $('camera-ai-enabled').checked = true; });
$('camera-ai-enabled').addEventListener('change', () => { if (!$('camera-ai-enabled').checked) $('camera-time-tracking').checked = false; });
$('camera-profile-form').addEventListener('submit', async event => {
  event.preventDefault();
  if (!camera || controlsBusy || !$('camera-profile-form').reportValidity()) return;
  const payload = {
    expected_revision: profileRevision,
    workplace_id: $('camera-workplace').value ? Number($('camera-workplace').value) : null,
    zone: controls.querySelector('[name="camera-zone"]:checked').value,
    ai_enabled: $('camera-ai-enabled').checked,
    time_tracking: $('camera-time-tracking').checked,
    ai_fps: $('camera-ai-fps').value === '' ? null : Number($('camera-ai-fps').value),
  };
  const result = await controlAction(`/api/cameras/${cameraId}/profile`, 'PUT', payload, 'Настройки камеры сохранены.');
  if (result) { profileDirty = false; renderControls(); }
});

let boxesLoading = false;
async function refreshBoxes() {
  // Metadata still drives source-staleness guards while the mapping tab hides the stream.
  if (document.hidden) { boxesEl.replaceChildren(); return; }
  if (boxesLoading) return;
  boxesLoading = true;
  let data;
  try {
    const res = await fetch(`/api/cameras/${cameraId}/detections`);
    if (!res.ok) throw new Error();
    data = await res.json();
  } catch (_) { boxesEl.replaceChildren(); return; }
  finally { boxesLoading = false; }
  boxesEl.replaceChildren();
  if (!document.hidden && cameraTabs.current() === 'overview' && window.CameraOverlay.isFresh(data)) {
    window.CameraOverlay.render(boxesEl, data.tracks, { image: $('stream'), links: true, metrics: true });
  }
  window.dispatchEvent(new CustomEvent('camera-detections', {detail: data}));
}
$('stream').onerror = () => { if (cameraTabs.current() === 'overview') $('video-overlay').hidden = false; };
for (const id of ['camera-edit', 'camera-edit-archive']) {
  const button = $(id);
  button.addEventListener('click', () => window.CameraSetup.openEdit(cameraId, button));
}
window.addEventListener('cameras-changed', () => { refresh(); loadWorkplaces(); });
window.addEventListener('camera-attendance-changed', refresh);
refresh(); loadWorkplaces(); refreshBoxes();
setInterval(refresh, 2000); setInterval(refreshBoxes, 500);
function alignBoxes() {
  const stream = $('stream');
  const rect = window.CameraOverlay.fitRect(stream.clientWidth, stream.clientHeight, stream.naturalWidth, stream.naturalHeight, getComputedStyle(stream).objectFit);
  if (rect) Object.assign(boxesEl.style, { inset: 'auto', left: `${stream.offsetLeft + rect.left}px`, top: `${stream.offsetTop + rect.top}px`, width: `${rect.width}px`, height: `${rect.height}px` });
}
new ResizeObserver(alignBoxes).observe($('stream'));
$('stream').addEventListener('load', alignBoxes);
document.addEventListener('visibilitychange', () => { if (document.hidden) boxesEl.replaceChildren(); else refreshBoxes(); });
alignBoxes();
