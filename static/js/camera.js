// страница камеры: MJPEG + overlay bounding boxes + статус.

const cameraId = window.CAMERA_ID;
const $ = (id) => document.getElementById(id);
const boxesEl = $('boxes');

async function refresh() {
  let cam;
  try {
    const res = await fetch(`/api/cameras/${cameraId}`);
    if (!res.ok) throw new Error('not found');
    cam = await res.json();
  } catch (e) {
    $('video-overlay').hidden = false;
    $('video-overlay').textContent = 'Камера не найдена';
    return;
  }

  document.title = `${cam.name}`;
  $('camera-title').textContent = cam.name;

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
    base = `Unknown${conf ? ' ' + conf : ''}`;
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

    const label = document.createElement('span');
    label.className = 'bbox-label';
    label.textContent = labelFor(track);
    box.appendChild(label);
    const footpoint = document.createElement('span');
    footpoint.className = 'bbox-footpoint';
    footpoint.title = 'Точка пересечения: низ рамки по центру';
    box.appendChild(footpoint);
    boxesEl.appendChild(box);
  }
}

// Если <img> не смог открыть MJPEG-поток (например, камера удалена)
$('stream').onerror = () => { $('video-overlay').hidden = false; };

refresh();
refreshBoxes();
setInterval(refresh, 2000);
setInterval(refreshBoxes, 500);

// Doorway calibration: a frozen bitmap has no object-fit/letterbox coordinates.
(() => {
  const canvas = $('calibration-canvas');
  const context = canvas.getContext('2d');
  const names = ['start', 'end', 'inside'];
  const fields = names.map(name => ['x', 'y'].map(axis => $(`gate-${name}-${axis}`)));
  let config = null;
  let runtime = null;
  let frozen = null;
  let image = null;
  let dirty = false;
  let busy = false;
  let cameraStatus = 'OFFLINE';
  let clickIndex = 0;

  function message(text, kind = 'pending') {
    const node = $('calibration-message');
    node.textContent = text;
    node.className = `test-result ${kind}`;
    node.hidden = !text;
  }

  async function jsonRequest(url, options) {
    let response;
    try {
      response = await fetch(url, options);
    } catch (_) {
      throw new Error('Сеть недоступна. Изменения не подтверждены; повторите попытку.');
    }
    let body;
    try { body = await response.json(); } catch (_) { body = null; }
    if (!response.ok) {
      const error = new Error(response.status === 409
        ? 'Конфигурация или источник изменились. Зафиксируйте новый кадр, проверьте координаты и сохраните повторно.'
        : response.status === 503
          ? 'Камера или база недоступны. Изменения не подтверждены; повторите попытку после восстановления.'
          : response.status === 404
            ? 'Камера не найдена. Обновите список камер.'
            : 'Проверьте роль, координаты и размеры. Линия и внутренняя точка должны быть достаточно разнесены.');
      error.status = response.status;
      throw error;
    }
    return body;
  }

  function points() {
    return fields.map(pair => pair.map(input => input.value === '' ? NaN : Number(input.value)));
  }

  function geometryError() {
    const [a, b, inside] = points();
    if ([...a, ...b, ...inside].some(value => !Number.isFinite(value) || value < 0 || value > 1)) {
      return 'Введите все координаты от 0 до 1.';
    }
    const dx = b[0] - a[0], dy = b[1] - a[1];
    const length = Math.hypot(dx, dy);
    if (length < .05 - 1e-12) return 'Разнесите концы линии минимум на 0,05.';
    if (Math.abs(dx * (inside[1] - a[1]) - dy * (inside[0] - a[0])) / length < .05 - 1e-12) {
      return 'Внутренняя точка должна быть минимум в 0,05 от линии.';
    }
    if (!$('calibration-role').value) return 'Выберите назначение камеры.';
    return '';
  }

  function updateButtons() {
    $('calibration-freeze').disabled = busy;
    $('calibration-clear').disabled = busy || !frozen;
    $('calibration-save').disabled = busy || !config || !frozen || Boolean(geometryError());
    $('calibration-disable').disabled = busy || !config || !config.role || !config.enabled;
  }

  function setBusy(value) {
    busy = value;
    for (const control of $('calibration-form').elements) control.disabled = value;
    $('calibration-freeze').textContent = value ? 'Подождите…' : 'Зафиксировать свежий кадр';
    updateButtons();
  }

  function showRuntime() {
    if (!runtime) return;
    const role = runtime.role === 'ENTRY' ? 'Вход' : runtime.role === 'EXIT' ? 'Выход' : 'Роль не назначена';
    const states = [role, runtime.enabled ? 'учёт включён' : 'учёт отключён',
                    `поток: ${cameraStatus}`];
    if (runtime.camera_enabled === false) states.push('камера отключена');
    if (runtime.ai_enabled === false) states.push('AI выключен');
    else if (runtime.ai_included === false) states.push('камера исключена из AI');
    else if (runtime.ai_available === false) states.push('AI недоступен');
    states.push(runtime.commissioning_required
      ? 'наладка не подтверждена — автоматический учёт запрещён'
      : 'акт наладки сохранён; текущую готовность проверяет монитор учёта');
    $('calibration-state').textContent = states.join(' · ');
  }

  function applyConfiguration(value) {
    config = value;
    $('calibration-role').value = value.role || '';
    $('calibration-enabled').checked = value.enabled;
    const coordinates = [value.line_start, value.line_end, value.inside_point];
    fields.forEach((pair, index) => pair.forEach((input, axis) => {
      input.value = coordinates[index] ? coordinates[index][axis] : '';
    }));
    clickIndex = coordinates.findIndex(point => !point);
    if (clickIndex < 0) clickIndex = 3;
    updateButtons();
  }

  async function loadConfiguration(fill) {
    const value = await jsonRequest(`/api/attendance/cameras/${cameraId}`);
    runtime = value;
    if (fill) applyConfiguration(value);
    showRuntime();
    return value;
  }

  function draw() {
    if (!image) return;
    context.clearRect(0, 0, canvas.width, canvas.height);
    context.drawImage(image, 0, 0, canvas.width, canvas.height);
    const values = points();
    const style = getComputedStyle(document.documentElement);
    context.strokeStyle = style.getPropertyValue('--vid-warn').trim();
    context.fillStyle = context.strokeStyle;
    const scale = Math.max(1, canvas.width / 640);
    context.lineWidth = 3 * scale;
    context.font = `${16 * scale}px sans-serif`;
    values.forEach((point, index) => {
      if (!point.every(Number.isFinite)) return;
      const x = point[0] * canvas.width, y = point[1] * canvas.height;
      context.beginPath();
      context.arc(x, y, 5 * scale, 0, Math.PI * 2);
      context.fill();
      context.fillText(index === 2 ? 'Внутри' : String(index + 1), x + 8 * scale, y - 8 * scale);
    });
    const [a, b, inside] = values;
    if (!a.every(Number.isFinite) || !b.every(Number.isFinite)) return;
    context.beginPath();
    context.moveTo(a[0] * canvas.width, a[1] * canvas.height);
    context.lineTo(b[0] * canvas.width, b[1] * canvas.height);
    context.stroke();
    if (!inside.every(Number.isFinite) || Math.hypot(b[0] - a[0], b[1] - a[1]) < .05 - 1e-12) return;
    const mid = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];
    let nx = -(b[1] - a[1]), ny = b[0] - a[0];
    const norm = Math.hypot(nx, ny);
    const sign = (inside[0] - mid[0]) * nx + (inside[1] - mid[1]) * ny >= 0 ? 1 : -1;
    const direction = $('calibration-role').value === 'EXIT' ? -sign : sign;
    nx = nx / norm * direction * .10;
    ny = ny / norm * direction * .10;
    const from = [(mid[0] - nx) * canvas.width, (mid[1] - ny) * canvas.height];
    const to = [(mid[0] + nx) * canvas.width, (mid[1] + ny) * canvas.height];
    const angle = Math.atan2(to[1] - from[1], to[0] - from[0]);
    context.beginPath();
    context.moveTo(...from);
    context.lineTo(...to);
    context.moveTo(to[0] - 14 * scale * Math.cos(angle - .45), to[1] - 14 * scale * Math.sin(angle - .45));
    context.lineTo(...to);
    context.lineTo(to[0] - 14 * scale * Math.cos(angle + .45), to[1] - 14 * scale * Math.sin(angle + .45));
    context.stroke();
  }

  function updateStep() {
    $('calibration-step').textContent = clickIndex === 0 ? 'Нажмите начало линии в проёме.'
      : clickIndex === 1 ? 'Нажмите конец линии в проёме.'
        : clickIndex === 2 ? 'Нажмите точку внутри помещения.'
          : geometryError() || 'Разметка готова. Стрелка показывает учитываемое направление. Точки на живом видео — низ рамки человека.';
  }

  $('calibration-freeze').addEventListener('click', async () => {
    setBusy(true);
    message('Получение свежего кадра…');
    try {
      const current = await loadConfiguration(false);
      const snapshot = await jsonRequest(`/api/attendance/cameras/${cameraId}/snapshot`);
      if (snapshot.source_revision !== current.source_revision) {
        throw new Error('Источник сменился во время фиксации. Повторите фиксацию кадра.');
      }
      const bitmap = new Image();
      bitmap.src = `data:image/jpeg;base64,${snapshot.jpeg_base64}`;
      try { await bitmap.decode(); } catch (_) {
        throw new Error('Не удалось открыть кадр. Повторите фиксацию.');
      }
      if (!dirty) applyConfiguration(current);
      else config = current; // Explicit refresh preserves the operator's entered geometry.
      frozen = snapshot;
      image = bitmap;
      canvas.width = snapshot.width;
      canvas.height = snapshot.height;
      $('calibration-image').hidden = false;
      $('calibration-metadata').textContent =
        `${snapshot.width} × ${snapshot.height} · источник ${snapshot.source_revision} · поток ${snapshot.stream_id} · кадр ${snapshot.seq} · получение сервером: ${snapshot.received_at}`;
      draw();
      updateStep();
      message('Кадр зафиксирован. Проверьте видимость ног и отметьте проём.', 'ok');
    } catch (error) {
      frozen = null;
      message(error.message, 'err');
    } finally { setBusy(false); }
  });

  $('calibration-clear').addEventListener('click', () => {
    for (const pair of fields) for (const input of pair) input.value = '';
    clickIndex = 0;
    dirty = true;
    draw();
    updateStep();
    updateButtons();
    fields[0][0].focus();
  });

  canvas.addEventListener('click', event => {
    if (!frozen || !image || busy || clickIndex >= 3) return;
    const rect = canvas.getBoundingClientRect();
    // The canvas itself is the displayed bitmap, not a surrounding panel.
    const x = (event.clientX - rect.left) / rect.width;
    const y = (event.clientY - rect.top) / rect.height;
    if (x < 0 || x > 1 || y < 0 || y > 1) return;
    fields[clickIndex][0].value = x.toFixed(3);
    fields[clickIndex][1].value = y.toFixed(3);
    clickIndex += 1;
    dirty = true;
    draw();
    updateStep();
    updateButtons();
  });

  $('calibration-form').addEventListener('input', () => {
    dirty = true;
    const values = points();
    clickIndex = values.findIndex(point => !point.every(Number.isFinite));
    if (clickIndex < 0) clickIndex = 3;
    draw();
    updateStep();
    updateButtons();
  });

  async function save(payload) {
    setBusy(true);
    message('Сохранение конфигурации…');
    try {
      const saved = await jsonRequest(`/api/attendance/cameras/${cameraId}`, {
        method: 'PUT', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      applyConfiguration(saved);
      dirty = false;
      message('Конфигурация сохранена. Изменение разметки требует повторной наладки.', 'ok');
      try { await loadConfiguration(false); } catch (_) {
        message('Конфигурация сохранена, но статус недоступен. Обновите страницу.', 'pending');
      }
    } catch (error) {
      if (error.status === 409) frozen = null;
      message(error.message, 'err');
    } finally { setBusy(false); }
  }

  $('calibration-form').addEventListener('submit', event => {
    event.preventDefault();
    const error = geometryError();
    if (busy || !config || !frozen || error) {
      message(error || 'Сначала зафиксируйте свежий кадр.', 'err');
      return;
    }
    const [start, end, inside] = points();
    save({ role: $('calibration-role').value, enabled: $('calibration-enabled').checked,
      line_start: start, line_end: end, inside_point: inside,
      reference_width: frozen.width, reference_height: frozen.height,
      expected_revision: config.revision, expected_source_revision: frozen.source_revision });
  });

  $('calibration-disable').addEventListener('click', () => {
    if (busy || !config || !config.role || !config.enabled) return;
    save({ role: config.role, enabled: false, line_start: config.line_start,
      line_end: config.line_end, inside_point: config.inside_point,
      reference_width: config.reference_width, reference_height: config.reference_height,
      expected_revision: config.revision, expected_source_revision: config.source_revision });
  });

  window.addEventListener('camera-state', event => {
    cameraStatus = event.detail.status || 'OFFLINE';
    showRuntime();
  });
  loadConfiguration(true).catch(error => message(error.message, 'err'));
  setInterval(() => loadConfiguration(false).catch(error => {
    $('calibration-state').textContent = `Статус учёта недоступен. ${error.message}`;
  }), 5000);

  // The live footpoints use the image rectangle too, even when the grid row
  // stretches the surrounding video panel beyond the image's actual height.
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
})();
