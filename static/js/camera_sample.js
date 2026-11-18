// Freeze a real native frame and append a verified face through the existing atomic photo-set API.
(() => {
  'use strict';
  const { normalizeRect, nativeCrop, pointInImage } = window.CameraCrop;
  const root = document.getElementById('camera-sample');
  if (!root) return;
  const ui = window.OperatorUI;
  const $ = (id) => document.getElementById(id);
  const image = $('sample-image'), stage = $('sample-stage'), selectionBox = $('sample-selection');
  const canvas = $('sample-preview'), select = $('sample-employee'), confirm = $('sample-confirm');
  const inputs = ['sample-x1', 'sample-y1', 'sample-x2', 'sample-y2'].map($);
  let frame = null, employee = null, selection = null, drag = null;
  let frameVersion = 0, employeeVersion = 0, loadingFrame = false, loadingEmployee = false, saving = false, conflict = false;

  function status(message, error = false) {
    $('sample-status').textContent = message;
    $('sample-status').className = error ? 'form-status err' : 'form-status';
  }
  function sync() {
    const busy = saving || loadingFrame;
    $('sample-freeze').disabled = busy;
    $('sample-cancel').disabled = busy || !frame;
    select.disabled = busy || loadingEmployee;
    confirm.disabled = busy || loadingEmployee || !employee || !selection || conflict;
    for (const input of inputs) input.disabled = busy || !frame;
    $('sample-reload-employee').disabled = busy || loadingEmployee;
    $('sample-save').disabled = busy || loadingEmployee || !frame || !selection || !employee
      || !confirm.checked || conflict || !employee.face_model_id || employee.faces.length >= 15;
    stage.setAttribute('aria-disabled', String(busy));
  }
  function releaseFrame() {
    frameVersion++; frame = selection = drag = null; confirm.checked = false;
    image.removeAttribute('src'); selectionBox.setAttribute('hidden', '');
    canvas.width = canvas.height = 0; canvas.hidden = true;
    $('sample-frame').hidden = true; $('sample-frame-info').textContent = '';
    inputs.forEach((input) => { input.value = ''; });
    sync();
  }
  function paint(preview = true) {
    selectionBox.toggleAttribute('hidden', !selection);
    if (selection) {
      const [x1, y1, x2, y2] = selection;
      for (const [key, value] of Object.entries({ x: x1, y: y1, width: x2 - x1, height: y2 - y1 })) selectionBox.setAttribute(key, value);
      inputs.forEach((input, i) => { input.value = String(selection[i]); });
    }
    if (preview) {
      const crop = frame && nativeCrop(selection, image.naturalWidth, image.naturalHeight);
      canvas.hidden = !crop;
      if (crop) {
        const scale = Math.min(1, 256 / Math.max(crop.width, crop.height));
        canvas.width = Math.max(1, Math.round(crop.width * scale));
        canvas.height = Math.max(1, Math.round(crop.height * scale));
        canvas.getContext('2d').drawImage(image, crop.x, crop.y, crop.width, crop.height, 0, 0, canvas.width, canvas.height);
        $('sample-crop-info').textContent = `${crop.width} × ${crop.height} исходных пикселей`;
      } else $('sample-crop-info').textContent = 'Выберите прямоугольник на кадре.';
    }
    sync();
  }
  function describeEmployee() {
    if (!employee) return;
    $('sample-employee-info').textContent = `${employee.name} · фотографий ${employee.faces.length}/15${employee.active ? '' : ' · сотрудник отключён'}`;
    if (employee.faces.length >= 15) status('У сотрудника уже 15 фотографий. Управляйте набором в его карточке.', true);
    else if (!employee.face_model_id) status('Модель лиц недоступна. Подготовьте модели; кадр и выбор можно оставить.', true);
    else status(frame ? 'Проверьте фрагмент и подтвердите личность сотрудника.' : 'Зафиксируйте кадр и выберите одно чёткое лицо.');
  }
  async function loadEmployee() {
    const ticket = ++employeeVersion;
    employee = null; confirm.checked = false; conflict = false; loadingEmployee = true;
    $('sample-reload-employee').hidden = true; $('sample-employee-info').textContent = '';
    sync();
    try {
      if (!select.value) return;
      const data = await ui.api(`/api/employees/${Number(select.value)}`);
      if (ticket !== employeeVersion) return;
      employee = data; describeEmployee();
    } catch (error) { if (ticket === employeeVersion) status(error.message, true); }
    finally { if (ticket === employeeVersion) { loadingEmployee = false; sync(); } }
  }
  $('sample-freeze').onclick = async () => {
    if (saving || loadingFrame) return;
    releaseFrame();
    const ticket = ++frameVersion;
    loadingFrame = true; sync(); status('Получаем исходный кадр…');
    try {
      const data = await ui.api(`/api/attendance/cameras/${Number(window.CAMERA_ID)}/snapshot`);
      if (ticket !== frameVersion) return;
      image.src = `data:image/jpeg;base64,${data.jpeg_base64}`;
      await image.decode();
      if (ticket !== frameVersion) return;
      if (image.naturalWidth !== data.width || image.naturalHeight !== data.height) throw new Error('Размер исходного кадра не совпал. Снимите кадр заново.');
      frame = data; $('sample-frame').hidden = false;
      $('sample-frame-info').textContent = `${data.received_at.slice(0, 10)} ${data.received_at.slice(11, 19)} · ${data.width} × ${data.height}`;
      status('Выделите человека или лицо. В прямоугольнике должно быть ровно одно чёткое лицо, не меньше 80 px.');
    } catch (error) {
      if (ticket === frameVersion) { releaseFrame(); status(`Кадр не получен. ${error.message}`, true); }
    } finally { if (ticket === frameVersion || !frame) { loadingFrame = false; sync(); } }
  };
  $('sample-cancel').onclick = () => { releaseFrame(); status('Кадр удалён. Живой поток не останавливался.'); };
  select.onchange = loadEmployee;
  $('sample-reload-employee').onclick = loadEmployee;
  confirm.onchange = sync;
  inputs.forEach((input) => input.addEventListener('change', () => {
    confirm.checked = false;
    selection = inputs.every((node) => node.value !== '') ? normalizeRect(...inputs.map((node) => Number(node.value))) : null;
    paint();
  }));
  stage.onpointerdown = (event) => {
    if (!frame || saving || loadingFrame || event.button !== 0) return;
    event.preventDefault();
    drag = pointInImage(event.clientX, event.clientY, image.getBoundingClientRect());
    stage.setPointerCapture(event.pointerId); selection = null; confirm.checked = false; paint();
  };
  stage.onpointermove = (event) => {
    if (!drag) return;
    selection = normalizeRect(...drag, ...pointInImage(event.clientX, event.clientY, image.getBoundingClientRect()));
    paint(false);
  };
  function finishDrag(event) {
    if (!drag) return;
    selection = normalizeRect(...drag, ...pointInImage(event.clientX, event.clientY, image.getBoundingClientRect()));
    drag = null;
    if (stage.hasPointerCapture(event.pointerId)) stage.releasePointerCapture(event.pointerId);
    paint();
  }
  stage.onpointerup = finishDrag;
  stage.onpointercancel = finishDrag;
  $('sample-form').onsubmit = async (event) => {
    event.preventDefault();
    if ($('sample-save').disabled) return;
    const crop = nativeCrop(selection, image.naturalWidth, image.naturalHeight);
    if (!crop) return;
    saving = true; sync(); status('Проверяем лицо и сохраняем фотографию…');
    const revision = employee.revision, employeeId = employee.id;
    const keep = employee.faces.map((face) => face.id);
    try {
      const native = document.createElement('canvas'); native.width = crop.width; native.height = crop.height;
      native.getContext('2d').drawImage(image, crop.x, crop.y, crop.width, crop.height, 0, 0, crop.width, crop.height);
      const blob = await new Promise((resolve) => native.toBlob(resolve, 'image/png'));
      native.width = native.height = 0;
      if (!blob) throw new Error('Не удалось подготовить исходный фрагмент.');
      const body = new FormData();
      body.append('metadata', JSON.stringify({ expected_revision: revision, keep_face_ids: keep }));
      body.append('files', blob, `camera-${Number(window.CAMERA_ID)}-${frame.seq}.png`);
      employee = await ui.api(`/api/employees/${employeeId}/faces`, {
        method: 'PUT', headers: { 'X-Operator-Request': '1' }, body,
      });
      releaseFrame(); describeEmployee();
      status(`Фото добавлено: ${employee.faces.length}/15. Оно поможет при следующих узнаваниях; прошлые часы не изменены.`);
    } catch (error) {
      if (error.status === 409) {
        conflict = true; confirm.checked = false; $('sample-reload-employee').hidden = false;
        status('Карточка сотрудника изменилась. Обновите её, проверьте выбор и подтвердите личность заново. Кадр сохранён.', true);
      } else {
        const codes = { FACE_MODEL_UNAVAILABLE: 'Модель лиц недоступна.', ENROLLMENT_UNAVAILABLE: 'Добавление фотографий недоступно.', ENROLLMENT_BUSY: 'Проверка другой фотографии уже выполняется.' };
        const detail = error.detail;
        const message = detail?.files?.map((file) => file.error).filter(Boolean).join('; ') || codes[detail?.code] || error.message;
        status(`${message} Кадр сохранён; автоматического повтора нет.`, true);
      }
    } finally { saving = false; sync(); }
  };
  async function loadChoices() {
    try {
      for (const person of await ui.api('/api/employees')) {
        const option = document.createElement('option'); option.value = String(person.id);
        option.textContent = `${person.name}${person.active ? '' : ' (отключён)'}`; select.append(option);
      }
    } catch (error) { status(`Список сотрудников не загружен. ${error.message}`, true); }
  }
  window.addEventListener('pagehide', () => { employeeVersion++; releaseFrame(); });
  sync(); loadChoices();
})();
