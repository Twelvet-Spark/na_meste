// Each domain owns one frozen request. An uncertain response never changes its target or bytes.
(() => {
  'use strict';
  const $ = (id) => document.getElementById(id), UI = window.OperatorUI, crop = window.CameraCrop;
  const root = $('visit-detail');
  if (!root) return;
  const endpoint = `/api/global-persons/visits/${encodeURIComponent(root.dataset.visitId)}`;
  const correctionNames = ['identity', 'sample', 'time'];
  const states = Object.fromEntries([...correctionNames, 'learning'].map((name) => [name, { busy: false, pending: null }]));
  const errors = {
    FACE_MODEL_UNAVAILABLE: 'Модель лиц недоступна. Добавление фото отключено; исправление личности и удаление образцов доступны. Проверьте диагностику и обновите наблюдение после восстановления модели.',
    ENROLLMENT_UNAVAILABLE: 'Подготовка фото недоступна. Исправление личности и удаление образцов доступны. Проверьте диагностику и обновите наблюдение.',
    VISIT_SAMPLE_ALREADY_REMOVED: 'Этот образец уже удалён. История обновлена.',
    VISIT_SAMPLE_NOT_FOUND: 'Запись об этом образце не найдена. История обновлена.',
    IDENTITY_REVISION_CONFLICT: 'Личность уже изменена. Обновите наблюдение и проверьте решение заново.',
    VISIT_FACE_MISSING: 'Сохранённого лица нет. Выберите фрагмент снимка или загрузите фотографию.',
    VISIT_FACE_EXPIRED: 'Срок хранения лица истёк. Загрузите фотографию; личность можно исправить без неё.',
    VISIT_PHOTO_EXPIRED: 'Срок хранения снимка истёк. Загрузите фотографию.',
    VISIT_FACE_CHANGED: 'Лицо изменилось. Обновите наблюдение и проверьте новое изображение.',
    VISIT_PHOTO_CHANGED: 'Снимок изменился. Обновите наблюдение и выделите фрагмент заново.',
    VISIT_PHOTO_MISSING: 'Снимка нет. Загрузите фотографию.',
    VISIT_FACE_QUALITY: 'Лицо недостаточно чёткое или маленькое. Выберите другое фото с одним лицом от 80 px.',
    VISIT_FACE_INVALID: 'Сохранённое лицо не прошло проверку. Выберите другое фото.',
    FACE_LIMIT_REACHED: 'У сотрудника уже 15 образцов. Управляйте фотографиями в его карточке.',
    EMPLOYEE_REVISION_CONFLICT: 'Набор фотографий изменился. Обновите версию сотрудника и подтвердите выбор заново.',
    GALLERY_CHANGED: 'Галерея изменилась. Обновите версию сотрудника и подтвердите выбор заново.',
    VISIT_SAMPLE_CHANGED: 'Этот образец уже изменён или удалён. Обновите историю.',
    VISIT_TIME_PREVIEW_CHANGED: 'Расчёт устарел. Рассчитайте изменение заново и проверьте новые итоги.',
    VISIT_TIME_IDENTITY_CHANGED: 'Сначала укажите сотрудника в журнале и обновите расчёт.',
    VISIT_TIME_OUTSIDE_OBSERVATION: 'Интервал должен находиться внутри первого и последнего наблюдения.',
    VISIT_TIME_FUTURE: 'Нельзя засчитать время в будущем.',
    PRESENCE_FLUSH_FAILED: 'Учёт времени не смог сохранить наблюдения. Проверьте диагностику и повторите запрос.',
    PRESENCE_UNAVAILABLE: 'Учёт времени недоступен. Исправление личности и фото работают отдельно.',
    VISIT_TIME_CORRECTION_CHANGED: 'Связанные исправления изменены. Обновите историю; посторонние записи не отменены.',
  };
  let visit = null, digests = {}, employees = [], employee = null, employeeVersion = 0, employeeLoading = false;
  let selection = [0, 0, 1, 1], drag = null, uploadFile = null, uploadURL = null, uploadVersion = 0;
  let preview = null, previewBusy = false, previewVersion = 0, offset = null, historyVersion = 0, refreshBusy = false;
  let learningAnalyzing = false;
  const bounds = ['sample-x1', 'sample-y1', 'sample-x2', 'sample-y2'].map($);
  const locked = () => refreshBusy || Object.values(states).some((state) => state.busy || state.pending) || previewBusy;
  const occupied = (name) => states[name].busy || Boolean(states[name].pending);
  const learningBlocked = () => refreshBusy || previewBusy || correctionNames.some(occupied);
  const blocked = (name) => refreshBusy || occupied(name) || (name !== 'learning' && occupied('learning'));
  const message = (name, text, error = false) => UI.message($(name + '-status'), text, error ? 'err' : '');
  const stamp = (value) => value ? value.replace('T', ' ') : '—';
  const errorText = (error) => errors[error.detail?.code] || error.message;
  function audit(name) {
    const actor = $(name + '-actor'), reason = $(name + '-reason');
    for (const input of [actor, reason]) {
      if (!input.value.trim()) { input.focus(); input.reportValidity(); message(name, 'Укажите оператора и причину.', true); return null; }
    }
    return { actor: actor.value.trim(), reason: reason.value.trim(), request_id: UI.newRequestId() };
  }
  function invalidatePreview() {
    previewVersion++;
    if (states.time.pending) return;
    preview = null; $('time-result').hidden = true; $('time-confirm').checked = false;
  }
  function sourceReady() {
    switch ($('sample-source').value) {
      case 'ALIGNED_FACE': return visit?.employee_sample_eligible && digests.face && digests.face === visit.face_sha256;
      case 'PHOTO_CROP': return selection && digests.photo && digests.photo === visit?.photo_sha256;
      case 'UPLOAD': return Boolean(uploadFile);
      default: return false;
    }
  }
  function sync() {
    for (const name of correctionNames) {
      $(name + '-fields').disabled = blocked(name) || !visit;
      $(name + '-form').setAttribute('aria-busy', String(states[name].busy));
      for (const button of $(name + '-history').querySelectorAll('button')) button.disabled = blocked(name) || !visit;
    }
    const identityPending = states.identity.pending;
    $('identity-submit').disabled = refreshBusy || occupied('learning') || states.identity.busy || !visit || (!identityPending && $('identity-mode').value === 'EMPLOYEE' && !$('identity-employee').value);
    $('identity-submit').textContent = identityPending ? 'Повторить тот же запрос' : 'Сохранить личность';
    $('identity-employee').disabled = $('identity-mode').value !== 'EMPLOYEE';
    $('identity-search').disabled = $('identity-mode').value !== 'EMPLOYEE';
    const chosen = employees.find((row) => String(row.id) === $('identity-employee').value);
    $('identity-employee-status').textContent = chosen && !chosen.active ? 'Сотрудник неактивен. Историческое исправление разрешено; оно не включает сотрудника.' : '';
    $('sample-submit').disabled = refreshBusy || occupied('learning') || states.sample.busy || !visit || (!states.sample.pending && (!visit.enrollment_available || employeeLoading || !employee || employee.faces_count >= 15 || !sourceReady() || !$('sample-confirm').checked));
    $('sample-submit').textContent = states.sample.pending ? 'Повторить тот же запрос' : 'Добавить образец';
    $('sample-reconcile').disabled = blocked('sample') || employeeLoading;
    $('sample-employee').disabled = employeeLoading;
    $('sample-search').disabled = employeeLoading;
    $('time-enabled').disabled = blocked('time') || previewBusy;
    $('time-fields').disabled = blocked('time') || previewBusy || !visit;
    $('time-preview').disabled = blocked('time') || previewBusy || !visit?.employee_id;
    $('time-confirm').disabled = blocked('time');
    $('time-apply').disabled = refreshBusy || occupied('learning') || states.time.busy || previewBusy || (!states.time.pending && (!preview || !$('time-confirm').checked));
    $('time-apply').textContent = states.time.pending ? 'Повторить тот же запрос' : 'Применить изменение часов';
    $('time-owner').textContent = visit?.employee_id ? `Сотрудник: ${visit.employee_name}. Версия личности: ${visit.identity_revision}.` : 'Сначала укажите сотрудника в разделе «Кто на изображении».';
    $('time-retry').hidden = !states.time.pending;
    $('time-retry').disabled = occupied('learning') || states.time.busy;
    root.dispatchEvent(new Event('journal-lock'));
    window.JournalLearning?.sync();
  }
  async function mutate(name, request) {
    const state = states[name];
    if (state.busy || refreshBusy || (name === 'learning'
      ? learningBlocked() || (!state.pending && learningAnalyzing)
      : occupied('learning'))) return;
    if (!state.pending && request) state.pending = Object.freeze({ ...request, options: Object.freeze({ ...request.options }) });
    const frozen = state.pending;
    if (!frozen) return;
    state.busy = true; sync(); message(name, 'Сохраняем… Не закрывайте страницу до подтверждения.');
    let result;
    try {
      result = await UI.api(frozen.url, { method: 'POST', ...frozen.options });
    } catch (error) {
      const code = error.detail?.code;
      const unavailable = error.status === 503 && (
        (name === 'sample' && ['FACE_MODEL_UNAVAILABLE', 'ENROLLMENT_UNAVAILABLE'].includes(code))
        || (name === 'learning' && ['BODY_MODEL_UNAVAILABLE', 'ENROLLMENT_UNAVAILABLE'].includes(code)));
      const uncertain = !unavailable && (!error.status || error.status < 400 || error.status === 408 || error.status >= 500);
      if (uncertain) message(name, errorText(error) + ' Результат не подтверждён. Повторите тот же запрос: сотрудник, данные и номер запроса зафиксированы. Не обновляйте страницу.', true);
      else {
        state.pending = null;
        if (name === 'sample') {
          $('sample-confirm').checked = false;
          if (error.status === 409) employee = null;
          if (unavailable) visit.enrollment_available = false;
          if (error.detail?.code === 'VISIT_PHOTO_CHANGED') delete digests.photo;
          if (error.detail?.code === 'VISIT_FACE_CHANGED') delete digests.face;
          sourceChanged();
        }
        if (name === 'time') invalidatePreview();
        message(name, errorText(error) + ' Эта операция не выполнена.', true);
      }
      state.busy = false;
      if (!uncertain && name === 'sample') {
        loadHistory();
        if (error.status === 409) loadEmployee();
      }
      if (name === 'learning') window.JournalLearning?.onMutationError(error, { uncertain });
      sync(); return;
    }
    state.pending = null; state.busy = false;
    if (name === 'identity') {
      Object.assign(visit, result); root.dispatchEvent(new CustomEvent('journal-identity', { detail: result }));
      invalidatePreview();
      window.JournalLearning?.invalidate();
      const removedFaces = result.removed_face_ids?.length || 0;
      const faceNotice = removedFaces === 1 ? ' Образец лица удалён из галереи.'
        : removedFaces > 1 ? ` Из галереи удалены образцы лица: ${removedFaces}.` : '';
      message(name, result.revoked_feedback_ids?.length
        ? 'Личность в журнале сохранена. Связанное обучение отменено.' + faceNotice + ' Прошлые часы не изменены.'
        : 'Личность в журнале сохранена. Фотографии и часы не изменены.');
    } else if (name === 'sample') {
      $('sample-confirm').checked = false;
      employee = null;
      message(name, frozen.url.endsWith('/remove') ? 'Образец удалён из галереи. Личность в журнале и часы не изменены.' : 'Образец принят в галерею для следующих узнаваний. Прежние фотографии сохранены; точность узнавания не гарантируется.');
      loadEmployee();
      window.JournalLearning?.invalidate();
    } else if (name === 'learning') {
      if (visit) {
        for (const key of ['learning_revision', 'gallery_revision']) {
          if (Number.isSafeInteger(result[key])) visit[key] = Math.max(visit[key] ?? 0, result[key]);
        }
        if (result.identity?.visit_id === root.dataset.visitId) {
          Object.assign(visit, result.identity);
          root.dispatchEvent(new CustomEvent('journal-identity', { detail: result.identity }));
          invalidatePreview();
        }
      }
      window.JournalLearning?.onMutationResult(result, frozen);
    } else {
      invalidatePreview(); message(name, frozen.url.endsWith('/undo') ? 'Связанные исправления часов отменены. Другие записи сохранены.' : 'Показанное исправление часов применено. Проверьте итоговый отчёт.');
    }
    sync(); loadHistory();
  }
  function jsonRequest(path, payload) {
    return { url: endpoint + path, options: { headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) } };
  }
  function optionsFor(name) {
    const select = $(name + '-employee'), chosen = select.value;
    const search = $(name + '-search').value.trim().toLocaleLowerCase();
    select.replaceChildren(new Option('Выберите сотрудника вручную', ''));
    for (const row of employees) {
      if (String(row.id) === chosen || row.name.toLocaleLowerCase().includes(search)) select.append(new Option(row.name + (row.active ? '' : ' (неактивен)'), String(row.id)));
    }
    select.value = chosen;
  }
  async function loadEmployees() {
    $('employees-retry').hidden = true;
    try {
      employees = await UI.api('/api/employees', { cache: 'no-store' });
      for (const name of ['identity', 'sample']) optionsFor(name);
      const chosen = $('time-exclude').value;
      $('time-exclude').replaceChildren(new Option('Не исключать', ''));
      for (const row of employees) $('time-exclude').append(new Option(row.name + (row.active ? '' : ' (неактивен)'), row.id));
      $('time-exclude').value = chosen;
    } catch (error) { message('identity', 'Список сотрудников не загружен. ' + errorText(error), true); $('employees-retry').hidden = false; }
    sync();
  }
  async function loadEmployee() {
    if (blocked('sample')) return;
    const ticket = ++employeeVersion, id = $('sample-employee').value;
    employee = null; employeeLoading = true; $('sample-confirm').checked = false; sync();
    $('sample-enrollment').href = id ? `/employees/${id}#photos` : '/employees';
    $('sample-employee-status').textContent = id ? 'Загружаем версию сотрудника…' : '';
    try {
      if (!id) return;
      const result = await UI.api(`/api/employees/${id}`, { cache: 'no-store' });
      if (ticket !== employeeVersion) return;
      employee = result;
      $('sample-employee-status').textContent = `${result.name} · образцов ${result.faces_count}/15 · версия ${result.revision}${result.active ? '' : ' · неактивен'}`;
    } catch (error) { if (ticket === employeeVersion) $('sample-employee-status').textContent = errorText(error) + ' Нажмите «Обновить версию сотрудника».'; }
    finally { if (ticket === employeeVersion) { employeeLoading = false; sync(); } }
  }
  function paint() {
    const image = $('sample-image'), box = $('sample-selection'), canvas = $('sample-preview');
    box.toggleAttribute('hidden', !selection);
    if (selection) {
      const [x1, y1, x2, y2] = selection;
      for (const [key, value] of Object.entries({ x: x1, y: y1, width: x2 - x1, height: y2 - y1 })) box.setAttribute(key, value);
    }
    const native = digests.photo && image.complete && crop.nativeCrop(selection, image.naturalWidth, image.naturalHeight);
    canvas.hidden = !native;
    if (native) {
      const scale = Math.min(1, 256 / Math.max(native.width, native.height));
      canvas.width = Math.max(1, Math.round(native.width * scale)); canvas.height = Math.max(1, Math.round(native.height * scale));
      canvas.getContext('2d').drawImage(image, native.x, native.y, native.width, native.height, 0, 0, canvas.width, canvas.height);
      $('sample-crop-info').textContent = `${native.width} × ${native.height} исходных пикселей`;
    } else $('sample-crop-info').textContent = 'Нет доступного снимка или выделенного фрагмента.';
    sync();
  }
  function sourceChanged() {
    $('sample-confirm').checked = false;
    const source = $('sample-source').value;
    $('sample-crop').hidden = source !== 'PHOTO_CROP'; $('sample-upload').hidden = source !== 'UPLOAD';
    bounds.forEach((input) => { input.disabled = source !== 'PHOTO_CROP'; });
    $('sample-file').disabled = source !== 'UPLOAD';
    const ready = sourceReady();
    $('sample-eligibility').textContent = !visit?.enrollment_available ? errors.FACE_MODEL_UNAVAILABLE
      : source === 'UPLOAD' ? 'Выберите JPEG или PNG с одним чётким лицом. Файл проверяется сервером.'
      : ready ? 'Проверьте показанное изображение и выберите сотрудника вручную.'
        : source === 'PHOTO_CROP' ? 'Снимок недоступен. Обновите наблюдение или загрузите фотографию.'
          : errors[visit?.employee_sample_reason] || 'Подходящее сохранённое лицо недоступно. Выберите фрагмент снимка или загрузите фотографию.';
    paint();
  }
  async function loadUpload() {
    const ticket = ++uploadVersion;
    uploadFile = null; $('sample-confirm').checked = false;
    if (uploadURL) URL.revokeObjectURL(uploadURL);
    uploadURL = null; $('sample-upload-preview').hidden = true; sync();
    const file = $('sample-file').files[0];
    if (!file) return;
    if (file.size > 15 * 1024 * 1024 || !['image/jpeg', 'image/png'].includes(file.type)) { message('sample', 'Нужен JPEG или PNG размером до 15 МиБ.', true); return; }
    try {
      const blob = new Blob([await file.arrayBuffer()], { type: file.type });
      if (ticket !== uploadVersion) return;
      const url = URL.createObjectURL(blob); uploadURL = url;
      const decoded = new Image(); decoded.src = url; await decoded.decode();
      if (ticket !== uploadVersion) return;
      if (decoded.naturalWidth * decoded.naturalHeight > 20000000) throw new Error('Изображение больше 20 мегапикселей. Выберите меньший исходник.');
      $('sample-upload-preview').src = url; $('sample-upload-preview').hidden = false;
      uploadFile = Object.freeze({ blob, name: file.name });
      message('sample', 'Фотография загружена для проверки. Подтвердите сотрудника.');
    } catch (error) { if (ticket === uploadVersion) message('sample', 'Не удалось открыть фотографию. ' + error.message, true); }
    if (ticket === uploadVersion) sync();
  }
  function historyRow(target, text, action, handler) {
    const row = document.createElement('li'), description = document.createElement('p'); description.textContent = text; row.append(description);
    if (handler) { const button = document.createElement('button'); button.type = 'button'; button.className = 'btn'; button.textContent = action; button.onclick = handler; row.append(button); }
    target.append(row);
  }
  async function loadHistory() {
    const ticket = ++historyVersion;
    await Promise.all(['identity', 'sample', 'time'].map(async (name) => {
      const target = $(name + '-history');
      try {
        const path = name === 'sample' ? '/employee-samples' : `/${name}/history`;
        const result = await UI.api(endpoint + path, { cache: 'no-store' });
        if (ticket !== historyVersion) return;
        target.replaceChildren();
        for (const row of result.items) {
          const auditText = `${stamp(row.created_at)} · ${row.actor || 'Оператор не указан'} · ${row.reason || 'Причина не указана'}`;
          if (name === 'identity') historyRow(target, `Версия ${row.revision}: ${row.employee_name || (row.mode === 'AI' ? 'Результат AI' : 'Неизвестный')} · ${auditText}`, 'Восстановить эту версию', () => {
            if (blocked(name)) return;
            const human = audit(name); if (human) mutate(name, jsonRequest('/identity/restore', { ...human, expected_revision: visit.identity_revision, restore_revision: row.revision }));
          });
          else if (name === 'sample') historyRow(target, `${row.employee_name} · ${row.source} · ${row.present === null ? 'Старый образец: наличие не подтверждено, замена через карточку сотрудника' : row.present ? 'В галерее' : 'Удалён'} · ${auditText}${row.removal ? ` · Удаление: ${stamp(row.removal.created_at)} · ${row.removal.actor} · ${row.removal.reason}` : ''}`, 'Удалить этот образец', row.present && row.removable ? async () => {
            if (blocked(name)) return;
            const human = audit(name); if (!human) return;
            states.sample.busy = true; sync();
            let current;
            try { current = await UI.api(`/api/employees/${row.employee_id}`, { cache: 'no-store' }); }
            catch (error) { message(name, errorText(error), true); }
            states.sample.busy = false;
            if (current) await mutate(name, jsonRequest('/employee-sample/remove', { ...human, sample_request_id: row.sample_request_id, expected_revision: current.revision }));
            sync();
          } : null);
          else {
            const revoked = row.created.every((item) => item.revoked_at), unavailable = row.created.some((item) => item.unavailable);
            historyRow(target, `${stamp(row.selection.started_at)} — ${stamp(row.selection.ended_at)} · ${revoked ? 'Отменено' : unavailable ? 'Записи изменены' : 'Применено'} · ${auditText}`, 'Отменить эти исправления часов', !revoked && !unavailable ? () => {
              if (blocked(name)) return;
              // Reveal audit fields without enabling a new time application.
              $('time-enabled').checked = true; $('time-form').hidden = false;
              const human = audit(name); if (human) mutate(name, jsonRequest('/time/undo', { ...human, operation_id: row.operation_id }));
            } : null);
          }
        }
        if (!result.items.length) target.textContent = 'Операций пока нет.';
      } catch (error) { if (ticket === historyVersion) { target.replaceChildren(); historyRow(target, 'История не загружена. ' + errorText(error), 'Повторить загрузку истории', loadHistory); } }
    }));
    sync();
  }
  function hours(ms) {
    const seconds = Math.floor(Math.abs(ms) / 1000);
    const fraction = Math.abs(ms) % 1000 ? '.' + String(Math.abs(ms) % 1000).padStart(3, '0') : '';
    return `${ms < 0 ? '−' : ''}${Math.floor(seconds / 3600)}:${String(Math.floor(seconds / 60) % 60).padStart(2, '0')}:${String(seconds % 60).padStart(2, '0')}${fraction}`;
  }
  $('identity-form').onsubmit = (event) => {
    event.preventDefault(); if (states.identity.pending) return mutate('identity');
    const human = audit('identity'); if (!human || !visit) return;
    const mode = $('identity-mode').value;
    if (mode === 'EMPLOYEE' && !$('identity-employee').value) return;
    mutate('identity', jsonRequest('/identity', { ...human, expected_revision: visit.identity_revision, mode, employee_id: mode === 'EMPLOYEE' ? Number($('identity-employee').value) : null }));
  };
  $('sample-form').onsubmit = (event) => {
    event.preventDefault(); if (states.sample.pending) return mutate('sample');
    const human = audit('sample'); if (!human || $('sample-submit').disabled) return;
    const source = $('sample-source').value;
    const payload = { ...human, employee_id: employee.id, expected_revision: employee.revision };
    if (source === 'ALIGNED_FACE') return mutate('sample', jsonRequest('/employee-sample', { ...payload, face_sha256: digests.face }));
    Object.assign(payload, { source });
    if (source === 'PHOTO_CROP') Object.assign(payload, { crop: [...selection], photo_sha256: digests.photo });
    const body = new FormData(); body.append('metadata', JSON.stringify(payload));
    if (source === 'UPLOAD') body.append('file', uploadFile.blob, uploadFile.name);
    mutate('sample', { url: endpoint + '/employee-photo', options: { body } });
  };
  $('time-form').onsubmit = async (event) => {
    event.preventDefault(); if (blocked('time') || previewBusy || !visit?.employee_id) return;
    const human = audit('time'); if (!human) return;
    invalidatePreview(); const ticket = previewVersion;
    const payload = { ...human, expected_identity_revision: visit.identity_revision, employee_id: visit.employee_id,
      started_at: $('time-start').value + offset, ended_at: $('time-end').value + offset,
      exclude_employee_id: $('time-exclude').value ? Number($('time-exclude').value) : null };
    previewBusy = true; sync(); message('time', 'Рассчитываем реальные дневные итоги…');
    try {
      const result = await UI.api(endpoint + '/time/preview', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
      if (ticket !== previewVersion) return;
      preview = { ...payload, ...result.selection, preview_sha256: result.preview_sha256 };
      $('time-rows').replaceChildren();
      for (const row of result.rows) {
        const tr = document.createElement('tr');
        for (const value of [`${row.employee_name} · ${row.date}`, row.method, hours(row.before_ms), hours(row.after_ms), hours(row.delta_ms)]) {
          const td = document.createElement('td'); td.textContent = value; tr.append(td);
        }
        $('time-rows').append(tr);
      }
      $('time-result').hidden = false; message('time', 'Проверьте итоги. Часы ещё не изменены.');
    } catch (error) { if (ticket === previewVersion) message('time', errorText(error), true); }
    finally { previewBusy = false; sync(); }
  };
  $('time-apply').onclick = () => {
    if (states.time.pending) return mutate('time');
    if (preview && $('time-confirm').checked) mutate('time', jsonRequest('/time/apply', preview));
  };
  $('time-fields').addEventListener('input', () => { invalidatePreview(); sync(); });
  $('time-confirm').onchange = sync;
  $('time-enabled').onchange = () => { $('time-form').hidden = !$('time-enabled').checked; invalidatePreview(); sync(); };
  $('identity-mode').onchange = sync; $('identity-employee').onchange = sync;
  for (const name of ['identity', 'sample']) $(name + '-search').oninput = () => { optionsFor(name); sync(); };
  for (const name of ['identity', 'sample']) $(name + '-search').onkeydown = (event) => {
    if (event.key === 'Enter') event.preventDefault();
  };
  $('employees-retry').onclick = loadEmployees;
  $('sample-employee').onchange = loadEmployee; $('sample-reconcile').onclick = loadEmployee;
  $('sample-confirm').onchange = sync; $('sample-source').onchange = sourceChanged; $('sample-file').onchange = loadUpload;
  bounds.forEach((input) => input.addEventListener('input', () => {
    selection = bounds.every((node) => node.value !== '') ? crop.normalizeRect(...bounds.map((node) => Number(node.value))) : null;
    $('sample-confirm').checked = false; paint();
  }));
  const stage = $('sample-stage');
  stage.onpointerdown = (event) => {
    if (blocked('sample') || !digests.photo || event.button !== 0) return;
    event.preventDefault(); drag = crop.pointInImage(event.clientX, event.clientY, $('sample-image').getBoundingClientRect());
    stage.setPointerCapture(event.pointerId); $('sample-confirm').checked = false;
  };
  function move(event) {
    if (!drag || blocked('sample')) return;
    selection = crop.normalizeRect(...drag, ...crop.pointInImage(event.clientX, event.clientY, $('sample-image').getBoundingClientRect()));
    bounds.forEach((input, index) => { input.value = selection ? String(selection[index]) : ''; }); paint();
  }
  stage.onpointermove = move;
  stage.onpointerup = (event) => { move(event); drag = null; if (stage.hasPointerCapture(event.pointerId)) stage.releasePointerCapture(event.pointerId); };
  stage.onpointercancel = () => { drag = null; };
  $('sample-image').onload = paint;
  $('time-retry').onclick = () => mutate('time');
  window.JournalCorrections = {
    locked,
    samplePending: () => refreshBusy || occupied('sample'),
    mutateLearning: (request) => mutate('learning', request),
    learningState: () => ({ busy: states.learning.busy, pending: Boolean(states.learning.pending),
      loading: learningBlocked(), analyzing: learningAnalyzing }),
    setLearningAnalyzing(value) { learningAnalyzing = Boolean(value); sync(); },
    setLoading(value) { refreshBusy = value; sync(); },
    update(value, imageDigests, { evidenceOnly = false } = {}) {
      visit = value; digests = { ...imageDigests };
      if (!blocked('sample')) {
        const src = $('visit-photo').getAttribute('src');
        if (digests.photo && src) $('sample-image').src = src;
        else $('sample-image').removeAttribute('src');
        sourceChanged();
      }
      if (!offset && visit.time_bounds) {
        offset = visit.time_bounds.started_at.match(/(Z|[+-]\d{2}:\d{2})$/)[0];
        // HTML datetime-local accepts milliseconds, not Python's six fractional digits.
        $('time-start').value = visit.time_bounds.started_at.slice(0, -offset.length).replace(/(\.\d{3})\d+$/, '$1');
        $('time-end').value = visit.time_bounds.ended_at.slice(0, -offset.length).replace(/(\.\d{3})\d+$/, '$1');
        $('time-zone').textContent = `Время организации: UTC${offset === 'Z' ? '+00:00' : offset}, не часовой пояс браузера. Интервал не расширяется автоматически.`;
      }
      if (!evidenceOnly) invalidatePreview();
      window.JournalLearning?.update(visit, digests, { evidenceOnly });
      sync(); loadHistory();
    },
  };
  window.addEventListener('beforeunload', (event) => { if (locked()) { event.preventDefault(); event.returnValue = ''; } });
  window.addEventListener('pagehide', (event) => { if (!event.persisted && uploadURL) URL.revokeObjectURL(uploadURL); });
  sync(); loadEmployees();
})();
