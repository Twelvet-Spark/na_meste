// The displayed face and append receipt stay frozen across uncertain retries.
(() => {
  'use strict';
  const $ = (id) => document.getElementById(id);
  const UI = window.OperatorUI;
  const root = $('visit-detail');
  if (!root) return;
  const endpoint = `/api/global-persons/visits/${encodeURIComponent(root.dataset.visitId)}`;
  const form = $('sample-form');
  const select = $('sample-employee');
  const labels = {
    PROVISIONAL: 'Предварительное', ACTIVE: 'Активное', CLOSED: 'Завершено',
    UNKNOWN: 'Неизвестный человек', FACE_CONFIRMED: 'Подтверждено лицом', CONFLICT: 'Конфликт личности',
    AVAILABLE: 'Сохранено', MISSING: 'Изображение не было сохранено',
    EXPIRED: 'Срок хранения истёк. Данные наблюдения сохранены.',
    PURGED: 'Изображение удалено по сроку хранения. Данные наблюдения сохранены.',
    FIRST_SEEN: 'Первое наблюдение', IDENTIFIED: 'Личность подтверждена лицом',
    IDENTITY_CONFLICT: 'Конфликт личности', LOST: 'Трек потерян', RECOVERED: 'Трек восстановлен',
    LINKED: 'Межкамерная связь по внешности',
  };
  const reasons = {
    FACE_MODEL_UNAVAILABLE: 'Модель лиц недоступна. Используйте обычное добавление фотографий после восстановления модели.',
    VISIT_FACE_MISSING: 'Нет сохранённого подходящего лица. Добавьте обычную фотографию сотрудника.',
    VISIT_FACE_EXPIRED: 'Срок хранения лица истёк. Добавьте обычную фотографию сотрудника.',
    VISIT_FACE_INVALID: 'Сохранённое лицо не подходит для образца. Добавьте обычную фотографию сотрудника.',
    VISIT_FACE_QUALITY: 'Недостаточное качество лица. Нужны исходный размер не менее 80 px, резкость и однозначная связь с человеком.',
    FACE_LIMIT_REACHED: 'У сотрудника уже 15 образцов. Управляйте набором фотографий в карточке сотрудника.',
    ENROLLMENT_BUSY: 'Другая обработка фотографий ещё выполняется. Подождите и повторите тот же запрос.',
    GALLERY_CHANGED: 'Галерея изменилась во время проверки. Загрузите текущую версию сотрудника и проверьте выбор.',
  };
  let visit = null;
  let employee = null;
  let faceDigest = null;
  let pending = null;
  let busy = false;
  let loading = false;
  let employeeLoading = false;
  let employeesReady = false;
  let completed = false;
  let needsReview = false;
  let expiryTimer = null;
  let version = 0;
  let employeeVersion = 0;
  const urls = new Map();

  function time(value) { return value ? value.replace('T', ' ').replace(/Z$/, ' UTC') : '—'; }
  function duration(value) {
    const seconds = Math.floor(Math.max(0, value));
    return `${Math.floor(seconds / 60)} мин ${seconds % 60} с`;
  }
  function facts(target, values) {
    target.replaceChildren();
    for (const [name, value] of values) {
      const term = document.createElement('dt'); term.textContent = name;
      const description = document.createElement('dd');
      if (value instanceof Node) description.append(value);
      else description.textContent = value ?? '—';
      target.append(term, description);
    }
  }
  function link(href, text) {
    const element = document.createElement('a'); element.href = href; element.textContent = text; return element;
  }
  function controls() {
    const locked = busy || loading;
    const eligible = visit?.employee_sample_eligible && faceDigest === visit?.face_sha256 && !needsReview;
    $('sample-fields').disabled = locked || !employeesReady || (!eligible && !pending);
    select.disabled = Boolean(pending) || employeeLoading;
    $('sample-confirm').disabled = Boolean(pending) || employeeLoading;
    $('sample-submit').disabled = locked || employeeLoading || completed ||
      (!pending && (!eligible || !employee || employee.faces_count >= 15 || !$('sample-confirm').checked));
    $('sample-submit').textContent = pending ? 'Повторить тот же запрос' : 'Добавить проверенный образец';
    $('visit-refresh').disabled = locked || Boolean(pending);
    $('sample-reconcile').disabled = locked || employeeLoading || Boolean(pending);
    $('sample-retry-employees').disabled = locked || Boolean(pending);
    root.setAttribute('aria-busy', String(loading));
    form.setAttribute('aria-busy', String(busy || employeeLoading));
  }
  function renderVisit() {
    $('visit-state').textContent = labels[visit.status] || visit.status;
    $('visit-state').className = 'status ' + (visit.status === 'ACTIVE' ? 'online' : visit.status === 'PROVISIONAL' ? 'connecting' : 'offline');
    const identity = visit.employee_id && visit.identity_state === 'FACE_CONFIRMED'
      ? link(`/employees/${visit.employee_id}`, visit.employee_name || `Сотрудник #${visit.employee_id}`)
      : labels[visit.identity_state] || visit.identity_state;
    const global = visit.global_id ? link(`/global/${visit.global_id}`, `G#${visit.global_id} · связь по внешности`) : 'Нет межкамерной связи';
    facts($('visit-facts'), [
      ['Анонимный ID наблюдения', visit.id], ['Камера', visit.camera_name || `Камера #${visit.camera_id}`],
      ['Первое наблюдение', time(visit.started_at)], ['Последнее наблюдение', time(visit.last_seen_at)],
      ['Длительность наблюдений', duration(visit.duration_seconds)], ['Состояние', labels[visit.status] || visit.status],
      ['Личность', identity], ['Завершено', time(visit.ended_at)], ['Причина завершения', visit.end_reason || '—'],
      ['Межкамерный архив', global], ['Поколение трекера / трек', `${visit.tracking_generation} / ${visit.track_id}`],
      ['Версия источника / поток', `${visit.source_revision} / ${visit.stream_id}`],
    ]);
    const quality = visit.face_quality || {};
    facts($('visit-image-facts'), [
      ['Изображения доступны до', time(visit.image_expires_at)], ['Удалены по сроку', time(visit.images_purged_at)],
      ['SHA256 показанного лица', faceDigest || 'Подходящее лицо не загружено'],
      ['Исходный размер лица', quality.min_size != null ? `${quality.min_size} px` : '—'],
      ['Оценка детектора / резкость', `${quality.det_score ?? '—'} / ${quality.blur ?? '—'}`],
      ['Связь лица с человеком', quality.unambiguous === true ? 'Однозначная' : 'Не подтверждена'],
    ]);
    const explanation = visit.employee_sample_eligible
      ? 'После загрузки лица выберите сотрудника и подтвердите его вручную.'
      : reasons[visit.employee_sample_reason] || 'Этот снимок не подходит для добавления. Используйте обычную загрузку фотографий в карточке сотрудника.';
    UI.message($('sample-eligibility'), explanation);
  }
  function clearImage(kind) {
    const image = $(`visit-${kind}`); image.hidden = true; image.removeAttribute('src');
    if (urls.has(kind)) { URL.revokeObjectURL(urls.get(kind)); urls.delete(kind); }
  }
  async function loadImage(kind, ticket) {
    clearImage(kind);
    const state = visit[`${kind}_status`];
    UI.message($(`visit-${kind}-status`), labels[state] || state);
    if (state !== 'AVAILABLE') return;
    try {
      const response = await fetch(`${endpoint}/${kind}`, { cache: 'no-store' });
      if (!response.ok) {
        const error = new Error(response.status === 410 ? labels.EXPIRED : response.status === 404 ? labels.MISSING : 'Не удалось загрузить изображение. Обновите наблюдение.');
        throw error;
      }
      const digest = response.headers.get('ETag')?.replace(/^"|"$/g, '');
      if (kind === 'face' && digest !== visit.face_sha256) {
        throw new Error('Лицо изменилось после загрузки данных. Обновите наблюдение и проверьте новый снимок.');
      }
      const blob = await response.blob();
      if (ticket !== version) return;
      const url = URL.createObjectURL(blob); urls.set(kind, url);
      const image = $(`visit-${kind}`); image.src = url;
      await image.decode();
      if (ticket !== version) return;
      image.hidden = false;
      if (kind === 'face') faceDigest = digest;
      UI.message($(`visit-${kind}-status`), 'Сохранённый снимок · ' + time(visit.image_expires_at) + ' — срок хранения');
    } catch (error) {
      if (ticket !== version) return;
      clearImage(kind);
      if (kind === 'face') faceDigest = null;
      UI.message($(`visit-${kind}-status`), error.message, 'error');
    }
  }
  async function loadEvents(ticket) {
    UI.message($('visit-events-status'), 'Загружаем события…');
    try {
      const events = await UI.api(`${endpoint}/events?limit=1000`);
      if (ticket !== version) return;
      $('visit-events').replaceChildren();
      for (const event of events) {
        const row = document.createElement('li');
        const payload = event.payload || {};
        row.textContent = [time(event.occurred_at), labels[event.type] || event.type,
          payload.reason, payload.global_id ? `G#${payload.global_id}` : '',
          payload.employee_id ? `Сотрудник #${payload.employee_id}` : ''].filter(Boolean).join(' · ');
        $('visit-events').append(row);
      }
      UI.message($('visit-events-status'), events.length ? (events.length === 1000 ? 'Показаны последние 1000 событий.' : '') : 'Событий пока нет.');
    } catch (error) {
      if (ticket === version) UI.message($('visit-events-status'), 'События не загружены. ' + error.message + ' Обновите наблюдение для повторной загрузки.', 'error');
    }
  }
  function scheduleExpiry() {
    clearTimeout(expiryTimer);
    const remaining = Date.parse(visit.image_expires_at) - Date.now();
    if (remaining > 0) {
      expiryTimer = setTimeout(scheduleExpiry, Math.min(remaining, 2147483647));
      return;
    }
    faceDigest = null; needsReview = true;
    visit.employee_sample_eligible = false;
    visit.employee_sample_reason = 'VISIT_FACE_EXPIRED';
    for (const kind of ['photo', 'face']) {
      clearImage(kind); UI.message($(`visit-${kind}-status`), labels.EXPIRED);
    }
    renderVisit(); controls();
  }

  async function loadVisit() {
    if (busy || loading || pending) return;
    const ticket = ++version;
    clearTimeout(expiryTimer);
    loading = true; faceDigest = null; controls();
    UI.message($('visit-status'), 'Загружаем наблюдение…');
    try {
      const result = await UI.api(endpoint, { cache: 'no-store' });
      if (ticket !== version) return;
      visit = result; completed = false; needsReview = false;
      $('sample-confirm').checked = false;
      renderVisit();
      await Promise.all([loadImage('photo', ticket), loadImage('face', ticket), loadEvents(ticket)]);
      if (ticket !== version) return;
      renderVisit();
      scheduleExpiry();
      if (visit.employee_sample_eligible && faceDigest) UI.message($('sample-eligibility'), 'Показанное лицо загружено. Выберите сотрудника и подтвердите его вручную.');
      UI.message($('visit-status'), '');
    } catch (error) {
      if (ticket === version) UI.message($('visit-status'), error.status === 404
        ? 'Наблюдение не найдено. Вернитесь к журналу.' : 'Не удалось загрузить наблюдение. ' + error.message + ' Повторите загрузку.', 'error');
    } finally { if (ticket === version) { loading = false; controls(); } }
  }
  async function loadEmployee() {
    if (pending || busy) return;
    const ticket = ++employeeVersion;
    employee = null; completed = false;
    $('sample-confirm').checked = false;
    $('sample-reconcile').hidden = true;
    $('sample-enrollment').href = select.value ? `/employees/${select.value}#photos` : '/employees';
    if (!select.value) { UI.message($('sample-employee-status'), ''); controls(); return; }
    employeeLoading = true; controls();
    UI.message($('sample-employee-status'), 'Загружаем текущую версию сотрудника…');
    try {
      const result = await UI.api(`/api/employees/${select.value}`, { cache: 'no-store' });
      if (ticket !== employeeVersion) return;
      employee = result;
      UI.message($('sample-employee-status'), `${employee.name}. Сохранено образцов: ${employee.faces_count}/15. Версия: ${employee.revision}.` + (employee.faces_count >= 15 ? ' Набор заполнен — откройте карточку сотрудника.' : ''));
    } catch (error) {
      if (ticket !== employeeVersion) return;
      UI.message($('sample-employee-status'), 'Версия сотрудника не загружена. ' + error.message, 'error');
      $('sample-reconcile').hidden = false;
    } finally { if (ticket === employeeVersion) { employeeLoading = false; controls(); } }
  }
  async function loadEmployees() {
    if (pending || busy) return;
    $('sample-retry-employees').hidden = true;
    try {
      const rows = await UI.api('/api/employees');
      const chosen = select.value;
      select.replaceChildren(new Option('Выберите сотрудника вручную', ''));
      for (const row of rows) select.append(new Option(row.name + (row.active ? '' : ' (неактивен)'), String(row.id)));
      select.value = chosen; employeesReady = true;
    } catch (error) {
      UI.message($('sample-status'), 'Список сотрудников не загружен. ' + error.message, 'error');
      $('sample-retry-employees').hidden = false;
    } finally { controls(); }
  }
  form.addEventListener('submit', async (event) => {
    event.preventDefault();
    if (busy || loading || employeeLoading || completed) return;
    if (!pending) {
      if (!visit?.employee_sample_eligible || needsReview || !employee || !faceDigest || faceDigest !== visit.face_sha256 || !$('sample-confirm').checked || employee.faces_count >= 15) return;
      pending = Object.freeze({ employee_id: employee.id, expected_revision: employee.revision,
        request_id: UI.newRequestId(), face_sha256: faceDigest });
    }
    busy = true; controls();
    UI.message($('sample-status'), 'Проверяем и добавляем образец… Не закрывайте страницу до подтверждения.');
    try {
      employee = await UI.api(`${endpoint}/employee-sample`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(pending) });
      UI.message($('sample-employee-status'), `Результат операции: сохранено образцов ${employee.faces_count}/15. Версия: ${employee.revision}.`);
      pending = null; completed = true; $('sample-confirm').checked = false;
      UI.message($('sample-status'), 'Проверенный образец добавлен. Все прежние образцы сохранены. Откройте карточку сотрудника для просмотра набора.');
    } catch (error) {
      const code = error.detail?.code;
      if (!error.status || error.status < 400 || error.status === 408 || error.status >= 500) {
        UI.message($('sample-status'), (reasons[code] || error.message) + ' Результат не подтверждён. Выбор, снимок и номер запроса сохранены в этой странице. Повторите тот же запрос, не меняя сотрудника и не обновляя страницу.', 'error');
      } else {
        pending = null; $('sample-confirm').checked = false;
        if (code === 'VISIT_FACE_CHANGED' || error.status === 410) {
          needsReview = true;
          UI.message($('sample-status'), code === 'VISIT_FACE_CHANGED'
            ? 'Сохранённое лицо изменилось. Обновите наблюдение и вручную проверьте новый образец. Ничего не добавлено.'
            : 'Срок хранения лица истёк. Ничего не добавлено. Используйте обычную загрузку фотографии.', 'error');
        } else {
          if (error.status === 409) {
            employee = null;
            $('sample-reconcile').hidden = false;
          }
          UI.message($('sample-status'), (reasons[code] || error.message) + ' Ничего не добавлено. Выбранный сотрудник сохранён; проверьте текущую версию перед новой попыткой.', 'error');
        }
      }
    } finally { busy = false; controls(); }
  });
  select.addEventListener('change', loadEmployee);
  $('sample-confirm').addEventListener('change', controls);
  $('sample-reconcile').addEventListener('click', loadEmployee);
  $('sample-retry-employees').addEventListener('click', loadEmployees);
  $('visit-refresh').addEventListener('click', loadVisit);
  window.addEventListener('beforeunload', (event) => {
    if (pending || busy) { event.preventDefault(); event.returnValue = ''; }
  });
  window.addEventListener('pagehide', (event) => {
    if (!event.persisted) {
      clearTimeout(expiryTimer);
      for (const url of urls.values()) URL.revokeObjectURL(url);
    }
  });
  loadVisit(); loadEmployees();
})();
