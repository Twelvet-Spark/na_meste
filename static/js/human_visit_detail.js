// Read-only evidence stays separate from independently retriable corrections.
(() => {
  'use strict';
  const $ = (id) => document.getElementById(id);
  const UI = window.OperatorUI;
  const root = $('visit-detail');
  if (!root) return;
  const endpoint = `/api/global-persons/visits/${encodeURIComponent(root.dataset.visitId)}`;
  const labels = {
    PROVISIONAL: 'Предварительное', ACTIVE: 'Активное', CLOSED: 'Завершено',
    UNKNOWN: 'Неизвестный человек', FACE_CONFIRMED: 'Подтверждено лицом', CONFLICT: 'Конфликт личности',
    OPERATOR_CONFIRMED: 'Подтверждено оператором', OPERATOR_UNKNOWN: 'Неизвестный · решение оператора',
    AVAILABLE: 'Сохранено', MISSING: 'Изображение не было сохранено',
    EXPIRED: 'Срок хранения истёк. Данные наблюдения сохранены.',
    PURGED: 'Изображение удалено по сроку хранения. Данные наблюдения сохранены.',
    FIRST_SEEN: 'Первое наблюдение', IDENTIFIED: 'Личность подтверждена лицом',
    IDENTITY_CONFLICT: 'Конфликт личности', LOST: 'Трек потерян', RECOVERED: 'Трек восстановлен',
    LINKED: 'Межкамерная связь по внешности',
  };
  let visit = null;
  let faceDigest = null;
  let expiryTimer = null;
  let version = 0;
  const urls = new Map();
  let loading = false;
  const digests = {};
  const corrections = window.JournalCorrections;

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
    $('visit-refresh').disabled = loading || corrections.locked();
    root.setAttribute('aria-busy', String(loading));
  }
  function renderVisit() {
    $('visit-state').textContent = labels[visit.status] || visit.status;
    $('visit-state').className = 'status ' + (visit.status === 'ACTIVE' ? 'online' : visit.status === 'PROVISIONAL' ? 'connecting' : 'offline');
    const identity = visit.employee_id
      ? link(`/employees/${visit.employee_id}`, visit.employee_name || `Сотрудник #${visit.employee_id}`)
      : labels[visit.identity_state] || visit.identity_state;
    const global = visit.global_id ? link(`/global/${visit.global_id}`, `G#${visit.global_id} · связь по внешности`) : 'Нет межкамерной связи';
    facts($('visit-facts'), [
      ['Анонимный ID наблюдения', visit.id], ['Камера', visit.camera_name || `Камера #${visit.camera_id}`],
      ['Первое наблюдение', time(visit.started_at)], ['Последнее наблюдение', time(visit.last_seen_at)],
      ['Длительность наблюдений', duration(visit.duration_seconds)], ['Состояние', labels[visit.status] || visit.status],
      ['Личность', identity], ['Завершено', time(visit.ended_at)], ['Причина завершения', visit.end_reason || '—'],
      ['Источник личности', visit.identity_source === 'OPERATOR' ? 'Оператор' : 'AI'],
      ['Исходное решение AI', `${visit.ai_employee_name || 'Без сотрудника'} · ${labels[visit.ai_identity_state] || visit.ai_identity_state}`],
      ['Версия исправления', visit.identity_revision],
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
  }
  function clearImage(kind) {
    const image = $(`visit-${kind}`); image.hidden = true; image.removeAttribute('src');
    if (urls.has(kind)) { URL.revokeObjectURL(urls.get(kind)); urls.delete(kind); }
    delete digests[kind];
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
      if (digest !== visit[`${kind}_sha256`]) {
        throw new Error('Изображение изменилось после загрузки данных. Обновите наблюдение и проверьте новый снимок.');
      }
      const blob = await response.blob();
      if (ticket !== version) return;
      const url = URL.createObjectURL(blob); urls.set(kind, url);
      const image = $(`visit-${kind}`); image.src = url;
      await image.decode();
      if (ticket !== version) return;
      image.hidden = false;
      digests[kind] = digest;
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
    // Keep the exact already-displayed evidence only while its mutation is unresolved.
    if (corrections.samplePending()) {
      expiryTimer = setTimeout(scheduleExpiry, 1000);
      return;
    }
    faceDigest = null;
    visit.employee_sample_eligible = false;
    visit.employee_sample_reason = 'VISIT_FACE_EXPIRED';
    for (const kind of ['photo', 'face']) {
      clearImage(kind); UI.message($(`visit-${kind}-status`), labels.EXPIRED);
    }
    renderVisit(); corrections.update(visit, digests, { evidenceOnly: true }); controls();
  }

  async function loadVisit() {
    if (loading || corrections.locked()) return;
    const ticket = ++version;
    clearTimeout(expiryTimer);
    loading = true; faceDigest = null; corrections.setLoading(true); controls();
    UI.message($('visit-status'), 'Загружаем наблюдение…');
    try {
      const result = await UI.api(endpoint, { cache: 'no-store' });
      if (ticket !== version) return;
      visit = result;
      renderVisit();
      await Promise.all([loadImage('photo', ticket), loadImage('face', ticket), loadEvents(ticket)]);
      if (ticket !== version) return;
      renderVisit();
      scheduleExpiry();
      UI.message($('visit-status'), '');
    } catch (error) {
      if (ticket === version) UI.message($('visit-status'), error.status === 404
        ? 'Наблюдение не найдено. Вернитесь к журналу.' : 'Не удалось загрузить наблюдение. ' + error.message + ' Повторите загрузку.', 'error');
    } finally {
      if (ticket === version) {
        loading = false; corrections.setLoading(false);
        if (visit) corrections.update(visit, digests);
        controls();
      }
    }
  }
  root.addEventListener('journal-identity', (event) => {
    Object.assign(visit, event.detail); renderVisit();
  });
  root.addEventListener('journal-lock', controls);
  $('visit-refresh').addEventListener('click', loadVisit);
  window.addEventListener('pagehide', (event) => {
    if (!event.persisted) {
      clearTimeout(expiryTimer);
      for (const url of urls.values()) URL.revokeObjectURL(url);
    }
  });
  loadVisit();
})();
