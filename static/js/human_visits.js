// Camera-local journal. Explicit refresh never replaces a filter draft.
(() => {
  'use strict';
  const $ = (id) => document.getElementById(id);
  const UI = window.OperatorUI;
  const form = $('visits-filters');
  const labels = { PROVISIONAL: 'Предварительное', ACTIVE: 'Активное', CLOSED: 'Завершено',
    UNKNOWN: 'Неизвестный человек', FACE_CONFIRMED: 'Подтверждено лицом', CONFLICT: 'Конфликт личности',
    OPERATOR_CONFIRMED: 'Подтверждено оператором', OPERATOR_UNKNOWN: 'Неизвестный · решение оператора',
  };
  let active = '';
  let sequence = 0;
  let loading = false;
  let loaded = false;
  let cursor = null;
  let filters = new URLSearchParams({ limit: '50' });
  const shown = new Set();
  const media = new Map(), queued = new Set();
  const maxMediaRequests = 4;
  const expired = 'Срок хранения истёк', missing = 'Снимок не сохранён';
  let inFlight = 0, generation = 0, observer = null, deadlineTimer = null, listRequest = null;

  function releaseImage(entry) {
    if (entry.url) URL.revokeObjectURL(entry.url);
    entry.url = null;
    entry.frame.replaceChildren();
  }

  function expireImage(entry) {
    queued.delete(entry);
    observer?.unobserve(entry.cell);
    entry.controller?.abort();
    entry.state = 'expired';
    releaseImage(entry);
    entry.caption.textContent = expired;
  }

  function checkDeadlines() {
    clearTimeout(deadlineTimer);
    deadlineTimer = null;
    const now = Date.now();
    let next = Infinity;
    for (const entry of media.values()) {
      if (entry.state === 'expired') continue;
      if (entry.deadline <= now) expireImage(entry);
      else next = Math.min(next, entry.deadline);
    }
    if (next !== Infinity) deadlineTimer = setTimeout(checkDeadlines, Math.min(next - now, 2147483647));
  }

  function disposeRows() {
    ++generation;
    observer?.disconnect();
    observer = null;
    clearTimeout(deadlineTimer);
    deadlineTimer = null;
    queued.clear();
    for (const entry of media.values()) {
      entry.controller?.abort();
      releaseImage(entry);
    }
    media.clear();
  }

  function requestButton(entry, message) {
    entry.caption.textContent = message;
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'btn visit-thumbnail-retry';
    button.textContent = message ? 'Повторить загрузку' : 'Загрузить снимок';
    button.addEventListener('click', () => {
      if (entry.generation !== generation || !['idle', 'failed'].includes(entry.state)) return;
      queued.add(entry);
      pumpMedia();
    });
    entry.caption.append(button);
  }

  async function fetchMedia(entry) {
    const controller = new AbortController();
    entry.controller = controller;
    entry.state = 'loading';
    entry.caption.textContent = `${entry.label} · загрузка…`;
    ++inFlight;
    const current = () => entry.generation === generation && !controller.signal.aborted;
    try {
      const response = await fetch(`/api/global-persons/visits/${encodeURIComponent(entry.id)}/${entry.kind}`,
        { cache: 'no-store', signal: controller.signal });
      if (!current()) return;
      if (entry.deadline <= Date.now() || response.status === 410) { expireImage(entry); return; }
      if (response.status === 404) {
        entry.state = 'missing';
        entry.caption.textContent = missing;
        observer?.unobserve(entry.cell);
        return;
      }
      if (!response.ok) throw new Error('THUMBNAIL_REQUEST_FAILED');
      const blob = await response.blob();
      if (!current()) return;
      if (entry.deadline <= Date.now()) { expireImage(entry); return; }
      if (blob.type !== 'image/jpeg') throw new Error('THUMBNAIL_FORMAT_INVALID');
      const image = document.createElement('img');
      image.alt = `${entry.label} в наблюдении ${entry.id.slice(0, 8)}`;
      image.width = 96; image.height = 120;
      const link = document.createElement('a');
      link.href = `/global/visits/${encodeURIComponent(entry.id)}`;
      link.setAttribute('aria-label', `Открыть наблюдение ${entry.id}: ${entry.label}`);
      link.append(image);
      const url = URL.createObjectURL(blob);
      entry.url = url;
      entry.state = 'shown';
      entry.caption.textContent = entry.label;
      observer?.unobserve(entry.cell);
      image.addEventListener('error', () => {
        if (!current() || entry.url !== url) return;
        releaseImage(entry);
        entry.state = 'failed';
        requestButton(entry, 'Не удалось загрузить снимок. ');
      });
      image.src = url;
      entry.frame.replaceChildren(link);
    } catch {
      if (!current()) return;
      if (entry.deadline <= Date.now()) expireImage(entry);
      else {
        entry.state = 'failed';
        requestButton(entry, 'Не удалось загрузить снимок. ');
      }
    } finally {
      entry.controller = null;
      --inFlight;
      pumpMedia();
    }
  }

  function pumpMedia() {
    for (const entry of queued) {
      if (inFlight >= maxMediaRequests) break;
      queued.delete(entry);
      if (entry.generation !== generation || !['idle', 'failed'].includes(entry.state)) continue;
      if (entry.deadline <= Date.now()) expireImage(entry);
      else void fetchMedia(entry);
    }
  }

  function thumbnail(visit) {
    const node = cell('');
    node.className = 'visit-thumbnail';
    const frame = document.createElement('div'), caption = document.createElement('div');
    frame.className = 'visit-thumbnail-frame';
    caption.className = 'visit-thumbnail-status muted';
    node.append(frame, caption);
    const kind = visit.photo_status === 'AVAILABLE' ? 'photo' : visit.face_status === 'AVAILABLE' ? 'face' : null;
    const stamp = Date.parse(visit.image_expires_at);
    const entry = { id: visit.id, kind, label: kind === 'face' ? 'Лицо' : 'Снимок человека',
      deadline: Number.isFinite(stamp) ? stamp : Infinity, generation, state: 'idle',
      cell: node, frame, caption, url: null, controller: null };
    media.set(node, entry);
    if (entry.deadline <= Date.now() || [visit.photo_status, visit.face_status].every(
      status => status === 'EXPIRED' || status === 'PURGED')) expireImage(entry);
    else if (!kind) { entry.state = 'missing'; caption.textContent = missing; }
    else if (typeof IntersectionObserver !== 'function') requestButton(entry, '');
    else {
      caption.textContent = `${entry.label} · ожидает загрузки`;
      if (!observer) {
        const observedGeneration = generation;
        observer = new IntersectionObserver(changes => {
          if (observedGeneration !== generation) return;
          for (const { target, isIntersecting } of changes) {
            const item = media.get(target);
            if (item?.state !== 'idle') continue;
            if (isIntersecting) queued.add(item);
            else queued.delete(item);
          }
          pumpMedia();
        });
      }
      observer.observe(node);
    }
    return node;
  }

  function time(value) { return value ? value.replace('T', ' ').replace(/Z$/, ' UTC') : '—'; }
  function duration(value) {
    const seconds = Math.floor(Math.max(0, value));
    return `${Math.floor(seconds / 60)} мин ${seconds % 60} с`;
  }
  function cell(text) {
    const element = document.createElement('td'); element.textContent = text; return element;
  }
  function render(items) {
    const body = $('visits-table').querySelector('tbody');
    for (const visit of items) {
      if (shown.has(visit.id)) continue;
      shown.add(visit.id);
      const row = document.createElement('tr');
      const id = cell('');
      const link = document.createElement('a');
      link.href = `/global/visits/${encodeURIComponent(visit.id)}`;
      link.textContent = visit.id.slice(0, 8); link.title = visit.id;
      link.setAttribute('aria-label', `Открыть наблюдение ${visit.id}`); id.append(link);
      const identity = cell(labels[visit.identity_state] || visit.identity_state);
      if (visit.employee_id) {
        const employee = document.createElement('a');
        employee.href = `/employees/${visit.employee_id}`;
        employee.textContent = visit.employee_name || `Сотрудник #${visit.employee_id}`;
        identity.append(document.createElement('br'), employee);
      }
      const observed = cell(time(visit.started_at));
      observed.append(document.createElement('br'), document.createTextNode(time(visit.last_seen_at)));
      const status = cell(labels[visit.status] || visit.status);
      if (visit.end_reason) status.append(document.createElement('br'), document.createTextNode(visit.end_reason));
      row.append(id, cell(visit.camera_name || `Камера #${visit.camera_id}`), identity, observed,
        cell(duration(visit.duration_seconds)), status, thumbnail(visit));
      body.append(row);
    }
    checkDeadlines();
  }
  async function load(more = false) {
    if (loading || (more && !cursor)) return;
    const ticket = ++sequence;
    const params = new URLSearchParams(filters);
    if (more) params.set('cursor', cursor);
    const controller = new AbortController();
    listRequest = controller;
    loading = true;
    UI.setBusy($('panel-visits'), true);
    UI.message($('visits-status'), 'Загружаем наблюдения…');
    try {
      const page = await UI.api(`/api/global-persons/visits?${params}`, { signal: controller.signal });
      if (ticket !== sequence) return;
      if (!more) { disposeRows(); shown.clear(); $('visits-table').querySelector('tbody').replaceChildren(); }
      render(page.items); cursor = page.next_cursor; loaded = true;
      $('visits-empty').hidden = shown.size > 0;
      $('visits-more').hidden = !cursor;
      $('visits-count').textContent = `Показано: ${shown.size}`;
      UI.message($('visits-status'), '');
      if (active === 'visits') $('summary').textContent = `Наблюдений: ${shown.size} · не рабочие часы`;
    } catch (error) {
      if (ticket !== sequence) return;
      UI.message($('visits-status'), 'Не удалось загрузить наблюдения. ' + error.message + ' Выбранные фильтры сохранены. Повторите загрузку.', 'error');
    } finally {
      if (ticket === sequence) { loading = false; listRequest = null; UI.setBusy($('panel-visits'), false); }
    }
  }
  function cancelList() {
    ++sequence;
    listRequest?.abort();
    listRequest = null;
    loading = false;
    disposeRows();
  }
  function applyFilters() {
    filters = new URLSearchParams({ limit: '50' });
    for (const name of ['camera_id', 'employee_id']) {
      if (form.elements[name].value) filters.set(name, form.elements[name].value);
    }
    for (const name of ['from', 'to']) {
      // These controls are explicitly UTC, not the browser's local timezone.
      if (form.elements[name].value) filters.set(name, form.elements[name].value + 'Z');
    }
    if (form.elements.unknown_only.checked) filters.set('unknown_only', 'true');
    cancelList();
    shown.clear();
    $('visits-table').querySelector('tbody').replaceChildren();
    cursor = null;
    loaded = false;
    $('visits-more').hidden = true;
    load();
  }
  function tabChanged(name) {
    active = name;
    $('summary').textContent = active === 'visits' ? `Наблюдений: ${shown.size} · не рабочие часы` : 'Межкамерный архив · только чтение';
    document.dispatchEvent(new CustomEvent('journal-tabchange', { detail: active }));
    if (active === 'visits' && !loaded) load();
  }
  form.addEventListener('submit', (event) => { event.preventDefault(); applyFilters(); });
  form.addEventListener('reset', () => { queueMicrotask(applyFilters); });
  $('visits-refresh').addEventListener('click', () => load());
  $('visits-more').addEventListener('click', () => load(true));
  document.addEventListener('visibilitychange', () => { if (!document.hidden) checkDeadlines(); });
  window.addEventListener('pagehide', cancelList);
  window.addEventListener('pageshow', (event) => { if (event.persisted && active === 'visits') load(); });

  async function choices(path, target) {
    try {
      const rows = await UI.api(path);
      for (const row of rows) {
        const option = document.createElement('option'); option.value = row.id;
        option.textContent = row.name || `#${row.id}`; target.append(option);
      }
    } catch (error) {
      UI.message($('visits-filter-status'), 'Не удалось загрузить варианты фильтра. ' + error.message + ' Обновите страницу, чтобы повторить.', 'error');
    }
  }
  choices('/api/cameras', $('visits-camera'));
  choices('/api/employees', $('visits-employee'));
  UI.tabs(document.querySelector('[role="tablist"]'), { onChange: tabChanged });
})();
