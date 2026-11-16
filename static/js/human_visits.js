// Camera-local journal. Explicit refresh never replaces a filter draft.
(() => {
  'use strict';
  const $ = (id) => document.getElementById(id);
  const UI = window.OperatorUI;
  const form = $('visits-filters');
  const tabs = ['visits', 'archive'];
  const labels = { PROVISIONAL: 'Предварительное', ACTIVE: 'Активное', CLOSED: 'Завершено',
    UNKNOWN: 'Неизвестный человек', FACE_CONFIRMED: 'Подтверждено лицом', CONFLICT: 'Конфликт личности',
    AVAILABLE: 'сохранено', MISSING: 'нет', EXPIRED: 'срок истёк', PURGED: 'удалено по сроку' };
  let active = '';
  let sequence = 0;
  let loading = false;
  let loaded = false;
  let cursor = null;
  let filters = new URLSearchParams({ limit: '50' });
  const shown = new Set();

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
      if (visit.employee_id && visit.identity_state === 'FACE_CONFIRMED') {
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
        cell(duration(visit.duration_seconds)), status,
        cell(`Человек: ${labels[visit.photo_status] || visit.photo_status}; лицо: ${labels[visit.face_status] || visit.face_status}`));
      body.append(row);
    }
  }
  async function load(more = false) {
    if (loading || (more && !cursor)) return;
    const ticket = ++sequence;
    const params = new URLSearchParams(filters);
    if (more) params.set('cursor', cursor);
    loading = true;
    UI.setBusy($('panel-visits'), true);
    UI.message($('visits-status'), 'Загружаем наблюдения…');
    try {
      const page = await UI.api(`/api/global-persons/visits?${params}`);
      if (ticket !== sequence) return;
      if (!more) { shown.clear(); $('visits-table').querySelector('tbody').replaceChildren(); }
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
      if (ticket === sequence) { loading = false; UI.setBusy($('panel-visits'), false); }
    }
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
    load();
  }
  function selectTab(name, focus = false) {
    active = tabs.includes(name) ? name : 'visits';
    for (const tab of tabs) {
      const selected = active === tab;
      $(`tab-${tab}`).setAttribute('aria-selected', String(selected));
      $(`tab-${tab}`).tabIndex = selected ? 0 : -1;
      $(`panel-${tab}`).hidden = !selected;
    }
    if (focus) $(`tab-${active}`).focus();
    $('summary').textContent = active === 'visits' ? `Наблюдений: ${shown.size} · не рабочие часы` : 'Межкамерный архив · только чтение';
    document.dispatchEvent(new CustomEvent('journal-tabchange', { detail: active }));
    if (active === 'visits' && !loaded) load();
  }
  for (const tab of tabs) {
    $(`tab-${tab}`).addEventListener('click', () => {
      history.replaceState(null, '', `#${tab}`); selectTab(tab);
    });
    $(`tab-${tab}`).addEventListener('keydown', (event) => {
      if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
      event.preventDefault();
      const next = event.key === 'Home' ? 'visits' : event.key === 'End' ? 'archive' : tabs[1 - tabs.indexOf(tab)];
      history.replaceState(null, '', `#${next}`); selectTab(next, true);
    });
  }
  window.addEventListener('hashchange', () => selectTab(location.hash.slice(1)));
  form.addEventListener('submit', (event) => { event.preventDefault(); applyFilters(); });
  form.addEventListener('reset', () => { queueMicrotask(applyFilters); });
  $('visits-refresh').addEventListener('click', () => load());
  $('visits-more').addEventListener('click', () => load(true));

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
  selectTab(location.hash.slice(1));
})();
