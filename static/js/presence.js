// Work time from every camera with «Учёт времени». All times are shown in the workplace
// offset the server reports; the browser's own time zone is never used for display.
'use strict';
const $ = (id) => document.getElementById(id);
const UI = window.OperatorUI;
const api = UI.api;
const state = { tab: null, employees: [], status: null, reportLoaded: false, warnings: '', activeLoad: 0, reportLoad: 0, correctionsLoad: 0 };
const sourceLabels = { FACE: 'по лицу', REID: 'по внешности', MANUAL: 'вручную', DOOR: 'проход' };
const directionLabels = { ENTRY: 'Вход', EXIT: 'Выход' };

function el(tag, text, className) { const node = document.createElement(tag); if (text != null) node.textContent = String(text); if (className) node.className = className; return node; }
function button(text, action, className = 'btn') { const node = el('button', text, className); node.type = 'button'; node.onclick = action; return node; }
function link(text, href) { const node = el('a', text); node.href = href; return node; }
function row(values) { const tr = el('tr'); for (const value of values) { const td = el('td'); if (value instanceof Node) td.append(value); else td.textContent = value ?? '—'; tr.append(td); } return tr; }
function employeeLink(item) { return link(item.employee_name || `#${item.employee_id}`, `/employees/${item.employee_id}`); }
function showError(id, error) { $(id).hidden = !error; $(id).textContent = error?.message || ''; }

// Server timestamps already carry the workplace offset; slice, never reparse.
const clock = (iso) => iso ? iso.slice(11, 16) : '—';
const stamp = (iso) => iso ? `${iso.slice(0, 10)} ${iso.slice(11, 16)}` : '—';
const span = (start, end) => start.slice(0, 10) === end.slice(0, 10) ? `${stamp(start)}–${clock(end)}` : `${stamp(start)} – ${stamp(end)}`;
function hm(ms) { const minutes = Math.floor(Math.max(0, ms || 0) / 60000); return `${Math.floor(minutes / 60)}:${String(minutes % 60).padStart(2, '0')}`; }
function offset() {
  const minutes = state.status.utc_offset_minutes;
  return `${minutes < 0 ? '-' : '+'}${String(Math.floor(Math.abs(minutes) / 60)).padStart(2, '0')}:${String(Math.abs(minutes) % 60).padStart(2, '0')}`;
}
function today() { return state.status ? state.status.as_of.slice(0, 10) : new Date().toISOString().slice(0, 10); }
function nextDay(date) { const d = new Date(`${date}T00:00:00Z`); d.setUTCDate(d.getUTCDate() + 1); return d.toISOString().slice(0, 10); }
const methodLabel = (method) => (method === 'FINAL' ? 'Итог дня' : 'День идёт');

// ---------- status ----------
function explainer(status) {
  return `Время считается по всем камерам с включённым «Учёт времени»: сотрудник на месте, пока его узнаёт хотя бы одна камера — по лицу или по внешности после узнавания по лицу в этот день. Пока день идёт, перерывы дольше ${status.live_gap_minutes} мин не считаются; после окончания дня считается время от первого до последнего появления за вычетом отсутствий дольше ${status.day_gap_minutes} мин. Выход через дверь и последующий вход не считаются рабочим временем.`;
}
function renderWarnings(status) {
  const items = [];
  const cameras = status.cameras.filter((item) => item.time_tracking);
  if (!cameras.length) items.push(['alert', 'Включите «Учёт времени» на нужных камерах', true]);
  if (!status.worker_alive) items.push(['alert', 'Распознавание не запущено — время не считается']);
  if (!status.appearance_available) items.push(['muted', 'Узнавание по внешности недоступно — считается только узнавание по лицу']);
  if (status.recorder.last_error) items.push(['alert', 'Ошибка записи наблюдений; данные хранятся в памяти и будут записаны повторно']);
  // Rebuild only on change so a polled status does not re-announce an unchanged alert.
  const key = JSON.stringify(items);
  if (key === state.warnings) return;
  state.warnings = key;
  $('attendance-warnings').replaceChildren(...items.map(([kind, text, withLink]) => {
    const node = el('p', text, kind === 'alert' ? 'test-result err' : 'muted');
    if (kind === 'alert') node.setAttribute('role', 'alert');
    if (withLink) { node.append(' '); node.append(link('Настроить камеры', '/')); }
    return node;
  }));
}
async function loadStatus() {
  try {
    const status = await api('/api/attendance/status');
    state.status = status;
    const cameras = status.cameras.filter((item) => item.time_tracking);
    const online = cameras.filter((item) => item.status === 'ONLINE').length;
    $('work-explainer').textContent = explainer(status);
    $('time-status').textContent = `Камер с учётом времени: ${cameras.length}, онлайн: ${online}`;
    $('workplace-zone').textContent = `UTC${offset()}`;
    document.querySelectorAll('.zone-label').forEach((node) => { node.textContent = `UTC${offset()}`; });
    renderWarnings(status);
    showError('status-error', null);
    return status;
  } catch (error) {
    showError('status-error', error);
    return null;
  }
}

// ---------- employees ----------
function fillEmployees(select, placeholder) {
  const old = select.value;
  select.replaceChildren();
  const first = el('option', placeholder); first.value = ''; select.append(first);
  for (const employee of state.employees) {
    const option = el('option', employee.name + (employee.active ? '' : ' (неактивен)'));
    option.value = employee.id; select.append(option);
  }
  select.value = old; select.disabled = !state.employees.length;
}
async function loadEmployees() {
  try {
    state.employees = await api('/api/employees');
    fillEmployees($('report-filter').elements.employee_id, 'Все');
    fillEmployees($('correction-form').elements.employee_id, 'Выберите сотрудника');
  } catch (error) { showError('report-error', error); }
}

// ---------- tab: now ----------
async function loadActive() {
  const load = ++state.activeLoad;
  try {
    const data = await api('/api/attendance/active');
    if (load !== state.activeLoad) return;
    showError('active-error', null);
    $('active-summary').textContent = `Обновлено ${clock(data.as_of)}`;
    $('active-rows').replaceChildren(...data.items.map((item) => row([
      employeeLink(item), clock(item.since), clock(item.last_seen_at), item.last_camera_name || `#${item.last_camera_id}`,
      hm(item.today_ms), sourceLabels[item.source] || '—'])));
    $('active-empty').hidden = data.items.length > 0;
  } catch (error) {
    if (load === state.activeLoad) showError('active-error', error);
  }
}

// ---------- tab: report ----------
function reportParams() {
  const form = $('report-filter');
  const params = new URLSearchParams();
  for (const name of ['from_date', 'to_date', 'employee_id']) if (form.elements[name].value) params.set(name, form.elements[name].value);
  return params;
}
function table(headers, bodyRows) {
  const wrap = el('div', null, 'table-wrap');
  const tbl = el('table', null, 'table');
  const head = el('tr');
  for (const title of headers) head.append(el('th', title));
  const thead = el('thead'); thead.append(head);
  const tbody = el('tbody'); tbody.append(...bodyRows);
  tbl.append(thead, tbody); wrap.append(tbl);
  return wrap;
}
function notes(item) {
  const values = [];
  if (item.has_manual) values.push('вручную');
  else if (!item.face_seen) values.push('только по внешности');
  return values.join(', ');
}
async function loadReport() {
  const form = $('report-filter');
  if (form.getAttribute('aria-busy') === 'true') return;
  const load = ++state.reportLoad;
  const params = reportParams();
  UI.setBusy(form, true);
  $('report-csv').hidden = true;
  $('report-results').replaceChildren();
  $('report-summary').textContent = 'Загрузка отчёта…';
  showError('report-error', null);
  try {
    const data = await api(`/api/attendance/report?${params}`);
    if (load !== state.reportLoad) return;
    state.reportLoaded = true;
    $('report-summary').textContent = `${data.from_date} — ${data.to_date} · на ${stamp(data.as_of)}`;
    const results = $('report-results');
    results.replaceChildren(table(
      ['Дата', 'Сотрудник', 'Отработано', 'Первое', 'Последнее', 'Сессий', 'Расчёт', 'Примечания'],
      data.rows.map(reportRow)));
    if (!data.rows.length) results.append(el('p', 'За выбранный период данных нет.', 'empty'));
    else results.append(el('h3', 'Итого за период'), table(
      ['Сотрудник', 'Отработано', 'Дней'],
      data.totals.map((item) => row([employeeLink(item), hm(item.worked_ms), item.days]))));
    $('report-csv').href = `/api/attendance/report.csv?${params}`;
    $('report-csv').hidden = false;
  } catch (error) {
    if (load !== state.reportLoad) return;
    state.reportLoaded = false;
    $('report-summary').textContent = 'Отчёт не загружен. Проверьте условия и нажмите «Сформировать».';
    showError('report-error', error);
  } finally {
    if (load === state.reportLoad) UI.setBusy(form, false);
  }
}
function reportRow(item) {
  const toggle = button(item.date, () => toggleTimeline(tr, toggle, item), 'btn ghost');
  toggle.setAttribute('aria-expanded', 'false');
  const tr = row([toggle, employeeLink(item), hm(item.worked_ms), clock(item.first_seen_at), clock(item.last_seen_at),
    item.sessions, methodLabel(item.method), notes(item)]);
  return tr;
}
async function toggleTimeline(tr, toggle, item) {
  if (tr.nextElementSibling?.dataset.detailFor === `${item.employee_id}:${item.date}`) {
    tr.nextElementSibling.remove(); toggle.setAttribute('aria-expanded', 'false'); return;
  }
  const detail = el('tr'); detail.dataset.detailFor = `${item.employee_id}:${item.date}`;
  const cell = el('td'); cell.colSpan = 8; cell.append(el('p', 'Загрузка…', 'muted'));
  detail.append(cell); tr.after(detail); toggle.setAttribute('aria-expanded', 'true');
  try {
    const data = await api(`/api/attendance/timeline?${new URLSearchParams({ employee_id: item.employee_id, date: item.date })}`);
    cell.replaceChildren(...timelineNodes(data));
  } catch (error) {
    cell.replaceChildren(el('p', error.status === 404 ? 'Сотрудник не найден.' : `Не удалось загрузить день: ${error.message}`, 'test-result err'));
  }
}
// Minutes since local midnight of `date`; a timestamp on a later date is the end of the bar.
function dayMinutes(iso, date) {
  const day = iso.slice(0, 10);
  if (day > date) return 1440;
  if (day < date) return 0;
  return Number(iso.slice(11, 13)) * 60 + Number(iso.slice(14, 16)) + Number(iso.slice(17, 19) || 0) / 60;
}
function dayBar(date, sessions, gaps) {
  const bar = el('div');
  bar.setAttribute('role', 'img');
  bar.setAttribute('aria-label', sessions.length ? `Рабочие сессии: ${sessions.map((s) => `${clock(s.started_at)}–${clock(s.ended_at)}`).join(', ')}` : 'Рабочих сессий нет');
  bar.style.cssText = 'position:relative;height:22px;border-radius:6px;overflow:hidden;background:var(--md-surface-container-high)';
  const block = (item, css, title) => {
    const from = dayMinutes(item.started_at, date), to = dayMinutes(item.ended_at, date);
    const node = el('div');
    node.style.cssText = `position:absolute;top:0;bottom:0;left:${from / 14.4}%;width:${Math.max(0.3, (to - from) / 14.4)}%;${css}`;
    node.title = `${title} ${clock(item.started_at)}–${clock(item.ended_at)}`;
    return node;
  };
  bar.append(
    ...gaps.map((gap) => block(gap, 'background:repeating-linear-gradient(45deg,var(--md-error) 0 3px,transparent 3px 6px)', 'Проход, не считается')),
    ...sessions.map((session) => block(session, 'background:var(--md-primary)', 'Работа')));
  const ticks = el('div', null, 'muted');
  ticks.style.cssText = 'display:flex;justify-content:space-between;font-size:12px';
  for (const hour of [0, 6, 12, 18, 24]) ticks.append(el('span', hour));
  const wrap = el('div'); wrap.append(bar, ticks);
  return wrap;
}
function list(title, items, render) {
  const nodes = [el('h4', title)];
  if (!items.length) { nodes.push(el('p', 'Нет', 'muted')); return nodes; }
  const ul = el('ul');
  for (const item of items) {
    const li = el('li', render(item));
    if (item.excluded) { li.style.textDecoration = 'line-through'; li.title = 'Исключено корректировкой'; }
    ul.append(li);
  }
  nodes.push(ul);
  return nodes;
}
function timelineNodes(data) {
  const correction = (kind) => (item) => `${kind} ${span(item.started_at, item.ended_at)} · ${item.actor}: ${item.reason}`;
  return [
    el('p', `${methodLabel(data.method)} · отработано ${hm(data.worked_ms)}`),
    dayBar(data.date, data.sessions, data.door_gaps),
    ...list('Сессии', data.sessions, (s) => `${clock(s.started_at)}–${clock(s.ended_at)} (${hm(Date.parse(s.ended_at) - Date.parse(s.started_at))})`),
    ...list('Выход и вход через дверь — не считается', data.door_gaps, (g) => `${clock(g.started_at)}–${clock(g.ended_at)}`),
    ...list('Наблюдения', data.segments, (s) => `${s.camera_name} · ${clock(s.started_at)}–${clock(s.last_seen_at)} · ${s.face_seen ? 'по лицу' : 'по внешности'}${s.excluded ? ' (исключено)' : ''}`),
    ...list('Проходы', data.crossings, (c) => `${directionLabels[c.direction] || c.direction} · ${c.camera_name} · ${stamp(c.occurred_at)}${c.excluded ? ' (исключено)' : ''}`),
    ...list('Корректировки', [...data.manual.map((item) => ({ ...item, text: correction('Добавлено')(item) })), ...data.exclusions.map((item) => ({ ...item, text: correction('Исключено')(item) }))], (item) => item.text),
  ];
}
$('report-filter').onsubmit = (event) => { event.preventDefault(); loadReport(); };

// ---------- tab: corrections ----------
function correctionRange() {
  const form = $('report-filter').elements;
  return { from: form.from_date.value, to: form.to_date.value };
}
async function loadCorrections() {
  const load = ++state.correctionsLoad;
  const { from, to } = correctionRange();
  $('corrections-range').textContent = `${from} — ${to}`;
  try {
    const data = await api(`/api/attendance/corrections?${new URLSearchParams({ from_date: from, to_date: to })}`);
    if (load !== state.correctionsLoad) return;
    showError('corrections-error', null);
    $('corrections-rows').replaceChildren(...data.items.map(correctionRow));
    $('corrections-empty').hidden = data.items.length > 0;
  } catch (error) {
    if (load === state.correctionsLoad) showError('corrections-error', error);
  }
}
function correctionRow(item) {
  const revoked = Boolean(item.revoked_at);
  const status = revoked ? `Отменено ${stamp(item.revoked_at)}: ${item.revoked_by}, ${item.revoke_reason}` : 'Действует';
  const action = el('td');
  if (!revoked) action.append(button('Отменить', () => openRevoke(action, item)));
  const tr = row([item.kind === 'MANUAL' ? 'Добавлено время' : 'Исключено время', employeeLink(item),
    span(item.started_at, item.ended_at), item.actor, item.reason, stamp(item.created_at), status]);
  tr.append(action);
  return tr;
}
function openRevoke(cell, item) {
  const form = el('form', null, 'attendance-form');
  const actor = el('input'); actor.name = 'actor'; actor.maxLength = 100; actor.required = true; actor.placeholder = 'Оператор'; actor.setAttribute('aria-label', 'Оператор');
  actor.value = $('correction-form').elements.actor.value;
  const reason = el('input'); reason.name = 'reason'; reason.maxLength = 1000; reason.required = true; reason.placeholder = 'Причина отмены'; reason.setAttribute('aria-label', 'Причина отмены');
  const submit = el('button', 'Отменить корректировку', 'btn primary'); submit.type = 'submit';
  const error = el('p', null, 'test-result err'); error.setAttribute('role', 'alert'); error.hidden = true;
  form.append(actor, reason, submit, button('Не отменять', () => { form.remove(); cell.querySelector('button')?.focus(); }), error);
  form.onsubmit = async (event) => {
    event.preventDefault();
    UI.setBusy(form, true);
    try {
      await api(`/api/attendance/corrections/${item.kind.toLowerCase()}/${item.id}/revoke`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ actor: actor.value.trim(), reason: reason.value.trim() }),
      });
      await refreshAfterCorrection();
    } catch (failure) {
      UI.setBusy(form, false);
      error.textContent = failure.status === 409 ? 'Корректировка уже отменена. Обновите список.' : failure.message; error.hidden = false;
    }
  };
  cell.append(form);
  actor.value ? reason.focus() : actor.focus();
}
async function refreshAfterCorrection() {
  await loadCorrections();
  if (state.reportLoaded) await loadReport();
}
const correctionForm = $('correction-form');
correctionForm.onsubmit = async (event) => {
  event.preventDefault();
  if (correctionForm.getAttribute('aria-busy') === 'true') return;
  const f = correctionForm.elements;
  showError('correction-error', null);
  $('correction-result').textContent = '';
  if (!state.status) { showError('correction-error', new Error('Часовой пояс рабочего места не загружен. Обновите страницу.')); return; }
  const seconds = (value) => (value.length === 5 ? `${value}:00` : value);
  const endDate = f.time_to.value <= f.time_from.value ? nextDay(f.date.value) : f.date.value;
  const kind = f.kind.value;
  const payload = {
    employee_id: Number(f.employee_id.value),
    started_at: `${f.date.value}T${seconds(f.time_from.value)}${offset()}`,
    ended_at: `${endDate}T${seconds(f.time_to.value)}${offset()}`,
    actor: f.actor.value.trim(), reason: f.reason.value.trim(),
  };
  // A retry of an unchanged form keeps its UUID, so a lost response replays safely.
  const signature = JSON.stringify({ kind, payload });
  if (correctionForm.pendingWrite?.signature !== signature) correctionForm.pendingWrite = { signature, request_id: UI.newRequestId() };
  UI.setBusy(correctionForm, true);
  try {
    await api(`/api/attendance/corrections/${kind}`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ...payload, request_id: correctionForm.pendingWrite.request_id }),
    });
    correctionForm.pendingWrite = null;
    f.reason.value = '';
    $('correction-result').textContent = kind === 'manual' ? 'Время добавлено.' : 'Время исключено.';
    await refreshAfterCorrection();
  } catch (error) {
    showError('correction-error', error);
  } finally {
    UI.setBusy(correctionForm, false);
  }
};

// ---------- tabs and startup ----------
function tabChanged(name) {
  state.tab = name;
  if (name === 'now') loadActive();
  else if (name === 'report' && !state.reportLoaded) loadReport();
  else if (name === 'corrections') loadCorrections();
}
async function start() {
  await Promise.all([loadEmployees(), loadStatus()]);
  const report = $('report-filter').elements;
  const params = new URLSearchParams(location.search);
  report.from_date.value = params.get('from_date') || `${today().slice(0, 8)}01`;
  report.to_date.value = params.get('to_date') || today();
  if (params.get('employee_id')) report.employee_id.value = params.get('employee_id');
  correctionForm.elements.date.value = today();
  UI.tabs(document.querySelector('[role="tablist"]'), { onChange: tabChanged });
}
start();
// Poll only read-only summaries, and only while visible; forms and report results are never replaced.
setInterval(() => {
  if (document.visibilityState !== 'visible') return;
  loadStatus();
  if (state.tab === 'now') loadActive();
}, 10000);
