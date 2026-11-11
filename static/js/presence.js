// Work attendance: server-offset times, audited writes, no visibility-based hours.
'use strict';
const $ = (id) => document.getElementById(id);
const state = { tab: 'active', employees: [], status: null, event: null, gap: null,
  offsets: { events: 0, review: 0, legacy: 0 }, selected: new Map(), imageUrls: [], eventLoad: 0 };
const labels = { ENTRY: 'Вход', EXIT: 'Выход', ACCEPTED: 'Принято', REVIEW: 'На проверке',
  VISITOR: 'Посетитель', VOID: 'Аннулировано', OPEN: 'Открыт — предварительно', CLOSED: 'Закрыт',
  FRAME_GAP: 'Пропуск кадров', STREAM_RESET: 'Переподключение потока', CLOCK_DISCONTINUITY: 'Скачок часов',
  COMMISSIONING_REQUIRED: 'Нужен ввод в эксплуатацию', CAMERA_UNCONFIGURED: 'Не настроены обе роли камер',
  CAMERA_OFFLINE: 'Камера недоступна', AI_UNAVAILABLE: 'AI-модели недоступны', AI_EXCLUDED: 'Камера исключена из AI',
  PERFORMANCE_WARMUP: 'Измеряется производительность', PERFORMANCE_DEGRADED: 'Недостаточная производительность',
  PERSISTENCE_ERROR: 'Ошибка сохранения', COVERAGE_UNAVAILABLE: 'Нет надёжного наблюдения',
  EVIDENCE_UNAVAILABLE: 'Не сохранено изображение', EVIDENCE_LOST: 'Потеряно доказательство прохода',
  PROCESS_DOWNTIME: 'Перерыв работы сервера', NO_FACE: 'Нет подтверждённого лица', AMBIGUOUS_FACE: 'Неоднозначное лицо',
  ASSOCIATION_UNCERTAIN: 'Неоднозначное сопровождение', INCOMPLETE_CROSSING: 'Неполный проход',
  DUPLICATE_ENTRY: 'Повторный вход без выхода', MISSING_ENTRY: 'Выход без входа', SIMULTANEOUS_EVENTS: 'Одновременные события',
  GALLERY_CHANGED: 'Изменена база лиц', SOURCE_CHANGED: 'Изменён источник', CALIBRATION_CHANGED: 'Изменена калибровка',
  EMPLOYEE_INACTIVE: 'Сотрудник неактивен', LOW_FACE_SCORE: 'Недостаточное сходство', BLURRY_FACE: 'Размытое лицо',
  BAD_ALIGNMENT: 'Лицо не выровнено', FACE_TOO_SMALL: 'Лицо слишком мало', LOW_FACE_DETECTION: 'Низкое качество детекции',
  UNMONITORED: 'Период до начала наблюдения', RUNNING: 'Предварительно', CONFIRMED: 'Подтверждено' };
const label = (value) => labels[value] || value || '—';
const reasons = (values) => (values || []).map(label).join('; ') || '—';
const time = (iso) => iso ? iso.replace('T', ' ') : '—';
const hms = (ms) => { const s = Math.floor(Math.max(0, ms || 0) / 1000); return `${String(Math.floor(s / 3600)).padStart(2, '0')}:${String(Math.floor(s / 60) % 60).padStart(2, '0')}:${String(s % 60).padStart(2, '0')}`; };
const inputTime = (iso) => iso ? iso.replace(/(?:Z|[+-]\d{2}:\d{2})$/, '').replace(/(\.\d{3})\d+$/, '$1') : '';
function offset() { const minutes = state.status?.utc_offset_minutes ?? 300; return `${minutes < 0 ? '-' : '+'}${String(Math.floor(Math.abs(minutes) / 60)).padStart(2, '0')}:${String(Math.abs(minutes) % 60).padStart(2, '0')}`; }
function timestamp(input) { if (!input.value) return null; return input.value + offset(); }
function el(tag, text, className) { const node = document.createElement(tag); if (text != null) node.textContent = String(text); if (className) node.className = className; return node; }
function button(text, action, className = 'btn') { const node = el('button', text, className); node.type = 'button'; node.onclick = action; return node; }
function link(text, href) { const node = el('a', text); node.href = href; return node; }
function row(values) { const tr = el('tr'); for (const value of values) { const td = el('td'); if (value instanceof Node) td.append(value); else td.textContent = value ?? '—'; tr.append(td); } return tr; }
function badge(value) { return el('span', label(value), `status ${value === 'REVIEW' ? 'connecting' : ['ACCEPTED', 'CLOSED'].includes(value) ? 'online' : 'offline'}`); }
function employeeLink(item) { return item.employee_id ? link(item.employee_name || `#${item.employee_id}`, `/employees/${item.employee_id}`) : el('span', 'Не установлен'); }
function requestId() { const bytes = crypto.getRandomValues(new Uint8Array(16)); bytes[6] = (bytes[6] & 15) | 64; bytes[8] = (bytes[8] & 63) | 128; const hex = Array.from(bytes, (v) => v.toString(16).padStart(2, '0')).join(''); return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20)}`; }
async function api(path, options = {}) {
  let response;
  try { response = await fetch(path, { cache: 'no-store', ...options }); }
  catch (cause) { const error = new Error('Нет связи с сервером. Поля сохранены: повторите тот же запрос.'); error.status = 0; throw error; }
  if (!response.ok) {
    let detail = '';
    try { const data = await response.json(); detail = Array.isArray(data.detail) ? data.detail.map((item) => `${item.loc?.join('.') || ''}: ${item.msg}`).join('; ') : data.detail || ''; } catch (_) { /* Non-JSON network/proxy errors still have a status. */ }
    const error = new Error(response.status === 409 ? 'Данные уже изменены. Обновите версию, сравните изменения и повторно примените решение.' : response.status === 503 ? 'Сохранение или чтение недоступно (503). Поля сохранены; повторите запрос после восстановления.' : detail || `Ошибка ${response.status}`);
    error.status = response.status; throw error;
  }
  return response.json();
}
function showError(id, error) { $(id).hidden = !error; $(id).textContent = error?.message || ''; }
function fields(form) { return form.elements; }
function human(form) { return { actor: fields(form).actor.value.trim(), reason: fields(form).reason.value.trim() }; }
async function write(form, path, payload) {
  // A retry of an unchanged form keeps its UUID, including after an unknown
  // network outcome. Editing the payload creates a distinct operator request.
  const signature = JSON.stringify({ path, payload });
  if (form.pendingWrite?.signature !== signature) form.pendingWrite = { signature, request_id: requestId() };
  const submits = [...form.querySelectorAll('[type="submit"]')];
  submits.forEach((item) => { item.disabled = true; }); form.setAttribute('aria-busy', 'true');
  try {
    const result = await api(path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ ...payload, request_id: form.pendingWrite.request_id }) });
    form.pendingWrite = null; return result;
  } finally { submits.forEach((item) => { item.disabled = false; }); form.removeAttribute('aria-busy'); }
}
function options(select, employees, all = false) {
  const old = select.value; select.replaceChildren();
  const placeholder = el('option', all ? 'Все сотрудники' : 'Выберите сотрудника');
  placeholder.value = '';
  select.append(placeholder);
  for (const employee of employees) {
    const option = el('option', employee.name + (employee.active ? '' : ' (неактивен)'));
    option.value = employee.id;
    select.append(option);
  }
  select.value = old; select.disabled = !employees.length;
}
async function loadEmployees() {
  try {
    state.employees = await api('/api/employees');
    document.querySelectorAll('.attendance-employees').forEach((select) => options(select, state.employees, select.dataset.all === 'true'));
    if (!state.employees.length) $('manual-result').textContent = 'Нет сотрудников. Создайте сотрудника на странице «Сотрудники»; проверка посетителей доступна без регистрации сотрудников.';
  } catch (error) { showError('manual-error', error); }
}
function commissionVersions() {
  const f = fields($('commission-form'));
  for (const role of ['ENTRY', 'EXIT']) { const config = state.status?.configuration.find((item) => item.enabled && item.role === role); f[role === 'ENTRY' ? 'expected_entry_revision' : 'expected_exit_revision'].value = config?.revision ?? ''; }
  $('commission-form').dataset.versionsSet = 'true';
}
async function loadStatus() {
  try {
    const status = await api('/api/attendance/status'); state.status = status;
    showError('status-error', null);
    $('technical-state').textContent = status.technical_ready ? 'Техника готова' : 'Техника не готова';
    $('technical-state').className = `status ${status.technical_ready ? 'online' : 'connecting'}`;
    $('automatic-state').textContent = status.automatic_ready ? 'Автоматический учёт разрешён' : 'Автоматический учёт заблокирован';
    $('automatic-state').className = `status ${status.automatic_ready ? 'online' : 'connecting'}`;
    $('readiness-reasons').textContent = `${reasons(status.reasons)} · технически здоровых секунд: ${Math.floor(status.technically_healthy_seconds)} · в очереди записей: ${status.pending_write_count}, уведомлений о сбоях: ${status.pending_fault_count}`;
    document.querySelectorAll('.zone-label').forEach((node) => { node.textContent = `UTC${offset()}`; }); $('workplace-zone').textContent = `UTC${offset()} · фиксированный часовой пояс`;
    const alerts = $('attendance-alerts'); alerts.replaceChildren();
    alerts.append(link(`Неопознанных/нерешённых проходов: ${status.unresolved_crossings}`, '/presence?kind=event#review'), document.createTextNode(' · '), link(`Потерь наблюдения: ${status.unreviewed_gaps}`, '/presence?kind=gap#review'), document.createTextNode(` · Конфликтов порядка: ${status.projection_conflicts}`));
    const configs = status.configuration.filter((item) => item.enabled);
    $('configuration-state').textContent = ['ENTRY', 'EXIT'].every((role) => configs.some((item) => item.role === role)) ? configs.map((item) => `${label(item.role)}: камера #${item.camera_id}, калибровка ${item.revision}, источник ${item.source_revision}`).join(' · ') : 'Не настроены обе направленные камеры. Калибруйте линию прохода и сторону помещения; ручная работа и проверка остаются доступны.';
    const metrics = $('role-metrics'); metrics.replaceChildren();
    for (const role of ['ENTRY', 'EXIT']) { const data = status.roles[role]; metrics.append(row(data ? [`${label(role)} / #${data.camera_id}`, data.processed_fps.toFixed(1), data.frame_age_seconds?.toFixed(3) ?? '—', `${data.processing_age_p95_seconds?.toFixed(3) ?? '—'} / ${data.completion_age_p95_seconds?.toFixed(3) ?? '—'}`, reasons(data.reasons)] : [label(role), '—', '—', '—', 'Не настроена'])); }
    const providers = $('provider-list'); providers.replaceChildren(); for (const [name, values] of Object.entries(status.providers)) providers.append(el('dt', name), el('dd', values.length ? values.join(', ') : 'Не загружен'));
    $('commission-state').textContent = status.commission ? `Последнее подтверждение #${status.commission.id}: ${time(status.commission.created_at)}, ${status.commission.actor}. ${status.commissioning_required ? 'Текущая конфигурация не подтверждена.' : 'Текущая конфигурация подтверждена.'}` : 'Испытания не подтверждены: автоматические проходы сохраняются на проверку.';
    if (!$('commission-form').dataset.versionsSet) commissionVersions();
    return status;
  } catch (error) {
    showError('status-error', error);
    $('automatic-state').textContent = 'Готовность неизвестна: нет актуальных данных';
    $('automatic-state').className = 'status error';
    $('technical-state').textContent = 'Техническая готовность неизвестна';
    $('technical-state').className = 'status error';
    return null;
  }
}
async function loadActive() {
  try {
    const data = await api('/api/attendance/active'); showError('active-error', null);
    const tbody = $('active-rows'); tbody.replaceChildren(); $('active-empty').hidden = data.items.length > 0;
    $('active-summary').textContent = `${data.total} незакрытых · на ${time(data.as_of)}`;
    for (const item of data.items) { const status = el('div'); status.append(badge(item.status), el('p', reasons(item.reason_codes), 'muted small')); tbody.append(row([employeeLink(item), time(item.started_at), hms(item.running_ms), hms(item.review_ms), status])); }
  } catch (error) { showError('active-error', error); }
}
function query(form) { const params = new URLSearchParams(); for (const [key, value] of new FormData(form)) if (value) params.set(key, value); return params; }
function pager(name, data) {
  const container = $(`${name}-pager`); container.replaceChildren();
  const previous = button('Назад', () => { state.offsets[name] = Math.max(0, data.offset - data.limit); loadPage(name); }); previous.disabled = data.offset === 0;
  const next = button('Далее', () => { state.offsets[name] = data.offset + data.limit; loadPage(name); }); next.disabled = data.offset + data.items.length >= data.total;
  container.append(previous, el('span', data.total ? `${data.offset + 1}–${data.offset + data.items.length} из ${data.total}` : 'Всего: 0', 'muted'), next);
}
async function loadPage(name) {
  const params = query($(`${name}-filter`)); params.set('limit', '50'); params.set('offset', state.offsets[name]);
  const path = name === 'legacy' ? '/api/presence/legacy' : `/api/attendance/${name}`;
  try {
    const data = await api(`${path}?${params}`); showError(`${name}-error`, null);
    const tbody = $(`${name}-rows`); tbody.replaceChildren(); $(`${name}-empty`).hidden = data.items.length > 0;
    for (const item of data.items) {
      if (name === 'events') {
        const status = el('div'); status.append(badge(item.status), el('p', reasons([...item.reason_codes, ...item.projection_reasons]), 'muted small'));
        tbody.append(row([time(item.occurred_at), employeeLink(item), label(item.direction), item.source === 'MANUAL' ? 'Ручное событие' : item.camera_name || `Камера #${item.camera_id}`, status, button('Открыть', () => openEvent(item.id))]));
      } else if (name === 'review') {
        if (item.kind === 'event') {
          const event = item.event; const box = el('input'); box.type = 'checkbox'; box.setAttribute('aria-label', `Посетитель: ${time(event.occurred_at)}`); box.disabled = event.status !== 'REVIEW' || event.employee_id != null; box.checked = state.selected.has(event.id);
          box.onchange = () => { if (box.checked && state.selected.size >= 100) { box.checked = false; showError('bulk-error', new Error('В одном запросе можно выбрать не более 100 проходов.')); return; } if (box.checked) state.selected.set(event.id, event.revision); else state.selected.delete(event.id); selectionCount(); };
          tbody.append(row([box, time(event.occurred_at), employeeLink(event), reasons([...event.reason_codes, ...event.projection_reasons]), button('Проверить проход', () => openEvent(event.id))]));
        } else { const gap = item.gap; tbody.append(row(['—', `${time(gap.started_at)} — ${gap.ended_at ? time(gap.ended_at) : 'ещё открыт'}`, 'Потеря наблюдения', reasons(gap.reason_codes), button('Сверить период', () => openGap(gap))])); }
      } else tbody.append(row([`${item.employee_name || '—'} / ${item.camera_name || '—'}`, time(item.started_at), time(item.last_seen_at), item.ended_at ? time(item.ended_at) : 'Историческое незакрытое наблюдение', item.confidence?.toFixed(3) ?? '—']));
    }
    pager(name, data);
  } catch (error) { showError(`${name}-error`, error); }
}
function selectionCount() { $('bulk-count').textContent = `Выбрано: ${state.selected.size} (включая другие страницы)`; }
function activate(name, { focus = false } = {}) {
  if (!['active', 'events', 'review', 'report', 'legacy'].includes(name)) name = 'active'; state.tab = name;
  document.querySelectorAll('[data-tab]').forEach((node) => { const active = node.dataset.tab === name; node.setAttribute('aria-selected', String(active)); node.tabIndex = active ? 0 : -1; });
  document.querySelectorAll('.attendance-tabpanel').forEach((node) => { node.hidden = node.id !== `panel-${name}`; });
  if (focus) $(`tab-${name}`).focus();
  if (name === 'active') loadActive(); else if (name !== 'report') loadPage(name);
}
document.querySelectorAll('[data-tab]').forEach((node) => { node.onclick = () => { location.hash = node.dataset.tab; }; node.onkeydown = (event) => { const tabs = [...document.querySelectorAll('[data-tab]')]; let index = tabs.indexOf(node); if (event.key === 'ArrowRight') index = (index + 1) % tabs.length; else if (event.key === 'ArrowLeft') index = (index + tabs.length - 1) % tabs.length; else if (event.key === 'Home') index = 0; else if (event.key === 'End') index = tabs.length - 1; else return; event.preventDefault(); location.hash = tabs[index].dataset.tab; activate(tabs[index].dataset.tab, { focus: true }); }; });
window.addEventListener('hashchange', () => activate(location.hash.slice(1)));
for (const name of ['events', 'review', 'legacy']) $(`${name}-filter`).onsubmit = (event) => { event.preventDefault(); state.offsets[name] = 0; if (name === 'review') { state.selected.clear(); selectionCount(); } loadPage(name); };
function clearImages() { for (const url of state.imageUrls) URL.revokeObjectURL(url); state.imageUrls = []; }
async function evidenceImage(event, kind, name, target, generation) {
  const figure = el('figure'); const caption = el('figcaption', name); const message = el('p', event.evidence.purged ? 'Срок хранения изображения истёк' : !event.evidence[kind] ? 'Изображение не сохранено' : 'Загрузка изображения…', 'muted'); figure.append(caption, message); target.append(figure);
  if (event.evidence.purged || !event.evidence[kind]) return;
  try {
    const response = await fetch(`/api/attendance/events/${encodeURIComponent(event.id)}/evidence/${kind}`, { cache: 'no-store' });
    if (generation !== state.eventLoad) return;
    if (!response.ok) { message.textContent = response.status === 410 ? 'Срок хранения изображения истёк' : response.status === 404 ? 'Изображение отсутствует' : 'Изображение временно недоступно — откройте событие повторно'; return; }
    const blob = await response.blob(); if (generation !== state.eventLoad) return;
    const url = URL.createObjectURL(blob); state.imageUrls.push(url); const image = el('img'); image.src = url; image.alt = `${name}: событие ${time(event.occurred_at)}`; message.replaceWith(image);
  } catch (_) { if (generation === state.eventLoad) message.textContent = 'Нет связи: изображение не загружено. Откройте событие повторно.'; }
}
async function openEvent(identifier, preserve = false) {
  const generation = ++state.eventLoad;
  try {
    const data = await api(`/api/attendance/events/${encodeURIComponent(identifier)}`); if (generation !== state.eventLoad) return;
    state.event = data; clearImages(); $('event-editor').hidden = false; $('gap-editor').hidden = true;
    $('event-effective').textContent = `${time(data.occurred_at)} · ${label(data.direction)} · ${data.employee_name || 'Сотрудник не установлен'} · ${label(data.status)} · версия ${data.revision}. Причины: ${reasons([...data.reason_codes, ...data.projection_reasons])}`;
    const images = $('event-images'); images.replaceChildren(); for (const [kind, name] of [['before', 'До прохода'], ['after', 'После прохода'], ['face', 'Лицо']]) evidenceImage(data, kind, name, images, generation);
    const candidates = $('event-candidates'); candidates.replaceChildren(el('h3', 'Кандидаты и качество лица'));
    const observations = data.raw.face_evidence || [];
    if (!observations.length) candidates.append(el('p', 'В исходной записи нет наблюдений лица. Ручное событие не содержит вымышленных изображений или оценок.', 'muted'));
    for (const face of observations) { const p = el('p'); p.textContent = `Кадр ${face.frame_seq} · кандидат: ${face.candidate_employee_name || face.candidate_employee_id || 'нет'} · сходство: ${face.score ?? face.candidate_score ?? '—'} · второй: ${face.runner_up_employee_name || face.runner_up_employee_id || 'нет'} (${face.runner_up_score ?? '—'}) · разница: ${face.margin ?? '—'} · качество: ${JSON.stringify(face.quality || {})}`; candidates.append(p); }
    candidates.append(el('p', `Пригодность до эксплуатационного допуска: ${data.raw.would_auto_accept ? 'да' : 'нет'}. Это не вероятность и не разрешение начислять часы.`, 'muted'));
    $('event-raw').textContent = JSON.stringify(data.raw, null, 2);
    const history = $('decision-history'); history.replaceChildren();
    for (const decision of data.decisions) { const item = el('li'); item.append(el('p', `Версия ${decision.revision} · ${time(decision.created_at)} · ${decision.actor}: ${decision.reason}`)); const changes = el('details'); changes.append(el('summary', 'До и после решения'), el('pre', JSON.stringify({ before: decision.before, after: decision.after }, null, 2))); item.append(changes); history.append(item); }
    if (!data.decisions.length) history.append(el('li', 'Решений оператора ещё нет.'));
    const f = fields($('decision-form')); f.expected_revision.value = data.revision;
    if (!preserve) { f.action.value = 'ACCEPT'; f.employee_id.value = data.employee_id || ''; f.direction.value = data.direction || ''; f.occurred_at.value = inputTime(data.occurred_at); f.reason.value = ''; $('decision-result').textContent = ''; $('decision-form').pendingWrite = null; }
    showError('decision-error', null); $('decision-rebase').hidden = true; decisionFields();
    if (!preserve) { $('event-editor').scrollIntoView({ behavior: 'auto', block: 'start' }); $('event-title').focus(); }
    else $('decision-result').textContent = 'Загружена актуальная версия. Ваши поля сохранены: сравните их с текущим состоянием выше, затем явно сохраните решение.';
  } catch (error) { showError(state.tab === 'review' ? 'review-error' : 'events-error', error); }
}
function decisionFields() { const f = fields($('decision-form')); const accept = f.action.value === 'ACCEPT'; for (const name of ['employee_id', 'direction', 'occurred_at']) { f[name].required = accept; f[name].disabled = !accept || (name === 'employee_id' && !state.employees.length); } }
fields($('decision-form')).action.onchange = decisionFields;
$('event-close').onclick = () => { $('event-editor').hidden = true; state.eventLoad++; clearImages(); };
$('decision-rebase').onclick = () => openEvent(state.event.id, true);
$('decision-form').onsubmit = async (event) => {
  event.preventDefault(); const form = event.currentTarget; const f = fields(form); const data = { ...human(form), expected_revision: Number(f.expected_revision.value), action: f.action.value };
  if (data.action === 'ACCEPT') Object.assign(data, { employee_id: Number(f.employee_id.value), direction: f.direction.value, occurred_at: timestamp(f.occurred_at) });
  try { const result = await write(form, `/api/attendance/events/${encodeURIComponent(state.event.id)}/decision`, data); await openEvent(result.id, true); $('decision-result').textContent = 'Решение сохранено. Рабочие интервалы пересчитаны; исходные данные не изменены.'; loadActive(); loadStatus(); if (['events', 'review'].includes(state.tab)) loadPage(state.tab); }
  catch (error) { showError('decision-error', error); $('decision-rebase').hidden = error.status !== 409; }
};
async function openGap(gap, preserve = false) {
  const generation = ++state.eventLoad;
  clearImages();
  $('gap-editor').hidden = false;
  $('event-editor').hidden = true;
  const form = $('gap-form');
  form.querySelector('[type="submit"]').disabled = true;
  try {
    const current = await api(`/api/attendance/gaps/${gap.id}`);
    if (generation !== state.eventLoad) return;
    state.gap = current;
    $('gap-description').textContent = `${time(current.started_at)} — ${current.ended_at ? time(current.ended_at) : 'период ещё открыт; дождитесь восстановления'}. ${reasons(current.reason_codes)}. Версия ${current.revision}. ${current.reviewed ? 'Период уже проверен.' : ''}`;
    fields(form).expected_revision.value = current.revision;
    form.querySelector('[type="submit"]').disabled = !current.ended_at;
    if (!preserve) {
      fields(form).reason.value = '';
      fields(form).reconciled.checked = false;
      form.pendingWrite = null;
      $('gap-result').textContent = '';
    }
    const history = $('gap-history');
    history.replaceChildren();
    for (const decision of current.decisions) {
      const item = el('li');
      item.append(el('p', `Версия ${decision.revision} · ${time(decision.created_at)} · ${decision.actor}: ${decision.reason}`));
      const changes = el('details');
      changes.append(el('summary', 'До и после сверки'), el('pre', JSON.stringify({ before: decision.before, after: decision.after }, null, 2)));
      item.append(changes);
      history.append(item);
    }
    if (!current.decisions.length) history.append(el('li', 'Сверок оператора ещё нет.'));
    showError('gap-error', null);
    $('gap-rebase').hidden = true;
    $('gap-editor').scrollIntoView({ block: 'start' });
    $('gap-title').focus();
  } catch (error) { showError('gap-error', error); }
}
$('gap-close').onclick = () => { $('gap-editor').hidden = true; state.eventLoad++; };
$('gap-rebase').onclick = async () => {
  await openGap(state.gap, true);
  $('gap-result').textContent = 'Ваши поля сохранены. Сравните новую версию и повторно сверьте период перед подтверждением.';
};
$('gap-form').onsubmit = async (event) => {
  event.preventDefault();
  const form = event.currentTarget;
  try {
    const result = await write(form, `/api/attendance/gaps/${state.gap.id}/review`, {
      ...human(form), expected_revision: Number(fields(form).expected_revision.value),
    });
    await openGap(result, true);
    $('gap-result').textContent = 'Сверка сохранена. История потери наблюдения остаётся в журнале.';
    loadActive(); loadStatus(); loadPage('review');
  } catch (error) {
    showError('gap-error', error);
    $('gap-rebase').hidden = error.status !== 409;
  }
};
$('manual-details').ontoggle = () => {
  if ($('manual-details').open && !fields($('manual-form')).occurred_at.value) {
    fields($('manual-form')).occurred_at.value = inputTime(state.status?.as_of);
  }
};
$('manual-form').onsubmit = async (event) => {
  event.preventDefault();
  const form = event.currentTarget;
  const f = fields(form);
  try {
    const result = await write(form, '/api/attendance/manual', {
      ...human(form), employee_id: Number(f.employee_id.value), direction: f.direction.value,
      occurred_at: timestamp(f.occurred_at),
    });
    showError('manual-error', null);
    $('manual-result').replaceChildren(document.createTextNode('Событие сохранено. '),
      button('Открыть запись', () => openEvent(result.id), 'btn ghost'));
    f.reason.value = '';
    loadActive(); loadStatus();
  } catch (error) { showError('manual-error', error); }
};
$('bulk-form').onsubmit = async (event) => {
  event.preventDefault();
  const form = event.currentTarget;
  if (!state.selected.size) {
    showError('bulk-error', new Error('Сначала выберите проверенные неопознанные проходы в таблице.'));
    return;
  }
  if (!confirm(`Отметить ${state.selected.size} выбранных проходов посетителями? Они не будут создавать рабочее время.`)) return;
  const items = [...state.selected].map(([event_id, expected_revision]) => ({ event_id, expected_revision }))
    .sort((a, b) => a.event_id.localeCompare(b.event_id));
  try {
    await write(form, '/api/attendance/visitors', { ...human(form), items });
    state.selected.clear(); selectionCount();
    showError('bulk-error', null);
    fields(form).reason.value = '';
    loadPage('review'); loadActive(); loadStatus();
  } catch (error) {
    showError('bulk-error', error);
    $('bulk-rebase').hidden = error.status !== 409;
  }
};
$('bulk-rebase').onclick = () => {
  state.selected.clear(); selectionCount();
  $('bulk-form').pendingWrite = null;
  $('bulk-rebase').hidden = true;
  loadPage('review');
};
$('commission-details').ontoggle = () => {
  if ($('commission-details').open && !$('commission-form').dataset.versionsSet) commissionVersions();
};
$('commission-rebase').onclick = async () => {
  if (!await loadStatus()) return;
  commissionVersions();
  $('commission-rebase').hidden = true;
  $('commission-result').textContent = 'Версии обновлены. Убедитесь, что испытания относятся к текущей конфигурации.';
};
$('commission-form').onsubmit = async (event) => {
  event.preventDefault();
  const form = event.currentTarget;
  const f = fields(form);
  const acceptance_record = {};
  for (const name of ['entry_trials', 'exit_trials', 'employee_count', 'visitor_trials', 'retreat_trials',
    'wrong_employee', 'duplicate_trips', 'missed_unflagged', 'frontal_trials', 'frontal_eligible', 'performance_minutes']) {
    acceptance_record[name] = Number(f[name].value);
  }
  acceptance_record.site_confirmed = f.site_confirmed.checked;
  if (!f.expected_entry_revision.value || !f.expected_exit_revision.value) {
    showError('commission-error', new Error('Сначала настройте обе камеры и обновите версии конфигурации.'));
    return;
  }
  try {
    const result = await write(form, '/api/attendance/commission', {
      ...human(form), expected_entry_revision: Number(f.expected_entry_revision.value),
      expected_exit_revision: Number(f.expected_exit_revision.value), acceptance_record,
    });
    showError('commission-error', null);
    $('commission-result').textContent = `Подтверждение #${result.commission_id} сохранено. Прежние пробные события остаются на проверке; аннулируйте испытательные записи с явным основанием.`;
    loadStatus();
  } catch (error) {
    showError('commission-error', error);
    $('commission-rebase').hidden = error.status !== 409;
  }
};
function reportTable(headers, items, render) {
  const wrap = el('div', null, 'table-wrap');
  const table = el('table', null, 'table');
  const head = el('thead');
  const tr = el('tr');
  for (const title of headers) tr.append(el('th', title));
  head.append(tr);
  const body = el('tbody');
  for (const item of items) body.append(row(render(item)));
  table.append(head, body); wrap.append(table);
  return wrap;
}
$('report-filter').onsubmit = async (event) => {
  event.preventDefault();
  const params = query(event.currentTarget);
  $('report-csv').hidden = true;
  try {
    const data = await api(`/api/attendance/report?${params}`);
    showError('report-error', null);
    $('report-summary').textContent = `${data.from_date} — ${data.to_date} · на ${time(data.as_of)} · ${data.finalizable ? 'Полные завершённые дни, можно использовать итог' : 'Неокончательный отчёт: есть открытые интервалы или неполнота наблюдения'}`;
    $('report-warnings').replaceChildren(
      link(`Нерешённых проходов: ${data.unresolved_crossings}`, '/presence?kind=event#review'),
      document.createTextNode(' · '), link(`Потерь наблюдения: ${data.unreviewed_gaps}`, '/presence?kind=gap#review'));
    const results = $('report-results');
    results.replaceChildren(el('h3', 'По дням'));
    results.append(reportTable(
      ['Дата', 'Сотрудник', 'Подтверждено', 'Текущее', 'На проверке', 'Визитов', 'Первый вход', 'Последний выход', 'Нерешено', 'Состояние / покрытие'],
      data.rows, (item) => [item.date, item.employee_name, hms(item.confirmed_ms), hms(item.running_ms),
        hms(item.review_ms), item.visit_count, time(item.first_entry), time(item.last_exit), item.pending_count,
        `${label(item.status)} · ${reasons(item.coverage_reasons)}`]));
    results.append(el('h3', 'Итоги по сотрудникам'), reportTable(
      ['Сотрудник', 'Подтверждено', 'Текущее', 'На проверке', 'Нерешено'], data.totals,
      (item) => [item.employee_name, hms(item.confirmed_ms), hms(item.running_ms), hms(item.review_ms), item.pending_count]));
    if (!data.rows.length) results.append(el('p', 'Нет сотрудников или рабочих записей за выбранный период.', 'empty'));
    $('report-csv').href = `/api/attendance/report.csv?${params}`;
    $('report-csv').hidden = false;
  } catch (error) {
    if (error.status === 404) error.message = 'Сервис отчётов недоступен. Рабочие интервалы и проверка доступны в соседних вкладках.';
    showError('report-error', error);
  }
};
$('refresh-all').onclick = () => {
  loadEmployees(); loadStatus(); loadActive();
  if (['events', 'review', 'legacy'].includes(state.tab)) loadPage(state.tab);
};
async function start() {
  await loadEmployees();
  const status = await loadStatus();
  if (status) {
    const report = fields($('report-filter'));
    report.from_date.value = status.as_of.slice(0, 10);
    report.to_date.value = status.as_of.slice(0, 10);
  }
  const params = new URLSearchParams(location.search);
  if (params.get('employee_id')) {
    for (const form of ['events-filter', 'review-filter', 'legacy-filter', 'report-filter']) {
      fields($(form)).employee_id.value = params.get('employee_id');
    }
  }
  if (params.get('kind')) fields($('review-filter')).kind.value = params.get('kind');
  activate(location.hash.slice(1) || 'active');
  if (params.get('event_id')) openEvent(params.get('event_id'));
}
start();
// Poll only read-only summaries. Never replace review/manual/commission fields,
// selected visitor rows, audit editors or report results while the operator types.
setInterval(() => { loadStatus(); loadActive(); }, 5000);
