// Person-appearance learning: read-only candidate analysis, audited decisions and undo.
// Writes go through JournalCorrections.mutateLearning so they share its freeze/retry/lock contract.
(() => {
  'use strict';
  const $ = (id) => document.getElementById(id), UI = window.OperatorUI, crop = window.CameraCrop;
  const corrections = window.JournalCorrections;
  const root = $('visit-detail');
  if (!root || !corrections) return;
  const visitId = root.dataset.visitId;
  const endpoint = `/api/global-persons/visits/${encodeURIComponent(visitId)}`;
  const bounds = ['learning-x1', 'learning-y1', 'learning-x2', 'learning-y2'].map($);
  const FULL = [0, 0, 1, 1];
  const MAX_UPLOAD = 15 * 1024 * 1024;
  const reasons = {
    BODY_MODEL_UNAVAILABLE: 'модель внешности недоступна',
    ENROLLMENT_UNAVAILABLE: 'подготовка изображений недоступна',
  };
  const faceCopy = {
    ADDED: 'Образец лица тоже добавлен в галерею лиц.',
    INELIGIBLE: 'Лицо не добавлено: на выбранном фрагменте нет одного достаточно чёткого лица от 80 px.',
    LIMIT: 'Лицо не добавлено: у сотрудника уже 15 образцов лица.',
    CONFLICT: 'Лицо не добавлено: оно слишком похоже на другого сотрудника.',
    UNAVAILABLE: 'Лицо не добавлено: модель лиц недоступна.',
    NEGATIVE_ADDED: 'Лицо тоже исключено как отрицательный пример.',
  };
  const modalityCopy = { BODY: 'внешность', FACE: 'лицо' };
  const staleCopy = 'Данные изменились. Нажмите «Предложить сотрудников», проверьте результат и подтвердите заново. Оператор и причина сохранены.';
  const conflictCopy = {
    IDENTITY_REVISION_CONFLICT: 'Личность в журнале уже изменена. ' + staleCopy,
    EMPLOYEE_REVISION_CONFLICT: 'Сотрудник или его образцы изменились. ' + staleCopy,
    LEARNING_REVISION_CONFLICT: 'Обучение уже изменено. ' + staleCopy,
    GALLERY_REVISION_CONFLICT: 'Галерея лиц изменилась. ' + staleCopy,
    LEARNING_REVISION_CHANGED: 'Данные менялись во время расчёта. ' + staleCopy,
    LEARNING_EVIDENCE_CHANGED: 'Выбранный фрагмент теперь даёт другое изображение. ' + staleCopy,
    BODY_MODEL_CHANGED: 'Модель внешности изменилась. ' + staleCopy,
    FACE_MODEL_CHANGED: 'Модель лиц изменилась. ' + staleCopy,
    VISIT_PHOTO_CHANGED: 'Снимок изменился. Обновите наблюдение, затем ' + staleCopy,
  };

  let visit = null, digests = {}, outage = false;
  let selection = [...FULL], drag = null, generation = 0;
  let shownSrc = null, savedDigest = null;
  let uploadFile = null, pendingUpload = null, uploadTicket = 0;
  let analysis = null, analyzeTicket = 0, analyzeController = null, analyzingFlag = false, autoKey = null;
  let known = { learning: -1, gallery: -1 };
  let employees = [], selected = null, employeeTicket = 0, employeeLoading = false;
  let cardButtons = [];

  const el = (tag, text, cls) => {
    const node = document.createElement(tag);
    if (text != null) node.textContent = text;
    if (cls) node.className = cls;
    return node;
  };
  const button = (text, handler) => {
    const node = el('button', text, 'btn'); node.type = 'button'; node.onclick = handler; return node;
  };
  const message = (id, text, error = false) => UI.message($(id), text, error ? 'error' : '');
  const stamp = (value) => value ? String(value).replace('T', ' ').replace(/Z$/, ' UTC') : '—';
  const count = (value) => Number.isInteger(value) ? String(value) : '—';
  const source = () => $('learning-source').value;
  const state = () => corrections.learningState();
  const bump = (key, value) => { if (Number.isSafeInteger(value)) known[key] = Math.max(known[key], value); };

  // ---- evidence (source image, numeric crop, capture time) ----
  function offsetOf() {
    return visit?.time_bounds?.started_at?.match(/(Z|[+-]\d{2}:\d{2})$/)?.[0] ?? null;
  }
  function capturedAt() {
    if (source() !== 'UPLOAD') return { value: null };
    const raw = $('learning-captured-at').value;
    if (!raw) return { value: null };
    const offset = offsetOf();
    if (!offset) return { error: 'Часовой пояс организации неизвестен. Обновите наблюдение.' };
    const local = raw.length === 16 ? raw + ':00' : raw, ms = Date.parse(local + offset);
    if (!Number.isFinite(ms)) return { error: 'Дата съёмки указана неверно.' };
    if (ms > Date.now()) return { error: 'Дата съёмки не может быть в будущем.' };
    return { value: local + offset };
  }
  function savedReady() {
    return Boolean(digests.photo && visit && digests.photo === visit.photo_sha256);
  }
  function imageReady() {
    const image = $('learning-image');
    const base = source() === 'UPLOAD' ? uploadFile && shownSrc === uploadFile.url : savedReady() && shownSrc;
    return Boolean(base && image.complete && image.naturalWidth > 0);
  }
  function setImage(src) {
    const image = $('learning-image');
    if (src) { if (shownSrc !== src) { shownSrc = src; image.src = src; } }
    else { shownSrc = null; image.removeAttribute('src'); }
    image.hidden = !src;
  }
  function showSourceImage() {
    setImage(source() === 'UPLOAD' ? (pendingUpload?.url ?? null)
      : savedReady() ? $('visit-photo').getAttribute('src') : null);
  }
  function resetSelection() {
    selection = [...FULL];
    bounds.forEach((input, index) => { input.value = String(FULL[index]); });
  }
  function dropUpload() {
    uploadTicket++;
    if (pendingUpload) URL.revokeObjectURL(pendingUpload.url);
    pendingUpload = null; uploadFile = null;
    if (source() === 'UPLOAD') setImage(null);
  }
  function cancelAnalysis() {
    analyzeTicket++;
    if (analyzeController) { analyzeController.abort(); analyzeController = null; }
    if (analyzingFlag) { analyzingFlag = false; corrections.setLearningAnalyzing(false); }
  }
  function evidenceChanged() {
    generation++; analysis = null; cancelAnalysis();
    $('learning-person-confirm').checked = false;
    message('learning-analysis-status', '');
    renderCandidates();
  }
  function analysisValid() {
    if (!analysis || analysis.gen !== generation || !visit) return false;
    const data = analysis.data;
    return data.learning_revision >= known.learning && data.gallery_revision >= known.gallery
      && data.identity_revision >= (visit.identity_revision ?? 0);
  }
  function evidenceText() {
    if (source() === 'UPLOAD') return 'Образец сотрудника — не меняет личность в этом визите.';
    return 'Снимок этого наблюдения: подтверждение запишет личность визита, отклонение снимет её, если она совпадает.';
  }
  function eligibilityText() {
    const data = analysis && analysis.gen === generation ? analysis.data : null;
    const body = source() === 'UPLOAD' ? ' Дата описывает внешность образца, а не приход.' : '';
    if (data) {
      const anchor = data.evidence.daily_anchor || {};
      if (anchor.today) return 'Снимок сегодняшний: подтверждение сможет служить опорой для автоматического узнавания сегодня.' + body;
      return (anchor.date ? `Дата снимка ${anchor.date}, не сегодня` : 'Дата снимка неизвестна') + ': только подсказки, автоматической опорой не станет.' + body;
    }
    if (source() === 'UPLOAD') return 'Без даты съёмки образец даёт только подсказки.' + body;
    return visit?.photo_captured_at ? '' : 'У этого снимка нет точной даты съёмки: он даёт только подсказки.';
  }
  function paint() {
    const image = $('learning-image'), box = $('learning-selection'), canvas = $('learning-preview');
    box.toggleAttribute('hidden', !selection);
    if (selection) {
      const [x1, y1, x2, y2] = selection;
      for (const [key, value] of Object.entries({ x: x1, y: y1, width: x2 - x1, height: y2 - y1 })) box.setAttribute(key, value);
    }
    const native = imageReady() && selection && crop.nativeCrop(selection, image.naturalWidth, image.naturalHeight);
    canvas.hidden = !native;
    if (native) {
      const scale = Math.min(1, 256 / Math.max(native.width, native.height));
      canvas.width = Math.max(1, Math.round(native.width * scale)); canvas.height = Math.max(1, Math.round(native.height * scale));
      canvas.getContext('2d').drawImage(image, native.x, native.y, native.width, native.height, 0, 0, canvas.width, canvas.height);
      const small = native.width < 16 || native.height < 32;
      $('learning-crop-info').textContent = `${native.width} × ${native.height} исходных пикселей`
        + (small ? ' — меньше минимума 16 × 32, сервер отклонит' : native.height < 128 ? ' — мало деталей' : '');
    } else {
      $('learning-crop-info').textContent = !selection ? 'Границы заданы неверно: левая меньше правой, верхняя меньше нижней.'
        : 'Нет проверенного снимка для выбора.';
    }
    const when = capturedAt();
    message('learning-captured-help', when.error || (source() === 'UPLOAD'
      ? `Время организации ${offsetOf() === 'Z' ? 'UTC+00:00' : 'UTC' + (offsetOf() || '')}; пусто — дата неизвестна.` : ''), Boolean(when.error));
    $('learning-captured-saved').textContent = source() === 'PHOTO_CROP'
      ? `Дата съёмки сохранённого снимка: ${visit?.photo_captured_at ? stamp(visit.photo_captured_at) : 'неизвестна (старый снимок)'}.` : '';
    $('learning-evidence-note').textContent = evidenceText();
    $('learning-eligibility').textContent = eligibilityText();
    sync();
  }

  // ---- analysis ----
  function canAnalyze() {
    const s = state();
    return Boolean(visit && visit.learning_available && !outage && !s.busy && !s.pending && !s.loading && !s.analyzing
      && imageReady() && selection && !capturedAt().error);
  }
  function analysisFailure(error) {
    const code = error.detail?.code;
    if (code === 'BODY_MODEL_UNAVAILABLE' || code === 'ENROLLMENT_UNAVAILABLE') return 'Модель внешности сейчас недоступна. Исправление личности, история и отмена обучения работают отдельно; повторите позже.';
    if (code === 'ENROLLMENT_BUSY') return 'Подготовка изображений занята. Повторите через несколько секунд.';
    if (code === 'VISIT_PHOTO_EXPIRED') return 'Срок хранения снимка истёк. Загрузите фотографию.';
    if (code === 'PERSON_CROP_TOO_SMALL') return `Фрагмент слишком мал: ${error.detail.width} × ${error.detail.height} px, нужно не меньше ${error.detail.min_width} × ${error.detail.min_height}.`;
    if (code === 'CAPTURED_AT_IN_FUTURE') return 'Дата съёмки не может быть в будущем.';
    if (conflictCopy[code]) return conflictCopy[code];
    return error.message || 'Не удалось подготовить подсказки. Повторите попытку.';
  }
  function evidenceRequest() {
    const when = capturedAt();
    if (when.error) { message('learning-analysis-status', when.error, true); return null; }
    if (!selection || !imageReady()) { message('learning-analysis-status', 'Выберите проверенный снимок и корректный фрагмент.', true); return null; }
    const evidence = { source: source(), crop: [...selection] };
    if (evidence.source === 'PHOTO_CROP') evidence.photo_sha256 = digests.photo;
    else if (when.value) evidence.captured_at = when.value;
    return evidence;
  }
  function formFor(metadata) {
    const body = new FormData(); body.append('metadata', JSON.stringify(metadata));
    if (metadata.source === 'UPLOAD') body.append('photo', uploadFile.blob, uploadFile.name);
    return body;
  }
  async function analyze() {
    if (!canAnalyze()) return;
    const evidence = evidenceRequest(); if (!evidence) return;
    cancelAnalysis();
    const ticket = analyzeTicket, gen = generation, controller = new AbortController();
    analyzeController = controller; analysis = null; renderCandidates();
    analyzingFlag = true; corrections.setLearningAnalyzing(true);
    message('learning-analysis-status', 'Ищем сходство… Ничего не сохраняется.');
    try {
      const data = await UI.api(`${endpoint}/learning/analyze`, { method: 'POST', body: formFor(evidence), signal: controller.signal });
      if (ticket !== analyzeTicket || gen !== generation) return;
      const stale = data.visit_id !== visitId || data.evidence?.source !== evidence.source
        || (evidence.source === 'PHOTO_CROP' && data.evidence.source_sha256 !== evidence.photo_sha256)
        || data.learning_revision < known.learning || data.gallery_revision < known.gallery
        || data.identity_revision < (visit?.identity_revision ?? 0);
      if (stale) { message('learning-analysis-status', 'Ответ устарел. Нажмите «Предложить сотрудников» ещё раз.', true); return; }
      bump('learning', data.learning_revision); bump('gallery', data.gallery_revision);
      analysis = { gen, data };
      const total = data.candidates.face.length + data.candidates.body.length;
      message('learning-analysis-status', total ? 'Подсказки готовы. Это сходство, не вероятность личности; ничего не сохранено.'
        : 'Похожих сотрудников нет. Можно выбрать сотрудника вручную. Ничего не сохранено.');
    } catch (error) {
      if (ticket !== analyzeTicket) return;
      message('learning-analysis-status', analysisFailure(error), true);
    } finally {
      if (ticket === analyzeTicket) {
        analyzeController = null;
        if (analyzingFlag) { analyzingFlag = false; corrections.setLearningAnalyzing(false); }
        renderCandidates(); paint();
      }
    }
  }
  function maybeAuto() {
    if (source() !== 'PHOTO_CROP' || !visit || visit.employee_id || !visit.learning_available || outage) return;
    if (visit.photo_status && visit.photo_status !== 'AVAILABLE') return;
    if (selection?.join() !== FULL.join() || !canAnalyze()) return;
    const key = JSON.stringify([visit.id, visit.photo_sha256, selection, visit.body_model_id ?? null, visit.face_model_id ?? null,
      visit.learning_revision ?? null, visit.gallery_revision ?? null]);
    if (key === autoKey) return;
    autoKey = key; analyze();
  }

  // ---- candidates ----
  function candidateCard(group, c) {
    const li = el('li', null, 'learning-card');
    li.append(el('p', c.name, 'learning-name'));
    li.append(el('p', `Сходство (${group === 'face' ? 'по лицу' : 'по внешности'}): ${Number(c.score).toFixed(3)}`));
    if (c.negative_score != null) li.append(el('p', `Сходство с отклонённым образцом: ${Number(c.negative_score).toFixed(3)}`));
    li.append(el('p', c.rejected ? 'Отклонён прежним решением оператора: похожий образец уже отвергнут.'
      : c.automatic ? 'Проходит порог и разницу с другими: при совпадении за сегодня может назвать человека автоматически.'
        : 'Ниже порога или неоднозначно: только подсказка, автоматического имени не будет.'));
    li.append(el('p', 'Сходство, не вероятность личности', 'muted'));
    for (const id of c.rejection_feedback_ids || []) {
      const link = el('a', 'Показать решение оператора, из-за которого карточка отклонена'); link.href = `#learning-example-${id}`;
      link.onclick = (event) => { event.preventDefault(); showFeedback(id, c.employee_id); };
      li.append(link);
    }
    const row = el('div', null, 'form-actions');
    const uploaded = source() === 'UPLOAD', target = { id: c.employee_id, name: c.name, revision: c.employee_revision };
    const yes = button(`Это ${c.name} — ${uploaded ? 'сохранить образец сотрудника' : 'подтвердить визит и обучить'}`, () => decide('ACCEPT', target));
    yes.className = 'btn primary';
    const no = button(`Это не ${c.name}`, () => decide('REJECT', target));
    cardButtons.push(yes, no); row.append(yes, no); li.append(row);
    return li;
  }
  function renderCandidates() {
    const box = $('learning-candidates'); cardButtons = []; box.replaceChildren();
    if (!analysisValid()) {
      box.append(el('p', 'Нажмите «Предложить сотрудников»: подсказки появятся только для выбранной области.', 'muted')); return;
    }
    const { candidates } = analysis.data;
    for (const [group, title] of [['face', 'По лицу'], ['body', 'По внешности']]) {
      const section = el('div', null, 'learning-group');
      section.append(el('h3', title));
      const items = candidates[group] || [];
      if (!items.length) section.append(el('p', 'Кандидатов нет. Выберите сотрудника вручную ниже.', 'muted'));
      else { const list = el('ul', null, 'learning-cards'); for (const c of items) list.append(candidateCard(group, c)); section.append(list); }
      box.append(section);
    }
  }

  // ---- employees ----
  function optionsFor() {
    const select = $('learning-employee'), chosen = select.value;
    const search = $('learning-search').value.trim().toLocaleLowerCase();
    select.replaceChildren(new Option('Выберите сотрудника вручную', ''));
    for (const row of employees) {
      if (String(row.id) === chosen || row.name.toLocaleLowerCase().includes(search)) select.append(new Option(row.name + (row.active ? '' : ' (неактивен)'), String(row.id)));
    }
    select.value = chosen;
  }
  async function loadEmployees() {
    $('learning-employees-retry').hidden = true;
    try { employees = await UI.api('/api/employees', { cache: 'no-store' }); optionsFor(); }
    catch (error) { message('learning-employee-status', 'Список сотрудников не загружен. ' + error.message, true); $('learning-employees-retry').hidden = false; }
    sync();
  }
  async function selectEmployee(id) {
    const ticket = ++employeeTicket;
    selected = null; employeeLoading = Boolean(id); lists.employee.ticket++; $('learning-employee-more').hidden = true; sync();
    message('learning-employee-status', id ? 'Загружаем версию сотрудника…' : '');
    if (!id) { loadFeedback('employee'); return; }
    try {
      const row = await UI.api(`/api/employees/${id}`, { cache: 'no-store' });
      if (ticket !== employeeTicket) return;
      selected = { id: row.id, name: row.name, revision: row.revision, active: row.active };
      message('learning-employee-status', `${row.name} · образцов лица ${row.faces_count}/15 · версия ${row.revision}${row.active ? '' : ' · неактивен: образец сохранится, но не будет участвовать, пока сотрудник не активирован'}`);
      await loadFeedback('employee');
    } catch (error) {
      if (ticket === employeeTicket) message('learning-employee-status', (error.detail?.code === 'EMPLOYEE_NOT_FOUND' ? 'Сотрудник не найден. ' : '') + error.message + ' Выберите сотрудника снова.', true);
    } finally { if (ticket === employeeTicket) { employeeLoading = false; sync(); } }
  }

  // ---- decisions ----
  function audit() {
    const actor = $('learning-actor'), reason = $('learning-reason');
    for (const input of [actor, reason]) {
      if (!input.value.trim()) { input.focus(); input.reportValidity(); message('learning-status', 'Укажите оператора и причину.', true); return null; }
    }
    return { actor: actor.value.trim(), reason: reason.value.trim(), request_id: UI.newRequestId() };
  }
  function canDecide() {
    const s = state();
    return Boolean(visit && visit.learning_available && !outage && analysisValid() && $('learning-person-confirm').checked
      && !s.busy && !s.pending && !s.loading && !s.analyzing);
  }
  function decide(action, target) {
    if (!canDecide() || !target) return;
    const human = audit(); if (!human) return;
    const data = analysis.data, when = capturedAt();
    if (when.error) { message('learning-status', when.error, true); return; }
    const metadata = { ...human, source: source(), crop: [...selection], action, employee_id: target.id,
      expected_employee_revision: target.revision, expected_identity_revision: data.identity_revision,
      expected_learning_revision: data.learning_revision, expected_gallery_revision: data.gallery_revision,
      expected_image_sha256: data.evidence.image_sha256, expected_body_model_id: data.body_model_id,
      expected_face_model_id: data.face_model_id ?? null, person_confirmed: true };
    if (metadata.source === 'PHOTO_CROP') metadata.photo_sha256 = digests.photo;
    else if (when.value) metadata.captured_at = when.value;
    $('learning-links').replaceChildren();
    corrections.mutateLearning({ url: `${endpoint}/learning/feedback`, options: { body: formFor(metadata) },
      meta: { kind: 'feedback', action, employee_id: target.id, name: target.name } });
  }
  function revoke(item, restore) {
    const s = state();
    if (s.busy || s.pending || s.loading) return;
    const human = audit(); if (!human) return;
    cancelAnalysis();
    $('learning-links').replaceChildren();
    corrections.mutateLearning({ url: `/api/recognition-learning/${encodeURIComponent(item.feedback_id)}/revoke`,
      options: { headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ...human, expected_identity_revision: restore ? visit.identity_revision : null }) },
      meta: { kind: 'revoke', id: item.feedback_id } });
  }
  function afterWrite() {
    evidenceChanged();
    loadFeedback('visit');
    if (selected) selectEmployee(selected.id);
  }
  function acceptCopy(result) {
    const parts = ['Внешность сохранена.', faceCopy[result.face_status] || ''];
    parts.push(result.daily_anchor?.today
      ? 'Образец может служить опорой для автоматического узнавания сегодня.'
      : 'Сегодняшней опорой он не станет (дата снимка не сегодня или неизвестна): только подсказки.');
    if (result.source === 'UPLOAD') parts.push('Личность в этом наблюдении не изменена.');
    else if (result.identity) parts.push('Личность в этом наблюдении записана.');
    if (result.revoked_feedback_ids?.length) parts.push('Прежнее обучение, связанное с этим наблюдением, отменено.');
    if (result.employee_active === false) parts.push('Сотрудник неактивен и не будет участвовать, пока не активирован.');
    parts.push('Прошлые часы не изменены.');
    return parts.filter(Boolean).join(' ');
  }
  function rejectCopy(result) {
    const modalities = (result.learned_modalities || []).map((m) => modalityCopy[m] || m);
    const parts = [`Отклонено: исключено сходство по: ${modalities.length ? modalities.join(', ') : 'внешности'}. Это не запрет на сотрудника, только похожий вид.`,
      faceCopy[result.face_status] && result.face_status !== 'ADDED' ? faceCopy[result.face_status] : ''];
    if (result.source !== 'UPLOAD' && result.identity) parts.push('Личность в этом наблюдении обновлена.');
    parts.push('Прошлые часы не изменены.');
    return parts.filter(Boolean).join(' ');
  }
  function onMutationResult(result, frozen) {
    bump('learning', result.learning_revision); bump('gallery', result.gallery_revision);
    $('learning-retry').hidden = true;
    if (frozen?.meta?.kind === 'revoke' || result.identity_outcome) {
      const outcome = { RESTORED: 'Личность восстановлена.', KEPT: 'Более позднее решение сохранено.', SAMPLE_ONLY: 'Удалён только образец.' }[result.identity_outcome] || 'Обучение удалено.';
      const face = result.removed_face_ids?.length ? ' Образец лица удалён из галереи.'
        : result.face_removal_skipped_ids?.length ? ' Образец лица оставлен: он уже изменён.' : '';
      message('learning-status', `${outcome}${face} Прошлые часы не изменены.`);
    } else message('learning-status', result.action === 'REJECT' ? rejectCopy(result) : acceptCopy(result));
    $('learning-person-confirm').checked = false;
    afterWrite();
  }
  function onMutationError(error, { uncertain } = {}) {
    const code = error.detail?.code;
    $('learning-retry').hidden = !state().pending;
    if (uncertain) {
      if (code === 'ENROLLMENT_BUSY') message('learning-status', 'Подготовка изображений занята. Повторите тот же запрос: данные и номер запроса зафиксированы.', true);
      sync(); return;
    }
    $('learning-person-confirm').checked = false;
    const links = $('learning-links'); links.replaceChildren();
    const addLink = (text, href, handler) => {
      const item = el('li'), link = el('a', text); link.href = href;
      if (handler) link.onclick = (event) => { event.preventDefault(); handler(); };
      item.append(link); links.append(item);
    };
    if (error.status === 503 && ['BODY_MODEL_UNAVAILABLE', 'ENROLLMENT_UNAVAILABLE'].includes(code)) {
      outage = true;
      message('learning-status', 'Обучение по человеку сейчас недоступно: модель не работает. Ничего не сохранено. Исправление личности, история и отмена обучения доступны; можно обновить наблюдение после восстановления.', true);
    } else if (code === 'LEARNING_DUPLICATE') {
      message('learning-status', 'Такое решение по этому изображению уже сохранено. Новый образец не добавлен.', true);
      const id = error.detail.feedback_id;
      if (id) addLink('Показать существующее решение', `#learning-example-${id}`,
        () => showFeedback(id, error.detail.employee_id));
    } else if (code === 'LEARNING_CONFLICT') {
      message('learning-status', 'Решение противоречит прежнему по этому изображению. Сначала удалите прежнее обучение.', true);
      for (const id of error.detail.feedback_ids || []) addLink('Показать противоречащее решение',
        `#learning-example-${id}`, () => showFeedback(id, error.detail.feedback_owners?.[id]));
    } else if (code === 'LEARNING_LIMIT_REACHED') {
      message('learning-status', `У сотрудника уже ${error.detail.limit} образцов этого вида. Удалите лишние вручную: автоматически ничего не заменяется.`, true);
      if (error.detail.employee_id) {
        $('learning-search').value = '';
        optionsFor();
        $('learning-employee').value = String(error.detail.employee_id);
        selectEmployee($('learning-employee').value);
      }
      addLink('Управлять образцами сотрудника', '#learning-employee-examples', () => {
        $('learning-employee-examples').closest('details').open = true;
        $('learning-employee-examples').scrollIntoView();
      });
    } else if (code === 'LEARNING_FEEDBACK_REVOKED') {
      message('learning-status', 'Это обучение уже удалено. История обновлена.', true); loadFeedback('visit'); loadFeedback('employee');
    } else if (code === 'VISIT_PHOTO_EXPIRED') {
      message('learning-status', 'Срок хранения снимка истёк. Загрузите фотографию; личность можно исправить отдельно.', true);
    } else if (code === 'PERSON_CROP_TOO_SMALL' || code === 'CAPTURED_AT_IN_FUTURE' || code === 'EMPLOYEE_NOT_FOUND' || code === 'LEARNING_FEEDBACK_NOT_FOUND') {
      message('learning-status', (code === 'EMPLOYEE_NOT_FOUND' ? 'Сотрудник не найден. ' : '') + analysisFailure(error), true);
    } else if (error.status === 409) {
      message('learning-status', conflictCopy[code] || staleCopy, true);
      if (code === 'EMPLOYEE_REVISION_CONFLICT' && selected) selectEmployee(selected.id);
    } else message('learning-status', (error.message || 'Операция не выполнена.') + ' Ничего не сохранено.', true);
    if (error.status === 409) { generation++; analysis = null; renderCandidates(); }
    sync();
  }

  // ---- feedback history / selected employee examples ----
  function thumbSet() {
    const urls = new Set(), queue = []; let active = 0, controller = new AbortController();
    const pump = () => {
      while (active < 4 && queue.length) { active++; queue.shift()().finally(() => { active--; pump(); }); }
    };
    return {
      add(task) { queue.push(task); pump(); },
      signal: () => controller.signal,
      track: (url) => urls.add(url),
      release() { controller.abort(); controller = new AbortController(); queue.length = 0; for (const url of urls) URL.revokeObjectURL(url); urls.clear(); },
    };
  }
  const lists = {
    visit: { target: 'learning-history', more: 'learning-history-more', prefix: 'learning-history-', cursor: null, ticket: 0, thumbs: thumbSet(), buttons: [] },
    employee: { target: 'learning-employee-examples', more: 'learning-employee-more', prefix: 'learning-example-', cursor: null, ticket: 0, thumbs: thumbSet(), buttons: [] },
  };
  function thumb(set, item) {
    const box = el('div', null, 'learning-thumb'), note = el('p', null, 'muted'), image = el('img');
    image.hidden = true; image.alt = `Образец обучения: ${item.employee_name}`;
    box.append(image, note);
    if (!item.photo_url || !item.photo_url.startsWith('/api/recognition-learning/')) {
      note.textContent = item.active ? 'Изображение не сохранено' : 'Изображение удалено при отмене'; return box;
    }
    const signal = set.signal(), retry = button('Повторить загрузку', () => set.add(load));
    retry.hidden = true; box.append(retry);
    async function load() {
      retry.hidden = true; note.textContent = 'Загружаем…';
      try {
        const response = await fetch(item.photo_url, { cache: 'no-store', signal });
        if (response.status === 410) { note.textContent = 'Изображение удалено'; return; }
        if (response.status === 404) { note.textContent = 'Изображение не найдено'; return; }
        if (!response.ok) throw new Error('image');
        const blob = await response.blob();
        if (signal.aborted) return;
        const url = URL.createObjectURL(blob); set.track(url);
        image.src = url; image.hidden = false; note.textContent = '';
      } catch { if (!signal.aborted) { note.textContent = 'Не удалось загрузить изображение.'; retry.hidden = false; } }
    }
    set.add(load);
    return box;
  }
  function feedbackRow(list, item) {
    const li = el('li'); li.id = list.prefix + item.feedback_id;
    li.append(thumb(list.thumbs, item));
    const modalities = (item.examples || []).map((e) => `${modalityCopy[e.modality.toUpperCase()] || e.modality} ${e.polarity === 'POSITIVE' ? '+' : '−'}${e.compatible ? '' : ' (другая модель, не участвует)'}`).join(', ');
    li.append(el('p', `${item.action === 'ACCEPT' ? 'Подтверждено' : 'Отклонено'}: ${item.employee_name}${item.employee_active ? '' : ' (неактивен)'} · ${item.source === 'UPLOAD' ? 'загруженный образец сотрудника' : 'снимок наблюдения'}`));
    li.append(el('p', `${item.active ? 'Активно' : 'Отозвано'} · образцы: ${modalities || 'нет'}${item.face_status ? ' · лицо: ' + item.face_status : ''}`));
    li.append(el('p', `Дата съёмки: ${item.captured_at ? stamp(item.captured_at) : 'неизвестна'}${item.daily_anchor?.today ? ' · опора на сегодня' : ''} · записано ${stamp(item.created_at)} · ${item.actor} · ${item.reason}`));
    if (item.revocation) li.append(el('p', `Удалено ${stamp(item.revocation.revoked_at)} · ${item.revocation.revoked_by} · ${item.revocation.reason}`));
    if (item.visit_id !== visitId) { const link = el('a', 'Исходное наблюдение'); link.href = `/global/visits/${encodeURIComponent(item.visit_id)}`; li.append(link); }
    if (item.active) {
      const restore = item.visit_id === visitId && item.source === 'PHOTO_CROP' && item.identity_revision != null;
      const undo = button(restore ? 'Отменить решение' : 'Удалить обучение', () => revoke(item, restore));
      list.buttons.push(undo); li.append(undo);
    }
    return li;
  }
  async function loadFeedback(kind, more = false) {
    const list = lists[kind], target = $(list.target), moreButton = $(list.more);
    const employeeId = kind === 'employee' ? selected?.id : null;
    if (!more) { list.thumbs.release(); list.buttons = []; list.cursor = null; }
    if (kind === 'employee' && !employeeId) {
      list.ticket++; target.replaceChildren(); moreButton.hidden = true;
      $('learning-counts').textContent = ''; sync(); return;
    }
    const ticket = ++list.ticket, params = new URLSearchParams();
    if (kind === 'employee') params.set('employee_id', String(employeeId));
    params.set('limit', '50'); if (more && list.cursor) params.set('cursor', list.cursor);
    try {
      const result = await UI.api(kind === 'visit' ? `${endpoint}/learning/history?${params}` : `/api/recognition-learning/examples?${params}`, { cache: 'no-store' });
      if (ticket !== list.ticket) return;
      if (!more) target.replaceChildren();
      const visible = more ? new Set(Array.from(target.children, node => node.id)) : null;
      for (const item of result.items || []) {
        if (!visible?.has(list.prefix + item.feedback_id)) target.append(feedbackRow(list, item));
      }
      if (!more && !(result.items || []).length) target.textContent = 'Записей обучения пока нет.';
      list.cursor = result.next_cursor || null; moreButton.hidden = !list.cursor;
      const c = result.counts || {};
      const text = `Образцы сейчас: внешность ${count(c.body_positive)} (отклонено ${count(c.body_negative)}), лицо ${count(c.face_positive)} (отклонено ${count(c.face_negative)}), опорных внешностей за сегодня ${count(c.today_body)}.`;
      if (kind === 'employee') $('learning-counts').textContent = (selected ? selected.name + ': ' : '') + text;
      else $('learning-history-counts').textContent = 'В этом наблюдении: ' + text;
    } catch (error) {
      if (ticket !== list.ticket) return;
      if (!more) { target.replaceChildren(); list.thumbs.release(); }
      const row = el('li', 'История обучения не загружена. ' + error.message + ' '); row.append(button('Повторить загрузку', () => loadFeedback(kind, more))); target.append(row);
    }
    sync();
  }
  async function showFeedback(id, employeeId) {
    if (employeeId && (!selected || selected.id !== employeeId)) {
      $('learning-search').value = ''; optionsFor(); $('learning-employee').value = String(employeeId);
      await selectEmployee(String(employeeId));
    }
    const list = lists.employee, target = $(list.target);
    let node = $(`learning-example-${id}`) || $(`learning-history-${id}`);
    if (!node && employeeId && selected?.id === employeeId) {
      const ticket = list.ticket;
      try {
        const result = await UI.api(`/api/recognition-learning/examples?${new URLSearchParams({
          employee_id: String(employeeId), feedback_id: id,
        })}`, { cache: 'no-store' });
        if (ticket !== list.ticket || selected?.id !== employeeId) return;
        const item = result.items.find((row) => row.feedback_id === id);
        if (item) {
          node = $(`learning-example-${id}`);
          if (!node) {
            node = feedbackRow(list, item);
            target.prepend(node);
          }
          sync();
        }
      } catch (error) {
        message('learning-status', 'Не удалось открыть пример. ' + error.message, true);
        return;
      }
    }
    if (!node) {
      message('learning-status', 'Пример больше не доступен в этой истории.', true);
      return;
    }
    const details = node.closest('details'); if (details) details.open = true;
    node.scrollIntoView();
    node.setAttribute('tabindex', '-1');
    node.focus();
  }

  // ---- state ----
  function sync() {
    const s = state(), locked = s.busy || s.pending || s.loading;
    const confirmed = canDecide();
    $('learning-fields').disabled = locked || !visit;
    $('learning-decision').disabled = locked || !visit;
    const uploading = source() === 'UPLOAD';
    $('learning-file').disabled = !uploading; $('learning-captured-at').disabled = !uploading;
    $('learning-analyze').disabled = !canAnalyze();
    $('learning-person-confirm').disabled = locked || !analysisValid();
    $('learning-employee').disabled = employeeLoading; $('learning-search').disabled = employeeLoading;
    for (const id of ['learning-manual-accept', 'learning-manual-reject']) $(id).disabled = !confirmed || !selected || employeeLoading;
    $('learning-manual-accept').textContent = selected ? `Это ${selected.name} — ${uploading ? 'сохранить образец сотрудника' : 'подтвердить визит и обучить'}` : 'Подтвердить и обучить';
    $('learning-manual-reject').textContent = selected ? `Это не ${selected.name}` : 'Это не выбранный сотрудник';
    for (const node of cardButtons) node.disabled = !confirmed;
    for (const list of Object.values(lists)) for (const node of list.buttons) node.disabled = locked;
    $('learning-retry').hidden = !s.pending; $('learning-retry').disabled = s.busy || s.loading;
    const unavailable = visit && (!visit.learning_available || outage);
    $('learning-unavailable').hidden = !unavailable;
    if (unavailable) $('learning-unavailable').textContent = outage ? 'Обучение по человеку недоступно: модель не работает. История, исправление личности и отмена обучения доступны.'
      : `Обучение по человеку недоступно${visit.learning_unavailable_reason ? ': ' + (reasons[visit.learning_unavailable_reason] || visit.learning_unavailable_reason) : ''}. История, исправление личности и отмена обучения доступны.`;
  }
  function update(value, imageDigests, { evidenceOnly = false } = {}) {
    visit = value; digests = { ...imageDigests };
    bump('learning', visit.learning_revision); bump('gallery', visit.gallery_revision);
    if (visit.learning_available) { if (!evidenceOnly) outage = false; }
    const photo = savedReady() ? digests.photo : null;
    if (photo !== savedDigest) {
      savedDigest = photo;
      if (source() === 'PHOTO_CROP') { evidenceChanged(); resetSelection(); }
    }
    if (!visit.learning_available || outage) { cancelAnalysis(); analysis = null; }
    showSourceImage();
    if (source() === 'PHOTO_CROP' && !photo) message('learning-analysis-status', 'Сохранённый снимок недоступен или срок хранения истёк. Загрузите фотографию.', true);
    renderCandidates(); paint();
    if (!evidenceOnly) loadFeedback('visit');
    maybeAuto();
  }
  function invalidate() {
    const active = Boolean(analysis) || analyzingFlag;
    evidenceChanged();
    if (active) message('learning-analysis-status', 'Данные изменились. Предложите сотрудников заново.');
    loadFeedback('visit');
    if (selected) selectEmployee(selected.id);
    paint();
  }
  async function loadSource() {
    uploadTicket++; const ticket = uploadTicket;
    if (pendingUpload) URL.revokeObjectURL(pendingUpload.url);
    pendingUpload = null; uploadFile = null; setImage(null);
    evidenceChanged(); resetSelection();
    const file = $('learning-file').files[0];
    if (!file) { paint(); return; }
    if (file.size > MAX_UPLOAD || !['image/jpeg', 'image/png'].includes(file.type)) {
      message('learning-analysis-status', 'Нужен JPEG или PNG размером до 15 МиБ.', true); paint(); return;
    }
    try {
      // The bytes are read once; preview, analysis and every retry use this immutable Blob.
      const blob = new Blob([await file.arrayBuffer()], { type: file.type });
      if (ticket !== uploadTicket) return;
      pendingUpload = { blob, name: file.name, url: URL.createObjectURL(blob) };
      setImage(pendingUpload.url);
      message('learning-analysis-status', 'Фотография открывается…');
    } catch (error) {
      if (ticket === uploadTicket) message('learning-analysis-status', 'Не удалось прочитать фотографию. ' + error.message, true);
    }
    if (ticket === uploadTicket) paint();
  }

  $('learning-source').onchange = () => {
    if (source() !== 'UPLOAD') { dropUpload(); $('learning-file').value = ''; $('learning-captured-at').value = ''; }
    $('learning-upload').hidden = source() !== 'UPLOAD';
    evidenceChanged(); resetSelection(); showSourceImage(); paint();
  };
  $('learning-file').onchange = loadSource;
  $('learning-captured-at').oninput = () => { evidenceChanged(); paint(); };
  bounds.forEach((input) => { input.oninput = () => {
    selection = bounds.every((node) => node.value !== '') ? crop.normalizeRect(...bounds.map((node) => Number(node.value))) : null;
    evidenceChanged(); paint();
  }; });
  const image = $('learning-image');
  image.onload = () => {
    if (pendingUpload && shownSrc === pendingUpload.url && !uploadFile) {
      if (image.naturalWidth * image.naturalHeight > 20000000) {
        message('learning-analysis-status', 'Изображение больше 20 мегапикселей. Выберите меньший исходник.', true); dropUpload(); paint(); return;
      }
      uploadFile = Object.freeze({ ...pendingUpload });
      message('learning-analysis-status', 'Фотография загружена. Выделите одного человека и запросите подсказки.');
    }
    paint(); maybeAuto();
  };
  image.onerror = () => {
    if (pendingUpload && shownSrc === pendingUpload.url) {
      message('learning-analysis-status', 'Не удалось открыть фотографию. Выберите другой JPEG или PNG.', true); dropUpload();
    }
    paint();
  };
  const stage = $('learning-stage');
  const pointer = (event) => crop.pointInImage(event.clientX, event.clientY, image.getBoundingClientRect());
  stage.onpointerdown = (event) => {
    if (!imageReady() || event.button !== 0 || $('learning-fields').disabled) return;
    event.preventDefault(); drag = pointer(event); stage.setPointerCapture(event.pointerId);
  };
  function move(event) {
    if (!drag) return;
    selection = crop.normalizeRect(...drag, ...pointer(event));
    bounds.forEach((input, index) => { input.value = selection ? String(selection[index]) : ''; });
    evidenceChanged(); paint();
  }
  stage.onpointermove = move;
  stage.onpointerup = (event) => { move(event); drag = null; if (stage.hasPointerCapture(event.pointerId)) stage.releasePointerCapture(event.pointerId); };
  stage.onpointercancel = () => { drag = null; };
  $('learning-analyze').onclick = analyze;
  $('learning-person-confirm').onchange = sync;
  $('learning-search').oninput = () => { optionsFor(); sync(); };
  $('learning-search').onkeydown = (event) => { if (event.key === 'Enter') event.preventDefault(); };
  $('learning-employee').onchange = () => selectEmployee($('learning-employee').value);
  $('learning-employees-retry').onclick = loadEmployees;
  $('learning-manual-accept').onclick = () => decide('ACCEPT', selected);
  $('learning-manual-reject').onclick = () => decide('REJECT', selected);
  $('learning-retry').onclick = () => corrections.mutateLearning();
  $('learning-history-more').onclick = () => loadFeedback('visit', true);
  $('learning-employee-more').onclick = () => loadFeedback('employee', true);
  window.addEventListener('pagehide', (event) => {
    if (event.persisted) return;
    cancelAnalysis(); drag = null;
    for (const list of Object.values(lists)) list.thumbs.release();
    if (pendingUpload) URL.revokeObjectURL(pendingUpload.url);
    pendingUpload = null; uploadFile = null;
  });
  window.JournalLearning = { update, sync, invalidate, onMutationResult, onMutationError };
  $('learning-upload').hidden = true;
  renderCandidates(); sync(); loadEmployees();
})();
