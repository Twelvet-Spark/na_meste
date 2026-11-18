// Camera directory: stable card nodes, safe text, explicit setup and no automatic recording.
(() => {
  'use strict';
  const ui = window.OperatorUI;
  const $ = (id) => document.getElementById(id);
  const tablist = $('camera-tabs');
  const store = $('camera-card-store');
  const bulk = $('camera-bulk');
  const summary = document.getElementById('summary');
  const empty = document.getElementById('empty');
  const retry = document.getElementById('camera-retry');
  const cards = new Map();
  const selected = new Set();
  const tabNodes = new Map();
  let cameras = [], workplaces = [], tabs = null, sequence = 0, bulkBusy = false;
  let previewRequests = 0;
  const states = { ONLINE: 'Онлайн', OFFLINE: 'Офлайн', CONNECTING: 'Подключение', RECONNECTING: 'Переподключение', ERROR: 'Ошибка' };
  function node(tag, className, text) {
    const value = document.createElement(tag); value.className = className;
    if (text) value.textContent = text;
    return value;
  }
  function gateLabel(role) {
    return ({ BOTH: 'Проход: вход и выход', ENTRY: 'Проход: только вход', EXIT: 'Проход: только выход' })[role] || 'Проход';
  }
  function directoryTabs(items, places) {
    const known = new Map(places.map((place) => [place.id, place]));
    const counts = new Map();
    let unassigned = 0;
    for (const camera of items) {
      const profile = camera.profile;
      if (profile.workplace_id == null) { unassigned++; continue; }
      counts.set(profile.workplace_id, (counts.get(profile.workplace_id) || 0) + 1);
      if (!known.has(profile.workplace_id)) known.set(profile.workplace_id, {
        id: profile.workplace_id, name: profile.workplace_name, active: profile.workplace_active,
      });
    }
    const groups = [...known.values()].filter((place) => place.active || counts.has(place.id))
      .sort((a, b) => a.name.localeCompare(b.name, 'ru'))
      .map((place) => ({ name: `wp-${place.id}`, label: place.name + (place.active ? '' : ' (архив)'), count: counts.get(place.id) || 0 }));
    return [{ name: 'all', label: 'Все', count: items.length }, ...groups,
      { name: 'unassigned', label: 'Без рабочего места', count: unassigned }];
  }
  function tabCameras(items, name) {
    if (name === 'all') return items;
    return items.filter((camera) => name === 'unassigned'
      ? camera.profile.workplace_id == null : camera.profile.workplace_id === Number(name.slice(3)));
  }
  function placeNode(parent, element, index) {
    if (parent.children[index] !== element) parent.insertBefore(element, parent.children[index] || null);
  }
  function renderTabs() {
    const groups = directoryTabs(cameras, workplaces);
    const wanted = new Set(groups.map((group) => group.name));
    for (const [name, tab] of tabNodes) {
      if (!wanted.has(name)) { tab.button.remove(); tabNodes.delete(name); }
    }
    groups.forEach((group, index) => {
      let tab = tabNodes.get(group.name);
      if (!tab) {
        const button = node('button', ''); button.type = 'button'; button.id = `camera-tab-${group.name}`;
        button.dataset.tab = group.name; button.setAttribute('role', 'tab');
        button.setAttribute('aria-controls', 'camera-directory-panel');
        const label = node('span', ''), count = node('span', 'tab-count');
        button.append(label, count); tab = { button, label, count }; tabNodes.set(group.name, tab);
      }
      tab.label.textContent = group.label; tab.count.textContent = `(${group.count})`;
      placeNode(tablist, tab.button, index);
    });
    if (!tabs) tabs = ui.tabs(tablist, { onChange: (name) => renderGroups(name) });
    else tabs.refresh();
    renderGroups(tabs.current());
  }
  function renderGroups(name) {
    const visible = tabCameras(cameras, name);
    const wanted = new Set(visible.map((camera) => camera.id));
    for (const [id, card] of cards) {
      card.workplace.hidden = name !== 'all';
      if (!wanted.has(id) && card.element.parentNode !== store) store.append(card.element);
    }
    for (const [zone, key] of [['ENTRANCE', 'entrances'], ['INTERIOR', 'interior']]) {
      const items = visible.filter((camera) => camera.profile.zone === zone);
      $(`camera-${key}`).hidden = items.length === 0;
      $(`camera-${key}-count`).textContent = `(${items.length})`;
      items.forEach((camera, index) => placeNode($(`camera-${key}-grid`), cards.get(camera.id).element, index));
    }
    $('camera-tab-empty').hidden = !cameras.length || visible.length > 0;
  }
  function renderWorkplaces() {
    const select = $('camera-bulk-workplace');
    const places = workplaces.filter((place) => place.active).sort((a, b) => a.name.localeCompare(b.name, 'ru'));
    const signature = JSON.stringify(places.map(({ id, name }) => [id, name]));
    if (select.dataset.options === signature) return;
    const value = select.value;
    select.replaceChildren(new Option('Без рабочего места', ''), ...places.map((place) => new Option(place.name, String(place.id))));
    select.value = places.some((place) => String(place.id) === value) ? value : '';
    select.dataset.options = signature;
  }
  function renderSelection() {
    bulk.hidden = selected.size === 0;
    $('camera-selected-count').textContent = `Выбрано: ${selected.size}`;
    for (const [id, card] of cards) {
      card.selection.checked = selected.has(id); card.selection.disabled = bulkBusy;
      card.element.classList.toggle('is-selected', selected.has(id));
    }
  }
  function newCard(camera) {
    const element = node('article', 'cam-card');
    const selectionLabel = node('label', 'cam-card-selection');
    const selection = node('input', ''); selection.type = 'checkbox';
    selectionLabel.append(selection, node('span', '', 'Выбрать'));
    selection.addEventListener('change', () => {
      if (bulkBusy) return;
      if (selection.checked && selected.size >= 200) {
        selection.checked = false;
        $('camera-bulk-result').textContent = 'Можно выбрать не более 200 камер за один раз.';
        return;
      }
      if (selection.checked) selected.add(camera.id); else selected.delete(camera.id);
      renderSelection();
    });
    const link = node('a', 'camera-preview'); link.href = `/cameras/${camera.id}`;
    const image = node('img', ''); image.alt = `Предпросмотр: ${camera.name}`; image.hidden = true;
    const unavailable = node('span', 'muted', 'Предпросмотр недоступен');
    const boxes = node('div', 'boxes-overlay'); boxes.setAttribute('aria-hidden', 'true');
    link.append(image, unavailable, boxes);
    image.addEventListener('load', () => {
      if (card.camera.enabled && card.camera.status === 'ONLINE') { image.hidden = false; unavailable.hidden = true; }
    });
    image.addEventListener('error', () => { image.hidden = true; unavailable.hidden = false; card.boxes.replaceChildren(); });
    const head = node('div', 'cam-card-head');
    const title = node('a', 'cam-name'); title.href = link.href;
    const status = node('span', 'status offline'); head.append(title, status);
    const meta = node('p', 'cam-meta muted');
    const people = node('p', 'cam-meta');
    const recording = node('span', 'p-badge');
    const badges = node('div', 'cam-card-badges');
    const zone = node('span', 'p-badge'), ai = node('span', 'p-badge'), timeTracking = node('span', 'p-badge', 'Учёт времени'), attendance = node('span', 'p-badge');
    badges.append(zone, ai, timeTracking, attendance);
    const workplace = node('p', 'cam-meta muted');
    const result = node('p', 'cam-card-result'); result.hidden = true; result.setAttribute('role', 'status');
    const actions = node('div', 'cam-card-actions');
    const edit = node('button', 'btn ghost', 'Изменить'); edit.type = 'button';
    edit.addEventListener('click', () => window.CameraSetup.openEdit(camera.id, edit));
    const remove = node('button', 'btn danger', 'Удалить'); remove.type = 'button';
    const reconnect = node('button', 'btn ghost', 'Переподключить'); reconnect.type = 'button';
    const card = { element, title, status, meta, people, recording, image, unavailable, selection, boxes,
      zone, ai, timeTracking, attendance, workplace, reconnect, result, camera, previewAt: 0, busy: false,
      previewAbort: null, imageUrl: null };
    reconnect.addEventListener('click', async () => {
      if (card.busy) return;
      card.busy = true; ui.setBusy(actions, true); result.hidden = false;
      result.textContent = 'Переподключение…'; result.classList.remove('error');
      try {
        await ui.api(`/api/cameras/${camera.id}/reconnect`, { method: 'POST' });
        result.textContent = 'Переподключение запрошено.'; await loadCameras();
      } catch (error) { result.textContent = error.message; result.classList.add('error'); }
      finally { card.busy = false; ui.setBusy(actions, false); reconnect.disabled = !card.camera.enabled; }
    });
    remove.addEventListener('click', async () => {
      if (card.busy) return;
      if (!window.confirm(`Удалить камеру «${card.camera.name}»? Камеру с историей можно только отключить.`)) return;
      card.busy = true; ui.setBusy(actions, true);
      try { await ui.api(`/api/cameras/${camera.id}`, { method: 'DELETE' }); await loadCameras(); }
      catch (error) { summary.textContent = `Удаление не выполнено. ${error.message}`; retry.hidden = false; }
      finally { card.busy = false; ui.setBusy(actions, false); reconnect.disabled = !card.camera.enabled; }
    });
    actions.append(reconnect, edit, remove);
    element.append(selectionLabel, link, head, workplace, badges, meta, people, recording, actions, result);
    return card;
  }
  function updateCard(card, camera) {
    card.camera = camera; card.title.textContent = camera.name;
    card.selection.setAttribute('aria-label', `Выбрать ${camera.name}`);
    card.zone.textContent = camera.profile.zone === 'ENTRANCE' ? 'Вход' : 'Внутренняя';
    card.ai.textContent = !camera.profile.ai_enabled ? 'AI выкл'
      : camera.profile.ai_fps == null ? 'AI' : `AI ${camera.profile.ai_fps.toLocaleString('ru-RU')} к/с`;
    card.timeTracking.hidden = !camera.profile.time_tracking;
    card.attendance.hidden = !camera.attendance.enabled;
    card.attendance.textContent = gateLabel(camera.attendance.role);
    card.workplace.textContent = camera.profile.workplace_name
      ? `${camera.profile.workplace_name}${camera.profile.workplace_active ? '' : ' (архив)'}` : 'Без рабочего места';
    card.reconnect.disabled = card.busy || !camera.enabled;
    card.status.textContent = states[camera.status] || 'Неизвестно';
    card.status.className = `status ${Object.hasOwn(states, camera.status) ? camera.status.toLowerCase() : 'offline'}`;
    card.meta.textContent = `${camera.nvr_host} · канал ${camera.channel} · ${camera.stream_type === 'sub' ? 'дополнительный' : 'основной'}${camera.resolution ? ` · ${camera.resolution}` : ''}`;
    card.people.textContent = `В кадре: ${camera.people_count || 0} · сотрудники: ${camera.known_count || 0} · неизвестные: ${camera.unknown_count || 0}`;
    const recordingStates = { STARTING: 'Запись запускается', RECORDING: 'Идёт запись', RECONNECTING: 'Запись: переподключение', STOPPING: 'Запись завершается' };
    card.recording.textContent = camera.recording_available === false ? 'Запись: состояние недоступно'
      : recordingStates[camera.recording?.status] || 'Запись не ведётся';
    card.image.alt = `Предпросмотр: ${camera.name}`;
    if (!camera.enabled || camera.status !== 'ONLINE') {
      card.previewAbort?.abort();
      card.boxes.replaceChildren();
      card.image.hidden = true; card.unavailable.hidden = false;
      card.unavailable.textContent = camera.enabled ? 'Нет свежего видео' : 'Камера отключена';
    }
  }
  function previewVisible(card) {
    if (document.hidden || !card.element.isConnected || !card.element.getClientRects().length) return false;
    const rect = card.element.getBoundingClientRect();
    return rect.bottom > 0 && rect.top < window.innerHeight && rect.right > 0 && rect.left < window.innerWidth;
  }
  function disposePreview(card) {
    card.previewAbort?.abort();
    card.boxes.replaceChildren();
    if (card.imageUrl) URL.revokeObjectURL(card.imageUrl);
    card.imageUrl = null;
    card.image.removeAttribute('src');
  }
  async function refreshPreview(card) {
    if (card.previewAbort || previewRequests >= 4) return;
    const controller = new AbortController();
    card.previewAbort = controller; card.previewAt = Date.now(); previewRequests++;
    let nextUrl = null;
    try {
      const [response, detections] = await Promise.all([
        fetch(`/api/cameras/${card.camera.id}/snapshot?size=thumbnail&t=${card.previewAt}`, { signal: controller.signal }),
        ui.api(`/api/cameras/${card.camera.id}/detections`, { signal: controller.signal }),
      ]);
      if (!response.ok) throw new Error('Нет свежего видео');
      const blob = await response.blob();
      if (controller.signal.aborted || !previewVisible(card)) return;
      nextUrl = URL.createObjectURL(blob);
      card.image.src = nextUrl;
      await card.image.decode();
      if (controller.signal.aborted || !previewVisible(card) || !card.camera.enabled || card.camera.status !== 'ONLINE') return;
      if (card.imageUrl) URL.revokeObjectURL(card.imageUrl);
      card.imageUrl = nextUrl; nextUrl = null;
      if (window.CameraOverlay.isFresh(detections)
          && detections.source_revision === card.camera.attendance.source_revision) {
        window.CameraOverlay.render(card.boxes, detections.tracks, { image: card.image, compact: true });
      } else card.boxes.replaceChildren();
    } catch (error) {
      card.boxes.replaceChildren();
      if (!controller.signal.aborted) {
        card.image.hidden = true; card.unavailable.hidden = false;
        card.unavailable.textContent = 'Нет свежего видео';
      }
    } finally {
      if (nextUrl) URL.revokeObjectURL(nextUrl);
      card.previewAbort = null; previewRequests--;
    }
  }
  function refreshPreviews() {
    for (const card of cards.values()) if (!previewVisible(card)) {
      card.previewAbort?.abort(); card.boxes.replaceChildren();
    }
    const visible = [...cards.values()].filter((card) => previewVisible(card) && card.camera.enabled
      && card.camera.status === 'ONLINE' && Date.now() - card.previewAt >= 1000)
      .sort((a, b) => a.previewAt - b.previewAt);
    for (const card of visible) {
      if (previewRequests >= 4) break;
      refreshPreview(card);
    }
  }
  async function loadCameras() {
    const ticket = ++sequence;
    const results = await Promise.allSettled([ui.api('/api/cameras'), ui.api('/api/workplaces')]);
    if (ticket !== sequence) return;
    const errors = [];
    if (results[1].status === 'fulfilled') { workplaces = results[1].value; renderWorkplaces(); }
    else errors.push(`Не удалось обновить рабочие места. ${results[1].reason.message}`);
    if (results[0].status === 'fulfilled') {
      cameras = results[0].value;
      const wanted = new Set(cameras.map((camera) => camera.id));
      for (const [id, card] of cards) {
        if (!wanted.has(id)) { disposePreview(card); card.element.remove(); cards.delete(id); selected.delete(id); }
      }
      for (const camera of cameras) {
        if (!cards.has(camera.id)) cards.set(camera.id, newCard(camera));
        updateCard(cards.get(camera.id), camera);
      }
      empty.hidden = cameras.length > 0;
    } else errors.push(`Не удалось обновить камеры. Показаны последние данные. ${results[0].reason.message}`);
    renderTabs(); renderSelection();
    const online = cameras.filter((camera) => camera.status === 'ONLINE').length;
    summary.textContent = [cameras.length ? `Камер: ${cameras.length} · онлайн: ${online}` : 'Камеры ещё не добавлены.', ...errors].join(' ');
    retry.hidden = errors.length === 0;
    refreshPreviews();
  }
  async function bulkAction(action) {
    if (bulkBusy || !selected.size) return;
    const cameraIds = [...selected];
    const names = new Map(cameraIds.map((id) => [id, cards.get(id).camera.name]));
    const payload = { camera_ids: cameraIds, action };
    if (action === 'ASSIGN_WORKPLACE') payload.workplace_id = $('camera-bulk-workplace').value ? Number($('camera-bulk-workplace').value) : null;
    bulkBusy = true; ui.setBusy(bulk, true); renderSelection();
    const result = $('camera-bulk-result'), failures = $('camera-bulk-failures');
    result.textContent = 'Выполнение…'; failures.replaceChildren(); failures.hidden = true;
    try {
      const data = await ui.api('/api/cameras/bulk', {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload),
      });
      const failed = data.results.filter((item) => !item.ok);
      result.textContent = `Готово: ${data.results.length - failed.length}; не выполнено: ${failed.length}`;
      for (const item of failed) failures.append(node('li', '', `${names.get(item.camera_id) || `Камера #${item.camera_id}`}: ${item.detail}`));
      failures.hidden = failed.length === 0;
      await loadCameras();
    } catch (error) { result.textContent = `Действие не выполнено. ${error.message}`; }
    finally { bulkBusy = false; ui.setBusy(bulk, false); renderSelection(); }
  }
  $('camera-bulk-assign').addEventListener('click', () => bulkAction('ASSIGN_WORKPLACE'));
  $('camera-bulk-actions').addEventListener('click', (event) => {
    const button = event.target.closest('[data-camera-action]');
    if (button && !button.disabled) bulkAction(button.dataset.cameraAction);
  });
  $('camera-selection-clear').addEventListener('click', () => {
    if (bulkBusy) return;
    selected.clear(); renderSelection();
    tabNodes.get(tabs.current())?.button.focus();
  });
  for (const id of ['btn-add', 'camera-empty-add']) {
    const button = document.getElementById(id);
    button.addEventListener('click', () => window.CameraSetup.openAdd(button));
  }
  retry.addEventListener('click', loadCameras);
  window.addEventListener('cameras-changed', loadCameras);
  document.addEventListener('visibilitychange', () => {
    if (!document.hidden) loadCameras();
    else for (const card of cards.values()) { card.previewAbort?.abort(); card.boxes.replaceChildren(); }
  });
  loadCameras();
  setInterval(() => { if (!document.hidden) loadCameras(); }, 5000);
  setInterval(refreshPreviews, 750);
  window.addEventListener('pagehide', () => { for (const card of cards.values()) disposePreview(card); });
})();
