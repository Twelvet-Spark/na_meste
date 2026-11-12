// Camera directory: stable card nodes, safe text, explicit setup and no automatic recording.
(() => {
  'use strict';
  const ui = window.OperatorUI;
  const grid = document.getElementById('camera-grid');
  const summary = document.getElementById('summary');
  const empty = document.getElementById('empty');
  const retry = document.getElementById('camera-retry');
  const cards = new Map();
  let sequence = 0;
  const states = { ONLINE: 'Онлайн', OFFLINE: 'Офлайн', CONNECTING: 'Подключение', RECONNECTING: 'Переподключение', ERROR: 'Ошибка' };
  function node(tag, className, text) {
    const value = document.createElement(tag); value.className = className;
    if (text) value.textContent = text;
    return value;
  }
  function newCard(camera) {
    const element = node('article', 'cam-card');
    const link = node('a', 'camera-preview'); link.href = `/cameras/${camera.id}`;
    const image = node('img', ''); image.alt = `Предпросмотр: ${camera.name}`; image.hidden = true;
    const unavailable = node('span', 'muted', 'Предпросмотр недоступен');
    link.append(image, unavailable);
    image.addEventListener('load', () => {
      if (card.camera.enabled && card.camera.status === 'ONLINE') { image.hidden = false; unavailable.hidden = true; }
    });
    image.addEventListener('error', () => { image.hidden = true; unavailable.hidden = false; });
    const head = node('div', 'cam-card-head');
    const title = node('a', 'cam-name'); title.href = link.href;
    const status = node('span', 'status offline'); head.append(title, status);
    const meta = node('p', 'cam-meta muted');
    const people = node('p', 'cam-meta');
    const recording = node('span', 'p-badge');
    const actions = node('div', 'cam-card-actions');
    const edit = node('button', 'btn ghost', 'Изменить'); edit.type = 'button';
    edit.addEventListener('click', () => window.CameraSetup.openEdit(camera.id, edit));
    const remove = node('button', 'btn danger', 'Удалить'); remove.type = 'button';
    const card = { element, title, status, meta, people, recording, image, unavailable, camera, previewAt: 0 };
    remove.addEventListener('click', async () => {
      if (!window.confirm(`Удалить камеру «${card.camera.name}»? Камеру с историей можно только отключить.`)) return;
      ui.setBusy(actions, true);
      try { await ui.api(`/api/cameras/${camera.id}`, { method: 'DELETE' }); await loadCameras(); }
      catch (error) { summary.textContent = `Удаление не выполнено. ${error.message}`; retry.hidden = false; }
      finally { ui.setBusy(actions, false); }
    });
    actions.append(edit, remove); element.append(link, head, meta, people, recording, actions);
    grid.append(element); return card;
  }
  function updateCard(card, camera) {
    card.camera = camera; card.title.textContent = camera.name;
    card.status.textContent = states[camera.status] || 'Неизвестно';
    card.status.className = `status ${Object.hasOwn(states, camera.status) ? camera.status.toLowerCase() : 'offline'}`;
    card.meta.textContent = `${camera.nvr_host} · канал ${camera.channel} · ${camera.stream_type === 'sub' ? 'дополнительный' : 'основной'}${camera.resolution ? ` · ${camera.resolution}` : ''}`;
    card.people.textContent = `В кадре: ${camera.people_count || 0} · сотрудники: ${camera.known_count || 0} · неизвестные: ${camera.unknown_count || 0}`;
    const recordingStates = { STARTING: 'Запись запускается', RECORDING: 'Идёт запись', RECONNECTING: 'Запись: переподключение', STOPPING: 'Запись завершается' };
    card.recording.textContent = camera.recording_available === false ? 'Запись: состояние недоступно'
      : recordingStates[camera.recording?.status] || 'Запись не ведётся';
    card.image.alt = `Предпросмотр: ${camera.name}`;
    if (camera.enabled && camera.status === 'ONLINE' && Date.now() - card.previewAt >= 10000) {
      card.previewAt = Date.now(); card.image.src = `/api/cameras/${camera.id}/snapshot?t=${card.previewAt}`;
    } else if (!camera.enabled || camera.status !== 'ONLINE') {
      card.image.hidden = true; card.unavailable.hidden = false;
      card.unavailable.textContent = camera.enabled ? 'Нет свежего видео' : 'Камера отключена';
    }
  }
  async function loadCameras() {
    const ticket = ++sequence;
    try {
      const cameras = await ui.api('/api/cameras');
      if (ticket !== sequence) return;
      const wanted = new Set(cameras.map((camera) => camera.id));
      for (const [id, card] of cards) {
        if (!wanted.has(id)) { card.element.remove(); cards.delete(id); }
      }
      for (const camera of cameras) {
        if (!cards.has(camera.id)) cards.set(camera.id, newCard(camera));
        updateCard(cards.get(camera.id), camera);
      }
      empty.hidden = cameras.length > 0;
      const online = cameras.filter((camera) => camera.status === 'ONLINE').length;
      summary.textContent = cameras.length ? `Камер: ${cameras.length} · онлайн: ${online}` : 'Камеры ещё не добавлены.';
      retry.hidden = true;
    } catch (error) {
      if (ticket !== sequence) return;
      summary.textContent = `Не удалось обновить камеры. Показаны последние данные. ${error.message}`;
      retry.hidden = false;
    }
  }
  for (const id of ['btn-add', 'camera-empty-add']) {
    const button = document.getElementById(id);
    button.addEventListener('click', () => window.CameraSetup.openAdd(button));
  }
  retry.addEventListener('click', loadCameras);
  window.addEventListener('cameras-changed', loadCameras);
  document.addEventListener('visibilitychange', () => { if (!document.hidden) loadCameras(); });
  loadCameras();
  setInterval(() => { if (!document.hidden) loadCameras(); }, 5000);
})();
