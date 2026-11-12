// Process-local archive playback. Never store camera credentials or player drafts.
(() => {
  'use strict';
  const $ = (id) => document.getElementById(id);
  const ui = window.OperatorUI;
  const video = $('recordings-video');
  const query = new URLSearchParams(location.search);
  let source = query.get('source') === 'dahua' ? 'dahua' : 'local';
  let generation = 0, searchGeneration = 0, transition = Promise.resolve();
  let session = null, hls = null, attached = null, nativeOrigin = null;
  let position = null, pausedAt = null, nextAt = null, wanted = null;
  let polling = false, tearingDown = false, leaving = false, starting = false;
  let intervalData = null;
  const states = { PREPARING: 'Подготовка первых фрагментов…', READY: 'Видео готово. Воспроизведение 1×.',
    ENDED: 'Доступный фрагмент подготовлен полностью.', CANCELLED: 'Просмотр завершён.', FAILED: 'Не удалось подготовить видео.', EXPIRED: 'Кэш просмотра истёк.' };
  const errors = {
    ARCHIVE_NOT_CONFIGURED: 'Архив не настроен. Откройте подключение камеры и укажите HTTP-параметры и смещение часов.',
    DEVICE_AUTH_FAILED: 'Регистратор отклонил доступ. Проверьте логин, пароль и разрешение на просмотр архива.',
    DEVICE_UNREACHABLE: 'Регистратор недоступен. Проверьте адрес и подключение к сети.',
    DEVICE_TIMEOUT: 'Регистратор не ответил вовремя. Проверьте сеть и повторите запрос.',
    DEVICE_TLS_FAILED: 'Не удалось проверить TLS-сертификат регистратора. Исправьте сертификат или настройки подключения.',
    ARCHIVE_UNSUPPORTED: 'Прошивка не поддерживает доступные профили архива. Проверьте модель, прошивку и разрешения Dahua.',
    DEVICE_PROTOCOL_ERROR: 'Регистратор вернул некорректный ответ. Проверьте совместимость прошивки и настройки архива.',
    SEARCH_LIMIT: 'Архив слишком большой для одного поиска. Выберите более короткий период.',
    PLAYBACK_BUSY: 'Другой просмотр уже использует преобразование видео. Завершите его и повторите запрос.',
    NO_RECORDING_AT_TIME: 'В выбранное время нет доступной записи. Автоматического перехода через разрыв не будет.',
    STORAGE_FULL: 'Недостаточно места для просмотра. Освободите место или завершите другие операции с видео.',
    SOURCE_CHANGED: 'Подключение регистратора изменилось. Выполните поиск заново.',
    LIBX264_UNAVAILABLE: 'В установленном FFmpeg отсутствует libx264. Установите сборку с этим кодировщиком и перезапустите приложение.',
    FFMPEG_UNAVAILABLE: 'FFmpeg или ffprobe недоступен. Проверьте установку на сервере.',
    MEDIA_PROCESS_CONTAINMENT_UNAVAILABLE: 'Windows не разрешила безопасный запуск медиа. Проверьте запуск приложения и перезапустите сервер.',
    PLAYBACK_PARTIAL: 'Запись закончилась раньше заявленного времени. Доступная часть сохранена, оставшаяся часть не подтверждена.',
    PLAYBACK_NO_PROGRESS: 'Подготовка видео остановилась. Проверьте источник и повторите запрос с сохранённого времени.',
    PLAYBACK_DEADLINE: 'Просмотр достиг предельного времени подготовки. Запустите новый запрос.',
    PLAYBACK_INVALID_SEGMENT: 'Полученный фрагмент не декодируется. Выберите другую запись или проверьте источник.',
    PLAYBACK_OUTPUT_GAP: 'Обнаружен разрыв выходного видео. Начните новый просмотр с выбранного времени.',
    PLAYBACK_NO_VIDEO: 'Источник не передал пригодное видео. Проверьте наличие записи и разрешения.',
    PLAYBACK_EXPIRED: 'Кэш истёк. Продолжите новым запросом с сохранённого времени.',
    PLAYBACK_CACHE_EXPIRED: 'Фрагменты кэша уже удалены. Продолжите с сохранённого времени.',
    PLAYBACK_UNAVAILABLE: 'Сервис просмотра недоступен. Проверьте FFmpeg и хранилище на сервере.',
  };
  const format = (value) => new Date(value).toLocaleString('ru-RU', { year: 'numeric', month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit', second: '2-digit' });
  function localInput(value) {
    const date = new Date(value);
    return new Date(date.getTime() - date.getTimezoneOffset() * 60000).toISOString().slice(0, 23);
  }
  function inputDate(id) {
    const value = new Date($(id).value);
    if (!Number.isFinite(value.getTime())) throw new Error('Укажите корректную дату и время.');
    return value;
  }
  function message(text, error = false) { ui.message($('recordings-player-status'), text, error ? 'error' : ''); }
  function errorText(error) {
    if (error instanceof TypeError && !error.status) return 'Сервер недоступен. Проверьте подключение и повторите запрос.';
    const code = typeof error.detail === 'string' ? error.detail : error.detail?.code;
    return errors[code] || (error.status === 410 ? errors.PLAYBACK_EXPIRED : error.status === 422 ? 'Проверьте время и период: не больше 24 часов.' : error.message || 'Не удалось выполнить запрос. Повторите попытку.');
  }
  function offerNext(value) {
    nextAt = value ? new Date(value).getTime() : null;
    $('recordings-next').hidden = nextAt === null;
    $('recordings-next').textContent = nextAt === null ? 'Следующая доступная запись' : `Следующая доступная запись: ${format(nextAt)}`;
  }
  function absoluteTime() {
    const date = hls?.playingDate;
    if (date && Number.isFinite(date.getTime())) return date.getTime();
    if (nativeOrigin !== null && Number.isFinite(video.currentTime)) return nativeOrigin + video.currentTime * 1000;
    return position;
  }
  function displayPosition(value) {
    if (!Number.isFinite(value)) return;
    position = value;
    $('recordings-position').textContent = format(value);
    $('recordings-position').dateTime = new Date(value).toISOString();
    if (document.activeElement !== $('recordings-time')) {
      $('recordings-time').value = String(value);
      $('recordings-time').setAttribute('aria-valuetext', format(Number($('recordings-time').value)));
    }
  }
  function detach() {
    tearingDown = true;
    video.pause();
    hls?.destroy(); hls = null;
    video.removeAttribute('src'); video.load();
    attached = null; nativeOrigin = null;
    tearingDown = false;
  }
  async function cancelServer(id, keepalive = false) {
    if (!id) return;
    try { await ui.api(`/api/recordings/playbacks/${id}`, { method: 'DELETE', keepalive }); }
    catch (error) { if (![404, 410].includes(error.status)) throw error; }
  }
  function closePlayer({ clear = true } = {}) {
    const token = ++generation;
    detach();
    const old = session; session = null;
    $('recordings-stop').disabled = true;
    if (clear) { wanted = null; pausedAt = null; offerNext(null); $('recordings-resume').hidden = true; }
    transition = transition.catch(() => {}).then(async () => {
      await cancelServer(old?.id);
      if (token === generation) starting = false;
    }).catch((error) => { if (token === generation) message(errorText(error), true); });
    return transition;
  }
  function requestAt(value) {
    if (!Number.isFinite(value) || !$('recordings-camera').value) return;
    const token = ++generation;
    const camera = Number($('recordings-camera').value), selectedSource = source;
    wanted = value; pausedAt = null; displayPosition(value);
    $('recordings-at').value = localInput(value);
    $('recordings-resume').hidden = true; offerNext(null);
    detach();
    const previous = session; session = null;
    starting = true; message('Подготовка первых фрагментов…');
    transition = transition.catch(() => {}).then(async () => {
      await cancelServer(previous?.id);
      if (token !== generation || leaving) return;
      const result = await ui.api('/api/recordings/playbacks', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ camera_id: camera, source: selectedSource, at: new Date(value).toISOString(), request_id: ui.newRequestId() }),
      });
      if (token !== generation || leaving) { await cancelServer(result.id, leaving); return; }
      session = result;
      $('recordings-stop').disabled = false;
      await updateSession(result, token);
    }).catch((error) => {
      if (token !== generation) return;
      message(errorText(error), true);
      if (error.detail?.next_at) offerNext(error.detail.next_at);
      $('recordings-resume').hidden = error.detail?.code === 'NO_RECORDING_AT_TIME';
    }).finally(() => { if (token === generation) starting = false; });
  }
  function seek(value) {
    if (!Number.isFinite(value)) return;
    const current = absoluteTime();
    if (session && attached && current !== null && !video.ended) {
      const mediaTime = video.currentTime + (value - current) / 1000;
      for (let index = 0; index < video.buffered.length; index++) {
        if (mediaTime >= video.buffered.start(index) && mediaTime < video.buffered.end(index)) {
          video.currentTime = mediaTime; displayPosition(value); pausedAt = video.paused ? value : null; return;
        }
      }
    }
    requestAt(value);
  }
  function playerFailure(text) {
    const saved = pausedAt ?? absoluteTime() ?? wanted;
    closePlayer({ clear: false });
    position = saved; pausedAt = saved;
    message(text, true);
    $('recordings-resume').hidden = false;
  }
  async function attach(result, token) {
    if (attached === result.id) return;
    attached = result.id;
    video.playbackRate = 1;
    const play = () => {
      if (token !== generation) return;
      video.play().catch(() => { if (token === generation) message('Видео готово. Нажмите воспроизведение в плеере.'); });
    };
    if (window.Hls?.isSupported()) {
      // hls.js 1.7.4 requires the default sync count explicitly with a max count.
      hls = new Hls({ startPosition: 0, liveSyncDurationCount: 3, liveMaxLatencyDurationCount: Infinity, maxLiveSyncPlaybackRate: 1 });
      hls.on(Hls.Events.MANIFEST_PARSED, play);
      hls.on(Hls.Events.ERROR, (_event, data) => {
        if (token !== generation || !data.fatal) return;
        playerFailure('Видео недоступно или кэш истёк. Продолжите новым запросом с сохранённого времени.');
      });
      hls.loadSource(result.manifest_url); hls.attachMedia(video);
    } else if (video.canPlayType('application/vnd.apple.mpegurl')) {
      const response = await fetch(result.manifest_url, { cache: 'no-store' });
      if (!response.ok) throw new Error(errors.PLAYBACK_CACHE_EXPIRED);
      const text = await response.text();
      if (token !== generation) return;
      const date = /^#EXT-X-PROGRAM-DATE-TIME:(.+)$/m.exec(text)?.[1];
      const firstDate = date ? new Date(date).getTime() : NaN;
      if (!Number.isFinite(firstDate)) throw new Error('В записи отсутствует привязка ко времени.');
      video.addEventListener('loadedmetadata', () => {
        if (token !== generation) return;
        const start = video.seekable.length ? video.seekable.start(0) : 0;
        nativeOrigin = firstDate - start * 1000;
        video.currentTime = start; play();
      }, { once: true });
      video.src = result.manifest_url; video.load();
    } else {
      playerFailure('Этот браузер не поддерживает HLS. Откройте страницу в актуальном Chrome, Edge, Firefox или Safari.');
    }
  }
  async function updateSession(result, token) {
    if (token !== generation) return;
    session = result;
    if (result.next_at) offerNext(result.next_at);
    message(result.error_code ? errors[result.error_code] || 'Подготовка видео завершилась с ошибкой. Повторите запрос.' : states[result.state], Boolean(result.error_code));
    if (result.manifest_url && !attached) {
      try { await attach(result, token); }
      catch (error) { if (token === generation) playerFailure(errorText(error)); }
    }
    if (['FAILED', 'CANCELLED', 'EXPIRED'].includes(result.state)) {
      if (result.state !== 'CANCELLED' || result.error_code) $('recordings-resume').hidden = false;
      if (result.state !== 'ENDED') {
        const saved = pausedAt ?? absoluteTime() ?? wanted;
        detach(); position = saved; pausedAt = saved;
      }
    }
  }
  async function poll() {
    if (!session || polling || leaving) return;
    const token = generation, id = session.id;
    polling = true;
    try {
      const result = await ui.api(`/api/recordings/playbacks/${id}`, { cache: 'no-store' });
      if (token === generation && session?.id === id) await updateSession(result, token);
    } catch (error) {
      if (token !== generation) return;
      message(errorText(error), true);
      if (error.status === 410) {
        const saved = pausedAt ?? absoluteTime() ?? wanted;
        detach(); position = saved; pausedAt = saved; session = null;
        $('recordings-resume').hidden = false;
      }
    } finally { polling = false; }
  }
  function renderTimeline(data) {
    intervalData = data;
    const lower = new Date(data.from).getTime(), upper = new Date(data.to).getTime();
    const slider = $('recordings-time');
    slider.min = String(lower); slider.max = String(upper - 1); slider.disabled = false;
    slider.value = String(Math.min(upper - 1, Math.max(lower, wanted ?? lower)));
    slider.setAttribute('aria-valuetext', format(Number(slider.value)));
    $('recordings-left').textContent = format(lower); $('recordings-right').textContent = format(upper);
    const track = $('recordings-track'), list = $('recordings-intervals');
    track.replaceChildren(); list.replaceChildren();
    const entries = [...data.intervals.map((item) => ({ ...item, gap: false })), ...data.gaps.map((item) => ({ ...item, gap: true }))];
    entries.sort((a, b) => new Date(a.from) - new Date(b.from));
    const labels = { READY: 'Доступно', EXPIRED: 'Удалено по сроку хранения или лимиту', MISSING: 'Файл отсутствует', CORRUPT: 'Файл повреждён', DELETING: 'Удаляется', UNINDEXED: 'Время не подтверждено' };
    for (const item of entries) {
      const start = new Date(item.from).getTime(), end = new Date(item.to).getTime();
      const bar = document.createElement('span');
      bar.className = item.gap ? 'recordings-gap' : item.state === 'READY' ? '' : 'recordings-unavailable';
      bar.style.left = `${Math.max(0, (start - lower) / (upper - lower) * 100)}%`;
      bar.style.width = `${Math.max(0, (end - start) / (upper - lower) * 100)}%`;
      track.append(bar);
      const row = document.createElement('li'), time = document.createElement('p'), label = document.createElement('p');
      time.textContent = `${format(start)} — ${format(end)}`;
      label.textContent = item.gap ? `Разрыв записи: ${item.reason}` : labels[item.state] || item.state;
      label.className = 'muted small'; row.append(time, label);
      if (!item.gap && item.state === 'READY') {
        const button = document.createElement('button'); button.type = 'button'; button.className = 'btn';
        button.textContent = 'Смотреть с начала фрагмента'; button.onclick = () => requestAt(start); row.append(button);
      }
      list.append(row);
    }
    $('recordings-empty').hidden = entries.length > 0;
    $('recordings-empty').textContent = 'За выбранный период записей нет. Измените период или источник.';
  }
  async function search() {
    const token = ++searchGeneration;
    let start, end;
    try {
      start = inputDate('recordings-from'); end = inputDate('recordings-to');
      if (end <= start || end - start > 86400000) throw new Error('Выберите положительный период не больше 24 часов.');
      if (!$('recordings-camera').value) throw new Error('Выберите камеру.');
    } catch (error) { ui.message($('recordings-search-status'), error.message, 'error'); return; }
    ui.message($('recordings-search-status'), 'Поиск записей…');
    const params = new URLSearchParams({ camera_id: $('recordings-camera').value, source, from: start.toISOString(), to: end.toISOString() });
    $('recordings-find').disabled = true;
    try {
      const data = await ui.api(`/api/recordings/timeline?${params}`, { cache: 'no-store' });
      if (token !== searchGeneration) return;
      renderTimeline(data);
      ui.message($('recordings-search-status'), `Найдено фрагментов: ${data.intervals.length}. Разрывов: ${data.gaps.length}.`);
      return token;
    } catch (error) {
      if (token === searchGeneration) {
        ui.message($('recordings-search-status'), `${errorText(error)} ${intervalData ? 'Показан предыдущий результат. ' : ''}Нажмите «Найти записи», чтобы повторить.`, 'error');
        if (!intervalData) $('recordings-empty').textContent = 'Список записей недоступен до успешного поиска.';
      }
    } finally { if (token === searchGeneration) $('recordings-find').disabled = false; }
  }
  function sourceView() {
    for (const button of document.querySelectorAll('[data-source]')) {
      const selected = button.dataset.source === source;
      button.setAttribute('aria-selected', String(selected)); button.tabIndex = selected ? 0 : -1;
    }
    $('recordings-results').setAttribute('aria-labelledby', `source-${source}`);
    $('recordings-basis').textContent = source === 'local'
      ? 'Локальное видео: приблизительное серверное время записи, не время экспозиции камеры и не доказательство отметок рабочего времени.'
      : 'Архив Dahua: время пересчитано по настроенному смещению часов устройства. Убедитесь, что смещение подходит для выбранной даты; переход на летнее время не определяется автоматически.';
  }
  async function switchSelection() {
    ++searchGeneration; intervalData = null;
    $('recordings-track').replaceChildren(); $('recordings-intervals').replaceChildren();
    $('recordings-empty').hidden = false; $('recordings-empty').textContent = 'Выполняется новый поиск…';
    $('recordings-time').disabled = true;
    const camera = $('recordings-camera').value;
    $('recordings-camera-link').href = camera ? `/cameras/${encodeURIComponent(camera)}` : '/';
    sourceView(); await closePlayer(); await search();
  }
  $('recordings-search').addEventListener('submit', (event) => { event.preventDefault(); search(); });
  $('recordings-camera').addEventListener('change', switchSelection);
  for (const button of document.querySelectorAll('[data-source]')) {
    button.addEventListener('click', () => { if (source !== button.dataset.source) { source = button.dataset.source; switchSelection(); } });
    button.addEventListener('keydown', (event) => {
      if (['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) {
        event.preventDefault(); const other = $(button.id === 'source-local' ? 'source-dahua' : 'source-local'); other.focus(); other.click();
      }
    });
  }
  $('recordings-seek').addEventListener('submit', (event) => { event.preventDefault(); try { seek(inputDate('recordings-at').getTime()); } catch (error) { message(error.message, true); } });
  function skip(seconds) {
    try { seek((pausedAt ?? absoluteTime() ?? wanted ?? inputDate('recordings-at').getTime()) + seconds * 1000); }
    catch (error) { message(error.message, true); }
  }
  $('recordings-back').onclick = () => skip(-10);
  $('recordings-forward').onclick = () => skip(10);
  $('recordings-time').addEventListener('input', () => { $('recordings-time').setAttribute('aria-valuetext', format(Number($('recordings-time').value))); });
  $('recordings-time').addEventListener('change', () => seek(Number($('recordings-time').value)));
  $('recordings-next').onclick = () => { if (nextAt !== null) requestAt(nextAt); };
  $('recordings-resume').onclick = () => { const target = pausedAt ?? position ?? wanted; if (target !== null) requestAt(target); };
  $('recordings-stop').onclick = () => { closePlayer(); message('Просмотр закрыт. Ручная запись камеры продолжает работать независимо.'); };
  video.addEventListener('timeupdate', () => { if (!tearingDown && pausedAt === null) displayPosition(absoluteTime()); });
  video.addEventListener('pause', () => { if (!tearingDown && attached && !video.ended) pausedAt = absoluteTime(); });
  video.addEventListener('play', () => {
    if (tearingDown || starting || pausedAt === null) return;
    const saved = pausedAt;
    const available = session?.available_from && saved >= new Date(session.available_from).getTime() && saved < new Date(session.available_to).getTime();
    if (!available) { video.pause(); requestAt(saved); } else { pausedAt = null; seek(saved); }
  });
  video.addEventListener('ratechange', () => { if (video.playbackRate !== 1) video.playbackRate = 1; });
  video.addEventListener('ended', () => {
    if (!session || tearingDown) return;
    const end = new Date(session.to).getTime();
    displayPosition(Math.min(absoluteTime() ?? end, end));
    if (!session.error_code && nextAt === end && end - new Date(session.from).getTime() >= 3599000) requestAt(nextAt);
    else { message(session.error_code ? errors[session.error_code] || 'Доступна только часть записи.' : 'Фрагмент закончился. Выберите следующую доступную запись.'); offerNext(session.next_at); }
  });
  video.addEventListener('error', () => {
    if (!attached || tearingDown || hls) return;
    playerFailure('Не удалось прочитать видео. Продолжите с сохранённого времени.');
  });
  window.addEventListener('pagehide', () => {
    leaving = true; ++generation; ++searchGeneration;
    position = pausedAt ?? absoluteTime() ?? position;
    const id = session?.id; detach(); session = null;
    cancelServer(id, true).catch(() => {});
  });
  let pollTimer = setInterval(poll, 5000);
  window.addEventListener('pagehide', () => clearInterval(pollTimer));
  window.addEventListener('pageshow', (event) => {
    if (!event.persisted) return;
    leaving = false; starting = false; polling = false;
    pollTimer = setInterval(poll, 5000);
    $('recordings-resume').hidden = position === null;
    message('Предыдущий просмотр закрыт. Можно продолжить с сохранённого времени.');
    search();
  });
  async function loadCameras() {
    const select = $('recordings-camera'), selected = select.value || query.get('camera_id');
    const retry = $('recordings-cameras-retry');
    retry.disabled = true; select.disabled = true; $('recordings-find').disabled = true;
    try {
      const cameras = await ui.api('/api/cameras');
      if (leaving) return false;
      select.replaceChildren();
      for (const camera of cameras) {
        const option = document.createElement('option');
        option.value = camera.id; option.textContent = camera.name; select.append(option);
      }
      retry.hidden = true;
      if (!cameras.length) {
        ui.message($('recordings-search-status'), 'Камеры ещё не добавлены. Откройте раздел «Камеры».');
        return false;
      }
      if (cameras.some((camera) => String(camera.id) === selected)) select.value = selected;
      $('recordings-camera-link').href = `/cameras/${encodeURIComponent(select.value)}`;
      return true;
    } catch (error) {
      ui.message($('recordings-search-status'), `${errorText(error)} Повторите загрузку камер.`, 'error');
      retry.hidden = false;
      return false;
    } finally {
      retry.disabled = false; select.disabled = false;
      $('recordings-find').disabled = !select.value;
    }
  }
  $('recordings-cameras-retry').onclick = async () => { if (await loadCameras()) await search(); };
  async function initialize() {
    const initialPlayerGeneration = generation;
    const initialSource = source;
    const at = query.get('at') ? new Date(query.get('at')) : new Date();
    const initial = Number.isFinite(at.getTime()) ? at : new Date();
    const start = new Date(initial); start.setHours(0, 0, 0, 0);
    const end = new Date(start.getTime() + 86400000);
    $('recordings-from').value = localInput(start); $('recordings-to').value = localInput(end);
    $('recordings-at').value = localInput(initial);
    $('recordings-zone').textContent = `Ввод и отображение: часовой пояс браузера (${Intl.DateTimeFormat().resolvedOptions().timeZone}). Период — не больше 24 часов. Сервер получает время с явным UTC-смещением.`;
    sourceView();
    if (await loadCameras()) {
      const initialCamera = $('recordings-camera').value;
      const initialSearchGeneration = await search();
      if (initialSearchGeneration === searchGeneration &&
          initialPlayerGeneration === generation && initialSource === source &&
          initialCamera === $('recordings-camera').value &&
          query.has('at') && Number.isFinite(at.getTime())) requestAt(at.getTime());
    }
  }
  initialize();
})();
