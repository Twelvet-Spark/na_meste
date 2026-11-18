// One truthful overlay for live view and directory thumbnails. Scores are not identity probabilities.
(() => {
  'use strict';
  const score = (value) => typeof value === 'number' && Number.isFinite(value) && value >= 0 && value <= 1;
  function label(track) {
    const known = track.employee_id != null;
    const parts = [known ? (track.employee_name || `Сотрудник #${track.employee_id}`) : 'Неизвестный'];
    if (known) parts.push(track.identity_source === 'REID' ? 'по внешности' : 'по лицу');
    if (score(track.person_score)) parts.push(`человек ${Math.round(track.person_score * 100)}%`);
    if (known && track.identity_source !== 'REID' && score(track.face_similarity)) parts.push(`сходство лица ${track.face_similarity.toFixed(2)}`);
    parts.push(`T${track.track_id}${track.global_id ? `·G${track.global_id}` : ''}`);
    return parts.join(' · ');
  }
  function fitRect(width, height, nativeWidth, nativeHeight, fit = 'contain') {
    if (![width, height, nativeWidth, nativeHeight].every((n) => Number.isFinite(n) && n > 0)) return null;
    const scale = (fit === 'cover' ? Math.max : Math.min)(width / nativeWidth, height / nativeHeight);
    const w = nativeWidth * scale, h = nativeHeight * scale;
    return { left: (width - w) / 2, top: (height - h) / 2, width: w, height: h };
  }
  function isFresh(data, now = Date.now()) {
    return Number.isFinite(data?.updated_at) && now / 1000 - data.updated_at >= -1 && now / 1000 - data.updated_at <= 2;
  }
  function render(container, tracks, { image = null, links = false, metrics = false, compact = false } = {}) {
    container.replaceChildren();
    if (image) {
      const rect = fitRect(image.clientWidth, image.clientHeight, image.naturalWidth, image.naturalHeight, getComputedStyle(image).objectFit);
      if (!rect) return;
      Object.assign(container.style, { inset: 'auto', left: `${image.offsetLeft + rect.left}px`, top: `${image.offsetTop + rect.top}px`, width: `${rect.width}px`, height: `${rect.height}px` });
    }
    for (const track of tracks || []) {
      if (!Array.isArray(track.bbox) || track.bbox.length !== 4 || !track.bbox.every(Number.isFinite)) continue;
      const [x1, y1, x2, y2] = track.bbox.map((n) => Math.max(0, Math.min(1, n)));
      if (x2 <= x1 || y2 <= y1) continue;
      const box = document.createElement('div');
      box.className = `bbox ${track.employee_id != null ? 'recognized' : 'unknown'}`;
      Object.assign(box.style, { left: `${x1 * 100}%`, top: `${y1 * 100}%`, width: `${(x2 - x1) * 100}%`, height: `${(y2 - y1) * 100}%` });
      const acknowledged = links && typeof track.visit_id === 'string' && /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(track.visit_id);
      const text = document.createElement(acknowledged ? 'a' : 'span');
      text.className = 'bbox-label';
      if (compact) {
        text.title = label(track);
        const lines = [track.employee_id != null ? (track.employee_name || `#${track.employee_id}`) : 'Неизвестный'];
        if (score(track.person_score)) lines.push(`чел. ${Math.round(track.person_score * 100)}%`);
        if (track.employee_id != null) {
          if (track.identity_source === 'REID') lines.push('по внешности');
          else if (score(track.face_similarity)) lines.push(`лицо ${track.face_similarity.toFixed(2)}`);
        }
        for (const line of lines) {
          const row = document.createElement('span'); row.textContent = line; text.append(row);
        }
        text.style.maxWidth = `${Math.min(100, Math.max(0, (1 - x1) * container.clientWidth))}px`;
      } else text.textContent = label(track);
      if (acknowledged) {
        text.href = `/global/visits/${encodeURIComponent(track.visit_id)}`;
        text.style.pointerEvents = 'auto';
        text.title = 'Открыть сохранённый визит — не запись рабочего времени';
      }
      if (y1 * container.clientHeight < 24) text.style.top = '0';
      const reasons = Array.isArray(track.identity_reason_codes) ? track.identity_reason_codes.join(', ') : '';
      if (reasons) text.title += `${text.title ? '. ' : ''}Причина состояния личности: ${reasons}`;
      box.append(text);
      if (metrics) {
        const foot = document.createElement('span'); foot.className = 'bbox-footpoint';
        foot.title = 'Точка пересечения: низ рамки по центру';
        const metric = document.createElement('span'); metric.className = 'bbox-metric';
        metric.dataset.trackId = String(track.track_id); box.append(foot, metric);
      }
      container.append(box);
    }
  }
  window.CameraOverlay = Object.freeze({ label, fitRect, isFresh, render });
})();
