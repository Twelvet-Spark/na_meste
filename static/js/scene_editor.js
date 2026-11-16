// Frozen native-image editor. No live inference, client image upload or implicit save.
(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const root = $('scene-editor');
  if (!root) return;
  const cameraId = window.CAMERA_ID;
  const base = `/api/cameras/${cameraId}/scene`;
  const attendance = `/api/attendance/cameras/${cameraId}`;
  const canvas = $('calibration-canvas'), ctx = canvas.getContext('2d');
  const clone = value => JSON.parse(JSON.stringify(value));
  const emptyGeometry = () => ({version: 1, floor: null, objects: [], metric: null});
  const emptyGate = () => ({points: [null, null, null], role: '', enabled: false});
  const state = {
    saved: null, config: null, runtime: null, geometry: emptyGeometry(), gate: emptyGate(),
    snapshot: null, image: null, pixels: null, validSnapshot: false, review: false,
    busy: false, tool: 'select', selection: '', vertex: -1, drawing: false,
    undo: [], redo: [], zoom: 1, pan: [0, 0], space: false, gesture: null,
    layers: {floor: true, objects: true, gate: true, metric: true, suggestions: true},
    prompts: {positive_points: [], negative_points: [], box: null},
    candidates: [], job: null, jobEpoch: 0, jobTimer: null, ttlTimer: null,
    previewTimer: null, metric: null, cameraStatus: 'OFFLINE', sceneCurrent: false,
    sourceEpoch: 0, previewKey: '', liveSource: null, liveStream: null,
    completedJob: null,
  };
  const gateNames = ['start', 'end', 'inside'];
  const hints = {
    select: 'Выберите вершину и перетащите её. Двойной щелчок по ребру вставляет вершину.',
    floor: 'Отметьте видимый пол по контуру, затем завершите. Не дорисовывайте скрытую геометрию по догадке.',
    object: 'Введите название объекта и отметьте его силуэт. Новый объект создаётся отдельной кнопкой.',
    gate: 'Отметьте начало линии, конец линии, затем точку внутри помещения.',
    metric: 'Четыре угла: (0,0), (ширина,0), (ширина,глубина), (0,глубина). Затем введите измерения.',
    ai: 'Выберите положительные/отрицательные точки или протяните рамку, затем запросите контур.',
  };
  function message(text, tone = 'pending', id = 'scene-message') {
    const node = $(id); node.textContent = text; node.className = `test-result ${tone}`; node.hidden = !text;
  }
  function point(p) { return Array.isArray(p) && p.length === 2 && p.every(x => Number.isFinite(x) && x >= 0 && x <= 1); }
  const cross = (a, b, c) => (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
  function onSegment(p, a, b) {
    return Math.abs(cross(a, b, p)) < 1e-10 && p[0] >= Math.min(a[0], b[0]) - 1e-10 && p[0] <= Math.max(a[0], b[0]) + 1e-10
      && p[1] >= Math.min(a[1], b[1]) - 1e-10 && p[1] <= Math.max(a[1], b[1]) + 1e-10;
  }
  function contains(p, polygon) {
    if (!point(p) || !polygon || polygon.length < 3) return false;
    let inside = false;
    for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
      const a = polygon[j], b = polygon[i];
      if (onSegment(p, a, b)) return true;
      if ((a[1] > p[1]) !== (b[1] > p[1]) && p[0] < (b[0] - a[0]) * (p[1] - a[1]) / (b[1] - a[1]) + a[0]) inside = !inside;
    }
    return inside;
  }
  function intersects(a, b, c, d) {
    return (cross(a, b, c) * cross(a, b, d) < 0 && cross(c, d, a) * cross(c, d, b) < 0)
      || onSegment(a, c, d) || onSegment(b, c, d) || onSegment(c, a, b) || onSegment(d, a, b);
  }
  function polygonError(p) {
    if (!p || p.length < 3 || p.length > 128 || !p.every(point)) return 'Нужно от 3 до 128 вершин в пределах изображения.';
    let area = 0;
    for (let i = 0; i < p.length; i++) {
      const a = p[i], b = p[(i + 1) % p.length], prev = p[(i + p.length - 1) % p.length];
      if (p.some((q, j) => j !== i && q[0] === a[0] && q[1] === a[1])) return 'Вершины не должны совпадать.';
      if (Math.abs(cross(prev, a, b)) < 1e-12 && (prev[0] - a[0]) * (b[0] - a[0]) + (prev[1] - a[1]) * (b[1] - a[1]) > 0) return 'Рёбра перекрываются.';
      area += a[0] * b[1] - b[0] * a[1];
      for (let j = i + 2; j < p.length; j++) {
        if (i === 0 && j === p.length - 1) continue;
        if (intersects(a, b, p[j], p[(j + 1) % p.length])) return 'Контур пересекает сам себя.';
      }
    }
    return Math.abs(area) < 1e-12 ? 'Контур не имеет площади.' : '';
  }
  function project(matrix, p) {
    const z = matrix[2][0] * p[0] + matrix[2][1] * p[1] + matrix[2][2];
    if (!Number.isFinite(z) || Math.abs(z) < 1e-12) return null;
    const q = [(matrix[0][0] * p[0] + matrix[0][1] * p[1] + matrix[0][2]) / z,
      (matrix[1][0] * p[0] + matrix[1][1] * p[1] + matrix[1][2]) / z];
    return q.every(Number.isFinite) ? q : null;
  }
  function homography(from, to) {
    const rows = [];
    from.forEach(([x, y], i) => {
      const [u, v] = to[i];
      rows.push([x, y, 1, 0, 0, 0, -u*x, -u*y, u], [0, 0, 0, x, y, 1, -v*x, -v*y, v]);
    });
    for (let c = 0; c < 8; c++) {
      let pivot = c;
      for (let r = c + 1; r < 8; r++) if (Math.abs(rows[r][c]) > Math.abs(rows[pivot][c])) pivot = r;
      if (Math.abs(rows[pivot][c]) < 1e-12) throw new Error('Углы не задают устойчивую плоскую проекцию.');
      [rows[c], rows[pivot]] = [rows[pivot], rows[c]];
      const div = rows[c][c]; for (let k = c; k <= 8; k++) rows[c][k] /= div;
      for (let r = 0; r < 8; r++) if (r !== c) {
        const scale = rows[r][c]; for (let k = c; k <= 8; k++) rows[r][k] -= scale * rows[c][k];
      }
    }
    const h = rows.map(row => row[8]).concat(1), scale = Math.max(...h.map(Math.abs));
    return [h.slice(0, 3), h.slice(3, 6), h.slice(6, 9)].map(row => row.map(x => x / scale));
  }
  function compileMetric() {
    const m = state.geometry.metric, floor = state.geometry.floor;
    if (!m) return null;
    if (polygonError(floor)) throw new Error('Сначала завершите простой контур пола.');
    if (m.corners.length !== 4 || polygonError(m.corners)) throw new Error('Отметьте четыре разных угла прямоугольника по порядку.');
    if (![m.width_m, m.depth_m].every(v => Number.isFinite(v) && v > 0 && v <= 1000)) throw new Error('Введите измеренную ширину и глубину больше 0 и не больше 1000 м.');
    const c = m.corners, turns = c.map((p, i) => cross(p, c[(i + 1) % 4], c[(i + 2) % 4]));
    if (turns.some(x => Math.abs(x) < 1e-12 || Math.sign(x) !== Math.sign(turns[0])) || !c.every(p => contains(p, floor))) throw new Error('Углы должны образовывать выпуклый прямоугольник внутри пола.');
    const area = Math.abs(c.reduce((s, p, i) => s + p[0] * c[(i+1)%4][1] - c[(i+1)%4][0] * p[1], 0)) / 2;
    if (area < 1e-4) throw new Error('Опорный прямоугольник слишком мал в кадре.');
    const target = [[0, 0], [m.width_m, 0], [m.width_m, m.depth_m], [0, m.depth_m]];
    const forward = homography(c, target), inverse = homography(target, c);
    const denominators = floor.map(p => forward[2][0]*p[0] + forward[2][1]*p[1] + forward[2][2]);
    if (denominators.some(z => Math.abs(z) < 1e-6 || Math.sign(z) !== Math.sign(denominators[0]))) throw new Error('Пол пересекает горизонт проекции. Исправьте углы или сохраните без масштаба.');
    for (const p of [...floor, ...c]) {
      const q = project(forward, p), r = q && project(inverse, q);
      if (!r || Math.hypot(r[0] - p[0], r[1] - p[1]) > 1e-5) throw new Error('Проекция неустойчива. Исправьте опорные углы.');
    }
    return {forward, inverse, width: m.width_m, depth: m.depth_m};
  }
  function geometryError() {
    const g = state.geometry;
    if (g.floor !== null && polygonError(g.floor)) return `Пол: ${polygonError(g.floor)}`;
    for (const o of g.objects) {
      if (!o.label.trim() || o.label.length > 80) return 'Каждому объекту нужно название от 1 до 80 символов.';
      if (polygonError(o.polygon)) return `${o.label}: ${polygonError(o.polygon)}`;
    }
    try { compileMetric(); } catch (e) { return e.message; }
    return '';
  }
  function gateError() {
    const [a, b, inside] = state.gate.points;
    if (![a, b, inside].every(point)) return 'Введите три точки входа в пределах кадра.';
    const length = Math.hypot(b[0] - a[0], b[1] - a[1]);
    if (length < .05 - 1e-12) return 'Разнесите концы линии минимум на 0,05.';
    if (Math.abs(cross(a, b, inside)) / length < .05 - 1e-12) return 'Внутренняя точка должна быть минимум в 0,05 от линии.';
    if (!state.gate.role) return 'Выберите назначение камеры.';
    if (state.config?.role && state.gate.role !== state.config.role) return 'Назначенную роль нельзя заменить здесь: используйте настройку учёта.';
    return '';
  }
  function gateFrom(config) { return {points: [config.line_start, config.line_end, config.inside_point].map(p => p ? [...p] : null), role: config.role || '', enabled: Boolean(config.enabled)}; }
  const sceneDirty = () => JSON.stringify(state.geometry) !== JSON.stringify(state.saved?.geometry || emptyGeometry());
  const gateDirty = () => JSON.stringify(state.gate) !== JSON.stringify(state.config ? gateFrom(state.config) : emptyGate());
  const editable = () => Boolean(state.image && state.snapshot && !state.busy);
  const saveable = () => editable() && state.validSnapshot && !state.review;
  function checkpoint() {
    state.undo.push({geometry: clone(state.geometry), gate: clone(state.gate), selection: state.selection});
    if (state.undo.length > 60) state.undo.shift();
    state.redo.length = 0;
  }
  function history(undo) {
    if (!editable()) return;
    const from = undo ? state.undo : state.redo, to = undo ? state.redo : state.undo;
    if (!from.length) return;
    to.push({geometry: clone(state.geometry), gate: clone(state.gate), selection: state.selection});
    Object.assign(state, from.pop()); state.vertex = -1; state.drawing = false; sync();
  }
  function selectedPoints() {
    if (state.selection === 'floor') return state.geometry.floor;
    if (state.selection === 'gate') return state.gate.points;
    if (state.selection === 'metric') return state.geometry.metric?.corners;
    return state.geometry.objects.find(o => o.id === state.selection)?.polygon;
  }
  function allShapes() {
    return [{id: 'floor', points: state.geometry.floor, visible: state.layers.floor},
      ...state.geometry.objects.map(o => ({id: o.id, points: o.polygon, visible: state.layers.objects})),
      {id: 'metric', points: state.geometry.metric?.corners, visible: state.layers.metric},
      {id: 'gate', points: state.gate.points, visible: state.layers.gate}];
  }
  function setTool(tool) {
    state.tool = tool; state.drawing = ['floor', 'object', 'metric', 'gate'].includes(tool);
    if (tool === 'floor') state.selection = 'floor';
    if (tool === 'gate') state.selection = 'gate';
    if (tool === 'metric') state.selection = 'metric';
    if (tool === 'object' && !state.geometry.objects.some(o => o.id === state.selection)) state.selection = '';
    state.vertex = -1;
    sync();
  }
  function newObject() {
    if (!editable()) return null;
    if (state.geometry.objects.length >= 32) { message('Допускается не больше 32 объектов.', 'err'); return null; }
    const label = $('scene-object-label').value.trim();
    if (!label || label.length > 80) { message('Введите название объекта перед созданием контура.', 'err'); $('scene-object-label').focus(); return null; }
    const object = {id: window.OperatorUI.newRequestId(), label, polygon: []};
    state.geometry.objects.push(object); state.selection = object.id; return object;
  }
  function transform() {
    const rect = canvas.getBoundingClientRect(), w = rect.width, h = rect.height;
    const scale = state.image ? Math.min(w / state.image.naturalWidth, h / state.image.naturalHeight) * state.zoom : 1;
    const iw = (state.image?.naturalWidth || 0) * scale, ih = (state.image?.naturalHeight || 0) * scale;
    return {w, h, iw, ih, x: (w - iw) / 2 + state.pan[0], y: (h - ih) / 2 + state.pan[1]};
  }
  const display = (p, t) => [t.x + p[0] * t.iw, t.y + p[1] * t.ih];
  function eventPosition(event) {
    const rect = canvas.getBoundingClientRect(); return [event.clientX - rect.left, event.clientY - rect.top];
  }
  function imagePoint(position, clamp = false) {
    const t = transform(); if (!t.iw || !t.ih) return null;
    const p = [(position[0] - t.x) / t.iw, (position[1] - t.y) / t.ih];
    return clamp ? p.map(x => Math.max(0, Math.min(1, x))) : point(p) ? p : null;
  }
  function path(points, t, close = true) {
    ctx.beginPath(); let started = false;
    for (const p of points || []) if (point(p)) {
      const q = display(p, t); if (!started) { ctx.moveTo(...q); started = true; } else ctx.lineTo(...q);
    }
    if (close && started) ctx.closePath();
  }
  function stroke(points, t, color, closed = true, dashed = false) {
    ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.setLineDash(dashed ? [7, 5] : []);
    path(points, t, closed); ctx.stroke(); ctx.setLineDash([]);
  }
  function drawArrow(t, color) {
    const [a, b, inside] = state.gate.points;
    stroke([a, b], t, color, false);
    if (![a, b, inside].every(point)) return;
    const aa = display(a, t), bb = display(b, t), ii = display(inside, t);
    const mid = [(aa[0] + bb[0]) / 2, (aa[1] + bb[1]) / 2];
    let nx = -(bb[1] - aa[1]), ny = bb[0] - aa[0];
    const length = Math.hypot(nx, ny); if (length < 1e-6) return;
    const sign = (ii[0] - mid[0]) * nx + (ii[1] - mid[1]) * ny >= 0 ? 1 : -1;
    nx *= sign * 30 / length; ny *= sign * 30 / length;
    const from = [mid[0] - nx, mid[1] - ny], to = [mid[0] + nx, mid[1] + ny];
    const angle = Math.atan2(ny, nx);
    ctx.strokeStyle = color; ctx.fillStyle = color; ctx.lineWidth = 3; ctx.beginPath();
    ctx.moveTo(...from); ctx.lineTo(...to);
    ctx.moveTo(to[0] - 12 * Math.cos(angle - .45), to[1] - 12 * Math.sin(angle - .45));
    ctx.lineTo(...to); ctx.lineTo(to[0] - 12 * Math.cos(angle + .45), to[1] - 12 * Math.sin(angle + .45)); ctx.stroke();
    ctx.font = '13px sans-serif'; ctx.lineWidth = 4; ctx.strokeStyle = '#1C1B1F';
    const label = 'Снаружи → внутрь', x = Math.max(t.x + 4, Math.min(mid[0] + 8, t.x + t.iw - ctx.measureText(label).width - 4)), y = mid[1] - 38;
    ctx.strokeText(label, x, y); ctx.fillText(label, x, y);
  }
  function gridSpec(metric) {
    const mapped = state.geometry.floor.map(p => project(metric.forward, p));
    const xs = mapped.map(p => p[0]), ys = mapped.map(p => p[1]);
    const bounds = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
    let spacing = .5;
    const count = () => Math.max(0, Math.floor(bounds[1]/spacing) - Math.ceil(bounds[0]/spacing) + 1)
      + Math.max(0, Math.floor(bounds[3]/spacing) - Math.ceil(bounds[2]/spacing) + 1);
    while (count() > 100) spacing *= 2;
    return {bounds, spacing};
  }
  function drawGrid(t, color) {
    if (!state.metric) return;
    const {bounds: [xmin, xmax, ymin, ymax], spacing} = gridSpec(state.metric);
    const h = state.metric.forward;
    ctx.save(); path(state.geometry.floor, t); ctx.clip(); ctx.globalAlpha = .7;
    // Pull metre-coordinate lines back into image space, without projecting
    // endpoints across an inverse horizon outside a concave floor.
    const line = (axis, value) => {
      const [a,b,c] = h[axis].map((v,i) => v - value*h[2][i]), ends = [];
      if (Math.abs(b) > 1e-12) for (const x of [0,1]) {
        const y = -(a*x+c)/b; if (y >= 0 && y <= 1) ends.push([x,y]);
      }
      if (Math.abs(a) > 1e-12) for (const y of [0,1]) {
        const x = -(b*y+c)/a; if (x >= 0 && x <= 1 && !ends.some(p => Math.hypot(p[0]-x,p[1]-y)<1e-10)) ends.push([x,y]);
      }
      if (ends.length < 2) return;
      ctx.beginPath(); ctx.moveTo(...display(ends[0],t)); ctx.lineTo(...display(ends[1],t));
      ctx.strokeStyle = color; ctx.lineWidth = 1; ctx.stroke();
    };
    for (let i = Math.ceil(xmin/spacing); i <= Math.floor(xmax/spacing); i++) line(0,i*spacing);
    for (let i = Math.ceil(ymin/spacing); i <= Math.floor(ymax/spacing); i++) line(1,i*spacing);
    ctx.restore();
  }
  function draw() {
    const t = transform(); if (!t.w || !t.h) return;
    const dpr = window.devicePixelRatio || 1, width = Math.round(t.w*dpr), height = Math.round(t.h*dpr);
    if (canvas.width !== width || canvas.height !== height) { canvas.width = width; canvas.height = height; }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, t.w, t.h);
    if (!state.image) return;
    ctx.drawImage(state.image, t.x, t.y, t.iw, t.ih);
    ctx.save(); ctx.beginPath(); ctx.rect(t.x, t.y, t.iw, t.ih); ctx.clip();
    const css = getComputedStyle(root), green = css.getPropertyValue('--vid-ok').trim(), amber = css.getPropertyValue('--vid-warn').trim(), orange = css.getPropertyValue('--vid-track').trim();
    if (state.layers.floor) stroke(state.geometry.floor, t, green);
    if (state.layers.objects) for (const o of state.geometry.objects) stroke(o.polygon, t, orange);
    if (state.layers.metric) { drawGrid(t, green); stroke(state.geometry.metric?.corners, t, amber); }
    if (state.layers.gate) drawArrow(t, amber);
    if (state.layers.suggestions) state.candidates.forEach((candidate, i) => { ctx.globalAlpha = String(i) === $('scene-ai-candidate').value ? 1 : .4; stroke(candidate.points, t, '#FFFFFF', true, true); });
    ctx.globalAlpha = 1;
    for (const shape of allShapes()) {
      if (!shape.visible) continue;
      (shape.points || []).forEach((p, i) => {
        if (!point(p)) return;
        const [x, y] = display(p, t), selected = shape.id === state.selection;
        ctx.beginPath(); ctx.arc(x, y, selected && i === state.vertex ? 7 : 4, 0, Math.PI*2);
        ctx.fillStyle = selected ? amber : '#FFFFFF'; ctx.fill(); ctx.strokeStyle = '#1C1B1F'; ctx.lineWidth = 1.5; ctx.stroke();
        if (selected) { ctx.font = '12px sans-serif'; ctx.lineWidth = 3; ctx.strokeText(String(i+1), x+9, y-7); ctx.fillText(String(i+1), x+9, y-7); }
      });
    }
    if (state.tool === 'ai') {
      for (const [key, color, symbol] of [['positive_points', green, '+'], ['negative_points', orange, '−']]) {
        for (const p of state.prompts[key]) { const [x,y] = display(p,t); ctx.fillStyle = '#1C1B1F'; ctx.beginPath(); ctx.arc(x,y,10,0,Math.PI*2); ctx.fill(); ctx.fillStyle = color; ctx.font = 'bold 17px sans-serif'; ctx.fillText(symbol,x-5,y+5); }
      }
      const box = state.prompts.box;
      if (box) stroke([[box[0],box[1]],[box[2],box[1]],[box[2],box[3]],[box[0],box[3]]],t,amber,true,true);
    }
    ctx.restore();
  }
  function drawRectified() {
    const metric = state.metric;
    $('scene-rectified').hidden = !metric || !state.image;
    if (!metric || !state.image) { state.previewKey = ''; return; }
    const key = JSON.stringify([state.snapshot.snapshot_id, state.geometry.floor, state.geometry.metric, sceneDirty()]);
    if (key === state.previewKey) return;
    // Bound raster allocation by aspect, never by metre dimensions.
    const scale = 1280 / Math.max(metric.width, metric.depth);
    const width = Math.max(1, Math.round(metric.width*scale)), height = Math.max(1, Math.round(metric.depth*scale));
    const target = $('scene-rectified-canvas'); target.width = width; target.height = height;
    const out = target.getContext('2d'), result = out.createImageData(width, height);
    if (!state.pixels) {
      const native = document.createElement('canvas'); native.width = state.image.naturalWidth; native.height = state.image.naturalHeight;
      const source = native.getContext('2d', {willReadFrequently: true}); source.drawImage(state.image, 0, 0);
      state.pixels = source.getImageData(0, 0, native.width, native.height);
    }
    const source = state.pixels, floor = state.geometry.floor;
    for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
      const p = project(metric.inverse, [(x+.5)/width*metric.width, (y+.5)/height*metric.depth]);
      if (!p || !point(p)) continue;
      const sx = Math.min(source.width-1, Math.floor(p[0]*source.width)), sy = Math.min(source.height-1, Math.floor(p[1]*source.height));
      if (sx < 0 || sy < 0) continue;
      const a = (sy*source.width+sx)*4, b = (y*width+x)*4;
      result.data[b] = source.data[a]; result.data[b+1] = source.data[a+1]; result.data[b+2] = source.data[a+2]; result.data[b+3] = 255;
    }
    out.putImageData(result,0,0);
    // Mask once in the rasterizer rather than testing 128 edges per pixel.
    out.save(); out.globalCompositeOperation = 'destination-in'; out.beginPath();
    floor.forEach((p,i) => {
      const q = project(metric.forward,p), x = q[0]/metric.width*width, y = q[1]/metric.depth*height;
      if (i) out.lineTo(x,y); else out.moveTo(x,y);
    });
    out.closePath(); out.fill(); out.restore();
    const {spacing} = gridSpec(metric);
    out.save(); out.globalCompositeOperation = 'source-atop'; out.strokeStyle = getComputedStyle(root).getPropertyValue('--vid-ok').trim(); out.lineWidth = 1;
    out.beginPath();
    for (let x=0; x<=metric.width; x+=spacing) { out.moveTo(x/metric.width*width,0); out.lineTo(x/metric.width*width,height); }
    for (let y=0; y<=metric.depth; y+=spacing) { out.moveTo(0,y/metric.depth*height); out.lineTo(width,y/metric.depth*height); }
    out.stroke(); out.restore();
    $('scene-grid-info').textContent = `Шаг сетки ${spacing} м · ${width} × ${height} пикс. · ${sceneDirty() ? 'несохранённый черновик' : 'сохранённые координаты'}. Показана область измеренного прямоугольника; прозрачное — вне видимого пола.`;
    state.previewKey = key;
  }
  function schedulePreview() {
    clearTimeout(state.previewTimer);
    state.previewTimer = setTimeout(() => { try { drawRectified(); } catch (_) { $('scene-rectified').hidden = true; message('Не удалось построить проекцию в браузере. Черновик сохранён.', 'err'); } }, 180);
  }
  function syncGateFields() {
    gateNames.forEach((name, i) => ['x','y'].forEach((axis,j) => { $(`gate-${name}-${axis}`).value = state.gate.points[i]?.[j] ?? ''; }));
    $('calibration-role').value = state.gate.role; $('calibration-enabled').checked = state.gate.enabled;
    for (const option of $('calibration-role').options) option.disabled = Boolean(state.config?.role && option.value !== state.config.role);
  }
  function renderCoordinates() {
    const list = $('scene-coordinate-list'); list.replaceChildren();
    const points = selectedPoints();
    if (!points) { const p = document.createElement('p'); p.textContent = 'Выберите контур, вход или масштаб.'; list.append(p); return; }
    points.forEach((p, i) => {
      const row = document.createElement('div'); row.className = 'scene-coordinate-row';
      const select = document.createElement('button'); select.type = 'button'; select.className = 'btn'; select.textContent = String(i+1); select.setAttribute('aria-label', `Выбрать вершину ${i+1}`); select.setAttribute('aria-pressed', String(state.vertex === i));
      select.addEventListener('click', () => { state.vertex = i; draw(); renderCoordinates(); }); row.append(select);
      ['X','Y'].forEach((axis,j) => {
        const label = document.createElement('label'); label.className = 'field'; label.textContent = axis;
        const input = document.createElement('input'); input.type = 'number'; input.min = '0'; input.max = '1'; input.step = 'any'; input.value = p?.[j] ?? ''; input.disabled = !editable();
        input.setAttribute('aria-label', `Вершина ${i+1}, ${axis}`);
        input.addEventListener('change', () => {
          if (!editable()) return;
          const value = input.value === '' ? NaN : Number(input.value);
          if (!Number.isFinite(value) || value < 0 || value > 1) { message('Координата должна быть от 0 до 1.', 'err'); return; }
          checkpoint(); const next = [...(points[i] || [null,null])]; next[j] = value; points[i] = next; state.vertex = i; sync();
        }); label.append(input); row.append(label);
      });
      const remove = document.createElement('button'); remove.className = 'btn'; remove.type = 'button'; remove.setAttribute('aria-label', `Удалить вершину ${i+1}`); remove.disabled = !editable();
      const icon = document.createElementNS('http://www.w3.org/2000/svg','svg');
      icon.setAttribute('viewBox','0 0 24 24'); icon.setAttribute('width','20'); icon.setAttribute('height','20'); icon.setAttribute('aria-hidden','true');
      const lines = document.createElementNS('http://www.w3.org/2000/svg','path');
      lines.setAttribute('d','M6 6L18 18M18 6L6 18'); lines.setAttribute('stroke','currentColor'); lines.setAttribute('stroke-width','2'); lines.setAttribute('stroke-linecap','round');
      icon.append(lines); remove.append(icon);
      remove.addEventListener('click', () => { state.vertex = i; deleteVertex(); }); row.append(remove); list.append(row);
    });
  }
  function sync() {
    const options = [['','Нет выбора'],['floor','Пол'], ...state.geometry.objects.map(o => [o.id, o.label || 'Без названия']), ['gate','Вход'], ['metric','Масштаб']];
    $('scene-selection').replaceChildren(...options.map(([value, label]) => new Option(label,value)));
    $('scene-selection').value = state.selection;
    const object = state.geometry.objects.find(o => o.id === state.selection);
    if (object) $('scene-object-label').value = object.label;
    $('scene-width').value = state.geometry.metric?.width_m ?? ''; $('scene-depth').value = state.geometry.metric?.depth_m ?? '';
    syncGateFields();
    root.querySelectorAll('[data-scene-tool]').forEach(button => button.setAttribute('aria-pressed', String(button.dataset.sceneTool === state.tool)));
    $('scene-ai-panel').hidden = state.tool !== 'ai'; $('scene-tool-hint').textContent = hints[state.tool];
    state.metric = null;
    try { state.metric = compileMetric(); $('scene-metric-status').textContent = state.metric ? 'Проекция черновика построена. Окончательная численная проверка выполняется при сохранении.' : 'Масштаб не задан; положение и ширина в метрах не определяются.'; }
    catch (e) { $('scene-metric-status').textContent = e.message; }
    $('scene-draft-status').textContent = `${sceneDirty() ? 'Сцена: есть несохранённые изменения.' : 'Сцена: без изменений.'} ${gateDirty() ? 'Вход: есть несохранённые изменения.' : 'Вход: без изменений.'} Эти две разметки сохраняются отдельно.`;
    buttons(); renderCoordinates(); draw(); schedulePreview(); gateWidth();
  }
  function buttons() {
    root.setAttribute('aria-busy', String(state.busy));
    $('calibration-freeze').disabled = state.busy;
    $('scene-undo').disabled = !editable() || !state.undo.length; $('scene-redo').disabled = !editable() || !state.redo.length;
    $('scene-save').disabled = !saveable() || !state.saved || Boolean(geometryError());
    $('calibration-save').disabled = !saveable() || !state.config || Boolean(gateError());
    $('calibration-disable').disabled = state.busy || !state.config?.role || !state.config?.enabled;
    $('calibration-clear').disabled = !editable();
    $('scene-reconcile').hidden = !state.review;
    $('scene-keep-draft').disabled = !state.validSnapshot || state.busy || !$('scene-reviewed').checked;
    $('scene-use-saved').disabled = !state.validSnapshot || state.busy;
    $('scene-ai-submit').disabled = !saveable() || Boolean(state.job);
    $('scene-ai-cancel').disabled = !state.job;
    $('scene-ai-accept').disabled = !saveable() || !state.candidates.length;
    root.querySelectorAll('[data-scene-tool]').forEach(button => { button.disabled = state.busy; });
    for (const id of ['scene-finish','scene-new-object','scene-clear-selected','scene-add-vertex','scene-remove-metric','scene-ai-clear','scene-width','scene-depth','scene-object-label','calibration-role','calibration-enabled']) $(id).disabled = !editable();
    gateNames.forEach(name => ['x','y'].forEach(axis => { $(`gate-${name}-${axis}`).disabled = !editable(); }));
    $('scene-add-vertex').disabled = !editable() || !selectedPoints()?.length || ['gate','metric'].includes(state.selection);
    $('scene-clear-selected').disabled = !editable() || !state.selection;
  }
  function showRuntime() {
    const r = state.runtime; if (!r) return;
    const parts = [r.role === 'ENTRY' ? 'Вход' : r.role === 'EXIT' ? 'Выход' : 'Роль не назначена', r.enabled ? 'учёт включён' : 'учёт отключён', `поток: ${state.cameraStatus}`];
    if (r.camera_enabled === false) parts.push('камера отключена');
    if (r.ai_enabled === false) parts.push('ИИ выключен'); else if (r.ai_included === false) parts.push('камера исключена из ИИ'); else if (r.ai_available === false) parts.push('ИИ недоступен');
    parts.push(r.commissioning_required ? 'наладка не подтверждена — автоматический учёт запрещён' : 'акт наладки сохранён; готовность проверяет монитор учёта');
    $('calibration-state').textContent = parts.join(' · ');
  }
  function showScene() {
    const s = state.saved;
    $('scene-state').textContent = !s ? 'Сцена недоступна.' : `${s.status === 'CURRENT' && state.sceneCurrent ? 'Сохранённая сцена актуальна' : s.status === 'UNSET' ? 'Сцена ещё не сохранена' : 'Сцена требует проверки на текущем источнике'} · версия ${s.revision}.`;
  }
  function markStale(text, sceneStale = true) {
    ++state.sourceEpoch;
    state.validSnapshot = false; state.review = true; $('scene-reviewed').checked = false;
    if (sceneStale) state.sceneCurrent = false;
    $('scene-reconcile-text').textContent = `${text} Черновик не потерян. Зафиксируйте свежий кадр; затем проверьте и явно примените координаты либо загрузите сохранённое.`;
    cancelJob(); buttons(); showScene(); gateWidth();
    if (sceneStale) document.querySelectorAll('.bbox-metric').forEach(node => { node.textContent = 'Положение не определено: сцена устарела'; });
  }
  function errorMessage(error, id = 'scene-message') {
    const code = error.detail?.code;
    if (error.status === 409 || error.status === 410) markStale(code || 'Кадр истёк, источник или конфигурация изменились.', error.status !== 410);
    const known = {SCENE_MODEL_UNAVAILABLE: 'Модель ИИ-подсказок недоступна. Ручная разметка работает.', SCENE_INFERENCE_FAILED: 'Ошибка модели ИИ. Предложение не создано; ручная разметка работает.', SCENE_BUSY: 'Сервис занят или лимит замороженных кадров исчерпан. Дождитесь завершения/истечения кадров и повторите.', SCENE_GEOMETRY_INVALID: 'Геометрия отклонена сервером. Проверьте пересечения и масштаб; можно убрать масштаб и сохранить контуры.'};
    message(known[code] || (error.status === 409 || error.status === 410 ? 'Сохранение не подтверждено. Черновик сохранён; требуется новый кадр и сверка.' : error.detail?.message || error.message || 'Нет связи. Черновик сохранён.'), 'err', id);
  }
  async function request(url, method, payload) {
    return window.OperatorUI.api(url, {method, headers: {'Content-Type':'application/json'}, ...(payload !== undefined ? {body: JSON.stringify(payload)} : {})});
  }
  async function freeze() {
    if (state.busy) return;
    state.busy = true; cancelJob(); buttons(); message('Получение исходного кадра…');
    const epoch = state.sourceEpoch;
    try {
      const [sceneResult, gateResult] = await Promise.allSettled([window.OperatorUI.api(base), window.OperatorUI.api(attendance)]);
      if (sceneResult.status !== 'fulfilled') throw sceneResult.reason;
      const saved = sceneResult.value, config = gateResult.status === 'fulfilled' ? gateResult.value : null;
      if (!config) message('Статус учёта недоступен. Сцена редактируется независимо; сохранение входа недоступно до новой фиксации.', 'err', 'calibration-message');
      const snapshot = await request(`${base}/snapshot`, 'POST', {});
      if (snapshot.camera_id !== cameraId || snapshot.source_revision !== saved.current_source_revision || (config && snapshot.source_revision !== config.source_revision)) throw Object.assign(new Error('Источник сменился при фиксации.'), {status:409});
      const image = new Image(); image.src = `data:image/jpeg;base64,${snapshot.jpeg_base64}`; await image.decode();
      if (image.naturalWidth !== snapshot.width || image.naturalHeight !== snapshot.height || !image.naturalWidth) throw new Error('Размер декодированного JPEG не совпадает с метаданными. Ввод по этому кадру запрещён.');
      if (epoch !== state.sourceEpoch) throw Object.assign(new Error('Источник изменился во время получения кадра.'), {status:409});
      const review = state.review || sceneDirty() || gateDirty() || Boolean(state.snapshot)
        || (saved.status === 'STALE') || (saved.reference_width && (saved.reference_width !== snapshot.width || saved.reference_height !== snapshot.height))
        || (config?.reference_width && (config.reference_width !== snapshot.width || config.reference_height !== snapshot.height));
      if (!review) { state.geometry = clone(saved.geometry || emptyGeometry()); if (config) state.gate = gateFrom(config); }
      state.saved = saved; state.config = config; state.runtime = config; state.sceneCurrent = saved.status === 'CURRENT';
      state.snapshot = snapshot; state.image = image; state.pixels = null; state.validSnapshot = true; state.review = Boolean(review);
      state.zoom = 1; state.pan = [0,0]; state.prompts = {positive_points: [],negative_points: [],box:null};
      $('scene-reviewed').checked = false; $('calibration-image').hidden = false;
      $('scene-reconcile-text').textContent = 'Новый кадр показан вместе с сохранённым черновиком. Координаты ещё не привязаны к этому кадру. Исправьте контуры при необходимости, затем подтвердите сверку.';
      $('calibration-metadata').textContent = `${snapshot.width} × ${snapshot.height} · источник ${snapshot.source_revision} · поток ${snapshot.stream_id} · кадр ${snapshot.seq} · получен сервером ${snapshot.received_at}`;
      clearTimeout(state.ttlTimer); state.ttlTimer = setTimeout(() => markStale('Срок замороженного кадра истёк.', false), Math.max(0, snapshot.expires_in_seconds*1000));
      message(review ? 'Кадр получен. Черновик сохранён; перед сохранением нужна явная сверка.' : 'Кадр зафиксирован. Выберите инструмент разметки.', 'ok'); showRuntime(); showScene();
    } catch (e) { errorMessage(e); }
    finally { state.busy = false; sync(); }
  }
  async function saveScene() {
    if (!saveable() || !state.saved) return;
    if (!$('scene-width').checkValidity() || !$('scene-depth').checkValidity()
        || Array.from($('scene-coordinate-list').querySelectorAll('input')).some(input => !input.checkValidity())) {
      message('Исправьте введённые координаты и измерения перед сохранением.','err'); return;
    }
    const error = geometryError(); if (error) { message(error,'err'); return; }
    state.busy = true; buttons();
    const epoch = state.sourceEpoch;
    try {
      const saved = await request(base,'PUT',{snapshot_id:state.snapshot.snapshot_id,expected_revision:state.saved.revision,expected_source_revision:state.snapshot.source_revision,geometry:clone(state.geometry)});
      state.saved = saved; state.geometry = clone(saved.geometry); state.sceneCurrent = saved.status === 'CURRENT' && epoch === state.sourceEpoch;
      message('Сцена сохранена. Линия входа и наладка не изменены.','ok'); showScene();
      window.dispatchEvent(new CustomEvent('scene-saved',{detail:saved}));
    } catch (e) { errorMessage(e); }
    finally { state.busy = false; sync(); }
  }
  async function saveGate(disable = false) {
    if (state.busy || !state.config || (!disable && !saveable())) return;
    if (!disable && gateError()) { message(gateError(),'err','calibration-message'); return; }
    if (!disable && !$('calibration-form').reportValidity()) return;
    const c = state.config, g = disable ? gateFrom(c) : state.gate;
    const payload = {role:g.role,enabled:disable ? false : g.enabled,line_start:g.points[0],line_end:g.points[1],inside_point:g.points[2],
      reference_width:disable ? c.reference_width : state.snapshot.width,reference_height:disable ? c.reference_height : state.snapshot.height,
      expected_revision:c.revision,expected_source_revision:disable ? c.source_revision : state.snapshot.source_revision};
    state.busy = true; buttons();
    try {
      const saved = await request(attendance,'PUT',payload); state.config = saved; state.runtime = saved;
      if (!disable) state.gate = gateFrom(saved); else state.gate.enabled = false;
      message(disable ? 'Учёт отключён. Черновик геометрии сохранён.' : 'Линия входа сохранена. Требуется повторная наладка; сцена отдельно не сохранялась.','ok','calibration-message'); showRuntime();
    } catch (e) { errorMessage(e,'calibration-message'); }
    finally { state.busy = false; sync(); }
  }
  function gateWidth() {
    let text = 'Ширина входа: не откалибрована.';
    const s = state.saved, [a,b] = state.gate.points;
    const dimensionsMatch = !state.snapshot || (state.snapshot.source_revision === s?.source_revision
      && state.snapshot.width === s?.reference_width && state.snapshot.height === s?.reference_height);
    if (state.sceneCurrent && s?.status === 'CURRENT' && s.metric_transform && dimensionsMatch
        && point(a) && point(b) && contains(a,s.geometry.floor) && contains(b,s.geometry.floor)
        && (!state.review || !gateDirty())) {
      const aa=project(s.metric_transform.image_to_metres,a), bb=project(s.metric_transform.image_to_metres,b);
      if (aa && bb) text = `Приблизительная ширина входа: ${Math.hypot(bb[0]-aa[0],bb[1]-aa[1]).toFixed(2)} м · по сохранённому полу${gateDirty() ? ', линия пока в черновике' : ''}.`;
    }
    $('scene-gate-width').textContent = text;
  }
  function liveMetrics(data) {
    const s = state.saved;
    const changed = (data.source_revision != null && state.liveSource != null && data.source_revision !== state.liveSource)
      || (data.stream_id && state.liveStream && data.stream_id !== state.liveStream);
    const snapshotMismatch = state.validSnapshot && state.snapshot
      && ((data.source_revision != null && data.source_revision !== state.snapshot.source_revision) || (data.stream_id && data.stream_id !== state.snapshot.stream_id));
    if (changed || snapshotMismatch || (state.sceneCurrent && data.source_revision != null && data.source_revision !== s?.source_revision)) markStale('Живой поток заменён.');
    if (data.source_revision != null) state.liveSource = data.source_revision;
    if (data.stream_id) state.liveStream = data.stream_id;
    const labels = new Map(Array.from(document.querySelectorAll('.bbox-metric'), node => [node.dataset.trackId,node]));
    for (const track of data.tracks || []) {
      const node = labels.get(String(track.track_id)); if (!node) continue;
      const box=track.bbox, p=Array.isArray(box) ? [(box[0]+box[2])/2,box[3]] : null;
      const usable = state.sceneCurrent && s?.status === 'CURRENT' && data.source_revision === s.source_revision && s.metric_transform && point(p)
        && data.frame_width === s.reference_width && data.frame_height === s.reference_height
        && box.every(Number.isFinite) && box[0]>0 && box[1]>0 && box[2]<1 && box[3]<1
        && contains(p,s.geometry.floor) && !s.geometry.objects.some(o=>contains(p,o.polygon));
      const mapped=usable ? project(s.metric_transform.image_to_metres,p) : null;
      node.textContent = mapped ? `≈ ${mapped[0].toFixed(2)}, ${mapped[1].toFixed(2)} м` : 'Положение не определено';
      node.title = mapped ? 'Оценка по нижнему центру рамки, не измеренная точка стопы; не доказательство рабочего времени.' : 'Нет актуального масштаба, низ рамки обрезан, вне пола или скрыт силуэтом предмета. Человек остаётся в журнале.';
    }
  }
  function deleteVertex() {
    const p=selectedPoints(); if (!editable() || !p || state.vertex<0 || state.vertex>=p.length) return;
    checkpoint(); if (state.selection==='gate') p[state.vertex]=null; else p.splice(state.vertex,1);
    state.vertex=Math.min(state.vertex,p.length-1); sync();
  }
  function insertVertex(index, value) {
    const p=selectedPoints(); if (!editable() || !p || !p.length || ['gate','metric'].includes(state.selection)) return;
    if (p.length>=128) { message('Не больше 128 вершин в контуре.','err'); return; }
    checkpoint(); p.splice(index+1,0,value); state.vertex=index+1; sync();
  }
  function clearSelected() {
    if (!editable() || !state.selection) return;
    checkpoint();
    if (state.selection==='floor') { state.geometry.floor=null; state.geometry.metric=null; }
    else if (state.selection==='metric') state.geometry.metric=null;
    else if (state.selection==='gate') state.gate.points=[null,null,null];
    else state.geometry.objects=state.geometry.objects.filter(o=>o.id!==state.selection);
    state.vertex=-1; sync();
  }
  function hitVertex(position) {
    const t=transform(); let best=null, distance=14;
    for (const shape of allShapes()) if (shape.visible) (shape.points||[]).forEach((p,index)=>{
      if (!point(p)) return; const q=display(p,t), d=Math.hypot(q[0]-position[0],q[1]-position[1]);
      if (d<distance) { distance=d; best={id:shape.id,index}; }
    }); return best;
  }
  function edgeHit(position) {
    const t=transform(); let best=null, distance=12;
    for (const shape of allShapes()) if (shape.visible && !['gate','metric'].includes(shape.id) && shape.points?.length>=2) {
      const p=shape.points;
      for (let i=0;i<p.length;i++) {
        const a=display(p[i],t), b=display(p[(i+1)%p.length],t), dx=b[0]-a[0],dy=b[1]-a[1], l=dx*dx+dy*dy;
        if (!l) continue; const u=Math.max(0,Math.min(1,((position[0]-a[0])*dx+(position[1]-a[1])*dy)/l));
        const q=[a[0]+u*dx,a[1]+u*dy], d=Math.hypot(q[0]-position[0],q[1]-position[1]);
        if (d<distance) { distance=d; best={id:shape.id,index:i,point:imagePoint(q)}; }
      }
    } return best;
  }
  function addPoint(p) {
    if (!editable()) return;
    if (state.tool==='select') {
      const match=allShapes().filter(s=>s.visible && s.id!=='gate' && contains(p,s.points)).pop();
      state.selection=match?.id||''; state.vertex=-1; sync(); return;
    }
    if (state.tool==='ai') {
      const kind=$('scene-ai-prompt').value; if (kind==='box') return;
      if (state.prompts.positive_points.length+state.prompts.negative_points.length>=32) { message('Не больше 32 точек подсказки.','err'); return; }
      cancelJob(); state.prompts[`${kind}_points`].push(p); draw(); return;
    }
    if (!state.drawing) return;
    checkpoint();
    if (state.tool==='floor') { state.selection='floor'; if (!state.geometry.floor) state.geometry.floor=[]; }
    if (state.tool==='object' && !state.geometry.objects.some(o=>o.id===state.selection) && !newObject()) { state.undo.pop(); return; }
    if (state.tool==='metric') { state.selection='metric'; if (!state.geometry.metric) state.geometry.metric={corners:[],width_m:null,depth_m:null}; }
    if (state.tool==='gate') {
      state.selection='gate'; const index=state.gate.points.findIndex(p=>!point(p));
      if (index<0) { state.undo.pop(); message('Три точки уже заданы. Перетащите их или разметьте вход заново.'); return; }
      state.gate.points[index]=p; state.vertex=index;
    } else {
      const points=selectedPoints(), limit=state.tool==='metric'?4:128;
      if (points.length>=limit) { state.undo.pop(); message(`Допускается не больше ${limit} вершин. Перетащите имеющиеся.`); return; }
      points.push(p); state.vertex=points.length-1;
    }
    sync();
  }
  function zoom(factor, anchor = null) {
    if (!state.image) return;
    const before=transform(), at=anchor||[before.w/2,before.h/2], p=imagePoint(at,true);
    state.zoom=Math.max(.5,Math.min(12,state.zoom*factor));
    const after=transform(), q=display(p,after); state.pan[0]+=at[0]-q[0]; state.pan[1]+=at[1]-q[1]; draw();
  }
  canvas.addEventListener('pointerdown',event=>{
    if (!editable() || state.gesture || (event.button!==0 && event.button!==1)) return;
    const position=eventPosition(event), p=imagePoint(position);
    if (!p) return;
    canvas.focus({preventScroll:true}); canvas.setPointerCapture(event.pointerId); event.preventDefault();
    if (state.space || event.button===1 || $('scene-pan').checked) { state.gesture={kind:'pan',start:position,pan:[...state.pan],id:event.pointerId}; return; }
    if (state.tool==='ai' && $('scene-ai-prompt').value==='box') { cancelJob(); state.gesture={kind:'box',start:p,id:event.pointerId}; return; }
    const hit=state.tool==='ai'?null:hitVertex(position);
    if (hit) { state.selection=hit.id; state.vertex=hit.index; checkpoint(); state.gesture={kind:'vertex',id:event.pointerId}; renderCoordinates(); draw(); return; }
    state.gesture={kind:'click',start:position,id:event.pointerId};
  });
  canvas.addEventListener('pointermove',event=>{
    const g=state.gesture; if (!g || g.id!==event.pointerId) return;
    const position=eventPosition(event), p=imagePoint(position,true);
    if (g.kind==='pan') { state.pan=[g.pan[0]+position[0]-g.start[0],g.pan[1]+position[1]-g.start[1]]; draw(); }
    if (g.kind==='vertex') { selectedPoints()[state.vertex]=p; state.metric=null; draw(); }
    if (g.kind==='box') { state.prompts.box=[Math.min(g.start[0],p[0]),Math.min(g.start[1],p[1]),Math.max(g.start[0],p[0]),Math.max(g.start[1],p[1])]; draw(); }
  });
  function endPointer(event) {
    const g=state.gesture; if (!g || g.id!==event.pointerId) return;
    state.gesture=null;
    if (canvas.hasPointerCapture(event.pointerId)) canvas.releasePointerCapture(event.pointerId);
    if (g.kind==='click' && event.type==='pointerup') {
      const pos=eventPosition(event), p=imagePoint(pos); if (p && Math.hypot(pos[0]-g.start[0],pos[1]-g.start[1])<5) addPoint(p);
    }
    sync();
  }
  canvas.addEventListener('pointerup',endPointer); canvas.addEventListener('pointercancel',endPointer);
  canvas.addEventListener('dblclick',event=>{
    if (!editable() || state.tool!=='select' || !imagePoint(eventPosition(event))) return;
    const edge=edgeHit(eventPosition(event)); if (edge?.point) { state.selection=edge.id; insertVertex(edge.index,edge.point); }
  });
  canvas.addEventListener('wheel',event=>{ if (!state.image || !imagePoint(eventPosition(event))) return; event.preventDefault(); zoom(Math.exp(-event.deltaY*.001),eventPosition(event)); },{passive:false});
  canvas.addEventListener('keydown',event=>{
    if (event.code==='Space') { state.space=true; event.preventDefault(); }
    if (event.key==='Escape') { state.drawing=false; setTool('select'); }
    if (event.key==='Delete' || event.key==='Backspace') { event.preventDefault(); deleteVertex(); }
    if (event.key.startsWith('Arrow') && editable()) {
      const points=selectedPoints(), p=points?.[state.vertex]; if (!point(p)) return;
      event.preventDefault(); checkpoint(); const step=event.shiftKey?10:1;
      const dx=event.key==='ArrowLeft'?-step:event.key==='ArrowRight'?step:0,dy=event.key==='ArrowUp'?-step:event.key==='ArrowDown'?step:0;
      points[state.vertex]=[Math.max(0,Math.min(1,p[0]+dx/state.image.naturalWidth)),Math.max(0,Math.min(1,p[1]+dy/state.image.naturalHeight))]; sync();
    }
  });
  canvas.addEventListener('keyup',event=>{ if (event.code==='Space') state.space=false; });
  canvas.addEventListener('blur',()=>{state.space=false;});
  root.addEventListener('keydown',event=>{
    if (['INPUT','TEXTAREA','SELECT'].includes(event.target.tagName)) return;
    if ((event.ctrlKey||event.metaKey) && event.key.toLowerCase()==='z') { event.preventDefault(); history(!event.shiftKey); }
  });
  function clearCandidates() { state.candidates=[]; $('scene-ai-candidate').replaceChildren(new Option('Нет предложений','')); }
  async function cancelJob() {
    ++state.jobEpoch; clearTimeout(state.jobTimer); const job=state.job || state.completedJob; state.job=null; state.completedJob=null; clearCandidates(); buttons(); draw();
    if (!job?.id) return;
    try { await request(`${base}/suggestions/${encodeURIComponent(job.id)}`,'DELETE'); }
    catch (e) { if (e.status!==410) $('scene-ai-status').textContent='Отмена на сервере не подтверждена. Поздний результат будет проигнорирован; повторный запрос может временно получить SCENE_BUSY.'; }
  }
  async function submitSuggestion() {
    if (!saveable() || state.job) return;
    const prompts=state.prompts;
    if (!prompts.positive_points.length && !prompts.negative_points.length && !prompts.box) { message('Добавьте точки или рамку для ИИ.','err'); return; }
    if (prompts.box && (prompts.box[0]>=prompts.box[2] || prompts.box[1]>=prompts.box[3])) { message('Рамка должна иметь ненулевую ширину и высоту.','err'); return; }
    if ($('scene-ai-kind').value==='object' && !$('scene-object-label').value.trim()) { message('Введите название объекта.','err'); return; }
    // Superseded terminal records are cancelled too; the server owns their TTL.
    cancelJob();
    clearCandidates(); const epoch=++state.jobEpoch, snapshot=state.snapshot.snapshot_id, started=performance.now(), kind=$('scene-ai-kind').value;
    state.job={id:null,snapshot,epoch,kind,started}; buttons(); $('scene-ai-status').textContent='Отправка запроса…';
    try {
      const result=await request(`${base}/suggestions`,'POST',{snapshot_id:snapshot,kind,...clone(prompts)});
      if (epoch!==state.jobEpoch || state.snapshot?.snapshot_id!==snapshot || !saveable()) {
        await request(`${base}/suggestions/${encodeURIComponent(result.job_id)}`,'DELETE'); return;
      }
      state.job.id=result.job_id; pollJob(state.job);
    } catch (e) { if (epoch===state.jobEpoch) { state.job=null; $('scene-ai-status').textContent=`Предложение недоступно: ${e.detail?.code||e.message}. Ручная разметка работает.`; errorMessage(e); buttons(); } }
  }
  async function pollJob(job) {
    try {
      const result=await window.OperatorUI.api(`${base}/suggestions/${encodeURIComponent(job.id)}`);
      if (job.epoch!==state.jobEpoch || state.job!==job || state.snapshot?.snapshot_id!==job.snapshot || !state.validSnapshot) return;
      if (result.snapshot_id!==job.snapshot || result.job_id!==job.id) { markStale('Ответ ИИ относится к другому кадру.'); return; }
      if (result.state==='QUEUED' || result.state==='RUNNING') {
        $('scene-ai-status').textContent=result.state==='QUEUED'?'В очереди локального CPU…':'Модель строит контур на CPU…';
        state.jobTimer=setTimeout(()=>pollJob(job),600); return;
      }
      state.job=null; state.completedJob=job;
      if (result.state==='SUCCEEDED') {
        state.candidates=(result.polygons||[]).slice(0,3).filter(p=>p.needs_review===true && Number.isFinite(p.score) && !polygonError(p.points)).map(p=>({...p,kind:job.kind}));
        $('scene-ai-candidate').replaceChildren(...state.candidates.map((p,i)=>new Option(`Кандидат ${i+1} · оценка модели ${p.score.toFixed(3)}`,String(i))));
        $('scene-ai-status').textContent=`${((performance.now()-job.started)/1000).toFixed(1)} с · ${state.candidates.length ? 'Проверьте пунктирный контур и примите в черновик. Ничего не сохранено.' : 'Пригодных простых контуров нет. Уточните подсказки или обведите вручную.'}`;
      } else $('scene-ai-status').textContent=result.state==='CANCELLED'?'Запрос отменён.':`ИИ не создал контур: ${result.error_code||'SCENE_INFERENCE_FAILED'}. Ручная разметка доступна.`;
      buttons(); draw();
    } catch (e) { if (job.epoch===state.jobEpoch) { state.job=null; $('scene-ai-status').textContent=`Запрос не завершён: ${e.detail?.code||e.message}.`; errorMessage(e); buttons(); } }
  }
  function acceptSuggestion() {
    if (!saveable()) return;
    const candidate=state.candidates[Number($('scene-ai-candidate').value)]; if (!candidate) return;
    if (candidate.kind==='object' && (!$('scene-object-label').value.trim() || state.geometry.objects.length>=32)) { message('Введите название; допустимо не больше 32 объектов.','err'); return; }
    checkpoint();
    if (candidate.kind==='floor') { state.geometry.floor=clone(candidate.points); state.selection='floor'; }
    else { const object=newObject(); if (!object) { state.undo.pop(); return; } object.polygon=clone(candidate.points); }
    cancelJob(); state.tool='select'; state.drawing=false; state.vertex=-1; message('Предложение принято только в черновик. Проверьте вершины и сохраните сцену отдельно.','ok'); sync();
  }
  root.querySelectorAll('[data-scene-tool]').forEach(button=>button.addEventListener('click',()=>setTool(button.dataset.sceneTool)));
  root.querySelectorAll('[data-scene-layer]').forEach(input=>input.addEventListener('change',()=>{state.layers[input.dataset.sceneLayer]=input.checked;draw();}));
  $('calibration-freeze').addEventListener('click',freeze);
  $('scene-save').addEventListener('click',saveScene);
  $('calibration-form').addEventListener('submit',event=>{event.preventDefault();saveGate();});
  $('calibration-disable').addEventListener('click',()=>saveGate(true));
  $('scene-reviewed').addEventListener('change',buttons);
  $('scene-keep-draft').addEventListener('click',()=>{
    if (!state.validSnapshot || !$('scene-reviewed').checked || state.busy) return;
    state.review=false; state.undo=[]; state.redo=[]; message('Проверенный черновик привязан к новому кадру. Сохраните сцену и вход отдельными действиями.'); sync();
  });
  $('scene-use-saved').addEventListener('click',()=>{
    if (!state.validSnapshot || state.busy) return;
    state.undo=[]; state.redo=[]; state.geometry=clone(state.saved.geometry||emptyGeometry()); if (state.config) state.gate=gateFrom(state.config);
    // Imported or dimension-mismatched saved coordinates still require explicit review.
    state.review=state.saved.status==='STALE' || Boolean(state.saved.reference_width && (state.saved.reference_width!==state.snapshot.width || state.saved.reference_height!==state.snapshot.height))
      || Boolean(state.config?.reference_width && (state.config.reference_width!==state.snapshot.width || state.config.reference_height!==state.snapshot.height));
    $('scene-reviewed').checked=false; state.selection=''; state.vertex=-1; sync();
  });
  $('scene-undo').addEventListener('click',()=>history(true)); $('scene-redo').addEventListener('click',()=>history(false));
  $('scene-selection').addEventListener('change',()=>{state.selection=$('scene-selection').value;state.vertex=-1;state.tool='select';state.drawing=false;sync();});
  $('scene-object-label').addEventListener('change',()=>{const o=state.geometry.objects.find(o=>o.id===state.selection);if(o&&editable()){checkpoint();o.label=$('scene-object-label').value.trim();sync();}});
  $('scene-new-object').addEventListener('click',()=>{if(!editable())return;checkpoint();const o=newObject();if(!o){state.undo.pop();return;}state.tool='object';state.drawing=true;state.vertex=-1;sync();});
  $('scene-finish').addEventListener('click',()=>{
    const p=selectedPoints(), error=state.selection==='gate'?gateError():polygonError(p);
    if(error){message(error,'err');return;}state.drawing=false;state.tool='select';sync();message('Контур завершён в черновике. Можно перемещать и добавлять вершины.','ok');
  });
  $('scene-clear-selected').addEventListener('click',clearSelected);
  $('calibration-clear').addEventListener('click',()=>{if(!editable())return;checkpoint();state.gate.points=[null,null,null];setTool('gate');});
  $('scene-add-vertex').addEventListener('click',()=>{
    const p=selectedPoints();if(!p?.length)return;const i=state.vertex>=0?state.vertex:p.length-1,a=p[i],b=p[(i+1)%p.length];if(point(a)&&point(b))insertVertex(i,[(a[0]+b[0])/2,(a[1]+b[1])/2]);
  });
  $('scene-remove-metric').addEventListener('click',()=>{if(!editable())return;checkpoint();state.geometry.metric=null;sync();});
  for(const [id,key] of [['scene-width','width_m'],['scene-depth','depth_m']]) $(id).addEventListener('change',()=>{
    if(!editable())return;checkpoint();if(!state.geometry.metric)state.geometry.metric={corners:[],width_m:null,depth_m:null};
    state.geometry.metric[key]=$(id).value===''?null:Number($(id).value);sync();
  });
  gateNames.forEach((name,i)=>['x','y'].forEach((axis,j)=>$(`gate-${name}-${axis}`).addEventListener('change',()=>{
    if(!editable())return;const value=$(`gate-${name}-${axis}`).value;if(value==='' || !Number.isFinite(Number(value)) || Number(value)<0 || Number(value)>1){message('Введите координату от 0 до 1.','err','calibration-message');return;}
    checkpoint();const p=state.gate.points[i]?[...state.gate.points[i]]:[null,null];p[j]=Number(value);state.gate.points[i]=p;sync();
  })));
  $('calibration-role').addEventListener('change',()=>{if(!editable())return;checkpoint();state.gate.role=$('calibration-role').value;sync();});
  $('calibration-enabled').addEventListener('change',()=>{if(!editable())return;checkpoint();state.gate.enabled=$('calibration-enabled').checked;sync();});
  $('scene-fit').addEventListener('click',()=>{state.zoom=1;state.pan=[0,0];draw();});
  $('scene-zoom-in').addEventListener('click',()=>zoom(1.25)); $('scene-zoom-out').addEventListener('click',()=>zoom(.8));
  $('scene-ai-submit').addEventListener('click',submitSuggestion); $('scene-ai-cancel').addEventListener('click',cancelJob);
  $('scene-ai-accept').addEventListener('click',acceptSuggestion); $('scene-ai-candidate').addEventListener('change',draw);
  $('scene-ai-clear').addEventListener('click',()=>{cancelJob();state.prompts={positive_points:[],negative_points:[],box:null};draw();});
  $('scene-ai-kind').addEventListener('change',cancelJob);
  new ResizeObserver(draw).observe($('calibration-image'));
  window.addEventListener('resize',draw);
  window.addEventListener('camera-state',event=>{
    const c=event.detail;state.cameraStatus=c.status||'OFFLINE';showRuntime();
    if ((c.enabled===false && state.sceneCurrent) || (state.validSnapshot && state.snapshot && c.source_revision!=null && c.source_revision!==state.snapshot.source_revision)
        || (state.sceneCurrent && c.source_revision!=null && c.source_revision!==state.saved?.source_revision)) markStale('Источник камеры изменён или отключён.');
  });
  window.addEventListener('camera-detections',event=>liveMetrics(event.detail));
  window.addEventListener('cameras-changed',()=>markStale('Подключение камеры изменено.'));
  window.addEventListener('beforeunload',event=>{if(sceneDirty()||gateDirty()){event.preventDefault();event.returnValue='';}});
  window.addEventListener('pagehide',()=>{
    clearTimeout(state.jobTimer);clearTimeout(state.ttlTimer);clearTimeout(state.previewTimer);
    if(state.job?.id)fetch(`${base}/suggestions/${encodeURIComponent(state.job.id)}`,{method:'DELETE',headers:{'X-Operator-Request':'1'},keepalive:true}).catch(()=>{});
  });
  async function load() {
    state.busy=true;buttons();
    const results=await Promise.allSettled([window.OperatorUI.api(base),window.OperatorUI.api(attendance)]);
    if(results[0].status==='fulfilled'){state.saved=results[0].value;state.geometry=clone(state.saved.geometry||emptyGeometry());state.sceneCurrent=state.saved.status==='CURRENT';showScene();}else errorMessage(results[0].reason);
    if(results[1].status==='fulfilled'){state.config=results[1].value;state.runtime=state.config;state.gate=gateFrom(state.config);showRuntime();}else errorMessage(results[1].reason,'calibration-message');
    state.busy=false;sync();
  }
  load();
  // Reuses the prior attendance status cadence, never a second detection poll.
  setInterval(async()=>{try{state.runtime=await window.OperatorUI.api(attendance);showRuntime();}catch(e){$('calibration-state').textContent=`Статус учёта недоступен. ${e.message}`;}},5000);
})();
