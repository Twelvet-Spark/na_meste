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
  const emptyGeometry = () => ({version: 2, floor: null, objects: [], doors: [], metric: null});
  const emptyGate = () => ({threshold: [null, null], top: [null, null], inside: null, side: 1, kind: 'DOOR', role: 'BOTH', enabled: false});
  const state = {
    saved: null, config: null, runtime: null, camera: null, geometry: emptyGeometry(), gate: emptyGate(),
    snapshot: null, image: null, pixels: null, validSnapshot: false, review: false,
    busy: false, mode: 'scene', tool: 'select', selection: '', vertex: -1, drawing: false,
    undo: [], redo: [], zoom: 1, pan: [0, 0], space: false, gesture: null,
    layers: {floor: true, objects: true, doors: true, gate: true, metric: true, suggestions: true},
    prompts: {positive_points: [], negative_points: [], box: null},
    candidates: [], analysisItems: [], job: null, jobEpoch: 0, jobTimer: null, ttlTimer: null,
    previewTimer: null, metric: null, cameraStatus: 'OFFLINE', sceneCurrent: false,
    sourceEpoch: 0, previewKey: '', liveSource: null, liveStream: null, completedJob: null,
  };
  const hints = {
    select: 'Выберите вершину и перетащите её. У обычного контура двойной щелчок по ребру вставляет вершину.',
    floor: 'Отметьте видимый пол по контуру, затем завершите. Не дорисовывайте скрытую геометрию по догадке.',
    object: 'Введите название объекта и отметьте его силуэт. Новый объект создаётся отдельной кнопкой.',
    door: 'Два щелчка задают порог. Верх появится автоматически; затем подгоните четыре угла под проём.',
    gate: 'Два щелчка задают порог учёта. Подгоните четыре угла и проверьте, где внутри и снаружи.',
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
  function doorError(quad, minimum = .02) {
    if (!Array.isArray(quad) || quad.length !== 4 || !quad.every(point)) return 'У прохода должны быть четыре угла в пределах кадра.';
    if (Math.hypot(quad[1][0] - quad[0][0], quad[1][1] - quad[0][1]) < minimum - 1e-12) return `Разнесите концы порога минимум на ${String(minimum).replace('.', ',')}.`;
    if (quad[3][1] >= quad[0][1] || quad[2][1] >= quad[1][1]) return 'Верх каждой стойки должен быть строго выше её основания.';
    return '';
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
  function compileMetric(geometry = state.geometry) {
    const m = geometry.metric, floor = geometry.floor;
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
  function geometryError(g = state.geometry) {
    if (g.version !== 2 || !Array.isArray(g.objects) || !Array.isArray(g.doors)) return 'Нужна геометрия сцены версии 2.';
    if (g.objects.length > 32 || g.doors.length > 8) return 'Допускается не больше 32 объектов и 8 проходов.';
    const ids = new Set();
    for (const item of [...g.objects, ...g.doors]) {
      if (typeof item.id !== 'string' || !/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(item.id) || ids.has(item.id.toLowerCase())) return 'Идентификаторы объектов и проходов должны быть уникальными UUID.';
      ids.add(item.id.toLowerCase());
      if (typeof item.label !== 'string' || !item.label.trim() || item.label.length > 80) return 'Каждому объекту и проходу нужно название от 1 до 80 символов.';
    }
    if (g.floor !== null && polygonError(g.floor)) return `Пол: ${polygonError(g.floor)}`;
    for (const o of g.objects) if (polygonError(o.polygon)) return `${o.label}: ${polygonError(o.polygon)}`;
    for (const d of g.doors) {
      if (!['DOOR', 'PASSAGE'].includes(d.kind)) return 'Выберите тип: дверь или проход.';
      if (doorError(d.quad)) return `${d.label}: ${doorError(d.quad)}`;
    }
    try { compileMetric(g); } catch (e) { return e.message; }
    return '';
  }
  const gateQuad = gate => [gate.threshold[0], gate.threshold[1], gate.top[1], gate.top[0]];
  function gateError(g = state.gate, config = state.config) {
    const error = doorError(gateQuad(g), .05);
    if (error) return error;
    if (!point(g.inside)) return 'Проход у края кадра: внутренняя сторона не помещается — сдвиньте порог';
    const [a, b] = g.threshold, length = Math.hypot(b[0] - a[0], b[1] - a[1]);
    if (Math.abs(cross(a, b, g.inside)) / length < .05 - 1e-12) return 'Внутренняя точка должна быть минимум в 0,05 от порога.';
    if (!['DOOR', 'PASSAGE'].includes(g.kind)) return 'Выберите тип прохода.';
    if (!['ENTRY', 'EXIT', 'BOTH'].includes(g.role)) return config?.role ? 'Назначенную роль нельзя стереть.' : 'Выберите, что считать.';
    return '';
  }
  function gateFrom(config) {
    const threshold = [config.line_start, config.line_end].map(p => p ? [...p] : null);
    const top = config.door_top ? clone(config.door_top) : [null, null];
    const inside = config.inside_point ? [...config.inside_point] : null;
    const side = threshold.every(point) && point(inside) ? (cross(...threshold, inside) < 0 ? -1 : 1) : 1;
    return {threshold, top, inside, side, kind: config.door_kind || 'DOOR', role: config.role || 'BOTH', enabled: Boolean(config.enabled)};
  }
  const sceneDirty = () => JSON.stringify(state.geometry) !== JSON.stringify(state.saved?.geometry || emptyGeometry());
  const gateDirty = () => JSON.stringify(state.gate) !== JSON.stringify(state.config ? gateFrom(state.config) : emptyGate());
  const editable = () => Boolean(state.image && state.snapshot && !state.busy);
  const saveable = () => editable() && state.validSnapshot && !state.review;
  const selectedDoor = () => state.geometry.doors.find(d => d.id === state.selection);
  const fixedShape = id => id === 'gate' || state.geometry.doors.some(d => d.id === id);
  // A running analysis would overwrite hand edits; a pending MobileSAM suggestion never touches the geometry.
  function cancelAnalysis() { if ((state.job || state.completedJob)?.kind === 'analysis') cancelJob(); }
  function checkpoint() {
    cancelAnalysis();
    state.undo.push({geometry: clone(state.geometry), gate: clone(state.gate), selection: state.selection, analysisItems: clone(state.analysisItems)});
    if (state.undo.length > 60) state.undo.shift();
    state.redo.length = 0;
  }
  function history(undo) {
    if (!editable()) return;
    cancelAnalysis();
    const from = undo ? state.undo : state.redo, to = undo ? state.redo : state.undo;
    if (!from.length) return;
    to.push({geometry: clone(state.geometry), gate: clone(state.gate), selection: state.selection, analysisItems: clone(state.analysisItems)});
    Object.assign(state, from.pop()); state.vertex = -1; state.drawing = false; sync();
  }
  function selectedPoints() {
    if (state.selection === 'floor') return state.geometry.floor;
    if (state.selection === 'gate') return gateQuad(state.gate);
    if (state.selection === 'metric') return state.geometry.metric?.corners;
    return selectedDoor()?.quad || state.geometry.objects.find(o => o.id === state.selection)?.polygon;
  }
  function moveVertex(index, value) {
    const points = selectedPoints();
    if (!points || !point(value)) return false;
    if (fixedShape(state.selection)) {
      const bottom = index === 2 ? 1 : 0, top = index === 0 ? 3 : 2;
      if ((index >= 2 && point(points[bottom]) && value[1] >= points[bottom][1])
          || (index < 2 && point(points[top]) && value[1] <= points[top][1])) {
        message('Верх стойки должен быть строго выше основания.', 'err'); return false;
      }
    }
    if (state.selection === 'gate') {
      if (index < 2) state.gate.threshold[index] = value;
      else state.gate.top[index === 3 ? 0 : 1] = value;
      state.gate.inside = state.gate.threshold.every(point)
        ? window.Door3D.insidePoint(state.gate.threshold, state.gate.top, state.gate.side) : null;
    } else points[index] = value;
    return true;
  }
  function allShapes() {
    const scene = state.mode === 'scene';
    return [{id: 'floor', points: state.geometry.floor, visible: scene && state.layers.floor},
      ...state.geometry.objects.map(o => ({id: o.id, points: o.polygon, visible: scene && state.layers.objects})),
      ...state.geometry.doors.map(d => ({id: d.id, points: d.quad, visible: scene && state.layers.doors})),
      {id: 'metric', points: state.geometry.metric?.corners, visible: scene && state.layers.metric},
      {id: 'gate', points: gateQuad(state.gate), visible: !scene && state.layers.gate}];
  }
  function setTool(tool) {
    cancelAnalysis();
    state.tool = tool; state.drawing = ['floor', 'object', 'door', 'metric', 'gate'].includes(tool);
    if (tool === 'floor') state.selection = 'floor';
    if (tool === 'gate') state.selection = 'gate';
    if (tool === 'metric') state.selection = 'metric';
    if (tool === 'door') state.selection = '';
    if (tool === 'object' && !state.geometry.objects.some(o => o.id === state.selection)) state.selection = '';
    state.vertex = -1; sync();
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
  function gateMetric() {
    const s = state.saved, [a, b] = state.gate.threshold;
    const dimensionsMatch = !state.snapshot || (state.snapshot.source_revision === s?.source_revision
      && state.snapshot.width === s?.reference_width && state.snapshot.height === s?.reference_height);
    return state.sceneCurrent && s?.status === 'CURRENT' && s.metric_transform && dimensionsMatch
      && !geometryError(s.geometry) && contains(a, s.geometry.floor) && contains(b, s.geometry.floor)
      && (!state.review || !gateDirty()) ? s.metric_transform : null;
  }
  function drawDoorFrame(quad, kind, t, inside = null, role = null, selected = false, metric = null) {
    if (!quad.every(point)) { stroke(quad.slice(0, 2), t, getComputedStyle(root).getPropertyValue('--vid-warn').trim(), false); return; }
    window.Door3D.draw(ctx, {quad, inside, kind, role, display: p => display(p, t), metric, selected});
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
    if (state.layers.doors) for (const d of state.geometry.doors) drawDoorFrame(d.quad, d.kind, t, null, null, state.selection === d.id);
    if (state.mode === 'gate' && state.layers.gate) drawDoorFrame(gateQuad(state.gate), state.gate.kind, t, state.gate.inside, state.gate.role, state.selection === 'gate', gateMetric());
    if (state.layers.suggestions) state.candidates.forEach((candidate, i) => { ctx.globalAlpha = String(i) === $('scene-ai-candidate').value ? 1 : .4; stroke(candidate.points, t, '#FFFFFF', true, true); });
    ctx.globalAlpha = 1;
    for (const shape of allShapes()) {
      if (!shape.visible || !showsHandles(shape)) continue;
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
    $('calibration-role').value = state.gate.role; $('calibration-enabled').checked = state.gate.enabled;
    $('calibration-kind').value = state.gate.kind;
    const selected = $('calibration-scene-door').value;
    $('calibration-scene-door').replaceChildren(new Option('Выберите дверь или проход', ''),
      ...state.geometry.doors.filter(d => !doorError(d.quad)).map(d => new Option(d.label, d.id)));
    $('calibration-scene-door').value = selected;
  }
  function renderCoordinates() {
    const list = $('scene-coordinate-list'); list.replaceChildren();
    const points = selectedPoints();
    if (!points) { const p = document.createElement('p'); p.textContent = 'Выберите контур, проход или масштаб.'; list.append(p); return; }
    points.forEach((p, i) => {
      const row = document.createElement('div'); row.className = 'scene-coordinate-row';
      const select = document.createElement('button'); select.type = 'button'; select.className = 'btn'; select.textContent = String(i+1); select.setAttribute('aria-label', `Выбрать вершину ${i+1}`); select.setAttribute('aria-pressed', String(state.vertex === i));
      select.addEventListener('click', () => { state.vertex = i; draw(); renderCoordinates(); }); row.append(select);
      ['X','Y'].forEach((axis,j) => {
        const label = document.createElement('label'); label.className = 'field'; label.textContent = axis;
        const input = document.createElement('input'); input.type = 'number'; input.min = '0'; input.max = '1'; input.step = 'any'; input.value = p?.[j] ?? ''; input.disabled = !editable() || !point(p);
        input.setAttribute('aria-label', `Вершина ${i+1}, ${axis}`);
        input.addEventListener('change', () => {
          if (!editable()) return;
          const value = input.value === '' ? NaN : Number(input.value);
          if (!Number.isFinite(value) || value < 0 || value > 1) { message('Координата должна быть от 0 до 1.', 'err'); return; }
          checkpoint(); const next = [...p]; next[j] = value; moveVertex(i, next); state.vertex = i; sync();
        }); label.append(input); row.append(label);
      });
      const remove = document.createElement('button'); remove.className = 'btn'; remove.type = 'button'; remove.setAttribute('aria-label', `Удалить вершину ${i+1}`); remove.disabled = !editable() || fixedShape(state.selection);
      const icon = document.createElementNS('http://www.w3.org/2000/svg','svg');
      icon.setAttribute('viewBox','0 0 24 24'); icon.setAttribute('width','20'); icon.setAttribute('height','20'); icon.setAttribute('aria-hidden','true');
      const lines = document.createElementNS('http://www.w3.org/2000/svg','path');
      lines.setAttribute('d','M6 6L18 18M18 6L6 18'); lines.setAttribute('stroke','currentColor'); lines.setAttribute('stroke-width','2'); lines.setAttribute('stroke-linecap','round');
      icon.append(lines); remove.append(icon);
      remove.addEventListener('click', () => { state.vertex = i; deleteVertex(); }); row.append(remove); list.append(row);
    });
  }
  function renderAnalysis() {
    const list = $('scene-analysis-results'); list.replaceChildren();
    for (const item of state.analysisItems) {
      const shape = item.id === 'floor' ? (state.geometry.floor && {label: 'Пол'}) :
        [...state.geometry.objects, ...state.geometry.doors].find(shape => shape.id === item.id);
      if (!shape) continue;
      const row = document.createElement('div'); row.className = 'scene-analysis-row';
      const select = document.createElement('button'); select.type = 'button'; select.className = 'scene-analysis-select';
      const kind = {FLOOR: 'пол', OBJECT: 'предмет', DOOR: 'дверь', PASSAGE: 'проход'}[item.kind];
      select.textContent = `${shape.label} · ${kind}${Number.isFinite(item.score) ? ` · ${(item.score * 100).toFixed(0)}%` : ''}${item.outline === 'BOX' ? ' · рамка' : ''}`;
      select.setAttribute('aria-pressed', String(state.selection === item.id));
      select.addEventListener('click', () => { state.selection = item.id; state.vertex = -1; state.tool = 'select'; state.drawing = false; sync(); });
      const remove = document.createElement('button'); remove.type = 'button'; remove.className = 'btn'; remove.textContent = 'Удалить'; remove.disabled = !editable();
      remove.setAttribute('aria-label', `Удалить: ${shape.label}`);
      remove.addEventListener('click', () => { state.selection = item.id; clearSelected(); });
      row.append(select, remove); list.append(row);
    }
  }
  function sync() {
    const options = state.mode === 'gate' ? [['', 'Нет выбора'], ['gate', 'Проход учёта']] :
      [['', 'Нет выбора'], ['floor', 'Пол'], ...state.geometry.objects.map(o => [o.id, o.label || 'Без названия']),
        ...state.geometry.doors.map(d => [d.id, d.label]), ['metric', 'Масштаб']];
    $('scene-selection').replaceChildren(...options.map(([value, label]) => new Option(label,value)));
    $('scene-selection').value = state.selection;
    const object = selectedDoor() || state.geometry.objects.find(o => o.id === state.selection);
    if (object) $('scene-object-label').value = object.label;
    $('scene-width').value = state.geometry.metric?.width_m ?? ''; $('scene-depth').value = state.geometry.metric?.depth_m ?? '';
    syncGateFields();
    $('calibration-validation').textContent = state.gate.threshold.some(point) ? gateError() : 'Отметьте два конца порога или возьмите проход из сцены.';
    root.querySelectorAll('[data-scene-tool]').forEach(button => {
      button.setAttribute('aria-pressed', String(button.dataset.sceneTool === state.tool));
      button.hidden = Boolean(button.dataset.mappingMode && button.dataset.mappingMode !== state.mode);
    });
    $('scene-ai-panel').hidden = state.tool !== 'ai'; $('scene-tool-hint').textContent = hints[state.tool];
    state.metric = null;
    try { state.metric = compileMetric(); $('scene-metric-status').textContent = state.metric ? 'Проекция черновика построена. Окончательная численная проверка выполняется при сохранении.' : 'Масштаб не задан; положение и ширина в метрах не определяются.'; }
    catch (e) { $('scene-metric-status').textContent = e.message; }
    $('scene-draft-status').textContent = `${sceneDirty() ? 'Сцена: есть несохранённые изменения.' : 'Сцена: без изменений.'} ${gateDirty() ? 'Проход: есть несохранённые изменения.' : 'Проход: без изменений.'} Эти две разметки сохраняются отдельно.`;
    buttons(); renderCoordinates(); renderAnalysis(); draw(); schedulePreview(); gateWidth();
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
    $('scene-ai-submit').disabled = !saveable() || Boolean(state.job) || !state.saved?.assistance_available;
    $('scene-ai-cancel').disabled = !state.job || state.job.kind === 'analysis';
    $('scene-ai-accept').disabled = !saveable() || !state.candidates.length;
    $('scene-analysis').disabled = !saveable() || !state.saved?.analysis_available || Boolean(state.job);
    $('scene-analysis-cancel').hidden = state.job?.kind !== 'analysis';
    $('scene-analysis-help').textContent = !state.saved?.analysis_available ? 'Модель распознавания не установлена — пересоберите модели' :
      !state.validSnapshot ? 'Зафиксируйте свежий кадр для распознавания.' : state.review ? 'Сначала подтвердите сверку разметки с новым кадром.' : 'Результат — черновик. Проверьте его и сохраните сцену.';
    root.querySelectorAll('[data-scene-tool]').forEach(button => { button.disabled = state.busy; });
    for (const id of ['scene-finish','scene-new-object','scene-clear-selected','scene-add-vertex','scene-remove-metric','scene-ai-clear','scene-width','scene-depth','scene-object-label','scene-door-kind','calibration-role','calibration-enabled','calibration-kind','calibration-scene-door','calibration-flip']) $(id).disabled = !editable();
    $('scene-add-vertex').disabled = !editable() || !selectedPoints()?.length || fixedShape(state.selection) || state.selection === 'metric';
    $('scene-clear-selected').disabled = !editable() || !state.selection;
    $('calibration-flip').disabled = !editable() || !state.gate.threshold.every(point);
  }
  function showRuntime() {
    const r = state.runtime; if (!r) return;
    const parts = [{ENTRY: 'Только вход', EXIT: 'Только выход', BOTH: 'Вход и выход'}[r.role] || 'Роль не назначена', r.enabled ? 'учёт включён' : 'учёт отключён', `поток: ${state.cameraStatus}`];
    if (r.camera_enabled === false) parts.push('камера отключена');
    if (r.ai_enabled === false) parts.push('ИИ выключен'); else if (r.ai_included === false) parts.push('камера исключена из ИИ'); else if (r.ai_available === false) parts.push('ИИ недоступен');
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
    if ((error.status === 409 && code !== 'SCENE_JOB_KIND_MISMATCH') || error.status === 410) markStale(code || 'Кадр истёк, источник или конфигурация изменились.', error.status !== 410);
    const known = {
      SCENE_MODEL_UNAVAILABLE: 'Модель ИИ-подсказок недоступна. Ручная разметка работает.',
      SCENE_DETECTOR_UNAVAILABLE: 'Модель распознавания не установлена — пересоберите модели. Ручная разметка работает.',
      SCENE_INFERENCE_FAILED: 'Ошибка модели ИИ. Разметка не создана; ручная разметка работает.',
      SCENE_BUSY: 'Модель занята. Дождитесь завершения текущего запроса и повторите распознавание.',
      SCENE_JOB_KIND_MISMATCH: 'Получен результат другого типа запроса. Повторите распознавание на свежем кадре.',
      SCENE_GEOMETRY_INVALID: 'Геометрия отклонена сервером. Проверьте пересечения и масштаб; можно убрать масштаб и сохранить контуры.',
    };
    message(known[code] || (typeof error.detail === 'string' ? error.detail : error.detail?.message)
      || (error.status === 409 || error.status === 410 ? 'Сохранение не подтверждено. Черновик сохранён; требуется новый кадр и сверка.' : error.message || 'Нет связи. Черновик сохранён.'), 'err', id);
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
    state.busy = true; cancelJob(); buttons();
    const epoch = state.sourceEpoch;
    try {
      const saved = await request(base,'PUT',{snapshot_id:state.snapshot.snapshot_id,expected_revision:state.saved.revision,expected_source_revision:state.snapshot.source_revision,geometry:clone(state.geometry)});
      state.saved = {...state.saved, ...saved}; state.geometry = clone(saved.geometry); state.sceneCurrent = saved.status === 'CURRENT' && epoch === state.sourceEpoch;
      message('Сцена сохранена. Проход не изменён.', 'ok'); showScene();
      window.dispatchEvent(new CustomEvent('scene-saved',{detail:saved}));
    } catch (e) { errorMessage(e); }
    finally { state.busy = false; sync(); }
  }
  async function saveGate(disable = false) {
    if (state.busy || !state.config || (!disable && !saveable())) return;
    if (!disable && gateError()) { message(gateError(), 'err', 'calibration-message'); return; }
    if (!disable && !$('calibration-form').reportValidity()) return;
    const c = state.config, g = disable ? gateFrom(c) : state.gate;
    const payload = {role: g.role, enabled: disable ? false : g.enabled,
      line_start: g.threshold[0], line_end: g.threshold[1], inside_point: g.inside,
      door_kind: g.kind, door_top: clone(g.top),
      reference_width: disable ? c.reference_width : state.snapshot.width,
      reference_height: disable ? c.reference_height : state.snapshot.height,
      expected_revision: c.revision, expected_source_revision: disable ? c.source_revision : state.snapshot.source_revision};
    state.busy = true; cancelJob(); buttons();
    try {
      const saved = await request(attendance, 'PUT', payload); state.config = saved; state.runtime = saved;
      if (!disable) state.gate = gateFrom(saved); else state.gate.enabled = false;
      message(disable ? 'Учёт проходов отключён. Черновик геометрии сохранён.' : `Проход сохранён.${saved.enabled && !state.camera?.profile.time_tracking ? ' «Учёт времени» для камеры выключен — проходы не учитываются.' : ''}`, 'ok', 'calibration-message');
      showRuntime();
      window.dispatchEvent(new CustomEvent('camera-attendance-changed'));
    } catch (e) { errorMessage(e, 'calibration-message'); }
    finally { state.busy = false; sync(); }
  }
  function gateWidth() {
    let text = 'Ширина прохода: не откалибрована.';
    const metric = gateMetric(), [a, b] = state.gate.threshold;
    if (metric) {
      const aa = project(metric.image_to_metres, a), bb = project(metric.image_to_metres, b);
      if (aa && bb) text = `Приблизительная ширина прохода: ${Math.hypot(bb[0]-aa[0],bb[1]-aa[1]).toFixed(2)} м · по сохранённому полу${gateDirty() ? ', проход пока в черновике' : ''}.`;
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
    const p = selectedPoints(); if (!editable() || !p || fixedShape(state.selection) || state.vertex < 0 || state.vertex >= p.length) return;
    checkpoint(); p.splice(state.vertex, 1);
    state.vertex=Math.min(state.vertex,p.length-1); sync();
  }
  function insertVertex(index, value) {
    const p = selectedPoints(); if (!editable() || !p?.length || fixedShape(state.selection) || state.selection === 'metric') return;
    if (p.length>=128) { message('Не больше 128 вершин в контуре.','err'); return; }
    checkpoint(); p.splice(index+1,0,value); state.vertex=index+1; sync();
  }
  function clearGate() {
    state.gate = {...state.gate, threshold: [null, null], top: [null, null], inside: null};
  }
  function clearSelected() {
    if (!editable() || !state.selection) return;
    checkpoint();
    if (state.selection === 'floor') { state.geometry.floor = null; state.geometry.metric = null; }
    else if (state.selection === 'metric') state.geometry.metric = null;
    else if (state.selection === 'gate') clearGate();
    else {
      state.geometry.objects = state.geometry.objects.filter(o => o.id !== state.selection);
      state.geometry.doors = state.geometry.doors.filter(d => d.id !== state.selection);
      state.selection = '';
    }
    state.vertex = -1; sync();
  }
  // A model outline can have a hundred vertices; its handles appear once it is selected.
  function showsHandles(shape) { return shape.id === state.selection || (shape.points || []).length <= 12; }
  function hitVertex(position) {
    const t=transform(); let best=null, distance=14;
    for (const shape of allShapes()) if (shape.visible && showsHandles(shape)) (shape.points||[]).forEach((p,index)=>{
      if (!point(p)) return; const q=display(p,t), d=Math.hypot(q[0]-position[0],q[1]-position[1]);
      if (d<distance) { distance=d; best={id:shape.id,index}; }
    }); return best;
  }
  function edgeHit(position) {
    const t=transform(); let best=null, distance=12;
    for (const shape of allShapes()) if (shape.visible && !fixedShape(shape.id) && shape.id !== 'metric' && shape.points?.length >= 2) {
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
    if (state.tool === 'select') {
      const match = allShapes().filter(s => s.visible && s.points?.every(point) && contains(p, s.points)).pop();
      state.selection = match?.id || ''; state.vertex = -1; sync(); return;
    }
    if (state.tool === 'ai') {
      const kind = $('scene-ai-prompt').value; if (kind === 'box') return;
      if (state.prompts.positive_points.length + state.prompts.negative_points.length >= 32) { message('Не больше 32 точек подсказки.', 'err'); return; }
      cancelJob(); state.prompts[`${kind}_points`].push(p); draw(); return;
    }
    if (!state.drawing) return;
    checkpoint();
    if (state.tool === 'door' || state.tool === 'gate') {
      const gate = state.tool === 'gate';
      let door = selectedDoor();
      if (!gate && !door) {
        if (state.geometry.doors.length >= 8) { state.undo.pop(); message('Допускается не больше 8 дверей и проходов.', 'err'); return; }
        const kind = $('scene-door-kind').value;
        door = {id: window.OperatorUI.newRequestId(), label: kind === 'DOOR' ? 'Дверь' : 'Проход', kind, quad: [null, null, null, null]};
        state.geometry.doors.push(door); state.selection = door.id;
      }
      if (gate) state.selection = 'gate';
      const threshold = gate ? state.gate.threshold : door.quad.slice(0, 2);
      const index = threshold.findIndex(p => !point(p));
      if (index < 0) { state.undo.pop(); message('Порог уже задан. Перетащите углы или начните новый проход.'); return; }
      threshold[index] = p;
      if (!gate) door.quad[index] = p;
      state.vertex = index;
      if (threshold.every(point)) {
        const kind = gate ? state.gate.kind : door.kind;
        const top = window.Door3D.extrude(threshold, kind, state.image.naturalWidth, state.image.naturalHeight);
        const quad = [threshold[0], threshold[1], top?.[1], top?.[0]];
        const error = doorError(quad, gate ? .05 : .02);
        if (error) {
          if (gate) state.gate.threshold[index] = null; else door.quad[index] = null;
          message(top ? error : 'Порог у верхнего края: сдвиньте его ниже, чтобы поместились стойки.', 'err');
        } else {
          if (gate) {
            state.gate.top = top; state.gate.side = window.Door3D.defaultSide(threshold, top);
            state.gate.inside = window.Door3D.insidePoint(threshold, top, state.gate.side);
          } else door.quad = quad;
          state.tool = 'select'; state.drawing = false;
        }
      }
    } else {
      if (state.tool === 'floor') { state.selection = 'floor'; if (!state.geometry.floor) state.geometry.floor = []; }
      if (state.tool === 'object' && !state.geometry.objects.some(o => o.id === state.selection) && !newObject()) { state.undo.pop(); return; }
      if (state.tool === 'metric') { state.selection = 'metric'; if (!state.geometry.metric) state.geometry.metric = {corners: [], width_m: null, depth_m: null}; }
      const points = selectedPoints(), limit = state.tool === 'metric' ? 4 : 128;
      if (points.length >= limit) { state.undo.pop(); message(`Допускается не больше ${limit} вершин. Перетащите имеющиеся.`); return; }
      points.push(p); state.vertex = points.length - 1;
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
    if (g.kind==='vertex') { moveVertex(state.vertex, p); state.metric=null; draw(); }
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
      moveVertex(state.vertex, [Math.max(0,Math.min(1,p[0]+dx/state.image.naturalWidth)),Math.max(0,Math.min(1,p[1]+dy/state.image.naturalHeight))]); sync();
    }
  });
  canvas.addEventListener('keyup',event=>{ if (event.code==='Space') state.space=false; });
  canvas.addEventListener('blur',()=>{state.space=false;});
  root.addEventListener('keydown',event=>{
    if (['INPUT','TEXTAREA','SELECT'].includes(event.target.tagName)) return;
    if ((event.ctrlKey||event.metaKey) && event.key.toLowerCase()==='z') { event.preventDefault(); history(!event.shiftKey); }
  });
  function clearCandidates() { state.candidates=[]; $('scene-ai-candidate').replaceChildren(new Option('Нет предложений','')); }
  const jobPath = job => `${base}/${job.kind === 'analysis' ? 'analysis' : 'suggestions'}${job.id ? `/${encodeURIComponent(job.id)}` : ''}`;
  const jobStatus = job => $(job.kind === 'analysis' ? 'scene-analysis-status' : 'scene-ai-status');
  async function cancelJob() {
    const epoch = ++state.jobEpoch; clearTimeout(state.jobTimer);
    const job = state.job || state.completedJob; state.job = null; state.completedJob = null;
    clearCandidates(); buttons(); draw();
    if (!job) return;
    jobStatus(job).textContent = 'Запрос отменён. Поздний результат будет проигнорирован.';
    if (!job.id) return;
    try { await request(jobPath(job), 'DELETE'); }
    catch (e) { if (e.status !== 410 && epoch === state.jobEpoch) jobStatus(job).textContent = 'Отмена на сервере не подтверждена. Поздний результат будет проигнорирован; дождитесь освобождения модели перед повтором.'; }
  }
  async function submitSuggestion() {
    if (!saveable() || state.job || !state.saved?.assistance_available) return;
    const prompts = state.prompts;
    if (!prompts.positive_points.length && !prompts.negative_points.length && !prompts.box) { message('Добавьте точки или рамку для ИИ.', 'err'); return; }
    if (prompts.box && (prompts.box[0] >= prompts.box[2] || prompts.box[1] >= prompts.box[3])) { message('Рамка должна иметь ненулевую ширину и высоту.', 'err'); return; }
    const kind = $('scene-ai-kind').value;
    if (kind === 'object' && !$('scene-object-label').value.trim()) { message('Введите название объекта.', 'err'); return; }
    submitJob(kind, {kind, ...clone(prompts)});
  }
  function submitAnalysis() {
    if (!saveable() || state.job || !state.saved?.analysis_available) return;
    submitJob('analysis', {});
  }
  async function submitJob(kind, payload) {
    cancelJob();
    const job = {id: null, snapshot: state.snapshot.snapshot_id, epoch: ++state.jobEpoch, kind, started: performance.now()};
    state.job = job; buttons(); jobStatus(job).textContent = 'Отправка запроса…';
    try {
      const result = await request(jobPath(job), 'POST', {snapshot_id: job.snapshot, ...payload});
      if (job.epoch !== state.jobEpoch || state.job !== job || state.snapshot?.snapshot_id !== job.snapshot || !saveable()) {
        await request(jobPath({...job, id: result.job_id}), 'DELETE'); return;
      }
      job.id = result.job_id; pollJob(job);
    } catch (e) {
      if (job.epoch === state.jobEpoch) {
        state.job = null; jobStatus(job).textContent = 'Запрос не принят. Ручная разметка доступна.';
        errorMessage(e); buttons();
      }
    }
  }
  function analysisGeometry(draft, metric) {
    return {version: 2, floor: draft.floor ? clone(draft.floor) : null,
      objects: (draft.objects || []).map(({id, label, polygon}) => ({id, label, polygon: clone(polygon)})),
      doors: (draft.doors || []).map(({id, label, kind, quad}) => ({id, label, kind, quad: clone(quad)})),
      metric: metric ? clone(metric) : null};
  }
  function acceptAnalysis(result) {
    const draft = result.draft;
    const warnings = {FLOOR_NOT_FOUND: 'Пол не найден — обведите вручную', NO_DOORS_FOUND: 'Двери не найдены', NO_OBJECTS_FOUND: 'Предметы не найдены'};
    const warningText = (result.warnings || []).map(code => warnings[code] || code).join('. ');
    const hasDraft = draft && (draft.floor || draft.objects?.length || draft.doors?.length);
    if (!hasDraft) { $('scene-analysis-status').textContent = `${warningText || 'Пригодной разметки нет.'} Текущий черновик не изменён.`; return; }
    const existing = state.geometry.floor || state.geometry.objects.length || state.geometry.doors.length;
    if (existing && !confirm('Заменить текущую разметку результатом распознавания?')) {
      $('scene-analysis-status').textContent = `Результат не применён. ${warningText}`; return;
    }
    const geometry = analysisGeometry(draft, state.geometry.metric);
    let removedMetric = false;
    if (geometry.metric && geometryError(geometry)) { geometry.metric = null; removedMetric = true; }
    const error = geometryError(geometry);
    if (error) { message(`Результат распознавания отклонён: ${error}`, 'err'); return; }
    checkpoint();
    state.geometry = geometry;
    state.analysisItems = [
      ...(draft.floor ? [{id: 'floor', kind: 'FLOOR'}] : []),
      ...(draft.objects || []).map(({id, score, outline}) => ({id, kind: 'OBJECT', score, outline})),
      ...(draft.doors || []).map(({id, kind, score}) => ({id, kind, score})),
    ];
    state.selection = ''; state.vertex = -1; state.tool = 'select'; state.drawing = false;
    modeTabs.select('scene');
    $('scene-analysis-status').textContent = `${warningText}${removedMetric ? ' Масштаб удалён: опорные углы не подходят к новому полу.' : ''} Разметка применена только в черновик. Проверьте и нажмите «Сохранить сцену».`;
    message('Распознавание ничего не сохраняет автоматически. Проверьте все контуры.', 'ok'); sync();
  }
  async function pollJob(job) {
    try {
      const result = await window.OperatorUI.api(jobPath(job));
      if (job.epoch !== state.jobEpoch || state.job !== job || state.snapshot?.snapshot_id !== job.snapshot || !state.validSnapshot || state.review) return;
      if (result.snapshot_id !== job.snapshot || result.job_id !== job.id) { markStale('Ответ ИИ относится к другому кадру.'); return; }
      const elapsed = `${((performance.now() - job.started) / 1000).toFixed(1)} с`;
      if (result.state === 'QUEUED' || result.state === 'RUNNING') {
        jobStatus(job).textContent = `${elapsed} · ${result.state === 'QUEUED' ? 'В очереди локальной модели…' : job.kind === 'analysis' ? 'Распознавание сцены…' : 'Модель строит контур…'}`;
        state.jobTimer = setTimeout(() => pollJob(job), 600); return;
      }
      state.job = null; state.completedJob = job;
      if (result.state === 'SUCCEEDED' && job.kind === 'analysis') {
        acceptAnalysis(result);
        $('scene-analysis-status').textContent = `${elapsed} · ${$('scene-analysis-status').textContent}`;
      } else if (result.state === 'SUCCEEDED') {
        state.candidates = (result.polygons || []).slice(0, 3).filter(p => p.needs_review === true && Number.isFinite(p.score) && !polygonError(p.points)).map(p => ({...p, kind: job.kind}));
        $('scene-ai-candidate').replaceChildren(...state.candidates.map((p, i) => new Option(`Кандидат ${i+1} · оценка модели ${p.score.toFixed(3)}`, String(i))));
        jobStatus(job).textContent = `${elapsed} · ${state.candidates.length ? 'Проверьте пунктирный контур и примите в черновик. Ничего не сохранено.' : 'Пригодных простых контуров нет. Уточните подсказки или обведите вручную.'}`;
      } else {
        jobStatus(job).textContent = result.state === 'CANCELLED' ? 'Запрос отменён.' : 'Модель не создала разметку. Ручная разметка доступна.';
        if (result.state !== 'CANCELLED') errorMessage({detail: {code: result.error_code || 'SCENE_INFERENCE_FAILED'}});
      }
      buttons(); draw();
    } catch (e) {
      if (job.epoch === state.jobEpoch) {
        state.job = null; jobStatus(job).textContent = 'Запрос не завершён. Ручная разметка доступна.';
        errorMessage(e); buttons();
      }
    }
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
    state.review=false; state.undo=[]; state.redo=[]; message('Проверенный черновик привязан к новому кадру. Сохраните сцену и проход отдельными действиями.'); sync();
  });
  $('scene-use-saved').addEventListener('click',()=>{
    if (!state.validSnapshot || state.busy) return;
    state.undo=[]; state.redo=[]; state.geometry=clone(state.saved.geometry||emptyGeometry()); if (state.config) state.gate=gateFrom(state.config);
    cancelJob(); state.analysisItems = [];
    // Imported or dimension-mismatched saved coordinates still require explicit review.
    state.review=state.saved.status==='STALE' || Boolean(state.saved.reference_width && (state.saved.reference_width!==state.snapshot.width || state.saved.reference_height!==state.snapshot.height))
      || Boolean(state.config?.reference_width && (state.config.reference_width!==state.snapshot.width || state.config.reference_height!==state.snapshot.height));
    $('scene-reviewed').checked=false; state.selection=''; state.vertex=-1; sync();
  });
  $('scene-undo').addEventListener('click',()=>history(true)); $('scene-redo').addEventListener('click',()=>history(false));
  $('scene-selection').addEventListener('change',()=>{state.selection=$('scene-selection').value;state.vertex=-1;state.tool='select';state.drawing=false;sync();});
  $('scene-object-label').addEventListener('change', () => { const o = selectedDoor() || state.geometry.objects.find(o => o.id === state.selection); if (o && editable()) { checkpoint(); o.label = $('scene-object-label').value.trim(); sync(); } });
  $('scene-new-object').addEventListener('click',()=>{if(!editable())return;checkpoint();const o=newObject();if(!o){state.undo.pop();return;}state.tool='object';state.drawing=true;state.vertex=-1;sync();});
  $('scene-finish').addEventListener('click',()=>{
    const p = selectedPoints(), error = state.selection === 'gate' ? gateError() : selectedDoor() ? doorError(p) : polygonError(p);
    if (error) { message(error, 'err'); return; } state.drawing = false; state.tool = 'select'; sync(); message('Контур завершён в черновике. Проверьте его перед сохранением.', 'ok');
  });
  $('scene-clear-selected').addEventListener('click',clearSelected);
  $('calibration-clear').addEventListener('click', () => { if (!editable()) return; checkpoint(); clearGate(); setTool('gate'); });
  $('scene-add-vertex').addEventListener('click',()=>{
    const p=selectedPoints();if(!p?.length)return;const i=state.vertex>=0?state.vertex:p.length-1,a=p[i],b=p[(i+1)%p.length];if(point(a)&&point(b))insertVertex(i,[(a[0]+b[0])/2,(a[1]+b[1])/2]);
  });
  $('scene-remove-metric').addEventListener('click',()=>{if(!editable())return;checkpoint();state.geometry.metric=null;sync();});
  for(const [id,key] of [['scene-width','width_m'],['scene-depth','depth_m']]) $(id).addEventListener('change',()=>{
    if(!editable())return;checkpoint();if(!state.geometry.metric)state.geometry.metric={corners:[],width_m:null,depth_m:null};
    state.geometry.metric[key]=$(id).value===''?null:Number($(id).value);sync();
  });
  $('calibration-flip').addEventListener('click', () => {
    if (!editable() || !state.gate.threshold.every(point)) return;
    checkpoint(); state.gate.side *= -1;
    state.gate.inside = window.Door3D.insidePoint(state.gate.threshold, state.gate.top, state.gate.side); sync();
  });
  $('calibration-kind').addEventListener('change', () => {
    if (!editable()) return;
    checkpoint(); state.gate.kind = $('calibration-kind').value; sync();
  });
  $('calibration-scene-door').addEventListener('change', () => {
    if (!editable()) return;
    const door = state.geometry.doors.find(d => d.id === $('calibration-scene-door').value);
    if (!door || doorError(door.quad)) return;
    checkpoint();
    state.gate.threshold = clone(door.quad.slice(0, 2)); state.gate.top = clone([door.quad[3], door.quad[2]]);
    state.gate.kind = door.kind; state.gate.side = window.Door3D.defaultSide(state.gate.threshold, state.gate.top);
    state.gate.inside = window.Door3D.insidePoint(state.gate.threshold, state.gate.top, state.gate.side);
    state.selection = 'gate'; state.vertex = -1; sync();
  });
  $('calibration-role').addEventListener('change',()=>{if(!editable())return;checkpoint();state.gate.role=$('calibration-role').value;sync();});
  $('calibration-enabled').addEventListener('change',()=>{if(!editable())return;checkpoint();state.gate.enabled=$('calibration-enabled').checked;sync();});
  $('scene-fit').addEventListener('click',()=>{state.zoom=1;state.pan=[0,0];draw();});
  $('scene-zoom-in').addEventListener('click',()=>zoom(1.25)); $('scene-zoom-out').addEventListener('click',()=>zoom(.8));
  $('scene-ai-submit').addEventListener('click',submitSuggestion); $('scene-ai-cancel').addEventListener('click',cancelJob);
  $('scene-ai-accept').addEventListener('click',acceptSuggestion); $('scene-ai-candidate').addEventListener('change',draw);
  $('scene-ai-clear').addEventListener('click',()=>{cancelJob();state.prompts={positive_points:[],negative_points:[],box:null};draw();});
  $('scene-ai-kind').addEventListener('change',cancelJob);
  $('scene-analysis').addEventListener('click', submitAnalysis);
  $('scene-analysis-cancel').addEventListener('click', cancelJob);
  new ResizeObserver(draw).observe($('calibration-image'));
  window.addEventListener('resize',draw);
  window.addEventListener('camera-state',event=>{
    const c = event.detail; updateCamera(c);
    if ((c.enabled===false && state.sceneCurrent) || (state.validSnapshot && state.snapshot && c.source_revision!=null && c.source_revision!==state.snapshot.source_revision)
        || (state.sceneCurrent && c.source_revision!=null && c.source_revision!==state.saved?.source_revision)) markStale('Источник камеры изменён или отключён.');
  });
  window.addEventListener('camera-detections',event=>liveMetrics(event.detail));
  window.addEventListener('cameras-changed',()=>markStale('Подключение камеры изменено.'));
  window.addEventListener('beforeunload',event=>{if(sceneDirty()||gateDirty()){event.preventDefault();event.returnValue='';}});
  window.addEventListener('pagehide', () => {
    ++state.jobEpoch; clearTimeout(state.jobTimer); clearTimeout(state.ttlTimer); clearTimeout(state.previewTimer);
    const job = state.job || state.completedJob;
    if (job?.id) window.OperatorUI.api(jobPath(job), {method: 'DELETE', keepalive: true}).catch(() => {});
  });
  const modeTabs = window.OperatorUI.tabs($('mapping-mode-tabs'), {hash: false, onChange(mode) {
    cancelJob(); state.mode = mode; state.selection = mode === 'gate' ? 'gate' : '';
    state.vertex = -1; state.tool = 'select'; state.drawing = false; sync();
  }});
  function updateCamera(camera) {
    state.camera = camera; state.cameraStatus = camera.status || 'OFFLINE';
    showRuntime(); buttons();
  }
  async function load() {
    state.busy = true; buttons();
    const results = await Promise.allSettled([window.OperatorUI.api(base), window.OperatorUI.api(attendance), window.OperatorUI.api(`/api/cameras/${cameraId}`)]);
    if (results[0].status === 'fulfilled') {
      state.saved = results[0].value; state.geometry = clone(state.saved.geometry || emptyGeometry());
      state.sceneCurrent = state.saved.status === 'CURRENT'; showScene();
    } else errorMessage(results[0].reason);
    if (results[1].status === 'fulfilled') {
      state.config = results[1].value; state.runtime = state.config; state.gate = gateFrom(state.config); showRuntime();
    } else errorMessage(results[1].reason, 'calibration-message');
    if (results[2].status === 'fulfilled') updateCamera(results[2].value);
    else message('Настройки камеры недоступны. Проход учёта станет доступен после восстановления связи.', 'err', 'calibration-message');
    state.busy = false; sync();
  }
  load();
  // Reuses the prior attendance status cadence, never a second detection poll.
  setInterval(async()=>{try{state.runtime=await window.OperatorUI.api(attendance);showRuntime();}catch(e){$('calibration-state').textContent=`Статус учёта недоступен. ${e.message}`;}},5000);
})();
