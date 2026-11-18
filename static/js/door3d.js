// Normalized door geometry; display maps an image point to canvas CSS pixels.
(() => {
  'use strict';
  const EPSILON = 1e-12;
  const cross = (a, b, p) => (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]);
  const midpoint = (a, b) => [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];
  const validPoint = (point) => Array.isArray(point) && point.length === 2 && point.every(Number.isFinite);

  function extrude(threshold, kind, imageW, imageH) {
    if (threshold.some((point) => point[1] === 0)) return null;
    const [a, b] = threshold;
    const length = Math.hypot((b[0] - a[0]) * imageW, (b[1] - a[1]) * imageH);
    const height = (kind === 'DOOR' ? 1.8 : 1.0) * length;
    return threshold.map(([x, y]) => [x, Math.max(0, y * imageH - height) / imageH]);
  }

  function insidePoint(threshold, top, side) {
    const [a, b] = threshold;
    const dx = b[0] - a[0], dy = b[1] - a[1], length = Math.hypot(dx, dy);
    if (length < EPSILON) return null;
    const offset = [-dy / length * 0.08 * side, dx / length * 0.08 * side];
    for (const t of [0.5, 0.25, 0.75]) {
      const point = [a[0] + t * dx + offset[0], a[1] + t * dy + offset[1]];
      if (point.every((value) => value >= 0 && value <= 1)) return point;
    }
    return null;
  }

  function defaultSide(threshold, top) {
    const [a, b] = threshold;
    const direction = top?.every(validPoint) ? cross(a, b, midpoint(...top)) : 0;
    if (Math.abs(direction) > EPSILON) return direction > 0 ? -1 : 1;
    return cross(a, b, [0.5, 0.5]) < 0 ? -1 : 1;
  }

  function project(matrix, point) {
    const z = matrix[2][0] * point[0] + matrix[2][1] * point[1] + matrix[2][2];
    if (!Number.isFinite(z) || Math.abs(z) < EPSILON) return null;
    const result = [(matrix[0][0] * point[0] + matrix[0][1] * point[1] + matrix[0][2]) / z,
      (matrix[1][0] * point[0] + matrix[1][1] * point[1] + matrix[1][2]) / z];
    return result.every(Number.isFinite) ? result : null;
  }

  // The caller supplies metric only for a CURRENT scene with the threshold in its floor.
  function metricBands(threshold, inside, metric, display) {
    if (!metric?.image_to_metres || !metric.metres_to_image) return null;
    const [a, b, inner] = [...threshold, inside].map((point) => project(metric.image_to_metres, point));
    if (![a, b, inner].every(validPoint)) return null;
    const dx = b[0] - a[0], dy = b[1] - a[1], length = Math.hypot(dx, dy);
    if (length < EPSILON) return null;
    const side = cross(a, b, inner) < 0 ? -1 : 1;
    const nx = -dy / length * side, ny = dx / length * side;
    const bands = [1, -1].map((sign) => [a, b, [b[0] + sign * nx, b[1] + sign * ny],
      [a[0] + sign * nx, a[1] + sign * ny]].map((point) => project(metric.metres_to_image, point)));
    if (!bands.flat().every(validPoint)) return null;
    return bands.map((band) => band.map(display));
  }

  function imageBands(a, b, inside) {
    const dx = b[0] - a[0], dy = b[1] - a[1];
    const side = cross(a, b, inside) < 0 ? -1 : 1;
    // Rotating the pixel threshold gives its unit normal times 0.6 * its length.
    const offset = [-dy * 0.6 * side, dx * 0.6 * side];
    return [1, -1].map((sign) => [a, b, [b[0] + sign * offset[0], b[1] + sign * offset[1]],
      [a[0] + sign * offset[0], a[1] + sign * offset[1]]]);
  }

  function path(ctx, points, closed = false) {
    ctx.beginPath(); ctx.moveTo(...points[0]);
    for (let index = 1; index < points.length; index += 1) ctx.lineTo(...points[index]);
    if (closed) ctx.closePath();
  }

  function caption(ctx, text, point, color = '#FFFBFE') {
    ctx.font = '500 12px "Segoe UI", sans-serif';
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.lineWidth = 4; ctx.strokeStyle = '#1C1B1F'; ctx.fillStyle = color;
    ctx.lineJoin = 'round'; ctx.strokeText(text, ...point); ctx.fillText(text, ...point);
  }

  const centre = (points) => points.reduce((sum, point) => [sum[0] + point[0] / points.length,
    sum[1] + point[1] / points.length], [0, 0]);

  function arrow(ctx, from, to, text, color) {
    const dx = to[0] - from[0], dy = to[1] - from[1], length = Math.hypot(dx, dy);
    if (length < 1) return;
    const ux = dx / length, uy = dy / length, size = Math.min(10, length * 0.3);
    const left = [to[0] - ux * size - uy * size * 0.55, to[1] - uy * size + ux * size * 0.55];
    const right = [to[0] - ux * size + uy * size * 0.55, to[1] - uy * size - ux * size * 0.55];
    for (const [width, stroke] of [[6, '#1C1B1F'], [3, color]]) {
      ctx.lineWidth = width; ctx.strokeStyle = stroke;
      path(ctx, [from, to]); ctx.stroke(); path(ctx, [left, to, right]); ctx.stroke();
    }
    caption(ctx, text, [to[0] + ux * 16, to[1] + uy * 16], color);
  }

  function draw(ctx, { quad, inside = null, kind = 'DOOR', role = null, display, metric = null, selected = false }) {
    if (!Array.isArray(quad) || quad.length !== 4 || !quad.every(validPoint)) return;
    const points = quad.map(display);
    if (!points.every(validPoint)) return;
    const [a, b, topEnd, topStart] = points;
    const length = Math.hypot(b[0] - a[0], b[1] - a[1]);
    const middle = midpoint(a, b);
    const green = '#4ADE80', amber = '#FACC15', frame = selected ? amber : '#EADDFF';
    ctx.save(); ctx.lineJoin = 'round'; ctx.lineCap = 'round'; ctx.setLineDash([]);

    if (validPoint(inside) && length > EPSILON) {
      const bands = metricBands([quad[0], quad[1]], inside, metric, display) || imageBands(a, b, display(inside));
      bands.forEach((band, index) => {
        path(ctx, band, true); ctx.fillStyle = index === 0 ? 'rgba(74, 222, 128, 0.20)' : 'rgba(250, 204, 21, 0.20)'; ctx.fill();
        ctx.lineWidth = 1; ctx.strokeStyle = index === 0 ? green : amber; ctx.stroke();
      });
      const inner = centre(bands[0]), outer = centre(bands[1]);
      // Keep zone labels toward the outer band edges, clear of the walking arrows.
      caption(ctx, 'Внутри', [inner[0] * 1.6 - middle[0] * 0.6, inner[1] * 1.6 - middle[1] * 0.6], green);
      caption(ctx, 'Снаружи', [outer[0] * 1.6 - middle[0] * 0.6, outer[1] * 1.6 - middle[1] * 0.6], amber);
      const tangent = [(b[0] - a[0]) / length, (b[1] - a[1]) / length];
      const offset = role === 'BOTH' ? Math.min(22, length * 0.18) : 0;
      const shifted = (point, sign) => [point[0] + tangent[0] * offset * sign, point[1] + tangent[1] * offset * sign];
      if (role === 'ENTRY' || role === 'BOTH') arrow(ctx, shifted(outer, -1), shifted(inner, -1), 'Вход', green);
      if (role === 'EXIT' || role === 'BOTH') arrow(ctx, shifted(inner, 1), shifted(outer, 1), 'Выход', amber);
    }

    path(ctx, points, true);
    if (kind === 'PASSAGE') {
      const topMiddle = midpoint(topStart, topEnd);
      const curtain = ctx.createLinearGradient(topMiddle[0], topMiddle[1], middle[0], middle[1]);
      curtain.addColorStop(0, 'rgba(234, 221, 255, 0.02)'); curtain.addColorStop(1, 'rgba(234, 221, 255, 0.28)');
      ctx.fillStyle = curtain; ctx.fill();
      ctx.setLineDash([10, 7]);
      ctx.lineWidth = 6; ctx.strokeStyle = '#1C1B1F'; ctx.stroke();
      ctx.lineWidth = 3; ctx.strokeStyle = frame; ctx.stroke(); ctx.setLineDash([]);
    } else {
      ctx.fillStyle = 'rgba(234, 221, 255, 0.16)'; ctx.fill();
      // Posts and lintel retain a darker inner edge against both bright and dark frames.
      path(ctx, [a, topStart, topEnd, b]);
      for (const [width, color] of [[9, '#1C1B1F'], [6, frame], [2, 'rgba(28, 27, 31, 0.65)']]) {
        ctx.lineWidth = width; ctx.strokeStyle = color; ctx.stroke();
      }
    }
    path(ctx, [a, b]); ctx.lineWidth = 9; ctx.strokeStyle = '#1C1B1F'; ctx.stroke();
    ctx.lineWidth = 5; ctx.strokeStyle = selected ? amber : green; ctx.stroke();
    if (validPoint(inside) || selected) caption(ctx, 'Порог — считается пересечение', [middle[0], middle[1] - 15]);
    if (selected) {
      for (const point of points) {
        ctx.beginPath(); ctx.arc(...point, 6, 0, Math.PI * 2);
        ctx.fillStyle = '#FFFBFE'; ctx.fill(); ctx.lineWidth = 2; ctx.strokeStyle = '#1C1B1F'; ctx.stroke();
      }
    }
    ctx.restore();
  }

  window.Door3D = { extrude, insidePoint, defaultSide, draw };
})();
