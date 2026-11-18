// Shared normalized coordinates; crops always use original image pixels.
(() => {
  'use strict';
  const clamp = (n) => Math.max(0, Math.min(1, n));
  function normalizeRect(x1, y1, x2, y2) {
    if (![x1, y1, x2, y2].every(Number.isFinite)) return null;
    const rect = [clamp(Math.min(x1, x2)), clamp(Math.min(y1, y2)), clamp(Math.max(x1, x2)), clamp(Math.max(y1, y2))];
    return rect[2] > rect[0] && rect[3] > rect[1] ? rect : null;
  }
  function nativeCrop(rect, width, height) {
    if (!rect || ![width, height].every((n) => Number.isInteger(n) && n > 0)) return null;
    const normalized = normalizeRect(...rect);
    if (!normalized) return null;
    const x = Math.floor(normalized[0] * width), y = Math.floor(normalized[1] * height);
    return { x, y, width: Math.ceil(normalized[2] * width) - x, height: Math.ceil(normalized[3] * height) - y };
  }
  function pointInImage(x, y, rect) {
    return [clamp((x - rect.left) / rect.width), clamp((y - rect.top) / rect.height)];
  }
  window.CameraCrop = Object.freeze({ normalizeRect, nativeCrop, pointInImage });
})();
