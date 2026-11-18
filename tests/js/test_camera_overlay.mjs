import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
import test from 'node:test';
class Element {
  constructor() { this.children = []; this.style = {}; this.clientWidth = 640; this.clientHeight = 360; this.title = ''; }
  append(...nodes) { this.children.push(...nodes); }
  replaceChildren() { this.children = []; }
}
const context = { window: {}, console, document: { createElement: () => new Element() }, AbortController, performance };
vm.runInNewContext(fs.readFileSync(new URL('../../static/js/camera_overlay.js', import.meta.url), 'utf8'), context);
const overlay = context.window.CameraOverlay;
const known = { track_id: 7, state: 'unknown', employee_id: 1, employee_name: 'Анна', identity_source: 'REID', person_score: .91, confidence: 0, face_similarity: null };
const appearance = overlay.label(known);
assert.match(appearance, /Анна/);
assert.match(appearance, /по внешности/);
assert.match(appearance, /91%/);
assert.doesNotMatch(appearance, /Неизвестный|сходство/);
const face = overlay.label({ ...known, identity_source: 'FACE', face_similarity: .82 });
assert.match(face, /сходство лица 0\.82/);
assert.match(face, /человек 91%/);
const unknown = overlay.label({ ...known, employee_id: null, employee_name: null, identity_source: null, face_similarity: null });
assert.match(unknown, /Неизвестный/);
assert.doesNotMatch(unknown, /Анна|по внешности|сходство/);
assert.doesNotMatch(overlay.label({ ...known, person_score: NaN }), /NaN|%/);
assert.deepEqual(JSON.parse(JSON.stringify(overlay.fitRect(640, 360, 640, 480))), { left: 80, top: 0, width: 480, height: 360 });
assert.deepEqual(JSON.parse(JSON.stringify(overlay.fitRect(640, 360, 640, 480, 'cover'))), { left: 0, top: -60, width: 640, height: 480 });
assert.equal(overlay.fitRect(640, 360, 0, 0), null);
test('Server age keeps slow inference visible regardless of client wall clock', () => {
  assert.equal(overlay.isFresh({ updated_at: 100, age_ms: 4000, max_age_ms: 6000 }, 9999999999999), true);
  assert.equal(overlay.isFresh({ updated_at: 100, age_ms: 6001, max_age_ms: 6000 }, 101000), false);
  assert.equal(overlay.isFresh({ updated_at: null }), false);
});
test('Unchecked people remain detecting in full and compact previews', () => {
  const pending = { ...known, employee_id: null, employee_name: null, identity_source: null, state: 'detecting' };
  assert.match(overlay.label(pending), /Распознавание/);
  assert.doesNotMatch(overlay.label(pending), /Неизвестный/);
  for (const compact of [false, true]) {
    const boxes = new Element();
    overlay.render(boxes, [{ ...pending, bbox: [.1, .1, .4, .8] }], { compact });
    assert.equal(boxes.children[0].className, 'bbox detecting');
    overlay.render(boxes, [{ ...known, bbox: [.1, .1, .4, .8] }], { compact });
    assert.equal(boxes.children[0].className, 'bbox recognized');
  }
});
test('Stalled detection requests abort and release their deadline', async () => {
  let deadline, cleared = false;
  context.setTimeout = (callback, delay) => { deadline = { callback, delay }; return 1; };
  context.clearTimeout = () => { cleared = true; };
  context.fetch = (_url, { signal }) => new Promise((_resolve, reject) => {
    signal.addEventListener('abort', () => reject(new Error('Aborted')), { once: true });
  });
  const pending = overlay.fetchDetections(1);
  assert.equal(deadline.delay, 2500);
  deadline.callback();
  await assert.rejects(pending, /Aborted/);
  assert.equal(cleared, true);
});
