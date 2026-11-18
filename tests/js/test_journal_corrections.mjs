// Controller state contracts, without a model, network, or production data.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';

class Node {
  constructor(id) { this.id = id; this.value = ''; this.hidden = false; this.disabled = false; this.checked = false; this.children = []; this.dataset = { visitId: 'visit' }; }
  setAttribute() {}
  removeAttribute() {}
  toggleAttribute() {}
  addEventListener() {}
  dispatchEvent() {}
  querySelectorAll() { return []; }
  replaceChildren(...children) { this.children = children; }
  append(child) { this.children.push(child); }
  getAttribute() { return null; }
  focus() {}
  reportValidity() { return Boolean(this.value); }
}
const nodes = new Map();
const get = (id) => { if (!nodes.has(id)) nodes.set(id, new Node(id)); return nodes.get(id); };
get('identity-mode').value = 'UNKNOWN';
get('sample-source').value = 'ALIGNED_FACE';
const attempts = [];
let rejectIdentity = true, sequence = 0, photoError = null, sampleItems = [], historyReads = 0;
const learningRequests = [], learningEvents = [];
let learningError = null, learningResponse = null;
let deferEmployees = false;
const employeeRequests = [];
const context = {
  console, Event: class {}, CustomEvent: class {},
  Option: class { constructor(text, value) { this.textContent = text; this.value = value; } },
  document: { getElementById: get, createElement: (name) => new Node(name) },
  window: { addEventListener() {}, CameraCrop: { nativeCrop: () => null }, JournalLearning: {
    update() {}, sync() {}, invalidate() {},
    onMutationResult(result, frozen) { learningEvents.push({ result, frozen }); },
    onMutationError(error, meta) { learningEvents.push({ error, uncertain: meta.uncertain }); },
  }, OperatorUI: {
    newRequestId: () => `request-${++sequence}`,
    message: (node, text) => { node.textContent = text; },
    api: async (url, options) => {
      if (url === '/api/employees') return [];
      if (url.startsWith('/api/employees/')) {
        if (deferEmployees) return new Promise((resolve) => employeeRequests.push(resolve));
        return { id: 7, name: 'Selected employee', revision: 2, faces_count: 1, active: true };
      }
      if (url.endsWith('/employee-samples')) { historyReads++; return { items: sampleItems }; }
      if (url.endsWith('/employee-sample') || url.endsWith('/employee-sample/remove')) {
        if (photoError) throw Object.assign(new Error('Request failed'), photoError);
        return { id: 7, faces_count: 2 };
      }
      if (url.endsWith('/time/preview')) {
        const selection = JSON.parse(options.body);
        return { selection, preview_sha256: 'f'.repeat(64), rows: [{ employee_name: 'Employee', date: '2026-03-10', method: 'FINAL', before_ms: 0, after_ms: 1000, delta_ms: 1000 }] };
      }
      if (url.endsWith('/identity')) {
        attempts.push(options.body);
        if (rejectIdentity) throw Object.assign(new Error('Unavailable'), { status: 503 });
        return { employee_id: null, employee_name: null, identity_revision: 1, identity_source: 'OPERATOR',
          identity_state: 'OPERATOR_UNKNOWN', revoked_feedback_ids: ['old-crop'], removed_face_ids: [3] };
      }
      if (url.endsWith('/learning/feedback') || url.endsWith('/revoke')) {
        learningRequests.push({ url, body: options.body });
        if (learningError) throw Object.assign(new Error('Learning failed'), learningError);
        return learningResponse;
      }
      return { items: [] };
    },
  } },
};
vm.runInNewContext(fs.readFileSync(new URL('../../static/js/journal_corrections.js', import.meta.url), 'utf8'), context);
const controller = context.window.JournalCorrections;
const visit = { enrollment_available: true, employee_id: 7, employee_name: 'Employee', identity_revision: 0, time_bounds: {
  started_at: '2026-03-10T09:00:00.123000+05:00', ended_at: '2026-03-10T09:00:01.123000+05:00',
} };
controller.update(visit, {});
await new Promise(setImmediate);
assert.equal(get('time-start').value, '2026-03-10T09:00:00.123', 'Native datetime-local defaults must not contain six fractional digits');
assert.equal(get('time-end').value, '2026-03-10T09:00:01.123');
controller.setLoading(true);
assert.equal(get('identity-fields').disabled, true, 'Refresh must not race an identity mutation');
assert.equal(get('identity-submit').disabled, true);
assert.equal(get('sample-fields').disabled, true);
controller.setLoading(false);
assert.equal(get('identity-fields').disabled, false, 'No face/model is needed to correct identity');

get('time-actor').value = 'Operator'; get('time-reason').value = 'Verified interval';
await get('time-form').onsubmit({ preventDefault() {} });
assert.equal(get('time-result').hidden, false);
controller.update(visit, {}, { evidenceOnly: true });
assert.equal(get('time-result').hidden, false, 'Image expiry must not discard an independent time preview');
controller.update(visit, {});
assert.equal(get('time-result').hidden, true, 'Explicit metadata refresh must invalidate the preview');

get('identity-actor').value = 'Operator'; get('identity-reason').value = 'Unknown person';
get('identity-form').onsubmit({ preventDefault() {} });
await new Promise(setImmediate);
assert.equal(get('identity-fields').disabled, true, 'Uncertain mutation freezes its inputs');
assert.equal(get('sample-fields').disabled, false, 'Uncertain identity does not disable photo controls');
assert.equal(controller.locked(), true, 'Refresh must not discard an uncertain mutation');
get('identity-mode').value = 'EMPLOYEE'; get('identity-employee').value = '99';
get('identity-reason').value = 'Different target must not affect retry';
rejectIdentity = false;
await get('identity-form').onsubmit({ preventDefault() {} });
assert.equal(attempts.length, 2);
assert.equal(attempts[0], attempts[1], 'Retry must retain the complete original payload and request ID');
assert.equal(JSON.parse(attempts[1]).mode, 'UNKNOWN');
assert.match(get('identity-status').textContent, /Связанное обучение отменено/);
assert.doesNotMatch(get('identity-status').textContent, /Фотографии и часы не изменены/);
assert.equal(controller.locked(), false);
console.log('Journal millisecond inputs, refresh ownership, independent expiry, and frozen retry checks passed.');

const failures = [];
function check(name, run) { try { run(); } catch (error) { failures.push(`${name}: ${error.message}`); } }
for (const name of ['identity', 'sample']) {
  let prevented = false;
  get(name + '-search').onkeydown?.({ key: 'Enter', preventDefault() { prevented = true; } });
  check('Search Enter must not submit ' + name, () => assert.equal(prevented, true));
}
get('sample-source').value = 'UPLOAD';
get('sample-x1').value = '1.5';
get('sample-source').onchange();
check('Hidden invalid crop bounds cannot block uploads', () => assert.equal(get('sample-x1').disabled, true));

get('sample-employee').value = '7';
await get('sample-employee').onchange();
Object.assign(visit, { employee_sample_eligible: true, face_sha256: 'exact-face', enrollment_available: false });
get('sample-source').value = 'ALIGNED_FACE';
controller.update(visit, { face: 'exact-face' });
get('sample-confirm').checked = true; get('sample-confirm').onchange();
check('Model-down addition is disabled', () => assert.equal(get('sample-submit').disabled, true));
check('Model-down explanation is actionable', () => assert.match(get('sample-eligibility').textContent, /Модель лиц недоступна/));
assert.equal(get('identity-fields').disabled, false);
visit.enrollment_available = true;
controller.update(visit, { face: 'exact-face' });
get('sample-actor').value = 'Operator'; get('sample-reason').value = 'Verified sample';
get('sample-confirm').checked = true; get('sample-confirm').onchange();
photoError = { status: 503, detail: { code: 'FACE_MODEL_UNAVAILABLE' } };
await get('sample-form').onsubmit({ preventDefault() {} });
check('Definite pre-write model failure does not freeze navigation', () => assert.equal(controller.locked(), false));
check('Model failure still permits sample removal', () => assert.equal(get('sample-fields').disabled, false));

// Finish any old-behavior pending request so independent assertions can run.
photoError = null;
if (controller.locked()) { await get('sample-form').onsubmit({ preventDefault() {} }); }
visit.enrollment_available = true;
controller.update(visit, {});
sampleItems = [{ employee_id: 7, employee_name: 'Selected employee', sample_request_id: 'sample',
  source: 'UPLOAD', present: true, removable: true, actor: 'Operator', reason: 'Enrollment' }];
controller.update(visit, {});
await new Promise(setImmediate);
photoError = { status: 409, detail: { code: 'VISIT_SAMPLE_ALREADY_REMOVED' } };
sampleItems[0] = { ...sampleItems[0], present: false, removal: { actor: 'Auditor', reason: 'Wrong face', created_at: '2026-03-10T09:00:00+05:00' } };
const beforeHistory = historyReads;
await get('sample-history').children[0].children[1].onclick();
await new Promise(setImmediate);
check('Stale sample failure refreshes history', () => assert.ok(historyReads > beforeHistory));
check('Removal audit is visible', () => assert.match(get('sample-history').children[0].children[0].textContent, /Auditor.*Wrong face/));
check('Employee selection recovers after conflict', () => assert.match(get('sample-employee-status').textContent, /версия 2/));

// Out-of-order fetches must never retarget a sample to the earlier employee.
deferEmployees = true;
get('sample-employee').value = '7'; const first = get('sample-employee').onchange();
get('sample-employee').value = '8'; const second = get('sample-employee').onchange();
employeeRequests[1]({ id: 8, name: 'New selection', revision: 5, faces_count: 1, active: true });
await second;
employeeRequests[0]({ id: 7, name: 'Stale selection', revision: 1, faces_count: 1, active: true });
await first;
check('Stale employee response is ignored', () => assert.match(get('sample-employee-status').textContent, /New selection/));
assert.deepEqual(failures, []);
console.log('Photo availability, definite failures, stale selection, input isolation, and removal audit checks passed.');

controller.setLearningAnalyzing(true);
assert.equal(controller.learningState().analyzing, true);
assert.equal(controller.locked(), false, 'Read-only analysis must not lock identity, time, or navigation');
controller.setLearningAnalyzing(false);
const learning = { url: '/api/global-persons/visits/visit/learning/feedback',
  options: { body: new Blob(['immutable original crop']) } };
learningError = { status: 503, detail: { code: 'BODY_MODEL_UNAVAILABLE' } };
await controller.mutateLearning(learning);
assert.equal(controller.learningState().pending, false);
assert.equal(controller.locked(), false);
assert.equal(learningEvents.at(-1).uncertain, false);
learningError = { status: 503, detail: { code: 'ENROLLMENT_BUSY' } };
await controller.mutateLearning(learning);
assert.equal(controller.learningState().pending, true);
assert.equal(controller.locked(), true);
learningError = null;
learningResponse = { identity: null, historical_hours_changed: false };
await controller.mutateLearning({ url: learning.url, options: { body: new Blob(['different selected file']) } });
assert.equal(learningRequests.at(-1).body, learning.options.body);
assert.equal(learningEvents.at(-1).frozen.url, learning.url);
assert.equal(controller.locked(), false);
learningError = { status: 409, detail: { code: 'LEARNING_CONFLICT' } };
await controller.mutateLearning(learning);
assert.equal(controller.learningState().pending, false);
assert.equal(learningEvents.at(-1).uncertain, false);
learningError = null;
const originalRevision = visit.identity_revision;
learningResponse = { identity: { visit_id: 'different-visit', identity_revision: 88 } };
const revoke = { url: '/api/recognition-learning/frozen-receipt/revoke', options: { body: '{}' } };
await controller.mutateLearning(revoke);
assert.equal(learningRequests.at(-1).url, revoke.url);
assert.equal(visit.identity_revision, originalRevision, 'Undo may not project another visit onto the open one');
learningResponse = { identity: { visit_id: 'visit', identity_revision: 89 } };
await controller.mutateLearning(revoke);
assert.equal(visit.identity_revision, 89);
console.log('Learning read/isolation locks, definitive errors, frozen upload retry and undo projection passed.');
