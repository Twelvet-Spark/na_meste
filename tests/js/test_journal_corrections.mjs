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
let rejectIdentity = true, sequence = 0;
const context = {
  console, Event: class {}, CustomEvent: class {},
  Option: class { constructor(text, value) { this.textContent = text; this.value = value; } },
  document: { getElementById: get, createElement: (name) => new Node(name) },
  window: { addEventListener() {}, CameraCrop: { nativeCrop: () => null }, OperatorUI: {
    newRequestId: () => `request-${++sequence}`,
    message: (node, text) => { node.textContent = text; },
    api: async (url, options) => {
      if (url === '/api/employees') return [];
      if (url.endsWith('/time/preview')) {
        const selection = JSON.parse(options.body);
        return { selection, preview_sha256: 'f'.repeat(64), rows: [{ employee_name: 'Employee', date: '2026-03-10', method: 'FINAL', before_ms: 0, after_ms: 1000, delta_ms: 1000 }] };
      }
      if (url.endsWith('/identity')) {
        attempts.push(options.body);
        if (rejectIdentity) throw Object.assign(new Error('Unavailable'), { status: 503 });
        return { employee_id: null, employee_name: null, identity_revision: 1, identity_source: 'OPERATOR', identity_state: 'OPERATOR_UNKNOWN' };
      }
      return { items: [] };
    },
  } },
};
vm.runInNewContext(fs.readFileSync(new URL('../../static/js/journal_corrections.js', import.meta.url), 'utf8'), context);
const controller = context.window.JournalCorrections;
const visit = { employee_id: 7, employee_name: 'Employee', identity_revision: 0, time_bounds: {
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
assert.equal(controller.locked(), false);
console.log('Journal millisecond inputs, refresh ownership, independent expiry, and frozen retry checks passed.');
