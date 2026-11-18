import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';

const script = fs.readFileSync(new URL('../../static/js/journal_learning.js', import.meta.url), 'utf8');
const crop = fs.readFileSync(new URL('../../static/js/image_crop.js', import.meta.url), 'utf8');
const flush = () => new Promise(resolve => setImmediate(resolve));
const digest = 'a'.repeat(64), resultDigest = 'b'.repeat(64);

class Element extends EventTarget {
  constructor(tag) {
    super(); this.tag = tag; this.dataset = { visitId: 'visit' }; this.children = [];
    this.value = ''; this.checked = false; this.hidden = false; this.disabled = false;
    this.complete = true; this.naturalWidth = 320; this.naturalHeight = 240; this.files = [];
    this.style = {};
  }
  set textContent(value) { this.text = value; this.children = []; }
  get textContent() { return (this.text || '') + this.children.map(child => child.textContent).join(''); }
  getAttribute(name) { return this[name] ?? null; }
  setAttribute(name, value) { this[name] = value; }
  toggleAttribute(name, value) { this[name] = value; }
  removeAttribute(name) { delete this[name]; }
  replaceChildren(...children) { this.children = children; this.text = ''; }
  append(...children) { this.children.push(...children); }
  prepend(child) { this.children.unshift(child); }
  closest(tag) { return tag === 'details' ? (this.details ??= { open: false }) : null; }
  scrollIntoView() { this.scrolled = true; }
  querySelectorAll(tag) { return this.children.flatMap(node => node.tag === tag ? [node] : node.querySelectorAll?.(tag) || []); }
  getContext() { return { drawImage() {}, clearRect() {} }; }
  focus() { this.focused = true; }
  reportValidity() { return true; }
  fire(type) {
    const event = new Event(type, { cancelable: true });
    this.dispatchEvent(event);
    this[`on${type}`]?.(event);
    return event;
  }
}

function rig() {
  const nodes = new Map(), reads = [], writes = [], urls = new Set();
  const get = id => {
    if (!nodes.has(id)) nodes.set(id, new Element(id));
    return nodes.get(id);
  };
  get('learning-source').value = 'PHOTO_CROP';
  for (const [i, name] of ['x1', 'y1', 'x2', 'y2'].entries()) get(`learning-${name}`).value = i < 2 ? '0' : '1';
  get('visit-photo').src = 'blob:saved';
  const requests = [], route = '/api/global-persons/visits/visit/learning';
  const state = { busy: false, pending: false, loading: false, analyzing: false };
  let nextUrl = 0, requestId = 0;
  const server = { revision: 3, history: [], examples: [], cap: false };
  const doc = Object.assign(new EventTarget(), { getElementById(id) {
    if (id.startsWith('learning-example-') || (id.startsWith('learning-history-') &&
      id !== 'learning-history-more' && id !== 'learning-history-counts')) {
      return ['learning-employee-examples', 'learning-history'].flatMap(list => get(list).children).find(node => node.id === id);
    }
    return get(id);
  }, createElement: tag => new Element(tag), hidden: false,
    createTextNode: text => ({ textContent: text }) });
  const win = Object.assign(new EventTarget(), { OperatorUI: {
    api: async (url, options) => {
      reads.push({ url, options });
      if (url === '/api/employees') return [{ id: 7, name: 'Manual person', active: true }, { id: 8, name: 'Second person', active: true }];
      if (/^\/api\/employees\/[78]$/.test(url)) {
        const id = Number(url.at(-1));
        return { id, name: id === 7 ? 'Manual person' : 'Second person', active: true, revision: server.revision, faces_count: 2 };
      }
      if (url.includes('/learning/history?')) return { items: server.history, counts: {} };
      if (url.includes('feedback_id=off-page')) return { items: [server.offPage] };
      if (url.includes('/recognition-learning/examples?')) return { items: server.examples, counts: { body_positive: server.cap ? 64 : 0 } };
      if (url.endsWith('/analyze')) {
        const request = { url, options };
        requests.push(request);
        return new Promise((resolve, reject) => Object.assign(request, { resolve, reject }));
      }
      return { items: [], counts: {} };
    },
    message(node, text) { node.textContent = text; }, newRequestId: () => `request-${++requestId}`,
  }, JournalCorrections: {
    learningState: () => ({ ...state }),
    setLearningAnalyzing(value) { state.analyzing = value; win.JournalLearning.sync(); },
    async mutateLearning(request) { writes.push(request); },
  } });
  const context = vm.createContext({ document: doc, window: win, console, Blob, FormData, Event, CustomEvent, queueMicrotask,
    AbortController, Date, URLSearchParams, Option: class extends Element {
      constructor(text, value) { super('option'); this.textContent = text; this.value = value; }
    }, URL: {
      createObjectURL() { const url = `blob:test-${++nextUrl}`; urls.add(url); return url; },
      revokeObjectURL(url) { urls.delete(url); },
    }, setTimeout, clearTimeout });
  vm.runInContext(crop, context);
  vm.runInContext(script, context);
  return { get, reads, writes, urls, state, requests, route, server, learning: win.JournalLearning };
}

function scan(face = [], body = []) {
  return { visit_id: 'visit', identity_revision: 0, gallery_revision: 5, learning_revision: 2,
    body_model_id: '1'.repeat(64), face_model_id: null, face_status: 'NO_FACE',
    evidence: { source: 'PHOTO_CROP', source_sha256: digest, image_sha256: resultDigest,
      crop: [0, 0, 1, 1], crop_width: 320, crop_height: 240, captured_at: null,
      daily_anchor: { date: null, today: false }, warnings: [] },
    candidates: { face, body } };
}

const fixture = rig(), { get, learning, requests, writes, state } = fixture;
const visit = { id: 'visit', employee_id: null, identity_revision: 0, learning_available: true,
  gallery_revision: 5, learning_revision: 2, photo_sha256: digest, photo_captured_at: null,
  photo_status: 'AVAILABLE', enrollment_available: false, body_model_id: '1'.repeat(64), face_model_id: null };
learning.update(visit, { photo: digest });
get('learning-image').fire('load');
await flush();
assert.equal(requests.length, 1, 'Unknown visit opens with one read-only scan of its verified photo even without face enrollment');
assert.equal(state.analyzing, true);
requests[0].resolve(scan([{ employee_id: 7, name: 'Face candidate', employee_revision: 3, score: .84321, rejected: false }],
  [{ employee_id: 8, name: 'Body candidate', employee_revision: 1, score: .98123, rejected: true, negative_score: .99 }]));
await flush();
assert.equal(state.analyzing, false);
assert.match(get('learning-candidates').textContent, /Face candidate/);
assert.match(get('learning-candidates').textContent, /Body candidate/);
assert.match(get('learning-candidates').textContent, /0[.,]843/);
assert.doesNotMatch(get('learning-candidates').textContent, /84%/);
assert.equal(writes.length, 0);
learning.update(visit, { photo: digest }, { evidenceOnly: true });
await flush();
assert.equal(requests.length, 1, 'Verified same evidence does not trigger background polling');
get('learning-x2').value = '.75';
get('learning-x2').fire('input');
assert.doesNotMatch(get('learning-candidates').textContent, /Face candidate/);
get('learning-analyze').fire('click');
await flush();
assert.equal(requests.length, 2);
get('learning-source').value = 'UPLOAD';
get('learning-source').fire('change');
requests[1].resolve(scan([], [{ employee_id: 8, name: 'Stale result', employee_revision: 1, score: .8 }]));
await flush();
assert.doesNotMatch(get('learning-candidates').textContent, /Stale result/);
assert.equal(state.analyzing, false);
assert.equal(writes.length, 0);
const enter = new Event('keydown', { cancelable: true });
Object.defineProperty(enter, 'key', { value: 'Enter' });
get('learning-search').dispatchEvent(enter);
get('learning-search').onkeydown?.(enter);
assert.equal(enter.defaultPrevented, true);
assert.equal(writes.length, 0);
learning.update({ ...visit, learning_available: false }, { photo: digest });
assert.equal(get('learning-analyze').disabled, true);
assert.equal(state.pending, false);
assert.equal(state.analyzing, false);
console.log('Learning candidates, read-only startup, evidence invalidation, stale responses and unavailable model passed.');

// Review conflicts require a fresh manual-employee DTO, not picking a different name and back.
const verify = rig(), v = verify.get, result = () => scan();
const shown = { ...visit, identity_revision: 4, time_bounds: { started_at: '2026-01-05T12:00:00+05:00' } };
verify.learning.update(shown, { photo: digest });
v('learning-image').fire('load');
await flush();
const resolveScan = async (value = result()) => {
  const req = verify.requests.at(-1);
  const metadata = JSON.parse(req.options.body.get('metadata'));
  req.resolve({ ...value, identity_revision: 4,
    evidence: { ...value.evidence, source: metadata.source, crop: metadata.crop } });
  await flush();
};
await resolveScan();
v('learning-employee').value = '7';
v('learning-employee').fire('change');
await flush();
v('learning-actor').value = 'Auditor';
v('learning-reason').value = 'Selected pixels';
v('learning-person-confirm').checked = true;
v('learning-person-confirm').fire('change');
v('learning-manual-accept').fire('click');
const readPayload = mutation => JSON.parse(mutation.options.body.get('metadata'));
const first = verify.writes[0], payload = readPayload(first);
assert.equal(first.url, verify.route + '/feedback');
assert.equal(payload.employee_id, 7);
assert.equal(payload.expected_employee_revision, 3);
assert.equal(payload.expected_identity_revision, 4);
assert.equal(payload.expected_learning_revision, 2);
assert.equal(payload.expected_gallery_revision, 5);
assert.equal(payload.expected_body_model_id, '1'.repeat(64));
assert.equal(payload.expected_face_model_id, null);
assert.equal(payload.expected_image_sha256, resultDigest);
assert.equal(payload.photo_sha256, digest);
assert.equal(payload.person_confirmed, true);
verify.server.revision = 4;
verify.learning.onMutationError({ status: 409, detail: { code: 'EMPLOYEE_REVISION_CONFLICT', employee_revision: 4 } });
await flush();
assert.match(v('learning-status').textContent, /Предложить сотрудников/);
assert.equal(v('learning-manual-accept').disabled, true);
v('learning-analyze').fire('click');
await resolveScan();
v('learning-person-confirm').checked = true;
v('learning-manual-accept').fire('click');
assert.equal(readPayload(verify.writes.at(-1)).expected_employee_revision, 4);
assert.equal(v('learning-employee').value, '7');

const saved = (id, source, visitId, identityRevision) => ({
  feedback_id: id, visit_id: visitId, employee_name: 'Second person', employee_active: true,
  active: true, action: 'ACCEPT', source, actor: 'Auditor', reason: 'Evidence',
  identity_revision: identityRevision, examples: [],
});
verify.server.cap = true;
verify.server.examples = [saved('from-another-visit', 'PHOTO_CROP', 'other', 1)];
verify.server.history = [saved('sets-identity', 'PHOTO_CROP', 'visit', 4), saved('independent', 'UPLOAD', 'visit', null)];
verify.learning.onMutationError({ status: 409, detail: { code: 'LEARNING_LIMIT_REACHED', employee_id: 8, limit: 64 } });
await flush();
assert.equal(v('learning-employee').value, '8');
assert.match(v('learning-counts').textContent, /64/);
assert.match(v('learning-links').textContent, /Управлять образцами/);
verify.learning.update(shown, { photo: digest });
await flush();
const undo = (list, index) => {
  v(list).children[index].querySelectorAll('button').at(-1).fire('click');
  const req = verify.writes.at(-1);
  return [req.url.split('/').at(-2), JSON.parse(req.options.body).expected_identity_revision];
};
assert.deepEqual(undo('learning-history', 0), ['sets-identity', 4]);
assert.deepEqual(undo('learning-history', 1), ['independent', null]);
assert.deepEqual(undo('learning-employee-examples', 0), ['from-another-visit', null]);
verify.server.offPage = saved('off-page', 'UPLOAD', 'other', null);
verify.learning.onMutationError({ status: 409, detail: { code: 'LEARNING_CONFLICT',
  feedback_ids: ['off-page'], feedback_owners: { 'off-page': 8 } } });
v('learning-links').querySelectorAll('a')[0].fire('click');
await flush();
const linked = v('learning-employee-examples').children[0];
assert.equal(linked.id, 'learning-example-off-page');
assert.equal(linked.closest('details').open, true);
assert.equal(linked.focused, true);
assert.deepEqual(undo('learning-employee-examples', 0), ['off-page', null]);

const file = (contents, name) => Object.assign(new Blob([contents], { type: 'image/png' }), { name });
v('learning-source').value = 'UPLOAD';
v('learning-source').fire('change');
v('learning-captured-at').value = '2026-01-05T12:00:00';
v('learning-captured-at').fire('input');
v('learning-file').files = [file('Original capture', 'original.png')];
v('learning-file').fire('change');
await flush();
v('learning-image').fire('load');
v('learning-analyze').fire('click');
await resolveScan();
v('learning-person-confirm').checked = true;
v('learning-manual-accept').fire('click');
const upload = verify.writes.at(-1);
const uploadPayload = readPayload(upload);
assert.equal(uploadPayload.source, 'UPLOAD');
assert.equal(uploadPayload.captured_at, '2026-01-05T12:00:00+05:00');
assert.equal('photo_sha256' in uploadPayload, false);
v('learning-file').files = [file('Different capture', 'different.png')];
v('learning-file').fire('change');
await flush();
assert.equal(await upload.options.body.get('photo').text(), 'Original capture');
assert.equal(upload.options.body.get('photo').name, 'original.png');
v('learning-source').value = 'PHOTO_CROP';
v('learning-source').fire('change');
assert.equal(v('learning-captured-at').disabled, true);
assert.equal(v('learning-captured-at').value, '');
verify.learning.onMutationError({ status: 503, detail: { code: 'BODY_MODEL_UNAVAILABLE' } });
assert.equal(v('learning-analyze').disabled, true);
assert.equal(v('learning-employee-examples').children[0].querySelectorAll('button').at(-1).disabled, false);
console.log('Exact decision payload, manual conflict recovery, cap management, upload bytes and scoped undo passed.');
