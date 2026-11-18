import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';

const script = fs.readFileSync(new URL('../../static/js/human_visits.js', import.meta.url), 'utf8');
const start = Date.parse('2026-11-19T12:00:00Z');
const flush = () => new Promise(setImmediate);

class Element extends EventTarget {
  constructor(tag) { super(); this.tag = tag; this.children = []; this.text = ''; this.hidden = false; }
  set textContent(value) { this.children = []; this.text = value; }
  get textContent() { return this.text + this.children.map(child => child.textContent).join(''); }
  setAttribute(name, value) { this[name] = value; }
  append(...children) { this.children.push(...children); }
  replaceChildren(...children) { this.children = children; this.text = ''; }
  querySelector(tag) { return this.children.find(child => child.tag === tag); }
  removeAttribute(name) { delete this[name]; }
}

function fixture(pages) {
  let now = start, nextTimer = 0, nextUrl = 0;
  const ids = new Map(), observers = [], requests = [], metadata = [], urls = new Set(), timers = new Map();
  const get = id => {
    if (!ids.has(id)) ids.set(id, new Element(id));
    return ids.get(id);
  };
  const body = new Element('tbody');
  get('visits-table').append(body);
  const form = get('visits-filters');
  form.elements = Object.fromEntries(['camera_id', 'employee_id', 'from', 'to', 'unknown_only'].map(id => [id, { value: '', checked: false }]));
  const document = Object.assign(new EventTarget(), {
    getElementById: get, querySelector: () => new Element('tabs'), createElement: tag => new Element(tag),
    createTextNode: text => ({ textContent: text }), hidden: false,
  });
  const window = Object.assign(new EventTarget(), { OperatorUI: {
    setBusy() {}, message(node, text) { node.textContent = text; },
    tabs(_el, options) { options.onChange('visits'); },
    async api(url, options) {
      if (!url.includes('/visits?')) return [];
      metadata.push({ url, signal: options?.signal });
      return pages.shift();
    },
  } });
  class IntersectionObserver {
    constructor(callback) { this.callback = callback; this.targets = new Set(); observers.push(this); }
    observe(target) { this.targets.add(target); }
    unobserve(target) { this.targets.delete(target); }
    disconnect() { this.targets.clear(); }
  }
  vm.runInNewContext(script, {
    window, document, AbortController, URLSearchParams, IntersectionObserver, queueMicrotask, CustomEvent,
    Date: class extends Date { static now() { return now; } },
    URL: {
      createObjectURL() { const url = `blob:test/${++nextUrl}`; urls.add(url); return url; },
      revokeObjectURL(url) { assert.equal(urls.delete(url), true); },
    },
    setTimeout(callback, delay) { const id = ++nextTimer; timers.set(id, [now + delay, callback]); return id; },
    clearTimeout(id) { timers.delete(id); },
    fetch(url, options) { return new Promise(resolve => requests.push({ url, signal: options.signal, respond(status) {
      resolve({ status, ok: status === 200, blob: async () => ({ type: 'image/jpeg' }) });
    } })); },
  });
  const clock = (ms, { timersRun = true } = {}) => {
    now += ms;
    if (timersRun) for (const [id, [at, callback]] of [...timers]) {
      if (at <= now && timers.delete(id)) callback();
    }
  };
  const expose = (amount = 100) => {
    const observer = observers.at(-1);
    observer.callback([...observer.targets].slice(0, amount).map(target => ({ target, isIntersecting: true })));
  };
  const thumb = row => body.children[row].children.at(-1);
  const find = (node, tag) => node.tag === tag ? node : node.children?.map(child => find(child, tag)).find(Boolean);
  return { get, window, document, body, metadata, requests, urls, clock, expose, thumb, find, observers };
}

function item(n, photo = 'AVAILABLE', face = 'MISSING', expires = 5000) {
  return { id: `visit-${n}`, camera_name: 'Office', status: 'ACTIVE', identity_state: 'UNKNOWN',
    duration_seconds: 1, photo_status: photo, face_status: face, image_expires_at: new Date(start + expires).toISOString() };
}

{
  const test = fixture([{ items: [item(1), item(2, 'MISSING', 'AVAILABLE'), item(3), item(4), item(5),
    item(6), item(7, 'PURGED', 'PURGED'), item(8, 'MISSING', 'MISSING')], next_cursor: 'page-two' },
  { items: [item(9)], next_cursor: null }]);
  await flush();
  assert.equal(test.requests.length, 0, 'Rows out of the viewport do not fetch media');
  test.expose();
  assert.equal(test.requests.length, 4, 'A full viewport starts at most four requests');
  assert.ok(test.requests[0].url.endsWith('/visit-1/photo'));
  assert.ok(test.requests[1].url.endsWith('/visit-2/face'));
  test.requests[0].respond(200); test.requests[1].respond(200);
  test.requests[2].respond(404); test.requests[3].respond(500);
  await flush();
  assert.equal(test.requests.length, 6, 'Queued visible rows use slots as requests finish');
  assert.ok(test.find(test.thumb(0), 'img').alt.includes('Снимок человека'));
  assert.ok(test.find(test.thumb(0), 'a').href.endsWith('/visit-1'));
  assert.ok(test.thumb(1).textContent.includes('Лицо'));
  assert.ok(test.find(test.thumb(1), 'img').alt.includes('Лицо'));
  assert.ok(test.thumb(2).textContent.includes('Снимок не сохранён'));
  assert.ok(!test.find(test.thumb(2), 'img'));
  assert.ok(test.thumb(3).textContent.includes('Не удалось загрузить'));
  const retry = test.find(test.thumb(3), 'button');
  assert.equal(retry.type, 'button');
  retry.dispatchEvent(new Event('click'));
  assert.equal(test.requests.length, 7);
  test.requests[6].respond(410);
  await flush();
  assert.ok(test.thumb(3).textContent.includes('Срок хранения истёк'));
  assert.ok(test.thumb(6).textContent.includes('Срок хранения истёк'));
  assert.ok(test.thumb(7).textContent.includes('Снимок не сохранён'));
  test.get('visits-more').dispatchEvent(new Event('click'));
  await flush();
  assert.ok(test.metadata[1].url.includes('cursor=page-two'));
  assert.equal(test.body.children.length, 9);
  assert.equal(test.urls.size, 2, 'Pagination retains live thumbnails');
  test.clock(5000);
  assert.equal(test.urls.size, 0, 'Timer revokes already-displayed thumbnails');
  assert.ok(!test.find(test.thumb(0), 'img'));
  test.requests[4].respond(200); test.requests[5].respond(200);
  await flush();
  assert.equal(test.urls.size, 0, 'Pending JPEGs cannot beat the expiry fence');
  assert.equal(test.requests.length, 7, 'Expiry never fetches to renew an image');
}

{
  const test = fixture([{ items: [item(10), item(11)], next_cursor: null },
    { items: [item(12, 'AVAILABLE', 'MISSING', 15000)], next_cursor: null }]);
  await flush();
  test.expose();
  test.requests[0].respond(200);
  await flush();
  assert.equal(test.urls.size, 1);
  test.get('visits-filters').elements.unknown_only.checked = true;
  test.get('visits-filters').dispatchEvent(new Event('submit', { cancelable: true }));
  await flush();
  assert.equal(test.observers[0].targets.size, 0);
  assert.equal(test.requests[1].signal.aborted, true);
  assert.equal(test.urls.size, 0);
  test.requests[1].respond(200);
  test.expose();
  test.requests[2].respond(200);
  await flush();
  assert.equal(test.urls.size, 1, 'Only filtered generation can publish');
  assert.ok(test.metadata[1].url.includes('unknown_only=true'));
  test.document.hidden = true;
  test.clock(16000, { timersRun: false });
  test.document.hidden = false;
  test.document.dispatchEvent(new Event('visibilitychange'));
  assert.equal(test.urls.size, 0);
  assert.ok(test.thumb(0).textContent.includes('Срок хранения истёк'));
  assert.equal(test.requests.length, 3);
  test.window.dispatchEvent(new Event('pagehide'));
  assert.ok(test.observers.every(observer => !observer.targets.size));
}
console.log('Journal thumbnail visibility, error handling, deadlines and disposal passed.');
