import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const source = readFileSync(new URL('../../static/js/ui.js', import.meta.url), 'utf8');
function fixture(names, hash = '', sharedPanel = false) {
  const panels = new Map();
  const windowEvents = new Map();
  const changes = [];
  const location = { hash };
  const history = { state: { retained: true }, writes: 0, replaceState(state, _, value) {
    assert.deepEqual(state, { retained: true });
    location.hash = value; this.writes += 1;
  } };
  const document = { activeElement: null, getElementById: (id) => panels.get(id) || null };
  const tablist = { ownerDocument: document, buttons: [], events: new Map(),
    querySelectorAll: () => tablist.buttons,
    addEventListener: (name, listener) => tablist.events.set(name, listener),
    contains: (button) => tablist.buttons.includes(button) };
  function add(name) {
    const panelId = sharedPanel ? 'shared-panel' : `panel-${name}`;
    if (!panels.has(panelId)) panels.set(panelId, { id: panelId, hidden: true, attributes: {},
      setAttribute(key, value) { this.attributes[key] = value; } });
    const attributes = { 'aria-controls': panelId, 'aria-selected': String(!tablist.buttons.length) };
    const button = { id: `tab-${name}`, dataset: { tab: name }, disabled: false, hidden: false,
      getAttribute: (key) => attributes[key], setAttribute: (key, value) => { attributes[key] = value; },
      closest: () => button, focus: () => { document.activeElement = button; } };
    tablist.buttons.push(button);
    return button;
  }
  names.forEach(add);
  const window = { addEventListener: (name, listener) => windowEvents.set(name, listener) };
  new Function('window', 'document', 'location', 'history', source)(window, document, location, history);
  assert.equal(typeof window.OperatorUI.tabs, 'function', 'Shared tabs helper is exported');
  function click(button) {
    tablist.events.get('click')({ target: { closest: () => button } });
  }
  function key(button, value) {
    let prevented = false;
    tablist.events.get('keydown')({ target: button, key: value, preventDefault() { prevented = true; } });
    return prevented;
  }
  return { tablist, panels, document, location, history, changes, add, click, key,
    hashchange: () => windowEvents.get('hashchange')(),
    initialize: (options = {}) => window.OperatorUI.tabs(tablist, { onChange: (name) => changes.push(name), ...options }) };
}

const f = fixture(['first', 'second', 'third'], '#unknown');
const tabs = f.initialize();
assert.equal(tabs.current(), 'first');
assert.equal(f.location.hash, '#first');
assert.equal(f.panels.get('panel-first').hidden, false);
assert.equal(f.panels.get('panel-second').hidden, true);
assert.equal(f.tablist.buttons[0].tabIndex, 0);
assert.equal(f.tablist.buttons[1].tabIndex, -1);
f.click(f.tablist.buttons[1]);
assert.equal(tabs.current(), 'second');
assert.equal(f.tablist.buttons[1].getAttribute('aria-selected'), 'true');
assert.equal(f.panels.get('panel-first').hidden, true);
assert.equal(f.panels.get('panel-second').hidden, false);
assert.equal(f.location.hash, '#second');
assert.deepEqual(f.changes, ['first', 'second']);
tabs.select('second');
assert.deepEqual(f.changes, ['first', 'second'], 'Reselecting must not refetch tab data');
assert.equal(f.key(f.tablist.buttons[1], 'ArrowRight'), true);
assert.equal(tabs.current(), 'third');
assert.equal(f.document.activeElement, f.tablist.buttons[2]);
f.key(f.tablist.buttons[2], 'ArrowRight');
assert.equal(tabs.current(), 'first');
f.key(f.tablist.buttons[0], 'ArrowLeft');
assert.equal(tabs.current(), 'third');
f.key(f.tablist.buttons[2], 'Home');
assert.equal(tabs.current(), 'first');
f.key(f.tablist.buttons[0], 'End');
assert.equal(tabs.current(), 'third');
assert.equal(f.key(f.tablist.buttons[2], 'Tab'), false);
f.location.hash = '#second'; f.hashchange();
assert.equal(tabs.current(), 'second');
const fourth = f.add('fourth'); tabs.refresh();
assert.equal(tabs.current(), 'second');
assert.equal(fourth.tabIndex, -1);
f.click(fourth);
assert.equal(tabs.current(), 'fourth');
f.tablist.buttons = f.tablist.buttons.filter((button) => button !== fourth);
tabs.refresh();
assert.equal(tabs.current(), 'first', 'Removing the selected dynamic tab selects the first remaining tab');

const shared = fixture(['local', 'dahua'], '#unrelated', true);
const sourceTabs = shared.initialize({ hash: false });
sourceTabs.select('dahua', { focus: true });
assert.equal(sourceTabs.current(), 'dahua');
assert.equal(shared.panels.get('shared-panel').hidden, false, 'Shared recording results remain visible');
assert.equal(shared.panels.get('shared-panel').attributes['aria-labelledby'], 'tab-dahua');
assert.equal(shared.document.activeElement, shared.tablist.buttons[1]);
assert.equal(shared.location.hash, '#unrelated');
assert.equal(shared.history.writes, 0);
console.log('PASS: shared tabs selection, keyboard, hashes, dynamic refresh and shared panels');
