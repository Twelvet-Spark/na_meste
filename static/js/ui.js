// Shared vanilla-JS operator controls. No draft or credential persistence.
(() => {
  'use strict';
  function errorText(detail) {
    if (typeof detail === 'string') return detail;
    if (detail?.code === 'PHOTO_VALIDATION_FAILED') {
      return 'Ничего не сохранено. ' + (detail.files || []).map((file) => `${file.filename}: ${file.error}`).join('; ');
    }
    if (Array.isArray(detail)) return detail.map((error) => `${error.loc?.join('.') || ''}: ${error.msg}`).join('; ');
    return 'Не удалось выполнить запрос. Повторите попытку.';
  }
  async function api(url, options = {}) {
    const headers = new Headers(options.headers);
    if (options.method && !['GET', 'HEAD'].includes(options.method.toUpperCase())) headers.set('X-Operator-Request', '1');
    const response = await fetch(url, { ...options, headers });
    let data = null;
    if (response.status !== 204) {
      try { data = await response.json(); }
      catch { /* A proxy's non-JSON error is still a recoverable request failure. */ }
    }
    if (!response.ok || (response.status !== 204 && data === null)) {
      const error = new Error(errorText(data?.detail));
      error.status = response.status; error.detail = data?.detail;
      throw error;
    }
    return data;
  }
  function newRequestId() {
    // Trusted-LAN HTTP does not always expose crypto.randomUUID().
    const bytes = crypto.getRandomValues(new Uint8Array(16));
    bytes[6] = (bytes[6] & 15) | 64; bytes[8] = (bytes[8] & 63) | 128;
    const hex = Array.from(bytes, (byte) => byte.toString(16).padStart(2, '0')).join('');
    return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20)}`;
  }
  const disabledStates = new WeakMap();
  const busyFocus = new WeakMap();
  function setBusy(root, busy) {
    root.setAttribute('aria-busy', String(busy));
    if (busy && root.contains(document.activeElement) && !busyFocus.has(root)) busyFocus.set(root, document.activeElement);
    for (const control of root.querySelectorAll('button, input, select, textarea, fieldset')) {
      if (busy && !disabledStates.has(control)) {
        disabledStates.set(control, control.disabled); control.disabled = true;
      } else if (!busy && disabledStates.has(control)) {
        control.disabled = disabledStates.get(control); disabledStates.delete(control);
      }
    }
    if (!busy) {
      const focus = busyFocus.get(root); busyFocus.delete(root);
      if (focus?.isConnected && document.activeElement === document.body) focus.focus({ preventScroll: true });
    }
  }
  function message(target, text, tone = '') {
    target.textContent = text;
    target.dataset.tone = tone;
  }
  function dialog(element, trigger, { isBusy = () => false } = {}) {
    let returnTo = trigger;
    const close = () => { if (!isBusy()) element.close(); };
    element.querySelector('[data-dialog-cancel]')?.addEventListener('click', close);
    element.addEventListener('cancel', (event) => { if (isBusy()) event.preventDefault(); });
    element.addEventListener('close', () => {
      if (returnTo?.isConnected) returnTo.focus();
    });
    return {
      open(opener = trigger) {
        returnTo = opener || document.activeElement;
        element.showModal();
        element.querySelector('[autofocus], input, button')?.focus();
      },
      close,
    };
  }
  function conflict(panel, reload, rebase) {
    let busy = false;
    reload.addEventListener('click', async () => {
      if (busy) return;
      busy = true; setBusy(panel, true);
      try { if (await rebase()) panel.hidden = true; }
      finally { busy = false; setBusy(panel, false); }
    });
    return {
      show(error) { if (error?.status === 409) panel.hidden = false; },
      clear() { panel.hidden = true; },
    };
  }
  window.OperatorUI = { api, newRequestId, setBusy, message, dialog, conflict };

  const toggle = document.getElementById('nav-toggle');
  const nav = document.getElementById('app-nav');
  if (!toggle || !nav) return;
  function closeNav(restoreFocus = false) {
    nav.classList.remove('is-open'); toggle.setAttribute('aria-expanded', 'false');
    if (restoreFocus) toggle.focus();
  }
  toggle.addEventListener('click', () => {
    const open = toggle.getAttribute('aria-expanded') !== 'true';
    toggle.setAttribute('aria-expanded', String(open)); nav.classList.toggle('is-open', open);
  });
  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape' && nav.classList.contains('is-open')) closeNav(true);
  });
  document.addEventListener('click', (event) => {
    if (!nav.contains(event.target) && !toggle.contains(event.target)) closeNav();
  });
  nav.addEventListener('click', (event) => { if (event.target.closest('a')) closeNav(); });
  matchMedia('(min-width: 1025px)').addEventListener('change', () => closeNav());
})();
