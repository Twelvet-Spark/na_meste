// Shared in-memory photo drafts and optional workplace picker.
(() => {
  'use strict';
  const UI = window.OperatorUI;
  class PhotoDraft {
    constructor({ input, grid, count, drop, status, onChange = () => {} }) {
      Object.assign(this, { input, grid, count, drop, status, onChange });
      this.accepted = []; this.kept = new Set(); this.staged = []; this.busy = false;
      this.replaceInput = document.createElement('input');
      this.replaceInput.type = 'file'; this.replaceInput.accept = 'image/jpeg,image/png';
      this.replaceInput.hidden = true; drop.after(this.replaceInput);
      input.addEventListener('change', () => { this.add(Array.from(input.files)); input.value = ''; });
      this.replaceInput.addEventListener('change', () => {
        const file = this.replaceInput.files[0];
        if (file && this.replacing && !this.busy) this.replace(this.replacing, file);
        this.replaceInput.value = ''; this.replacing = null;
      });
      drop.addEventListener('dragover', (event) => {
        event.preventDefault();
        if (!this.busy) { event.dataTransfer.dropEffect = 'copy'; drop.classList.add('is-dragging'); }
      });
      drop.addEventListener('dragleave', (event) => { if (!drop.contains(event.relatedTarget)) drop.classList.remove('is-dragging'); });
      drop.addEventListener('drop', (event) => {
        event.preventDefault(); drop.classList.remove('is-dragging');
        if (!this.busy) this.add(Array.from(event.dataTransfer.files));
      });
      window.addEventListener('pagehide', (event) => { if (!event.persisted) this.release(); });
      this.render();
    }
    get total() { return this.kept.size + this.staged.length; }
    get dirty() { return this.staged.length > 0 || this.kept.size !== this.accepted.length; }
    get files() { return this.staged.map((entry) => entry.file); }
    setAccepted(faces, preserve = false) {
      const omitted = new Set(this.accepted.filter((face) => !this.kept.has(face.id)).map((face) => face.id));
      if (!preserve) this.release();
      this.accepted = faces;
      this.kept = new Set(faces.filter((face) => !preserve || !omitted.has(face.id)).map((face) => face.id));
      this.render();
    }
    add(files) {
      if (this.busy || !files.length) return;
      if (this.total + files.length > 15) {
        UI.message(this.status, 'Максимум 15 фотографий в итоговом наборе. Уберите лишние или нажмите «Заменить» у фотографии.', 'error');
        return;
      }
      this.staged.push(...files.map((file) => ({ file, url: URL.createObjectURL(file), error: '' })));
      this.changed();
    }
    replace(entry, file) {
      const index = this.staged.indexOf(entry);
      const replacement = { file, url: URL.createObjectURL(file), error: '' };
      if (entry.face) { this.kept.delete(entry.face.id); this.staged.push(replacement); }
      else if (index >= 0) { URL.revokeObjectURL(entry.url); this.staged.splice(index, 1, replacement); }
      else { URL.revokeObjectURL(replacement.url); return; }
      this.changed();
      this.grid.querySelectorAll('[data-photo-replace]')[this.kept.size + this.staged.indexOf(replacement)]?.focus();
    }
    changed() {
      this.staged.forEach((entry) => { entry.error = ''; });
      this.render(); this.onChange();
      UI.message(this.status, 'Черновик изменён. Файлы ещё не отправлены на сервер.');
    }
    setErrors(detail) {
      this.staged.forEach((entry) => { entry.error = ''; });
      if (detail?.code === 'PHOTO_VALIDATION_FAILED') {
        for (const error of detail.files || []) if (this.staged[error.index]) this.staged[error.index].error = error.error;
      }
      this.render();
    }
    setBusy(busy) {
      this.busy = busy; this.drop.setAttribute('aria-disabled', String(busy));
      this.input.disabled = busy; this.replaceInput.disabled = busy;
      for (const button of this.grid.querySelectorAll('button')) button.disabled = busy;
    }
    release() {
      this.staged.forEach((entry) => URL.revokeObjectURL(entry.url)); this.staged = [];
    }
    reset() { this.release(); this.kept = new Set(this.accepted.map((face) => face.id)); this.render(); this.onChange(); }
    render() {
      this.count.textContent = `Принято: ${this.accepted.length}/15 · выбрано новых: ${this.staged.length} · итоговый набор: ${this.total}/15`;
      const entries = [
        ...this.accepted.filter((face) => this.kept.has(face.id)).map((face) => ({ face, url: face.thumbnail_url, label: `Сохранённое фото ${face.id}` })),
        ...this.staged,
      ];
      this.grid.replaceChildren(...entries.map((entry, index) => {
        const cell = document.createElement('figure'); cell.className = 'face-cell';
        const image = document.createElement('img'); image.src = entry.url; image.alt = entry.label || entry.file.name;
        const label = document.createElement('figcaption'); label.textContent = entry.label || entry.file.name;
        const kind = document.createElement('span'); kind.className = 'photo-kind muted'; kind.textContent = entry.face ? 'Принято' : 'Не сохранено';
        const actions = document.createElement('div'); actions.className = 'photo-actions';
        const replace = document.createElement('button'); replace.type = 'button'; replace.className = 'btn ghost';
        replace.textContent = 'Заменить'; replace.dataset.photoReplace = '';
        replace.setAttribute('aria-label', `Заменить: ${label.textContent}`); replace.disabled = this.busy;
        replace.addEventListener('click', () => { this.replacing = entry; this.replaceInput.click(); });
        const remove = document.createElement('button'); remove.type = 'button'; remove.className = 'btn ghost';
        remove.textContent = 'Убрать'; remove.disabled = this.busy; remove.setAttribute('aria-label', `Убрать: ${label.textContent}`);
        remove.addEventListener('click', () => {
          if (entry.face) this.kept.delete(entry.face.id);
          else { URL.revokeObjectURL(entry.url); this.staged = this.staged.filter((item) => item !== entry); }
          this.changed();
          const next = this.grid.querySelectorAll('[data-photo-replace]')[Math.min(index, this.total - 1)];
          (next || this.input).focus();
        });
        actions.append(replace, remove); cell.append(image, label, kind, actions);
        if (entry.error) {
          const error = document.createElement('p'); error.className = 'photo-error'; error.textContent = entry.error; cell.append(error);
        }
        return cell;
      }));
    }
  }
  class WorkplacePicker {
    constructor({ select, trigger, dialog, form, error, onChange = () => {} }) {
      Object.assign(this, { select, trigger, form, error, onChange });
      this.places = []; this.created = new Map(); this.existing = new Set(); this.sequence = 0; this.busy = false;
      this.dialog = UI.dialog(dialog, trigger, { isBusy: () => this.busy });
      select.addEventListener('change', () => this.onChange());
      trigger.addEventListener('click', () => { UI.message(error, ''); this.dialog.open(); });
      form.addEventListener('submit', async (event) => {
        event.preventDefault(); if (this.busy) return;
        this.busy = true; UI.setBusy(form, true); UI.message(error, 'Создаём рабочее место…');
        try {
          const place = await UI.api('/api/workplaces', { method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name: form.elements.name.value.trim() }) });
          this.created.set(place.id, place);
          this.places = [...this.places.filter((item) => item.id !== place.id), place];
          this.render([...this.selected, place.id]); this.onChange({ created: true }); form.reset();
          this.busy = false; this.dialog.close();
        } catch (error) { UI.message(this.error, error.message, 'error'); }
        finally { this.busy = false; UI.setBusy(form, false); }
      });
    }
    get selected() { return Array.from(this.select.selectedOptions, (option) => Number(option.value)); }
    async load(selected = null, existing = []) {
      const sequence = ++this.sequence;
      const places = await UI.api('/api/workplaces');
      if (sequence !== this.sequence) return;
      const known = new Set(places.map((place) => place.id));
      this.places = [...places, ...Array.from(this.created.values()).filter((place) => !known.has(place.id))];
      for (const id of known) this.created.delete(id);
      this.existing = new Set(existing.map((place) => place.id));
      this.render(selected === null ? this.selected : selected);
    }
    render(selected) {
      const keep = new Set(selected);
      this.select.replaceChildren(...this.places.filter((place) => place.active || this.existing.has(place.id)).map((place) => {
        const option = new Option(place.name + (place.active ? '' : ' (архив)'), place.id);
        option.selected = keep.has(place.id); return option;
      }));
    }
  }
  window.EmployeeForm = { PhotoDraft, WorkplacePicker };

  const form = document.querySelector('[data-employee-create]');
  if (!form) return;
  const $ = (id) => document.getElementById(id);
  const status = $('create-status');
  let requestId = null, busy = false;
  const changed = () => { requestId = null; };
  const photos = new PhotoDraft({ input: $('create-files'), grid: $('draft-photos'), count: $('draft-count'),
    drop: $('photo-drop'), status, onChange: changed });
  const picker = new WorkplacePicker({ select: form.elements.workplace_ids, trigger: $('btn-workplace'),
    dialog: $('workplace-dialog'), form: $('workplace-form'), error: $('workplace-error'), onChange: changed });
  form.addEventListener('input', () => { if (!busy) changed(); });
  form.addEventListener('submit', async (event) => {
    event.preventDefault(); if (busy) return;
    if (!photos.total) { UI.message(status, 'Выберите хотя бы одну фотографию.', 'error'); $('create-files').focus(); return; }
    requestId ||= UI.newRequestId();
    const body = new FormData();
    body.append('metadata', JSON.stringify({
      first_name: form.elements.first_name.value.trim(), last_name: form.elements.last_name.value.trim(),
      external_id: form.elements.external_id.value.trim() || null, request_id: requestId, workplace_ids: picker.selected,
    }));
    photos.files.forEach((file) => body.append('files', file));
    busy = true; UI.setBusy(form, true); photos.setBusy(true);
    UI.message(status, 'Проверяем все фотографии. Сотрудник появится только после успешной проверки всего набора…');
    try {
      const employee = await UI.api('/api/employees', { method: 'POST', body });
      location.assign(`/employees/${employee.id}`);
    } catch (error) {
      photos.setErrors(error.detail);
      UI.message(status, error.message + ' Черновик и файлы оставлены на этой странице для исправления.', 'error');
    } finally { busy = false; UI.setBusy(form, false); photos.setBusy(false); }
  });
  async function loadPlaces() {
    $('places-retry').disabled = true;
    try { await picker.load(); UI.message($('places-status'), ''); $('places-retry').hidden = true; }
    catch (error) {
      UI.message($('places-status'), 'Рабочие места не загружены. Их можно не указывать: ' + error.message, 'error');
      $('places-retry').hidden = false;
    } finally { $('places-retry').disabled = false; }
  }
  $('places-retry').addEventListener('click', loadPlaces);
  loadPlaces();
})();
