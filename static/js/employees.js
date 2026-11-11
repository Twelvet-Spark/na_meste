// Directory refreshes never discard the last successful content or filter state.
(() => {
  'use strict';
  const UI = window.OperatorUI;
  const $ = (id) => document.getElementById(id);
  let employees = [], places = [], loaded = false, sequence = 0;
  let deleting = null, deleteBusy = false;
  const deletion = UI.dialog($('delete-dialog'), null, { isBusy: () => deleteBusy });
  const normalize = (value) => String(value || '').toLocaleLowerCase('ru').replace(/ё/g, 'е').trim();
  function render() {
    const query = normalize($('employee-search').value);
    const activity = $('activity-filter').value, place = $('workplace-filter').value;
    const visible = employees.filter((employee) => {
      const name = normalize([employee.name, employee.first_name, employee.last_name, employee.external_id].filter(Boolean).join(' '));
      return query.split(/\s+/).every((word) => name.includes(word))
        && (activity === 'all' || employee.active === (activity === 'active'))
        && (!place || (place === 'none' ? !employee.workplaces.length : employee.workplaces.some((item) => item.id === Number(place))));
    });
    $('employee-count').textContent = loaded ? String(employees.length) : '';
    $('summary').textContent = loaded ? `Показано: ${visible.length} из ${employees.length} · активных: ${employees.filter((employee) => employee.active).length}` : 'Загружаем список…';
    $('employee-list').replaceChildren(...visible.map(renderEmployee));
    $('empty').hidden = !loaded || visible.length > 0;
    const filtered = employees.length > 0;
    $('empty-title').textContent = filtered ? 'По этим условиям никто не найден' : 'Добавьте первого сотрудника';
    $('empty-description').textContent = filtered ? 'Измените запрос или сбросьте фильтры — данные сотрудников не изменятся.' : 'Укажите имя и фамилию и выберите от 1 до 15 фотографий. Рабочее место — необязательно.';
    $('empty-create').hidden = filtered; $('clear-filters').hidden = !filtered;
  }
  function renderPlaces() {
    const selected = $('workplace-filter').value;
    const all = new Map(places.map((place) => [place.id, place]));
    employees.forEach((employee) => employee.workplaces.forEach((place) => all.set(place.id, place)));
    const options = [new Option('Все рабочие места', ''), new Option('Без рабочего места', 'none'),
      ...Array.from(all.values()).sort((a, b) => a.name.localeCompare(b.name, 'ru')).map((place) => new Option(place.name + (place.active ? '' : ' (архив)'), place.id))];
    // Preserve a chosen filter even if its workplace disappeared from a later response.
    if (selected && !options.some((option) => option.value === selected)) {
      options.push(new Option($('workplace-filter').selectedOptions[0]?.textContent || 'Выбранное рабочее место', selected));
    }
    $('workplace-filter').replaceChildren(...options); $('workplace-filter').value = selected;
  }
  function renderEmployee(employee) {
    const item = document.createElement('li'); item.className = 'employee-row';
    const thumb = document.createElement('div'); thumb.className = 'employee-thumb';
    const firstFace = employee.faces?.find((face) => face.thumbnail_url);
    if (firstFace) {
      const image = document.createElement('img'); image.src = firstFace.thumbnail_url; image.alt = ''; image.loading = 'lazy'; thumb.append(image);
    } else { thumb.textContent = 'Нет фото'; }
    const info = document.createElement('div'); info.className = 'employee-info';
    const name = document.createElement('a'); name.className = 'employee-name';
    name.href = `/employees/${employee.id}`; name.textContent = employee.name;
    const meta = document.createElement('p'); meta.className = 'meta muted';
    meta.textContent = [employee.active ? 'Активен' : 'Неактивен', employee.external_id ? `ID: ${employee.external_id}` : ''].filter(Boolean).join(' · ');
    const chips = document.createElement('div'); chips.className = 'workplace-chips';
    if (!employee.workplaces.length) { const none = document.createElement('span'); none.className = 'muted small'; none.textContent = 'Без рабочего места'; chips.append(none); }
    employee.workplaces.forEach((place) => {
      const chip = document.createElement('span'); chip.className = 'workplace-chip'; chip.textContent = place.name + (place.active ? '' : ' (архив)'); chips.append(chip);
    });
    info.append(name, meta, chips);
    const readiness = document.createElement('div'); readiness.className = 'employee-readiness';
    const count = document.createElement('span'); count.className = 'photo-count'; count.textContent = `${employee.faces_count}/15 принято`;
    const state = document.createElement('span');
    const needsCorrection = employee.needs_name_completion || employee.needs_photo_remediation;
    state.className = `status ${needsCorrection ? 'connecting' : employee.active ? 'online' : 'offline'}`;
    state.textContent = needsCorrection ? 'Нужно дополнить' : employee.active ? 'Готов к распознаванию' : 'Распознавание отключено';
    readiness.append(count, state);
    if (needsCorrection) {
      const help = document.createElement('span'); help.className = 'small muted';
      help.textContent = [employee.needs_name_completion ? 'Уточните имя и фамилию' : '', employee.needs_photo_remediation ? 'Исправьте набор фотографий' : ''].filter(Boolean).join(' · ');
      readiness.append(help);
    }
    const actions = document.createElement('div'); actions.className = 'employee-row-actions';
    const open = document.createElement('a'); open.className = 'btn'; open.href = name.href; open.textContent = 'Открыть'; open.setAttribute('aria-label', `Открыть: ${employee.name}`);
    const remove = document.createElement('button'); remove.className = 'btn ghost'; remove.type = 'button'; remove.textContent = 'Удалить'; remove.setAttribute('aria-label', `Удалить: ${employee.name}`);
    remove.addEventListener('click', () => {
      deleting = employee; $('delete-description').textContent = `«${employee.name}» и фотографии будут удалены.`;
      UI.message($('delete-error'), ''); deletion.open(remove);
    });
    actions.append(open, remove); item.append(thumb, info, readiness, actions); return item;
  }
  async function loadEmployees() {
    const active = ++sequence;
    $('employee-list').setAttribute('aria-busy', 'true');
    const [employeeResult, placeResult] = await Promise.allSettled([UI.api('/api/employees'), UI.api('/api/workplaces')]);
    if (active !== sequence) return;
    const errors = [];
    if (employeeResult.status === 'fulfilled') { employees = employeeResult.value; loaded = true; }
    else errors.push('Не удалось обновить сотрудников: ' + employeeResult.reason.message);
    if (placeResult.status === 'fulfilled') places = placeResult.value;
    else errors.push('Не удалось обновить рабочие места: ' + placeResult.reason.message);
    renderPlaces();
    if (employeeResult.status === 'fulfilled') render();
    else if (!loaded) $('summary').textContent = 'Список пока не загружен.';
    $('employee-list').setAttribute('aria-busy', 'false');
    $('list-error').hidden = !errors.length;
    $('list-error-message').textContent = errors.join(' ') + (loaded && errors.length ? ' Последний загруженный список и фильтры сохранены.' : '');
  }
  $('delete-form').addEventListener('submit', async (event) => {
    event.preventDefault(); if (deleteBusy || !deleting) return;
    deleteBusy = true; UI.setBusy(event.currentTarget, true);
    try {
      await UI.api(`/api/employees/${deleting.id}`, { method: 'DELETE' });
      ++sequence; $('employee-list').setAttribute('aria-busy', 'false');
      employees = employees.filter((employee) => employee.id !== deleting.id); render();
      deleteBusy = false; deletion.close(); $('btn-refresh').focus();
    } catch (error) { UI.message($('delete-error'), error.message, 'error'); }
    finally { deleteBusy = false; UI.setBusy($('delete-form'), false); }
  });
  $('employee-filters').addEventListener('submit', (event) => event.preventDefault());
  $('employee-search').addEventListener('input', render);
  $('workplace-filter').addEventListener('change', render);
  $('activity-filter').addEventListener('change', render);
  $('clear-filters').addEventListener('click', () => { $('employee-filters').reset(); render(); $('employee-search').focus(); });
  $('btn-refresh').addEventListener('click', loadEmployees); $('btn-retry').addEventListener('click', loadEmployees);
  loadEmployees();
})();
