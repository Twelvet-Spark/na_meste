// Setup transfer drafts and secrets live only in this page's memory.
(() => {
  'use strict';
  const { api, message, newRequestId, setBusy } = window.OperatorUI;
  const $ = (id) => document.getElementById(id);
  const root = $('setup-transfer');
  const controls = $('transfer-controls');
  const exportForm = $('transfer-export-form');
  const importForm = $('transfer-import-form');
  const output = $('transfer-message');
  const limit = 64 * 1024 * 1024;
  const errors = {
    TRANSFER_LOCAL_ONLY: 'Откройте http://localhost:8000/settings непосредственно на компьютере приложения (можно через удалённый рабочий стол). Перенос через сеть или прокси запрещён.',
    TRANSFER_OPERATOR_REQUIRED: 'Запрос запрещён. Откройте настройки через localhost на компьютере приложения.',
    TRANSFER_PASSWORD_INVALID: 'Пароль должен содержать от 12 до 1024 байт UTF-8.',
    TRANSFER_PASSWORD_OR_FILE_INVALID: 'Неверный пароль или повреждённый файл. Проверьте пароль и выберите исходный файл снова.',
    TRANSFER_INVALID: 'Файл или запрос некорректен. Используйте файл переноса, созданный этим приложением.',
    TRANSFER_INCOMPATIBLE: 'Версия файла или базы несовместима. Установите одинаковую версию приложения на обоих компьютерах.',
    TRANSFER_MODEL_MISMATCH: 'Файл содержит несовместимые данные модели. Проверьте файл переноса и диагностику.',
    TRANSFER_MODEL_UNAVAILABLE: 'Для подготовки миниатюр недоступна модель лиц. Проверьте диагностику и повторите проверку файла.',
    ENROLLMENT_BUSY: 'Обработка фотографий уже выполняется. Подождите и повторите проверку; файл и пароль сохранены.',
    TRANSFER_DESTINATION_NOT_EMPTY: 'Импорт запрещён: установка уже содержит данные. Используйте новую пустую установку; существующие данные не изменены.',
    TRANSFER_DIGEST_MISMATCH: 'Файл изменился после проверки. Проверьте его заново.',
    TRANSFER_REQUEST_CONFLICT: 'Этот запрос уже использован для другого файла. Очистите форму и проверьте файл заново.',
    TRANSFER_TOO_LARGE: 'Файл превышает 64 МиБ или запрос слишком большой. Данные не импортированы.',
    TRANSFER_BUSY: 'Другой перенос ещё выполняется. Подождите и повторите тот же запрос.',
    TRANSFER_CONTENT_TYPE: 'Неверный формат запроса. Обновите страницу и выберите файл снова.',
    TRANSFER_STORAGE_UNAVAILABLE: 'База данных недоступна. Проверьте диагностику; при неизвестном результате повторите импорт с тем же файлом и паролем.',
    TRANSFER_UNAVAILABLE: 'Перенос недоступен. Проверьте зависимости и диагностику, затем повторите попытку.',
  };
  const warnings = {
    TRANSFER_NO_HISTORY_OR_VIDEO: 'История, видео и настройки этого компьютера не переносятся.',
    TRANSFER_LEGACY_REMEDIATION: 'Есть старые карточки с неполными именами или набором фото вне 1–15. Они сохранятся для исправления.',
    TRANSFER_MISSING_THUMBNAILS: 'У части образцов нет миниатюр. Неподтверждённые старые векторы сохранятся, но не участвуют в распознавании.',
    TRANSFER_FACES_NEED_REENROLLMENT: 'Есть непригодные для текущей модели образцы. После импорта замените их новыми фотографиями; исходные фото не хранятся.',
    TRANSFER_SCENES_NEED_REVIEW: 'Сцены импортируются как устаревшие. Проверьте геометрию на новом снимке и сохраните её явно.',
    TRANSFER_RESTART_FOR_GALLERY: 'Данные сохранены, но галерея не обновилась. Требуется обычный перезапуск.',
  };
  const labels = { cameras: 'Камеры', workplaces: 'Рабочие места', employees: 'Сотрудники', faces: 'Сохранённые образцы лиц', assignments: 'Назначения рабочих мест', scenes: 'Сцены камер' };
  let preview = null, requestId = null, busy = false, attempted = false;
  const urls = new Set();
  const explain = (error) => errors[error.detail] || 'Соединение прервано или ответ не получен. Если импорт уже был отправлен, повторите его без смены файла и пароля: повтор не создаст дубликаты.';
  const passwordValid = (value) => { const size = new TextEncoder().encode(value).length; return size >= 12 && size <= 1024; };

  function invalidate() {
    preview = null; requestId = null;
    attempted = false;
    $('transfer-preview-button').disabled = false;
    $('transfer-preview').hidden = true;
    $('transfer-counts').replaceChildren();
    $('transfer-warnings').replaceChildren();
  }
  function clear() {
    importForm.reset();
    exportForm.reset();
    invalidate();
    message(output, '');
  }
  function list(target, lines) {
    target.replaceChildren(...lines.map((text) => {
      const row = document.createElement('li'); row.textContent = text; return row;
    }));
  }
  async function availability() {
    $('transfer-retry').hidden = true;
    try {
      const view = await api('/api/setup-transfer/status', { headers: { 'X-Operator-Request': '1' }, cache: 'no-store' });
      controls.disabled = false;
      message($('transfer-status'), view.restart_required ? 'Перенос сохранён. Перезапустите приложение обычным способом; ручная запись не включится автоматически.' :
        view.destination_empty ? 'Пустая установка: можно импортировать файл с другого компьютера.' : 'Данные уже есть: экспорт доступен, новый импорт запрещён.');
    } catch (error) {
      controls.disabled = true;
      message($('transfer-status'), explain(error), 'error');
      $('transfer-retry').hidden = false;
    }
  }
  function formData(importing = false) {
    const file = $('transfer-file').files[0];
    if (!file || file.size > limit) throw { detail: file ? 'TRANSFER_TOO_LARGE' : 'TRANSFER_INVALID' };
    const password = $('transfer-import-password').value;
    if (!passwordValid(password)) throw { detail: 'TRANSFER_PASSWORD_INVALID' };
    const data = new FormData();
    data.append('file', file); data.append('password', password);
    if (importing) { data.append('request_id', requestId); data.append('expected_sha256', preview.sha256); }
    return data;
  }
  async function operation(action) {
    if (busy) return;
    busy = true; setBusy(root, true); message(output, 'Выполняется перенос…');
    try { await action(); }
    catch (error) { message(output, explain(error), 'error'); }
    finally {
      setBusy(root, false); busy = false;
      $('transfer-preview-button').disabled = attempted;
      $('transfer-import-button').disabled = !preview?.destination_empty;
    }
  }
  exportForm.addEventListener('submit', (event) => {
    event.preventDefault();
    if (!exportForm.reportValidity()) return;
    operation(async () => {
      const password = $('transfer-export-password').value;
      if (!passwordValid(password)) throw { detail: 'TRANSFER_PASSWORD_INVALID' };
      if (password !== $('transfer-export-confirm').value) {
        message(output, 'Пароли не совпадают. Повторите пароль файла.', 'error'); return;
      }
      const response = await fetch('/api/setup-transfer/export', { method: 'POST', cache: 'no-store',
        headers: { 'Content-Type': 'application/json', 'X-Operator-Request': '1' }, body: JSON.stringify({ password }) });
      if (!response.ok) {
        const data = await response.json().catch(() => null);
        throw { detail: data?.detail };
      }
      if (!response.headers.get('content-type')?.startsWith('application/octet-stream')) throw {};
      const blob = await response.blob();
      const url = URL.createObjectURL(blob); urls.add(url);
      const anchor = document.createElement('a');
      anchor.href = url; anchor.download = 'setup.nmsetup';
      root.append(anchor); anchor.click(); anchor.remove();
      setTimeout(() => { URL.revokeObjectURL(url); urls.delete(url); }, 60000);
      exportForm.reset();
      message(output, 'Файл передан браузеру для скачивания. Сохраните его и пароль отдельно. Видео и история не включены.');
    });
  });
  importForm.addEventListener('submit', (event) => {
    event.preventDefault();
    if (attempted || busy || !importForm.reportValidity()) return;
    operation(async () => {
      invalidate();
      const result = await api('/api/setup-transfer/preview', { method: 'POST', body: formData() });
      preview = result; requestId = newRequestId();
      list($('transfer-counts'), [
        ...Object.entries(labels).map(([key, label]) => `${label}: ${result.counts[key]}`),
        `Перекодировано миниатюр: ${result.faces_reencoded}`,
        `Нужно заменить образцов: ${result.faces_need_reenrollment}`,
        `Сцен для проверки: ${result.scenes_need_review}`,
      ]);
      list($('transfer-warnings'), result.warnings.map((code) => warnings[code]).filter(Boolean));
      $('transfer-preview').hidden = false;
      message(output, result.destination_empty ? 'Проверка завершена. Сверьте состав и подтвердите импорт.' : errors.TRANSFER_DESTINATION_NOT_EMPTY, result.destination_empty ? '' : 'error');
    });
  });
  $('transfer-import-button').addEventListener('click', () => {
    if (!preview || !requestId || !preview.destination_empty) return;
    operation(async () => {
      attempted = true;
      const result = await api('/api/setup-transfer/import', { method: 'POST', body: formData(true) });
      if (result.committed !== true) throw {};
      clear();
      await availability();
      const extra = result.warnings.includes('TRANSFER_RESTART_FOR_GALLERY') ? ' Галерея обновится при запуске.' : '';
      message(output, `Импорт завершён. Перезапустите приложение обычным способом. Затем проверьте камеры и повторно введите учёт в эксплуатацию.${extra}`);
    });
  });
  $('transfer-file').addEventListener('change', invalidate);
  $('transfer-import-password').addEventListener('input', invalidate);
  $('transfer-clear').addEventListener('click', clear);
  $('transfer-retry').addEventListener('click', availability);
  window.addEventListener('pagehide', () => { clear(); for (const url of urls) URL.revokeObjectURL(url); urls.clear(); });
  availability();
})();
