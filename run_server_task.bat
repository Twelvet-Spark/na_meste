@echo off
setlocal
rem PALEVO — самовосстанавливающаяся обёртка сервера.
rem Только одна копия обёртки. Если настроенный APP_PORT занят — ждём;
rem занятый порт не доказывает, что там работает именно это приложение.
rem После выхода сервера перезапуск через 5 секунд.
rem Автозапуск при входе в Windows — ярлык на этот файл в
rem   shell:startup  (не требует прав администратора).
rem В Планировщике задач запускайте только одну обёртку от нужной учётной записи.
rem Параметры APP_PORT считываются при запуске обёртки; после их смены перезапустите её.
cd /d "%~dp0" || exit /b 1
set "PYTHON=.venv\Scripts\python.exe"
if not exist "%PYTHON%" set "PYTHON=venv\Scripts\python.exe"
if not exist "%PYTHON%" (
  echo Ошибка: Python venv не найден. Ожидался .venv или venv в папке проекта.
  exit /b 1
)
if not exist "data\" mkdir "data"
if not exist "data\" (
  echo Не удалось создать каталог data для журнала сервера.
  exit /b 1
)
set "SERVER_PORT="
for /f "delims=" %%P in ('%PYTHON% -c "import sys; from app.config import settings; port = settings.app_port; print(port) if port in range(1, 65536) else sys.exit(1)"') do set "SERVER_PORT=%%P"
if not defined SERVER_PORT (
  echo Не удалось прочитать APP_PORT. Проверьте конфигурацию приложения.
  exit /b 1
)
:loop
netstat -ano | findstr /R /C:":%SERVER_PORT% .*LISTENING" >nul 2>&1
if not errorlevel 1 (
  rem Порт занят; не запускаем второй процесс на том же порту.
  timeout /t 30 /nobreak >nul 2>&1 || "%PYTHON%" -c "import time; time.sleep(30)"
  goto loop
)
echo === server-restart %date% %time% === >> "data\app_run.log"
"%PYTHON%" main.py >> "data\app_run.log" 2>&1
echo === exit code %errorlevel%, перезапуск через 5 с === >> "data\app_run.log"
timeout /t 5 /nobreak >nul 2>&1 || "%PYTHON%" -c "import time; time.sleep(5)"
goto loop
