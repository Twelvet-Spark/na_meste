@echo off
setlocal
rem PALEVO — запуск сервера. Использовать этот файл, а не "python main.py":
rem системный Python не имеет зависимостей проекта (uvicorn и т.д.),
rem они установлены в venv.
cd /d "%~dp0"
echo Запуск PALEVO...
set "PYTHON=%~dp0.venv\Scripts\python.exe"
if not exist "%PYTHON%" set "PYTHON=%~dp0venv\Scripts\python.exe"
if not exist "%PYTHON%" (
  echo Ошибка: Python venv не найден. Ожидался .venv или venv в папке проекта.
  pause
  exit /b 1
)
"%PYTHON%" main.py
set "EXIT_CODE=%ERRORLEVEL%"
pause
exit /b %EXIT_CODE%
