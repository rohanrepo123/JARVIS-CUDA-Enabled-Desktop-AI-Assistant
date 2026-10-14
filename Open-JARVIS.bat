@echo off
setlocal
cd /d "%~dp0"
set "HF_HUB_DISABLE_SYMLINKS_WARNING=1"
set "HF_HOME=%~dp0.cache\huggingface"
set "PIP_CACHE_DIR=%~dp0.pip-cache"
set "TEMP=%~dp0.tmp"
set "TMP=%~dp0.tmp"
set "JARVIS_VENV=%~dp0..\JEnv"
set "JARVIS_DATABASE_PATH=%~dp0data\jarvis.db"
if not exist "%HF_HOME%" mkdir "%HF_HOME%"
if not exist "%PIP_CACHE_DIR%" mkdir "%PIP_CACHE_DIR%"
if not exist "%TEMP%" mkdir "%TEMP%"
if not exist "%~dp0data" mkdir "%~dp0data"
if not exist "%JARVIS_VENV%\Scripts\pythonw.exe" (
  if exist "%~dp0.python312\python.exe" (
    "%~dp0.python312\python.exe" -m venv "%JARVIS_VENV%"
  ) else (
    where py >nul 2>nul
    if errorlevel 1 (
      echo Python 3.12 is required. Install Python 3.12 and enable the Python launcher, then try again.
      pause
      exit /b 1
    )
    py -3.12 --version >nul 2>nul
    if errorlevel 1 (
      echo Python 3.12 is required. Install Python 3.12 and enable the Python launcher, then try again.
      pause
      exit /b 1
    )
    py -3.12 -m venv "%JARVIS_VENV%"
  )
  if errorlevel 1 goto setup_error
)
if not exist "%JARVIS_VENV%\.jarvis-ready" (
  "%JARVIS_VENV%\Scripts\python.exe" -m pip install --upgrade pip
  if errorlevel 1 goto setup_error
  "%JARVIS_VENV%\Scripts\python.exe" -m pip install -r requirements-lock.txt
  if errorlevel 1 goto setup_error
  "%JARVIS_VENV%\Scripts\python.exe" -m pip install -r requirements-cuda.txt
  if errorlevel 1 goto setup_error
  >"%JARVIS_VENV%\.jarvis-ready" echo ready
)
"%JARVIS_VENV%\Scripts\python.exe" -c "from importlib.metadata import version; assert version('kokoro') == '0.9.4' and version('soundfile') == '0.14.0' and version('torch') == '2.14.1+cu130'" >nul 2>nul
if errorlevel 1 (
  "%JARVIS_VENV%\Scripts\python.exe" -m pip install -r requirements-kokoro-cuda.txt
  if errorlevel 1 goto setup_error
)
start "JARVIS" "%JARVIS_VENV%\Scripts\pythonw.exe" "%~dp0desktop_app.py"
exit /b 0
:setup_error
echo Setup failed. Check your internet connection and Python installation, then run this file again.
pause
exit /b 1
