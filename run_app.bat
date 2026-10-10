@echo off
REM PhishLens launcher for Windows. Double-click it, or run .\run_app.bat from PowerShell.
cd /d "%~dp0"
if not exist venv (
  echo Creating virtual environment...
  python -m venv venv || (echo Python 3.10 or newer is required. & pause & exit /b 1)
  call venv\Scripts\activate.bat
  pip install -r requirements.txt || (pause & exit /b 1)
) else (
  call venv\Scripts\activate.bat
)
if not exist models\ensemble_config.joblib (
  echo Training models for the first time...
  python train.py || (pause & exit /b 1)
)
start "" http://127.0.0.1:8000
python app.py
pause
