@echo off
REM Starts MedPredict: the Flask backend serves both the API and the website
REM on http://localhost:5000. Always uses the project's .venv so the right
REM packages (Flask, scikit-learn, pandas) are available.
cd /d "%~dp0"

set VENV_PY=.venv\Scripts\python.exe

if not exist "%VENV_PY%" (
    echo Creating virtual environment in .venv ...
    python -m venv .venv || goto :fail
)

"%VENV_PY%" -c "import flask, flask_cors, sklearn, pandas, numpy" 2>nul
if errorlevel 1 (
    echo Installing dependencies from backend\requirements.txt ...
    "%VENV_PY%" -m pip install -r backend\requirements.txt || goto :fail
)

echo.
echo Starting MedPredict on http://localhost:5000
echo Keep this window open while you use the app. Press Ctrl+C to stop.
echo.
start "" cmd /c "timeout /t 3 /nobreak >nul && start http://localhost:5000"
"%VENV_PY%" backend\app.py
pause
exit /b 0

:fail
echo.
echo Setup failed. See the messages above.
pause
exit /b 1
