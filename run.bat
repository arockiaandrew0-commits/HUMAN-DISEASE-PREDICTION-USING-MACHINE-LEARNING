@echo off
REM Starts the backend API (port 5000) and the frontend (port 8000), then opens the app.
cd /d "%~dp0"

set PY=python
if exist ".venv\Scripts\python.exe" set PY=.venv\Scripts\python.exe

echo Starting backend API on http://localhost:5000 ...
start "MedPredict Backend" cmd /k "%PY% backend\app.py"

echo Starting frontend on http://localhost:8000 ...
start "MedPredict Frontend" cmd /k "%PY% -m http.server 8000 --directory frontend"

timeout /t 3 /nobreak >nul
start "" http://localhost:8000
echo.
echo Both servers are running in their own windows. Close those windows to stop them.
