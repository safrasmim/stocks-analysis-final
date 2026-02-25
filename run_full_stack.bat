@echo off
setlocal ENABLEDELAYEDEXPANSION

REM ==========================================================
REM Full local setup + training + evaluation + run (Windows)
REM Repo root: stocks-analysis-final
REM ==========================================================

cd /d %~dp0

echo [1/8] Creating Python venv (if missing)...
if not exist .venv (
    py -m venv .venv
)

echo [2/8] Activating venv...
call .venv\Scripts\activate.bat

echo [3/8] Installing backend dependencies...
python -m pip install --upgrade pip
pip install -r backend\requirements.txt

echo [4/8] Generating sample data...
cd backend
python scripts\generate_sample_data.py

echo [5/8] Training models...
python scripts\train_all_models.py
if errorlevel 1 (
    echo Training failed. Exiting.
    exit /b 1
)

echo [6/8] Evaluating models...
python scripts\evaluate_all_models.py
if errorlevel 1 (
    echo Evaluation failed. Exiting.
    exit /b 1
)

cd ..

echo [7/8] Installing frontend dependencies...
cd frontend
call npm install
if errorlevel 1 (
    echo npm install failed. Exiting.
    exit /b 1
)
cd ..

echo [8/8] Starting backend and frontend in separate windows...
start "Backend API" cmd /k "cd /d %cd%\backend && call ..\.venv\Scripts\activate.bat && uvicorn src.app:app --host 0.0.0.0 --port 8000 --reload"
start "Frontend UI" cmd /k "cd /d %cd%\frontend && npm start"

echo.
echo Done.
echo Backend:  http://localhost:8000/docs
echo Frontend: http://localhost:3000
echo.
pause
