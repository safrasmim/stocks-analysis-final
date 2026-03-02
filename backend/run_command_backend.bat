@echo off
cd /d "%~dp0"

if not exist "venv\Scripts\python.exe" (
  python -m venv venv
)

call ".\venv\Scripts\activate.bat"

REM python -m pip install --upgrade pip

if exist requirements.txt (
   REM pip install -r requirements.txt
   python -m pip install -r requirements.txt
)

python ".\scripts\generate_sample_data.py"
if errorlevel 1 pause & exit /b 1

python ".\scripts\train_all_models.py" --allow-synthetic
if errorlevel 1 pause & exit /b 1

python ".\scripts\evaluate_all_models.py"
if errorlevel 1 pause & exit /b 1

uvicorn src.app:app --host 0.0.0.0 --port 8000 --reload

REM python -m pip install --upgrade pip REM Not always
REM pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cpu  REM only in new env
REM pip install -r backend/requirements.txt REM only in new env or new lib added