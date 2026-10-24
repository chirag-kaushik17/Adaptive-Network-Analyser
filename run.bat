@echo off
setlocal
cd /d "%~dp0"

set "PYTHON=python"
if exist "%~dp0.venv\Scripts\python.exe" set "PYTHON=%~dp0.venv\Scripts\python.exe"
if not exist "%~dp0.venv\Scripts\python.exe" if exist "%~dp0venv\Scripts\python.exe" set "PYTHON=%~dp0venv\Scripts\python.exe"

set "PYTHONPATH=%~dp0src;%PYTHONPATH%"
"%PYTHON%" -c "import fastapi, uvicorn, streamlit" >nul 2>&1
if errorlevel 1 (
    echo Required packages are missing. Install them with:
    echo   "%PYTHON%" -m pip install -r requirements.txt
    pause
    exit /b 1
)

start "ATA Backend" cmd /k "cd /d ""%~dp0"" && ""%PYTHON%"" -m ata.serve --host 127.0.0.1 --port 8000"
timeout /t 2 /nobreak >nul
start "ATA Frontend" cmd /k "cd /d ""%~dp0"" && ""%PYTHON%"" -m streamlit run src\ata\dashboard.py --server.address 127.0.0.1 --server.port 8501"

echo ATA backend:  http://127.0.0.1:8000
echo ATA dashboard: http://127.0.0.1:8501
echo Close the backend and frontend terminal windows to stop the services.
endlocal
