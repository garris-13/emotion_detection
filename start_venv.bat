@echo off
chcp 65001 >nul
echo ========================================
echo  EmoCare - Emotion Detection System
echo  .venv Virtual Environment Starter
echo ========================================
echo.
color 0A

REM ========================================================
REM  Configure Virtual Environment Path
REM ========================================================
set "VENV_NAME=.venv"
set "VENV_DIR=%~dp0%VENV_NAME%"
set "VENV_PYTHON=%VENV_DIR%\Scripts\python.exe"

REM Check if virtual environment exists, create if not
if not exist "%VENV_PYTHON%" (
    echo [Init] Virtual environment not found, creating...
    echo Target path: %VENV_DIR%

    REM Try to create virtual environment with system Python
    python -m venv "%VENV_DIR%"

    if %errorlevel% neq 0 (
        echo.
        echo [ERROR] Failed to create virtual environment!
        echo Please ensure Python is installed and in PATH.
        pause
        exit /b
    )
    echo [OK] Virtual environment created successfully!
) else (
    echo [Init] Existing virtual environment detected, preparing to start...
)

echo [1/7] Using Python: %VENV_PYTHON%

REM 1. Clean up processes
echo [2/7] Cleaning up old Python processes...
taskkill /F /IM python.exe /T 2>nul 2>nul
timeout /t 2 /nobreak >nul

REM 2. Check port usage
echo [3/7] Checking port usage...
netstat -ano | findstr :7860 >nul
if %errorlevel% equ 0 (
    echo Port 7860 is in use, cleaning up...
    for /f "tokens=5" %%a in ('netstat -ano ^| findstr :7860') do taskkill /F /PID %%a
    timeout /t 1 /nobreak >nul
)

netstat -ano | findstr :8000 >nul
if %errorlevel% equ 0 (
    echo Port 8000 is in use, cleaning up...
    for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8000') do taskkill /F /PID %%a
    timeout /t 1 /nobreak >nul
)

REM 3. Create directory structure
echo [4/7] Creating directory structure...
if not exist "data\monitor_results\images" mkdir "data\monitor_results\images"
if not exist "data\monitor_results\results" mkdir "data\monitor_results\results"
if not exist "data\comprehensive_results" mkdir "data\comprehensive_results"

REM 4. Install project dependencies (fix Python 3.13 compatibility)
echo [5/7] Checking and installing dependencies...
echo Upgrading pip...
"%VENV_PYTHON%" -m pip install --upgrade pip --quiet

echo Installing compatibility dependencies...
REM Install base compatible versions first
"%VENV_PYTHON%" -m pip install flask==2.3.3 werkzeug==2.3.7 --quiet

REM Force upgrade flask-cors to Python 3.13 compatible version
echo Installing Python 3.13 compatible flask-cors...
"%VENV_PYTHON%" -m pip install flask-cors>=4.0.0 --upgrade --quiet

REM Install other dependencies
if exist "backend\requirements.txt" (
    echo Installing other dependencies from backend\requirements.txt...
    REM Skip already installed dependencies
    "%VENV_PYTHON%" -m pip install -r backend\requirements.txt --quiet --ignore-installed flask flask-cors werkzeug
)

REM 5. Start API server
echo [6/7] Starting API server...
start "EmoCare API Server" cmd /k "cd /d %~dp0 && "%VENV_PYTHON%" backend\api\api_server.py"

echo Waiting for API server to start (5 seconds)...
timeout /t 5 /nobreak >nul

REM 6. Start HTTP server
echo [7/7] Starting HTTP file server...
start "EmoCare HTTP Server" cmd /k "cd /d %~dp0 && "%VENV_PYTHON%" -m http.server 8000"

echo.
echo ========================================
echo [OK] System startup complete!
echo.
echo Access URLs:
echo     API Server:    http://localhost:7860
echo     Frontend:      http://127.0.0.1:8000/frontend/examples/emotion_ui.html
echo.
echo Features:
echo     Emotion recognition (image upload/camera monitor)
echo     Smart conversation (session management/summary memory)
echo     User login/registration
echo     MySQL data storage
echo.
echo Usage steps:
echo     1. Open frontend interface
echo     2. Login with default account (username: default_user, password: default123)
echo     3. Start using features
echo ========================================
echo.
REM Auto open browser
echo Opening browser...
start http://127.0.0.1:8000/frontend/examples/emotion_ui.html

pause