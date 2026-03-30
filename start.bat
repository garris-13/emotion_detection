@echo off
chcp 65001 >nul
echo ========================================
echo  Emotion Detection System
echo ========================================
echo.
color 0A

REM ========================================================
REM  Use conda emotiondetection environment
REM ========================================================
echo [1/5] Activating conda emotiondetection env...

echo [2/5] Cleaning up old Python processes...
taskkill /F /IM python.exe >nul 2>&1
timeout /t 1 /nobreak >nul

echo [3/5] Starting API server...
start "Emotion Detection API" cmd /k "cd /d %~dp0 && conda activate emotiondetection && python backend\api\api_server.py"

echo Waiting for API server to start, about 5 seconds...
timeout /t 5 /nobreak >nul

echo [4/5] Starting HTTP file server...
start "HTTP File Server" cmd /k "cd /d %~dp0 && conda activate emotiondetection && python -m http.server 8000"

echo.
echo ========================================
echo System started successfully!
echo Please visit: http://localhost:8000/frontend/examples/emotion_ui.html
echo ========================================
start http://127.0.0.1:8000/frontend/examples/emotion_ui.html
pause