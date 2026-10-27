@echo off
title EcomML Studio v2 - Dashboard Server
echo.
echo ====================================================
echo   EcomML Studio v2  --  Starting Dashboard...
echo ====================================================
echo.

:: Kill anything on port 8080
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8080 ^| findstr LISTENING') do (
    echo   Freeing port 8080 (PID %%a)...
    taskkill /PID %%a /F >nul 2>&1
    timeout /t 1 /nobreak >nul
)

:: Open browser after 2 seconds
start "" timeout /t 2 /nobreak >nul
start "" "http://localhost:8080/ecomml_dashboard.html"

:: Start the server (stays open in this window)
cd /d "%~dp0"
python run.py

pause
