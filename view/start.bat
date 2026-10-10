@echo off
setlocal EnableExtensions

title ReliSign Quick Start
cd /d "%~dp0"

set "PORT=8001"
set "URL=http://localhost:%PORT%"
set "APP=%~dp0web_app\app.py"
set "PY=C:\Users\Nemo518\.conda\envs\torch13\python.exe"

rem --- locate python ---
if not exist "%PY%" set "PY=python"

if not exist "%PY%" (
    echo [ERROR] Python not found.
    echo         Edit this file and set PY to your python.exe full path.
    pause
    exit /b 1
)

rem --- already running? ---
netstat -ano | findstr ":%PORT%" | findstr "LISTENING" >nul 2>&1
if not errorlevel 1 (
    echo [ReliSign] Server already running on port %PORT%.
    echo [ReliSign] Opening browser: %URL%
    start "" "%URL%"
    exit /b 0
)

rem --- start backend ---
echo [ReliSign] Starting backend on port %PORT% ...
start "ReliSign Backend" "%PY%" "%APP%"

rem --- wait until ready (max 60s) ---
echo [ReliSign] Waiting for server to become ready ...
set /a N=0
:wait
timeout /t 1 /nobreak >nul
netstat -ano | findstr ":%PORT%" | findstr "LISTENING" >nul 2>&1
if not errorlevel 1 goto ready
set /a N+=1
if %N% geq 60 (
    echo [ERROR] Server did not start within 60 seconds.
    echo         Check the "ReliSign Backend" window for errors.
    pause
    exit /b 1
)
goto wait

:ready
echo [ReliSign] Ready. Opening browser: %URL%
start "" "%URL%"
echo.
echo [ReliSign] Started. Close the "ReliSign Backend" window to stop the server.
timeout /t 3 /nobreak >nul
exit /b 0
