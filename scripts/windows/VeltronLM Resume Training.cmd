@echo off
REM ============================================================================
REM  VeltronLM - Resume Training
REM
REM  Launcher for the desktop shortcut. Double-click this.
REM
REM  Runs a bounded training window (default 8 hours) and stops cleanly at the end.
REM  Change the default by editing HOURS below, or run:
REM      run_veltron_training_window.ps1 -Hours 2
REM ============================================================================
setlocal

set "REPO=%~dp0..\.."
set "HOURS=8"

REM Prefer the repo's own interpreter; fall back to whatever python is on PATH.
set "PY="
if exist "%REPO%\.venv\Scripts\python.exe" set "PY=%REPO%\.venv\Scripts\python.exe"
if not defined PY (
    for /f "delims=" %%p in ('where python 2^>nul') do (
        if not defined PY set "PY=%%p"
    )
)
if not defined PY (
    echo ERROR: no Python interpreter found.
    echo Install one, or set VELTRON_PYTHON.
    pause
    exit /b 1
)

title VeltronLM Training
echo.
echo   VeltronLM - Resume Training
echo   repository : %REPO%
echo   interpreter: %PY%
echo   window     : %HOURS% hours
echo.
echo   Closing this window will NOT stop training safely.
echo   Use "VeltronLM - Stop Training" instead, or press Ctrl+C.
echo.

powershell.exe -NoProfile -ExecutionPolicy Bypass -Command ^
  "& '%REPO%\scripts\windows\run_veltron_training_window.ps1' -Hours %HOURS% -RunName 'mini-pretrain' -GraceMinutes 10"

set "RC=%ERRORLEVEL%"
echo.
if "%RC%"=="10" (
    echo Training was already running; nothing was started.
) else if "%RC%"=="0" (
    echo Window complete. Latest checkpoint preserved.
) else (
    echo Finished with exit code %RC%.
)
echo.
pause
endlocal