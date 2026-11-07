@echo off
REM ---------------------------------------------------------------------------
REM  Launcher for the Dog Diving jump-distance scorer GUI  (M8 / PLAN section 5)
REM
REM  Resolves a Python interpreter automatically so the same file works on any
REM  Windows machine:
REM    1. DD_PY, if set and runnable            (explicit override, highest priority)
REM    2. a venv beside this script             (.venv\Scripts\python.exe, venv\...)
REM    3. a venv one or two levels up           (..\dd-venv\Scripts\python.exe, ...)
REM    4. python.exe / python3.exe / py.exe on PATH
REM  A candidate only wins if it actually runs ("-c import sys"), so a broken
REM  venv, a missing base interpreter, or a Windows Store alias is skipped.
REM
REM  Usage:  run_app.cmd                       (no clip; use "Open clip..." in the GUI)
REM          run_app.cmd --clip data\clips\18-07.mp4
REM          run_app.cmd --which-python        (print the interpreter and exit)
REM  Override the interpreter by setting DD_PY before calling.
REM ---------------------------------------------------------------------------
setlocal EnableExtensions

set "SCRIPT_DIR=%~dp0"
set "PY="

REM 1) explicit override
if defined DD_PY call :accept "%DD_PY%"

REM 2) venvs beside / above this script
if not defined PY call :accept "%SCRIPT_DIR%.venv\Scripts\python.exe"
if not defined PY call :accept "%SCRIPT_DIR%venv\Scripts\python.exe"
if not defined PY call :accept "%SCRIPT_DIR%..\.venv\Scripts\python.exe"
if not defined PY call :accept "%SCRIPT_DIR%..\venv\Scripts\python.exe"
if not defined PY call :accept "%SCRIPT_DIR%..\dd-venv\Scripts\python.exe"
if not defined PY call :accept "%SCRIPT_DIR%..\.venv_jetson\Scripts\python.exe"
if not defined PY call :accept "%SCRIPT_DIR%..\..\dd-venv\Scripts\python.exe"

REM 3) interpreters on PATH (each one is run-tested before it is accepted)
if not defined PY for /f "delims=" %%P in ('where python 2^>nul') do if not defined PY call :accept "%%P"
if not defined PY for /f "delims=" %%P in ('where python3 2^>nul') do if not defined PY call :accept "%%P"
if not defined PY for /f "delims=" %%P in ('where py 2^>nul') do if not defined PY call :accept "%%P"

if not defined PY goto :fail

REM optional diagnostic: print the resolved interpreter and stop
if /I "%~1"=="--which-python" (
    echo %PY%
    exit /b 0
)

cd /d "%~dp0"
"%PY%" "src\app_gui.py" %*
if errorlevel 1 (
    echo.
    echo The app exited with an error ^(see the traceback above^).
    pause
    exit /b 1
)
exit /b 0

REM ---- helper: accept a candidate only if it exists AND runs -----------------
:accept
if "%~1"=="" exit /b 1
if not exist "%~1" exit /b 1
"%~1" -c "import sys" >nul 2>&1
if errorlevel 1 exit /b 1
set "PY=%~1"
exit /b 0

:fail
echo ERROR: no working Python interpreter found.
echo.
echo Looked for DD_PY, a venv beside/above this script, then python/python3/py on PATH.
echo Set DD_PY to a working interpreter and retry, e.g.
echo     set DD_PY=C:\path\to\python.exe ^&^& run_app.cmd
echo.
pause
exit /b 2
