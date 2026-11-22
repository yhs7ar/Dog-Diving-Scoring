@echo off
REM ---------------------------------------------------------------------------
REM  Launcher for the Dog Diving jump-distance scorer GUI  (M8 / PLAN section 5)
REM
REM  Resolves a Python interpreter automatically so the same file works on any
REM  Windows machine:
REM    1. DD_PY, if set and runnable            (explicit override, highest priority)
REM    2. a venv beside this script             (.venv\Scripts\python.exe, venv\...)
REM    3. a venv one or two levels up           (..\dd-venv\Scripts\python.exe, ...)
REM    4. the project venv in the user's home   (%USERPROFILE%\dd-venv\Scripts\python.exe)
REM    5. python.exe / python3.exe / py.exe on PATH
REM  A candidate only wins if it actually runs ("-c import sys"), so a broken
REM  venv, a missing base interpreter, or a Windows Store alias is skipped.
REM  Once an interpreter is chosen its import-time dependencies are VERIFIED
REM  (cv2, numpy, torch, tkinter, matplotlib); if any are missing the launcher prints exactly
REM  which ones and how to fix, instead of letting the app crash on `import cv2`.
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

REM 4) the project venv in the user's home directory (the common install spot)
if not defined PY call :accept "%USERPROFILE%\dd-venv\Scripts\python.exe"

REM 5) interpreters on PATH (each one is run-tested before it is accepted)
if not defined PY for /f "delims=" %%P in ('where python 2^>nul') do if not defined PY call :accept "%%P"
if not defined PY for /f "delims=" %%P in ('where python3 2^>nul') do if not defined PY call :accept "%%P"
if not defined PY for /f "delims=" %%P in ('where py 2^>nul') do if not defined PY call :accept "%%P"

if not defined PY goto :fail

REM optional diagnostic: print the resolved interpreter and stop (before deps)
if /I "%~1"=="--which-python" (
    echo %PY%
    exit /b 0
)

REM verify import-time dependencies so the app cannot crash on a bare import
call :checkdeps
if errorlevel 1 goto :depsfail

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

REM ---- helper: verify cv2/numpy/torch/tkinter; echo each missing one ---------
:checkdeps
set "HAS_MISSING="
"%PY%" -c "import cv2" >nul 2>&1
if errorlevel 1 (
    echo   - cv2 ^(pip: opencv-python^)
    set "HAS_MISSING=1"
)
"%PY%" -c "import numpy" >nul 2>&1
if errorlevel 1 (
    echo   - numpy
    set "HAS_MISSING=1"
)
"%PY%" -c "import torch" >nul 2>&1
if errorlevel 1 (
    echo   - torch
    set "HAS_MISSING=1"
)
"%PY%" -c "import tkinter" >nul 2>&1
if errorlevel 1 (
    echo   - tkinter ^(ships with CPython on Windows; on Linux: python3-tk^)
    set "HAS_MISSING=1"
)
"%PY%" -c "import matplotlib" >nul 2>&1
if errorlevel 1 (
    echo   - matplotlib ^(pip: matplotlib^)
    set "HAS_MISSING=1"
)
if defined HAS_MISSING exit /b 1
exit /b 0

:depsfail
echo.
echo ERROR: the resolved interpreter is missing required packages:
echo.
echo     %PY%
echo.
echo Fix by pointing DD_PY at a virtualenv that has them, e.g.
echo     set DD_PY=C:\Users\you\dd-venv\Scripts\python.exe ^&^& run_app.cmd
echo or install the project requirements into that interpreter:
echo     "%PY%" -m pip install -r requirements.txt
echo.
pause
exit /b 3

:fail
echo ERROR: no working Python interpreter found.
echo.
echo Looked for DD_PY, a venv beside/above this script, %%USERPROFILE%%\dd-venv,
echo then python/python3/py on PATH.
echo Set DD_PY to a working interpreter and retry, e.g.
echo     set DD_PY=C:\path\to\python.exe ^&^& run_app.cmd
echo.
pause
exit /b 2
