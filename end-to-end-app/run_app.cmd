@echo off
REM ---------------------------------------------------------------------------
REM  Launcher for the Dog Diving jump-distance scorer GUI  (M8 / PLAN section 5)
REM
REM  This file is the single source of truth for starting the app. The shortcut
REM  in the Projects folder points HERE, so if the interpreter moves, edit this
REM  file -- never the .lnk.
REM
REM  Why a local venv and not the project's own?  G:\...\.venv CANNOT run: its base
REM  interpreter is not installed on this host, and G: is a Google Drive *stream*
REM  mount where importing torch exceeds a 300 s timeout.  dd-venv is a local copy.
REM  See HANDOFF.md, "Environment -- READ THIS FIRST".
REM
REM  Usage:  run_app.cmd                       (no clip; use "Open clip..." in the GUI)
REM          run_app.cmd --clip data\clips\18-07.mp4
REM  Override the interpreter by setting DD_PY before calling.
REM ---------------------------------------------------------------------------
setlocal

if not defined DD_PY set "DD_PY=C:\Users\harri\dd-venv\Scripts\python.exe"

if not exist "%DD_PY%" (
    echo ERROR: interpreter not found at "%DD_PY%"
    echo.
    echo The project's own .venv cannot run from G: -- see HANDOFF.md, "Environment".
    echo Set DD_PY to a working interpreter and retry, e.g.
    echo     set DD_PY=C:\path\to\python.exe ^&^& run_app.cmd
    echo.
    pause
    exit /b 2
)

cd /d "%~dp0"
"%DD_PY%" "src\app_gui.py" %*

if errorlevel 1 (
    echo.
    echo The app exited with an error ^(see the traceback above^).
    pause
    exit /b 1
)

endlocal
