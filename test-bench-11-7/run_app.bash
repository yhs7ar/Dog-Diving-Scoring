#!/usr/bin/env bash
# ---------------------------------------------------------------------------
#  Launcher for the Dog Diving jump-distance scorer GUI  (M8 / PLAN section 5)
#
#  Resolves a Python interpreter automatically so the same file works on any
#  Linux (or macOS) machine:
#    1. $DD_PY, if set and runnable            (explicit override, highest priority)
#    2. a venv beside this script             ( .venv/bin/python3, venv/bin/python3 )
#    3. a venv one or two levels up           ( ../.venv, ../dd-venv, ../.venv_jetson )
#    4. the project venv in the user's home   ( $HOME/dd-venv/bin/python3 )
#    5. python3 / python on PATH
#  A candidate only wins if it actually runs ("-c import sys"), so a broken venv
#  or an interpreter whose base is missing is skipped.
#  Once an interpreter is chosen its import-time dependencies are VERIFIED
#  (cv2, numpy, torch, tkinter); if any are missing the launcher prints exactly
#  which ones and how to fix, instead of letting the app crash on `import cv2`.
#
#  Usage:  ./run_app.bash                     (no clip; use "Open clip..." in the GUI)
#          ./run_app.bash --clip data/clips/18-07.mp4
#          ./run_app.bash --which-python      (print the interpreter and exit)
#  Override the interpreter by setting DD_PY before calling, e.g.:
#          DD_PY=/path/to/python3 ./run_app.bash
# ---------------------------------------------------------------------------
set -u

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd -P)"
PY=""

_accept() {
    local cand="$1"
    [ -n "$cand" ] || return 1
    [ -x "$cand" ] || return 1
    "$cand" -c "import sys" >/dev/null 2>&1 || return 1
    PY="$cand"
    return 0
}

# 1) explicit override
if [ -n "${DD_PY:-}" ]; then _accept "$DD_PY" || true; fi

# 2) venvs beside / above this script, 3) in the user's home
if [ -z "$PY" ]; then
    for rel in \
        ".venv/bin/python3" ".venv/bin/python" \
        "venv/bin/python3" "venv/bin/python" \
        "../.venv/bin/python3" "../.venv/bin/python" \
        "../venv/bin/python3" "../venv/bin/python" \
        "../dd-venv/bin/python3" "../dd-venv/bin/python" \
        "../.venv_jetson/bin/python3" "../.venv_jetson/bin/python" \
        "../../dd-venv/bin/python3" "../../dd-venv/bin/python"
    do
        if _accept "$SCRIPT_DIR/$rel"; then break; fi
    done
    # the project venv in the user's home directory (the common install spot)
    if [ -z "$PY" ]; then
        for rel in "dd-venv/bin/python3" "dd-venv/bin/python"; do
            if _accept "${HOME:-}/$rel"; then break; fi
        done
    fi
fi

# 4) interpreters on PATH
if [ -z "$PY" ]; then
    for name in python3 python; do
        if command -v "$name" >/dev/null 2>&1; then
            _accept "$(command -v "$name")" && break
        fi
    done
fi

if [ -z "$PY" ]; then
    echo "ERROR: no working Python interpreter found." >&2
    echo "" >&2
    echo "Looked for \$DD_PY, a venv beside/above this script, \$HOME/dd-venv," >&2
    echo "then python3/python on PATH." >&2
    echo "Set DD_PY to a working interpreter and retry, e.g." >&2
    echo "    export DD_PY=/path/to/python3 && ./run_app.bash" >&2
    exit 2
fi

# optional diagnostic: print the resolved interpreter and stop (before deps)
if [ "${1:-}" = "--which-python" ]; then
    echo "$PY"
    exit 0
fi

# verify import-time dependencies so the app cannot crash on a bare import
_missing=""
for mod in cv2 numpy torch tkinter; do
    "$PY" -c "import $mod" >/dev/null 2>&1 || _missing="$_missing $mod"
done
if [ -n "$_missing" ]; then
    echo "ERROR: the resolved interpreter is missing required packages:$_missing" >&2
    echo "" >&2
    echo "    $PY" >&2
    echo "" >&2
    echo "Fix by pointing DD_PY at a virtualenv that has them, e.g." >&2
    echo "    export DD_PY=\$HOME/dd-venv/bin/python3 && ./run_app.bash" >&2
    echo "or install the project requirements into that interpreter:" >&2
    echo "    $PY -m pip install -r requirements.txt" >&2
    echo "  (tkinter ships with CPython on Windows; on Linux: apt install python3-tk)" >&2
    exit 3
fi

cd "$SCRIPT_DIR" || exit 1
"$PY" "src/app_gui.py" "$@"
EXIT_CODE=$?
if [ "$EXIT_CODE" -ne 0 ]; then
    echo ""
    echo "The app exited with an error (see the traceback above)."
    exit "$EXIT_CODE"
fi
