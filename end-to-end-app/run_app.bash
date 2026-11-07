#!/usr/bin/env bash
# ---------------------------------------------------------------------------
#  Launcher for the Dog Diving jump-distance scorer GUI  (M8 / PLAN section 5)
#
#  Resolves a Python interpreter automatically so the same file works on any
#  Linux (or macOS) machine:
#    1. $DD_PY, if set and runnable            (explicit override, highest priority)
#    2. a venv beside this script             ( .venv/bin/python3, venv/bin/python3 )
#    3. a venv one or two levels up           ( ../.venv, ../dd-venv, ../.venv_jetson )
#    4. python3 / python on PATH
#  A candidate only wins if it actually runs ("-c import sys"), so a broken venv
#  or an interpreter whose base is missing is skipped.
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

# 2) venvs beside / above this script
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
fi

# 3) interpreters on PATH
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
    echo "Looked for \$DD_PY, a venv beside/above this script, then python3/python on PATH." >&2
    echo "Set DD_PY to a working interpreter and retry, e.g." >&2
    echo "    export DD_PY=/path/to/python3 && ./run_app.bash" >&2
    exit 2
fi

# optional diagnostic: print the resolved interpreter and stop
if [ "${1:-}" = "--which-python" ]; then
    echo "$PY"
    exit 0
fi

cd "$SCRIPT_DIR" || exit 1
"$PY" "src/app_gui.py" "$@"
EXIT_CODE=$?
if [ "$EXIT_CODE" -ne 0 ]; then
    echo ""
    echo "The app exited with an error (see the traceback above)."
    exit "$EXIT_CODE"
fi
