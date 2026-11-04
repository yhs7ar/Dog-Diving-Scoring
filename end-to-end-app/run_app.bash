#!/usr/bin/env bash
# ---------------------------------------------------------------------------
#  Launcher for the Dog Diving jump-distance scorer GUI  (M8 / PLAN section 5)
#
#  This file is the single source of truth for starting the app.
#
#  Usage:  ./run_app.sh                       (no clip; use "Open clip..." in the GUI)
#          ./run_app.sh --clip data/clips/18-07.mp4
#  Override the interpreter by setting DD_PY before calling, e.g.:
#          DD_PY=/path/to/python3 ./run_app.sh
# ---------------------------------------------------------------------------

# Set default interpreter if DD_PY is not already defined
DD_PY="${DD_PY:-/home/dogdiving/Documents/Captstone-Repo/.venv_jetson/bin/python3}"

if [ ! -f "$DD_PY" ]; then
    echo "ERROR: interpreter not found at \"$DD_PY\""
    echo ""
    echo "The project's own .venv cannot run from G: -- see HANDOFF.md, "Environment"."
    echo "Set DD_PY to a working interpreter and retry, e.g."
    echo "    export DD_PY=/path/to/python3 && ./run_app.sh"
    echo ""
    read -p "Press [Enter] to exit..."
    exit 2
fi

# Change to the directory where this script is located
cd "$(dirname "$0")"

# Run the python script, passing all arguments along
"$DD_PY" "src/app_gui.py" "$@"
EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    echo ""
    echo "The app exited with an error (see the traceback above)."
    read -p "Press [Enter] to exit..."
    exit $EXIT_CODE
fi