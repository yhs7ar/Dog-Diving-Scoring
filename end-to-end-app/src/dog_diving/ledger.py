"""Score ledger (step 5a): a small CSV that accumulates accepted jump scores.

The design ref (§5a) names this the score ledger, distinct from ``report.py``'s
``batch_rollup.csv`` (a per-clip validation-batch artifact with no dog-name field and no
Accept trigger).  A ledger row is one *accepted* jump: the dog's name plus the distance the
operator accepted.  The CSV is written with ``report``'s RFC-4180 quote/cell helpers
(reused, not re-written), and the dog-names list is a small JSON file so previously entered
names survive between sessions.

Convention (decided conservatively, recorded in the mission ROADMAP):

* **CSV, not ``.xlsx``** — ``requirements.txt`` ships no spreadsheet library and the design
  body says "CSV"; a CSV is what Excel opens directly.
* **Default location ``end-to-end-app/data/``** — ``score_ledger.csv`` + ``dog_names.json``,
  next to the other site-specific ``data/`` trees.  ``DD_LEDGER`` / ``DD_DOG_NAMES`` env
  vars override, so a rig run can stage them elsewhere without editing the tree.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from dog_diving.report import _csv_cell  # reuse the step-4b-era RFC-4180 helper (design §5a)

# The ledger's fixed column order.  ``route`` / ``frame`` are informative extras read from
# the accepted measurement; they are empty when absent, never guessed.
LEDGER_COLUMNS = ("dog", "display", "total_inches", "clip", "timestamp", "route", "frame")


def ledger_row(dog: str, pending: dict | None) -> dict[str, Any]:
    """One accepted jump's row: the dog name plus the step-4b-parked accept fields.

    ``pending`` is the ``App.pending_accept`` dict 4b builds
    (``{clip, timestamp, display, total_inches, measurement}``).  ``measurement`` is read
    only for the optional ``route`` / ``frame`` extras, so a partial dict is fine.
    """
    pending = pending or {}
    meas = pending.get("measurement") or {}
    return {
        "dog": dog,
        "display": pending.get("display"),
        "total_inches": pending.get("total_inches"),
        "clip": pending.get("clip"),
        "timestamp": pending.get("timestamp"),
        "route": meas.get("route"),
        "frame": meas.get("frame"),
    }


def _ensure_header(path: Path) -> None:
    """Create the file (with its header) if absent or empty; leave a non-empty file alone."""
    if not path.exists() or path.stat().st_size == 0:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(",".join(LEDGER_COLUMNS) + "\n", encoding="utf-8")


def append_ledger_row(path: str | Path, row: dict[str, Any]) -> Path:
    """Append one row, creating the file (with its header) on first write."""
    path = Path(path)
    _ensure_header(path)
    line = ",".join(_csv_cell(row.get(c)) for c in LEDGER_COLUMNS)
    with path.open("a", encoding="utf-8", newline="") as fh:
        fh.write(line + "\n")
    return path


def load_dog_names(path: str | Path) -> list[str]:
    """The remembered dog names, unique and in saved order; ``[]`` for a missing/corrupt file.

    De-duplication is enforced here (not only on save) so the invariant "unique names" holds
    even for a hand-edited file — the combobox values are never fed a duplicate.
    """
    path = Path(path)
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return []
    if not isinstance(data, list):
        return []
    seen: list[str] = []
    for x in data:
        name = str(x).strip()
        if name and name not in seen:
            seen.append(name)
    return seen


def save_dog_names(path: str | Path, names: Iterable[str]) -> Path:
    """Persist ``names`` (de-duplicated, order-preserving, empties dropped)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    seen: list[str] = []
    for n in names:
        n = str(n).strip()
        if n and n not in seen:
            seen.append(n)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(seen, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)
    return path


def remember_dog_name(path: str | Path, name: str) -> list[str]:
    """Add ``name`` to the persisted list (front, de-duplicated) and return the new list.

    An empty/whitespace name is a no-op that returns the current list unchanged — the
    caller refuses empty names before this point, but the guard keeps the file honest.
    """
    name = str(name).strip()
    if not name:
        return load_dog_names(path)
    names = [name] + [n for n in load_dog_names(path) if n != name]
    save_dog_names(path, names)
    return names
