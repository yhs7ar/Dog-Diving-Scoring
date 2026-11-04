"""Step 5a — the score ledger (CSV) + dog names, tested headlessly.

The GUI controls live in ``app_gui.App._build_results`` (a Tk widget tree that is flaky to
construct on this host), so — same pattern as 4b/4c — this file asserts the *wiring* is
present on ``App``, then drives ``on_save_ledger`` against a stub with the handful of
attributes the method actually reads.  ``ledger.py`` itself is a pure module and is tested
directly against ``tmp_path`` (no real ``data/`` writes — the app redirects via
``self.ledger_path`` / ``self.dog_names_path`` for exactly this reason).

What is under test:

* ``ledger.ledger_row`` / ``append_ledger_row`` — header-once, append, RFC-4180 round-trip;
* ``ledger.load_dog_names`` / ``save_dog_names`` / ``remember_dog_name`` — dedupe/order/empties;
* ``App`` exposes ``on_save_ledger`` / ``on_open_ledger`` + the redirectable paths;
* Save guards (no run / empty name), the happy path (CSV row + name persisted + combobox
  refresh + label + log), and the self-sufficient re-capture when nothing is parked.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import app_gui  # noqa: E402
from dog_diving import ledger  # noqa: E402


# --- ledger.py: the pure CSV + names module --------------------------------
def test_ledger_row_builds_dog_plus_parked_fields() -> None:
    pending = {
        "clip": "data/clips/clip.mp4",
        "timestamp": "2026-11-04T05:00:00",
        "display": "9' 3\"",
        "total_inches": 111.18,
        "measurement": {"route": "plane", "frame": 86},
    }
    row = ledger.ledger_row("Rex", pending)
    assert row["dog"] == "Rex"
    assert row["display"] == "9' 3\""
    assert row["total_inches"] == 111.18
    assert row["clip"] == "data/clips/clip.mp4"
    assert row["timestamp"] == "2026-11-04T05:00:00"
    assert row["route"] == "plane"
    assert row["frame"] == 86


def test_ledger_row_tolerates_missing_measurement() -> None:
    row = ledger.ledger_row("Rex", {"display": "n/a", "total_inches": None})
    assert row["route"] is None
    assert row["frame"] is None
    assert row["total_inches"] is None


def test_append_ledger_row_writes_header_once_and_round_trips(tmp_path: Path) -> None:
    p = tmp_path / "ledger.csv"
    ledger.append_ledger_row(p, ledger.ledger_row("Rex", {
        "display": "9' 3\"", "total_inches": 111.18, "clip": "c.mp4",
        "timestamp": "2026-11-04T05:00:00", "measurement": {"route": "plane", "frame": 86},
    }))
    ledger.append_ledger_row(p, ledger.ledger_row("Bella", {
        "display": "7' 1\"", "total_inches": 85.0, "clip": None,
        "timestamp": None, "measurement": {},
    }))
    lines = p.read_text(encoding="utf-8").strip().splitlines()
    assert lines[0] == ",".join(ledger.LEDGER_COLUMNS)  # header exactly once
    assert len(lines) == 3
    rows = list(csv.DictReader(lines))
    assert rows[0]["dog"] == "Rex"
    assert rows[0]["total_inches"] == "111.18"
    assert rows[0]["display"] == "9' 3\""
    assert rows[1]["dog"] == "Bella"
    assert rows[1]["clip"] == ""  # None -> empty cell, not the string "None"


def test_load_dog_names_missing_empty_corrupt_valid(tmp_path: Path) -> None:
    assert ledger.load_dog_names(tmp_path / "nope.json") == []
    p = tmp_path / "names.json"
    p.write_text("not json", encoding="utf-8")
    assert ledger.load_dog_names(p) == []
    p.write_text(json.dumps({"not": "a list"}), encoding="utf-8")
    assert ledger.load_dog_names(p) == []
    p.write_text(json.dumps(["Rex", "", "Bella", "Rex", "  "]), encoding="utf-8")
    assert ledger.load_dog_names(p) == ["Rex", "Bella"]  # deduped, empties dropped


def test_save_dog_names_dedupes_orders_and_drops_empties(tmp_path: Path) -> None:
    p = tmp_path / "names.json"
    ledger.save_dog_names(p, ["Rex", "", "Bella", "Rex", "  ", "Milo"])
    assert ledger.load_dog_names(p) == ["Rex", "Bella", "Milo"]


def test_remember_dog_name_fronts_and_dedupes(tmp_path: Path) -> None:
    p = tmp_path / "names.json"
    ledger.save_dog_names(p, ["Rex", "Bella"])
    assert ledger.remember_dog_name(p, "Milo") == ["Milo", "Rex", "Bella"]
    assert ledger.remember_dog_name(p, "Rex") == ["Rex", "Milo", "Bella"]  # moved to front
    assert ledger.remember_dog_name(p, "   ") == ["Rex", "Milo", "Bella"]  # empty is a no-op


# --- app_gui wiring ---------------------------------------------------------
def _stub(*, review, name="Rex", pending=None, clip=None):
    logs: list[str] = []
    box_values: list[list[str]] = []
    label_texts: list[str] = []

    s = SimpleNamespace(
        review=review,
        clip=clip,
        pending_accept=pending,
        ledger_path=None,   # set by each test to its tmp path
        dog_names_path=None,
        dog_names=["Rex"],
        log=lambda m: logs.append(m),
        dog_name_var=SimpleNamespace(get=lambda: name),
        dog_name_box=SimpleNamespace(configure=lambda **k: box_values.append(list(k.get("values", [])))),
        save_label=SimpleNamespace(configure=lambda **k: label_texts.append(k.get("text", ""))),
    )
    return s, logs, box_values, label_texts


def test_app_exposes_the_save_ledger_wiring() -> None:
    assert hasattr(app_gui.App, "on_save_ledger")
    assert hasattr(app_gui.App, "on_open_ledger")
    src = Path(app_gui.__file__).read_text(encoding="utf-8")
    # the redirectable paths are initialised in __init__, not left to ad-hoc assignment
    assert "self.ledger_path" in src
    assert "self.dog_names_path" in src
    assert "self.dog_names" in src


def test_save_with_no_run_logs_and_writes_nothing(tmp_path: Path) -> None:
    s, logs, _bv, _lt = _stub(review=None)
    s.ledger_path = tmp_path / "ledger.csv"
    s.dog_names_path = tmp_path / "names.json"
    app_gui.App.on_save_ledger(s)
    assert "no run to save" in logs
    assert not (tmp_path / "ledger.csv").exists()


def test_save_with_empty_name_is_refused(tmp_path: Path) -> None:
    meas = {"display": "9' 3\"", "total_inches": 111.18, "route": "plane", "frame": 86}
    s, logs, _bv, _lt = _stub(review=SimpleNamespace(measurement=lambda: meas), name="   ")
    s.ledger_path = tmp_path / "ledger.csv"
    s.dog_names_path = tmp_path / "names.json"
    app_gui.App.on_save_ledger(s)
    assert "enter a dog name to save" in logs
    assert not (tmp_path / "ledger.csv").exists()


def test_save_persists_row_and_name_and_refreshes_combobox(tmp_path: Path) -> None:
    meas = {"display": "9' 3\"", "total_inches": 111.18, "route": "plane", "frame": 86}
    pending = {
        "clip": "data/clips/clip.mp4", "timestamp": "2026-11-04T05:00:00",
        "display": meas["display"], "total_inches": meas["total_inches"], "measurement": meas,
    }
    s, logs, box_values, label_texts = _stub(
        review=SimpleNamespace(measurement=lambda: meas), name="Milo", pending=pending,
        clip=Path("data/clips/clip.mp4"))
    s.ledger_path = tmp_path / "ledger.csv"
    s.dog_names_path = tmp_path / "names.json"
    # The real app loads ``dog_names`` from this same file at __init__; seed it so the stub
    # matches reality (``remember_dog_name`` re-reads from disk, not from the stub attribute).
    ledger.save_dog_names(s.dog_names_path, ["Rex"])
    app_gui.App.on_save_ledger(s)

    rows = list(csv.DictReader((tmp_path / "ledger.csv").read_text(encoding="utf-8").splitlines()))
    assert len(rows) == 1
    assert rows[0]["dog"] == "Milo"
    assert rows[0]["display"] == "9' 3\""
    assert rows[0]["total_inches"] == "111.18"
    assert rows[0]["route"] == "plane"
    # the name was persisted and the combobox refreshed with the new list
    assert ledger.load_dog_names(s.dog_names_path) == ["Milo", "Rex"]
    assert box_values and box_values[-1] == ["Milo", "Rex"]
    assert label_texts and label_texts[-1] == "saved: 9' 3\" (Milo)"
    assert any("saved \u2192 ledger: Milo @ 9' 3\"" in m for m in logs)


def test_save_recaptures_measurement_when_nothing_parked(tmp_path: Path) -> None:
    """A new run clears ``pending_accept``; Save must still work from ``review.measurement()``."""
    meas = {"display": "7' 1\"", "total_inches": 85.0, "route": "arc3d", "frame": 42}
    s, logs, _bv, _lt = _stub(review=SimpleNamespace(measurement=lambda: meas), pending=None)
    s.ledger_path = tmp_path / "ledger.csv"
    s.dog_names_path = tmp_path / "names.json"
    app_gui.App.on_save_ledger(s)

    rows = list(csv.DictReader((tmp_path / "ledger.csv").read_text(encoding="utf-8").splitlines()))
    assert len(rows) == 1
    assert rows[0]["dog"] == "Rex"
    assert rows[0]["total_inches"] == "85"
    assert rows[0]["route"] == "arc3d"
    assert any("saved \u2192 ledger: Rex @ 7' 1\"" in m for m in logs)


def test_open_ledger_references_self_ledger_path(tmp_path: Path) -> None:
    # ``on_open_ledger`` shells out to explorer; here we only assert it keys on the
    # redirectable instance path (a stub with no ledger yet logs the refusal, not a crash).
    logs: list[str] = []
    s = SimpleNamespace(ledger_path=tmp_path / "not-yet.csv", log=lambda m: logs.append(m))
    app_gui.App.on_open_ledger(s)
    assert any("ledger not yet written" in m for m in logs)
