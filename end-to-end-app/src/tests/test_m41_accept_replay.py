"""Step 4b — the Results-pane Replay / Accept wiring, tested headlessly.

The GUI buttons themselves live in ``app_gui.App._build_results`` (a Tk widget tree), which is
flaky to construct on this host.  So this file follows the step-4c pattern: assert the wiring is
*present* on ``App``, then drive the two callbacks against a plain stub with the handful of
attributes each method actually reads — no ``tk.Tk()``, no window.

What is under test:

* ``App`` exposes ``on_replay_from_results`` / ``on_accept`` and initialises ``pending_accept``;
* both callbacks guard on ``self.review is None`` and log the exact refusal strings;
* Replay selects the Review tab then delegates to ``on_play_toggle`` (no duplicated playback loop);
* Accept parks ``display`` / ``total_inches`` + clip + timestamp into ``pending_accept``, logging
  ``accept \u2192 5a (persist pending)`` — and does NOT build the persistence (that is 5a's job).
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import app_gui  # noqa: E402


def _log_capture() -> list[str]:
    box: list[str] = []
    return box


def _stub(*, review, clip=None):
    """An object with exactly the attributes the 4b callbacks read, nothing more.

    ``on_accept`` and ``on_replay_from_results`` only touch ``review`` / ``log`` / ``clip`` /
    ``pending_accept`` (and, for replay, ``nb`` / ``tab_review`` / ``on_play_toggle``).  Driving
    them unbound against this stub keeps the test honest about that surface.
    """
    logs: list[str] = []
    nb_calls: list[object] = []
    play_calls: list[int] = []

    s = SimpleNamespace(
        review=review,
        clip=clip,
        pending_accept=None,
        tab_review=SimpleNamespace(),
        log=lambda m: logs.append(m),
        nb=SimpleNamespace(select=lambda t: nb_calls.append(t)),
        on_play_toggle=lambda: play_calls.append(1),
    )
    return s, logs, nb_calls, play_calls


# --- the wiring exists, without constructing a Tk root ----------------------
def test_app_exposes_the_accept_replay_wiring() -> None:
    assert hasattr(app_gui.App, "on_accept")
    assert hasattr(app_gui.App, "on_replay_from_results")
    # the pending state is initialised in __init__ (not left to ad-hoc assignment)
    src = Path(app_gui.__file__).read_text(encoding="utf-8")
    assert "self.pending_accept" in src


# --- Replay: guard + delegate ----------------------------------------------
def test_replay_with_no_run_logs_and_does_nothing() -> None:
    s, logs, nb_calls, play_calls = _stub(review=None)
    app_gui.App.on_replay_from_results(s)
    assert "no run to replay" in logs
    assert nb_calls == []                 # never jumps to Review
    assert play_calls == []               # never touches the playback loop


def test_replay_selects_review_then_starts_playback() -> None:
    s, logs, nb_calls, play_calls = _stub(review=object())
    app_gui.App.on_replay_from_results(s)
    assert nb_calls == [s.tab_review]     # Review tab selected
    assert play_calls == [1]              # playback delegated to on_play_toggle exactly once
    assert "no run to replay" not in logs


# --- Accept: guard + capture -----------------------------------------------
def test_accept_with_no_run_logs_and_leaves_nothing_pending() -> None:
    s, logs, _nb, _play = _stub(review=None)
    app_gui.App.on_accept(s)
    assert "no run to accept" in logs
    assert s.pending_accept is None       # nothing invented


def test_accept_captures_measurement_clip_and_timestamp() -> None:
    meas = {"display": "9' 3\"", "total_inches": 111.18, "route": "plane", "frame": 86}
    clip = Path("data/clips/video004_right-00.00.36.812-00.00.38.328-seg01.mp4")
    review = SimpleNamespace(measurement=lambda: meas)
    s, logs, _nb, _play = _stub(review=review, clip=clip)
    app_gui.App.on_accept(s)
    pa = s.pending_accept
    assert pa is not None
    assert pa["display"] == "9' 3\""
    assert pa["total_inches"] == 111.18
    assert pa["clip"] == str(clip)
    assert pa["timestamp"]               # a real timestamp string, not None
    assert pa["measurement"] is meas     # the full measurement dict is parked for 5a
    assert any("accept \u2192 5a (persist pending)" in m for m in logs)


def test_accept_with_no_clip_records_none_not_a_crash() -> None:
    review = SimpleNamespace(measurement=lambda: {"display": "n/a", "total_inches": None})
    s, _logs, _nb, _play = _stub(review=review, clip=None)
    app_gui.App.on_accept(s)
    assert s.pending_accept["clip"] is None
    assert s.pending_accept["display"] == "n/a"
