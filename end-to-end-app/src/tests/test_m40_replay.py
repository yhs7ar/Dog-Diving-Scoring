"""Step 4c — the replay scheduler (slow-motion playback), tested against a fake clock.

The GUI half (the ``\u25b6 Play 0.25\u00d7`` button in ``app_gui.App._build_review`` and the
``on_play_toggle`` / ``_replay_frame`` wiring into ``on_slider``) needs a Tk root, which is
flaky on this host — so this file exercises the Tk-free ``dog_diving.replay.ReplayScheduler``
with a deterministic fake clock and asserts the wiring is *present* on ``App`` without
constructing a window.  The playback contract under test:

* tick interval = ``1000 / (speed * fps)`` ms (0.25\u00d7 of the clip's fps);
* one frame per tick, in order, starting one past ``from_frame``;
* stops cleanly AT the last frame (``n_frames - 1``) — never past it — and calls ``on_finish`` once;
* ``pause`` / ``resume`` / ``toggle`` / ``stop`` give the button its states;
* a non-positive fps/speed refuses to play rather than scheduling a zero/negative delay.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dog_diving import replay  # noqa: E402
from dog_diving.replay import REPLAY_SPEED, ReplayScheduler  # noqa: E402


class FakeClock:
    """A ``root.after``-compatible clock the test advances one tick at a time.

    ``after`` records ``(delay_ms, fn)`` under a fresh id; ``after_cancel`` drops it.  The
    test then calls ``fire_one`` / ``fire_all`` to run the pending callbacks in the order
    they were scheduled, exactly as Tk would when their delay elapses.
    """

    def __init__(self) -> None:
        self._next = 0
        self.timers: dict[int, tuple[int, object]] = {}  # id -> (delay_ms, fn)

    def after(self, ms: int, fn) -> int:
        self._next += 1
        self.timers[self._next] = (ms, fn)
        return self._next

    def after_cancel(self, iid: int) -> None:
        self.timers.pop(iid, None)

    def pending(self) -> list[int]:
        return sorted(ms for ms, _ in self.timers.values())

    def fire_one(self) -> None:
        assert len(self.timers) == 1, f"expected exactly one pending timer, got {self.timers}"
        iid = next(iter(self.timers))
        _ms, fn = self.timers.pop(iid)
        fn()

    def fire_all(self) -> None:
        while self.timers:
            self.fire_one()


def make(*, fps: float = 60.0, n_frames: int | None = 92, speed: float = REPLAY_SPEED):
    clock = FakeClock()
    sched = ReplayScheduler(fps, speed=speed, n_frames=n_frames,
                            after=clock.after, after_cancel=clock.after_cancel)
    return sched, clock


# --- the interval -----------------------------------------------------------
def test_replay_speed_is_a_named_quarter() -> None:
    assert REPLAY_SPEED == 0.25


def test_tick_interval_is_inverse_of_speed_times_fps() -> None:
    sched, _ = make(fps=60.0, speed=0.25)               # shipped clip: 60 fps -> 1/15 s
    assert sched.tick_ms == round(1000 / (0.25 * 60)) == 67
    sched2, _ = make(fps=25.0, speed=0.25)
    assert sched2.tick_ms == 160
    sched3, _ = make(fps=30.0, speed=0.5)               # speed factor is a parameter
    assert sched3.tick_ms == round(1000 / (0.5 * 30)) == 67


def test_tick_interval_refuses_non_positive_fps_or_speed() -> None:
    for fps, speed in ((0.0, 0.25), (-30.0, 0.25), (30.0, 0.0), (30.0, -0.25)):
        sched, _ = make(fps=fps, speed=speed)
        assert sched.tick_ms is None
        sched.start()
        assert not sched.playing            # nothing scheduled
        assert sched.frame == 0             # position is still defined


# --- playback advances one frame per tick, in order -------------------------
def test_playback_steps_each_frame_and_stops_at_the_last() -> None:
    sched, clock = make(fps=10.0, n_frames=5)          # stop_frame = 4
    seen: list[int] = []
    finished: list[int] = []
    sched.on_frame = seen.append
    sched.on_finish = lambda: finished.append(1)
    sched.start(from_frame=0)
    assert sched.playing
    clock.fire_all()
    assert seen == [1, 2, 3, 4]                        # never emits 5 (past the end)
    assert not sched.playing
    assert len(finished) == 1                          # on_finish fires exactly once
    assert clock.timers == {}                          # no dangling timer after the end


def test_playback_starts_one_past_from_frame() -> None:
    sched, clock = make(fps=10.0, n_frames=10)
    seen: list[int] = []
    sched.on_frame = seen.append
    sched.start(from_frame=6)
    clock.fire_one()
    assert seen == [7]
    sched.pause()


def test_start_without_from_frame_begins_at_zero() -> None:
    sched, clock = make(fps=10.0, n_frames=10)
    seen: list[int] = []
    sched.on_frame = seen.append
    sched.start()
    clock.fire_one()
    assert seen == [1]                                 # first emitted frame is 1


# --- pause / resume / toggle / stop -----------------------------------------
def test_pause_cancels_the_pending_tick() -> None:
    sched, clock = make(fps=10.0, n_frames=10)
    sched.on_frame = lambda _f: None
    sched.start(from_frame=0)
    clock.fire_one()                                   # frame 1
    sched.pause()
    assert not sched.playing
    assert clock.timers == {}                          # the next tick was cancelled
    assert sched.frame == 1                            # position preserved


def test_resume_continues_from_the_current_frame() -> None:
    sched, clock = make(fps=10.0, n_frames=10)
    seen: list[int] = []
    sched.on_frame = seen.append
    sched.start(from_frame=0)
    clock.fire_one()                                   # -> 1
    sched.pause()
    clock.fire_all()                                   # nothing scheduled: still paused
    sched.resume()
    assert sched.playing
    clock.fire_one()
    assert seen == [1, 2]
    sched.pause()


def test_toggle_flips_play_and_pause() -> None:
    sched, clock = make(fps=10.0, n_frames=10)
    sched.on_frame = lambda _f: None
    sched.toggle()                                     # not started -> start
    assert sched.playing
    sched.toggle()                                     # -> pause
    assert not sched.playing
    sched.toggle()                                     # -> resume
    assert sched.playing
    sched.pause()


def test_stop_forgets_the_position() -> None:
    sched, clock = make(fps=10.0, n_frames=10)
    sched.on_frame = lambda _f: None
    sched.start(from_frame=4)
    clock.fire_one()                                   # -> 5
    sched.stop()
    assert not sched.playing
    assert sched.frame is None
    # a stop-then-start begins at frame 0 again
    seen: list[int] = []
    sched.on_frame = seen.append
    sched.start()
    clock.fire_one()
    assert seen == [1]
    sched.pause()


def test_on_frame_and_on_finish_are_optional() -> None:
    sched, clock = make(fps=10.0, n_frames=3)
    sched.start(from_frame=0)                          # no callbacks set
    clock.fire_all()                                   # must not raise
    assert not sched.playing
    assert sched.frame == 2                            # stopped at the last frame


# --- the GUI wiring exists, without constructing a Tk root ------------------
def test_app_exposes_the_play_wiring() -> None:
    import app_gui  # noqa: F401
    assert hasattr(app_gui.App, "on_play_toggle")
    assert hasattr(app_gui.App, "_replay_frame")
    assert hasattr(app_gui.App, "_replay_finish")
    assert hasattr(app_gui.App, "_sync_play_btn")
    assert hasattr(app_gui.App, "_stop_replay")
    # the speed factor is wired from the named constant, not a magic number
    src = Path(app_gui.__file__).read_text(encoding="utf-8")
    assert "replay_mod.REPLAY_SPEED" in src
    assert "ReplayScheduler" in src
