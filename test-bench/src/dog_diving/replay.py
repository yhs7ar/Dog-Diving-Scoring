"""Replay scheduler — time-based slow-motion playback for the Review pane (step 4c).

The Review pane already has frame *navigation* (a slider + prev/next/±5/±10 buttons
driving ``ReviewState.choose``).  What it lacks is *playback*: a loop that advances
the frame at a reduced rate so the operator watches the jump in slow motion instead
of clicking frame-by-frame.

This module is the Tk-free half of that feature.  ``ReplayScheduler`` is a pure
state machine that emits frame indices on a timer; the GUI injects ``root.after`` /
``root.after_cancel`` as the clock and routes ``on_frame`` into the *same* ``on_slider``
path manual scrubbing uses, so the overlay, the measurement and the plan view follow
playback exactly as they follow a hand-drag.

The clock is **dependency-injected** on purpose: the unit tests drive it against a
fake clock with no display and no ``sleep``, which is the only way to test a
``root.after``-driven loop headlessly (a real Tk root is flaky on this host).
"""

from __future__ import annotations

import threading
from typing import Callable

#: The slow-motion playback factor, as a fraction of real time.  0.25 means one frame
#: every ``1 / (0.25 * fps)`` seconds — i.e. 4x slower than the source.  A named
#: constant so a later operator control (a speed dropdown) has one thing to change.
REPLAY_SPEED = 0.25

#: A ``root.after``-compatible clock: ``after(ms, fn) -> id`` schedules ``fn`` after
#: ``ms`` and returns a cancellable id; ``after_cancel(id)`` cancels it.  Injected so
#: the scheduler can run on Tk's main loop or on a test's fake clock.
AfterFn = Callable[[int, Callable[[], None]], object]
AfterCancelFn = Callable[[object], None]


def _default_after(ms: int, fn: Callable[[], None]) -> object:
    """Fallback clock (a daemon thread).  The GUI always injects ``root.after``."""
    t = threading.Timer(ms / 1000.0, fn)
    t.daemon = True
    t.start()
    return t


def _default_after_cancel(timer: object) -> None:
    getattr(timer, "cancel", lambda: None)()


class ReplayScheduler:
    """Advance a frame counter at ``speed * fps``, one frame per tick.

    ``frame`` is the index last *emitted* (``None`` before the first start).  Each
    tick advances by exactly one frame and calls ``on_frame(frame)``; when the next
    frame would pass ``stop_frame`` (= ``n_frames - 1``) it stops, calls ``on_finish``
    (if set) and schedules nothing further.  ``pause`` / ``resume`` / ``toggle`` /
    ``stop`` give the play button its states, and ``tick_ms`` is the interval so the
    unit test can assert ``tick == 1 / (speed * fps)`` without running a clock.
    """

    def __init__(
        self,
        fps: float,
        *,
        speed: float = REPLAY_SPEED,
        n_frames: int | None = None,
        after: AfterFn | None = None,
        after_cancel: AfterCancelFn | None = None,
    ) -> None:
        self.fps = float(fps)
        self.speed = float(speed)
        self.n_frames = n_frames
        self._after = after if after is not None else _default_after
        self._after_cancel = after_cancel if after_cancel is not None else _default_after_cancel
        self._after_id: object | None = None
        self._playing = False
        #: The frame index currently displayed (the last one handed to ``on_frame``).
        self.frame: int | None = None
        #: Called with each frame index as playback advances.  The GUI points this at
        #: ``on_slider`` so playback and manual scrubbing share one code path.
        self.on_frame: Callable[[int], None] | None = None
        #: Called once when playback reaches the end (never when paused/stopped).
        self.on_finish: Callable[[], None] | None = None

    # -- derived quantities ---------------------------------------------------

    @property
    def playing(self) -> bool:
        return self._playing

    @property
    def tick_ms(self) -> int | None:
        """Interval between frames in ms: ``1000 / (speed * fps)``, floored at 1.

        ``None`` when the interval is undefined (non-positive fps or speed), which is
        how a misconfigured clip refuses to play rather than scheduling a zero/negative
        delay.
        """
        if self.fps <= 0 or self.speed <= 0:
            return None
        return max(1, round(1000.0 / (self.speed * self.fps)))

    @property
    def stop_frame(self) -> int | None:
        """The last playable index: the final frame, inclusive."""
        return None if self.n_frames is None else int(self.n_frames) - 1

    # -- scheduling -----------------------------------------------------------

    def _schedule(self) -> None:
        ms = self.tick_ms
        if ms is None:
            self._playing = False
            return
        self._after_id = self._after(ms, self._tick)

    def start(self, from_frame: int | None = None) -> None:
        """Begin (or restart) playback from ``from_frame`` (default: the current frame)."""
        if self._playing:
            return
        if from_frame is not None:
            self.frame = int(from_frame)
        if self.frame is None:
            self.frame = 0
        self._playing = True
        self._schedule()

    def pause(self) -> None:
        """Halt the timer; the frame stays where it is."""
        self._playing = False
        if self._after_id is not None:
            self._after_cancel(self._after_id)
            self._after_id = None

    def resume(self) -> None:
        """Continue from the current frame (no-op if already playing or never started)."""
        if self._playing or self.frame is None:
            return
        self._playing = True
        self._schedule()

    def toggle(self) -> None:
        if self._playing:
            self.pause()
        elif self.frame is not None:
            self.resume()
        else:
            self.start()

    def stop(self) -> None:
        """Pause and forget the position, so the next start begins at frame 0."""
        self.pause()
        self.frame = None

    # -- the tick -------------------------------------------------------------

    def _tick(self) -> None:
        if not self._playing:
            return
        self._after_id = None
        nxt = (0 if self.frame is None else int(self.frame)) + 1
        if self.stop_frame is not None and nxt > self.stop_frame:
            # Reached the end: stop cleanly and tell the caller once.
            self._playing = False
            if self.on_finish is not None:
                self.on_finish()
            return
        self.frame = nxt
        if self.on_frame is not None:
            self.on_frame(nxt)
        # Re-check: on_frame may have paused us (e.g. the operator pressed Pause or
        # a new run replaced the scheduler mid-tick).
        if self._playing:
            self._schedule()
