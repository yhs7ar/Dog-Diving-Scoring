"""dual_view — show the paired camera's clip in the Calibrate pane (display only).

The rig records every jump with both ZED X cameras, as two MP4s that share a name except for a
camera token.  This module finds the other clip of a pair and adds a small switch under the
Calibrate view so the operator can look at either feed.  It is deliberately **display only**:

* the loaded clip (``App.clip``) is unchanged — the plane, New Plane, Run and Review all keep using
  it, exactly as before;
* while the other feed is shown, left-click / drag / arrow keys on the canvas are swallowed, so the
  plane can never be edited against the wrong camera's image (wheel zoom and middle-drag pan still
  work);
* anything that redraws the plane (a button, a finished waterline seed, loading a clip) switches the
  view back to the loaded clip first, so the plane is only ever drawn over its own camera.

Hooking it in is three lines in ``app_gui.py`` (search for ``dual_view``): ``attach_pair_view`` in
``_build_calibrate``, ``on_clip_loaded`` in ``load_clip`` and ``release_view`` in ``_redraw_plane``.
Deleting those lines removes the feature; this file then has no effect.

Naming conventions (decided elsewhere in the project — read here, never renamed)
---------------------------------------------------------------------------------
* recorder (``zed_recorder.clip_name``): ``<prefix>_right.mp4`` / ``<prefix>_left.mp4``, where
  ``right`` / ``left`` name the CAMERA on the rig (cam1 / cam2), not a lens.  Extra tokens after the
  side (e.g. a dog name, ``..._right_sapper.mp4``) are kept.
* test-bench 11-7: ``..._camera0_right[_dog].mp4`` / ``..._camera1_right[_dog].mp4`` — the
  ``camera0`` / ``camera1`` token is what differs; the ``_right`` after it is the same in both.

If a name carries a ``camera0`` / ``camera1`` token that token is swapped; otherwise the last
``right`` / ``left`` token is.  A name with neither has no pair to look for.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

#: Same list as ``app_gui.VIDEO_SUFFIXES`` (duplicated so this module never imports ``app_gui``).
VIDEO_SUFFIXES = (".mp4", ".mov", ".avi", ".mkv", ".m4v")

_CAMERA_SWAP = {"camera0": "camera1", "camera1": "camera0"}
_SIDE_SWAP = {"right": "left", "left": "right"}

#: Placeholder size when the loaded clip's size is not known (the rig's SVGA clips are 960x600).
FALLBACK_SIZE = (960, 600)


# ---------------------------------------------------------------------------
# Pairing (pure; no Tk, no video decoding)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ClipPair:
    """A clip and the other camera's clip of the same jump.

    ``partner`` is the existing file or ``None``; ``expected_name`` is the file name that was
    looked for (``None`` when the clip's name carries no camera token at all).
    """

    clip: Path
    clip_label: str
    partner: Path | None
    partner_label: str
    expected_name: str | None


def _label(token: str) -> str:
    return token if token in _CAMERA_SWAP else f"{token} camera"


def swap_camera_token(stem: str) -> tuple[str, str, str] | None:
    """``stem`` -> ``(this_token, other_token, partner_stem)``, or ``None`` if it has no token."""
    tokens = stem.split("_")
    lowered = [t.lower() for t in tokens]
    for i, tok in enumerate(lowered):                       # test-bench: camera0 <-> camera1
        if tok in _CAMERA_SWAP:
            swapped = list(tokens)
            swapped[i] = _CAMERA_SWAP[tok]
            return tok, _CAMERA_SWAP[tok], "_".join(swapped)
    for i in range(len(lowered) - 1, -1, -1):               # recorder: last right <-> left
        tok = lowered[i]
        if tok in _SIDE_SWAP:
            swapped = list(tokens)
            swapped[i] = _SIDE_SWAP[tok]
            return tok, _SIDE_SWAP[tok], "_".join(swapped)
    return None


def find_pair(clip: str | Path) -> ClipPair:
    """Find the other camera's clip of ``clip`` in the same folder (any video suffix)."""
    clip = Path(clip)
    swap = swap_camera_token(clip.stem)
    if swap is None:
        return ClipPair(clip, "loaded clip", None, "paired camera", None)
    mine, other, stem = swap
    suffixes = [clip.suffix] + [s for s in VIDEO_SUFFIXES if s != clip.suffix.lower()]
    for suffix in suffixes:
        cand = clip.with_name(stem + suffix)
        if cand.is_file() and cand.resolve() != clip.resolve():
            return ClipPair(clip, _label(mine), cand, _label(other), cand.name)
    return ClipPair(clip, _label(mine), None, _label(other), stem + clip.suffix)


def read_first_frame(clip: str | Path, index: int = 0):
    """One BGR frame by index (same reader ``app_gui.read_frame`` uses), or ``None``."""
    from dog_diving import video as video_mod

    idx = max(0, int(index))
    try:
        for _i, fr in video_mod.iter_frames(clip, start=idx, stop=idx + 1):
            return fr
    except Exception:  # noqa: BLE001 — an unreadable partner is shown as a message, not raised
        return None
    return None


# ---------------------------------------------------------------------------
# The Calibrate-pane switch
# ---------------------------------------------------------------------------

#: Canvas events that edit the plane; swallowed while the other feed is shown.
_BLOCKED_EVENTS = ("<Button-1>", "<B1-Motion>", "<ButtonRelease-1>",
                   "<Left>", "<Right>", "<Up>", "<Down>")


class PairView:
    """Two toggle buttons under the Calibrate view: the loaded clip / the other camera's clip."""

    def __init__(self, app: Any, holder: Any) -> None:
        self.app = app
        self.view = app.calib_view
        self.pair: ClipPair | None = None
        self.showing_partner = False
        self._partner_cache: tuple[Path | None, Any] | None = None
        self._saved: tuple[Any, Any] | None = None   # (overlay, content_box) of the clip's view

        tk, ttk = app.tk, app.ttk
        self.row = ttk.Frame(holder.master)
        self.row.pack(after=holder, anchor="w", pady=(2, 0))
        ttk.Label(self.row, text="camera view:").pack(side="left", padx=(0, 4))
        # Plain ttk buttons, styled like the zoom bar's; the feed on screen carries a check mark.
        self.choice = tk.StringVar(value="clip")
        self.btn_clip = ttk.Button(self.row, text="loaded clip",
                                   command=lambda: self._select("clip"))
        self.btn_clip.pack(side="left", padx=1)
        self.btn_partner = ttk.Button(self.row, text="paired camera",
                                      command=lambda: self._select("partner"))
        self.btn_partner.pack(side="left", padx=1)
        self.choice.trace_add("write", lambda *_: self._sync_widgets())
        self.note = ttk.Label(self.row, text="", font=("", 8), foreground="#666")
        self.note.pack(side="left", padx=8)

        # A bindtag in FRONT of the canvas's own bindings: while the other feed is shown it returns
        # "break", so the plane handlers in app_gui never see the click.
        self._block_tag = f"PairViewBlock{id(self)}"
        canvas = self.view.canvas
        canvas.bindtags((self._block_tag,) + tuple(canvas.bindtags()))
        for seq in _BLOCKED_EVENTS:
            canvas.bind_class(self._block_tag, seq, self._block)
        self._sync_widgets()

    # -- hooks called from app_gui ------------------------------------------
    def on_clip_loaded(self, clip: str | Path) -> None:
        """A clip was opened: go back to its own view and look for its pair."""
        self.release_view()
        self.pair = find_pair(clip)
        self._partner_cache = None
        self._sync_widgets()

    def release_view(self) -> bool:
        """Leave the other camera's view (if shown) without drawing anything.

        Returns ``True`` when it did, so the caller re-renders the clip's frame from scratch.
        """
        if not self.showing_partner:
            return False
        self.showing_partner = False
        overlay, box = self._saved if self._saved is not None else (None, None)
        self.view.overlay = overlay
        self.view.content_box = box
        self._saved = None
        # The view's photo is the partner's: force app_gui to re-show its own frame.
        if hasattr(self.app, "_plane_shown_key"):
            self.app._plane_shown_key = None
        self.choice.set("clip")
        self.note.configure(text="")
        return True

    # -- the switch ------------------------------------------------------------
    def _select(self, which: str) -> None:
        self.choice.set(which)
        self._on_choice()

    def _on_choice(self) -> None:
        if self.choice.get() == "partner":
            self.show_partner()
        else:
            self.show_clip()

    def show_partner(self) -> None:
        if self.pair is None:
            self.choice.set("clip")
            return
        if not self.showing_partner:
            self._saved = (self.view.overlay, self.view.content_box)
            self.showing_partner = True
            self.choice.set("partner")
        frame, message = self._partner_image()
        self.view.content_box = None
        self.view.overlay = (lambda v, m=message: self._draw_message(v, m)) if message else None
        self.view.show(frame, reset_view=True)
        shown = self.pair.partner.name if self.pair.partner is not None else "no file"
        self.note.configure(text=f"display only — {shown}  ·  plane tools and Run use "
                                 f"{self.pair.clip.name}")

    def show_clip(self) -> None:
        if not self.release_view():
            return
        app = self.app
        if getattr(app, "plane_session", None) is not None and getattr(app, "pool_frame", None) is not None:
            app._redraw_plane(reset_view=True)
            return
        frame = read_first_frame(self.pair.clip, self._frame_index()) if self.pair else None
        if frame is not None:
            self.view.show(frame, reset_view=True)

    # -- helpers ---------------------------------------------------------------
    def _frame_index(self) -> int:
        var = getattr(self.app, "calib_frame_idx", None)
        try:
            return int(var.get()) if var is not None else 0
        except Exception:  # noqa: BLE001
            return 0

    def _placeholder_size(self) -> tuple[int, int]:
        info = getattr(self.app, "info", None)
        w, h = int(getattr(info, "width", 0) or 0), int(getattr(info, "height", 0) or 0)
        return (w, h) if w > 0 and h > 0 else FALLBACK_SIZE

    def _partner_image(self) -> tuple[np.ndarray, str | None]:
        """The partner's frame, or a black frame + the message to draw on it."""
        pair = self.pair
        if self._partner_cache is not None and self._partner_cache[0] == pair.partner:
            return self._partner_cache[1]
        message = None
        frame = None
        if pair.partner is None:
            message = f"associated video for {pair.partner_label} not found"
            if pair.expected_name:
                message += f"\n(looked for {pair.expected_name})"
        else:
            frame = read_first_frame(pair.partner, self._frame_index())
            if frame is None:
                message = f"could not read {pair.partner.name}"
        if frame is None:
            w, h = self._placeholder_size()
            frame = np.zeros((h, w, 3), dtype=np.uint8)
        result = (frame, message)
        self._partner_cache = (pair.partner, result)
        return result

    @staticmethod
    def _draw_message(view: Any, message: str) -> None:
        """Centred text as an overlay item, so it stays crisp at any zoom."""
        size = view.image_size
        if size is None:
            return
        z = float(view.zoom or 1.0)
        w, h = size
        view.canvas.create_text(w * z / 2.0, h * z / 2.0, text=message, fill="#e6e6e6",
                                font=("", 12, "bold"), justify="center",
                                width=max(200, int(w * z * 0.8)), tags=("overlay",))

    def _block(self, _event) -> str | None:
        return "break" if self.showing_partner else None

    def _sync_widgets(self) -> None:
        pair = self.pair
        state = "normal" if pair is not None else "disabled"
        on_partner = self.choice.get() == "partner"
        mark = lambda shown: "✓ " if (pair is not None and shown) else ""   # noqa: E731
        self.btn_clip.configure(
            state=state,
            text=mark(not on_partner) + (f"{pair.clip_label} (loaded)" if pair else "loaded clip"))
        self.btn_partner.configure(
            state=state,
            text=mark(on_partner) + (pair.partner_label if pair else "paired camera"))


def attach_pair_view(app: Any, holder: Any) -> PairView:
    """Add the camera-view switch under the Calibrate view (``holder`` = the view's frame)."""
    return PairView(app, holder)
