"""calibrate_tab — the Calibrate pane, lifted out of ``end-to-end-app/src/app_gui.py``.

A self-contained ``CalibrateTab(ttk.Frame)`` that is a FAITHFUL copy of the parent
``App``'s Calibrate tab: waterline detection, semi-manual quad/plane placement with
corner + side drag, snap-to-waterline with per-edge locks, the plane dropdown, the
``Plane Confirmed`` build, and save-to-planes-folder.

The parent's ``App`` is a notebook of five tabs (Load/Calibrate/Process/Review/Results).
This class is ONLY the Calibrate pane, so a handful of cross-tab hooks the copied methods
touch are stubbed or removed — and every one of those is noted where it happens:

* ``_redraw_plan`` (the plan view lives on the Review tab) is a no-op.
* ``on_arrow_key`` no longer gates on "is the Calibrate tab selected" (there is no
  notebook here; the pane is always the active surface).
* ``load_clip`` drops the Load/Review/Process/Results wiring (clip label, slider,
  replay scheduler, stability check) and keeps probe + read-frame + start-plane.
* ``_drain`` only dispatches the message kinds the Calibrate worker actually puts on
  the queue (``log`` / ``seed`` / ``seed_error`` / ``seed_done``).

Everything else — the plane session, the seed cache, the waterline detector worker
thread, the snap/lock primitives, the overlay drawing, the persistence path — is copied
verbatim from the parent.

The backend is the ``dog_diving`` package under ``src/`` (see the parent's own imports).
``sys.path`` is patched so ``import dog_diving`` works when this file sits in ``src/``.
"""

from __future__ import annotations

import json
import math
import os
import queue
import sys
import threading
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import tkinter as tk  # noqa: E402
from tkinter import ttk  # noqa: E402

from dog_diving.pool import plane as plane_mod  # noqa: E402
from dog_diving.pool import seed as seed_mod  # noqa: E402
from dog_diving.pool import calib as pool_calib_mod  # noqa: E402
from dog_diving import waterline as waterline_mod  # noqa: E402
from dog_diving import video as video_mod  # noqa: E402
from dog_diving.config import (  # noqa: E402
    ACTIVE_PLANE_PATH,
    PLANES_DIR,
)

# --- module constants (copied from app_gui.py) --------------------------------
CALIB_FRAME_DEFAULT = 0
ZOOM_STEPS = (1.0, 2.0, 4.0, 8.0, 16.0)

#: How much of a clip the waterline detector looks at (operator order 2026-10-21).
SEED_FRAME_COUNT = 90

#: The two +x anchors, in feet from the dock.
SEED_ANCHOR_NEAR_FT = 10.0
SEED_ANCHOR_FAR_FT = 15.0

#: The snap toggle's four edges, plane side -> label.
SNAP_SIDE_LABELS = (("back", "Top"), ("front", "Bottom"), ("left", "Left"), ("right", "Right"))

#: The blue primary-action colours.
PRIMARY_BG = "#1f6feb"
PRIMARY_BG_ACTIVE = "#388bfd"


# ===========================================================================
# Non-Tk helpers (copied from app_gui.py)
# ===========================================================================
def _finite2(p) -> bool:
    """True when a projected point is a real, finite 2-D pixel."""
    try:
        return bool(math.isfinite(float(p[0])) and math.isfinite(float(p[1])))
    except (TypeError, ValueError, IndexError):
        return False


def guard_frame(canvas_w: int, canvas_h: int, img_w: int, img_h: int) -> float:
    """Scale factor that fits an image inside a canvas without upscaling."""
    if img_w <= 0 or img_h <= 0:
        return 1.0
    return min(1.0, canvas_w / img_w, canvas_h / img_h)


def read_frame(clip: str | Path, index: int):
    """One frame by index, via ``video.iter_frames`` (never CAP_PROP_POS_FRAMES seeking)."""
    idx = max(0, int(index))
    for _i, fr in video_mod.iter_frames(clip, start=idx, stop=idx + 1):
        return fr
    return None


# ===========================================================================
# The zoom/pan canvas (copied verbatim from app_gui.py)
# ===========================================================================
class FrameView:
    """A Canvas showing one BGR frame with **zoom + pan**, mapping clicks back to image px.

    Rendering crops to the visible region *before* scaling.  The ``overlay`` callback draws
    vector overlays (the plane) on the canvas after each photo render.
    """

    def __init__(self, canvas, max_w: int = 900, max_h: int = 520) -> None:
        self.canvas = canvas
        self.max_w, self.max_h = max_w, max_h
        self.zoom = 1.0
        self.fit_zoom = 1.0
        self.on_change: Callable[["FrameView"], None] | None = None
        self.overlay: Callable[["FrameView"], None] | None = None
        self.cursor_var = None            # optional tk.StringVar for an image-px readout
        self._src = None                  # RGB numpy array of the current frame
        self._photo = None                # keep a reference or Tk garbage-collects it
        self._shown = False               # False until the first frame is displayed
        self._at_fit = True               # True while the operator has not zoomed away from fit
        self._last_size: tuple[int, int] | None = None
        self.content_box: tuple[float, float, float, float] | None = None
        self._forced_viewport: tuple[int, int] | None = None   # for tests

    # -- geometry -----------------------------------------------------------
    def _viewport(self) -> tuple[int, int]:
        if self._forced_viewport is not None:
            return self._forced_viewport
        return max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height())

    def _fit_size(self) -> tuple[int, int]:
        if self._forced_viewport is not None:
            return self._forced_viewport
        cw, ch = int(self.canvas.winfo_width()), int(self.canvas.winfo_height())
        if cw <= 1 or ch <= 1:
            return self.max_w, self.max_h
        return cw, ch

    @property
    def image_size(self) -> tuple[int, int] | None:
        return None if self._src is None else (self._src.shape[1], self._src.shape[0])

    def _clamp_zoom(self, z: float) -> float:
        return max(0.02, min(64.0, float(z)))

    def _fit_source_size(self) -> tuple[int, int] | None:
        if self._src is None:
            return None
        h, w = self._src.shape[:2]
        box = self.content_box
        if box is None:
            return w, h
        bw = max(1.0, float(box[2]) - float(box[0]))
        bh = max(1.0, float(box[3]) - float(box[1]))
        return int(round(bw)), int(round(bh))

    def _centre_on_content(self) -> None:
        box, size = self.content_box, self.image_size
        if box is None or size is None:
            return
        W, H = size
        z = self.zoom or 1.0
        sw, sh = max(1.0, W * z), max(1.0, H * z)
        cx = 0.5 * (float(box[0]) + float(box[2])) * z
        cy = 0.5 * (float(box[1]) + float(box[3])) * z
        cw, ch = self._viewport()
        self.canvas.xview_moveto(min(1.0, max(0.0, (cx - cw / 2.0) / sw)))
        self.canvas.yview_moveto(min(1.0, max(0.0, (cy - ch / 2.0) / sh)))

    def _update_scrollregion(self) -> None:
        size = self.image_size
        if size is None:
            return
        w, h = size
        self.canvas.configure(scrollregion=(0, 0, w * self.zoom, h * self.zoom))

    # -- display ------------------------------------------------------------
    def show(self, bgr, *, reset_view: bool = False) -> None:
        from PIL import Image  # local: keeps the module importable without Pillow

        if bgr is None:
            self._src = None
            self.canvas.delete("all")
            self.canvas.create_text(self.max_w // 2, self.max_h // 2, text="no frame", fill="#888")
            self._notify()
            return
        self._src = bgr[:, :, ::-1]                     # BGR -> RGB
        h, w = self._src.shape[:2]
        cw, ch = self._fit_size()
        src_w, src_h = self._fit_source_size() or (w, h)
        self.fit_zoom = guard_frame(cw, ch, src_w, src_h)
        first = reset_view or not self._shown
        if first:
            self.zoom = self.fit_zoom
            self._shown = True
            self._at_fit = True
        else:
            self.zoom = self._clamp_zoom(self.zoom)
            self._at_fit = abs(self.zoom - self.fit_zoom) < 1e-6
        self._update_scrollregion()
        if first:
            self._centre_on_content()
        self._render()

    def on_resize(self, event=None) -> None:
        if self._src is None:
            return
        if event is not None and getattr(event, "widget", None) is not self.canvas:
            return
        size = (int(getattr(event, "width", 0) or self._fit_size()[0]),
                int(getattr(event, "height", 0) or self._fit_size()[1]))
        if size[0] <= 1 or size[1] <= 1 or size == self._last_size:
            return
        self._last_size = size
        h, w = self._src.shape[:2]
        src_w, src_h = self._fit_source_size() or (w, h)
        self.fit_zoom = guard_frame(size[0], size[1], src_w, src_h)
        was_fit = self._at_fit
        self.zoom = self.fit_zoom if was_fit else self._clamp_zoom(self.zoom)
        self._update_scrollregion()
        if was_fit:
            self._centre_on_content()
        self._render()

    def refresh(self) -> None:
        self._render()

    def redraw_overlay(self) -> None:
        self.canvas.delete("overlay")
        if self.overlay is not None and self._src is not None:
            self.overlay(self)

    def _render(self) -> None:
        from PIL import Image, ImageTk

        if self._src is None:
            return
        H, W = self._src.shape[:2]
        z = self.zoom
        cw, ch = self._viewport()
        self.canvas.delete("all")

        x0, y0 = self.canvas.canvasx(0) / z, self.canvas.canvasy(0) / z
        pad = 4
        cx0, cy0 = max(0, int(x0) - pad), max(0, int(y0) - pad)
        cx1, cy1 = min(W, int(x0 + cw / z) + pad + 1), min(H, int(y0 + ch / z) + pad + 1)
        if cx1 <= cx0 or cy1 <= cy0:
            cx0, cy0, cx1, cy1 = 0, 0, min(W, max(1, cx1)), min(H, max(1, cy1))

        crop = self._src[cy0:cy1, cx0:cx1]
        resample = Image.NEAREST if z > 1.0 else Image.BILINEAR
        img = Image.fromarray(crop).resize(
            (max(1, int(round(crop.shape[1] * z))), max(1, int(round(crop.shape[0] * z)))),
            resample,
        )
        self._photo = ImageTk.PhotoImage(img)
        self.canvas.create_image(cx0 * z, cy0 * z, anchor="nw", image=self._photo)
        self.redraw_overlay()
        self._notify()

    def _notify(self) -> None:
        if self.on_change is not None:
            self.on_change(self)

    # -- zoom / pan ---------------------------------------------------------
    def set_zoom(self, z: float, anchor: tuple[float, float] | None = None) -> float:
        if self._src is None:
            return self.zoom
        if anchor is None and self.content_box is not None:
            self.zoom = self._clamp_zoom(z)
            self._at_fit = abs(self.zoom - self.fit_zoom) < 1e-6
            self._update_scrollregion()
            self._centre_on_content()
            self._render()
            return self.zoom
        if anchor is None:
            cw, ch = self._viewport()
            anchor = (cw / 2.0, ch / 2.0)
        ix = self.canvas.canvasx(anchor[0]) / self.zoom
        iy = self.canvas.canvasy(anchor[1]) / self.zoom
        self.zoom = self._clamp_zoom(z)
        self._at_fit = abs(self.zoom - self.fit_zoom) < 1e-6
        self._update_scrollregion()
        H, W = self._src.shape[:2]
        self.canvas.xview_moveto(min(1.0, max(0.0, (ix * self.zoom - anchor[0]) / max(1.0, W * self.zoom))))
        self.canvas.yview_moveto(min(1.0, max(0.0, (iy * self.zoom - anchor[1]) / max(1.0, H * self.zoom))))
        self._render()
        return self.zoom

    def zoom_by(self, factor: float, anchor: tuple[float, float] | None = None) -> float:
        return self.set_zoom(self.zoom * factor, anchor=anchor)

    def fit(self) -> float:
        return self.set_zoom(self.fit_zoom)

    def on_wheel(self, event) -> None:
        delta = getattr(event, "delta", 0) or 0
        if not delta:
            return
        self.zoom_by(1.25 ** (delta / 120.0), anchor=(event.x, event.y))

    def bind_zoom(self) -> None:
        self.canvas.bind("<MouseWheel>", self.on_wheel)
        self.canvas.bind("<Button-4>", lambda e: self.zoom_by(1.25, anchor=(e.x, e.y)))
        self.canvas.bind("<Button-5>", lambda e: self.zoom_by(0.8, anchor=(e.x, e.y)))
        self.canvas.bind("<minus>", lambda e: self.zoom_by(0.8))
        self.canvas.bind("<KP_Subtract>", lambda e: self.zoom_by(0.8))
        self.canvas.bind("<equal>", lambda e: self.zoom_by(1.25))
        self.canvas.bind("<plus>", lambda e: self.zoom_by(1.25))
        self.canvas.bind("<KP_Add>", lambda e: self.zoom_by(1.25))
        self.canvas.bind("<Configure>", self.on_resize)
        self.canvas.bind("<Button-2>", lambda e: self.canvas.scan_mark(e.x, e.y))
        self.canvas.bind("<B2-Motion>", lambda e: (self.canvas.scan_dragto(e.x, e.y, gain=1),
                                                   self._render()))
        self.canvas.bind("<Motion>", self.on_motion)

    def on_motion(self, event) -> None:
        if self.cursor_var is None:
            return
        if self._src is None:
            self.cursor_var.set("")
            return
        ix, iy = self.to_image(event.x, event.y)
        h, w = self._src.shape[:2]
        inside = 0 <= ix < w and 0 <= iy < h
        self.cursor_var.set(f"x={ix:7.2f}  y={iy:7.2f} px" + ("" if inside else "  (outside)"))

    # -- mapping ------------------------------------------------------------
    def to_image(self, cx: float, cy: float) -> tuple[float, float]:
        z = self.zoom or 1.0
        return self.canvas.canvasx(cx) / z, self.canvas.canvasy(cy) / z


# ===========================================================================
# The standalone Calibrate tab
# ===========================================================================
class CalibrateTab(ttk.Frame):
    """The Calibrate pane: a faithful lift of ``App``'s Calibrate tab.

    Usage (the test-bench imports this class and adds it as a tab)::

        tab = CalibrateTab(parent, planes_dir=None, on_plane_saved=None)
        tab.load_clip(Path("data/clips/18-07.mp4"))
        tab.refresh_planes()
    """

    def __init__(self, parent, *, planes_dir=None, on_plane_saved=None) -> None:
        super().__init__(parent, padding=8)
        # NB: ``ttk.Frame`` already owns ``self.tk`` (the Tcl interpreter) and ``Misc`` methods
        # such as ``winfo_toplevel`` call it.  The parent ``App`` was a plain object, so its
        # ``self.tk`` alias was free; here the module aliases are ``_tk``/``_ttk`` to avoid the
        # collision.
        self._tk = tk
        self._ttk = ttk
        self.root = self.winfo_toplevel()

        # --- overridable seams (a headless test can replace these) ----------
        from tkinter import messagebox as _messagebox
        self.warn: Callable[..., Any] = (
            (lambda *a, **k: None) if "pytest" in sys.modules else _messagebox.showwarning)
        #: The reset prompt for *New Plane*.  ``None`` means "ask with a real dialog"; a test (or an
        #: unattended run) may set a callable returning True/False instead of blocking on a modal.
        self.confirm_reset: Callable[[], bool] | None = None

        # --- state (initial values are the parent App.__init__'s) ----------
        self.clip: Path | None = None
        self.info = None
        #: A finished pipeline run (Review's), read only for the dock-side hint / accepted frames.
        self.outcome = None
        #: The single aligned plane driving the 3-D route (one sequence at a time).
        self.plane_session: plane_mod.PlaneSession | None = None
        #: Last waterline detection, kept so its fitted lines can be drawn over the photo.
        self.seed_edges: waterline_mod.WaterlineEdges | None = None
        #: Waterline detections cached per clip sha256 — "detect once per clip".
        self._seed_cache: dict[str, tuple[Any, Any]] = {}
        #: Canvas-px positions of the four per-edge lock icons, rebuilt on every overlay draw.
        self._lock_hotspots: dict[str, tuple[float, float]] = {}
        #: The clip's sha256, computed on load; keys the waterline seed cache (NOT the plane).
        self.clip_sha: str = ""
        #: The selected plane's identity (its prefix / setup id).
        self._plane_setup_id: str | None = None
        #: Where saved planes live.
        self.plane_dir = Path(planes_dir) if planes_dir is not None else Path(PLANES_DIR)
        #: Where the remembered active-plane pointer lives.
        self.active_plane_path: Path = Path(ACTIVE_PLANE_PATH)
        #: The plane remembered from a previous session, if it still exists.
        self._active_plane_path: Path | None = self._read_active_plane_path()
        #: An optional MEASURED lens (a checkerboard still).  None = the focal prior.
        self.pool_lens = None
        #: The frame the Calibrate pane's photo shows.
        self.pool_frame = None
        #: What the Calibrate pane's photo currently shows — (clip path, id(plane), padded shape).
        self._plane_shown_key: tuple[object, ...] | None = None
        #: The built pool document (``None`` until Plane Confirmed / a saved plane is adopted).
        self.pool_calib: dict | None = None
        self.pool_calib_path: Path | None = None
        self.pool_result: dict | None = None
        #: Hook: called after a successful save as ``on_plane_saved(setup_id, path)``.
        self.on_plane_saved = on_plane_saved

        # --- worker-thread queue (the waterline detector) -------------------
        self.cancel_event = threading.Event()
        self.worker: threading.Thread | None = None
        self.q: queue.Queue = queue.Queue()

        # --- build the pane ------------------------------------------------
        self._build_widgets()
        self.root.after(80, self._drain)
        self._refresh_plane_choices()

    # -- ui construction ----------------------------------------------------
    def _build_widgets(self) -> None:
        """The tab's own shell: the calibrate content, plus a status strip and session log.

        The parent puts ``status`` and ``log_text`` on the root window (shared across tabs);
        here they live on the tab itself, and ``log`` writes to them exactly as the parent does.
        """
        # status strip (bottom) and session log (bottom) come first so the calibrate content
        # (packed top, expand) fills the rest.
        self.status = ttk.Label(self, text="ready", anchor="w", relief="sunken")
        self.status.pack(side="bottom", fill="x")
        log_frame = ttk.LabelFrame(self, text="session log", padding=4)
        log_frame.pack(side="bottom", fill="x", pady=(0, 4))
        self.log_text = tk.Text(log_frame, height=7, wrap="none")
        self.log_text.pack(fill="x")

        self._build_calibrate()

    def _zoomable(self, parent, w: int, h: int):
        """Canvas + scrollbars + zoom toolbar + image-px cursor readout (copied from the parent)."""
        tk, ttk = self._tk, self._ttk
        holder = ttk.Frame(parent)
        holder.columnconfigure(0, weight=1)
        holder.rowconfigure(0, weight=1)
        canvas = tk.Canvas(holder, width=w, height=h, bg="#202020",
                           highlightthickness=1, highlightbackground="#444")
        hbar = ttk.Scrollbar(holder, orient="horizontal", command=canvas.xview)
        vbar = ttk.Scrollbar(holder, orient="vertical", command=canvas.yview)
        canvas.configure(xscrollcommand=hbar.set, yscrollcommand=vbar.set)
        def on_scroll(*args):
            canvas.xview(*args)
            view.refresh()
        def on_vscroll(*args):
            canvas.yview(*args)
            view.refresh()
        hbar.configure(command=on_scroll)
        vbar.configure(command=on_vscroll)
        canvas.bind("<Configure>", lambda e: view.refresh(), add="+")
        canvas.bind("<B1-Motion>", lambda e: view.refresh(), add="+")
        canvas.grid(row=0, column=0, sticky="nsew")
        vbar.grid(row=0, column=1, sticky="ns")
        hbar.grid(row=1, column=0, sticky="ew")

        view = FrameView(canvas, w, h)
        view.bind_zoom()

        bar = ttk.Frame(holder)
        bar.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(2, 0))
        ttk.Button(bar, text="Fit", width=4, command=view.fit).pack(side="left", padx=1)
        for z in ZOOM_STEPS:
            ttk.Button(bar, text=f"{z:g}\u00d7", width=4,
                       command=lambda zz=z: view.set_zoom(zz)).pack(side="left", padx=1)
        zoom_var = tk.StringVar(value="zoom \u2014")
        ttk.Label(bar, textvariable=zoom_var).pack(side="left", padx=8)
        cursor = tk.StringVar(value="")
        ttk.Label(bar, textvariable=cursor, width=26).pack(side="left", padx=8)
        view.cursor_var = cursor

        def changed(v: FrameView) -> None:
            zz = v.zoom or 1.0
            zoom_var.set(f"zoom {zz * 100:.0f}%   1 px = {1.0 / zz:.2f} image px")

        view.on_change = changed
        ttk.Label(bar, text="wheel = zoom at cursor \u00b7 middle-drag = pan").pack(
            side="left", padx=8)
        return view, holder

    def _build_calibrate(self) -> None:
        tk, ttk = self._tk, self._ttk
        f = self
        self.calib_frame_idx = tk.IntVar(value=CALIB_FRAME_DEFAULT)
        self.grid_var = tk.BooleanVar(value=True)

        # The one plane for the whole competition.
        pick = ttk.Frame(f)
        pick.pack(anchor="w")
        ttk.Label(pick, text="plane:").pack(side="left")
        self.plane_pick_var = tk.StringVar(value="")
        self.plane_pick = ttk.Combobox(pick, textvariable=self.plane_pick_var, width=34,
                                       state="readonly")
        self.plane_pick.pack(side="left", padx=4)
        self.plane_pick.bind("<<ComboboxSelected>>", self.on_pick_plane)
        self.save_plane_btn = ttk.Button(pick, text="Save New Plane", command=self.on_save_pool)
        self.pool_status = ttk.Label(pick, text="no plane selected")
        self.pool_status.pack(side="left", padx=(12, 0))

        # --- the 3-D route: one plane on the water surface (2026-10-19) ------
        plane_row = ttk.Frame(f)
        plane_row.pack(anchor="w", pady=(8, 0))
        self.new_plane_btn = self._blue_button(plane_row, "New Plane", self.on_new_plane)
        self.new_plane_btn.pack(side="left")
        self.detect_btn = ttk.Button(plane_row, text="Re-fit Waterline",
                                     command=self.on_refit_waterline)
        self.detect_btn.pack(side="left", padx=(8, 0))
        # Snapping is a per-EDGE property now (2026-10-27): every edge carries a lock icon on the
        # canvas, all four are locked to begin with, and the master switch behind them stays on.
        self.snap_on_var = tk.BooleanVar(value=True)
        self.snap_btn = ttk.Button(plane_row, text="TOGGLE: Snap to Waterline (ON)",
                                   command=self.on_toggle_snap)
        self.snap_side_vars: dict[str, Any] = {}
        # The two LONG edges start LOCKED; the two SHORT ends start RELEASED (order 2026-10-27).
        for side, label in SNAP_SIDE_LABELS:
            self.snap_side_vars[side] = tk.BooleanVar(value=(side in ("back", "front")))
        self.snap_far_var = self.snap_side_vars["back"]
        ttk.Button(plane_row, text="Flip Plane (left-right)",
                   command=self.on_plane_flip_x).pack(side="left", padx=(10, 0))
        self.complete_btn = self._blue_button(plane_row, "Plane Confirmed", self.on_build_plane,
                                              primary_big=True)
        self.complete_btn.pack(side="left", padx=(10, 0))
        self.plane_label = ttk.Label(f, text="no plane yet")

        dim_row = ttk.Frame(f)
        dim_row.pack(anchor="w", pady=(6, 0))
        ttk.Label(dim_row, text="• Anchor (near dock):").pack(side="left")
        self.plane_near_var = tk.StringVar(value=f"{SEED_ANCHOR_NEAR_FT:g}")
        ttk.Entry(dim_row, textvariable=self.plane_near_var, width=6).pack(side="left", padx=2)
        ttk.Label(dim_row, text="• Anchor (opposite dock):").pack(side="left", padx=(8, 2))
        self.plane_len_var = tk.StringVar(value=f"{SEED_ANCHOR_FAR_FT:g}")
        ttk.Entry(dim_row, textvariable=self.plane_len_var, width=6).pack(side="left", padx=2)
        ttk.Label(dim_row, text="width (ft):").pack(side="left", padx=(8, 2))
        self.plane_wid_var = tk.StringVar(value=f"{plane_mod.DEFAULT_WIDTH_FT:g}")
        ttk.Entry(dim_row, textvariable=self.plane_wid_var, width=6).pack(side="left")
        ttk.Button(dim_row, text="Apply size", command=self.on_plane_dims).pack(side="left", padx=4)
        self.dock_side_var = tk.StringVar(value="auto")

        self.calib_view, _cal_holder = self._zoomable(f, 900, 430)
        _cal_holder.pack(fill="both", expand=True)
        self.calib_view.overlay = self._draw_plane_overlay
        self.calib_canvas = self.calib_view.canvas
        self.calib_canvas.bind("<Button-1>", self.on_calib_click)
        self.calib_canvas.bind("<B1-Motion>", self.on_calib_drag)
        self.calib_canvas.bind("<ButtonRelease-1>", self.on_calib_release)
        for key, dx, dy in (("<Left>", -1, 0), ("<Right>", 1, 0), ("<Up>", 0, -1), ("<Down>", 0, 1)):
            self.calib_canvas.bind(key, lambda e, a=dx, b=dy: self.on_arrow_key(e, a, b))
        self._build_nudge_pad(f)

    # -- fine placement: the arrow pad ---------------------------------------
    def _blue_button(self, parent, text: str, command, *, primary_big: bool = False):
        if primary_big:
            return self._tk.Button(parent, text=text, command=command, bg=PRIMARY_BG, fg="white",
                                  activebackground=PRIMARY_BG_ACTIVE, activeforeground="white",
                                  relief="flat", bd=0, padx=30, pady=12, width=18,
                                  font=("TkDefaultFont", 11, "bold"), cursor="hand2",
                                  highlightthickness=0)
        return self._tk.Button(parent, text=text, command=command, bg=PRIMARY_BG, fg="white",
                              activebackground=PRIMARY_BG_ACTIVE, activeforeground="white",
                              relief="flat", bd=0, padx=12, pady=3, cursor="hand2",
                              highlightthickness=0)

    def _build_nudge_pad(self, parent) -> None:
        ttk = self._ttk
        pad = ttk.Frame(parent)
        pad.pack(anchor="w", pady=(4, 0))
        ttk.Label(pad, text="fine placement:", font=("", 9, "bold")).pack(side="left", padx=(0, 6))
        cross = ttk.Frame(pad)
        cross.pack(side="left")
        self._nudge_button(cross, "\u25b2", 0, -1).grid(row=0, column=1, padx=1)
        self._nudge_button(cross, "\u25c0", -1, 0).grid(row=1, column=0, padx=1)
        self._nudge_button(cross, "\u25bc", 0, 1).grid(row=1, column=1, padx=1)
        self._nudge_button(cross, "\u25b6", 1, 0).grid(row=1, column=2, padx=1)
        self.nudge_var = self._tk.StringVar(value="no corner selected — click one on the canvas")
        ttk.Label(pad, textvariable=self.nudge_var).pack(side="left", padx=10)
        ttk.Label(pad, text="1 image px per click · Shift 10 · Ctrl 0.1 · arrow keys work too",
                  foreground="#777").pack(side="left", padx=6)

    def _nudge_button(self, parent, text: str, dx: int, dy: int):
        b = self._ttk.Button(parent, text=text, width=2)
        b.bind("<Button-1>", lambda e, a=dx, c=dy: self.on_nudge_handle(a, c, e))
        return b

    @staticmethod
    def _nudge_step(state: int) -> float:
        mods = int(state or 0)
        if mods & 0x1:                       # Shift
            return 10.0
        if mods & 0x4 or mods & 0x20000:     # Control, or Alt on Windows
            return 0.1
        return 1.0

    def on_nudge_handle(self, dx: float, dy: float, event=None) -> None:
        s = self.plane_session
        if s is None:
            self.log("start the plane first")
            return
        if s.selected is None:
            placed = [t for t in plane_mod.CORNER_TAGS if s.corners.get(t) is not None]
            if not placed:
                self.log("no corner to move — click a corner on the canvas first")
                return
            s.select(placed[0])
        step = self._nudge_step(getattr(event, "state", 0) if event is not None else 0)
        if not s.nudge(dx * step, dy * step):
            self.log(f"nothing movable at {s.selected!r}")
            return
        self._maybe_snap_edges()
        self._redraw_plane()
        self._sync_plane_widgets()

    def on_arrow_key(self, event, dx: int, dy: int):
        """Arrow keys nudge the selected handle.

        The parent gates this on "is the Calibrate tab selected" (``self.nb``); this standalone
        tab has no notebook, so the pane is always the active surface and the keys always nudge.
        """
        self.on_nudge_handle(dx, dy, event)
        return "break"

    def _sync_nudge_label(self) -> None:
        var = getattr(self, "nudge_var", None)
        if var is None:
            return
        s = self.plane_session
        if s is None or s.selected is None:
            var.set("no corner selected — click one on the canvas")
            return
        p = (s.corners.get(s.selected) if s.selected in plane_mod.CORNER_TAGS
             else s.side_midpoint(s.selected))
        where = "" if p is None else f"  ({p[0]:.2f}, {p[1]:.2f}) px"
        var.set(f"moving {s.selected}{where}")

    # -- logging / status ---------------------------------------------------
    def log(self, msg: str) -> None:
        self.log_text.insert("end", msg + "\n")
        self.log_text.see("end")
        self.status.configure(text=msg)

    # -- the plane editor ----------------------------------------------------
    def _draw_lock(self, c, x: float, y: float, locked: bool) -> None:
        col = "#ffd166" if locked else "#8a8f98"
        c.create_rectangle(x - 5, y - 1, x + 5, y + 7, outline=col, width=1,
                           fill=col if locked else "", tags="overlay")
        if locked:
            c.create_arc(x - 4, y - 8, x + 4, y + 1, start=0, extent=180, style="arc",
                         outline=col, width=1, tags="overlay")
        else:
            c.create_arc(x - 1, y - 9, x + 7, y + 1, start=0, extent=180, style="arc",
                         outline=col, width=1, tags="overlay")

    def _lock_at(self, event) -> str | None:
        cx = float(self.calib_canvas.canvasx(event.x))
        cy = float(self.calib_canvas.canvasy(event.y))
        for name, (lx, ly) in self._lock_hotspots.items():
            if abs(cx - lx) <= 9.0 and abs(cy - ly) <= 9.0:
                return name
        return None

    def on_toggle_side_snap(self, side: str) -> None:
        var = self.snap_side_vars.get(side)
        if var is None:
            return
        var.set(not bool(var.get()))
        self.snap_on_var.set(True)
        label = dict(SNAP_SIDE_LABELS).get(side, side)
        if var.get():
            moved = self._maybe_snap_edges()
            self.log(f"{label} edge LOCKED to its detected waterline"
                     + (" (moved back onto it)" if moved else ""))
        else:
            self.log(f"{label} edge released — drag or pull it freely")
        self._redraw_plane()

    def on_calib_click(self, event) -> None:
        lock = self._lock_at(event)
        if lock is not None:
            self.on_toggle_side_snap(lock)
            return
        if self.plane_session is None:
            return
        x, y = self.calib_view.to_image(event.x, event.y)
        if self.plane_session.click(x, y, zoom=self.calib_view.zoom):
            self.calib_canvas.focus_set()
            self._redraw_plane()
            self._sync_plane_widgets()

    def on_calib_drag(self, event) -> None:
        if self.plane_session is None or self.plane_session.dragging is None:
            return
        x, y = self.calib_view.to_image(event.x, event.y)
        self.plane_session.drag_to(x, y)
        self._maybe_snap_edges()
        self._redraw_plane()

    def on_calib_release(self, _event) -> None:
        if self.plane_session is None:
            return
        if self.plane_session.release() is not None:
            self._maybe_snap_edges()
            self._sync_plane_widgets()
            self._redraw_plane()

    def on_new_plane(self, seed: bool = True) -> None:
        """New Plane: make/reset the quad, detect the waterlines, snap to them, arm the toggle."""
        if not self.clip:
            self.log("load a clip first")
            return
        if self._plane_in_progress() and not self._confirm_reset_plane():
            self.log("new plane cancelled — the plane on screen is unchanged")
            return
        idx = int(self.calib_frame_idx.get())
        frame = read_frame(self.clip, idx)
        if frame is None:
            self.log(f"could not read frame {idx}")
            return
        _far, _near, Wd = self._plane_dims_or_default()
        self.pool_frame = frame
        self.plane_session = plane_mod.PlaneSession(
            frame_shape=(int(frame.shape[0]), int(frame.shape[1])),
            length_ft=float(plane_mod.METRIC_LENGTH_FT), width_ft=Wd, dock_offset_ft=0.0)
        self.plane_session.fit_to_frame()
        self.pool_calib = None
        self.pool_calib_path = None
        self._plane_setup_id = None
        self.pool_status.configure(text="new plane — not saved yet")
        self.snap_on_var.set(True)
        self._sync_snap_btn()
        self._sync_plane_widgets()
        self._redraw_plane(reset_view=True)
        if not seed:
            self.log(f"pool plane started on frame {idx}: drag the corners onto the pool's corners "
                     f"(X = 0 is the dock lip); the canvas extends past the frame")
            return
        cached = self._seed_cache.get(self.clip_sha or "")
        if cached is not None:
            edges, res = cached
            self.log("plane opened from the cached waterline fit for this clip")
            self._apply_seed(edges, res)
            return
        self.log(f"pool plane started on frame {idx}: fitting the detected waterlines — "
                 f"drag from there; the canvas extends past the frame")
        self._start_waterline_detection()

    def _plane_in_progress(self) -> bool:
        s = self.plane_session
        if s is None:
            return False
        return bool(s.placed or any(p is not None for p in s.corners.values()))

    def _confirm_reset_plane(self) -> bool:
        ask = getattr(self, "confirm_reset", None)
        if ask is not None:
            return bool(ask())
        from tkinter import messagebox
        return bool(messagebox.askyesno(
            "New Plane",
            "A plane is already on the canvas.\n\nReset it and fit a new one to the detected "
            "waterlines?",
            parent=self.root, icon="warning", default="no"))

    def on_toggle_snap(self) -> None:
        on = not bool(self.snap_on_var.get())
        self.snap_on_var.set(on)
        self._sync_snap_btn()
        if not on:
            self.log("snap to waterline OFF — corners and edges move freely")
            return
        moved = self._maybe_snap_edges()
        if self.plane_session is not None:
            self._sync_plane_widgets()
            self._redraw_plane()
        held = ", ".join(label for side, label in SNAP_SIDE_LABELS
                         if self.snap_side_vars[side].get())
        self.log(f"snap to waterline ON (holding {held or 'nothing'}) — "
                 + ("edges slid onto the detected lines" if moved
                    else "no detected line has an edge to hold yet"))

    def _sync_snap_btn(self) -> None:
        on = bool(self.snap_on_var.get())
        self.snap_btn.configure(text=f"TOGGLE: Snap to Waterline ({'ON' if on else 'OFF'})")

    def on_make_parallelogram(self) -> None:
        s = self.plane_session
        if s is None or not s.placed:
            self.log("place the plane's four corners first (New Plane)")
            return
        s.make_parallelogram()
        moved = self._maybe_snap_edges()
        self._sync_plane_widgets()
        self._redraw_plane()
        self.log("plane made a parallelogram — drag from here"
                 + (" (edges held on the detected waterlines)" if moved else ""))

    def _snap_lines(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        edges = self.seed_edges
        s = self.plane_session
        if edges is None:
            return {}
        out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for side, name in (("back", "far"), ("front", "near")):
            line = seed_mod.line_of(getattr(edges, name, None))
            if line is not None:
                if name == "near":
                    line = (line[0] + np.array([0.0, float(seed_mod.NEAR_WALL_DROP_PX)]),
                            line[1])
                out[side] = line

        try:
            H = s.homography()
        except Exception:
            H = None
        if H is None:
            return out
        w_ft = float(s.width_ft)
        for name, x_ft in (("left", float(s.dock_offset_ft)),
                           ("right", float(s.dock_offset_ft) + float(s.length_ft))):
            p0 = np.asarray(plane_mod.project(H, x_ft, 0.0), float)
            p1 = np.asarray(plane_mod.project(H, x_ft, w_ft), float)
            if np.all(np.isfinite(p0)) and np.all(np.isfinite(p1)):
                out[name] = (p0, p1 - p0)
        return out

    @staticmethod
    def _x0_end_is_image_left(s) -> bool:
        if s is None or not s.placed:
            return True
        lx = 0.5 * (float(s.corners["00"][0]) + float(s.corners["01"][0]))
        rx = 0.5 * (float(s.corners["10"][0]) + float(s.corners["11"][0]))
        return lx <= rx

    def _maybe_snap_edges(self) -> bool:
        s = self.plane_session
        if s is None or not s.placed or not self.snap_on_var.get():
            return False
        lines = self._snap_lines()
        if not lines:
            return False
        moved = False
        for _pass in range(2):
            for side, _label in SNAP_SIDE_LABELS:
                if not self.snap_side_vars[side].get() or side not in lines:
                    continue
                point, direction = lines[side]
                moved = s.snap_side_to_line(side, point, direction) or moved
        return moved

    def on_plane_fit(self) -> None:
        if self.plane_session is None:
            self.log("start the plane first")
            return
        self.plane_session.fit_to_frame()
        self._sync_plane_widgets()
        self._redraw_plane()

    def on_plane_flip_x(self) -> None:
        s = self.plane_session
        if s is None or not s.placed:
            self.log("place the plane's four corners first, then flip")
            return
        before = {t: s.corners[t] for t in plane_mod.CORNER_TAGS}
        s.flip_x()
        same = all(s.corners[t] == before[t] for t in plane_mod.CORNER_TAGS)
        self._sync_plane_widgets()
        self._redraw_plane()
        self.log(f"plane flipped left-right: X=0 is now the end at x={s.corners['00'][0]:.0f} px; "
                 f"the quad did not move{' (verified: every corner kept its pixel)' if same else ''}, "
                 f"only the number scale changed")

    # -- automatic placement from the detected waterlines (PLAN §17) ----------
    def _accepted_frames(self) -> list[int]:
        det = getattr(self.outcome, "detect", None) if self.outcome is not None else None
        if det is None:
            return []
        try:
            return list(det.accepted_frames())
        except Exception:                                            # noqa: BLE001
            return []

    def _dock_hint_x(self) -> float | None:
        det = getattr(self.outcome, "detect", None) if self.outcome is not None else None
        if det is None:
            return None
        xs: list[float] = []
        for r in getattr(det, "records", []) or []:
            if r.get("accepted") and r.get("tailset"):
                xs.append(float(r["tailset"][0]))
                if len(xs) >= 3:
                    break
        return float(np.median(xs)) if xs else None

    def on_refit_waterline(self) -> None:
        if not self.clip:
            self.log("load a clip first")
            return
        if self.plane_session is None:
            self.on_new_plane()
            return
        cached = self._seed_cache.get(self.clip_sha or "")
        session = self.plane_session
        if cached is not None:
            edges, res = cached
            if res.corners:
                block = seed_mod.build_block(res, edges, upstream=waterline_mod.UPSTREAM,
                                             file_sha256=waterline_mod.vendored_sha256())
                session.apply_seed(res.corners, block)
                session.ensure_far_edge_short()
                self.seed_edges = edges
                self.log(f"re-fit: the quad is back on the waterlines detected for this clip "
                         f"({res.level}) — corners "
                         + ", ".join(f"{k} ({v[0]:.0f},{v[1]:.0f})"
                                     for k, v in res.corners.items()))
                self._sync_plane_widgets()
                self._redraw_plane()
                return
        self.log("no detected fit cached for this clip yet — fitting it now")
        self._start_waterline_detection()

    def on_detect_waterline(self) -> None:
        if not self.clip:
            self.log("load a clip first")
            return
        if self.plane_session is None:
            self.on_new_plane()
            if self.plane_session is None:
                return
        self._start_waterline_detection()

    def _start_waterline_detection(self) -> None:
        s = self.plane_session
        if s is None or not self.clip:
            return
        L, Wd = float(plane_mod.METRIC_LENGTH_FT), float(s.width_ft)
        shape = (int(s.frame_shape[0]), int(s.frame_shape[1]))
        snapshot = plane_mod.PlaneSession.from_doc({"plane": s.as_dict()})
        roi = self._seed_roi(snapshot)
        clip = Path(self.clip)
        window = (0, SEED_FRAME_COUNT)
        accepted = self._accepted_frames()
        forced = str(self.dock_side_var.get())
        if forced in ("left", "right"):
            hint = 0.0 if forced == "left" else float(shape[1]) - 1.0
            self.log(f"dock side forced to the {forced} by the pane control — X = 0 will be the "
                     f"{forced} short-end waterline")
        else:
            hint = self._dock_hint_x()
            if hint is None:
                self.log("no detections yet, so the dock side is unknown — set the dock-side control "
                         "or run the clip first; X = 0 will be whichever short end the detector picks")
        self.detect_btn.configure(state="disabled")
        self.log(f"detecting waterlines on {clip.name}: frames {window[0]}-{window[1] - 1} "
                 f"(the clip's opening {SEED_FRAME_COUNT}) — a second or two")

        def work() -> None:
            try:
                edges = waterline_mod.detect_edges(clip, window=window, roi_poly=roi)
                if not (edges.far.found and edges.near.found):
                    found = ", ".join(edges.found) or "none"
                    self.q.put(("log", f"the first {SEED_FRAME_COUNT} frames gave only: {found} — "
                                       f"widening the waterline window to the clip's calm stretch"))
                    edges = waterline_mod.detect_edges(clip, accepted_frames=accepted, roi_poly=roi)
                res = seed_mod.seed_from_edges(edges, length_ft=L, width_ft=Wd,
                                               frame_shape=shape, current=snapshot,
                                               dock_side=forced if forced in ("left", "right") else None,
                                               dock_hint_x=hint)
                self.q.put(("seed", (edges, res)))
            except Exception as exc:                                 # noqa: BLE001
                self.q.put(("seed_error", f"{type(exc).__name__}: {exc}"))
            finally:
                self.q.put(("seed_done", None))

        threading.Thread(target=work, daemon=True).start()

    @staticmethod
    def _seed_roi(session):
        if str(os.environ.get("DD_SEED_ROI", "0")).strip() == "0":
            return None
        if session is None or not session.placed:
            return None
        pts = np.array([session.corners[t] for t in plane_mod.CORNER_TAGS], dtype=float)
        centre = pts.mean(axis=0)
        grown = centre + (pts - centre) * 1.3
        h, w = int(session.frame_shape[0]), int(session.frame_shape[1])
        grown[:, 0] = np.clip(grown[:, 0], 0.0, max(0.0, w - 1.0))
        grown[:, 1] = np.clip(grown[:, 1], 0.0, max(0.0, h - 1.0))
        return grown

    def _apply_seed(self, edges, res) -> None:
        self.seed_edges = edges
        s = self.plane_session
        gates = " ".join(("warn" if (v.get("pass") and v.get("warn")) else
                          "ok" if v.get("pass") else "FAIL") + f" {k}"
                         for k, v in res.gates.items())
        self.log(f"waterlines: {edges.level} (far {'ok' if edges.far.found else 'no'}, "
                 f"near {'ok' if edges.near.found else 'no'}, "
                 f"dock {'ok' if edges.dock.found else 'no'}, side {edges.dock_side}) · "
                 f"{edges.method} fisher {edges.fisher:.2f} · short-edge method "
                 f"{getattr(edges, 'short_method', 'current')} · frames "
                 f"{edges.window[0]}-{edges.window[1]} · {gates}")
        for name in ("left", "right"):
            e = getattr(edges, name, None)
            if e is not None:
                self.log(f"  short end {name}: {e.status} — {e.reason or 'fitted'}")
        for note in res.notes:
            self.log(f"  seed: {note}")
        for w in edges.warnings:
            self.log(f"  waterline warning: {w}")
        if s is None:
            return
        if res.refused or res.corners is None:
            self.log(f"seed REFUSED — {res.reason}")
            self.status.configure(
                text=f"seed refused — the plane was NOT fitted ({res.reason.split(':')[0]})")
            moved = self._maybe_snap_edges()
            self.log("the quad was left as it was"
                     + (" but its checked edges were slid onto the detected lines" if moved
                        else " and no edge could be snapped")
                     + "; the detected lines stay on the canvas for reference")
            self._sync_plane_widgets()
            self._redraw_plane()
            return
        block = seed_mod.build_block(res, edges, upstream=waterline_mod.UPSTREAM,
                                     file_sha256=waterline_mod.vendored_sha256())
        s.apply_seed(res.corners, block)
        swapped = s.ensure_far_edge_short()
        self.plane_near_var.set(f"{s.dock_offset_ft:g}")
        self.plane_len_var.set(f"{s.dock_offset_ft + s.length_ft:g}")
        self.log(f"plane placed over the full pool: X 0 → {s.dock_offset_ft + s.length_ft:g} ft")
        if self.clip_sha:
            self._seed_cache[self.clip_sha] = (edges, res)
        self.pool_calib = None
        self.pool_calib_path = None
        self._maybe_snap_edges()
        self._place_default_anchors()
        self._sync_plane_widgets()
        self._redraw_plane()
        if swapped:
            self.log("far edge is now the short one (the seed had the pool the other way round)")
        if np.isfinite(res.px_per_ft_x) and np.isfinite(res.px_per_ft_y):
            scale_txt = (f"{res.px_per_ft_x:.1f} px/ft along X at the dock edge "
                         f"({res.px_per_ft_y:.1f} across)")
        else:
            scale_txt = "extent taken from the four detected lines — no px/ft involved"
        self.log(f"seed {res.level}: anchor {res.anchor_ft:g} ft · {scale_txt} · "
                 f"drag a corner or a side to correct it, then Plane Confirmed")

    def on_plane_dims(self) -> None:
        if self.plane_session is None:
            self.log("start the plane first")
            return
        try:
            far = float(self.plane_len_var.get())
            near = float(self.plane_near_var.get() or 0.0)
            Wd = float(self.plane_wid_var.get())
            if Wd <= 0:
                raise ValueError("width must be positive")
            if near < 0:
                raise ValueError("the near-dock anchor cannot be before the dock (0 ft)")
            if far <= near:
                raise ValueError(f"the anchor opposite the dock ({far:g} ft) must be beyond the "
                                 f"near-dock anchor ({near:g} ft)")
            if far > plane_mod.METRIC_LENGTH_FT:
                raise ValueError(f"the +x axis only runs to {plane_mod.METRIC_LENGTH_FT:g} ft — "
                                 f"the far edge cannot be anchored beyond it")
        except ValueError as exc:
            self.log(f"plane size rejected: {exc}")
            return
        self.plane_session.width_ft = Wd
        try:
            info = self.plane_session.apply_anchors(near, far)
        except ValueError as exc:
            self.log(f"anchors not applied: {exc}")
            return
        self.plane_session.notes.append(
            f"anchors applied: {near:g} -> {far:g} ft, width {Wd:g} ft · the quad's short edges "
            f"moved {info['moved_px']:.1f} px, the plane {info['plane_shift_px']:.2e} px")
        self._maybe_snap_edges()
        self._sync_plane_widgets()
        self._redraw_plane()
        self.log(f"anchors {near:g} / {far:g} ft: the quad's short edges moved {info['moved_px']:.1f} px; "
                 f"the plane moved {info['plane_shift_px']:.1e} px (it does not move)")
        if near > 0:
            self.log(f"the quad's dock-side edge now sits at X = {near:g} ft on the fixed "
                     f"0 → {plane_mod.METRIC_LENGTH_FT:g} ft axis; the grid is unchanged")

    def _plane_dims_or_default(self) -> tuple[float, float, float]:
        def num(var, fallback: float, name: str) -> float:
            try:
                return float(var.get())
            except (TypeError, ValueError):
                self.log(f"plane {name} {var.get()!r} is not a number — using {fallback:g} ft")
                return fallback
        far = num(self.plane_len_var, SEED_ANCHOR_FAR_FT, "anchor opposite the dock")
        near = num(self.plane_near_var, SEED_ANCHOR_NEAR_FT, "anchor near the dock")
        Wd = num(self.plane_wid_var, float(plane_mod.DEFAULT_WIDTH_FT), "width")
        if far <= near:
            near = 0.0
        return far, near, Wd

    def _sync_plane_widgets(self) -> None:
        s = self.plane_session
        if s is None:
            self.plane_label.configure(text="no plane yet")
            self._sync_nudge_label()
            return
        n = sum(1 for v in s.corners.values() if v is not None)
        seed_tag = ""
        if s.seed:
            lvl = (s.seed.get("seed") or {}).get("level") or "?"
            seed_tag = f" · seed {lvl} ({s.seed.get('placement', 'auto')})"
        self.plane_label.configure(
            text=(f"anchors {s.dock_offset_ft:g} \u2192 "
                  f"{s.dock_offset_ft + s.length_ft:g} ft · axis 0\u2013"
                  f"{plane_mod.METRIC_LENGTH_FT:g} ft · {s.width_ft:g} ft wide · {n}/4 corners"
                  f"{seed_tag}")
                 + ("" if s.placed else " — click the missing ones"))
        try:
            self.plane_near_var.set(f"{s.dock_offset_ft:g}")
            self.plane_len_var.set(f"{s.dock_offset_ft + s.length_ft:g}")
            self.plane_wid_var.set(f"{s.width_ft:g}")
        except Exception:                                            # noqa: BLE001
            pass
        self._sync_plane_prompt()
        self._sync_nudge_label()

    def _redraw_plane(self, reset_view: bool = False) -> None:
        s = self.plane_session
        if s is None or self.pool_frame is None:
            return
        base = plane_mod.padded_canvas(self.pool_frame, s.pad)
        fh, fw = self.pool_frame.shape[:2]
        self.calib_view.content_box = (float(s.pad[0]), float(s.pad[1]),
                                       float(s.pad[0]) + float(fw),
                                       float(s.pad[1]) + float(fh))
        # The clip is part of the shown key: a *different* clip with the *same* frame size
        # must still re-render the photo (only id(plane)+shape used to change here, so a
        # same-sized clip left the previous clip's frame on screen).
        key = (str(self.clip) if self.clip is not None else "", id(s), base.shape)
        if reset_view or key != self._plane_shown_key:
            self._plane_shown_key = key
            self.calib_view.show(base, reset_view=reset_view)
        else:
            self.calib_view.redraw_overlay()
        self._sync_plane_prompt()

    def _draw_plane_overlay(self, view) -> None:
        c = view.canvas
        c.delete("overlay")
        self._lock_hotspots = {}
        s = self.plane_session
        if s is None:
            return
        z = float(view.zoom or 1.0)
        ox, oy = float(s.pad[0]), float(s.pad[1])

        def P(p) -> tuple[float, float]:
            return ((float(p[0]) + ox) * z, (float(p[1]) + oy) * z)

        if self.seed_edges is not None:
            for name, colour in (("left", "#8affc1"), ("right", "#ffb08a"),
                                 ("far", "#ffcc44"), ("near", "#ff8ce0"), ("dock", "#7cf0ff")):
                e = getattr(self.seed_edges, name, None)
                if e is None or not getattr(e, "found", False) or e.fit is None:
                    continue
                (ax, ay), (bx, by) = e.fit["endpoints"]
                x0, y0 = P((ax, ay))
                x1, y1 = P((bx, by))
                if max(abs(x0), abs(y0), abs(x1), abs(y1)) > 1e5:
                    continue
                c.create_line(x0, y0, x1, y1, fill=colour, width=2, dash=(6, 4), tags="overlay")
                if e.points is not None and len(e.points) > 1:
                    step = max(1, len(e.points) // 40)
                    for q in e.points[::step]:
                        qx, qy = P(q)
                        c.create_rectangle(qx - 1.5, qy - 1.5, qx + 1.5, qy + 1.5,
                                           outline=colour, tags="overlay")

        if not s.placed:
            for tag in plane_mod.CORNER_TAGS:
                p = s.corners.get(tag)
                if p is not None:
                    x, y = P(p)
                    c.create_oval(x - 4, y - 4, x + 4, y + 4, outline="#00dcff", width=1,
                                  tags="overlay")
            return

        try:
            H = s.homography()
        except Exception as exc:                                      # noqa: BLE001
            c.create_text(12, 12, anchor="nw", fill="#ff8a8a", tags="overlay",
                          text=f"plane not drawable: {exc}"[:160])
            return
        Wd = float(s.width_ft)
        L = float(s.length_ft)
        off = float(s.dock_offset_ft)

        def seg(X0: float, Y0: float, X1: float, Y1: float, colour: str, width: int,
                dash: tuple[int, int] | None = None) -> None:
            a = plane_mod.project(H, X0, Y0)
            b = plane_mod.project(H, X1, Y1)
            if not (_finite2(a) and _finite2(b)):
                return
            (x0, y0), (x1, y1) = P(a), P(b)
            lim = 1e5
            if max(abs(x0), abs(y0), abs(x1), abs(y1)) > lim:
                return
            kw = {"fill": colour, "width": width, "tags": "overlay"}
            if dash:
                kw["dash"] = dash
            c.create_line(x0, y0, x1, y1, **kw)

        if bool(self.grid_var.get()):
            minor, major = plane_mod.MINOR_FT, plane_mod.MAJOR_FT
            for i in range(int(round(plane_mod.METRIC_LENGTH_FT / minor)) + 1):
                X = i * minor
                is_major = abs(X / major - round(X / major)) < 1e-9
                seg(X, 0.0, X, Wd, "#78d278" if is_major else "#44804a", 1)
                if is_major:
                    a = plane_mod.project(H, X, 0.0)
                    if _finite2(a):
                        x, y = P(a)
                        if abs(x) < 1e5 and abs(y) < 1e5:
                            c.create_text(x + 3, y - 5, text=f"{X:g}'", fill="#96f096",
                                          anchor="sw", font=("", 8), tags="overlay")
            for j in range(int(round(Wd / minor)) + 1):
                Y = j * minor
                is_major = abs(Y / major - round(Y / major)) < 1e-9
                seg(0.0, Y, plane_mod.METRIC_LENGTH_FT, Y,
                    "#63a063" if is_major else "#456b4f", 1)

        seg(0.0, 0.0, 0.0, Wd, "#3c78ff", 2)
        seg(off, 0.0, off, Wd, "#ff9f43", 1, dash=(2, 3))
        seg(off + L, 0.0, off + L, Wd, "#e6d24a", 1, dash=(4, 3))
        seg(off, 0.0, off + L, 0.0, "#00c8ff", 1)
        seg(off, Wd, off + L, Wd, "#00c8ff", 1)

        placed_pts = [p for p in s.corners.values() if p is not None]
        cx0 = sum(p[0] for p in placed_pts) / max(1, len(placed_pts))
        cy0 = sum(p[1] for p in placed_pts) / max(1, len(placed_pts))
        for tag in plane_mod.CORNER_TAGS:
            p = s.corners.get(tag)
            if p is None:
                continue
            x, y = P(p)
            c.create_oval(x - 4, y - 4, x + 4, y + 4, fill="#ffffff", outline="#3c78ff",
                          width=1, tags="overlay")
            right, below = p[0] >= cx0, p[1] >= cy0
            anchor = ("nw" if right and below else "ne" if below else "sw" if right else "se")
            c.create_text(x + (18 if right else -18), y + (20 if below else -20), text=tag,
                          fill="#dfe7ff", anchor=anchor, font=("", 7), tags="overlay")
        for name in plane_mod.SIDE_TAGS:
            m = s.side_midpoint(name)
            if m is None:
                continue
            x, y = P(m)
            if name in ("back", "front"):
                c.create_oval(x - 6, y - 6, x + 6, y + 6, fill="#00c8ff", outline="#ffffff",
                              width=1, tags="overlay")
            else:
                c.create_oval(x - 3, y - 3, x + 3, y + 3, outline="#c8c8c8", width=1,
                              tags="overlay")
            cxq, cyq = P((cx0, cy0))
            ddx, ddy = x - cxq, y - cyq
            dn = float(np.hypot(ddx, ddy)) or 1.0
            lx, ly = x + 21.0 * ddx / dn, y + 21.0 * ddy / dn
            self._draw_lock(c, lx, ly, bool(self.snap_side_vars[name].get()))
            self._lock_hotspots[name] = (lx, ly)
        for handle in {s.selected, s.dragging}:
            if handle is None:
                continue
            p = (s.corners.get(handle) if handle in plane_mod.CORNER_TAGS
                 else s.side_midpoint(handle))
            if p is None:
                continue
            x, y = P(p)
            c.create_oval(x - 9, y - 9, x + 9, y + 9,
                          outline="#ffee66" if handle == s.selected else "#ffffff",
                          width=2, tags="overlay")

    def _sync_plane_prompt(self) -> None:
        return

    def on_build_plane(self) -> None:
        s = self.plane_session
        if (s is not None and s.placed
                and abs(s.dock_offset_ft - SEED_ANCHOR_NEAR_FT) < 1e-9
                and abs(s.length_ft - (SEED_ANCHOR_FAR_FT - SEED_ANCHOR_NEAR_FT)) < 1e-9):
            try:
                s.apply_anchors(0.0, float(plane_mod.METRIC_LENGTH_FT))
            except Exception:                                        # noqa: BLE001
                pass
        doc = self._build_pool_doc()
        if doc is not None:
            from tkinter import messagebox
            choice = messagebox.askyesnocancel(
                "Plane Confirmed",
                "Save this plane to disk?\n\n"
                "Yes = save it (you name it next), No = use it without saving, Cancel = abort")
            if choice is True:
                self.on_save_pool()
            elif choice is False:
                self._show_pool_doc(doc, source="aligned plane")
                self.log("plane calibration applied (not saved)")
            else:
                return

            self._place_default_anchors()
            self._refresh_plane_choices()

    def _place_default_anchors(self) -> None:
        s = self.plane_session
        if s is None or not s.placed:
            return
        try:
            info = s.apply_anchors(SEED_ANCHOR_NEAR_FT, SEED_ANCHOR_FAR_FT)
        except ValueError as exc:
            self.log(f"default anchors not placed: {exc}")
            return
        except Exception as exc:                                     # noqa: BLE001
            self.log(f"default anchors not placed: {type(exc).__name__}: {exc}")
            return
        self._sync_plane_widgets()
        self._redraw_plane()
        self.log(f"anchors now {SEED_ANCHOR_NEAR_FT:g} / {SEED_ANCHOR_FAR_FT:g} ft: the quad's short "
                 f"edges moved {info['moved_px']:.1f} px, the plane "
                 f"{info['plane_shift_px']:.1e} px (it does not move)")

    def _plane_id(self) -> str:
        if self._plane_setup_id:
            return str(self._plane_setup_id)
        return "plane"

    def _build_pool_doc(self) -> dict | None:
        s = self.plane_session
        if s is None:
            self.log("start the plane first")
            return None
        try:
            return s.build(self._plane_id(), lens=self.pool_lens,
                           notes="water-surface plane (app_gui)")
        except Exception as exc:  # noqa: BLE001 - a bad quad must not kill the pane
            self.log(f"plane document not buildable: {type(exc).__name__}: {exc}")
            return None

    def _show_pool_doc(self, doc: dict, *, source: str, path: str | Path | None = None) -> None:
        gates = doc.get("gates") or pool_calib_mod.validate_gates(doc)
        ok = pool_calib_mod.gates_pass(gates)
        pgates = doc.get("plane_gates") or {}
        pok = all(g["pass"] for g in pgates.values()) if pgates else True
        plane_doc = bool(doc.get("plane"))
        delivery_ok = pok if plane_doc else (ok and pok)

        self.pool_calib = doc
        self.pool_calib_path = None if path is None else Path(path)
        self.pool_status.configure(text=f"{doc.get('setup_id')}")
        for name, g in list(pgates.items()) + list(gates.items()):
            if not g.get("pass"):
                tag = "" if name.startswith("P") else " (informational, not a delivery gate)"
                self.log(f"  gate {name}{tag} FAIL: {g.get('detail')}")
        if plane_doc and gates:
            self.log("  the W gates grade the reconstructed pose — informational on this route: the "
                     "answer is the plane homography, so they can fail while every distance is right")
        if not delivery_ok:
            self.log("this document fails its own gate — any distance it produces is wrong by an "
                     "unmeasured amount; fix the plane before using it")
        mp = doc.get("mapping") or {}
        if mp.get("probe"):
            worst = max(r["roundtrip_error_ft"] for r in mp["probe"])
            at0 = mp.get("px_per_ft_along_at_0")
            self.log(f"  mapping: {'n/a' if at0 is None else f'{at0:.1f} px/ft along X at 0 ft'}, "
                     f"round-trip worst {worst:.1e} ft")
        self.log(f"pool calibration from {source}")
        self._redraw_plan()

    def on_save_pool(self, prefix: str | None = None) -> None:
        if self.plane_session is None or not self.plane_session.placed:
            self.log("nothing to save — make a plane and place its four corners first")
            return
        if prefix is None:
            from tkinter import simpledialog
            prefix = simpledialog.askstring(
                "Save Plane", "Plane name prefix (e.g. event or pool):",
                initialvalue=(self._plane_setup_id or "plane"))
            if not prefix:
                self.log("plane save cancelled")
                return
        prefix = str(prefix).strip().replace(" ", "_") or "plane"
        doc = self._build_pool_doc()
        if doc is None:
            return
        from datetime import datetime
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        base = f"{prefix}_{stamp}"
        path = pool_calib_mod.setup_path(base, self.plane_dir)
        k = 2
        while path.exists() and k <= 99:   # never overwrite: a same-second clash gets a counter
            base = f"{prefix}_{stamp}_{k}"
            path = pool_calib_mod.setup_path(base, self.plane_dir)
            k += 1
        setup_id = base
        doc["setup_id"] = setup_id
        doc.pop("clips", None)
        try:
            pool_calib_mod.save(doc, path)
        except Exception as exc:  # noqa: BLE001
            self.log(f"plane save failed: {type(exc).__name__}: {exc}")
            self.pool_status.configure(text="SAVE FAILED — see the log")
            return
        self._plane_setup_id = setup_id
        self.pool_calib = doc
        self.pool_calib_path = path
        self._set_active_plane(path)
        self._refresh_plane_choices()
        if hasattr(self, "plane_pick_var"):
            self.plane_pick_var.set(setup_id)
        self.pool_status.configure(text=f"plane saved — {path.name}")
        self.log(f"plane saved to {path} (setup id {setup_id}) — reused for every clip")
        self._redraw_plan()
        if self.on_plane_saved is not None:
            try:
                self.on_plane_saved(setup_id, str(path))
            except Exception as exc:  # noqa: BLE001
                self.log(f"on_plane_saved callback failed: {type(exc).__name__}: {exc}")

    def _clip_sha(self) -> str:
        if not self.clip_sha and self.clip is not None:
            try:
                self.clip_sha = video_mod.sha256(self.clip)
            except Exception as exc:                                 # noqa: BLE001
                self.log(f"could not hash the clip ({type(exc).__name__}: {exc})")
                self.clip_sha = ""
        return self.clip_sha or ""

    # -- the remembered active plane (one per competition) ----------
    def _read_active_plane_path(self) -> Path | None:
        try:
            data = json.loads(Path(self.active_plane_path).read_text(encoding="utf-8"))
            p = Path(str(data.get("path") or ""))
            return p if p.is_file() else None
        except Exception:                                            # noqa: BLE001
            return None

    def _set_active_plane(self, path: str | Path) -> None:
        self._active_plane_path = Path(path)
        try:
            self.active_plane_path.parent.mkdir(parents=True, exist_ok=True)
            self.active_plane_path.write_text(
                json.dumps({"path": str(Path(path))}), encoding="utf-8")
        except Exception as exc:                                     # noqa: BLE001
            self.log(f"could not remember the active plane ({type(exc).__name__}: {exc})")

    def _restore_plane_for_view(self) -> None:
        if self.plane_session is not None:
            if self.clip:
                try:
                    self.pool_frame = read_frame(self.clip, int(self.calib_frame_idx.get()))
                except Exception:                                    # noqa: BLE001
                    self.pool_frame = None
                if (self.pool_frame is not None and self.plane_session.frame_shape[0]
                        and tuple(self.plane_session.frame_shape[:2])
                        != tuple(self.pool_frame.shape[:2])):
                    self.log("WARNING: the active plane was placed on a different frame size "
                             f"({tuple(self.plane_session.frame_shape[:2])} vs "
                             f"{tuple(self.pool_frame.shape[:2])}) — re-place it if the camera moved")
            self._sync_plane_widgets()
            self._redraw_plane(reset_view=False)
            self.pool_status.configure(text=f"{self._plane_setup_id or 'plane'} (active)")
            return
        if self._active_plane_path is not None and Path(self._active_plane_path).is_file():
            try:
                doc = pool_calib_mod.load(self._active_plane_path)
            except Exception as exc:                                 # noqa: BLE001
                self.log(f"could not restore the active plane ({type(exc).__name__}: {exc})")
                self.pool_status.configure(text="no plane selected")
                return
            self._apply_plane_doc(doc, self._active_plane_path)
            return
        self.pool_status.configure(text="no plane selected")

    def _apply_plane_doc(self, doc: dict, path: str | Path | None = None) -> None:
        self._show_pool_doc(doc, source=str(path or "saved plane"), path=path)
        self._plane_setup_id = doc.get("setup_id")
        if path is not None:
            self._set_active_plane(path)
        self._refresh_plane_choices()
        try:
            display = str(doc.get("setup_id") or "")
            if not display and path is not None:
                display = Path(path).stem
            self.plane_pick_var.set(display)
        except Exception:                                            # noqa: BLE001
            pass
        try:
            self.plane_session = plane_mod.PlaneSession.from_doc(doc)
        except Exception as exc:  # noqa: BLE001
            self.plane_session = None
            self.log(f"no editable plane in this document ({exc}) — it can still be used as input")
        if self.plane_session is not None:
            if self.clip:
                self.pool_frame = read_frame(self.clip, int(self.calib_frame_idx.get()))
                if self.pool_frame is not None and not self.plane_session.frame_shape[0]:
                    self.plane_session.frame_shape = self.pool_frame.shape[:2]
            self._sync_plane_widgets()
            self._redraw_plane(reset_view=True)
            self.log("plane loaded — it is reused for every clip (no per-clip matching)")

    def on_pick_plane(self, _event=None) -> None:
        sid = str(self.plane_pick_var.get())
        if not sid:
            return

        if self.plane_session is not None and self.plane_session.placed:
            from tkinter import messagebox
            response = messagebox.askyesnocancel(
                "Warning",
                "You have an unsaved plane on screen. Do you want to save it before loading a new one?")
            if response is True:
                self.on_build_plane()
            elif response is False:
                pass
            else:
                return

        found = False
        for path in pool_calib_mod.list_setups(self.plane_dir):
            try:
                doc = pool_calib_mod.load(path)
            except Exception:
                continue
            stem = path.stem
            if stem.endswith(".pool"):
                stem = stem[:-5]
            if str(doc.get("setup_id")) == sid or stem == sid or path.stem == sid:
                self._apply_plane_doc(doc, path)
                found = True
                break

        if found:
            self.log(f"plane {sid} adopted")
        else:
            self.log(f"saved plane {sid!r} is no longer on disk")

    def _refresh_plane_choices(self) -> None:
        pick = getattr(self, "plane_pick", None)
        if pick is None:
            return

        plane_paths: list[Path] = []
        try:
            for p in pool_calib_mod.list_setups(self.plane_dir):
                pp = Path(p)
                if pp.exists():
                    plane_paths.append(pp)
        except Exception:
            pass

        seen_paths: set = set()
        unique_planes: list[tuple[float, Path]] = []
        for p in plane_paths:
            try:
                resolved = p.resolve()
            except Exception:
                resolved = p
            if resolved in seen_paths:
                continue
            seen_paths.add(resolved)
            try:
                mtime = resolved.stat().st_mtime
            except Exception:
                mtime = 0.0
            unique_planes.append((mtime, resolved))

        unique_planes.sort(key=lambda x: x[0], reverse=True)

        values: list[str] = []
        for _mtime, p in unique_planes:
            sid = p.stem
            if sid.endswith(".pool"):
                sid = sid[:-5]
            try:
                doc = json.loads(p.read_text(encoding="utf-8"))
                pool_calib_mod.require_schema(doc)
                sid = str(doc.get("setup_id") or sid)
            except Exception:
                continue
            if sid and sid not in values:
                values.append(sid)

        try:
            pick.configure(values=values)
        except Exception:                                            # noqa: BLE001
            pass
        current = str(self.plane_pick_var.get() or "")
        if current and current not in values:
            self.plane_pick_var.set("")

    # -- worker-thread queue ------------------------------------------------
    def _drain(self) -> None:
        try:
            while True:
                kind, payload = self.q.get_nowait()
                if kind == "log":
                    self.log(str(payload))
                elif kind == "seed":
                    self._apply_seed(*payload)
                elif kind == "seed_error":
                    self.detect_btn.configure(state="normal")
                    self.log(f"waterline detection failed: {payload}")
                elif kind == "seed_done":
                    self.detect_btn.configure(state="normal")
        except queue.Empty:
            pass
        self.root.after(80, self._drain)
        self._refresh_plane_choices()

    # -- cross-tab hooks (no-ops here: the plan view lives on the Review tab) --
    def _redraw_plan(self) -> None:
        """No-op in the standalone tab.

        The parent's ``_redraw_plan`` draws the landing-point plan view on the Review pane's
        ``plan_canvas``, which this tab does not contain.  It is called by ``_show_pool_doc``
        and ``on_save_pool``; the plane document is still adopted / saved here, but the plan
        view is simply not drawn.
        """

    # -- public interface ----------------------------------------------------
    def load_clip(self, clip_path) -> None:
        """Probe the clip, read its frame, and start the plane (parent's ``load_clip`` + ``on_new_plane``).

        The parent's ``load_clip`` also wires the Load/Process/Review/Results tabs and kicks off the
        stability check; none of that exists here, so this keeps only probe + read + start.
        """
        clip = Path(clip_path)
        try:
            self.info = video_mod.probe(clip, count_frames=False)
            self.clip = clip
        except Exception as exc:  # noqa: BLE001
            self.log(f"could not open {clip}: {type(exc).__name__}: {exc}")
            return
        self.log(
            f"clip: {self.info.width}x{self.info.height} @ {self.info.fps:.3f} fps · "
            f"{self.info.n_frames} frames · codec={self.info.codec} · "
            f"{self.info.duration_s:.2f}s")
        for w in self.info.warnings:
            self.log(f"clip warning: {w}")
        self.pool_result = None
        self.seed_edges = None
        self.clip_sha = ""
        self._clip_sha()                     # still needed: it keys the waterline seed cache
        self._refresh_plane_choices()
        self._restore_plane_for_view()
        self._sync_plane_widgets()
        self.log(f"loaded {clip.name} ({self.info.n_frames} frames)")
        # Start the plane (the parent's stability check auto-starts a new plane when none is active;
        # without that path here, ``on_new_plane`` is the direct entry point).
        if self.plane_session is None:
            self.on_new_plane()

    def refresh_planes(self) -> None:
        """Repopulate the plane dropdown (the parent's ``_refresh_plane_choices``)."""
        self._refresh_plane_choices()
