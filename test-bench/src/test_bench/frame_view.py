"""``FrameView`` — a zoomable/panable image canvas with pixel-exact click mapping.

Adapted from the dock-diving scorer's Calibrate pane (the ``image-measurement-ui``
skill's reference).  It renders only the visible crop (never the whole frame at
high zoom), anchors wheel-zoom at the cursor, maps clicks through the scroll-aware
transform divided by zoom, and draws measurement overlays in image coordinates.

Why this exists: a fit-to-window pane quantises a 960 px frame at ~2.5 image px
per canvas px, so a ground-truth click needs zoom to be meaningful.  This is the
measurement surface the GT pane and the Compare pane share.

Needs PIL (``pillow``) + numpy + cv2 — all present in the bench venv.
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
from PIL import Image, ImageTk

MAX_ZOOM = 16.0
ZOOM_STEPS = (1.0, 2.0, 4.0, 8.0, 16.0)


class FrameView:
    def __init__(self, canvas, max_w: int = 900, max_h: int = 560):
        self.canvas = canvas
        self.max_w, self.max_h = max_w, max_h
        self.zoom = 1.0
        self.fit_zoom = 1.0
        self.on_change = None            # callable(self) after a render
        self.cursor_var = None           # optional tk.StringVar for a live px readout
        self._src = None                 # RGB ndarray of the current frame
        self._photo = None               # keep a ref or Tk garbage-collects the image
        self._shown = False
        self._forced_viewport = None     # (w, h) override for headless tests
        self._click_cb = None            # callable(ix, iy) in image px
        self._overlays: list[tuple] = [] # (kind, image-coord args, colour)

    # -- mapping -----------------------------------------------------------
    def to_image(self, cx: float, cy: float) -> tuple[float, float]:
        """Canvas px -> image px, honouring zoom AND scroll offset."""
        z = self.zoom or 1.0
        return self.canvas.canvasx(cx) / z, self.canvas.canvasy(cy) / z

    def to_display(self, ix: float, iy: float) -> tuple[float, float]:
        """Image px -> canvas px (the transform the renderer uses, inverted)."""
        return ix * self.zoom, iy * self.zoom

    def _viewport(self) -> tuple[int, int]:
        if self._forced_viewport is not None:
            return self._forced_viewport
        return max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height())

    def _clamp_zoom(self, z: float) -> float:
        return max(self.fit_zoom, min(MAX_ZOOM, float(z)))

    def _update_scrollregion(self) -> None:
        h, w = self._src.shape[:2]
        self.canvas.configure(scrollregion=(0, 0, w * self.zoom, h * self.zoom))

    # -- display -----------------------------------------------------------
    def show(self, bgr, *, reset_view: bool = False) -> None:
        self._src = bgr[:, :, ::-1]          # BGR -> RGB
        h, w = self._src.shape[:2]
        self.fit_zoom = min(1.0, self.max_w / w, self.max_h / h)
        if reset_view or not self._shown:
            self.zoom = self.fit_zoom
            self._shown = True
        else:
            self.zoom = self._clamp_zoom(self.zoom)
        self._update_scrollregion()
        self._render()

    def _render(self) -> None:
        z = self.zoom
        H, W = self._src.shape[:2]
        cw, ch = self._viewport()
        self.canvas.delete("all")
        x0, y0 = self.canvas.canvasx(0) / z, self.canvas.canvasy(0) / z
        pad = 4
        cx0, cy0 = max(0, int(x0) - pad), max(0, int(y0) - pad)
        cx1 = min(W, int(x0 + cw / z) + pad + 1)
        cy1 = min(H, int(y0 + ch / z) + pad + 1)
        crop = self._src[cy0:cy1, cx0:cx1]
        resample = Image.NEAREST if z > 1.0 else Image.BILINEAR
        img = Image.fromarray(crop).resize(
            (max(1, int(round(crop.shape[1] * z))), max(1, int(round(crop.shape[0] * z)))),
            resample)
        self._photo = ImageTk.PhotoImage(img, master=self.canvas)
        self.canvas.create_image(cx0 * z, cy0 * z, anchor="nw", image=self._photo)
        self._draw_overlays()
        if self.on_change:
            self.on_change(self)

    # -- zoom / pan --------------------------------------------------------
    def set_zoom(self, z: float, anchor=None) -> float:
        if self._src is None:
            return self.zoom
        if anchor is None:
            cw, ch = self._viewport()
            anchor = (cw / 2.0, ch / 2.0)
        ix = self.canvas.canvasx(anchor[0]) / self.zoom
        iy = self.canvas.canvasy(anchor[1]) / self.zoom
        self.zoom = self._clamp_zoom(z)
        self._update_scrollregion()
        H, W = self._src.shape[:2]
        self.canvas.xview_moveto(
            min(1.0, max(0.0, (ix * self.zoom - anchor[0]) / max(1.0, W * self.zoom))))
        self.canvas.yview_moveto(
            min(1.0, max(0.0, (iy * self.zoom - anchor[1]) / max(1.0, H * self.zoom))))
        self._render()
        return self.zoom

    def zoom_by(self, factor: float, anchor=None) -> float:
        return self.set_zoom(self.zoom * factor, anchor=anchor)

    def fit(self) -> float:
        return self.set_zoom(self.fit_zoom)

    def bind_zoom(self) -> None:
        self.canvas.bind("<MouseWheel>",
                         lambda e: self.zoom_by(1.25 ** (e.delta / 120.0), (e.x, e.y)))
        self.canvas.bind("<Button-2>", lambda e: self.canvas.scan_mark(e.x, e.y))
        self.canvas.bind("<B2-Motion>",
                         lambda e: (self.canvas.scan_dragto(e.x, e.y, gain=1), self._render()))
        self.canvas.bind("<Motion>", self.on_motion)
        self.canvas.bind("<Button-1>", self._on_click)

    def on_motion(self, event) -> None:
        if self.cursor_var is None or self._src is None:
            return
        ix, iy = self.to_image(event.x, event.y)
        h, w = self._src.shape[:2]
        inside = 0 <= ix < w and 0 <= iy < h
        self.cursor_var.set(f"x={ix:7.2f}  y={iy:7.2f} px" + ("" if inside else "  (outside)"))

    # -- click capture (the measurement) -----------------------------------
    def on_click(self, callback: Callable[[float, float], None]) -> None:
        """Register a handler receiving (ix, iy) in image px on a left click."""
        self._click_cb = callback

    def _on_click(self, event) -> None:
        if self._click_cb is None or self._src is None:
            return
        ix, iy = self.to_image(event.x, event.y)
        h, w = self._src.shape[:2]
        if 0 <= ix < w and 0 <= iy < h:
            self._click_cb(float(ix), float(iy))

    # -- overlays (image-coordinate geometry) ------------------------------
    def clear_overlays(self) -> None:
        self._overlays = []
        self._render()

    def set_overlays(self, overlays: Sequence[tuple]) -> None:
        """``(kind, args, colour)``: kind in {'marker','curve'}; args in image px."""
        self._overlays = list(overlays)
        self._render()

    def _draw_overlays(self) -> None:
        for kind, args, colour in self._overlays:
            if kind == "marker":
                ix, iy, r = args
                cx, cy = self.to_display(ix, iy)
                rr = max(2, r * self.zoom)
                self.canvas.create_oval(cx - rr, cy - rr, cx + rr, cy + rr,
                                        outline=colour, width=2)
            elif kind == "curve":
                pts = [self.to_display(px, py) for px, py in args]
                if len(pts) >= 2:
                    self.canvas.create_line(*[c for p in pts for c in p],
                                            fill=colour, width=2, smooth=True)
