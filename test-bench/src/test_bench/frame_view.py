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
        self._marquee_cb = None          # callable(x0, y0, x1, y1) in image px
        self._drag_start = None          # canvas-px press anchor
        self._drag_img = None            # image-px press anchor
        self._dragged = False
        self._marquee_rect = None        # canvas rect id while dragging
        self._overlays: list[tuple] = [] # (kind, image-coord args, colour[, alpha])
        self.element_opacity = 1.0       # 0..1 — trajectory / points / entry marks
        self.grid_opacity = 1.0          # 0..1 — the metric lattice only

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
        if self._src is None:
            return
        h, w = self._src.shape[:2]
        self.canvas.configure(scrollregion=(0, 0, w * self.zoom, h * self.zoom))

    # -- display -----------------------------------------------------------
    def show(self, bgr, *, reset_view: bool = False) -> None:
        self._src = bgr[:, :, ::-1]          # BGR -> RGB
        if self._src is None:
            return
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
        if self._src is None:
            return
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
        if self._src is None:
                    return
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
        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)

    def on_motion(self, event) -> None:
        if self.cursor_var is None or self._src is None:
            return
        ix, iy = self.to_image(event.x, event.y)
        if self._src is None:
            return
        h, w = self._src.shape[:2]
        inside = 0 <= ix < w and 0 <= iy < h
        self.cursor_var.set(f"x={ix:7.2f}  y={iy:7.2f} px" + ("" if inside else "  (outside)"))

    # -- click capture (the measurement) -----------------------------------
    def on_click(self, callback: Callable[[float, float], None]) -> None:
        """Register a handler receiving (ix, iy) in image px on a left click."""
        self._click_cb = callback

    def on_marquee(self, callback) -> None:
        """Register a handler receiving ``(x0, y0, x1, y1)`` in image px on a
        box-drag release.  ``None`` disarms the marquee (clicks pass through)."""
        self._marquee_cb = callback

    def _on_press(self, event) -> None:
        if self._src is None:
            return
        self._drag_start = (event.x, event.y)
        self._drag_img = self.to_image(event.x, event.y)
        self._dragged = False
        self._marquee_rect = None
        if self._marquee_cb is not None:
            self._marquee_rect = self.canvas.create_rectangle(
                event.x, event.y, event.x, event.y,
                outline="#00e5ff", dash=(4, 4), width=1)

    def _on_drag(self, event) -> None:
        if getattr(self, "_drag_start", None) is None:
            return
        if getattr(self, "_marquee_rect", None) is not None:
            x0, y0 = self._drag_start
            self.canvas.coords(self._marquee_rect, x0, y0, event.x, event.y)
            if abs(event.x - x0) + abs(event.y - y0) > 4:
                self._dragged = True

    def _on_release(self, event) -> None:
        start_img = getattr(self, "_drag_img", None)
        rect = getattr(self, "_marquee_rect", None)
        dragged = bool(getattr(self, "_dragged", False))
        self._drag_start = None
        self._drag_img = None
        self._marquee_rect = None
        if rect is not None:
            self.canvas.delete(rect)
        if start_img is None:
            return
        if dragged and self._marquee_cb is not None:
            ix0, iy0 = start_img
            ix1, iy1 = self.to_image(event.x, event.y)
            x0, x1 = sorted((ix0, ix1))
            y0, y1 = sorted((iy0, iy1))
            self._marquee_cb(float(x0), float(y0), float(x1), float(y1))
        else:
            self._on_click(event)

    def _on_click(self, event) -> None:
        if self._click_cb is None or self._src is None:
            return
        ix, iy = self.to_image(event.x, event.y)
        if self._src is None:
            return
        h, w = self._src.shape[:2]
        if 0 <= ix < w and 0 <= iy < h:
            self._click_cb(float(ix), float(iy))

    # -- overlays (image-coordinate geometry) ------------------------------
    def clear_overlays(self) -> None:
        self._overlays = []
        self._render()

    def set_overlays(self, overlays: Sequence[tuple]) -> None:
        """``(kind, args, colour[, alpha])`` in image px.

        ``kind`` in ``{'marker', 'dot', 'curve', 'segment', 'label', 'grid'}``.
        The optional 4th element is a per-item opacity override (0..1); otherwise
        the view's ``element_opacity`` (or ``grid_opacity`` for ``grid``) applies.
        """
        self._overlays = list(overlays)
        self._render()

    # -- the dynamic line style (ported from the parent's Review canvas) ------
    @staticmethod
    def _review_stroke(k: float) -> tuple[int, tuple[int, int] | None, float]:
        """``(line width, dash, dot radius)`` for zoom factor ``k = zoom / fit``.

        Operator order 2026-10-21: at fit the whole frame sits on the canvas, so a
        solid 2 px stroke is a bar across the subject — draw a 1 px dashed hairline;
        past 1.5x close to a 2 px stroke, past 4x to 3 px solid, because at those
        zooms the same width covers a small part of the picture.
        """
        if k < 1.5:
            return 1, (3, 3), 2.0
        if k < 4.0:
            return 2, (5, 2), 2.5
        return 3, None, 3.5

    def _hex_rgb(self, colour: str) -> tuple[int, int, int]:
        try:
            r, g, b = self.canvas.winfo_rgb(colour)
            return r >> 8, g >> 8, b >> 8
        except Exception:                                            # noqa: BLE001
            return 128, 128, 128

    def _bg_at(self, at) -> tuple[int, int, int]:
        """The frame's own RGB under an image point (the alpha-blend backdrop)."""
        src = self._src
        if src is None or at is None:
            return (128, 128, 128)
        h, w = src.shape[:2]
        x = int(min(max(0, round(float(at[0]))), w - 1))
        y = int(min(max(0, round(float(at[1]))), h - 1))
        r, g, b = src[y, x][:3]
        return int(r), int(g), int(b)

    def _blend(self, colour: str, at=None, alpha: float | None = None) -> str:
        """``colour`` at opacity over the frame pixel at ``at`` (Tk has no alpha)."""
        a = self.element_opacity if alpha is None else max(0.0, min(1.0, float(alpha)))
        if a >= 0.999:
            return colour
        ir, ig, ib = self._hex_rgb(colour)
        br, bg, bb = self._bg_at(at)

        def mix(c: int, d: int) -> int:
            return max(0, min(255, int(round(a * c + (1.0 - a) * d))))

        return f"#{mix(ir, br):02x}{mix(ig, bg):02x}{mix(ib, bb):02x}"

    def _ink_segments(self, pts, colour: str, w: int, dash, alpha: float) -> None:
        """Polyline in display px, blended per segment when faded (backdrop varies)."""
        if len(pts) < 2:
            return
        if alpha >= 0.999:
            flat = [c for p in pts for c in p]
            kw: dict = {"fill": colour, "width": w}
            if dash:
                kw["dash"] = dash
            self.canvas.create_line(*flat, **kw)
            return
        z = self.zoom or 1.0
        for a, b in zip(pts, pts[1:]):
            mid_img = ((a[0] + b[0]) * 0.5 / z, (a[1] + b[1]) * 0.5 / z)
            kw = {"fill": self._blend(colour, mid_img, alpha), "width": w}
            if dash:
                kw["dash"] = dash
            self.canvas.create_line(a[0], a[1], b[0], b[1], **kw)

    def _draw_overlays(self) -> None:
        if not self._overlays:
            return
        z = self.zoom or 1.0
        fit = max(1e-6, float(self.fit_zoom or 1.0))
        k = max(1.0, z / fit)
        line_w, dash, dot_r = self._review_stroke(k)

        for item in self._overlays:
            kind, args, colour = item[0], item[1], item[2]
            alpha = item[3] if len(item) > 3 else None
            if kind == "grid":
                ink = self._blend(colour, None, self.grid_opacity)
            else:
                ink = self._blend(colour, None,
                                  alpha if alpha is not None else self.element_opacity)

            if kind in ("marker", "dot"):
                ix, iy, wt = args
                cx, cy = self.to_display(ix, iy)
                rr = max(1.5 if kind == "dot" else 2.0, dot_r * float(wt))
                if kind == "marker":
                    self.canvas.create_oval(cx - rr, cy - rr, cx + rr, cy + rr,
                                            outline=ink, width=max(1, line_w))
                else:  # dot: a filled detection point
                    self.canvas.create_oval(cx - rr, cy - rr, cx + rr, cy + rr,
                                            fill=ink, outline=ink, width=1)
            elif kind == "curve":
                pts = [self.to_display(px, py) for px, py in args]
                self._ink_segments(pts, ink, line_w, dash,
                                   alpha if alpha is not None else self.element_opacity)
            elif kind == "segment":
                x0, y0, x1, y1 = args
                ax, ay = self.to_display(x0, y0)
                bx, by = self.to_display(x1, y1)
                self._ink_segments([(ax, ay), (bx, by)], ink, line_w, dash,
                                   alpha if alpha is not None else self.element_opacity)
            elif kind == "label":
                ix, iy, text = args
                cx, cy = self.to_display(ix, iy)
                self.canvas.create_text(cx + 8, cy - 8, text=str(text), fill=ink,
                                        anchor="w", font=("TkDefaultFont", 8))
            elif kind == "grid":
                for a, b, major in args:
                    ax, ay = self.to_display(a[0], a[1])
                    bx, by = self.to_display(b[0], b[1])
                    self.canvas.create_line(ax, ay, bx, by, fill=ink,
                                            width=2 if major else 1)

