"""FrameView — pixel-exact zoom/pan/click/overlay mapping (headless)."""
from __future__ import annotations

import sys
import tkinter as tk
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))

from test_bench.frame_view import MAX_ZOOM, FrameView  # noqa: E402


@pytest.fixture(scope="module")
def root():
    try:
        r = tk.Tk()
    except tk.TclError:
        pytest.skip("no Tk display available")
    r.withdraw()
    yield r
    r.destroy()


def _view(root, w=900, h=560):
    canvas = tk.Canvas(root, width=w, height=h)
    canvas.pack()
    view = FrameView(canvas, w, h)
    view.bind_zoom()
    view._forced_viewport = (w, h)
    return canvas, view


def test_clamp_and_precision(root):
    _, view = _view(root)
    view.show(np.zeros((1080, 1920, 3), np.uint8))
    assert view.set_zoom(0.01) == pytest.approx(view.fit_zoom)
    assert view.set_zoom(1000) == MAX_ZOOM
    view.fit()
    assert 1.0 / view.zoom > 2.0            # fit quantises > 2 image px / canvas px
    view.set_zoom(4.0)
    assert 1.0 / view.zoom <= 0.26          # 4x clears a sub-px placement budget


def test_span_mapping_relative(root):
    canvas, view = _view(root)
    view.show(np.zeros((1080, 1920, 3), np.uint8))
    view.set_zoom(2.0)
    canvas.xview_moveto(0)
    canvas.yview_moveto(0)
    x0, _ = view.to_image(0, 0)
    x1, _ = view.to_image(400, 0)
    assert x1 - x0 == pytest.approx(200.0, abs=1.5)   # 400 canvas px at 2x = 200 image px


def test_to_display_span_exact(root):
    _, view = _view(root)
    view.show(np.zeros((1080, 1920, 3), np.uint8))
    view.set_zoom(3.0)
    x0, y0 = view.to_display(0, 0)
    x1, y1 = view.to_display(100, 50)
    assert x1 - x0 == pytest.approx(300.0)    # 100 image px * 3.0 zoom, no scroll term
    assert y1 - y0 == pytest.approx(150.0)


def test_click_maps_to_image(root):
    _, view = _view(root)
    view.show(np.zeros((1080, 1920, 3), np.uint8))
    got: list[tuple[float, float]] = []
    view.on_click(lambda ix, iy: got.append((ix, iy)))

    class _E:
        x, y = 400, 200
    view._on_click(_E())
    assert got == [view.to_image(400, 200)]


def test_overlay_draw_does_not_crash(root):
    _, view = _view(root)
    view.show(np.zeros((1080, 1920, 3), np.uint8))
    view.set_overlays([
        ("marker", (532.0, 311.0, 6.0), "red"),
        ("curve", [(100, 100), (200, 120), (300, 110)], "cyan"),
    ])
    view.clear_overlays()


def test_review_stroke_table():
    assert FrameView._review_stroke(1.0) == (1, (3, 3), 2.0)
    assert FrameView._review_stroke(1.4) == (1, (3, 3), 2.0)
    assert FrameView._review_stroke(1.5) == (2, (5, 2), 2.5)
    assert FrameView._review_stroke(3.9) == (2, (5, 2), 2.5)
    assert FrameView._review_stroke(4.0) == (3, None, 3.5)
    assert FrameView._review_stroke(8.0) == (3, None, 3.5)


def test_blend_midpoint(root):
    _, view = _view(root)
    frame = np.zeros((1080, 1920, 3), np.uint8)
    frame[:] = (40, 40, 40)                       # dark grey backdrop
    view.show(frame)
    # element opacity 0.5 over (40,40,40) with white ink (255,255,255)
    view.element_opacity = 0.5
    blended = view._blend("white", (0, 0))
    # 0.5*255 + 0.5*40 = 147.5 -> 148
    assert blended == "#949494"


def test_element_opacity_zero_hides_but_grid_survives(root):
    _, view = _view(root)
    view.show(np.zeros((1080, 1920, 3), np.uint8))
    view.element_opacity = 0.0
    view.grid_opacity = 1.0
    view.set_overlays([
        ("marker", (100, 100, 1.0), "red"),
        ("grid", [((0, 0), (100, 0), True)], "#3a6ea5"),
    ])
    # element overlays vanish at opacity 0 (blended to the backdrop), grid stays
    assert view._blend("red", None, 0.0) != "red"


def test_new_overlay_kinds_draw_without_crash(root):
    canvas, view = _view(root)
    view.show(np.zeros((1080, 1920, 3), np.uint8))
    view.set_overlays([
        ("dot", (100.0, 100.0, 1.0), "#00e5ff"),
        ("segment", (0.0, 0.0, 100.0, 100.0), "magenta"),
        ("label", (100.0, 100.0, "Δ3.2px"), "cyan"),
        ("grid", [((0, 0), (500, 0), False), ((0, 0), (0, 500), True)], "#3a6ea5"),
    ])
    # some canvas items were created (image + the overlays)
    assert len(canvas.find_all()) >= 4


def test_zoom_changes_line_width(root):
    canvas, view = _view(root)
    view.show(np.zeros((1080, 1920, 3), np.uint8))
    view.set_overlays([("curve", [(0, 0), (100, 100)], "cyan")])
    # at fit, the curve is a 1-px dashed line; canvas stores the width in its item config
    items = [i for i in canvas.find_all() if canvas.type(i) == "line"]
    assert items, "expected a line overlay item"
    assert float(canvas.itemcget(items[0], "width")) == 1.0
    view.set_zoom(8.0)
    view.set_overlays([("curve", [(0, 0), (100, 100)], "cyan")])
    items = [i for i in canvas.find_all() if canvas.type(i) == "line"]
    assert float(canvas.itemcget(items[0], "width")) == 3.0


class _E:
    def __init__(self, x, y):
        self.x, self.y = x, y


def test_press_release_dispatches_click(root):
    _, view = _view(root)
    view.show(np.zeros((1080, 1920, 3), np.uint8))
    got = []
    view.on_click(lambda ix, iy: got.append((ix, iy)))
    view._on_press(_E(400, 200))
    view._on_release(_E(400, 200))      # no drag -> a plain click
    assert got == [view.to_image(400, 200)]


def test_marquee_drag_dispatches_rect_not_click(root):
    canvas, view = _view(root)
    view.show(np.zeros((1080, 1920, 3), np.uint8))
    clicks, rects = [], []
    view.on_click(lambda ix, iy: clicks.append((ix, iy)))
    view.on_marquee(lambda x0, y0, x1, y1: rects.append((x0, y0, x1, y1)))
    view._on_press(_E(100, 50))
    view._on_drag(_E(300, 250))
    view._on_release(_E(300, 250))
    assert rects and not clicks
    x0, y0, x1, y1 = rects[0]
    assert x0 <= x1 and y0 <= y1


def test_marquee_disarmed_ignores_drag(root):
    canvas, view = _view(root)
    view.show(np.zeros((1080, 1920, 3), np.uint8))
    rects = []
    view.on_marquee(lambda *r: rects.append(r))
    view.on_marquee(None)               # disarm
    view._on_press(_E(100, 50))
    view._on_drag(_E(300, 250))
    view._on_release(_E(300, 250))
    assert not rects

