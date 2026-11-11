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
