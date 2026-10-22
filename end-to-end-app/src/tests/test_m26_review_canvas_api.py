"""M26 — Review's canvas UI: element opacity, zoom-dependent strokes, a draggable tailset X.

Operator orders (2026-10-21), quoted:

1. *"Add a slider below the table to modify the transparency of all elements.  The initial opacity
   should be 10%."*  Tk has no item alpha, so ``App._blend`` does the alpha arithmetic against the
   frame pixel under each element — that is what these tests check: the ink is mixed with the
   backdrop, and the slider is what sets the mix.
2. *"Remove 'Correct Entry' and instead add a 'Correct Entry' button that allows the user to
   override the trajectory and X marker to place the location of the tailset."*  One button, armed
   as a drag mode; the drop point becomes the measured point AND the frame the trajectory stops at.
3. *"The X should turn into a small dot when the user hovers, clicks, and drags the point."*
4. *"the X for the last detection still does not match ... the other UI."*  Pane and exported still
   now draw one mark: a tilted X.
5. *"Make the line style dynamic and dependent on zoom level, currently it obstructs the view."*
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import app_gui                                               # noqa: E402
from dog_diving import annotate as A                           # noqa: E402
from tests import pool_scene as PS                           # noqa: E402
from tests.test_m19_measure import plane_doc, traj_with      # noqa: E402


def _root_or_skip():
    import tkinter as tk
    last = None
    for _ in range(3):
        try:
            root = tk.Tk()
            root.withdraw()
            return root
        except Exception as exc:                            # noqa: BLE001
            last = exc
    pytest.skip(f"no Tk display available after 3 attempts: {type(last).__name__}: {last}")


@pytest.fixture(autouse=True)
def synthetic_frame_source(monkeypatch):
    """Every redraw path decodes a frame; here it is a synthetic one (there is no clip in a test)."""
    monkeypatch.setattr(app_gui, "read_frame", lambda clip, idx: _frame())


class Ev:
    """The only thing Tk hands a canvas handler."""

    def __init__(self, x: float, y: float) -> None:
        self.x, self.y = x, y


def _canvas_xy(view, ix: float, iy: float) -> tuple[float, float]:
    z = view.zoom
    return (ix * z - view.canvas.canvasx(0.0), iy * z - view.canvas.canvasy(0.0))


class _Contact:
    """Just enough of ``contact.Contact`` for ReviewState (it only reads ``chosen_frame``)."""

    def __init__(self, frame: int) -> None:
        self.chosen_frame = int(frame)


def _frame(w: int = 480, h: int = 320) -> np.ndarray:
    """A synthetic BGR frame with two clearly different halves, so a blend is measurable."""
    f = np.zeros((h, w, 3), np.uint8)
    f[:, : w // 2] = (40, 200, 40)          # left half bright green (BGR)
    f[:, w // 2:] = (200, 40, 40)           # right half blue
    return f


def _app_with_review(root):
    """An App holding a real ReviewState over synthetic points — no clip, no model."""
    app = app_gui.App(root, clip=None)
    root.update()
    sc = PS.deck_marks_scene()
    doc = plane_doc(sc)
    pts = []
    for i, X in enumerate((4.0, 7.0, 10.0, 13.0, 16.0)):
        v = sc.project(X, 8.0, sc.water_z_ft)
        pts.append({"frame": 100 + 10 * i, "x": float(v[0]), "y": float(v[1]), "kp_conf": 0.95})
    app.review = app_gui.ReviewState(
        clip=Path("synthetic.mp4"), traj=traj_with(pts), contact=_Contact(pts[-1]["frame"]),
        calib=None, pool_doc=doc, image_size=(sc.image_size[0], sc.image_size[1]),
        n_frames=200,
    )
    app.review_view._forced_viewport = (880, 420)
    app.review_view.show(_frame())
    root.update()
    return app, pts


def _overlay_lines(app) -> list:
    """Overlay polylines only — the lattice has its own tag and its own weights."""
    c = app.review_view.canvas
    return [i for i in c.find_withtag("overlay")
            if c.type(i) == "line" and "grid" not in c.gettags(i)]


def _grid_items(app) -> list:
    return list(app.review_view.canvas.find_withtag("grid"))


def _overlay_ovals(app) -> list:
    c = app.review_view.canvas
    return [i for i in c.find_withtag("overlay") if c.type(i) == "oval"]


# ===========================================================================
# 1. the opacity slider
# ===========================================================================
def test_the_opacity_slider_sits_under_the_table_and_starts_at_90_percent():
    root = _root_or_skip()
    try:
        app = app_gui.App(root, clip=None)
        root.update()
        assert app.opacity_scale.cget("from") == 0
        assert app.opacity_scale.cget("to") == 100
        assert float(app.entry_opacity.get()) == pytest.approx(90.0)
        assert app_gui.OPACITY_DEFAULT_PCT == 90.0
        # it is in the points-table column, packed BELOW whatever holds the table
        assert app.opacity_scale.winfo_manager() == "pack"
        holder = app.opacity_scale.master
        parent = holder.master
        above = app.pts_tree
        while above.master is not parent:
            above = above.master
            assert above is not app.tab_review, "the slider is not in the table's column"
        siblings = list(parent.pack_slaves())
        assert siblings.index(holder) > siblings.index(above), "the slider is not below the table"
    finally:
        root.destroy()


def test_opacity_mixes_the_ink_with_the_frame_underneath():
    root = _root_or_skip()
    try:
        app, _pts = _app_with_review(root)
        # 100 %: the ink is drawn as itself
        app.entry_opacity.set(100)
        assert app._opacity() == pytest.approx(1.0)
        assert app._blend("#ffe14d", (10.0, 10.0)) == "#ffe14d"
        # 10 %: the ink is nearly the backdrop the element sits on …
        app.entry_opacity.set(10)
        faded = app._blend("#ffe14d", (10.0, 10.0))
        assert faded != "#ffe14d"
        ink = np.array(app_gui._hex_rgb("#ffe14d"), float)
        bg = np.array(app._bg_at((10.0, 10.0)), float)
        got = np.array(app_gui._hex_rgb(faded), float)
        assert np.linalg.norm(got - bg) < np.linalg.norm(got - ink), (faded, bg, ink)
        # … and 10 % of the way from the backdrop towards the ink, numerically
        want = 0.10 * ink + 0.90 * bg
        assert np.allclose(got, want, atol=1.5), (got, want)
        # the two halves of the frame are different backdrops, so the blend must differ too
        assert app._blend("#ffe14d", (10.0, 10.0)) != app._blend("#ffe14d", (400.0, 10.0))
    finally:
        root.destroy()


def test_the_slider_repaints_the_overlay_at_the_new_opacity():
    root = _root_or_skip()
    try:
        app, _pts = _app_with_review(root)
        app.entry_opacity.set(100)
        app.on_opacity(100)
        strong = {app.review_view.canvas.itemcget(i, "fill") for i in _overlay_lines(app)}
        app.entry_opacity.set(10)
        app.on_opacity(10)
        faint = {app.review_view.canvas.itemcget(i, "fill") for i in _overlay_lines(app)}
        assert strong and faint
        assert not (strong & faint), "the fill did not change with the slider"
    finally:
        root.destroy()


# ===========================================================================
# 2. zoom-dependent line style
# ===========================================================================
def test_the_stroke_thins_and_dashes_as_the_frame_is_zoomed_out():
    w_fit, dash_fit, dot_fit = app_gui.App._review_stroke(1.0)
    w_mid, dash_mid, _ = app_gui.App._review_stroke(2.0)
    w_in, dash_in, dot_in = app_gui.App._review_stroke(8.0)
    assert w_fit == 1 and dash_fit is not None, "at fit the line must be a dashed hairline"
    assert w_mid == 2 and w_in == 3
    assert dash_in is None, "zoomed in, the stroke must be closed"
    assert dot_fit < dot_in


def test_the_canvas_stroke_follows_the_zoom():
    root = _root_or_skip()
    try:
        app, _pts = _app_with_review(root)
        app.entry_opacity.set(100)
        v = app.review_view
        v.zoom = v.fit_zoom
        v.refresh()
        at_fit = {int(float(v.canvas.itemcget(i, "width"))) for i in _overlay_lines(app)}
        dashes = {v.canvas.itemcget(i, "dash") for i in _overlay_lines(app)}
        assert at_fit == {1}, at_fit
        assert any(d for d in dashes), "no dash pattern at fit"

        v.set_zoom(min(4.0, app_gui.MAX_ZOOM))
        zoomed = {int(float(v.canvas.itemcget(i, "width"))) for i in _overlay_lines(app)}
        assert max(zoomed) >= 2, zoomed
    finally:
        root.destroy()


def test_a_faded_polyline_is_drawn_per_segment():
    root = _root_or_skip()
    try:
        app, pts = _app_with_review(root)
        app.entry_opacity.set(100)
        app.review_view.redraw_overlay()
        solid = len(_overlay_lines(app))
        app.entry_opacity.set(10)
        app.review_view.redraw_overlay()
        faded = len(_overlay_lines(app))
        # one item per segment of the trajectory + the arc, instead of one per polyline
        assert faded > solid + len(pts) - 2, (solid, faded)
    finally:
        root.destroy()


# ===========================================================================
# 3. Correct Entry: arm, hover, drag, commit
# ===========================================================================
def test_the_button_is_called_correct_entry_and_arms_a_drag_mode():
    root = _root_or_skip()
    try:
        app, _pts = _app_with_review(root)
        assert app.correct_btn.cget("text") == "Correct Entry"
        app.on_toggle_correct()
        assert app.correct_entry_mode is True
        assert app.correct_point_mode is True                 # the old name still reads
        assert "ARMED" in app.correct_btn.cget("text")
        assert app.review_view.canvas.cget("cursor") == "crosshair"
        app.on_toggle_correct()
        assert app.correct_entry_mode is False
        assert app.correct_btn.cget("text") == "Correct Entry"
    finally:
        root.destroy()


def test_hovering_the_x_turns_it_into_a_dot():
    root = _root_or_skip()
    try:
        app, _pts = _app_with_review(root)
        app.entry_opacity.set(100)
        app.on_toggle_correct()
        mark = app._entry_point()
        assert mark is not None
        v = app.review_view

        app.on_review_leave()
        before = [i for i in _overlay_ovals(app) if v.canvas.itemcget(i, "fill")]
        assert not before, "a filled dot was drawn with no pointer on the mark"

        app.on_review_motion(Ev(*_canvas_xy(v, *mark)))       # hover ON the X
        dots = [i for i in _overlay_ovals(app) if v.canvas.itemcget(i, "fill")]
        assert len(dots) == 1, "hovering the X did not turn it into a dot"
        assert app.entry_cursor is not None

        # and away from the mark the X comes back (only the dashed preview ring is left)
        app.on_review_motion(Ev(*_canvas_xy(v, mark[0] + 120.0, mark[1] + 90.0)))
        assert not [i for i in _overlay_ovals(app) if v.canvas.itemcget(i, "fill")]
    finally:
        root.destroy()


def test_dragging_the_tailset_sets_the_point_and_cuts_the_trajectory():
    root = _root_or_skip()
    try:
        app, pts = _app_with_review(root)
        app.on_toggle_correct()
        v = app.review_view
        target = pts[2]                                        # the middle detection
        app.on_review_click(Ev(*_canvas_xy(v, float(target["x"]), float(target["y"]))))
        assert app._entry_drag is not None, "the press did not take hold of the mark"
        assert app.review.manual_point == pytest.approx(
            (float(target["x"]), float(target["y"])), abs=1.0)
        # while dragging the X is a dot, not an X
        app.on_review_drag(Ev(*_canvas_xy(v, float(target["x"]) + 3.0, float(target["y"]) + 2.0)))
        dots = [i for i in _overlay_ovals(app) if v.canvas.itemcget(i, "fill")]
        assert dots, "the marker did not become a dot while dragging"

        app.on_review_release(Ev(*_canvas_xy(v, float(target["x"]), float(target["y"]))))
        assert app._entry_drag is None
        assert app.correct_entry_mode is False, "the mode stayed armed after the drop"
        assert app.review.manual_point == pytest.approx(
            (float(target["x"]), float(target["y"])), abs=1.0)
        # the trajectory is overridden too: it now stops at the dropped-on frame
        assert app.review.entry_frame == int(target["frame"])
        assert all(int(p["frame"]) <= int(target["frame"]) for p in app.review.draw_points())
        assert app.review.measurement()["point_source"] == "manual"
    finally:
        root.destroy()


def test_auto_entry_drops_the_dragged_point():
    root = _root_or_skip()
    try:
        app, pts = _app_with_review(root)
        app.review.manual_point = (float(pts[0]["x"]), float(pts[0]["y"]))
        app.review.entry_frame = int(pts[0]["frame"])
        app.on_clear_entry()
        assert app.review.manual_point is None
        assert app.review.entry_frame is None
        assert app.review.measurement()["point_source"] != "manual"
    finally:
        root.destroy()


# ===========================================================================
# 4. one mark for one quantity: the pane and the exported still
# ===========================================================================
def test_the_exported_still_draws_the_same_x_as_the_pane():
    img = np.full((200, 200, 3), 0, np.uint8)
    out = A.draw_measurement(img.copy(), {"x_px": 100.0, "y_px": 100.0})
    red = (0, 0, 255)
    on_diagonal = tuple(int(v) for v in out[92, 92])
    on_vertical = tuple(int(v) for v in out[100, 92])
    assert on_diagonal == red, f"no X arm on the diagonal: {on_diagonal} (a plus, not an X?)"
    assert on_vertical != red, "a vertical arm is still drawn: that is the old plus mark"


# ===========================================================================
# 5. the lattice follows the opacity rule, and the mark is drawn once
# ===========================================================================
def test_the_grid_is_a_canvas_layer_that_follows_the_slider():
    """Operator order 2026-10-21: *"the grid does not adhere to the opacity rule."*"""
    root = _root_or_skip()
    try:
        app, _pts = _app_with_review(root)
        app.entry_opacity.set(100)
        app.review_view.redraw_overlay()
        items = _grid_items(app)
        assert len(items) > 20, f"the lattice is not on the canvas: {len(items)} item(s)"
        c = app.review_view.canvas
        strong = {c.itemcget(i, "fill") for i in items if c.type(i) == "line"}
        assert len(strong) > 2, strong

        app.entry_opacity.set(20)
        app.review_view.redraw_overlay()
        faint = {c.itemcget(i, "fill") for i in _grid_items(app) if c.type(i) == "line"}
        assert not (strong & faint), "the lattice colour did not change with the slider"

        app.entry_opacity.set(0)                     # 0 %: none of the overlay is left
        app.review_view.redraw_overlay()
        assert not _grid_items(app)
    finally:
        root.destroy()


def test_the_grid_spec_lands_on_the_planes_own_points():
    """The canvas lattice and the baked one come from one function, so they cannot drift."""
    from dog_diving.pool import plane as PL

    sc = PS.deck_marks_scene()
    doc = plane_doc(sc)
    spec = PL.grid_spec(doc["homography"], length_ft=25.0,
                        width_ft=float(sc.width_ft), lens=None)
    assert len(spec["lines"]) >= 25 + int(round(sc.width_ft))
    assert {"a", "b", "major"} <= set(spec["lines"][0])
    assert spec["labels"] and spec["labels"][0]["text"].endswith("'")
    # the first lattice line is X = 0, running the pool's width through the document's own map
    first = spec["lines"][0]
    assert first["a"] == pytest.approx(PL.project(doc["homography"], 0.0, 0.0))
    assert first["b"] == pytest.approx(PL.project(doc["homography"], 0.0, float(sc.width_ft)))
    assert "datum" in spec


def test_the_review_frame_carries_no_baked_overlay(monkeypatch):
    """The frame is the photo; the lattice, trajectory and X are canvas items — drawn once each."""
    root = _root_or_skip()
    try:
        app, _pts = _app_with_review(root)
        seen: dict = {}
        real = app_gui.annotated_frame

        def spy(clip, index, **kw):
            seen.update(kw)
            return real(clip, index, **kw)

        monkeypatch.setattr(app_gui, "annotated_frame", spy)
        app._redraw_review()
        assert seen.get("show_grid") is False, "the lattice is still baked into the frame"
        assert seen.get("pool_result") is None, "the entry mark is still baked into the frame"
        assert seen.get("entry_mark") is False
        assert seen.get("measurement") is None
    finally:
        root.destroy()


def test_the_still_still_draws_the_entry_mark():
    """Taking it off the pane must not take it out of an export."""
    img = np.full((120, 120, 3), 0, np.uint8)
    result = {"entry": {"X_ft": 10.0, "Y_ft": 4.0, "x_px": 60.0, "y_px": 60.0}}
    on = A.draw_frame(img.copy(), pool_result=result, banner=False)
    off = A.draw_frame(img.copy(), pool_result=result, banner=False, entry_mark=False)
    assert np.any(on != 0), "the export lost its entry mark"
    assert not np.any(off != 0), "entry_mark=False still drew the mark"
