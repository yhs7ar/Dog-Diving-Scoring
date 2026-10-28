"""M20 — plane placement: a fixed 0-35 ft axis, an arrow pad for fine placement, and an overlay
whose lines stay 1-2 **screen** px at any zoom.

Operator order, 2026-10-20:
* the +x axis always extends to 35 ft; the ``length (ft)`` entry is only an anchor for refined
  placement against detail visible in the video, never the length of the measured pool;
* a fixed arrow pad under the canvas moves the selected point, one image px per click
  (Shift 10, Ctrl/Alt 0.1);
* the thick baked-in lines are replaced by canvas vectors so they thin out as the operator zooms;
* the viewer scales with the window.

The Tk tests are skipped, not failed, when no display is available — same contract as
``test_m8_gui``.  They build the real ``App`` and read the real canvas item list, because the
defect they guard against (a width that scales with the zoom) is invisible to a pure function.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import app_gui                                              # noqa: E402
import dog_diving.config as DDC                               # noqa: E402
import dog_diving.pool.plane as PL                            # noqa: E402


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


def _frame(w: int = 960, h: int = 600) -> np.ndarray:
    f = np.full((h, w, 3), 40, dtype=np.uint8)
    f[:, :, 1] = 90
    return f


def _plane(length_ft: float = 25.0) -> PL.PlaneSession:
    s = PL.PlaneSession(frame_shape=(600, 960), length_ft=length_ft)
    s.fit_to_frame()
    return s


class _Ev:
    """Minimal Tk event stand-in.  ``state`` drives the nudge step; the rest is for ``on_resize``."""
    def __init__(self, state: int = 0, **kw):
        self.state = state
        for k, v in kw.items():
            setattr(self, k, v)


def _app(root, monkeypatch=None, *, start: bool = True):
    def fake_read_frame(clip, idx):
        return _frame()

    if monkeypatch is not None:
        monkeypatch.setattr(app_gui, "read_frame", fake_read_frame)
    else:
        app_gui.read_frame = fake_read_frame
    app = app_gui.App(root, clip=None)
    app.clip = Path("synthetic.mp4")
    app.confirm_reset = lambda: True      # New Plane asks before replacing a plane; no modal here
    root.update()
    if start:
        app.on_new_plane(seed=False)
        root.update()
    return app


def _line_widths(canvas) -> dict[float, int]:
    out: dict[float, int] = {}
    for item in canvas.find_withtag("overlay"):
        if canvas.type(item) == "line":
            w = float(canvas.itemcget(item, "width"))
            out[w] = out.get(w, 0) + 1
    return out


def _datum_span(canvas) -> float:
    """Length of the only 2 px line in the overlay — the X = 0 dock-lip datum."""
    best = 0.0
    for item in canvas.find_withtag("overlay"):
        if canvas.type(item) != "line" or float(canvas.itemcget(item, "width")) != 2.0:
            continue
        x0, y0, x1, y1 = canvas.coords(item)
        best = max(best, float(np.hypot(x1 - x0, y1 - y0)))
    return best


# ---------------------------------------------------------------------------
# 1. the metric axis is FIXED at 35 ft; the entry is only an anchor
# ---------------------------------------------------------------------------
def test_the_metric_axis_runs_to_35_ft_whatever_the_anchor_is():
    for anchor in (18.0, 25.0, 35.0):
        s = _plane(anchor)
        assert s.extent()["x_max"] == pytest.approx(PL.METRIC_LENGTH_FT), (anchor, s.extent())
        assert PL.METRIC_LENGTH_FT == pytest.approx(35.0)
        assert DDC.POOL_LENGTH_FT == pytest.approx(35.0)
        block = s.as_dict()
        assert block["metric_length_ft"] == pytest.approx(35.0)
        assert block["length_ft"] == pytest.approx(anchor), "the anchor must survive the change"


def test_the_mapping_reports_past_the_anchored_far_edge():
    """A point beyond the quad is still X ft from the dock lip — that is the point of the order."""
    s = _plane(25.0)
    px = s.pixel_of(33.0, 8.0)
    X, Y = s.world_at(*px)
    assert X == pytest.approx(33.0, abs=1e-6), X
    assert Y == pytest.approx(8.0, abs=1e-6), Y


def test_the_mapping_summary_probes_the_whole_axis():
    xs = [row["X_ft"] for row in PL.mapping_summary(_plane(20.0))["probe"]]
    assert xs[0] == 0.0 and xs[-1] == pytest.approx(35.0), xs
    assert len(xs) == 8, xs                                    # 0, 5, ..., 35


def test_the_document_records_the_axis_and_keeps_the_anchor():
    s = _plane(22.0)
    doc = s.build("m20-axis")
    assert doc["extent"]["x_max"] == pytest.approx(35.0)
    assert doc["plane"]["metric_length_ft"] == pytest.approx(35.0)
    assert doc["plane"]["length_ft"] == pytest.approx(22.0)
    assert all(g["pass"] for g in doc["plane_gates"].values()), doc["plane_gates"]


def test_the_rendered_grid_is_driven_by_the_metric_axis_not_the_anchor(monkeypatch):
    """If the grid were still driven by ``length_ft``, shortening the axis would change nothing."""
    s = _plane(20.0)
    frame = _frame()
    full = PL.draw_plane(frame, s, grid=True)
    monkeypatch.setattr(PL, "METRIC_LENGTH_FT", 20.0)
    clipped = PL.draw_plane(frame, s, grid=True)
    assert not np.array_equal(full, clipped), (
        "draw_plane ignored METRIC_LENGTH_FT — the grid is still bounded by the anchor")


# ---------------------------------------------------------------------------
# 2. the arrow pad / nudge arithmetic
# ---------------------------------------------------------------------------
def test_a_corner_nudges_by_exactly_the_step_given():
    s = _plane()
    s.select("10")
    x0, y0 = s.corners["10"]
    assert s.nudge(1.0, 0.0) is True
    assert s.corners["10"] == pytest.approx((x0 + 1.0, y0))
    assert s.nudge(0.0, -10.0) is True
    assert s.corners["10"] == pytest.approx((x0 + 1.0, y0 - 10.0))


def test_sub_pixel_steps_accumulate_instead_of_rounding_away():
    s = _plane()
    s.select("00")
    x0 = s.corners["00"][0]
    for _ in range(3):
        s.nudge(0.1, 0.0)
    assert s.corners["00"][0] == pytest.approx(x0 + 0.3, abs=1e-9)


def test_a_side_nudge_translates_both_of_its_corners():
    s = _plane()
    s.select("front")
    a, b = s.corners["11"], s.corners["01"]
    assert s.nudge(0.0, 2.0) is True
    assert s.corners["11"] == pytest.approx((a[0], a[1] + 2.0))
    assert s.corners["01"] == pytest.approx((b[0], b[1] + 2.0))


def test_nothing_selected_means_nothing_moves():
    s = _plane()
    assert s.nudge(1.0, 1.0) is False
    s.select("nonsense")
    assert s.selected is None
    assert s.nudge(1.0, 1.0) is False


def test_the_grab_radius_is_screen_pixels_so_zoom_does_not_lose_the_handle():
    """At 8x, 10 image px away is 80 screen px — too far.  At fit, the same 10 px is a hit."""
    s = _plane()
    cx, cy = s.corners["00"]
    disp = s.to_display(cx + 10.0, cy)
    assert s.handle_at(*disp, tol=PL.GRAB_PX / 1.0) == "00", "a 10 px gap must be grabbable at 1x"
    assert s.handle_at(*disp, tol=PL.GRAB_PX / 8.0) is None, "at 8x that gap is 80 screen px"


# ---------------------------------------------------------------------------
# 3. the live pane: vectors, thickness, pad, resize
# ---------------------------------------------------------------------------
def test_the_overlay_is_zoom_independent_and_positions_scale(monkeypatch):
    root = _root_or_skip()
    try:
        app = _app(root, monkeypatch)
        app.calib_view.set_zoom(1.0)
        root.update()
        w1 = _line_widths(app.calib_canvas)
        span1 = _datum_span(app.calib_canvas)
        assert w1, "the plane drew no lines at all"
        assert max(w1) <= 2.0, w1

        app.calib_view.set_zoom(8.0)
        root.update()
        w8 = _line_widths(app.calib_canvas)
        span8 = _datum_span(app.calib_canvas)

        assert w8 == w1, f"line widths changed with zoom: {w1} -> {w8}"
        assert span8 == pytest.approx(span1 * 8.0, rel=0.02), (span1, span8)
    finally:
        root.destroy()


def test_the_plane_survives_the_grid_being_switched_off(monkeypatch):
    root = _root_or_skip()
    try:
        app = _app(root, monkeypatch)
        app.grid_var.set(False)
        app._redraw_plane()
        root.update()
        widths = _line_widths(app.calib_canvas)
        assert 2.0 in widths, "the dock-lip datum disappeared with the grid"
        assert widths.get(1.0, 0) <= 4, widths      # datum-adjacent outline only, no 1 ft rules
    finally:
        root.destroy()


def test_the_pad_moves_the_selected_corner_by_1_10_and_0_1_px(monkeypatch):
    root = _root_or_skip()
    try:
        app = _app(root, monkeypatch)
        s = app.plane_session
        s.select("10")
        x0, y0 = s.corners["10"]

        app.on_nudge_handle(1, 0, _Ev(0))
        assert s.corners["10"][0] == pytest.approx(x0 + 1.0)

        app.on_nudge_handle(0, 1, _Ev(0x1))                    # Shift
        assert s.corners["10"][1] == pytest.approx(y0 + 10.0)

        app.on_nudge_handle(1, 0, _Ev(0x4))                    # Control
        assert s.corners["10"][0] == pytest.approx(x0 + 1.1, abs=1e-9)

        app.on_nudge_handle(1, 0, _Ev(0x20000))                # Alt on Windows
        assert s.corners["10"][0] == pytest.approx(x0 + 1.2, abs=1e-9)
    finally:
        root.destroy()


def test_a_nudge_does_not_re_render_the_photo(monkeypatch):
    """The point of the vector overlay: moving a corner must not re-resize the frame."""
    root = _root_or_skip()
    try:
        app = _app(root, monkeypatch)
        calls = []
        original = app.calib_view._render
        app.calib_view._render = lambda: (calls.append(1), original())[1]
        app.plane_session.select("10")
        for _ in range(5):
            app.on_nudge_handle(1, 0, _Ev(0))
        assert calls == [], f"each nudge re-rendered the image ({len(calls)} renders)"
    finally:
        root.destroy()


def test_the_arrow_keys_do_not_nudge_from_another_tab(monkeypatch):
    root = _root_or_skip()
    try:
        app = _app(root, monkeypatch)
        s = app.plane_session
        s.select("10")
        before = s.corners["10"]

        app.nb.select(1)                                       # Process — not the plane tab
        assert app.on_arrow_key(_Ev(0), 1, 0) is None
        assert s.corners["10"] == before, "an arrow key on another tab moved the plane"

        app.nb.select(0)                                       # Calibrate
        assert app.on_arrow_key(_Ev(0), 1, 0) == "break"
        assert s.corners["10"][0] == pytest.approx(before[0] + 1.0)
    finally:
        root.destroy()


def test_a_pane_left_at_fit_follows_the_window_size(monkeypatch):
    root = _root_or_skip()
    try:
        app = _app(root, monkeypatch)
        view = app.calib_view
        # Width-bound on purpose: a 960x600 frame in a 600-wide pane fits by width, so widening the
        # pane must raise the fit scale.  (In a 900x400 pane the fit is height-bound and widening
        # changes nothing, which is correct and is why the numbers are chosen this way.)
        view._forced_viewport = (600, 900)
        view.show(_frame(), reset_view=True)
        first = view.zoom
        assert view._at_fit is True

        view._forced_viewport = (900, 900)
        view.on_resize(_Ev(0, widget=view.canvas, width=900, height=900))
        assert view.fit_zoom > first, "a wider window must give a larger fit scale"
        assert view.zoom == pytest.approx(view.fit_zoom), "a pane at fit must follow the window"

        view.set_zoom(4.0)
        view._forced_viewport = (700, 900)
        view.on_resize(_Ev(0, widget=view.canvas, width=700, height=900))
        assert view.zoom == pytest.approx(4.0), "resizing threw away the operator's zoom"
    finally:
        root.destroy()


def test_fit_is_computed_on_the_photo_not_on_the_padding(monkeypatch):
    """A padded canvas fitted to the pane shows the frame at a quarter of the size and the rest
    black — the pane must fit the PHOTO and reach the padding by panning."""
    root = _root_or_skip()
    try:
        app = _app(root, monkeypatch)
        view = app.calib_view
        view._forced_viewport = (900, 400)
        view.show(PL.padded_canvas(app.pool_frame, app.plane_session.pad), reset_view=True)

        # frame 960x600 in a 900x400 pane fits by height; the padded canvas (1632x1272) gives 0.314
        assert view.fit_zoom == pytest.approx(400 / 600, rel=1e-3), view.fit_zoom
        assert view.content_box is not None

        box = view.content_box
        ix, iy = view.to_image(450, 200)
        assert box[0] <= ix <= box[2] and box[1] <= iy <= box[3], (ix, iy, box)
    finally:
        root.destroy()


def test_a_zoom_button_keeps_the_photo_in_view(monkeypatch):
    root = _root_or_skip()
    try:
        app = _app(root, monkeypatch)
        view = app.calib_view
        view._forced_viewport = (900, 400)
        view.show(PL.padded_canvas(app.pool_frame, app.plane_session.pad), reset_view=True)
        box = view.content_box

        view.set_zoom(4.0)                          # no anchor == the zoom-button path
        ix, iy = view.to_image(450, 200)
        assert box[0] <= ix <= box[2] and box[1] <= iy <= box[3], (
            f"4x zoom landed at ({ix:.0f}, {iy:.0f}), outside the photo {box}")
    finally:
        root.destroy()


def test_the_length_entry_is_an_anchor_and_35_is_the_ceiling(monkeypatch):
    root = _root_or_skip()
    try:
        app = _app(root, monkeypatch)
        # Before the corners are placed the boxes describe the INITIAL FIT: the full 0-35 ft axis.
        assert app.plane_len_var.get() == "35"
        assert app.plane_near_var.get() == "0"
        assert app.plane_session.length_ft == pytest.approx(PL.METRIC_LENGTH_FT)
        assert app.plane_session.dock_offset_ft == pytest.approx(0.0)

        app.plane_len_var.set("40")
        app.on_plane_dims()
        assert app.plane_session.length_ft == pytest.approx(PL.METRIC_LENGTH_FT), (
            "an anchor beyond the 35 ft axis must be refused, leaving the plane unchanged")

        # Applying anchors moves the SELECTABLE short edges to those X values ...
        H0 = app.plane_session.homography()
        before = dict(app.plane_session.corners)
        app.plane_near_var.set("10")
        app.plane_len_var.set("15")
        app.on_plane_dims()
        assert app.plane_session.dock_offset_ft == pytest.approx(10.0)
        assert app.plane_session.length_ft == pytest.approx(5.0)
        assert app.plane_session.corners != before, "the selectable short edges did not move"
        for tag, want_x in (("00", 10.0), ("01", 10.0), ("10", 15.0), ("11", 15.0)):
            wx, _wy = app.plane_session.world_at(*app.plane_session.corners[tag])
            assert wx == pytest.approx(want_x, abs=1e-6), (tag, wx)

        # ... and the PLANE does not move with them.
        H1 = app.plane_session.homography()
        for X, Y in ((0.0, 0.0), (5.0, 5.0), (17.5, 10.0), (35.0, 20.0)):
            a = PL.project(H0, X, Y)
            b = PL.project(H1, X, Y)
            d = ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5
            assert d < 1e-6, f"the plane moved {d:.3g} px at X={X}, Y={Y}"
        assert app.plane_session.extent()["x_max"] == pytest.approx(PL.METRIC_LENGTH_FT)
    finally:
        root.destroy()
