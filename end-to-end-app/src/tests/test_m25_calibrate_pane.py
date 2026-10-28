"""M25 — the 2026-10-21 Calibrate-pane order: one New Plane, a snap toggle with four boxes.

Operator order, in his words:

* *"'New Plane', 'Make parallelogram' and 'Fit to frame' are redundant"* — one button makes the
  plane, runs the waterline detector on **the clip's first 90 frames**, snaps the quad onto the
  detected lines with the far edge short and the near edge long, and arms the snap toggle;
* *"The Snap to Waterline Toggle should have its own button"*, with boxes for **Top, Bottom, Left,
  Right**;
* *"Flip Plane (left-right) should keep the plane orientation completely and only change the number
  scale used"*;
* *"The Save New Plane button should be next to the 'saved plane:' dropdown"*;
* the anchors read ``• Anchor (near dock):`` / ``• Anchor (opposite dock):`` and default to 0 / 35 ft;
* *"Hide calibration frame index and show 1-ft grid, these should be assumed"*;
* *"Add two new waterlines, the waterline at the short ends of the pool.  Snap +x = 0 to the right or
  left waterline depending on orientation selected by the user."*

The Tk half builds the real ``App`` (skipped, not failed, with no display — same contract as
``test_m8_gui``) because a button's text, colour and order are exactly what these orders are about.
"""
from __future__ import annotations

import sys
import time
import dataclasses
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import app_gui                                            # noqa: E402
from dog_diving import waterline as WL                      # noqa: E402
from dog_diving.pool import plane as PL                     # noqa: E402
from dog_diving.pool import seed as SD                      # noqa: E402

FRAME = np.full((600, 960, 3), 60, dtype=np.uint8)


def _root_or_skip():
    import tkinter as tk
    last = None
    for _ in range(3):
        try:
            root = tk.Tk()
            root.withdraw()
            return root
        except Exception as exc:                          # noqa: BLE001
            last = exc
    pytest.skip(f"no Tk display available after 3 attempts: {type(last).__name__}: {last}")


# ---------------------------------------------------------------------------
# synthetic waterlines
# ---------------------------------------------------------------------------
def _edge(name: str, point, direction, n: int = 40) -> WL.EdgeFit:
    p = np.asarray(point, float)
    d = np.asarray(direction, float)
    d = d / np.linalg.norm(d)
    pts = p + np.outer(np.linspace(-150.0, 150.0, n), d)
    return WL.EdgeFit(name=name, found=True, points=pts,
                      fit={"point": p, "direction": d, "endpoints": [pts[0], pts[-1]],
                           "angle_deg": float(np.degrees(np.arctan2(d[1], d[0]))),
                           "slope": float(d[1] / d[0]) if abs(d[0]) > 1e-9 else float("inf"),
                           "intercept": 0.0, "rms_residual": 0.5, "p95_residual": 1.0,
                           "rms_center": 0.5, "rms_ends": 0.5},
                      inlier_fraction=0.95, stability_px=0.4)


def _scene_edges(dock_side: str = "left") -> WL.WaterlineEdges:
    """A pool seen down its length: two long waterlines and both short ends."""
    far = _edge("far", (100.0, 260.0), (1.0, 0.0))
    near = _edge("near", (60.0, 470.0), (1.0, 0.0))
    left = _edge("left", (150.0, 200.0), (0.0, 1.0))
    right = _edge("right", (800.0, 300.0), (0.0, 1.0))
    dock = dataclasses.replace(left if dock_side == "left" else right, name="dock")
    # The mask fields matter: S2 refuses a detection whose water fraction is unknown, so a scene
    # without them can only ever seed by refusing (which is how this used to be masked by snapping).
    return WL.WaterlineEdges(far=far, near=near, dock=dock, left=left, right=right,
                             dock_side=dock_side, frame_size=(960, 600),
                             water_fraction=0.6, n_components=1, components_kept=1,
                             dominant_share=0.95)


# ===========================================================================
# 1. the waterline module: both short ends
# ===========================================================================
def test_the_opposite_short_edge_is_a_vertical_bridging_the_two_long_edges():
    """The opposite side is CONSTRUCTED: one x, intersected with both long waterlines.

    Order 2026-10-28: *"the opposite short edge of the pool should be placed as a vertical line
    bridging the near and far long edges."*  Before this, it was either the detector's own line (a
    ``frame_cut`` vertical at the picture border, or a diagonal reported ``observed``) or the dock
    waterline's direction copied through the far end — three rules, all of them able to put the side
    somewhere the operator did not expect.  One vertical through the far end of the visible water is
    all of them collapsed into one.

    Regression history: with the cut in the fit the X-pencil swept 78 deg (16.8 / 20.7 deg edges
    against a 12.2 deg dock waterline); with the dock direction copied the quad stopped being
    vertical on the far side; with a border line taken for the far wall the quad became
    00 (638,302) 10 (0,309) 11 (0,426) 01 (1187,425).
    """
    ang = np.radians(12.2)
    far = _edge("far", (100.0, 260.0), (1.0, 0.0))
    near = _edge("near", (60.0, 470.0), (1.0, 0.0))
    dock = _edge("right", (800.0, 300.0), (float(np.cos(ang)), float(np.sin(ang))))
    cut = _edge("left", (2.0, 470.0), (0.0, -1.0))
    cut.status = "frame_cut"                      # a vertical at the border, NOT a waterline
    edges = WL.WaterlineEdges(far=far, near=near, dock=dock, left=cut, right=dock,
                              dock_side="right", frame_size=(960, 600),
                              water_fraction=0.6, n_components=1, components_kept=1,
                              dominant_share=0.95)
    res = SD.seed_from_edges(edges, length_ft=35.0, width_ft=20.0, frame_shape=(600, 960))
    assert not res.refused, res.reason
    assert res.level == "L4"
    assert any("VERTICAL" in n for n in res.notes), res.notes

    p10 = np.asarray(res.corners["10"], float)
    p11 = np.asarray(res.corners["11"], float)
    assert p11[0] == pytest.approx(p10[0], abs=1e-6), (
        f"the opposite edge is not vertical: x = {p10[0]:.1f} -> {p11[0]:.1f}")
    # it really does bridge the two long waterlines: same x, one on each line (the near quad edge is
    # the detected near waterline dropped by NEAR_WALL_DROP_PX — that line is the top of the wall)
    assert p10[1] == pytest.approx(260.0, abs=2.0), p10
    assert p11[1] == pytest.approx(470.0 + float(SD.NEAR_WALL_DROP_PX), abs=2.0), p11


def test_a_border_line_reported_observed_is_never_the_far_wall():
    """A vertical AT the frame border, reported ``observed``, must not become the far side.

    The x of the constructed vertical comes from where the far waterline ENDS, not from a detector
    line, so the status of the short-side detection cannot move it.
    """
    ang = np.radians(12.2)
    far = _edge("far", (100.0, 260.0), (1.0, 0.0))
    near = _edge("near", (60.0, 470.0), (1.0, 0.0))
    dock = _edge("right", (800.0, 300.0), (float(np.cos(ang)), float(np.sin(ang))))
    border = _edge("left", (0.0, 470.0), (0.0, -1.0))          # vertical, ON the frame border
    border.status = "observed"                                  # ... and reported as a real line
    edges = WL.WaterlineEdges(far=far, near=near, dock=dock, left=border, right=dock,
                              dock_side="right", frame_size=(960, 600),
                              water_fraction=0.6, n_components=1, components_kept=1,
                              dominant_share=0.95)
    res = SD.seed_from_edges(edges, length_ft=35.0, width_ft=20.0, frame_shape=(600, 960))
    assert not res.refused, res.reason
    p10 = np.asarray(res.corners["10"], float)
    p11 = np.asarray(res.corners["11"], float)
    assert p11[0] == pytest.approx(p10[0], abs=1e-6)            # vertical
    assert abs(p11[1] - p10[1]) > 100.0, (p10, p11)             # bridges far -> near
    assert p10[1] == pytest.approx(260.0, abs=2.0), p10         # on the far waterline
    assert p11[1] == pytest.approx(470.0 + float(SD.NEAR_WALL_DROP_PX), abs=2.0), p11


def test_both_short_ends_are_exposed_and_dock_is_one_of_them():
    e = _scene_edges()
    assert [x.name for x in e.ends] == ["left", "right"]
    assert [x.name for x in e.all_edges] == ["far", "near", "dock", "left", "right"]
    assert e.dock_side == "left"
    # dock is a copy of the winning end, not a fourth line: same geometry, different label
    assert np.allclose(e.dock.fit["point"], e.left.fit["point"])


def test_the_short_ends_reach_the_document_and_the_overlay():
    e = _scene_edges()
    d = e.as_dict()
    assert set(d["short_ends"]) == {"left", "right"}
    assert {"far", "near", "dock", "left", "right"} <= set(d["edges"])
    assert WL.as_overlay(e)["right"] is not None


# ===========================================================================
# 2. the seed: X = 0 is whichever short end the orientation control names
# ===========================================================================
def test_the_seed_follows_the_orientation_control_over_the_detector():
    e = _scene_edges(dock_side="left")          # the detector votes "left"
    res = SD.seed_from_edges(e, length_ft=35.0, width_ft=float(PL.DEFAULT_WIDTH_FT),
                             frame_shape=(600, 960), dock_side="right")
    assert res.dock_side == "right"
    assert any("X=0 snapped to the right waterline" in n for n in res.notes), res.notes


def test_the_seed_keeps_the_detector_vote_when_no_orientation_is_chosen():
    e = _scene_edges(dock_side="left")
    res = SD.seed_from_edges(e, length_ft=35.0, width_ft=float(PL.DEFAULT_WIDTH_FT),
                             frame_shape=(600, 960), dock_side=None)
    assert res.dock_side == "left"
    assert not any("orientation" in n for n in res.notes), res.notes


def test_a_missing_short_end_falls_back_and_says_so():
    e = _scene_edges(dock_side="left")
    e.right = WL.EdgeFit(name="right", found=False, reason="not found")
    res = SD.seed_from_edges(e, length_ft=35.0, width_ft=float(PL.DEFAULT_WIDTH_FT),
                             frame_shape=(600, 960), dock_side="right")
    assert any("right short-end waterline was not found" in n for n in res.notes), res.notes


# ===========================================================================
# 3. the far edge is the short one
# ===========================================================================
def test_ensure_far_edge_short_swaps_the_end_that_is_too_big():
    s = PL.PlaneSession(frame_shape=(600, 960), length_ft=35.0, width_ft=12.0)
    # far end drawn FATTER than the near end: the pool the wrong way round
    s.corners = {"00": (100.0, 300.0), "10": (900.0, 300.0),
                 "11": (700.0, 420.0), "01": (300.0, 420.0)}
    before = dict(s.corners)
    assert s.ensure_far_edge_short() is True
    assert sorted(s.corners.values()) == sorted(before.values()), "a pixel moved"
    far = np.hypot(*(np.subtract(s.corners["10"], s.corners["00"])))
    near = np.hypot(*(np.subtract(s.corners["11"], s.corners["01"])))
    assert far < near
    assert s.ensure_far_edge_short() is False, "a correct plane was swapped again"


# ===========================================================================
# 4. flip: the numbering changes, the quad does not
# ===========================================================================
def test_flip_keeps_every_corner_pixel_and_moves_x0_to_the_other_end():
    s = PL.PlaneSession(frame_shape=(600, 960), length_ft=35.0, width_ft=12.0)
    s.corners = {"00": (100.0, 300.0), "10": (800.0, 330.0),
                 "11": (800.0, 500.0), "01": (100.0, 470.0)}
    before = dict(s.corners)
    assert s.flip_x() is True
    assert s.corners == {"00": before["10"], "10": before["00"],
                         "01": before["11"], "11": before["01"]}
    assert sorted(s.corners.values()) == sorted(before.values())


# ===========================================================================
# 5. the pane itself
# ===========================================================================
def _app(root, monkeypatch=None):
    if monkeypatch is not None:
        monkeypatch.setattr(app_gui, "read_frame", lambda clip, idx: FRAME.copy())
    else:
        app_gui.read_frame = lambda clip, idx: FRAME.copy()
    app = app_gui.App(root, clip=None)
    app.clip = Path("synthetic.mp4")
    app.confirm_reset = lambda: True
    root.update()
    return app


def _texts(widget) -> list[str]:
    """Every button/checkbutton/label caption in the Calibrate tab, in packing order."""
    out: list[str] = []

    def walk(w) -> None:
        for child in w.winfo_children():
            try:
                out.append(str(child.cget("text")))
            except Exception:                             # noqa: BLE001
                pass
            walk(child)

    walk(widget)
    return out


def test_the_button_row_is_the_operator_order():
    root = _root_or_skip()
    try:
        app = _app(root, None)
        texts = _texts(app.tab_calib)
        for wanted in ("New Plane", "Re-fit Waterline", "Flip Plane (left-right)",
                       "Plane Confirmed"):
            assert wanted in texts, f"{wanted!r} missing from the Calibrate tab: {texts}"
        # the two redundant buttons and the old names are gone
        for gone in ("Make parallelogram", "Fit to frame", "Flip X", "Build document",
                     "Save plane", "calibration frame index:", "Show 1-ft grid",
                     "Re-detect Waterline"):
            assert gone not in texts, f"{gone!r} is still a control: {texts}"
        # the order is the operator's
        order = [t for t in texts if t in ("New Plane", "Re-fit Waterline",
                                           "Flip Plane (left-right)", "Plane Confirmed")]
        assert order == ["New Plane", "Re-fit Waterline", "Flip Plane (left-right)",
                         "Plane Confirmed"]
        # the confirmation is the big one: taller and wider than a default primary button
        assert int(app.complete_btn.cget("pady")) >= 10, app.complete_btn.cget("pady")
        assert int(app.complete_btn.cget("padx")) >= 24, app.complete_btn.cget("padx")
    finally:
        root.destroy()


def test_the_primary_buttons_are_blue_with_white_text():
    root = _root_or_skip()
    try:
        app = app_gui.App(root, clip=None)
        app.clip = Path("synthetic.mp4")
        app.confirm_reset = lambda: True
        root.update()
        for btn in (app.new_plane_btn, app.complete_btn):
            assert btn.cget("text") in ("New Plane", "Plane Confirmed")
            assert btn.cget("bg").lower() == app_gui.PRIMARY_BG.lower(), btn.cget("bg")
            assert btn.cget("fg").lower() == "white", btn.cget("fg")
    finally:
        root.destroy()


def test_save_new_plane_is_hidden_but_kept_for_the_document_route():
    root = _root_or_skip()
    try:
        app = app_gui.App(root, clip=None)
        app.clip = Path("synthetic.mp4")
        app.confirm_reset = lambda: True
        root.update()
        assert app.save_plane_btn.cget("text") == "Save New Plane"
        # Hidden by operator order 2026-10-27: Complete commits the plane.  The widget survives
        # (the document route and its handler still exist) but no geometry manager owns it.
        assert app.save_plane_btn.winfo_manager() == "", "Save New Plane is still on screen"
    finally:
        root.destroy()


def test_the_anchor_fields_are_labelled_and_default_to_10_and_15():
    root = _root_or_skip()
    try:
        app = app_gui.App(root, clip=None)
        app.clip = Path("synthetic.mp4")
        app.confirm_reset = lambda: True
        root.update()
        texts = _texts(app.tab_calib)
        assert "• Anchor (near dock):" in texts, texts
        assert "• Anchor (opposite dock):" in texts, texts
        assert app.plane_near_var.get() == "10"
        assert app.plane_len_var.get() == "15"
    finally:
        root.destroy()


def test_the_snap_toggle_has_its_own_button_and_four_boxes():
    root = _root_or_skip()
    try:
        app = app_gui.App(root, clip=None)
        app.clip = Path("synthetic.mp4")
        app.confirm_reset = lambda: True
        root.update()
        assert app.snap_btn.cget("text").startswith("TOGGLE: Snap to Waterline")
        assert "(ON)" in app.snap_btn.cget("text")
        # HIDDEN as a checkbox row (2026-10-27) — the per-edge LOCKS on the canvas replaced it.
        assert app.snap_btn.winfo_manager() == "", "the snap toggle is still on screen"
        assert all(not cb.winfo_manager() for cb in app.snap_btn.master.winfo_children()
                   if cb.winfo_class() == "TCheckbutton")
        assert [lab for _side, lab in app_gui.SNAP_SIDE_LABELS] == \
               ["Top", "Bottom", "Left", "Right"]
        assert set(app.snap_side_vars) == {"back", "front", "left", "right"}
        # the long edges are locked (they were fitted); the short ends are released — they sit at
        # the anchor X values in mid-pool, where there is no waterline to hold (2026-10-27)
        for side in ("back", "front"):
            assert app.snap_side_vars[side].get() is True, side
        for side in ("left", "right"):
            assert app.snap_side_vars[side].get() is False, side
        # each edge has a lock icon on the canvas once a quad is on it
        app.on_new_plane(seed=False)
        app._redraw_plane()
        assert {"back", "front", "left", "right"} <= set(app._lock_hotspots), app._lock_hotspots
        # clicking the Bottom lock releases exactly that edge
        app.on_toggle_side_snap("front")
        assert app.snap_side_vars["front"].get() is False
        assert app.snap_side_vars["back"].get() is True
        app.on_toggle_side_snap("front")
        assert app.snap_side_vars["front"].get() is True
    finally:
        root.destroy()


def test_new_plane_asks_before_resetting_an_existing_plane():
    root = _root_or_skip()
    try:
        app = app_gui.App(root, clip=None)
        app.clip = Path("synthetic.mp4")
        app.confirm_reset = lambda: True
        root.update()
        app.on_new_plane(seed=False)
        first = app.plane_session
        assert first is not None

        app.confirm_reset = lambda: False              # the operator says "no"
        app.on_new_plane(seed=False)
        assert app.plane_session is first, "the plane was replaced after a cancelled prompt"

        app.confirm_reset = lambda: True
        app.on_new_plane(seed=False)
        assert app.plane_session is not first, "the confirmed reset did not make a new plane"
    finally:
        root.destroy()


def test_new_plane_detects_on_the_opening_frames_and_arms_the_snap_toggle(monkeypatch):
    root = _root_or_skip()
    seen: list[tuple[int, int]] = []

    def fake_detect(clip, **kw):
        win = kw.get("window")
        if win is not None:
            seen.append((int(win[0]), int(win[1])))
        return _scene_edges("left")

    monkeypatch.setattr(app_gui.waterline_mod, "detect_edges", fake_detect)
    try:
        app = app_gui.App(root, clip=None)
        app.clip = Path("synthetic.mp4")
        app.confirm_reset = lambda: True
        root.update()
        app.on_new_plane()

        deadline = time.time() + 10.0
        while time.time() < deadline and app.seed_edges is None:
            root.update()
            app._drain()
            time.sleep(0.02)
        assert app.seed_edges is not None, "the waterline worker never delivered"

        assert seen[0] == (0, app_gui.SEED_FRAME_COUNT), seen
        assert app_gui.SEED_FRAME_COUNT == 90
        # the toggle is armed again: the per-edge LOCKS decide what actually moves (2026-10-27)
        assert app.snap_on_var.get() is True
        assert "(ON)" in app.snap_btn.cget("text")
        # The detected lines are on the canvas and available to the pull / re-fit primitives.
        s = app.plane_session
        lines = app._snap_lines()
        assert {"back", "front", "left", "right"} <= set(lines), lines

        def perp(line, tag: str) -> float:
            point, direction = line
            q = np.asarray(s.corners[tag], float)
            return abs(float((q - point) @ np.array([-direction[1], direction[0]])))

        pairs = (("00", "10"), ("01", "11"))
        # One long side of the quad is ON the detected far waterline (the seed measures it), and the
        # other is on the near waterline DROPPED by NEAR_WALL_DROP_PX (the pool wall above the
        # water).  Tag-agnostic: the seed may swap the Y ends so the far edge is the short one.
        assert min(max(perp(lines["back"], t) for t in pair) for pair in pairs) < 1.0, (
            "no side of the quad sits on the detected far waterline")
        assert min(max(perp(lines["front"], t) for t in pair) for pair in pairs) < 1.0, (
            "no side of the quad sits on the wall-dropped near waterline")
        # and that near target really is the detected line dropped by the wall height
        raw = app.seed_edges.near.fit
        p_raw = np.asarray(raw["point"], float)
        d_raw = np.asarray(raw["direction"], float)
        d_raw = d_raw / np.linalg.norm(d_raw)
        drop = abs(float((np.asarray(lines["front"][0], float) - p_raw)
                         @ np.array([-d_raw[1], d_raw[0]])))
        assert drop == pytest.approx(SD.NEAR_WALL_DROP_PX, abs=0.5)
    finally:
        root.destroy()


def test_snapping_only_touches_the_checked_boxes(monkeypatch):
    root = _root_or_skip()
    monkeypatch.setattr(app_gui.waterline_mod, "detect_edges",
                        lambda clip, **kw: _scene_edges("left"))
    try:
        app = app_gui.App(root, clip=None)
        app.clip = Path("synthetic.mp4")
        app.confirm_reset = lambda: True
        root.update()
        app.on_new_plane(seed=False)                   # no detection, no snap
        app.seed_edges = _scene_edges("left")
        # ALL FOUR edges are monitored; the long pair starts locked, the short pair released.
        assert app.snap_side_vars["front"].get() is True
        for side in ("left", "right"):
            assert app.snap_side_vars[side].get() is False
        app.snap_side_vars["left"].set(False)
        app.snap_side_vars["right"].set(False)
        app.snap_side_vars["front"].set(False)
        app.snap_side_vars["back"].set(True)
        before = dict(app.plane_session.corners)
        app.snap_on_var.set(True)
        assert app._maybe_snap_edges() is True
        # Bottom, Left and Right are released: 01 and 11 belong to those and must not move
        for tag in ("01", "11"):
            assert app.plane_session.corners[tag] == pytest.approx(before[tag]), tag

        # checking Bottom as well is what makes the near edge follow its line
        app.snap_side_vars["front"].set(True)
        before = dict(app.plane_session.corners)
        app._maybe_snap_edges()
        for tag in ("01", "11"):
            assert app.plane_session.corners[tag] != pytest.approx(before[tag]), tag
    finally:
        root.destroy()


def test_x0_snaps_to_the_short_end_the_orientation_names():
    """A short edge holds its OWN X — not whichever pool wall the detector saw.

    Regression (2026-10-27): the short-end lock used the detected waterline as its target, so with
    the quad's short edges at the 10/15 ft anchors, locking Right dragged the whole band onto the
    dock wall — a 60 px jump that landed X = 15 on X = 0.  With the quad on the walls (the 0..35
    fit) the plane's own X = 0 line IS the detected waterline, so the old behaviour is preserved
    where it was correct.
    """
    root = _root_or_skip()
    try:
        app = app_gui.App(root, clip=None)
        app.clip = Path("synthetic.mp4")
        app.confirm_reset = lambda: True
        root.update()
        app.on_new_plane(seed=False)
        app.seed_edges = _scene_edges("left")
        app.dock_side_var.set("left")
        # seed the quad onto the walls first: the 0..35 fit puts its X = 0 edge ON the detected end
        res = SD.seed_from_edges(app.seed_edges, length_ft=35.0, width_ft=20.0,
                                 frame_shape=(600, 960), current=app.plane_session)
        assert not res.refused, res.reason
        app.plane_session.apply_seed(res.corners)
        app.plane_session.ensure_far_edge_short()
        lines = app._snap_lines()
        want = np.asarray(app.seed_edges.left.fit["point"], float)
        d = np.asarray(lines["left"][1], float)
        off = float(d[0] * (want - lines["left"][0])[1] - d[1] * (want - lines["left"][0])[0])
        assert abs(off) < 5.0, f"the X = 0 lock target misses the detected end by {off:.1f} px"

        # off the walls (the default 10/15 anchors) the target stays in the band: snapping it moves
        # nothing, instead of travelling to the pool wall
        app.plane_session.apply_anchors(10.0, 15.0)
        lines = app._snap_lines()
        mid = 0.5 * (np.asarray(app.plane_session.corners["10"], float)
                     + np.asarray(app.plane_session.corners["11"], float))
        d = np.asarray(lines["right"][1], float)
        d = d / float(np.linalg.norm(d))
        v = mid - np.asarray(lines["right"][0], float)
        assert abs(float(d[0] * v[1] - d[1] * v[0])) < 1.0, "the Right lock target is not the edge"
        before = dict(app.plane_session.corners)
        for side in ("left", "right"):
            app.snap_side_vars[side].set(True)
        app._maybe_snap_edges()
        for tag in ("00", "01", "10", "11"):
            assert app.plane_session.corners[tag] == pytest.approx(before[tag], abs=1e-6), tag
    finally:
        root.destroy()
