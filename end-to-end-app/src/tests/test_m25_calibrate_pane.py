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
    return WL.WaterlineEdges(far=far, near=near, dock=dock, left=left, right=right,
                             dock_side=dock_side, frame_size=(960, 600))


# ===========================================================================
# 1. the waterline module: both short ends
# ===========================================================================
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
        for wanted in ("New Plane", "Re-detect Waterline", "Flip Plane (left-right)", "Complete"):
            assert wanted in texts, f"{wanted!r} missing from the Calibrate tab: {texts}"
        # the two redundant buttons and the old names are gone
        for gone in ("Make parallelogram", "Fit to frame", "Flip X", "Build document",
                     "Save plane", "calibration frame index:", "Show 1-ft grid"):
            assert gone not in texts, f"{gone!r} is still a control: {texts}"
        # the order is the operator's
        order = [t for t in texts if t in ("New Plane", "Re-detect Waterline",
                                           "Flip Plane (left-right)", "Complete")]
        assert order == ["New Plane", "Re-detect Waterline", "Flip Plane (left-right)", "Complete"]
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
            assert btn.cget("text") in ("New Plane", "Complete")
            assert btn.cget("bg").lower() == app_gui.PRIMARY_BG.lower(), btn.cget("bg")
            assert btn.cget("fg").lower() == "white", btn.cget("fg")
    finally:
        root.destroy()


def test_save_new_plane_sits_with_the_saved_plane_picker():
    root = _root_or_skip()
    try:
        app = app_gui.App(root, clip=None)
        app.clip = Path("synthetic.mp4")
        app.confirm_reset = lambda: True
        root.update()
        assert app.save_plane_btn.cget("text") == "Save New Plane"
        assert app.save_plane_btn.master is app.plane_pick.master, (
            "Save New Plane is not in the saved-plane row")
        assert app.save_plane_btn.pack_info()["side"] == "left"
    finally:
        root.destroy()


def test_the_anchor_fields_are_labelled_and_default_to_0_and_35():
    root = _root_or_skip()
    try:
        app = app_gui.App(root, clip=None)
        app.clip = Path("synthetic.mp4")
        app.confirm_reset = lambda: True
        root.update()
        texts = _texts(app.tab_calib)
        assert "• Anchor (near dock):" in texts, texts
        assert "• Anchor (opposite dock):" in texts, texts
        assert app.plane_near_var.get() == "0"
        assert app.plane_len_var.get() == "35"
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
        assert "(OFF)" in app.snap_btn.cget("text")
        assert [lab for _side, lab in app_gui.SNAP_SIDE_LABELS] == \
               ["Top", "Bottom", "Left", "Right"]
        assert set(app.snap_side_vars) == {"back", "front", "left", "right"}
        # only the far waterline is held to begin with (operator order 2026-10-21)
        assert app.snap_side_vars["back"].get() is True
        for side in ("front", "left", "right"):
            assert app.snap_side_vars[side].get() is False, side
        app.on_toggle_snap()
        assert app.snap_on_var.get() is True
        assert "(ON)" in app.snap_btn.cget("text")
        app.on_toggle_snap()
        assert app.snap_on_var.get() is False
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
        # the toggle is armed by the button itself
        assert app.snap_on_var.get() is True
        assert "(ON)" in app.snap_btn.cget("text")
        # every checked edge is on its detected line
        s = app.plane_session
        lines = app._snap_lines()
        assert {"back", "front", "left", "right"} <= set(lines), lines
        # ONLY the checked edge is held: Top to the detected far waterline.  The other three are
        # free — pinning all four would make the quad impossible to drag by hand.
        point, direction = lines["back"]
        for tag in ("00", "10"):
            q = np.asarray(s.corners[tag], float)
            foot = point + direction * float(np.dot(q - point, direction))
            assert float(np.hypot(*(q - foot))) < 5.0, (tag, q, foot)
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
        # the pane's default is Top only, and Top is the far waterline
        assert app.snap_side_vars["back"].get() is True
        before = dict(app.plane_session.corners)
        app.snap_on_var.set(True)
        assert app._maybe_snap_edges() is True
        # Bottom, Left and Right are unchecked: 01 and 11 belong to those and must not move
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
    root = _root_or_skip()
    try:
        app = app_gui.App(root, clip=None)
        app.clip = Path("synthetic.mp4")
        app.confirm_reset = lambda: True
        root.update()
        app.on_new_plane(seed=False)
        app.seed_edges = _scene_edges("left")
        app.dock_side_var.set("left")
        lines = app._snap_lines()
        assert np.allclose(lines["left"][0], app.seed_edges.left.fit["point"])

        # after a flip the X = 0 edge is at the other image end, and the toggle follows it there
        app.plane_session.flip_x()
        lines = app._snap_lines()
        assert np.allclose(lines["left"][0], app.seed_edges.right.fit["point"])
    finally:
        root.destroy()
