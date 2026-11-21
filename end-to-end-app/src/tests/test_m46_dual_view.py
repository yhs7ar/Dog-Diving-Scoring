"""M46 — the Calibrate pane's camera-pair switch (``dual_view.py``), display only.

Covers the pairing rules for both naming conventions (pure), and — when a Tk display exists — that
the switch shows the other camera's frame, a black "not found" frame when there is no pair, blocks
plane edits while the other feed is up, and hands the view back to the loaded clip.

Run:  <venv-python> -m pytest src/tests/test_m46_dual_view.py -s
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import dual_view  # noqa: E402


# -- pairing (pure) ---------------------------------------------------------------

@pytest.mark.parametrize("stem, expected", [
    ("live_jump_20261120_191517_right", ("right", "left", "live_jump_20261120_191517_left")),
    ("live_jump_20261120_191517_left", ("left", "right", "live_jump_20261120_191517_right")),
    ("live_jump_19691231_191517_right_sapper", ("right", "left", "live_jump_19691231_191517_left_sapper")),
    ("live_jump_19691231_191517_camera0_right", ("camera0", "camera1", "live_jump_19691231_191517_camera1_right")),
    ("live_jump_19691231_191517_camera1_right_sapper",
     ("camera1", "camera0", "live_jump_19691231_191517_camera0_right_sapper")),
    ("191230_camera0_right", ("camera0", "camera1", "191230_camera1_right")),
    ("Dog Dock Diving Slow Mo 2-seg01", None),
    ("video004_seg01", None),
])
def test_swap_camera_token(stem, expected) -> None:
    assert dual_view.swap_camera_token(stem) == expected


def _touch(p: Path) -> Path:
    p.write_bytes(b"")
    return p


def test_find_pair_recorder_convention(tmp_path: Path) -> None:
    right = _touch(tmp_path / "live_jump_20261120_191517_right.mp4")
    left = _touch(tmp_path / "live_jump_20261120_191517_left.mp4")
    pair = dual_view.find_pair(right)
    assert pair.partner == left and pair.clip_label == "right camera" and pair.partner_label == "left camera"
    assert dual_view.find_pair(left).partner == right


def test_find_pair_test_bench_convention(tmp_path: Path) -> None:
    c0 = _touch(tmp_path / "live_jump_19691231_191517_camera0_right_sapper.mp4")
    c1 = _touch(tmp_path / "live_jump_19691231_191517_camera1_right_sapper.mp4")
    pair = dual_view.find_pair(c1)
    assert pair.partner == c0 and pair.clip_label == "camera1" and pair.partner_label == "camera0"


def test_find_pair_missing_and_unnamed(tmp_path: Path) -> None:
    lonely = _touch(tmp_path / "live_jump_20261120_191517_right.mp4")
    pair = dual_view.find_pair(lonely)
    assert pair.partner is None
    assert pair.expected_name == "live_jump_20261120_191517_left.mp4"
    plain = dual_view.find_pair(_touch(tmp_path / "video004_seg01.mp4"))
    assert plain.partner is None and plain.expected_name is None


def test_find_pair_other_suffix(tmp_path: Path) -> None:
    right = _touch(tmp_path / "jump_right.mp4")
    left = _touch(tmp_path / "jump_left.mov")
    assert dual_view.find_pair(right).partner == left


# -- the Calibrate switch (needs a Tk display) ------------------------------------

def _write_clip(path: Path, bgr: tuple[int, int, int], n: int = 3, size=(160, 100)) -> Path:
    cv2 = pytest.importorskip("cv2")
    w, h = size
    wr = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))
    if not wr.isOpened():
        pytest.skip("cv2 cannot write mp4v here")
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:] = bgr
    for _ in range(n):
        wr.write(frame)
    wr.release()
    return path


def _app_or_skip():
    try:
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"no Tk display available: {type(exc).__name__}: {exc}")
    import app_gui
    return root, app_gui.App(root)


def _shown_mean_bgr(view) -> np.ndarray:
    return view._src[:, :, ::-1].reshape(-1, 3).mean(axis=0)   # _src is RGB


def _overlay_texts(view) -> list[str]:
    c = view.canvas
    return [c.itemcget(i, "text") for i in c.find_withtag("overlay") if c.type(i) == "text"]


def test_switch_shows_partner_and_returns(tmp_path: Path) -> None:
    right = _write_clip(tmp_path / "live_jump_20261120_191517_right.mp4", (0, 0, 200))   # red
    _write_clip(tmp_path / "live_jump_20261120_191517_left.mp4", (200, 0, 0))            # blue
    root, app = _app_or_skip()
    try:
        app.load_clip(right)
        pv = app.pair_view
        assert pv.pair is not None and pv.pair.partner is not None
        assert pv.btn_partner.cget("text") == "left camera"
        assert pv.btn_clip.cget("text") == "✓ right camera (loaded)"

        pv.btn_partner.invoke()
        assert pv.showing_partner
        assert pv.btn_partner.cget("text") == "✓ left camera"
        assert pv.btn_clip.cget("text") == "right camera (loaded)"
        b, g, r = _shown_mean_bgr(app.calib_view)
        assert b > 150 and r < 50, "the other camera's (blue) frame is not on the canvas"
        assert pv._block(None) == "break", "plane edits must be swallowed on the other feed"

        pv.choice.set("clip")
        pv._on_choice()
        assert not pv.showing_partner and pv._block(None) is None
        b, g, r = _shown_mean_bgr(app.calib_view)
        assert r > 150 and b < 50, "the loaded clip's (red) frame did not come back"
    finally:
        root.destroy()


def test_missing_pair_shows_black_with_message(tmp_path: Path) -> None:
    right = _write_clip(tmp_path / "live_jump_20261120_191517_right.mp4", (0, 0, 200))
    root, app = _app_or_skip()
    try:
        app.load_clip(right)
        pv = app.pair_view
        pv.choice.set("partner")
        pv._on_choice()
        assert float(app.calib_view._src.max()) == 0.0, "placeholder is not black"
        assert app.calib_view.image_size == (app.info.width, app.info.height)
        texts = _overlay_texts(app.calib_view)
        assert any("associated video for left camera not found" in t for t in texts), texts
    finally:
        root.destroy()


def test_loading_a_clip_leaves_the_partner_view(tmp_path: Path) -> None:
    right = _write_clip(tmp_path / "live_jump_20261120_191517_right.mp4", (0, 0, 200))
    left = _write_clip(tmp_path / "live_jump_20261120_191517_left.mp4", (200, 0, 0))
    root, app = _app_or_skip()
    try:
        app.load_clip(right)
        pv = app.pair_view
        pv.choice.set("partner")
        pv._on_choice()
        app.load_clip(left)
        assert not pv.showing_partner and pv.choice.get() == "clip"
        assert pv.pair.clip == left and pv.pair.partner == right
        assert app.calib_view.overlay == app._draw_plane_overlay, "plane overlay not restored"
    finally:
        root.destroy()
