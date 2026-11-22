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


def _photo(view) -> np.ndarray:
    """The photo inside the dark margin (``content_box``), as BGR."""
    x0, y0, x1, y1 = (int(round(v)) for v in view.content_box)
    return view._src[y0:y1, x0:x1, ::-1]                          # _src is RGB


def _shown_mean_bgr(view) -> np.ndarray:
    return _photo(view).reshape(-1, 3).mean(axis=0)


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
        photo = _photo(app.calib_view)
        assert float(photo.max()) == 0.0, "placeholder is not black"
        assert photo.shape[:2] == (app.info.height, app.info.width)
        texts = _overlay_texts(app.calib_view)
        assert any("associated video for left camera not found" in t for t in texts), texts
    finally:
        root.destroy()


def test_partner_sits_where_the_loaded_clip_sits(tmp_path: Path) -> None:
    """Both feeds use the plane's margin + content box, so they land in the same place on screen."""
    right = _write_clip(tmp_path / "live_jump_20261120_191517_right.mp4", (0, 0, 200))
    _write_clip(tmp_path / "live_jump_20261120_191517_left.mp4", (200, 0, 0))
    root, app = _app_or_skip()
    try:
        app.load_clip(right)
        app.on_new_plane(seed=False)                  # the loaded clip, drawn by app_gui itself
        clip_box, clip_size = app.calib_view.content_box, app.calib_view.image_size
        assert clip_box is not None
        app.pair_view.btn_partner.invoke()
        assert app.calib_view.content_box == clip_box
        assert app.calib_view.image_size == clip_size
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


# -- the derived plane on the paired camera (H12) ----------------------------------

import pair_homography as ph  # noqa: E402

_H12_TRUE = np.array([[0.92, 0.06, 18.0],
                      [-0.03, 0.97, 9.0],
                      [0.00012, -0.00005, 1.0]])


def _textured(size=(480, 300), seed=3) -> np.ndarray:
    """A feature-rich synthetic "pool": random blobs + lines, so matching has something to bite."""
    cv2 = pytest.importorskip("cv2")
    w, h = size
    rng = np.random.default_rng(seed)
    img = np.full((h, w, 3), 90, np.uint8)
    for _ in range(160):
        c = tuple(int(v) for v in rng.integers(0, 255, 3))
        x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
        if rng.random() < 0.5:
            cv2.circle(img, (x, y), int(rng.integers(3, 14)), c, -1)
        else:
            cv2.rectangle(img, (x, y), (x + int(rng.integers(4, 25)), y + int(rng.integers(4, 25))), c, -1)
    return img


def _warp(img: np.ndarray, H: np.ndarray) -> np.ndarray:
    cv2 = pytest.importorskip("cv2")
    return cv2.warpPerspective(img, H, (img.shape[1], img.shape[0]))


def test_estimate_h12_recovers_a_known_mapping() -> None:
    a = _textured()
    b = _warp(a, _H12_TRUE)
    r = ph.estimate_h12_from_images(a, b)
    assert r.ok, r.message
    pts = np.array([[100, 80], [380, 90], [360, 240], [120, 230]], float)
    err = np.abs(ph.map_points(r.H, pts) - ph.map_points(_H12_TRUE, pts)).max()
    assert err < 1.5, f"H12 off by {err:.2f} px"


def test_estimate_h12_fails_cleanly_on_unrelated_images() -> None:
    r = ph.estimate_h12_from_images(_textured(seed=1), _textured(seed=2))
    assert not r.ok and r.message


def test_plane_mask_and_compose() -> None:
    H1 = np.array([[10.0, 0, 20], [0, 10.0, 30], [0, 0, 1]])        # 1 ft = 10 px
    m = ph.plane_mask((300, 480), H1, length_ft=35, width_ft=20, margin_px=0)
    assert m[30 + 100, 20 + 175] == 255 and m[5, 5] == 0
    H2 = ph.compose(_H12_TRUE, H1)
    want = ph.map_points(_H12_TRUE, ph.map_points(H1, [[7.0, 4.0]]))
    assert np.allclose(ph.map_points(H2, [[7.0, 4.0]]), want)


def _write_frames(path: Path, frame: np.ndarray, n: int = 3) -> Path:
    cv2 = pytest.importorskip("cv2")
    h, w = frame.shape[:2]
    wr = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))
    if not wr.isOpened():
        pytest.skip("cv2 cannot write mp4v here")
    for _ in range(n):
        wr.write(frame)
    wr.release()
    return path


def _real_reader(monkeypatch) -> None:
    """Older tests (m20, m25) replace ``app_gui.read_frame`` without restoring it; these tests need
    the loaded clip's real pixels, so pin the real reader for their duration."""
    import app_gui
    monkeypatch.setattr(app_gui, "read_frame", lambda clip, idx: dual_view.read_first_frame(clip, idx))


def _overlay_lines(view) -> int:
    c = view.canvas
    return sum(1 for i in c.find_withtag("overlay") if c.type(i) == "line")


def test_paired_view_draws_the_derived_plane(tmp_path: Path, monkeypatch) -> None:
    _real_reader(monkeypatch)
    a = _textured()
    right = _write_frames(tmp_path / "jump_right.mp4", a)
    _write_frames(tmp_path / "jump_left.mp4", _warp(a, _H12_TRUE))
    root, app = _app_or_skip()
    try:
        app.load_clip(right)
        pv = app.pair_view
        pv.btn_partner.invoke()                              # no plane yet: frame only, a hint
        assert _overlay_lines(app.calib_view) == 0
        assert "fit the plane" in pv.note.cget("text")

        pv.btn_clip.invoke()
        app.on_new_plane(seed=False)
        pv.btn_partner.invoke()
        assert pv.h12 is not None and pv.h12.ok, pv.h12 and pv.h12.message
        assert _overlay_lines(app.calib_view) > 10, "derived plane not drawn"
        assert "derived via H12" in pv.note.cget("text")
        assert pv._block(None) == "break", "the paired view must stay read-only"

        # The derived corner = H12(loaded corner), within a few px of the true mapping.
        s = app.plane_session
        H2 = ph.compose(pv.h12.H, s.homography())
        got = ph.map_points(H2, [[0.0, 0.0]])[0]
        want = ph.map_points(_H12_TRUE, [s.corners["00"]])[0]
        assert np.abs(got - want).max() < 3.0
    finally:
        root.destroy()


def test_h12_source_is_swappable(tmp_path: Path, monkeypatch) -> None:
    _real_reader(monkeypatch)
    a = _textured()
    right = _write_frames(tmp_path / "jump_right.mp4", a)
    _write_frames(tmp_path / "jump_left.mp4", a)
    root, app = _app_or_skip()
    try:
        app.load_clip(right)
        app.on_new_plane(seed=False)
        calls = []

        def fake(f1, f2, mask):
            calls.append(mask is not None)
            return ph.H12Result(np.eye(3), "identity (test source)", source="test")

        app.pair_view.set_h12_source(fake)
        app.pair_view.btn_partner.invoke()
        assert calls == [True], "source not called once with the plane mask"
        assert "identity (test source)" in app.pair_view.note.cget("text")
        app.pair_view.btn_clip.invoke()
        app.pair_view.btn_partner.invoke()
        assert calls == [True], "an unchanged plane must reuse the cached H12"
    finally:
        root.destroy()
