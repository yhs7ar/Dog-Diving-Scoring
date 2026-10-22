"""Tests for detect_waterline_lite.py. Run with:  python -m pytest tests -q"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import detect_waterline_lite as lite  # noqa: E402


def test_noise_map_ignores_linear_drift():
    rng = np.random.default_rng(0)
    base = rng.integers(50, 150, (40, 60)).astype(np.float32)
    frames = np.stack([np.clip(base + 40 * t / 29, 0, 255).astype(np.uint8) for t in range(30)])
    assert lite.noise_map(frames).mean() < 1.0      # plain variance would be ~130


def test_fit_line_rejects_outliers():
    x = np.arange(500, dtype=np.float64)
    y = -0.02 * x + 300 + np.random.default_rng(1).normal(0, 1, x.size)
    y[100:160] -= 60                                 # e.g. a person merged into the water
    fit, inlier = lite.fit_line(np.stack([x, y], axis=1))
    assert abs(fit["slope"] + 0.02) < 0.002 and abs(fit["intercept"] - 300) < 1
    assert not inlier[100:160].any() and inlier[200:].mean() > 0.95
    assert fit["rms_residual_px"] < 2


def test_roi_edge_is_not_a_waterline():
    mask = np.zeros((80, 100), np.uint8)
    mask[30:] = 255
    roi = lite.roi_mask(mask.shape, "0,40,100,80")  # ROI cuts the water at y=40
    assert len(lite.scan_waterline(mask & roi, roi)) == 0
    roi = lite.roi_mask(mask.shape, "0,20,100,80")  # real edge at y=30 inside the ROI
    pts = lite.scan_waterline(mask & roi, roi)
    assert len(pts) == 100 and np.all(pts[:, 1] == 30)


def test_cli_on_synthetic_video(tmp_path):
    from make_synthetic_video import make
    vid = str(tmp_path / "syn")
    make(vid, calm=60, active=0, drift=20)
    out = tmp_path / "out"
    lite.main(["--video", vid + ".mp4", "--end-frame", "60", "--roi", "0,100,640,300",
               "--output-dir", str(out)])
    res = json.load(open(out / "waterline.json"))
    truth = np.array(json.load(open(vid + "_truth.json"))["waterline_y"])
    pts = np.array(res["points_xy"])
    assert np.abs(pts[:, 1] - truth[pts[:, 0].astype(int)]).mean() < 2.0
    assert abs(res["line"]["slope"] - 0.08) < 0.02   # synthetic edge slopes at ~0.08
    for f in ["overlay.png", "noise_map.png", "mask.png"]:
        assert (out / f).exists()

    # Same clip selected in seconds (synthetic video is 30 fps) gives the same result.
    out_t = tmp_path / "out_t"
    lite.main(["--video", vid + ".mp4", "--start-time", "0", "--end-time", "2.0",
               "--roi", "0,100,640,300", "--output-dir", str(out_t)])
    res_t = json.load(open(out_t / "waterline.json"))
    assert res_t["frame_range"] == [0, 60] and res_t["time_range_s"] == [0.0, 2.0]
    assert res_t["line"] == res["line"]
