"""Tests for detect_waterline.py. Run with:  python -m pytest tests -q"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import detect_waterline as dw  # noqa: E402

H, W, N = 80, 120, 60


def synthetic_stack(boundary: int = 40, side: str = "below", drift: float = 0.0, seed: int = 0):
    """Static texture on one side of `boundary`, flickering 'water' on the other."""
    rng = np.random.default_rng(seed)
    static = rng.integers(80, 160, (H, W)).astype(np.float32)
    yy, xx = np.mgrid[0:H, 0:W]
    water = {"below": yy >= boundary, "above": yy < boundary,
             "right": xx >= boundary, "left": xx < boundary}[side]
    frames = []
    for t in range(N):
        f = static.copy()
        f[water] = 100 + 15 * np.sin(0.3 * xx[water] + 0.2 * yy[water] - 0.7 * t)
        f += rng.normal(0, 1.0, (H, W)) + drift * t / (N - 1)
        frames.append(np.clip(f, 0, 255).astype(np.uint8))
    return np.stack(frames), water


def test_noise_maps_higher_on_water():
    frames, water = synthetic_stack()
    maps = dw.compute_noise_maps(frames, pre_blur=0)
    for m in dw.NOISE_METHODS:
        assert maps[m][water].mean() > 5 * maps[m][~water].mean(), m


def test_variance_matches_numpy():
    frames, _ = synthetic_stack()
    maps = dw.compute_noise_maps(frames, pre_blur=0)
    np.testing.assert_allclose(maps["variance"], frames.astype(np.float64).var(axis=0), rtol=1e-4, atol=1e-3)


def test_detrended_removes_linear_drift():
    rng = np.random.default_rng(1)
    base = rng.integers(50, 150, (H, W)).astype(np.float32)
    frames = np.stack([np.clip(base + 40 * t / (N - 1), 0, 255).astype(np.uint8) for t in range(N)])
    maps = dw.compute_noise_maps(frames, pre_blur=0)
    assert maps["variance"].mean() > 100      # ~40^2/12 = 133
    assert maps["detrended"].mean() < 1.0     # only quantisation residue left


def test_diff_lag():
    frames, water = synthetic_stack()
    d1 = dw.compute_noise_maps(frames, pre_blur=0, diff_lag=1)["diff"]
    d4 = dw.compute_noise_maps(frames, pre_blur=0, diff_lag=4)["diff"]
    assert d4[water].mean() > d1[water].mean()  # slow ripples: larger lag sees more change


def test_parse_roi():
    box = dw.parse_roi("10,20,50,60")
    assert box.shape == (4, 2) and box.min(axis=0).tolist() == [10, 20]
    poly = dw.parse_roi("0,0;10,0;5,8")
    assert poly.shape == (3, 2)
    assert dw.parse_roi(None) is None
    with pytest.raises(ValueError):
        dw.parse_roi("1,2,3")


def test_threshold_manual_and_otsu():
    m = np.zeros((10, 10), np.uint8)
    m[5:] = 200
    mask, t = dw.threshold_noise_map(m, "otsu")
    assert mask[5:].all() and not mask[:5].any()
    mask, t = dw.threshold_noise_map(m, "manual", 250)
    assert not mask.any() and t == 250
    with pytest.raises(ValueError):
        dw.threshold_noise_map(m, "manual")


def test_clean_mask_removes_specks_and_holes():
    m = np.zeros((H, W), np.uint8)
    m[40:] = 255
    m[60, 60] = 0           # hole in water
    m[5, 5] = 255           # speck on structure
    out = dw.clean_mask(m, 3, 5)
    assert out[60, 60] == 255 and out[5, 5] == 0


@pytest.mark.parametrize("side", ["below", "above", "left", "right"])
def test_end_to_end_on_arrays(side):
    boundary = 40
    frames, _ = synthetic_stack(boundary, side)
    roi = np.full((H, W), 255, np.uint8)
    r = dw.detect_in_window(frames, roi, dw.PipelineParams(method="variance"))
    assert r["side"] == side
    pts = r["points"]
    pos = pts[:, 1] if side in ("below", "above") else pts[:, 0]
    expected = boundary if side in ("below", "right") else boundary - 1
    assert len(pts) > 0.9 * (W if side in ("below", "above") else H)
    assert np.abs(pos - expected).mean() < 1.5


def test_roi_edge_is_not_a_waterline():
    # Water from y=30 down. Left part of the ROI starts at y=20 (real edge
    # inside the ROI); right part starts at y=50 (ROI cuts the water).
    water = np.zeros((H, W), np.uint8)
    water[30:] = 255
    roi = dw.roi_to_mask(dw.parse_roi("0,20;60,20;60,50;119,50;119,79;0,79"), (H, W))
    mask = water & roi
    pts, _ = dw.extract_waterline(mask, roi, "scan", "below", smooth=0)
    assert set(pts[:, 1]) == {30}
    assert pts[:, 0].max() <= 60
    pts, _ = dw.extract_waterline(mask, roi, "contour", "below")
    assert np.all(pts[:, 1] == 30)


def test_contour_method():
    frames, _ = synthetic_stack(40)
    roi = np.full((H, W), 255, np.uint8)
    r = dw.detect_in_window(frames, roi, dw.PipelineParams(method="variance", line_method="contour"))
    assert len(r["points"]) > 0.8 * W
    assert np.abs(r["points"][:, 1] - 40).mean() < 1.5


def test_stability_low_on_calm_water():
    frames, _ = synthetic_stack(40)
    roi = np.full((H, W), 255, np.uint8)
    p = dw.PipelineParams(method="variance")
    r = dw.detect_in_window(frames, roi, p)
    st = dw.waterline_stability(frames, roi, p, r["norm"], r["side"], window=30)["stats"]
    assert st["n_windows"] >= 2
    assert st["per_position_std_median"] < 1.0


def test_fit_straight_line_ignores_outliers():
    x = np.arange(200, dtype=np.float64)
    y = 0.1 * x + 50
    y[90:100] += 40          # a glare blob that fooled the mask
    fit = dw.fit_straight_line(np.stack([x, y], axis=1), "below")
    assert abs(fit["slope"] - 0.1) < 0.005
    assert abs(fit["intercept"] - 50) < 1.0
    assert abs(fit["angle_deg"] - np.degrees(np.arctan(0.1))) < 0.3
    assert fit["max_residual"] > 30           # the outliers are reported, not hidden
    np.testing.assert_allclose(fit["endpoints"][0][0], 0, atol=1)
    np.testing.assert_allclose(fit["endpoints"][1][0], 199, atol=1)


def test_fit_straight_line_vertical():
    y = np.arange(100, dtype=np.float64)
    x = 30 + 0.05 * y
    fit = dw.fit_straight_line(np.stack([x, y], axis=1), "right")
    assert fit["form"].startswith("x =")
    assert abs(fit["slope"] - 0.05) < 1e-3 and abs(fit["intercept"] - 30) < 0.1
    assert fit["rms_residual"] < 1e-3


def test_fit_flags_bent_ends():
    # A straight edge bowed at both ends, like barrel lens distortion.
    x = np.linspace(-1, 1, 300)
    y = 100 + 20 * x ** 4
    fit = dw.fit_straight_line(np.stack([x * 300 + 300, y], axis=1))
    assert fit["rms_ends"] > 2 * fit["rms_center"]


def test_cli_on_synthetic_video(tmp_path):
    from make_synthetic_video import make
    vid = str(tmp_path / "syn")
    make(vid, calm=90, active=10, drift=20)
    out = tmp_path / "out"
    rc = dw.main(["--video", vid + ".mp4", "--end-frame", "90", "--roi", "0,100,640,300",
                  "--output-dir", str(out), "--scale", "0.5", "--fit", "line"])
    assert rc == 0
    truth = np.array(json.load(open(vid + "_truth.json"))["waterline_y"])
    pts = np.load(out / "waterline.npy")
    err = pts[:, 1] - truth[np.clip(pts[:, 0].astype(int), 0, len(truth) - 1)]
    assert np.abs(err).mean() < 2.0   # within 2 px at original resolution despite 0.5x processing
    for f in ["waterline.json", "overlay.png", "noise_map_comparison.png", "mask_clean.png"]:
        assert (out / f).exists()
    # The synthetic waterline is slanted (slope 0.08) with a gentle wave, so a
    # straight fit should recover roughly that slope with a few px of residual.
    fit = json.load(open(out / "waterline.json"))["line_fit"]
    assert abs(fit["slope"] - 0.08) < 0.02
    assert fit["rms_residual"] < 6
