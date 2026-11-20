"""M5 — scoring registry contract + arc_2d.

Covers the standardized-contract deliverable: every method's ``ScoreResult``
validates against one schema, methods register/unregister through the registry,
and the v1 ``arc_2d`` method runs deterministically on the parent's cached seg01
detections with each axis behaving as documented.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1]
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SRC))

from test_bench.scoring import contract, registry  # noqa: E402
from test_bench.scoring.arc_2d import arc_2d, _select  # noqa: E402
from test_bench.scoring.arc_2d_centerline import arc_2d_centerline  # noqa: E402

CLIP_DIR = REPO / "end-to-end-app" / "out" / "video004_right-00.00.36.812-00.00.38.328-seg01"
PLANE = REPO / "end-to-end-app" / "planes" / "plane_test-clip-video004_right.pool.json"
MODEL_SHA = "bdb7688775ee39201d4a77ef0041c7c3e9eb9202f7a5df287a11c98e80ef82e4"


# --- contract ---------------------------------------------------------------

def test_new_result_is_valid():
    assert contract.validate(contract.new_result("arc_2d", {"n_points": 8})) == []


def test_validate_catches_problems():
    r = contract.new_result("arc_2d", {})
    del r["entry_px"]
    assert "entry_px" in " ".join(contract.validate(r))
    r2 = contract.new_result("arc_2d", {})
    r2["method"] = "bogus"
    assert contract.validate(r2)
    r3 = contract.new_result("arc_2d", {})
    r3["status"] = "weird"
    assert contract.validate(r3)
    r4 = contract.new_result("arc_2d", {})
    r4["entry_px"] = [1, 2, 3]
    assert contract.validate(r4)


# --- registry ---------------------------------------------------------------

def test_registry():
    assert "arc_2d" in registry.list_methods()
    assert registry.get("arc_2d") is arc_2d
    with pytest.raises(KeyError):
        registry.get("arc_3d")  # reserved slot, not yet implemented
    with pytest.raises(ValueError):
        registry.register("not_a_method", arc_2d)
    with pytest.raises(ValueError):
        registry.register("arc_2d", arc_2d)


# --- selection rules --------------------------------------------------------

def _pts(n=10):
    return [{"frame": i, "x": float(20 + i * 30), "y": float(400 - i * 5),
             "kp_conf": float((i % 3) + 1) / 4.0} for i in range(n)]


def test_select_rules():
    p = _pts(10)
    assert [q["frame"] for q in _select(p, "all", "all")] == list(range(10))
    assert [q["frame"] for q in _select(p, "last_n", 3)] == [7, 8, 9]
    assert len(_select(p, "top_kp_conf_n", 4)) == 4
    assert len(_select(p, "flight_run_only", "all")) == 10  # single run
    assert len(_select(p, "ransac_inliers", "all")) >= 5  # clean parabola -> mostly inliers


# --- arc_2d on real seg01 data ----------------------------------------------

@pytest.fixture(scope="module")
def seg01():
    if not (CLIP_DIR / "tracks.json").is_file():
        pytest.skip("parent cached artifacts not present on this machine")
    from test_bench import analyze as A
    from dog_diving import trajectory
    records, _ = A.load_cached_records(CLIP_DIR / "tracks.json", model_sha=MODEL_SHA)
    points = trajectory.assemble(records)
    plane = json.loads(PLANE.read_text(encoding="utf-8"))
    return points, plane


def test_arc_2d_valid_and_deterministic(seg01):
    points, plane = seg01
    cfg = {"selection": "all", "n_points": "all", "frame_size": [960, 600]}
    r1 = registry.run("arc_2d", points, plane, cfg)
    r2 = registry.run("arc_2d", points, plane, cfg)
    assert r1["status"] == "ok"
    assert r1["distance_in"] > 0
    assert contract.validate(r1) == []
    assert r1 == r2


def test_arc_2d_sweep_n_points(seg01):
    points, plane = seg01
    for n in (4, 6, 8, 12, 20, "all"):
        r = registry.run("arc_2d", points, plane,
                         {"selection": "last_n", "n_points": n, "frame_size": [960, 600]})
        assert contract.validate(r) == []
        if n == "all":
            assert r["status"] == "ok"
        else:
            # a low n_points can legitimately fail the x-span gate (no_fit) — that
            # is a real finding the table reports, not an error.
            assert r["status"] in ("ok", "few_points", "no_fit")


def test_arc_2d_few_points(seg01):
    points, plane = seg01
    r = registry.run("arc_2d", points, plane,
                     {"selection": "last_n", "n_points": 2, "frame_size": [960, 600]})
    assert r["status"] == "few_points"


def test_arc_2d_no_calibration(seg01):
    points, _ = seg01
    r = registry.run("arc_2d", points, None,
                     {"selection": "all", "n_points": "all", "frame_size": [960, 600]})
    assert r["status"] == "no_calibration"


def test_arc_2d_emits_selected_px_and_curve_space(seg01):
    points, plane = seg01
    r = registry.run("arc_2d", points, plane,
                     {"selection": "last_n", "n_points": 8, "frame_size": [960, 600]})
    fit = r["fit"]
    assert fit["curve_space"] == "px"
    expected = [[float(p["x"]), float(p["y"])] for p in _select(points, "last_n", 8)]
    assert fit["selected_px"] == expected
    assert fit["selected_frames"] == [int(p["frame"]) for p in _select(points, "last_n", 8)]


# --- arc_2d_centerline -------------------------------------------------------

def test_arc_2d_centerline_registered():
    assert "arc_2d_centerline" in registry.list_methods()
    assert registry.get("arc_2d_centerline") is arc_2d_centerline


def test_arc_2d_centerline_valid_and_deterministic(seg01):
    points, plane = seg01
    cfg = {"selection": "all", "n_points": "all", "frame_size": [960, 600]}
    r1 = registry.run("arc_2d_centerline", points, plane, cfg)
    r2 = registry.run("arc_2d_centerline", points, plane, cfg)
    assert r1["status"] == "ok"
    assert r1["distance_in"] > 0
    assert contract.validate(r1) == []
    assert r1 == r2


def test_arc_2d_centerline_lands_on_centerline(seg01):
    """The method's promise: the scored entry sits on the pool centerline (Y == yc)."""
    import numpy as np
    from dog_diving.pool import calib, homography, lens as _lens
    points, plane = seg01
    r = registry.run("arc_2d_centerline", points, plane,
                     {"selection": "all", "n_points": "all", "frame_size": [960, 600]})
    yc = r["fit"]["centerline_yc_ft"]
    ln = calib.lens_from_document(plane)
    H = np.asarray(plane["homography"], float).reshape(3, 3)
    pxu = _lens.undistort_points(np.array([r["entry_px"]], float), ln)[0]
    XY = homography.apply_homography_inv(H, np.array([[pxu[0], pxu[1]]], float))
    assert abs(float(XY[0, 1]) - yc) < 1e-6


def test_arc_2d_centerline_differs_from_arc_2d(seg01):
    """The whole point: the centerline crossing is not the terminal point."""
    points, plane = seg01
    cfg = {"selection": "all", "n_points": "all", "frame_size": [960, 600]}
    a = registry.run("arc_2d", points, plane, cfg)
    c = registry.run("arc_2d_centerline", points, plane, cfg)
    assert a["status"] == "ok" and c["status"] == "ok"
    assert c["distance_in"] != a["distance_in"]


def test_arc_2d_centerline_few_points(seg01):
    points, plane = seg01
    r = registry.run("arc_2d_centerline", points, plane,
                     {"selection": "last_n", "n_points": 2, "frame_size": [960, 600]})
    assert r["status"] == "few_points"


def test_arc_2d_centerline_no_calibration(seg01):
    points, _ = seg01
    r = registry.run("arc_2d_centerline", points, None,
                     {"selection": "all", "n_points": "all", "frame_size": [960, 600]})
    assert r["status"] == "no_calibration"


def test_arc_2d_centerline_no_centerline_source(seg01):
    """No extent y-bounds and no corners -> the centerline is undefined -> error."""
    points, plane = seg01
    p = json.loads(json.dumps(plane))
    p["extent"] = {}
    p["plane"]["corners"] = {}
    p["plane"]["length_ft"] = None
    p["plane"]["width_ft"] = None
    r = registry.run("arc_2d_centerline", points, p,
                     {"selection": "all", "n_points": "all", "frame_size": [960, 600]})
    assert r["status"] == "error"


def test_arc_2d_centerline_centerline_from_corners(seg01):
    """Extent without y-bounds falls back to plane corners: yc == width/2."""
    points, plane = seg01
    p = json.loads(json.dumps(plane))
    p["extent"] = {"x_min": 0.0, "x_max": 35.0}  # no y bounds
    r = registry.run("arc_2d_centerline", points, p,
                     {"selection": "all", "n_points": "all", "frame_size": [960, 600]})
    assert r["status"] == "ok"
    assert abs(r["fit"]["centerline_yc_ft"] - plane["plane"]["width_ft"] / 2.0) < 1e-9
