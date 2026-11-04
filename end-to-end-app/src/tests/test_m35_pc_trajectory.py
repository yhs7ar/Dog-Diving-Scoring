"""Tests for ``dog_diving.pc_trajectory`` — 3-D trajectory from per-frame point clouds (PC-TRAJ).

All tests run offline (no pyzed, no GPU): the point-cloud lookup, the metres->feet + ZED->world
re-axing, and the trajectory/plane/crossing chain are exercised against a **synthetic** fixture
(a flat water plane at ``Z = 0`` with a parabola of dog points above it), which is the whole
deliverable of step 13.  The real ``frame_{i:06d}.npy`` read (NEURAL depth on the Jetson) is
hardware-verify-pending — that is step 12's output consumed here.
"""

from __future__ import annotations

import numpy as np
import pytest

from dog_diving import pc_trajectory as PC
from dog_diving.pc_trajectory import FEET_PER_METRE

FPS = 30.0
G = 32.174


# ============================================================================
# coordinate transforms
# ============================================================================
def test_metres_to_feet_scales_only() -> None:
    assert np.allclose(PC.metres_to_feet([1.0, 2.0, 3.0]),
                       [FEET_PER_METRE, 2 * FEET_PER_METRE, 3 * FEET_PER_METRE])


def test_zed_to_world_maps_and_scales() -> None:
    # ZED (x right, y up, z forward) -> world (X = z, Y = x, Z = y), metres -> feet.
    zed = np.array([1.0, 2.0, 3.0])  # metres
    world = PC.zed_to_world(zed)
    assert np.allclose(world, np.array([3.0, 1.0, 2.0]) * FEET_PER_METRE)


def test_zed_to_world_round_trips() -> None:
    # world feet -> ZED metres is the inverse of zed_to_world; a point survives the loop.
    X, Y, Z = 8.16, -0.5, 2.0  # world feet
    zed = np.array([Y, Z, X]) / FEET_PER_METRE  # metres, ZED frame
    assert np.allclose(PC.zed_to_world(zed), [X, Y, Z], atol=1e-9)


# ============================================================================
# lookup
# ============================================================================
def test_frame_path_naming() -> None:
    assert PC.frame_path("out", 7).name == "frame_000007.npy"


def test_lookup_point_indexes_row_col() -> None:
    cloud = np.full((10, 12, 3), np.nan, dtype=np.float32)
    cloud[4, 6] = [1.0, 2.0, 3.0]  # row 4, col 6 -> pixel (x=6, y=4)
    assert np.allclose(PC.lookup_point(cloud, 6.0, 4.0), [1.0, 2.0, 3.0])


def test_lookup_point_rounds_to_nearest_pixel() -> None:
    cloud = np.full((10, 12, 3), np.nan, dtype=np.float32)
    cloud[4, 6] = [1.0, 2.0, 3.0]
    assert np.allclose(PC.lookup_point(cloud, 6.4, 3.6), [1.0, 2.0, 3.0])


def test_lookup_point_rejects_nonfinite() -> None:
    cloud = np.full((10, 12, 3), np.nan, dtype=np.float32)
    cloud[4, 6] = [np.nan, 0.0, 1.0]
    assert PC.lookup_point(cloud, 6.0, 4.0) is None
    cloud[4, 6] = [np.inf, 0.0, 1.0]
    assert PC.lookup_point(cloud, 6.0, 4.0) is None


def test_lookup_point_rejects_out_of_bounds() -> None:
    cloud = np.zeros((10, 12, 3), dtype=np.float32)
    assert PC.lookup_point(cloud, -1.0, 4.0) is None
    assert PC.lookup_point(cloud, 6.0, 10.0) is None  # row == height (out)
    assert PC.lookup_point(cloud, 12.0, 4.0) is None  # col == width (out)


# ============================================================================
# points_at_detections
# ============================================================================
def _write_frames(tmp_path, frames: dict[int, np.ndarray]) -> None:
    d = tmp_path / "frames"
    d.mkdir(exist_ok=True)
    for i, cloud in frames.items():
        np.save(d / f"frame_{i:06d}.npy", np.asarray(cloud, dtype=np.float32))
    return d


def test_points_at_detections_round_trip(tmp_path) -> None:
    # A dog point in world feet, stored in ZED metres at a known pixel.
    X, Y, Z = 8.16, -0.5, 2.0
    zed = np.array([Y, Z, X]) / FEET_PER_METRE
    cloud = np.full((8, 8, 3), np.nan, dtype=np.float32)
    cloud[3, 5] = zed
    d = _write_frames(tmp_path, {2: cloud})
    dets = [{"frame": 2, "x": 5.0, "y": 3.0}]

    r = PC.points_at_detections(d, dets)
    assert r["n_valid"] == 1 and r["n_skipped"] == 0
    p = r["points"][0]
    assert p["frame"] == 2 and p["x"] == 5.0 and p["y"] == 3.0
    assert np.allclose([p["X_ft"], p["Y_ft"], p["Z_ft"]], [X, Y, Z], atol=1e-5)
    assert np.allclose(p["pos_m"], zed, atol=1e-6)


def test_points_at_detections_skips_missing_frame(tmp_path) -> None:
    cloud = np.full((4, 4, 3), np.nan, dtype=np.float32)
    cloud[1, 1] = [1.0, 2.0, 3.0]
    d = _write_frames(tmp_path, {0: cloud})
    r = PC.points_at_detections(d, [(0, 1, 1), (9, 1, 1)])
    assert r["n_valid"] == 1 and r["n_skipped"] == 1
    assert r["skipped"][0]["reason"] == "missing_frame"
    assert r["skipped"][0]["frame"] == 9


def test_points_at_detections_skips_no_depth_and_never_invents_origin(tmp_path) -> None:
    # A NaN pixel must be skipped, and no phantom (0,0,0) point may appear in its place.
    cloud = np.full((4, 4, 3), np.nan, dtype=np.float32)
    cloud[1, 1] = [1.0, 2.0, 3.0]   # valid
    # pixel (2,2) is NaN already; pixel (3,0) also NaN
    d = _write_frames(tmp_path, {0: cloud})
    r = PC.points_at_detections(d, [(0, 1, 1), (0, 2, 2)])
    assert r["n_valid"] == 1 and r["n_skipped"] == 1
    assert r["skipped"][0]["reason"] == "no_depth"
    # the lone valid point is the real one, not the origin
    assert not np.allclose([r["points"][0]["X_ft"], r["points"][0]["Y_ft"],
                            r["points"][0]["Z_ft"]], [0.0, 0.0, 0.0])


# ============================================================================
# synthetic ballistic fixture + end-to-end crossing
# ============================================================================
def _ballistic(n=24, vx=12.0, vz=8.0, z0=2.0, fps=FPS):
    """World-feet dog trajectory: X = vx·t, Y = 0, Z = z0 + vz·t − (g/2)t².  Returns the crossing."""
    t = np.arange(n) / fps
    X = vx * t
    Y = np.zeros_like(t)
    Z = z0 + vz * t - 0.5 * G * t * t
    roots = np.roots([-0.5 * G, vz, z0])
    t_cross = max(float(r.real) for r in roots if abs(r.imag) < 1e-9 and r.real >= 0)
    return t, X, Y, Z, t_cross, vx * t_cross


def _fixture(tmp_path, X, Y, Z, *, px0=20, py0=55, h=72, w=120):
    """Write one cloud per frame with the dog's ZED metres at (px0+i, py0-i); return detections."""
    d = tmp_path / "frames"
    d.mkdir(exist_ok=True)
    dets = []
    for i in range(len(X)):
        cloud = np.full((h, w, 3), np.nan, dtype=np.float32)
        zed = np.array([Y[i], Z[i], X[i]]) / FEET_PER_METRE
        px, py = px0 + i, py0 - i
        assert 0 <= px < w and 0 <= py < h
        cloud[py, px] = zed
        np.save(d / f"frame_{i:06d}.npy", cloud)
        dets.append((i, float(px), float(py)))
    return d, dets


def test_trajectory_from_cloud_recovers_crossing(tmp_path) -> None:
    _t, X, Y, Z, _t_cross, X_cross = _ballistic(n=24)
    d, dets = _fixture(tmp_path, X, Y, Z)

    r = PC.trajectory_from_cloud(d, dets, water_z_ft=0.0)
    assert r["status"] == "ok", r["detail"]
    assert r["distance_ft"] == pytest.approx(X_cross, abs=0.02)
    assert r["distance_in"] == pytest.approx(X_cross * 12.0, abs=0.25)
    # the free-fall acceleration is recovered from the fit (the signature of a real ballistic arc)
    cz2 = r["traj"].coeffs["z"][0]
    assert cz2 == pytest.approx(-G / (2.0 * FPS ** 2), abs=1e-4)
    # the crossing is on the descending branch (past the apex, past the last arc sample)
    assert r["crossing"]["apex_frame"] is not None
    assert r["crossing"]["frame"] > r["crossing"]["apex_frame"]
    # plane fit is near-perfect on a clean vertical jump (Y == 0 constant)
    assert r["plane_rms_ft"] < 1e-6


def test_trajectory_from_cloud_reports_world_point_fields(tmp_path) -> None:
    _t, X, Y, Z, _tc, _xc = _ballistic(n=24)
    d, dets = _fixture(tmp_path, X, Y, Z)
    r = PC.trajectory_from_cloud(d, dets, water_z_ft=0.0)
    assert r["status"] == "ok"
    # the crossing point's Z equals the water level, by construction
    assert r["crossing"]["point"][2] == pytest.approx(0.0, abs=1e-6)
    # the arc RMS is reported and tiny (clean data)
    assert r["arc_rms_ft"] is not None and r["arc_rms_ft"] < 1e-3


def test_trajectory_from_cloud_canonical_fixture_identity_transform(tmp_path) -> None:
    # The same fixture, but with the cloud stored in the canonical world frame (metres) and the
    # identity transform (metres -> feet only) — the tests must not depend on the ZED re-axing.
    _t, X, Y, Z, _tc, X_cross = _ballistic(n=24)
    d = tmp_path / "frames"
    d.mkdir(exist_ok=True)
    dets = []
    for i in range(len(X)):
        cloud = np.full((72, 120, 3), np.nan, dtype=np.float32)
        px, py = 20 + i, 55 - i
        cloud[py, px] = np.array([X[i], Y[i], Z[i]]) / FEET_PER_METRE
        np.save(d / f"frame_{i:06d}.npy", cloud)
        dets.append((i, float(px), float(py)))
    r = PC.trajectory_from_cloud(d, dets, water_z_ft=0.0, to_world=PC.metres_to_feet)
    assert r["status"] == "ok", r["detail"]
    assert r["distance_ft"] == pytest.approx(X_cross, abs=0.02)


# ============================================================================
# failure modes (each reports its own reason, never a placeholder)
# ============================================================================
def test_trajectory_from_cloud_no_points(tmp_path) -> None:
    d = tmp_path / "frames"
    d.mkdir(exist_ok=True)
    r = PC.trajectory_from_cloud(d, [(0, 1, 1), (1, 2, 2)])  # no frame files at all
    assert r["status"] == "no_points"
    assert "missing_frame" in r["detail"]


def test_trajectory_from_cloud_few_points(tmp_path) -> None:
    cloud = np.full((8, 8, 3), np.nan, dtype=np.float32)
    cloud[3, 3] = [1.0, 0.0, 2.0]
    d = _write_frames(tmp_path, {0: cloud, 1: cloud})
    r = PC.trajectory_from_cloud(d, [(0, 3, 3), (1, 3, 3)])
    assert r["status"] == "few_points"


def test_trajectory_from_cloud_no_crossing(tmp_path) -> None:
    _t, X, Y, Z, _tc, _xc = _ballistic(n=24)
    d, dets = _fixture(tmp_path, X, Y, Z)
    # the dog's apex is ~3 ft; a water level above it is unreachable -> no real crossing
    r = PC.trajectory_from_cloud(d, dets, water_z_ft=5.0)
    assert r["status"] == "no_crossing"


def test_trajectory_from_cloud_no_plane_span(tmp_path) -> None:
    # The dog never moves horizontally (X constant), so no flight direction exists to fit.
    n = 10
    X = np.full(n, 2.0)
    Y = np.zeros(n)
    Z = np.linspace(2.0, 0.5, n)
    d, dets = _fixture(tmp_path, X, Y, Z, px0=20)
    r = PC.trajectory_from_cloud(d, dets, water_z_ft=0.0)
    assert r["status"].startswith("no_plane_")
