"""Tests for ``dog_diving.pc_water_plane`` + the water-plane fit in ``pc_trajectory`` (step 6 / 1b-auto).

All tests run offline (no pyzed, no GPU) against synthetic clouds:

* a dense horizontal water plane (canonical ``zed_to_world``, water at ``Z = 0``), optionally with a
  ballistic dog parabola above it — the fixture the trajectory cluster already uses;
* a camera-pose fixture (camera above the water, behind the dock lip) that exercises
  ``seed_plane_session``'s world->image projection against a ground-truth pinhole projection.

The real fused/per-frame cloud (NEURAL depth on the Jetson) and the true floor-aligned camera pose
remain hardware-verify-pending — the ``to_world`` / ``world_to_zed`` seams are where those plug in.
"""

from __future__ import annotations

import numpy as np
import pytest

from dog_diving import pc_distance as PD
from dog_diving import pc_trajectory as PC
from dog_diving import pc_water_plane as WP
from dog_diving.pc_trajectory import FEET_PER_METRE

FPS = 30.0
G = 32.174


# ============================================================================
# coordinate seams
# ============================================================================
def test_world_to_zed_is_inverse_of_zed_to_world() -> None:
    X, Y, Z = 8.16, -0.5, 2.0
    zed = PC.world_to_zed([X, Y, Z])
    assert np.allclose(PC.zed_to_world(zed), [X, Y, Z], atol=1e-9)


def test_intrinsics_matrix_from_dict_and_array() -> None:
    K = WP.intrinsics_matrix({"fx": 600.0, "fy": 600.0, "cx": 480.0, "cy": 300.0})
    assert np.allclose(K, [[600, 0, 480], [0, 600, 300], [0, 0, 1]])
    assert np.allclose(WP.intrinsics_matrix(K), K)


def test_project_zed_to_pixel_pinhole() -> None:
    K = WP.intrinsics_matrix({"fx": 100.0, "fy": 100.0, "cx": 10.0, "cy": 20.0})
    # a point on the optical axis, 2 m out -> principal point
    assert WP.project_zed_to_pixel([0.0, 0.0, 2.0], K) == pytest.approx((10.0, 20.0))
    assert WP.project_zed_to_pixel([0.0, 0.0, 0.0], K) is None   # at the camera
    assert WP.project_zed_to_pixel([0.0, 0.0, -1.0], K) is None  # behind it


# ============================================================================
# the water-plane fit
# ============================================================================
def _water_plane_zed(water_z_ft: float = 0.0, n: int = 50,
                     x_range=(0.0, 12.0), y_range=(-3.0, 3.0)) -> np.ndarray:
    """A dense water plane at world ``Z = water_z_ft``, in ZED metres (canonical mapping)."""
    xs = np.linspace(*x_range, n)
    ys = np.linspace(*y_range, n)
    X, Y = np.meshgrid(xs, ys)
    Z = np.full_like(X, float(water_z_ft))
    world = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    return np.array([world[:, 1], world[:, 2], world[:, 0]]).T / FEET_PER_METRE


def test_fit_water_plane_recovers_zero() -> None:
    r = PC.fit_water_plane(_water_plane_zed(0.0))
    assert r["status"] == "ok", r["detail"]
    assert r["water_z_ft"] == pytest.approx(0.0, abs=1e-6)
    assert r["n_inliers"] > 100
    assert r["inlier_fraction"] == pytest.approx(1.0, abs=1e-3)


def test_fit_water_plane_recovers_nonzero() -> None:
    r = PC.fit_water_plane(_water_plane_zed(2.5))
    assert r["status"] == "ok", r["detail"]
    assert r["water_z_ft"] == pytest.approx(2.5, abs=1e-4)


def test_fit_water_plane_ignores_sparse_outliers() -> None:
    # A dense water plane plus a few dog points above it: RANSAC must pick the water, not the arc.
    water = _water_plane_zed(0.0, n=40)
    _t, X, Y, Z, _tc, _xc = _ballistic()
    dog = np.array([Y, Z, X]).T / FEET_PER_METRE
    r = PC.fit_water_plane(np.vstack([water, dog]))
    assert r["status"] == "ok", r["detail"]
    assert r["water_z_ft"] == pytest.approx(0.0, abs=1e-4)
    # the water dominates: the inlier fraction is below 1 because the dog points are outliers
    assert r["inlier_fraction"] > 0.9


def test_fit_water_plane_rejects_vertical_wall() -> None:
    # A wall across the pool (world X = const) has a HORIZONTAL world normal -> not the water.
    ys = np.linspace(-3.0, 3.0, 30)
    zs = np.linspace(0.0, 4.0, 30)
    Y, Z = np.meshgrid(ys, zs)
    X = np.full_like(Y, 5.0)
    world = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    wall = np.array([world[:, 1], world[:, 2], world[:, 0]]).T / FEET_PER_METRE
    r = PC.fit_water_plane(wall)
    assert r["status"] == "not_horizontal"
    assert r["water_z_ft"] is None


def test_fit_water_plane_rejects_few_inliers() -> None:
    # 5 coplanar points form a plane but cannot be "the dominant water".
    pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
                    [1.0, 1.0, 0.0], [0.5, 0.5, 0.0]]) / FEET_PER_METRE
    r = PC.fit_water_plane(pts)
    assert r["status"] == "few_inliers"
    assert r["water_z_ft"] is None


def test_fit_water_plane_rejects_no_points() -> None:
    r = PC.fit_water_plane(np.full((0, 3), np.nan))
    assert r["status"] == "no_points"
    r2 = PC.fit_water_plane(np.full((5, 3), np.nan))  # all NaN -> no finite points
    assert r2["status"] == "no_points"


def test_merge_frames_cloud_stacks_finite(tmp_path) -> None:
    d = tmp_path / "frames"
    d.mkdir()
    a = np.full((4, 4, 3), np.nan, dtype=np.float32)
    a[1, 1] = [1.0, 2.0, 3.0]
    b = np.full((4, 4, 3), np.nan, dtype=np.float32)
    b[2, 2] = [4.0, 5.0, 6.0]
    np.save(d / "frame_000000.npy", a)
    np.save(d / "frame_000001.npy", b)
    merged = PC.merge_frames_cloud(d)
    assert merged.shape == (2, 3)
    assert set(map(tuple, np.round(merged, 3))) == {(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)}


# ============================================================================
# ballistic fixture (mirrors test_m35/36)
# ============================================================================
def _ballistic(n=24, vx=12.0, vz=8.0, z0=2.0, fps=FPS):
    t = np.arange(n) / fps
    X = vx * t
    Y = np.zeros_like(t)
    Z = z0 + vz * t - 0.5 * G * t * t
    roots = np.roots([-0.5 * G, vz, z0])
    t_cross = max(float(r.real) for r in roots if abs(r.imag) < 1e-9 and r.real >= 0)
    return t, X, Y, Z, t_cross, vx * t_cross


def _fixture(tmp_path, X, Y, Z, *, px0=20, py0=55, h=72, w=120):
    d = tmp_path / "frames"
    d.mkdir(exist_ok=True)
    dets = []
    for i in range(len(X)):
        cloud = np.full((h, w, 3), np.nan, dtype=np.float32)
        cloud[py0 - i, px0 + i] = np.array([Y[i], Z[i], X[i]]) / FEET_PER_METRE
        np.save(d / f"frame_{i:06d}.npy", cloud)
        dets.append((i, float(px0 + i), float(py0 - i)))
    return d, dets


# ============================================================================
# the closed loop: water_z_ft auto-fit feeds the trajectory / distance
# ============================================================================
def test_trajectory_from_cloud_autofits_water(tmp_path) -> None:
    _t, X, Y, Z, _tc, X_cross = _ballistic()
    d, dets = _fixture(tmp_path, X, Y, Z)
    water = _water_plane_zed(0.0)

    r = PC.trajectory_from_cloud(d, dets, water_cloud=water)   # water_z_ft = None
    assert r["status"] == "ok", r["detail"]
    assert r["water_z_source"] == "cloud_fit"
    assert r["water_z_ft"] == pytest.approx(0.0, abs=1e-4)
    assert r["distance_ft"] == pytest.approx(X_cross, abs=0.02)


def test_pc_distance_autofits_water(tmp_path) -> None:
    _t, X, Y, Z, _tc, X_cross = _ballistic()
    d, dets = _fixture(tmp_path, X, Y, Z)
    water = _water_plane_zed(0.0)

    r = PD.pc_distance(d, dets, water_cloud=water)
    assert r["status"] == "ok", r["detail"]
    assert r["water_z_source"] == "cloud_fit"
    assert r["water_z_ft"] == pytest.approx(0.0, abs=1e-4)
    assert r["distance_ft"] == pytest.approx(X_cross, abs=0.02)
    assert r["total_inches"] == pytest.approx(X_cross * 12.0, abs=0.25)


def test_trajectory_from_cloud_no_water_level(tmp_path) -> None:
    # Only the dog's vertical flight plane is present (no water): no dominant horizontal plane, so
    # the water height cannot be derived and must be reported as a refusal, never zero.
    _t, X, Y, Z, _tc, _xc = _ballistic()
    d, dets = _fixture(tmp_path, X, Y, Z)
    r = PC.trajectory_from_cloud(d, dets)   # no water_z_ft, no water_cloud
    assert r["status"] == "no_water_level"
    assert r["water_z_ft"] is None


def test_pc_distance_no_water_level_is_not_zero(tmp_path) -> None:
    _t, X, Y, Z, _tc, _xc = _ballistic()
    d, dets = _fixture(tmp_path, X, Y, Z)
    r = PD.pc_distance(d, dets)
    assert r["status"] == "no_water_level"
    assert r["total_inches"] is None and r["distance_ft"] is None


def test_supplied_water_z_still_works(tmp_path) -> None:
    # Backward compatible: an explicit water_z_ft skips the fit entirely.
    _t, X, Y, Z, _tc, X_cross = _ballistic()
    d, dets = _fixture(tmp_path, X, Y, Z)
    r = PD.pc_distance(d, dets, water_z_ft=0.0)
    assert r["status"] == "ok"
    assert r["water_z_source"] == "supplied"
    assert r["distance_ft"] == pytest.approx(X_cross, abs=0.02)


# ============================================================================
# seeding the PlaneSession from the fitted plane
# ============================================================================
#: A realistic camera pose: 3 ft behind the dock lip, 5.9 ft above the water, level, looking down the
#: pool (+X).  This is exactly the floor-aligned pose that is hardware-verify-pending; the test plugs
#: it into the same ``to_world`` / ``world_to_zed`` seams the rig measurement will fill.
CAM = np.array([-3.0, 0.0, 5.9])


def _to_world_cam(zed_m: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(zed_m, dtype=float).ravel()[:3]
    return np.array([z, x, y]) * FEET_PER_METRE + CAM


def _world_to_zed_cam(world_ft: np.ndarray) -> np.ndarray:
    X, Y, Z = np.asarray(world_ft, dtype=float).ravel()[:3]
    p = np.array([X, Y, Z]) - CAM
    return np.array([p[1], p[2], p[0]]) / FEET_PER_METRE


def _cam_water_cloud(n: int = 40, x_range=(0.0, 28.0), y_range=(-10.0, 10.0)) -> np.ndarray:
    xs = np.linspace(*x_range, n)
    ys = np.linspace(*y_range, n)
    X, Y = np.meshgrid(xs, ys)
    world = np.stack([X.ravel(), Y.ravel(), np.zeros_like(X.ravel())], axis=1)
    return np.array([_world_to_zed_cam(p) for p in world])


def test_seed_plane_session_projects_and_roundtrips() -> None:
    K = {"fx": 600.0, "fy": 600.0, "cx": 480.0, "cy": 300.0}
    frame_shape = (600, 960)
    L, Wd = 25.0, 20.0
    cloud = _cam_water_cloud()

    r = WP.seed_plane_session(
        cloud, intrinsics=K, length_ft=L, width_ft=Wd, frame_shape=frame_shape,
        to_world=_to_world_cam, world_to_zed=_world_to_zed_cam)
    assert r["status"] == "ok", r["detail"]
    assert r["water_z_ft"] == pytest.approx(0.0, abs=1e-3)

    # Each corner equals the independent ground-truth projection.
    world_corners = {"00": (0.0, 0.0), "10": (L, 0.0), "11": (L, Wd), "01": (0.0, Wd)}
    for tag in WP.CORNER_TAGS:
        X, Y = world_corners[tag]
        zed = _world_to_zed_cam([X, Y, 0.0])
        gt = WP.project_zed_to_pixel(zed, WP.intrinsics_matrix(K))
        assert r["corners"][tag] == pytest.approx(gt, abs=1e-6)

    # The seeded session round-trips world -> pixel -> world, and the quad passes the P-gates.
    session = r["session"]
    assert session is not None and session.placed
    assert r["gates"]["G4_quad"]["pass"] is True
    for X, Y in ((0.0, 0.0), (L, 0.0), (L, Wd), (0.0, Wd)):
        px, py = session.pixel_of(X, Y)
        back = session.world_at(px, py)
        assert back == pytest.approx((X, Y), abs=1e-4)


def test_seed_plane_session_records_auto_provenance() -> None:
    r = WP.seed_plane_session(
        _cam_water_cloud(), intrinsics={"fx": 600, "fy": 600, "cx": 480, "cy": 300},
        length_ft=25.0, width_ft=20.0, frame_shape=(600, 960),
        to_world=_to_world_cam, world_to_zed=_world_to_zed_cam)
    assert r["status"] == "ok"
    assert r["session"].seed is not None
    assert r["session"].seed["placement"] == "auto"
    assert r["session"].seed["upstream"] == "point_cloud"
    assert r["session"].seed["water_z_ft"] == pytest.approx(0.0, abs=1e-3)


def test_seed_plane_session_refuses_when_no_plane() -> None:
    # A cloud with no dominant horizontal plane must refuse and leave the session untouched.
    r = WP.seed_plane_session(
        np.full((5, 3), np.nan), intrinsics={"fx": 600, "fy": 600, "cx": 480, "cy": 300},
        length_ft=25.0, width_ft=20.0, frame_shape=(600, 960),
        to_world=_to_world_cam, world_to_zed=_world_to_zed_cam)
    assert r["status"].startswith("no_water_plane_")
    assert r["session"] is None and r["corners"] is None
