"""Tests for ``dog_diving.fused_trajectory`` — the fused-cloud 3-D trajectory (step 4a).

All tests run offline (no pyzed, no GPU): the fused path is exercised against the
``test_m33_fusion`` synthetic fixture (two per-camera clouds + a known (R, T)) for the water
plane, plus the ``test_m35`` ballistic fixture (a flat water plane at world ``Z = 0`` with a
parabola of dog points above it) for the dog trajectory.  The real SVO2 -> NEURAL depth ->
per-camera cloud -> fusion path is hardware-verify-pending.

What is pinned here:
* ``fused_water_plane`` fuses via the ``PointCloudSource`` seam + ``FusionConfig`` and fits a
  horizontal plane at a known world ``Z``.
* ``trajectory_from_fused_cloud`` / ``pc_distance_fused`` hand the merged cloud to the
  existing world-frame chain and recover the ballistic crossing, with ``fused`` provenance and
  ``water_fit_source == "fused_cloud"``.
* A supplied ``water_z_ft`` skips the fusion entirely (a CPU host can still score a run).
"""

from __future__ import annotations

import numpy as np
import pytest

from dog_diving import fused_trajectory as FT
from dog_diving.fusion import (
    HAS_ZED_SDK,
    FusionConfig,
    PointCloudSource,
    SVO2PointCloudSource,
    SyntheticPointCloudSource,
)
from dog_diving.pc_trajectory import FEET_PER_METRE

FPS = 30.0
G = 32.174

REF_SERIAL = 42250580   # cameras[0] — reference frame (identity pose)
CAM1_SERIAL = 40847387  # cameras[1] — carries the (R, T) transform

R_Z90 = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
T_FIXTURE = np.array([1.0, 0.5, 0.0])  # metres


def _fusion_doc():
    return {
        "world": {"override_gravity": True, "rotation": [0.0, 0.0, 0.0],
                  "translation": [0.0, 0.0, 0.0]},
        "cameras": [
            {"serial_number": REF_SERIAL,
             "position": {"translation": [0.0, 0.0, 0.0], "rotation": [0.0, 0.0, 0.0]}},
            {"serial_number": CAM1_SERIAL,
             "position": {"translation": T_FIXTURE.tolist(), "rotation": R_Z90.tolist()}},
        ],
    }


def _config():
    return FusionConfig.from_dict(_fusion_doc())


# --- fused-water fixture: two water planes that fuse onto one horizontal plane -----------

def _ref_water_zed(y_m: float, n: int = 40) -> np.ndarray:
    """A dense water plane at ZED y = ``y_m`` (world ``Z = y_m * ft/m``), in ZED metres."""
    xs = np.linspace(-2.0, 2.0, n)
    zs = np.linspace(1.0, 8.0, n)
    X, Z = np.meshgrid(xs, zs)
    Y = np.full_like(X, y_m)
    return np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)


def _wall_zed(n: int = 30) -> np.ndarray:
    """A vertical wall across the pool: constant ZED z -> constant world X (not water)."""
    xs = np.linspace(-2.0, 2.0, n)
    ys = np.linspace(0.0, 3.0, n)
    X, Y = np.meshgrid(xs, ys)
    Z = np.full_like(X, 5.0)
    return np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)


def _cam1_from_ref(ref_points: np.ndarray) -> np.ndarray:
    """Map reference-frame points into cam1's own frame, so fusion lands them back exactly."""
    return (R_Z90.T @ (ref_points - T_FIXTURE).T).T


def _water_source(y_m: float, n: int = 40) -> SyntheticPointCloudSource:
    ref = _ref_water_zed(y_m, n=n)
    return SyntheticPointCloudSource({REF_SERIAL: ref, CAM1_SERIAL: _cam1_from_ref(ref)})


def _wall_source() -> SyntheticPointCloudSource:
    wall = _wall_zed()
    return SyntheticPointCloudSource({REF_SERIAL: wall, CAM1_SERIAL: _cam1_from_ref(wall)})


# --- ballistic dog fixture (mirrors test_m35) -------------------------------------------

def _ballistic(n=24, vx=12.0, vz=8.0, z0=2.0, fps=FPS):
    t = np.arange(n) / fps
    X = vx * t
    Y = np.zeros_like(t)
    Z = z0 + vz * t - 0.5 * G * t * t
    roots = np.roots([-0.5 * G, vz, z0])
    t_cross = max(float(r.real) for r in roots if abs(r.imag) < 1e-9 and r.real >= 0)
    return t, X, Y, Z, t_cross, vx * t_cross


def _frames(tmp_path, X, Y, Z, *, px0=20, py0=55, h=72, w=120):
    d = tmp_path / "frames"
    d.mkdir(exist_ok=True)
    dets = []
    for i in range(len(X)):
        cloud = np.full((h, w, 3), np.nan, dtype=np.float32)
        cloud[py0 - i, px0 + i] = np.array([Y[i], Z[i], X[i]]) / FEET_PER_METRE
        np.save(d / f"frame_{i:06d}.npy", cloud)
        dets.append((i, float(px0 + i), float(py0 - i)))
    return d, dets


# --- fused_water_plane -------------------------------------------------------------------

def test_fused_water_plane_recovers_zero() -> None:
    r = FT.fused_water_plane(_water_source(0.0), _config())
    assert r["status"] == "ok", r["water_fit"]["detail"]
    assert r["water_z_ft"] == pytest.approx(0.0, abs=1e-4)
    assert r["reference_serial"] == REF_SERIAL
    assert r["counts"] == {REF_SERIAL: 1600, CAM1_SERIAL: 1600}
    assert r["n_points"] == 3200


def test_fused_water_plane_recovers_nonzero() -> None:
    y_m = 2.5 / FEET_PER_METRE  # world Z = 2.5 ft
    r = FT.fused_water_plane(_water_source(y_m), _config())
    assert r["status"] == "ok", r["water_fit"]["detail"]
    assert r["water_z_ft"] == pytest.approx(2.5, abs=1e-3)


def test_fused_water_plane_refuses_non_horizontal() -> None:
    r = FT.fused_water_plane(_wall_source(), _config())
    assert r["status"] == "not_horizontal"
    assert r["water_z_ft"] is None


# --- trajectory_from_fused_cloud --------------------------------------------------------

def test_trajectory_from_fused_cloud_recovers_crossing(tmp_path) -> None:
    _t, X, Y, Z, _tc, X_cross = _ballistic()
    d, dets = _frames(tmp_path, X, Y, Z)

    r = FT.trajectory_from_fused_cloud(d, dets, _water_source(0.0), _config())
    assert r["status"] == "ok", r["detail"]
    assert r["water_z_source"] == "cloud_fit"
    assert r["water_fit_source"] == "fused_cloud"
    assert r["water_z_ft"] == pytest.approx(0.0, abs=1e-4)
    assert r["distance_ft"] == pytest.approx(X_cross, abs=0.02)
    assert r["fused"]["reference_serial"] == REF_SERIAL
    assert r["fused"]["counts"] == {REF_SERIAL: 1600, CAM1_SERIAL: 1600}


def test_trajectory_from_fused_cloud_no_water_level(tmp_path) -> None:
    _t, X, Y, Z, _tc, _xc = _ballistic()
    d, dets = _frames(tmp_path, X, Y, Z)
    # A fused cloud with no dominant horizontal plane -> the water height cannot be derived.
    r = FT.trajectory_from_fused_cloud(d, dets, _wall_source(), _config())
    assert r["status"] == "no_water_level"


def test_trajectory_from_fused_cloud_supplied_water_skips_fusion(tmp_path) -> None:
    _t, X, Y, Z, _tc, X_cross = _ballistic()
    d, dets = _frames(tmp_path, X, Y, Z)

    class _Raising(PointCloudSource):
        def load(self):
            raise RuntimeError("must not be called when water_z_ft is supplied")

    r = FT.trajectory_from_fused_cloud(d, dets, _Raising(), _config(), water_z_ft=0.0)
    assert r["status"] == "ok", r["detail"]
    assert r["water_z_source"] == "supplied"
    assert "fused" not in r
    assert r["distance_ft"] == pytest.approx(X_cross, abs=0.02)


# --- pc_distance_fused ------------------------------------------------------------------

def test_pc_distance_fused_recovers_crossing(tmp_path) -> None:
    _t, X, Y, Z, _tc, X_cross = _ballistic()
    d, dets = _frames(tmp_path, X, Y, Z)

    r = FT.pc_distance_fused(d, dets, _water_source(0.0), _config())
    assert r["route"] == "pc"
    assert r["status"] == "ok", r["detail"]
    assert r["total_inches"] == pytest.approx(X_cross * 12.0, abs=0.25)
    assert r["water_z_source"] == "cloud_fit"
    assert r["water_fit_source"] == "fused_cloud"
    assert r["fused"]["counts"] == {REF_SERIAL: 1600, CAM1_SERIAL: 1600}


def test_pc_distance_fused_supplied_water(tmp_path) -> None:
    _t, X, Y, Z, _tc, X_cross = _ballistic()
    d, dets = _frames(tmp_path, X, Y, Z)

    r = FT.pc_distance_fused(d, dets, _water_source(0.0), _config(), water_z_ft=0.0)
    assert r["route"] == "pc"
    assert r["status"] == "ok"
    assert r["water_z_source"] == "supplied"
    assert r["total_inches"] == pytest.approx(X_cross * 12.0, abs=0.25)


# --- hardware refusal (CPU host) --------------------------------------------------------

@pytest.mark.skipif(HAS_ZED_SDK, reason="host has pyzed; offline refusal test is for CPU hosts")
def test_fused_water_plane_svo2_source_refuses(tmp_path) -> None:
    src = SVO2PointCloudSource([tmp_path / "jump_cam1.svo2", tmp_path / "jump_cam2.svo2"])
    with pytest.raises(RuntimeError):
        FT.fused_water_plane(src, _config())
