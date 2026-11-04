"""Tests for ``dog_diving.pc_datum`` — auto-locating the dock lip in the point cloud (1b-auto follow-on).

All tests run offline (no pyzed, no GPU) against a **synthetic** fixture: a flat water plane at
``Z = 0`` and a dock-top plane at ``Z = 1.625 ft`` (the documented 19.5 in freeboard), with the lip
at ``X = 0`` running along ``Y``.  The fixture is written in world feet and converted to ZED metres
through ``pc_trajectory.world_to_zed`` (the canonical inverse of ``zed_to_world``), so the module is
exercised through the same ``to_world`` seam the real rig will use.  The real fused/per-frame cloud
(NEURAL depth on the Jetson) and the true floor-aligned camera->pool transform are
hardware-verify-pending.

The water region is made deliberately **dominant** (more points, wider extent) so the RANSAC water
fit picks it rather than a tilted plane spanning water + dock — mirroring the real scene, where the
water surface is far larger than the dock strip.
"""

from __future__ import annotations

import numpy as np
import pytest

from dog_diving import pc_datum as DAT
from dog_diving import pc_distance as PD
from dog_diving.pc_trajectory import FEET_PER_METRE, world_to_zed

FPS = 30.0
G = 32.174
DOCK_Z = 1.625  # 19.5 in freeboard


def _cloud(points_ft):
    """World-feet points -> ZED-metres ``(N, 3)`` cloud."""
    return np.array([world_to_zed(p) for p in points_ft], dtype=float)


def _dock_water(seed=0, *, n_water=1200, n_dock=400, dock_z=DOCK_Z, rotate_deg=0.0):
    """A synthetic dock + water scene in world feet: water ``X in [0.5, 25]``, ``Z = 0``; dock
    ``X in [-4, 0]``, ``Y in [0, 8]``, ``Z = dock_z``.  ``rotate_deg`` spins both in the ``(X, Y)``
    plane (to exercise an angled lip)."""
    rng = np.random.default_rng(seed)
    pts = []
    for _ in range(n_water):
        pts.append([rng.uniform(0.5, 25.0), rng.uniform(0.0, 8.0), 0.0])
    for _ in range(n_dock):
        pts.append([rng.uniform(-4.0, 0.0), rng.uniform(0.0, 8.0), dock_z])
    pts = np.array(pts, dtype=float)
    if rotate_deg:
        t = np.radians(rotate_deg)
        R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
        pts[:, :2] = pts[:, :2] @ R.T
    return pts


def test_locate_fronto_parallel_lip() -> None:
    # This one lets the water plane be auto-fitted (the water-dominant fixture must land it at Z = 0).
    res = DAT.locate_dock_lip(_cloud(_dock_water()))
    assert res["status"] == "ok", res["detail"]
    assert res["water_z_ft"] == pytest.approx(0.0, abs=0.1)
    assert res["dock_z_ft"] == pytest.approx(DOCK_Z, abs=0.1)
    # pool direction points +X (away from the dock, into the water)
    assert res["pool_dir"][0] > 0.9
    assert abs(res["pool_dir"][1]) < 0.1
    d = res["datum"]
    assert d["valid"] is True
    # the lip sits at the dock edge X = 0 — the measurement-critical fact (a perpendicular distance
    # to the lip line is independent of the line's extent)
    assert abs(d["a"][0]) < 0.25 and abs(d["b"][0]) < 0.25
    assert 6.0 < d["length_ft"] < 9.0
    assert d["a"][2] == pytest.approx(DOCK_Z, abs=0.1)
    assert d["b"][2] == pytest.approx(DOCK_Z, abs=0.1)


def test_near_far_invariant() -> None:
    # near (a) has the smaller world X (closer to the camera in the canonical frame), and u is a->b.
    res = DAT.locate_dock_lip(_cloud(_dock_water()), water_z_ft=0.0)
    d = res["datum"]
    assert d["valid"] is True
    assert d["a"][0] <= d["b"][0] + 1e-9
    u = np.asarray(d["u"])
    ab = (np.asarray(d["b"]) - np.asarray(d["a"]))
    ab = ab / np.linalg.norm(ab)
    assert np.allclose(u, ab, atol=1e-6)


def test_supplied_water_z_ft_skips_fit() -> None:
    res = DAT.locate_dock_lip(_cloud(_dock_water()), water_z_ft=0.0)
    assert res["status"] == "ok", res["detail"]
    assert res["water_fit"]["status"] == "ok"
    assert res["water_fit"]["detail"] == "supplied"
    assert res["water_z_ft"] == 0.0


def test_angled_lip_found_and_ordered() -> None:
    # Rotate the whole scene 30 deg in the (X, Y) plane: the lip is still a straight edge, still
    # perpendicular to the pool direction, but no longer axis-aligned.
    res = DAT.locate_dock_lip(_cloud(_dock_water(rotate_deg=30.0)), water_z_ft=0.0)
    assert res["status"] == "ok", res["detail"]
    d = res["datum"]
    assert d["valid"] is True
    u = np.asarray(d["u"])[:2]
    pd = np.asarray(res["pool_dir"])[:2]
    assert abs(float(u @ pd)) < 0.15         # lip perpendicular to the pool axis
    assert 6.0 < d["length_ft"] < 9.0
    assert d["a"][0] <= d["b"][0] + 1e-9     # near is the smaller-X corner


def test_ignores_dog_blob_above_water() -> None:
    # A dog-sized blob above the dock must not win the dock-plane fit.
    pts = _dock_water().tolist()
    rng = np.random.default_rng(7)
    for _ in range(40):  # a small 0.3 ft ball at Z = 3 ft, above the dock
        pts.append([rng.uniform(0.5, 1.0), rng.uniform(3.0, 4.0), rng.uniform(2.85, 3.15)])
    res = DAT.locate_dock_lip(_cloud(np.array(pts, dtype=float)), water_z_ft=0.0)
    assert res["status"] == "ok", res["detail"]
    assert res["dock_z_ft"] == pytest.approx(DOCK_Z, abs=0.1)  # not the dog's height


def test_refuses_no_water() -> None:
    # A vertical wall only (no horizontal plane) -> the water fit refuses.
    rng = np.random.default_rng(1)
    wall = [[5.0, rng.uniform(0, 8), rng.uniform(0, 4)] for _ in range(300)]
    res = DAT.locate_dock_lip(_cloud(np.array(wall, dtype=float)))
    assert res["status"] == "no_water_plane"
    assert res["datum"] is None


def test_refuses_no_dock() -> None:
    # Water only, nothing above the plane -> no dock-top surface.
    rng = np.random.default_rng(2)
    water = [[rng.uniform(0, 20), rng.uniform(0, 8), 0.0] for _ in range(400)]
    res = DAT.locate_dock_lip(_cloud(np.array(water, dtype=float)))
    assert res["status"] == "no_dock"
    assert res["datum"] is None


def test_refuses_no_pool_side() -> None:
    # Dock points only, with a supplied water height -> no water-region points to point toward.
    rng = np.random.default_rng(3)
    dock = [[rng.uniform(-4, 0), rng.uniform(0, 8), DOCK_Z] for _ in range(300)]
    res = DAT.locate_dock_lip(_cloud(np.array(dock, dtype=float)), water_z_ft=0.0)
    assert res["status"] == "no_pool_side"
    assert res["datum"] is None


def test_refuses_empty_cloud() -> None:
    assert DAT.locate_dock_lip(np.empty((0, 3)))["status"] == "no_points"


# ============================================================================
# integration: the auto-located datum feeds the "pc" distance route
# ============================================================================
def _ballistic(n=24, vx=12.0, vz=8.0, z0=2.0, fps=FPS):
    t = np.arange(n) / fps
    X = vx * t
    Y = np.zeros_like(t)
    Z = z0 + vz * t - 0.5 * G * t * t
    roots = np.roots([-0.5 * G, vz, z0])
    t_cross = max(float(r.real) for r in roots if abs(r.imag) < 1e-9 and r.real >= 0)
    return X, Y, Z, vx * t_cross


def test_auto_datum_scores_the_same_as_hand_datum(tmp_path) -> None:
    # The whole point: locate the lip in the cloud, hand it to pc_distance, and get the same lip
    # distance the operator-supplied datum would give (the lip is at X = 0, so lip distance == X).
    X, Y, Z, X_cross = _ballistic()
    d = tmp_path / "frames"
    d.mkdir(exist_ok=True)
    dets = []
    for i in range(len(X)):
        cloud = np.full((72, 120, 3), np.nan, dtype=np.float32)
        cloud[55 - i, 20 + i] = np.array([Y[i], Z[i], X[i]]) / FEET_PER_METRE
        np.save(d / f"frame_{i:06d}.npy", cloud)
        dets.append((i, 20.0 + i, 55.0 - i))

    located = DAT.locate_dock_lip(_cloud(_dock_water()))
    assert located["status"] == "ok"
    datum = located["datum"]
    assert datum["valid"] is True

    r = PD.pc_distance(d, dets, water_z_ft=0.0, lip=datum)
    assert r["status"] == "ok", r["detail"]
    assert r["lip"]["valid"] is True
    assert r["lip_distance_ft"] == pytest.approx(X_cross, abs=0.3)
    assert r["lip_distance_in"] == pytest.approx(X_cross * 12.0, abs=3.6)
    # and the horizontal distance from the crossing point to the located line agrees directly
    dl = PD.distance_to_lip([r["distance_ft"], 0.0, 0.0], datum)
    assert dl["distance_ft"] == pytest.approx(X_cross, abs=0.3)
