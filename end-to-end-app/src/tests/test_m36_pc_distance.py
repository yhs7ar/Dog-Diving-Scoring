"""Tests for ``dog_diving.pc_distance`` — route "pc" (step 14 / PC-DIST).

All tests run offline (no pyzed, no GPU) against a **synthetic** fixture: a flat water plane at
``Z = 0``, a dock-lip line at ``X = 0`` (the world frame's documented datum), and a ballistic dog
parabola of world-feet points written into per-frame clouds at known pixels — the same fixture
``test_m35`` builds.  The real per-frame cloud read (NEURAL depth on the Jetson) and the true
camera->pool rigid transform are hardware-verify-pending.
"""

from __future__ import annotations

import numpy as np
import pytest

from dog_diving import pc_distance as PD
from dog_diving.pc_trajectory import FEET_PER_METRE

FPS = 30.0
G = 32.174


# ============================================================================
# datum line + line distance
# ============================================================================
def test_datum_line_valid() -> None:
    lip = PD.datum_line([0.0, -1.0, 0.5], [0.0, 1.0, 0.5])
    assert lip["valid"] is True
    assert lip["length_ft"] == pytest.approx(2.0)
    assert np.allclose(lip["u"], [0.0, 1.0, 0.0])


def test_datum_line_rejects_degenerate() -> None:
    lip = PD.datum_line([0.0, 0.0, 0.5], [0.0, 0.0, 0.5])
    assert lip["valid"] is False
    assert "degenerate" in lip["reason"]


def test_datum_line_rejects_nonfinite() -> None:
    lip = PD.datum_line([np.nan, 0.0, 0.5], [0.0, 1.0, 0.5])
    assert lip["valid"] is False
    assert "non-finite" in lip["reason"]


def test_horizontal_distance_ignores_z() -> None:
    # Same (X, Y) line at two different heights -> same horizontal distance.
    d1 = PD.horizontal_distance_to_line([3.0, 0.0, 5.0], [0.0, -1.0, 0.5], [0.0, 1.0, 0.5])
    d2 = PD.horizontal_distance_to_line([3.0, 0.0, 99.0], [0.0, -1.0, 7.0], [0.0, 1.0, 7.0])
    assert d1 == pytest.approx(3.0)
    assert d2 == pytest.approx(3.0)


def test_horizontal_distance_degenerate_line() -> None:
    assert PD.horizontal_distance_to_line([3.0, 0.0, 0.0], [0.0, 1.0, 0.0],
                                          [0.0, 1.0, 0.0]) is None


def test_signed_distance_flips_with_corner_order() -> None:
    a, b = [0.0, -1.0, 0.0], [0.0, 1.0, 0.0]
    s1 = PD.signed_horizontal_distance([3.0, 0.0, 0.0], a, b)
    s2 = PD.signed_horizontal_distance([3.0, 0.0, 0.0], b, a)
    assert s1 == pytest.approx(-s2)
    assert abs(s1) == pytest.approx(3.0)


def test_distance_to_lip_missing_datum_is_not_zero() -> None:
    r = PD.distance_to_lip([3.0, 0.0, 0.0], None)
    assert r["valid"] is False and r["distance_ft"] is None
    assert "no valid dock-lip datum" in r["detail"]


# ============================================================================
# synthetic ballistic fixture (mirrors test_m35)
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
        zed = np.array([Y[i], Z[i], X[i]]) / FEET_PER_METRE
        px, py = px0 + i, py0 - i
        cloud[py, px] = zed
        np.save(d / f"frame_{i:06d}.npy", cloud)
        dets.append((i, float(px), float(py)))
    return d, dets


# ============================================================================
# route "pc"
# ============================================================================
def test_pc_distance_recovers_crossing(tmp_path) -> None:
    _t, X, Y, Z, _tc, X_cross = _ballistic()
    d, dets = _fixture(tmp_path, X, Y, Z)

    r = PD.pc_distance(d, dets, water_z_ft=0.0)
    assert r["status"] == "ok", r["detail"]
    assert r["route"] == "pc"
    assert r["distance_ft"] == pytest.approx(X_cross, abs=0.02)
    assert r["total_inches"] == pytest.approx(X_cross * 12.0, abs=0.25)
    assert r["transverse_ft"] == pytest.approx(0.0, abs=0.02)
    assert r["frame"] is not None
    assert r["point_source"] == "model" and r["point_origin"] == "point_cloud"


def test_pc_distance_lip_at_origin_equals_x(tmp_path) -> None:
    # With the documented fixture (lip at X = 0, perpendicular to the pool axis), the horizontal
    # distance to the lip IS the crossing X.
    _t, X, Y, Z, _tc, X_cross = _ballistic()
    d, dets = _fixture(tmp_path, X, Y, Z)
    lip = PD.datum_line([0.0, -1.0, 0.5], [0.0, 1.0, 0.5])

    r = PD.pc_distance(d, dets, water_z_ft=0.0, lip=lip)
    assert r["status"] == "ok"
    assert r["lip"]["valid"] is True
    assert r["lip_distance_ft"] == pytest.approx(X_cross, abs=0.02)
    assert r["lip_distance_in"] == pytest.approx(X_cross * 12.0, abs=0.25)


def test_pc_distance_lip_offset_is_constant_bias(tmp_path) -> None:
    # Move the lip back off the world origin (X = +0.5 ft): the lip distance drops by exactly that
    # constant — the datum bias the project re-finds, now expressed as a number, not assumed away.
    _t, X, Y, Z, _tc, X_cross = _ballistic()
    d, dets = _fixture(tmp_path, X, Y, Z)
    lip = PD.datum_line([0.5, -1.0, 0.5], [0.5, 1.0, 0.5])

    r = PD.pc_distance(d, dets, water_z_ft=0.0, lip=lip)
    assert r["status"] == "ok"
    assert r["distance_ft"] == pytest.approx(X_cross, abs=0.02)          # along-pool X unchanged
    assert r["lip_distance_ft"] == pytest.approx(abs(X_cross - 0.5), abs=0.02)


def test_pc_distance_skips_no_depth_and_never_scores_zero(tmp_path) -> None:
    # Every detection pixel has no depth (NaN): the route must report no_points, NOT distance 0.
    d = tmp_path / "frames"
    d.mkdir(exist_ok=True)
    dets = []
    for i in range(10):
        cloud = np.full((72, 120, 3), np.nan, dtype=np.float32)
        np.save(d / f"frame_{i:06d}.npy", cloud)
        dets.append((i, 20.0 + i, 55.0 - i))

    r = PD.pc_distance(d, dets, water_z_ft=0.0)
    assert r["status"] == "no_points"
    assert r["total_inches"] is None and r["distance_ft"] is None
    assert any("no_points" in w for w in r["warnings"])


def test_pc_distance_no_crossing(tmp_path) -> None:
    _t, X, Y, Z, _tc, _xc = _ballistic()
    d, dets = _fixture(tmp_path, X, Y, Z)
    r = PD.pc_distance(d, dets, water_z_ft=5.0)  # apex ~3 ft, unreachable
    assert r["status"] == "no_crossing"
    assert r["total_inches"] is None


# ============================================================================
# agreement table + cross-check
# ============================================================================
def test_agreement_table_pairwise_inches() -> None:
    routes = {
        "pc": {"total_inches": 111.18},
        "plane": {"total_inches": 113.18},     # +2 in
        "arc3d": {"total_inches": 110.18},     # -1 in
    }
    t = PD.agreement_table(routes)
    assert t["answered"] == ["arc3d", "pc", "plane"]
    assert t["missing"] == []
    by_pair = {(p["a"], p["b"]): p["disagreement_in"] for p in t["pairs"]}
    assert by_pair[("arc3d", "pc")] == pytest.approx(1.0)
    assert by_pair[("pc", "plane")] == pytest.approx(2.0)
    assert by_pair[("arc3d", "plane")] == pytest.approx(3.0)
    assert t["max_disagreement_in"] == pytest.approx(3.0)


def test_agreement_table_skips_unanswered_routes() -> None:
    routes = {
        "pc": {"total_inches": 111.18},
        "plane": {"total_inches": None},
    }
    t = PD.agreement_table(routes)
    assert t["answered"] == ["pc"]
    assert t["missing"] == ["plane"]
    assert t["pairs"] == [] and t["max_disagreement_in"] is None


def test_agreement_table_tolerance_gate() -> None:
    routes = {"pc": {"total_inches": 100.0}, "plane": {"total_inches": 100.6}}
    assert PD.agreement_table(routes, tolerance_in=1.0)["within_tolerance"] is True
    assert PD.agreement_table(routes, tolerance_in=0.5)["within_tolerance"] is False
    assert PD.agreement_table(routes)["within_tolerance"] is None


def test_cross_check_integrates_pc_route(tmp_path) -> None:
    # The real pc result feeding the same agreement seam the report will use, against two
    # synthetic mono results a couple of inches off.
    _t, X, Y, Z, _tc, X_cross = _ballistic()
    d, dets = _fixture(tmp_path, X, Y, Z)
    pc = PD.pc_distance(d, dets, water_z_ft=0.0)

    t = PD.cross_check(pc,
                       plane_result={"total_inches": X_cross * 12.0 + 2.0},
                       arc3d_result={"total_inches": X_cross * 12.0 - 1.0})
    assert t["answered"] == ["arc3d", "pc", "plane"]
    by_pair = {(p["a"], p["b"]): p["disagreement_in"] for p in t["pairs"]}
    assert by_pair[("arc3d", "pc")] == pytest.approx(1.0, abs=0.3)
    assert by_pair[("pc", "plane")] == pytest.approx(2.0, abs=0.3)
    assert t["max_disagreement_in"] == pytest.approx(3.0, abs=0.3)
