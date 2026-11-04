"""Tests for ``pc_score`` — the headless auto-datum point-cloud scoring CLI (convenience follow-on).

All offline (no pyzed, no GPU): the same synthetic dock + water + ballistic-dog fixture the
``pc_datum`` / ``pc_distance`` tests use, fed through :func:`pc_score.run_pc_score` and the CLI
``main``.  The real fused/per-frame NEURAL-depth cloud and the floor-aligned camera->pool transform
stay hardware-verify-pending — these tests prove the *composition* of the two already-proven halves,
not the rig.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from dog_diving import pc_datum as DAT
from dog_diving.pc_trajectory import FEET_PER_METRE, world_to_zed

import pc_score

FPS = 30.0
G = 32.174
DOCK_Z = 1.625  # 19.5 in freeboard


def _cloud(points_ft):
    return np.array([world_to_zed(p) for p in points_ft], dtype=float)


def _dock_water(seed=0, *, n_water=1200, n_dock=400, dock_z=DOCK_Z):
    rng = np.random.default_rng(seed)
    pts = [[rng.uniform(0.5, 25.0), rng.uniform(0.0, 8.0), 0.0] for _ in range(n_water)]
    pts += [[rng.uniform(-4.0, 0.0), rng.uniform(0.0, 8.0), dock_z] for _ in range(n_dock)]
    return np.array(pts, dtype=float)


def _ballistic(n=24, vx=12.0, vz=8.0, z0=2.0, fps=FPS):
    t = np.arange(n) / fps
    X = vx * t
    Y = np.zeros_like(t)
    Z = z0 + vz * t - 0.5 * G * t * t
    roots = np.roots([-0.5 * G, vz, z0])
    t_cross = max(float(r.real) for r in roots if abs(r.imag) < 1e-9 and r.real >= 0)
    return X, Y, Z, vx * t_cross


def _write_frames(tmp_path, X, Y, Z):
    d = tmp_path / "frames"
    d.mkdir(exist_ok=True)
    dets = []
    for i in range(len(X)):
        cloud = np.full((72, 120, 3), np.nan, dtype=np.float32)
        cloud[55 - i, 20 + i] = np.array([Y[i], Z[i], X[i]]) / FEET_PER_METRE
        np.save(d / f"frame_{i:06d}.npy", cloud)
        dets.append((i, 20.0 + i, 55.0 - i))
    return d, dets


# ============================================================================
# the composition
# ============================================================================
def test_run_pc_score_auto_datum_matches_hand_datum(tmp_path) -> None:
    X, Y, Z, X_cross = _ballistic()
    d, dets = _write_frames(tmp_path, X, Y, Z)

    res = pc_score.run_pc_score(d, dets, cloud=_cloud(_dock_water()))
    assert res["status"] == "ok", res["detail"]
    assert res["lip"]["status"] == "ok"
    assert res["lip"]["datum"]["valid"] is True
    assert res["distance"]["status"] == "ok"
    # the located lip is at X = 0, so the lip distance == the crossing X
    assert res["distance"]["lip_distance_ft"] == pytest.approx(X_cross, abs=0.3)
    assert res["distance"]["lip_distance_in"] == pytest.approx(X_cross * 12.0, abs=3.6)
    # the water height the two halves share is the one the locate step fitted
    assert res["water_z_ft"] == pytest.approx(0.0, abs=0.1)
    assert res["distance"]["water_z_ft"] == pytest.approx(0.0, abs=0.1)
    assert res["distance"]["water_z_source"] == "supplied"  # handed down from the locate step


def test_run_pc_score_merges_frames_when_cloud_omitted(tmp_path) -> None:
    # No --cloud: the water fit must come from merging the frame_*.npy files.  The frame clouds
    # here hold only the dog's pixels, so they cannot fit a water plane -> the locate step must
    # refuse honestly (no_water_plane / no_dock), not invent a datum.
    X, Y, Z, _ = _ballistic()
    d, dets = _write_frames(tmp_path, X, Y, Z)
    res = pc_score.run_pc_score(d, dets, cloud=None)
    assert res["status"] != "ok"
    assert res["distance"] is None
    assert res["lip"]["datum"] is None


def test_run_pc_score_supplied_water_z_ft(tmp_path) -> None:
    X, Y, Z, X_cross = _ballistic()
    d, dets = _write_frames(tmp_path, X, Y, Z)
    res = pc_score.run_pc_score(d, dets, cloud=_cloud(_dock_water()), water_z_ft=0.0)
    assert res["status"] == "ok", res["detail"]
    assert res["lip"]["water_fit"]["detail"] == "supplied"
    assert res["distance"]["lip_distance_ft"] == pytest.approx(X_cross, abs=0.3)


def test_run_pc_score_refuses_vertical_wall(tmp_path) -> None:
    rng = np.random.default_rng(1)
    wall = [[5.0, rng.uniform(0, 8), rng.uniform(0, 4)] for _ in range(300)]
    X, Y, Z, _ = _ballistic()
    d, dets = _write_frames(tmp_path, X, Y, Z)
    res = pc_score.run_pc_score(d, dets, cloud=_cloud(np.array(wall, dtype=float)))
    assert res["status"] == "no_water_plane"
    assert res["distance"] is None


def test_result_is_json_serialisable(tmp_path) -> None:
    X, Y, Z, _ = _ballistic()
    d, dets = _write_frames(tmp_path, X, Y, Z)
    res = pc_score.run_pc_score(d, dets, cloud=_cloud(_dock_water()))
    json.dumps(res)  # must not raise (no numpy arrays / scalars leak through)


# ============================================================================
# input handling
# ============================================================================
def test_load_detections_accepts_dicts_and_triples(tmp_path) -> None:
    p = tmp_path / "det.json"
    p.write_text(json.dumps([{"frame": 0, "x": 10.0, "y": 20.0}, [1, 11.0, 21.0]]))
    loaded = pc_score._load_detections(p)
    assert len(loaded) == 2


def test_load_detections_rejects_non_list(tmp_path) -> None:
    p = tmp_path / "det.json"
    p.write_text(json.dumps({"frame": 0, "x": 10, "y": 20}))
    with pytest.raises(ValueError):
        pc_score._load_detections(p)


# ============================================================================
# the CLI
# ============================================================================
def test_main_end_to_end_writes_out(tmp_path) -> None:
    X, Y, Z, X_cross = _ballistic()
    d, dets = _write_frames(tmp_path, X, Y, Z)
    cloud_path = tmp_path / "fused.npy"
    np.save(cloud_path, _cloud(_dock_water()))
    det_path = tmp_path / "det.json"
    det_path.write_text(json.dumps(dets))
    out_path = tmp_path / "pc_score.json"

    rc = pc_score.main(["--frames-dir", str(d), "--detections", str(det_path),
                        "--cloud", str(cloud_path), "--out", str(out_path)])
    assert rc == 0
    written = json.loads(out_path.read_text(encoding="utf-8"))
    assert written["status"] == "ok"
    assert written["distance"]["lip_distance_ft"] == pytest.approx(X_cross, abs=0.3)


def test_main_refusal_exits_nonzero(tmp_path) -> None:
    X, Y, Z, _ = _ballistic()
    d, dets = _write_frames(tmp_path, X, Y, Z)
    rng = np.random.default_rng(1)
    wall = [[5.0, rng.uniform(0, 8), rng.uniform(0, 4)] for _ in range(300)]
    cloud_path = tmp_path / "wall.npy"
    np.save(cloud_path, _cloud(np.array(wall, dtype=float)))
    det_path = tmp_path / "det.json"
    det_path.write_text(json.dumps(dets))

    rc = pc_score.main(["--frames-dir", str(d), "--detections", str(det_path),
                        "--cloud", str(cloud_path)])
    assert rc == 1


def test_main_missing_inputs_exit_two(tmp_path) -> None:
    assert pc_score.main(["--frames-dir", str(tmp_path / "nope"),
                          "--detections", str(tmp_path / "also_nope.json")]) == 2
