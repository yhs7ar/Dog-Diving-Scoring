"""Tests for fusion.py — runtime two-camera point-cloud fusion (step 2d).

All tests run offline (no pyzed, no hardware, no Tk): the fusion core is pure
numpy and is exercised against a synthetic two-cloud fixture with a *known*
transform, so the assertion is "the merged cloud lands where the transform says
it should" — not a regression against a real rig.

Also pins the Fusion-JSON schema the two calibration scripts produce, including
the rotation-representation ambiguity (3×3 matrix vs Rodrigues vs Euler-degrees).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dog_diving.fusion import (  # noqa: E402
    HAS_ZED_SDK,
    FusedCloud,
    FusionConfig,
    FusionTransform,
    PointCloudSource,
    SVO2PointCloudSource,
    SyntheticPointCloudSource,
    fuse_from_source,
    fuse_point_clouds,
    rotation_from_json,
)


# --- fixtures -----------------------------------------------------------------

REF_SERIAL = 42250580   # cameras[0] — reference frame (identity pose)
CAM1_SERIAL = 40847387  # cameras[1] — carries the (R, T) transform

# Rotation about Z by +90°:  x' = -y, y' = x, z' = z.
R_Z90 = np.array(
    [[0.0, -1.0, 0.0],
     [1.0, 0.0, 0.0],
     [0.0, 0.0, 1.0]],
    dtype=np.float64,
)
T_FIXTURE = np.array([1.0, 0.5, 0.0], dtype=np.float64)  # metres


def _fusion_doc(rotation, translation=T_FIXTURE.tolist()):
    """A minimal Fusion JSON in the schema both calibration scripts emit."""
    return {
        "world": {"override_gravity": True, "rotation": [0.0, 0.0, 0.0],
                  "translation": [0.0, 0.0, 0.0]},
        "cameras": [
            {"serial_number": REF_SERIAL,
             "position": {"translation": [0.0, 0.0, 0.0], "rotation": [0.0, 0.0, 0.0]}},
            {"serial_number": CAM1_SERIAL,
             "position": {"translation": translation, "rotation": rotation}},
        ],
    }


def _cloud_ref(n=6):
    """Some points in the reference camera's own frame."""
    return np.array(
        [[float(i), float(i * 2), 0.5] for i in range(n)], dtype=np.float64
    )


def _cloud_cam1_from_true(true_points, R=R_Z90, T=T_FIXTURE):
    """Invert (R, T) so that ``transform.apply(B)`` lands back on ``true_points``."""
    return (R.T @ (true_points - T).T).T


# --- rotation parsing --------------------------------------------------------

def test_rotation_from_matrix_nested() -> None:
    R = rotation_from_json(R_Z90.tolist())
    assert np.allclose(R, R_Z90)


def test_rotation_from_matrix_flat() -> None:
    flat = [v for row in R_Z90.tolist() for v in row]
    R = rotation_from_json(flat)
    assert np.allclose(R, R_Z90)


def test_rotation_from_rodrigues_z90() -> None:
    rvec = [0.0, 0.0, np.pi / 2.0]
    R = rotation_from_json(rvec)
    assert np.allclose(R, R_Z90, atol=1e-12)


def test_rotation_from_identity_rodrigues() -> None:
    assert np.allclose(rotation_from_json([0.0, 0.0, 0.0]), np.eye(3))


def test_rotation_from_euler_deg() -> None:
    # Z-rotation by 90° expressed in Euler degrees (sensor_placer_auto.py fusion format).
    R = rotation_from_json([0.0, 0.0, 90.0], convention="euler_deg")
    assert np.allclose(R, R_Z90, atol=1e-12)


def test_rotation_bad_arity_raises() -> None:
    with pytest.raises(ValueError):
        rotation_from_json([1.0, 2.0])  # 2 values


# --- FusionConfig parsing ----------------------------------------------------

def test_config_reference_is_cameras_zero() -> None:
    cfg = FusionConfig.from_dict(_fusion_doc(R_Z90.tolist()))
    assert cfg.reference_serial == REF_SERIAL
    assert len(cfg.cameras) == 2


def test_config_transform_for_cam1() -> None:
    cfg = FusionConfig.from_dict(_fusion_doc(R_Z90.tolist()))
    t = cfg.transform_for(CAM1_SERIAL)
    assert np.allclose(t.rotation, R_Z90)
    assert np.allclose(t.translation, T_FIXTURE)


def test_config_reference_transform_is_identity() -> None:
    cfg = FusionConfig.from_dict(_fusion_doc(R_Z90.tolist()))
    t = cfg.transform_for(REF_SERIAL)
    assert np.allclose(t.rotation, np.eye(3))
    assert np.allclose(t.translation, np.zeros(3))


def test_config_unknown_serial_raises() -> None:
    cfg = FusionConfig.from_dict(_fusion_doc(R_Z90.tolist()))
    with pytest.raises(KeyError):
        cfg.transform_for(99999999)


def test_config_from_file_round_trip(tmp_path) -> None:
    p = tmp_path / "fusion.json"
    p.write_text(json.dumps(_fusion_doc(R_Z90.tolist())))
    cfg = FusionConfig.from_file(p)
    assert cfg.reference_serial == REF_SERIAL
    assert np.allclose(cfg.transform_for(CAM1_SERIAL).rotation, R_Z90)


def test_config_rejects_single_camera() -> None:
    with pytest.raises(ValueError):
        FusionConfig.from_dict({"cameras": [{"serial_number": 1, "position": {}}]})


# --- the fusion core (synthetic fixture) -------------------------------------

def test_fused_cloud_lands_where_transform_says() -> None:
    A = _cloud_ref(6)                    # reference-frame points
    A2 = _cloud_ref(4) + np.array([7.0, -3.0, 2.0])   # where cam1 points should land
    B = _cloud_cam1_from_true(A2)        # cam1's own-frame points

    cfg = FusionConfig.from_dict(_fusion_doc(R_Z90.tolist()))
    fused = fuse_point_clouds({REF_SERIAL: A, CAM1_SERIAL: B}, cfg)

    assert isinstance(fused, FusedCloud)
    assert fused.points.shape == (len(A) + len(B), 3)
    assert fused.reference_serial == REF_SERIAL
    # Reference cloud passes through untouched; cam1 cloud lands on A2.
    assert np.allclose(fused.points[: len(A)], A)
    assert np.allclose(fused.points[len(A):], A2, atol=1e-9)
    assert fused.counts == {REF_SERIAL: len(A), CAM1_SERIAL: len(B)}


def test_fused_cloud_counts_sum_to_points() -> None:
    A = _cloud_ref(3)
    B = _cloud_cam1_from_true(_cloud_ref(5))
    cfg = FusionConfig.from_dict(_fusion_doc(R_Z90.tolist()))
    fused = fuse_point_clouds({REF_SERIAL: A, CAM1_SERIAL: B}, cfg)
    assert len(fused) == sum(fused.counts.values()) == len(A) + len(B)


def test_fuse_single_point_cloud_reshapes_1d() -> None:
    cfg = FusionConfig.from_dict(_fusion_doc(R_Z90.tolist()))
    fused = fuse_point_clouds({REF_SERIAL: np.array([1.0, 2.0, 3.0])}, cfg)
    assert fused.points.shape == (1, 3)


def test_fuse_unknown_serial_refuses_by_default() -> None:
    cfg = FusionConfig.from_dict(_fusion_doc(R_Z90.tolist()))
    with pytest.raises(KeyError):
        fuse_point_clouds({REF_SERIAL: _cloud_ref(3), 123456: _cloud_ref(3)}, cfg)


def test_fuse_unknown_serial_skipped_when_asked() -> None:
    cfg = FusionConfig.from_dict(_fusion_doc(R_Z90.tolist()))
    fused = fuse_point_clouds(
        {REF_SERIAL: _cloud_ref(3), 123456: _cloud_ref(4)}, cfg, skip_missing=True
    )
    assert fused.counts == {REF_SERIAL: 3}
    assert fused.points.shape == (3, 3)


def test_fusion_transform_apply_known_result() -> None:
    t = FusionTransform(R_Z90, T_FIXTURE)
    p = np.array([[1.0, 0.0, 0.0]])
    out = t.apply(p)
    # R_Z90 @ [1,0,0] = [0,1,0]; + T = [1.0, 1.5, 0.0]
    assert np.allclose(out, [[1.0, 1.5, 0.0]])


# --- the source seam ---------------------------------------------------------

def test_synthetic_source_returns_clouds() -> None:
    A = _cloud_ref(3)
    src = SyntheticPointCloudSource({REF_SERIAL: A, CAM1_SERIAL: _cloud_ref(2)})
    loaded = src.load()
    assert set(loaded) == {REF_SERIAL, CAM1_SERIAL}
    assert np.array_equal(loaded[REF_SERIAL], A)
    assert isinstance(src, PointCloudSource)


def test_fuse_from_source_end_to_end() -> None:
    A = _cloud_ref(4)
    A2 = _cloud_ref(3) + np.array([2.0, 2.0, 0.0])
    B = _cloud_cam1_from_true(A2)
    src = SyntheticPointCloudSource({REF_SERIAL: A, CAM1_SERIAL: B})
    cfg = FusionConfig.from_dict(_fusion_doc(R_Z90.tolist()))

    fused = fuse_from_source(src, cfg)
    assert fused.points.shape == (len(A) + len(B), 3)
    assert np.allclose(fused.points[: len(A)], A)
    assert np.allclose(fused.points[len(A):], A2, atol=1e-9)


@pytest.mark.skipif(HAS_ZED_SDK, reason="host has pyzed; offline refusal test is for CPU hosts")
def test_svo2_source_refuses_without_pyzed(tmp_path) -> None:
    # On a CPU host the hardware path must refuse (not return a fake cloud).
    src = SVO2PointCloudSource([tmp_path / "jump_cam1.svo2", tmp_path / "jump_cam2.svo2"])
    with pytest.raises(RuntimeError):
        src.load()
