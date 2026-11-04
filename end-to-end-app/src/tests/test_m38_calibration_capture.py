"""Tests for calibration_capture.py — the calibration-capture mode (step 1a).

All tests run offline (no pyzed, no hardware, no Tk): the orchestration is driven
through :class:`FakeCalibrationCaptureBackend`, which writes synthetic SVO2 stubs and
a Fusion JSON with a known ``(R, T)``, so the assertion is "the saved Fusion JSON
loads back through ``fusion.FusionConfig.from_file`` and the fused cloud lands where
the transform says it should" — not a regression against a real rig.

Also pins the argv the rig backend will shell out with (the proven scripts + serials)
and the rotation-convention mapping (ICP 3×3 matrix vs sensor-placer Euler degrees),
so the ambiguity recorded in fusion.py is resolved *explicitly*, not by accident.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dog_diving.calibration_capture import (  # noqa: E402
    ICP_CALIBRATION,
    SENSOR_PLACER,
    DUAL_SVGA_RECORDER,
    FakeCalibrationCaptureBackend,
    RigCalibrationCaptureBackend,
    _discover_svo_pair,
    fusion_convention,
    load_fusion_config,
    run_calibration_capture,
)
from dog_diving.config import FUSION_JSON_PATH, SVO_RECORDINGS_DIR  # noqa: E402
from dog_diving.fusion import (  # noqa: E402
    HAS_ZED_SDK,
    FusionConfig,
    fuse_point_clouds,
)


REF_SERIAL = 42250580
CAM1_SERIAL = 40847387
R_Z90 = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
T_FIXTURE = np.array([1.0, 0.5, 0.0])


def _cloud_ref(n=6):
    return np.array([[float(i), float(i * 2), 0.5] for i in range(n)], dtype=np.float64)


def _cloud_cam1_from_true(true_points):
    return (R_Z90.T @ (true_points - T_FIXTURE).T).T


# --- fusion_convention ------------------------------------------------------

def test_convention_icp_is_rodrigues_default() -> None:
    # ICP writes a 3×3 matrix (arity 9); the loader default is fine for it.
    assert fusion_convention("icp") == "rodrigues"
    assert fusion_convention("kabsch") == "rodrigues"


def test_convention_sensor_placer_is_euler_deg() -> None:
    assert fusion_convention("sensor_placer") == "euler_deg"
    assert fusion_convention("sensor-placer") == "euler_deg"


def test_convention_unknown_method_raises() -> None:
    with pytest.raises(ValueError):
        fusion_convention("nonsense")


# --- known paths ------------------------------------------------------------

def test_known_paths_are_project_constants() -> None:
    assert isinstance(FUSION_JSON_PATH, Path)
    assert isinstance(SVO_RECORDINGS_DIR, Path)
    assert FUSION_JSON_PATH.name == "fusion_config.json"


# --- the offline orchestration (fake backend) -------------------------------

def test_run_calibration_capture_end_to_end(tmp_path) -> None:
    fusion_out = tmp_path / "fusion_config.json"
    result = run_calibration_capture(
        FakeCalibrationCaptureBackend(),
        prefix="calib_test",
        output_dir=tmp_path / "svos",
        fusion_path=fusion_out,
        method="icp",
    )
    assert result["reference_serial"] == REF_SERIAL
    assert result["method"] == "icp"
    assert result["rotation_convention"] == "rodrigues"
    assert len(result["svos"]) == 2
    # Two SVO2 stubs written, cam0 first.
    assert result["svos"][0].endswith("_cam0_SN42250580.svo2")
    assert result["svos"][1].endswith("_cam1_SN40847387.svo2")
    assert fusion_out.is_file()


def test_run_calibration_capture_reads_back_correct_transform(tmp_path) -> None:
    fusion_out = tmp_path / "fusion_config.json"
    result = run_calibration_capture(
        FakeCalibrationCaptureBackend(),
        prefix="calib_test",
        output_dir=tmp_path / "svos",
        fusion_path=fusion_out,
    )
    cfg = result["config"]
    assert isinstance(cfg, FusionConfig)
    t = cfg.transform_for(CAM1_SERIAL)
    assert np.allclose(t.rotation, R_Z90)
    assert np.allclose(t.translation, T_FIXTURE)


def test_run_calibration_capture_sensor_placer_euler_deg(tmp_path) -> None:
    fusion_out = tmp_path / "fusion_config.json"
    result = run_calibration_capture(
        FakeCalibrationCaptureBackend(),
        prefix="calib_test",
        output_dir=tmp_path / "svos",
        fusion_path=fusion_out,
        method="sensor_placer",
    )
    assert result["rotation_convention"] == "euler_deg"
    # The Euler-degree file must load to the same rotation the matrix would.
    assert np.allclose(result["config"].transform_for(CAM1_SERIAL).rotation, R_Z90)


def test_load_fusion_config_wraps_known_path(tmp_path) -> None:
    fusion_out = tmp_path / "fusion_config.json"
    run_calibration_capture(
        FakeCalibrationCaptureBackend(), prefix="x", output_dir=tmp_path / "s",
        fusion_path=fusion_out,
    )
    cfg = load_fusion_config(fusion_out)
    assert cfg.reference_serial == REF_SERIAL


def test_wiring_feeds_fuse_point_clouds(tmp_path) -> None:
    fusion_out = tmp_path / "fusion_config.json"
    result = run_calibration_capture(
        FakeCalibrationCaptureBackend(), prefix="x", output_dir=tmp_path / "s",
        fusion_path=fusion_out,
    )
    cfg = result["config"]

    A = _cloud_ref(6)
    A2 = _cloud_ref(4) + np.array([7.0, -3.0, 2.0])
    B = _cloud_cam1_from_true(A2)

    fused = fuse_point_clouds({REF_SERIAL: A, CAM1_SERIAL: B}, cfg)
    assert fused.reference_serial == REF_SERIAL
    assert np.allclose(fused.points[: len(A)], A)
    assert np.allclose(fused.points[len(A):], A2, atol=1e-9)


# --- refusal / failure modes ------------------------------------------------

def test_refuses_when_capture_returns_one_svo(tmp_path) -> None:
    class OneSvo(FakeCalibrationCaptureBackend):
        def capture(self, prefix, duration, output_dir):
            return [Path(output_dir) / f"{prefix}_cam0.svo2"]

    with pytest.raises(RuntimeError, match="two SVO2"):
        run_calibration_capture(OneSvo(), prefix="x", output_dir=tmp_path / "s",
                                fusion_path=tmp_path / "f.json")


def test_refuses_when_extrinsics_writes_nothing(tmp_path) -> None:
    class NoOutput(FakeCalibrationCaptureBackend):
        def compute_extrinsics(self, svo0, svo1, output_path, method):
            return Path(output_path)  # claims success, writes no file

    with pytest.raises(RuntimeError, match="did not produce"):
        run_calibration_capture(NoOutput(), prefix="x", output_dir=tmp_path / "s",
                                fusion_path=tmp_path / "f.json")


def test_discover_svo_pair_sorts_cam0_first(tmp_path) -> None:
    (tmp_path / "p_cam1_SN40847387.svo2").write_text("{}")
    (tmp_path / "p_cam0_SN42250580.svo2").write_text("{}")
    (tmp_path / "p_cam2_ignored.svo2").write_text("{}")  # a third file must not leak in
    pair = _discover_svo_pair(tmp_path, "p")
    assert [p.name for p in pair] == ["p_cam0_SN42250580.svo2", "p_cam1_SN40847387.svo2"]


# --- the rig backend's command construction (pure, no hardware) -------------

def test_rig_capture_argv_points_at_proven_recorder(tmp_path) -> None:
    be = RigCalibrationCaptureBackend()
    argv = be._capture_argv("calib_p", 10, tmp_path)
    assert argv[0] == sys.executable
    assert argv[1] == str(DUAL_SVGA_RECORDER)
    assert "-d" in argv and "10" in argv
    assert "-p" in argv and "calib_p" in argv
    assert str(tmp_path) in argv
    # The recorder's LOSSLESS SVGA config lives in that script; nothing here re-declares it.


def test_rig_extrinsics_argv_icp() -> None:
    be = RigCalibrationCaptureBackend()
    argv = be._extrinsics_argv("a.svo2", "b.svo2", "out.json", "icp")
    assert argv[1] == str(ICP_CALIBRATION)
    assert argv[argv.index("--svo0") + 1] == "a.svo2"
    assert argv[argv.index("--svo1") + 1] == "b.svo2"
    assert argv[argv.index("--output") + 1] == "out.json"
    assert "--format" not in argv


def test_rig_extrinsics_argv_sensor_placer_fusion_format() -> None:
    be = RigCalibrationCaptureBackend()
    argv = be._extrinsics_argv("a.svo2", "b.svo2", "out.json", "sensor_placer")
    assert argv[1] == str(SENSOR_PLACER)
    assert argv[argv.index("--format") + 1] == "fusion"


def test_rig_extrinsics_argv_unknown_method_raises() -> None:
    be = RigCalibrationCaptureBackend()
    with pytest.raises(ValueError):
        be._extrinsics_argv("a.svo2", "b.svo2", "out.json", "bogus")


@pytest.mark.skipif(HAS_ZED_SDK, reason="host has pyzed; offline refusal is for CPU hosts")
def test_rig_backend_refuses_without_pyzed(tmp_path) -> None:
    be = RigCalibrationCaptureBackend()
    with pytest.raises(RuntimeError):
        be.capture("x", 10, tmp_path)
    with pytest.raises(RuntimeError):
        be.compute_extrinsics("a.svo2", "b.svo2", tmp_path / "o.json", "icp")
