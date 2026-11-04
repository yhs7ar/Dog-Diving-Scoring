"""Tests for generate_point_cloud.py — single-SVO2 → depth → XYZ cloud (step 12 / PC-GEN).

All tests run offline (no pyzed, no GPU): the pure helpers (output-path derivation, JSON/CSV
writers, cloud save/load round-trips, frame merging, validation) are exercised directly, and
the hardware entry point :func:`generate` is pinned to *refuse* on a CPU host rather than
invent a cloud.  The real SVO2 read is hardware-verify-pending (Jetson + NEURAL depth).
"""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "dual_recording"
    / "generate_point_cloud.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("generate_point_cloud", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


mod = _load_module()


# --- output-path derivation --------------------------------------------------

def test_output_paths_npy(tmp_path) -> None:
    p = mod.output_paths("some/dir/jump_cam1.svo2", tmp_path, "npy")
    assert p["cloud"] == [tmp_path / "jump_cam1.npy"]
    assert p["intrinsics"] == tmp_path / "jump_cam1_intrinsics.json"
    assert p["timestamps"] == tmp_path / "jump_cam1_timestamps.csv"
    assert p["frames_dir"] == tmp_path / "jump_cam1_frames"


def test_output_paths_xyz_and_both(tmp_path) -> None:
    xyz = mod.output_paths("jump.svo2", tmp_path, "xyz")
    assert xyz["cloud"] == [tmp_path / "jump.xyz"]
    both = mod.output_paths("jump.svo2", tmp_path, "both")
    assert both["cloud"] == [tmp_path / "jump.npy", tmp_path / "jump.xyz"]


def test_output_paths_bad_format_raises(tmp_path) -> None:
    with pytest.raises(ValueError):
        mod.output_paths("jump.svo2", tmp_path, "ply")


# --- JSON / CSV writers ------------------------------------------------------

def test_write_intrinsics_round_trip(tmp_path) -> None:
    doc = {
        "serial_number": 40847387,
        "fx": 300.5, "fy": 302.1, "cx": 600.0, "cy": 360.0,
        "width": 1200, "height": 720,
    }
    path = mod.write_intrinsics(doc, tmp_path / "intrinsics.json")
    loaded = json.loads(path.read_text())
    assert loaded == doc


def test_write_timestamps_csv_round_trip(tmp_path) -> None:
    rows = [(0, 1000), (1, 1033), (2, 1066)]
    path = mod.write_timestamps_csv(rows, tmp_path / "ts.csv")
    with path.open() as f:
        reader = list(csv.reader(f))
    assert reader[0] == ["frame_index", "timestamp_ms"]
    assert reader[1:] == [["0", "1000"], ["1", "1033"], ["2", "1066"]]


# --- cloud save / load -------------------------------------------------------

def test_save_load_cloud_npy_round_trip(tmp_path) -> None:
    xyz = np.array([[0.1, 0.2, 1.0], [0.3, -0.4, 2.0]], dtype=np.float64)
    path = mod.save_cloud(xyz, tmp_path / "cloud.npy")
    loaded = mod.load_cloud(path)
    assert loaded.dtype == np.float64
    assert np.allclose(loaded, xyz)


def test_save_load_cloud_xyz_round_trip(tmp_path) -> None:
    xyz = np.array([[0.1, 0.2, 1.0], [0.3, -0.4, 2.0]], dtype=np.float64)
    path = mod.save_cloud(xyz, tmp_path / "cloud.xyz")
    # ASCII .xyz: x y z per line, 6-decimal precision.
    lines = path.read_text().strip().splitlines()
    assert len(lines) == 2
    loaded = mod.load_cloud(path)
    assert np.allclose(loaded, xyz, atol=1e-6)


def test_save_cloud_bad_suffix_raises(tmp_path) -> None:
    with pytest.raises(ValueError):
        mod.save_cloud(np.zeros((1, 3)), tmp_path / "cloud.ply")


def test_save_frame_cloud_preserves_nan_and_shape(tmp_path) -> None:
    xyz = np.full((4, 5, 3), np.nan, dtype=np.float32)
    xyz[1, 2] = [1.0, 2.0, 3.0]
    xyz[3, 4] = [np.inf, 0.0, -np.inf]  # inf/-inf coerced to NaN; finite 0.0 kept
    path = mod.save_frame_cloud(xyz, tmp_path / "frame_000000.npy")
    loaded = mod.load_frame_cloud(path)
    assert loaded.shape == (4, 5, 3)
    assert loaded.dtype == np.float32
    assert np.allclose(loaded[1, 2], [1.0, 2.0, 3.0])
    assert np.isnan(loaded[0, 0]).all()
    # non-finite values (the two infs) become NaN; the finite 0.0 is preserved.
    assert np.isnan(loaded[3, 4, 0])
    assert loaded[3, 4, 1] == 0.0
    assert np.isnan(loaded[3, 4, 2])


# --- merge_frames ------------------------------------------------------------

def test_merge_frames_filters_nonfinite() -> None:
    a = np.array([[[0.0, 0.0, 1.0], [np.nan, 0.0, 1.0]]])  # 1 valid
    b = np.array([[[1.0, 1.0, 2.0], [2.0, 2.0, 3.0]]])      # 2 valid
    merged = mod.merge_frames([a, b])
    assert merged.shape == (3, 3)
    assert np.allclose(merged[0], [0.0, 0.0, 1.0])
    assert np.allclose(merged[1], [1.0, 1.0, 2.0])
    assert np.allclose(merged[2], [2.0, 2.0, 3.0])


def test_merge_frames_empty_returns_empty() -> None:
    merged = mod.merge_frames([])
    assert merged.shape == (0, 3)
    assert merged.dtype == np.float64


def test_merge_frames_all_invalid_is_empty() -> None:
    a = np.full((2, 3, 3), np.nan, dtype=np.float64)
    merged = mod.merge_frames([a])
    assert merged.shape == (0, 3)


# --- generate() validation + offline refusal --------------------------------

def test_generate_validates_stride() -> None:
    with pytest.raises(ValueError):
        mod.generate("x.svo2", stride=0)


def test_generate_validates_max_frames() -> None:
    with pytest.raises(ValueError):
        mod.generate("x.svo2", max_frames=-1)


def test_generate_validates_skip_frames() -> None:
    with pytest.raises(ValueError):
        mod.generate("x.svo2", skip_frames=-1)


@pytest.mark.skipif(mod.HAS_ZED_SDK, reason="host has pyzed; offline refusal is for CPU hosts")
def test_generate_refuses_without_pyzed(tmp_path) -> None:
    with pytest.raises(RuntimeError):
        mod.generate(tmp_path / "jump.svo2", out_dir=tmp_path)
