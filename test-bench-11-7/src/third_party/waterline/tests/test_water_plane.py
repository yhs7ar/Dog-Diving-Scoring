"""Tests for estimate_water_plane.py against the known stub scene.
Run with:  python -m pytest tests -q"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import estimate_water_plane as wp  # noqa: E402
from zed_stub import StubCamera, StubScene, write_waterline  # noqa: E402


def run(tmp_path, scene=None, origin=None, **kw):
    scene = scene or StubScene()
    waterline = write_waterline(tmp_path / "waterline.json", scene)
    cam = StubCamera(scene)
    return wp.estimate(cam, cam.open(), waterline, origin_pixel=origin, **kw), scene


def normal_error_deg(res, scene):
    n = np.array(res["plane"]["normal"])
    return np.degrees(np.arccos(np.clip(n @ scene.down(), -1, 1)))


def test_recovers_true_plane(tmp_path):
    res, scene = run(tmp_path)
    assert normal_error_deg(res, scene) < 0.1
    assert abs(res["plane"]["d"] - scene.camera_height) < 0.01        # within 1 cm
    tilt = res["quality"]["camera_tilt"]
    assert abs(tilt["pitch_deg"] - scene.pitch_deg) < 0.1
    assert abs(tilt["roll_deg"] - scene.roll_deg) < 0.1


def test_plane_is_below_camera(tmp_path):
    res, _ = run(tmp_path)
    assert res["plane"]["d"] > 0
    assert res["plane"]["normal"][1] > 0.9    # 'down' is mostly +y in the camera frame


def test_projection_removes_pixels_above_offset(tmp_path):
    # Sampling 30 px above the line puts the raw points ~20 cm above the water at 8 m.
    with_proj, scene = run(tmp_path, pixels_above=30)
    without, _ = run(tmp_path, pixels_above=30, project=False)
    assert abs(with_proj["plane"]["d"] - scene.camera_height) < 0.01
    assert scene.camera_height - without["plane"]["d"] > 0.1


def test_robust_to_outliers(tmp_path):
    noisy = StubScene(outlier_fraction=0.15)
    res, scene = run(tmp_path, scene=noisy)
    assert abs(res["plane"]["d"] - scene.camera_height) < 0.01


def test_strict_confidence_drops_data_but_still_works(tmp_path):
    scene = StubScene()
    waterline = write_waterline(tmp_path / "waterline.json", scene)
    loose = StubCamera(scene, confidence_threshold=90)
    strict = StubCamera(scene, confidence_threshold=40)
    r_loose = wp.estimate(loose, loose.open(), waterline)
    r_strict = wp.estimate(strict, strict.open(), waterline)
    assert r_strict["quality"]["valid_depth_fraction"] < r_loose["quality"]["valid_depth_fraction"]
    assert abs(r_strict["plane"]["d"] - scene.camera_height) < 0.01


def test_origin_height(tmp_path):
    res, scene = run(tmp_path, origin=(300, 540))
    true_o = scene.true_xyz(300, 540)
    true_h = scene.camera_height - true_o @ scene.down()
    assert abs(res["origin"]["height_above_water_m"] - true_h) < 0.01
    on_water = np.array(res["origin"]["on_water_m"])
    assert abs(on_water @ scene.down() - scene.camera_height) < 0.01  # lies on the water plane


def test_image_size_mismatch_is_rejected(tmp_path):
    scene = StubScene()
    waterline = write_waterline(tmp_path / "waterline.json", scene)
    waterline["image_size"] = [1280, 720]
    cam = StubCamera(scene)
    with pytest.raises(ValueError, match="image size"):
        wp.estimate(cam, cam.open(), waterline)


def test_cli(tmp_path):
    write_waterline(tmp_path / "waterline.json")
    out = tmp_path / "plane.json"
    wp.main(["--waterline", str(tmp_path / "waterline.json"), "--stub", "--output", str(out)])
    res = json.load(open(out))
    assert set(res) >= {"plane", "quality", "inputs", "samples"}
