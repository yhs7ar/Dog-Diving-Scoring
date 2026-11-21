"""M4 — ground truth + judge (data model, gt_in via homography)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1]
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SRC))

from test_bench import ground_truth as gt_mod  # noqa: E402

PLANE = REPO / "end-to-end-app" / "planes" / "plane_test-clip-video004_right.pool.json"


@pytest.fixture(scope="module")
def plane():
    if not PLANE.is_file():
        pytest.skip("parent plane document not present on this machine")
    return json.loads(PLANE.read_text(encoding="utf-8"))


def _direct_gt_in(plane, x_px, y_px):
    """The same entry_coordinate call, written out independently in the test."""
    from dog_diving.pool import calib as _calib
    from dog_diving.pool import intersection as _inter
    lens = _calib.lens_from_document(plane)
    pose = _calib.pose_from_document(plane)
    water = _calib.water_from_document(plane)
    H = np.asarray(plane["homography"], dtype=float).reshape(3, 3)
    return _inter.entry_coordinate(pose, water, (x_px, y_px), homography=H,
                                   extent=plane.get("extent"), lens=lens)["distance_in"]


def test_gt_in_matches_direct_entry_coordinate(plane):
    gt = gt_mod.set_ground_truth(plane, frame=85, x_px=532.8985, y_px=311.032, judge_in=96.5)
    assert gt["gt_in"] == pytest.approx(_direct_gt_in(plane, 532.8985, 311.032))
    assert gt["gt_in_source"] == "plane_homography"


def test_judge_present_vs_absent(plane):
    with_judge = gt_mod.set_ground_truth(plane, frame=85, x_px=532.9, y_px=311.0, judge_in=96.5)
    assert with_judge["judge_present"] is True
    assert with_judge["judge_in"] == pytest.approx(96.5)
    no_judge = gt_mod.set_ground_truth(plane, frame=85, x_px=532.9, y_px=311.0)
    assert no_judge["judge_present"] is False
    assert no_judge["judge_in"] is None


def test_round_trip(tmp_path, plane):
    gt = gt_mod.set_ground_truth(plane, frame=85, x_px=532.9, y_px=311.0, judge_in=96.5)
    gt_mod.write_ground_truth(gt, tmp_path)
    back = gt_mod.read_ground_truth(tmp_path)
    assert back == gt


def test_requires_plane():
    with pytest.raises(ValueError):
        gt_mod.set_ground_truth(None, frame=1, x_px=10, y_px=10)
    with pytest.raises(ValueError):
        gt_mod.set_ground_truth({}, frame=1, x_px=10, y_px=10)


def test_set_judge_preserves_gt_pixel():
    gt = {"schema": "x", "frame": 85, "x_px": 532.9, "y_px": 311.0, "gt_in": 98.0,
          "judge_in": None, "judge_present": False, "updated": "t1"}
    out = gt_mod.set_judge(gt, 96.5, updated="t2")
    # the GT pixel and derived gt_in are untouched; only the judge changes
    assert out["x_px"] == 532.9 and out["y_px"] == 311.0 and out["gt_in"] == 98.0
    assert out["judge_in"] == pytest.approx(96.5)
    assert out["judge_present"] is True
    assert out["updated"] == "t2"
    # clearing the judge is also expressible
    cleared = gt_mod.set_judge(out, None)
    assert cleared["judge_present"] is False and cleared["judge_in"] is None
