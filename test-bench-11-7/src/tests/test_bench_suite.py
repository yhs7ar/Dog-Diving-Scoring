"""M6 — metrics arithmetic + suite runner.

Hand-computed fixtures for the three errors (pixel / score / judge) and the
averages, including a judge-absent clip that contributes to pixel/score but is
excluded from the judge average.  Plus a structural check that ``run_suite``
dispatches ``arc_2d`` through the registry over the real seg01 data.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1]
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SRC))

from test_bench import metrics, suite  # noqa: E402

CLIP_DIR = REPO / "end-to-end-app" / "out" / "video004_right-00.00.36.812-00.00.38.328-seg01"
PLANE = REPO / "end-to-end-app" / "planes" / "plane_test-clip-video004_right.pool.json"
MODEL_SHA = "bdb7688775ee39201d4a77ef0041c7c3e9eb9202f7a5df287a11c98e80ef82e4"


# --- metric arithmetic (hand-computed) ---------------------------------------

def test_pixel_score_judge_errors():
    gt = {"x_px": 100.0, "y_px": 200.0, "gt_in": 98.0, "judge_in": 96.5, "judge_present": True}
    score = {"entry_px": [103.0, 204.0], "distance_in": 99.1}
    e = metrics.compute_errors(score, gt)
    assert e["pixel_error_px"] == pytest.approx(5.0)      # hypot(3, 4)
    assert e["score_error_in"] == pytest.approx(1.1)      # |99.1 - 98.0|
    assert e["judge_error_in"] == pytest.approx(1.5)      # |96.5 - 98.0|
    assert e["accuracy_pct"] == pytest.approx(100.0 * (1.0 - 1.1 / 98.0))


def test_score_accuracy_pct_clamps_and_nulls():
    gt = {"gt_in": 98.0}
    assert metrics.score_accuracy_pct({"distance_in": 98.0}, gt) == 100.0
    assert metrics.score_accuracy_pct({"distance_in": 49.0}, gt) == 50.0
    assert metrics.score_accuracy_pct({"distance_in": 0.0}, gt) == 0.0      # clamped
    assert metrics.score_accuracy_pct({"distance_in": 196.0}, gt) == 0.0    # clamped
    assert metrics.score_accuracy_pct({"distance_in": None}, gt) is None
    assert metrics.score_accuracy_pct({"distance_in": 98.0}, {}) is None
    assert metrics.score_accuracy_pct({"distance_in": 98.0}, {"gt_in": 0.0}) is None


def test_judge_absent_is_null_and_still_comparable():
    gt = {"x_px": 100.0, "y_px": 200.0, "gt_in": 98.0, "judge_present": False}
    score = {"entry_px": [100.0, 200.0], "distance_in": 98.0}
    e = metrics.compute_errors(score, gt)
    assert e["pixel_error_px"] == 0.0
    assert e["score_error_in"] == 0.0
    assert e["judge_error_in"] is None
    assert e["accuracy_pct"] == 100.0


def test_mean_ignores_none():
    assert metrics.mean([2.0, 4.0, None]) == pytest.approx(3.0)
    assert metrics.mean([None, None]) is None
    assert metrics.mean([]) is None


# --- suite -------------------------------------------------------------------

def test_grid_configs():
    cfgs = suite.grid_configs([4, 8], ["last_n", "all"])
    assert len(cfgs) == 4
    assert {c["n_points"] for c in cfgs} == {4, 8}
    assert {c["selection"] for c in cfgs} == {"last_n", "all"}
    assert {c["method"] for c in cfgs} == {"arc_2d"}


def test_grid_configs_multi_method():
    cfgs = suite.grid_configs([4], ["all"], methods=["arc_2d", "arc_3d"])
    assert len(cfgs) == 2
    assert {c["method"] for c in cfgs} == {"arc_2d", "arc_3d"}
    # a lone string is accepted too
    assert suite.grid_configs([4], ["all"], methods="arc_2d")[0]["method"] == "arc_2d"


@pytest.fixture(scope="module")
def seg01():
    if not (CLIP_DIR / "tracks.json").is_file():
        pytest.skip("parent cached artifacts not present on this machine")
    from test_bench import analyze as A
    from dog_diving import trajectory
    records, _ = A.load_cached_records(CLIP_DIR / "tracks.json", model_sha=MODEL_SHA)
    return trajectory.assemble(records), json.loads(PLANE.read_text(encoding="utf-8"))


def test_run_suite_dispatch_and_average(seg01):
    points, plane = seg01
    gt = {"x_px": 532.8985, "y_px": 311.032, "gt_in": 98.0, "judge_in": 96.5, "judge_present": True}
    clips = [{"clip_id": "seg01", "points": points, "plane_doc": plane,
              "gt": gt, "frame_size": [960, 600]}]
    res = suite.run_suite(clips, [{"method": "arc_2d", "n_points": "all", "selection": "all"}])
    assert len(res["rows"]) == 1
    run = res["rows"][0]["runs"][0]
    assert run["score"]["status"] == "ok"
    assert isinstance(run["pixel_error_px"], float)
    assert isinstance(run["score_error_in"], float)
    assert isinstance(run["judge_error_in"], float)
    assert isinstance(run["accuracy_pct"], float)
    assert res["averages"]["per_method"]["arc_2d"]["pixel_error_px"] == pytest.approx(run["pixel_error_px"])
    assert res["averages"]["overall"]["judge_error_in"] == pytest.approx(run["judge_error_in"])
    assert res["averages"]["overall"]["accuracy_pct"] == pytest.approx(run["accuracy_pct"])


def test_run_suite_judge_absent_average(seg01):
    points, plane = seg01
    gt_no_judge = {"x_px": 532.8985, "y_px": 311.032, "gt_in": 98.0, "judge_present": False}
    clips = [{"clip_id": "seg01", "points": points, "plane_doc": plane,
              "gt": gt_no_judge, "frame_size": [960, 600]}]
    res = suite.run_suite(clips, [{"method": "arc_2d", "n_points": "all", "selection": "all"}])
    assert res["averages"]["overall"]["judge_error_in"] is None  # no judge anywhere
    assert res["averages"]["overall"]["score_error_in"] is not None


# --- provenance / invalidation (#1) ------------------------------------------

def test_clip_provenance_and_stale():
    man = {"detections": {"model_sha": "m1"}}
    plane = {"setup_id": "s1"}
    gt = {"updated": "t1"}
    prov = suite.clip_provenance(man, plane, gt)
    assert prov == {"model_sha": "m1", "plane_setup_id": "s1", "gt_updated": "t1",
                    "excluded_frames": []}
    # unchanged -> fresh
    assert suite.is_stale(prov, man, plane, gt) is False
    # re-place GT (new updated timestamp)
    assert suite.is_stale(prov, man, plane, {"updated": "t2"}) is True
    # re-assign plane (new setup_id)
    assert suite.is_stale(prov, man, {"setup_id": "s2"}, gt) is True
    # re-detect with a different model
    assert suite.is_stale(prov, {"detections": {"model_sha": "m2"}}, plane, gt) is True
    # deselect a detection (new excluded_frames)
    assert suite.is_stale(prov, {**man, "excluded_frames": [5]}, plane, gt) is True
    # missing / empty recorded provenance is conservatively stale
    assert suite.is_stale(None, man, plane, gt) is True
    assert suite.is_stale({}, man, plane, gt) is True


def test_provenance_nulls():
    man = {"detections": {}}
    prov = suite.clip_provenance(man, None, None)
    assert prov == {"model_sha": None, "plane_setup_id": None, "gt_updated": None,
                    "excluded_frames": []}
