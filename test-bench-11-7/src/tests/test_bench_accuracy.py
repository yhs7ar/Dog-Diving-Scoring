"""Accuracy tab — step 1: scalar metrics (median, percent errors, bin counts).

Hand-computed fixtures only; no I/O, no run files.  The percentage metrics are
``err/GT × 100`` (NOT ``metrics.score_accuracy_pct``), and every one of them must
return ``None`` rather than a number when its denominator is undefined.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))

from test_bench import accuracy  # noqa: E402


# --- median ------------------------------------------------------------------

def test_median_odd_even_empty_and_none():
    assert accuracy.median([3.0, 1.0, 2.0]) == pytest.approx(2.0)          # sorted 1,2,3
    assert accuracy.median([4.0, 1.0, 3.0, 2.0]) == pytest.approx(2.5)     # (2+3)/2
    assert accuracy.median([]) is None
    assert accuracy.median([None, None]) is None
    # None is dropped, not treated as zero
    assert accuracy.median([1.0, None, 3.0]) == pytest.approx(2.0)
    assert accuracy.median([5.0]) == pytest.approx(5.0)


# --- percent errors ----------------------------------------------------------

def test_score_error_pct_value():
    gt = {"gt_in": 120.0}
    score = {"distance_in": 114.0}
    assert accuracy.score_error_pct(score, gt) == pytest.approx(5.0)       # 6/120 = 5 %

    # independent of sign of the miss
    assert accuracy.score_error_pct({"distance_in": 126.0}, gt) == pytest.approx(5.0)


def test_score_error_pct_none_cases():
    assert accuracy.score_error_pct({"distance_in": 114.0}, {}) is None               # no GT
    assert accuracy.score_error_pct({"distance_in": 114.0}, {"gt_in": None}) is None  # null GT
    assert accuracy.score_error_pct({"distance_in": 114.0}, {"gt_in": 0.0}) is None   # 0 GT
    assert accuracy.score_error_pct({"distance_in": 114.0}, {"gt_in": -5.0}) is None  # negative GT
    assert accuracy.score_error_pct({"distance_in": None}, {"gt_in": 120.0}) is None  # no score


def test_judge_error_pct_value_and_none_cases():
    gt = {"gt_in": 120.0, "judge_in": 108.0, "judge_present": True}
    assert accuracy.judge_error_pct(gt) == pytest.approx(10.0)            # 12/120

    assert accuracy.judge_error_pct({"gt_in": 120.0, "judge_present": False,
                                     "judge_in": 108.0}) is None          # no judge
    assert accuracy.judge_error_pct({"gt_in": 120.0, "judge_present": True,
                                     "judge_in": None}) is None           # judge flag, no value
    assert accuracy.judge_error_pct({"gt_in": 0.0, "judge_present": True,
                                     "judge_in": 5.0}) is None            # non-positive GT
    assert accuracy.judge_error_pct({}) is None                           # nothing at all


def test_percent_is_error_over_gt_not_accuracy():
    """Guards the whole point of the tab: ±6 in on a 120 in jump reads 5 %, not 95 %."""
    gt = {"gt_in": 120.0}
    assert accuracy.score_error_pct({"distance_in": 114.0}, gt) == pytest.approx(5.0)
    assert accuracy.score_error_pct({"distance_in": 114.0}, gt) != pytest.approx(95.0)


# --- bin counts --------------------------------------------------------------

def test_bin_counts_counts_and_pcts():
    errors = [1.0, 2.0, 3.0, 5.0, 9.0]          # 5 jumps
    b = accuracy.bin_counts(errors)
    assert b[2]["count"] == 2                   # 1.0, 2.0 (edge inclusive)
    assert b[2]["pct"] == pytest.approx(40.0)
    assert b[4]["count"] == 3                   # 1,2,3
    assert b[4]["pct"] == pytest.approx(60.0)
    assert b[6]["count"] == 4                   # 1,2,3,5
    assert b[6]["pct"] == pytest.approx(80.0)


def test_bin_counts_boundary_ties_are_inclusive():
    b = accuracy.bin_counts([2.0, 4.0, 6.0])
    assert b[2]["count"] == 1 and b[2]["pct"] == pytest.approx(100.0 / 3.0)
    assert b[4]["count"] == 2 and b[4]["pct"] == pytest.approx(200.0 / 3.0)
    assert b[6]["count"] == 3 and b[6]["pct"] == pytest.approx(100.0)


def test_bin_counts_empty_and_none_only():
    empty = accuracy.bin_counts([])
    assert sorted(empty) == [2, 4, 6]
    assert all(v["count"] == 0 and v["pct"] == 0.0 for v in empty.values())

    nunn = accuracy.bin_counts([None, None])
    assert all(v["count"] == 0 and v["pct"] == 0.0 for v in nunn.values())


def test_bin_counts_custom_bins_and_negative_input():
    b = accuracy.bin_counts([-1.5, -3.0], bins=(2,))
    assert b == {2: {"count": 1, "pct": pytest.approx(50.0)}}   # abs applied


# --- nominal in/px -----------------------------------------------------------

def test_nominal_in_per_px_hand_computed():
    """Two marks 10 ft apart, 300 px apart → 120 in / 300 px = 0.4 in/px."""
    doc = {"marks": [
        {"ft": 0.0, "inch": 0.0, "x_px": 100.0, "y_px": 100.0},
        {"ft": 10.0, "inch": 0.0, "x_px": 400.0, "y_px": 100.0},
    ]}
    assert accuracy.nominal_in_per_px(doc) == pytest.approx(0.4)


def test_nominal_in_per_px_7_and_12_foot_marks():
    """The plan's worked example: 7′/12′ marks → 60 in over ~214 px ≈ 0.28 in/px."""
    doc = {"marks": [
        {"ft": 7.0, "inch": 0.0, "x_px": 1090.4279434035109, "y_px": 759.0141391514494},
        {"ft": 12.0, "inch": 0.0, "x_px": 876.5512110350842, "y_px": 758.4102719729003},
    ]}
    assert accuracy.nominal_in_per_px(doc) == pytest.approx(0.2805, abs=1e-3)


def test_nominal_in_per_px_picks_farthest_pair():
    """Three marks; the metric must use the min/max pair, not two neighbours."""
    doc = {"marks": [
        {"ft": 0.0, "inch": 0.0, "x_px": 0.0, "y_px": 0.0},
        {"ft": 5.0, "inch": 0.0, "x_px": 60.0, "y_px": 0.0},     # neighbour pair → 1.0
        {"ft": 15.0, "inch": 0.0, "x_px": 120.0, "y_px": 0.0},
    ]}
    # farthest pair = 0′ and 15′ → 180 in / 120 px = 1.5 in/px
    assert accuracy.nominal_in_per_px(doc) == pytest.approx(1.5)


def test_nominal_in_per_px_fewer_than_two_marks():
    assert accuracy.nominal_in_per_px({}) is None
    assert accuracy.nominal_in_per_px({"marks": []}) is None
    assert accuracy.nominal_in_per_px({"marks": [{"ft": 7.0, "x_px": 1.0, "y_px": 2.0}]}) is None


def test_nominal_in_per_px_identical_marks():
    # same real distance → None
    assert accuracy.nominal_in_per_px({"marks": [
        {"ft": 7.0, "inch": 0.0, "x_px": 1.0, "y_px": 2.0},
        {"ft": 7.0, "inch": 0.0, "x_px": 99.0, "y_px": 99.0},
    ]}) is None
    # same pixel → None
    assert accuracy.nominal_in_per_px({"marks": [
        {"ft": 0.0, "inch": 0.0, "x_px": 5.0, "y_px": 5.0},
        {"ft": 10.0, "inch": 0.0, "x_px": 5.0, "y_px": 5.0},
    ]}) is None


# --- local in/px at entry ----------------------------------------------------

def test_local_in_per_px_uniform_scale():
    """Uniform homography: 48 px/ft → 0.25 in/px, independent of the entry point."""
    H = [[48.0, 0.0, 100.0], [0.0, 48.0, 50.0], [0.0, 0.0, 1.0]]
    doc = {"homography": H}
    # world (10, 5) → pixel (580, 290)
    assert accuracy.local_in_per_px_at_entry(doc, (580.0, 290.0)) == pytest.approx(0.25, abs=1e-9)
    # world (2, 3) → pixel (196, 194)
    assert accuracy.local_in_per_px_at_entry(doc, (196.0, 194.0)) == pytest.approx(0.25, abs=1e-9)


def test_local_in_per_px_singular_homography():
    assert accuracy.local_in_per_px_at_entry({"homography": [[0.0] * 3] * 3},
                                             (100.0, 100.0)) is None


def test_local_in_per_px_missing_inputs():
    assert accuracy.local_in_per_px_at_entry({}, (100.0, 100.0)) is None
    assert accuracy.local_in_per_px_at_entry({"homography": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]},
                                             None) is None


# --- aggregate ---------------------------------------------------------------

def _cell(distance_in, entry_px=None, method="arc_2d", n_points=4, selection="last_n"):
    return {"method": method, "n_points": n_points, "selection": selection,
            "score": {"distance_in": distance_in, "entry_px": entry_px}}


def _plane_doc():
    return {
        # uniform 48 px/ft → 0.25 in/px at any entry point
        "homography": [[48.0, 0.0, 100.0], [0.0, 48.0, 50.0], [0.0, 0.0, 1.0]],
        # 0′ and 10′ marks 300 px apart → 120 in / 300 px = 0.4 in/px
        "marks": [
            {"ft": 0.0, "inch": 0.0, "x_px": 100.0, "y_px": 100.0},
            {"ft": 10.0, "inch": 0.0, "x_px": 400.0, "y_px": 100.0},
        ],
    }


def test_aggregate_full_run():
    cfg = {"method": "arc_2d", "n_points": 4, "selection": "last_n"}
    rows = [
        {"clip_id": "a", "gt": {"gt_in": 100.0, "judge_in": 108.0, "judge_present": True},
         "runs": [_cell(104.0, entry_px=(580.0, 290.0))], "plane_doc": _plane_doc()},
        {"clip_id": "b", "gt": {"gt_in": 120.0, "judge_present": False},
         "runs": [_cell(126.0)]},
        {"clip_id": "c", "gt": {}, "runs": [_cell(50.0)]},
    ]
    out = accuracy.aggregate(rows, cfg)
    clips = out["clips"]
    s = out["summary"]
    assert [c["clip_id"] for c in clips] == ["a", "b", "c"]

    # score errors: a=4, b=6, c excluded (no GT)
    assert s["n_clips"] == 3
    assert s["n_with_judge"] == 1
    assert s["mean_in"] == pytest.approx(5.0)
    assert s["median_in"] == pytest.approx(5.0)
    assert s["mean_pct"] == pytest.approx(4.5)            # 4 % and 5 %
    assert s["median_pct"] == pytest.approx(4.5)
    assert s["bins"][2] == {"count": 0, "pct": pytest.approx(0.0)}
    assert s["bins"][4] == {"count": 1, "pct": pytest.approx(50.0)}
    assert s["bins"][6] == {"count": 2, "pct": pytest.approx(100.0)}
    # judge: clip a only → 8 in, 8 %
    assert s["judge_mean_in"] == pytest.approx(8.0)
    assert s["judge_median_in"] == pytest.approx(8.0)
    assert s["judge_mean_pct"] == pytest.approx(8.0)

    a, b, c = clips
    # in/px comes from the plane doc (nominal) and homography at entry (local)
    assert a["in_per_px_nominal"] == pytest.approx(0.4)
    assert a["in_per_px_local"] == pytest.approx(0.25, abs=1e-9)
    assert b["in_per_px_nominal"] is None                 # no plane doc
    assert b["in_per_px_local"] is None
    assert a["score_error_in"] == pytest.approx(4.0)
    assert a["score_error_pct"] == pytest.approx(4.0)
    assert a["judge_error_in"] == pytest.approx(8.0)
    assert a["judge_error_pct"] == pytest.approx(8.0)
    assert a["within_2"] is False and a["within_4"] is True and a["within_6"] is True
    assert b["score_error_in"] == pytest.approx(6.0)
    assert b["judge_error_in"] is None
    assert b["within_2"] is False and b["within_4"] is False and b["within_6"] is True
    # clip c has no GT → in/px only, no errors, never passes a bin
    assert c["score_error_in"] is None
    assert c["score_error_pct"] is None
    assert c["within_2"] is False and c["within_4"] is False and c["within_6"] is False


def test_aggregate_clip_filter_and_missing_config():
    cfg = {"method": "arc_2d", "n_points": 4, "selection": "last_n"}
    rows = [
        {"clip_id": "a", "gt": {"gt_in": 100.0}, "runs": [_cell(104.0)]},
        {"clip_id": "b", "gt": {"gt_in": 120.0}, "runs": [_cell(126.0)]},
    ]
    out = accuracy.aggregate(rows, cfg, clip_ids=["a"])
    assert [c["clip_id"] for c in out["clips"]] == ["a"]
    assert out["summary"]["n_clips"] == 1
    assert out["summary"]["mean_in"] == pytest.approx(4.0)

    # clip_ids=None → every clip
    assert accuracy.aggregate(rows, cfg)["summary"]["n_clips"] == 2

    # a config with no matching run cell → error and in/px are None, bins empty
    bad_cfg = {"method": "arc_2d", "n_points": 99, "selection": "last_n"}
    out3 = accuracy.aggregate(rows, bad_cfg)
    assert all(c["score_error_in"] is None for c in out3["clips"])
    assert out3["summary"]["mean_in"] is None
    assert out3["summary"]["bins"][2]["count"] == 0


def test_aggregate_matches_all_n_points():
    cfg = {"method": "arc_2d", "n_points": "all", "selection": "last_n"}
    rows = [{"clip_id": "a", "gt": {"gt_in": 100.0},
             "runs": [_cell(104.0, n_points="all")]}]
    out = accuracy.aggregate(rows, cfg)
    assert out["clips"][0]["score_error_in"] == pytest.approx(4.0)

