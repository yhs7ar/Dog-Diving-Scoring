"""M3 — analyze-wiring reproduction pins (torch-free).

These prove the copied ``trajectory`` / ``measure`` / ``pool`` modules behave
identically to the parent on this machine, using the parent's *cached* detection
records (``out/<clip>/tracks.json``) and its recorded ``trajectory.json`` /
``result.json``.  No YOLO re-detection (that is the torch-dependent M3b).

Pins (parent clip ``video004_right-…-seg01``, model ``bdb76887…``):

  * ``build_trajectory`` reproduces ``trajectory.json`` byte-for-byte (with the
    parent's ``clip_id``).
  * ``plane_distance`` reproduces the recorded pixel entry ``x_px / y_px / frame``
    exactly.

``total_inches`` is *not* pinned to the historical value: the plane doc the parent
used on 2026-11-06 was later recalibrated, so the same pixel now maps to a
different distance.  The assertion is that the homography *is* applied (finite,
positive inches) — wiring correctness — not the stale absolute number.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1]          # test-bench/src
REPO = Path(__file__).resolve().parents[3]          # Dog-Diving-Scoring
sys.path.insert(0, str(SRC))

from test_bench import analyze as A  # noqa: E402

CLIP_DIR = (REPO / "end-to-end-app" / "out" / "video004_right-00.00.36.812-00.00.38.328-seg01")
PLANE_DOC = REPO / "end-to-end-app" / "planes" / "plane_test-clip-video004_right.pool.json"
MODEL_SHA = "bdb7688775ee39201d4a77ef0041c7c3e9eb9202f7a5df287a11c98e80ef82e4"

needs_parent = pytest.mark.skipif(
    not (CLIP_DIR / "tracks.json").is_file(),
    reason="parent cached artifacts not present on this machine",
)


@needs_parent
def _fixtures():
    records, _ = A.load_cached_records(CLIP_DIR / "tracks.json", model_sha=MODEL_SHA)
    parent_traj = json.loads((CLIP_DIR / "trajectory.json").read_text(encoding="utf-8"))
    result = json.loads((CLIP_DIR / "result.json").read_text(encoding="utf-8"))
    plane = json.loads(PLANE_DOC.read_text(encoding="utf-8"))
    clip_id = parent_traj["clip_id"]
    return records, parent_traj, result, plane, clip_id


@needs_parent
def test_trajectory_reproduces_parent():
    records, parent_traj, _, _, clip_id = _fixtures()
    traj = A.build_trajectory(records, clip_id=clip_id, frame_size=(960, 600))
    assert traj.as_dict() == parent_traj


@needs_parent
def test_measure_reproduces_pixel_entry():
    records, _, result, plane, clip_id = _fixtures()
    meas = A.measure(records, plane, clip_id=clip_id, frame_size=(960, 600))
    target = result["measurement"]
    assert meas["x_px"] == target["x_px"]
    assert meas["y_px"] == target["y_px"]
    assert meas["frame"] == target["frame"]
    assert meas["route"] == result["measurement_route"]


@needs_parent
def test_measure_applies_homography():
    records, _, _, plane, clip_id = _fixtures()
    meas = A.measure(records, plane, clip_id=clip_id, frame_size=(960, 600))
    assert meas["total_inches"] is not None and meas["total_inches"] > 0
    assert meas["trajectory"] is not None  # trajectory attached for trajectory.json artifact


def test_load_cached_records_selects_and_refuses(tmp_path):
    doc = {"entries": {
        "k1": {"model_sha": "aaa", "records": [{"frame": 1, "tailset": [1, 2, 0.9], "accepted": True}]},
        "k2": {"model_sha": "bbb", "records": [{"frame": 2, "tailset": [3, 4, 0.8], "accepted": True}]},
    }}
    p = tmp_path / "tracks.json"
    p.write_text(json.dumps(doc), encoding="utf-8")
    # refuses ambiguous
    with pytest.raises(ValueError):
        A.load_cached_records(p)
    # selects by model_sha
    records, meta = A.load_cached_records(p, model_sha="bbb")
    assert records[0]["frame"] == 2
    # unknown model_sha
    with pytest.raises(ValueError):
        A.load_cached_records(p, model_sha="zzz")
