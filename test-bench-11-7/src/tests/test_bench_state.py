"""Derived clip state, plane resolution, and GT recovery from run snapshots.

These pin the two regressions the 11-7 batch shipped with: a clip whose manifest
cache blocks are empty (``detections: {}``) must still count its accepted points
from ``detections.json``, and a manifest naming a plane that was re-saved under a
timestamped id must still resolve the plane its camera belongs to.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1]
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SRC))

from test_bench import manifest as man_mod  # noqa: E402
import recover_gt_from_results as recover  # noqa: E402

PLANE = REPO / "end-to-end-app" / "planes" / "plane_test-clip-video004_right.pool.json"


# --- fixtures ---------------------------------------------------------------

def _record(frame: int, *, accepted=True, deselected=False):
    return {"frame": frame, "nose": [1.0, 2.0, 0.9], "tailset": [3.0, 4.0, 0.9],
            "box_conf": 0.8, "track_id": None, "accepted": accepted,
            "deselected": deselected}


def _write_clip(root: Path, cid: str, *, frames=(1, 2, 3, 4, 5, 6), setup_id=None,
                camera=None, gt=None, w=960, h=600, excluded=None,
                detections_block=None) -> Path:
    cdir = root / cid
    cdir.mkdir(parents=True, exist_ok=True)
    man = {
        "schema": man_mod.SCHEMA,
        "clip_id": cid,
        "meta": {"width": w, "height": h, "n_frames": 20, "fps": 30.0},
        "plane": {"setup_id": setup_id if setup_id is not None else f"{cid}_plane",
                  "present": bool(setup_id)},
        "detections": detections_block if detections_block is not None else {},
        "trajectory": {},
        "ground_truth": {},
        "labels": {"camera": camera or cid.split("_", 1)[1]},
    }
    if excluded:
        man["excluded_frames"] = list(excluded)
    (cdir / "manifest.json").write_text(json.dumps(man), encoding="utf-8")
    (cdir / "detections.json").write_text(json.dumps([_record(f) for f in frames]),
                                         encoding="utf-8")
    (cdir / "trajectory.json").write_text(json.dumps(
        {"clip_id": cid, "points": [{"frame": f, "x": 10.0 * f, "y": 20.0 * f} for f in frames],
         "inliers": [], "coeffs": None}), encoding="utf-8")
    (cdir / "ground_truth.json").write_text(json.dumps(
        gt if gt is not None else {"schema": "test_bench.ground_truth/1", "frame": None,
                                   "x_px": None, "y_px": None, "gt_in": None,
                                   "judge_in": 238.0, "judge_present": True}),
        encoding="utf-8")
    return cdir


def _plane_doc(setup_id: str, *, created="2026-11-20T00:00:00+00:00", size=(960, 600)):
    return {"schema": "dog_diving.pool_plane/1", "setup_id": setup_id, "created": created,
            "image_size": list(size), "homography": [1, 0, 0, 0, 1, 0, 0, 0, 1],
            "notes": "", "route": "homography"}


@pytest.fixture()
def planes_dir(tmp_path, monkeypatch):
    """An empty saved-planes folder, with PLANES_DIR pointed at it."""
    d = tmp_path / "planes"
    d.mkdir()
    monkeypatch.setattr(man_mod, "PLANES_DIR", d)
    return d


def _save_plane(planes_dir: Path, doc: dict) -> None:
    (planes_dir / f"{doc['setup_id']}.pool.json").write_text(json.dumps(doc), encoding="utf-8")


# --- accepted-point derivation ---------------------------------------------

def test_accepted_frames_honours_flags_and_exclusions():
    recs = [_record(1), _record(2, accepted=False), _record(3, deselected=True),
            _record(4), _record(5), _record(4)]
    assert man_mod.accepted_frames(recs) == [1, 4, 5]
    assert man_mod.accepted_frames(recs, excluded=[5]) == [1, 4]
    # a record with no "accepted" key is accepted (the vendored reference rule)
    assert man_mod.accepted_frames([{"frame": 7}]) == [7]


def test_clip_state_reads_artifacts_not_the_empty_manifest_cache(tmp_path, planes_dir):
    """The 11-7 regression: detections {} in the manifest, real points on disk."""
    doc = _plane_doc("camera0_right_2026-11-20_01-44-19")
    _save_plane(planes_dir, doc)
    cdir = _write_clip(tmp_path, "190321_camera0_right", setup_id="camera0_right_11-7",
                       camera="camera0_right",
                       detections_block={})          # the stale cache

    st = man_mod.clip_state(cdir)
    assert st["n_detections"] == 6
    assert st["n_accepted"] == 6                      # NOT the manifest's 0
    assert st["n_trajectory_points"] == 6
    assert st["plane_source"] == "camera-auto"
    assert st["blockers"] == ["ground-truth point not placed"]


def test_clip_state_ready_when_points_plane_and_gt_present(tmp_path, planes_dir):
    _save_plane(planes_dir, _plane_doc("cam"))
    cdir = _write_clip(tmp_path, "190321_camera0_right", setup_id="cam",
                       gt={"frame": 3, "x_px": 100.0, "y_px": 200.0, "gt_in": 180.0,
                           "judge_in": 238.0, "judge_present": True})
    st = man_mod.clip_state(cdir)
    assert st["ready"] is True
    assert st["blockers"] == []
    assert st["plane_source"] == "manifest"


def test_suite_blockers_report_each_missing_input(tmp_path, planes_dir):
    cdir = _write_clip(tmp_path, "190321_camera0_right", frames=(1, 2),
                       setup_id="camera0_right_11-7", camera="camera0_right")
    blockers = man_mod.clip_state(cdir)["blockers"]
    assert any("need 4" in b for b in blockers)
    assert any("ground-truth point not placed" in b for b in blockers)
    assert any("camera" in b and "plane" in b for b in blockers)


# --- plane resolution -------------------------------------------------------

def test_plane_exact_id_wins_over_camera_match(tmp_path, planes_dir):
    _save_plane(planes_dir, _plane_doc("cam_old", created="2026-01-01T00:00:00+00:00"))
    _save_plane(planes_dir, _plane_doc("cam_new", created="2026-11-20T00:00:00+00:00"))
    man = {"clip_id": "190321_camera0_right", "labels": {"camera": "camera0_right"},
           "meta": {"width": 960, "height": 600}, "plane": {"setup_id": "cam_old"}}
    info = man_mod.plane_resolution(man)
    assert info["source"] == "manifest"
    assert info["setup_id"] == "cam_old"


def test_plane_falls_back_to_newest_camera_plane(tmp_path, planes_dir):
    _save_plane(planes_dir, _plane_doc("camera0_right_2026-11-19_10-00-00",
                                       created="2026-11-19T10:00:00+00:00"))
    _save_plane(planes_dir, _plane_doc("camera0_right_2026-11-20_01-44-19",
                                       created="2026-11-20T01:44:19+00:00"))
    _save_plane(planes_dir, _plane_doc("camera1_right_2026-11-20_01-48-37",
                                       created="2026-11-20T01:48:37+00:00"))
    man = {"clip_id": "190321_camera0_right", "labels": {"camera": "camera0_right"},
           "meta": {"width": 960, "height": 600}, "plane": {"setup_id": "camera0_right_11-7"}}
    info = man_mod.plane_resolution(man)
    assert info["source"] == "camera-auto"
    assert info["setup_id"] == "camera0_right_2026-11-20_01-44-19"
    assert "camera0_right" in (info["note"] or "")


def test_plane_fallback_can_be_refused(tmp_path, planes_dir):
    _save_plane(planes_dir, _plane_doc("camera0_right_2026-11-20_01-44-19"))
    man = {"clip_id": "190321_camera0_right", "labels": {"camera": "camera0_right"},
           "meta": {"width": 960, "height": 600}, "plane": {"setup_id": "camera0_right_11-7"}}
    assert man_mod.clip_plane(man, allow_camera_fallback=False) is None
    assert man_mod.plane_resolution(man, allow_camera_fallback=False)["error_kind"] == "not_found"


def test_plane_size_mismatch_is_loud(tmp_path, planes_dir):
    _save_plane(planes_dir, _plane_doc("cam", size=(1280, 720)))
    man = {"clip_id": "190321_camera0_right", "meta": {"width": 960, "height": 600},
           "plane": {"setup_id": "cam"}}
    info = man_mod.plane_resolution(man)
    assert info["error_kind"] == "size_mismatch"
    with pytest.raises(ValueError, match="960x600"):
        man_mod.clip_plane(man)


def test_plane_mismatched_candidate_is_not_auto_matched(tmp_path, planes_dir):
    _save_plane(planes_dir, _plane_doc("camera0_right_big", size=(1280, 720)))
    man = {"clip_id": "190321_camera0_right", "labels": {"camera": "camera0_right"},
           "meta": {"width": 960, "height": 600}, "plane": {"setup_id": "camera0_right_11-7"}}
    info = man_mod.plane_resolution(man)
    assert info["doc"] is None
    assert "no saved plane matches" in (info["error"] or "")


# --- GT recovery ------------------------------------------------------------

@pytest.fixture(scope="module")
def real_plane():
    if not PLANE.is_file():
        pytest.skip("parent plane document not present on this machine")
    return json.loads(PLANE.read_text(encoding="utf-8"))


def test_recover_gt_from_snapshot(tmp_path, planes_dir, real_plane):
    size = real_plane.get("image_size") or [960, 600]
    _save_plane(planes_dir, real_plane)
    cdir = _write_clip(tmp_path, "190321_camera0_right", setup_id=real_plane["setup_id"],
                       w=int(size[0]), h=int(size[1]),
                       gt={"frame": None, "x_px": None, "y_px": None, "gt_in": None,
                           "judge_in": 238.0, "judge_present": True,
                           "reference": {"dog": "lida ford", "score_raw": "19'10\"",
                                         "csv_line": 3}})
    res = tmp_path / "results"
    res.mkdir()
    snap = {"run_id": "2026-11-20T07-32-40.207077Z00-00",
            "rows": [{"clip_id": "190321_camera0_right",
                      "gt": {"frame": 15, "x_px": 532.8985, "y_px": 311.032,
                             "gt_in": 0.0, "judge_in": 238.0}}]}
    (res / "run.json").write_text(json.dumps(snap), encoding="utf-8")

    assert recover.main(["--root", str(tmp_path), "--apply"]) == 0

    gt = json.loads((cdir / "ground_truth.json").read_text(encoding="utf-8"))
    assert gt["x_px"] == pytest.approx(532.8985)
    assert gt["gt_in"] > 0
    assert gt["reference"]["csv_line"] == 3            # provenance preserved
    man = man_mod.read_manifest(cdir)
    assert man["ground_truth"]["gt_in"] == pytest.approx(gt["gt_in"])
    # the recovered record now makes the clip suite-ready (points/plane/GT)
    assert man_mod.clip_state(cdir)["blockers"] == []

    # idempotent: a placed GT is never overwritten
    assert recover.main(["--root", str(tmp_path), "--apply"]) == 0
    again = json.loads((cdir / "ground_truth.json").read_text(encoding="utf-8"))
    assert again["frame"] == 15
