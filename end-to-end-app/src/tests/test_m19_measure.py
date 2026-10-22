"""M19 — the plane is the only ground truth, and the detector is the fork's.

Two rules, both from the operator (2026-10-20), pinned here:

1. *"Assume that the final point detected for the dog's tailset is the point of intersection with
   the plane. The +x distance at this point will be the reported distance."*  —
   :func:`dog_diving.measure.plane_distance` samples the **last accepted detection**, maps it through
   the plane document's own homography, and reports its X. No contact candidates, no arc, no 1-D fit.
2. The detector is ``image_set.detect_frames`` (``model.predict``, no tracker), so there is no
   dominant-track filter to discard the dog.  ``detect_clip`` must report the tracker fields as
   absent rather than inventing them.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests import pool_scene as PS

from dog_diving import detect as D
from dog_diving import measure as M
from dog_diving import trajectory as T
from dog_diving.pool import calib as C
from dog_diving.pool import plane as PL
from dog_diving.pool.lens import LensModel

CLIP = "data/clips/video004_right-00.00.36.812-00.00.38.328-seg01.mp4"


@pytest.fixture(scope="module")
def sc():
    return PS.deck_marks_scene()


def plane_doc(scene, length: float = 25.0) -> dict:
    W = float(scene.width_ft)

    def px(X, Y):
        v = scene.project(X, Y, scene.water_z_ft)
        return (float(v[0]), float(v[1]))

    s = PL.PlaneSession(frame_shape=(scene.image_size[1], scene.image_size[0]),
                        length_ft=length, width_ft=W)
    for tag, (X, Y) in zip(PL.CORNER_TAGS, ((0, 0), (length, 0), (length, W), (0, W))):
        s.set_corner(tag, *px(X, Y))
    return s.build("m19", lens=LensModel(K=scene.K, dist=scene.dist,
                                         image_size=scene.image_size, source="checkerboard"))


def traj_with(points: list[dict]):
    """A Trajectory whose accepted points are exactly ``points``."""
    recs = []
    for p in points:
        recs.append({"frame": p["frame"], "tailset": [p["x"], p["y"], p.get("kp_conf", 0.9)],
                     "nose": [p["x"] - 40, p["y"] - 20, 0.9], "box_conf": 0.4,
                     "track_id": None, "accepted": True})
    return T.build_trajectory(recs, clip_id="m19")


# ---------------------------------------------------------------------------
# 1. the last detection is the measurement
# ---------------------------------------------------------------------------
def test_the_final_detection_is_the_reported_distance(sc):
    doc = plane_doc(sc)
    # three detections, the LAST one at 12 ft from the dock edge, 9 ft across
    pts = []
    for f, (X, Y) in ((10, (2.0, 8.0)), (20, (7.0, 6.0)), (30, (12.0, 9.0))):
        raw = sc.project(X, Y, sc.water_z_ft)
        pts.append({"frame": f, "x": float(raw[0]), "y": float(raw[1])})
    traj = traj_with(pts)

    out = M.plane_distance(traj, doc)
    assert out["route"] == M.ROUTE
    assert out["frame"] == 30, "the measurement must be the LAST accepted detection"
    assert out["total_inches"] == pytest.approx(12.0 * 12.0, abs=0.2)
    assert out["distance_ft"] == pytest.approx(12.0, abs=0.02)
    assert out["transverse_ft"] == pytest.approx(9.0, abs=0.05)
    assert out["point_source"] == "model"
    assert out["warnings"] == []
    print(f"\n  final f30 -> {out['display']} ({out['total_inches']:.2f} in), "
          f"X {out['distance_ft']:.2f} ft, Y {out['transverse_ft']:.2f} ft")


def test_a_manual_point_overrides_under_the_same_assumption(sc):
    doc = plane_doc(sc)
    raw = sc.project(5.0, 4.0, sc.water_z_ft)
    traj = traj_with([{"frame": 5, "x": float(raw[0]), "y": float(raw[1])}])
    clicked = sc.project(20.0, 2.0, sc.water_z_ft)
    out = M.plane_distance(traj, doc, manual_point=(float(clicked[0]), float(clicked[1])))
    assert out["point_source"] == "manual" and out["point_origin"] == "click"
    assert out["total_inches"] == pytest.approx(20.0 * 12.0, abs=0.2)


def test_without_a_plane_there_is_no_number(sc):
    raw = sc.project(5.0, 4.0, sc.water_z_ft)
    traj = traj_with([{"frame": 5, "x": float(raw[0]), "y": float(raw[1])}])
    out = M.plane_distance(traj, None)
    assert out["total_inches"] is None and out["display"] == "n/a"
    assert any("no plane calibration" in w for w in out["warnings"])


def test_a_frame_restriction_uses_that_frames_detection(sc):
    doc = plane_doc(sc)
    pts = []
    for f, X in ((10, 3.0), (20, 8.0)):
        raw = sc.project(X, 8.0, sc.water_z_ft)
        pts.append({"frame": f, "x": float(raw[0]), "y": float(raw[1])})
    traj = traj_with(pts)
    assert M.plane_distance(traj, doc, frame=10)["distance_ft"] == pytest.approx(3.0, abs=0.02)
    assert M.plane_distance(traj, doc)["distance_ft"] == pytest.approx(8.0, abs=0.02)


# ---------------------------------------------------------------------------
# 2. the detector is the fork's: no tracker, and the cache says so
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not __import__("pathlib").Path(CLIP).is_file(), reason="clip missing")
def test_detect_clip_uses_predict_and_reports_no_tracker():
    r = D.detect_clip(CLIP, D.DetectConfig(), frame_range=(40, 50), use_cache=False)
    assert D.DETECTOR == "predict_v1"
    assert r.dominant_track is None and r.tracking_available is False and r.track_coverage == 0.0
    assert r.records and all(rec["track_id"] is None for rec in r.records)
    # every frame is decided on the keypoint, not on a track id
    assert r.n_accepted == sum(1 for rec in r.records if rec.get("accepted"))
    print(f"\n  {len(r.records)} frames, {r.n_accepted} accepted, no tracker")


@pytest.mark.skipif(not __import__("pathlib").Path(CLIP).is_file(), reason="clip missing")
def test_the_cache_entry_records_its_own_provenance(tmp_path):
    r = D.detect_clip(CLIP, D.DetectConfig(), frame_range=(40, 44), out_dir=tmp_path)
    import json
    doc = json.loads((tmp_path / __import__("pathlib").Path(CLIP).stem / "tracks.json")
                     .read_text(encoding="utf-8"))
    entry = doc["entries"][r.cache_key_hash]
    # the file-level header is rewritten by every writer, so an entry that only trusts it cannot
    # say which model/detector produced it (measured 2026-10-20: it could not be answered from disk)
    assert entry["model_sha"] == r.model_sha
    assert entry["detector"] == "predict_v1"
    assert r.from_cache is False
    again = D.detect_clip(CLIP, D.DetectConfig(), frame_range=(40, 44), out_dir=tmp_path)
    assert again.from_cache is True
