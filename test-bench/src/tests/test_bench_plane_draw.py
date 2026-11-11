"""Plane re-draw — round-trip identity + validity against the real plane doc."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1]
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SRC))

from test_bench import plane_draw  # noqa: E402

PLANE = REPO / "end-to-end-app" / "planes" / "plane_test-clip-video004_right.pool.json"


@pytest.fixture(scope="module")
def plane():
    if not PLANE.is_file():
        pytest.skip("no plane doc present")
    return json.loads(PLANE.read_text(encoding="utf-8"))


def test_redraw_identity_reproduces_homography(plane):
    """Re-drawing with the same corners reproduces (close to) the same homography."""
    corners = plane["plane"]["corners"]
    redrawn = plane_draw.redraw_plane(plane, corners, setup_id="s-redrawn", updated="t")
    H0 = np.asarray(plane["homography"], dtype=float).reshape(3, 3)
    H1 = np.asarray(redrawn["homography"], dtype=float).reshape(3, 3)
    # same scale-normalised direction
    H0 = H0 / np.linalg.norm(H0)
    H1 = H1 / np.linalg.norm(H1)
    assert np.allclose(H0, H1, atol=1e-4), "redraw of the same corners must not move the homography"
    assert redrawn["setup_id"] == "s-redrawn"
    assert redrawn["plane"]["corners"] == {k: [float(v) for v in plane["plane"]["corners"][k]]
                                           for k in ("00", "10", "11", "01")}


def test_redraw_changes_homography_when_corners_move(plane):
    corners = {k: list(v) for k, v in plane["plane"]["corners"].items()}
    corners["11"] = [corners["11"][0] + 12.0, corners["11"][1] + 8.0]  # nudge one corner
    redrawn = plane_draw.redraw_plane(plane, corners, setup_id="s-moved", updated="t")
    H0 = np.asarray(plane["homography"], dtype=float).reshape(3, 3)
    H1 = np.asarray(redrawn["homography"], dtype=float).reshape(3, 3)
    H0 = H0 / np.linalg.norm(H0)
    H1 = H1 / np.linalg.norm(H1)
    assert not np.allclose(H0, H1, atol=1e-4), "a moved corner must move the homography"


def test_redrawn_doc_still_measures(plane):
    from dog_diving import measure
    from test_bench import analyze
    # a trajectory-free smoke: plane_distance on the redrawn doc still maps a pixel
    corners = plane["plane"]["corners"]
    redrawn = plane_draw.redraw_plane(plane, corners)
    # build a trivial traj-like object with one accepted point at a real corner
    class T:
        def __init__(self, pts):
            self.points = pts
    pts = [{"frame": 85, "x": corners["10"][0], "y": corners["10"][1], "kp_conf": 0.9}]
    m = measure.plane_distance(T(pts), redrawn)
    assert m["total_inches"] is not None and m["total_inches"] > 0


def test_redraw_rejects_bad_input(plane):
    with pytest.raises(ValueError):
        plane_draw.redraw_plane({"plane": {}}, {"00": [0, 0], "10": [1, 1], "11": [2, 2], "01": [3, 3]})
    with pytest.raises(ValueError):
        plane_draw.redraw_plane(plane, {"00": [0, 0]})
