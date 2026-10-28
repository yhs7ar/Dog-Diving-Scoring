"""M23 — Review's deselection set: the other-dog fix.

The operator's report (2026-10-21): *"the current model has high confidence and if there is another
dog in the frame issues arise, but this can be resolved by allowing the user to manually deselect
detected points."*

So a deselected detection has to leave the measurement as well as the drawing.  Before this file,
``ReviewState.excluded`` only changed the fitted arc — ``measure.plane_distance`` still took the
highest-frame detection regardless, which is exactly the point a second dog would supply.  These
tests pin the four places the set now reaches:

1. ``measure.final_point`` / ``measure.plane_distance`` — the measured point;
2. ``ReviewState.measurement`` / ``.arc`` — the number Review shows;
3. ``ReviewState.point_rows`` — the list it is edited from;
4. ``report.build_result`` / ``build_report_text`` — what the export records.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tests import pool_scene as PS
from tests.test_m19_measure import plane_doc, traj_with

from dog_diving import annotate as A
from dog_diving import measure as M
from dog_diving import report as R

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLIP = PROJECT_ROOT / "data" / "clips" / "video004_right-00.00.36.812-00.00.38.328-seg01.mp4"
NEEDS_CLIP = pytest.mark.skipif(not CLIP.is_file(), reason="sample clip missing")


@pytest.fixture(scope="module")
def sc():
    return PS.deck_marks_scene()


@pytest.fixture(scope="module")
def doc(sc):
    return plane_doc(sc)


def arc_points(scene, doc, xs_ft=(4.0, 6.0, 8.0, 10.0, 12.0), y_ft=8.0):
    """Five on-plane detections, frame 100, 110, ... — the last one is the "other dog"."""
    out = []
    for i, X in enumerate(xs_ft):
        v = scene.project(X, y_ft, scene.water_z_ft)
        out.append({"frame": 100 + 10 * i, "x": float(v[0]), "y": float(v[1]), "kp_conf": 0.95})
    return out


# ---------------------------------------------------------------------------
# 1. the measured point
# ---------------------------------------------------------------------------
def test_final_point_skips_deselected_frames():
    pts = [{"frame": f, "x": float(f), "y": 1.0} for f in (10, 20, 30)]
    traj = traj_with(pts)
    assert M.final_point(traj)["frame"] == 30
    assert M.final_point(traj, {30})["frame"] == 20
    assert M.final_point(traj, {20, 30})["frame"] == 10
    assert M.final_point(traj, {10, 20, 30}) is None


def test_plane_distance_reports_the_highest_kept_point(sc, doc):
    pts = arc_points(sc, doc)
    traj = traj_with(pts)
    base = M.plane_distance(traj, doc)
    assert base["total_inches"] is not None
    assert base["frame"] == pts[-1]["frame"]

    # the operator clicks the last detection out (a second dog): the distance becomes the previous
    # kept point, not the one that was clicked away
    after = M.plane_distance(traj, doc, excluded={pts[-1]["frame"]})
    assert after["frame"] == pts[-2]["frame"]
    assert after["total_inches"] != pytest.approx(base["total_inches"])
    assert after["excluded_frames"] == [pts[-1]["frame"]]
    prev = M.plane_distance(traj, doc, frame=pts[-2]["frame"])
    assert after["total_inches"] == pytest.approx(prev["total_inches"])


def test_deselecting_everything_reports_no_distance_not_a_guess(sc, doc):
    pts = arc_points(sc, doc)
    traj = traj_with(pts)
    out = M.plane_distance(traj, doc, excluded={p["frame"] for p in pts})
    assert out["total_inches"] is None
    assert any("deselected" in w for w in out["warnings"]), out["warnings"]


def test_a_deselected_frame_cannot_be_named_as_the_measured_one(sc, doc):
    traj = traj_with(arc_points(sc, doc))
    out = M.plane_distance(traj, doc, frame=140, excluded={140})
    assert out["total_inches"] is None
    assert any("deselected" in w for w in out["warnings"])


# ---------------------------------------------------------------------------
# 2. what Review shows
# ---------------------------------------------------------------------------
@pytest.fixture()
def review(sc, doc):
    app_gui = pytest.importorskip("app_gui")
    traj = traj_with(arc_points(sc, doc))
    return app_gui, app_gui.ReviewState(clip=Path("x.mp4"), traj=traj, contact=None,
                                        calib=None, image_size=sc.image_size, pool_doc=doc)


def test_review_measurement_follows_the_deselections(review):
    _app_gui, st = review
    before = st.measurement()
    last = int(st.traj.points[-1]["frame"])
    assert st.toggle_point(last) is True
    after = st.measurement()
    assert before["frame"] == last
    assert after["frame"] == last - 10
    assert after["total_inches"] != pytest.approx(before["total_inches"])
    assert after["excluded_frames"] == [last]

    st.include_all()
    assert st.measurement()["total_inches"] == pytest.approx(before["total_inches"])
    assert st.measurement()["excluded_frames"] == []


def test_point_rows_mirror_the_excluded_set(review):
    _app_gui, st = review
    rows = st.point_rows()
    assert [r["frame"] for r in rows] == [int(p["frame"]) for p in st.traj.points]
    assert all(r["included"] for r in rows)
    st.toggle_point(rows[1]["frame"])
    rows = st.point_rows()
    assert rows[1]["included"] is False
    assert sum(1 for r in rows if not r["included"]) == 1
    st.include_all()
    assert all(r["included"] for r in st.point_rows())


# ---------------------------------------------------------------------------
# 3. what the export records
# ---------------------------------------------------------------------------
def test_result_json_gains_a_review_block_only_when_something_is_excluded(sc, doc):
    traj = traj_with(arc_points(sc, doc))
    meas = M.plane_distance(traj, doc, excluded={traj.points[-1]["frame"]})
    kwargs = dict(clip_id="c", traj=traj, measurement=meas, still_frame=1)

    plain = R.build_result(**kwargs)
    assert "review" not in plain, "an untouched run must keep its previous key set (PLAN §11)"

    marked = R.build_result(**kwargs, excluded_frames=meas["excluded_frames"],
                            arc_after_exclusion={"n": 4, "rms_px": 0.7, "frame_range": [100, 130]})
    rev = marked["review"]
    assert rev["excluded_frames"] == [int(traj.points[-1]["frame"])]
    assert rev["n_excluded"] == 1
    assert rev["arc_n"] == 4
    assert marked["measurement"]["frame"] == int(traj.points[-2]["frame"])


def test_report_text_states_the_deselection(sc, doc):
    traj = traj_with(arc_points(sc, doc))
    meas = M.plane_distance(traj, doc, excluded={traj.points[-1]["frame"]})
    result = R.build_result(clip_id="c", traj=traj, measurement=meas,
                            excluded_frames=meas["excluded_frames"])
    text = R.build_report_text(result)
    assert "review" in text
    assert str(int(traj.points[-1]["frame"])) in text
    assert "highest-frame KEPT detection" in text


# ---------------------------------------------------------------------------
# 4. the exported picture
# ---------------------------------------------------------------------------
@NEEDS_CLIP
def test_annotated_still_drops_the_excluded_marker(tmp_path: Path, sc, doc):
    traj = traj_with(arc_points(sc, doc))
    meas = M.plane_distance(traj, doc)
    plain = A.write_still(CLIP, tmp_path / "plain.png", 0, traj=traj, measurement=meas,
                          show_grid=False)
    out = A.write_still(CLIP, tmp_path / "out.png", 0, traj=traj, measurement=meas,
                        show_grid=False, arc=None, excluded={traj.points[-1]["frame"]})
    import cv2
    a, b = cv2.imread(plain["path"]), cv2.imread(out["path"])
    assert b is not None
    assert a.shape == b.shape
    assert int(np.abs(a.astype(int) - b.astype(int)).sum()) > 0, \
        "the exported still ignored the deselection"


# ---------------------------------------------------------------------------
# 5. the list opens on the points the arc uses (order 2026-10-28)
# ---------------------------------------------------------------------------
def _traj_55():
    """55 accepted detections with the parabola fitted on the last 20 — a real run's shape."""
    pts = [{"frame": 32 + i, "x": 800.0 - 3.0 * i, "y": 300.0 + 1.5 * i, "kp_conf": 0.9}
           for i in range(55)]
    traj = traj_with(pts)
    traj.fit_points = pts[-20:]                 # exactly what build_trajectory records
    return traj, pts


def test_the_list_opens_on_the_points_the_arc_uses():
    """Order 2026-10-28: *"automatically deselect the points not being used for the trajectory"*.

    A 55-detection run read "55 of 55 point(s) included" while the arc fitted 20 of them.
    """
    app_gui = pytest.importorskip("app_gui")
    traj, pts = _traj_55()
    st = app_gui.ReviewState(clip=Path("x.mp4"), traj=traj, contact=None, calib=None,
                             image_size=(960, 600))
    assert sum(r["included"] for r in st.point_rows()) == 55

    dropped = st.focus_trajectory_window()
    assert dropped == 35

    rows = st.point_rows()
    assert sum(r["included"] for r in rows) == 20
    # exactly the twenty the fit used — the LAST twenty, not the first
    assert [r["frame"] for r in rows if r["included"]] == [p["frame"] for p in pts[-20:]]
    # and the measured point is untouched: the final detection was, and still is, included
    assert int(M.final_point(traj, st.excluded)["frame"]) == pts[-1]["frame"]


def test_the_drawn_points_are_exactly_the_fitted_window():
    """The canvas keeps the same twenty the quadratic was fitted on — not a nearby set."""
    app_gui = pytest.importorskip("app_gui")
    traj, _pts = _traj_55()
    st = app_gui.ReviewState(clip=Path("x.mp4"), traj=traj, contact=None, calib=None,
                             image_size=(960, 600))
    st.focus_trajectory_window()
    assert [int(p["frame"]) for p in st.draw_points()] == [int(p["frame"]) for p in traj.fit_points]


def test_focus_is_a_no_op_when_there_is_no_window():
    """No parabola window (the gates rejected it, or no frame size): nothing is hidden."""
    app_gui = pytest.importorskip("app_gui")
    pts = [{"frame": f, "x": float(f), "y": 1.0} for f in (10, 20, 30)]
    st = app_gui.ReviewState(clip=Path("x.mp4"), traj=traj_with(pts), contact=None, calib=None,
                             image_size=(960, 600))
    assert st.focus_trajectory_window() == 0
    assert sum(r["included"] for r in st.point_rows()) == 3
