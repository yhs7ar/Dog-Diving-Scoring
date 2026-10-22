"""M24 — the 2026-10-21 Calibrate/Review order: near anchor, parallelogram, snap, entry frame.

Pinned here because each one is a rule, not a preference:

* **Two +x anchors.**  ``dock_offset_ft`` is the X of the quad's DOCK-SIDE edge, so an operator can
  put the corners on the 6 ft pool marks instead of on a corner outside the frame.  The axis still
  runs 0 -> 35 ft and reported X is still measured from the dock, which is the whole point.
* **Parallelogram is a starting shape, not a constraint.**  ``make_parallelogram`` averages the two
  edge directions and rebuilds the corners; dragging afterwards is free (the homography needs the
  perspective skew).
* **The far edge can be held on the detected far waterline** while the operator drags.
* **The entry frame is an operator override of the measured point** and it is recorded as one.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tests import pool_scene as PS
from tests.test_m19_measure import plane_doc, traj_with

import app_gui
from dog_diving import measure as M
from dog_diving import report as R
from dog_diving.pool import calib as C
from dog_diving.pool import plane as PL


@pytest.fixture(scope="module")
def sc():
    return PS.deck_marks_scene()


@pytest.fixture()
def doc(sc):
    return plane_doc(sc)


def _corners(sc, length=25.0, offset=0.0):
    W = float(sc.width_ft)

    def px(X, Y):
        v = sc.project(X, Y, sc.water_z_ft)
        return (float(v[0]), float(v[1]))

    return {"00": px(offset, 0), "10": px(offset + length, 0),
            "11": px(offset + length, W), "01": px(offset, W)}


def _session(sc, *, offset=0.0, length=25.0, width=None):
    s = PL.PlaneSession(frame_shape=(sc.image_size[1], sc.image_size[0]),
                        length_ft=length, width_ft=width or float(sc.width_ft),
                        dock_offset_ft=offset)
    for tag, p in _corners(sc, length, offset).items():
        s.set_corner(tag, *p)
    return s


# ---------------------------------------------------------------------------
# 1. the near anchor (dock_offset_ft)
# ---------------------------------------------------------------------------
def test_the_near_anchor_moves_the_quad_without_moving_the_axis(sc) -> None:
    s = _session(sc, offset=6.0, length=19.0)
    Xs = sorted(c["X_ft"] for c in s.correspondences())
    assert Xs[0] == pytest.approx(6.0), Xs
    assert Xs[-1] == pytest.approx(25.0), Xs
    # the axis is unchanged: a distance may still be reported from 0 to 35 ft
    assert s.extent()["x_min"] == 0.0 and s.extent()["x_max"] == PL.METRIC_LENGTH_FT
    # and the quad's own corner maps back to the same absolute X
    px = s.corners["00"]
    X, Y = s.world_at(*px)
    assert X == pytest.approx(6.0, abs=1e-6), (X, Y)


def test_a_zero_anchor_leaves_an_old_document_byte_identical(sc) -> None:
    s = _session(sc)
    doc0 = s.build("t", lens=None)
    again = PL.PlaneSession.from_doc(doc0)
    assert again.dock_offset_ft == 0.0
    assert [c["X_ft"] for c in again.correspondences()] == \
           [c["X_ft"] for c in s.correspondences()]


def test_the_offset_survives_a_document_round_trip(sc) -> None:
    s = _session(sc, offset=6.0, length=19.0)
    doc = s.build("offset", lens=None)
    assert doc["plane"]["dock_offset_ft"] == pytest.approx(6.0)
    back = PL.PlaneSession.from_doc(doc)
    assert back.dock_offset_ft == pytest.approx(6.0)
    assert back.length_ft == pytest.approx(19.0)
    for tag in PL.CORNER_TAGS:
        assert back.corners[tag] == pytest.approx(s.corners[tag])


def test_the_measured_distance_still_counts_from_the_dock(sc) -> None:
    """A point 12 ft from the dock must read 12 ft even when the quad starts at 6 ft."""
    s = _session(sc, offset=6.0, length=19.0)
    doc = s.build("offset", lens=None)
    traj = traj_with([{"frame": 10, "x": 100.0, "y": 100.0},
                      {"frame": 11, "x": 101.0, "y": 101.0}])
    raw = sc.project(12.0, 8.0, sc.water_z_ft)
    traj.points = [{"frame": 10, "x": float(raw[0]), "y": float(raw[1]), "kp_conf": 0.9}]
    m = M.plane_distance(traj, doc)
    assert m["total_inches"] is not None, m["warnings"]
    assert m["distance_ft"] == pytest.approx(12.0, abs=0.05), m


# ---------------------------------------------------------------------------
# 2. the parallelogram starting shape
# ---------------------------------------------------------------------------
def test_make_parallelogram_keeps_the_centre_and_parallelises_the_edges(sc) -> None:
    s = _session(sc)
    # skew one corner so the quad is a general quad, like a hand placement
    s.corners["10"] = (s.corners["10"][0] + 18.0, s.corners["10"][1] - 9.0)
    before = np.array([s.corners[t] for t in PL.CORNER_TAGS], float)
    centre_before = before.mean(axis=0)

    s.make_parallelogram()
    p = {t: np.array(s.corners[t], float) for t in PL.CORNER_TAGS}
    assert np.allclose(p["10"] - p["00"], p["11"] - p["01"], atol=1e-9), "X edges not parallel"
    assert np.allclose(p["01"] - p["00"], p["11"] - p["10"], atol=1e-9), "Y edges not parallel"
    assert np.allclose(np.array(list(p.values())).mean(axis=0), centre_before, atol=1e-6), (
        "the shape moved instead of being re-squared")


def test_make_parallelogram_refuses_an_unplaced_quad(sc) -> None:
    s = PL.PlaneSession(frame_shape=(sc.image_size[1], sc.image_size[0]))
    assert s.make_parallelogram()  # the prompt, not a crash
    assert not s.placed


# ---------------------------------------------------------------------------
# 3. edge snapping — one side at a time, named the way the pane names it
# ---------------------------------------------------------------------------
def _quad(sc):
    s = PL.PlaneSession(frame_shape=(600, 960), length_ft=20.0, width_ft=12.0)
    for tag, p in (("00", (100.0, 400.0)), ("10", (300.0, 400.0)),
                   ("11", (300.0, 300.0)), ("01", (100.0, 300.0))):
        s.set_corner(tag, *p)
    return s


def _dist_to_line(q, point, direction):
    d = np.asarray(direction, float) / np.linalg.norm(direction)
    foot = np.asarray(point, float) + d * float(np.dot(np.asarray(q, float) - point, d))
    return float(np.hypot(*(np.asarray(q, float) - foot)))


def test_snapping_one_side_moves_only_that_side(sc) -> None:
    """``back`` is the plane's Top edge, which is the far waterline the seed was built from."""
    s = _quad(sc)
    untouched = {t: s.corners[t] for t in ("01", "11")}
    line_p, line_d = np.array([0.0, 440.0]), np.array([1.0, 0.1])
    assert s.snap_side_to_line("back", line_p, line_d) is True
    for tag in ("00", "10"):
        assert _dist_to_line(s.corners[tag], line_p, line_d) < 1e-6, tag
    for tag, p in untouched.items():
        assert s.corners[tag] == pytest.approx(p), "snapping moved the opposite edge"


def test_snapping_the_right_side_moves_the_x_equals_l_edge(sc) -> None:
    """The pool's far END waterline: the two far-X corners, not the far-Y ones."""
    s = _quad(sc)
    untouched = {t: s.corners[t] for t in ("00", "01")}
    line_p, line_d = np.array([320.0, 0.0]), np.array([0.2, 1.0])
    assert s.snap_side_to_line("right", line_p, line_d) is True
    for tag in ("10", "11"):
        assert _dist_to_line(s.corners[tag], line_p, line_d) < 1e-6, tag
    for tag, p in untouched.items():
        assert s.corners[tag] == pytest.approx(p)


def test_snapping_a_placed_edge_twice_is_idempotent() -> None:
    s = _quad(None)
    line = (np.array([0.0, 420.0]), np.array([1.0, 0.0]))
    assert s.snap_far_edge_to_line(*line) is True
    assert s.snap_far_edge_to_line(*line) is False, "a second snap reported another move"


def test_snapping_an_unknown_side_is_a_no_op() -> None:
    s = _quad(None)
    before = dict(s.corners)
    assert s.snap_side_to_line("middle", np.array([0.0, 0.0]), np.array([1.0, 0.0])) is False
    assert s.corners == before


# ---------------------------------------------------------------------------
# 4. the operator's entry frame
# ---------------------------------------------------------------------------
class _Contact:
    """The minimum of ``contact.ContactChoice`` that ReviewState actually uses."""

    def __init__(self) -> None:
        self.chosen_frame: int | None = None
        self.chosen_source = "smoke"

    def choose(self, frame: int, source: str = "manual"):
        self.chosen_frame, self.chosen_source = int(frame), source
        return self

    def nudge(self, delta: int):
        self.chosen_frame = int((self.chosen_frame or 0) + int(delta))
        return self

    def ranked(self):
        return []


def _review(sc, doc, frames=8):
    pts = [{"frame": 10 + 3 * i, "x": 400.0 + 6.0 * i, "y": 300.0 - 12.0 * i}
           for i in range(frames)]
    traj = traj_with(pts)
    con = _Contact().choose(int(pts[-1]["frame"]), "arc_waterline")
    return app_gui.ReviewState(clip=Path("x.mp4"), traj=traj, contact=con, calib=None,
                               image_size=(sc.image_size[0], sc.image_size[1]),
                               pool_doc=doc)


def test_the_entry_frame_overrides_the_final_detection(sc, doc) -> None:
    st = _review(sc, doc)
    assert st.cutoff() is None
    assert st.measurement()["frame"] == st.traj.points[-1]["frame"]
    assert st.measurement()["entry_override"] is False

    want = st.traj.points[2]["frame"]
    st.set_entry_frame(want)
    m = st.measurement()
    assert m["frame"] == want, m
    assert m["entry_override"] is True
    assert st.arc()["frame_range"][1] <= want, "the drawn arc runs past the entry frame"
    assert all(int(p["frame"]) <= want for p in st.draw_points())


def test_an_entry_frame_with_no_detection_snaps_to_the_nearest_one(sc, doc) -> None:
    st = _review(sc, doc)
    frames = [int(p["frame"]) for p in st.traj.points]
    empty = next(f for f in range(min(frames), max(frames)) if f not in frames)
    got = st.set_entry_frame(empty)
    assert got in frames, (empty, got)
    assert st.measurement()["frame"] == got


def test_the_entry_frame_cannot_be_a_deselected_point(sc, doc) -> None:
    st = _review(sc, doc)
    target = st.traj.points[-1]["frame"]
    st.toggle_point(target)
    got = st.set_entry_frame(target)
    assert got != target, "an excluded detection was allowed to supply the measured point"


def test_live_entry_follows_the_slider(sc, doc) -> None:
    st = _review(sc, doc)
    st.entry_live = True
    st.choose(int(st.traj.points[4]["frame"]), "smoke")
    assert st.cutoff() == int(st.traj.points[4]["frame"])
    m = st.measurement()
    assert m["frame"] == int(st.traj.points[4]["frame"]) and m["entry_override"] is True


# ---------------------------------------------------------------------------
# 5. the record: result.json + report.txt
# ---------------------------------------------------------------------------
def test_the_result_keeps_the_entry_frame_and_says_who_chose_it(sc, doc) -> None:
    traj = traj_with([{"frame": 10 + i, "x": 400.0 + i, "y": 300.0 - i} for i in range(8)])
    res = R.build_result(
        clip_id="c", clip_sha="0" * 64, clip_name="c.mp4", video=None, detect=None,
        traj=traj, contact=None, measurement={"total_inches": 120.0, "display": "10' 0\""},
        calib=None, calib_path=None, detect_config={}, artifacts={}, entry_frame=12,
        arc_after_exclusion={"n": 3, "rms_px": 1.5, "frame_range": [10, 12]},
    )
    rev = res["review"]
    assert rev["entry_frame"] == 12 and rev["entry_is_operator_choice"] is True
    text = R.build_report_text(res)
    assert "entry frame 12 chosen by the operator" in text, text
    # and an untouched run still has no review block at all
    plain = R.build_result(
        clip_id="c", clip_sha="0" * 64, clip_name="c.mp4", video=None, detect=None,
        traj=traj, contact=None, measurement={"total_inches": 120.0, "display": "10' 0\""},
        calib=None, calib_path=None, detect_config={}, artifacts={},
    )
    assert "review" not in plain
