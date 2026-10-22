"""M22 — the plane/grid/trajectory tools added on 2026-10-21.

Four operator orders, four small pieces of behaviour, each tested where it can be tested without a
display:

1. **Flip X** — reverse the plane's orientation at any time.  Nothing in P1-P4 can catch a mirrored
   plane (the foreshortening is symmetric), so the test has to be geometric: X must actually run the
   other way afterwards.
2. **Thin trajectory lines** — the overlay's stroke weights follow the plane grid, so a 960-px frame is
   not covered by the annotation.
3. **Faded grid above the landing point** — Review's lattice is blended above the landing row and
   untouched below it.
4. **Click a detection out of the arc** — ``ReviewState`` keeps the excluded set and the arc is refit
   from what remains.

The Plan tab is Tk-only and is covered by the pane tests that build the real ``App``; its geometry is
one equal-scale linear map (feet -> canvas), which is checked here as arithmetic.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dog_diving import trajectory as T                              # noqa: E402
from dog_diving.pool import plane as P                              # noqa: E402

H, W_PX = 1080, 1920


def _session() -> P.PlaneSession:
    s = P.PlaneSession(frame_shape=(H, W_PX), length_ft=25.0, width_ft=16.0)
    s.corners = {"00": (1200.0, 300.0), "10": (300.0, 330.0),
                 "11": (300.0, 520.0), "01": (1200.0, 700.0)}
    return s


# ---------------------------------------------------------------------------
# 1. flip X — the NUMBER scale changes, the quad does not
# ---------------------------------------------------------------------------
def test_flip_x_changes_the_numbering_and_no_pixel():
    s = _session()
    before = {t: s.corners[t] for t in P.CORNER_TAGS}
    d_before = np.sign(s.pixel_of(20.0, 8.0)[0] - s.pixel_of(0.0, 8.0)[0])
    assert s.flip_x() is True
    # the image direction of increasing X is now the other way: that is the whole point
    d_after = np.sign(s.pixel_of(20.0, 8.0)[0] - s.pixel_of(0.0, 8.0)[0])
    assert d_after == -d_before
    # not one pixel moved: the same four points, and X=0 is the pair that used to be X=L
    assert sorted(s.corners.values()) == sorted(before.values())
    assert s.corners["00"] == before["10"] and s.corners["01"] == before["11"]
    assert s.corners["10"] == before["00"] and s.corners["11"] == before["01"]


def test_flip_x_turns_a_reversed_plane_into_a_valid_one():
    """The flip exists to fix X=0 sitting on the far end; P4 is what says which end that is."""
    correct = _session()
    assert P.validate_plane({"homography": np.asarray(correct.homography()).tolist()},
                            correct)["P4"]["pass"]
    # the same quad numbered the wrong way round: X=0 on the small end
    rev = P.PlaneSession(frame_shape=(H, W_PX), length_ft=25.0, width_ft=16.0)
    rev.corners = {"00": correct.corners["10"], "10": correct.corners["00"],
                   "11": correct.corners["01"], "01": correct.corners["11"]}
    gates_before = P.validate_plane({"homography": np.asarray(rev.homography()).tolist()}, rev)
    assert not gates_before["P4"]["pass"], gates_before["P4"]["detail"]

    rev.flip_x()
    gates_after = P.validate_plane({"homography": np.asarray(rev.homography()).tolist()}, rev)
    assert all(v["pass"] for v in gates_after.values()), {
        k: v["detail"] for k, v in gates_after.items()}


def test_flip_x_is_an_involution():
    s = _session()
    before = {t: s.corners[t] for t in P.CORNER_TAGS}
    s.flip_x()
    s.flip_x()
    assert s.corners == before


def test_flip_x_marks_an_automatic_placement_as_edited():
    s = _session()
    s.seed = {"placement": "auto"}
    s.flip_x()
    assert s.seed["placement"] == "auto+edited"


def test_flip_x_refuses_an_empty_plane():
    s = P.PlaneSession(frame_shape=(H, W_PX))
    assert s.flip_x() is False


# ---------------------------------------------------------------------------
# 2/3. the grid: unchanged without a fade, blended only above the row
# ---------------------------------------------------------------------------
def _grid_energy(img: np.ndarray, y0: int, y1: int) -> float:
    """How much ink the overlay put down in rows [y0, y1)."""
    return float(np.abs(img[y0:y1].astype(np.int32)).sum())


def test_grid_draws_identically_when_no_fade_is_asked_for():
    s = _session()
    frame = np.zeros((H, W_PX, 3), np.uint8)
    plain = P.draw_grid_on(frame, s.homography(), length_ft=25.0, width_ft=16.0)
    same = P.draw_grid_on(frame, s.homography(), length_ft=25.0, width_ft=16.0, fade_above_y=None)
    assert np.array_equal(plain, same)
    assert plain.any(), "the grid drew nothing"


def test_grid_fades_only_above_the_landing_row():
    s = _session()
    frame = np.zeros((H, W_PX, 3), np.uint8)
    y_split = 420
    plain = P.draw_grid_on(frame, s.homography(), length_ft=25.0, width_ft=16.0)
    faded = P.draw_grid_on(frame, s.homography(), length_ft=25.0, width_ft=16.0,
                           fade_above_y=float(y_split), fade_alpha=0.25)
    above_plain = _grid_energy(plain, 0, y_split)
    above_faded = _grid_energy(faded, 0, y_split)
    assert above_plain > 0
    assert above_faded < 0.75 * above_plain          # the lattice above is visibly knocked back
    # Below the row the lattice keeps full strength.  cv2's anti-aliased rasterisation is
    # endpoint-dependent, so a line that crosses the split differs from the one-piece line by a DN or
    # two anywhere along its length — compare energy, and cap the per-pixel difference, instead of
    # demanding bit equality.
    y_clear = y_split + 6
    below_plain = _grid_energy(plain, y_clear, plain.shape[0])
    below_faded = _grid_energy(faded, y_clear, faded.shape[0])
    assert below_plain > 0
    assert 0.95 < below_faded / below_plain < 1.05
    assert int(np.abs(plain[y_clear:].astype(np.int32) - faded[y_clear:].astype(np.int32)).max()) < 40


def test_grid_fade_leaves_the_photo_visible_behind_it():
    """A faded line must still be a line — half-blended, not erased."""
    s = _session()
    frame = np.full((H, W_PX, 3), 20, np.uint8)
    out = P.draw_grid_on(frame, s.homography(), length_ft=25.0, width_ft=16.0,
                         fade_above_y=420.0, fade_alpha=0.5)
    above = out[:420]
    assert above.max() > 20, "the faded lattice vanished entirely"


# ---------------------------------------------------------------------------
# 4. the arc: refit and the Review toggle
# ---------------------------------------------------------------------------
def _track(n: int = 12, seed: int = 3) -> list[dict]:
    """A parabolic track in image pixels with one planted outlier."""
    rng = np.random.default_rng(seed)
    pts = []
    for i in range(n):
        f = 10 + i * 3
        x = 300.0 + 9.0 * i + float(rng.normal(0, 0.6))
        y = 500.0 - 26.0 * i + 1.9 * i * i + float(rng.normal(0, 0.6))
        pts.append({"frame": f, "x": x, "y": y})
    pts[5]["y"] += 90.0                     # the frame the operator will click out
    return pts


def test_refit_fits_the_clean_points_and_ignores_the_outlier():
    pts = _track()
    clean = [p for i, p in enumerate(pts) if i != 5]
    fit = T.refit(clean)
    assert fit["coeffs"] is not None
    assert fit["n"] == len(clean)
    assert fit["frame_range"] == [clean[0]["frame"], clean[-1]["frame"]]
    assert fit["rms_px"] < 2.0
    assert len(fit["inliers"]) == len(clean)


def test_refit_refuses_a_quadratic_it_cannot_determine():
    fit = T.refit(_track()[:2])
    assert fit["coeffs"] is None and fit["inliers"] == []


def test_review_toggle_excludes_and_restores_a_frame():
    app_gui = pytest.importorskip("app_gui")

    class Traj:
        points = _track()
        coeffs = {"t_ref": 26.0, "x": [0, 0, 400], "y": [0, -20, 500]}
        inliers = points[:4]
        airborne_frame_range = (10, 43)
        arc_rms_px = 1.11

    st = app_gui.ReviewState(clip=Path("x.mp4"), traj=Traj(), contact=None, calib=None,
                             image_size=(960, 600))
    base = st.arc()
    assert base["coeffs"] is Traj.coeffs              # untouched while nothing is excluded
    assert base["n"] == len(Traj.points)

    assert st.toggle_point(Traj.points[5]["frame"]) is True
    assert Traj.points[5]["frame"] in st.excluded
    after = st.arc()
    assert after["coeffs"] is not Traj.coeffs         # refit, not passed through
    assert after["n"] == len(Traj.points) - 1
    assert after["rms_px"] < T.refit(Traj.points)["rms_px"] + 1e-9

    assert st.toggle_point(Traj.points[5]["frame"]) is False
    assert st.arc()["n"] == len(Traj.points)


def test_nearest_point_picks_the_closest_detection():
    app_gui = pytest.importorskip("app_gui")

    class Traj:
        points = _track()

    st = app_gui.ReviewState(clip=Path("x.mp4"), traj=Traj(), contact=None, calib=None,
                             image_size=(960, 600))
    target = Traj.points[3]
    d, p = st.nearest_point(target["x"] + 2.0, target["y"] - 2.0)
    assert p is target and d < 5.0


# ---------------------------------------------------------------------------
# 2. the thin trajectory overlay
# ---------------------------------------------------------------------------
def test_overlay_is_thin_by_default_and_marks_excluded_points():
    import cv2

    from dog_diving.trajectory import Trajectory

    pts = _track(8)
    traj = Trajectory(clip_id="t", n_points=len(pts), points=pts, runs=[], flight_run=0,
                      airborne_frame_range=(pts[0]["frame"], pts[-1]["frame"]),
                      n_before_leave=0, n_before_entry=0, coeffs=None, inliers=pts[:3],
                      n_outliers=0, arc_rms_px=None, arc_rms_all_px=None,
                      max_inlier_res_px=None, residuals=None, flight_points=pts,
                      last_arc_frame=None, last_contact_candidate=pts[-1], cfg={})
    frame = np.zeros((600, 960, 3), np.uint8)
    thin = T.draw_overlay(frame, traj, show_conf=False)
    thick = T.draw_overlay(frame, traj, thickness=4, show_conf=False)

    def yellow(img):
        return int(((img[:, :, 0] < 60) & (img[:, :, 1] > 200) & (img[:, :, 2] > 200)).sum())

    assert yellow(thin) > 0
    assert yellow(thin) < yellow(thick)              # the order: no more 4 px strokes

    marked = T.draw_overlay(frame, traj, show_conf=False, excluded={pts[2]["frame"]})
    x, y = int(round(pts[2]["x"])), int(round(pts[2]["y"]))
    patch = marked[max(0, y - 4):y + 5, max(0, x - 4):x + 5]
    assert ((patch[:, :, 0] > 120) & (patch[:, :, 1] > 120) & (patch[:, :, 2] > 120)).any(), \
        "the excluded point was not marked"
