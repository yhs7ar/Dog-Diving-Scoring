"""M11 — Phase 3 (transverse scaling) and Phase 4 (planar homography) against ground truth.

The scene gives an exact answer for every quantity here — the homography, the pose, the camera
centre — so the assertions are tight rather than generous, and a route that is subtly wrong fails.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.pool_scene import deck_marks_scene, waterline_marks_scene
from dog_diving.pool import homography as H
from dog_diving.pool import lens as L
from dog_diving.pool import vanishing as V


# ============================================================================
# fixtures
# ============================================================================
def _clean_lens(sc):
    return L.LensModel(K=sc.K, dist=sc.dist, image_size=sc.image_size, source="lines")


def _undistorted_marks(sc, lm=None):
    lm = lm or _clean_lens(sc)
    marks = sc.marks()
    und = L.undistort_points(np.array([[m["x_px"], m["y_px"]] for m in marks]), lm)
    return [{"ft": m["ft"], "x_px": float(u[0]), "y_px": float(u[1])}
            for m, u in zip(marks, und)]


def _edge_families(sc):
    """The two line families the plan's Phase 3 asks for, as clicked segments."""
    def seg(p, q):
        return [*sc.project_clean(*p)[:2], *sc.project_clean(*q)[:2]]
    along = [seg((0.0, 0.0, 0.0), (31.0, 0.0, 0.0)),
             seg((0.0, 16.0, 0.0), (31.0, 16.0, 0.0)),
             seg((0.0, 8.0, 0.0), (31.0, 8.0, 0.0))]
    across = [seg((0.0, 0.0, 0.0), (0.0, 16.0, 0.0)),
              seg((31.0, 0.0, 0.0), (31.0, 16.0, 0.0))]
    return along, across


# ============================================================================
# Phase 3 — vanishing points
# ============================================================================
def test_vanish_recovers_the_exact_intersection_of_a_two_line_family():
    """The estimator check that matters: two lines have an exact intersection, and it must be found.

    The naive implementation — smallest singular vector of the line-coefficient matrix — is a
    *different* estimator, and on a two-line family it returns a point on the bisector.  Measured on
    this scene it gave 174 px of RMS against a truth reproduced here to 1e-9.
    """
    sc = deck_marks_scene()
    along, _ = _edge_families(sc)
    two = [V.line_from_points(s[:2], s[2:]) for s in along[:2]]
    vp = V.vanish(two, image_size=sc.image_size)
    truth = sc.vp_x()
    truth_xy = truth[:2] / truth[2]
    assert vp.xy is not None
    assert np.allclose(vp.xy, truth_xy, atol=1e-6)
    assert vp.rms_px < 1e-9


def test_vanish_reports_a_direction_when_the_family_is_parallel():
    """A single line cannot give a point, and pretending otherwise is a silent error."""
    sc = deck_marks_scene()
    along, _ = _edge_families(sc)
    vp = V.vanish([V.line_from_points(along[0][:2], along[0][2:])])
    assert vp.finite is False
    assert vp.xy is None
    assert vp.usable is False
    assert any("DIRECTION" in n for n in vp.notes)


def test_focal_from_orthogonal_vanishing_points_recovers_the_true_focal():
    sc = deck_marks_scene()
    along, across = _edge_families(sc)
    vpx = V.vanish([V.line_from_points(s[:2], s[2:]) for s in along], image_size=sc.image_size)
    vpy = V.vanish([V.line_from_points(s[:2], s[2:]) for s in across], image_size=sc.image_size)
    f = V.focal_from_orthogonal_vps(vpx.xy, vpy.xy, (959.5, 539.5))
    assert f == pytest.approx(sc.K[0, 0], rel=1e-6)


def test_focal_from_non_orthogonal_vanishing_points_returns_nan_not_a_wrong_number():
    """A negative ``f^2`` turned into a positive ``f`` is a camera that is wrong by 90 degrees."""
    assert np.isnan(V.focal_from_orthogonal_vps((100.0, 100.0), (200.0, 150.0), (0.0, 0.0)))


def test_intrinsics_from_vps_flags_a_pair_it_cannot_use():
    sc = deck_marks_scene()
    lm0 = L.placeholder_intrinsics(sc.image_size)
    bad = V.vanish([V.line_from_points([0.0, 0.0, 0.0][:2], [0.0, 0.0, 0.0][:2])]
                   if False else [np.array([0.0, 1.0, -100.0]), np.array([0.0, 1.0, -400.0])],
                   image_size=sc.image_size)
    lm, rep = V.intrinsics_from_orthogonal_vps(bad, bad, sc.image_size)
    assert rep["ok"] is False
    assert lm.fx == pytest.approx(lm0.fx)          # falls back to the prior, and says so
    assert rep["notes"]


def test_principal_point_from_orthocentre_of_three_orthogonal_vps():
    sc = deck_marks_scene()
    K = sc.K
    vps = [K @ sc.R[:, i] for i in range(3)]
    pp = V.principal_point_from_orthocentre(*vps)
    assert pp is not None
    assert pp[0] == pytest.approx(959.5, abs=1e-6)
    assert pp[1] == pytest.approx(539.5, abs=1e-6)


def test_vanishing_points_by_consensus_finds_both_pool_families_exactly():
    """The correct clustering: consensus over line *pairs*, not over image angles.

    This is the plan's "detect parallel lines of lanes/walls to calculate vanishing points" done
    properly, and it matters because the naive alternative cannot work here — see the next test.
    """
    sc = deck_marks_scene()
    along, across = _edge_families(sc)
    lines = [V.line_from_points(s[:2], s[2:]) for s in along + across]
    vps = V.vanishing_points_from_lines(lines, image_size=sc.image_size)
    assert len(vps) == 2
    found = {vp.n_lines: vp for vp in vps}
    assert set(found) == {2, 3}
    truth_x = (sc.vp_x()[:2] / sc.vp_x()[2])
    truth_y = (sc.vp_y()[:2] / sc.vp_y()[2])
    assert np.allclose(found[3].xy, truth_x, atol=1e-6)
    assert np.allclose(found[2].xy, truth_y, atol=1e-6)
    assert found[3].rms_px < 1e-9


def test_clustering_parallel_world_lines_by_image_angle_is_wrong_here():
    """Why the consensus method exists: this scene's truly-parallel pool lines are 5.7 deg apart.

    The three lines along the pool's length are physically parallel, but their common vanishing point
    sits at (3310, 111) rather than at infinity, so their *image* angles are 167.8, 171.5 and 173.5
    deg.  Any angle tolerance loose enough to group them also swallows the across-pool family.  The
    angle method is therefore documented as an approximation and no longer used.
    """
    sc = deck_marks_scene()
    along, across = _edge_families(sc)
    lines = [V.line_from_points(s[:2], s[2:]) for s in along + across]
    angles = sorted(np.degrees(np.arctan2(l[1], l[0])) % 180.0 for l in lines[:3])
    assert angles[-1] - angles[0] > 5.0, f"the fixture must show the effect ({angles})"
    groups = V.cluster_lines_by_angle(lines, angle_tol_deg=3.0)
    assert max(len(g) for g in groups) < 3, (
        "angle clustering must fail to group all three truly-parallel pool lines")


def test_family_report_orders_by_support_then_residual():
    sc = deck_marks_scene()
    along, across = _edge_families(sc)
    lines = [V.line_from_points(s[:2], s[2:]) for s in along + across]
    rep = V.family_report(lines, image_size=sc.image_size)
    assert len(rep) == 2
    assert rep[0].n_lines >= rep[1].n_lines


# ============================================================================
# Route A — direct correspondences
# ============================================================================
def test_route_a_recovers_the_homography_exactly_from_six_known_points():
    sc = deck_marks_scene()
    world = np.array([[0.0, 0.0], [31.0, 0.0], [0.0, 16.0], [31.0, 16.0], [15.0, 0.0],
                      [16.0, 16.0]])
    image = sc.project_many_clean([(x, y, 0.0) for x, y in world])
    Hh, info = H.homography_from_correspondences(world, image)
    Htrue = H.normalise_homography(sc.K @ np.column_stack([sc.R[:, 0], sc.R[:, 1], sc.t]))
    assert info["residual"]["max_ft"] < 1e-9
    assert np.max(np.abs(Hh - Htrue)) < 1e-9


def test_route_a_refuses_collinear_points_instead_of_returning_a_wrong_homography():
    """The measured failure this guard exists for: six collinear pool marks.

    A DLT fed those points reproduces them perfectly and reconstructs the world 12.7 ft RMS wrong
    (up to 18.8 ft).  A silent 18-foot error is worse than a crash, so this raises.
    """
    sc = deck_marks_scene()
    world = sc.true_world_xy()
    image = sc.project_many_clean([(x, 0.0, 0.0) for x, _ in world])
    with pytest.raises(ValueError, match="collinear"):
        H.homography_from_correspondences(world, image)


def test_route_a_needs_at_least_four_points():
    sc = deck_marks_scene()
    with pytest.raises(ValueError, match="needs >= 4"):
        H.dlt_homography(np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
                         np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]))


def test_route_a_is_exact_from_four_points_and_stays_bounded_with_click_noise():
    """Four points is the minimum, and a realistic 1 px of click noise must stay sub-inch at 30 ft."""
    sc = deck_marks_scene()
    world = np.array([[0.0, 0.0], [31.0, 0.0], [0.0, 16.0], [31.0, 16.0]])
    image = sc.project_many_clean([(x, y, 0.0) for x, y in world])
    Hh, info = H.homography_from_correspondences(world, image)
    assert info["residual"]["max_ft"] < 1e-9

    rng = np.random.default_rng(3)
    errs = []
    for _ in range(25):
        noisy = image + rng.normal(0.0, 1.0, image.shape)
        Hn, _ = H.homography_from_correspondences(world, noisy)
        probe = H.apply_homography_inv(Hn, np.array([[1800.0, 450.0]]))
        truth = H.apply_homography_inv(Hh, np.array([[1800.0, 450.0]]))
        errs.append(float(np.linalg.norm(probe - truth) * 12.0))
    assert max(errs) < 36.0, f"1 px of click noise moved the 30 ft reading by {max(errs):.1f} in"


def test_robust_fit_downweights_a_single_misclicked_point():
    """One badly wrong click, and the robust pass must reject it rather than bend to it.

    Measured on a 12-point grid with one click 72 px away: plain DLT 1.85 in of error at 30 ft,
    robust 0.00 in, with the bad point's weight driven to 0 over 6 reweighting iterations.
    """
    sc = deck_marks_scene()
    gx, gy = np.meshgrid([0.0, 10.0, 20.0, 31.0], [0.0, 8.0, 16.0])
    world = np.column_stack([gx.ravel(), gy.ravel()])
    image = sc.project_many_clean([(x, y, 0.0) for x, y in world])
    bad = image.copy()
    bad[4] += np.array([60.0, -40.0])              # one badly wrong click

    Htrue = H.normalise_homography(sc.K @ np.column_stack([sc.R[:, 0], sc.R[:, 1], sc.t]))
    probe = np.array([[1500.0, 600.0]])
    good = H.apply_homography_inv(Htrue, probe)[0]

    plain, _ = H.homography_from_correspondences(world, bad)
    robust, info = H.homography_from_correspondences(world, bad, robust=True)
    e_plain = abs(float(H.apply_homography_inv(plain, probe)[0][0]) - good[0]) * 12.0
    e_robust = abs(float(H.apply_homography_inv(robust, probe)[0][0]) - good[0]) * 12.0
    assert info["weights"][4] < 0.1, info["weights"]
    assert info["iterations"] > 1
    assert e_robust < e_plain / 2.0, f"robust {e_robust:.3f} in vs plain {e_plain:.3f} in"


# ============================================================================
# Route A' — two orthogonal VPs + two known (possibly collinear) points
# ============================================================================
def test_route_a_prime_recovers_the_pose_from_two_vps_and_the_mark_row():
    """The route that makes the plan's Phase 3 into a Phase 4 homography.

    Two vanishing points hand over two rotation columns, so the mark row alone — a *collinear*
    set — suffices for the rest, which is exactly what a pool's painted ruler provides.
    """
    sc = deck_marks_scene()
    along, across = _edge_families(sc)
    vpx = V.vanish([V.line_from_points(s[:2], s[2:]) for s in along], image_size=sc.image_size)
    vpy = V.vanish([V.line_from_points(s[:2], s[2:]) for s in across], image_size=sc.image_size)
    marks = _undistorted_marks(sc)
    W = np.array([[m["ft"], 0.0] for m in marks])
    I = np.array([[m["x_px"], m["y_px"]] for m in marks])

    R, t, info = H.pose_from_vps_and_points(sc.K, vpx.v, vpy.v, W, I)
    assert info["orthogonality_cos"] < 1e-9
    assert info["rms_px"] < 1e-4
    assert np.max(np.abs(R - sc.R)) < 1e-9
    assert np.max(np.abs(t - sc.t)) < 1e-9


def test_route_a_prime_refuses_two_vps_that_are_not_orthogonal():
    """The guard that stops a wrong line family being read as a right angle."""
    sc = deck_marks_scene()
    along, _ = _edge_families(sc)
    vpx = V.vanish([V.line_from_points(s[:2], s[2:]) for s in along], image_size=sc.image_size)
    marks = _undistorted_marks(sc)
    W = np.array([[m["ft"], 0.0] for m in marks])
    I = np.array([[m["x_px"], m["y_px"]] for m in marks])
    R, t, info = H.pose_from_vps_and_points(sc.K, vpx.v, vpx.v, W, I)   # same family twice
    assert info["orthogonality_cos"] > 0.15
    assert any("not 90" in n or "apart" in n for n in info["notes"])
    assert np.allclose(R, np.eye(3)) and np.allclose(t, 0.0)             # a refusal, not a guess


def test_route_a_prime_needs_two_points():
    sc = deck_marks_scene()
    with pytest.raises(ValueError, match=">= 2"):
        H.pose_from_vps_and_points(sc.K, sc.vp_x(), sc.vp_y(), np.array([[0.0, 0.0]]),
                                   np.array([[100.0, 100.0]]))


# ============================================================================
# Route B — the plan's Phase 3/4 as written
# ============================================================================
def test_route_b_recovers_the_homography_from_a_row_a_vp_an_edge_and_a_width_prior():
    """The current corpus's route: marks + transverse VP + far edge + the 14-20 ft width prior."""
    sc = deck_marks_scene()
    _, across = _edge_families(sc)
    vpy = V.vanish([V.line_from_points(s[:2], s[2:]) for s in across], image_size=sc.image_size)
    marks = _undistorted_marks(sc)
    Hh, info = H.homography_from_row_and_edge(
        marks, vp_transverse=vpy.v, edge_image_line=sc.edge_line(sc.width_ft),
        width_ft=sc.width_ft, marks_on_edge="near")
    Htrue = H.normalise_homography(sc.K @ np.column_stack([sc.R[:, 0], sc.R[:, 1], sc.t]))
    assert info["route"] == "B_row_vp_edge_width"
    assert info["residual"]["rms_ft"] < 1e-6
    assert np.max(np.abs(Hh - Htrue)) < 1e-5


def test_route_b_refuses_a_width_prior_without_a_transverse_extent():
    """A length cannot fix a transverse scale with nothing to measure it against, and must say so."""
    sc = deck_marks_scene()
    _, across = _edge_families(sc)
    vpy = V.vanish([V.line_from_points(s[:2], s[2:]) for s in across], image_size=sc.image_size)
    marks = _undistorted_marks(sc)
    with pytest.raises(ValueError, match="width prior alone"):
        H.homography_from_row_and_edge(marks, vp_transverse=vpy.v, edge_image_line=None,
                                      width_ft=16.0)


def test_route_b_needs_three_marks():
    sc = deck_marks_scene()
    _, across = _edge_families(sc)
    vpy = V.vanish([V.line_from_points(s[:2], s[2:]) for s in across], image_size=sc.image_size)
    marks = _undistorted_marks(sc)[:2]
    with pytest.raises(ValueError, match=">= 3 marks"):
        H.homography_from_row_and_edge(marks, vp_transverse=vpy.v,
                                      edge_image_line=sc.edge_line(16.0), width_ft=16.0)


def test_width_prior_sensitivity_is_measured_not_asserted():
    """How many inches ride on the plan's "14 to 20 ft" allowance.

    Measured on this scene at the operator's 20 ft default: **4.8 in of distance per foot of pool
    width**, i.e. 28.8 in across the plan's full 6 ft prior band — against a ±2 in accuracy
    criterion.  The number scales as 1/width (it was 6.0 in/ft when the default was 16 ft), which is
    why it is measured from the scene rather than written down as a constant.
    """
    sc = deck_marks_scene()
    _, across = _edge_families(sc)
    vpy = V.vanish([V.line_from_points(s[:2], s[2:]) for s in across], image_size=sc.image_size)
    marks = _undistorted_marks(sc)
    s = H.width_sensitivity(marks, vp_transverse=vpy.v, edge_image_line=sc.edge_line(sc.width_ft),
                            width_ft=sc.width_ft, probe_world_xy=[(31.0, 8.0)])
    assert s["in_per_ft_of_width"] == pytest.approx(4.8, rel=0.1)
    assert s["in_across_prior_band"] == pytest.approx(28.8, rel=0.1)
    assert any("prior band" in n for n in s["notes"])


# ============================================================================
# The pose — what makes Phase 5 reachable at all
# ============================================================================
def test_pose_decomposition_recovers_the_rotation_translation_and_camera_centre():
    sc = deck_marks_scene()
    world = np.array([[0.0, 0.0], [31.0, 0.0], [0.0, 16.0], [31.0, 16.0]])
    image = sc.project_many_clean([(x, y, 0.0) for x, y in world])
    Hh, _ = H.homography_from_correspondences(world, image)
    pose = H.decompose_plane_homography(Hh, sc.K)
    assert pose.info["orthonormality"] < 1e-9
    assert np.max(np.abs(pose.R - sc.R)) < 1e-9
    assert np.max(np.abs(pose.t - sc.t)) < 1e-9
    assert np.allclose(pose.camera_centre, sc.eye, atol=1e-9)


def test_plane_homography_at_zero_height_is_the_marked_plane():
    sc = deck_marks_scene()
    pose = _pose(sc)
    H0 = pose.plane_homography(0.0)
    assert np.max(np.abs(H0 - pose.H)) < 1e-9


def test_plane_homography_moves_the_plane_by_exactly_the_requested_height():
    """Phase 5 in one assertion: the water plane differs from the marked one only in its 3rd column.

    Compared as *proportions*: :func:`normalise_homography` divides by ``H[2, 2]``, and ``H[2, 2]``
    is itself a function of the plane height, so two shared columns come out scaled by that common
    factor.  Multiplying by the *normalised* ratio cannot undo it (that ratio is 1 by construction);
    the right test is that the two columns are parallel vectors, i.e. their cross product vanishes.
    """
    sc = deck_marks_scene()
    pose = _pose(sc)
    H0, Hw = pose.plane_homography(0.0), pose.plane_homography(sc.water_z_ft)

    def parallel(u, v, tol=1e-9):
        return float(np.linalg.norm(np.cross(u, v))) < tol * float(
            np.linalg.norm(u) * np.linalg.norm(v))

    assert parallel(H0[:, 0], Hw[:, 0])
    assert parallel(H0[:, 1], Hw[:, 1])
    assert not parallel(H0[:, 2], Hw[:, 2])


def test_plane_homography_reproduces_the_scenes_own_water_plane_projection():
    """The third column change must be the *right* one: it has to project the water surface correctly."""
    sc = deck_marks_scene()
    pose = _pose(sc)
    Hw = pose.plane_homography(sc.water_z_ft)
    world = np.array([[0.5, 1.0], [12.0, 6.0], [30.5, 15.5], [22.0, 0.25]])
    pred = H.apply_homography(Hw, world)
    truth = sc.project_many_clean([(x, y, sc.water_z_ft) for x, y in world])
    assert np.max(np.abs(pred - truth)) < 1e-8


def test_resolve_up_recovers_the_true_up_axis_sign_from_a_rim_pair():
    """The rim pair decides which way +Z points — a coin flip the decomposition cannot settle alone.

    Measured: the naive test (comparing reconstructed heights) returned a rim at 19.4 ft on a scene
    where the rim is at 0.33 ft.  The vanishing-point test below needs no world coordinates at all.
    """
    sc = deck_marks_scene()
    lm = _clean_lens(sc)
    pose = _pose(sc)
    top_raw, con_raw = sc.rim_pair(X=4.0)
    top = L.undistort_points(top_raw, lm)
    con = L.undistort_points(con_raw, lm)
    sign, detail = pose.resolve_up(con, top)
    assert sign == 1.0
    assert detail["vp_mode"] == "finite"
    assert detail["projection"] > 0.0

    # The answer is a property of the *geometry*, not of the pose's current convention: `resolve_up`
    # is asking "which sign makes world +Z point up", so it returns the same answer whether the pose
    # arrived with z_sign = +1 or -1.  That is what makes it usable as the fix for a decomposition
    # whose sign is unknown.
    flipped = H.PlanePose(K=pose.K, R=pose.R, t=pose.t, H=pose.H, z_sign=-1.0)
    assert flipped.resolve_up(con, top)[0] == 1.0
    assert detail["step_px"] > 0.0


def test_resolve_up_says_so_when_it_cannot_decide():
    """Two clicks with no displacement *along* the vertical's image direction carry no height signal."""
    sc = deck_marks_scene()
    pose = _pose(sc)
    v = pose.vertical_vp(z_sign=1.0)
    vp_img = np.asarray(v[:2], dtype=float) / float(v[2])
    p0 = np.array([900.0, 500.0])
    motion = (vp_img - p0) * (1.0 if float(v[2]) >= 0.0 else -1.0)   # the +Z motion direction
    perp = np.array([-motion[1], motion[0]], dtype=float)
    perp /= np.linalg.norm(perp)
    sign, detail = pose.resolve_up(p0, p0 + perp * 40.0)
    assert sign == 1.0
    assert "undetermined" in detail["rule"]

    # ... and along the motion direction it decides, with the opposite answer for reversed clicks.
    up = pose.resolve_up(p0, p0 + motion / np.linalg.norm(motion) * 40.0)[0]
    down = pose.resolve_up(p0, p0 - motion / np.linalg.norm(motion) * 40.0)[0]
    assert {up, down} == {1.0, -1.0}


def _pose(sc):
    world = np.array([[0.0, 0.0], [31.0, 0.0], [0.0, 16.0], [31.0, 16.0]])
    image = sc.project_many_clean([(x, y, 0.0) for x, y in world])
    Hh, _ = H.homography_from_correspondences(world, image)
    return H.decompose_plane_homography(Hh, sc.K)


def test_waterline_marks_scene_also_builds_a_valid_pose():
    """The other venue arrangement (marks at the waterline) must work, not just the deck one."""
    sc = waterline_marks_scene()
    pose = _pose(sc)
    assert pose.info["orthonormality"] < 1e-9
    assert np.allclose(pose.camera_centre, sc.eye, atol=1e-9)


def test_homography_condition_reports_anisotropy_and_fold_back():
    sc = deck_marks_scene()
    pose = _pose(sc)
    cond = H._homography_condition(pose.H)
    assert cond["px_per_ft_along"] > 0
    assert cond["px_per_ft_across"] > 0
    assert np.isfinite(cond["cond"])


def test_plane_from_three_points():
    n, d = H.plane_from_three_points((0.0, 0.0, 2.0), (1.0, 0.0, 2.0), (0.0, 1.0, 2.0))
    assert np.allclose(np.abs(n), np.array([0.0, 0.0, 1.0]), atol=1e-9)
    assert d == pytest.approx(2.0)
    with pytest.raises(ValueError, match="collinear"):
        H.plane_from_three_points((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (2.0, 2.0, 2.0))
