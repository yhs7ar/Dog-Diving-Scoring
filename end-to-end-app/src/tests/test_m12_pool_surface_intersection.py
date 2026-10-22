"""M12 — Phase 5 (water-plane extrapolation) and Phase 6 (tail-set intersection), on ground truth.

Phase 5's headline measurement here is the rim/coping pair, and the test file records *why* the
obvious formulation of it fails: two rays from one camera centre share no vertical, so "solve for
(X, Y, z_top, z_contact) by reprojection" is scale-degenerate and runs to infinity.  The wall-plane
formulation returns the water level to 1e-9 ft; the degenerate one returned −4.561 ft on water that
is truly at −1.625 ft, with a *small* residual of 0.269 px.  Both are asserted below so a regression
to the degenerate version cannot pass quietly.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.pool_scene import deck_marks_scene, waterline_marks_scene
from dog_diving.pool import geometry as G
from dog_diving.pool import homography as H
from dog_diving.pool import intersection as X
from dog_diving.pool import lens as L
from dog_diving.pool import surface as S
from dog_diving.pool import vanishing as V


# ============================================================================
# fixtures
# ============================================================================
def _pose(sc):
    world = np.array([[0.0, 0.0], [31.0, 0.0], [0.0, 16.0], [31.0, 16.0]])
    image = sc.project_many_clean([(x, y, 0.0) for x, y in world])
    Hh, _ = H.homography_from_correspondences(world, image)
    pose = H.decompose_plane_homography(Hh, sc.K)
    lm = L.LensModel(K=sc.K, dist=sc.dist, image_size=sc.image_size, source="lines")
    top, con = sc.rim_pair(X=4.0)
    sign, _ = pose.resolve_up(L.undistort_points(con, lm), L.undistort_points(top, lm))
    return H.PlanePose(K=pose.K, R=pose.R, t=pose.t, H=pose.H, z_sign=sign, info=pose.info)


def _clicks(sc, lm, x=4.0):
    top, con = sc.rim_pair(X=x)
    return L.undistort_points(top, lm), L.undistort_points(con, lm)


WALL_NEAR = {"axis": "y", "value": 0.0}


# ============================================================================
# Ray / plane primitives
# ============================================================================
def test_ray_from_pixel_passes_through_the_camera_centre():
    sc = deck_marks_scene()
    pose = _pose(sc)
    ray = G.ray_from_pixel(pose, (900.0, 500.0))
    assert np.allclose(ray.origin, pose.camera_centre)
    assert np.isclose(np.linalg.norm(ray.direction), 1.0)


def test_world_from_pixel_at_zero_height_lands_on_the_marked_plane():
    sc = deck_marks_scene()
    pose = _pose(sc)
    world = np.array([[24.0, 6.0]])
    px = H.apply_homography(pose.H, world)[0]
    back = G.world_from_pixel(pose, px, 0.0)
    assert back["status"] == "ok"
    assert np.allclose(back["point"][:2], world[0], atol=1e-6)
    assert abs(float(back["point"][2])) < 1e-9


def test_intersect_ray_plane_reports_parallel_and_behind_rather_than_none():
    """Three outcomes, all of which a caller has to act on differently — so none is ``None``."""
    origin = np.array([0.0, 0.0, 5.0])
    parallel = G.Ray(origin=origin, direction=np.array([1.0, 0.0, 0.0]), pixel=(0.0, 0.0))
    hit = G.intersect_ray_plane(parallel, (0.0, 0.0, 1.0), 0.0)
    assert hit["status"] == "parallel" and hit["point"] is None

    away = G.Ray(origin=origin, direction=np.array([0.0, 0.0, 1.0]), pixel=(0.0, 0.0))
    hit = G.intersect_ray_plane(away, (0.0, 0.0, 1.0), 0.0)
    assert hit["status"] == "behind" and "behind the camera" in hit["detail"]

    down = G.Ray(origin=origin, direction=np.array([0.0, 0.0, -1.0]), pixel=(0.0, 0.0))
    hit = G.intersect_ray_plane(down, (0.0, 0.0, 1.0), -2.0)
    assert hit["status"] == "ok"
    assert hit["point"][2] == pytest.approx(-2.0)


def test_intersect_ray_vertical_plane_gives_the_height_on_a_known_wall():
    sc = deck_marks_scene()
    pose = _pose(sc)
    lm = L.LensModel(K=sc.K, dist=sc.dist, image_size=sc.image_size, source="lines")
    px = L.undistort_points(sc.project(9.0, 0.0, sc.water_z_ft), lm)
    hit = G.height_on_wall(pose, px, "y", 0.0)
    assert hit["status"] == "ok"
    assert hit["z_ft"] == pytest.approx(sc.water_z_ft, abs=1e-9)
    assert np.asarray(hit["point"])[0] == pytest.approx(9.0, abs=1e-6)

    with pytest.raises(ValueError, match="axis must be"):
        G.height_on_wall(pose, px, "z", 0.0)


# ============================================================================
# Phase 5 — the water plane
# ============================================================================
def test_water_level_from_a_wall_contact_is_exact():
    sc = deck_marks_scene()
    pose, lm = _pose(sc), L.LensModel(K=sc.K, dist=sc.dist, image_size=sc.image_size,
                                      source="lines")
    _, con = _clicks(sc, lm, x=4.0)
    wp = S.water_plane_from_wall_contact(pose, con, WALL_NEAR)
    assert wp.method == "wall_contact"
    assert wp.z_ft == pytest.approx(sc.water_z_ft, abs=1e-9)
    assert wp.below_marked_plane_ft == pytest.approx(sc.freeboard_in / 12.0, abs=1e-9)
    assert wp.residual_px == 0.0
    assert wp.sigma_ft is not None and wp.sigma_ft * 12.0 < 0.5


def test_rim_pair_water_level_is_exact_and_self_checks_the_vertical():
    """The plan's Phase 5 anchor, done on a known wall: the contact's own height is the water level."""
    sc = deck_marks_scene()
    pose, lm = _pose(sc), L.LensModel(K=sc.K, dist=sc.dist, image_size=sc.image_size,
                                      source="lines")
    top, con = _clicks(sc, lm, x=4.0)
    wp = S.water_plane_from_rim_pair(pose, top, con, WALL_NEAR)
    assert wp.z_ft == pytest.approx(sc.water_z_ft, abs=1e-9)
    assert wp.rim_top_z_ft == pytest.approx(sc.rim_z_ft, abs=1e-9)
    assert wp.residual_ft is not None and wp.residual_ft < 1e-6
    assert any("VERTICAL REFERENCE" in n for n in wp.notes)


def test_rim_pair_needs_a_wall_and_says_so():
    """Without a wall the pair does not constrain a height at all — the module's central finding."""
    sc = deck_marks_scene()
    pose, lm = _pose(sc), L.LensModel(K=sc.K, dist=sc.dist, image_size=sc.image_size,
                                      source="lines")
    top, con = _clicks(sc, lm)
    with pytest.raises(ValueError, match="wall"):
        S.water_plane_from_rim_pair(pose, top, con, None)   # type: ignore[arg-type]


def test_a_wall_shape_that_is_not_a_wall_is_reported_not_trusted():
    """A line whose world Y varies along its length was not a single wall edge."""
    sc = deck_marks_scene()
    pose = _pose(sc)
    # A line across the pool (X varies, Y constant at 8 ft) fed in as if it were the near edge.
    across = [*sc.project_clean(0.0, 8.0, 0.0)[:2], *sc.project_clean(31.0, 8.0, 0.0)[:2]]
    wall = S.wall_from_edge_line(pose, across)
    assert wall["value"] == pytest.approx(8.0, abs=1e-6)
    assert wall["spread_ft"] < 1e-6

    # A genuinely non-straight-world line (a diagonal) must show its spread instead.
    diagonal = [*sc.project_clean(0.0, 0.0, 0.0)[:2], *sc.project_clean(31.0, 16.0, 0.0)[:2]]
    wall2 = S.wall_from_edge_line(pose, diagonal)
    assert wall2["spread_ft"] > 1.0
    assert wall2["spread_in"] > 12.0


def test_known_freeboard_is_labelled_as_a_specification():
    wp = S.water_plane_from_known_freeboard(19.5)
    assert wp.method == "known_freeboard"
    assert wp.z_ft == pytest.approx(-19.5 / 12.0)
    assert wp.sigma_ft is None
    assert any("specification" in n for n in wp.notes)


def test_click_sensitivity_is_a_few_inches_not_thousands():
    """The number that exposed the degenerate formulation: it reported 2405 in per pixel.

    Measured here: ~0.25 in of water level per pixel of click error on the contact, with the pair
    adding a comparable amount through the vertical consistency check.
    """
    sc = deck_marks_scene()
    pose, lm = _pose(sc), L.LensModel(K=sc.K, dist=sc.dist, image_size=sc.image_size,
                                      source="lines")
    top, con = _clicks(sc, lm)
    s = S.sensitivity_from_clicks(pose, top, con, WALL_NEAR, sigma_px=1.0)
    assert s["ok"]
    assert 0.0 < s["spread_in"] < 5.0, s
    assert any("click error" in n for n in s["notes"])


def test_the_degenerate_rim_formulation_would_not_have_worked():
    """Record of the failure, so nobody re-derives it: ray-pair reprojection runs to infinity.

    The residual of a point *on a ray* against its own pixel is zero at any distance, so a
    least-squares fit over (X, Y, z_top, z_contact) has no lower bound and no scale.  The direct
    demonstration: pushing the two solved points far away lowers the residual without moving the
    pixels.
    """
    sc = deck_marks_scene()
    pose, lm = _pose(sc), L.LensModel(K=sc.K, dist=sc.dist, image_size=sc.image_size,
                                      source="lines")
    top, con = _clicks(sc, lm)
    ray_t = G.ray_from_pixel(pose, top)
    for s in (10.0, 100.0, 1000.0):
        p = ray_t.at(s)
        assert G.reprojection_error_px(pose, p, top) < 1e-6, (
            "any point on the ray reprojects exactly onto the pixel — which is why the "
            "reprojection objective is unbounded below and cannot determine a height")


# ============================================================================
# Phase 6 — the intersection
# ============================================================================
def test_entry_coordinate_is_exact_at_five_points_across_the_pool():
    sc = deck_marks_scene()
    pose, lm = _pose(sc), L.LensModel(K=sc.K, dist=sc.dist, image_size=sc.image_size,
                                      source="lines")
    top, con = _clicks(sc, lm)
    wp = S.water_plane_from_rim_pair(pose, top, con, WALL_NEAR)
    for xt, yt in [(31.0, 8.0), (22.5, 3.0), (5.0, 12.0), (30.0, 0.5), (1.0, 0.2)]:
        px = L.undistort_points(sc.entry_pixel(xt, yt), lm)
        e = X.entry_coordinate(pose, wp, px)
        assert e["X_ft"] == pytest.approx(xt, abs=1e-8)
        assert e["Y_ft"] == pytest.approx(yt, abs=1e-8)
        assert e["distance_in"] == pytest.approx(xt * 12.0, abs=1e-7)
        assert e["homography_agreement_ft"] < 1e-9
        assert e["ray_depth_ft"] > 0.0
        assert "negative_distance" not in [k for k, v in e["flags"].items() if v]


def test_the_two_routes_agree_to_machine_precision():
    """The project's verification protocol, as one call: homography vs ray/plane."""
    sc = deck_marks_scene()
    pose, lm = _pose(sc), L.LensModel(K=sc.K, dist=sc.dist, image_size=sc.image_size,
                                      source="lines")
    top, con = _clicks(sc, lm)
    wp = S.water_plane_from_rim_pair(pose, top, con, WALL_NEAR)
    px = L.undistort_points(sc.project_many(
        [(x, y, sc.water_z_ft) for x, y in [(0.0, 0.0), (1.0, 1.0), (15.0, 8.0), (31.0, 16.0),
                                            (22.0, 4.0), (5.0, 12.0)]]), lm)
    ra = X.route_agreement_ft(pose, wp, px)
    assert ra["ok"] and ra["n"] == 6
    assert ra["max_ft"] < 1e-9


def test_entry_coordinate_flags_an_impossible_entry_rather_than_hiding_it():
    sc = deck_marks_scene()
    pose, lm = _pose(sc), L.LensModel(K=sc.K, dist=sc.dist, image_size=sc.image_size,
                                      source="lines")
    top, con = _clicks(sc, lm)
    wp = S.water_plane_from_rim_pair(pose, top, con, WALL_NEAR)
    extent = {"x_min": 0.0, "x_max": 31.0, "y_min": 0.0, "y_max": 16.0}

    inside = X.entry_coordinate(pose, wp, L.undistort_points(sc.entry_pixel(20.0, 8.0), lm),
                                extent=extent)
    assert not inside["flags"]["outside_extent"]

    # A point behind the dock edge is a negative distance, and that must be visible, not clamped.
    behind_px = L.undistort_points(sc.project(-3.0, 8.0, sc.water_z_ft), lm)
    e = X.entry_coordinate(pose, wp, behind_px, extent=extent)
    assert e["X_ft"] == pytest.approx(-3.0, abs=1e-6)
    assert e["flags"]["negative_distance"]
    assert e["flags"]["outside_extent"]
    assert any("dock side" in n.lower() for n in e["notes"])


def test_entry_coordinate_without_a_pixel_is_an_honest_empty():
    sc = deck_marks_scene()
    pose = _pose(sc)
    wp = S.water_plane_from_known_freeboard(19.5)
    e = X.entry_coordinate(pose, wp, None)          # type: ignore[arg-type]
    assert e["X_ft"] is None and e["flags"]["no_point"]


# ============================================================================
# Parallax — the plan's literal step is only valid at zero height
# ============================================================================
def test_the_literal_step_is_exact_when_the_tailset_is_at_the_water():
    """``height_ft = 0`` is the plan's Phase 6 statement, and it must be the correct limit."""
    sc = deck_marks_scene()
    pose, lm = _pose(sc), L.LensModel(K=sc.K, dist=sc.dist, image_size=sc.image_size,
                                     source="lines")
    top, con = _clicks(sc, lm)
    wp = S.water_plane_from_rim_pair(pose, top, con, WALL_NEAR)
    px = L.undistort_points(sc.entry_pixel(28.0, 9.0), lm)
    corr = X.entry_with_height(pose, wp, px, height_ft=0.0)
    lit = X.entry_coordinate(pose, wp, px)
    assert corr["X_ft"] == pytest.approx(lit["X_ft"], abs=1e-9)
    assert corr["shift_in"] == pytest.approx(0.0, abs=1e-9)


def test_height_correction_recovers_the_true_entry_from_an_airborne_frame():
    """Given the true height, the plumb-corrected entry is the true (X, Y) — exactly."""
    sc = deck_marks_scene()
    pose, lm = _pose(sc), L.LensModel(K=sc.K, dist=sc.dist, image_size=sc.image_size,
                                      source="lines")
    top, con = _clicks(sc, lm)
    wp = S.water_plane_from_rim_pair(pose, top, con, WALL_NEAR)
    for h in (0.5, 2.0, 4.0):
        px = L.undistort_points(sc.entry_pixel(31.0, 8.0, Z=sc.water_z_ft + h), lm)
        corr = X.entry_with_height(pose, wp, px, height_ft=h)
        assert corr["X_ft"] == pytest.approx(31.0, abs=1e-7), f"h={h}"
        assert corr["Y_ft"] == pytest.approx(8.0, abs=1e-7), f"h={h}"


def test_parallax_bias_is_signed_long_for_a_camera_behind_the_dog():
    """The sign is a property of the camera's side, and is reported rather than assumed.

    Measured on this scene (camera 17 ft behind and to the side of the entry, 5.6 ft up): a tail-set
    0.5 ft above the water reconstructs 33.8 in LONG, and the bias grows to ~555 in at 4 ft.  The
    shallow view of the water is why: under 10 deg of elevation, a foot of height is worth ~7 ft of
    ground.
    """
    sc = deck_marks_scene()
    pose, lm = _pose(sc), L.LensModel(K=sc.K, dist=sc.dist, image_size=sc.image_size,
                                      source="lines")
    top, con = _clicks(sc, lm)
    wp = S.water_plane_from_rim_pair(pose, top, con, WALL_NEAR)
    px = L.undistort_points(sc.entry_pixel(31.0, 8.0, Z=sc.water_z_ft + 0.5), lm)
    pb = X.parallax_bias(pose, wp, px, height_ft=0.5)
    assert pb["ok"] and pb["short"] is False
    assert pb["dX_in"] == pytest.approx(33.8, abs=1.0)
    assert any("camera is behind the dog" in n for n in pb["notes"])

    big = X.parallax_bias(pose, wp,
                          L.undistort_points(sc.entry_pixel(31.0, 8.0,
                                                           Z=sc.water_z_ft + 4.0), lm),
                          height_ft=4.0)
    assert big["dX_in"] > pb["dX_in"] * 5.0, "the bias must be first-order in the height"


# ============================================================================
# A whole track
# ============================================================================
def test_trajectory_intersections_and_the_plateau_candidate():
    sc = deck_marks_scene()
    pose, lm = _pose(sc), L.LensModel(K=sc.K, dist=sc.dist, image_size=sc.image_size,
                                      source="lines")
    top, con = _clicks(sc, lm)
    wp = S.water_plane_from_rim_pair(pose, top, con, WALL_NEAR)

    # A dog descending: airborne at 3 ft, then genuinely at the water for the last frames.  The
    # pixels are raw (the scene projects through the lens), so the undistortion is requested here.
    track = []
    for i in range(14):
        h = max(0.0, 3.0 - 0.5 * i)
        px = sc.entry_pixel(31.0, 8.0, Z=sc.water_z_ft + h)
        track.append({"frame": 100 + i, "x": float(px[0]), "y": float(px[1]), "kp_conf": 0.9})

    rows = X.trajectory_intersections(pose, wp, track, lens=lm)
    assert len(rows) == 14
    # The last frames are at the water, so they must read the true distance exactly.
    assert rows[-1]["X_ft"] == pytest.approx(31.0, abs=1e-7)
    # ... and a raw pixel passed without `lens` is measurably wrong, which is why the parameter
    # exists: measured 30.8716 ft instead of 31 ft, i.e. 1.5 in short.
    naive = X.trajectory_intersections(pose, wp, track)[-1]
    assert abs(naive["X_ft"] - 31.0) > 0.05

    plateau = X.crossing_frame_from_sequence(rows)
    assert plateau["ok"]
    assert plateau["frame"] == 113
    assert plateau["distance_ft"] == pytest.approx(31.0, abs=1e-7)
    assert "candidate" in plateau["detail"]


def test_crossing_frame_needs_enough_frames():
    r = X.crossing_frame_from_sequence([])
    assert r["ok"] is False and "plausible" not in r["detail"]
    r2 = X.crossing_frame_from_sequence([{"frame": 1, "distance_ft": 5.0},
                                         {"frame": 2, "distance_ft": 5.0}])
    assert r2["ok"] is False


def test_waterline_marks_scene_has_a_zero_offset_water_plane():
    """The other venue arrangement: marks at the waterline mean Phase 5 has nothing to extend."""
    sc = waterline_marks_scene()
    pose, lm = _pose(sc), L.LensModel(K=sc.K, dist=sc.dist, image_size=sc.image_size,
                                      source="lines")
    top, con = _clicks(sc, lm)
    wp = S.water_plane_from_rim_pair(pose, top, con, WALL_NEAR)
    assert wp.z_ft == pytest.approx(0.0, abs=1e-9)
    assert not wp.is_below()
    px = L.undistort_points(sc.entry_pixel(24.0, 5.0), lm)
    e = X.entry_coordinate(pose, wp, px)
    assert e["X_ft"] == pytest.approx(24.0, abs=1e-8)
