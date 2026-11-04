"""M30 — the 3-D trajectory plane route (``intersection-algo.txt`` phases 1-5).

What this file is for
---------------------
The 2-D route decides the entry frame by intersecting a *pixel-space* curve with a *waterline row*
and then intersects one *pixel's* ray with the water plane.  The 3-D route replaces both with one
statement about the pool: the flight is a curve in world space, and the entry is where that curve
crosses ``Z = z_water``.  This file tests that arithmetic on ground truth, and it also records the
one number the route cannot fix by itself — what the *single-pass* bootstrap costs on a shallow
camera — so the cost is pinned in a test rather than living in a commit message.

The measured numbers below (110.01 in, 2.598 ft plane RMS, heights reaching -5.24 ft) come from
this file's own scene, not from speculation: each is asserted, and if the bootstrap is ever
replaced they will fail and say so.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.pool_scene import deck_marks_scene, raw_clicks
from dog_diving import contact as K
from dog_diving import trajectory as T
from dog_diving.pool import calib as C
from dog_diving.pool import geometry as G
from dog_diving.pool import homography as H
from dog_diving.pool import intersection as X
from dog_diving.pool import lens as L
from dog_diving.pool import pipeline as P
from dog_diving.pool import surface as S

FPS = 30.0
G_FT_PER_FRAME2 = 32.174 / FPS ** 2        # 0.035749 ft/frame² — a free fall at 30 fps


# ============================================================================
# fixtures — a scene, a pose, and a flight with a known answer
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


def _water(sc):
    return S.WaterPlane(z_ft=float(sc.water_z_ft), method="known_freeboard", sigma_ft=0.0)


def _flight(sc, *, x0=0.5, y0=1.0, vx=0.60, vz=0.50, n=24):
    """A ballistic tail-set in a vertical plane at ``Y = y0``, with its pixels.

    The truth is generated here and only the pixels are handed to the pipeline — the world columns
    stay behind as the answer key, which is the only way this test can tell a correct
    reconstruction from a self-consistent wrong one.
    """
    out = []
    for k in range(n):
        t = float(k)
        X_ft = x0 + vx * t
        Z_ft = sc.water_z_ft + vz * t - 0.5 * G_FT_PER_FRAME2 * t * t
        px = sc.project_clean(X_ft, y0, Z_ft)
        out.append({"frame": k, "X_ft": X_ft, "Y_ft": y0, "Z_ft": Z_ft,
                    "x": float(px[0]), "y": float(px[1])})
    return out


def _crossing(point, z_water, g=G_FT_PER_FRAME2):
    """Closed-form descending crossing of the generated arc: ``(X_ft, t)``."""
    vz = (point[1]["Z_ft"] - point[0]["Z_ft"]) + 0.5 * g
    disc = vz ** 2 - 2 * (g / 2) * (point[0]["Z_ft"] - z_water)
    t = (-vz - np.sqrt(disc)) / (-g)
    return point[0]["X_ft"] + (point[1]["X_ft"] - point[0]["X_ft"]) * t, float(t)


def _pixels_only(pts):
    """The track as a detector emits it: pixels and frames, no world columns."""
    return [{"frame": p["frame"], "x": p["x"], "y": p["y"], "kp_conf": 0.9, "accepted": True}
            for p in pts]


def _coeffs_from_ballistic(z_water, t_ref=10.0, vz=0.5, x0=2.0, vx=0.6):
    """``{t_ref, x, y, z}`` for a known parabola, in the project's centred-time convention."""
    return {"t_ref": float(t_ref),
            "x": [0.0, vx, x0],
            "y": [0.0, 0.0, 1.0],
            "z": [-0.5 * G_FT_PER_FRAME2, vz, z_water + 1.0]}


def _doc(sc):
    return C.build("pool3D", clicks=raw_clicks(sc), lens=L.LensModel(
        K=sc.K, dist=sc.dist, image_size=sc.image_size, source="lines"), width_ft=sc.width_ft)


# ============================================================================
# Phase 1 — the vertical trajectory plane
# ============================================================================
def test_a_flight_in_one_vertical_plane_is_recovered_exactly():
    pts = [(0.5 + 0.6 * t, 2.0) for t in range(20)]
    fit = G.vertical_plane_from_points(pts)
    assert fit["status"] == "ok"
    plane = fit["plane"]
    assert plane.is_vertical
    assert np.allclose(np.abs(plane.normal), [0.0, 1.0, 0.0], atol=1e-12)
    assert abs(float(fit["rms_ft"])) < 1e-12
    # the plane's offset is the world Y of the flight, up to the normal's sign
    assert plane.d == pytest.approx(2.0 if plane.normal[1] > 0 else -2.0, abs=1e-12)
    assert fit["span_ft"] == pytest.approx(0.6 * 19, rel=1e-9)


def test_the_plane_sign_does_not_depend_on_point_order():
    """A track fed backwards must fit the same plane — otherwise ``d`` flips with frame order."""
    pts = [(0.5 + 0.6 * t, 2.0) for t in range(20)]
    a = G.vertical_plane_from_points(pts)["plane"]
    b = G.vertical_plane_from_points(list(reversed(pts)))["plane"]
    assert np.allclose(a.normal, b.normal)
    assert a.d == pytest.approx(b.d, abs=1e-12)


def test_the_plane_fit_reports_why_it_cannot_and_never_falls_back_to_pixels():
    """Three failure modes, three statuses — plus the one that used to pass silently.

    A flat detection row keys its pixel columns ``x``/``y``.  A world-space plane fitted through
    those and then used as a world plane is nonsense that *looks* like a fit, so bare ``x``/``y``
    dicts are rejected instead of being read as feet.
    """
    two = G.vertical_plane_from_points([(1.0, 1.0), (2.0, 1.0)])
    assert two["status"] == "few_points" and two["plane"] is None

    static = G.vertical_plane_from_points([(6.0, 4.0)] * 5)
    assert static["status"] == "no_span"
    assert "no flight direction" in static["detail"]

    pixels = [{"frame": i, "x": 900.0 + 12 * i, "y": 480.0 - 3 * i} for i in range(8)]
    assert G.vertical_plane_from_points(pixels)["status"] == "few_points"


# ============================================================================
# Phase 1 — a curve crossing a fixed Z
# ============================================================================
def test_the_water_crossing_is_solved_exactly_and_takes_the_descending_branch():
    z_water = -1.625
    coeffs = _coeffs_from_ballistic(z_water)
    t_ref = coeffs["t_ref"]
    # apex of Z(t): t_ref - a1 / (2 a2)
    apex = t_ref - coeffs["z"][1] / (2.0 * coeffs["z"][0])
    cross = G.intersect_curve_with_elevation(coeffs, z_water, t_min=apex)
    assert cross["status"] == "ok"
    assert len(cross["t_all"]) == 2                      # a ballistic curve crosses twice
    assert cross["t"] == cross["t_all"][1]               # the later one is the descent
    assert cross["point"][2] == pytest.approx(z_water, abs=1e-12)
    # independent arithmetic: the analytic roots of a2 tc² + a1 tc + (a0 - z) = 0, of which the
    # descending branch is the LATER one (a2 < 0 for free fall, so the ± order flips)
    a2, a1, a0 = coeffs["z"]
    disc = a1 ** 2 - 4 * a2 * (a0 - z_water)
    roots = sorted(((-a1 + s * np.sqrt(disc)) / (2 * a2) + t_ref) for s in (1.0, -1.0))
    assert cross["t"] == pytest.approx(roots[-1], rel=1e-12)
    assert cross["t_all"] == pytest.approx(roots, rel=1e-12)
    # and the point is the curve at that time
    assert np.allclose(cross["point"], G.curve_point(coeffs, cross["t"]))


def test_the_crossing_refuses_a_2d_fit_and_an_unreachable_height():
    flat = {"t_ref": 3.0, "x": [0.0, 1.0, 0.0], "y": [0.0, 1.0, 0.0]}
    assert G.intersect_curve_with_elevation(flat, 0.0)["status"] == "no_third_axis"

    high = _coeffs_from_ballistic(-1.625)
    out = G.intersect_curve_with_elevation(high, 99.0)
    assert out["status"] == "no_crossing"
    assert "never reaches" in out["detail"]


# ============================================================================
# Phase 2 — the fit in world space
# ============================================================================
def test_the_world_fit_recovers_the_true_acceleration_and_refuses_pixel_columns():
    sc = deck_marks_scene()
    pts = _flight(sc)
    traj = T.build_trajectory(pts, clip_id="synth", space="3d")
    assert traj.space == "3d"
    assert traj.coeffs is not None and "z" in traj.coeffs
    # the quadratic's half-acceleration IS -g/2 for a real free fall
    assert traj.coeffs["z"][0] == pytest.approx(-0.5 * G_FT_PER_FRAME2, rel=1e-6)
    assert traj.arc_rms_ft is not None and traj.arc_rms_ft < 1e-9
    assert traj.arc_rms_px is None          # a px field never carries feet
    assert traj.max_inlier_res_ft is not None

    with pytest.raises(ValueError, match="Unproject the track first"):
        T.build_trajectory(_pixels_only(pts), clip_id="synth", space="3d")


def test_the_2d_fit_is_untouched_by_the_third_axis():
    """The parity-pinned route: same keys, same numbers, when z is not supplied."""
    recs = [{"frame": i, "x": 100.0 + 20 * i, "y": 500.0 - 40 * i + 1.5 * i * i, "kp_conf": 0.9}
            for i in range(12)]
    t = np.array([r["frame"] for r in recs], float)
    x = np.array([r["x"] for r in recs], float)
    y = np.array([r["y"] for r in recs], float)
    a = T.robust_parabola(t, x, y, 20.0)
    b = T.robust_parabola(t, x, y, 20.0, z=None)
    assert a == b
    assert set(a) == {"t_ref", "x", "y", "inliers", "residuals", "n_inliers",
                      "arc_rms_px", "max_inlier_res_px", "arc_rms_all_px"}

    z = np.zeros_like(x)
    three = T.robust_parabola(t, x, y, 20.0, z=z)
    assert three["arc_rms_px"] is None and three["arc_rms_ft"] is not None
    assert three["residual_unit"] == "ft"
    assert T.predict3(three, float(t[0]))[2] == pytest.approx(0.0, abs=1e-9)


def test_predict3_refuses_a_2d_fit_rather_than_inventing_a_zero_height():
    with pytest.raises(ValueError, match="no world Z axis"):
        T.predict3({"t_ref": 0.0, "x": [0, 1, 0], "y": [0, 1, 0]}, 1.0)


def test_the_review_refit_follows_the_world_curve_on_the_3d_route():
    """Deselecting a point has to move the *world* curve, not a pixel-space lookalike of it."""
    sc = deck_marks_scene()
    pts = _flight(sc)
    keep = pts[:-3]
    out = T.refit(keep, space="3d")
    assert out["coeffs"] is not None and "z" in out["coeffs"]
    assert out["rms_ft"] is not None and out["rms_ft"] < 1e-9
    assert out["rms_px"] is None
    assert out["coeffs"]["z"][0] == pytest.approx(-0.5 * G_FT_PER_FRAME2, rel=1e-6)
    assert out["frame_range"] == [0, len(keep) - 1]
    # and the 2-D refit is untouched when no space is asked for
    flat = T.refit(_pixels_only(keep))
    assert flat["rms_px"] is not None and "z" not in (flat["coeffs"] or {})


# ============================================================================
# Phase 4 — the intersection arithmetic, on ground truth
# ============================================================================
def test_the_3d_entry_is_exact_when_the_points_are_really_3d():
    """Hand the fit true world points and the arithmetic must be exact to a rounding error.

    This is the test that separates *this* route's arithmetic from *this* route's bootstrap: if it
    ever fails, the maths broke; if the bootstrap test below fails instead, the inputs broke.
    """
    sc = deck_marks_scene()
    pose, water = _pose(sc), _water(sc)
    pts = _flight(sc)
    x_true, t_true = _crossing(pts, sc.water_z_ft)

    traj = T.build_trajectory(pts, clip_id="synth", space="3d", plane=None)
    e = X.entry_coordinate(pose, water, None, trajectory=traj)

    assert e["route"] == "arc3d"
    assert e["X_ft"] == pytest.approx(x_true, abs=0.005)
    assert e["distance_in"] == pytest.approx(x_true * 12.0, abs=0.06)
    assert e["Y_ft"] == pytest.approx(1.0, abs=1e-6)
    assert e["Z_ft"] == pytest.approx(sc.water_z_ft, abs=1e-12)
    assert e.get("crossing_frame") == pytest.approx(t_true, abs=1e-6)
    # the pixel is an OUTPUT on this route: it must be the reprojection of that point
    uv = G.project_point(pose, (e["X_ft"], e["Y_ft"], e["Z_ft"]))
    assert (e["x_px"], e["y_px"]) == pytest.approx((float(uv[0]), float(uv[1])), abs=1e-6)


def test_the_3d_entry_reports_how_far_past_the_data_it_had_to_extrapolate():
    sc = deck_marks_scene()
    pose, water = _pose(sc), _water(sc)
    pts = _flight(sc, n=18)                      # track stops at frame 17, crossing is later
    traj = T.build_trajectory(pts, clip_id="synth", space="3d")
    e = X.entry_coordinate(pose, water, None, trajectory=traj)
    x_true, t_true = _crossing(pts, sc.water_z_ft)
    assert e["X_ft"] == pytest.approx(x_true, abs=0.02)
    assert e["extrap_frames"] == pytest.approx(t_true - pts[-1]["frame"], abs=1e-6)
    assert e["flags"]["arc_extrapolated"] is True


def test_a_2d_trajectory_is_refused_by_the_3d_entry_not_silently_used():
    sc = deck_marks_scene()
    pose, water = _pose(sc), _water(sc)
    flat = T.build_trajectory(_pixels_only(_flight(sc)), clip_id="synth")
    out = X.entry_coordinate(pose, water, None, trajectory=flat)
    assert out["X_ft"] is None
    assert out["flags"]["arc_failed"] is True
    assert "image space" in out["notes"][0]


def test_the_single_pixel_route_still_works_and_is_still_the_default():
    """Phase 4 must not have narrowed the old call: a click with no trajectory still measures."""
    sc = deck_marks_scene()
    pose, water = _pose(sc), _water(sc)
    px = sc.project_clean(20.0, 1.0, sc.water_z_ft)
    e = X.entry_coordinate(pose, water, px)
    assert e["route"] in ("ray_plane", "homography")
    assert e["X_ft"] == pytest.approx(20.0, abs=0.05)


# ============================================================================
# Phase 5 — the bootstrap, and the cost of a single pass (measured)
# ============================================================================
def test_the_single_pass_bootstrap_cost_is_what_it_was_measured_to_be():
    """The single pass fits the *parallax*, not the flight, on this shallow camera.

    Unprojecting an airborne tail-set to ``z = z_water`` slides it ~5 ft along its ray.  The plane
    fitted through those slid points is therefore wrong, and the entry inherits the error: +110 in
    against ground truth on this scene (the plain single-pixel answer is +65 in).  The numbers are
    asserted so that a future bootstrap change has to come here and state its own.
    """
    sc = deck_marks_scene()
    pose, water = _pose(sc), _water(sc)
    pts = _flight(sc)
    x_true, _t = _crossing(pts, sc.water_z_ft)

    world = P.track_to_world(pose, water, _pixels_only(pts))
    assert world["status"] == "ok"
    assert world["rms_ft"] > 1.0                        # measured 2.598 ft
    assert world["flags"]["plane_wide"] is True
    assert world["height_range_ft"][0] < -1.0           # measured -5.24 ft: below the water
    assert abs(world["max_reprojection_px"]) < 1e-6     # the arithmetic itself is clean

    traj = T.build_trajectory(world["points"], clip_id="synth", space="3d", plane=world["plane"])
    e = X.entry_coordinate(pose, water, None, trajectory=traj)
    err_in = (e["X_ft"] - x_true) * 12.0
    assert err_in == pytest.approx(110.0, abs=12.0), f"bootstrap cost moved: {err_in:+.1f} in"


def test_the_pipeline_declines_the_3d_route_when_its_own_plane_is_too_wide():
    """A wrong-but-confident number is worse than a refusal: the gate has to be decisive."""
    sc = deck_marks_scene()
    doc = _doc(sc)
    pts = _flight(sc)
    res = P.translate_track(_pixels_only(pts), doc=doc, contact_frame=pts[-1]["frame"],
                            clip_id="synth", trajectory_space="3d")
    assert res["flags"]["arc3d_plane_too_wide"] is True
    assert "arc_entry" not in res
    assert any("declined to answer" in n for n in res["notes"])
    # the pixel route still answered — the refusal is of the 3-D route, not of the measurement
    assert res["entry"] is not None and res["entry"]["route"] != "arc3d"


def test_a_steep_view_passes_the_plane_gate_and_is_still_73_in_wrong():
    """The gate is necessary and not sufficient — so the record has to say so itself.

    From a 60 ft camera the smear is small enough that the single pass fits one plane well
    (RMS 0.095 ft, gate passes) while *under*-estimating every height: the entry lands 73 in short
    of ground truth.  A caller that only checked `flags` would ship that number.  Hence the standing
    ``arc3d_unverified_bootstrap`` flag and the note on every arc entry.
    """
    sc = deck_marks_scene(eye=(19.0, 10.0, 60.0))
    doc = _doc(sc)
    pts = _flight(sc, y0=3.0)
    x_true, _t = _crossing(pts, sc.water_z_ft)

    res = P.translate_track(_pixels_only(pts), doc=doc, contact_frame=pts[-1]["frame"],
                            clip_id="steep", trajectory_space="3d")
    assert res["flags"].get("arc3d_plane_too_wide") is None       # the gate passes...
    assert res["world"]["rms_ft"] < 0.5
    entry = res["entry"]
    assert entry["route"] == "arc3d"                              # ...and the arc answers
    err_in = (entry["X_ft"] - x_true) * 12.0
    assert err_in == pytest.approx(-73.0, abs=15.0), f"steep-view cost moved: {err_in:+.1f} in"
    # the arbitration itself is wired: the pixel answer is kept beside it, and the difference stated
    assert res["entry_pixel"]["route"] == "ray_plane"
    assert res["arc_vs_pixel_in"] == pytest.approx(
        entry["distance_in"] - res["entry_pixel"]["distance_in"])
    assert entry["flags"]["arc3d_unverified_bootstrap"] is True
    assert any("UNVERIFIED BOOTSTRAP" in n for n in entry["notes"])


def test_the_pipeline_still_defaults_to_the_2d_route_and_never_mixes_the_two():
    sc = deck_marks_scene()
    doc = _doc(sc)
    pts = _flight(sc)
    res = P.translate_track(_pixels_only(pts), doc=doc, contact_frame=pts[-1]["frame"],
                            clip_id="synth")
    assert res["trajectory_space"] == "2d"
    assert "world" not in res and "arc_entry" not in res
    assert res["entry"] is not None and res["entry"]["route"] != "arc3d"

    with pytest.raises(ValueError, match="trajectory_space must be"):
        P.translate_track(_pixels_only(pts), doc=doc, trajectory_space="3.5d")


# ============================================================================
# Phase 3 — the contact candidates on the world route
# ============================================================================
def test_candidate_two_solves_the_water_plane_and_the_reasons_say_so():
    sc = deck_marks_scene()
    pts = _flight(sc)
    x_true, t_true = _crossing(pts, sc.water_z_ft)
    traj = T.build_trajectory(pts, clip_id="synth", space="3d")

    res = K.contact_candidates(_pixels_only(pts), traj=traj, space="3d",
                               water_z_ft=sc.water_z_ft, clip_id="synth")
    assert res.space == "3d" and res.water_z_ft == pytest.approx(sc.water_z_ft)
    assert res.waterline_y is None                  # there is no row on this route
    c2 = res.by_source(K.S_ARC3D)
    assert c2.available is True
    assert c2.frame == int(round(t_true))
    assert c2.evidence["x_at_crossing"] == pytest.approx(x_true, abs=0.01)
    assert "z = " in c2.reason
    # the 2-D candidate is not in the set at all — the two routes never mix
    assert all(c.source != K.S_ARC for c in res.candidates)
    assert [c.source for c in res.ranked()] == list(K.ORDER_3D)
    assert res.arc_evidence["cz2"] == pytest.approx(-0.5 * G_FT_PER_FRAME2, rel=1e-6)


def test_the_world_rules_read_feet_per_frame_and_report_it():
    sc = deck_marks_scene()
    pts = _flight(sc)
    traj = T.build_trajectory(pts, clip_id="synth", space="3d")
    ok, reason, ev = K.arc_is_freefall_3d(traj)
    assert ok and "ft/frame" in reason
    assert ev["residual_unit"] == "ft"
    assert "z_span_ft" in ev and ev["z_span_ft"] > 1.0
    # the fitted descent rate at the end of the run is negative (falling), in ft/frame
    rate = K.world_descent_rate(traj.coeffs, float(pts[-1]["frame"]))
    assert rate < 0.0
    assert rate == pytest.approx(
        traj.coeffs["z"][1] + 2 * traj.coeffs["z"][0] * (pts[-1]["frame"] - traj.coeffs["t_ref"]))
    # and the image-space screen refuses to pretend it can answer in feet
    flat = T.build_trajectory(_pixels_only(pts), clip_id="synth")
    ok2, reason2, _ = K.arc_is_freefall_3d(flat)
    assert ok2 is False and "image space" in reason2


def test_the_2d_candidate_set_is_unchanged():
    sc = deck_marks_scene()
    pts = _flight(sc)
    flat = T.build_trajectory(_pixels_only(pts), clip_id="synth")
    res = K.contact_candidates(_pixels_only(pts), traj=flat, waterline_y=480.0, clip_id="synth")
    assert [c.source for c in res.ranked()] == [K.S_LAST, K.S_ARC, K.S_COLLAPSE, K.S_VEL]
    assert res.space == "2d" and res.waterline_y == 480.0
