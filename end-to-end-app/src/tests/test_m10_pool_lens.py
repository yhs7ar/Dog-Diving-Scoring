"""M10 — Phase 1 (lens calibration and distortion correction) against synthetic ground truth.

Every assertion here is a measured number on the scene in :mod:`pool_scene`, not a tolerance chosen
to make the test pass.  The scene has real distortion (``k1 = -0.12``) so a phase that forgets to
undistort fails rather than quietly passing.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from tests.pool_scene import deck_marks_scene, frame_from_scene
from dog_diving.pool import lens as L


# ============================================================================
# The model and its self-consistency
# ============================================================================
def test_placeholder_intrinsics_is_labelled_as_assumed():
    lm = L.placeholder_intrinsics((1920, 1080))
    assert lm.source == "assumed"
    assert lm.deterministic is False
    assert lm.fx == pytest.approx(1920.0)                     # ratio 1.0 * max(w, h)
    assert lm.principal_point == pytest.approx((959.5, 539.5))
    assert not lm.is_distorted


def test_focal_from_hfov_is_exact_for_a_known_field_of_view():
    # A 90 deg horizontal FOV on a 1000 px sensor is exactly f = w/2.
    assert L.focal_from_hfov((1000, 500), 90.0) == pytest.approx(500.0)


def test_roundtrip_error_is_float_noise_on_a_realistic_model():
    """The distortion model has to invert itself; a model that cannot is a bug, not a tolerance.

    Measured at the frame corners, where the distortion is strongest: 1.6e-13 px with the converged
    inversion, against 3.9e-5 px with ``cv2.undistortPoints``' default stopping rule.  The assertion
    is set at the *default-rule* value so it still fails if the converged path silently stops being
    used.
    """
    sc = deck_marks_scene()
    lm = L.LensModel(K=sc.K, dist=sc.dist, image_size=sc.image_size, source="lines")
    corners = np.array([[0, 0], [1919, 0], [0, 1079], [1919, 1079], [960, 540]], dtype=float)
    assert L.roundtrip_error(corners, lm) < 3.9e-5
    assert L._undistort_criteria(), "the converged undistortion path must be available here"


def test_undistort_points_recovers_the_clean_projection():
    """Ground truth: the scene can project with and without distortion, so the correction is checkable.

    This is the strongest form of the Phase 1 test available without a checkerboard: the exact answer
    is known analytically.
    """
    sc = deck_marks_scene()
    lm = L.LensModel(K=sc.K, dist=sc.dist, image_size=sc.image_size, source="lines")
    world = [(x, y, 0.0) for x in (0.0, 1.0, 8.0, 16.0, 24.0, 31.0) for y in (0.0, 8.0, 16.0)]
    raw = sc.project_many(world)
    clean = sc.project_many_clean(world)
    und = L.undistort_points(raw, lm)
    err = np.linalg.norm(und - clean, axis=1)
    assert float(err.max()) < 1e-8, f"max undistort error {err.max():.3g} px"


def test_distortion_actually_moves_pixels_so_the_test_can_fail():
    """A sanity guard on the fixture: if the scene lost its distortion, the test above is vacuous."""
    sc = deck_marks_scene()
    raw = sc.project_many([(0.0, 0.0, 0.0), (31.0, 16.0, 0.0)])
    clean = sc.project_many_clean([(0.0, 0.0, 0.0), (31.0, 16.0, 0.0)])
    assert float(np.linalg.norm(raw - clean, axis=1).max()) > 1.0


def test_undistort_image_keeps_the_same_intrinsics():
    """``undistort_image`` must not change K, or every pixel coordinate the pipeline holds is void."""
    sc = deck_marks_scene()
    lm = L.LensModel(K=sc.K, dist=sc.dist, image_size=sc.image_size, source="lines")
    frame = frame_from_scene(sc)
    out = L.undistort_image(frame, lm)
    assert out.shape == frame.shape
    # A model with no distortion is a copy, and a copy must not alias the original buffer.
    same = L.undistort_image(frame, L.placeholder_intrinsics(sc.image_size))
    assert np.array_equal(same, frame)
    assert same is not frame


# ============================================================================
# Straightness: the metric the line route is built on
# ============================================================================
def test_straightness_is_zero_for_a_mathematically_straight_line():
    t = np.linspace(0.0, 1.0, 50)
    pts = np.column_stack([100 + 300 * t, 400 - 120 * t])
    assert L.straightness_residual(pts) < 1e-9


def test_straightness_grows_with_bow():
    """The residual is the deviation from the set's **own** best-fit line, not from its endpoints.

    Measured on a 5-unit-deep bow: 3.3 px, not 5.0 — because the best-fit line splits the bow's
    range, so the deepest point is closer to the line than to the chord.  Recording the real number
    here matters, because a residual metric that has been mentally replaced by "the sagitta" will be
    misread by a factor of ~1.5 on every real measurement.
    """
    t = np.linspace(-1.0, 1.0, 101)
    straight = np.column_stack([200 + 100 * t, 300 + 0.0 * t])
    bowed = np.column_stack([200 + 100 * t, 300 + 5.0 * (1 - t ** 2)])
    assert L.straightness_residual(straight) < 1e-9
    assert L.straightness_residual(bowed) == pytest.approx(3.3, abs=1e-6)


def test_estimate_from_lines_recovers_k1_from_the_pools_own_edges():
    """The plan's "straight pool-edge line fitting", verified against a known coefficient.

    Two long edges plus two transverse lines, spanning most of the frame, with the focal frozen at
    its true value: ``k1`` must come back to within a few percent of the truth.
    """
    sc = deck_marks_scene()
    truth = L.LensModel(K=sc.K, dist=sc.dist, image_size=sc.image_size, source="assumed")
    lines = []
    for y in (0.0, 5.0, 10.0, 16.0):
        lines.append(sc.project_many([(x, y, 0.0) for x in np.linspace(0.0, 31.0, 60)]))
    for x in (0.0, 15.0, 31.0):
        lines.append(sc.project_many([(x, y, 0.0) for y in np.linspace(0.0, 16.0, 60)]))

    before = L.total_straightness(lines)
    fitted = L.estimate_from_lines(lines, sc.image_size, lens0=truth, fit=("k1", "k2"))
    after = L.total_straightness([L.undistort_points(ln, fitted) for ln in lines])

    assert before > 0.5, f"the fixture must actually bend: measured {before:.3f} px"
    assert fitted.source == "lines"
    assert after < before / 10.0, f"straightness {before:.3f} -> {after:.3f} px"
    assert fitted.dist[0] == pytest.approx(truth.dist[0], abs=0.02)
    assert fitted.dist[1] == pytest.approx(truth.dist[1], abs=0.02)


def test_estimate_from_lines_refuses_to_claim_a_line_family_that_cannot_constrain_it():
    """ID-1: lines hugging the image centre do not determine radial distortion, and must say so.

    The guard matters because the failure is otherwise silent — the fit returns coefficients that
    are barely different from the prior while reporting a large improvement in a residual that was
    never constraining ``k1`` in the first place.
    """
    sc = deck_marks_scene()
    base = L.placeholder_intrinsics(sc.image_size)
    cx, cy = base.principal_point
    # Two short lines straddling the principal point, well inside the 35% coverage gate.
    lines = [np.column_stack([np.linspace(cx - 60, cx + 60, 15),
                              np.linspace(cy - 8, cy + 8, 15)]),
             np.column_stack([np.linspace(cx - 60, cx + 60, 15),
                              np.linspace(cy + 20, cy - 20, 15)])]
    assert L.line_coverage(lines, sc.image_size)["max_radius_frac"] < 0.35
    fitted = L.estimate_from_lines(lines, sc.image_size, lens0=base, fit=("k1", "k2"))
    assert any("ID-1" in n for n in fitted.notes), fitted.notes
    assert fitted.dist[0] == pytest.approx(0.0, abs=0.05)


def test_estimate_from_lines_warns_when_focal_and_k1_cannot_be_separated():
    """ID-2: fitting ``f`` alongside ``k1`` from straightness alone is the classic degeneracy."""
    sc = deck_marks_scene()
    lines = [sc.project_many([(x, 0.0, 0.0) for x in np.linspace(0.0, 31.0, 60)]),
             sc.project_many([(x, 16.0, 0.0) for x in np.linspace(0.0, 31.0, 60)])]
    base = L.LensModel(K=sc.K, dist=np.zeros(5), image_size=sc.image_size, source="assumed")
    fitted = L.estimate_from_lines(lines, sc.image_size, lens0=base, fit=("f", "k1", "k2"))
    assert any("ID-2" in n or "correlation" in n for n in fitted.notes), fitted.notes


def test_estimate_from_lines_with_no_parameters_is_an_honest_no_op():
    sc = deck_marks_scene()
    lines = [sc.project_many([(x, 0.0, 0.0) for x in np.linspace(0.0, 31.0, 40)])]
    fitted = L.estimate_from_lines(lines, sc.image_size, fit=())
    assert any("no distortion parameters requested" in n for n in fitted.notes)
    assert not fitted.is_distorted


def test_estimate_from_lines_rejects_a_family_that_is_not_a_line():
    sc = deck_marks_scene()
    with pytest.raises(ValueError):
        L.estimate_from_lines([np.array([[0.0, 0.0], [1.0, 1.0]])], sc.image_size)


# ============================================================================
# Checkerboard route and the JSON round trip
# ============================================================================
def test_checkerboard_calibration_recovers_a_known_camera():
    """Render an exactly-projected chessboard through a known camera and recover it.

    The corpus ships no chessboard image, so this is the only way to exercise the route — and
    exercising it matters, because it is the only path that measures the *whole* model (focal,
    principal point and all five distortion terms) rather than assuming most of it.

    Two fixture mistakes were found and fixed on the way here, both of which looked like calibrator
    failures: drawing one dot per corner (not a checkerboard, correctly rejected), and warping a
    straight-edged board through the corner homography (wrong under real distortion — it recovered a
    focal 24% off).  The board is now rendered per-pixel by inverse mapping
    (:func:`tests.pool_scene.render_checkerboard`).
    """
    import cv2

    from tests.pool_scene import render_checkerboard

    image_size = (1280, 960)
    pattern = (9, 6)
    square = 0.55
    K_true = np.array([[1100.0, 0.0, 640.0], [0.0, 1100.0, 480.0], [0.0, 0.0, 1.0]])
    dist_true = np.array([-0.18, 0.05, 0.0005, -0.0005, 0.0])

    objp = np.zeros((pattern[0] * pattern[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape(-1, 2)
    objp *= square

    frames = []
    rng = np.random.default_rng(11)
    for _ in range(18):
        # Strong tilts and a spread of depths, not just small in-plane rotations: with a planar
        # target, focal length and radial distortion are separated almost entirely by the *view
        # tilt* (measured: tilt +/-0.4 rad recovers f 24% off, +/-0.75 rad recovers it to 1.3%).
        rvec = np.array([rng.uniform(-0.75, 0.75), rng.uniform(-0.75, 0.75),
                         rng.uniform(-0.5, 0.5)])
        tvec = np.array([rng.uniform(-0.6, 0.6), rng.uniform(-0.4, 0.4), rng.uniform(6.0, 12.0)])
        frames.append(render_checkerboard(image_size, K_true, dist_true, rvec, tvec, pattern,
                                         square=square))

    lm = L.calibrate_from_checkerboard(frames, pattern, square_size=square)
    assert lm.source == "checkerboard"
    assert lm.fx == pytest.approx(K_true[0, 0], rel=0.05)
    assert lm.fy == pytest.approx(K_true[1, 1], rel=0.05)
    assert lm.rms_px is not None and lm.rms_px < 0.5

    # The individual k1/k2 split is genuinely ill-conditioned on a planar target (they are strongly
    # correlated — that is why every calibrator reports them together), so the honest assertion is on
    # the *combined radial displacement* they produce, which is what the pipeline actually uses.
    def radial(model, r):
        return r * (model[0] * r ** 2 + model[1] * r ** 4)

    for r in (0.3, 0.6, 0.9):
        assert radial(lm.dist, r) == pytest.approx(radial(dist_true, r), abs=0.01), f"r={r}"


def test_render_checkerboard_is_not_a_homography_of_a_flat_board():
    """Guard on the fixture itself: the render must bend the board, or the test above is vacuous.

    Stated geometrically rather than pixel-by-pixel, because a pixel comparison also measures
    resampling conventions: the *corner positions* of an undistorted projection are an exact
    homography image of the board lattice, and the corner positions of a distorted projection are
    not.  If someone later "simplifies" the render back to a homography warp, the residual below
    would collapse and this guard fires.
    """
    import cv2

    from tests.pool_scene import render_checkerboard

    image_size = (960, 720)
    pattern = (9, 6)
    K = np.array([[900.0, 0.0, 480.0], [0.0, 900.0, 360.0], [0.0, 0.0, 1.0]])
    dist = np.array([-0.25, 0.08, 0.0, 0.0, 0.0])
    rvec = np.array([0.4, -0.3, 0.2])
    # z=9: the bordered board spans ~7.7 world units, so at f=900 anything closer than ~9 projects
    # wider than the 960 px frame, gets cropped, and is correctly refused as a board.
    tvec = np.array([0.0, 0.0, 9.0])
    lattice = np.array([[c, r] for r in range(pattern[1]) for c in range(pattern[0])], np.float32)

    def corners_of(img):
        found, corners = cv2.findChessboardCorners(img, pattern,
                                                   cv2.CALIB_CB_ADAPTIVE_THRESH
                                                   + cv2.CALIB_CB_NORMALIZE_IMAGE)
        assert found, "the render must be detectable as a chessboard"
        return corners.reshape(-1, 2).astype(np.float32)

    def homography_residual(img):
        pts = corners_of(img)
        H, _ = cv2.findHomography(lattice, pts)
        pred = cv2.perspectiveTransform(lattice.reshape(-1, 1, 2), H).reshape(-1, 2)
        return float(np.max(np.linalg.norm(pred - pts, axis=1)))

    clean_res = homography_residual(render_checkerboard(image_size, K, np.zeros(5), rvec, tvec,
                                                        pattern, square=0.55))
    dist_res = homography_residual(render_checkerboard(image_size, K, dist, rvec, tvec,
                                                       pattern, square=0.55))
    # The absolute floor is corner-*detection* noise on a nearest-neighbour render, so the assertion
    # is relative: the distorted projection must fit a homography materially worse than a clean one
    # does.  An absolute threshold here would only be measuring the detector.
    assert dist_res > max(1.5, 3.0 * clean_res), (
        f"distorted corners must fit a homography far worse than clean ones "
        f"(clean {clean_res:.3f} px, distorted {dist_res:.3f} px)")


def test_checkerboard_route_raises_when_no_board_is_found():
    lm = None
    with pytest.raises(ValueError, match="no checkerboard found"):
        L.calibrate_from_checkerboard([np.full((120, 160), 30, np.uint8)], (9, 6))


def test_lens_model_json_round_trip_is_exact():
    sc = deck_marks_scene()
    lm = L.LensModel(K=sc.K, dist=sc.dist, image_size=sc.image_size, source="lines",
                     rms_px=0.25, notes=["hello"])
    back = L.LensModel.from_dict(lm.as_dict())
    assert np.allclose(back.K, lm.K)
    assert np.allclose(back.dist, lm.dist)
    assert back.image_size == lm.image_size
    assert back.source == lm.source
    assert back.notes == lm.notes


def test_lens_model_from_dict_pads_a_short_distortion_vector():
    doc = {"K": np.eye(3).tolist(), "dist": [0.1, 0.2], "image_size": [640, 480]}
    lm = L.LensModel.from_dict(doc)
    assert lm.dist.shape == (5,)
    assert lm.dist[0] == pytest.approx(0.1)
    assert lm.dist[4] == 0.0


def test_focal_prior_scales_with_the_long_edge():
    assert L.focal_prior_px((1920, 1080), 1.0) == pytest.approx(1920.0)
    assert L.focal_prior_px((1080, 1920), 1.0) == pytest.approx(1920.0)
    assert L.HAVE_SCIPY is True, "the plan names SciPy; it must be present in this environment"
