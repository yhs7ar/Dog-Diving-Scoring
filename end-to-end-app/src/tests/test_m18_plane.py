"""M18 — the pool plane: one aligned quad on the water surface, and the mapping it gives.

Replaces the four bars (2026-10-19).  The fixture is the known-camera synthetic scene, so the
numbers below are checked against ground truth: the plane's four corners are placed at the true
water-surface corners, and an entry at a known ``(X, Y)`` must come back as that ``X`` — which *is*
the distance from the dock edge.

Two things this file pins deliberately:

* the mapping is the **plane homography**, and the pipeline is told to answer from the document's
  own matrix.  Answering from the decomposed pose instead costs 3.9-7.9 in under an assumed lens
  (measured), because ``water.homography(pose)`` and ``doc["homography"]`` are both called "the
  homography" and are not the same matrix;
* no height is reported.  The aligned plane *is* the water surface, so the water offset is zero by
  construction and the 19.5 in Dog Diving freeboard is recorded as a specification only.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tests import pool_scene as PS

from dog_diving.pool import calib as C
from dog_diving.pool import plane as PL
from dog_diving.pool import pipeline as P
from dog_diving.pool.lens import LensModel

FREEBOARD_IN = 19.5
L_DEFAULT = 25.0


@pytest.fixture(scope="module")
def sc():
    return PS.deck_marks_scene()


def true_lens(scene) -> LensModel:
    return LensModel(K=scene.K, dist=scene.dist, image_size=scene.image_size, source="checkerboard")


def placed(scene, length: float = L_DEFAULT) -> PL.PlaneSession:
    """The plane at the scene's true water-surface corners (what an operator aims for)."""
    W = float(scene.width_ft)

    def px(X, Y):
        v = scene.project(X, Y, scene.water_z_ft)
        return (float(v[0]), float(v[1]))

    s = PL.PlaneSession(frame_shape=(scene.image_size[1], scene.image_size[0]),
                        length_ft=length, width_ft=W)
    s.set_corner("00", *px(0, 0))
    s.set_corner("10", *px(length, 0))
    s.set_corner("11", *px(length, W))
    s.set_corner("01", *px(0, W))
    return s


# ---------------------------------------------------------------------------
# 1. the quad model
# ---------------------------------------------------------------------------
def test_corner_world_puts_x_zero_on_the_dock_lip():
    assert PL.corner_world("00", 25.0, 16.0) == (0.0, 0.0)
    assert PL.corner_world("10", 25.0, 16.0) == (25.0, 0.0)
    assert PL.corner_world("11", 25.0, 16.0) == (25.0, 16.0)
    assert PL.corner_world("01", 25.0, 16.0) == (0.0, 16.0)


def test_fit_to_frame_gives_a_pool_shaped_quad_that_passes_its_gates(sc):
    s = PL.PlaneSession(frame_shape=(sc.image_size[1], sc.image_size[0]))
    assert not s.placed
    s.fit_to_frame()
    assert s.placed
    doc = s.build("fit", lens=true_lens(sc))
    assert all(g["pass"] for g in doc["plane_gates"].values()), doc["plane_gates"]
    # the X = 0 edge is the left-hand one, i.e. the dock side, and the back edge is higher
    assert s.corners["00"][0] < s.corners["10"][0]
    assert s.corners["00"][1] < s.corners["01"][1]


def test_clicking_places_the_missing_corners_in_order_in_display_coordinates(sc):
    s = PL.PlaneSession(frame_shape=(sc.image_size[1], sc.image_size[0]))
    pad = s.pad
    for i, tag in enumerate(PL.CORNER_TAGS):
        assert s.corners[tag] is None
        assert s.click(pad[0] + 100.0 + 40 * i, pad[1] + 200.0) is True
        # the stored corner is in IMAGE coordinates, i.e. the padding is removed
        assert s.corners[tag] == pytest.approx((100.0 + 40 * i, 200.0))
    assert s.placed
    assert s.click(5.0, 5.0) is False          # nowhere near a handle: nothing happens


def test_only_the_nearest_handle_is_grabbed_and_sides_move_both_corners(sc):
    s = placed(sc)
    pad = s.pad
    c00 = s.corners["00"]
    d = (pad[0] + c00[0], pad[1] + c00[1])
    assert s.click(*d) is True and s.dragging == "00"
    s.drag_to(d[0] + 30.0, d[1] + 12.0)
    assert s.corners["00"] == pytest.approx((c00[0] + 30.0, c00[1] + 12.0))
    s.release()

    # a side drag translates the edge: both of its corners move by the same delta
    mid = s.side_midpoint("back")
    before = {t: s.corners[t] for t in ("00", "10")}
    assert s.click(pad[0] + mid[0], pad[1] + mid[1]) is True and s.dragging == "back"
    s.drag_to(pad[0] + mid[0], pad[1] + mid[1] - 25.0)
    for t in ("00", "10"):
        assert s.corners[t] == pytest.approx((before[t][0], before[t][1] - 25.0))
    s.release()
    assert s.dragging is None


# ---------------------------------------------------------------------------
# 2. the mapping against ground truth
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("X,Y", [(0.5, 8.0), (5.0, 3.0), (12.0, 12.0), (22.0, 6.5)])
def test_entry_distance_is_x_with_a_measured_lens(sc, X, Y):
    doc = placed(sc).build("m18", lens=true_lens(sc))
    raw = sc.project(X, Y, sc.water_z_ft)
    out = P.translate_track([], doc=doc, manual_point=(float(raw[0]), float(raw[1])))
    e = out["entry"]
    assert e is not None, out["notes"]
    assert e["route"] == "homography", "the plane route must answer from the plane's homography"
    assert e["X_ft"] == pytest.approx(X, abs=0.01)
    assert e["distance_in"] == pytest.approx(X * 12.0, abs=0.12)
    assert e["homography_agreement_ft"] == pytest.approx(0.0, abs=1e-6)


def test_the_assumed_lens_costs_under_two_inches_and_says_so(sc):
    """The pose route costs 3.9-7.9 in here; the plane homography must not inherit that."""
    doc = placed(sc).build("m18-prior", lens=None)
    assert doc["lens"]["source"] == "assumed"
    worst = 0.0
    agree = 0.0
    for X, Y in ((0.5, 8.0), (5.0, 3.0), (12.0, 12.0), (22.0, 6.5)):
        raw = sc.project(X, Y, sc.water_z_ft)
        e = P.translate_track([], doc=doc, manual_point=(float(raw[0]), float(raw[1])))["entry"]
        worst = max(worst, abs(e["X_ft"] - X))
        agree = max(agree, float(e["homography_agreement_ft"]))
    assert worst * 12.0 < 2.0, f"assumed-lens error {worst * 12:.2f} in"
    assert agree > 0.05, "the pose/homography disagreement was not reported"
    print(f"\n  assumed lens: worst {worst * 12:.2f} in, pose-vs-plane disagreement "
          f"{agree * 12:.1f} in (reported, not hidden)")


def test_no_height_is_reported_and_the_freeboard_is_only_a_specification(sc):
    doc = placed(sc).build("m18-z", lens=true_lens(sc))
    assert doc["water"]["z_ft"] == pytest.approx(0.0, abs=1e-9)
    assert doc["water"]["method"] == "known_freeboard"
    assert doc["plane"]["freeboard_in_spec"] == pytest.approx(FREEBOARD_IN)
    assert "NOT used" in doc["plane"]["freeboard_note"]
    assert "derived" not in doc, "the plane route must not report heights"


def test_mapping_round_trips_and_resolves_distance_coarsely_at_the_far_end(sc):
    der = placed(sc).build("m18-map", lens=true_lens(sc))["mapping"]
    rows = der["probe"]
    # The metric axis runs 0 -> 35 ft (2026-10-20), whatever the quad's anchor is: this plane is
    # anchored at DEFAULT_LENGTH_FT, so the probe still ends at 35 and not at the anchor.
    assert len(rows) == 8                      # 0,5,10,...,35 ft
    assert rows[-1]["X_ft"] == pytest.approx(35.0)
    assert max(r["roundtrip_error_ft"] for r in rows) < 1e-6
    assert rows[0]["px_per_ft_along"] > 2.0 * rows[-1]["px_per_ft_along"]


# ---------------------------------------------------------------------------
# 3. what the plane can fail on, and the editor's edge cases
# ---------------------------------------------------------------------------
def test_a_self_intersecting_quad_fails_p1(sc):
    s = placed(sc)
    s.corners["10"], s.corners["11"] = s.corners["11"], s.corners["10"]   # bow tie
    doc = s.build("m18-bowtie", lens=true_lens(sc))
    assert doc["plane_gates"]["P1"]["pass"] is False
    assert "crosses itself" in doc["plane_gates"]["P1"]["detail"]


def test_a_reversed_plane_fails_p4(sc):
    """X=0 must sit at the dock end: reversing the axis makes a foot subtend MORE px at the far end."""
    s = placed(sc)
    a, b = s.corners["00"], s.corners["10"]
    c, d = s.corners["01"], s.corners["11"]
    s.corners["00"], s.corners["10"] = b, a
    s.corners["01"], s.corners["11"] = d, c
    doc = s.build("m18-flip", lens=true_lens(sc))
    assert doc["plane_gates"]["P4"]["pass"] is False
    assert "reversed" in doc["plane_gates"]["P4"]["detail"]


def test_corners_may_live_outside_the_frame(sc):
    """The pool's far end is normally out of shot, so a corner there must be legal and accurate.

    This is a *genuine* continuation of the plane past the frame (the same camera, a longer ruler),
    not an arbitrary quad: the geometry has to stay consistent or the test would be measuring the
    nonsense it created.
    """
    W = float(sc.width_ft)
    w, h = sc.image_size
    chosen = None
    for L in (35.0, 40.0, 45.0, 50.0, 60.0):
        pts = [sc.project(X, Y, sc.water_z_ft) for X in (0.0, L) for Y in (0.0, W)]
        if not all(np.all(np.isfinite(p)) for p in pts):
            continue
        # a real overshoot, not a corner 5 px past the border: 60 px minimum
        overflow = max(max(p[0] - w, p[1] - h, -p[0], -p[1]) for p in pts)
        if overflow > 60.0:
            chosen = (L, pts)
            break
    assert chosen is not None, "no plane length put a corner outside the frame"
    L, pts = chosen
    s = PL.PlaneSession(frame_shape=(h, w), length_ft=L, width_ft=W)
    (p00, p01, pL0, pLW) = pts          # project order: X=0 Y=0 ; X=0 Y=W ; X=L Y=0 ; X=L Y=W
    s.set_corner("00", float(p00[0]), float(p00[1]))
    s.set_corner("01", float(p01[0]), float(p01[1]))
    s.set_corner("10", float(pL0[0]), float(pL0[1]))
    s.set_corner("11", float(pLW[0]), float(pLW[1]))
    doc = s.build("m18-offframe", lens=true_lens(sc))
    assert all(g["pass"] for g in doc["plane_gates"].values()), doc["plane_gates"]
    worst = max(pts, key=lambda p: max(p[0] - w, p[1] - h, -p[0], -p[1]))
    over = max(worst[0] - w, worst[1] - h, -worst[0], -worst[1])
    print(f"\n  plane L={L:g} ft: a corner sits at ({worst[0]:.0f}, {worst[1]:.0f}), "
          f"{over:.0f} px outside a {w}x{h} frame")
    for X, Y in ((5.0, 3.0), (12.0, 12.0), (22.0, 6.5)):
        raw = sc.project(X, Y, sc.water_z_ft)
        e = P.translate_track([], doc=doc, manual_point=(float(raw[0]), float(raw[1])))["entry"]
        assert e["X_ft"] == pytest.approx(X, abs=0.02)


def test_the_editor_canvas_extends_beyond_the_frame(sc):
    s = PL.PlaneSession(frame_shape=(sc.image_size[1], sc.image_size[0]))
    frame = PS.frame_from_scene(sc)
    s.fit_to_frame()
    vis = PL.draw_plane(frame, s, grid=True)
    assert vis.shape[0] == frame.shape[0] + 2 * round(s.pad[1])
    assert vis.shape[1] == frame.shape[1] + 2 * round(s.pad[0])
    # a corner outside the image still draws inside the padded canvas
    s.corners["11"] = (float(sc.image_size[0] + 100), float(sc.image_size[1] + 50))
    vis2 = PL.draw_plane(frame, s, grid=True)
    assert vis2.shape == vis.shape


def test_the_grid_is_one_foot_minor_and_five_foot_major(sc):
    assert PL.MINOR_FT == 1.0 and PL.MAJOR_FT == 5.0
    s = placed(sc)
    frame = PS.frame_from_scene(sc)
    with_grid = PL.draw_plane(frame, s, grid=True)
    without = PL.draw_plane(frame, s, grid=False)
    assert with_grid.shape == without.shape
    assert not np.array_equal(with_grid, without), "the grid flag changed nothing"
    # a 5 ft step along the dock edge must span ~5x a 1 ft step (the mapping, not the drawing)
    a, b, c = (np.asarray(s.pixel_of(X, 0.0)) for X in (0.0, 1.0, 5.0))
    assert np.linalg.norm(c - a) > 3.0 * np.linalg.norm(b - a)


# ---------------------------------------------------------------------------
# 4. the document round-trips and stays editable
# ---------------------------------------------------------------------------
def test_a_plane_document_round_trips_and_rebuilds(sc, tmp_path):
    s = placed(sc)
    doc = s.build("m18-rt", lens=true_lens(sc))
    path = tmp_path / "m18.pool.json"
    C.save(doc, path)
    loaded = C.load(path)
    assert loaded["schema"] == C.SCHEMA
    assert loaded["plane"]["schema"] == PL.SCHEMA_BLOCK

    again = PL.PlaneSession.from_doc(loaded)
    assert again.length_ft == pytest.approx(s.length_ft)
    for tag in PL.CORNER_TAGS:
        assert again.corners[tag] == pytest.approx(s.corners[tag])

    # nudge a corner and rebuild: the mapping must follow the edit, not the old document
    x, y = again.corners["10"]
    again.corners["10"] = (x - 12.0, y + 6.0)
    doc2 = again.build("m18-rt", lens=true_lens(sc))
    assert doc2["homography"] != loaded["homography"]
    assert all(g["pass"] for g in doc2["plane_gates"].values()), doc2["plane_gates"]


# ---------------------------------------------------------------------------
# 5. Review: the plane the operator aligned is the plane the reviewer sees
# ---------------------------------------------------------------------------
def test_the_review_grid_is_drawn_from_the_documents_own_matrix(sc):
    from dog_diving import annotate as AN

    doc = placed(sc).build("m18-review", lens=true_lens(sc))
    frame = PS.frame_from_scene(sc)
    drawn = AN.draw_pool_surface(frame, doc, labels=True)
    assert drawn is not frame and not np.array_equal(drawn, frame), "no grid was drawn"
    same = PL.draw_grid_on(frame, doc["homography"], length_ft=25.0, width_ft=float(sc.width_ft),
                           lens=C.lens_from_document(doc), labels=True)
    assert np.array_equal(drawn, same), ("Review drew a different lattice from the document's "
                                         "matrix — that is the pose-derived grid, not the plane")

    # With a MEASURED lens the pose-derived matrix is the same map, so the two grids coincide —
    # which is why this bug is invisible on good intrinsics.  With the pane's ASSUMED lens they do
    # not, and drawing the pose-derived one puts the review lattice where the plane never was.
    prior = placed(sc).build("m18-review-prior", lens=None)
    kw = dict(length_ft=25.0, width_ft=float(sc.width_ft),
              lens=C.lens_from_document(prior), labels=True)
    from_doc = PL.draw_grid_on(frame, prior["homography"], **kw)
    from_pose = PL.draw_grid_on(frame, C.pose_from_document(prior).plane_homography(0.0), **kw)
    assert not np.array_equal(from_doc, from_pose), (
        "with an assumed lens the pose-derived grid must differ from the document's — if it does "
        "not, this test is no longer testing the wiring")


def test_review_measures_x_from_the_plane_not_the_1d_route(sc):
    import app_gui

    doc = placed(sc).build("m18-review2", lens=true_lens(sc))
    X, Y = 12.0, 9.0
    raw = sc.project(X, Y, sc.water_z_ft)

    class Traj:
        points = [{"frame": 40, "x": float(raw[0]), "y": float(raw[1]), "kp_conf": 0.9}]

    class Contact:
        chosen_frame = 40

    st = app_gui.ReviewState(clip=Path("x.mp4"), traj=Traj(), contact=Contact(),
                             calib=None, image_size=sc.image_size, n_frames=100,
                             pool_doc=doc)
    m = st.measurement()
    assert m.get("route") == "plane", m
    assert m["total_inches"] == pytest.approx(X * 12.0, abs=0.2), m
    assert "plane" in st.readout()
    # a manual click takes precedence over the trajectory point, as everywhere else
    st.manual_point = (float(sc.project(5.0, 4.0, sc.water_z_ft)[0]),
                       float(sc.project(5.0, 4.0, sc.water_z_ft)[1]))
    m2 = st.measurement()
    assert m2["total_inches"] == pytest.approx(5.0 * 12.0, abs=0.2), m2
    assert m2["point_source"] == "manual"


def test_a_document_without_a_plane_block_is_refused_by_from_doc():
    with pytest.raises(ValueError, match="no 'plane' block"):
        PL.PlaneSession.from_doc({"schema": C.SCHEMA})
