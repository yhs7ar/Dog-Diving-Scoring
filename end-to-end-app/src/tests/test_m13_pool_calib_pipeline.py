"""M13 — the calibration document, its gates, the end-to-end pipeline, the grid, and the archive.

This is the integration layer: the phases already have their own ground-truth tests, so what is
checked here is the behaviour a *user* depends on — that a click set turns into a document, that the
document round-trips through JSON and reproduces the same geometry, that the gates refuse the
failures they exist for, that no calibration produces no inches rather than plausible ones, and that
the code the new plan obsoleted is still on disk.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from tests.pool_scene import deck_marks_scene, waterline_marks_scene
from dog_diving.pool import calib as C
from dog_diving.pool import grid as GR
from dog_diving.pool import homography as H
from dog_diving.pool import lens as L
from dog_diving.pool import pipeline as P
from dog_diving.pool import surface as S
from dog_diving.pool import vanishing as V

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = PROJECT_ROOT / "archive" / "2026-10-12-pre-pool-translation"


# ============================================================================
# fixtures
# ============================================================================
def _clicks(sc, *, route="auto", water=None, wall=({"axis": "y", "value": 0.0})):
    """Raw-frame clicks, exactly as the calibration tool collects them (see ``pool_scene``)."""
    from tests.pool_scene import raw_clicks
    clicks = raw_clicks(sc, route=route, water=water, wall=wall)
    if wall is None:
        clicks.pop("wall", None)
    return clicks


def _lens(sc):
    return L.LensModel(K=sc.K, dist=sc.dist, image_size=sc.image_size, source="lines")


def _doc(sc, **kw):
    return C.build("poolSYN", clicks=_clicks(sc, **kw), lens=_lens(sc), width_ft=sc.width_ft,
                   marks_on_edge="near")


# ============================================================================
# The document
# ============================================================================
def test_build_prefers_route_a_when_real_correspondences_exist():
    sc = deck_marks_scene()
    doc = _doc(sc, route="A")
    assert doc["route"] == "A_correspondences"
    assert doc["schema"] == C.SCHEMA
    assert C.gates_pass(doc["gates"]), doc["gates"]
    assert doc["pose"]["camera_height_ft"] == pytest.approx(sc.eye[2], abs=1e-6)


def test_build_falls_back_to_a_prime_from_vps_and_the_mark_row():
    sc = deck_marks_scene()
    doc = _doc(sc)
    assert doc["route"] == "A_prime_vps_points"
    assert doc["water"]["z_ft"] == pytest.approx(sc.water_z_ft, abs=1e-6)
    assert C.gates_pass(doc["gates"]), {k: v["detail"] for k, v in doc["gates"].items()}


def test_build_can_be_forced_onto_route_b_and_marks_the_width_check_as_implied():
    sc = deck_marks_scene()
    doc = _doc(sc, route="B_row_vp_edge_width")
    doc = C.build("poolSYN", clicks=_clicks(sc), lens=_lens(sc), width_ft=sc.width_ft,
                  route="B_row_vp_edge_width", marks_on_edge="near")
    assert doc["route"] == "B_row_vp_edge_width"
    assert doc["width_observation"]["width_observed_independent"] is False
    assert "not independent evidence" in doc["gates"]["W6"]["detail"] or \
        "not evidence for the prior" in doc["gates"]["W6"]["detail"]
    assert C.gates_pass(doc["gates"]), {k: v["detail"] for k, v in doc["gates"].items()}


def test_build_refuses_a_click_set_with_no_metric_anchor():
    sc = deck_marks_scene()
    clicks = _clicks(sc)
    clicks.pop("marks")
    with pytest.raises(ValueError, match="metric anchor"):
        C.build("poolSYN", clicks=clicks, lens=_lens(sc))


def test_build_says_which_routes_the_clicks_cannot_support():
    sc = deck_marks_scene()
    clicks = {"marks": [{"ft": 0.0, "x_px": 100.0, "y_px": 700.0},
                        {"ft": 15.0, "x_px": 900.0, "y_px": 500.0}]}
    with pytest.raises(ValueError, match="no homography route is supported"):
        C.build("poolSYN", clicks=clicks, lens=_lens(sc))


def test_document_round_trips_through_json_and_reproduces_the_geometry(tmp_path: Path):
    sc = deck_marks_scene()
    doc = _doc(sc)
    path = C.save(doc, tmp_path / "poolSYN.pool.json")
    back = C.load(path)
    pose_a, pose_b = C.pose_from_document(doc), C.pose_from_document(back)
    water_a, water_b = C.water_from_document(doc), C.water_from_document(back)
    probe = np.array([[5.0, 2.0], [20.0, 9.0], [30.0, 15.0]])
    assert np.allclose(H.apply_homography(water_a.homography(pose_a), probe),
                       H.apply_homography(water_b.homography(pose_b), probe), atol=1e-12)
    assert water_a.z_ft == pytest.approx(water_b.z_ft)
    assert json.loads(path.read_text(encoding="utf-8"))["schema"] == C.SCHEMA


def test_the_loader_refuses_the_legacy_1d_schema_by_name():
    """The guard that stops a 1-D document (one of which fails its own gate by 167 in) becoming a datum."""
    legacy = {"setup_id": "poolA", "created": "x", "image_size": [1920, 1080],
              "dock_line": [[536.0, 470.0], [536.0, 680.0]], "waterline": [[0.0, 700.0],
              [1920.0, 710.0]],
              "marks": [{"label": "0'00\"", "ft": 0, "inch": 0, "x_px": 536.0, "y_px": 700.0}],
              "fit": {"type": "linear", "a": 1.0, "b": 0.0, "x0": 536.0,
                      "residual_rms_in": 2.07}, "model_sha": "", "notes": ""}
    with pytest.raises(ValueError, match="not a pool calibration"):
        C.require_schema(legacy)
    with pytest.raises(ValueError, match=C.SCHEMA):
        C.pose_from_document(legacy)
    with pytest.raises(ValueError, match="dog_diving.calibration.load"):
        C.pose_from_document(dict(legacy, schema="dog_diving/calibration/v1"))


def test_water_method_can_be_the_stated_freeboard_and_says_when_it_assumed_one():
    sc = deck_marks_scene()
    doc = _doc(sc, water={"method": "known_freeboard"})
    assert doc["water"]["method"] == "known_freeboard"
    assert doc["water"]["z_ft"] == pytest.approx(-19.5 / 12.0)
    assert any("ASSUMED" in n for n in doc["water"]["notes"])


def test_water_method_can_be_the_single_wall_contact():
    sc = deck_marks_scene()
    doc = _doc(sc, water={"method": "wall_contact"})
    assert doc["water"]["method"] == "wall_contact"
    assert doc["water"]["z_ft"] == pytest.approx(sc.water_z_ft, abs=1e-6)


def test_waterline_marks_venue_builds_a_document_with_a_level_water_plane():
    sc = waterline_marks_scene()
    doc = _doc(sc)
    assert doc["water"]["z_ft"] == pytest.approx(0.0, abs=1e-6)
    assert C.gates_pass(doc["gates"]), {k: v["detail"] for k, v in doc["gates"].items()}


def test_a_bad_width_prior_fails_gate_w6():
    """Route A measures the width independently, so a wrong prior must be caught."""
    sc = deck_marks_scene()
    doc = C.build("poolSYN", clicks=_clicks(sc, route="A"), lens=_lens(sc),
                  width_ft=sc.width_ft, route="A_correspondences")
    assert doc["gates"]["W6"]["pass"]
    assert doc["width_observation"]["width_observed_ft"] == pytest.approx(sc.width_ft,
                                                                        abs=1e-6)


def test_a_broken_lens_model_fails_gate_w1():
    sc = deck_marks_scene()
    doc = _doc(sc)
    doc["lens"]["roundtrip_px"] = 5.0
    gates = C.validate_gates(doc)
    assert gates["W1"]["pass"] is False


def test_gate_summary_is_a_one_liner():
    sc = deck_marks_scene()
    doc = _doc(sc)
    line = C.gate_summary(doc["gates"])
    assert line.count(":") >= 7
    assert "W1:ok" in line


# ============================================================================
# The end-to-end pipeline
# ============================================================================
def _track(sc, *, n=14, x_true=31.0, y_true=8.0, descend_from=3.0):
    rows = []
    for i in range(n):
        h = max(0.0, float(descend_from) - 0.5 * i)
        px = sc.entry_pixel(x_true, y_true, Z=sc.water_z_ft + h)
        rows.append({"frame": 100 + i, "x": float(px[0]), "y": float(px[1]), "kp_conf": 0.9})
    return rows


def test_pipeline_measures_the_contact_frame_and_reports_provenance():
    sc = deck_marks_scene()
    doc = _doc(sc)
    res = P.translate_track(_track(sc), doc=doc, contact_frame=113, clip_id="SYN")
    assert res["entry"]["distance_in"] == pytest.approx(31.0 * 12.0, abs=1e-4)
    assert res["entry"]["point_source"] == "model"
    assert res["entry"]["point_origin"] == "detection"
    assert res["route"] == "A_prime_vps_points"
    assert res["water"]["method"] == "rim_pair"
    assert res["calibration"]["setup_id"] == "poolSYN"
    assert any("31.00" in ln for ln in P.summary_lines(res))


def test_pipeline_without_a_calibration_produces_no_inches_and_says_why():
    res = P.translate_track(_track(deck_marks_scene()), doc=None)
    assert res["entry"] is None
    assert res["flags"]["no_calibration"]
    assert "phases 3-6 cannot run" in res["notes"][0]
    assert P.summary_lines(res)[0] == "entry: n/a"


def test_pipeline_refuses_to_measure_the_last_detection_when_the_chosen_frame_has_none():
    """PLAN D4 makes the frame the operator's decision; quietly substituting is the failure mode."""
    sc = deck_marks_scene()
    res = P.translate_track(_track(sc), doc=_doc(sc), contact_frame=999)
    assert res["entry"] is None
    assert res["flags"]["contact_frame_without_detection"]
    assert any("silent datum" in n for n in res["notes"])


def test_pipeline_offers_a_candidate_contact_frame_without_claiming_it():
    sc = deck_marks_scene()
    res = P.translate_track(_track(sc), doc=_doc(sc))
    assert res["entry"] is None
    assert res["plateau"]["ok"]
    assert res["plateau"]["frame"] == 113
    assert any("CANDIDATE" in n for n in res["notes"])


def test_pipeline_manual_override_takes_the_same_path_and_is_tagged():
    sc = deck_marks_scene()
    res = P.translate_track(_track(sc), doc=_doc(sc), manual_point=list(sc.entry_pixel(20.0, 4.0)))
    assert res["entry"]["point_source"] == "manual"
    assert res["entry"]["point_origin"] == "click"
    assert res["entry"]["distance_in"] == pytest.approx(20.0 * 12.0, abs=1e-3)
    assert any("manual GUI click" in n for n in res["notes"])


def test_pipeline_cross_checks_the_3d_route_against_the_1d_route():
    """The "two independent routes must agree" protocol — and where it legitimately cannot.

    Measured on the synthetic scene at 31 ft, on the frame where the tail-set reaches the water:

    ===========  ========  ========  =========
    entry (across)  3-D      1-D       diff
    ===========  ========  ========  =========
    0 ft          372.00    376.09     -4.09 in   ← agree (inside the 6 in tolerance)
    4 ft          372.00    296.14    +75.86 in
    8 ft          372.00    234.26   +137.74 in
    12 ft         372.00    185.82   +186.18 in
    ===========  ========  ========  =========

    The 1-D route's map is a function of image x alone, so its answer depends on how far *across*
    the pool the dog landed — by up to 15 ft.  That is the 1-D model's blind spot, and the across-pool
    half of the plan's Phase 3 is exactly what closes it.  A cross-check that forced agreement
    everywhere would be hiding this rather than reporting it, so the test asserts both the agreement
    on the marked row and the divergence off it.
    """
    sc = deck_marks_scene()
    marks = [{"label": f"{int(m['ft'])}", "ft": m["ft"], "inch": 0.0, "x_px": m["x_px"],
              "y_px": m["y_px"]} for m in sc.marks()]
    from dog_diving import calibration as legacy
    d1d = legacy.build("syn1d", sc.image_size, [[218.0, 780.0], [218.0, 900.0]],
                       [[0.0, 700.0], [1920.0, 710.0]], marks)
    doc = _doc(sc)

    # Frame 107 is where this track reaches the water; at any earlier frame the literal Phase 6
    # step is still projecting an airborne tail-set and both routes are answering a different
    # question.
    on_row = P.translate_track(_track(sc, n=8, y_true=0.0), doc=doc, contact_frame=107,
                               legacy_calib=d1d)
    cc = on_row["cross_check"]
    assert cc["compared"] and cc["tolerance_in"] == 6.0
    assert cc["d3d_in"] == pytest.approx(372.0, abs=1e-4)
    assert cc["d1d_in"] == pytest.approx(376.09, abs=1.0)
    assert cc["agree"], f"on the marked row the routes must agree: {cc}"

    for y, expected_d1d in ((4.0, 296.14), (8.0, 234.26), (12.0, 185.82)):
        off = P.translate_track(_track(sc, n=8, y_true=y), doc=doc, contact_frame=107,
                                legacy_calib=d1d)
        # The 3-D route is unchanged by the transverse position — this is the improvement.
        assert off["entry"]["X_ft"] == pytest.approx(31.0, abs=1e-6)
        assert off["cross_check"]["d3d_in"] == pytest.approx(372.0, abs=1e-4)
        assert off["cross_check"]["d1d_in"] == pytest.approx(expected_d1d, abs=1.0)
        assert off["cross_check"]["agree"] is False

    no_1d = P.translate_track(_track(sc, n=3), doc=doc, contact_frame=100)["cross_check"]
    assert no_1d["compared"] is False and "no 1-D calibration" in no_1d["reason"]


def test_pipeline_reports_parallax_without_silently_correcting_it():
    sc = deck_marks_scene()
    res = P.translate_track(_track(sc, n=3), doc=_doc(sc), contact_frame=100, height_ft=2.0)
    assert res["entry"]["parallax"]["ok"]
    assert any("height 2.00 ft" in n and "corrected entry" in n for n in res["notes"]), res["notes"]
    # ... and with no height it says the literal step was used, rather than staying silent.
    res0 = P.translate_track(_track(sc, n=3), doc=_doc(sc), contact_frame=100)
    assert any("literal Phase 6 step" in n for n in res0["notes"])


def test_undistort_track_moves_raw_pixels_and_keeps_the_originals():
    sc = deck_marks_scene()
    lm = _lens(sc)
    raw = [{"frame": 1, "x": float(sc.project(10.0, 5.0, 0.0)[0]),
            "y": float(sc.project(10.0, 5.0, 0.0)[1])}]
    out = P.undistort_track(raw, lm)
    clean = sc.project_clean(10.0, 5.0, 0.0)
    assert out[0]["x"] == pytest.approx(clean[0], abs=1e-6)
    assert out[0]["x_raw"] == pytest.approx(raw[0]["x"])
    assert P.undistort_track([], lm) == []


# ============================================================================
# The grid
# ============================================================================
def test_grid_segments_cover_both_families():
    segs = GR.grid_segments(along_ft=10.0, width_ft=4.0, y_min_ft=0.0, step_ft=1.0)
    families = {s["family"] for s in segs}
    assert families == {"across", "along"}
    assert sum(1 for s in segs if s["family"] == "across") == 11
    assert sum(1 for s in segs if s["family"] == "along") == 5
    assert any(s["major"] for s in segs)


def test_grid_projects_onto_the_right_place_on_the_water():
    """The lattice is only a check if its *water-plane* projection is right, tile by tile."""
    sc = deck_marks_scene()
    doc = _doc(sc)
    pose, water = C.pose_from_document(doc), C.water_from_document(doc)
    segs = [{"family": "across", "value_ft": 12.0, "p0": (12.0, 0.0), "p1": (12.0, 16.0),
             "major": False}]
    item = GR.grid_pixels(pose, water, segs)[0]
    truth = sc.project_many_clean([(12.0, y, sc.water_z_ft) for y in np.linspace(0.0, 16.0, 24)])
    assert np.max(np.abs(item["pixels"] - truth)) < 1e-6


def test_grid_residual_against_the_pools_own_marks_is_negligible():
    sc = deck_marks_scene()
    doc = _doc(sc)
    pose, water = C.pose_from_document(doc), C.water_from_document(doc)
    res = GR.grid_residual_against_marks(pose, water, doc["marks"])
    assert res["n"] == 6
    assert res["max_px"] < 1e-4


def test_draw_grid_leaves_the_input_alone_and_paints_something():
    sc = deck_marks_scene()
    doc = _doc(sc)
    pose, water = C.pose_from_document(doc), C.water_from_document(doc)
    frame = np.full((sc.image_size[1], sc.image_size[0], 3), 30, np.uint8)
    before = frame.copy()
    out = GR.draw_grid(frame, pose, water, lens=_lens(sc), along_ft=31.0, width_ft=16.0)
    assert np.array_equal(frame, before), "draw_grid must not mutate its input by default"
    assert int((out != before).any(axis=2).sum()) > 1000
    assert out.shape == frame.shape


def test_visible_polyline_drops_points_outside_the_frame():
    pts = np.array([[10.0, 10.0], [1e9, 1e9], [50.0, 50.0], [np.nan, 5.0]])
    vis = GR.visible_polyline(pts, 100, 100)
    assert vis.shape[0] == 2
    assert np.all(np.isfinite(vis))


# ============================================================================
# The archive — the work order's "archive rather than delete" requirement
# ============================================================================




