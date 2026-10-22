"""M21 — automatic plane placement from the detected waterlines (PLAN §17).

Two layers are tested separately, because they fail differently:

* ``dog_diving.waterline`` turns a noise mask into three classified edges.  Tested on synthetic masks
  and stacks with a *known* boundary, since a real clip has no ground truth to assert against.
* ``dog_diving.pool.seed`` turns three lines into a quad.  Tested against a **model-consistent** scene:
  three lines generated from the same projective law the seed inverts, so the recovery is exact to
  machine precision.  That is the only way to tell an algebra error from a modelling choice.

The gates are tested by planting one defect each.  Deliberately **not** tested: that the seed agrees
with the line it was seeded from — that number is ~0 px by construction and is not a gate (the same
class of bug as a reversed plane scoring 1.00 on the old P4).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dog_diving import waterline as W                                  # noqa: E402
from dog_diving.pool import plane as P                                 # noqa: E402
from dog_diving.pool import seed as S                                  # noqa: E402

H, W_PX = 1080, 1920


# ---------------------------------------------------------------------------
# fixtures / helpers
# ---------------------------------------------------------------------------
def unit(v):
    v = np.asarray(v, dtype=float)
    return v / float(np.linalg.norm(v))


def edge(name, p, d, *, rms=1.0, inlier=0.9, stability=0.5, found=True, n=200):
    """A real ``EdgeFit`` with a hand-written fit dict, in the shape ``fit_straight_line`` returns."""
    p = np.asarray(p, dtype=float)
    d = unit(d)
    pts = np.stack([p, p + 300.0 * d])
    return W.EdgeFit(
        name=name, found=found, points=pts,
        fit={"point": list(map(float, p)), "direction": list(map(float, d)),
             "rms_residual": rms, "p95_residual": 2 * rms, "rms_center": rms * 0.9,
             "rms_ends": rms * 1.2, "angle_deg": float(np.degrees(np.arctan2(d[1], d[0]))),
             "slope": 0.0, "intercept": 0.0, "endpoints": [list(map(float, pts[0])),
                                                           list(map(float, pts[1]))],
             "n": n},
        inlier_fraction=inlier, stability_px=stability, reason="")


def edges3(far, near, dock, *, dock_side="right", water=0.6, comps=1, kept=1, share=0.95):
    return W.WaterlineEdges(far=far, near=near, dock=dock, dock_side=dock_side,
                            water_fraction=water, n_components=comps, components_kept=kept,
                            dominant_share=share)


def scene(*, pxx=30.0, pxy=24.0, width_ft=16.0, anchor_ft=25.0, dock_dir=(0.06, 1.0),
          vp=(-900.0, 250.0), origin=(1420.0, 320.0)):
    """A model-consistent scene: the three lines a true quad of this shape would produce."""
    p00 = np.asarray(origin, float)
    uy = unit(dock_dir)
    p01 = p00 + width_ft * pxy * uy
    vp = np.asarray(vp, float)
    at, _s = S.pixel_of(p00, p01, vp, anchor_ft=anchor_ft, width_ft=width_ft, px_per_ft_x=pxx)
    corners = {"00": p00, "10": at(anchor_ft, 0.0), "11": at(anchor_ft, width_ft), "01": p01}
    session = P.PlaneSession(frame_shape=(H, W_PX), length_ft=anchor_ft, width_ft=width_ft)
    for tag, pt in corners.items():
        session.corners[tag] = (float(pt[0]), float(pt[1]))
    far = edge("far", p00, vp - p00)
    near = edge("near", p01, vp - p01)
    dock = edge("dock", p00, uy)
    return p00, p01, corners, session, far, near, dock


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------
def test_projective_anchor_is_monotone_and_shrinks_with_distance():
    p00 = np.array([1400.0, 300.0])
    vp = np.array([-800.0, 300.0])
    a25, s25, _A = S.projective_anchor(p00, vp, 30.0, 25.0)
    a35, s35, _A2 = S.projective_anchor(p00, vp, 30.0, 35.0)
    assert 0.0 < s25 < s35 < 1.0                       # strictly short of the vanishing point
    assert np.linalg.norm(a35 - p00) > np.linalg.norm(a25 - p00)
    # px/ft at the dock edge is the requested 30, and it decays
    near_ft = np.linalg.norm(S.projective_anchor(p00, vp, 30.0, 1.0)[0] - p00)
    far_ft = np.linalg.norm(S.projective_anchor(p00, vp, 30.0, 26.0)[0]
                            - S.projective_anchor(p00, vp, 30.0, 25.0)[0])
    assert near_ft == pytest.approx(30.0, rel=0.02)
    assert far_ft < near_ft


def test_seed_recovers_a_model_consistent_quad_exactly():
    p00, p01, corners, session, far, near, dock = scene()
    res = S.seed_from_edges(edges3(far, near, dock), length_ft=25.0, width_ft=16.0,
                            frame_shape=(H, W_PX), current=session)
    assert not res.refused, res.reason
    assert res.level == "L3"
    for tag in S.CORNER_TAGS:
        got = np.asarray(res.corners[tag], float)
        assert np.linalg.norm(got - np.asarray(corners[tag], float)) < 1e-6, tag
    assert res.gates["S4"]["pass"]


def test_seed_refuses_when_the_two_long_lines_are_parallel():
    p00, p01, _c, session, far, near, dock = scene()
    parallel_near = edge("near", p01, far.fit["direction"])
    res = S.seed_from_edges(edges3(far, parallel_near, dock), length_ft=25.0, width_ft=16.0,
                            frame_shape=(H, W_PX), current=session)
    assert res.refused and res.corners is None
    assert "S3" in res.reason


def test_seed_refuses_a_mirrored_dock_side():
    """The vanishing point must sit on the far side of the origin from the dock.

    With the side flipped, X would increase *towards* the dock: every distance still positive, every
    one measured the wrong way.  This is the one geometric error a foreshortening gate cannot see.
    """
    _p00, _p01, _c, session, far, near, dock = scene()
    res = S.seed_from_edges(edges3(far, near, dock, dock_side="left"), length_ft=25.0,
                            width_ft=16.0, frame_shape=(H, W_PX), current=session)
    assert res.refused and res.corners is None
    assert "S3" in res.reason


def test_seed_refuses_a_quad_that_collapses():
    """A near line dragged onto the far line gives a degenerate rectangle: S3 (width) or S5 (area)."""
    p00, p01, _c, session, far, near, dock = scene()
    collapsed = edge("near", p01 - np.array([0.0, 300.0]), near.fit["direction"])
    res = S.seed_from_edges(edges3(far, collapsed, dock), length_ft=25.0, width_ft=16.0,
                            frame_shape=(H, W_PX), current=session)
    assert res.refused and res.corners is None
    assert "S3" in res.reason or "S5" in res.reason


def test_seed_refuses_a_mask_that_is_mostly_not_water():
    p00, p01, _c, session, far, near, dock = scene()
    res = S.seed_from_edges(edges3(far, near, dock, water=0.01), length_ft=25.0, width_ft=16.0,
                            frame_shape=(H, W_PX), current=session)
    assert res.refused and res.corners is None
    assert "S2" in res.reason


def test_seed_refuses_when_two_blobs_are_equally_large():
    p00, p01, _c, session, far, near, dock = scene()
    res = S.seed_from_edges(edges3(far, near, dock, share=0.4), length_ft=25.0, width_ft=16.0,
                            frame_shape=(H, W_PX), current=session)
    assert res.refused and "S2" in res.reason


def test_no_far_line_means_nothing_is_placed():
    p00, p01, _c, session, far, near, dock = scene()
    missing = W.EdgeFit(name="far", found=False, reason="no run")
    res = S.seed_from_edges(edges3(missing, near, dock), length_ft=25.0, width_ft=16.0,
                            frame_shape=(H, W_PX), current=session)
    assert res.refused and res.level == "none"
    assert res.corners is None and "far" in res.reason


def test_l2_takes_the_width_from_the_current_quad_and_says_so():
    p00, p01, corners, session, far, _near, dock = scene()
    res = S.seed_from_edges(edges3(far, W.EdgeFit(name="near", found=False),
                                   dock), length_ft=25.0, width_ft=16.0,
                            frame_shape=(H, W_PX), current=session)
    assert not res.refused, res.reason
    assert res.level == "L2"
    assert np.allclose(res.corners["01"], corners["01"], atol=1e-6)
    assert any("near waterline not found" in n for n in res.notes)


def test_l1_warns_that_the_origin_is_a_default():
    p00, p01, corners, session, far, near, _dock = scene()
    res = S.seed_from_edges(edges3(far, near, W.EdgeFit(name="dock", found=False)),
                            length_ft=25.0, width_ft=16.0, frame_shape=(H, W_PX),
                            current=session)
    assert not res.refused, res.reason
    assert res.level == "L1"
    assert res.gates["S6"]["pass"] and res.gates["S6"]["warn"]
    assert "S6" not in res.failing()


def test_l0_moves_the_quad_onto_the_far_line_only():
    p00, p01, corners, session, far, _near, _dock = scene()
    res = S.seed_from_edges(edges3(far, W.EdgeFit(name="near", found=False),
                                   W.EdgeFit(name="dock", found=False)),
                            length_ft=25.0, width_ft=16.0, frame_shape=(H, W_PX),
                            current=session)
    assert not res.refused, res.reason
    assert res.level == "L0"
    got = np.asarray(res.corners["00"], float)
    # corner 00 sits ON the detected far line (perpendicular distance ~0)
    d = unit(np.asarray(far.fit["direction"], float))
    n = np.array([-d[1], d[0]])
    assert abs(float((got - np.asarray(far.fit["point"], float)) @ n)) < 1e-6
    # and the quad kept its size (rotation + translation only)
    assert np.linalg.norm(np.asarray(res.corners["10"]) - got) == pytest.approx(
        np.linalg.norm(corners["10"] - corners["00"]), rel=1e-6)


def test_mirror_session_flips_x_and_keeps_the_tags_metric():
    session = P.PlaneSession(frame_shape=(H, W_PX), length_ft=25.0, width_ft=16.0)
    session.fit_to_frame()
    m = S.mirror_session(session)
    for tag in S.CORNER_TAGS:
        assert m.corners[tag][0] == pytest.approx(W_PX - 1 - session.corners[tag][0])
        assert m.corners[tag][1] == pytest.approx(session.corners[tag][1])
    # the metric origin moved to the other side — that is the point of mirroring, and it is why the
    # tags must NOT swap as well
    assert (m.corners["00"][0] < 0.5 * W_PX) != (session.corners["00"][0] < 0.5 * W_PX)
    assert S.quad_area(m.corners) == pytest.approx(S.quad_area(session.corners))
    gates = P.validate_plane({"homography": np.asarray(m.homography()).tolist()}, m)
    assert all(v["pass"] for v in gates.values())


def test_dock_hint_mirrors_only_when_the_dock_line_is_missing():
    """A detected dock line defines the side by itself; the hint must not overrule it."""
    p00, p01, _c, session, far, near, dock = scene()
    # L3: dock detected, hint claiming the opposite side -> no mirror note, seed still exact
    res = S.seed_from_edges(edges3(far, near, dock), length_ft=25.0, width_ft=16.0,
                            frame_shape=(H, W_PX), current=session, dock_hint_x=100.0)
    assert not res.refused
    assert not any("mirrored" in n for n in res.notes)

    # L1 with the same hint: the default quad assumed the dock on the left, the hint says right
    default = P.PlaneSession(frame_shape=(H, W_PX), length_ft=25.0, width_ft=16.0)
    default.fit_to_frame()
    res2 = S.seed_from_edges(edges3(far, near, W.EdgeFit(name="dock", found=False)),
                             length_ft=25.0, width_ft=16.0, frame_shape=(H, W_PX),
                             current=default, dock_hint_x=1700.0)
    assert any("mirrored" in n for n in res2.notes)
    assert res2.corners["00"][0] > 0.5 * W_PX          # origin moved to the dock (right) side


def test_the_gate_set_is_exactly_s1_to_s6():
    """Guards the circularity rule: no gate may compare the quad to the lines it came from."""
    p00, p01, _c, session, far, near, dock = scene()
    res = S.seed_from_edges(edges3(far, near, dock), length_ft=25.0, width_ft=16.0,
                            frame_shape=(H, W_PX), current=session)
    assert set(res.gates) == {"S1", "S2", "S3", "S4", "S5", "S6"}


# ---------------------------------------------------------------------------
# adapter: boundaries, runs, the calm window
# ---------------------------------------------------------------------------
def _synthetic_stack(top=120, bottom=220, right=260, n=40, h=240, w=320, seed=0):
    """Static everywhere except a water rectangle [top:bottom, 0:right] that flickers."""
    rng = np.random.default_rng(seed)
    spatial = rng.integers(0, 40, size=(h, w)).astype(np.float32)
    frames = np.repeat(spatial[None, :, :], n, axis=0)
    flicker = rng.normal(0.0, 18.0, size=(n, bottom - top, right))
    frames[:, top:bottom, :right] = frames[:, top:bottom, :right] + flicker
    return np.clip(frames, 0, 255).astype(np.uint8)


def test_column_boundaries_find_a_known_top_and_bottom():
    mask = np.zeros((240, 320), np.uint8)
    mask[120:220, 20:300] = 255
    top, bottom = W.column_boundaries(mask)
    assert top[:, 1].min() == top[:, 1].max() == 120
    assert bottom[:, 1].min() == bottom[:, 1].max() == 219
    assert top[:, 0].min() >= 20 and top[:, 0].max() <= 299


def test_boundaries_that_coincide_with_the_frame_edge_are_dropped():
    mask = np.zeros((240, 320), np.uint8)
    mask[120:220, 20:320] = 255                    # water runs into the right edge
    left, right = W.row_boundaries(mask)
    assert len(right) == 0                          # that boundary is a crop artefact
    assert len(left) == 100                         # the other one is real and must survive
    assert left[:, 0].min() == left[:, 0].max() == 20
    mask2 = np.zeros((240, 320), np.uint8)
    mask2[120:220, 20:300] = 255
    _left2, right2 = W.row_boundaries(mask2)
    assert right2[:, 0].min() == right2[:, 0].max() == 299


def test_split_runs_breaks_at_a_gap_and_not_at_jitter():
    xs = np.arange(0, 100, 1.0)
    ys = np.full_like(xs, 50.0) + np.tile([0.0, 0.3], 50)
    assert len(W.split_runs(np.stack([xs, ys], 1))) == 1
    xs2 = np.concatenate([xs[:40], xs[60:]])
    ys2 = np.full_like(xs2, 50.0)
    assert len(W.split_runs(np.stack([xs2, ys2], 1))) == 2


def test_fit_run_scores_inliers_and_rejects_a_bent_run():
    straight = np.stack([np.arange(0, 200, 1.0), np.full(200, 80.0)], 1)
    fit = W.fit_run(straight)
    assert fit is not None and fit["inlier_fraction"] == 1.0
    # a genuinely bent boundary (quadratic bow, like a waterline curving round a pool corner)
    bent = straight.copy()
    bent[:, 1] += 0.004 * (bent[:, 0] - 100.0) ** 2
    fit_bent = W.fit_run(bent)
    assert fit_bent is None or fit_bent["rms_residual"] > W.MAX_RMS_PX


def test_analyze_finds_all_three_edges_on_a_synthetic_stack():
    frames = _synthetic_stack()
    out = W.analyze(frames, scale=1.0, frame_size=(320, 240), window=(0, 40))
    assert out.far.found and out.near.found and out.dock.found
    assert out.far.midpoint is not None and abs(out.far.midpoint[1] - 120) <= 2
    assert out.near.midpoint is not None and abs(out.near.midpoint[1] - 219) <= 2
    assert out.dock_side == "right"
    assert out.dock.midpoint is not None and abs(out.dock.midpoint[0] - 259) <= 3
    assert out.level == "L3"
    assert out.far.stability_px is not None and out.far.stability_px < W.MAX_STABILITY_PX


def test_analyze_picks_a_method_and_reports_provenance():
    out = W.analyze(_synthetic_stack(), frame_size=(320, 240), window=(5, 35))
    assert out.method in W.wl.NOISE_METHODS
    assert out.fisher_by_method and out.fisher == pytest.approx(
        max(out.fisher_by_method.values()))
    d = out.as_dict()
    assert d["window"] == [5, 35] and d["frame_size"] == [320, 240]
    assert d["edges"]["far"]["found"] is True


def test_calm_window_prefers_the_stretch_clear_of_detections():
    start, end, why = W.calm_window(300, accepted_frames=list(range(40, 90)))
    assert start == 90 or start > 80          # after the blocked span
    assert end - start >= W.MIN_WINDOW
    assert "clear" in why


def test_calm_window_uses_the_opening_frames_when_nothing_is_clean():
    start, end, why = W.calm_window(80, accepted_frames=list(range(0, 80, 3)))
    assert start == 0 and end <= 80
    assert "check the noise map" in why


def test_calm_window_caps_a_long_clean_clip():
    start, end, _why = W.calm_window(2000, accepted_frames=None)
    assert (start, end) == (0, W.MAX_WINDOW)


# ---------------------------------------------------------------------------
# provenance in the document
# ---------------------------------------------------------------------------
def _doc(session):
    doc = session.build("seed_probe")
    return doc


def test_seed_block_reaches_the_document_and_flips_on_the_first_edit():
    p00, p01, corners, session, far, near, dock = scene()
    res = S.seed_from_edges(edges3(far, near, dock), length_ft=25.0, width_ft=16.0,
                            frame_shape=(H, W_PX), current=session)
    block = S.build_block(res, edges3(far, near, dock), upstream=W.UPSTREAM,
                          file_sha256=W.vendored_sha256())
    fresh = P.PlaneSession(frame_shape=(H, W_PX), length_ft=25.0, width_ft=16.0)
    fresh.apply_seed(res.corners, block)

    doc = _doc(fresh)
    seeded = doc["plane"]["waterline_seed"]
    assert seeded["placement"] == "auto"
    assert seeded["seed"]["level"] == "L3"
    assert seeded["detection"]["edges"]["far"]["found"] is True
    assert len(seeded["file_sha256"]) == 64

    fresh.select("right")
    fresh.nudge(1.0, 0.0)
    assert fresh.seed["placement"] == "auto+edited"
    assert _doc(fresh)["plane"]["waterline_seed"]["placement"] == "auto+edited"


def test_a_hand_placed_plane_document_has_no_seed_block():
    session = P.PlaneSession(frame_shape=(H, W_PX), length_ft=25.0, width_ft=16.0)
    session.fit_to_frame()
    assert "waterline_seed" not in session.as_dict()
    assert "waterline_seed" not in _doc(session)["plane"]


def test_the_vendored_detector_hash_matches_its_manifest():
    manifest = (Path(W.VENDORED_FILE).parent / "SHA256SUMS").read_text(encoding="utf-8")
    assert W.vendored_sha256() in manifest
    assert W.UPSTREAM.endswith("Waterline-Detection@e0c412607ea92d557da127b1ac72f791f23efb96")


def test_a_seeded_plane_measures_through_its_own_homography():
    """The seed must not touch the measurement path: same corners -> same feet."""
    p00, p01, corners, session, far, near, dock = scene()
    res = S.seed_from_edges(edges3(far, near, dock), length_ft=25.0, width_ft=16.0,
                            frame_shape=(H, W_PX), current=session)
    manual = P.PlaneSession(frame_shape=(H, W_PX), length_ft=25.0, width_ft=16.0)
    for tag, pt in res.corners.items():
        manual.set_corner(tag, *pt)
    seeded = P.PlaneSession(frame_shape=(H, W_PX), length_ft=25.0, width_ft=16.0)
    seeded.apply_seed(res.corners, {"placement": "auto"})
    for tag in S.CORNER_TAGS:
        assert np.allclose(seeded.corners[tag], manual.corners[tag])
    probe = (960.0, 400.0)
    assert seeded.world_at(*probe) == pytest.approx(manual.world_at(*probe))
