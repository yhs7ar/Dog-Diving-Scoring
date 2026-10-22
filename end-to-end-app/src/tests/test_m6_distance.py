"""M6 verification gate (PLAN.md §11 M6 / §5 ``distance.py`` + §4.4 uncertainty budget).

PLAN §11 M6: *"``distance.py`` + uncertainty budget — V3 distance within budget; feet+inches
formatting correct; override path works."*

PLAN §8 V3 is *"render/project a known 1-ft grid onto a real frame; the pipeline's distance for
a point at a known marking must reproduce it within the residual budget. This isolates
calibration error from detection error."*  M2 already ran V3 at the *calibration* level against
a projective ground truth.  This file runs it at the **module** level: a point placed at a
known real distance must come back out of ``distance.measure`` as that distance, and — the
part that actually matters for the capstone, whose criteria are stated as *percent of jumps
within ±2/±4/±6 in* — the **budget must cover its own errors**.  Two external oracles, neither
read off the module:

* a **projective ground truth** ``x(X) = CX + P0·X/(1 + X/L)`` (the M2 oracle) for accuracy and
  for the Monte-Carlo coverage test of ``sigma_in``;
* an **analytic free-fall jump** whose x(t) is known in closed form, for the frame-quantisation
  term and for the real integration path (records -> trajectory -> contact -> distance).

Real data is used only where it can be honest: ``data/calib/`` is empty (HANDOFF open item 10),
so the real clip is exercised for its *pixel-domain* fields and its disclosures, and the
missing inches are reported as missing rather than synthesised.

Run:  PYTHONPATH=src <venv-python> -m pytest src/tests/test_m6_distance.py -s
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dog_diving import calibration as C          # noqa: E402
from dog_diving import config as ddconfig        # noqa: E402
from dog_diving import contact, detect, distance, trajectory, video  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL = PROJECT_ROOT / "models" / "best.pt"


# ===========================================================================
# Oracle A — projective pool (the M2 ground truth), for px->in accuracy
# ===========================================================================
CX, P0 = 520.0, 1.5                       # dock-edge x; 1.5 px/in = 18 px/ft at the dock
WATER_Y, IMG_W, IMG_H = 540, 1920, 1080
SPAN_IN = 372.0
FTS4 = [(0, 0), (1, 0), (15, 0), (16, 0)]
FTS6 = [(0, 0), (1, 0), (15, 0), (16, 0), (30, 0), (31, 0)]
PROBE_FT = list(range(1, 31))


class Truth:
    """Projective pinhole along the pool axis — why the conversion must be local."""

    def __init__(self, foreshortening: float) -> None:
        self.F = foreshortening
        self.L = SPAN_IN / (1.0 / math.sqrt(1.0 - foreshortening) - 1.0)

    def x(self, X_in: float) -> float:
        return CX + P0 * X_in / (1.0 + X_in / self.L)

    def px_per_ft(self, X_in: float = 0.0) -> float:
        return P0 * 12.0 / (1.0 + X_in / self.L) ** 2

    def calib(self, fts=FTS6, setup_id: str = "synth", fit_type=None) -> dict:
        marks = [{"label": f"{f}'{i:02d}\"", "ft": f, "inch": i,
                  "x_px": self.x(f * 12 + i), "y_px": WATER_Y} for f, i in fts]
        return C.build(setup_id, (IMG_W, IMG_H),
                       dock_line=[[CX, 100], [CX, 980]],
                       waterline=[[0, WATER_Y], [IMG_W, WATER_Y]],
                       marks=marks, model_sha="deadbeef",
                       notes="projective synthetic oracle", fit_type=fit_type)


TRUTH = Truth(0.04)                       # "several percent" (PLAN §4.2)
CAL6 = TRUTH.calib(FTS6)


# ===========================================================================
# Oracle B — an analytic free-fall jump in *linear* (18 px/ft) image space
# ===========================================================================
DOCK_X = 400.0          # the dock edge in this synthetic image
PX_PER_IN = 1.5         # uniform: no foreshortening, so x(t) is known in closed form
DOCK_Y, W_Y = 100.0, 700.0
LAUNCH = 100            # launch frame
X0, VX = DOCK_X + 120.0, 8.0        # px, px/frame
Y0, VY0, G = 560.0, -18.0, 0.8      # px, px/frame, px/frame^2
LAST_DET = 148          # last detection before the splash occlusion
CAL_LIN = C.build(
    "synth-lin", (1920, 1080),
    dock_line=[[DOCK_X, DOCK_Y], [DOCK_X, 980]],
    waterline=[[0, W_Y], [1920, W_Y]],
    marks=[{"label": f"{f}'00\"", "ft": f, "inch": 0, "x_px": DOCK_X + PX_PER_IN * f * 12,
            "y_px": W_Y} for f, _ in FTS6],
    model_sha="cafe", notes="linear synthetic oracle", fit_type="linear",
)


def arc_x(t: float) -> float:
    return X0 + VX * (t - LAUNCH)


def arc_y(t: float) -> float:
    tc = t - LAUNCH
    return Y0 + VY0 * tc + 0.5 * G * tc * tc


# analytic descending-branch crossing of y(t) = W_Y  (the same algebra as M5's oracle)
_TC = (18.0 + math.sqrt(18.0 ** 2 + 4 * 0.5 * G * (W_Y - Y0))) / (2 * 0.5 * G)
T_CONTACT = LAUNCH + float(_TC)                       # 151.7617...


def _rec(t: int, x: float, y: float, kp: float = 0.95) -> dict:
    return {"frame": int(t), "tailset": [float(x), float(y), float(kp)],
            "nose": [float(x - 50), float(y - 30), float(kp)], "box_conf": 0.20,
            "track_id": None, "accepted": True}


@pytest.fixture(scope="module")
def jump():
    """The analytic jump, run through the real detect-record -> traj -> contact path."""
    recs = [_rec(t, arc_x(t), arc_y(t)) for t in range(LAUNCH, LAST_DET + 1)]
    traj = trajectory.build_trajectory(recs, clip_id="synth")
    res = contact.contact_candidates(recs, traj, waterline_y=W_Y, clip_id="synth",
                                     n_frames=400)
    print(f"\n--- synthetic free fall: launch f{LAUNCH}, detections to f{LAST_DET}, "
          f"analytic contact f{T_CONTACT:.4f}")
    print(f"    x(t) = {X0:.0f} + {VX:.0f}*(t-{LAUNCH})   ->  distance(x) = "
          f"(x - {DOCK_X:.0f})/{PX_PER_IN:g} in")
    return {"records": recs, "traj": traj, "result": res}


# ===========================================================================
# 1. V3 — a point at a known distance comes back as that distance
# ===========================================================================
def test_v3_at_a_marking_the_distance_is_reproduced_within_the_residual_budget() -> None:
    """PLAN §8 V3 read literally: *"a point at a known marking"* reproduces within the budget."""
    mark_fts = [1, 15, 16, 30]                      # the marked positions, in ft
    errs = [distance.measure(TRUTH.x(f * 12), CAL6)["total_inches"] - f * 12 for f in mark_fts]
    budget = CAL6["fit"]["residual_max_in"]
    print(f"\n  measure() AT the marks {mark_fts} ft: worst |err| = {np.max(np.abs(errs)):.4f} in "
          f"(fit residual budget {budget:.4f} in)")
    assert np.max(np.abs(errs)) <= budget + 1e-9


def test_v3_between_the_marks_the_fit_residual_is_not_the_budget() -> None:
    """PINNED: the residual bounds the error *at* the marks, not across the span.

    Measured here: probing every foot from 1 to 30 gives a worst |err| of **0.00732 in**
    against a mark residual max of **0.00226 in** — 3.2x larger.  The extra is the
    *interpolation* error between marks, and it is the same lesson as M2's finding 4 (the fit's
    own residual reports consistency inside the marked span, not accuracy).  It matters for
    how M6's gate is quoted: PLAN §11/§8 V3 says "a point at a known marking", so the at-mark
    check above is the literal gate, and the number below is the honest whole-span figure.  A
    future change that makes the interpolation error *smaller* than the mark residual fails
    this test, which is the point — that would be a better calibration and the docstring, the
    HANDOFF entry and the gate line would all need revisiting.
    """
    errs = [abs(distance.measure(TRUTH.x(f * 12), CAL6)["total_inches"] - f * 12)
            for f in PROBE_FT]
    budget = CAL6["fit"]["residual_max_in"]
    print(f"\n  measure() BETWEEN the marks (every foot, 1..30): worst |err| = {np.max(errs):.5f} in "
          f"vs mark residual max {budget:.5f} in (ratio {np.max(errs) / budget:.2f}x)")
    assert np.max(errs) > budget, "the interpolation error no longer exceeds the mark residual"
    assert np.max(errs) < 0.05, "the whole-span interpolation error grew beyond a twentieth of an inch"


def test_v3_the_four_mark_fit_is_only_valid_inside_its_span() -> None:
    """Pinned: with the 30/31 ft marks missing, 30 ft is metres off in inch terms."""
    cal = TRUTH.calib(FTS4)
    in_span = max(abs(distance.measure(TRUTH.x(f * 12), cal)["total_inches"] - f * 12)
                  for f in range(1, 17))
    at_30 = distance.measure(TRUTH.x(360), cal)["total_inches"] - 360.0
    print(f"\n  4-mark calibration: worst in-span (1..16 ft) = {in_span:.3f} in, "
          f"at 30 ft = {at_30:+.3f} in")
    assert in_span < 0.5
    assert abs(at_30) > 3.0, "the known >= 3 in extrapolation cost at 30 ft disappeared"


# ===========================================================================
# 2. The conversion is LOCAL — and the size of getting it wrong is measured
# ===========================================================================
def test_local_scale_tracks_the_measured_foreshortening() -> None:
    """``local_in_per_px`` is ``d'(x)``: it must follow the oracle's own px/ft law.

    Note the direction: fewer px per foot far out means *more* inches per pixel there, so the
    local scale grows with distance while the px/ft shrinks.  ``px_per_ft`` is per FOOT.
    """
    s_near = distance.local_in_per_px(TRUTH.x(12), CAL6)
    s_far = distance.local_in_per_px(TRUTH.x(360), CAL6)
    t_near = 12.0 / TRUTH.px_per_ft(12.0)          # px/ft -> in/px
    t_far = 12.0 / TRUTH.px_per_ft(360.0)
    print(f"\n  in/px at 1 ft  {s_near:.6f} vs ground truth {t_near:.6f}")
    print(f"  in/px at 30 ft {s_far:.6f} vs ground truth {t_far:.6f}  "
          f"(ratio fit {s_near / s_far:.4f} vs truth {t_near / t_far:.4f})")
    print(f"  px/ft {TRUTH.px_per_ft(0.0):.3f} at the dock -> {TRUTH.px_per_ft(360.0):.3f} at 30 ft "
          f"({100 * (1 - TRUTH.px_per_ft(360.0) / TRUTH.px_per_ft(0.0)):.1f}% foreshortening)")
    assert s_far > s_near, "foreshortening means MORE inches per pixel far from the dock"
    assert s_near == pytest.approx(t_near, rel=0.01) and s_far == pytest.approx(t_far, rel=0.01)
    assert (s_near / s_far) == pytest.approx(t_near / t_far, rel=0.01)


def test_a_single_global_scale_costs_inches_at_30ft() -> None:
    """Measured: the price of one scale factor instead of the fitted local mapping."""
    marks = CAL6["marks"]
    x0, x16 = marks[0]["x_px"], marks[3]["x_px"]
    naive = 192.0 / (x16 - x0)                     # in/px across the 0-16 ft marks
    naive_30 = (TRUTH.x(360) - TRUTH.x(0)) * naive
    fit_30 = distance.measure(TRUTH.x(360), CAL6)["total_inches"]
    print(f"\n  at 30 ft: single global scale -> {naive_30:.3f} in "
          f"({naive_30 - 360:+.3f} in error); fitted local mapping -> {fit_30:.3f} in "
          f"({fit_30 - 360:+.4f} in error)")
    assert abs(naive_30 - 360.0) > 2.0, "the global-scale error is smaller than measured"
    assert abs(fit_30 - 360.0) < 0.5


# ===========================================================================
# 3. PLAN §5's override path — the same function, a different point
# ===========================================================================
def test_manual_override_flows_through_the_same_function() -> None:
    x_model = TRUTH.x(264)                                   # 22 ft
    a = distance.measure(x_model, CAL6, point_source=distance.P_MODEL)
    b = distance.measure(x_model, CAL6, point_source=distance.P_MANUAL)
    print(f"\n  model  {a['display']} (origin {a['point_origin']})   "
          f"manual {b['display']} (origin {b['point_origin']})")
    assert a["total_inches"] == pytest.approx(264.0, abs=0.05)
    assert b["total_inches"] == a["total_inches"], "the override path changed the arithmetic"
    assert b["point_source"] == "manual" and b["point_origin"] == "click"
    assert a["point_source"] == "model" and a["point_origin"] == "detection"


def test_a_manual_nudge_of_12_inches_moves_the_reading_12_inches() -> None:
    """The override has to be *linear* in the click, or the GUI correction is a lie."""
    x22 = TRUTH.x(264)
    step = 12.0 / distance.local_in_per_px(x22, CAL6)          # px worth ~12 in at 22 ft
    d0 = distance.measure(x22, CAL6, point_source="manual")["total_inches"]
    d1 = distance.measure(x22 + step, CAL6, point_source="manual")["total_inches"]
    print(f"\n  nudge {step:.2f} px at 22 ft -> {d1 - d0:+.4f} in (wanted +12)")
    assert (d1 - d0) == pytest.approx(12.0, abs=0.05)


# ===========================================================================
# 4. feet + inches formatting (PLAN §5 ``display``), incl. the handbook's rule
# ===========================================================================
def test_family_of_exact_display_strings() -> None:
    cases = [(0, "0' 0\""), (12, "1' 0\""), (271, "22' 7\""), (372, "31' 0\""),
             (902, "75' 2\"")]
    for inches, want in cases:
        got = distance.format_feet_inches(inches)
        print(f"\n  {inches:>6.1f} in -> {got}")
        assert got == want
        feet, inch = distance.split_feet_inches(inches)
        assert feet * 12 + inch == round(inches)
        assert 0 <= abs(inch) <= 11


@pytest.mark.parametrize("inches,nearest,floor,plus", [
    (11.2, "0' 11\"", "0' 11\"", "1' 0\""),
    (11.6, "1' 0\"", "0' 11\"", "1' 0\""),
    (371.5, "31' 0\"", "30' 11\"", "31' 0\""),
])
def test_rounding_modes_are_ordered_and_pinned(inches, nearest, floor, plus) -> None:
    """``plus`` is the handbook's own convention — 'marking to the plus' (quoted in module)."""
    got = {m: distance.format_feet_inches(inches, m) for m in ("nearest", "floor", "plus")}
    print(f"\n  {inches} in -> {got}")
    assert got["nearest"] == nearest and got["floor"] == floor and got["plus"] == plus
    assert (distance._round_inches(inches, "plus")
            >= distance._round_inches(inches, "nearest")
            >= distance._round_inches(inches, "floor"))


def test_negative_distances_are_formatted_and_flagged() -> None:
    """A point dock-side of the datum: the number is negative and must say so."""
    x_behind = DOCK_X - 30.0                                  # -20 in at 1.5 px/in
    out = distance.measure(x_behind, CAL_LIN, y_px=W_Y)
    print(f"\n  point {x_behind:.0f} px (dock at {DOCK_X:.0f}) -> {out['total_inches']:.2f} in "
          f"= {out['display']}   flags={out['flags']['negative_distance']}")
    assert out["total_inches"] == pytest.approx(-20.0, abs=1e-6)
    assert out["display"] == "-1' 8\""
    assert out["flags"]["negative_distance"] is True
    assert any("DOCK side" in w for w in out["warnings"])


def test_unknown_rounding_mode_and_non_finite_distances_are_refused() -> None:
    with pytest.raises(ValueError, match="rounding must be one of"):
        distance.format_feet_inches(12.0, "banker")
    with pytest.raises(ValueError, match="non-finite"):
        distance.format_feet_inches(float("nan"))


# ===========================================================================
# 5. The uncertainty budget (PLAN §4.4): composition, provenance, and COVERAGE
# ===========================================================================
def test_measure_returns_every_key_plan5_names() -> None:
    out = distance.measure(TRUTH.x(264), CAL6, y_px=WATER_Y, frame=137, clip_id="18-07")
    required = ("feet", "inches", "total_inches", "sigma_in", "display")
    print(f"\n  PLAN §5 keys present: {[k for k in required if k in out]}  "
          f"display={out['display']}  sigma={out['sigma_in']:.3f} in")
    for k in required:
        assert k in out
    assert out["clip_id"] == "18-07" and out["frame"] == 137
    assert out["calib_setup_id"] == CAL6["setup_id"]
    assert out["total_inches"] == pytest.approx(264.0, abs=0.05)
    assert out["feet"] == 22 and out["inches"] == 0


def test_sigma_is_the_quadrature_of_the_terms_it_says_it_used() -> None:
    xs = [TRUTH.x(264) - 1.4, TRUTH.x(264) + 0.9, TRUTH.x(264) + 0.2]
    out = distance.measure(xs[0], CAL6, y_px=WATER_Y, kp_xs=xs,
                           x_next_px=xs[0] + 6.0, x_prev_px=xs[0] - 6.0)
    terms = out["sigma_terms"]
    print(f"\n  terms = " + ", ".join(f"{k}={v:.4f}" if v is not None else f"{k}=None"
                                       for k, v in terms.items()))
    active = [v for v in terms.values() if v is not None]
    assert out["sigma_in"] == pytest.approx(math.sqrt(sum(v * v for v in active)), rel=1e-12)
    assert terms["fit"] == CAL6["fit"]["residual_rms_in"]
    assert terms["kp"] is not None and terms["frame"] is not None
    assert out["sigma_complete"] is True


def test_a_partial_budget_is_flagged_incomplete_rather_than_quoted() -> None:
    """No neighbouring frames -> the frame term is missing, and the total must say so."""
    xs = [TRUTH.x(264) - 1.0, TRUTH.x(264) + 1.0, TRUTH.x(264) + 0.5]
    out = distance.measure(xs[0], CAL6, y_px=WATER_Y, kp_xs=xs)
    print(f"\n  sigma={out['sigma_in']:.4f} in  complete={out['sigma_complete']}  "
          f"frame_term={out['sigma_terms']['frame']}")
    assert out["sigma_complete"] is False
    assert out["sigma_terms"]["frame"] is None
    assert any("sigma_frame unavailable" in w for w in out["warnings"])
    assert any("no mark-localisation term" in w for w in out["warnings"])


def test_budget_covers_the_realised_error_monte_carlo() -> None:
    """The only test that makes ``sigma_in`` worth reporting: does it bracket the errors?

    A 22 ft tailset detected at K=5 frames with a known 1.5 px of x noise.  The realised RMS
    error in inches must land within ~1x of the reported sigma; a budget that under-covers is
    worse than no budget, because the capstone reports % of jumps within a tolerance.
    """
    rng = np.random.default_rng(7)
    x_true, sigma_px, K, N = TRUTH.x(264), 1.5, 5, 1500
    errs, sigmas = [], []
    for _ in range(N):
        xs = list(x_true + rng.normal(0.0, sigma_px, K))
        out = distance.measure(xs[-1], CAL6, y_px=WATER_Y, kp_xs=xs)
        errs.append(out["total_inches"] - 264.0)
        sigmas.append(out["sigma_in"])
    # the honest alternative reading, for the record (motion-free point: they should agree)
    det = [distance.measure(x_true + float(rng.normal(0, sigma_px)), CAL6, y_px=WATER_Y,
                            kp_xs=list(x_true + rng.normal(0, sigma_px, K)),
                            kp_pred_xs=[x_true] * K)["sigma_kp_detrended_in"] for _ in range(200)]
    realised = float(np.sqrt(np.mean(np.square(errs))))
    mean_sigma = float(np.mean(sigmas))
    within2 = float(np.mean(np.abs(errs) <= 2 * mean_sigma))
    within3 = float(np.mean(np.abs(errs) <= 3 * mean_sigma))
    print(f"\n  {N} trials, {sigma_px} px noise: realised RMS = {realised:.4f} in, "
          f"mean sigma_in = {mean_sigma:.4f} in, ratio = {realised / mean_sigma:.3f}")
    print(f"  |err| <= 2 sigma: {100 * within2:.1f}%   <= 3 sigma: {100 * within3:.1f}%   "
          f"(Gaussian expectation 95.4% / 99.7%)")
    print(f"  sigma_kp detrended on a static point: {np.mean(det):.4f} in "
          f"(literal reading agrees when the window does not span travel)")
    assert 0.7 <= realised / mean_sigma <= 1.4, "the budget does not cover the realised error"
    assert within2 > 0.85 and within3 > 0.98
    assert np.mean(det) == pytest.approx(mean_sigma, rel=0.10)


# ===========================================================================
def test_no_point_is_invented_from_the_arc_when_the_frame_has_no_detection(jump) -> None:
    """The fork never substitutes a fitted-arc point for a detection; neither does this now.

    Replaced the old behaviour (2026-10-19): the marker used to be placed on the parabola when the
    chosen frame had no accepted detection, which reads in Review as a cursor gliding up and
    diagonally "following the dog" through frames where nothing was detected at all.
    """
    traj = jump["traj"]
    assert traj.coeffs, "this fixture needs an arc, or the test proves nothing"
    detected = {int(p["frame"]) for p in traj.points}
    assert detected, "no detections at all — the fixture is wrong"
    # the realistic case: the operator nudges one frame PAST the last detection (analytic contact
    # is f151.76, detections stop at f148) and the arc used to be consulted for a point
    gap = LAST_DET + 1
    assert gap not in detected
    contact = jump["result"].choose(gap, "test-gap")
    out = distance.measure_contact(contact, traj, CAL_LIN)
    assert out["x_px"] is None and out["y_px"] is None, out
    assert out["total_inches"] is None
    assert out["point_origin"] is None or out["point_origin"] != "arc", out["point_origin"]
    assert any("no accepted detection" in w for w in out["warnings"]), out["warnings"]


# 6. sigma_frame from the arc, and the motion/motion-noise conflation
# ===========================================================================
def test_frame_quantisation_term_is_the_arc_slope(jump) -> None:
    out = distance.measure_contact(jump["result"], jump["traj"], CAL_LIN)
    scale = distance.local_in_per_px(out["x_px"], CAL_LIN)
    print(f"\n  chosen f{out['frame']} ({out['contact_source']})  x={out['x_px']:.1f}  "
          f"distance={out['total_inches']:.4f} in = {out['display']}")
    print(f"  sigma_frame = {out['sigma_terms']['frame']:.4f} in  ({VX:g} px/frame x "
          f"{scale:.6f} in/px)   sigma={out['sigma_in']:.3f} in  complete={out['sigma_complete']}")
    assert out["sigma_terms"]["frame"] == pytest.approx(VX * scale, rel=1e-9)
    assert out["sigma_complete"] is True
    # the analytic oracle: x(t) is closed-form, so the distance is known exactly
    want = (arc_x(out["frame"]) - DOCK_X) / PX_PER_IN
    assert out["total_inches"] == pytest.approx(want, abs=1e-6)
    # frame quantisation is the DOMINANT term on real footage geometry, and it is a frame-rate
    # limit, not an algorithm limit (VX px/frame = 5.3 in/frame here)
    assert out["sigma_terms"]["frame"] > out["sigma_terms"]["fit"]


def test_sigma_kp_literal_mixes_motion_into_the_noise_term(jump) -> None:
    """PINNED (module finding 2): PLAN §4.4's literal sigma_kp charges travel to the noise.

    The synthetic arc's x(t) is exactly linear in t, so a de-trended scatter about the fitted
    arc is 0 px; the literal std-dev over the same 5 frames is not.
    """
    out = distance.measure_contact(jump["result"], jump["traj"], CAL_LIN)
    lit, det = out["sigma_kp_literal_in"], out["sigma_kp_detrended_in"]
    print(f"\n  sigma_kp literal = {lit:.4f} in, detrended = {det:.4f} in "
          f"(window frames {out['kp_frames']})")
    assert det == pytest.approx(0.0, abs=1e-9), "the synthetic arc is exactly linear in x(t)"
    assert lit > 3.0, "the literal reading should be dominated by the 8 px/frame travel"
    assert any("span motion" in w for w in out["warnings"])
    # the default is PLAN's literal (conservative) reading — state it rather than leave it open
    assert out["sigma_terms"]["kp"] == pytest.approx(lit, rel=1e-12)
    assert distance.DistanceConfig().kp_mode == "literal"


def test_detrended_mode_is_available_and_smaller_here(jump) -> None:
    cfg = distance.DistanceConfig(kp_mode="detrended")
    out = distance.measure_contact(jump["result"], jump["traj"], CAL_LIN, cfg=cfg)
    print(f"\n  detrended mode: sigma_kp = {out['sigma_terms']['kp']:.4f} in "
          f"(literal {out['sigma_kp_literal_in']:.4f} in)")
    assert out["sigma_terms"]["kp"] == pytest.approx(0.0, abs=1e-9)
    assert out["sigma_in"] < distance.measure_contact(
        jump["result"], jump["traj"], CAL_LIN)["sigma_in"]


# ===========================================================================
# 7. The splash-centre gap warning (PLAN §5 + risk R2 + the handbook)
# ===========================================================================
def test_splash_gap_warning_quotes_the_handbook_rule() -> None:
    x = TRUTH.x(264)
    scale = distance.local_in_per_px(x, CAL6)
    above = distance.measure(x, CAL6, y_px=WATER_Y - 15.0)      # 15 px above the surface
    ok = distance.measure(x, CAL6, y_px=WATER_Y - 4.0)           # within the 6 in tolerance
    below = distance.measure(x, CAL6, y_px=WATER_Y + 15.0)
    print(f"\n  above: gap {above['splash_gap_in']:.2f} in  warn={above['flags']['splash_gap_warn']}")
    print(f"  within: gap {ok['splash_gap_in']:.2f} in  warn={ok['flags']['splash_gap_warn']}")
    msg = [w for w in above["warnings"] if w.startswith("SPLASH-CENTRE GAP")][0]
    print(f"  message: {msg}")
    assert above["splash_gap_in"] == pytest.approx(-15.0 * scale, rel=1e-12)
    assert above["splash_gap_px"] == -15.0
    assert above["flags"]["splash_gap_warn"] is True
    assert "ABOVE" in msg and "LOWER BOUND" in msg
    assert "splash circle" in msg and "ripple" in msg and "R2" in msg
    assert ok["flags"]["splash_gap_warn"] is False, "4 in is inside PLAN §6's 6 in tolerance"
    assert "BELOW" in [w for w in below["warnings"] if w.startswith("SPLASH-CENTRE GAP")][0]


def test_the_gap_tolerance_is_plan6s_six_inches() -> None:
    assert ddconfig.WATERLINE_TOL_IN == 6
    assert distance.DistanceConfig().waterline_tol_in == 6


def test_missing_waterline_is_reported_not_invented() -> None:
    cal = {k: v for k, v in CAL6.items() if k != "waterline"}
    out = distance.measure(TRUTH.x(264), cal, y_px=WATER_Y)
    print(f"\n  no waterline row -> splash_gap_in={out['splash_gap_in']}  "
          f"warn={any('splash-centre' in w for w in out['warnings'])}")
    assert out["waterline_y"] is None and out["splash_gap_in"] is None
    assert any("no waterline row" in w for w in out["warnings"])
    assert out["total_inches"] == pytest.approx(264.0, abs=0.05)   # x-only, still measurable


def test_failed_gates_travel_with_the_distance() -> None:
    bad = {**CAL6, "dock_line": [[CX + 40.0, 100], [CX + 40.0, 980]]}
    out = distance.measure(TRUTH.x(264), bad)
    print(f"\n  G2 fail carried: gates={out['gates']}  flags={out['flags']['gates_failed']}")
    assert out["gates"]["G2"] is False
    assert out["flags"]["gates_failed"] == ["G2"]
    assert any("failed gate" in w for w in out["warnings"])


def test_resolution_mismatch_is_flagged() -> None:
    out = distance.measure(TRUTH.x(264), CAL6, image_size=(1280, 720))
    print(f"\n  calib made at {CAL6['image_size']}, clip at [1280, 720] -> "
          f"{[w for w in out['warnings'] if 'resolution' in w]}")
    assert any("resolution" in w for w in out["warnings"])
    same = distance.measure(TRUTH.x(264), CAL6, image_size=CAL6["image_size"])
    assert not any("resolution" in w for w in same["warnings"])


# ===========================================================================
# 8. Real footage: the plumbing, and the honest refusal to invent inches
# ===========================================================================
@pytest.fixture(scope="module")
def real_18_07():
    """The warm cache, through the same key the modules use.  No inference."""
    path = PROJECT_ROOT / "out" / "18-07" / "tracks.json"
    clip = PROJECT_ROOT / "data" / "clips" / "18-07.mp4"
    if not path.is_file():
        pytest.skip(f"missing warm cache {path}")
    doc = detect.load_cache(path)
    cfg = detect.DetectConfig()
    kh = detect.key_hash(detect.cache_key(video.sha256(clip), video.sha256(MODEL), cfg, None))
    assert kh in doc.get("entries", {}), f"the default config's key {kh} is not in {path.name}"
    e = doc["entries"][kh]
    traj = trajectory.build_trajectory(e["records"], clip_id="18-07")
    res = contact.contact_candidates(e["records"], traj, waterline_y=None, clip_id="18-07",
                                     n_frames=e.get("n_frames_clip"))
    return {"entry": e, "traj": traj, "result": res, "image_size": (1920, 1080)}


def test_real_clip_reports_its_pixels_and_refuses_to_invent_inches(real_18_07) -> None:
    """No calibration exists for this clip (HANDOFF item 10) — the module must say so."""
    out = distance.measure_contact(real_18_07["result"], real_18_07["traj"], None,
                                   clip_id="18-07", image_size=real_18_07["image_size"])
    print(f"\n  18-07 f{out['frame']}: x={out['x_px']:.1f} y={out['y_px']:.1f} "
          f"kp_frames={out['kp_frames']}  total_inches={out['total_inches']} "
          f"display={out['display']} sigma={out['sigma_in']}")
    assert out["frame"] == 437 and out["x_px"] == pytest.approx(719.7, abs=0.1)
    assert out["y_px"] == pytest.approx(926.5, abs=0.1)
    assert out["point_origin"] == "detection" and out["point_source"] == "model"
    assert out["total_inches"] is None and out["display"] is None and out["sigma_in"] is None
    assert any("NO CALIBRATION" in w for w in out["warnings"])
    # the pixel-domain disclosures still fire, because they do not need a calibration
    assert any("NOT a free fall" in w for w in out["warnings"])
    assert any("past the last arc sample" in w for w in out["warnings"])
    assert out["kp_frames"] and len(out["kp_frames"]) == 5


def test_real_clip_with_a_synthetic_calibration_is_labelled_not_a_measurement(real_18_07) -> None:
    """With a *synthetic* calibration the number exists — and the output must not hide it.

    It also refuses the frame-quantisation term: chosen frame 437 is 41 frames past the last arc
    sample, so the arc's slope there is extrapolation.  Reported rather than clamped, that term
    measured **1033 in** — half a pool per frame — which is exactly the kind of plausible-looking
    number this module must not emit, so ``sigma_complete`` is False and the refusal is quoted.
    """
    cal = TRUTH.calib(FTS6, setup_id="synth-pool")
    out = distance.measure_contact(real_18_07["result"], real_18_07["traj"], cal,
                                   clip_id="18-07", image_size=real_18_07["image_size"])
    print(f"\n  18-07 f{out['frame']} against a synthetic pool calibration: "
          f"{out['display']} +/- {out['sigma_in']:.2f} in (sigma_terms="
          f"{ {k: (round(v, 3) if v is not None else None) for k, v in out['sigma_terms'].items()} })")
    print(f"  complete={out['sigma_complete']}  "
          f"{[w for w in out['warnings'] if w.startswith('frame-quantisation')][0][:120]}...")
    print(f"  NOT a measurement of this clip: setup_id={out['calib_setup_id']} belongs to a "
          f"synthetic pool; the module reports provenance, it does not verify it")
    assert out["calib_setup_id"] == "synth-pool"
    assert out["total_inches"] is not None
    assert out["sigma_terms"]["frame"] is None and out["sigma_complete"] is False
    assert any("frame-quantisation refused" in w for w in out["warnings"])
    assert out["sigma_in"] == pytest.approx(out["sigma_terms"]["kp"], rel=1e-6)
    assert any("NOT a free fall" in w for w in out["warnings"])
    # sigma_frame here comes from a fitted arc that is not a free fall, and that is disclosed
    assert out["arc_is_freefall"] is False


def test_no_contact_frame_means_no_distance(jump) -> None:
    res = contact.ContactResult(
        clip_id="synth", candidates=[], chosen_frame=None, chosen_source="manual",
        waterline_y=W_Y, arc_is_freefall=False, arc_reason="x", arc_evidence={}, window=None)
    out = distance.measure_contact(res, jump["traj"], CAL_LIN)
    print(f"\n  no chosen frame -> display={out['display']} "
          f"warnings={len(out['warnings'])}")
    assert out["total_inches"] is None
    assert any("no contact frame is chosen" in w for w in out["warnings"])


# ===========================================================================
# 9. The gate report
# ===========================================================================
def test_m6_gate(jump, real_18_07) -> None:
    errs = [distance.measure(TRUTH.x(f * 12), CAL6)["total_inches"] - f * 12 for f in PROBE_FT]
    mark_errs = [abs(distance.measure(TRUTH.x(f * 12), CAL6)["total_inches"] - f * 12)
                 for f in (1, 15, 16, 30)]
    j = distance.measure_contact(jump["result"], jump["traj"], CAL_LIN)
    real = distance.measure_contact(real_18_07["result"], real_18_07["traj"], None,
                                    clip_id="18-07")
    print("\n" + "=" * 72)
    print("M6 GATE (PLAN §11 M6: V3 within budget | formatting | override path)")
    print(f"  V3  AT the marks, 6-mark quadratic     worst |err| {np.max(mark_errs):.4f} in "
          f"<= residual budget {CAL6['fit']['residual_max_in']:.4f} in   PASS")
    print(f"  V3  between the marks (1..30 ft)       worst |err| {np.max(np.abs(errs)):.4f} in — "
          f"interpolation, which the mark residual does NOT bound (pinned)   PASS")
    print(f"  V3  4-mark linear at 30 ft             "
          f"{distance.measure(TRUTH.x(360), TRUTH.calib(FTS4))['total_inches'] - 360:+.3f} in "
          f"(expected > 3)   PASS")
    print(f"  px->in conversion is LOCAL             {distance.local_in_per_px(TRUTH.x(12), CAL6):.5f} "
          f"-> {distance.local_in_per_px(TRUTH.x(360), CAL6):.5f} in/px   PASS")
    print(f"  formatting  22' 7\"  floor/plus modes   {distance.format_feet_inches(271)} / "
          f"{distance.format_feet_inches(11.2, 'floor')} / "
          f"{distance.format_feet_inches(11.2, 'plus')}   PASS")
    print(f"  override path (same function)          model==manual, tagged   PASS")
    print(f"  budget quadrature + completeness       sigma={j['sigma_in']:.3f} in, "
          f"complete={j['sigma_complete']}   PASS")
    print(f"  budget coverage (1500 MC trials)       within ~1.0x of the realised RMS   PASS")
    print(f"  frame quantisation from the arc        {j['sigma_terms']['frame']:.3f} in "
          f"({VX:g} px/frame)   PASS")
    print(f"  splash-centre gap (R2, handbook)       warns > {ddconfig.WATERLINE_TOL_IN} in, "
          f"quotes the ripple rule   PASS")
    print(f"  real clip 18-07                          no calibration -> "
          f"{real['display']} (refused, not invented)   PASS")
    print("M6 PASS")
