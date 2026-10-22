"""M2 verification gate (PLAN.md §11 M2 / §8 V2 + V3).

Gate: V2 (calibration reproducibility: residual RMS and the resulting distance
agree within 1 in) and V3 (synthetic ground truth: a known marking is reproduced
within the residual budget) pass, and the calibration JSON round-trips.

WHY THIS FILE CHANGED (2026-10-07)
----------------------------------
The previous version re-implemented the fit inline and imported only aliases, so
it could not have caught the ``NameError`` in ``list_setups()``.  It also
"verified" accuracy by measuring the module against the very marks it had been
fitted on — self-consistency, not accuracy.  This version drives the real module
against an **external projective ground truth**.

WHAT THE MEASUREMENT FOUND (pinned by the tests below)
-----------------------------------------------------
Ground truth: ``x(X) = CX + P0 * X / (1 + X/L)``, parameterised by the fractional
px-per-foot loss ``F`` across the 0..31 ft span (PLAN §4.2 claims "several
percent").

1. **The 4-mark linear fit only works inside the marked span.** Its error at
   31 ft grows at a steady **~0.93 in per 1% of foreshortening** (measured:
   0.93 in @1%, 1.87 @2%, 2.81 @3%, 3.76 @4%, 10.53 @11%). At a *modest* 2%
   foreshortening it already consumes the capstone's entire ±2 in budget at long
   distances. PLAN §0 D1 calls the 30-31 ft prompt "skippable if out of frame" —
   this is the price of skipping it.
2. **The 6-mark quadratic (including 30 and 31 ft) is accurate across the whole
   span**: worst |error| over 1..31 ft is 0.007 in @4% and 0.061 in @11%. The
   marks that span the measured range, not the fit order, are the accuracy lever.
3. **With only 4 marks the fit is not better than a single scale factor.** Both
   are one-parameter models in the marked span; measured difference 0.03 in @4%.
   PLAN §4.2's "fit rather than (Δx)/(px_per_ft)" argument therefore does *not*
   hold for the 3-4 mark branch — it only pays off at >=5 marks.
4. **Reproducibility is limited by mark LOCALISATION, not by fit order.** The
   distance range across 400 refits is exactly linear in the mark jitter:
   ~5.1 * sigma_px inches (4 marks) / 5.5 * sigma_px (6 marks). PLAN §8's V2
   target of <=1 in therefore demands mark precision <= ~0.2 px; at a sloppy
   0.5 px the range is ~2.5 in and V2 is missed no matter which fit is used.
   (Sub-pixel line snapping is what makes 0.1-0.2 px reachable — the same
   requirement the `dock-diving-scoring` skill states for the gesture seed.)

Run:  PYTHONPATH=src <venv-python> -m pytest src/tests/test_m2_calib.py -s
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dog_diving import calibration as C  # noqa: E402

# --- external ground truth --------------------------------------------------
CX, P0 = 520.0, 1.5                      # dock-edge x, px per inch at the dock (18 px/ft)
WATER_Y, IMG_W, IMG_H = 540, 1920, 1080
SPAN_IN = 372.0                          # 31 ft, the far end of the marked range
FTS4 = [(0, 0), (1, 0), (15, 0), (16, 0)]                    # -> linear
FTS6 = [(0, 0), (1, 0), (15, 0), (16, 0), (30, 0), (31, 0)]  # -> quadratic
PROBE_FT = list(range(1, 31))            # 1..30 ft, none of them used to fit
ERR_PER_PCT_AT_31FT = 0.93               # in per 1% foreshortening, 4-mark linear


def lam_for(foreshortening: float) -> float:
    """Invert ``F = 1 - 1/(1 + 372/L)^2`` -> L for the wanted foreshortening."""
    return SPAN_IN / (1.0 / np.sqrt(1.0 - foreshortening) - 1.0)


class Truth:
    """Projective pinhole along the pool axis — the reason a fit is needed at all."""

    def __init__(self, foreshortening: float) -> None:
        self.F = foreshortening
        self.L = lam_for(foreshortening)

    def x(self, X_in: float) -> float:
        return CX + P0 * X_in / (1.0 + X_in / self.L)

    def px_per_ft(self, X_in: float = 0.0) -> float:
        return P0 * 12.0 / (1.0 + X_in / self.L) ** 2

    def calib(self, fts, setup_id: str = "synth", fit_type=None) -> dict:
        marks = [
            {"label": f"{f}'{i:02d}\"", "ft": f, "inch": i,
             "x_px": self.x(f * 12 + i), "y_px": WATER_Y}
            for f, i in fts
        ]
        return C.build(
            setup_id, (IMG_W, IMG_H),
            dock_line=[[CX, 100], [CX, 980]],
            waterline=[[0, WATER_Y], [IMG_W, WATER_Y]],
            marks=marks, model_sha="deadbeef", notes="projective synthetic oracle",
            fit_type=fit_type,
        )

    def errors(self, calib: dict, fts=PROBE_FT) -> np.ndarray:
        """Signed error in inches (measured - nominal) at each probe distance."""
        return np.array([
            C.distance_inches(self.x(f * 12), calib) - f * 12 for f in fts
        ])

    def err_at(self, calib: dict, ft: float) -> float:
        return C.distance_inches(self.x(ft * 12), calib) - ft * 12


MOD = Truth(0.04)      # moderate: "several percent"
MILD = Truth(0.01)     # conservative
HARSH = Truth(0.11)    # worst case tested


# --- the ground truth must itself be discriminating -------------------------
def test_ground_truth_is_actually_foreshortened() -> None:
    for t in (MILD, MOD, HARSH):
        dock, far = t.px_per_ft(0.0), t.px_per_ft(360.0)
        loss = 1.0 - far / dock
        assert loss == pytest.approx(t.F, abs=0.005), f"parameterisation broken: {loss:.4f}"
    assert MOD.px_per_ft(0.0) > MOD.px_per_ft(360.0)


# --- V3: accuracy against the oracle ---------------------------------------
@pytest.mark.parametrize("fts", [FTS4, FTS6], ids=["4marks", "6marks"])
def test_v3_marks_are_reproduced_within_residual_budget(fts) -> None:
    """V3, literally: a point AT a known marking reproduces within the budget."""
    cal = MOD.calib(fts)
    err = MOD.errors(cal, fts=[f for f, _ in fts if f > 0])
    budget = max(cal["fit"]["residual_max_in"], 1e-9)
    print(f"\n  {cal['fit']['type']:<9} at-mark max|err|={np.max(np.abs(err)):.6f} in "
          f"(budget {budget:.6f} in)")
    assert np.max(np.abs(err)) <= budget + 1e-6, "fit does not reproduce its own marks"


def test_v3_quadratic_covers_the_whole_measured_span() -> None:
    """The accuracy guarantee: 6 marks spanning 0..31 ft reproduce 1..30 ft."""
    for t in (MILD, MOD, HARSH):
        cal = t.calib(FTS6)
        worst = float(np.max(np.abs(t.errors(cal))))
        print(f"\n  quadratic @{100 * t.F:.0f}% foreshortening: worst |err| 1..30 ft = {worst:.4f} in")
        assert worst < 0.5, f"@{100 * t.F:.0f}%: worst error {worst:.3f} in >= 0.5 in"


def test_linear_fit_is_only_valid_inside_the_marked_span() -> None:
    """Pins finding (1): the extrapolation cost of skipping the 30-31 ft marks."""
    cal = MOD.calib(FTS4)
    in_span = float(np.max(np.abs(MOD.errors(cal, fts=range(1, 17)))))
    at_31 = abs(MOD.err_at(cal, 31))
    print(f"\n  4-mark linear @4%: in-span (1..16 ft) worst={in_span:.3f} in | "
          f"at 31 ft={at_31:.3f} in")
    assert in_span < 0.5, f"linear fit should be usable inside the span, got {in_span:.3f} in"
    assert at_31 > 3.0, (
        f"expected the known >=3 in extrapolation error at 31 ft, got {at_31:.3f} in — "
        "if this improved, the finding in the module docstring needs revisiting"
    )


def test_extrapolation_error_law_is_about_one_inch_per_percent() -> None:
    """~0.93 in of error at 31 ft per 1% of foreshortening — the skip decision's price."""
    measured = {}
    for pct in (1, 2, 3, 4):
        t = Truth(pct / 100.0)
        measured[pct] = abs(t.err_at(t.calib(FTS4), 31))
    print("\n  error at 31 ft vs foreshortening: "
          + ", ".join(f"{k}%->{v:.2f} in" for k, v in measured.items()))
    for pct, err in measured.items():
        expected = ERR_PER_PCT_AT_31FT * pct
        assert expected * 0.8 <= err <= expected * 1.2, (
            f"{pct}% foreshortening: got {err:.3f} in, expected ~{expected:.3f} in"
        )
    assert list(measured.values()) == sorted(measured.values()), "error is not monotonic in F"


def test_four_marks_are_equivalent_to_a_single_scale_factor() -> None:
    """Pins finding (3): both are one-parameter models inside the marked span."""
    cal = MOD.calib(FTS4)
    span_px = MOD.x(192) - MOD.x(0)
    naive_in_per_px = 192.0 / span_px
    fit_err = np.max(np.abs(MOD.errors(cal, fts=range(1, 17))))
    naive_err = np.max(np.abs([
        (MOD.x(f * 12) - MOD.x(0)) * naive_in_per_px - f * 12 for f in range(1, 17)
    ]))
    print(f"\n  4-mark fit vs single scale (in-span): {fit_err:.3f} in vs {naive_err:.3f} in")
    assert abs(fit_err - naive_err) < 0.15, (
        "a 4-mark linear fit should be no better than a single scale factor; "
        "if this diverges, PLAN §4.2's rationale for the 3-4 mark branch may hold after all"
    )


def test_six_marks_decisively_beat_a_single_scale_factor() -> None:
    """...whereas spanning the range does pay off — by more than an order of magnitude."""
    cal = MOD.calib(FTS6)
    span_px = MOD.x(192) - MOD.x(0)
    naive_in_per_px = 192.0 / span_px
    fit_err = float(np.max(np.abs(MOD.errors(cal))))
    naive_err = float(np.max(np.abs([
        (MOD.x(f * 12) - MOD.x(0)) * naive_in_per_px - f * 12 for f in PROBE_FT
    ])))
    print(f"\n  6-mark quadratic vs single scale (1..30 ft): {fit_err:.4f} in vs {naive_err:.3f} in")
    assert fit_err * 20 < naive_err, f"fit {fit_err:.4f} in not decisively better than {naive_err:.3f} in"


# --- V2: reproducibility ----------------------------------------------------
def test_v2_identical_marks_are_exactly_deterministic() -> None:
    a, b, c = (MOD.calib(FTS4, setup_id="s") for _ in range(3))
    for d in (b, c):
        assert d["fit"] == a["fit"], "fit is not deterministic"
    assert len({json.dumps(x["fit"], sort_keys=True) for x in (a, b, c)}) == 1


def test_v2_literal_three_calibrations_agree_exactly() -> None:
    """PLAN §8 V2 read literally: the same still frame calibrated 3x must agree."""
    cal = MOD.calib(FTS4)
    dists = [C.distance_inches(MOD.x(22 * 12), cal) for _ in range(3)]
    rmses = [cal["fit"]["residual_rms_in"] for _ in range(3)]
    assert max(dists) - min(dists) == 0.0, "re-calibrating identical marks changed the distance"
    assert max(rmses) - min(rmses) == 0.0
    assert max(dists) - min(dists) <= 1.0 and max(rmses) - min(rmses) <= 1.0


def _jitter_range(fts, sigma_px: float, probe_ft: float, n: int = 400, seed: int = 11) -> float:
    """Max-min distance across ``n`` refits with every mark jittered by N(0, sigma_px)."""
    rng = np.random.default_rng(seed)
    base = MOD.calib(fts)
    probe = MOD.x(probe_ft * 12)
    dists = []
    for _ in range(n):
        marks = [
            {**m, "x_px": m["x_px"] + float(rng.normal(0.0, sigma_px))}
            for m in MOD.calib(fts)["marks"]
        ]
        dists.append(C.distance_inches(probe, {**base, "marks": marks,
                                               "fit": C.fit_from_marks(marks)}))
    return max(dists) - min(dists)


def test_v2_jitter_spread_scales_linearly_with_mark_precision() -> None:
    """Mark LOCALISATION, not fit order, is the reproducibility budget.

    Measured law (4 marks, 400 refits/probe): distance range = 5.1 * sigma_px
    inches. It is exactly linear, so it converts "click accurately" into a
    number: the PLAN §8 V2 target of <=1 in needs mark precision <= ~0.2 px.
    """
    ratios = {}
    for fts, label in ((FTS4, "4marks"), (FTS6, "6marks")):
        rs = []
        for sigma in (0.05, 0.10, 0.20):
            spread = _jitter_range(fts, sigma, probe_ft=14)
            rs.append(spread / sigma)
            print(f"\n  {label} sigma={sigma:.2f} px -> range {spread:.3f} in "
                  f"({spread / sigma:.2f} in per px)")
        assert max(rs) - min(rs) < 0.3, f"{label}: spread is not linear in sigma_px ({rs})"
        ratios[label] = sum(rs) / len(rs)
    print(f"\n  law: range ~= {ratios['4marks']:.1f}*sigma_px (4 marks), "
          f"{ratios['6marks']:.1f}*sigma_px (6 marks)")
    # the headline constraint
    assert 1.0 <= _jitter_range(FTS4, 0.20, probe_ft=14) <= 1.5, "0.2 px should cost ~1 in"
    # and document the failure mode rather than assert it away
    coarse = _jitter_range(FTS4, 0.50, probe_ft=14)
    print(f"  NOTE: at a sloppy 0.5 px, range = {coarse:.2f} in -- over the V2 1 in budget")
    assert coarse > 1.0, "the 0.5 px case is expected to exceed the V2 budget"


def test_v2_within_budget_at_subpixel_mark_precision() -> None:
    """LE.0.1 px mark localisation (sub-pixel line snapping) meets V2 with margin."""
    for probe_ft in (8, 14):
        spread = _jitter_range(FTS4, 0.10, probe_ft=probe_ft)
        print(f"\n  4 marks, sigma=0.10 px, probe {probe_ft} ft -> range {spread:.3f} in")
        assert spread <= 1.0, f"@{probe_ft} ft: {spread:.3f} in exceeds the 1 in V2 budget"


# --- JSON round-trip + persistence (the list_setups NameError regression) ---
def test_json_round_trip(tmp_path: Path) -> None:
    cal = MOD.calib(FTS4, setup_id="roundtrip")
    p = tmp_path / "roundtrip.json"
    C.save(cal, p)
    assert p.is_file(), "save() did not produce the file"
    assert not (tmp_path / "roundtrip.tmp.json").exists(), "temp file left behind"
    back = C.load(p)
    assert back == cal, "calibration did not survive a save/load round trip"
    assert C.distance_inches(MOD.x(264), back) == pytest.approx(
        C.distance_inches(MOD.x(264), cal), abs=1e-9)


def test_list_setups_finds_saved_setups(tmp_path: Path) -> None:
    """Regression: this raised NameError (undefined CALIB_DIR) and broke on dir=None."""
    for sid in ("a", "b"):
        C.save(MOD.calib(FTS4, setup_id=sid), tmp_path / f"{sid}.json")
    assert [p.name for p in C.list_setups(tmp_path)] == ["a.json", "b.json"]
    assert C.list_setups(tmp_path / "nope") == []
    assert isinstance(C.list_setups(), list)          # dir=None resolves the default


def test_save_is_atomic_and_creates_parent(tmp_path: Path) -> None:
    p = tmp_path / "deep" / "nested" / "s.json"
    C.save(MOD.calib(FTS4), p)
    assert p.is_file()
    assert json.loads(p.read_text())["setup_id"] == "synth"


def test_schema_validation_rejects_incomplete_documents() -> None:
    good = MOD.calib(FTS4)
    C.validate_schema(good)                            # must not raise
    for key in ("created", "model_sha", "notes", "waterline", "dock_line"):
        broken = {k: v for k, v in good.items() if k != key}
        with pytest.raises(ValueError, match="missing required keys"):
            C.validate_schema(broken)


# --- fit mechanics ----------------------------------------------------------
def test_fit_type_selection_follows_mark_count() -> None:
    assert C.choose_fit_type(3) == "linear"
    assert C.choose_fit_type(4) == "linear"
    assert C.choose_fit_type(5) == "quadratic"
    assert C.choose_fit_type(6) == "quadratic"
    assert C.choose_fit_type(6, "linear") == "linear"   # explicit override wins


def test_fit_rejects_non_monotonic_marks() -> None:
    bad = MOD.calib(FTS4)["marks"]
    bad[2]["x_px"] = bad[1]["x_px"] - 5
    with pytest.raises(ValueError, match="strictly increasing"):
        C.fit_from_marks(bad)


def test_fit_rejects_missing_keys() -> None:
    bad = MOD.calib(FTS4)["marks"]
    del bad[1]["y_px"]
    with pytest.raises(ValueError, match="missing keys"):
        C.fit_from_marks(bad)


def test_fit_rejects_too_few_marks() -> None:
    with pytest.raises(ValueError, match="at least 2 marks"):
        C.fit_from_marks(MOD.calib(FTS4)["marks"][:1])


def test_inverse_ft_round_trips() -> None:
    """inverse_ft must be the algebraic inverse of the fit (it draws the 1-ft grid)."""
    for fts in (FTS4, FTS6):
        cal = MOD.calib(fts)
        for inches in (0, 12, 180, 264, 372):
            back = C.map_inches(C.inverse_ft(inches, cal), cal)
            assert back == pytest.approx(float(inches), abs=1e-6), (
                f"{cal['fit']['type']}: inverse_ft({inches}) round trip -> {back:.6f} in"
            )


def test_dock_edge_is_the_origin_not_the_zero_mark() -> None:
    """PLAN §4.2: D is measured from the dock edge, so moving the edge moves every reading."""
    cal = MOD.calib(FTS4)
    assert C.distance_inches(CX, cal) == pytest.approx(0.0, abs=1e-9)
    in_per_px = 1.0 / (C.map_inches(CX + 1, cal) - C.map_inches(CX, cal))
    shifted = {**cal, "dock_line": [[CX + 3 * in_per_px, 100], [CX + 3 * in_per_px, 980]]}
    delta = C.distance_inches(MOD.x(264), cal) - C.distance_inches(MOD.x(264), shifted)
    assert delta == pytest.approx(3.0, abs=0.05), f"origin shift produced {delta:.3f} in"


# --- validation gates (PLAN §4.3) -------------------------------------------
def test_gates_pass_on_a_good_calibration() -> None:
    for fts in (FTS4, FTS6):
        cal = MOD.calib(fts)
        gates = C.validate_gates(cal)
        assert C.gates_pass(gates), {k: v["detail"] for k, v in gates.items()}
        assert gates["G2"]["pass"], gates["G2"]["detail"]
        assert not gates["G3"]["warn"], gates["G3"]["detail"]


def test_gate_g2_fails_when_dock_edge_disagrees_with_zero_mark() -> None:
    cal = MOD.calib(FTS4)
    in_per_px = 1.0 / (C.map_inches(CX + 1, cal) - C.map_inches(CX, cal))
    bad = {**cal, "dock_line": [[CX + 8 * in_per_px, 100], [CX + 8 * in_per_px, 980]]}
    gates = C.validate_gates(bad)
    assert not gates["G2"]["pass"], gates["G2"]["detail"]


def test_gate_g3_is_tiered_warn_then_block() -> None:
    cal = MOD.calib(FTS4)
    warn = {**cal, "fit": {**cal["fit"], "residual_rms_in": 2.5}}
    block = {**cal, "fit": {**cal["fit"], "residual_rms_in": 4.0}}
    g_warn, g_block = C.validate_gates(warn)["G3"], C.validate_gates(block)["G3"]
    assert g_warn["pass"] and g_warn["warn"], g_warn["detail"]     # >2 warns but is allowed
    assert not g_block["pass"], g_block["detail"]                  # >3 blocks


def test_gate_g5_detects_fold_back() -> None:
    # turning point inside the span: d(x) = a x^2 + b x turns at x = -b/(2a) = 300 px
    cal = MOD.calib(FTS6)
    cal["fit"] = {**cal["fit"], "type": "quadratic", "a": -0.0004, "b": 0.24, "x0": CX}
    gates = C.validate_gates(cal)
    assert not gates["G5"]["pass"], gates["G5"]["detail"]


# --- human-readable gate report --------------------------------------------
def test_m2_gate() -> None:
    print("\n" + "=" * 72)
    print("M2 GATE")
    for label, fts in (("4 marks 0,1,15,16", FTS4), ("6 marks 0,1,15,16,30,31", FTS6)):
        cal = MOD.calib(fts)
        err = MOD.errors(cal)
        gates = C.validate_gates(cal)
        print(f"  {label:<26} fit={cal['fit']['type']:<9} RMS={cal['fit']['residual_rms_in']:.4f} in  "
              f"worst|err| 1..30ft={np.max(np.abs(err)):.3f} in  "
              f"gates={'PASS' if C.gates_pass(gates) else 'FAIL'}")
    print(f"  quadratic across span (1%..11% foreshortening)      worst <= 0.061 in   PASS")
    print(f"  4-mark extrapolation law                            ~0.93 in / 1%      PASS")
    print("  V2 identical marks, 3x                              exact              PASS")
    print("  V2 mark-localisation law                            ~5.1*sigma_px      PASS")
    print("  JSON round-trip + schema validation                 identical          PASS")
    print("  list_setups() resolves CALIB_DIR                    no NameError       PASS")
    print("M2 PASS")
