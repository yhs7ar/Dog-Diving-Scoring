"""Distance — pixels to inches, feet+inches, and the uncertainty budget (M6, PLAN.md §5/§4.4).

PLAN §5 ``distance.py`` verbatim:

    - ``measure(x_px, calib, point_source)`` -> ``{feet, inches, total_inches, sigma_in,
      display: "22' 7\\""}``.
    - Manual override: a corrected ``(x,y)`` from a GUI click flows through the same
      function, tagged ``point_source: "model" | "manual"``.
    - Also emits the **splash-centre gap warning**: if ``|y_point - waterline.y|`` exceeds
      a threshold, note that the entry point is above/below the waterline and that Dog Diving
      judges mark the ripple centre (handbook) — so the number is a lower bound on the
      rule's mark (risk R2).

and PLAN §4.4's budget::

    sigma_D = sqrt( sigma_fit^2 + sigma_kp^2 + sigma_frame^2 )
      sigma_fit   = calibration residual RMS, converted to inches at the measurement x
      sigma_kp    = std-dev of x over the last K accepted tailset frames, converted to inches
      sigma_frame = |D(frame) - D(frame +- 1)|   (frame-quantisation cost)

This module is the only place a pixel becomes an inch, and it stays thin on purpose: the
arithmetic lives in :mod:`dog_diving.calibration` (``distance_inches``), the arc in
:mod:`dog_diving.trajectory`, the contact frame in :mod:`dog_diving.contact`.  What is added
here is (a) the display formatting, (b) the budget *with its provenance*, and (c) the
disclosure PLAN §9 R2 demands — every number says which point it came from, how much of the
budget is actually computed, and what is wrong with the point.

Measured on this build (see ``test_m6_distance.py``)
---------------------------------------------------
1. **The px->in conversion has to be local, and the size of the mistake is measured.** A
   single global scale factor taken across the 0-16 ft marks is off by **3.2 in at 30 ft**
   at 4% foreshortening (6-mark fitted quadratic: 0.007 in).  ``local_in_per_px`` applies
   ``d'(x)`` at the measurement point; that derivative is the "converted to inches at the
   measurement x" clause of PLAN §4.4.
2. **PLAN §4.4's ``sigma_kp`` as written mixes the dog's motion into the noise term.** "Std-dev
   of x over the last K accepted tailset frames" is not a localisation uncertainty when those
   frames span travel: measured on a synthetic constant-velocity arc (12 px/frame, K=5) the
   literal reading gives **~17 px** of scatter where the point-to-point noise is exactly 0,
   while the *de-trended* scatter against the fitted arc gives **0.00 px**.  Both numbers are
   returned; the default is PLAN's literal reading because it is the conservative one, and a
   material disagreement is disclosed in ``notes`` rather than silently chosen.
3. **The budget is tested for coverage, not just for arithmetic.** A budget that does not
   bracket its own errors is decoration: the Monte-Carlo test perturbs the measurement point
   with a known 1.5 px and checks that the realised RMS error in inches lands within ~1x of
   the reported ``sigma_in`` (measured ratio 1.02).
4. **PLAN §8 V3's "within the residual budget" holds at the marks and not between them.** Probing
   every foot from 1 to 30 ft gives a worst |err| of **0.00732 in** against a mark residual max of
   **0.00226 in** (3.2x).  The extra is the interpolation error between marks; the residual only
   reports consistency inside the marked span.  V3 says "a point at a known marking", so the
   at-mark check is the literal gate — but a gate line quoting the residual against a
   whole-span probe would be over-claiming, so both numbers are pinned.
5. **The frame-quantisation term is refused outside the arc's observed support.** On 18-07 the
   chosen frame (437) sits 41 frames past the last arc sample (396); differencing the *extrapolated*
   arc there reports `sigma_frame = 1033 in` — half a pool per frame.  That is not an uncertainty, so
   `measure_contact` only computes the term while the frame is inside `airborne_frame_range`, and
   says so; `sigma_complete` goes False and `sigma_in` is labelled a lower bound.
6. **The handbook was read (2026-10-08) from the capstone folder.**  *Splash Circle* — "the
   center point of this ripple is what should be used to determine the break in the water's
   surface or the mark for distance"; *Scoring* (c) — "from the end of the dock to where the
   base of the dog's tail breaks the surface of the water"; *Consistency* — "we recommend
   always marking your distances to the plus vs. the negative benefit of the competitor"; and
   the manual-judge certification bar is **"a minimum of 60% of your scores be within 6" of
   the electronic score"** (a citable human baseline for M9's ±6 in column).  The rounding
   policy is therefore exposed (``nearest`` default, ``plus`` = the handbook's stated
   convention) instead of being assumed, and ``total_inches`` is always reported unrounded.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np

from . import calibration as _calibration
from . import trajectory as _trajectory
from .config import (
    DISTANCE_KP_MODE,
    DISTANCE_KP_WINDOW,
    DISTANCE_ROUNDING,
    WATERLINE_TOL_IN,
)

# display rounding modes (see ``format_feet_inches``)
R_NEAREST = "nearest"      # half away from zero — the honest report
R_FLOOR = "floor"          # toward -inf
R_PLUS = "plus"            # toward +inf — the handbook's "mark to the plus"
ROUND_MODES = (R_NEAREST, R_FLOOR, R_PLUS)

# sigma_kp readings
KP_LITERAL = "literal"      # PLAN §4.4 verbatim: std-dev of x over the last K accepted frames

# point provenance
P_MODEL = "model"          # PLAN §5's two documented values ...
P_MANUAL = "manual"
O_DETECTION = "detection"  # ... plus where a "model" point actually came from
O_CLICK = "click"


# --- configuration ----------------------------------------------------------
@dataclass(frozen=True)
class DistanceConfig:
    """M6 knobs.  ``waterline_tol_in`` is PLAN §6's key; the rest are this module's."""

    waterline_tol_in: float = WATERLINE_TOL_IN
    kp_window: int = DISTANCE_KP_WINDOW
    kp_mode: str = DISTANCE_KP_MODE
    rounding: str = DISTANCE_ROUNDING
    mark_sigma_px: float | None = None   # optional 4th term; see ``budget``

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


# --- the local scale (PLAN §4.4's "converted to inches at the measurement x") ------
def local_in_per_px(x_px: float, calib: dict) -> float:
    """``|d'(x)|`` — inches per pixel **at the measurement point**.

    The fit maps px -> inches, so its derivative is already an in/px scale; the point of
    evaluating it at ``x`` rather than using a constant is that px-per-foot varies by
    several percent across the pool (PLAN §4.2, and measured: 18.00 px/ft at the dock
    against 17.30 at 30 ft at 4% foreshortening).
    """
    f = calib.get("fit", calib)
    x = float(x_px) - float(f["x0"])
    if f["type"] == "quadratic":
        return abs(2.0 * float(f["a"]) * x + float(f["b"]))
    return abs(float(f["a"]))




def waterline_y(calib: dict) -> float | None:
    """The calibration's water row (``None`` when absent/unusable)."""
    if not calib:
        return None
    wl = calib.get("waterline")
    if not wl:
        return None
    ys = [float(p[1]) for p in wl]
    return float(np.mean(ys)) if ys else None


# --- feet + inches (PLAN §5's ``display``) ----------------------------------
def _round_inches(inches: float, rounding: str) -> int:
    if rounding not in ROUND_MODES:
        raise ValueError(f"rounding must be one of {ROUND_MODES}, got {rounding!r}")
    v = float(inches)
    if not math.isfinite(v):
        raise ValueError(f"cannot format a non-finite distance: {inches!r}")
    if rounding == R_FLOOR:
        return int(math.floor(v))
    if rounding == R_PLUS:
        return int(math.ceil(v))
    return int(math.floor(v + 0.5)) if v >= 0 else -int(math.floor(-v + 0.5))


def split_feet_inches(inches: float, rounding: str = R_NEAREST) -> tuple[int, int]:
    """Rounded distance -> signed ``(feet, inches)``; ``inches`` is always 0..11."""
    total = _round_inches(inches, rounding)
    sign = -1 if total < 0 else 1
    a = abs(total)
    return sign * (a // 12), sign * (a % 12)


def format_feet_inches(inches: float, rounding: str = R_NEAREST) -> str:
    """``22' 7"`` — Dog Diving' feet-and-inches form (PLAN §0, assumed *"as Dog Diving reports it"*).

    ``rounding``:
      ``nearest`` (default) — half away from zero; the measurement, reported without a bias.
      ``floor`` — toward -inf (a scorer who reads the inch *below* the mark).
      ``plus`` — toward +inf: the handbook's *"always marking your distances to the plus vs.
      the negative benefit of the competitor"*.  Exposed rather than assumed, because the
      handbook states the convention and neither it nor PLAN fixes an inch-rounding rule for
      this prototype; ``total_inches`` is always reported unrounded so the display can never
      hide the difference.
    """
    feet, inch = split_feet_inches(inches, rounding)
    sign = "-" if (feet < 0 or inch < 0) else ""
    return f"{sign}{abs(feet)}' {abs(inch)}\""


# --- the uncertainty budget (PLAN §4.4) -------------------------------------
def budget(
    x_px: float,
    calib: dict | None,
    *,
    cfg: DistanceConfig | None = None,
    kp_xs: Sequence[float] | None = None,
    kp_pred_xs: Sequence[float] | None = None,
    x_next_px: float | None = None,
    x_prev_px: float | None = None,
) -> dict[str, Any]:
    """PLAN §4.4's ``sigma_D`` with every term named, valued and *availability-flagged*.

    ``sigma_fit``  the fit's residual RMS in inches (PLAN's wording; it is measured inside the
                   marked span, so it says nothing about extrapolation — see the M2 finding).
    ``sigma_kp``   std-dev of the last K accepted tailset x's x the local scale.  PLAN's literal
                   reading; ``detrended`` (scatter about the fitted arc) is computed alongside
                   whenever ``kp_pred_xs`` is supplied, because the literal reading mixes motion
                   into the noise (module docstring finding 2).
    ``sigma_frame``|D(f) - D(f+-1)|, via the neighbouring frames' x and the local scale.
    ``mark``       OPTIONAL 4th term, off by default (``mark_sigma_px=None``): a mark-jitter
                   uncertainty, which is the one *pixel-domain* input for which "converted to
                   inches at the measurement x" actually bites.  The M2 study measured mark
                   localisation as the reproducibility budget (~5.1 x sigma_px inches), so
                   leaving it out is a deliberate, reported omission rather than an oversight.

    ``complete`` is False when a PLAN §4.4 term could not be computed; a partial quadrature is
    a *lower* bound on sigma_D and must never be quoted as if it were the whole budget.
    """
    cfg = cfg or DistanceConfig()
    notes: list[str] = []
    terms: dict[str, float | None] = {"fit": None, "kp": None, "frame": None, "mark": None}
    avail = {k: False for k in terms}
    scale = local_in_per_px(x_px, calib) if calib else None

    if calib:
        terms["fit"] = float(calib["fit"]["residual_rms_in"])
        avail["fit"] = True
        notes.append(
            f"sigma_fit = {terms['fit']:.3f} in is the fit's residual RMS *inside the marked "
            f"span*; it is not an extrapolation error and excludes mark localisation"
        )

    literal = detrended = None
    xs = [float(v) for v in (kp_xs or [])]
    if len(xs) >= 2 and scale is not None:
        literal = float(np.std(xs, ddof=1)) * scale
        preds = [float(v) for v in (kp_pred_xs or [])]
        if len(preds) == len(xs):
            detrended = float(np.std(np.array(xs) - np.array(preds), ddof=1)) * scale
        chosen = literal if cfg.kp_mode == KP_LITERAL else detrended
        if chosen is None:
            chosen = literal
        terms["kp"] = chosen
        avail["kp"] = True
        if literal is not None and detrended is not None and literal > 0.0:
            ratio = detrended / literal
            if ratio < 0.8:
                notes.append(
                    f"sigma_kp: literal {literal:.3f} in vs detrended {detrended:.3f} in "
                    f"({100 * ratio:.0f}%) — the last K frames span motion, so PLAN §4.4's "
                    f"literal reading charges that travel to the noise term; the value used "
                    f"is the '{cfg.kp_mode}' one"
                )
    elif scale is not None:
        notes.append(f"sigma_kp unavailable: {len(xs)} accepted frame(s) in the window (need >= 2)")
    else:
        notes.append("sigma_kp unavailable: no calibration, so there is no px->in scale to "
                     "convert the window's scatter into inches")

    deltas = [abs(float(v) - float(x_px)) for v in (x_next_px, x_prev_px) if v is not None]
    if deltas and scale is not None:
        terms["frame"] = max(deltas) * scale
        avail["frame"] = True
    else:
        notes.append("sigma_frame unavailable: no neighbouring frame to difference (needs the "
                     "fitted arc or an explicit point at frame +-1)")

    if cfg.mark_sigma_px is not None and scale is not None:
        terms["mark"] = abs(float(cfg.mark_sigma_px)) * scale
        avail["mark"] = True
    else:
        notes.append(
            "no mark-localisation term: PLAN §4.4 does not list one, and the M2 study measured "
            "mark placement as the *dominant* reproducibility budget (~5.1 x sigma_px inches) — "
            "pass mark_sigma_px to include it, and inflate any claim made without it"
        )

    active = [v for k, v in terms.items() if avail[k] and v is not None]
    sigma = float(math.sqrt(sum(v * v for v in active))) if active else None
    return {
        "sigma_in": sigma,
        "terms": terms,
        "available": avail,
        "complete": bool(avail["fit"] and avail["kp"] and avail["frame"]),
        "sigma_kp_literal_in": literal,
        "sigma_kp_detrended_in": detrended,
        "kp_k": len(xs),
        "kp_mode": cfg.kp_mode,
        "in_per_px": scale,
        "notes": notes,
    }


# --- the measurement ---------------------------------------------------------
def measure(
    x_px: float | None,
    calib: dict | None,
    *,
    y_px: float | None = None,
    point_source: str = P_MODEL,
    point_origin: str | None = None,
    cfg: DistanceConfig | None = None,
    frame: int | None = None,
    clip_id: str | None = None,
    image_size: Sequence[int] | None = None,
    kp_xs: Sequence[float] | None = None,
    kp_pred_xs: Sequence[float] | None = None,
    x_next_px: float | None = None,
    x_prev_px: float | None = None,
) -> dict[str, Any]:
    """PLAN §5's ``measure``.  Model point and manual override take the *same* path.

    ``calib=None`` is a supported, honest call: the pixel-domain fields (x, y, the fitted-arc
    frame cost, sigma_kp) are still reported, the inches fields are ``None``, and the reason is
    in ``warnings``.  That matters here because ``data/calib/`` is empty (HANDOFF open item 10),
    so "this clip has no calibration" is the current normal answer rather than an error.
    """
    cfg = cfg or DistanceConfig()
    warnings: list[str] = []
    out: dict[str, Any] = {
        "clip_id": clip_id, "frame": frame,
        "point_source": point_source, "point_origin": point_origin or (
            O_CLICK if point_source == P_MANUAL else O_DETECTION),
        "x_px": None if x_px is None else float(x_px),
        "y_px": None if y_px is None else float(y_px),
        "total_inches": None, "rounded_inches": None, "feet": None, "inches": None,
        "display": None, "sigma_in": None, "sigma_terms": None, "sigma_complete": False,
        "sigma_kp_literal_in": None, "sigma_kp_detrended_in": None, "kp_k": 0, "notes": [],
        "in_per_px": None, "calib_setup_id": None, "calib_fit_type": None,
        "calib_residual_rms_in": None, "waterline_y": None,
        "splash_gap_px": None, "splash_gap_in": None, "splash_gap_tol_in": float(cfg.waterline_tol_in),
        "flags": {}, "gates": {}, "warnings": warnings, "cfg": cfg.as_dict(),
    }

    if x_px is None:
        warnings.append("no measurement point — there is nothing to convert (the chosen frame has "
                        "no accepted detection; an arc prediction is no longer substituted for one)")
        return out

    b = budget(x_px, calib, cfg=cfg, kp_xs=kp_xs, kp_pred_xs=kp_pred_xs,
               x_next_px=x_next_px, x_prev_px=x_prev_px)
    out["sigma_in"] = b["sigma_in"]
    out["sigma_terms"] = b["terms"]
    out["sigma_complete"] = b["complete"]
    out["in_per_px"] = b["in_per_px"]
    out["sigma_kp_literal_in"] = b["sigma_kp_literal_in"]
    out["sigma_kp_detrended_in"] = b["sigma_kp_detrended_in"]
    out["kp_k"] = b["kp_k"]
    out["notes"] = b["notes"]
    warnings.extend(b["notes"])

    if calib is None:
        warnings.append("NO CALIBRATION for this clip (data/calib/ is empty): px->in needs "
                        "PLAN §4.2's marks, and PLAN §5 #2 needs the waterline row — the "
                        "pixel-domain fields above are real, the inches are not computable")
        return out

    # provenance + the gates, so a distance is never quoted from a calibration that failed
    out["calib_setup_id"] = calib.get("setup_id")
    out["calib_fit_type"] = calib.get("fit", {}).get("type")
    out["calib_residual_rms_in"] = calib.get("fit", {}).get("residual_rms_in")
    gates = _calibration.validate_gates(calib)
    out["gates"] = {k: bool(v["pass"]) for k, v in gates.items()}
    failed = [k for k, v in gates.items() if not v["pass"]]
    if failed:
        warnings.append(
            "calibration failed gate(s) "
            + ", ".join(f"{k} ({gates[k]['detail']})" for k in failed)
            + " — the distance below is computed from a calibration PLAN §4.3 would block"
        )
    if image_size is not None and list(calib.get("image_size", [])) != list(image_size):
        warnings.append(
            f"calibration was made at {list(calib.get('image_size', []))} but this clip is "
            f"{list(image_size)} — a calibration is bound to a camera setup, and a resolution "
            f"change invalidates the pixel geometry (PLAN §4.1)"
        )
    for k, v in gates.items():
        if v.get("warn"):
            warnings.append(f"calibration gate {k} warns: {v['detail']}")

    total = float(_calibration.distance_inches(x_px, calib))
    feet, inch = split_feet_inches(total, cfg.rounding)
    out.update({
        "total_inches": total,
        "rounded_inches": int(_round_inches(total, cfg.rounding)),
        "feet": feet, "inches": inch,
        "display": format_feet_inches(total, cfg.rounding),
    })

    # R2 / PLAN §5: the splash-centre gap.  y is converted with the *local x scale* — PLAN §4.1
    # keeps y as QA ("that keeps the prototype 1-D and honest about what it can't resolve") and
    # PLAN §10 excludes a homography, so this is a screening threshold, not a measurement.
    wy = waterline_y(calib)
    out["waterline_y"] = wy
    if y_px is None:
        warnings.append("no y for the measurement point, so the splash-centre gap cannot be "
                        "checked (PLAN §5 / risk R2)")
    elif wy is None:
        warnings.append("the calibration has no waterline row, so the splash-centre gap cannot "
                        "be checked (PLAN §5 / risk R2)")
    else:
        gap_px = float(y_px) - wy
        gap_in = gap_px * (b["in_per_px"] or 0.0)
        out["splash_gap_px"] = gap_px
        out["splash_gap_in"] = gap_in
        if abs(gap_in) > float(cfg.waterline_tol_in):
            where = "ABOVE" if gap_px < 0 else "BELOW"
            bound = ("LOWER" if gap_px < 0 else "UPPER")
            warnings.append(
                f"SPLASH-CENTRE GAP: the measurement point is {abs(gap_in):.1f} in {where} the "
                f"waterline (> {cfg.waterline_tol_in:g} in). Dog Diving judges mark the *centre of "
                f"the splash circle / ripple* — the base of the tail breaking the surface — not "
                f"the tailset keypoint at an arbitrary frame, so this distance is a {bound} "
                f"BOUND on the rule's mark and must be reported as one (PLAN §9 R2)."
            )

    if total < 0.0:
        warnings.append(
            f"the measurement point is {abs(total):.1f} in on the DOCK side of the dock edge, so "
            f"the distance is negative — the datum or the point is wrong (the measured distance "
            f"from the lip must come out positive)"
        )
    out["flags"] = {
        "negative_distance": bool(total < 0.0),
        "splash_gap_warn": bool(out["splash_gap_in"] is not None
                                and abs(out["splash_gap_in"]) > float(cfg.waterline_tol_in)),
        "sigma_complete": bool(b["complete"]),
        "gates_failed": failed,
        "no_calibration": False,
    }
    return out


# --- the contact-aware entry point ------------------------------------------
def measure_contact(
    contact_result: Any,
    traj: Any,
    calib: dict | None,
    *,
    cfg: DistanceConfig | None = None,
    manual_point: Sequence[float] | None = None,
    clip_id: str | None = None,
    image_size: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Measure the operator's chosen contact frame (PLAN §5 D4 -> §5 ``distance.py``).

    Resolves the point, the sigma_kp window and the frame-quantisation neighbours from the
    trajectory, then defers to :func:`measure` — so the manual-override path in the GUI is the
    same function, differing only in ``point_source`` (PLAN §5's requirement).

    ``manual_point=(x, y)`` is the GUI click: it overrides the model point and tags the result
    ``manual``; the model's own point stays in ``model_point`` for the A/B.
    """
    cfg = cfg or DistanceConfig()
    frame = getattr(contact_result, "chosen_frame", None)
    points = list(getattr(traj, "points", []) or [])
    coeffs = getattr(traj, "coeffs", None)
    if clip_id is None:
        clip_id = getattr(contact_result, "clip_id", None) or getattr(traj, "clip_id", None)

    model_point = None
    origin = None
    if frame is not None:
        for p in points:
            if int(p["frame"]) == int(frame):
                model_point, origin = (float(p["x"]), float(p["y"])), O_DETECTION
                break
        # NO arc fallback (removed 2026-10-19, replacing this logic with the trajectory fork's).
        # The fork's Review only ever marks accepted detections; this code used to place the marker
        # on the FITTED PARABOLA when the chosen frame had no detection, which produces a cursor
        # that sweeps smoothly up and diagonally across frames whether or not the dog is found —
        # measured on 18-07: frames 100-220 have 8 accepted points clustered at x ~ 283, y ~ 509 and
        # the marker still glided along the arc through the other 100+ frames.  An arc prediction
        # is not a detection and must never be presented as one.
    # NOTE: there is deliberately NO fallback to the last detection when no contact frame is
    # chosen.  PLAN §5 D4 makes the entry frame the operator's decision, and quietly measuring
    # the last detection instead is exactly the "silently emit a datum" failure mode the
    # dock-diving-scoring skill warns about — it is an aftermath point on this corpus.

    no_point_warnings: list[str] = []
    if model_point is None and manual_point is None and frame is not None:
        no_point_warnings.append(
            f"no accepted detection at frame {frame}, and no point is invented: the marker is "
            f"only ever placed on a detected tail-set (or a manual click). Pick a frame that has "
            f"one — the trajectory table lists them."
        )
    if manual_point is not None:
        x_px, y_px, source, origin = float(manual_point[0]), float(manual_point[1]), \
            P_MANUAL, O_CLICK
    elif model_point is not None:
        x_px, y_px, source = model_point[0], model_point[1], P_MODEL
    else:
        x_px = y_px = source = None

    # sigma_kp window: the last K accepted frames at or before the chosen one
    ref = frame
    window = ([] if ref is None
              else [p for p in points if int(p["frame"]) <= int(ref)][-int(cfg.kp_window):])
    kp_xs = [float(p["x"]) for p in window]
    kp_pred_xs = ([float(_trajectory.predict(coeffs, float(p["frame"]))[0]) for p in window]
                  if coeffs else None)

    # sigma_frame: the fitted arc one frame either side of the choice — but ONLY while the
    # chosen frame is inside the arc's observed support.  Extrapolated, the arc's slope is not a
    # fit: on 18-07 (chosen f437, last arc sample f396) the neighbouring-frame difference came
    # out at 1033 in, i.e. half the pool per frame.  That is not an uncertainty, so the term is
    # refused and the refusal is disclosed rather than reported as a number.
    support = tuple(getattr(traj, "airborne_frame_range", None) or ())
    inside_support = bool(ref is not None and support and support[0] <= ref <= support[1])
    x_next = x_prev = None
    if coeffs and inside_support:
        x_next = float(_trajectory.predict(coeffs, float(ref) + 1.0)[0])
        x_prev = float(_trajectory.predict(coeffs, float(ref) - 1.0)[0])

    out = measure(x_px, calib, y_px=y_px, point_source=source or P_MODEL, point_origin=origin,
                  cfg=cfg, frame=(None if ref is None else int(ref)), clip_id=clip_id,
                  image_size=image_size, kp_xs=kp_xs, kp_pred_xs=kp_pred_xs,
                  x_next_px=x_next, x_prev_px=x_prev)

    out["contact_source"] = getattr(contact_result, "chosen_source", None)
    out["arc_is_freefall"] = getattr(contact_result, "arc_is_freefall", None)
    out["arc_reason"] = getattr(contact_result, "arc_reason", None)
    out["kp_frames"] = [int(p["frame"]) for p in window]
    out["model_point"] = None if model_point is None else list(model_point)
    out["warnings"].extend(_contact_disclosures(contact_result, traj, ref, source,
                                                arc_used=inside_support))
    out["warnings"].extend(no_point_warnings)
    return out


def _contact_disclosures(contact_result: Any, traj: Any, frame: int | None,
                         source: str | None, *, arc_used: bool = True) -> list[str]:
    """What is wrong with the frame the operator picked (PLAN §5, risk R2)."""
    w: list[str] = []
    if frame is None:
        w.append("no contact frame is chosen (PLAN §5 D4: the operator confirms or nudges the "
                 "auto-suggestion before a distance is reported)")
        return w
    if source == P_MANUAL:
        w.append("point_source=manual: this distance comes from an operator click, not from the "
                 "detector (PLAN §5's override path)")
    arc_ok = getattr(contact_result, "arc_is_freefall", None)
    if arc_ok is False:
        w.append(
            "the fitted arc is NOT a free fall, so the chosen frame is a proxy rather than a "
            f"measured water entry: {getattr(contact_result, 'arc_reason', '')} "
            "(this is the whole corpus as of 2026-10-08 — HANDOFF open item 9)"
        )
    last_arc = getattr(traj, "last_arc_frame", None)
    if last_arc is not None and int(frame) > int(last_arc):
        w.append(f"the chosen frame {int(frame)} is {int(frame) - int(last_arc)} frames past the "
                 f"last arc sample ({int(last_arc)}) — an aftermath detection, so the measured "
                 f"point is not a point the dog's flight ever occupied")
    if not arc_used and getattr(traj, "coeffs", None):
        support = getattr(traj, "airborne_frame_range", None)
        w.append(
            f"frame-quantisation refused: the chosen frame is outside the fitted arc's observed "
            f"support ({support}), so the arc's slope there is extrapolation rather than a fit "
            f"(measured: it would have reported a frame term of half a pool per frame). "
            f"sigma_frame is therefore missing, and sigma_in is a LOWER bound (PLAN §4.4)"
        )
    return w
