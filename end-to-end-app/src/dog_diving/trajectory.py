"""Trajectory — assemble accepted tailset points, trim to the flight, robust parabola.

M4 deliverable (PLAN.md §5 ``trajectory.py``).

What this module is for
-----------------------
The per-frame records from :mod:`dog_diving.detect` are a *set of detections*, not a
trajectory.  A clip also contains the dog standing on the dock, the handler, and
the aftermath in the water — all of which this checkpoint happily gives confident
tailset keypoints to (see ``test_m4_trajectory.py``).  This module turns the
accepted detections into a *flight arc* that the rest of the pipeline can measure:

    assemble -> segment into temporal runs -> pick the flight run -> robust
    parabola fit inside it, dropping points further than ``arc_outlier_px``

PLAN §5 verbatim: *"Assemble accepted tailset points (dominant track only). Trim:
drop frames before the dog leaves the dock (first sustained upward motion) and
after entry. Outlier rejection: robust parabola fit (RANSAC or iteratively-
reweighted least squares) in image space; drop points with residual >
``cfg.arc_outlier_px`` (default 20 px)."*  Outputs it names: ``points[]``,
``inliers[]``, ``parabola coeffs``, ``arc_rms_px``, ``airborne_frame_range``,
``n_before_entry``, plus the *last accepted tailset frame* kept as
``last_contact_candidate`` for ``contact.py``.

The fit is in the TIME domain, deliberately
-------------------------------------------
The arc is fitted as a pair of quadratics in the frame index ``t``::

    x(t) = cx2·tc² + cx1·tc + cx0
    y(t) = cy2·tc² + cy1·tc + cy0          tc = t - t_ref

*Why not ``y = a·x² + b·x + c``:* the image x-coordinate of a jump is **not
monotonic** on this corpus (18-07 runs x 1579→1127 while y 885→383), so ``y(x)``
is multi-valued and un-fittable.  Frames are uniformly spaced in time, so a
quadratic in ``t`` *is* the free-fall model in image space, and it is the only
form that can be solved for a waterline-crossing **frame** — which is exactly
what ``contact.py`` candidate #2 needs (PLAN §5).

*Why ``tc`` is centred:* fitting a quadratic against raw frame indices in the
hundreds is numerically ill-conditioned.  ``np.polyfit`` on t=369…396 emits
``RankWarning: Polyfit may be poorly conditioned`` and returns coefficient values
that are unusable; centring on the run's mean frame fixes it.  ``t_ref`` is
carried in the output so a coefficient triple is always interpretable.  Pinned by
``test_m4_trajectory.py::test_centring_is_required``.

Trimming on a multi-event clip (fifth measured reality, not a PLAN correction)
-------------------------------------------------------------------------------
PLAN §5's trim assumes **one** jump per clip.  18-07.mp4 does not honour that: its
30 accepted frames fall into four temporal runs — 6-9, 342-347, 369-396, 428-437 —
and the *whole-clip* quadratic through all 30 has an RMS of **206 px**, i.e. a
single arc explains nothing.  (Frames 6-9 and 428-437 are two different handlers'
landings; 342-396 is the third event, where the keypoint lands in empty air beside
the handler rather than on a flying dog.)  So this module:

1. splits the accepted points into runs at any frame gap > ``max_frame_gap``
   (default 10; the choice is *not* knife-edge — every gap from 6 to 21 gives the
   same four runs on this clip, because within an event the gaps are <= 6 frames
   and between events they are >= 22);
2. takes the **flight run** = the run with the most points (tie-break: longest
   frame span, then lowest inlier RMS).  Measured on 18-07: the runs hold
   3 / 3 / **15** / 9 points, so the flight is 369-396.

This is a deterministic reading of "before the dog leaves the dock / after entry"
that still works when a clip holds more than one event.  It is recorded as an open
item in HANDOFF.md, not as a silent reinterpretation of PLAN.

Nothing is ever deleted
-----------------------
``points`` keeps *every* assembled accepted detection; trimming and outlier
rejection only *classify*.  ``inliers`` is the arc subset, ``n_before_leave`` /
``n_before_entry`` count the accepted frames outside it, and ``last_contact_candidate``
is the last accepted frame overall — PLAN §5 requires it survive into the output so
``contact.py`` candidate #1 can use it even when it sits past the fitted arc.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np

from .config import ARC_OUTLIER_PX, TRAJ_MAX_FRAME_GAP, TRAJ_SEED

# PLAN §6 does not list a trajectory section; these two live in config.py next to
# ARC_OUTLIER_PX (which PLAN §6 *does* list, default 20).
DEFAULT_RANSAC_ITERS = 3000


# --- configuration ----------------------------------------------------------
@dataclass(frozen=True)
class TrajConfig:
    """Knobs for arc assembly + cleaning.  ``arc_outlier_px`` is PLAN §6's key."""

    arc_outlier_px: float = ARC_OUTLIER_PX
    max_frame_gap: int = TRAJ_MAX_FRAME_GAP
    min_run_points: int = 3          # a quadratic needs three points
    ransac_iters: int = DEFAULT_RANSAC_ITERS
    seed: int = TRAJ_SEED            # fixed -> the RANSAC result is reproducible

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


# --- points -----------------------------------------------------------------
def _point_from_record(r: dict, kp_conf: float | None) -> dict | None:
    """A single accepted detection -> ``{frame, x, y, kp_conf}`` (or ``None``)."""
    ts = r.get("tailset")
    if not ts:
        return None
    accepted = r.get("accepted")
    if accepted is None:
        accepted = (kp_conf is None) or (float(ts[2]) > float(kp_conf))
    if not accepted:
        return None
    p = {"frame": int(r["frame"]), "x": float(ts[0]), "y": float(ts[1]),
         "kp_conf": float(ts[2])}
    # Provenance from detect's adaptive ladder — ONLY for rows the ladder judged, so a
    # fixed-threshold run's points keep exactly {frame, x, y, kp_conf} and a plain run's
    # trajectory.json stays byte-identical (PLAN §11).  Pinned by test_m4_trajectory.py's schema
    # test (plain) and test_m16_detect_ladder.py (ladder).
    if r.get("conf_used") is not None:
        p["box_conf"] = None if r.get("box_conf") is None else float(r["box_conf"])
        p["conf_used"] = float(r["conf_used"])
        p["kp_conf_used"] = float(r["kp_conf_used"])
        p["low_conf"] = bool(r.get("low_conf"))
    return p


def assemble(source: Any, kp_conf: float | None = None) -> list[dict]:
    """Accepted detections -> frame-ordered points.

    ``source`` is a :class:`dog_diving.detect.DetectResult` (its ``records`` are
    used), a sequence of detect records, or a sequence of already-assembled
    points.  Honours detect's ``accepted`` flag when present; a record without one
    is accepted iff ``kp_conf`` is ``None`` or its keypoint confidence exceeds it
    — this is how the vendored reference rule (``tail[2] > 0.3``) is replayed.
    """
    recs = getattr(source, "records", source)
    out: list[dict] = []
    for r in recs:
        if "tailset" in r:
            p = _point_from_record(r, kp_conf)
        elif "x" in r and "y" in r:
            p = {"frame": int(r["frame"]), "x": float(r["x"]), "y": float(r["y"]),
                 "kp_conf": float(r.get("kp_conf", float("nan")))}
        else:
            p = None
        if p is not None:
            out.append(p)
    out.sort(key=lambda p: p["frame"])
    return out


def temporal_runs(points: Sequence[dict], max_frame_gap: int = TRAJ_MAX_FRAME_GAP
                  ) -> list[tuple[int, int]]:
    """Index runs of points whose consecutive frames differ by <= ``max_frame_gap``.

    Returns half-open ``(i0, i1)`` index pairs into ``points``.
    """
    if not points:
        return []
    runs: list[tuple[int, int]] = []
    start = 0
    for i in range(1, len(points)):
        if points[i]["frame"] - points[i - 1]["frame"] > max_frame_gap:
            runs.append((start, i))
            start = i
    runs.append((start, len(points)))
    return runs


# --- fitting ----------------------------------------------------------------
def _quad(tc: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Quadratic least-squares coefficients ``[a2, a1, a0]`` in centred time."""
    return np.polyfit(tc, v, 2)


def _residuals(cx: np.ndarray, cy: np.ndarray, tc: np.ndarray,
               x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.sqrt((y - np.polyval(cy, tc)) ** 2 + (x - np.polyval(cx, tc)) ** 2)


def robust_parabola(t: np.ndarray, x: np.ndarray, y: np.ndarray, thr: float,
                    iters: int = DEFAULT_RANSAC_ITERS, seed: int = TRAJ_SEED
                    ) -> dict[str, Any] | None:
    """Robust (RANSAC + least-squares refinement) quadratic fit in centred time.

    ``t`` are absolute frame indices; the fit is done against ``tc = t - mean(t)``
    and ``t_ref`` is returned so the coefficients stay interpretable.  Inliers are
    points within ``thr`` px (Euclidean, image space) of the fitted curve.

    Returns ``{t_ref, x:[a2,a1,a0], y:[a2,a1,a0], inliers:bool[], residuals:[...],
    arc_rms_px, max_inlier_res_px, n_inliers}`` or ``None`` when fewer than three
    distinct frames are supplied (a quadratic is not determined).
    """
    t = np.asarray(t, float)
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = len(t)
    if n < 3 or len(set(t.tolist())) < 3:
        return None

    t_ref = float(t.mean())
    tc = t - t_ref

    if n == 3:                                    # exactly determined
        cx, cy = _quad(tc, x), _quad(tc, y)
        mask = np.ones(n, bool)
    else:
        rng = np.random.default_rng(seed)
        best: tuple[int, float, np.ndarray] | None = None
        for _ in range(int(iters)):
            idx = rng.choice(n, 3, replace=False)
            cx_s, cy_s = _quad(tc[idx], x[idx]), _quad(tc[idx], y[idx])
            r = _residuals(cx_s, cy_s, tc, x, y)
            m = r <= thr
            c = int(m.sum())
            if c == 0:
                continue
            rms = float(np.sqrt((r[m] ** 2).mean()))
            if best is None or c > best[0] or (c == best[0] and rms < best[1]):
                best = (c, rms, m.copy())
        if best is None:
            return None
        mask = best[2]
        # refinement: re-fit on the inliers until the inlier set stops growing
        for _ in range(50):
            cx, cy = _quad(tc[mask], x[mask]), _quad(tc[mask], y[mask])
            r = _residuals(cx, cy, tc, x, y)
            m2 = r <= thr
            if int(m2.sum()) == int(mask.sum()):
                mask = m2
                break
            mask = m2

    cx, cy = _quad(tc[mask], x[mask]), _quad(tc[mask], y[mask])
    res = _residuals(cx, cy, tc, x, y)
    in_res = res[mask]
    return {
        "t_ref": t_ref,
        "x": [float(v) for v in cx],
        "y": [float(v) for v in cy],
        "inliers": [bool(v) for v in mask],
        "residuals": [float(v) for v in res],
        "n_inliers": int(mask.sum()),
        "arc_rms_px": float(np.sqrt((in_res ** 2).mean())) if in_res.size else float("nan"),
        "max_inlier_res_px": float(in_res.max()) if in_res.size else float("nan"),
        "arc_rms_all_px": float(np.sqrt((res ** 2).mean())),
    }


def predict(coeffs: dict, frame: float) -> tuple[float, float]:
    """``(x, y)`` on the fitted curve at an absolute ``frame``."""
    tc = float(frame) - float(coeffs["t_ref"])
    return float(np.polyval(coeffs["x"], tc)), float(np.polyval(coeffs["y"], tc))


def solve_frame_for_y(coeffs: dict, y_value: float, t_min: float | None = None
                      ) -> float | None:
    """Smallest frame >= ``t_min`` where the arc reaches ``y_value`` (image y).

    The bridge to ``contact.py`` candidate #2 (PLAN §5): solve ``y(t) = waterline.y``.
    Returns ``None`` when the arc never reaches that row.  Image y grows downward,
    so the *first* crossing after ``t_min`` is the descending branch of the arc.
    """
    a2, a1, a0 = (float(v) for v in coeffs["y"])
    t_ref = float(coeffs["t_ref"])
    roots = np.roots([a2, a1, a0 - float(y_value)])
    frames = sorted(float(r.real) + t_ref for r in roots
                    if abs(float(r.imag)) < 1e-6)
    cands = [f for f in frames if t_min is None or f >= float(t_min)]
    return cands[0] if cands else None


# --- the assembled result ---------------------------------------------------
@dataclass
class Trajectory:
    """PLAN §5's ``trajectory.py`` outputs, plus the evidence behind each judgement."""

    clip_id: str
    n_points: int
    points: list[dict]                        # every accepted detection, frame order
    runs: list[dict]                          # [{i0, i1, start_frame, end_frame, n}]
    flight_run: int | None
    airborne_frame_range: list[int] | None
    n_before_leave: int                       # accepted frames before the flight run
    n_before_entry: int                       # accepted frames after the flight run
    coeffs: dict | None
    inliers: list[dict]                       # arc subset (points within arc_outlier_px)
    n_outliers: int
    arc_rms_px: float | None
    arc_rms_all_px: float | None
    max_inlier_res_px: float | None
    residuals: list[float] | None             # per flight-run point, aligned with `flight_points`
    flight_points: list[dict]
    last_arc_frame: int | None
    last_contact_candidate: dict | None
    cfg: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_trajectory(source: Any, cfg: TrajConfig | None = None,
                     clip_id: str | None = None, kp_conf: float | None = None
                     ) -> Trajectory:
    """Full pipeline: assemble -> runs -> flight run -> robust arc fit."""
    cfg = cfg or TrajConfig()
    if clip_id is None:
        clip_id = str(getattr(source, "clip_id", "") or "")
    pts = assemble(source, kp_conf=kp_conf)
    runs_idx = temporal_runs(pts, cfg.max_frame_gap)
    runs = [{"i0": i0, "i1": i1, "start_frame": pts[i0]["frame"],
             "end_frame": pts[i1 - 1]["frame"], "n": i1 - i0} for i0, i1 in runs_idx]

    coeffs = None
    inliers: list[dict] = []
    residuals: list[float] | None = None
    flight_points: list[dict] = []
    flight_run: int | None = None
    airborne: list[int] | None = None
    n_before_leave = n_before_entry = 0

    # the flight run = most points, then longest span, then lowest inlier RMS
    scored = [i for i, (i0, i1) in enumerate(runs_idx)
              if (i1 - i0) >= cfg.min_run_points]
    if scored:
        def _score(i: int):
            i0, i1 = runs_idx[i]
            sub = pts[i0:i1]
            tt = np.array([p["frame"] for p in sub], float)
            xx = np.array([p["x"] for p in sub], float)
            yy = np.array([p["y"] for p in sub], float)
            fit = robust_parabola(tt, xx, yy, cfg.arc_outlier_px, cfg.ransac_iters, cfg.seed)
            return (-(i1 - i0), -(pts[i1 - 1]["frame"] - pts[i0]["frame"]),
                    fit["arc_rms_px"] if fit else float("inf"))

        flight_run = min(scored, key=_score)
        i0, i1 = runs_idx[flight_run]
        flight_points = pts[i0:i1]
        tt = np.array([p["frame"] for p in flight_points], float)
        xx = np.array([p["x"] for p in flight_points], float)
        yy = np.array([p["y"] for p in flight_points], float)
        fit = robust_parabola(tt, xx, yy, cfg.arc_outlier_px, cfg.ransac_iters, cfg.seed)
        if fit is not None:
            coeffs = {k: fit[k] for k in ("t_ref", "x", "y")}
            residuals = fit["residuals"]
            inliers = [p for p, keep in zip(flight_points, fit["inliers"]) if keep]
            n_outliers = len(flight_points) - int(fit["n_inliers"])
            airborne = [flight_points[0]["frame"], flight_points[-1]["frame"]]
            n_before_leave = sum(1 for p in pts if p["frame"] < airborne[0])
            n_before_entry = sum(1 for p in pts if p["frame"] > airborne[1])
            last_arc_frame = inliers[-1]["frame"] if inliers else airborne[1]
            return Trajectory(
                clip_id=clip_id, n_points=len(pts), points=pts, runs=runs,
                flight_run=flight_run, airborne_frame_range=airborne,
                n_before_leave=n_before_leave, n_before_entry=n_before_entry,
                coeffs=coeffs, inliers=inliers, n_outliers=n_outliers,
                arc_rms_px=fit["arc_rms_px"], arc_rms_all_px=fit["arc_rms_all_px"],
                max_inlier_res_px=fit["max_inlier_res_px"], residuals=residuals,
                flight_points=flight_points, last_arc_frame=last_arc_frame,
                last_contact_candidate=(pts[-1] if pts else None), cfg=cfg.as_dict(),
            )

    return Trajectory(
        clip_id=clip_id, n_points=len(pts), points=pts, runs=runs,
        flight_run=flight_run, airborne_frame_range=None, n_before_leave=0,
        n_before_entry=0, coeffs=None, inliers=[], n_outliers=0, arc_rms_px=None,
        arc_rms_all_px=None, max_inlier_res_px=None, residuals=None,
        flight_points=[], last_arc_frame=None,
        last_contact_candidate=(pts[-1] if pts else None), cfg=cfg.as_dict(),
    )


# --- per-point confidence labels ---------------------------------------------
# PLAN §5's yellow polyline is drawn through *every accepted* point, so a clip whose
# accepted set contains arc outliers renders as a zig-zag and there is no way to tell a
# 0.99 keypoint from one scraped in at 0.31 by eye.  Printing each point's keypoint
# confidence makes the difference explicit rather than a matter of taste.
CONF_MIN_SPACING_PX = 24     # suppress a label this close to the previous one; 0 = all
CONF_FONT_SCALE = 0.45


def conf_colour(kp_conf: float) -> tuple[int, int, int]:
    """BGR for one point's confidence: green trusted, orange ok, red marginal."""
    if kp_conf >= 0.90:
        return (0, 220, 0)
    if kp_conf >= 0.70:
        return (0, 165, 255)
    return (0, 0, 255)


def conf_labels(
    points: Any,
    *,
    min_spacing_px: int = CONF_MIN_SPACING_PX,
    digits: int = 2,
) -> list[tuple[int, int, str, tuple[int, int, int]]]:
    """``(x, y, text, colour)`` for every point whose confidence should be printed.

    ``min_spacing_px`` suppresses a label sitting within that many pixels on *both* axes
    of the previous drawn label.  A dog that is hovering, or standing on the dock, yields
    dozens of accepted points in one spot with near-identical confidences, and printing
    all of them overprints into an unreadable blob.  ``0`` labels every point; the count
    of points vs the count of labels is reported by the gate so the thinning is never
    silent.  Points without a finite ``kp_conf`` are skipped rather than labelled ``nan``.
    """
    out: list[tuple[int, int, str, tuple[int, int, int]]] = []
    last: tuple[int, int] | None = None
    for p in points:
        c = p.get("kp_conf")
        if c is None or not np.isfinite(float(c)):
            continue
        c = float(c)
        x, y = int(round(float(p["x"]))), int(round(float(p["y"])))
        if min_spacing_px > 0 and last is not None:
            if abs(x - last[0]) < min_spacing_px and abs(y - last[1]) < min_spacing_px:
                continue
        out.append((x, y, f"{c:.{digits}f}", conf_colour(c)))
        last = (x, y)
    return out


def refit(points: Sequence[dict], cfg: "TrajConfig | None" = None) -> dict[str, Any]:
    """Fit the arc to an explicit point list — Review's include/exclude.

    Review lets the operator click a detection out of the arc (a handler's leg, a splash point, a box
    that latched onto a banner).  This is the same robust fit :func:`build_trajectory` uses, over
    whatever subset survives, so the arc panel and the arc drawn on the frame agree by construction.

    Returns ``{coeffs, inliers, frame_range, n, rms_px}``; ``coeffs`` is ``None`` when the remaining
    points cannot determine a quadratic (fewer than 3 distinct frames) — the caller should then keep
    the previous arc rather than draw a line through nothing.
    """
    cfg = cfg or TrajConfig()
    pts = list(points)
    out: dict[str, Any] = {"coeffs": None, "inliers": [], "frame_range": None, "n": len(pts),
                           "rms_px": None}
    if len(pts) < 3:
        return out
    t = np.array([p["frame"] for p in pts], float)
    x = np.array([p["x"] for p in pts], float)
    y = np.array([p["y"] for p in pts], float)
    fit = robust_parabola(t, x, y, cfg.arc_outlier_px, cfg.ransac_iters, cfg.seed)
    if fit is None:
        return out
    out["coeffs"] = {k: fit[k] for k in ("t_ref", "x", "y")}
    out["inliers"] = [p for p, keep in zip(pts, fit["inliers"]) if keep]
    out["frame_range"] = [int(min(t)), int(max(t))]
    out["rms_px"] = fit["arc_rms_px"]
    return out


# --- overlay (M4's visual check; annotate.py owns the full frame in M7) ------
def draw_overlay(
    frame,
    traj: Trajectory,
    thickness: int = 2,
    *,
    show_conf: bool = True,
    conf_min_spacing_px: int = CONF_MIN_SPACING_PX,
    conf_font_scale: float = CONF_FONT_SCALE,
    arc: dict[str, Any] | None = None,
    excluded: set[int] | tuple[int, ...] = (),
):
    """Draw the track (thin polyline) + the fitted arc, in the plane overlay's weights.

    Operator order (2026-10-21): *"copy the new formatting of the lines instead of the old bulky,
    obstructive lines"*.  The weights follow ``plane.draw_grid_on`` — 1-2 px lines and 3 px handles —
    instead of the vendored reference's ``thickness=4`` polyline with 5 px rings, which on a 960-px
    frame covered the dog it was meant to annotate.  This is therefore **no longer** byte-identical to
    ``draw_tail_trajectory.py``'s output, and the reference comparison in M4 is about geometry, not
    stroke width.

    ``arc`` overrides the fitted curve (Review's clicked-out subset: see :func:`refit`), and
    ``excluded`` marks the points that were taken out so the operator can see what is off the arc.
    """
    import cv2  # local: cv2 is only needed to draw

    out = frame.copy()
    drop = {int(f) for f in (excluded or ())}
    kept = [p for p in traj.points if int(p["frame"]) not in drop]
    pts = [(int(round(p["x"])), int(round(p["y"]))) for p in kept]
    if len(pts) > 1:
        cv2.polylines(out, [np.array(pts, np.int32).reshape(-1, 1, 2)], False,
                      (0, 255, 255), max(1, int(thickness)))

    coeffs = (arc or {}).get("coeffs") if arc is not None else traj.coeffs
    inliers = (arc or {}).get("inliers") if arc is not None else traj.inliers
    frame_range = ((arc or {}).get("frame_range") if arc is not None
                   else traj.airborne_frame_range)
    if coeffs and frame_range:
        f0, f1 = frame_range
        curve = [tuple(int(round(v)) for v in predict(coeffs, f)) for f in range(int(f0), int(f1) + 1)]
        if len(curve) > 1:
            cv2.polylines(out, [np.array(curve, np.int32).reshape(-1, 1, 2)], False,
                          (0, 200, 0), 2)

    for p in inliers or []:
        if int(p["frame"]) in drop:
            continue
        cv2.circle(out, (int(round(p["x"])), int(round(p["y"]))), 3, (255, 0, 255), 1)
    for f in excluded or ():
        for p in traj.points:
            if int(p["frame"]) == int(f):
                c = (int(round(p["x"])), int(round(p["y"])))
                cv2.circle(out, c, 3, (150, 150, 150), 1)          # hollow: out of the arc
                cv2.line(out, (c[0] - 3, c[1]), (c[0] + 3, c[1]), (150, 150, 150), 1)
                break
    if pts:
        cv2.circle(out, pts[-1], 4, (0, 0, 255), -1)

    if show_conf:
        h, w = out.shape[:2]
        for x, y, text, colour in conf_labels(traj.points, min_spacing_px=conf_min_spacing_px):
            if not (0 <= x < w and 0 <= y < h):
                continue
            cv2.circle(out, (x, y), 3, colour, -1)
            tw = int(11 * conf_font_scale * len(text))          # rough text width
            tx = x + 7 if x + 7 + tw < w else max(0, x - 9 - tw)
            ty = y - 6 if y - 6 > 12 else y + 18                # keep on the frame
            cv2.putText(out, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, conf_font_scale,
                        (0, 0, 0), 3)                           # outline: legible on any bg
            cv2.putText(out, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, conf_font_scale,
                        (255, 255, 255), 1)                     # face
    return out
