"""Contact — water-entry frame candidates + reasons (M5 deliverable, PLAN.md §5).

PLAN §5 ``contact.py``, verbatim:

    Ranked, each with a plain-language reason shown in the GUI:

    1. **Last accepted arc sample** — after splash the tailset is occluded/confused;
       the last confident detection is a strong contact proxy. (baseline, cheap,
       explainable)
    2. **Parabola extrapolation to the waterline** — solve the fitted arc for
       ``y = waterline.y``; gives the predicted crossing frame, correcting for 1–3
       frames lost to splash occlusion.  Reported as a *separate* candidate because
       it is a model prediction, not an observation.
    3. **Confidence collapse** — frame where ``tailset conf`` or ``box conf`` drops
       below threshold for >= 2 consecutive frames while the arc was still descending.
    4. **Vertical-velocity reversal** — last frame before ``dy/dt`` departs from the
       free-fall fit.

    GUI shows all four (frame numbers + one-line reasons), preselects #1, and lets the
    user nudge by ±1/±5/±10 frames or scrub.  The chosen frame and which candidate it
    came from go into ``result.json``.

This module builds that ranked set, deterministically, with the evidence for each
candidate carried alongside it.  It makes no judgement about which is right — that is
the operator's (PLAN §10: "no automatic scoring without human confirmation").

The module also screens the arc, and this is a deliberate addition to PLAN §5
---------------------------------------------------------------------------
Three of the four candidates are only meaningful if the fitted arc is a **free-fall
arc**.  In image coordinates (y down) gravity pulls the tailset *down*, so the
fitted ``y(t)`` quadratic must have **positive curvature** (``cy2 > 0``) and the
branch that reaches the water is the descending one (``t > apex``).  If ``cy2 <= 0``
the arc is not a fall: #2 has no descending branch to solve, #3's "while the arc was
still descending" never holds, and #4 compares ``dy/dt`` against a model that is not
a free fall.  Measured on the three warm caches, every one of them gives ``cy2 <= 0``
(see ``test_m5_contact.py``), so on this corpus the honest answer for #2/#3/#4 is
*"unavailable, here is why"* — not a plausible-looking frame number.

The screen is a screening test, not a proof: under strong perspective the image
curve of a projectile is a conic rather than a quadratic, so the sign of the fitted
curvature is evidence, not a theorem.  It exists to stop the module emitting a
confident contact frame from an arc that cannot be one.

Candidate #3 uses the KEYPOINT confidence, not the box confidence
-----------------------------------------------------------------
PLAN §5 #3 says "``tailset conf`` **or** ``box conf`` drops below threshold".  On the
vendored fine-tune the box confidence never reaches its own tracker threshold: over
the full 18-07 clip it runs 0.121/0.141/0.250 (min/median/max, measured by M3), so a
box-confidence trigger fires on essentially every frame and is not a signal about the
dog.  The trigger is therefore implemented but **off by default**
(``ContactConfig.box_conf_floor = None``); the keypoint confidence is the signal,
exactly as M3's accept rule and the reference recipe gate on ``tail[2] > 0.3``.
Set the floor explicitly to re-enable PLAN's literal OR.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Any, Sequence

import numpy as np

from . import trajectory as _trajectory
from .config import (
    CONTACT_BOX_CONF_FLOOR,
    CONTACT_COLLAPSE_RUN,
    CONTACT_FREEFALL_CURVATURE_MIN,
    CONTACT_KP_CONF,
    CONTACT_MAX_EXTRAP_FRAMES,
    CONTACT_MIN_DESCENT_PX_PER_FRAME,
    CONTACT_VEL_TOL_PX,
)

# candidate sources — the strings that land in result.json
S_LAST = "last_accepted"
S_ARC = "arc_waterline"
S_COLLAPSE = "conf_collapse"
S_VEL = "velocity_reversal"

RANK = {S_LAST: 1, S_ARC: 2, S_COLLAPSE: 3, S_VEL: 4}
ORDER = (S_LAST, S_ARC, S_COLLAPSE, S_VEL)

# PLAN §5: "lets the user nudge by ±1/±5/±10 frames or scrub"
SCRUB_STEPS: tuple[int, ...] = (1, 5, 10)


# --- configuration ----------------------------------------------------------
@dataclass(frozen=True)
class ContactConfig:
    """M5 knobs.  PLAN §6 lists none, so they live here next to ``ARC_OUTLIER_PX``."""

    kp_conf: float = CONTACT_KP_CONF
    collapse_run: int = CONTACT_COLLAPSE_RUN
    vel_tol_px: float = CONTACT_VEL_TOL_PX
    freefall_curvature_min: float = CONTACT_FREEFALL_CURVATURE_MIN
    min_descent_px_per_frame: float = CONTACT_MIN_DESCENT_PX_PER_FRAME
    max_extrap_frames: int = CONTACT_MAX_EXTRAP_FRAMES
    box_conf_floor: float | None = CONTACT_BOX_CONF_FLOOR

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


# --- arc geometry -----------------------------------------------------------
def arc_apex_frame(coeffs: dict) -> float | None:
    """Frame of the fitted arc's image-y extremum (``None`` when ``cy2 == 0``).

    For a free-fall arc (``cy2 > 0``) this is the *apex*: the highest point of the
    jump, and the boundary between the rising and the descending branch.
    """
    if not coeffs:
        return None
    a2, a1, _ = (float(v) for v in coeffs["y"])
    if a2 == 0.0:
        return None
    return float(coeffs["t_ref"]) - a1 / (2.0 * a2)


def arc_slope(coeffs: dict, frame: float) -> float:
    """Fitted ``dy/dframe`` at ``frame``.  Negative = rising, positive = falling."""
    a2, a1, _ = (float(v) for v in coeffs["y"])
    return 2.0 * a2 * (float(frame) - float(coeffs["t_ref"])) + a1


def arc_is_freefall(traj: Any, cfg: ContactConfig | None = None) -> tuple[bool, str, dict]:
    """Screen: does the fitted arc have the curvature a free fall must have?

    Returns ``(ok, reason, evidence)``.  ``evidence`` always carries the numbers, so a
    reader can see *why* rather than trust a boolean.
    """
    cfg = cfg or ContactConfig()
    coeffs = getattr(traj, "coeffs", None)
    ys = [float(p["y"]) for p in getattr(traj, "flight_points", []) or []]
    ev: dict[str, Any] = {
        "cy2": float(coeffs["y"][0]) if coeffs else None,
        "cx2": float(coeffs["x"][0]) if coeffs else None,
        "n_flight_points": len(getattr(traj, "flight_points", []) or []),
        "y_span_px": (max(ys) - min(ys)) if ys else None,
        "apex_frame": arc_apex_frame(coeffs) if coeffs else None,
        "curvature_min": float(cfg.freefall_curvature_min),
    }
    if not coeffs:
        ev["model_slopes_px_per_frame"] = []
        return False, "no fitted arc (fewer than three accepted points in one run)", ev

    cy2 = float(coeffs["y"][0])
    fp = getattr(traj, "flight_points", []) or []
    ev["model_slopes_px_per_frame"] = [round(arc_slope(coeffs, p["frame"]), 3) for p in fp]
    if cy2 <= float(cfg.freefall_curvature_min):
        return (
            False,
            f"fitted arc has no descending branch: y-curvature cy2 = {cy2:.4f} "
            f"(needs > {cfg.freefall_curvature_min:g} for image y to turn from rising to falling "
            f"with y growing downward) — PLAN §5 #2/#3/#4 are each defined on a descending arc, "
            f"so none of them can report a water-entry frame here",
            ev,
        )
    return True, f"arc turns from rising to falling: cy2 = {cy2:.4f} > " \
                 f"{cfg.freefall_curvature_min:g}, image-y minimum (apex) at frame " \
                 f"{ev['apex_frame']:.1f}", ev


# --- the candidate set ------------------------------------------------------
@dataclass
class Candidate:
    """One ranked water-entry candidate, with its reason and its evidence."""

    source: str
    frame: int | None
    reason: str
    available: bool = False
    confident: bool = False
    evidence: dict[str, Any] = field(default_factory=dict)

    @property
    def rank(self) -> int:
        return RANK[self.source]

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["rank"] = self.rank
        return d


@dataclass
class ContactResult:
    """PLAN §5's ranked candidate set + the operator's pick."""

    clip_id: str
    candidates: list[Candidate]
    chosen_frame: int | None
    chosen_source: str
    waterline_y: float | None
    arc_is_freefall: bool
    arc_reason: str
    arc_evidence: dict[str, Any]
    window: list[int] | None
    n_frames: int | None = None
    cfg: dict[str, Any] = field(default_factory=dict)

    # -- access ------------------------------------------------------------
    def by_source(self, source: str) -> Candidate:
        for c in self.candidates:
            if c.source == source:
                return c
        raise KeyError(source)

    def ranked(self) -> list[Candidate]:
        """The four candidates in PLAN §5 rank order (1..4)."""
        return sorted(self.candidates, key=lambda c: c.rank)

    def available_frames(self) -> list[int]:
        return [c.frame for c in self.candidates if c.available and c.frame is not None]

    def brackets(self, frame: float) -> bool:
        """Is ``frame`` inside the span of the *available* candidates?  (PLAN §11 M5.)"""
        f = self.available_frames()
        return bool(f) and min(f) <= float(frame) <= max(f)

    # -- the operator's pick ----------------------------------------------
    def choose(self, frame: int, source: str = "manual") -> "ContactResult":
        """Record the operator's frame (PLAN §5: chosen frame + which candidate)."""
        return replace(self, chosen_frame=nudge(frame, 0, self.n_frames), chosen_source=source)

    def nudge(self, delta: int) -> "ContactResult":
        """±1 / ±5 / ±10 nudge of the current pick, clamped to the clip."""
        if self.chosen_frame is None:
            return self
        return self.choose(nudge(self.chosen_frame, delta, self.n_frames), self.chosen_source)

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["candidates"] = [c.as_dict() for c in self.ranked()]
        return d


def nudge(frame: int, delta: int, n_frames: int | None = None) -> int:
    """Clamp a nudged frame into ``[0, n_frames - 1]`` (``n_frames`` unknown -> unbounded)."""
    f = int(round(float(frame))) + int(delta)
    if n_frames is not None:
        f = max(0, min(int(n_frames) - 1, f))
    return f


# --- the four rules ---------------------------------------------------------
def _first_collapse(frames_desc: Sequence[int], by_frame: dict[int, dict], cfg: ContactConfig
                    ) -> tuple[int | None, dict[str, Any]]:
    """First frame of the first ``collapse_run``-long confidence gap in ``frames_desc``."""
    run = 0
    start: int | None = None
    weakest: float | None = None
    for f in frames_desc:
        r = by_frame.get(f)
        kc = None
        if r is not None:
            ts = r.get("tailset")
            kc = float(ts[2]) if ts and ts[2] is not None else None
        weak = kc is None or kc <= cfg.kp_conf
        if cfg.box_conf_floor is not None:
            bc = r.get("box_conf") if r is not None else None
            weak = weak or (bc is not None and float(bc) <= cfg.box_conf_floor)
        if weak:
            if run == 0:
                start = f
                weakest = kc
            run += 1
            if run >= cfg.collapse_run:
                return start, {"collapse_start": start, "collapse_end": f,
                               "run_length": run, "kp_conf_at_start": weakest,
                               "kp_conf_threshold": float(cfg.kp_conf),
                               "box_conf_floor": cfg.box_conf_floor}
        else:
            run = 0
    return None, {"collapse_start": None, "kp_conf_threshold": float(cfg.kp_conf),
                  "box_conf_floor": cfg.box_conf_floor}


def _first_departure(traj: Any, coeffs: dict, cfg: ContactConfig
                     ) -> tuple[int | None, dict[str, Any]]:
    """First frame whose next observed ``dy/dt`` departs from the fitted free fall."""
    fp = list(getattr(traj, "flight_points", []) or [])
    worst: dict[str, Any] = {"vel_tol_px": float(cfg.vel_tol_px), "n_pairs": max(len(fp) - 1, 0)}
    for i in range(len(fp) - 1):
        f1, f2 = int(fp[i]["frame"]), int(fp[i + 1]["frame"])
        dt = f2 - f1
        if dt <= 0:
            continue
        obs = (float(fp[i + 1]["y"]) - float(fp[i]["y"])) / dt
        mod = arc_slope(coeffs, (f1 + f2) / 2.0)
        if abs(obs - mod) > float(cfg.vel_tol_px):
            worst.update({"departure_frame": f1, "next_frame": f2,
                          "observed_dy_per_frame": obs, "model_dy_per_frame": mod,
                          "deviation_px_per_frame": obs - mod})
            return f1, worst
    worst.update({"departure_frame": None})
    return None, worst


def contact_candidates(records: Sequence[dict] | Any, traj: Any = None,
                       waterline_y: float | None = None, cfg: ContactConfig | None = None,
                       clip_id: str | None = None, kp_conf: float | None = None,
                       n_frames: int | None = None) -> ContactResult:
    """Build PLAN §5's ranked water-entry candidates.

    ``records`` is detect's record list (or a ``DetectResult``); ``traj`` a
    :class:`dog_diving.trajectory.Trajectory` — built here from ``records`` when omitted.
    ``waterline_y`` is the calibration's water row (PLAN §5 #2 needs it); without it
    that candidate is reported unavailable rather than guessed.
    """
    if cfg is None:
        cfg = ContactConfig()
    if kp_conf is not None:
        cfg = replace(cfg, kp_conf=float(kp_conf))
    recs = list(getattr(records, "records", records) or [])
    if traj is None:
        traj = _trajectory.build_trajectory(recs, clip_id=clip_id)
    if clip_id is None:
        clip_id = str(getattr(traj, "clip_id", "") or "")
    if n_frames is None:
        # only a DetectResult knows the clip's length; guessing it from the last
        # detection would silently clamp a reviewer's scrub to the last detection.
        n_frames = getattr(records, "n_frames_clip", None)

    ok_ff, reason_ff, ev_ff = arc_is_freefall(traj, cfg)
    coeffs = getattr(traj, "coeffs", None)
    apex = arc_apex_frame(coeffs) if coeffs else None
    by_frame = {int(r["frame"]): r for r in recs}
    flight = list(getattr(traj, "flight_points", []) or [])
    last_arc_frame = int(flight[-1]["frame"]) if flight else None

    cands: list[Candidate] = []

    # 1. last accepted arc sample (never conditional: it is the baseline)
    last = getattr(traj, "last_contact_candidate", None)
    if last is None:
        cands.append(Candidate(S_LAST, None, "no accepted tailset detection in this clip",
                               available=False, confident=False, evidence={}))
    else:
        f = int(last["frame"])
        inside = bool(getattr(traj, "airborne_frame_range", None)
                      and traj.airborne_frame_range[0] <= f <= traj.airborne_frame_range[1])
        n_acc = len(getattr(traj, "points", []) or [])
        evidence_c1 = {"x": float(last["x"]), "y": float(last["y"]),
                       "kp_conf": float(last["kp_conf"]), "n_accepted": n_acc,
                       "inside_flight_run": inside,
                       "last_arc_frame": last_arc_frame,
                       "frames_after_last_arc": (f - last_arc_frame)
                       if last_arc_frame is not None else None}
        cands.append(Candidate(
            S_LAST, f,
            f"last accepted tailset detection (frame {f}, keypoint conf "
            f"{float(last['kp_conf']):.2f})"
            + ("; inside the fitted flight run"
               if inside or last_arc_frame is None
               else f"; {f - last_arc_frame} frames AFTER the last arc sample "
                    f"({last_arc_frame}) — an aftermath detection, not an arc sample")
            + "; the tailset is occluded or confused once the splash starts, so this proxy is "
              "systematically EARLY",
            available=True, confident=True,
            evidence=evidence_c1,
        ))

    # 2. arc extrapolated to the waterline
    if waterline_y is None:
        cands.append(Candidate(
            S_ARC, None,
            "unavailable: no waterline — PLAN §5 #2 solves y(t) = waterline.y, which comes "
            "from the M2 calibration (data/calib/); none exists for this clip",
            available=False, confident=False, evidence={"waterline_y": None}))
    elif not ok_ff:
        cands.append(Candidate(
            S_ARC, None, f"unavailable: {reason_ff}", available=False, confident=False,
            evidence=dict(ev_ff, waterline_y=float(waterline_y))))
    else:
        t_min = float(apex)
        t_cross = _trajectory.solve_frame_for_y(coeffs, float(waterline_y), t_min=t_min)
        y0 = float(np.polyval(coeffs["y"], float(flight[0]["frame"]) - coeffs["t_ref"])) \
            if flight else None
        if t_cross is None:
            cands.append(Candidate(
                S_ARC, None,
                f"unavailable: the fitted arc never reaches y = {float(waterline_y):.0f} on its "
                f"descending branch", available=False, confident=False,
                evidence={"waterline_y": float(waterline_y), "y_at_flight_start": y0}))
        else:
            f = int(round(t_cross))
            extrap = (t_cross - last_arc_frame) if last_arc_frame is not None else None
            confident = (extrap is None) or (-2.0 <= extrap <= float(cfg.max_extrap_frames))
            cands.append(Candidate(
                S_ARC, f,
                f"fitted arc crosses the waterline y = {float(waterline_y):.0f} on the descending "
                f"branch at frame {t_cross:.1f}"
                + (f" ({extrap:+.1f} frames past the last observation — a model prediction, not an "
                   f"observation)" if extrap is not None else ""),
                available=True, confident=confident,
                evidence={"waterline_y": float(waterline_y), "crossing_frame": float(t_cross),
                          "apex_frame": apex, "extrap_frames": extrap,
                          "extrap_max": int(cfg.max_extrap_frames),
                          "y_at_flight_start": y0,
                          "x_at_crossing": float(_trajectory.predict(coeffs, t_cross)[0])}))

    # 3. confidence collapse while still descending
    if not ok_ff or apex is None or last_arc_frame is None:
        why = reason_ff if not ok_ff else ("no fitted arc" if apex is None else "no observed arc")
        cands.append(Candidate(S_COLLAPSE, None, f"unavailable: {why}",
                               available=False, confident=False, evidence=dict(ev_ff)))
    else:
        f_end = int(round(apex)) + max(int(cfg.max_extrap_frames), 1)
        if n_frames:
            f_end = min(f_end, int(n_frames) - 1)
        first = int(np.ceil(apex))
        f_start = max(first, 0)
        frame, ev = _first_collapse(range(f_start, f_end + 1), by_frame, cfg)
        if frame is None:
            cands.append(Candidate(
                S_COLLAPSE, None,
                f"no {cfg.collapse_run}-frame keypoint-confidence collapse in frames "
                f"{f_start}..{f_end}", available=False, confident=False, evidence=ev))
        else:
            slope = arc_slope(coeffs, frame)
            ev.update({"apex_frame": apex, "slope_at_collapse_px_per_frame": slope,
                       "min_descent_px_per_frame": float(cfg.min_descent_px_per_frame)})
            if slope < float(cfg.min_descent_px_per_frame):
                cands.append(Candidate(
                    S_COLLAPSE, None,
                    f"unavailable: the fitted arc is not descending at frame {frame} "
                    f"(dy/dt = {slope:.2f} px/frame < {cfg.min_descent_px_per_frame:g}), so this "
                    f"confidence gap is not a water entry", available=False, confident=False,
                    evidence=ev))
            else:
                cands.append(Candidate(
                    S_COLLAPSE, int(frame),
                    f"keypoint confidence collapses for {cfg.collapse_run} consecutive frames "
                    f"starting at frame {frame} (<= {cfg.kp_conf:g}) while the fitted arc is "
                    f"still descending (dy/dt = {slope:.1f} px/frame)", available=True,
                    confident=True, evidence=ev))

    # 4. vertical-velocity departure from the free-fall fit
    if not ok_ff or coeffs is None:
        cands.append(Candidate(S_VEL, None, f"unavailable: {reason_ff}", available=False,
                               confident=False, evidence=dict(ev_ff)))
    else:
        frame, ev = _first_departure(traj, coeffs, cfg)
        if frame is None:
            f = last_arc_frame
            cands.append(Candidate(
                S_VEL, f,
                f"the observed dy/dt stays within {cfg.vel_tol_px:g} px/frame of the free-fall fit "
                f"across the whole observed run, so the last frame before departure is the last "
                f"observed frame {f}", available=True, confident=False, evidence=ev))
        else:
            cands.append(Candidate(
                S_VEL, int(frame),
                f"observed dy/dt departs from the free-fall fit after frame {frame} "
                f"(observed {ev['observed_dy_per_frame']:+.1f} vs model "
                f"{ev['model_dy_per_frame']:+.1f} px/frame, off by "
                f"{ev['deviation_px_per_frame']:+.1f})", available=True, confident=True,
                evidence=ev))

    avail = [c.frame for c in cands if c.available and c.frame is not None]
    conf = [c.frame for c in cands if c.confident and c.frame is not None]
    window = [min(conf), max(conf)] if conf else ([min(avail), max(avail)] if avail else None)

    # PLAN §5: preselect #1
    c1 = next(c for c in cands if c.source == S_LAST)
    return ContactResult(
        clip_id=clip_id, candidates=cands, chosen_frame=c1.frame, chosen_source=S_LAST,
        waterline_y=(None if waterline_y is None else float(waterline_y)),
        arc_is_freefall=ok_ff, arc_reason=reason_ff, arc_evidence=ev_ff,
        window=window, n_frames=n_frames, cfg=cfg.as_dict(),
    )


# --- overlay (the GUI's frame ticks; annotate.py owns the full frame in M7) --
def draw_candidates(frame, result: ContactResult, thickness: int = 2):
    """Label each candidate frame number on ``frame`` (the GUI's "shows all four").

    Available candidates are drawn in their rank colour; unavailable ones are listed
    with their short reason so the operator sees *why* the list is short rather than
    silently getting fewer options.
    """
    import cv2  # local: cv2 is only needed to draw

    out = frame.copy()
    h = out.shape[0]
    colours = {S_LAST: (0, 255, 255), S_ARC: (0, 200, 0),
               S_COLLAPSE: (255, 160, 0), S_VEL: (255, 0, 255)}
    y = 30
    for c in result.ranked():
        if c.available and c.frame is not None:
            txt = f"#{c.rank} {c.source} f{c.frame}"
        else:
            txt = f"#{c.rank} {c.source} n/a"
        cv2.putText(out, txt, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colours[c.source], thickness)
        y += 30
    if result.chosen_frame is not None:
        cv2.putText(out, f"chosen f{result.chosen_frame} ({result.chosen_source})",
                    (12, y + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), thickness + 1)
        cv2.line(out, (0, h - 12), (out.shape[1], h - 12), (0, 0, 255), thickness)
    return out
