"""Four waterline edges from Quintin's temporal-noise detector, in one call.

PLAN §17 / ``docs/PLAN_WATERLINE_SEED.md``.  Upstream
(``third_party/waterline/detect_waterline.py``, vendored unmodified) answers one question: **where is
the boundary between water and structure in this calm window?**  It returns *one* boundary per call,
chosen globally by ``water_side``, and it has no notion of which boundary is which.

The app's plane needs four of them (:mod:`dog_diving.pool.seed` turns them into a quad):

===========  ==================  ==========================================================
edge         plane line          what it looks like
===========  ==================  ==========================================================
``far``      ``Y = 0``           water against the **far** wall — per-column top boundary
``near``     ``Y = W``           water against the **near** wall — per-column bottom boundary
``left``     ``X = 0`` or ``X = anchor``  water against the **left** end wall — per-row boundary
``right``    ``X = anchor`` or ``X = 0``  water against the **right** end wall — per-row boundary
``dock``     ``X = 0``           water against the **dock's front face** — per-row boundary
===========  ==================  ==========================================================

Everything upstream provides is reused as-is (noise maps, normalization, Otsu/manual threshold,
morphology, the Huber line fit with its end-vs-centre distortion diagnostic, separability, glare and
drift reports).  What is added here, and why it could not live upstream:

* **both** column boundaries in one pass (upstream's ``scan_boundary`` returns one of them);
* per-row boundaries for the dock contact, which is near-vertical and needs
  ``water_side="left"/"right"`` (upstream's vertical fit exists; its scan does not separate sides);
* **run splitting** before fitting — a lane rope, a glare patch or a shadow interrupts the mask, and a
  single global fit bridges the gap and tilts;
* **classification** of a boundary into far / near / dock, plus the dock **side**, which upstream has
  no concept of;
* **stability**: the same edge re-detected on the two halves of the window — an independent second
  sample, which is the only non-circular check available on this route.

All coordinates returned are in **original video pixels** (the working scale is mapped back), because
that is the space ``PlaneSession`` corners and the pool document live in.
"""

from __future__ import annotations

import os
import math
import time
import dataclasses
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
import hashlib
from typing import Any, Sequence

import cv2
import numpy as np

from third_party.waterline import detect_waterline as wl

#: Working resolution factor.  A 1920x1080 clip at 0.5 is 540x960: the waterline is 2 px wide instead
#: of 4, and the whole pipeline stays inside a few hundred MB instead of a couple of GB.
DEFAULT_SCALE = 0.5

#: Window lengths, in frames.  >= 2 s so a sub-window spans a full ripple period (upstream's own
#: reason for ``auto_stability_window``: 30-frame windows jittered 2.5-5 px, 90-frame ones ~0.5 px).
MIN_WINDOW = 60
MAX_WINDOW = 150
WINDOW_MARGIN = 5

#: Edge acceptance, all in ORIGINAL pixels unless said otherwise.  These are the S1 thresholds.
INLIER_TOL_PX = 3.0
MAX_RMS_PX = 3.0
MIN_INLIER_FRACTION = 0.60
MAX_STABILITY_PX = 3.0
MIN_RUN_POINTS = 20
MIN_DOCK_POINTS = 30
MIN_DOCK_ANGLE_DEG = 55.0

#: A run is broken where the scan index jumps this far, or the boundary jumps this multiple of the
#: median step (both floored, so a clean run is never split by jitter).
MAX_RUN_GAP_PX = 16.0
RUN_JUMP_FACTOR = 4.0

#: How to place the pool's two SHORT ends (the water against each end wall).  ``current`` is the
#: original S1 extreme-pixel row scan; ``A`` and ``B`` are the two routes in
#: ``out/SHORT_EDGE_METHODS.md`` (A: band-consensus occupancy on the water strip, occlusion-first;
#: B: local change point on the continuous noise map, illumination-first).  ``DD_SHORT_EDGE_METHOD``
#: selects one per process so two GUI instances can be compared on the same clip.
SHORT_EDGE_METHODS = ("current", "A", "B", "M1")
#: The DEFAULT is M1 (order 2026-10-28).  ``current`` keeps the near-vertical model, and on this
#: corpus it reads the dock short end as a diagonal at 146 deg — which flips ``dock_side`` to the far
#: end and makes the seed refuse its own geometry gate (reproduced on video004 seg01: "S3 failed …
#: vanishing point x=-5678 vs origin x=478 with the dock on the left").  M1, which fits each short
#: end as a line through one crossing per band, finds the 12.2 deg dock waterline instead.
SHORT_EDGE_METHOD = (os.environ.get("DD_SHORT_EDGE_METHOD") or "M1").strip()

#: Band scan shared by methods A and B.  Bands are laid ABOVE the near waterline (towards the far
#: one), band 0 nearest the near waterline — the band an overhanging banner cannot reach, which is
#: where the domain statement puts the always-visible water/wall contact.
BAND_COUNT = 8
PLATEAU_LEVEL = 0.75          # occupancy that counts as "inside the pool" (method A)
CROSS_LEVEL = 0.50            # the crossing that is called the short edge (method A)
BAND_X_TOL = 20.0             # original px a band's answer may sit from the consensus
MIN_BAND_POINTS = 3           # bands that must agree before a line is claimed
STEP_WINDOW = 18              # columns either side of a candidate change point (method B)
STEP_RATIO_MIN = 1.30         # level ratio that counts as a real step (method B)

#: Method ``M1`` (the four-line fit) gates.  A short edge is a LINE fitted through one boundary
#: crossing per band — measured on video004 seg01 the dock edge is a single line at RMS 5.0 px with
#: dx/dy = 4.71, i.e. 12 deg from horizontal, so "near-vertical" is the wrong test for it.  What *is*
#: disqualifying is an edge parallel to the long waterlines (the far/near boundary continuing) or one
#: with too little support to fit.
MIN_SHORT_CROSSINGS = 3
MIN_SHORT_ANGLE_DEG = 8.0
SHORT_MAX_RMS_PX = 6.0

#: The pool's two SHORT ends (the water against each end wall).  ``dock`` is whichever of these the
#: detector believes X = 0 is on; the other one is the image of X = anchor.

#: Provenance of the vendored detector.  Recorded in every pool document this module seeds, so
#: "which algorithm placed this plane?" is answerable years later, offline.
UPSTREAM = "quintinlevy/Waterline-Detection@e0c412607ea92d557da127b1ac72f791f23efb96"
VENDORED_FILE = (Path(__file__).resolve().parents[1] / "third_party" / "waterline"
                 / "detect_waterline.py")


@lru_cache(maxsize=1)
def vendored_sha256() -> str:
    """sha256 of the vendored ``detect_waterline.py`` (must match third_party's SHA256SUMS)."""
    try:
        return hashlib.sha256(VENDORED_FILE.read_bytes()).hexdigest()
    except OSError:
        return ""


@dataclass
class EdgeFit:
    """One classified waterline edge, in original pixels."""

    name: str
    found: bool = False
    points: np.ndarray | None = None                 # (K, 2) original px
    fit: dict[str, Any] | None = None                # upstream fit_straight_line dict
    inlier_fraction: float = 0.0
    stability_px: float | None = None                # None = not checkable (no second sample)
    reason: str = ""
    #: ``observed`` (a fitted line from agreeing evidence), ``partial`` (fitted to the visible span
    #: only, the rest of the line is the extension of that fit) or ``off_shot`` (no line: the water
    #: runs off the frame at this end).  Only ``observed`` may be gated on as a measurement.
    status: str = "observed"

    @property
    def rms(self) -> float:
        return float(self.fit["rms_residual"]) if self.fit else float("nan")

    @property
    def midpoint(self) -> tuple[float, float] | None:
        if self.points is None or len(self.points) == 0:
            return None
        p = self.points.mean(axis=0)
        return (float(p[0]), float(p[1]))

    @property
    def angle_deg(self) -> float:
        return float(self.fit["angle_deg"]) if self.fit else float("nan")

    def as_dict(self) -> dict[str, Any]:
        if not self.found or self.fit is None:
            return {"found": False, "status": self.status, "reason": self.reason}
        return {"found": True, "status": self.status,
                "n": int(len(self.points)), "rms_px": round(self.rms, 2),
                "p95_px": round(float(self.fit["p95_residual"]), 2),
                "inlier_fraction": round(float(self.inlier_fraction), 3),
                "stability_px": (None if self.stability_px is None
                                 else round(float(self.stability_px), 2)),
                "angle_deg": round(self.angle_deg, 2),
                "slope": round(float(self.fit["slope"]), 6),
                "intercept": round(float(self.fit["intercept"]), 3),
                "endpoints": [[round(float(v), 2) for v in p] for p in self.fit["endpoints"]],
                "rms_center_px": round(float(self.fit["rms_center"]), 2),
                "rms_ends_px": round(float(self.fit["rms_ends"]), 2)}


@dataclass
class WaterlineEdges:
    """Result of one detection: the long edges, both pool ends, and everything needed to judge them.

    ``far``/``near`` are the long waterlines (the images of ``Y = 0`` and ``Y = W``); ``left`` and
    ``right`` are the pool's two SHORT ends — the water against each end wall, one of which is the
    dock.  Operator order (2026-10-21): *"add two new waterlines, the waterline at the short ends of
    the pool"* — the same row scan supplies both, and keeping the one that loses the dock vote is
    what gives the seed a measured line for ``X = anchor`` as well as for ``X = 0``.  ``dock`` is the
    end the detector believes the dock is on (whichever of ``left``/``right``) and keeps its old
    meaning: the image of ``X = 0``.
    """

    far: EdgeFit
    near: EdgeFit
    dock: EdgeFit
    left: EdgeFit | None = None                       # the pool's left short end (left wall)
    right: EdgeFit | None = None                      # the pool's right short end (right wall)
    dock_side: str = "unknown"                       # "left" | "right" | "unknown"
    dock_side_evidence: str = ""
    method: str = "detrended"
    short_method: str = "current"                    # "current" | "A" | "B" — how the short ends were placed
    fisher: float = float("nan")
    fisher_by_method: dict[str, float] = field(default_factory=dict)
    threshold: float = float("nan")
    window: tuple[int, int] = (0, 0)
    window_reason: str = ""
    scale: float = DEFAULT_SCALE
    water_fraction: float = float("nan")
    n_components: int = 0
    components_kept: int = 0
    dominant_share: float = float("nan")
    warnings: list[str] = field(default_factory=list)
    frame_size: tuple[int, int] = (0, 0)             # (w, h) of the ORIGINAL video
    elapsed_s: float = 0.0
    params: dict[str, Any] = field(default_factory=dict)

    @property
    def edges(self) -> tuple[EdgeFit, ...]:
        return (self.far, self.near, self.dock)

    @property
    def ends(self) -> tuple[EdgeFit, ...]:
        """The pool's two short ends (**left**, **right**) — ``dock`` is one of them."""
        return tuple(e for e in (self.left, self.right) if e is not None)

    @property
    def all_edges(self) -> tuple[EdgeFit, ...]:
        """Every fitted line, deduplicated: ``dock`` is the winning end, not a fourth line."""
        out = [self.far, self.near]
        seen = {id(self.dock)}
        out.append(self.dock)
        for e in self.ends:
            if id(e) not in seen:
                out.append(e)
        return tuple(out)

    @property
    def found(self) -> tuple[str, ...]:
        return tuple(e.name for e in self.edges if e.found)

    @property
    def level(self) -> str:
        got = set(self.found)
        ends = {e.name for e in self.ends if e.found}
        if {"far", "near", "dock"} <= got and ends == {"left", "right"}:
            return "L4"
        if {"far", "near", "dock"} <= got:
            return "L3"
        if {"far", "dock"} <= got:
            return "L2"
        if {"far", "near"} <= got:
            return "L1"
        if "far" in got:
            return "L0"
        return "none"

    def as_dict(self) -> dict[str, Any]:
        return {"method": self.method, "short_method": self.short_method,
                "fisher": round(float(self.fisher), 3),
                "fisher_by_method": {k: round(float(v), 3)
                                     for k, v in self.fisher_by_method.items()},
                "threshold": round(float(self.threshold), 1),
                "window": [int(self.window[0]), int(self.window[1])],
                "window_reason": self.window_reason,
                "scale": float(self.scale), "frame_size": list(self.frame_size),
                "water_fraction": round(float(self.water_fraction), 3),
                "n_components": int(self.n_components),
                "components_kept": int(self.components_kept),
                "dominant_share": round(float(self.dominant_share), 3),
                "dock_side": self.dock_side, "dock_side_evidence": self.dock_side_evidence,
                "level": self.level,
                "edges": {e.name: e.as_dict() for e in self.all_edges},
                "short_ends": {e.name: e.as_dict() for e in self.ends},
                "elapsed_s": round(float(self.elapsed_s), 2),
                "warnings": list(self.warnings), "params": dict(self.params)}


# ---------------------------------------------------------------------------
# When to look
# ---------------------------------------------------------------------------
def calm_window(n_frames: int, accepted_frames: Sequence[int] | None = None, *,
                min_frames: int = MIN_WINDOW, max_frames: int = MAX_WINDOW,
                margin: int = WINDOW_MARGIN) -> tuple[int, int, str]:
    """The longest stretch of clip with no accepted tail-set detection, capped to ``max_frames``.

    Upstream's central caveat is that a dog or a handler in frame corrupts the temporal statistics —
    so a window is *evidence*, and the clip's own detections are what say where the calm is.  The
    first accepted frame is only the common case; a clip where the dog is already on the dock at
    frame 0 still has calm later, and taking ``[0, first)`` there would hand the caller a window full
    of the very motion being excluded.

    Returns ``(start, end, reason)``.  ``end`` is exclusive.  When no stretch reaches
    ``min_frames``, the opening ``min_frames`` are used and the reason says so.
    """
    n = int(n_frames)
    if n <= 0:
        raise ValueError("an empty clip has no calm window")
    acc = sorted({int(f) for f in (accepted_frames or [])})
    limit = min(n, max(min_frames, 1))

    # Longest run of frames clear of any accepted detection (plus a margin either side).
    blocked = [False] * n
    for f in acc:
        for j in range(max(0, f - margin), min(n, f + margin + 1)):
            blocked[j] = True
    best = (0, 0)
    start = None
    for i in range(n + 1):
        free = i < n and not blocked[i]
        if free and start is None:
            start = i
        elif not free and start is not None:
            if (i - start) > (best[1] - best[0]):
                best = (start, i)
            start = None
    length = best[1] - best[0]
    if length >= limit:
        end = min(best[1], best[0] + max_frames)
        why = (f"{length} consecutive frames clear of accepted detections"
               if acc else "no accepted detections in the clip")
        return int(best[0]), int(end), why
    # No clean stretch long enough: the opening frames are the best remaining guess (a jump is rare
    # at the very start of a clip), and the reason says it is a guess.
    end = min(n, limit)
    return 0, int(end), (f"no {limit}-frame stretch clear of {len(acc)} accepted detection(s); "
                         f"using the opening {end} frames — check the noise map")


# ---------------------------------------------------------------------------
# Boundaries
# ---------------------------------------------------------------------------
def _roi_inner(roi_mask: np.ndarray) -> np.ndarray:
    """The ROI eroded by 5 px with the image border removed.

    A boundary that coincides with the ROI or image edge is a cropping artefact, not a waterline
    (upstream's rule for the contour method, applied to every scan here).
    """
    inner = cv2.erode(roi_mask, np.ones((5, 5), np.uint8))
    inner[0, :] = inner[-1, :] = 0
    inner[:, 0] = inner[:, -1] = 0
    return inner


def column_boundaries(mask: np.ndarray, roi_mask: np.ndarray | None = None
                      ) -> tuple[np.ndarray, np.ndarray]:
    """Top and bottom water boundary, one point per column, in **working** pixels.

    Both at once: the same mask gives the far wall's contact (top) and the near wall's (bottom), and
    running upstream's scan twice would rescan the whole image to find what one pass already has.
    """
    h, w = mask.shape[:2]
    b = mask > 0
    inner = _roi_inner(roi_mask if roi_mask is not None else np.full((h, w), 255, np.uint8))
    cols = np.arange(w)
    any_c = b.any(axis=0)
    top = np.argmax(b, axis=0)
    bot = (h - 1) - np.argmax(b[::-1, :], axis=0)
    # Each boundary is dropped independently: a water region that runs off the top of the frame has
    # no *top* boundary, but its bottom boundary is still real.  Rejecting the whole column instead
    # would throw away the good edge with the bad one.
    base = any_c & ((bot - top) >= 3)
    ok_top = base & (top > 0) & (inner[top, cols] > 0)
    ok_bot = base & (bot < h - 1) & (inner[bot, cols] > 0)
    return (np.stack([cols[ok_top].astype(float), top[ok_top].astype(float)], axis=1),
            np.stack([cols[ok_bot].astype(float), bot[ok_bot].astype(float)], axis=1))


def row_boundaries(mask: np.ndarray, roi_mask: np.ndarray | None = None
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Left and right water boundary, one point per row, in **working** pixels (the dock contact)."""
    h, w = mask.shape[:2]
    b = mask > 0
    inner = _roi_inner(roi_mask if roi_mask is not None else np.full((h, w), 255, np.uint8))
    rows = np.arange(h)
    any_r = b.any(axis=1)
    left = np.argmax(b, axis=1)
    right = (w - 1) - np.argmax(b[:, ::-1], axis=1)
    base = any_r & ((right - left) >= 3)
    ok_left = base & (left > 0) & (inner[rows, left] > 0)
    ok_right = base & (right < w - 1) & (inner[rows, right] > 0)
    return (np.stack([left[ok_left].astype(float), rows[ok_left].astype(float)], axis=1),
            np.stack([right[ok_right].astype(float), rows[ok_right].astype(float)], axis=1))


def _median_smooth(points: np.ndarray, *, vertical: bool, window: int = 5) -> np.ndarray:
    """Median-filter the boundary position along its scan axis (upstream's ``smooth``, applied here).

    A per-column boundary jitters by a pixel or two; without this, :func:`split_runs` reads those as
    jumps and shreds one real edge into fragments.  Measured on a synthetic mask: a clean 100-row
    vertical boundary came back as runs of 3-8 points and no dock edge at all.
    """
    if points is None or len(points) < max(3, window):
        return points
    pts = np.asarray(points, dtype=float).copy()
    scan_i, perp_i = (1, 0) if vertical else (0, 1)
    pts = pts[np.argsort(pts[:, scan_i], kind="stable")]
    pad = window // 2
    padded = np.pad(pts[:, perp_i], pad, mode="edge")
    win = np.lib.stride_tricks.sliding_window_view(padded, window)
    pts[:, perp_i] = np.median(win, axis=1)
    return pts


def split_runs(points: np.ndarray, *, vertical: bool = False,
               max_gap: float = MAX_RUN_GAP_PX,
               jump_factor: float = RUN_JUMP_FACTOR) -> list[np.ndarray]:
    """Break a boundary into contiguous runs before fitting.

    A lane rope, a glare patch, a shadow or the dock's own reflection interrupts the mask; one global
    fit spans the gap and tilts towards whichever side has more points.  Splitting first and keeping
    the best run is what makes the fit local to real evidence.
    """
    if points is None or len(points) < 2:
        return []
    pts = np.asarray(points, dtype=float)
    scan_i, perp_i = (1, 0) if vertical else (0, 1)
    order = np.argsort(pts[:, scan_i], kind="stable")
    pts = pts[order]
    d_scan = np.diff(pts[:, scan_i])
    d_perp = np.abs(np.diff(pts[:, perp_i]))
    jump = max(3.0, float(jump_factor * np.median(d_perp))) if len(d_perp) else 3.0
    breaks = np.nonzero((d_scan > float(max_gap)) | (d_perp > jump))[0]
    out, start = [], 0
    for b in breaks:
        out.append(pts[start:b + 1])
        start = b + 1
    out.append(pts[start:])
    return [r for r in out if len(r) >= 2]


def fit_run(run: np.ndarray, *, vertical: bool = False) -> dict[str, Any] | None:
    """Fit one run and score it, in the units of the points handed in (original px)."""
    if run is None or len(run) < 2:
        return None
    try:
        fit = wl.fit_straight_line(run, "left" if vertical else "below")
    except (ValueError, cv2.error):
        return None
    p0 = np.asarray(fit["point"], dtype=float)
    d = np.asarray(fit["direction"], dtype=float)
    n = np.array([-d[1], d[0]])
    resid = (np.asarray(run, dtype=float) - p0) @ n
    fit = dict(fit)
    fit["n"] = int(len(run))
    fit["inlier_fraction"] = float(np.mean(np.abs(resid) <= INLIER_TOL_PX))
    return fit


def _rms_limit(span_px: float) -> float:
    """RMS tolerance for a run, scaled by its length.

    A fixed 3.0 px gate is length-blind: half a degree of angular error over a 480 px waterline is
    2 px of RMS, so the longest and best-supported boundary in the frame is rejected in favour of a
    short clean fragment.  Measured on video004 seg01: the true far waterline is a 240-point run at
    RMS 3.17 (rejected by 0.17 px) and the run that won instead is a 44-point piece of a mask step
    at -22 deg.  The tolerance therefore grows gently with the span and never shrinks below the
    original absolute value.
    """
    return float(MAX_RMS_PX) + 0.01 * float(max(span_px, 0.0))


def _best_run(runs: Sequence[np.ndarray], *, vertical: bool,
              min_points: int = MIN_RUN_POINTS) -> tuple[np.ndarray, dict] | None:
    """The run that survives S1 best: enough points, straight enough, most inliers."""
    best: tuple[float, np.ndarray, dict] | None = None
    for run in runs:
        if len(run) < min_points:
            continue
        fit = fit_run(run, vertical=vertical)
        if fit is None:
            continue
        span = float(np.ptp(run[:, 1 if vertical else 0]))
        if fit["rms_residual"] > _rms_limit(span):
            continue
        score = float(fit["n"]) * float(fit["inlier_fraction"])
        if best is None or score > best[0]:
            best = (score, run, fit)
    if best is None:
        return None
    return best[1], best[2]


def _perp_offset(fit: dict, p: Sequence[float]) -> float:
    p0 = np.asarray(fit["point"], dtype=float)
    d = np.asarray(fit["direction"], dtype=float)
    n = np.array([-d[1], d[0]])
    return float(abs((np.asarray(p, dtype=float) - p0) @ n))


def _refit_here(mask: np.ndarray, roi_mask: np.ndarray, kind: str, side: str,
                baseline: np.ndarray, scale_back: float) -> dict | None:
    """Re-detect the SAME edge in another window's mask, keeping the run that overlaps it.

    Overlap, not "the best run": in the second sample a different wall may win the score, and
    comparing that line to the baseline would report a large "instability" that is really a different
    edge.
    """
    vertical = kind in ("dock", "left", "right")
    if kind == "far":
        cand = column_boundaries(mask, roi_mask)[0] * scale_back
    elif kind == "near":
        cand = column_boundaries(mask, roi_mask)[1] * scale_back
    else:
        cand = (row_boundaries(mask, roi_mask)[0] if side == "left"
                else row_boundaries(mask, roi_mask)[1]) * scale_back
    scan_i = 1 if vertical else 0
    lo, hi = float(baseline[:, scan_i].min()), float(baseline[:, scan_i].max())
    best = None
    for run in split_runs(cand, vertical=vertical):
        rlo, rhi = float(run[:, scan_i].min()), float(run[:, scan_i].max())
        overlap = min(hi, rhi) - max(lo, rlo)
        span = max(1e-6, hi - lo)
        if overlap < 0.5 * span:
            continue
        fit = fit_run(run, vertical=vertical)
        if fit is None:
            continue
        if best is None or fit["n"] > best["n"]:
            best = fit
    return best


def _edge(name: str, runs: Sequence[np.ndarray], *, vertical: bool,
          min_points: int = MIN_RUN_POINTS) -> EdgeFit:
    got = _best_run(runs, vertical=vertical, min_points=min_points)
    if got is None:
        why = ("no run with >= %d points" % min_points if not runs
               else "no run under %.0f px RMS with >= %d points" % (MAX_RMS_PX, min_points))
        return EdgeFit(name=name, found=False, reason=why)
    run, fit = got
    if fit["inlier_fraction"] < MIN_INLIER_FRACTION:
        return EdgeFit(name=name, found=False, points=run, fit=fit,
                       inlier_fraction=float(fit["inlier_fraction"]),
                       reason=f"inlier fraction {fit['inlier_fraction']:.2f} < "
                              f"{MIN_INLIER_FRACTION:.2f}")
    return EdgeFit(name=name, found=True, points=run, fit=fit,
                   inlier_fraction=float(fit["inlier_fraction"]))


# ---------------------------------------------------------------------------
# The two short ends, methods A and B
# ---------------------------------------------------------------------------
def _contiguous(idx: np.ndarray) -> list[np.ndarray]:
    """Split an increasing index array into contiguous runs."""
    if idx.size == 0:
        return []
    out, start = [], 0
    for g in np.nonzero(np.diff(idx) > 1)[0]:
        out.append(idx[start:g + 1])
        start = g + 1
    out.append(idx[start:])
    return out


def _strip_height(near_fit: dict, far_fit: dict | None, w: int, scale: float) -> float:
    """Vertical extent of the water strip between the long waterlines, in working px."""
    xs = np.linspace(0.05 * w, 0.95 * w, 11)
    if far_fit is None:
        return 60.0
    d = ((near_fit["slope"] - far_fit["slope"]) * xs
         + (near_fit["intercept"] - far_fit["intercept"]) * scale)
    d = d[np.isfinite(d) & (d > 1.0)]
    return float(np.clip(np.median(d), 12.0, 240.0)) if d.size else 60.0


def _band_profile(image: np.ndarray, near_fit: dict, k: int, thickness: float, *,
                  scale: float, h: int, w: int) -> np.ndarray:
    """Mean of ``image`` across band ``k`` (band 0 is nearest the near waterline), per column.

    The bands are laid in the water strip along the near waterline's own slope, so a tilted camera
    does not smear one band into the next, and band 0 is the part of the short edge a banner hanging
    over the dock cannot cover.
    """
    xs = np.arange(w, dtype=float)
    yn = near_fit["slope"] * xs + near_fit["intercept"] * scale
    r_hi = np.rint(yn - k * thickness).astype(int)
    r_lo = np.rint(yn - (k + 1) * thickness).astype(int)
    prof = np.full(w, np.nan, dtype=float)
    for x in range(w):
        r0, r1 = max(0, int(r_lo[x])), min(h, int(r_hi[x]))
        if r1 - r0 >= 2:
            prof[x] = float(image[r0:r1, x].mean())
    return prof


def _end_crossing(prof: np.ndarray, side: str, *, plateau: float = PLATEAU_LEVEL,
                  level: float = CROSS_LEVEL) -> tuple[float | None, str]:
    """Method A: where the band's water occupancy leaves its in-pool plateau (working px, sub-pixel).

    Works from the *plateau*, not from the outermost water pixel: an occluder that juts out past the
    wall (a ramp leg, a banner) is a second, narrower high region, not the end of the pool.
    """
    q = np.asarray(prof, dtype=float)
    good = np.isfinite(q)
    if good.sum() < 10:
        return None, "band has too few usable columns"
    q = np.where(good, q, 0.0)
    if q.size >= 5:
        q = np.convolve(q, np.ones(5) / 5.0, mode="same")
    inside = np.nonzero(q >= plateau)[0]
    if inside.size < 5:
        return None, "no in-pool plateau in this band"
    run = max(_contiguous(inside), key=len)
    i = int(run[-1]) if side == "right" else int(run[0])
    step = 1 if side == "right" else -1
    j = i
    while 0 <= j + step < q.size and q[j + step] >= level:
        j += step
    jn = j + step
    if not (0 <= jn < q.size):
        return None, "water runs off the frame border"
    a, b = float(q[j]), float(q[jn])
    frac = 0.0 if a == b else (a - level) / (a - b)
    return float(j) + step * float(np.clip(frac, 0.0, 1.0)), "occupancy crossing"


def _end_step(prof: np.ndarray, side: str, *, window: int = STEP_WINDOW,
              min_ratio: float = STEP_RATIO_MIN) -> tuple[float | None, str]:
    """Method B: the change point of the continuous noise profile (working px, sub-pixel).

    A two-sample median ratio, not a threshold: shade scales the water level *and* the structure
    level down together, so the ratio and therefore the step's position survive it.  This is the
    estimator for the shaded short edge, where a global Otsu cut halves the dock band.
    """
    q = np.asarray(prof, dtype=float)
    good = np.isfinite(q)
    if good.sum() < 2 * window + 6:
        return None, "band too short for a step fit"
    q = np.clip(np.where(good, q, float(np.nanmedian(q))), 0.0, None)
    best: tuple[float, int, float, float] | None = None
    for i in range(window, q.size - window):
        left = float(np.median(q[i - window:i]))
        right = float(np.median(q[i:i + window]))
        hi, lo = (left, right) if side == "right" else (right, left)
        ratio = hi / max(lo, 1e-9)
        if best is None or ratio > best[0]:
            best = (ratio, i, left, right)
    if best is None:
        return None, "band has no usable columns"
    ratio, i, left, right = best
    if ratio < min_ratio:
        return None, f"no step (best level ratio {ratio:.2f} < {min_ratio})"
    level = 0.5 * (left + right)
    lo_i, hi_i = max(1, i - window), min(q.size - 1, i + window)
    cand = [j for j in range(lo_i, hi_i + 1)
            if (q[j - 1] - level) * (q[j] - level) <= 0.0]
    if not cand:
        return float(i), f"step {ratio:.2f}x"
    j = min(cand, key=lambda c: abs(c - i))
    a, b = float(q[j - 1]), float(q[j])
    frac = 0.0 if a == b else (level - a) / (b - a)
    return float(j - 1) + float(np.clip(frac, 0.0, 1.0)), f"step {ratio:.2f}x"


def short_end_edges(image: np.ndarray, near: "EdgeFit", far: "EdgeFit | None", *,
                    method: str, scale: float,
                    roi_mask: np.ndarray | None = None) -> dict[str, "EdgeFit"]:
    """Both short ends from one band scan: ``image`` is the water mask (A) or the noise map (B).

    Both methods run the *same* bands and the same consensus rule, so a difference between them is
    the estimator (thresholded occupancy vs continuous step), not the geometry.  A band that times
    out at the frame border says the end is off-shot; bands that disagree say an occluder got in the
    way — neither is turned into a line.
    """
    h, w = image.shape[:2]
    if near is None or not near.found or near.fit is None:
        return {s: EdgeFit(name=s, found=False, status="off_shot",
                           reason="no near waterline to lay the bands against")
                for s in ("left", "right")}
    scale = float(scale) or 1.0
    back = 1.0 / scale
    strip = _strip_height(near.fit, far.fit if (far is not None and far.found) else None, w, scale)
    thickness = max(3.0, strip / float(BAND_COUNT))
    out: dict[str, EdgeFit] = {}
    for side in ("left", "right"):
        hits: list[tuple[int, float, float]] = []        # (band, x, y) in original px
        notes: list[str] = []
        band0_off = False
        for k in range(BAND_COUNT):
            prof = _band_profile(image, near.fit, k, thickness, scale=scale, h=h, w=w)
            pos, why = _end_crossing(prof, side) if method == "A" else _end_step(prof, side)
            if pos is None:
                notes.append(f"band {k}: {why}")
                if k == 0 and "off the frame border" in why:
                    band0_off = True
                continue
            y_w = near.fit["slope"] * pos + near.fit["intercept"] * scale - (k + 0.5) * thickness
            hits.append((k, pos * back, y_w * back))
        if not hits:
            off = any("off the frame border" in n for n in notes)
            out[side] = EdgeFit(
                name=side, found=False, status="off_shot" if off else "declined",
                reason=f"0/{BAND_COUNT} bands gave a position"
                       + ("; the water runs off the frame at this end" if off else "")
                       + (f" · {notes[0]}" if notes else ""))
            continue
        # Bottom-up: the band nearest the waterline is the one a banner hanging over the dock
        # cannot reach, so it sets the answer and the bands above it are kept only while they agree.
        # A median across all bands would let the ramp/banner diagonal outvote the wall.
        hits.sort(key=lambda t: t[0])
        base_x = hits[0][1]
        keep: list[tuple[int, float, float]] = []
        for hit in hits:
            if abs(hit[1] - base_x) > BAND_X_TOL:
                break
            keep.append(hit)
        arr = np.asarray([(x, y) for _k, x, y in keep], dtype=float)
        spread = float(np.max(np.abs(arr[:, 0] - base_x)))
        # A crossing that lands on the edge of the search region is the region ending, not the pool:
        # the Calibrate pane passes the quad grown 30 %, and on video004 seg01 that region stops at
        # x ~ 698 with the water still running to x = 958 — which read as a perfect four-band edge.
        if roi_mask is not None and roi_mask.any():
            cols = np.nonzero(roi_mask.any(axis=0))[0]
            lim = (float(cols.max()) * back) if side == "right" else (float(cols.min()) * back)
            if (side == "right" and base_x >= lim - 3.0) or (side == "left" and base_x <= lim + 3.0):
                out[side] = EdgeFit(
                    name=side, found=False, status="declined",
                    reason=f"x={base_x:.0f} px is the edge of the search region (ROI ends at "
                           f"x={lim:.0f}), not the end of the water — widen the ROI")
                continue
        if len(keep) >= 2:
            fit = fit_run(arr, vertical=True)
            status = "observed" if len(keep) >= MIN_BAND_POINTS else "partial"
            why = (f"{len(keep)}/{BAND_COUNT} bands agree from band {keep[0][0]}: x={base_x:.0f} px, "
                   f"spread {spread:.1f} px")
        elif keep[0][0] <= 1:
            # Exactly one band saw a termination, and it is one of the two nearest the waterline —
            # the position is evidence, the tilt is not, so the line is emitted vertical through that
            # point and labelled unconfirmed.
            fit = fit_run(np.array([[base_x, arr[0][1] - 30.0], [base_x, arr[0][1] + 30.0]]),
                          vertical=True)
            status = "partial"
            why = (f"only band {keep[0][0]} shows a termination (x={base_x:.0f} px)"
                   + ("; band 0 runs to the frame border" if band0_off else "")
                   + " — position only, tilt not measured")
        else:
            # A lone hit in an upper band is the corner where the far waterline meets the frame, or
            # the top of an occluder.  It is not the water/wall contact and must not become a line.
            out[side] = EdgeFit(
                name=side, found=False, status="declined",
                reason=f"only band {keep[0][0]} (x={base_x:.0f} px) terminated, above the bands where "
                       f"the wall contact can be seen — not treated as a short edge")
            continue
        if fit is None:
            out[side] = EdgeFit(name=side, found=False, status="declined",
                                reason="too few band points to fit a line")
            continue
        out[side] = EdgeFit(name=side, found=True, points=arr, fit=fit,
                            inlier_fraction=float(fit["inlier_fraction"]),
                            stability_px=spread, status=status, reason=why)
    return out


def short_end_lines(image: np.ndarray, near: "EdgeFit", far: "EdgeFit | None", *,
                    scale: float, roi_mask: np.ndarray | None = None,
                    frame_w: float | None = None) -> dict[str, "EdgeFit"]:
    """Method M1 — each short edge is a **line** fitted through one boundary crossing per band.

    The estimator is Method A's occupancy crossing; what changes is the model.  A band's crossing is
    one point *on* the short edge, and the edge slants: measured on video004 seg01 the dock edge's
    crossings are (951,369) (855,351) (741,333) (724,315) — one line, RMS 5.0 px, dx/dy = 4.71, i.e.
    **12 deg from horizontal**.  Requiring a common x across bands (``BAND_X_TOL``) therefore rejected
    the correct edge by construction, which is why this method drops that test and keeps only:
    enough crossings, straight enough, and not parallel to the long waterlines.

    When the water leaves the frame at an end, the edge is **cut at the frame border** and labelled
    ``frame_cut``: that is not a claim about the pool, it is the honest boundary of what is visible,
    and it is what lets a complete quad be generated for an end whose wall is off shot.
    """
    h, w = image.shape[:2]
    if near is None or not near.found or near.fit is None:
        return {s: EdgeFit(name=s, found=False, status="off_shot",
                           reason="no near waterline to lay the bands against")
                for s in ("left", "right")}
    scale = float(scale) or 1.0
    back = 1.0 / scale
    fw = float(frame_w) if frame_w else float(w) * back
    strip = _strip_height(near.fit, far.fit if (far is not None and far.found) else None, w, scale)
    thickness = max(3.0, strip / float(BAND_COUNT))
    near_angle = float(near.angle_deg)
    lo_ok, hi_ok = 0.0, fw - 1.0
    if roi_mask is not None and roi_mask.any():
        cols = np.nonzero(roi_mask.any(axis=0))[0]
        lo_ok, hi_ok = float(cols.min()) * back, float(cols.max()) * back
    out: dict[str, EdgeFit] = {}
    for side in ("left", "right"):
        pts: list[tuple[float, float]] = []
        notes: list[str] = []
        border = 0
        for k in range(BAND_COUNT):
            prof = _band_profile(image, near.fit, k, thickness, scale=scale, h=h, w=w)
            pos, why = _end_crossing(prof, side)
            if pos is not None and not (lo_ok + 3.0 <= pos * back <= hi_ok - 3.0):
                pos, why = None, "the crossing is the edge of the search region (ROI), not the water"
            if pos is None:
                if "off the frame border" in why:
                    border += 1
                notes.append(f"band {k}: {why}")
                continue
            y_w = near.fit["slope"] * pos + near.fit["intercept"] * scale - (k + 0.5) * thickness
            pts.append((pos * back, y_w * back))
        if len(pts) >= MIN_SHORT_CROSSINGS:
            arr = np.asarray(pts, dtype=float)
            fit = fit_run(arr, vertical=True)
            if fit is not None:
                ang = float(fit["angle_deg"])
                off = abs((ang - near_angle + 90.0) % 180.0 - 90.0)
                if float(fit["rms_residual"]) > SHORT_MAX_RMS_PX:
                    out[side] = EdgeFit(name=side, found=False, status="declined",
                                        reason=f"{len(pts)} crossings but RMS "
                                               f"{fit['rms_residual']:.1f} px > {SHORT_MAX_RMS_PX}")
                    continue
                if off < MIN_SHORT_ANGLE_DEG:
                    out[side] = EdgeFit(name=side, found=False, status="declined",
                                        reason=f"the line is {off:.1f} deg off the long waterlines "
                                               f"(< {MIN_SHORT_ANGLE_DEG}) — that is the same edge, "
                                               f"not a short end")
                    continue
                out[side] = EdgeFit(name=side, found=True, points=arr, fit=fit,
                                    inlier_fraction=float(fit["inlier_fraction"]),
                                    stability_px=None,
                                    status="observed",
                                    reason=f"{len(pts)} band crossings, one line: RMS "
                                           f"{fit['rms_residual']:.1f} px, {off:.0f} deg off the "
                                           f"long waterlines, dx/dy "
                                           f"{1.0 / max(abs(math.tan(np.radians(ang))), 1e-9):.2f}")
                continue
        if border >= max(2, BAND_COUNT // 2):
            # The water runs off this end of the frame: cut the quad at the border and say so.  The
            # cut is the vertical at that column, spanning the strip from the far waterline to the
            # near one — a real quad edge, and explicitly not a claim about the pool.
            x_b = 0.0 if side == "left" else fw - 1.0
            y_n = near.fit["slope"] * x_b + near.fit["intercept"]
            if far is not None and far.found and far.fit is not None:
                y_f = far.fit["slope"] * x_b + far.fit["intercept"]
            else:
                y_f = y_n - 90.0
            seg = np.array([[x_b, y_f], [x_b, y_n]], dtype=float)
            fit = fit_run(seg, vertical=True)
            out[side] = EdgeFit(name=side, found=True, points=seg, fit=fit,
                                inlier_fraction=1.0, stability_px=None, status="frame_cut",
                                reason=f"{border}/{BAND_COUNT} bands run off the frame at x={x_b:.0f} "
                                       f"— the quad is CUT at the frame here, not at a wall")
            continue
        out[side] = EdgeFit(name=side, found=False, status="declined",
                            reason=f"{len(pts)}/{MIN_SHORT_CROSSINGS} crossings needed"
                                   + (f" · {notes[0]}" if notes else ""))
    return out


# ---------------------------------------------------------------------------
# The detection
# ---------------------------------------------------------------------------
def analyze(frames: np.ndarray, *, scale: float = 1.0,
            frame_size: tuple[int, int] | None = None,
            roi_poly: np.ndarray | None = None,
            threshold_mode: str = "otsu", threshold_value: float | None = None,
            pre_blur: int = 3, post_blur: int = 5, diff_lag: int = 1,
            open_size: int = 5, close_size: int = 15,
            window: tuple[int, int] = (0, 0), window_reason: str = "",
            short_method: str | None = None) -> WaterlineEdges:
    """Detect the three edges in an in-memory stack of frames.

    ``frames`` are the *working* resolution (already scaled); every coordinate returned is multiplied
    back by ``1/scale`` so callers only ever see original pixels.  Kept separate from
    :func:`detect_edges` so tests can drive the whole classification with a synthetic stack.
    """
    t0 = time.time()
    if frames is None or len(frames) < 2:
        raise ValueError("need at least 2 frames for a temporal noise map")
    shape = frames.shape[1:3]
    roi_mask = wl.roi_to_mask(roi_poly, shape, 1.0 if roi_poly is None else scale)
    maps = wl.compute_noise_maps(frames, pre_blur, diff_lag)

    # Pick the statistic by upstream's own separability score, with a bias to `detrended`: it is the
    # one that survives the lighting drift these venues have (sun through a skylight, arena lights).
    fisher: dict[str, float] = {}
    norm_by_method: dict[str, np.ndarray] = {}
    for m in wl.NOISE_METHODS:
        norm = wl.normalize_noise_map(maps[m], wl.fit_normalization(maps[m], roi_mask), post_blur)
        norm_by_method[m] = norm
        fisher[m] = float(wl.separability(norm, roi_mask)["fisher_ratio"])
    best = max(fisher, key=lambda k: (fisher[k] * (1.10 if k == "detrended" else 1.0)))
    norm_map = norm_by_method[best]

    raw_mask, thr = wl.threshold_noise_map(norm_map, threshold_mode, threshold_value, roi_mask)
    # Components: keep every *sizeable* high-noise region, not just the largest.  Upstream's default
    # (largest only) is right for a clean pool, and wrong here: in a sunlit clip the water splits into
    # a bright half and a shaded half at the dock, and keeping the largest then makes the mask's "dock
    # side" boundary a component edge in mid-pool — measured on video004 seg01, the dock line landed at
    # x~620 instead of the real dock at x~900.  A region under a quarter of the largest is dropped, so
    # a glare spot still does not join the water.
    n_raw, labels, stats, _ = cv2.connectedComponentsWithStats(raw_mask, connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA] if n_raw > 1 else np.array([], dtype=int)
    kept = 0
    if len(areas) == 0:
        mask = np.zeros_like(raw_mask)
    else:
        min_area = max(int(0.25 * int(areas.max())), 400, int(0.002 * roi_mask.size))
        mask = wl.clean_mask(raw_mask, open_size, close_size, keep_largest=False,
                             min_area=min_area)
        kept = int(sum(1 for a in areas if a >= min_area))
    comp = wl.component_report(raw_mask, frames.max(axis=0))
    warnings: list[str] = []
    if comp["n_components"] > 1:
        warnings.append(f"{comp['n_components']} separate high-noise regions; only the largest is "
                        f"kept (areas {[c['area'] for c in comp['components'][:5]]})")
    for c in comp["glare_suspects"]:
        warnings.append(f"high-noise region at {c['centroid_xy']} (area {c['area']}) is "
                        f"{c['saturated_fraction']:.0%} saturated: likely glare, not water")

    scale_back = 1.0 / float(scale)
    top_w, bot_w = column_boundaries(mask, roi_mask)
    left_w, right_w = row_boundaries(mask, roi_mask)
    top, bot = _median_smooth(top_w * scale_back, vertical=False), _median_smooth(
        bot_w * scale_back, vertical=False)
    left, right = _median_smooth(left_w * scale_back, vertical=True), _median_smooth(
        right_w * scale_back, vertical=True)

    far = _edge("far", split_runs(top), vertical=False)
    near = _edge("near", split_runs(bot), vertical=False)

    # Both short ends are kept (operator order 2026-10-21): the pool's two end walls are the images
    # of X = 0 and X = anchor, and which is which is the operator's orientation choice, not a score
    # contest.  ``dock`` stays the detector's own vote so every existing consumer keeps working.
    short_how = (short_method or SHORT_EDGE_METHOD or "M1").strip()
    if short_how == "M1":
        # Method M1 (2026-10-27): each short edge is a LINE through one crossing per band.  The
        # dock edge measured on video004 seg01 is 12 deg from horizontal, so the old near-vertical
        # test and the common-x consensus both rejected the correct edge by construction.
        sides = short_end_lines((mask > 0).astype(np.float32), near, far, scale=float(scale),
                                roi_mask=roi_mask if roi_poly is not None else None,
                                frame_w=float(frame_size[0]) if frame_size else None)
    elif short_how in ("A", "B"):
        # Methods A/B (2026-10-27): the extreme-pixel row scan above is what latched onto the
        # building and the ramp on video004 seg01, so these scan the water strip in bands instead.
        sides = short_end_edges((mask > 0).astype(np.float32) if short_how == "A" else norm_map,
                               near, far, method=short_how, scale=float(scale),
                               roi_mask=roi_mask if roi_poly is not None else None)
    else:
        sides = {
            "left": _edge("left", split_runs(left, vertical=True), vertical=True,
                          min_points=MIN_DOCK_POINTS),
            "right": _edge("right", split_runs(right, vertical=True), vertical=True,
                           min_points=MIN_DOCK_POINTS),
        }
    for side, e in sides.items():
        # Legacy near-vertical test — ONLY for the original scan.  Measured on video004 seg01 a
        # correct short edge is 12 deg from horizontal, so methods A/B/M1 carry their own,
        # geometry-aware checks and this one would reject the right answer.
        if short_how == "current" and e.found and abs(e.angle_deg) < MIN_DOCK_ANGLE_DEG:
            e.found = False
            e.reason = f"line is {e.angle_deg:.0f}deg from horizontal, not vertical enough"
    ranked = sorted(sides.items(),
                    key=lambda kv: (kv[1].status == "observed", kv[1].found, _score(kv[1])),
                    reverse=True)
    dock_side, winner = ranked[0]
    dock = dataclasses.replace(winner, name="dock")      # provisional; rebuilt after stability
    if not winner.found:
        dock_side = "unknown"

    # Stability: the same edge in each half of the window.  This is the second, independent sample —
    # a comparison of the seed against the line it came from would be ~0 px by construction.
    h = len(frames)
    halves = [frames[: h // 2], frames[h // 2:]] if h >= 4 else []
    # Methods A/B supply their own second sample (the spread of the per-band answers), so the
    # half-window refit — which re-runs the *old* row scan — would compare unlike things.
    short_edges = () if short_how in ("A", "B", "M1") else (("left", sides["left"]), ("right", sides["right"]))
    for side, e in (("", far), ("", near), *short_edges):
        if not e.found or not halves:
            continue
        offsets: list[float] = []
        for half in halves:
            hmaps = wl.compute_noise_maps(half, pre_blur, diff_lag)
            hnorm = wl.normalize_noise_map(hmaps[best],
                                           wl.fit_normalization(hmaps[best], roi_mask), post_blur)
            hraw, _ = wl.threshold_noise_map(hnorm, threshold_mode, threshold_value, roi_mask)
            hmask = wl.clean_mask(hraw, open_size, close_size, keep_largest=False,
                                  min_area=MIN_RUN_POINTS * 4)
            refit = _refit_here(hmask, roi_mask, e.name, side or dock_side, e.points, scale_back)
            if refit is None or e.midpoint is None:
                continue
            # A half that locked onto a different edge is not evidence about this one.  Two parallel
            # candidates are common (in the arena clips the far banner base and the near waterline are
            # both ~horizontal, ~230 px apart) and an x-overlap test cannot tell them apart, so require
            # the same line to within 25 px as well — and report "not checkable" rather than a number
            # that reads as instability.
            offset = _perp_offset(refit, e.midpoint)
            if offset > 25.0:
                continue
            delta = abs((float(refit["angle_deg"]) - e.angle_deg + 90.0) % 180.0 - 90.0)
            if delta <= 15.0:
                offsets.append(offset)
        e.stability_px = max(offsets) if offsets else None
        if e.stability_px is not None and e.stability_px > MAX_STABILITY_PX:
            e.found = False
            e.reason = (f"unstable: the two halves disagree by {e.stability_px:.1f} px "
                        f"(> {MAX_STABILITY_PX:.0f})")

    # ``dock`` is the end the detector voted for, rebuilt now that stability may have demoted it — one
    # object per side, so a demoted dock cannot disagree with the ``left``/``right`` it came from.
    dock = dataclasses.replace(sides.get(dock_side) or winner, name="dock")
    if not dock.found:
        dock_side = "unknown"

    evidence = (f"left score {_score(sides['left']):.0f}, "
                f"right score {_score(sides['right']):.0f}")
    w_px, h_px = frame_size or (int(shape[1]), int(shape[0]))
    return WaterlineEdges(
        far=far, near=near, dock=dock, left=sides["left"], right=sides["right"],
        dock_side=dock_side, dock_side_evidence=evidence,
        method=best, short_method=short_how, fisher=fisher[best], fisher_by_method=fisher,
        threshold=float(thr),
        window=(int(window[0]), int(window[1])), window_reason=window_reason,
        scale=float(scale), water_fraction=float(np.mean(mask > 0)),
        n_components=int(comp["n_components"]), components_kept=int(kept),
        dominant_share=float(_dominant_share(comp)), warnings=warnings,
        frame_size=(int(w_px), int(h_px)), elapsed_s=time.time() - t0,
        params={"pre_blur": pre_blur, "post_blur": post_blur, "diff_lag": diff_lag,
                "open_size": open_size, "close_size": close_size,
                "threshold_mode": threshold_mode, "threshold_value": threshold_value,
                "roi": "poly" if roi_poly is not None else "frame",
                "short_edge_method": short_how,
                "scale": float(scale)})


def _dominant_share(comp: dict[str, Any]) -> float:
    """Largest high-noise region as a share of the listed ones (1.0 = one clean blob)."""
    comps = comp.get("components") or []
    total = sum(float(c["area"]) for c in comps)
    return float(comps[0]["area"]) / total if comps and total > 0 else float("nan")


def _score(e: EdgeFit) -> float:
    return float(e.inlier_fraction) * len(e.points if e.points is not None else [])


def detect_edges(clip: str | Path, *, first_accepted: int | None = None,
                 accepted_frames: Sequence[int] | None = None,
                 window: tuple[int, int] | None = None,
                 scale: float = DEFAULT_SCALE, step: int = 1,
                 roi_poly: np.ndarray | None = None,
                 threshold_mode: str = "otsu", threshold_value: float | None = None,
                 short_method: str | None = None,
                 ) -> WaterlineEdges:
    """Load a calm window from ``clip`` and detect the three edges.

    The window is :func:`calm_window`'s answer unless the caller pins one (the pane lets the operator
    edit it).  ``first_accepted`` is accepted for convenience; ``accepted_frames`` is the full list
    and is what actually drives the choice.
    """
    clip = Path(clip)
    info = wl.probe_video(clip)
    acc = list(accepted_frames or [])
    if first_accepted is not None:
        acc = acc or [int(first_accepted)]
    n = int(info.frame_count) or 0
    if window is None:
        start, end, reason = calm_window(n, acc)
    else:
        start, end = int(window[0]), int(window[1])
        reason = "pinned by the operator"
    end = max(start + 2, min(end, n)) if n else end
    frames = wl.load_frames(clip, start, end, False, int(step), float(scale))
    out = analyze(frames, scale=scale, frame_size=(info.width, info.height), roi_poly=roi_poly,
                  threshold_mode=threshold_mode, threshold_value=threshold_value,
                  window=(start, end), window_reason=reason, short_method=short_method)
    out.params["step"] = int(step)
    return out


def as_overlay(edges: WaterlineEdges) -> dict[str, Any]:
    """The fitted lines and their inliers, for a canvas overlay (original px)."""
    out: dict[str, Any] = {}
    for e in edges.all_edges:
        if not e.found or e.fit is None:
            out[e.name] = None
            continue
        out[e.name] = {"endpoints": [tuple(map(float, p)) for p in e.fit["endpoints"]],
                       "points": (e.points.round(1).tolist() if e.points is not None else []),
                       "inlier_fraction": float(e.inlier_fraction),
                       "rms_px": float(e.rms)}
    return out
