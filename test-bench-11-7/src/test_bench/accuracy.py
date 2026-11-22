"""Accuracy tab — the proposal's *simple* distance-accuracy metrics.

The bench's own deliverable is ``metrics.py`` (pixel / score / judge error,
mean-only averages, ``accuracy_pct``).  The Accuracy tab reports the same runs a
second, proposal-facing way: **error as a percentage of ground truth**, medians,
and the share of jumps inside ±2 / ±4 / ±6 in.  ``metrics.py`` stays frozen so
the existing green suite keeps its contract; everything new lands here.

Definitions (``g`` = ground-truth record, ``s`` = scored cell)::

  * ``score_error_pct`` = 100 · score_error_in / g.gt_in
  * ``judge_error_pct`` = 100 · judge_error_in / g.gt_in

Note this is **err/GT**, not ``metrics.score_accuracy_pct`` (which is
``100·(1 − err/GT)`` and therefore reads ~95 % on a 120 in jump even at ±6 in,
making the ±3 in target and the ±6 in fallback indistinguishable).

Any metric whose denominator is missing — no GT record, no judge, non-positive
``gt_in`` — returns ``None`` rather than 0, so the GUI can render ``—`` instead
of inventing a perfect score.
"""
from __future__ import annotations

import math
import statistics
from typing import Any, Iterable

import numpy as np

from . import metrics as _metrics

__all__ = [
    "median",
    "score_error_pct",
    "judge_error_pct",
    "nominal_in_per_px",
    "local_in_per_px_at_entry",
    "bin_counts",
    "find_run",
    "aggregate",
]


def median(values: Iterable[float | None]) -> float | None:
    """Median of the non-None values; None for an empty/all-None input."""
    xs = [float(v) for v in values if v is not None]
    return float(statistics.median(xs)) if xs else None


def _pct_of_gt(err_in: float | None, gt_in: Any) -> float | None:
    if err_in is None or gt_in is None or float(gt_in) <= 0:
        return None
    return float(100.0 * err_in / float(gt_in))


def score_error_pct(score: dict[str, Any], gt: dict[str, Any]) -> float | None:
    """``|score − GT|`` as a percentage of GT inches (None when undefined)."""
    return _pct_of_gt(_metrics.score_error_in(score, gt), gt.get("gt_in"))


def judge_error_pct(gt: dict[str, Any]) -> float | None:
    """``|judge − GT|`` as a percentage of GT inches (None when no judge)."""
    return _pct_of_gt(_metrics.judge_error_in(gt), gt.get("gt_in"))


def bin_counts(errors_in: Iterable[float | None], bins: tuple[int, ...] = (2, 4, 6)
               ) -> dict[int, dict[str, float]]:
    """Share of jumps whose absolute error lands inside each ± bin, in inches.

    Returns ``{2: {"count": n, "pct": p}, 4: {...}, 6: {...}}``.  The bin edge is
    **inclusive** (an error of exactly 2.0 counts as within ±2).  ``None`` values
    are dropped; with no usable errors every bin reports ``count 0, pct 0.0``.
    """
    xs = [abs(float(v)) for v in errors_in if v is not None]
    out: dict[int, dict[str, float]] = {}
    for b in bins:
        n = sum(1 for v in xs if v <= float(b))
        out[int(b)] = {"count": n, "pct": float(100.0 * n / len(xs)) if xs else 0.0}
    return out


def nominal_in_per_px(plane_doc: dict[str, Any]) -> float | None:
    """Inches-per-pixel from the plane's own distance marks.

    Takes the two marks farthest apart in real inches (each mark carries ``ft``,
    ``inch``, ``x_px``, ``y_px``) and divides that inch span by the pixel distance
    between them.  ``None`` when there are fewer than two marks, the two extremes
    sit at the same real distance, or they sit at the same pixel.
    """
    marks = (plane_doc or {}).get("marks") or []
    if not isinstance(marks, (list, tuple)) or len(marks) < 2:
        return None

    def _total_in(m: dict[str, Any]) -> float:
        return float(m.get("ft", 0.0) or 0.0) * 12.0 + float(m.get("inch", 0.0) or 0.0)

    totals = [_total_in(m) for m in marks]
    i_lo = min(range(len(totals)), key=lambda i: totals[i])
    i_hi = max(range(len(totals)), key=lambda i: totals[i])
    in_span = totals[i_hi] - totals[i_lo]
    if in_span <= 0.0:
        return None

    x0, y0 = float(marks[i_lo]["x_px"]), float(marks[i_lo]["y_px"])
    x1, y1 = float(marks[i_hi]["x_px"]), float(marks[i_hi]["y_px"])
    px_span = math.hypot(x1 - x0, y1 - y0)
    if px_span <= 0.0:
        return None
    return in_span / px_span


def _apply_h(H: np.ndarray, x: float, y: float) -> np.ndarray | None:
    """One 2-D point through a 3×3 homography; ``None`` on a degenerate divide."""
    homog = np.array([[x, y, 1.0]], dtype=float)
    out = (H @ homog.T).T
    w = float(out[0, 2])
    if abs(w) < 1e-15:
        return None
    return out[0, :2] / w


def local_in_per_px_at_entry(plane_doc: dict[str, Any], entry_px: Any) -> float | None:
    """Inches-per-pixel at the scored entry point, from the plane homography.

    Inverts ``H`` at ``entry_px`` to recover the world ``(X, Y)`` on the plane,
    finite-differences one foot along +X back into pixels, and returns
    ``12 / px_per_ft``.  This is the honest scale at the measurement point —
    perspective makes it vary across the frame.  ``None`` on a missing/degenerate
    homography, a missing entry pixel, or a zero foot-spacing.
    """
    H = (plane_doc or {}).get("homography")
    if H is None or entry_px is None:
        return None
    H = np.asarray(H, dtype=float).reshape(3, 3)
    px = np.asarray(entry_px, dtype=float).reshape(-1)
    if px.size < 2 or not np.all(np.isfinite(px)):
        return None
    try:
        H_inv = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return None

    world = _apply_h(H_inv, float(px[0]), float(px[1]))
    if world is None or not np.all(np.isfinite(world)):
        return None
    X, Y = float(world[0]), float(world[1])

    p0 = _apply_h(H, X, Y)
    p1 = _apply_h(H, X + 1.0, Y)
    if p0 is None or p1 is None:
        return None
    px_per_ft = float(np.linalg.norm(p1 - p0))
    if px_per_ft <= 0.0 or not np.isfinite(px_per_ft):
        return None
    return 12.0 / px_per_ft


def _find_run(runs: Iterable[dict[str, Any]] | None, cfg: dict[str, Any]
              ) -> dict[str, Any] | None:
    """The run cell whose method/n_points/selection match ``cfg`` (or None).

    ``n_points`` is compared string-wise so an int ``4`` in one place matches a
    ``"4"`` in the other and ``"all"`` matches ``"all"``; ``method`` and
    ``selection`` compare exactly.
    """
    method = cfg.get("method")
    selection = cfg.get("selection")
    n_points = cfg.get("n_points")
    for run in runs or ():
        if (run.get("method") == method and run.get("selection") == selection
                and str(run.get("n_points")) == str(n_points)):
            return run
    return None


def find_run(runs: Iterable[dict[str, Any]] | None, cfg: dict[str, Any]
             ) -> dict[str, Any] | None:
    """Public alias for ``_find_run`` (GUI consumers, e.g. the Compare cone)."""
    return _find_run(runs, cfg)


def aggregate(rows: list[dict[str, Any]], method_cfg: dict[str, Any],
              clip_ids: list[str] | None = None) -> dict[str, Any]:
    """Per-clip accuracy table + summary for one method config across a run.

    ``rows`` are a run document's ``rows`` list, optionally enriched with a
    ``plane_doc`` per row (the GUI loads each clip's plane from
    ``planes/<setup_id>.pool.json``; this function is pure and only reads what it
    is handed).  ``method_cfg`` is ``{method, n_points, selection}`` and selects
    one run cell per clip; ``clip_ids`` narrows the table (``None`` = every clip,
    preserving the run's order).

    A clip with no usable GT — or whose selected config has no run cell — still
    appears in ``clips`` (it shows in/px only) but is excluded from every error,
    bin, and judge statistic.  Judge statistics cover only judge-present clips.
    ``within_2`` / ``within_4`` / ``within_6`` are per-clip pass marks on the
    **score** error; the summary ``bins`` aggregate the same errors across clips.
    """
    if clip_ids is not None:
        wanted = set(clip_ids)
        rows = [r for r in rows if r.get("clip_id") in wanted]

    clips: list[dict[str, Any]] = []
    score_errs: list[float] = []
    score_pcts: list[float] = []
    judge_errs: list[float] = []
    judge_pcts: list[float] = []

    for row in rows:
        clip_id = row.get("clip_id")
        gt = row.get("gt") or {}
        plane_doc = row.get("plane_doc")
        run_cell = _find_run(row.get("runs"), method_cfg)
        score = (run_cell or {}).get("score") or {}

        in_per_px_nominal = nominal_in_per_px(plane_doc)
        in_per_px_local = (local_in_per_px_at_entry(plane_doc, score.get("entry_px"))
                           if score.get("entry_px") is not None else None)

        score_error = _metrics.score_error_in(score, gt)
        score_pct = score_error_pct(score, gt)
        judge_error = _metrics.judge_error_in(gt)
        judge_pct = judge_error_pct(gt)

        within = {b: (score_error is not None and score_error <= float(b))
                  for b in (2, 4, 6)}

        clips.append({
            "clip_id": clip_id,
            "n_points": method_cfg.get("n_points"),
            "in_per_px_nominal": in_per_px_nominal,
            "in_per_px_local": in_per_px_local,
            "score_error_in": score_error,
            "score_error_pct": score_pct,
            "judge_error_in": judge_error,
            "judge_error_pct": judge_pct,
            "within_2": within[2],
            "within_4": within[4],
            "within_6": within[6],
        })

        if score_error is not None:
            score_errs.append(score_error)
        if score_pct is not None:
            score_pcts.append(score_pct)
        if judge_error is not None:
            judge_errs.append(judge_error)
            judge_pcts.append(judge_pct)  # non-None whenever judge_error is

    return {
        "clips": clips,
        "summary": {
            "n_clips": len(clips),
            "n_with_judge": len(judge_errs),
            "mean_in": _metrics.mean(score_errs),
            "median_in": median(score_errs),
            "mean_pct": _metrics.mean(score_pcts),
            "median_pct": median(score_pcts),
            "bins": bin_counts(score_errs),
            "judge_mean_in": _metrics.mean(judge_errs),
            "judge_median_in": median(judge_errs),
            "judge_mean_pct": _metrics.mean(judge_pcts),
        },
    }
