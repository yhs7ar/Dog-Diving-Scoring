"""The bench's deliverable: per-clip error metrics and their averages.

Ground truth and judge live in one ``gt`` dict (``ground_truth.json`` shape)::

    { "x_px", "y_px", "gt_in", "judge_in", "judge_present", ... }

For a scored ``ScoreResult`` ``s`` and ground truth ``gt``:

  * **pixel error**  = ‖ s.entry_px − (gt.x_px, gt.y_px) ‖₂   (pixels)
  * **score error**  = | s.distance_in − gt.gt_in |           (inches)
  * **judge error**  = | gt.judge_in − gt.gt_in |             (inches; null when no judge)

Averages are the **mean** of each metric over the clips in a run; a clip with
``judge_present = false`` contributes to the pixel/score averages and is
excluded from the judge average (and is still fully comparable auto-vs-GT).
"""
from __future__ import annotations

import math
from typing import Any, Iterable


def pixel_error_px(score: dict[str, Any], gt: dict[str, Any]) -> float | None:
    ep = score.get("entry_px")
    if ep is None or gt.get("x_px") is None or gt.get("y_px") is None:
        return None
    return float(math.hypot(float(ep[0]) - float(gt["x_px"]),
                            float(ep[1]) - float(gt["y_px"])))


def score_error_in(score: dict[str, Any], gt: dict[str, Any]) -> float | None:
    if score.get("distance_in") is None or gt.get("gt_in") is None:
        return None
    return float(abs(score["distance_in"] - gt["gt_in"]))


def judge_error_in(gt: dict[str, Any]) -> float | None:
    if not gt.get("judge_present") or gt.get("judge_in") is None or gt.get("gt_in") is None:
        return None
    return float(abs(gt["judge_in"] - gt["gt_in"]))


def score_accuracy_pct(score: dict[str, Any], gt: dict[str, Any]) -> float | None:
    """The score's distance accuracy as a percentage of ground truth.

    ``100`` = exact; declines toward 0 as ``|score − GT|`` grows, clamped to
    ``[0, 100]``.  ``None`` when the score or GT distance is missing (or GT is
    non-positive, so a percentage is undefined).
    """
    err = score_error_in(score, gt)
    gt_in = gt.get("gt_in")
    if err is None or gt_in is None or float(gt_in) <= 0:
        return None
    return float(max(0.0, min(100.0, 100.0 * (1.0 - err / float(gt_in)))))


def compute_errors(score: dict[str, Any], gt: dict[str, Any]) -> dict[str, float | None]:
    """All error metrics for one (clip, method-config) cell."""
    return {
        "pixel_error_px": pixel_error_px(score, gt),
        "score_error_in": score_error_in(score, gt),
        "judge_error_in": judge_error_in(gt),
        "accuracy_pct": score_accuracy_pct(score, gt),
    }


def mean(values: Iterable[float | None]) -> float | None:
    """Mean of the non-None values; None for an empty/all-None input."""
    xs = [float(v) for v in values if v is not None]
    return float(sum(xs) / len(xs)) if xs else None
