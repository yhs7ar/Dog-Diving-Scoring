"""The test-suite runner: grid the three axes, score every cell, average.

A prepared clip is ``{clip_id, points, plane_doc, gt, frame_size}``.  A method
config is ``{method, n_points, selection}``; ``frame_size`` rides with the clip
(from its manifest ``meta``), so the same config list applies across clips.

``run_suite`` emits the results document (ROADMAP §5.4): one row per clip with
its ``gt`` and one ``runs`` cell per config (ScoreResult + errors), plus
``overall`` and ``per_method`` averages.
"""
from __future__ import annotations

from typing import Any, Iterable, Sequence

from . import metrics
from .scoring import registry


def grid_configs(n_points_values: Sequence[Any], selection_values: Sequence[str],
                 methods: Sequence[str] = ("arc_2d",)) -> list[dict[str, Any]]:
    """The cross-product of the axes (method x n_points x selection)."""
    if isinstance(methods, str):
        methods = (methods,)
    return [{"method": m, "n_points": n, "selection": s}
            for m in methods for n in n_points_values for s in selection_values]


def _errors_for(clip: dict, cfg: dict) -> dict[str, Any]:
    score = registry.run(
        cfg["method"], clip["points"], clip["plane_doc"],
        {"n_points": cfg["n_points"], "selection": cfg["selection"],
         "frame_size": clip["frame_size"]})
    err = metrics.compute_errors(score, clip["gt"])
    return {"method": cfg["method"], "n_points": cfg["n_points"],
            "selection": cfg["selection"], **err, "score": score}


def run_suite(clips: Iterable[dict], configs: Sequence[dict]) -> dict[str, Any]:
    rows = []
    for clip in clips:
        runs = [_errors_for(clip, cfg) for cfg in configs]
        rows.append({"clip_id": clip["clip_id"], "gt": clip["gt"], "runs": runs})

    all_runs = [run for row in rows for run in row["runs"]]
    judge_present_runs = [row["gt"] for row in rows if row["gt"].get("judge_present")]

    averages = {
        "overall": {
            "pixel_error_px": metrics.mean(r["pixel_error_px"] for r in all_runs),
            "score_error_in": metrics.mean(r["score_error_in"] for r in all_runs),
            "judge_error_in": metrics.mean(metrics.judge_error_in(g) for g in judge_present_runs),
        },
        "per_method": {},
    }
    for method in sorted({r["method"] for r in all_runs}):
        m_runs = [r for r in all_runs if r["method"] == method]
        averages["per_method"][method] = {
            "pixel_error_px": metrics.mean(r["pixel_error_px"] for r in m_runs),
            "score_error_in": metrics.mean(r["score_error_in"] for r in m_runs),
            "judge_error_in": metrics.mean(r["judge_error_in"] for r in m_runs),
        }
    return {"rows": rows, "averages": averages}


def clip_provenance(manifest: dict, plane_doc: dict | None, gt: dict | None) -> dict[str, Any]:
    """What a run's metrics depend on, recorded so a re-do is detected on load.

    ROADMAP §6.4/§9.3: re-place GT, re-detect with a different model, or re-assign
    the plane all change one of these three, so a stale ``results/*.json`` is
    flagged instead of silently shown.  Deselecting a detection (Review) changes
    ``excluded_frames`` and must invalidate the run the same way.
    """
    return {
        "model_sha": (manifest.get("detections") or {}).get("model_sha"),
        "plane_setup_id": (plane_doc or {}).get("setup_id"),
        "gt_updated": (gt or {}).get("updated"),
        "excluded_frames": (manifest.get("excluded_frames") or []),
    }


def is_stale(recorded: dict[str, Any] | None, manifest: dict,
             plane_doc: dict | None, gt: dict | None) -> bool:
    """True when a run's recorded provenance no longer matches the current inputs.

    A missing/None ``recorded`` (older run, or one written before provenance was
    captured) is conservatively treated as stale rather than silently trusted.
    """
    if not recorded:
        return True
    return recorded != clip_provenance(manifest, plane_doc, gt)
