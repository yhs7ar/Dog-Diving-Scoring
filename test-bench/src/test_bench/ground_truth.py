"""Ground truth + judge — the reference the bench scores against.

The ground-truth point is a *pixel* the operator places (the true tailset∩water
intersection).  ``gt_in`` is **always recomputed** from that pixel through the
plane document's own homography (``intersection.entry_coordinate`` — the same
primitive ``arc_2d`` and the parent's ``measure.plane_distance`` use); it is
never stored as a free-standing number, so re-drawing the plane re-derives the
GT distance.  The judge's distance is an optional, independently entered number.

Schema: ``test_bench.ground_truth/1`` (ROADMAP §5.2).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from dog_diving.pool import calib as _calib
from dog_diving.pool import intersection as _inter

SCHEMA = "test_bench.ground_truth/1"
GT_IN_SOURCE = "plane_homography"


def gt_in_from_pixel(plane_doc: dict[str, Any], x_px: float, y_px: float
                     ) -> tuple[float | None, dict[str, Any], list[str]]:
    """``(gt_in_inches, flags, notes)`` for one pixel, via the plane's homography."""
    if not plane_doc or not (plane_doc.get("plane") or {}):
        raise ValueError("a plane document is required to derive gt_in from a pixel")
    lens = _calib.lens_from_document(plane_doc)
    pose = _calib.pose_from_document(plane_doc)
    water = _calib.water_from_document(plane_doc)
    H = np.asarray(plane_doc["homography"], dtype=float).reshape(3, 3)
    entry = _inter.entry_coordinate(
        pose, water, (float(x_px), float(y_px)), homography=H,
        extent=plane_doc.get("extent"), lens=lens)
    gt_in = entry.get("distance_in")
    return (float(gt_in) if gt_in is not None else None,
            entry.get("flags") or {}, entry.get("notes") or [])


def set_ground_truth(plane_doc: dict[str, Any], *, frame: int, x_px: float, y_px: float,
                     judge_in: float | None = None, updated: str | None = None
                     ) -> dict[str, Any]:
    """Build a ground-truth record; ``gt_in`` is derived, never passed in."""
    gt_in, flags, notes = gt_in_from_pixel(plane_doc, x_px, y_px)
    gt: dict[str, Any] = {
        "schema": SCHEMA,
        "frame": int(frame),
        "x_px": float(x_px),
        "y_px": float(y_px),
        "gt_in": gt_in,
        "gt_in_source": GT_IN_SOURCE,
        "judge_in": float(judge_in) if judge_in is not None else None,
        "judge_present": judge_in is not None,
        "updated": updated,
    }
    if notes:
        gt["warnings"] = notes
    if flags.get("outside_extent"):
        gt.setdefault("warnings", []).append("GT pixel maps OUTSIDE the plane's extent")
    return gt


def set_judge(gt: dict[str, Any], judge_in: float | None, updated: str | None = None) -> dict[str, Any]:
    """Update the judge distance on an existing GT record (the GT pixel is untouched)."""
    gt = dict(gt)
    gt["judge_in"] = float(judge_in) if judge_in is not None else None
    gt["judge_present"] = judge_in is not None
    if updated is not None:
        gt["updated"] = updated
    return gt


def write_ground_truth(gt: dict[str, Any], clip_dir: str | Path) -> Path:
    path = Path(clip_dir) / "ground_truth.json"
    path.write_text(json.dumps(gt, indent=2), encoding="utf-8")
    return path


def read_ground_truth(clip_dir: str | Path) -> dict[str, Any]:
    return json.loads((Path(clip_dir) / "ground_truth.json").read_text(encoding="utf-8"))
