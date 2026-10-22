"""The measurement, with the plane as the only ground truth (2026-10-20).

Operator's rule, quoted so it cannot drift: *"Assume that the final point detected for the dog's
tailset is the point of intersection with the plane. The +x distance at this point will be the
reported distance."*

So there is **one** measurement in the app:

1. take the **last accepted tail-set detection** in the trajectory (or the operator's manual point,
   which is the same rule applied by hand);
2. assume that pixel lies **on the water plane** the operator aligned (PLAN §15) — no arc, no
   contact-frame candidates, no 1-D tick fit;
3. map it through the plane document's **own homography** and report its ``X`` in inches.

That is fewer moving parts than any previous route and it removes the two failure modes this
project kept re-finding: a fitted arc standing in for a detection, and a px->in fit on painted marks
whose span and localisation set the error budget.

What this replaced (all archived under ``archive/2026-10-20-pre-plane-only/``):
``calibration.py`` + ``calibrate_tool.py`` (the 1-D mark calibration and its click tool),
``contact.py``'s candidate ranking as the *measurement* input, and ``distance.measure_contact`` as the
number of record.  Those modules stay importable so the CLI and the old artifacts still open; the app
no longer calls them for a distance.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np

from . import distance as _distance
from .pool import calib as _calib
from .pool import intersection as _inter
from .pool import lens as _lens

ROUTE = "plane"


def kept_points(traj: Any, excluded: Iterable[int] | None = ()) -> list[dict[str, Any]]:
    """Every accepted detection the operator has NOT deselected in Review, in frame order.

    ``excluded`` is the Review step's frame set (see ``app_gui.ReviewState.excluded``).  Deselecting a
    detection there means *"this point is not my dog"* — so it must not be drawn on the arc, must not
    be fitted, and must not become the measured point.  All three read the same set through this
    function, which is what keeps the number and the picture from disagreeing.
    """
    drop = {int(f) for f in (excluded or ())}
    points = list(getattr(traj, "points", None) or [])
    return [p for p in points if int(p["frame"]) not in drop]


def final_point(traj: Any, excluded: Iterable[int] | None = ()) -> dict[str, Any] | None:
    """The final accepted tail-set detection (highest frame number), or None.

    With ``excluded`` set, the deselected frames are skipped before the maximum is taken, so the
    measurement falls back to the highest-frame detection the operator kept.
    """
    points = kept_points(traj, excluded)
    if not points:
        return None
    return max(points, key=lambda p: int(p["frame"]))


def row_at(traj: Any, frame: int) -> dict[str, Any] | None:
    """The accepted detection at ``frame``, or None.  Used by Review to place its marker."""
    for p in (getattr(traj, "points", None) or []):
        if int(p["frame"]) == int(frame):
            return p
    return None


def plane_distance(
    traj: Any,
    plane_doc: dict[str, Any] | None,
    *,
    manual_point: Sequence[float] | None = None,
    frame: int | None = None,
    margin_ft: float = 0.0,
    excluded: Iterable[int] | None = (),
    frame_is_operator_choice: bool = False,
) -> dict[str, Any]:
    """``X`` of the final detected point, in inches, from the plane homography alone.

    ``manual_point`` overrides the point (a click in Review) under the same rule: the clicked pixel
    is taken to be on the plane.  ``frame`` restricts the sample to one frame's detection (Review's
    frame stepper); the default is the *final* detection, which is the scoring rule.

    ``excluded`` is Review's deselection set: those frames are not candidates for the final point and
    therefore cannot set the distance.  This is the other-dog case — a confident detection of a second
    dog later in the clip would otherwise be the last point and score its position instead of the
    jumping dog's.
    """
    out: dict[str, Any] = {
        "route": ROUTE, "total_inches": None, "display": "n/a", "sigma_in": None,
        "distance_ft": None, "transverse_ft": None, "frame": None, "point_source": "unset",
        "x_px": None, "y_px": None, "warnings": [], "point_origin": None,
        "excluded_frames": sorted({int(f) for f in (excluded or ())}),
        "entry_override": bool(frame_is_operator_choice and frame is not None),
    }
    if not plane_doc or not (plane_doc.get("plane") or {}):
        out["warnings"].append(
            "no plane calibration: the distance needs the plane the operator aligned (the 1-D "
            "mark calibration is archived and no longer produces a number)")
        return out

    lens = _calib.lens_from_document(plane_doc)
    pose = _calib.pose_from_document(plane_doc)
    water = _calib.water_from_document(plane_doc)
    H = np.asarray(plane_doc["homography"], dtype=float).reshape(3, 3)

    if manual_point is not None:
        px = (float(manual_point[0]), float(manual_point[1]))
        source, origin = "manual", "click"
        out["frame"] = None if frame is None else int(frame)
    else:
        drop = {int(f) for f in (excluded or ())}
        if frame is not None and int(frame) in drop:
            out["warnings"].append(
                f"frame {int(frame)} was deselected in Review, so it cannot supply the measured point")
            return out
        row = row_at(traj, frame) if frame is not None else final_point(traj, drop)
        if row is None:
            out["warnings"].append(
                f"no accepted detection at frame {frame}" if frame is not None
                else ("every accepted detection was deselected in Review, so there is no final "
                      "point — re-include one, or click the true measurement point" if drop
                      else "the trajectory has no accepted detections, so there is no final point"))
            return out
        px = (float(row["x"]), float(row["y"]))
        source, origin = "model", "detection"
        out["frame"] = int(row["frame"])
        out["kp_conf"] = row.get("kp_conf")

    out["x_px"], out["y_px"] = px
    out["point_source"], out["point_origin"] = source, origin

    entry = _inter.entry_coordinate(
        pose, water, px, homography=H, extent=plane_doc.get("extent"),
        margin_ft=margin_ft, lens=lens,
    )
    if entry.get("X_ft") is None:
        out["warnings"].extend(entry.get("notes") or ["the point could not be mapped"])
        return out

    inches = float(entry["distance_in"])
    out.update({
        "total_inches": inches,
        "distance_ft": float(entry["X_ft"]),
        "transverse_ft": float(entry["Y_ft"]),
        "display": _distance.format_feet_inches(inches),
        "x_px": float(entry["x_px"]), "y_px": float(entry["y_px"]),
        "water_z_ft": float(water.z_ft),
        "homography_agreement_ft": entry.get("homography_agreement_ft"),
        "flags": entry.get("flags") or {},
    })
    if out["flags"].get("outside_extent"):
        out["warnings"].append(
            "the point maps OUTSIDE the plane's extent — it is reported, but it means either the "
            "detection or the plane is wrong")
    return out
