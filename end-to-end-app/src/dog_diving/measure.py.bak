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
from . import trajectory as _trajectory
from .pool import calib as _calib
from .pool import geometry as _geometry
from .pool import intersection as _inter
from .pool import lens as _lens
from .pool import pipeline as _pipeline

ROUTE = "plane"
#: What ``entry_coordinate`` tags a record built from a **fitted 3-D trajectory** (the plane method's
#: route).  Named here as well as in ``pool.intersection`` because the Results pane switches on it to
#: decide which rows make sense, and a literal repeated in two files is a literal that will drift.
ROUTE_ARC3D = "arc3d"
#: The two scoring methods the operator can switch between in Review.  ``"arc"`` is the rule in the
#: module docstring (final detection assumed on the plane); ``"plane"`` is the 3-D trajectory plane
#: (a world-space arc extrapolated to ``Z = z_water``).  They are two *answers to different
#: questions*, not two estimates of one, which is why they are named rather than blended.
SPACE_ARC = "arc"
SPACE_PLANE = "plane"
SPACES = (SPACE_ARC, SPACE_PLANE)


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


# ============================================================================
# The second scoring method: the trajectory PLANE (3-D world arc)
# ============================================================================
def trajectory_plane_fit(
    traj: Any,
    plane_doc: dict[str, Any] | None,
    *,
    excluded: Iterable[int] | None = (),
    frame: int | None = None,
    margin_ft: float = 0.0,
) -> dict[str, Any]:
    """Track -> world points -> world arc -> the water crossing.  One computation, two callers.

    Review draws this curve **and** scores with it, so it is computed once here rather than twice in
    the pane: a pane that drew one fit and measured another would be the exact failure this project
    keeps finding (the arc standing in for the detection).

    The chain, each step being a function that reports its own failure rather than returning a
    placeholder:

    * ``kept_points`` — Review's deselections apply, because a point the operator rejected must not
      pull the fit (and must not be the extrapolation's anchor);
    * ``pipeline.undistort_track`` — the document's lens, applied once, up front;
    * ``pipeline.track_to_world`` — pixels to rays to one vertical plane to 3-D points, **single
      pass** (see that function for its measured bias);
    * ``trajectory.build_trajectory(space="3d")`` — the same robust estimator, in feet;
    * ``intersection.entry_coordinate(trajectory=...)`` — where ``Z(t)`` crosses the water.

    Returns ``{status, detail, plane, traj3d, entry, curve, rms_ft, n_points}``.  ``status`` is
    ``"ok"``, ``"no_calibration"``, ``"few_points"``, ``"no_fit"``, ``"no_crossing"`` or a
    ``track_to_world`` failure — every one of which the caller reports as a warning, because no
    number is better than a number with a guessed input.
    """
    out: dict[str, Any] = {"status": None, "detail": "", "plane": None, "traj3d": None,
                           "entry": None, "curve": [], "rms_ft": None, "n_points": 0}
    fail, parts = _plane_doc_parts(plane_doc)
    if fail is not None:
        out.update(fail)
        return out
    pose, water, lens = parts["pose"], parts["water"], parts["lens"]
    H, extent = parts["H"], parts["extent"]

    pts = kept_points(traj, excluded)
    if frame is not None:
        pts = [p for p in pts if int(p["frame"]) <= int(frame)]
    # Operator order 2026-10-28: fit the last 20 detected points only.
    if len(pts) > 20:
        pts = pts[-20:]
    out["n_points"] = len(pts)
    if len(pts) < 3:
        out.update({"status": "few_points",
                    "detail": (f"{len(pts)} kept detection(s) at or before frame {frame} — the 3-D "
                               f"arc needs at least three to be a curve")})
        return out

    rows = _pipeline.undistort_track(pts, lens)
    world = _pipeline.track_to_world(pose, water, rows)
    out["plane"] = {k: (v.as_dict() if hasattr(v, "as_dict") else v)
                    for k, v in world.items() if k != "points"}
    out["points3d"] = world.get("points")
    if world["status"] != "ok":
        out.update({"status": f"world_{world['status']}", "detail": world["detail"]})
        return out
    out["rms_ft"] = world.get("rms_ft")

    traj3d = _trajectory.build_trajectory(world["points"], space="3d", plane=world["plane"])
    out["traj3d"] = traj3d
    if traj3d.coeffs is None:
        out.update({"status": "no_fit",
                    "detail": ("the world points were built but no 3-D arc survived the fit (fewer "
                               "than three in one run)")})
        return out

    entry = _inter.entry_coordinate(pose, water, None, extent=extent, margin_ft=margin_ft,
                                    homography=H, trajectory=traj3d)
    out["entry"] = entry
    if entry.get("X_ft") is None:
        out.update({"status": "no_crossing",
                    "detail": "; ".join(entry.get("notes") or ["the curve never met the water"])})
        return out

    # The drawn curve: from the first flight frame to the crossing (and a little past it, so the
    # operator can see the extrapolation land on the water rather than stop in mid-air).
    first = (traj3d.airborne_frame_range or [0, 0])[0]
    last = float(entry.get("crossing_frame") or (traj3d.airborne_frame_range or [0, 1])[1])
    t0, t1 = float(first), max(last, float(first) + 1.0)
    frames = [t0 + (t1 - t0) * i / 48.0 for i in range(49)]
    out["curve"] = _geometry.project_curve(pose, traj3d.coeffs, frames)
    out["status"] = "ok"
    out["detail"] = entry.get("notes", [""])[0] if entry.get("notes") else ""
    return out


def _plane_doc_parts(plane_doc: dict[str, Any] | None
                     ) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """``(failure, parts)`` for a pool document — exactly one of the two is meaningful.

    ``failure`` is ``None`` when the document can answer, and a ``{"status", "detail"}`` dict when it
    cannot; ``parts`` carries the rebuilt ``pose``/``water``/``lens``/``H``/``extent``.  One function,
    one contract, so the "is there a water plane at all" test cannot drift between the two places
    that need it.
    """
    if not plane_doc:
        return ({"status": "no_calibration",
                 "detail": "no pool calibration document is loaded"}, {})
    if not (plane_doc.get("plane") or plane_doc.get("water")):
        return ({"status": "no_calibration",
                 "detail": ("the loaded calibration has neither a plane block nor a water level, so "
                            "there is no water for the arc to cross")}, {})
    H = (np.asarray(plane_doc["homography"], dtype=float).reshape(3, 3)
         if plane_doc.get("homography") is not None else None)
    return None, {"pose": _calib.pose_from_document(plane_doc),
                  "water": _calib.water_from_document(plane_doc),
                  "lens": _calib.lens_from_document(plane_doc),
                  "H": H, "extent": plane_doc.get("extent")}


def trajectory_plane_distance(
    traj: Any,
    plane_doc: dict[str, Any] | None,
    *,
    manual_point: Sequence[float] | None = None,
    frame: int | None = None,
    margin_ft: float = 0.0,
    excluded: Iterable[int] | None = (),
    frame_is_operator_choice: bool = False,
    fit: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """``X`` of the water entry where the fitted **world** arc crosses the water plane.

    Same output shape as :func:`plane_distance`, so Review, the Results pane and ``result.json``
    read one schema whichever method answered; ``route`` says which one did (``"plane"`` vs
    ``"arc3d"``).  ``fit`` lets the caller pass a fit it already has (Review caches it for the
    canvas), so the drawn curve and the number cannot come from two different runs.

    A manual click has no track and therefore no curve: this returns a warning rather than quietly
    falling back to the single-pixel rule, because that fallback would report the other method's
    number under this method's name.
    """
    out: dict[str, Any] = {
        "route": "arc3d", "total_inches": None, "display": "n/a", "sigma_in": None,
        "distance_ft": None, "transverse_ft": None, "frame": None, "point_source": "unset",
        "x_px": None, "y_px": None, "warnings": [], "point_origin": None,
        "excluded_frames": sorted({int(f) for f in (excluded or ())}),
        "entry_override": bool(frame_is_operator_choice and frame is not None),
    }
    if manual_point is not None:
        out["point_source"], out["point_origin"] = "manual", "click"
        out["warnings"].append(
            "the trajectory-plane method fits a curve to the detections; a single clicked point has "
            "no curve to fit, so it cannot answer here — switch Review back to 'Trajectory arc' for "
            "a clicked measurement, or re-include the detections")
        return out
    if fit is None:
        fit = trajectory_plane_fit(traj, plane_doc, excluded=excluded, frame=frame,
                                   margin_ft=margin_ft)
    if fit.get("status") != "ok":
        out["warnings"].append(f"the trajectory-plane method could not answer ({fit.get('status')}): "
                               f"{fit.get('detail')}")
        return out

    entry = fit["entry"]
    inches = float(entry["distance_in"])
    out.update({
        "total_inches": inches,
        "distance_ft": float(entry["X_ft"]),
        "transverse_ft": float(entry["Y_ft"]),
        "display": _distance.format_feet_inches(inches),
        "x_px": (None if entry.get("x_px") is None else float(entry["x_px"])),
        "y_px": (None if entry.get("y_px") is None else float(entry["y_px"])),
        "water_z_ft": float(entry["Z_ft"]),
        "homography_agreement_ft": entry.get("homography_agreement_ft"),
        "flags": dict(entry.get("flags") or {}),
        "frame": (None if entry.get("crossing_frame") is None
                  else int(round(float(entry["crossing_frame"])))),
        "point_source": "model", "point_origin": "fitted_curve",
        "crossing_frame": entry.get("crossing_frame"),
        "extrap_frames": entry.get("extrap_frames"),
        "arc_rms_ft": entry.get("arc_rms_ft"),
        "plane_rms_ft": fit.get("rms_ft"),
    })
    # The entry's own notes are the honest provenance; keep the two that change how the number
    # should be read, and drop the mechanical ones.
    for note in (entry.get("notes") or []):
        if "UNVERIFIED" in note or "crosses the water plane" in note or "PAST the last" in note:
            out["warnings"].append(note)
    if out["flags"].get("arc3d_plane_too_wide"):
        out["warnings"].append(
            "the track fits no single vertical plane, which is what this method assumes")
    if out["flags"].get("outside_extent"):
        out["warnings"].append(
            "the entry maps OUTSIDE the plane's extent — reported, but it means the detections or "
            "the plane is wrong")
    return out
