"""``arc_2d`` — the v1 scoring method: a 2-D image-space parabola.

The bench's first conforming method.  It fits the fork's pixel-space parabola
(``trajectory.fit_parabolic_trajectory``, y = A·x² + B·x + C) over the *selected*
accepted detections, takes the fitted curve's **terminal point** as the entry,
and maps that pixel to a world distance through the plane document's own
homography (``pool.intersection.entry_coordinate`` — the same primitive the
parent's ``measure.plane_distance`` uses).

v1 crossing rule (ROADMAP §9.1, resolved at build time): the entry is the fitted
curve evaluated at the **last selected detection's x** — a trajectory-smoothed
final point.  The 2-D fit carries no Z, so it cannot itself locate the water
plane; extrapolating the fitted curve to the waterline is a follow-up option the
contract is deliberately agnostic to (add a ``crossing`` config key + a new fit
without touching the registry or the metrics).

The three axes (ROADMAP §6.6) live in ``config``:
  * ``selection``: ``all | last_n | top_kp_conf_n | flight_run_only | ransac_inliers``
  * ``n_points``: an int cap, or ``"all"``
  * ``frame_size``: ``[width, height]`` in px (from the manifest ``meta``)
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from dog_diving import trajectory as _trajectory
from dog_diving.pool import calib as _calib
from dog_diving.pool import intersection as _inter

from . import contract

METHOD = "arc_2d"
SELECTIONS = ("all", "last_n", "top_kp_conf_n", "flight_run_only", "ransac_inliers")
#: Degree-2 fit needs >= 3 points.  Deliberately lower than the fork's 6-point
#: gate because the bench's ``n_points`` axis sweeps down to 4.
MIN_POINTS = 3
RANSAC_THR_PX = 20.0


def _cap(n_points: Any, length: int) -> int:
    return length if n_points in ("all", None) else min(int(n_points), length)


def _ransac_inliers(points: list[dict]) -> list[dict]:
    t = np.asarray([float(p["frame"]) for p in points], dtype=float)
    x = np.asarray([float(p["x"]) for p in points], dtype=float)
    y = np.asarray([float(p["y"]) for p in points], dtype=float)
    fit = _trajectory.robust_parabola(t, x, y, RANSAC_THR_PX)
    if fit is None or fit.get("inliers") is None:
        return points
    return [p for p, keep in zip(points, fit["inliers"]) if keep]


def _select(points: list[dict], selection: str, n_points: Any) -> list[dict]:
    pts = list(points)
    if not pts:
        return pts
    if selection == "all":
        sel = pts
    elif selection == "last_n":
        return pts[-_cap(n_points, len(pts)):]
    elif selection == "top_kp_conf_n":
        k = _cap(n_points, len(pts))
        sel = sorted(pts, key=lambda p: -float(p.get("kp_conf", 0.0)))[:k]
        sel.sort(key=lambda p: p["frame"])
    elif selection == "flight_run_only":
        runs = _trajectory.temporal_runs(pts)
        i0, i1 = max(runs, key=lambda r: r[1] - r[0]) if runs else (0, len(pts))
        sel = pts[i0:i1]
    elif selection == "ransac_inliers":
        sel = _ransac_inliers(pts)
    else:
        raise ValueError(f"selection {selection!r} not in {SELECTIONS}")
    if n_points not in ("all", None):
        sel = sel[-_cap(n_points, len(sel)):]
    return sel


def arc_2d(points: Sequence[dict], plane_doc: dict | None, config: dict[str, Any]) -> dict[str, Any]:
    selection = config.get("selection", "all")
    n_points = config.get("n_points", "all")
    frame_size = config.get("frame_size")
    result = contract.new_result(METHOD, {
        "n_points": n_points, "selection": selection, "frame_size": frame_size})

    if not plane_doc or not (plane_doc.get("plane") or {}):
        result["status"] = "no_calibration"
        result["warnings"].append("no plane document — the distance needs the plane homography")
        return result
    if not frame_size or len(frame_size) != 2:
        result["status"] = "error"
        result["warnings"].append("config['frame_size'] must be [width, height]")
        return result
    w, h = int(frame_size[0]), int(frame_size[1])

    sel = _select(list(points), selection, n_points)
    result["fit"]["n_selected"] = len(sel)
    if len(sel) < MIN_POINTS:
        result["status"] = "few_points"
        result["warnings"].append(f"selected {len(sel)} points; arc_2d needs >= {MIN_POINTS}")
        return result

    last_x = float(sel[-1]["x"])
    curve, contact, coeffs = _trajectory.fit_parabolic_trajectory(
        sel, w, h, min_points=MIN_POINTS, x_end=last_x)
    if coeffs is None:
        result["status"] = "no_fit"
        result["warnings"].append("parabola fit rejected (insufficient x-span or fit failure)")
        return result

    # v1 entry = the fitted curve's terminal point (see module docstring).
    entry_px = curve[-1] if curve else None
    xs = np.asarray([float(p["x"]) for p in sel])
    ys = np.asarray([float(p["y"]) for p in sel])
    resid = coeffs["a"] * xs * xs + coeffs["b"] * xs + coeffs["c"] - ys
    result["fit"].update({
        "coeffs": coeffs,
        "n_points_used": len(sel),
        "n_curve_points": len(curve),
        "curve": [[float(px), float(py)] for px, py in curve],
        "rms_px": float(np.sqrt(np.mean(resid * resid))),
        "contact_px": list(contact) if contact else None,
        "last_point_frame": int(sel[-1]["frame"]),
    })
    if entry_px is None:
        result["status"] = "no_crossing"
        result["warnings"].append("no fitted curve produced an entry point")
        return result

    lens = _calib.lens_from_document(plane_doc)
    pose = _calib.pose_from_document(plane_doc)
    water = _calib.water_from_document(plane_doc)
    H = np.asarray(plane_doc["homography"], dtype=float).reshape(3, 3)
    entry = _inter.entry_coordinate(pose, water, entry_px, homography=H,
                                    extent=plane_doc.get("extent"), lens=lens)
    if entry.get("X_ft") is None:
        result["status"] = "no_crossing"
        result["warnings"].extend(entry.get("notes") or ["the entry point could not be mapped"])
        return result

    result["entry_px"] = [float(entry_px[0]), float(entry_px[1])]
    result["x_px"] = float(entry["x_px"])
    result["y_px"] = float(entry["y_px"])
    result["distance_in"] = float(entry["distance_in"])
    result["distance_ft"] = float(entry["X_ft"])
    result["entry_frame"] = None  # the entry is an extrapolated point, not a real frame
    if entry.get("flags", {}).get("outside_extent"):
        result["warnings"].append("entry maps OUTSIDE the plane's extent")
    return result
