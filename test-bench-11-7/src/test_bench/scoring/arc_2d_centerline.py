"""``arc_2d_centerline`` — the centerline-pinned 2-D image-space parabola.

A second conforming method, derived from ``arc_2d`` by changing **only** the
crossing rule.  It fits the same unweighted pixel-space parabola
(``trajectory.fit_parabolic_trajectory``, y = A·x² + B·x + C) over the same
selected detections, but instead of taking the fitted curve's terminal point as
the entry, it assumes the dog lands on the pool centerline and locates the entry
as the fitted parabola's intersection with the *image of the centerline*.

Centerline (world frame): ``Y_c = (y_min + y_max)/2`` from the document's
``extent``; when the extent carries no y-bounds, the plane's own corners
(``plane.corner_world``) supply them; when neither is present the method has no
centerline to pin to and reports ``error``.

Crossing rule: the centerline is a world line (Y = Y_c); its image is a line in
pixel space (two world points pushed through ``apply_homography`` then the
lens's forward ``distort_points`` so it lives in the *raw* space the parabola is
fitted in).  Substituting the parabola into that line's equation gives a
quadratic in x with up to two real roots.  The entry is the root with the
**largest world X that still lies within the pool** (far edge + a small splash
margin) — the parabola's other root is an extrapolation far outside the pool
where the homography diverges, and is excluded.  The landing is thus chosen by
pool geometry (dock at X = 0, jump +X, landing inside the pool), never by the
last detection.  No root within the pool → ``no_crossing``.

The fit itself is unchanged from ``arc_2d``: ``np.polyfit`` degree 2, all
selected points weighted equally.  ``x_end=last_x`` is retained only to bound
the sampled ``curve`` for the Compare overlay (the trajectory is based on the
last points, always including the last x), never to anchor the score.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from dog_diving import trajectory as _trajectory
from dog_diving.pool import calib as _calib
from dog_diving.pool import homography as _homography
from dog_diving.pool import intersection as _inter
from dog_diving.pool import lens as _lens
from dog_diving.pool import plane as _plane

from . import contract
from .arc_2d import MIN_POINTS, _select

METHOD = "arc_2d_centerline"

#: Root *selection* only: how far past the far edge (``x_max``) a centerline
#: crossing may sit and still count as the landing (a splash just past the far
#: wall).  The final ``entry_coordinate`` extent gate is still zero-margin.
SPLASH_MARGIN_FT = 5.0


def _centerline_ft(plane_doc: dict) -> tuple[float, float, float] | None:
    """``(y_min, y_max, yc)`` world feet from extent, else plane corners, else ``None``."""
    extent = plane_doc.get("extent") or {}
    y_min = extent.get("y_min")
    y_max = extent.get("y_max")
    if y_min is None or y_max is None:
        plane_block = plane_doc.get("plane") or {}
        corners = plane_block.get("corners") or {}
        length_ft = plane_block.get("length_ft")
        width_ft = plane_block.get("width_ft")
        if corners and length_ft is not None and width_ft is not None:
            ys = [float(_plane.corner_world(tag, float(length_ft), float(width_ft))[1])
                  for tag in corners]
            y_min, y_max = float(min(ys)), float(max(ys))
        else:
            return None
    y_min, y_max = float(y_min), float(y_max)
    return y_min, y_max, (y_min + y_max) / 2.0


def _real_roots(a: float, b: float, c: float) -> list[float]:
    """Real roots of ``a·x² + b·x + c = 0`` (degenerate linear case included)."""
    if abs(a) < 1e-12:
        if abs(b) < 1e-12:
            return []
        return [float(-c / b)]
    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        return []
    sq = float(np.sqrt(disc))
    return [float((-b + sq) / (2.0 * a)), float((-b - sq) / (2.0 * a))]


def arc_2d_centerline(points: Sequence[dict], plane_doc: dict | None,
                      config: dict[str, Any]) -> dict[str, Any]:
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

    lens = _calib.lens_from_document(plane_doc)
    pose = _calib.pose_from_document(plane_doc)
    water = _calib.water_from_document(plane_doc)
    H = np.asarray(plane_doc["homography"], dtype=float).reshape(3, 3)

    cl = _centerline_ft(plane_doc)
    if cl is None:
        result["status"] = "error"
        result["warnings"].append(
            "no pool centerline — extent has no y_min/y_max and the plane has no corners/width")
        return result
    _y_min, _y_max, yc = cl
    result["fit"]["centerline_yc_ft"] = yc

    sel = _select(list(points), selection, n_points)
    result["fit"]["n_selected"] = len(sel)
    if len(sel) < MIN_POINTS:
        result["status"] = "few_points"
        result["warnings"].append(f"selected {len(sel)} points; {METHOD} needs >= {MIN_POINTS}")
        return result

    last_x = float(sel[-1]["x"])
    curve, contact, coeffs = _trajectory.fit_parabolic_trajectory(
        sel, w, h, min_points=MIN_POINTS, x_end=last_x)
    if coeffs is None:
        result["status"] = "no_fit"
        result["warnings"].append("parabola fit rejected (insufficient x-span or fit failure)")
        return result

    xs = np.asarray([float(p["x"]) for p in sel])
    ys = np.asarray([float(p["y"]) for p in sel])
    resid = coeffs["a"] * xs * xs + coeffs["b"] * xs + coeffs["c"] - ys
    result["fit"].update({
        "coeffs": coeffs,
        "n_points_used": len(sel),
        "n_curve_points": len(curve),
        "curve": [[float(px), float(py)] for px, py in curve],
        "selected_px": [[float(p["x"]), float(p["y"])] for p in sel],
        "selected_frames": [int(p["frame"]) for p in sel],
        "curve_space": "px",
        "rms_px": float(np.sqrt(np.mean(resid * resid))),
        "contact_px": list(contact) if contact else None,
        "last_point_frame": int(sel[-1]["frame"]),
    })

    # --- the centerline image line, in the raw pixel space the parabola lives in ---
    extent = plane_doc.get("extent") or {}
    x0 = float(extent.get("x_min", 0.0))
    plane_block = plane_doc.get("plane") or {}
    x1 = float(extent.get("x_max") or plane_block.get("length_ft") or 50.0)
    px_und = _homography.apply_homography(
        H, np.array([[x0, yc], [x1, yc]], dtype=float))
    px_raw = _lens.distort_points(px_und, lens)
    p0, p1 = px_raw[0], px_raw[1]
    line = np.cross(np.array([p0[0], p0[1], 1.0]), np.array([p1[0], p1[1], 1.0]))
    result["fit"]["centerline_line_px"] = [
        [float(p0[0]), float(p0[1])], [float(p1[0]), float(p1[1])]]

    # --- substitute the parabola into the line: A·x² + B·x + C = 0 ---
    A = coeffs["a"] * line[1]
    B = coeffs["b"] * line[1] + line[0]
    C = coeffs["c"] * line[1] + line[2]
    roots = _real_roots(A, B, C)
    if not roots:
        result["status"] = "no_crossing"
        result["warnings"].append("the fitted parabola does not reach the pool centerline")
        return result

    # entry = the down-pool crossing (largest world X) *within the pool*.  The
    # parabola's other real root is an extrapolation far outside the pool (beyond
    # the far edge, where the homography diverges) and must not win — the pool
    # extent, not a detection, is what identifies the landing side.
    margin = SPLASH_MARGIN_FT
    entry_px: tuple[float, float] | None = None
    best_x = float("-inf")
    for xr in roots:
        yr = coeffs["a"] * xr * xr + coeffs["b"] * xr + coeffs["c"]
        pxu = _lens.undistort_points(np.array([[xr, yr]], dtype=float), lens)[0]
        XY = _homography.apply_homography_inv(
            H, np.array([[pxu[0], pxu[1]]], dtype=float))
        Xw = float(XY[0, 0])
        if (x0 - margin) <= Xw <= (x1 + margin) and Xw > best_x:
            best_x = Xw
            entry_px = (float(xr), float(yr))

    if entry_px is None:
        result["status"] = "no_crossing"
        result["warnings"].append(
            "no centerline crossing within the pool extent "
            f"[{x0:.1f}, {x1:.1f}] ft — the fitted parabola reaches the centerline "
            "only outside the pool")
        return result

    result["fit"].update({
        "crossing_x_px": float(entry_px[0]),
        "crossing_y_px": float(entry_px[1]),
        "crossing_side": "down_pool",
        "n_crossings": len(roots),
    })

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
    result["entry_frame"] = None  # the entry is an extrapolated crossing, not a real frame
    if entry.get("flags", {}).get("outside_extent"):
        result["warnings"].append("entry maps OUTSIDE the plane's extent")
    if entry_px[0] < 0 or entry_px[0] > w:
        result["warnings"].append("centerline crossing extrapolated outside the frame")
    return result
