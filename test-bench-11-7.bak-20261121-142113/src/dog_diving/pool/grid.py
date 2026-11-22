"""The metric grid — a 2-D foot lattice on the pool surface, drawn in the image.

This is the deliverable the plan's Phase 4 names directly: *"compute the 2D homography
transformation matrix to build a uniform metric grid extending outward from the dock"*.  It is
also the visible half of gate W2 in :mod:`dog_diving.pool.calib`: a grid whose ticks do not land on
the pool's own painted marks is a calibration that is wrong, and a reviewer can see that in one
look where a residual table cannot show it.

Two things this module exists to get right
------------------------------------------
1. **The grid is a 2-D lattice, not a row of ticks.**  The previous 1-D route drew vertical lines
   at increasing image x.  That cannot express the transverse direction at all, so it could never
   show a calibration whose across-pool scale was wrong — which, with a width prior in play, is
   the most likely way for this pipeline to be wrong.
2. **Drawing happens through the lens.**  Everything in :mod:`dog_diving.pool` lives in
   undistorted pixels, but the frame being drawn on is a raw one.  Compositing undistorted
   geometry straight onto a raw frame puts the grid a few pixels off the marks it is being checked
   against, and the error is largest at the frame edges — exactly where the check matters.  So the
   grid is projected with the pose and *then* pushed back through the distortion model.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import cv2
import numpy as np

from .homography import PlanePose, apply_homography
from .lens import LensModel, distort_points
from .surface import WaterPlane

# BGR, matching the existing overlay palette so the two overlays do not look like different tools.
C_GRID = (0, 200, 255)
C_GRID_MINOR = (0, 120, 160)
C_MARK = (0, 0, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def grid_segments(
    *,
    along_ft: float = 31.0,
    width_ft: float = 16.0,
    y_min_ft: float = 0.0,
    step_ft: float = 1.0,
    major_every_ft: float = 5.0,
) -> list[dict[str, Any]]:
    """A lattice of foot-spaced line segments in world coordinates on the pool surface.

    Both families are returned because both are checks: the along-pool lines test the transverse
    (width) calibration and the across-pool lines test the longitudinal one.  A grid with only one
    family would silently pass a calibration that is wrong in the other axis.

    ``y_min_ft`` lets the lattice be placed anywhere across the pool rather than always starting
    at the near edge — a pool whose marks are on the far wall needs exactly that.
    """
    step = float(step_ft)
    if step <= 0:
        raise ValueError("step_ft must be positive")
    y_top = float(y_min_ft) + float(width_ft)
    segs: list[dict[str, Any]] = []

    x = 0.0
    while x <= float(along_ft) + 1e-9:
        segs.append({"family": "across", "value_ft": x,
                     "p0": (x, float(y_min_ft)), "p1": (x, y_top),
                     "major": abs(x % float(major_every_ft)) < 1e-9})
        x += step

    y = float(y_min_ft)
    while y <= y_top + 1e-9:
        segs.append({"family": "along", "value_ft": y,
                     "p0": (0.0, y), "p1": (float(along_ft), y),
                     "major": abs((y - float(y_min_ft)) % float(major_every_ft)) < 1e-9})
        y += step
    return segs


def visible_polyline(
    pts: np.ndarray, width: int, height: int, *, margin: float = 20.0
) -> np.ndarray:
    """Clip a pixel-space polyline to the frame, keeping the parts that are on-screen.

    Necessary rather than cosmetic: :func:`apply_homography` returns ``nan`` for points behind the
    camera and enormous coordinates for points near the horizon, and both would destroy the drawing
    (``cv2.polylines`` with one bad vertex draws nothing, or fills the frame).  Points well outside
    the frame are dropped and the polyline is split there, so a grid line that leaves the image and
    comes back is drawn in two pieces instead of being connected across the gap.
    """
    arr = np.asarray(pts, dtype=float).reshape(-1, 2)
    lo = -float(margin)
    hi_x = float(width) + float(margin)
    hi_y = float(height) + float(margin)
    inside = (np.isfinite(arr).all(axis=1)
              & (arr[:, 0] >= lo) & (arr[:, 0] <= hi_x)
              & (arr[:, 1] >= lo) & (arr[:, 1] <= hi_y))
    return arr[inside]


def grid_pixels(
    pose: PlanePose,
    water: WaterPlane,
    segments: Sequence[dict],
    *,
    samples: int = 24,
) -> list[dict[str, Any]]:
    """World lattice -> undistorted-pixel polylines, with each segment's label anchor.

    Each world segment is sampled along its length before projection.  Sampling matters for the
    *across* family (constant X, varying Y): its image is a straight line under a homography, but
    the *clipped* drawing of it is not, and sampling also keeps the label anchor at a sane point.
    """
    out: list[dict[str, Any]] = []
    for seg in segments:
        p0 = np.asarray(seg["p0"], dtype=float)
        p1 = np.asarray(seg["p1"], dtype=float)
        ts = np.linspace(0.0, 1.0, max(2, int(samples)))
        world = np.array([p0 + t * (p1 - p0) for t in ts])
        px = apply_homography(water.homography(pose), world)
        out.append({**seg, "pixels": px, "label_at": px[len(px) // 2]})
    return out


def draw_grid(
    frame: np.ndarray,
    pose: PlanePose,
    water: WaterPlane,
    *,
    lens: LensModel | None = None,
    along_ft: float = 31.0,
    width_ft: float = 16.0,
    y_min_ft: float = 0.0,
    step_ft: float = 1.0,
    major_every_ft: float = 5.0,
    labels: bool = True,
    thickness: int = 1,
    copy: bool = True,
) -> np.ndarray:
    """Draw the 2-D metric lattice onto a frame and return it.

    ``lens`` is applied last, per the module docstring: the lattice is built in the undistorted
    space the homography lives in and then pushed back through the distortion model so it lands on
    the marks it is meant to be checked against.

    ``copy=True`` (the default) leaves the input untouched, matching
    :func:`dog_diving.annotate.draw_frame`'s "pure: frame is copied, never drawn onto in place"
    contract.  Callers inside a tight video loop who already own a scratch frame may pass
    ``copy=False`` for the allocation saving, but the default has to be the safe one: an overlay
    that silently overwrites its input makes the caller's next comparison against the original
    frame impossible, which is precisely the check a reviewer does by eye.
    """
    out = frame.copy() if copy else frame
    h, w = out.shape[:2]
    segs = grid_segments(along_ft=along_ft, width_ft=width_ft, y_min_ft=y_min_ft,
                         step_ft=step_ft, major_every_ft=major_every_ft)
    for item in grid_pixels(pose, water, segs):
        px = item["pixels"]
        if lens is not None and lens.is_distorted:
            px = distort_points(px, lens)
        vis = visible_polyline(px, w, h)
        if vis.shape[0] < 2:
            continue
        colour = C_GRID if item["major"] else C_GRID_MINOR
        pts = np.round(vis).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(out, [pts], False, colour,
                      thickness + 1 if item["major"] else thickness, cv2.LINE_AA)
        if labels and item["major"]:
            anchor = item["label_at"]
            anchor = anchor if np.all(np.isfinite(anchor)) else vis[len(vis) // 2]
            if lens is not None and lens.is_distorted:
                anchor = distort_points(np.asarray(anchor, dtype=float).reshape(1, 2), lens)[0]
            ax, ay = int(round(float(anchor[0]))), int(round(float(anchor[1])))
            if 0 <= ax < w - 40 and 0 <= ay < h - 8:
                text = (f"{item['value_ft']:.0f}" if item["family"] == "across"
                        else f"y{item['value_ft']:.0f}")
                cv2.putText(out, text, (ax + 4, ay - 4), FONT, 0.55, colour, 2, cv2.LINE_AA)
    return out


def grid_residual_against_marks(
    pose: PlanePose,
    water: WaterPlane,
    marks: Sequence[dict],
) -> dict[str, Any]:
    """Compare the projected grid against the pool's own painted marks, in feet.

    Gate W2's honest companion: the homography's residual only reports how well the *fit* matched
    its own inputs, and a fit can match its inputs and still be wrong elsewhere (that is the whole
    reason :func:`dog_diving.pool.homography.assert_not_collinear` exists).  Reprojecting the mark
    positions through the finished document and comparing against where the marks actually are is
    the check the reviewer performs by eye, done numerically.

    A mark sitting below the marked plane cannot be pushed onto it, so the *water plane* is used for
    the comparison when a mark carries a ``z_ft``; otherwise the marked plane.
    """
    rows: list[dict[str, Any]] = []
    for m in marks:
        if m.get("x_px") is None or m.get("y_px") is None or m.get("ft") is None:
            continue
        z = float(m.get("z_ft", 0.0))
        H_plane = pose.plane_homography(z) if z != float(water.z_ft) else water.homography(pose)
        pred = apply_homography(H_plane, np.array([[float(m["ft"]), float(m.get("y_ft", 0.0))]]))
        obs = np.array([[float(m["x_px"]), float(m["y_px"])]], dtype=float)
        if not np.all(np.isfinite(pred[0])):
            continue
        rows.append({
            "ft": float(m["ft"]),
            "label": m.get("label", ""),
            "err_px": float(np.linalg.norm(pred[0] - obs[0])),
            "pred_px": [float(pred[0][0]), float(pred[0][1])],
            "obs_px": [float(obs[0][0]), float(obs[0][1])],
        })
    errs = [r["err_px"] for r in rows]
    return {
        "n": len(rows),
        "rows": rows,
        "max_px": float(max(errs)) if errs else None,
        "rms_px": float(np.sqrt(np.mean(np.square(errs)))) if errs else None,
    }
