"""Ray and plane primitives — the shared geometry under Phases 5 and 6.

Lives in its own module so :mod:`dog_diving.pool.surface` (Phase 5) and
:mod:`dog_diving.pool.intersection` (Phase 6) can both use it without importing each other.

The one idea in here
--------------------
A pixel is a **ray**, and a ray is only a point once you intersect it with a surface.  Every
phase of this pipeline is an instance of that:

* the marked plane (Phase 4) — ``z = 0``;
* the water plane (Phase 5) — ``z = z_water``;
* a **vertical wall** (Phase 5's rim/coping/coping anchors) — ``y = 0`` or ``x = 0``, the two
  cases :func:`intersect_ray_vertical_plane` handles;
* the dog's actual height (Phase 6's parallax correction) — ``z = water + height``.

Written once here because the alternative is three subtly different copies of the same
intersection, which is exactly how a sign error survives a test suite.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .homography import PlanePose

# Below this the ray is parallel to the plane and the intersection does not exist.
PARALLEL_TOL = 1e-9


@dataclass
class Ray:
    """A world-space ray: the camera centre plus a unit direction."""

    origin: np.ndarray            # (3,) world, feet
    direction: np.ndarray         # (3,) world, unit
    pixel: tuple[float, float]    # the undistorted pixel it came from

    def at(self, s: float) -> np.ndarray:
        return self.origin + float(s) * self.direction


def ray_from_pixel(pose: PlanePose, pixel: Sequence[float]) -> Ray:
    """Undistorted pixel -> world ray, from the Phase 4 pose.

    ``d_cam = K^-1 (u, v, 1)`` back-projects the pixel; ``R^T`` carries the direction into the
    world frame and ``C = -R^T t`` is the camera centre.  No undistortion happens here — a
    pixel reaching this module is already in the undistorted space ``K`` describes.
    """
    K = np.asarray(pose.K, dtype=float)
    R = np.asarray(pose.R, dtype=float)
    u, v = float(pixel[0]), float(pixel[1])
    d_cam = np.linalg.solve(K, np.array([u, v, 1.0]))
    d_world = R.T @ d_cam
    n = float(np.linalg.norm(d_world))
    if n < 1e-15:
        raise ValueError(f"degenerate pixel {pixel!r}: the back-projected ray has no direction")
    return Ray(origin=np.asarray(pose.camera_centre, dtype=float).copy(),
               direction=d_world / n, pixel=(u, v))


def intersect_ray_plane(ray: Ray, normal: Sequence[float], d: float) -> dict[str, Any]:
    """``ray`` against the plane ``n . x = d``.

    Returns a dict rather than ``None``: *which* of the three outcomes happened is information
    the caller has to act on, and ``None`` throws it away.

    * ``"ok"`` — a real intersection in front of the camera;
    * ``"parallel"`` — the ray never meets the plane (an (X, Y) here would be infinite);
    * ``"behind"`` — the intersection lies behind the camera, so the pose or the plane offset
      has the wrong sign or scale, not the point.
    """
    n = np.asarray(normal, dtype=float).ravel()
    n = n / max(float(np.linalg.norm(n)), 1e-15)
    denom = float(n @ ray.direction)
    if abs(denom) < PARALLEL_TOL:
        return {"status": "parallel", "point": None, "s": None,
                "detail": f"the ray is parallel to the plane (n.d = {denom:.2e})"}
    s = (float(d) - float(n @ ray.origin)) / denom
    if s <= 0.0:
        return {"status": "behind", "point": None, "s": float(s),
                "detail": f"the intersection is {abs(s):.3f} ft behind the camera — the pose or "
                          f"the plane offset has the wrong sign or scale"}
    return {"status": "ok", "point": ray.at(s), "s": float(s), "detail": ""}


def world_from_pixel(pose: PlanePose, pixel: Sequence[float], z_ft: float) -> dict[str, Any]:
    """Pixel -> the 3-D world point on the horizontal plane at height ``z_ft``."""
    ray = ray_from_pixel(pose, pixel)
    hit = intersect_ray_plane(ray, (0.0, 0.0, 1.0), float(z_ft))
    return {"ray": ray, **hit}


def intersect_ray_vertical_plane(
    ray: Ray, axis: str, value: float
) -> dict[str, Any]:
    """``ray`` against a **vertical** plane ``y = value`` or ``x = value``.

    This is the workhorse of Phase 5, and the reason it is a separate function from
    :func:`intersect_ray_plane` is that a vertical plane is what a pool wall *is*.  The
    distinction matters more than it looks: a click on the water where it meets a wall is
    **one observable and one unknown** (its height), because its horizontal position is pinned
    by the wall — whereas the same click against the water plane is one observable and *two*
    unknowns and is therefore unsolvable on its own.

    Returns the same dict contract as :func:`intersect_ray_plane`, with ``point`` in world
    coordinates (so ``point[2]`` is the height on the wall).
    """
    ax = str(axis).lower()
    if ax in ("y", "y_ft", "across"):
        n = np.array([0.0, 1.0, 0.0])
    elif ax in ("x", "x_ft", "along"):
        n = np.array([1.0, 0.0, 0.0])
    else:
        raise ValueError(f"axis must be 'x' or 'y', got {axis!r}")
    return intersect_ray_plane(ray, n, float(value))


def height_on_wall(pose: PlanePose, pixel: Sequence[float], axis: str,
                   value: float) -> dict[str, Any]:
    """Convenience: pixel -> the point on a known vertical wall, plus its height."""
    ray = ray_from_pixel(pose, pixel)
    hit = intersect_ray_vertical_plane(ray, axis, value)
    out = {"ray": ray, **hit}
    out["z_ft"] = None if hit["point"] is None else float(np.asarray(hit["point"])[2])
    return out


def project_point(pose: PlanePose, point: Sequence[float]) -> np.ndarray:
    """World point -> undistorted pixel, via the pose's own ``K, R, t``.

    Not a residual on its own, but the piece a caller needs to compute one against the click it
    started from.  Living here means the reprojection is done with the **same** ``K, R, t`` the
    intersection used, so a check can never drift from the thing it is checking.
    """
    K = np.asarray(pose.K, dtype=float)
    R = np.asarray(pose.R, dtype=float)
    p = np.asarray(point, dtype=float)[:3]
    p_cam = R @ p + np.asarray(pose.t, dtype=float)
    if p_cam[2] <= 1e-12:
        return np.array([np.nan, np.nan])
    return (K @ p_cam)[:2] / p_cam[2]


def reprojection_error_px(pose: PlanePose, point: Sequence[float],
                          pixel: Sequence[float]) -> float:
    """How far a reconstructed world point lands from the pixel it came from (px)."""
    uv = project_point(pose, point)
    px = np.asarray(pixel, dtype=float).ravel()[:2]
    if not np.all(np.isfinite(uv)):
        return float("inf")
    return float(np.linalg.norm(uv - px))
