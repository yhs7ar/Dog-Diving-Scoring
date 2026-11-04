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
* the dog's actual height (Phase 6's parallax correction) — ``z = water + height``;
* the dog's **trajectory plane** (:class:`TrajectoryPlane`, the 3-D route) — a vertical plane
  fitted to the flight run, which turns every tail-set pixel into a 3-D point without needing
  a height at all.

Written once here because the alternative is three subtly different copies of the same
intersection, which is exactly how a sign error survives a test suite.

The two phases this module owns for the 3-D route
-------------------------------------------------
:func:`vertical_plane_from_points` fits the plane (Phase 1's primitive) and
:func:`intersect_curve_with_elevation` finds where a parametric 3-D curve crosses a fixed
height — the geometric answer to *"which point of the flight is the water entry?"*, with no
image-space row anywhere in it.
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


# ============================================================================
# Phase 1 — the vertical trajectory plane, and a curve crossing a height
# ============================================================================
#: A normal this close to horizontal means the plane is vertical.
VERTICAL_TOL = 1e-6
#: Below this horizontal extent (ft) a flight run has no direction to fit.
MIN_PLANE_SPAN_FT = 0.5
#: The plane's normal is fitted in the horizontal plane; below this |n_xy| it is degenerate.
MIN_NORMAL_XY = 1e-9


@dataclass
class TrajectoryPlane:
    """The vertical 3-D plane the dog's flight run is taken to lie in.
    A *vertical* plane, deliberately: the flight is ballistic in the world frame, so it has no
    reason to leave a plane that contains the vertical axis, and pinning that plane is what turns
    a single-camera pixel into one 3-D point — the ray meets the plane exactly once.  A general
    (tilted) plane would fit the same clicks but is not a physical statement about a jump.

    The plane is carried as ``normal . x = d`` in world feet, with ``normal`` horizontal, so the
    whole family of intersections reduces to :func:`intersect_ray_plane` — one implementation,
    one sign convention."""

    origin: np.ndarray            # (3,) a point on the plane, world ft
    normal: np.ndarray            # (3,) unit, horizontal (n[2] == 0)
    direction: np.ndarray         # (3,) unit, horizontal, inside the plane
    rms_ft: float = 0.0           # horizontal TLS residual of the fit points
    n_points: int = 0
    source: str = ""

    @property
    def d(self) -> float:
        return float(self.normal @ self.origin)

    @property
    def is_vertical(self) -> bool:
        return abs(float(self.normal[2])) <= VERTICAL_TOL

    def signed_distance(self, point: Sequence[float]) -> float:
        p = np.asarray(point, dtype=float).ravel()[:3]
        return float(self.normal @ p) - self.d

    def project(self, point: Sequence[float]) -> np.ndarray:
        """The closest point of the plane to ``point`` (the plumb drop onto the plane)."""
        p = np.asarray(point, dtype=float).ravel()[:3]
        return p - self.signed_distance(p) * self.normal

    def at(self, s: float) -> np.ndarray:
        """The point ``origin + s * direction`` — ``s`` in ft along the flight direction."""
        return self.origin + float(s) * self.direction

    def coordinate(self, point: Sequence[float]) -> float:
        """``s`` of the projection of ``point`` — the along-flight coordinate, in ft."""
        return float((np.asarray(point, dtype=float).ravel()[:3] - self.origin) @ self.direction)

    def intersect(self, ray: Ray) -> dict[str, Any]:
        """``ray`` against this plane, with the same dict contract as :func:`intersect_ray_plane`."""
        return intersect_ray_plane(ray, self.normal, self.d)

    def as_dict(self) -> dict[str, Any]:
        return {"origin_ft": [float(v) for v in self.origin],
                "normal": [float(v) for v in self.normal],
                "direction": [float(v) for v in self.direction],
                "d": float(self.d), "rms_ft": float(self.rms_ft),
                "n_points": int(self.n_points), "source": self.source,
                "is_vertical": bool(self.is_vertical)}



@dataclass
class CurvedTrajectoryPlane(TrajectoryPlane):
    """A vertical plane adapted to fit the trajectory curvature."""

    def project_curve_surface(self, pose: PlanePose, coeffs: dict, frames: Sequence[float]) -> list[tuple[float, float]]:
        # TODO: Implement curved surface projection
        return []


    @property
    def d(self) -> float:
        return float(self.normal @ self.origin)

    @property
    def is_vertical(self) -> bool:
        return abs(float(self.normal[2])) <= VERTICAL_TOL

    def signed_distance(self, point: Sequence[float]) -> float:
        p = np.asarray(point, dtype=float).ravel()[:3]
        return float(self.normal @ p) - self.d

    def project(self, point: Sequence[float]) -> np.ndarray:
        """The closest point of the plane to ``point`` (the plumb drop onto the plane)."""
        p = np.asarray(point, dtype=float).ravel()[:3]
        return p - self.signed_distance(p) * self.normal

    def at(self, s: float) -> np.ndarray:
        """The point ``origin + s * direction`` — ``s`` in ft along the flight direction."""
        return self.origin + float(s) * self.direction

    def coordinate(self, point: Sequence[float]) -> float:
        """``s`` of the projection of ``point`` — the along-flight coordinate, in ft."""
        return float((np.asarray(point, dtype=float).ravel()[:3] - self.origin) @ self.direction)

    def intersect(self, ray: Ray) -> dict[str, Any]:
        """``ray`` against this plane, with the same dict contract as :func:`intersect_ray_plane`."""
        return intersect_ray_plane(ray, self.normal, self.d)

    def as_dict(self) -> dict[str, Any]:
        return {"origin_ft": [float(v) for v in self.origin],
                "normal": [float(v) for v in self.normal],
                "direction": [float(v) for v in self.direction],
                "d": float(self.d), "rms_ft": float(self.rms_ft),
                "n_points": int(self.n_points), "source": self.source,
                "is_vertical": bool(self.is_vertical)}



def _xy_of(point: Any) -> np.ndarray | None:
    """``(X, Y)`` **in feet** from a dict carrying ``X_ft``/``Y_ft``, or from a pair.

    Deliberately does not fall back to a bare ``x``/``y``: a flat detection row uses those keys for
    *pixels*, and a plane fitted through pixel columns and then used as a world plane produces
    numbers that look plausible and are meaningless.  An unrecognised dict returns ``None`` and the
    caller reports ``few_points``.
    """
    if isinstance(point, dict):
        for kx, ky in (("X_ft", "Y_ft"), ("x_ft", "y_ft")):
            if point.get(kx) is not None and point.get(ky) is not None:
                return np.array([float(point[kx]), float(point[ky])], dtype=float)
        return None
    p = np.asarray(point, dtype=float).ravel()
    if p.size < 2 or not np.all(np.isfinite(p[:2])):
        return None
    return np.array([float(p[0]), float(p[1])], dtype=float)


def vertical_plane_from_points(
    points: Sequence[Any], *, min_points: int = 3, source: str = ""
) -> dict[str, Any]:
    """Fit a **vertical** plane to the horizontal positions of a flight run (Phase 1).

    The fit is a total-least-squares line through ``(X, Y)`` (SVD of the centred point set, so it
    is symmetric in the two axes and does not privilege ``Y = f(X)``); the plane is that line
    extruded along world ``+Z``, so its normal ``(-dy, dx, 0)`` is horizontal by construction.

    Returns the module's usual dict contract, because *why* a fit failed decides what the caller
    does next — ``"few_points"`` means gather more of the track, ``"no_span"`` means the dog moved
    too little across the pool for a direction to exist (a near-vertical jump would genuinely leave
    the plane undetermined), ``"degenerate"`` means the numbers are non-finite.

    ``rms_ft`` is the point-to-line residual in the horizontal plane: as long as the tail-set track
    wanders no further than a few inches off a straight line, the plane assumption is doing no
    damage, and this number is the evidence for that rather than a claim.
    """
    xy: list[np.ndarray] = []
    for p in points or ():
        q = _xy_of(p)
        if q is not None and np.all(np.isfinite(q)):
            xy.append(q)
    pts = np.asarray(xy, dtype=float) if xy else np.zeros((0, 2))
    n = int(pts.shape[0])
    fail = {"status": None, "plane": None, "rms_ft": None, "n_points": n, "span_ft": 0.0}
    if n < int(min_points):
        return {**fail, "status": "few_points",
                "detail": f"{n} usable (X, Y) point(s); a vertical plane needs at least "
                          f"{int(min_points)}"}
    centre = pts.mean(axis=0)
    centred = pts - centre
    try:
        _u, _sv, vt = np.linalg.svd(centred, full_matrices=False)
    except np.linalg.LinAlgError as exc:                              # noqa: BLE001
        return {**fail, "status": "degenerate", "detail": f"the point set is not fittable: {exc}"}
    direction_xy = np.asarray(vt[0], dtype=float)
    norm = float(np.linalg.norm(direction_xy))
    if not np.all(np.isfinite(direction_xy)) or norm < MIN_NORMAL_XY:
        return {**fail, "status": "degenerate", "detail": "the fitted direction is not finite"}
    direction_xy = direction_xy / norm
    along = centred @ direction_xy
    span_ft = float(np.ptp(along)) if n else 0.0
    if span_ft < float(MIN_PLANE_SPAN_FT):
        return {**fail, "status": "no_span", "span_ft": span_ft,
                "detail": f"the flight run spans {span_ft:.3f} ft across the pool (minimum "
                          f"{MIN_PLANE_SPAN_FT:g} ft), so no flight direction exists to fit"}
    residuals = np.abs(centred @ np.array([-direction_xy[1], direction_xy[0]]))
    rms = float(np.sqrt(np.mean(residuals ** 2))) if n else 0.0
    normal = np.array([-direction_xy[1], direction_xy[0], 0.0], dtype=float)
    # Sign fixed by the numbers, not by the point order: the same track must fit the same plane
    # whether it is fed forwards or backwards, or a reverse-ordered run would report a different
    # `d` for identical geometry.
    if normal[0] < 0.0 or (normal[0] == 0.0 and normal[1] < 0.0):
        normal = -normal
    plane = TrajectoryPlane(
        origin=np.array([centre[0], centre[1], 0.0], dtype=float),
        normal=normal, direction=np.array([direction_xy[0], direction_xy[1], 0.0], dtype=float),
        rms_ft=rms, n_points=n, source=str(source))
    return {"status": "ok", "plane": plane, "rms_ft": rms, "n_points": n,
            "span_ft": span_ft,
            "s_range_ft": [float(along.min()), float(along.max())],
            "detail": f"vertical plane through {n} point(s), horizontal RMS {rms:.3f} ft, "
                      f"span {span_ft:.2f} ft"}


def curve_point(coeffs: dict, t: float, t_ref: float | None = None) -> np.ndarray:
    """The 3-D point of a parametric quadratic curve at time ``t`` (world ft).

    ``coeffs`` holds one ``[a2, a1, a0]`` triple per axis, in **centred** time ``tc = t - t_ref``
    — the same convention :func:`dog_diving.trajectory.robust_parabola` returns, so the curve
    drawn and the curve solved are the same curve.
    """
    ref = float(coeffs["t_ref"] if t_ref is None else t_ref)
    tc = float(t) - ref
    return np.array([float(np.polyval(coeffs[ax], tc)) for ax in ("x", "y", "z")], dtype=float)


def intersect_curve_with_elevation(
    coeffs: dict, z_ft: float, *, t_ref: float | None = None,
    t_min: float | None = None,
) -> dict[str, Any]:
    """Where a parametric 3-D curve crosses ``z = z_ft`` — Phase 1's second primitive.

    This is the whole of the 3-D route's contact arithmetic: the flight is a curve in world space,
    the water is a fixed height, and the entry is where the curve meets it.  Nothing here knows
    about image rows or frame edges, which is the point of the rewrite — a waterline row is a
    property of one camera's picture, while ``Z = z_water`` is a property of the pool.

    ``t_min`` restricts the answer to crossings at or after that time (pass the apex to select the
    **descending** branch; a ballistic curve crosses a given height twice).  The smallest eligible
    root wins, and ``t_all`` carries every real root so a caller can see the one that was rejected.

    Returns ``{"status", "t", "point", "t_all", "detail"}`` with status ``"ok"``,
    ``"no_crossing"`` (the curve never reaches that height — the fit was extrapolated too far or
    the gate is wrong) or ``"no_third_axis"`` (``coeffs`` has no ``z`` — a 2-D fit was passed).
    """
    empty = {"status": None, "t": None, "point": None, "t_all": [], "detail": ""}
    if not isinstance(coeffs, dict) or coeffs.get("z") is None:
        return {**empty, "status": "no_third_axis",
                "detail": "the fit carries no world Z axis, so it has no height to cross"}
    ref = float(coeffs["t_ref"] if t_ref is None else t_ref)
    a2, a1, a0 = (float(v) for v in coeffs["z"])
    try:
        roots = np.roots([a2, a1, a0 - float(z_ft)])
    except (np.linalg.LinAlgError, ValueError) as exc:                 # noqa: BLE001
        return {**empty, "status": "degenerate", "detail": f"the Z quadratic is not solvable: {exc}"}
    frames = sorted(float(r.real) + ref for r in roots if abs(float(r.imag)) < 1e-6)
    eligible = [f for f in frames if t_min is None or f >= float(t_min)]
    if not eligible:
        return {**empty, "status": "no_crossing", "t_all": frames,
                "detail": (f"the curve never reaches z = {float(z_ft):.3f} ft"
                           + (f" at or after frame {float(t_min):.1f}" if t_min is not None else "")
                           + f" (roots: {[round(f, 2) for f in frames] or 'none'})")}
    t = eligible[0]
    return {"status": "ok", "t": float(t), "point": curve_point(coeffs, t, ref),
            "t_all": frames,
            "detail": f"z = {float(z_ft):.3f} ft crossed at frame {t:.2f} on the descending branch"}


def project_curve(pose: PlanePose, coeffs: dict, frames: Sequence[float],
                  *, t_ref: float | None = None) -> list[tuple[float, float]]:
    """A 3-D curve -> the undistorted pixels it images to, for drawing.

    The world arc and the image arc are two different curves, and the canvas can only speak in
    pixels.  Projecting the *solved* curve (rather than refitting a parabola through the projected
    points) is the honest translation: what the operator sees is the same geometry the distance came
    from, including where it was extrapolated past the last detection.

    Points that do not image in front of the camera are dropped rather than drawn at infinity.
    """
    out: list[tuple[float, float]] = []
    for t in frames:
        uv = project_point(pose, curve_point(coeffs, float(t), t_ref))
        if np.all(np.isfinite(uv)):
            out.append((float(uv[0]), float(uv[1])))
    return out

