"""Point-cloud dock-lip auto-location (1b-auto follow-on).

Where this sits
---------------
Step 14 (PC-DIST) scores the jump against a dock-lip datum that the *operator* supplies as a
``datum_line(near_corner, far_corner)`` in world feet.  Step 6 (1b-auto) closed the *water* half of
that gap — :func:`dog_diving.pc_trajectory.fit_water_plane` derives ``water_z_ft`` from the cloud.
This module closes the **dock** half: it locates the lip *in the point cloud itself*, so the datum
is auto-placed and then (per the operator's standing order) available for manual adjustment, rather
than drawn from scratch.

The measurement is unchanged.  The domain rule (``dock-diving-scoring`` skill) says the datum is
"the vertical projection of the lip onto the water plane", and a perpendicular distance does not
depend on the line's extent — so this module only has to find *the lip line and where the dock
ends*, not the dock's far corners.

How the lip is found (no learned model, pure geometry)
------------------------------------------------------
1. Fit the water plane (:func:`~dog_diving.pc_trajectory.fit_water_plane`) or accept a supplied
   ``water_z_ft`` — the reference height everything else is measured against.
2. Points more than ``freeboard_min_ft`` above water are "not water" (dock top, dog, railings).
   Fit the dominant **horizontal** plane to that subset -> the dock-top plane -> ``dock_z_ft``.
   RANSAC prunes non-horizontal samples, so a dog (a small blob) and vertical railings are
   outliers and the dock top wins.
3. The **pool direction** is the 2-D ``(X, Y)`` unit vector from the dock-top centroid to the
   water centroid.  It is measured, never assumed — the ``zed_to_world`` axis convention is a
   documented placeholder, so "which side is the pool" is read from the data.
4. The dock-top points are convex-hulled; the hull edge whose outward normal best aligns with the
   pool direction *is* the lip.  The dock-top points within ``lip_band_ft`` of that edge refine the
   line (least squares), and the span of those points along the line gives the two corners.

What is still a placeholder
---------------------------
``zed_to_world`` (and therefore the *metric* position of every corner) is the canonical
single-camera prototype map; the true floor-aligned camera->pool rigid transform is
hardware-verify-pending and plugs into the ``to_world`` seam exactly as it does for the trajectory
cluster.  The near/far ordering below is deterministic in the canonical frame (smaller world ``X``
= closer to the camera, tie-broken by smaller ``Y``); like the dock-edge keypoints, swapping the
two flips the *sign* of the reported signed distance but never the measured distance.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Sequence

import numpy as np

from . import pc_distance as _pcd
from . import pc_trajectory as _pc

#: Points this far (ft) above the fitted water plane count as "not water" (dock / dog / railing).
FREEBOARD_MIN_FT = 0.5
#: A fitted horizontal plane must sit at least this far (ft) above water to be "a dock".
DOCK_ABOVE_WATER_FT = 0.8
#: Points within this band (ft) of the water plane count as "water" for the pool-direction centroid.
WATER_BAND_FT = 0.3
#: Dock-top points within this perpendicular distance (ft) of the lip edge refine the lip line.
LIP_BAND_FT = 0.5
#: RANSAC horizontal-plane fit defaults, in world feet.
DOCK_DIST_THRESH_FT = 0.15
MIN_DOCK_POINTS = 16
MIN_LIP_POINTS = 4
MAX_ITER = 200
ANGLE_TOL_DEG = 15.0

#: Same ``to_world`` seam the trajectory cluster takes (world feet, ``X`` along pool, ``Z`` up).
ToWorld = _pc.ToWorld


def _finite_world(cloud: Any, to_world: ToWorld) -> np.ndarray:
    """Flatten a cloud (``(N,3)`` or image-shaped ``(H,W,3)`` metres) to finite ``(M,3)`` world ft."""
    arr = np.asarray(cloud, dtype=float)
    if arr.size == 0:
        return np.empty((0, 3), dtype=float)
    if arr.ndim == 3:
        arr = arr.reshape(-1, arr.shape[2])
    arr = arr[:, :3]
    arr = arr[np.isfinite(arr).all(axis=1)]
    if arr.shape[0] == 0:
        return np.empty((0, 3), dtype=float)
    world = np.stack([np.asarray(to_world(p), dtype=float).ravel()[:3] for p in arr])
    return world[np.isfinite(world).all(axis=1)]


def _plane_normal(p3: np.ndarray) -> np.ndarray | None:
    p = np.asarray(p3, dtype=float)
    n = np.cross(p[1] - p[0], p[2] - p[0])
    norm = float(np.linalg.norm(n))
    if not np.isfinite(norm) or norm < 1e-12:
        return None
    return n / norm


def _refine_plane(pts: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Least-squares plane (SVD) on ``pts``; returns ``(unit normal, offset, rms)`` in world ft."""
    pts = np.asarray(pts, dtype=float)
    c = pts.mean(axis=0)
    _u, _s, vt = np.linalg.svd(pts - c, full_matrices=False)
    n = np.asarray(vt[-1], dtype=float)
    norm = float(np.linalg.norm(n))
    n = n / norm if (np.isfinite(norm) and norm > 1e-12) else np.array([0.0, 1.0, 0.0])
    d = float(n @ c)
    rms = float(np.sqrt(np.mean(((pts - c) @ n) ** 2)))
    return n, d, rms


def _fit_horizontal_plane_world(
    pts: np.ndarray,
    *,
    dist_thresh_ft: float = DOCK_DIST_THRESH_FT,
    min_inliers: int = MIN_DOCK_POINTS,
    max_iter: int = MAX_ITER,
    angle_tol_deg: float = ANGLE_TOL_DEG,
    seed: int = 0,
) -> dict[str, Any]:
    """RANSAC a dominant **horizontal** plane in world feet, returning its inlier mask.

    Unlike :func:`dog_diving.pc_trajectory.fit_water_plane` (which fits *any* plane and checks
    horizontality afterwards), this fits *only* horizontal candidates: non-horizontal samples are
    pruned inside RANSAC so a vertical wall or the dog cannot win the inlier count.
    """
    out: dict[str, Any] = {"status": None, "z_ft": None, "normal": None, "inlier_mask": None,
                           "n_inliers": 0, "rms_ft": None, "detail": ""}
    pts = np.asarray(pts, dtype=float)
    if pts.shape[0] < 3:
        out.update(status="no_points", detail=f"{pts.shape[0]} point(s) — a plane needs at least 3")
        return out
    cos_tol = math.cos(math.radians(float(angle_tol_deg)))
    rng = np.random.default_rng(int(seed))
    best: tuple[int, np.ndarray, float, np.ndarray] | None = None
    for _ in range(int(max_iter)):
        idx = rng.choice(pts.shape[0], 3, replace=False)
        n = _plane_normal(pts[idx])
        if n is None or abs(float(n[2])) < cos_tol:
            continue  # degenerate or not horizontal — not a dock-top candidate
        d = float(n @ pts[idx[0]])
        dist = np.abs(pts @ n - d)
        mask = dist < float(dist_thresh_ft)
        cnt = int(mask.sum())
        if best is None or cnt > best[0]:
            best = (cnt, n, d, mask)
    if best is None:
        out.update(status="no_horizontal",
                   detail="no non-collinear horizontal sample — no dock-top surface in this cloud")
        return out
    cnt, n, d, mask = best
    if cnt < int(min_inliers):
        out.update(status="few_inliers", n_inliers=cnt,
                   detail=f"the dominant horizontal plane has {cnt} inlier(s) (need {int(min_inliers)})")
        return out
    n, d, rms = _refine_plane(pts[mask])
    if float(n[2]) < 0.0:
        n, d = -n, -d
    out.update(status="ok", normal=n.tolist(), n_inliers=int(mask.sum()), rms_ft=float(rms),
               inlier_mask=mask)
    # The plane's height is read at the centroid of its inliers (more robust than n @ 0).
    out["z_ft"] = float(pts[mask][:, 2].mean())
    out["detail"] = (f"horizontal plane at world Z = {out['z_ft']:+.2f} ft "
                     f"({out['n_inliers']} inliers, RMS {out['rms_ft']:.3f} ft)")
    return out


def _convex_hull_2d(points: np.ndarray) -> np.ndarray:
    """Monotone-chain convex hull of 2-D points, returned CCW (``(M,2)``, no repeated vertex)."""
    pts = sorted(set(map(tuple, np.asarray(points, dtype=float))))
    if len(pts) <= 1:
        return np.array(pts)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[tuple[float, float]] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list[tuple[float, float]] = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1], dtype=float)


def _order_near_far(p1: np.ndarray, p2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """near = smaller world ``X`` (closer to the camera, canonical frame), tie-broken by smaller ``Y``."""
    if p1[0] < p2[0] - 1e-9:
        return p1, p2
    if p2[0] < p1[0] - 1e-9:
        return p2, p1
    return (p1, p2) if p1[1] <= p2[1] else (p2, p1)


def locate_dock_lip(
    cloud: Any,
    *,
    to_world: ToWorld = _pc.zed_to_world,
    water_z_ft: float | None = None,
    freeboard_min_ft: float = FREEBOARD_MIN_FT,
    dock_above_water_ft: float = DOCK_ABOVE_WATER_FT,
    water_band_ft: float = WATER_BAND_FT,
    lip_band_ft: float = LIP_BAND_FT,
    min_dock_points: int = MIN_DOCK_POINTS,
    min_lip_points: int = MIN_LIP_POINTS,
    angle_tol_deg: float = ANGLE_TOL_DEG,
    seed: int = 0,
) -> dict[str, Any]:
    """Locate the dock lip in a point cloud -> a ``pc_distance.datum_line`` in world feet.

    ``cloud`` is a ZED-metres cloud (``(N,3)`` or image-shaped ``(H,W,3)``); ``to_world`` maps
    metres to world feet.  ``water_z_ft`` may be supplied (skipping the water fit) or derived.

    Returns ``{status, datum, water_z_ft, water_fit, dock_z_ft, dock_fit, pool_dir, gates,
    n_points, n_above_water, n_dock, n_lip, detail}``.  ``datum`` is the dict
    :func:`dog_diving.pc_distance.datum_line` returns (ready to pass as ``lip=``) — its ``a``/``b``
    are the two lip corners in world feet with the ``near -> far`` ordering applied.
    ``status`` is ``"ok"``, ``"no_points"``, ``"no_water_plane"``, ``"no_dock"``, ``"no_pool_side"``
    or ``"degenerate"`` — a refused locate never carries a ``datum``.
    """
    res: dict[str, Any] = {
        "status": None, "datum": None, "water_z_ft": None, "water_fit": None,
        "dock_z_ft": None, "dock_fit": None, "pool_dir": None, "gates": {},
        "n_points": 0, "n_above_water": 0, "n_dock": 0, "n_lip": 0, "detail": "",
    }

    world = _finite_world(cloud, to_world)
    res["n_points"] = int(world.shape[0])
    if world.shape[0] < 3:
        res.update(status="no_points", detail="too few finite points in the cloud")
        return res

    # -- water reference -----------------------------------------------------
    if water_z_ft is None:
        wfit = _pc.fit_water_plane(cloud, to_world=to_world, angle_tol_deg=angle_tol_deg)
    else:
        wfit = {"status": "ok", "water_z_ft": float(water_z_ft), "detail": "supplied"}
    res["water_fit"] = wfit
    res["gates"]["G_water"] = {"pass": wfit["status"] == "ok", "detail": wfit.get("detail", "")}
    if wfit["status"] != "ok":
        res.update(status="no_water_plane", detail=wfit.get("detail", "no water plane"))
        return res
    res["water_z_ft"] = float(wfit["water_z_ft"])

    # -- dock-top plane ------------------------------------------------------
    above = world[world[:, 2] > res["water_z_ft"] + float(freeboard_min_ft)]
    res["n_above_water"] = int(above.shape[0])
    if above.shape[0] < int(min_dock_points):
        res["gates"]["G_dock"] = {"pass": False, "detail": "too few above-water points"}
        res.update(status="no_dock",
                   detail=f"{above.shape[0]} point(s) above water (need {int(min_dock_points)})")
        return res
    dfit = _fit_horizontal_plane_world(above, min_inliers=int(min_dock_points),
                                       angle_tol_deg=angle_tol_deg, seed=seed)
    res["dock_fit"] = {k: dfit[k] for k in ("status", "z_ft", "normal", "n_inliers", "rms_ft",
                                            "detail")}
    res["gates"]["G_dock"] = {"pass": dfit["status"] == "ok", "detail": dfit["detail"]}
    if dfit["status"] != "ok":
        res.update(status="no_dock", detail=dfit["detail"])
        return res
    dock_z = float(dfit["z_ft"])
    if dock_z < res["water_z_ft"] + float(dock_above_water_ft):
        res["gates"]["G_dock"] = {"pass": False,
                                  "detail": f"plane at Z {dock_z:+.2f} ft is not clearly a dock "
                                            f"(need {dock_above_water_ft:g} ft above water)"}
        res.update(status="no_dock", detail=res["gates"]["G_dock"]["detail"])
        return res
    res["dock_z_ft"] = dock_z
    dock_pts = above[dfit["inlier_mask"]]
    res["n_dock"] = int(dock_pts.shape[0])

    # -- pool direction (measured, not assumed) ------------------------------
    water_pts = world[np.abs(world[:, 2] - res["water_z_ft"]) < float(water_band_ft)]
    if water_pts.shape[0] < 3:
        res["gates"]["G_pool_side"] = {"pass": False,
                                       "detail": "no points near the water plane to point toward"}
        res.update(status="no_pool_side",
                   detail="the pool side cannot be determined — no water-region points")
        return res
    pool_dir = water_pts[:, :2].mean(axis=0) - dock_pts[:, :2].mean(axis=0)
    norm = float(np.linalg.norm(pool_dir))
    if not np.isfinite(norm) or norm < 1e-9:
        res["gates"]["G_pool_side"] = {"pass": False,
                                       "detail": "water and dock centroids coincide"}
        res.update(status="no_pool_side", detail=res["gates"]["G_pool_side"]["detail"])
        return res
    pool_dir = pool_dir / norm
    res["pool_dir"] = [float(pool_dir[0]), float(pool_dir[1]), 0.0]
    res["gates"]["G_pool_side"] = {"pass": True, "detail": f"pool dir ~ [{pool_dir[0]:+.3f}, {pool_dir[1]:+.3f}]"}

    # -- lip = the dock-top hull edge facing the pool ------------------------
    hull = _convex_hull_2d(dock_pts[:, :2])
    if hull.shape[0] < 3:
        res.update(status="degenerate", detail="dock-top points are collinear — no area, no lip")
        return res
    best_edge: tuple[float, np.ndarray, np.ndarray] | None = None
    for i in range(hull.shape[0]):
        a = hull[i]
        b = hull[(i + 1) % hull.shape[0]]
        d = b - a
        denom = float(np.linalg.norm(d))
        if denom < 1e-12:
            continue
        outward = np.array([d[1], -d[0]]) / denom  # CCW hull -> interior on the left
        score = float(outward @ pool_dir)
        if best_edge is None or score > best_edge[0]:
            best_edge = (score, a.copy(), b.copy())
    if best_edge is None:
        res.update(status="degenerate", detail="no usable hull edge")
        return res
    edge_a, edge_b = best_edge[1], best_edge[2]

    # Refine the lip DIRECTION from dock-top points within ``lip_band_ft`` of that edge, but keep the
    # line anchored to the hull edge itself (the pool-side boundary).  Anchoring to the band centroid
    # would pull the line back into the dock by ~half the band — a constant datum bias.
    edge_dir = edge_b - edge_a
    edge_norm = float(np.linalg.norm(edge_dir))
    # perpendicular distance (2-D cross) from each dock point to the line through edge_a -> edge_b
    v = dock_pts[:, :2] - edge_a
    perp = np.abs(v[:, 0] * edge_dir[1] - v[:, 1] * edge_dir[0]) / edge_norm
    band = dock_pts[perp < float(lip_band_ft)]
    if band.shape[0] < int(min_lip_points):
        band = np.array([edge_a, edge_b])  # fall back to the hull vertices
    c = band[:, :2].mean(axis=0)
    _u, _s, vt = np.linalg.svd(band[:, :2] - c, full_matrices=False)
    u = np.asarray(vt[0], dtype=float)  # lip direction (unit, 2-D)
    anchor = (edge_a + edge_b) / 2.0     # on the boundary, not the band interior
    # span along the lip direction -> the two corners, on the line through the boundary anchor
    span_pts = np.vstack([band[:, :2], edge_a.reshape(1, 2), edge_b.reshape(1, 2)])
    t = (span_pts - anchor) @ u
    lo, hi = int(np.argmin(t)), int(np.argmax(t))
    corner_lo = np.array([anchor[0] + u[0] * t[lo], anchor[1] + u[1] * t[lo], dock_z])
    corner_hi = np.array([anchor[0] + u[0] * t[hi], anchor[1] + u[1] * t[hi], dock_z])
    near, far = _order_near_far(corner_lo, corner_hi)
    res["n_lip"] = int(band.shape[0])
    res["gates"]["G_lip"] = {"pass": True, "detail": f"{res['n_lip']} lip points, span "
                                                     f"{float(np.linalg.norm(corner_hi - corner_lo)):.2f} ft"}

    res["datum"] = _pcd.datum_line(near, far)
    res["status"] = "ok"
    res["detail"] = (f"dock lip located at world X ~ {float(near[0]):+.2f}..{float(far[0]):+.2f} ft "
                     f"(dock top Z = {dock_z:+.2f} ft, {res['n_lip']} lip points, "
                     f"{res['n_dock']} dock points)")
    return res
