"""Point-cloud trajectory — 3-D dog trajectory from per-frame point clouds (step 13 / PC-TRAJ).

Where the numbers come from
---------------------------
The mono route (:func:`dog_diving.pool.pipeline.track_to_world`) *unprojects* each detection
pixel to the water height, fits one vertical plane through the smeared ``(X, Y)``, then
re-intersects every ray with that plane to recover a height.  That single-pass bootstrap is
measurably biased — 110 in on the project's shallow synthetic view and 73 in on a steep one
(``test_m30``) — because a monocular pixel carries no depth.

This module is the **stereo / point-cloud replacement** for that bootstrap: the ZED X records
real depth, and ``generate_point_cloud.py`` (step 12) writes it as one image-shaped
``(H, W, 3)`` cloud per frame.  A detection pixel then answers its own 3-D position directly —
``cloud[row, col]``, the same ``[row, col]`` layout ``sensor_placer_auto.get_3d_points_at_pixels``
already indexes — so no unprojection, no plane bootstrap, no parallax smear.  The dog's genuine
3-D points feed the same downstream fit as the mono route (vertical plane + 3-D arc + water
crossing), so the two answers are comparable, and the point-cloud one is the honest distance the
mono route was built to approximate.

Coordinate frames (read before trusting a number)
-------------------------------------------------
``frame_{i}.npy`` holds **metres** in the ZED camera frame (``RIGHT_HANDED_Y_UP``: ``x`` right,
``y`` up, ``z`` forward).  The trajectory fit needs a world frame whose ``Z`` is *up*, because
"the water plane" is ``Z = z_water`` and "a vertical plane" has a horizontal normal.  The default
:func:`zed_to_world` does the single-camera mapping

    world X (along pool) = ZED z (forward)
    world Y (across)     = ZED x (right)
    world Z (up)         = ZED y (up)

and scales metres -> feet.  **The sign/axis convention is a documented prototype assumption for a
camera looking down the pool**; the true camera->pool rigid transform (the floor alignment from
``sensor_placer_auto.py``'s ``find_floor_plane`` + IMU leveling) is hardware-verify-pending and is
step 14's job to reconcile.  The ``to_world`` callable is the seam that transform plugs into.

What is NOT done here (step 14 / 1b-auto)
-----------------------------------------
``water_z_ft`` is an **input**, not estimated from the cloud.  The water height in the camera
frame is a separate problem (fit the dominant horizontal plane to the water region); this module
consumes it.  The synthetic fixture in the tests places the water at ``Z = 0`` by construction.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from . import trajectory as _trajectory
from .pool import geometry as _geometry

#: Exactly 0.3048 m/ft, inverted — the ZED SDK records in metres, the app scores in feet.
FEET_PER_METRE = 3.280839895013123

#: A ZED-camera-frame metre point -> app world feet.  The seam the floor-aligned rigid transform
#: plugs into once it is measured on the rig.
ToWorld = Callable[[np.ndarray], np.ndarray]

#: Below this many world points no flight plane or arc can be fitted.
MIN_POINTS = 3


def metres_to_feet(xyz_m: np.ndarray) -> np.ndarray:
    """Scale a 3-vector from metres to feet, no axis change (used by the canonical fixture)."""
    return np.asarray(xyz_m, dtype=float).ravel()[:3] * FEET_PER_METRE


def zed_to_world(xyz_m: np.ndarray) -> np.ndarray:
    """ZED camera frame (``RIGHT_HANDED_Y_UP``) -> app world feet, single-camera prototype map.

    ``X``/``Y``/``Z`` naming follows the app's world convention (``entry_coordinate`` docstring):
    ``X`` is along the pool (``+X`` = away from the dock edge), ``Y`` across, ``Z`` up.  The ZED
    frame is ``x`` right, ``y`` up, ``z`` forward, so a camera looking down the pool maps its
    forward axis to ``+X`` and its right axis to ``+Y`` (the sign of ``Y`` depends on whether the
    camera is mirrored, which the floor alignment resolves).  Documented assumption, not a
    measured fact — see the module docstring.
    """
    x, y, z = (float(v) for v in np.asarray(xyz_m, dtype=float).ravel()[:3])
    return np.array([z, x, y], dtype=float) * FEET_PER_METRE


def world_to_zed(world_ft: np.ndarray) -> np.ndarray:
    """App world feet -> ZED camera frame metres — the inverse of :func:`zed_to_world`.

    The default ``zed_to_world`` is ``world = (z, x, y) * FEET_PER_METRE``, so the inverse is
    ``zed = (world_Y, world_Z, world_X) / FEET_PER_METRE``.  This is the canonical placeholder for a
    camera looking down the pool with no camera-to-pool offset — the same documented assumption as
    ``zed_to_world``.  The true floor-aligned inverse (camera above the water, behind the dock lip)
    is hardware-verify-pending and plugs into the same seam the seed layer takes.
    """
    X, Y, Z = (float(v) for v in np.asarray(world_ft, dtype=float).ravel()[:3])
    return np.array([Y, Z, X], dtype=float) / FEET_PER_METRE


def frame_path(frames_dir: str | Path, frame_index: int) -> Path:
    """The per-frame cloud file for one ``frame_index`` — the name ``generate_point_cloud.py`` writes."""
    return Path(frames_dir) / f"frame_{int(frame_index):06d}.npy"


def load_frame_cloud(frames_dir: str | Path, frame_index: int) -> np.ndarray | None:
    """Load ``frame_{i:06d}.npy``, or ``None`` when the file is absent."""
    p = frame_path(frames_dir, frame_index)
    if not p.is_file():
        return None
    return np.load(p)


def lookup_point(frame_cloud: np.ndarray, x: float, y: float) -> np.ndarray | None:
    """The 3-D position (metres, cloud frame) at pixel ``(x, y)``, or ``None``.

    Indexes ``cloud[row, col]`` — i.e. ``[round(y), round(x)]`` — the exact layout
    ``get_3d_points_at_pixels`` assumes.  A detection pixel with no depth (``NaN``/``inf``) or
    outside the frame must not silently become ``(0, 0, 0)``: it returns ``None`` and the caller
    records *why*, because a dog point invented at the origin is a distance of zero with full
    confidence.
    """
    arr = np.asarray(frame_cloud, dtype=float)
    h, w = arr.shape[:2]
    row, col = int(round(float(y))), int(round(float(x)))
    if not (0 <= row < h and 0 <= col < w):
        return None
    pt = arr[row, col, :3]
    if not np.all(np.isfinite(pt)):
        return None
    return np.asarray(pt, dtype=float).ravel()[:3]


def _apex_frame(coeffs: dict) -> float | None:
    """Frame of the fitted ``Z(t)`` maximum, from the Z quadratic alone (descending-branch gate)."""
    z = (coeffs or {}).get("z")
    if not z:
        return None
    a2, a1, _ = (float(v) for v in z)
    if a2 == 0.0:
        return None
    return float(coeffs["t_ref"]) - a1 / (2.0 * a2)


def _detection_parts(det: Any) -> tuple[int, float, float] | None:
    """Normalise one detection to ``(frame, x, y)`` from a dict or a ``(frame, x, y)`` sequence."""
    if isinstance(det, dict):
        if det.get("x") is None or det.get("y") is None or det.get("frame") is None:
            return None
        return int(det["frame"]), float(det["x"]), float(det["y"])
    seq = list(det)
    if len(seq) < 3:
        return None
    return int(seq[0]), float(seq[1]), float(seq[2])


def points_at_detections(
    frames_dir: str | Path,
    detections: Iterable[Any],
    *,
    to_world: ToWorld = zed_to_world,
) -> dict[str, Any]:
    """Look up every detection's 3-D position in the per-frame clouds, in world feet.

    ``detections`` is any iterable of ``{frame, x, y}`` dicts or ``(frame, x, y)`` triples (the
    accepted tail-set / nose points).  Each is looked up in its own ``frame_{i:06d}.npy``; a
    missing file, an out-of-frame pixel or a non-finite cloud value skips it **with a reason**
    rather than silently contributing ``(0, 0, 0)``.

    Returns ``{points, skipped, n_total, n_valid, n_skipped}`` where each ``points`` entry carries
    the pixel columns (``x``/``y``, kept for the drawn arc and provenance) plus ``X_ft``/``Y_ft``/
    ``Z_ft`` (world feet) and ``pos_m`` (the raw metre lookup, before ``to_world``).
    """
    points: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for det in detections:
        parts = _detection_parts(det)
        if parts is None:
            skipped.append({"frame": None, "x": None, "y": None,
                            "reason": "malformed_detection"})
            continue
        frame, x, y = parts
        cloud = load_frame_cloud(frames_dir, frame)
        if cloud is None:
            skipped.append({"frame": frame, "x": x, "y": y, "reason": "missing_frame"})
            continue
        pos_m = lookup_point(cloud, x, y)
        if pos_m is None:
            skipped.append({"frame": frame, "x": x, "y": y, "reason": "no_depth"})
            continue
        world = np.asarray(to_world(pos_m), dtype=float).ravel()
        if world.size < 3 or not np.all(np.isfinite(world[:3])):
            skipped.append({"frame": frame, "x": x, "y": y, "reason": "bad_transform"})
            continue
        points.append({
            "frame": frame, "x": x, "y": y,
            "X_ft": float(world[0]), "Y_ft": float(world[1]), "Z_ft": float(world[2]),
            "pos_m": [float(v) for v in pos_m],
        })
    return {
        "points": points,
        "skipped": skipped,
        "n_total": len(points) + len(skipped),
        "n_valid": len(points),
        "n_skipped": len(skipped),
    }


# ============================================================================
# The water plane — where ``water_z_ft`` comes from (step 6 / 1b-auto)
# ============================================================================
#: RANSAC plane fit defaults.  ``dist_thresh_m`` is the inlier band in metres; the ZED's depth
#: noise on a specular water surface is coarse, so the band is generous but the inlier count is what
#: gates "dominant".
DIST_THRESH_M = 0.05
MIN_INLIERS = 16
MAX_ITER = 200
#: A plane whose world normal is within this many degrees of vertical counts as "the water".
ANGLE_TOL_DEG = 15.0


def _finite_points(cloud: Any) -> np.ndarray:
    """Flatten a cloud (``(N,3)`` or image-shaped ``(H,W,3)``) to finite ``(M,3)`` metres."""
    arr = np.asarray(cloud, dtype=float)
    if arr.size == 0:
        return np.empty((0, 3), dtype=float)
    if arr.ndim == 3:
        arr = arr.reshape(-1, arr.shape[2])
    arr = arr[:, :3]
    return arr[np.isfinite(arr).all(axis=1)]


def _plane_normal(p3: np.ndarray) -> np.ndarray | None:
    """Unit normal of three points, or ``None`` when they are collinear/coincident."""
    p = np.asarray(p3, dtype=float)
    v1, v2 = p[1] - p[0], p[2] - p[0]
    n = np.cross(v1, v2)
    norm = float(np.linalg.norm(n))
    if not np.isfinite(norm) or norm < 1e-12:
        return None
    return n / norm


def _refine_plane(pts: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Least-squares plane (SVD) on ``pts``; returns ``(unit normal, offset, rms_m)``."""
    pts = np.asarray(pts, dtype=float)
    c = pts.mean(axis=0)
    _u, _s, vt = np.linalg.svd(pts - c, full_matrices=False)
    n = np.asarray(vt[-1], dtype=float)
    norm = float(np.linalg.norm(n))
    if norm < 1e-12 or not np.isfinite(norm):
        n = np.array([0.0, 1.0, 0.0])
    else:
        n = n / norm
    d = float(n @ c)
    rms = float(np.sqrt(np.mean(((pts - c) @ n) ** 2)))
    return n, d, rms


def _dir_world(to_world: ToWorld, origin_m: np.ndarray, n_m: np.ndarray) -> np.ndarray | None:
    """Map a **direction** ``n_m`` through ``to_world`` (finite-difference, exact for a linear map)."""
    a = np.asarray(to_world(origin_m), dtype=float).ravel()[:3]
    b = np.asarray(to_world(np.asarray(origin_m, dtype=float) + np.asarray(n_m, dtype=float)),
                   dtype=float).ravel()[:3]
    v = b - a
    norm = float(np.linalg.norm(v))
    if not np.isfinite(norm) or norm < 1e-12:
        return None
    return v / norm


def fit_water_plane(
    cloud: Any,
    *,
    to_world: ToWorld = zed_to_world,
    dist_thresh_m: float = DIST_THRESH_M,
    min_inliers: int = MIN_INLIERS,
    max_iter: int = MAX_ITER,
    angle_tol_deg: float = ANGLE_TOL_DEG,
    seed: int = 0,
) -> dict[str, Any]:
    """Fit the dominant horizontal plane to a point cloud -> ``water_z_ft`` in world feet.

    The water is a single physical surface seen by (ideally) two cameras, so a **fused** cloud
    (``fusion.fuse_point_clouds``) is preferred, but a single image-shaped cloud or a
    ``merge_frames_cloud`` stack works too.  RANSAC finds the plane with the most inliers (the water
    dominates the scene), then a least-squares refinement pins it; the fitted normal is checked to be
    within ``angle_tol_deg`` of vertical **in world feet** (the ``to_world`` seam, so the check is
    independent of the ZED frame) and ``water_z_ft`` is that plane's world ``Z``.

    Returns ``{status, normal_m, offset_m, water_z_ft, rms_m, n_inliers, inlier_fraction, n_points,
    detail}`` with ``status`` ``\"ok\"``, ``\"no_points\"``, ``\"degenerate\"``, ``\"few_inliers\"``
    or ``\"not_horizontal\"`` — a refused fit never carries a ``water_z_ft``.
    """
    out: dict[str, Any] = {"status": None, "normal_m": None, "offset_m": None, "water_z_ft": None,
                           "rms_m": None, "n_inliers": 0, "inlier_fraction": None,
                           "n_points": 0, "detail": ""}
    pts = _finite_points(cloud)
    out["n_points"] = int(pts.shape[0])
    if pts.shape[0] < 3:
        out.update(status="no_points",
                   detail=f"{pts.shape[0]} finite point(s) — a plane needs at least 3")
        return out

    rng = np.random.default_rng(int(seed))
    best: tuple[int, np.ndarray, float, np.ndarray] | None = None
    for _ in range(int(max_iter)):
        idx = rng.choice(pts.shape[0], 3, replace=False)
        n = _plane_normal(pts[idx])
        if n is None:
            continue
        d = float(n @ pts[idx[0]])
        dist = np.abs(pts @ n - d)
        mask = dist < float(dist_thresh_m)
        cnt = int(mask.sum())
        if best is None or cnt > best[0]:
            best = (cnt, n, d, mask)
    if best is None:
        out.update(status="degenerate", detail="no 3 non-collinear samples produced a plane")
        return out
    cnt, n, d, mask = best
    if cnt < int(min_inliers):
        out.update(status="few_inliers", n_inliers=cnt,
                   detail=f"the dominant plane has {cnt} inlier(s) (need {int(min_inliers)}) — "
                          f"the water does not dominate this cloud")
        return out
    n, d, rms = _refine_plane(pts[mask])
    out["normal_m"] = n.tolist()
    out["offset_m"] = float(d)
    out["rms_m"] = float(rms)
    out["n_inliers"] = int(mask.sum())
    out["inlier_fraction"] = float(mask.sum() / pts.shape[0])

    origin_m = n * d
    nw = _dir_world(to_world, origin_m, n)
    if nw is None:
        out.update(status="degenerate", detail="the plane normal could not be mapped to world feet")
        return out
    if nw[2] < 0.0:
        n, d, nw = -n, -d, -nw
        origin_m = n * d
        out["normal_m"] = n.tolist()
        out["offset_m"] = float(d)
    cos_up = float(nw[2])
    if cos_up < float(math.cos(math.radians(angle_tol_deg))):
        out.update(status="not_horizontal",
                   detail=f"the dominant plane tilts {math.degrees(math.acos(min(max(cos_up, -1.0), 1.0))):.1f} deg "
                          f"from vertical in world feet (allow {angle_tol_deg:g}) — not a water surface")
        return out
    out["water_z_ft"] = float(np.asarray(to_world(origin_m), dtype=float).ravel()[2])
    out["status"] = "ok"
    out["detail"] = (f"water plane at world Z = {out['water_z_ft']:+.2f} ft "
                     f"({out['n_inliers']} inliers, {out['inlier_fraction']:.1%} of {out['n_points']} pts, "
                     f"RMS {out['rms_m']:.3f} m)")
    return out


def merge_frames_cloud(frames_dir: str | Path,
                       frame_indices: Iterable[int] | None = None) -> np.ndarray:
    """Stack every finite point of the per-frame clouds into one ``(M, 3)`` metres cloud.

    The fused cloud is the better input for the water-plane fit (one surface, two cameras), but when
    only the per-frame ``frame_{i:06d}.npy`` files exist this is the fallback — the water region in
    any single frame is enough to fit a horizontal plane.
    """
    frames_dir = Path(frames_dir)
    if frame_indices is None:
        files = sorted(frames_dir.glob("frame_*.npy"))
    else:
        files = [frame_path(frames_dir, int(i)) for i in frame_indices]
    blocks = [_finite_points(np.load(f)) for f in files if f.is_file()]
    if not blocks:
        return np.empty((0, 3), dtype=float)
    return np.vstack(blocks)


def trajectory_from_cloud(
    frames_dir: str | Path,
    detections: Iterable[Any],
    *,
    water_z_ft: float | None = None,
    water_cloud: Any = None,
    min_points: int = MIN_POINTS,
    to_world: ToWorld = zed_to_world,
) -> dict[str, Any]:
    """Point cloud -> world-feet trajectory -> water crossing -> distance.

    The chain, each step reporting its own failure rather than a placeholder number:

    * :func:`points_at_detections` — genuine 3-D points at the detection pixels;
    * :func:`dog_diving.pool.geometry.vertical_plane_from_points` — the vertical flight plane
      (fitted on the horizontal ``(X, Y)`` only, the same primitive the mono route uses);
    * :func:`dog_diving.trajectory.build_trajectory(space="3d")` — the robust 3-D arc;
    * :func:`dog_diving.pool.geometry.intersect_curve_with_elevation` — where ``Z(t)`` meets
      ``water_z_ft`` on the descending branch (apex-gated).

    Returns ``{status, detail, points, skipped, plane, plane_rms_ft, traj, arc_rms_ft, crossing,
    distance_ft, distance_in, water_z_ft, water_z_source, water_fit}``.  ``status`` is ``"ok"``,
    ``"no_water_level"``, ``"no_points"``, ``"few_points"``, ``"no_plane"``, ``"no_fit"`` or
    ``"no_crossing"``.  ``distance_ft`` is the
    ``X`` of the crossing point (the distance from the dock edge, which the world frame places at
    ``X = 0``), so it is the same quantity the plane/arc routes report — comparable, not a new unit.

    ``water_z_ft`` is resolved just before the crossing (the only step that needs it): pass it
    explicitly (the old path), or pass ``water_cloud`` (a fused or image-shaped cloud, or a
    ``merge_frames_cloud`` stack) and it is fitted by :func:`fit_water_plane`.  When neither is given
    and ``water_z_ft`` is ``None`` the per-frame clouds in ``frames_dir`` are merged and fitted as a
    last resort; if that fails the result is ``no_water_level`` — the water height is essential to
    the crossing, so it is never silently defaulted to zero.
    """
    out: dict[str, Any] = {
        "status": None, "detail": "", "points": [], "skipped": [],
        "plane": None, "plane_rms_ft": None, "traj": None, "arc_rms_ft": None,
        "crossing": None, "distance_ft": None, "distance_in": None,
        "water_z_ft": None, "water_z_source": None, "water_fit": None,
    }

    looked = points_at_detections(frames_dir, detections, to_world=to_world)
    out["points"], out["skipped"] = looked["points"], looked["skipped"]
    if not out["points"]:
        reasons = sorted({s["reason"] for s in out["skipped"]})
        out.update({"status": "no_points",
                    "detail": (f"no detection produced a finite 3-D point "
                               f"(skipped {looked['n_skipped']}: {', '.join(reasons)})")})
        return out
    if len(out["points"]) < int(min_points):
        out.update({"status": "few_points",
                    "detail": f"{len(out['points'])} finite 3-D point(s); a flight plane needs at "
                              f"least {int(min_points)}"})
        return out

    fit = _geometry.vertical_plane_from_points(
        out["points"], min_points=int(min_points), source="point cloud")
    if fit["status"] != "ok":
        out.update({"status": f"no_plane_{fit['status']}", "detail": fit["detail"],
                    "plane_rms_ft": fit.get("rms_ft")})
        return out
    out["plane"] = fit["plane"].as_dict()
    out["plane_rms_ft"] = float(fit["rms_ft"])

    traj = _trajectory.build_trajectory(out["points"], space="3d", plane=fit["plane"])
    out["traj"] = traj
    if traj.coeffs is None or traj.coeffs.get("z") is None:
        out.update({"status": "no_fit",
                    "detail": "the 3-D points were built but no arc survived the fit"})
        return out
    out["arc_rms_ft"] = traj.arc_rms_ft

    if water_z_ft is None:
        source_cloud = water_cloud
        if source_cloud is None:
            source_cloud = merge_frames_cloud(frames_dir)
        wfit = fit_water_plane(source_cloud, to_world=to_world)
        out["water_fit"] = wfit
        if wfit["status"] != "ok":
            out.update(status="no_water_level",
                       detail=f"no water plane could be fitted: {wfit['detail']}")
            return out
        water_z_ft = wfit["water_z_ft"]
        out["water_z_ft"] = float(water_z_ft)
        out["water_z_source"] = "cloud_fit"
    else:
        out["water_z_ft"] = float(water_z_ft)
        out["water_z_source"] = "supplied"

    apex = _apex_frame(traj.coeffs)
    cross = _geometry.intersect_curve_with_elevation(traj.coeffs, float(water_z_ft), t_min=apex)
    if cross["status"] != "ok":
        out.update({"status": "no_crossing", "detail": cross["detail"]})
        return out
    P = np.asarray(cross["point"], dtype=float)
    out["crossing"] = {
        "frame": float(cross["t"]),
        "point": [float(v) for v in P],
        "t_all": [float(v) for v in (cross.get("t_all") or [])],
        "apex_frame": apex,
        "extrap_frames": float(cross["t"]) - float(traj.last_arc_frame)
        if getattr(traj, "last_arc_frame", None) is not None else None,
    }
    out["distance_ft"] = float(P[0])
    out["distance_in"] = float(P[0]) * 12.0
    out["status"] = "ok"
    out["detail"] = (f"z = {float(water_z_ft):+.2f} ft crossed at frame {cross['t']:.2f} "
                     f"(X = {P[0]:.2f} ft, plane RMS {fit['rms_ft']:.3f} ft, "
                     f"arc RMS {traj.arc_rms_ft if traj.arc_rms_ft is not None else float('nan'):.3f} ft)")
    return out
