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


def trajectory_from_cloud(
    frames_dir: str | Path,
    detections: Iterable[Any],
    *,
    water_z_ft: float = 0.0,
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
    distance_ft, distance_in, water_z_ft}``.  ``status`` is ``"ok"``, ``"no_points"``,
    ``"few_points"``, ``"no_plane"``, ``"no_fit"`` or ``"no_crossing"``.  ``distance_ft`` is the
    ``X`` of the crossing point (the distance from the dock edge, which the world frame places at
    ``X = 0``), so it is the same quantity the plane/arc routes report — comparable, not a new unit.
    """
    out: dict[str, Any] = {
        "status": None, "detail": "", "points": [], "skipped": [],
        "plane": None, "plane_rms_ft": None, "traj": None, "arc_rms_ft": None,
        "crossing": None, "distance_ft": None, "distance_in": None,
        "water_z_ft": float(water_z_ft),
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
