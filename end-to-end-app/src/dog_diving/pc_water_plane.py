"""Point cloud -> water plane -> seed the pool ``PlaneSession`` (step 6 / 1b-auto).

The point-cloud cluster (steps 12-14) reads real ZED depth at detection pixels and scores a jump, but
it still needs two numbers the operator supplies by hand: ``water_z_ft`` (the water plane's height in
world feet) and the dock-lip datum.  This module closes the *water* half of that gap.

Two layers, split so each is testable without the other:

* :func:`dog_diving.pc_trajectory.fit_water_plane` (the fit) — RANSAC + least-squares on the fused /
  per-frame cloud, returning ``water_z_ft``.  The trajectory/distance modules consume it directly.
* :func:`seed_plane_session` (this module) — take the same cloud + the camera intrinsics, fit the
  water plane, project the metric rectangle's four corners to image pixels, and build the
  :class:`dog_diving.pool.plane.PlaneSession` the operator then drags to fine-tune.

The seed's whole point is the operator's standing order: *"the plane should be automatically placed
... and manually adjusted by the user using the existing controls."*  The fitted plane is the
**starting quad**, not the answer — ``PlaneSession.apply_seed`` records ``placement: "auto"`` and the
first drag flips it to ``"auto+edited"``, so a reviewer can tell a seeded plane from a hand-drawn one.

What is still a placeholder (read before trusting a corner)
-----------------------------------------------------------
``world_to_zed`` is the canonical inverse of ``pc_trajectory.zed_to_world``: a camera sitting AT the
dock edge, level, looking down the pool.  That is the same documented prototype assumption the
trajectory cluster already carries, and it is wrong in one important way the seed makes visible: a
corner at ``X = 0`` maps to ``ZED z = 0``, which is *at* the camera and does not project.  The true
floor-aligned camera pose (camera above the water, behind the lip) is hardware-verify-pending; it
plugs into the ``world_to_zed`` seam exactly where ``to_world`` plugs into ``zed_to_world``.  Until
it is measured, the seed is a shape-true prototype, not a metric answer.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np

from . import pc_trajectory as _pc
from .pool import plane as _plane

CORNER_TAGS = _plane.CORNER_TAGS

#: App world feet -> ZED metres.  The seam the floor-aligned camera pose plugs into.
WorldToZed = Callable[[np.ndarray], np.ndarray]


def intrinsics_matrix(intrinsics: Any) -> np.ndarray:
    """A 3×3 pinhole ``K`` from a ``fx/fy/cx/cy`` dict (the generator's JSON) or an existing 3×3."""
    if isinstance(intrinsics, dict):
        fx, fy = float(intrinsics["fx"]), float(intrinsics["fy"])
        cx = float(intrinsics.get("cx", 0.0))
        cy = float(intrinsics.get("cy", 0.0))
        return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=float)
    a = np.asarray(intrinsics, dtype=float)
    if a.shape == (3, 3):
        return a
    raise ValueError(f"intrinsics must be a fx/fy/cx/cy dict or a 3x3 matrix, got shape {a.shape}")


def project_zed_to_pixel(p_m: Sequence[float], K: np.ndarray) -> tuple[float, float] | None:
    """ZED metres -> image pixel via pinhole ``K``; ``None`` at/behind the camera plane."""
    p = np.asarray(p_m, dtype=float).ravel()[:3]
    if p[2] <= 1e-9:
        return None
    uv = K @ p
    return (float(uv[0] / uv[2]), float(uv[1] / uv[2]))


def _gate(pass_: bool, detail: str, *, warn: bool = False) -> dict[str, Any]:
    return {"pass": bool(pass_), "warn": bool(warn), "detail": detail}


def seed_plane_session(
    cloud: Any,
    *,
    intrinsics: Any,
    length_ft: float,
    width_ft: float,
    frame_shape: tuple[int, int],
    to_world: _pc.ToWorld = _pc.zed_to_world,
    world_to_zed: WorldToZed = _pc.world_to_zed,
    dock_offset_ft: float = 0.0,
    angle_tol_deg: float = _pc.ANGLE_TOL_DEG,
) -> dict[str, Any]:
    """Fit the water plane and seed a ``PlaneSession`` from it.

    The fitted plane's ``water_z_ft`` pins the quad's height; the metric rectangle
    ``(dock_offset_ft .. dock_offset_ft + length_ft) × (0 .. width_ft)`` on that plane is projected
    to four image corners through the camera intrinsics and the ``world_to_zed`` seam, then written
    into a :class:`~dog_diving.pool.plane.PlaneSession` via ``apply_seed`` (so the operator's drag
    controls remain the manual adjustment, and ``placement`` is recorded ``auto`` until the first
    drag).

    Returns ``{status, corners, session, water_z_ft, water_fit, gates, notes, seed_block, detail}``.
    ``session`` is the seeded ``PlaneSession`` (or ``None``); ``corners`` is the four image-px tags;
    ``gates`` carries the fit (G1), horizontality (G2), projection (G3) and quad (G4, P1-P4 through
    the app's own validator) so a refusal names exactly what failed.
    """
    res: dict[str, Any] = {
        "status": None, "corners": None, "session": None, "water_z_ft": None,
        "water_fit": None, "gates": {}, "notes": [], "seed_block": None, "detail": "",
    }

    fit = _pc.fit_water_plane(cloud, to_world=to_world, angle_tol_deg=angle_tol_deg)
    res["water_fit"] = fit
    res["gates"]["G1_fit"] = _gate(fit["status"] == "ok", fit["detail"])
    if fit["status"] != "ok":
        res["status"] = f"no_water_plane_{fit['status']}"
        res["detail"] = fit["detail"]
        res["notes"].append("no water plane was fitted — the PlaneSession was left untouched")
        return res
    res["water_z_ft"] = float(fit["water_z_ft"])

    K = intrinsics_matrix(intrinsics)
    corners_world = {"00": (0.0, 0.0), "10": (float(length_ft), 0.0),
                     "11": (float(length_ft), float(width_ft)), "01": (0.0, float(width_ft))}
    corners_px: dict[str, tuple[float, float]] = {}
    projection_ok = True
    for tag in CORNER_TAGS:
        X, Y = corners_world[tag]
        zed = world_to_zed(np.array([X + float(dock_offset_ft), Y, res["water_z_ft"]]))
        uv = project_zed_to_pixel(zed, K)
        if uv is None or not np.all(np.isfinite(uv)):
            res["notes"].append(
                f"corner {tag} lies at/behind the camera — the rectangle cannot be projected with "
                f"the current world->camera mapping")
            projection_ok = False
            continue
        corners_px[tag] = (float(uv[0]), float(uv[1]))
    res["gates"]["G3_projection"] = _gate(
        projection_ok,
        f"{len(corners_px)}/4 corners projected" +
        ("" if projection_ok else " — a corner is at/behind the camera (see the world_to_zed seam)"))
    if not projection_ok:
        res["status"] = "projection_failed"
        res["detail"] = "one or more corners could not be projected to the image"
        return res

    session = _plane.PlaneSession(frame_shape=(int(frame_shape[0]), int(frame_shape[1])),
                                  length_ft=float(length_ft), width_ft=float(width_ft),
                                  dock_offset_ft=float(dock_offset_ft))
    block = {
        "upstream": "point_cloud",
        "placement": "auto",
        "water_z_ft": res["water_z_ft"],
        "water_fit": {k: fit[k] for k in ("status", "normal_m", "offset_m", "rms_m",
                                          "n_inliers", "inlier_fraction", "n_points")},
        "notes": list(res["notes"]),
    }
    session.apply_seed(corners_px, block=block)
    res["session"] = session
    res["corners"] = dict(corners_px)
    res["seed_block"] = block

    try:
        pgates = _plane.validate_plane({"homography": session.homography().tolist()}, session)
    except Exception as exc:  # noqa: BLE001
        pgates = {"P": {"pass": False, "warn": False, "detail": f"not checkable: {exc}"}}
    bad = [k for k, v in pgates.items() if not v.get("pass") and not v.get("warn")]
    res["gates"]["G4_quad"] = {
        "pass": not bad, "warn": False,
        "detail": "; ".join(f"{k} {'ok' if v.get('pass') else 'FAIL'} ({v.get('detail', '')})"
                            for k, v in sorted(pgates.items())),
        "sub": {k: bool(v.get("pass")) for k, v in pgates.items()},
    }
    if bad:
        res["status"] = "refused"
        res["detail"] = "the seeded quad failed: " + ", ".join(f"{k} — {pgates[k]['detail']}" for k in bad)
        return res
    res["status"] = "ok"
    res["detail"] = (f"water plane at Z = {res['water_z_ft']:+.2f} ft seeded a "
                     f"{length_ft:g} x {width_ft:g} ft quad (4 corners, P-gates pass)")
    return res
