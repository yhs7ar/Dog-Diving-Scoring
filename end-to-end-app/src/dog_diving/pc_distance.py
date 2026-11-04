"""Point-cloud distance — route ``"pc"`` (step 14 / PC-DIST).

Where this sits
---------------
The point-cloud route answers the same question the two mono routes already do — *"how far from
the dock lip did the dog enter the water?"* — but from a different measurement source.  The three
routes and their inputs:

| route    | function                               | input                                  |
|----------|----------------------------------------|----------------------------------------|
| ``plane``  | :func:`dog_diving.measure.plane_distance`           | one pixel through the plane's homography |
| ``arc3d``  | :func:`dog_diving.measure.trajectory_plane_distance` | mono track -> world arc -> water crossing |
| ``pc``     | :func:`pc_distance` (this module)                  | per-frame point clouds at the detection pixels |

The first two unproject a pixel (one ray, no depth); the point-cloud route reads real ZED depth at
the detection pixel, so a dog point answers its own 3-D position without the single-pass plane
bootstrap that ``track_to_world`` uses and that is measurably biased on shallow views (110 in on the
project's synthetic shallow view — ``test_m30``).  ZED depth on water is itself approximate (the
surface is specular and moving), so this route is a **cross-check, not a replacement** for the
datum: three routes that agree is evidence; three routes that disagree is a measurement to find.

The dock-lip datum
------------------
``pc_trajectory.trajectory_from_cloud`` reports the crossing point's ``X`` (the along-pool
coordinate, in world feet, where the world frame places the dock edge at ``X = 0``).  But the dock
lip is *not yet located in the cloud* — placing it is step 1b-auto's job, still pending.  Until it
is, the datum is an **input**: the operator supplies the lip's two physical corners in world feet
(``near_corner`` / ``far_corner``, the same convention the dock-edge keypoints use), and
:func:`distance_to_lip` measures the **horizontal** perpendicular distance from the crossing point
to that line.  With the lip at ``X = 0`` (the synthetic fixture, and the documented assumption) the
two quantities coincide; with the lip offset from the origin they differ by exactly that offset,
which is the constant datum bias this project keeps re-finding (lip vs water-contact is ~2 in).

The distance is horizontal by construction: the dock lip's height and the entry point's height do
not enter "distance from the dock" — only the ``(X, Y)`` projection of the lip line matters.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from . import distance as _distance
from . import pc_trajectory as _pc

#: This route's name, next to ``measure.ROUTE`` (``"plane"``) and ``measure.ROUTE_ARC3D``
#: (``"arc3d"``).  A literal repeated in two files is a literal that will drift, so the Results
#: pane / report switch on it here.
ROUTE_PC = "pc"

#: Below this many world points no flight plane or arc can be fitted (re-exported from
#: ``pc_trajectory`` so a caller reads one module).
MIN_POINTS = _pc.MIN_POINTS

#: Two lip corners this close together (ft) make the datum line degenerate.
DEGENERATE_LIP_FT = 1e-9


def datum_line(near_corner: Sequence[float], far_corner: Sequence[float]) -> dict[str, Any]:
    """A dock-lip line in world feet, from its two physical corners.

    ``near_corner`` / ``far_corner`` are 3-vectors in world feet (``X`` along the pool, ``Y``
    across, ``Z`` up).  The order is the semantics — ``near -> far`` — and the signed distance
    (right-hand rule about that direction) flips if the two are swapped, exactly as the
    dock-edge keypoints' element order does (see the ``dock-diving-scoring`` skill).  The corners'
    ``Z`` is carried as provenance of where the lip sits above the water; the *measurement* drops
    it (horizontal distance only).

    Returns ``{a, b, u, length_ft, valid, reason}`` with ``a``/``b`` the two corners, ``u`` the
    unit direction ``a -> b``, and ``valid`` False (with ``reason``) for a non-finite or
    coincident pair.
    """
    a = np.asarray(near_corner, dtype=float).ravel()[:3]
    b = np.asarray(far_corner, dtype=float).ravel()[:3]
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        return {"a": a.tolist(), "b": b.tolist(), "u": None, "length_ft": None,
                "valid": False, "reason": "non-finite lip corner"}
    u = b - a
    length = float(np.linalg.norm(u))
    if length < DEGENERATE_LIP_FT:
        return {"a": a.tolist(), "b": b.tolist(), "u": None, "length_ft": 0.0,
                "valid": False, "reason": "degenerate lip line (coincident corners)"}
    return {"a": a.tolist(), "b": b.tolist(), "u": (u / length).tolist(),
            "length_ft": length, "valid": True, "reason": ""}


def horizontal_distance_to_line(point: Sequence[float], a: Sequence[float],
                                b: Sequence[float]) -> float | None:
    """Perpendicular distance in the ``(X, Y)`` plane from ``point`` to the line ``a -> b``.

    The dock distance is horizontal, so the ``Z`` of all three inputs is ignored — the lip's
    height and the entry point's height cannot change "distance from the dock".  Returns ``None``
    for a degenerate line (``a == b`` in the horizontal plane).
    """
    p = np.asarray(point, dtype=float).ravel()[:2]
    pa = np.asarray(a, dtype=float).ravel()[:2]
    pb = np.asarray(b, dtype=float).ravel()[:2]
    d = pb - pa
    if not np.all(np.isfinite(d)):
        return None
    denom = float(np.linalg.norm(d))
    if denom < DEGENERATE_LIP_FT:
        return None
    # 2-D cross product, written out: np.cross on 2-vectors is deprecated in NumPy 2.0.
    dx, dy = float(d[0]), float(d[1])
    vx, vy = float(p[0] - pa[0]), float(p[1] - pa[1])
    return float(abs(dx * vy - dy * vx) / denom)


def signed_horizontal_distance(point: Sequence[float], a: Sequence[float],
                               b: Sequence[float]) -> float | None:
    """The signed form of :func:`horizontal_distance_to_line`.

    Sign is the right-hand rule about ``a -> b`` (positive = ``point`` on the left of the directed
    line).  The sign's *physical* meaning (which side is the pool) depends on the ``near -> far``
    ordering and on which side of the dock the pool is — a rig question, not derivable from the
    numbers — so this is reported as a geometric signed value, never as a "pool side" verdict.
    """
    p = np.asarray(point, dtype=float).ravel()[:2]
    pa = np.asarray(a, dtype=float).ravel()[:2]
    pb = np.asarray(b, dtype=float).ravel()[:2]
    d = pb - pa
    if not np.all(np.isfinite(d)):
        return None
    denom = float(np.linalg.norm(d))
    if denom < DEGENERATE_LIP_FT:
        return None
    # 2-D cross product, written out (np.cross on 2-vectors is deprecated in NumPy 2.0).
    dx, dy = float(d[0]), float(d[1])
    vx, vy = float(p[0] - pa[0]), float(p[1] - pa[1])
    return (dx * vy - dy * vx) / denom


def distance_to_lip(point: Sequence[float], lip: dict[str, Any] | None) -> dict[str, Any]:
    """The horizontal distance from a world point to the dock-lip datum line.

    ``lip`` is the dict :func:`datum_line` returns.  A missing or invalid datum is reported as a
    reason, never a distance of zero — a dog point invented at the lip is a score of 0 in with
    full confidence, the exact failure this project's measurement rules exist to prevent.
    """
    out: dict[str, Any] = {"distance_ft": None, "distance_in": None, "signed_ft": None,
                           "valid": False, "detail": ""}
    if not lip or not lip.get("valid"):
        out["detail"] = ("no valid dock-lip datum — the lip has not been located in the point "
                         "cloud, so the along-pool X is reported without a lip-distance cross-check")
        return out
    a = np.asarray(lip["a"], dtype=float)
    b = np.asarray(lip["b"], dtype=float)
    d = horizontal_distance_to_line(point, a, b)
    s = signed_horizontal_distance(point, a, b)
    if d is None:
        out["detail"] = "degenerate lip line"
        return out
    out.update({"distance_ft": d, "distance_in": d * 12.0, "signed_ft": s, "valid": True})
    return out


def pc_distance(
    frames_dir: str,
    detections: Iterable[Any],
    *,
    lip: dict[str, Any] | None = None,
    water_z_ft: float = 0.0,
    min_points: int = MIN_POINTS,
    to_world: _pc.ToWorld = _pc.zed_to_world,
) -> dict[str, Any]:
    """Route ``"pc"`` — the point-cloud water crossing, in inches, plus the lip cross-check.

    The chain is :func:`pc_trajectory.trajectory_from_cloud` (genuine 3-D points at the detection
    pixels -> vertical flight plane -> 3-D arc -> water crossing), so every failure mode that
    function names (``no_points`` / ``few_points`` / ``no_plane_*`` / ``no_fit`` / ``no_crossing``)
    surfaces here as a warning rather than a placeholder number.  A non-finite or no-depth
    detection pixel is skipped upstream and therefore **cannot** contribute a ``(0, 0, 0)`` point
    that would read as a zero distance.

    ``distance_ft`` / ``total_inches`` are the crossing point's ``X`` (along the pool, world feet
    -> inches) — the same quantity the ``plane`` and ``arc3d`` routes report, so the three are
    directly comparable.  With a valid ``lip`` datum, ``lip_distance_ft`` / ``lip_distance_in`` are
    the **horizontal perpendicular** distance from that crossing point to the lip line — the
    distance the operator actually cares about when the world frame's ``X = 0`` is not the lip.

    The output shape mirrors ``measure.plane_distance`` / ``measure.trajectory_plane_distance``
    (``route`` / ``total_inches`` / ``display`` / ``distance_ft`` / ``transverse_ft`` / ``frame``
    / ``point_source`` / ``warnings``) so the Results pane and ``result.json`` read one schema.
    """
    out: dict[str, Any] = {
        "route": ROUTE_PC, "total_inches": None, "display": "n/a", "sigma_in": None,
        "distance_ft": None, "transverse_ft": None, "frame": None,
        "point_source": "unset", "point_origin": None, "x_px": None, "y_px": None,
        "warnings": [], "excluded_frames": [], "entry_override": False,
        "water_z_ft": float(water_z_ft), "status": None, "detail": "",
        "n_points": 0, "n_skipped": 0, "plane_rms_ft": None, "arc_rms_ft": None,
        "crossing_frame": None, "extrap_frames": None, "crossing_point_ft": None,
        "lip": None, "lip_distance_ft": None, "lip_distance_in": None, "lip_signed_ft": None,
    }

    cloud = _pc.trajectory_from_cloud(
        frames_dir, detections, water_z_ft=water_z_ft, min_points=min_points, to_world=to_world)
    out["status"] = cloud["status"]
    out["detail"] = cloud["detail"]
    out["n_points"] = len(cloud["points"])
    out["n_skipped"] = len(cloud["skipped"])
    out["plane_rms_ft"] = cloud["plane_rms_ft"]
    out["arc_rms_ft"] = cloud["arc_rms_ft"]

    if cloud["status"] != "ok":
        out["warnings"].append(
            f"the point-cloud route could not answer ({cloud['status']}): {cloud['detail']}")
        return out

    P = np.asarray(cloud["crossing"]["point"], dtype=float)
    out["point_source"] = "model"
    out["point_origin"] = "point_cloud"
    out["distance_ft"] = float(P[0])
    out["transverse_ft"] = float(P[1])
    out["total_inches"] = float(P[0]) * 12.0
    out["display"] = _distance.format_feet_inches(out["total_inches"])
    out["frame"] = int(round(float(cloud["crossing"]["frame"])))
    out["crossing_frame"] = cloud["crossing"]["frame"]
    out["extrap_frames"] = cloud["crossing"].get("extrap_frames")
    out["crossing_point_ft"] = [float(v) for v in P]

    if lip is not None:
        dl = distance_to_lip(P, lip)
        out["lip"] = dl
        if dl["valid"]:
            out["lip_distance_ft"] = dl["distance_ft"]
            out["lip_distance_in"] = dl["distance_in"]
            out["lip_signed_ft"] = dl["signed_ft"]
        else:
            out["warnings"].append(dl["detail"])
    return out


def agreement_table(routes: Mapping[str, Any], *,
                    tolerance_in: float | None = None) -> dict[str, Any]:
    """Pairwise agreement in inches between every route that produced a number.

    ``routes`` maps a route name to its result dict (any of ``plane_distance`` /
    ``trajectory_plane_distance`` / ``pc_distance``).  A result with ``total_inches is None`` is a
    route that could not answer and is listed under ``missing`` rather than compared.  This is a
    pure function of the route results, so it is the same seam a GUI report and a test both use.

    Returns ``{pairs, max_disagreement_in, answered, missing, within_tolerance, tolerance_in}``.
    ``within_tolerance`` is ``None`` until ``tolerance_in`` is given (the project's accuracy bar is
    the painted-ruler tick, ~0.5 in, but a pc-vs-mono agreement target is a rig decision — ZED
    depth on water is approximate).
    """
    answered = {str(k): float(v["total_inches"])
                for k, v in routes.items()
                if isinstance(v, dict) and v.get("total_inches") is not None}
    missing = sorted(set(str(k) for k in routes) - set(answered))
    pairs: list[dict[str, Any]] = []
    keys = sorted(answered)
    max_d: float | None = None
    for i, ka in enumerate(keys):
        for kb in keys[i + 1:]:
            d = abs(answered[ka] - answered[kb])
            pairs.append({"a": ka, "b": kb, "a_in": answered[ka], "b_in": answered[kb],
                          "disagreement_in": d})
            max_d = d if max_d is None else max(max_d, d)
    within = None
    if tolerance_in is not None and pairs:
        within = all(p["disagreement_in"] <= tolerance_in for p in pairs)
    return {"pairs": pairs, "max_disagreement_in": max_d,
            "answered": sorted(answered), "missing": missing,
            "within_tolerance": within, "tolerance_in": tolerance_in}


def cross_check(pc_result: dict[str, Any], *,
                plane_result: dict[str, Any] | None = None,
                arc3d_result: dict[str, Any] | None = None,
                tolerance_in: float | None = None) -> dict[str, Any]:
    """:func:`agreement_table` over ``"pc"`` plus whichever mono route answered.

    Convenience for the report: the point-cloud route against ``route="plane"`` (the homography)
    and ``route="arc3d"`` (the mono 3-D arc).  ``None`` routes are simply absent from the table and
    appear under ``missing``.
    """
    routes: dict[str, Any] = {"pc": pc_result}
    if plane_result is not None:
        routes["plane"] = plane_result
    if arc3d_result is not None:
        routes["arc3d"] = arc3d_result
    return agreement_table(routes, tolerance_in=tolerance_in)
