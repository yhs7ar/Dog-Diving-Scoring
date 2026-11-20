"""Phase 6 — tail-set trajectory intersection with the pool surface plane.

`pool-translation-plan.txt`, Phase 6::

    Objective: Calculate the exact (X, Y) impact point of the dog's tail set.
    Implementation:
      * Project the YOLO-detected tail-set coordinates into the calibrated 3D water plane.
      * Output the final intersection coordinates.

Two ways to the same point, on purpose
--------------------------------------
A pixel maps onto the water plane through either

* the **plane homography** — ``(X, Y) ~ H_water^-1 (u, v, 1)``, one matrix inverse, or
* the **ray / plane intersection** — unproject the pixel to a world ray with the camera pose,
  then intersect that ray with the plane.

They are the same map written two ways, which is exactly why both are here: the project's
verification protocol requires *two independent routes that agree*, and this pair agrees to
machine precision (:func:`route_agreement_ft`, measured ~1e-15 ft on synthetic ground truth).  A
homography inverse evaluated outside the calibrated region, or a pose whose translation drifted,
shows up as a disagreement instead of as a quietly wrong score.

The one physical subtlety the plan's literal step hides
------------------------------------------------------
"Project the tail-set coordinates into the water plane" is unambiguous **only at the instant the
tail-set is in the water**, where the ray through the dog and the vertical through the dog
coincide.  At any other instant the ray hits the water somewhere else, and the displacement is not
a rounding error: it is first-order in the tail-set's height, and its size scales with how shallow
the camera's view of the surface is (a fixed camera behind the dock, 45 ft away and 5.6 ft up,
sees the water at under 10 deg, where a foot of height is worth about seven feet of ground — see
the measured numbers in ``test_m12_pool_surface_intersection.py``).

**The sign is not fixed, and it is not "always short".**  The ray from the camera through the dog
continues *past* the dog to reach the water, so the reconstruction lands further along the ray than
the dog's own plumb line.  Whether that means a larger or a smaller measured distance depends on
which side of the dog the camera is:

* camera **behind** the dog (the usual fixed side view, and this project's configuration) → the
  literal step reads **long**;
* camera **beyond** the dog → the literal step reads **short**.

``PLAN.md`` §9 R2's "systematically early → underestimated distance" is a different effect — a
*late* contact frame being unavailable — and the two must not be conflated.  The code therefore
reports the signed difference with an explicit ``short`` flag instead of assuming a direction.

:func:`entry_with_height` implements the general case — project onto the plane at the dog's
*actual* height, then drop the vertical to the water — and the plan's literal step is its
``height_ft = 0`` special case.  :func:`parallax_bias` reports the difference in inches, so a jump
measured on a frame where the dog was still airborne carries a stated bias rather than a silent one.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np

from .geometry import (
    Ray,
    intersect_curve_with_elevation,
    intersect_ray_plane,
    project_point,
    ray_from_pixel,
    world_from_pixel,
)
from .homography import PlanePose, apply_homography_inv
from .lens import LensModel, undistort_points
from .surface import WaterPlane

# Re-exported so callers can reach the ray primitives through this module, which is the
# phase boundary the plan names.
__all__ = [
    "Ray", "ray_from_pixel", "intersect_ray_plane", "world_from_pixel",
    "entry_coordinate", "entry_with_height", "parallax_bias", "route_agreement_ft",
    "trajectory_intersections", "crossing_frame_from_sequence",
]


# ============================================================================
# The 3-D route: a fitted world curve against the water plane
# ============================================================================
def _curve_apex(coeffs: dict) -> float | None:
    """Frame of the fitted ``Z(t)`` maximum, from the coefficients alone.

    Duplicated from :func:`dog_diving.contact.world_apex_frame` on purpose: this module must not
    import ``contact`` (contact imports the pool's geometry, and a cycle there would be invisible
    until something reordered an import).  Five lines of quadratic, twice, beats an import edge
    that turns the pool layer and the frame-decision layer into one knot.
    """
    if not coeffs or coeffs.get("z") is None:
        return None
    a2, a1, _ = (float(v) for v in coeffs["z"])
    if a2 == 0.0:
        return None
    return float(coeffs["t_ref"]) - a1 / (2.0 * a2)


def _entry_from_arc3d(
    pose: PlanePose,
    water: WaterPlane,
    trajectory: Any,
    *,
    t_min: float | None = None,
    homography: Any | None = None,
) -> dict[str, Any]:
    """The fitted 3-D trajectory ∩ ``Z = water.z_ft`` -> an entry record (Phase 4).

    Returns ``{"status", "detail", "fields", "notes", "flags"}``.  ``fields`` is what
    :func:`entry_coordinate` folds in, so both routes emit **one** record shape — a consumer that
    has to branch on the route to find the distance is a consumer that will eventually read the
    wrong key.

    The only input besides the curve is the water's height.  ``t_min`` defaults to the apex, which
    selects the descending branch without any arithmetic about what "descending" looks like in an
    image: a ballistic curve crosses any height twice and the entry is the second one.

    **This arithmetic is exact; the numbers it is fed are not.**  Given a genuinely 3-D curve the
    crossing reproduces ground truth to a rounding error (``test_m30``: 0.000 in).  The curve
    normally comes from :func:`pipeline.track_to_world`'s single-pass bootstrap, whose heights are
    biased by tens of inches — so the record carries ``flags["arc3d_unverified_bootstrap"]`` and a
    note stating the measured bias rather than leaving a precise-looking number to be trusted.
    """
    coeffs = getattr(trajectory, "coeffs", None)
    if not coeffs:
        return {"status": "no_fit", "fields": {}, "notes": [], "flags": {},
                "detail": "the trajectory has no fitted arc (fewer than three points in one run)"}
    if coeffs.get("z") is None:
        return {"status": "not_3d", "fields": {}, "notes": [], "flags": {},
                "detail": "the trajectory was fitted in image space (no world Z axis) — build it "
                          "with space='3d' to use the geometric route"}
    apex = _curve_apex(coeffs)
    tm = float(t_min) if t_min is not None else apex
    cross = intersect_curve_with_elevation(coeffs, float(water.z_ft), t_min=tm)
    if cross["status"] != "ok":
        return {"status": cross["status"], "fields": {}, "notes": [], "flags": {},
                "detail": cross["detail"]}

    P = np.asarray(cross["point"], dtype=float)
    t_cross = float(cross["t"])
    px = project_point(pose, P)
    fields: dict[str, Any] = {
        "x_px": (None if not np.all(np.isfinite(px)) else float(px[0])),
        "y_px": (None if not np.all(np.isfinite(px)) else float(px[1])),
        "X_ft": float(P[0]), "Y_ft": float(P[1]), "Z_ft": float(water.z_ft),
        "distance_ft": float(P[0]), "transverse_ft": float(P[1]),
        "ray_depth_ft": None,
        "homography_agreement_ft": None,
        "route": "arc3d",
        "crossing_frame": t_cross,
        "apex_frame": apex,
        "arc_rms_ft": getattr(trajectory, "arc_rms_ft", None),
    }
    notes = [
        f"the entry is where the fitted world curve crosses the water plane (z = "
        f"{float(water.z_ft):+.2f} ft) at frame {t_cross:.2f} — X = {float(P[0]):.2f} ft, "
        f"Y = {float(P[1]):.2f} ft",
        "x_px/y_px are the REPROJECTION of that 3-D point, not the pixel it came from: on this "
        "route the pixel is an output, which is what makes the 1-D cross-check comparable",
        "UNVERIFIED BOOTSTRAP: this route's heights come from unprojecting each pixel to the water "
        "and fitting one vertical plane through the result (pipeline.track_to_world, single pass). "
        "That is measurably biased — 110 in on the project's shallow synthetic view and 73 in on a "
        "steep one, where the plane gate passes and gives no warning — so read this as a "
        "cross-check on the pixel route, not as a distance. See test_m30 for both numbers.",
    ]
    flags: dict[str, Any] = {"arc3d_unverified_bootstrap": True}
    last_arc = getattr(trajectory, "last_arc_frame", None)
    if last_arc is not None:
        extrap = t_cross - float(last_arc)
        fields["extrap_frames"] = float(extrap)
        if extrap > 0.5:
            flags["arc_extrapolated"] = True
            notes.append(
                f"the crossing is {extrap:+.1f} frames PAST the last accepted arc sample "
                f"({int(last_arc)}): the arc answered where the detector had already lost the dog "
                f"to the splash. That is the model's prediction, not an observation.")
    roots = cross.get("t_all") or []
    if len(roots) > 1:
        fields["crossing_frame_all"] = [float(v) for v in roots]
        notes.append(
            f"the curve crosses z = {float(water.z_ft):+.2f} ft twice (frames "
            f"{', '.join(f'{v:.1f}' for v in roots)}); the descending one was taken")
    if not np.all(np.isfinite(px)):
        notes.append("the entry point does not image in front of the camera, so x_px is None — "
                     "the (X, Y) is still the curve's answer, but the 1-D cross-check cannot run")
    if homography is not None and fields["x_px"] is not None:
        # The plane homography is an independent map from image to pool: pushing the reprojected
        # entry back through it says whether the curve's own 3-D point agrees with the operator's
        # calibration. It is a cross-check, never the answer.
        try:
            Hw = np.asarray(homography, dtype=float).reshape(3, 3)
            XY_h = apply_homography_inv(Hw, np.array([[fields["x_px"], fields["y_px"]]]))[0]
            agree = float(np.linalg.norm(np.asarray(XY_h, dtype=float) - P[:2]))
            fields["homography_agreement_ft"] = agree
            if agree > 0.05:
                notes.append(
                    f"the plane homography places the reprojected entry {agree * 12:.1f} in from "
                    f"where the 3-D curve puts it — the curve and the calibration disagree, and "
                    f"neither is privileged over the other here")
        except Exception as exc:                                      # noqa: BLE001
            notes.append(f"the homography cross-check could not run: {exc}")
    return {"status": "ok", "detail": cross["detail"], "fields": fields,
            "notes": notes, "flags": flags}


# ============================================================================
# The plan's literal step
# ============================================================================
def entry_coordinate(
    pose: PlanePose,
    water: WaterPlane,
    pixel: Sequence[float] | None,
    *,
    extent: dict[str, float] | None = None,
    margin_ft: float = 0.0,
    height_ft: float = 0.0,
    lens: LensModel | None = None,
    homography: Any | None = None,
    trajectory: Any | None = None,
    t_min: float | None = None,
) -> dict[str, Any]:
    """The tail-set pixel -> the ``(X, Y)`` water-entry coordinate.

    ``distance_ft`` is ``X``: the world frame puts the dock edge at ``X = 0`` and the jump
    direction along +``X``, so the distance from the end of the dock *is* the X coordinate.
    ``transverse_ft`` is ``Y`` — a number the previous 1-D pipeline could not produce at all, and
    the one that tells a reviewer whether the dog entered off-axis, which is the whole point of
    the across-pool half of the plan's Phase 3.

    **Pixels in, pixels out, undistorted.**  ``pixel`` must already be in the undistorted space
    ``pose`` lives in.  Passing a raw pixel is a mistake with no visible symptom — the distance just
    comes out wrong by a position-dependent amount — so ``lens`` exists to make it impossible to
    make silently: supply the model and the pixel is undistorted here (measured cost of getting it
    wrong on the synthetic scene: a 31 ft entry reads 30.87 ft, i.e. 1.5 in).

    ``extent`` (from :func:`dog_diving.pool.surface.extent_from_anchors`) gates the result: an
    (X, Y) outside the pool box is flagged in ``flags["outside_extent"]``.  The number is still
    returned, because hiding it would destroy the evidence that something upstream is wrong.

    ``homography`` answers from that matrix instead of the ray/plane intersection — **and it must be
    the document's own** ``doc["homography"]``, not ``water.homography(pose)``.  Those two are the
    same map only when the pose is self-consistent, and with an assumed lens it is not: measured on
    the synthetic scene (placeholder focal 1920 px against a true 2112), answering from the
    pose-derived matrix biased the entry by **3.9-7.9 in**, while the document's matrix was off only
    by the distortion mismatch (1.65 in worst).  Both matrices are *named* "the homography", which is
    exactly how a route switch can look correct and change nothing.

    ``trajectory`` switches this function onto the 3-D route (phases 1-4 of
    ``intersection-algo.txt``): instead of intersecting one pixel's ray with the water, it solves
    ``Z(t) = water.z_ft`` on the fitted world-space curve and reports that point.  Three consequences
    worth stating plainly, because they are the reason the rewrite exists:

    * **``pixel`` stops being an input and becomes an output.**  On this route the entry is a point
      in the pool, so ``x_px``/``y_px`` are filled by *reprojecting* it (:func:`geometry.project_point`).
      That is deliberate and it is what keeps :func:`pipeline.cross_check_against_1d` working — the
      1-D route needs an image column, and the honest one is the column the reconstructed entry
      images to, not the last detection's.
    * **Height needs no assumption.**  ``Z(t)`` is fitted, so the dog's height at every frame is in
      the curve; ``height_ft`` and :func:`entry_with_height` are not used on this route (they stay
      for the single-pixel route, where they are still the only way to price parallax).
    * **The answer is a model prediction, not an observation.**  The crossing is normally *past* the
      last accepted detection — the splash is why the track ends.  ``flags["arc_extrapolated"]``
      and ``crossing_frame`` say how far past, and the caller reports it as a candidate rather than
      as a measured pixel.
    """
    if pixel is not None and lens is not None:
        pixel = np.asarray(undistort_points(np.asarray(pixel, dtype=float).reshape(1, 2), lens),
                           dtype=float)[0]
    out: dict[str, Any] = {
        "x_px": None if pixel is None else float(pixel[0]),
        "y_px": None if pixel is None else float(pixel[1]),
        "X_ft": None, "Y_ft": None, "Z_ft": None,
        "distance_ft": None, "distance_in": None,
        "transverse_ft": None, "transverse_in": None,
        "ray_depth_ft": None, "water_z_ft": float(water.z_ft),
        "height_assumed_ft": float(height_ft),
        "homography_agreement_ft": None,
        "flags": {}, "notes": [],
    }
    if pixel is None and trajectory is None:
        out["notes"].append("no pixel supplied — there is no point to intersect")
        out["flags"] = {"no_point": True}
        return out

    # --- the 3-D route: solve the fitted world curve against the water plane -------------------
    arc = (_entry_from_arc3d(pose, water, trajectory, t_min=t_min, homography=homography)
           if trajectory is not None else None)
    if arc is not None and arc["status"] == "ok":
        out.update(arc.pop("fields"))
        out["notes"].extend(arc.pop("notes", []))
        out["flags"].update(arc.pop("flags", {}))
    elif pixel is None:
        # A trajectory was supplied and could not answer, and there is no pixel to fall back on.
        # Reporting the reason is the whole job; inventing a number would not be.
        why = arc["detail"] if arc is not None else "no trajectory and no pixel supplied"
        out["notes"].append(f"the 3-D route could not produce an entry: {why}")
        out["flags"] = {"arc_failed" if arc is not None else "no_point": True}
        return out
    else:
        if arc is not None:
            out["notes"].append(
                f"the 3-D route did not answer ({arc['detail']}), so this entry is the "
                f"single-pixel ray/plane intersection instead — a different question, and the "
                f"one that needs height_ft to be honest about parallax")

        # --- both routes, always computed: they are the same map only if the pose is consistent ----
        prefer_homography = homography is not None
        Hw = (np.asarray(homography, dtype=float).reshape(3, 3) if prefer_homography
              else water.homography(pose))
        XY_h = apply_homography_inv(Hw, np.array([[float(pixel[0]), float(pixel[1])]]))[0]
        hit = world_from_pixel(pose, pixel, water.z_ft)

        if hit["status"] != "ok":
            if not prefer_homography:
                out["notes"].append(f"ray/plane intersection failed: {hit['status']} — {hit['detail']}")
                out["flags"] = {"intersection_failed": True}
                return out
            out.update({"X_ft": float(XY_h[0]), "Y_ft": float(XY_h[1]), "Z_ft": float(water.z_ft),
                        "distance_ft": float(XY_h[0]), "transverse_ft": float(XY_h[1]),
                        "route": "homography"})
            out["notes"].append(
                f"the pose route failed ({hit['status']}) so the plane homography answered; that is "
                f"sound only because this calibration's plane *is* the water surface")

        else:
            P = np.asarray(hit["point"], dtype=float)
            agree = float(np.linalg.norm(XY_h - P[:2]))
            out["homography_agreement_ft"] = agree
            if prefer_homography:
                chosen, z = XY_h[:2], float(water.z_ft)
            else:
                chosen, z = P[:2], float(P[2])
            out.update({"X_ft": float(chosen[0]), "Y_ft": float(chosen[1]), "Z_ft": z,
                        "distance_ft": float(chosen[0]), "transverse_ft": float(chosen[1]),
                        "ray_depth_ft": float(hit["s"]),
                        "route": "homography" if prefer_homography else "ray_plane"})
            if agree > 1e-6:
                out["notes"].append(
                    f"the homography route and the ray/plane route disagree by {agree:.3e} ft "
                    f"({agree * 12:.3e} in); the answer came from "
                    f"{'the homography' if prefer_homography else 'the ray/plane intersection'}. They "
                    f"are the same map, so a disagreement means the pose is not self-consistent — with "
                    f"an assumed lens it is not, and on the synthetic scene that costs the ray/plane "
                    f"route several inches while the homography barely moves."
                )

    out["distance_in"] = float(out["distance_ft"]) * 12.0
    out["transverse_in"] = float(out["transverse_ft"]) * 12.0

    # --- parallax: is the literal step valid on this frame? -------------------
    # The 3-D route does not need this question answered: Z(t) is fitted, so the height of the
    # dog at the crossing frame is in the curve rather than assumed.  Saying so is better than
    # silently skipping a correction the caller asked for.
    if abs(float(height_ft)) > 1e-9 and out.get("route") == "arc3d":
        out["notes"].append(
            f"height {height_ft:.2f} ft was supplied but not applied: the 3-D route reconstructs "
            f"the height from the fitted Z(t) (the crossing is at z = {out['Z_ft']:+.2f} ft), so "
            f"there is no parallax left to correct. Pass a trajectory=None call to price it")
    elif abs(float(height_ft)) > 1e-9 and pixel is not None:
        corr = entry_with_height(pose, water, pixel, height_ft=float(height_ft))
        if corr.get("status") == "ok":
            out["parallax_correction"] = corr
            out["notes"].append(
                f"height {height_ft:.2f} ft supplied: the literal projection is "
                f"{corr['shift_in']:.1f} in from the plumb-corrected entry at "
                f"({corr['X_ft']:.2f}, {corr['Y_ft']:.2f}) ft"
            )

    # --- the extent gate -----------------------------------------------------
    # A bounding box is the only thing this gate can test, and a dict without any of the four bounds
    # makes `within_extent` return True for every point.  That is how this gate previously reported a
    # confident `outside_extent: False` for a point 2.3 ft outside the pool, so the absence of a bbox
    # is now reported as *not checked* instead of as a pass.
    BBOX_KEYS = ("x_min", "x_max", "y_min", "y_max")
    if extent and any(k in extent for k in BBOX_KEYS):
        inside = WaterPlane.within_extent([(out["X_ft"], out["Y_ft"])], extent, margin_ft)[0]
        out["flags"]["outside_extent"] = bool(not inside)
        if not inside:
            out["notes"].append(
                f"the entry (X, Y)=({out['X_ft']:.2f}, {out['Y_ft']:.2f}) ft falls outside the "
                f"pool extent {extent} — arithmetically fine, physically impossible; check the "
                f"calibration and the contact frame"
            )
    else:
        out["flags"]["extent_not_checked"] = True
        out["notes"].append(
            "the extent gate did not run: the calibration carries no pool bounding box (x_min/x_max/"
            "y_min/y_max), so nothing could be tested. The (X, Y) above is arithmetically valid and "
            "physically unchecked — it could be in the crowd. Supply the pool's corners or its long "
            "edges as anchors to enable the check."
        )
    out["flags"]["negative_distance"] = bool(float(out["distance_ft"]) < 0.0)
    if out["flags"]["negative_distance"]:
        out["notes"].append(
            f"the entry is {abs(out['distance_in']):.1f} in on the DOCK side of the dock edge, so "
            f"the distance is negative — the datum or the point is wrong (a measured distance "
            f"from the lip must come out positive)"
        )
    return out


# ============================================================================
# The height-aware version (and the parallax it reveals)
# ============================================================================
def entry_with_height(
    pose: PlanePose,
    water: WaterPlane,
    pixel: Sequence[float],
    *,
    height_ft: float,
) -> dict[str, Any]:
    """Entry coordinate for a tail-set observed at ``height_ft`` above the water surface.

    Two steps, both exact:

    1. the pixel's ray is intersected with the plane at ``water.z_ft + height_ft`` — that is
       where the dog actually is;
    2. the entry point is that point's **vertical plumb line** onto the water plane, i.e. keep
       ``(X, Y)`` and drop ``Z``.

    With ``height_ft = 0`` this reduces exactly to the plan's Phase 6 step (the dog is in the
    water, so ray and plumb coincide), which is why the two are not competing implementations:
    this one *contains* it.
    """
    z = float(water.z_ft) + float(height_ft)
    hit = world_from_pixel(pose, pixel, z)
    if hit["status"] != "ok":
        return {"status": hit["status"], "detail": hit["detail"], "X_ft": None, "Y_ft": None,
                "shift_in": None}
    P = np.asarray(hit["point"], dtype=float)
    lit = entry_coordinate(pose, water, pixel, height_ft=0.0)
    if lit.get("X_ft") is None:
        return {"status": "no_literal", "detail": "the literal projection failed",
                "X_ft": float(P[0]), "Y_ft": float(P[1]), "shift_in": None}
    shift = float(np.hypot(P[0] - lit["X_ft"], P[1] - lit["Y_ft"])) * 12.0
    return {"status": "ok", "detail": "", "X_ft": float(P[0]), "Y_ft": float(P[1]),
            "Z_ft": float(z), "shift_in": shift,
            "literal_X_ft": lit["X_ft"], "literal_Y_ft": lit["Y_ft"]}


def parallax_bias(
    pose: PlanePose,
    water: WaterPlane,
    pixel: Sequence[float],
    *,
    height_ft: float,
) -> dict[str, Any]:
    """How wrong the literal Phase 6 step is for a tail-set observed at ``height_ft``.

    Reported in inches along the measurement axis.  *Short* is the dangerous direction: a short
    entry is a recorded distance that is too small, which is risk R2's failure mode.  This is the
    number that justifies either confirming the contact frame with the operator or supplying the
    height, and it is why this module does not simply stop at the plan's literal step.
    """
    corr = entry_with_height(pose, water, pixel, height_ft=height_ft)
    lit = entry_coordinate(pose, water, pixel, height_ft=0.0)
    if corr.get("X_ft") is None or lit.get("X_ft") is None:
        return {"ok": False, "height_ft": float(height_ft),
                "detail": corr.get("detail") or "no intersection"}
    dX = float(lit["X_ft"]) - float(corr["X_ft"])
    dY = float(lit["Y_ft"]) - float(corr["Y_ft"])
    return {
        "ok": True,
        "height_ft": float(height_ft),
        "literal_X_ft": float(lit["X_ft"]), "corrected_X_ft": float(corr["X_ft"]),
        "literal_Y_ft": float(lit["Y_ft"]), "corrected_Y_ft": float(corr["Y_ft"]),
        "dX_in": dX * 12.0, "dY_in": dY * 12.0,
        "magnitude_in": float(np.hypot(dX, dY)) * 12.0,
        "short": bool(dX < 0.0),
        "notes": [
            f"a tail-set {height_ft:.2f} ft above the water reconstructs "
            f"{'short' if dX < 0 else 'long'} by {abs(dX) * 12.0:.1f} in on the measurement axis. "
            f"The ray continues past the dog to reach the water, so the sign is set by whether "
            f"the camera is behind the dog (long) or beyond it (short) — the plan's literal step "
            f"is only valid at zero height, and the bias is first-order in the height"
        ],
    }


def route_agreement_ft(pose: PlanePose, water: WaterPlane, pixels: Any) -> dict[str, Any]:
    """Max disagreement between the homography route and the ray/plane route, in feet.

    The project's own "two independent routes must agree" protocol, reduced to one call.  Anything
    above float noise means the pose, the plane height and the plane homography have drifted
    apart, and no distance from this calibration should be quoted until it is fixed.
    """
    pts = np.asarray(pixels, dtype=float).reshape(-1, 2)
    Hw = water.homography(pose)
    xy_h = apply_homography_inv(Hw, pts)
    diffs: list[float] = []
    for p, xy in zip(pts, xy_h):
        hit = world_from_pixel(pose, p, water.z_ft)
        if hit["status"] != "ok":
            continue
        diffs.append(float(np.linalg.norm(np.asarray(hit["point"])[:2] - xy)))
    return {
        "n": len(diffs),
        "max_ft": float(max(diffs)) if diffs else None,
        "max_in": float(max(diffs) * 12.0) if diffs else None,
        "ok": bool(diffs and max(diffs) < 1e-6),
    }


# ============================================================================
# A whole detection track
# ============================================================================
def trajectory_intersections(
    pose: PlanePose,
    water: WaterPlane,
    track: Iterable[dict],
    *,
    extent: dict[str, float] | None = None,
    margin_ft: float = 0.0,
    lens: LensModel | None = None,
    homography: Any | None = None,
) -> list[dict[str, Any]]:
    """Per-frame entry coordinates for a tail-set track.

    ``track`` is any iterable of records carrying ``frame``, ``x`` and ``y`` — ``detect.py``'s
    per-frame record shape and ``trajectory.assemble``'s point shape both qualify.  Pass ``lens``
    when the records hold **raw** pixels and the undistortion should happen here; otherwise the
    pixels are taken to be undistorted already (which is what
    :func:`dog_diving.pool.pipeline.undistort_track` guarantees).

    Each frame is treated at ``height_ft = 0`` (the plan's literal step), because a detection track
    carries no height: one view gives one ray per frame, and the height along it is the missing
    degree of freedom.

    So the reconstructed ``X`` as a function of frame is *the sequence of hypothetical entry
    distances if the tail-set were already at the water*.  It converges to the true entry as the
    dog descends — which is what makes the contact frame's value the one that means anything, and
    also what makes the earlier frames useful, since the sequence shows the parallax closing.
    """
    out: list[dict[str, Any]] = []
    for rec in track:
        if rec.get("x") is None or rec.get("y") is None:
            continue
        e = entry_coordinate(pose, water, (float(rec["x"]), float(rec["y"])),
                             extent=extent, margin_ft=margin_ft, lens=lens,
                             homography=homography)
        e["frame"] = rec.get("frame")
        e["kp_conf"] = rec.get("kp_conf")
        out.append(e)
    return out


def crossing_frame_from_sequence(
    entries: Sequence[dict[str, Any]],
    *,
    plateau_tol_ft: float = 0.15,
) -> dict[str, Any]:
    """Estimate the entry frame from the reconstructed-distance sequence alone.

    As the dog falls the parallax closes, so the reconstructed ``distance_ft`` **plateaus** at the
    true entry distance.  The end of the final plateau is therefore a data-driven candidate for
    the contact frame, obtained without any splash detection.

    It is offered as an extra *candidate*, never as a replacement for the operator's confirmation
    (`PLAN.md` D4), because a plateau is also produced by the detection simply being lost — which
    is why the caller is given the run length and the plateau's own spread, not just a frame
    number.
    """
    rows = [e for e in entries if e.get("distance_ft") is not None and e.get("frame") is not None]
    if len(rows) < 3:
        return {"ok": False, "frame": None,
                "detail": f"only {len(rows)} usable frame(s) — a plateau needs >= 3"}
    rows = sorted(rows, key=lambda r: int(r["frame"]))
    xs = np.array([float(r["distance_ft"]) for r in rows], dtype=float)
    frames = np.array([int(r["frame"]) for r in rows], dtype=int)
    last = float(xs[-1])
    i = len(xs) - 1
    while i > 0 and abs(float(xs[i - 1]) - last) <= float(plateau_tol_ft):
        i -= 1
    run = len(xs) - i
    span = float(np.ptp(xs[i:])) if run > 1 else 0.0
    return {
        "ok": True,
        "frame": int(frames[-1]),
        "plateau_start_frame": int(frames[i]),
        "plateau_run": int(run),
        "plateau_span_ft": span,
        "plateau_span_in": span * 12.0,
        "distance_ft": last,
        "detail": f"the reconstructed distance holds within {plateau_tol_ft:g} ft over the last "
                  f"{run} accepted frame(s) — a plateau is consistent with the water entry, but "
                  f"also with the detection simply being lost, so it is a candidate and not a "
                  f"confirmed contact frame",
    }
