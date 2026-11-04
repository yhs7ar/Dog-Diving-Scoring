"""End-to-end: a detection track plus a pool calibration -> an (X, Y) entry and a distance.

`pool-translation-plan.txt`'s six phases in the order they run for a clip::

    1  lens      undistort every pixel the track carries          (lens.py)
    2  features  the calibration document already holds them      (calib.py)
    3  transverse  the vanishing points and width prior           (vanishing.py)
    4  homography  pixels <-> feet, and the pose                  (homography.py)
    5  water plane the level, extrapolated where it is occluded   (surface.py)
    6  intersection  the tail-set ray -> (X, Y) on the surface    (intersection.py)

Two independent routes, and the difference between them is the answer's error bar
------------------------------------------------------------------------------
The skill this pipeline is built against requires *"two independent routes must agree"* as the
correctness proof for a calibration.  Both routes now exist and both run:

* the **3-D route** — Phase 6's ray/plane intersection, which is this plan's answer;
* the **1-D route** — the pre-existing :mod:`dog_diving.calibration` mark fit, which is the
  project's earlier answer and is still in the tree.

On the same clip they should agree to a few inches, and when they do not, that disagreement is
reported in ``cross_check`` rather than resolved by picking one.  This is the one place where
keeping the abandoned-looking 1-D code pays for itself: it is not dead weight, it is the only
independent estimate of the same quantity that the project has.

What this module refuses to do
------------------------------
It will not invent a contact frame, a height, or a water level.  With no calibration it returns the
pixel-domain fields and ``None`` for the inches; with no contact frame it refuses to measure the
last detection instead (the *"silently emit a datum"* failure mode the skill names); with an
airborne contact frame it reports the parallax bias instead of correcting it silently.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np

from ..config import TRAJ_PLANE_MAX_RMS_FT, TRAJ_SPACE
from . import calib as _calib
from . import geometry as _geometry
from . import intersection as _intersection
from . import lens as _lens
from .homography import PlanePose
from .lens import LensModel, undistort_points
from .surface import WaterPlane

#: Below this many world points a flight plane cannot be fitted.
MIN_WORLD_POINTS = 3


def track_to_world(
    pose: PlanePose,
    water: WaterPlane,
    track: Sequence[dict],
    *,
    min_points: int = MIN_WORLD_POINTS,
) -> dict[str, Any]:
    """Phase 5's first move: the 2-D track -> 3-D world points on a fitted vertical plane.

    **Single pass by default, as specified**: every point is unprojected to the water height first
    (``Z = z_water`` — the only height a monocular pixel can be placed at without assuming
    anything), a vertical plane is fitted to those ``(X, Y)``, and every ray is then intersected
    with that plane to give the point's **true** height.

    What that buys: ``Z_ft`` is no longer assumed to be the water level for every detection.  A
    tail-set two feet above the water gets a point two feet above the water, and the arc fitted
    through those points can be extrapolated to ``Z = z_water`` instead of to the last pixel that
    happened to survive the splash.  What it costs: the horizontal position now depends on the
    plane, so a flight that genuinely curves across the pool (or a badly fitted line) moves the
    answer.  ``rms_ft`` is that error, measured, and ``flags["plane_wide"]`` says when it exceeds
    ``config.TRAJ_PLANE_MAX_RMS_FT``.

    **The cost is not always small, and this is where that shows.**  Unprojecting an *airborne*
    point to the water plane moves it along its ray by an amount that grows with its height — on the
    project's own shallow synthetic view (camera 5.6 ft up, 17 ft out, water seen at <10 deg) a
    tail-set 3.5 ft up lands about 5 ft further out than it is.  Fitting the flight line through
    those smeared positions therefore fits the *parallax*, not the flight: measured on that scene,
    the single pass put the crossing 110 in from ground truth (vs +65 in for the plain single-pixel
    answer), with a plane RMS of 2.6 ft and fitted heights reaching 5 ft *below* the water.  That is
    why ``rms_ft`` is returned and gated rather than ignored.

    **Iterating this is a fixed point, so it is not offered as a knob.**  Re-fitting the plane from
    the points the previous pass reconstructed cannot move anything: those points lie *on* the plane
    by construction, so the next fit returns the same plane and the same answer.  Measured: passes
    1, 2, 3, 5 and 8 all returned X = 26.451 ft, +110.01 in from truth, with the RMS collapsing to
    0.000 ft on pass 2 — a converged fit of the wrong plane.  A real improvement needs an outside
    constraint (the known free-fall acceleration, or stereo disparity), not another pass of this.

    **The gate is necessary and not sufficient.**  On a steep view (a 60 ft camera) the plane RMS
    drops to 0.095 ft, the gate passes, and the entry is still 73 in from ground truth: the single
    pass *under*-estimates height rather than scattering it, so it can fit one plane beautifully
    through a compressed height profile.  ``flags`` therefore never certifies this route; the entry
    record it produces carries ``arc3d_unverified_bootstrap``, and the two measured numbers live in
    ``test_m30``.

    Returns every row with ``X_ft``/``Y_ft``/``Z_ft`` added (and the pixel columns kept, so the drawn
    arc and the 1-D cross-check still work), plus ``plane``, ``rms_ft`` and per-point
    ``reprojection_px`` — which on this construction is ~0 by definition, and is therefore a check on
    the arithmetic rather than on the physics.
    """
    rows = [dict(r) for r in track if r.get("x") is not None and r.get("y") is not None]
    out: dict[str, Any] = {"status": None, "points": rows, "plane": None, "rms_ft": None,
                           "span_ft": None, "detail": "", "flags": {}, "n_points": len(rows),
                           "n_unprojected": 0}
    if len(rows) < int(min_points):
        out["status"] = "few_points"
        out["detail"] = (f"{len(rows)} track point(s) usable; a flight plane needs at least "
                         f"{int(min_points)}")
        return out

    # pass 1 — every pixel's ray against the water plane: the only height we can name up front
    on_water: list[dict] = []
    for r in rows:
        hit = _geometry.world_from_pixel(pose, (float(r["x"]), float(r["y"])), float(water.z_ft))
        if hit["status"] != "ok":
            continue
        p = np.asarray(hit["point"], dtype=float)
        r["X_water_ft"], r["Y_water_ft"] = float(p[0]), float(p[1])
        on_water.append(r)
    out["n_unprojected"] = len(on_water)
    if len(on_water) < int(min_points):
        out["status"] = "few_rays"
        out["detail"] = (f"only {len(on_water)} of {len(rows)} ray(s) met the water plane at "
                         f"z = {float(water.z_ft):+.2f} ft; the rest are parallel or behind the "
                         f"camera, which is a pose/plane problem, not a track problem")
        return out

    # the plane's horizontal seed: the water-height positions, and nothing else — never whatever
    # world keys a caller's rows happen to carry (a row that already has a true X_ft/Y_ft would
    # otherwise fit its own answer into the plane it is measured with).
    seed = [(r["X_water_ft"], r["Y_water_ft"]) for r in on_water]
    fit = _geometry.vertical_plane_from_points(
        seed, min_points=int(min_points), source="water-height unprojection")
    if fit["status"] != "ok":
        out.update({"status": fit["status"], "detail": fit["detail"]})
        return out
    plane = fit["plane"]
    out.update({"plane": plane, "rms_ft": float(fit["rms_ft"]),
                "span_ft": float(fit["span_ft"]), "plane_detail": fit["detail"]})
    out["flags"]["plane_wide"] = bool(float(fit["rms_ft"]) > float(TRAJ_PLANE_MAX_RMS_FT))

    # the same rays against that plane: the 3-D point, with a height this pass believes in
    kept: list[dict] = []
    for r in rows:
        ray = _geometry.ray_from_pixel(pose, (float(r["x"]), float(r["y"])))
        hit = plane.intersect(ray)
        if hit["status"] != "ok":
            continue
        p = np.asarray(hit["point"], dtype=float)
        r["X_ft"], r["Y_ft"], r["Z_ft"] = float(p[0]), float(p[1]), float(p[2])
        r["height_above_water_ft"] = float(p[2]) - float(water.z_ft)
        r["reprojection_px"] = _geometry.reprojection_error_px(
            pose, p, (float(r["x"]), float(r["y"])))
        kept.append(r)

    out["points"] = kept
    out["n_points"] = len(kept)
    # The check the whole route stands or falls on: the reconstructed point must project back onto
    # the pixel it came from. Any real error here is arithmetic, and it is reported rather than
    # rounded away.
    reps = [float(r["reprojection_px"]) for r in kept
            if r.get("reprojection_px") is not None and np.isfinite(r["reprojection_px"])]
    out["max_reprojection_px"] = max(reps) if reps else None
    if len(kept) < int(min_points):
        out["status"] = "few_points"
        out["detail"] = f"only {len(kept)} point(s) survived the plane intersection"
        return out
    heights = [float(r["height_above_water_ft"]) for r in kept]
    out["height_range_ft"] = [min(heights), max(heights)]
    out["status"] = "ok"
    out["detail"] = (f"{len(kept)} point(s) on a vertical plane (horizontal RMS "
                     f"{float(fit['rms_ft']):.3f} ft, span "
                     f"{float(fit['span_ft']):.2f} ft); fitted height range {min(heights):+.2f} to "
                     f"{max(heights):+.2f} ft above the water")
    if out["flags"]["plane_wide"]:
        out["detail"] += ("; the plane is too wide to trust — the track does not lie in one "
                          "vertical plane, which is what the single pass assumes")
    return out



def undistort_track(track: Iterable[dict], lens: LensModel | None) -> list[dict]:
    """Phase 1 applied to a whole detection track: every ``(x, y)`` moved into the undistorted space.

    Done once, up front, rather than per-frame inside the intersection, for two reasons: it is the
    same transform for every point (so doing it repeatedly is waste), and having one entry point
    means a caller cannot accidentally mix distorted and undistorted pixels in one track — the bug
    that would put every distance out by a position-dependent amount with no visible symptom.
    """
    rows = [dict(r) for r in track if r.get("x") is not None and r.get("y") is not None]
    if not rows:
        return []
    pts = np.array([[float(r["x"]), float(r["y"])] for r in rows], dtype=float)
    und = undistort_points(pts, lens)
    for r, u, raw in zip(rows, und, pts):
        r["x_raw"] = float(raw[0])
        r["y_raw"] = float(raw[1])
        r["x"] = float(u[0])
        r["y"] = float(u[1])
    return rows


def records_to_track(records: Iterable[dict], *, kp_conf: float | None = None,
                     accepted_only: bool = True) -> list[dict]:
    """Normalise detector records into the flat ``(x, y, kp_conf)`` rows the phases consume.

    This exists because of a real, silent integration failure.  ``detect.py`` emits records shaped
    ``{"frame", "tailset": [x, y, conf], "nose": [...], "box_conf", "track_id", "accepted"}``, while
    :func:`undistort_track` selects rows with ``r.get("x") is not None``.  Handed detector records
    directly, it therefore dropped **every** row and returned an empty track — no exception, no
    warning, just ``n_track_points = 0`` and no entry.  On the real ``18-07`` cache (450 records
    present) that is what happened before this function existed.

    So the conversion is explicit and it is done *inside* :func:`translate_track`, not left to
    callers: a shape mismatch here is invisible downstream, and the only reliable place to catch it
    is where the track is first read.

    ``accepted_only`` honours ``detect.py``'s own accept rule (tailset present, keypoint confidence
    above ``kp_conf``, on the dominant track).  Passing ``False`` keeps every record carrying a
    tailset, which is what a QA probe wants and what a distance never does.
    """
    from dog_diving.config import DEFAULT_KP_CONF

    thr = float(DEFAULT_KP_CONF if kp_conf is None else kp_conf)
    rows: list[dict] = []
    n_raw = 0
    n_tail = 0
    for r in records:
        n_raw += 1
        if r.get("x") is not None and r.get("y") is not None:      # already flat
            rows.append(dict(r))
            continue
        ts = r.get("tailset")
        if not ts or ts[0] is None or ts[1] is None:
            continue
        n_tail += 1
        conf = None if ts[2] is None else float(ts[2])
        if accepted_only and (r.get("accepted") is False or conf is None or conf <= thr):
            continue
        rows.append({
            "frame": r.get("frame"),
            "x": float(ts[0]), "y": float(ts[1]),
            "kp_conf": conf,
            "accepted": bool(r.get("accepted", True)),
            "track_id": r.get("track_id"),
            "box_conf": r.get("box_conf"),
        })
    if n_raw and not rows and n_tail == 0:
        raise ValueError(
            f"none of the {n_raw} supplied record(s) carry either flat x/y or a tailset — the shape "
            f"is not one this pipeline understands, and converting it silently would produce an "
            f"empty track that looks like 'the dog was never detected'"
        )
    return rows


def translate_track(
    track: Iterable[dict],
    *,
    doc: dict[str, Any] | None = None,
    contact_frame: int | None = None,
    manual_point: Sequence[float] | None = None,
    height_ft: float = 0.0,
    legacy_calib: dict | None = None,
    clip_id: str | None = None,
    margin_ft: float = 0.0,
    trajectory_space: str | None = None,
) -> dict[str, Any]:
    """Phases 1-6 for one clip: per-frame (X, Y), the chosen entry, and the cross-check.

    ``contact_frame`` is the operator-confirmed water-entry frame (`PLAN.md` D4).  When it is
    ``None`` the result carries the per-frame coordinates and a **candidate** frame from the
    convergence plateau (:func:`dog_diving.pool.intersection.crossing_frame_from_sequence`) but no
    distance — measuring the last detection instead would be exactly the silent-datum failure.

    ``manual_point`` is an ``(x, y)`` in **raw** pixels from a GUI click; it is undistorted here and
    takes precedence over the track, and is tagged ``point_source="manual"`` so provenance survives
    into the report.

    ``trajectory_space="3d"`` runs the geometric route (phases 1-5 of ``intersection-algo.txt``):
    the track is unprojected to world rays (:func:`track_to_world`), a 3-D trajectory is fitted to
    those points, and the entry is where the fitted curve crosses the water plane — not where one
    pixel's ray hits it.  The extra keys are ``world`` (the plane fit and its residuals),
    ``trajectory3d`` (the fit summary) and ``arc_entry`` (the crossing, always computed); when a
    ``contact_frame`` is confirmed, ``arc_entry`` **is** ``entry`` and the single-pixel answer is
    kept beside it as ``entry_pixel`` so the two can be compared in inches.  With no confirmed
    frame the arc crossing is reported as a candidate only, exactly like the plateau — a fitted
    crossing is a model prediction, and the pipeline has never turned one of those into a distance
    on its own.
    """
    space = str(trajectory_space if trajectory_space is not None else TRAJ_SPACE).lower()
    if space not in ("2d", "3d"):
        raise ValueError(f"trajectory_space must be '2d' or '3d', got {space!r}")
    out: dict[str, Any] = {
        "clip_id": clip_id, "n_track_points": 0, "calibration": None,
        "water": None, "pose_ok": False, "route": None,
        "entry": None, "plateau": None, "per_frame": [],
        "cross_check": None, "flags": {}, "notes": [], "trajectory_space": space,
    }
    if doc is None:
        out["notes"].append(
            "no pool calibration supplied, so phases 3-6 cannot run: the 3-D translation needs the "
            "homography (Phase 4) and the water plane (Phase 5), and without them a pixel is only a "
            "ray. The pixel-domain fields below are real; the inches are not computable."
        )
        out["flags"] = {"no_calibration": True}
        return out

    lens = _calib.lens_from_document(doc)
    pose = _calib.pose_from_document(doc)
    water = _calib.water_from_document(doc)
    # A plane calibration (PLAN §15) *is* the water surface, so its mapping is the homography and the
    # decomposed pose is only a by-product.  Answering from the pose there means inheriting the
    # focal prior's error (measured 2.7-7.9 in on the synthetic scene against a few tenths for the
    # homography), so the document says which route to use rather than the caller guessing.
    plane_route = str((doc.get("plane") or {}).get("mapping") or "") == "homography"
    # The DOCUMENT's matrix, deliberately — not water.homography(pose), which is the same map only
    # when the decomposed pose is self-consistent (it is not, under an assumed lens).
    plane_H = (np.asarray(doc["homography"], dtype=float).reshape(3, 3) if plane_route
               and doc.get("homography") is not None else None)
    extent = doc.get("extent") or None
    out["calibration"] = {"setup_id": doc.get("setup_id"), "route": doc.get("route"),
                          "created": doc.get("created")}
    out["water"] = water.as_dict()
    if plane_route:
        out["notes"].append(
            "this calibration's plane IS the water surface (PLAN §15), so the map is the plane "
            "homography: the answer does not depend on the assumed focal length, only on the "
            "corners. The pose route is recorded as a cross-check (homography_agreement_ft).")
    out["route"] = doc.get("route")
    out["pose_ok"] = bool(doc.get("gates", {}).get("W4", {}).get("pass", False))

    rows = undistort_track(records_to_track(track, accepted_only=True), lens)
    out["n_track_points"] = len(rows)
    if not rows:
        out["flags"]["no_track_points"] = True
        out["notes"].append(
            "no accepted tail-set records survived the conversion to track points: every phase "
            "below is a function of that track, so there is nothing to intersect and no entry is "
            "produced. Check the detector's coverage for this clip rather than this calibration."
        )

    # --- Phase 5: unproject the track, then let the fitted curve find the crossing -------------
    # This runs before the entry selection on purpose: the entry selection then *chooses* between
    # the single-pixel answer and the curve's, rather than computing one and patching the other in
    # afterwards.
    arc_traj = None
    arc_entry = None
    if space == "3d" and rows:
        world = track_to_world(pose, water, rows)
        out["world"] = {k: (v.as_dict() if hasattr(v, "as_dict") else v)
                        for k, v in world.items() if k != "points"}
        if world["status"] != "ok":
            out["flags"]["world_failed"] = True
            out["notes"].append(
                f"the 3-D route could not build world points ({world['status']}): "
                f"{world['detail']}. The pixel-domain fields below are unaffected; no 3-D arc was "
                f"fitted, so the entry is the single-pixel one.")
        elif world["flags"].get("plane_wide"):
            # The single pass assumed the track lies in ONE vertical plane. When its own fit says
            # otherwise (RMS over TRAJ_PLANE_MAX_RMS_FT), the assumption is measurably false and the
            # arc built on it would be a confident-looking wrong number. Refuse, and say why.
            out["flags"]["arc3d_plane_too_wide"] = True
            out["notes"].append(
                f"the 3-D route declined to answer: the track fits no single vertical plane "
                f"(horizontal RMS {float(world['rms_ft']):.2f} ft against a "
                f"{TRAJ_PLANE_MAX_RMS_FT:g} ft limit), which is the assumption the whole method "
                f"rests on. Fitted heights ranged {world['height_range_ft'][0]:+.1f} to "
                f"{world['height_range_ft'][1]:+.1f} ft. The entry is the single-pixel one; see "
                f"pipeline.track_to_world for the measured cost of the single pass on a shallow "
                f"camera.")
        else:
            from .. import trajectory as _trajectory    # local: keeps pool/'s import graph unchanged
            arc_traj = _trajectory.build_trajectory(
                world["points"], clip_id=clip_id, space="3d", plane=world["plane"])
            out["trajectory3d"] = {
                "space": arc_traj.space, "n_points": arc_traj.n_points,
                "n_flight_points": len(arc_traj.flight_points),
                "airborne_frame_range": arc_traj.airborne_frame_range,
                "last_arc_frame": arc_traj.last_arc_frame,
                "arc_rms_ft": arc_traj.arc_rms_ft,
                "max_inlier_res_ft": arc_traj.max_inlier_res_ft,
                "plane": arc_traj.plane,
                "coeffs": arc_traj.coeffs,
                "reprojection_max_px": world.get("max_reprojection_px"),
            }
            if arc_traj.coeffs is None:
                out["flags"]["arc3d_no_fit"] = True
                out["notes"].append(
                    "the world points were built but no 3-D arc survived the fit (fewer than three "
                    "in one run, or the plane was too wide to trust); the entry falls back to the "
                    "single-pixel route.")
            else:
                arc_entry = _intersection.entry_coordinate(
                    pose, water, None, extent=extent, margin_ft=margin_ft, homography=plane_H,
                    trajectory=arc_traj)
                arc_entry["point_source"] = "arc3d"
                arc_entry["point_origin"] = "fitted_curve"
                if arc_entry.get("frame") is None and arc_entry.get("crossing_frame") is not None:
                    arc_entry["frame"] = int(round(float(arc_entry["crossing_frame"])))
                out["arc_entry"] = arc_entry
                out["notes"].append(
                    f"3-D route: {len(arc_traj.flight_points)} flight point(s) unprojected onto a "
                    f"vertical plane (horizontal RMS {float(world['rms_ft']):.3f} ft), fitted in "
                    f"world space (RMS "
                    + (f"{arc_traj.arc_rms_ft:.3f} ft" if arc_traj.arc_rms_ft is not None else "n/a")
                    + "); the entry is where that curve crosses the water plane")

    if manual_point is not None:
        px = undistort_points(np.asarray(manual_point, dtype=float).reshape(1, 2), lens)[0]
        entry = _intersection.entry_coordinate(pose, water, px, extent=extent,
                                               margin_ft=margin_ft, homography=plane_H)
        entry["frame"] = contact_frame
        entry["point_source"] = "manual"
        entry["point_origin"] = "click"
        out["entry"] = entry
        out["notes"].append("the measurement point is a manual GUI click, tagged 'manual'")
        if space == "3d":
            out["notes"].append(
                "the 3-D arc route was not applied to this click: a single pixel carries no track, "
                "so there is no curve to cross the water. The click's own ray/plane intersection "
                "is the answer, which is why height_ft still matters here.")
    elif contact_frame is not None:
        match = next((r for r in rows if int(r.get("frame", -1)) == int(contact_frame)), None)
        if match is None:
            out["notes"].append(
                f"the chosen contact frame {contact_frame} has no accepted detection, and measuring "
                f"the last available detection instead would be a silent datum — so no entry was "
                f"produced. Confirm a frame that has a tail-set, or supply a manual point."
            )
            out["flags"]["contact_frame_without_detection"] = True
        else:
            entry = _intersection.entry_coordinate(
                pose, water, (match["x"], match["y"]), extent=extent, margin_ft=margin_ft,
                height_ft=float(height_ft), homography=plane_H)
            entry["frame"] = int(contact_frame)
            entry["point_source"] = "model"
            entry["point_origin"] = "detection"
            entry["kp_conf"] = match.get("kp_conf")
            out["entry"] = entry
    else:
        # no confirmed frame: the 3-D crossing is a CANDIDATE, reported as such (see below), and
        # `entry` stays None.  Same rule as the plateau; a fit does not overrule PLAN.md D4.
        pass

    # --- which answer becomes THE entry --------------------------------------------------------
    # On the 3-D route the curve's crossing answers, and the single-pixel entry is kept beside it:
    # two numbers for the same clip, both recomputable, and the difference between them is the
    # parallax the old route was silently absorbing.  A confirmed contact frame is still required —
    # a fitted crossing is a model prediction and PLAN.md D4 is about who decides that frame.
    if (space == "3d" and arc_entry is not None and contact_frame is not None
            and out.get("entry") is not None and out["entry"].get("distance_in") is not None
            and arc_entry.get("distance_in") is not None):
        pixel_entry = out["entry"]
        out["entry_pixel"] = pixel_entry
        out["entry"] = arc_entry
        diff_in = float(arc_entry["distance_in"]) - float(pixel_entry["distance_in"])
        out["arc_vs_pixel_in"] = diff_in
        out["notes"].append(
            f"the 3-D arc answers: {arc_entry['distance_in']:.2f} in against the single-pixel "
            f"intersection at frame {contact_frame}'s {pixel_entry['distance_in']:.2f} in "
            f"({diff_in:+.2f} in). The difference is the height the old route assumed away — the "
            f"arc puts the tail-set at Z = "
            f"{float(arc_entry.get('Z_ft', 0.0)):+.2f} ft at the crossing, not at the water.")
    elif space == "3d" and arc_entry is not None and arc_entry.get("distance_in") is not None:
        out["notes"].append(
            f"the 3-D curve crosses the water at frame {arc_entry.get('crossing_frame', float('nan')):.1f} "
            f"implying {arc_entry['distance_in']:.2f} in, but no contact frame was confirmed, so this "
            f"is a CANDIDATE (like the plateau), not the distance")

    per_frame = _intersection.trajectory_intersections(pose, water, rows, extent=extent,
                                                       margin_ft=margin_ft,
                                                       homography=plane_H)
    out["per_frame"] = per_frame
    out["plateau"] = _intersection.crossing_frame_from_sequence(per_frame)

    if out["entry"] is None and out["plateau"].get("ok"):
        out["notes"].append(
            f"no contact frame was chosen; the convergence plateau suggests frame "
            f"{out['plateau']['frame']} (last {out['plateau']['plateau_run']} accepted frame(s), "
            f"spread {out['plateau']['plateau_span_in']:.1f} in) as a CANDIDATE. It is not a "
            f"distance until a human confirms it — a plateau is also what a lost detection looks "
            f"like."
        )

    # --- the parallax disclosure (Phase 6's own honest extra) ----------------
    if out["entry"] is not None:
        if abs(float(height_ft)) > 1e-9 and out["entry"].get("x_px") is not None:
            pb = _intersection.parallax_bias(
                pose, water, (out["entry"]["x_px"], out["entry"]["y_px"]),
                height_ft=float(height_ft))
            out["entry"]["parallax"] = pb
            if pb.get("ok"):
                out["notes"].append(
                    f"height {height_ft:.2f} ft was supplied: the literal Phase 6 projection reads "
                    f"{'short' if pb['short'] else 'long'} by {abs(pb['dX_in']):.1f} in against the "
                    f"plumb-corrected entry at ({pb['corrected_X_ft']:.2f}, "
                    f"{pb['corrected_Y_ft']:.2f}) ft"
                )
        else:
            out["notes"].append(
                "the entry was taken with height 0 ft, which is the plan's literal Phase 6 step and "
                "is only valid if the tail-set really is at the water surface on the chosen frame. "
                "Pass height_ft to price the residual parallax."
            )

    # --- the cross-check against the 1-D route -------------------------------
    out["cross_check"] = cross_check_against_1d(out["entry"], legacy_calib, out["per_frame"])
    if out["cross_check"] and out["cross_check"].get("compared"):
        cc = out["cross_check"]
        out["notes"].append(
            f"cross-check against the 1-D route: 3-D {cc['d3d_in']:.2f} in vs 1-D "
            f"{cc['d1d_in']:.2f} in — difference {cc['diff_in']:+.2f} in "
            f"({'AGREE' if cc['agree'] else 'DISAGREE'})"
        )
    return out


def cross_check_against_1d(
    entry: dict | None,
    legacy_calib: dict | None,
    per_frame: Sequence[dict],
    *,
    tolerance_in: float = 6.0,
) -> dict[str, Any]:
    """Compare the 3-D entry distance with what the 1-D mark fit would have said.

    The two routes share the mark clicks and nothing else, which is what makes the comparison
    informative: the 1-D route reduces the scene to a scale along image x, the 3-D route builds a
    camera and a plane.  Agreement within a tick of the painted scale (12 in; the tolerance here is
    the skill's 6 in) is the project's stated proof; disagreement is reported with both numbers so a
    reviewer can see which one moved.

    Returns ``{"compared": False, "reason": ...}`` whenever a side is missing, rather than a number
    that looks like agreement.
    """
    if entry is None or entry.get("distance_in") is None:
        return {"compared": False, "reason": "no 3-D entry to compare"}
    if not legacy_calib:
        return {"compared": False,
                "reason": "no 1-D calibration supplied, so there is no second route to compare "
                          "against"}
    try:
        from .. import calibration as _legacy  # local import: the 1-D route is optional
        x_px = float(entry["x_px"])
        d1d = float(_legacy.distance_inches(x_px, legacy_calib))
    except Exception as exc:  # noqa: BLE001 - a cross-check must never break a measurement
        return {"compared": False, "reason": f"the 1-D route could not be evaluated: {exc}"}
    d3d = float(entry["distance_in"])
    diff = d3d - d1d
    notes = [
        "the 1-D route reduces the scene to a scale along image x; the 3-D route builds a camera "
        "and a plane. They share the mark clicks and nothing else, so agreement is evidence and "
        "disagreement is a finding — neither number should be chosen on looks"
    ]
    if entry.get("route") == "arc3d":
        notes.append(
            "this 3-D distance came from the fitted curve crossing the water plane, and the 1-D "
            "route was fed the pixel that entry REPROJECTS to — not the last detection's pixel. "
            "That is what keeps the legacy 1-D calibration documents comparable without pretending "
            "the two routes measure the same thing: the 1-D route measures the dock scale at a "
            "column, the 3-D route measures a height-independent point on the water.")
    return {
        "compared": True,
        "d3d_in": d3d,
        "d1d_in": d1d,
        "diff_in": diff,
        "tolerance_in": float(tolerance_in),
        "agree": bool(abs(diff) <= float(tolerance_in)),
        "route_3d": entry.get("route"),
        "notes": notes,
    }


def summary_lines(result: dict[str, Any]) -> list[str]:
    """One-screen text for a report or a log line.  Reads the result, never recomputes it."""
    lines: list[str] = []
    e = result.get("entry")
    if e and e.get("distance_in") is not None:
        d = float(e["distance_in"])
        lines.append(f"entry (X, Y) = ({e['X_ft']:.2f}, {e['Y_ft']:.2f}) ft   "
                     f"distance {int(d // 12)}' {d - 12 * int(d // 12):.0f}\"  ({d:.1f} in)")
    else:
        lines.append("entry: n/a")
    if result.get("water"):
        w = result["water"]
        lines.append(f"water level {w['offset_in']:+.1f} in from the marked plane "
                     f"(method {w['method']})")
    if result.get("calibration"):
        c = result["calibration"]
        lines.append(f"calibration {c.get('setup_id')} via route {c.get('route')}")
    p = result.get("plateau")
    if p and p.get("ok"):
        lines.append(f"contact-frame candidate {p['frame']} "
                     f"(plateau {p['plateau_run']} frames, spread {p['plateau_span_in']:.1f} in)")
    cc = result.get("cross_check")
    if cc and cc.get("compared"):
        lines.append(f"1-D cross-check {cc['d1d_in']:.1f} in vs 3-D {cc['d3d_in']:.1f} in "
                     f"({cc['diff_in']:+.1f} in)")
    return lines
