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

from . import calib as _calib
from . import intersection as _intersection
from . import lens as _lens
from .homography import PlanePose
from .lens import LensModel, undistort_points
from .surface import WaterPlane


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
) -> dict[str, Any]:
    """Phases 1-6 for one clip: per-frame (X, Y), the chosen entry, and the cross-check.

    ``contact_frame`` is the operator-confirmed water-entry frame (`PLAN.md` D4).  When it is
    ``None`` the result carries the per-frame coordinates and a **candidate** frame from the
    convergence plateau (:func:`dog_diving.pool.intersection.crossing_frame_from_sequence`) but no
    distance — measuring the last detection instead would be exactly the silent-datum failure.

    ``manual_point`` is an ``(x, y)`` in **raw** pixels from a GUI click; it is undistorted here and
    takes precedence over the track, and is tagged ``point_source="manual"`` so provenance survives
    into the report.
    """
    out: dict[str, Any] = {
        "clip_id": clip_id, "n_track_points": 0, "calibration": None,
        "water": None, "pose_ok": False, "route": None,
        "entry": None, "plateau": None, "per_frame": [],
        "cross_check": None, "flags": {}, "notes": [],
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

    if manual_point is not None:
        px = undistort_points(np.asarray(manual_point, dtype=float).reshape(1, 2), lens)[0]
        entry = _intersection.entry_coordinate(pose, water, px, extent=extent,
                                               margin_ft=margin_ft, homography=plane_H)
        entry["frame"] = contact_frame
        entry["point_source"] = "manual"
        entry["point_origin"] = "click"
        out["entry"] = entry
        out["notes"].append("the measurement point is a manual GUI click, tagged 'manual'")
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
    return {
        "compared": True,
        "d3d_in": d3d,
        "d1d_in": d1d,
        "diff_in": diff,
        "tolerance_in": float(tolerance_in),
        "agree": bool(abs(diff) <= float(tolerance_in)),
        "notes": [
            "the 1-D route reduces the scene to a scale along image x; the 3-D route builds a camera "
            "and a plane. They share the mark clicks and nothing else, so agreement is evidence and "
            "disagreement is a finding — neither number should be chosen on looks"
        ],
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
