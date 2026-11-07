"""The pool calibration document — schema, route selection, gates, persistence.

This is the Phase 2/4 boundary: it takes an operator's clicks, decides **which** homography route
they can support, builds the pose and the water plane, runs the gates, and writes a single JSON
document that every later phase reads.  It is deliberately the only place that knows how to turn a
click set into geometry, so a second, subtly different construction cannot appear elsewhere.

The schema is versioned and named ``dog_diving_pool_calibration/v1``, which serves one more purpose
than tidiness: the pre-existing ``data/calib/*.json`` files are the **1-D** mark schema, and one of
them on this machine fails its own dock-edge gate by 167 in (``HANDOFF.md`` open item 23).  A
loader that refuses an unknown schema means that document cannot silently become a 3-D datum.

Routes, in the order the module prefers them
--------------------------------------------
``A_correspondences``
    >= 4 points with known ``(X, Y)``.  Best conditioned; needs data this corpus does not have yet.
``A_prime_vps_points``
    two orthogonal vanishing points plus >= 2 known points, which may be **collinear** — the whole
    point of supplying directions separately.  This is the plan's Phase 3 feeding Phase 4.
``B_row_vp_edge_width``
    the marked row + the transverse vanishing point + the far pool edge + the width prior.  This is
    the plan's Phase 3/4 exactly as written, and the route the current corpus can actually feed.

The gates are named ``W1..W7`` rather than ``G1..G5`` on purpose.  ``G1``–``G5`` in
:mod:`dog_diving.calibration` test a 1-D scale fit (strictly increasing mark x, a dock-edge-vs-0-ft
agreement, a residual RMS).  Reusing those names for tests of a different model would make the two
documents look comparable when they are not.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np

from .. import config as _config
from . import homography as _homography
from . import lens as _lens
from . import surface as _surface
from .homography import PlanePose
from .lens import LensModel, undistort_points
from .vanishing import VanishingPoint, vanish

SCHEMA = "dog_diving_pool_calibration/v1"

Route = Literal["A_correspondences", "A_prime_vps_points", "B_row_vp_edge_width"]

# Route-precedence order; the first route whose inputs exist and validate is used unless the
# caller names one explicitly.

ROUTE_LABEL = {
    "A_correspondences": "A — direct correspondences with known (X, Y)",
    "A_prime_vps_points": "A' — two orthogonal vanishing points + known points",
    "B_row_vp_edge_width": "B — marked row + transverse VP + far edge + width prior",
}


# ============================================================================
# Route selection
# ============================================================================
def _correspondence_array(correspondences: Sequence | None) -> np.ndarray:
    """Normalise route A's input to an ``(N, 4)`` array of ``(X_ft, Y_ft, x_px, y_px)``.

    Accepts dicts (the documented click shape) or bare sequences.  Done in one place because
    ``np.asarray`` on a list of dicts produces an object array, and the first version of
    :func:`choose_route` tried to ``float()`` it — a ``TypeError`` from inside a route-selection
    helper, which is a poor way to tell an operator that their clicks were fine.
    """
    rows: list[list[float]] = []
    for c in correspondences or []:
        if isinstance(c, dict):
            rows.append([float(c["X_ft"]), float(c["Y_ft"]), float(c["x_px"]), float(c["y_px"])])
        else:
            v = np.asarray(c, dtype=float).ravel()
            if v.size < 4:
                raise ValueError(f"a correspondence needs (X_ft, Y_ft, x_px, y_px); got {c!r}")
            rows.append([float(v[0]), float(v[1]), float(v[2]), float(v[3])])
    return np.asarray(rows, dtype=float).reshape(-1, 4)


def choose_route(*, correspondences: Sequence | None, vps: Sequence[Any] | None,
                 marks: Sequence[dict] | None, edge_line: Sequence[float] | None) -> str:
    """Pick the best route the supplied inputs can support, and say why not the others.

    Preference order is accuracy-of-conditioning, not convenience: real (X, Y) points beat
    directions-plus-two-points, which beats a width prior.  The check for route A is not just the
    count — collinear points are refused here for the same reason
    :func:`dog_diving.pool.homography.assert_not_collinear` refuses them inside the DLT.
    """
    reasons: list[str] = []
    corr = _correspondence_array(correspondences)
    if corr.shape[0] >= 4:
        try:
            _homography.assert_not_collinear(corr[:, :2], role="correspondence", min_perp=0.25)
            return "A_correspondences"
        except Exception as exc:  # noqa: BLE001
            reasons.append(f"A refused: {exc}")
    else:
        reasons.append(f"A needs >= 4 known (X, Y) points, got {corr.shape[0]}")

    if vps is not None and len(list(vps)) >= 2 and marks and len(marks) >= 2:
        usable = [v for v in vps if getattr(v, "v", None) is not None]
        if len(usable) >= 2:
            return "A_prime_vps_points"
        reasons.append("A' refused: fewer than two usable vanishing points")
    else:
        reasons.append("A' needs two vanishing points and >= 2 known points")

    if marks and len(marks) >= 3 and edge_line is not None and vps and len(list(vps)) >= 1:
        return "B_row_vp_edge_width"
    reasons.append("B needs >= 3 marks on one row, a transverse vanishing point and the far edge "
                   "line — a width prior alone cannot fix the transverse scale")

    raise ValueError("no homography route is supported by the supplied clicks: " + "; ".join(reasons))


# ============================================================================
# Building the document
# ============================================================================
def build(
    setup_id: str,
    *,
    clicks: dict[str, Any],
    lens: LensModel | None = None,
    route: str | None = None,
    water: dict[str, Any] | None = None,
    width_ft: float | None = None,
    marks_on_edge: str | None = None,
    extent: dict[str, float] | None = None,
    model_sha: str = "",
    notes: str = "",
    image_size: tuple[int, int] | None = None,
    created: str | None = None,
) -> dict[str, Any]:
    """Assemble a pool calibration from a set of clicks.

    ``clicks`` is the operator's raw input, in **raw pixel** coordinates (undistortion happens
    here, once):

    ``marks``
        ``[{"ft": 0, "x_px": .., "y_px": ..}, ...]`` — the pool's painted distance marks, all on one
        row.  Their ``ft`` values are the metric anchor: without at least one real length the whole
        document is scale-free.
    ``vp_transverse`` / ``vp_along``
        each ``{"lines": [[x1,y1,x2,y2], ...]}`` — sampled straight edges of a parallel family, one
        family running across the pool and one along it.  Vanishing points are *derived* from the
        lines rather than being supplied as points, because computing them here is the only way to
        also get their residual and their conditioning.
    ``edge_line``
        ``[x1,y1,x2,y2]`` — the long pool edge the marks are **not** on (route B's transverse
        extent).
    ``correspondences``
        ``[{"X_ft": .., "Y_ft": .., "x_px": .., "y_px": ..}, ...]`` — route A's known points.
    ``rim``
        ``{"top": [x, y], "contact": [x, y]}`` — the rim/coping pair, used for the water level via
        its known vertical wall.
    ``wall``
        ``{"axis": "y", "value": 0.0}`` — which vertical plane the water's contact lies on.  Route
        B derives it from ``edge_line`` when it is not given.

    ``water`` selects the water-level method: ``{"method": "wall_contact"|"rim_pair"|
    "known_freeboard", ...}``.  When omitted the rim pair is used if the clicks contain one,
    otherwise ``known_freeboard`` with ``freeboard_in``.
    """
    clicks = dict(clicks or {})
    lens = lens or _lens.placeholder_intrinsics(image_size or (1920, 1080))
    if image_size is None:
        image_size = lens.image_size

    marks_raw = [dict(m) for m in (clicks.get("marks") or [])]
    if not marks_raw:
        raise ValueError("a pool calibration needs the pool's distance marks: they are the only "
                         "metric anchor in the whole document")
    # The config surface is the declared default for the *venue*, not a per-call argument, so an
    # omitted value comes from it rather than from a literal repeated at the call site.  (Nine of the
    # ten POOL_* knobs were referenced by nothing when this was first written; a test now fails if a
    # knob is added and not consumed.)
    if width_ft is None:
        width_ft = float(_config.POOL_WIDTH_FT)
    if marks_on_edge is None:
        marks_on_edge = str(_config.POOL_MARKS_ON_EDGE)

    marks = _undistort_mark_clicks(marks_raw, lens)

    # Line clicks have to be undistorted too, and this was a real gap: marks, the rim pair and the
    # route-A correspondences were undistorted while the *lines* were not, so an operator clicking a
    # raw frame got vanishing points and a wall derived from distorted pixels. Undistorting the
    # endpoints is the right thing because a straight world edge stays straight under a projective
    # map — the two endpoints are all that is needed to carry the line.
    vps_by_role: dict[str, VanishingPoint] = {}
    for key in ("vp_transverse", "vp_along"):
        spec = clicks.get(key)
        if not spec:
            continue
        raw_lines = (spec.get("lines") if isinstance(spec, dict) else spec) or []
        lines = _undistort_segments(raw_lines, lens)
        try:
            vps_by_role[key] = vanish(
                [_seg_to_line(ln) for ln in lines], image_size=tuple(image_size),
                principal_point=lens.principal_point)
        except Exception:  # noqa: BLE001 - a missing VP just removes a route
            continue
    vps = list(vps_by_role.values())
    vp_transverse = vps_by_role.get("vp_transverse")
    vp_along = vps_by_role.get("vp_along")

    edge_line_raw = clicks.get("edge_line")
    edge_line = None if edge_line_raw is None else _undistort_segments([edge_line_raw], lens)[0]
    correspondences = clicks.get("correspondences")

    chosen = route or choose_route(correspondences=correspondences, vps=vps, marks=marks,
                                   edge_line=edge_line)
    anchor_notes: list[str] = []

    if chosen == "A_correspondences":
        corr = _correspondence_array(correspondences)
        img = undistort_points(corr[:, 2:4], lens)
        H, hinfo = _homography.homography_from_correspondences(corr[:, :2], img, robust=True)
        pose = _homography.decompose_plane_homography(H, lens.K)
        anchor_notes.append("metric anchor: points with known real-world (X, Y) in feet")
    elif chosen == "A_prime_vps_points":
        if vp_transverse is None or vp_along is None:
            raise ValueError("route A' needs both a transverse and an along-pool vanishing point")
        W = np.array([[float(m["ft"]), 0.0] for m in marks], dtype=float)
        I = np.array([[float(m["x_px"]), float(m["y_px"])] for m in marks], dtype=float)
        R, t, ainfo = _homography.pose_from_vps_and_points(
            lens.K, vp_along.v, vp_transverse.v, W, I)
        H = _homography.homography_from_pose(lens.K, R, t)
        pose = _homography.decompose_plane_homography(H, lens.K)
        hinfo = ainfo
        anchor_notes.append("metric anchor: the mark row's own foot spacing")
        if ainfo.get("notes"):
            anchor_notes.extend(ainfo["notes"])
    else:
        if vp_transverse is None:
            raise ValueError("route B needs the transverse vanishing point")
        H, hinfo = _homography.homography_from_row_and_edge(
            marks, vp_transverse=vp_transverse.v, edge_image_line=_seg_to_line_or_line(edge_line),
            width_ft=float(width_ft), marks_on_edge=str(marks_on_edge),
            vp_transverse_finite=bool(vp_transverse.finite))
        pose = _homography.decompose_plane_homography(H, lens.K)
        anchor_notes.append(f"transverse anchor: {width_ft:g} ft pool-width PRIOR against the "
                            f"observed far edge")
        anchor_notes.extend(hinfo.get("notes") or [])

    # --- up-axis sign, from the rim pair when there is one -------------------
    rim = clicks.get("rim")
    z_detail: dict[str, Any] = {"rule": "not resolved: no rim pair was clicked"}
    z_sign = 1.0
    if rim and rim.get("top") and rim.get("contact"):
        lower = undistort_points(np.asarray(rim["contact"], dtype=float).reshape(1, 2), lens)[0]
        upper = undistort_points(np.asarray(rim["top"], dtype=float).reshape(1, 2), lens)[0]
        z_sign, z_detail = pose.resolve_up(lower, upper)
        pose = PlanePose(K=pose.K, R=pose.R, t=pose.t, H=pose.H, z_sign=z_sign,
                         info=pose.info)
    else:
        anchor_notes.append(
            "the up-axis sign could not be resolved from a rim pair, so +Z is the decomposition's "
            "default. If the water comes out ABOVE the marked plane on a venue where it should be "
            "below, this is the reason — click a rim/coping pair and rebuild."
        )

    # --- the water plane -----------------------------------------------------
    wall = clicks.get("wall")
    if wall is None and edge_line is not None:
        try:
            wall = _surface.wall_from_edge_line(pose, _seg_to_line_or_line(edge_line))
        except Exception:  # noqa: BLE001 - the wall is optional
            wall = None
    # `water` may be passed as an argument or carried inside `clicks`; supporting both means an
    # operator's saved click set is self-contained, and a caller who passes only `clicks` cannot
    # silently get the default method because the key was in the wrong place.
    if water is None and clicks.get("water"):
        water = clicks["water"]
    water_plane = _build_water(pose, lens, rim, wall, water)

    # --- extent --------------------------------------------------------------
    # ``extent`` is a bounding box and *only* a bounding box.  It used to be overloaded with the
    # width-observation report, and the cost was a gate that could not fail: ``within_extent`` reads
    # x_min/x_max/y_min/y_max with +-inf defaults, so a report dict holding none of those keys made
    # it return ``True`` for *every* point — while ``flags["outside_extent"]`` read ``False``, i.e. a
    # confident pass.  Measured on the synthetic document: an entry at ``(3.54, -2.28)``, which is
    # 2.3 ft outside a 16 ft pool, was reported as inside.  A QA gate that always passes is worse
    # than no gate, because it gets cited as evidence.
    ext = dict(extent or {})
    if not ext and clicks.get("extent_anchors"):
        ext = _surface.extent_from_anchors(pose, clicks["extent_anchors"],
                                           fallback_z=water_plane.z_ft)

    y_row = 0.0 if str(marks_on_edge) == "near" else float(width_ft)
    width_obs: dict[str, Any] = {}
    if edge_line is not None:
        try:
            obs = _surface.extent_from_lines(pose, [_seg_to_line_or_line(edge_line)])
            y_other = obs.get("y_max", obs.get("y_min"))
            if y_other is not None:
                y_other = float(y_other)
                # The observed width is the one *independent* check on the width prior for routes
                # A/A', and it is implied by construction for route B (which uses the prior as an
                # input).  Recording which one it is stops W6 reporting a tautology as evidence.
                width_independent = chosen in ("A_correspondences", "A_prime_vps_points")
                width_obs = {
                    "width_observed_ft": abs(y_other - y_row),
                    "width_observed_independent": bool(width_independent),
                    "width_observed_note": (
                        "measured from the clicked far edge against the marked row"
                        if width_independent else
                        "implied by construction: route B uses the width prior as an input, so this "
                        "is a self-consistency value and NOT independent evidence for the prior"),
                }
                if obs.get("max_line_spread_ft") is not None:
                    width_obs["max_line_spread_ft"] = float(obs["max_line_spread_ft"])
                # A clicked far edge is a real bound, so fold it — and the marked row — into the
                # bbox.  Without this the common click set (marks on one edge, the other edge
                # clicked) produced no extent at all and the gate stayed vacuous.
                ext.setdefault("y_min", min(y_row, y_other))
                ext.setdefault("y_max", max(y_row, y_other))
        except Exception:  # noqa: BLE001 - the width observation is a QA aid
            pass

    # The pool's own dimensions, recorded rather than left to a caller's default.  ``along_ft`` is
    # the marks' span and is labelled as such: it is where the marks stop, not where the pool does.
    mark_fts = [float(m["ft"]) for m in marks if m.get("ft") is not None]
    pool_info: dict[str, Any] = {
        "width_ft": float(width_ft),
        "marks_on_edge": str(marks_on_edge),
        "y_min_ft": float(y_row),
        "along_ft": float(max(mark_fts)) if mark_fts else float(width_ft),
        "along_source": ("the marks' own span" if mark_fts else "fallback: no marks in the document"),
    }

    doc: dict[str, Any] = {
        "schema": SCHEMA,
        "setup_id": str(setup_id),
        "created": created or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "image_size": [int(image_size[0]), int(image_size[1])],
        "model_sha": model_sha,
        "notes": notes,
        "route": chosen,
        "route_label": ROUTE_LABEL.get(chosen, chosen),
        "world_frame": {
            "origin": "0 ft mark on the dock edge",
            "x_axis": "+X = jump direction, feet from the dock edge",
            "y_axis": "+Y = across the pool, feet",
            "z_axis": "+Z = up; the marks' plane is Z = 0",
            "pixel_space": "undistorted",
            "units": "feet",
        },
        "lens": lens.as_dict(),
        "marks": marks,
        "vp_transverse": None if vp_transverse is None else vp_transverse.as_dict(),
        "vp_along": None if vp_along is None else vp_along.as_dict(),
        "edge_line": None if edge_line is None else [float(v) for v in edge_line],
        "marks_on_edge": str(marks_on_edge),
        "width_prior_ft": float(width_ft),
        "homography": np.asarray(H, dtype=float).tolist(),
        "homography_info": hinfo,
        "pose": {
            "R": np.asarray(pose.R, dtype=float).tolist(),
            "t": [float(v) for v in np.asarray(pose.t, dtype=float).ravel()],
            "z_sign": float(z_sign),
            "z_sign_detail": z_detail,
            "orthonormality": float(pose.info.get("orthonormality", 0.0)),
            "camera_centre_ft": [float(v) for v in pose.camera_centre],
            "camera_height_ft": float(pose.camera_centre[2]),
        },
        "wall": wall,
        "water": water_plane.as_dict(),
        "pool": pool_info,
        "extent": ext,
        # The width observation is its own fact, not a bbox: keeping them in one dict is what made
        # the extent gate vacuous (see the comment above).
        "width_observation": width_obs,
        "anchors": anchor_notes,
    }
    doc["lens"]["roundtrip_px"] = _lens.roundtrip_error(
        [[0.0, 0.0], [image_size[0] - 1, 0.0], [0.0, image_size[1] - 1],
         [image_size[0] - 1, image_size[1] - 1],
         [image_size[0] / 2.0, image_size[1] / 2.0]], lens)
    doc["gates"] = validate_gates(doc)
    return doc


def _undistort_mark_clicks(marks: list[dict], lens: LensModel) -> list[dict]:
    pts = np.array([[float(m["x_px"]), float(m["y_px"])] for m in marks], dtype=float)
    und = undistort_points(pts, lens)
    out = []
    for m, u in zip(marks, und):
        rec = dict(m)
        rec["x_px_raw"] = float(m["x_px"])
        rec["y_px_raw"] = float(m["y_px"])
        rec["x_px"] = float(u[0])
        rec["y_px"] = float(u[1])
        out.append(rec)
    return out


def _undistort_segments(segments: Sequence[Sequence[float]], lens: LensModel
                        ) -> list[list[float]]:
    """Undistort line/segment clicks, endpoint by endpoint.

    A straight world line images as a straight line only in the **undistorted** frame, so the two
    endpoints carry the whole line and there is nothing to re-sample.  A homogeneous ``(3,)`` line
    handed in is returned unchanged: it is already a statement about geometry, not a click, and
    transforming it would require the full line-conic transform rather than a point map.
    """
    if not segments:
        return []
    out: list[list[float]] = []
    flat: list[list[float]] = []
    idx: list[int] = []
    for i, seg in enumerate(segments):
        s = np.asarray(seg, dtype=float).ravel()
        if s.size == 3:
            out.append([float(v) for v in s])
            continue
        if s.size < 4:
            raise ValueError(f"a segment needs 4 numbers, got {s.size}: {seg!r}")
        out.append([])                        # placeholder, filled below
        flat.append([float(s[0]), float(s[1])])
        idx.append(len(out) - 1)
        flat.append([float(s[2]), float(s[3])])
        idx.append(len(out) - 1)
    if flat:
        und = undistort_points(np.asarray(flat, dtype=float), lens)
        for k, slot in enumerate(idx):
            out[slot].extend([float(und[k][0]), float(und[k][1])])
    return out


def _seg_to_line(seg: Sequence[float]) -> np.ndarray:
    s = np.asarray(seg, dtype=float).ravel()
    if s.size == 3:
        return s
    if s.size < 4:
        raise ValueError(f"a segment needs 4 numbers, got {s.size}")
    from .vanishing import line_from_points
    return line_from_points(s[:2], s[2:4])


def _seg_to_line_or_line(spec: Any) -> np.ndarray:
    """Accept either a 4-vector segment or a 3-vector homogeneous line."""
    s = np.asarray(spec, dtype=float).ravel()
    if s.size == 3:
        return s
    return _seg_to_line(s)


def _build_water(pose: PlanePose, lens: LensModel, rim: dict | None,
                 wall: dict | None, water: dict | None) -> _surface.WaterPlane:
    """Resolve the water plane from the operator's choice, defaulting sensibly and saying so."""
    spec = dict(water or {})
    method = str(spec.get("method") or ("rim_pair" if rim else "known_freeboard"))
    if method == "known_freeboard":
        fb = spec.get("freeboard_in")
        if fb is None:
            fb = 19.5
            spec["freeboard_in_assumed"] = fb
        wp = _surface.water_plane_from_known_freeboard(
            float(fb), source=str(spec.get("source", "spec")),
            sigma_in=spec.get("sigma_in"))
        if "freeboard_in_assumed" in spec:
            wp.notes.append(
                f"no freeboard was supplied, so Dog Diving' standard {fb:g} in was ASSUMED; this is "
                f"the single biggest input to every distance and it must be replaced with the "
                f"venue's own measured value"
            )
        return wp
    if rim is None:
        raise ValueError(f"the '{method}' water method needs a rim pair "
                         f"({{'top': [x, y], 'contact': [x, y]}}) in the clicks")
    if wall is None:
        raise ValueError(
            f"the '{method}' water method needs the vertical wall the contact lies on — pass "
            f"clicks['wall'] = {{'axis': 'y', 'value': <world Y in ft>}}, or supply edge_line so "
            f"it can be derived. Without a wall the rim pair does not constrain a height at all "
            f"(see the surface module docstring)"
        )
    top = undistort_points(np.asarray(rim["top"], dtype=float).reshape(1, 2), lens)[0]
    con = undistort_points(np.asarray(rim["contact"], dtype=float).reshape(1, 2), lens)[0]
    if method == "wall_contact":
        return _surface.water_plane_from_wall_contact(pose, con, wall)
    if method == "rim_pair":
        return _surface.water_plane_from_rim_pair(pose, top, con, wall)
    raise ValueError(f"unknown water method {method!r}: expected 'wall_contact', 'rim_pair' or "
                     f"'known_freeboard'")


# ============================================================================
# Gates (W1..W7)
# ============================================================================
def validate_gates(doc: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """W1–W7.  Returns ``{gate: {"pass": bool, "warn": bool, "detail": str}}``.

    Each gate is a test the *document* can fail on its own, with the threshold chosen from a
    physical consequence rather than from convenience:

    W1  lens self-consistency — a distortion model that cannot invert itself is a bug.
    W2  homography residual — 3 in warn, 12 in block, because those are fractions of a jump.
    W3  pose orthonormality — a non-rigid pose means the fitted plane is not a plane.
    W4  the two independent routes agree (homography vs ray/plane) — the project's protocol.
    W5  the water level is inside the sane band and the rim pair was self-consistent.
    W6  the observed width (if measured) agrees with the width prior.
    W7  the mapping is non-degenerate and orientation-preserving (no fold-back).
    """
    g: dict[str, dict[str, Any]] = {}

    rt = float(doc.get("lens", {}).get("roundtrip_px") or 0.0)
    g["W1"] = {"pass": rt <= 0.05, "warn": False,
               "detail": f"lens undistort/distort round-trip max {rt:.4g} px (block > 0.05)"}

    res = (doc.get("homography_info") or {}).get("residual") or {}
    rms_ft = res.get("rms_ft")
    if rms_ft is None:
        g["W2"] = {"pass": False, "warn": False, "detail": "no homography residual recorded"}
    else:
        g["W2"] = {"pass": float(rms_ft) <= 1.0,
                   "warn": float(rms_ft) > 0.25,
                   "detail": f"homography residual RMS {float(rms_ft) * 12:.2f} in "
                             f"(warn > 3 in, block > 12 in)"}

    ortho = float((doc.get("pose") or {}).get("orthonormality") or 0.0)
    g["W3"] = {"pass": ortho <= 0.05, "warn": False,
               "detail": f"pose orthonormality residual {ortho:.4f} (block > 0.05)"}

    # W4 needs a pose rebuilt from the document; do it rather than trusting a stored number.
    try:
        pose = pose_from_document(doc)
        water = water_from_document(doc)
        pts = _probe_grid(pose, water)
        agree = _agreement(pose, water, pts)
        g["W4"] = {"pass": bool(agree["ok"]), "warn": False,
                   "detail": f"homography route vs ray/plane route: max disagreement "
                             f"{agree['max_in']:.2e} in over {agree['n']} probe point(s)"}
    except Exception as exc:  # noqa: BLE001 - a gate must never raise
        g["W4"] = {"pass": False, "warn": False, "detail": f"could not cross-check the routes: {exc}"}

    wd = doc.get("water") or {}
    z = wd.get("z_ft")
    rim_ok = True
    rim_detail = "no rim pair (method " + str(wd.get("method")) + ")"
    if wd.get("method") == "rim_pair" and wd.get("residual_ft") is not None:
        rim_ok = float(wd["residual_ft"]) <= 0.25
        rim_detail = (f"rim pair vertical disagreement {float(wd['residual_ft']) * 12:.1f} in "
                      f"(block > 3 in)")
    sane = z is not None and abs(float(z)) <= _surface.WATER_OFFSET_SANE_FT[1]
    g["W5"] = {"pass": bool(sane and rim_ok), "warn": False,
               "detail": f"water level {float(z) * 12:+.1f} in from the marked plane; {rim_detail}"}

    obs = (doc.get("width_observation") or {}).get("width_observed_ft")
    if obs is None:
        g["W6"] = {"pass": True, "warn": False,
                   "detail": "no observed pool width to compare with the prior (visual check only)"}
    elif not (doc.get("width_observation") or {}).get("width_observed_independent", False):
        g["W6"] = {"pass": True, "warn": False,
                   "detail": f"observed width {float(obs):.2f} ft vs prior "
                             f"{float(doc.get('width_prior_ft') or 16.0):g} ft — but route "
                             f"{doc.get('route')} used the prior as an INPUT, so this is a "
                             f"self-consistency value and not evidence for the prior"}
    else:
        diff = abs(float(obs) - float(doc.get("width_prior_ft") or 16.0))
        # ``POOL_WIDTH_PRIOR_BAND_FT`` is the plan's "standard dock-diving pool" family (14-20 ft).
        # It is an *additional* observation on top of the prior-difference thresholds, not a
        # replacement: a 15 ft pool with a 16 ft prior passes the difference test while telling you
        # the width prior was the wrong model for this venue.
        band = tuple(_config.POOL_WIDTH_PRIOR_BAND_FT)
        outside_band = not (float(band[0]) <= float(obs) <= float(band[1]))
        g["W6"] = {
            "pass": diff <= 3.0, "warn": bool(diff > 1.5 or outside_band),
            "detail": f"observed width {float(obs):.2f} ft vs prior "
                      f"{float(doc.get('width_prior_ft') or 16.0):g} ft "
                      f"(diff {diff:.2f} ft; warn > 1.5, block > 3) — independent check"
                      + (f"; the observed width is OUTSIDE the standard {float(band[0]):g}-"
                         f"{float(band[1]):g} ft band, so the width prior is the wrong model for "
                         f"this venue" if outside_band else "")}

    try:
        H = np.asarray(doc["homography"], dtype=float).reshape(3, 3)
        cond = _homography._homography_condition(H)
        det = float(np.linalg.det(H[:2, :2]))
        ok = np.isfinite(cond["cond"]) and cond["cond"] < 1e12 and abs(det) > 1e-12
        g["W7"] = {"pass": bool(ok), "warn": False,
                   "detail": f"px/ft along {cond['px_per_ft_along']:.1f}, across "
                             f"{cond['px_per_ft_across']:.1f}, anisotropy "
                             f"{cond['anisotropy']:.1f}x, cond {cond['cond']:.3g}"
                             + (" — the transverse axis is coarsely resolved, so transverse "
                                "errors are cheap" if cond["anisotropy"] > 20 else "")}
    except Exception as exc:  # noqa: BLE001
        g["W7"] = {"pass": False, "warn": False, "detail": f"degenerate homography: {exc}"}
    return g


def _probe_grid(pose: PlanePose, water: _surface.WaterPlane, n: int = 5):
    from .homography import apply_homography
    W = np.array([[(31.0 * i / max(n - 1, 1)), (16.0 * j / 2.0)]
                  for i in range(n) for j in range(3)], dtype=float)
    return apply_homography(water.homography(pose), W)


def _agreement(pose: PlanePose, water: _surface.WaterPlane, pixels) -> dict[str, Any]:
    from .intersection import route_agreement_ft
    return route_agreement_ft(pose, water, pixels)


def gates_pass(gates: dict[str, dict[str, Any]]) -> bool:
    return all(bool(g["pass"]) for g in gates.values())


def gate_summary(gates: dict[str, dict[str, Any]]) -> str:
    """One line, for a report or a banner: ``W1 ok  W2 ok  W3 FAIL``."""
    return "  ".join(f"{k}:{'ok' if v['pass'] else ('warn' if v.get('warn') else 'FAIL')}"
                     for k, v in gates.items())


# ============================================================================
# Document accessors — the only sanctioned way to rebuild live objects
# ============================================================================
def pose_from_document(doc: dict[str, Any]) -> PlanePose:
    """Rebuild the :class:`PlanePose` from a document.

    Rebuilding rather than storing live objects is what makes the JSON the single source of truth:
    a document copied into ``out/<clip>/calib_used.json`` reproduces the exact geometry that
    produced the score, months later, with no dependency on a pickled object.
    """
    require_schema(doc)
    lens = LensModel.from_dict(doc["lens"])
    H = np.asarray(doc["homography"], dtype=float).reshape(3, 3)
    p = doc["pose"]
    return PlanePose(K=lens.K, R=np.asarray(p["R"], dtype=float).reshape(3, 3),
                     t=np.asarray(p["t"], dtype=float).ravel(), H=H,
                     z_sign=float(p.get("z_sign", 1.0)),
                     info={"orthonormality": float(p.get("orthonormality", 0.0)),
                           "from_document": True})


def water_from_document(doc: dict[str, Any]) -> _surface.WaterPlane:
    require_schema(doc)
    wd = doc["water"]
    return _surface.WaterPlane(
        z_ft=float(wd["z_ft"]), method=str(wd.get("method", "unknown")),
        sigma_ft=None if wd.get("sigma_ft") is None else float(wd["sigma_ft"]),
        anchors=list(wd.get("anchors") or []),
        residual_px=None if wd.get("residual_px") is None else float(wd["residual_px"]),
        residual_ft=None if wd.get("residual_ft") is None else float(wd["residual_ft"]),
        wall=wd.get("wall"), rim_top_z_ft=wd.get("rim_top_z_ft"),
        notes=list(wd.get("notes") or []))


def lens_from_document(doc: dict[str, Any]) -> LensModel:
    require_schema(doc)
    return LensModel.from_dict(doc["lens"])


def require_schema(doc: dict[str, Any]) -> None:
    """Refuse anything that is not a v1 pool calibration, by name.

    This is the guard that keeps the pre-existing 1-D documents (``data/calib/poolA.json`` and
    friends) from being consumed as 3-D datums — one of them fails its own dock-edge gate by 167 in,
    and a loader that shrugged and read the fields it recognised would turn that into a wrong score
    instead of an error.
    """
    got = (doc or {}).get("schema")
    if got != SCHEMA:
        raise ValueError(
            f"not a pool calibration: expected schema {SCHEMA!r}, got {got!r}. The 1-D mark schema "
            f"in data/calib/ is a different document and must be read with "
            f"dog_diving.calibration.load, not this module."
        )


# ============================================================================
# Persistence
# ============================================================================
def save(doc: dict[str, Any], path: str | Path) -> Path:
    """Atomic write — a half-written calibration is worse than none."""
    require_schema(doc)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.json")
    tmp.write_text(json.dumps(doc, indent=2, default=str), encoding="utf-8")
    shutil.move(str(tmp), str(path))
    return path


def load(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        doc = json.load(f)
    require_schema(doc)
    return doc


def setup_path(setup_id: str, dir: str | Path | None = None) -> Path:
    from ..config import CALIB_DIR
    d = Path(dir) if dir is not None else Path(CALIB_DIR)
    return d / f"{setup_id}.pool.json"


def list_setups(dir: str | Path | None = None) -> list[Path]:
    from ..config import CALIB_DIR
    d = Path(dir) if dir is not None else Path(CALIB_DIR)
    if not d.exists():
        return []
    return sorted(d.glob("*.pool.json"))
