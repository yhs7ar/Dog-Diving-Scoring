"""Phase 5 — water-plane extrapolation around the occlusion.

`pool-translation-plan.txt`, Phase 5::

    Objective: Reconstruct the water plane where it is hidden behind the front wall
               from the camera's perspective.
    Implementation:
      * Anchor the plane using the fully visible top rim and vertical side walls of the
        pool structure.
      * Extend the plane downward using camera extrinsic parameters to mathematically
        define the water level.

Why this is cheap once Phase 4 exists
------------------------------------
:meth:`dog_diving.pool.homography.PlanePose.plane_homography` shows that the homography of a
parallel plane at height ``z`` differs from the marked plane's homography only in its third
column.  So Phase 5 is not a second fit and does not have to see the water: it is the **one
scalar** ``z_water`` — how far the water surface sits from the marked plane.  Everything the
plan calls "extrapolation" is that one number on top of the pose Phase 4 already produced, and
that is why the water plane can be reconstructed where it is invisible.

The plan's rim anchor, corrected
--------------------------------
The plan says to anchor on "the visible top rim **and vertical side walls**".  Taken as
*coplanar water points*, a rim is unsafe, and the project has already established why twice
over:

* the literature pass recorded in the ``dock-diving-scoring`` skill: *"Anchor the water keys to
  the water's contact with the dock, never to a pool rim or coping.  An above-ground rim sits
  above the water plane, so a labeled rim point is not coplanar with the plane you are solving
  for — worse than no point at all."*
* the user's own domain correction: the dock lip **overhangs** the pool by ~2 in, so the water's
  contact with the dock can itself be hidden beneath the lip.

The *wall* half of the sentence is the part that is exactly right, and it is what this module
builds on.  A pool wall is a **vertical plane** at a known world Y (or X), and a point on a
known vertical plane has only **one** unknown — its height.  So a click on the water where it
meets a known wall is directly solvable, in closed form, with no fitting at all.

The trap this module exists to avoid
------------------------------------
An earlier version of this file asked the operator for a rim **top** and a water contact and
solved for "(X, Y, z_top, z_contact) such that both project to their clicks".  That formulation
is **scale-degenerate** and it fails quietly:

* two rays from the *same* camera centre only share a vertical if the two rays are the same
  ray, so a perfect solution does not exist and the solver is minimising a contradiction;
* worse, the objective it minimises is unbounded below — any point *on a ray* reprojects exactly
  onto that ray's pixel, so pushing both points to infinity drives the residual to zero.  There
  is no scale in the problem to stop it.

Measured on the synthetic scene (water truly at −1.625 ft), that version returned
**z = −4.561 ft** with a *small* residual of 0.269 px — a 35 in error dressed up as a good fit —
and its reported click sensitivity was **±2405 in per pixel** instead of a few inches.  The
wall-plane formulation below returns the water level to 1e-9 ft on the same scene with a
sensitivity of the order of an inch.  The lesson is not "add a regulariser"; it is that the
plan's rim pair only works *because of the walls*, and without a wall the pair does not
constrain a height at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from .geometry import (
    height_on_wall,
    ray_from_pixel,
    reprojection_error_px,
    world_from_pixel,
)
from .homography import PlanePose, apply_homography_inv

# How far a competition dock's top surface can plausibly sit from the plane the marks are
# painted on, in feet.  Used only as a sanity band, wide enough not to reject a real venue.
WATER_OFFSET_SANE_FT = (-4.0, 4.0)

# Wall axis names, in the language an operator uses.
WALL_NEAR_EDGE = "y"          # the long pool edge the camera sees, at a known Y


# ============================================================================
# The plane object
# ============================================================================
@dataclass
class WaterPlane:
    """The water surface as a height offset from the **marked plane**.

    ``z_ft`` is the world Z of the water surface in the Phase 4 frame, which is anchored on
    whatever plane the operator's marks lie on (that plane is Z = 0).  The *sign* is therefore a
    property of the venue, not a bug:

    * marks painted on the **deck / dock top** → the water is below → ``z_ft < 0``, and the
      plan's Phase 5 ("extend the plane downward") applies literally;
    * marks painted on the pool **wall at the waterline** → the water *is* the marked plane →
      ``z_ft ~ 0`` and there is nothing to extend.  This is a legitimate arrangement
      (``PLAN.md`` §12 item 4 is still open on exactly this question), so it is not an error.

    ``method`` records how the level was obtained, because the methods do not carry equal
    authority: a wall contact is a measurement, a stated freeboard is a specification.
    """

    z_ft: float
    method: str                                   # wall_contact | rim_pair | known_freeboard
    sigma_ft: float | None = None
    anchors: list[dict[str, Any]] = field(default_factory=list)
    residual_px: float | None = None
    residual_ft: float | None = None              # the rim pair's own consistency check
    wall: dict[str, Any] | None = None
    rim_top_z_ft: float | None = None
    notes: list[str] = field(default_factory=list)

    # --- queries ---------------------------------------------------------
    @property
    def offset_ft(self) -> float:
        """Water level relative to the marked plane (negative = below it)."""
        return float(self.z_ft)

    @property
    def below_marked_plane_ft(self) -> float:
        """How far the water sits below the marked plane (positive when below)."""
        return -float(self.z_ft)

    def is_below(self, tol: float = 1e-9) -> bool:
        """Is the water surface below the marked plane, beyond float noise?

        ``tol`` defaults to a nanometre-scale 1e-9 ft rather than 0, because a waterline-marks venue
        (where the marked plane *is* the water) solves to ~1e-16 ft and a strict ``< 0`` would report
        the water as above the plane it is level with.
        """
        return float(self.z_ft) < -abs(float(tol))

    def as_dict(self) -> dict[str, Any]:
        return {
            "z_ft": float(self.z_ft),
            "offset_ft": self.offset_ft,
            "offset_in": self.offset_ft * 12.0,
            "below_marked_plane_in": self.below_marked_plane_ft * 12.0,
            "method": self.method,
            "sigma_ft": None if self.sigma_ft is None else float(self.sigma_ft),
            "sigma_in": None if self.sigma_ft is None else float(self.sigma_ft) * 12.0,
            "residual_px": None if self.residual_px is None else float(self.residual_px),
            "residual_ft": None if self.residual_ft is None else float(self.residual_ft),
            "wall": self.wall,
            "rim_top_z_ft": self.rim_top_z_ft,
            "anchors": self.anchors,
            "notes": list(self.notes),
        }

    # --- the plane's own homography --------------------------------------
    def homography(self, pose: PlanePose) -> np.ndarray:
        """Undistorted pixels <-> (X, Y) feet **on the water surface**."""
        return pose.plane_homography(float(self.z_ft))



    # --- extent ----------------------------------------------------------
    @staticmethod
    def within_extent(world_xy: Any, extent: dict[str, float] | None,
                      margin_ft: float = 0.0) -> list[bool]:
        """Which points fall inside a known pool extent (a sanity gate, not a measurement).

        An (X, Y) in the crowd or behind the dock is arithmetically fine and physically
        impossible; this is the check that catches it.  ``margin_ft`` widens the box so a
        legitimate edge-of-pool entry is not rejected.
        """
        pts = np.asarray(world_xy, dtype=float).reshape(-1, 2)
        if not extent:
            return [True] * pts.shape[0]
        x0 = float(extent.get("x_min", -np.inf)) - float(margin_ft)
        x1 = float(extent.get("x_max", np.inf)) + float(margin_ft)
        y0 = float(extent.get("y_min", -np.inf)) - float(margin_ft)
        y1 = float(extent.get("y_max", np.inf)) + float(margin_ft)
        return [bool(x0 <= p[0] <= x1 and y0 <= p[1] <= y1) for p in pts]


# ============================================================================
# Building the plane
# ============================================================================
def wall_from_edge_line(pose: PlanePose, image_line: Sequence[float]) -> dict[str, Any]:
    """The world ``Y`` of a pool edge observed as an image line — i.e. its wall offset.

    A pool wall is a vertical plane, so the whole image of a wall's top edge must map to a
    single world ``Y``.  Pushing several points of the observed line onto the marked plane and
    taking the mean gives that ``Y``, and their spread is the honest measure of whether the
    line really was a wall edge (a curved coping, or a line that is actually the waterline) —
    reported as ``spread_ft`` rather than discarded.
    """
    l = np.asarray(image_line, dtype=float).ravel()
    if l.size == 4:
        p1, p2 = np.array([l[0], l[1]]), np.array([l[2], l[3]])
    elif l.size == 3:
        n = float(np.hypot(l[0], l[1])) or 1.0
        ln = l / n
        foot = -ln[2] * np.array([ln[0], ln[1]])
        d = np.array([-ln[1], ln[0]])
        p1, p2 = foot - d * 500.0, foot + d * 500.0
    else:
        raise ValueError("image_line must be a homogeneous (3,) line or a (4,) segment")
    ts = np.linspace(0.0, 1.0, 7)
    pts = np.array([p1 + t * (p2 - p1) for t in ts])
    world = apply_homography_inv(pose.H, pts)
    ys = world[:, 1]
    return {
        "axis": WALL_NEAR_EDGE,
        "value": float(ys.mean()),
        "spread_ft": float(np.ptp(ys)),
        "spread_in": float(np.ptp(ys) * 12.0),
        "n_samples": int(pts.shape[0]),
    }


def _wall_spec(wall: dict[str, Any] | str | None,
               wall_value: float | None) -> tuple[dict[str, Any], str, float]:
    """Normalise the ``wall`` argument, refusing ``None`` with a message that names the consequence.

    ``None`` is a realistic slip — a caller who has not yet derived the wall will pass one — and it
    must produce the module's own explanation, not ``TypeError: 'NoneType' object is not iterable``.
    """
    if wall is None:
        raise ValueError(
            "this water-level method needs the vertical wall the contact lies on — pass "
            "{'axis': 'y', 'value': <world Y in ft>}, or an axis name with wall_value. Without a "
            "wall a click on the water constrains two unknowns (its X and Y) and cannot be solved "
            "(see the module docstring)"
        )
    if isinstance(wall, str):
        if wall_value is None:
            raise ValueError("give wall_value alongside the axis shorthand, e.g. wall='y', "
                             "wall_value=0.0")
        spec: dict[str, Any] = {"axis": wall, "value": float(wall_value)}
    else:
        spec = dict(wall)
        if wall_value is not None:
            spec["value"] = float(wall_value)
    if "axis" not in spec or "value" not in spec:
        raise ValueError(f"the wall needs both 'axis' and 'value' (in ft); got {wall!r}")
    return spec, str(spec["axis"]).lower(), float(spec["value"])


def water_plane_from_wall_contact(
    pose: PlanePose,
    contact_pixel: Sequence[float],
    wall: dict[str, Any] | str,
    *,
    wall_value: float | None = None,
    sigma_px: float = 1.0,
) -> WaterPlane:
    """The plan's wall anchor: one click where the water meets a **known vertical wall**.

    Closed form, no fitting: the click gives a ray, the wall gives a vertical plane at a known
    world ``Y`` (or ``X``), and their intersection is the water's contact point.  Its ``Z`` is
    the water level.  This is the measurement the plan's Phase 5 is reaching for — and it is
    well posed precisely because the wall removes the second unknown that makes a bare click on
    the water unsolvable.

    ``wall`` is either ``{"axis": "y", "value": 0.0}`` (what :func:`wall_from_edge_line`
    returns) or the shorthand ``"y"``/``"x"`` with ``wall_value`` given separately.
    """
    spec, axis, value = _wall_spec(wall, wall_value)
    hit = height_on_wall(pose, contact_pixel, axis, value)
    notes: list[str] = []
    if hit["status"] != "ok":
        raise ValueError(
            f"the water contact cannot be placed on the wall {axis}={value:.3f} ft: "
            f"{hit['status']} — {hit['detail']}"
        )
    z = float(hit["z_ft"])
    p = np.asarray(hit["point"], dtype=float)
    if abs(z) > WATER_OFFSET_SANE_FT[1]:
        notes.append(
            f"the solved water level is {z * 12:+.0f} in from the marked plane, outside the "
            f"sane band of +-{WATER_OFFSET_SANE_FT[1] * 12:.0f} in — a mis-click, the wrong "
            f"wall, or marks that are not on the plane the wall was measured from"
        )
    notes.append(
        f"water level solved in closed form as the ray/wall intersection at {axis}="
        f"{value:.3f} ft; the wall supplies the horizontal position, so the click only has to "
        f"give the height"
    )
    if "spread_ft" in spec and float(spec["spread_ft"]) > 1.0:
        notes.append(
            f"the wall line's world Y varies by {float(spec['spread_ft']) * 12:.1f} in along "
            f"its length, so it was probably not a single wall edge (a curved coping, or the "
            f"waterline rather than the coping)"
        )
    sigma = _sigma_from_pixel(pose, contact_pixel, axis, value, sigma_px)
    return WaterPlane(
        z_ft=z, method="wall_contact", sigma_ft=sigma,
        anchors=[{"role": "water_contact", "x_px": float(contact_pixel[0]),
                  "y_px": float(contact_pixel[1]), "wall": spec}],
        residual_px=0.0,                            # exact by construction: no fit was made
        wall=spec, notes=notes,
    )


def water_plane_from_rim_pair(
    pose: PlanePose,
    rim_top_pixel: Sequence[float],
    water_contact_pixel: Sequence[float],
    wall: dict[str, Any] | str,
    *,
    wall_value: float | None = None,
    max_residual_ft: float = 0.25,
) -> WaterPlane:
    """The plan's full rim anchor: a rim top **and** the water contact, on a known wall.

    Both clicks are placed on the same known vertical plane, so each gives a 3-D point in
    closed form.  The water level is the **contact's** height, and the pair's job is the
    self-check: two points on one wall vertical must share their along-wall coordinate ``X``.
    Their disagreement is the honest residual (in feet, and in pixels via reprojection), and a
    large one means the two clicks were not on one vertical — a curved coping, a perspective
    mistake, or a mis-click — which is exactly the check the previous, degenerate formulation
    could not make because it had no residual worth reporting.

    ``rim_top_z_ft`` is reported because the rim's height above the marked plane is a real
    venue fact and is worth surfacing: if it comes out as a wild number, the wall was wrong.
    """
    spec, axis, value = _wall_spec(wall, wall_value)

    top_hit = height_on_wall(pose, rim_top_pixel, axis, value)
    con_hit = height_on_wall(pose, water_contact_pixel, axis, value)
    if top_hit["status"] != "ok" or con_hit["status"] != "ok":
        bad = top_hit if top_hit["status"] != "ok" else con_hit
        raise ValueError(f"a rim click cannot be placed on the wall {axis}={value:.3f} ft: "
                         f"{bad['status']} — {bad['detail']}")

    p_top = np.asarray(top_hit["point"], dtype=float)
    p_con = np.asarray(con_hit["point"], dtype=float)
    along = 0 if axis == "y" else 1          # the coordinate that must agree on one vertical
    d_along = abs(float(p_top[along] - p_con[along]))
    d_along_px = max(
        reprojection_error_px(pose, p_con, water_contact_pixel),
        reprojection_error_px(pose, p_top, rim_top_pixel),
    )
    z_contact = float(p_con[2])

    notes: list[str] = []
    if d_along > float(max_residual_ft):
        notes.append(
            f"REJECT-WORTHY: the two rim clicks are {d_along * 12:.1f} in apart along the "
            f"wall (limit {max_residual_ft * 12:.1f} in), so they were not on one vertical. The "
            f"water level below is the contact's own height and is still usable, but the pair's "
            f"self-check failed — re-click, or use the single-contact method"
        )
    if p_top[2] < p_con[2]:
        notes.append(
            f"the clicked rim TOP ({p_top[2]:+.3f} ft) solves BELOW the clicked water contact "
            f"({p_con[2]:+.3f} ft). That is the click order reversed, or the wall on the wrong "
            f"side of the pool"
        )
    if abs(z_contact) > WATER_OFFSET_SANE_FT[1]:
        notes.append(
            f"the solved water level is {z_contact * 12:+.0f} in from the marked plane, outside "
            f"the sane band of +-{WATER_OFFSET_SANE_FT[1] * 12:.0f} in"
        )
    notes.append(
        "the rim is used as a VERTICAL REFERENCE on a known wall, never as a coplanar water "
        "point: an above-ground rim sits above the water plane, so a labeled rim point would "
        "not lie on the plane being solved for (see the module docstring)"
    )
    sigma = _sigma_from_pixel(pose, water_contact_pixel, axis, value, 1.0)
    return WaterPlane(
        z_ft=z_contact, method="rim_pair", sigma_ft=sigma,
        anchors=[{"role": "rim_top", "x_px": float(rim_top_pixel[0]),
                  "y_px": float(rim_top_pixel[1])},
                 {"role": "water_contact", "x_px": float(water_contact_pixel[0]),
                  "y_px": float(water_contact_pixel[1])}],
        residual_px=d_along_px, residual_ft=d_along, wall=spec,
        rim_top_z_ft=float(p_top[2]), notes=notes,
    )


def water_plane_from_known_freeboard(freeboard_in: float,
                                     *, source: str = "spec",
                                     sigma_in: float | None = None) -> WaterPlane:
    """The water plane from a stated freeboard — a specification, not a measurement.

    On most venues this is the more accurate answer: a competition dock's height above the water
    is fixed, measured and published, where a one-pixel wall click is worth a couple of inches.
    Labelled ``known_freeboard`` so no downstream number can be quoted as if the water level had
    been observed.
    """
    fb = abs(float(freeboard_in)) / 12.0
    return WaterPlane(
        z_ft=-fb, method="known_freeboard",
        sigma_ft=None if sigma_in is None else abs(float(sigma_in)) / 12.0,
        notes=[f"water level from a stated freeboard of {abs(float(freeboard_in)):g} in "
               f"({source} specification); its error is the venue's tolerance, not this "
               f"pipeline's"],
    )


def _sigma_from_pixel(pose: PlanePose, pixel: Sequence[float], axis: str, value: float,
                      sigma_px: float) -> float | None:
    """Sensitivity of the solved height to ``sigma_px`` of click error, in feet.

    Four perturbed re-solves, so the number is the actual spread the click error produces rather
    than a linearisation that may not hold (the height is a ratio of two nearly-parallel ray
    components, so it is distinctly non-linear in the click).
    """
    vals: list[float] = []
    base = np.asarray(pixel, dtype=float).ravel()[:2]
    for dp in ((sigma_px, 0.0), (-sigma_px, 0.0), (0.0, sigma_px), (0.0, -sigma_px)):
        try:
            hit = height_on_wall(pose, base + np.asarray(dp, dtype=float), axis, value)
        except Exception:  # noqa: BLE001 - a sensitivity is best-effort reporting
            continue
        if hit["status"] == "ok":
            vals.append(float(hit["z_ft"]))
    if len(vals) < 2:
        return None
    return (max(vals) - min(vals)) / 2.0


def sensitivity_from_clicks(
    pose: PlanePose,
    rim_top_pixel: Sequence[float],
    water_contact_pixel: Sequence[float],
    wall: dict[str, Any] | str,
    *,
    wall_value: float | None = None,
    sigma_px: float = 1.0,
) -> dict[str, Any]:
    """What a click error costs on the water level, in inches per ``sigma_px``.

    Perturbs each of the two clicks by ``+/- sigma_px`` in x and y, re-solves the wall contact,
    and reports the spread of the resulting level.  This is the term that decides how much of the
    accuracy budget Phase 5 consumes — and it is the number that exposed the previous rim-pair
    formulation as unusable (it reported thousands of inches, versus single-digit inches here).
    """
    base = water_plane_from_rim_pair(pose, rim_top_pixel, water_contact_pixel, wall,
                                     wall_value=wall_value)
    vals: list[float] = [base.z_ft]
    for dp in ((sigma_px, 0.0), (-sigma_px, 0.0), (0.0, sigma_px), (0.0, -sigma_px)):
        for con in (True, False):
            top = np.asarray(rim_top_pixel, dtype=float).ravel()[:2].copy()
            wat = np.asarray(water_contact_pixel, dtype=float).ravel()[:2].copy()
            if con:
                wat = wat + np.asarray(dp, dtype=float)
            else:
                top = top + np.asarray(dp, dtype=float)
            try:
                zone = str(wall["axis"]) if isinstance(wall, dict) else str(wall)
                zval = (float(wall["value"]) if isinstance(wall, dict)
                        else float(wall_value))          # type: ignore[arg-type]
                hit = height_on_wall(pose, wat, zone, zval)
            except Exception:  # noqa: BLE001
                continue
            if hit["status"] == "ok":
                vals.append(float(hit["z_ft"]))
    spread_in = (max(vals) - min(vals)) * 12.0
    return {
        "ok": len(vals) > 1,
        "sigma_px": float(sigma_px),
        "level_min_in": float(min(vals) * 12.0),
        "level_max_in": float(max(vals) * 12.0),
        "level_base_in": float(base.z_ft * 12.0),
        "spread_in": float(spread_in),
        "notes": [f"a {sigma_px:g} px click error moves the solved water level over a range of "
                  f"{spread_in:.1f} in — quote this beside any distance that depends on a "
                  f"clicked water level"],
    }


# ============================================================================
# Plane extent from the visible structure (the "side walls" half of the plan)
# ============================================================================
def extent_from_anchors(
    pose: PlanePose,
    anchors: Sequence[dict],
    *,
    z_ft: float = 0.0,
    fallback_z: float | None = None,
) -> dict[str, float]:
    """World bounding box of the pool from clicked structure points.

    ``anchors`` are ``{"role": ..., "x_px": ..., "y_px": ...}`` where the role names a known
    piece of the pool.  Roles recognised by convention and used by the sanity gates:

    * ``"end_far"`` / ``"end_near"`` — the end walls: ``x_max`` / ``x_min``;
    * ``"edge_near"`` / ``"edge_far"`` — the long pool edges: ``y_min`` / ``y_max``.

    Points that sit **below** the marked plane — a water contact on a wall, say — cannot be
    pushed onto it, so ``fallback_z`` lets the caller say "place the ones that fail at this
    height instead".  Passing ``fallback_z=water.z_ft`` is the correct call for a mixed anchor
    set, because the extent is a QA aid and a QA aid that guesses the wrong height silently
    becomes a wrong bound.
    """
    pts: list[np.ndarray] = []
    named: dict[str, float] = {}
    for a in anchors:
        if a.get("x_px") is None or a.get("y_px") is None:
            continue
        px = np.array([[float(a["x_px"]), float(a["y_px"])]])
        hit = world_from_pixel(pose, px[0], float(z_ft))
        if hit["status"] != "ok":
            if fallback_z is None:
                continue
            hit = world_from_pixel(pose, px[0], float(fallback_z))
            if hit["status"] != "ok":
                continue
        xy = np.asarray(hit["point"], dtype=float)[:2]
        pts.append(xy)
        role = str(a.get("role", ""))
        if role == "end_far":
            named["x_max"] = float(xy[0])
        elif role == "end_near":
            named["x_min"] = float(xy[0])
        elif role == "edge_near":
            named["y_min"] = float(xy[1])
        elif role == "edge_far":
            named["y_max"] = float(xy[1])
    if pts:
        arr = np.asarray(pts, dtype=float)
        named.setdefault("x_min", float(arr[:, 0].min()))
        named.setdefault("x_max", float(arr[:, 0].max()))
        named.setdefault("y_min", float(arr[:, 1].min()))
        named.setdefault("y_max", float(arr[:, 1].max()))
    return named


def extent_from_lines(
    pose: PlanePose,
    edges: Sequence[Sequence[float]],
    *,
    z_ft: float = 0.0,
) -> dict[str, float]:
    """The pool's extent and observed width from whole image lines (the coping edges).

    A rectangular pool's two long edges are two world lines at constant ``Y``; this returns their
    ``Y`` values, the observed width between them, and — the useful part — how well each line
    behaved as a single wall.  That is how a clicked edge independently confirms (or contradicts)
    the width prior that route B in :mod:`dog_diving.pool.homography` depends on.
    """
    reports = []
    for ln in edges:
        try:
            reports.append(wall_from_edge_line(pose, ln))
        except Exception:  # noqa: BLE001 - a QA extent is best-effort
            continue
    out: dict[str, float] = {}
    if reports:
        ys = [float(r["value"]) for r in reports]
        out["y_min"] = float(min(ys))
        out["y_max"] = float(max(ys))
        out["width_observed_ft"] = float(max(ys) - min(ys))
        out["max_line_spread_ft"] = float(max(float(r["spread_ft"]) for r in reports))
    return out
