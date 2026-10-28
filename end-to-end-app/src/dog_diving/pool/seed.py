"""Three detected waterlines -> the plane's four corners (PLAN §17).

The plane is a quad on the water surface whose corners are a metric rectangle, ``X = 0`` on the dock
lip and ``+X`` down the pool.  Detection supplies three of the pool's boundary lines

* ``far``  — water against the far wall   -> the image of ``Y = 0``
* ``near`` — water against the near wall  -> the image of ``Y = W``
* ``dock`` — water against the dock face  -> the image of ``X = 0``

and that is six of a homography's eight constraints.  Two things are still needed, and this module is
about choosing them honestly:

1. **The metric origin** ``p00`` is the intersection of ``far`` and ``dock``: the water's inner
   corner.  It is *measured*, not assumed.
2. **The X scale.**  ``far`` and ``near`` are parallel in the water plane, so their image
   intersection is the vanishing point of the X direction.  With ``p00``, ``p01 = near ∩ dock``
   (= ``W`` feet apart, a known width) and that vanishing point, the mapping is determined up to the
   px-per-foot at the dock edge.  That last number comes from the quad already in the session: its
   own homography gives the ratio between px/ft along X and along Y, and the *measured* width supplies
   the absolute scale.  Only then is the anchor (where ``35 ft`` lands) a single projective parameter,
   and the operator already has two controls for it.

So the seed never guesses an image fraction: the far edge, the near edge and the dock edge come from
the video, the origin is their intersection, and the only inherited quantity is the session's own
anisotropy — stated in the log and in the document.

Two properties worth knowing before reading the gates:

* the seeded quad satisfies **P4** (a foot must subtend fewer px at the far end) *by construction*,
  because the X map is ``s(X) = A·X / (1 + A·X)``, which decays monotonically towards the vanishing
  point.  P4 still earns its keep on hand-dragged quads; it cannot catch a bad seed on its own.
* comparing the seeded quad to the line it was seeded from scores ~0 px and is therefore **not** a
  gate — the same trap as a reversed plane scoring 1.00 on the old P4.  The only independent sample on
  this route is :func:`dog_diving.waterline.analyze`'s half-window stability, and it is reported, not
  hidden.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import cv2
import numpy as np

from . import homography as _homography
from . import plane as plane_mod

#: Corner tags, in the order the quad is drawn, and the metric coordinate each one means.
CORNER_TAGS = plane_mod.CORNER_TAGS

#: Gate thresholds.
MIN_WIDTH_FRAC = 0.15             # |p01 - p00| as a fraction of frame height

#: How far BELOW the detected near waterline the quad's near edge is placed, in original px.  The
#: near long edge the detector finds is the top of the pool wall (the coping); the wall stands above
#: the water, so the water surface meets it this far lower in the image.  Operator order 2026-10-27.
NEAR_WALL_DROP_PX = 30.0
MAX_ANCHOR_PROJECTIVE = 0.995     # s(L) must stay short of the vanishing point
AREA_RATIO_BAND = (0.15, 5.0)     # seeded area vs the default quad's area
RHO_BAND = (0.2, 5.0)             # px/ft(X) / px/ft(Y) inherited from the session


# ---------------------------------------------------------------------------
# Small geometry, all of it testable without a video
# ---------------------------------------------------------------------------
def line_of(edge: Any) -> tuple[np.ndarray, np.ndarray] | None:
    """``(point, unit direction)`` from a :class:`dog_diving.waterline.EdgeFit`, or ``None``."""
    if edge is None or not getattr(edge, "found", False) or getattr(edge, "fit", None) is None:
        return None
    p = np.asarray(edge.fit["point"], dtype=float)
    d = np.asarray(edge.fit["direction"], dtype=float)
    n = float(np.linalg.norm(d))
    if not np.isfinite(n) or n < 1e-9:
        return None
    return p, d / n


def intersect(a: tuple[np.ndarray, np.ndarray] | None,
              b: tuple[np.ndarray, np.ndarray] | None,
              eps: float = 1e-9) -> np.ndarray | None:
    """Intersection of two lines given as (point, unit direction); ``None`` when parallel."""
    if a is None or b is None:
        return None
    (p1, d1), (p2, d2) = a, b
    den = float(d1[0] * d2[1] - d1[1] * d2[0])
    if abs(den) < eps:
        return None
    t = ((p2[0] - p1[0]) * d2[1] - (p2[1] - p1[1]) * d2[0]) / den
    p = p1 + t * d1
    return p if np.all(np.isfinite(p)) else None


def foot_on_line(p: Sequence[float], line: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """Perpendicular projection of ``p`` onto the line."""
    p = np.asarray(p, dtype=float)
    p0, d = line
    t = float((p - p0) @ d)
    return p0 + t * d


def _rotation(deg: float) -> np.ndarray:
    a = np.radians(float(deg))
    c, s = float(np.cos(a)), float(np.sin(a))
    return np.array([[c, -s], [s, c]])


def pixel_of(p00: np.ndarray, p01: np.ndarray, vp: np.ndarray, *, anchor_ft: float,
             width_ft: float, px_per_ft_x: float) -> tuple[Any, float]:
    """The seeded mapping as a function ``(X, Y) ft -> image px``, in closed form.

    ``p(X, Y) = p00 + s(X)·(vp - p00) + (1 - s(X))·(Y/W)·(p01 - p00)`` with
    ``s(X) = A·X / (1 + A·X)``: projective along X (it converges on the detected vanishing point),
    affine along Y (the width is measured, not extrapolated), and consistent with the three lines —
    ``Y = 0`` is the far waterline, ``Y = W`` passes through ``p01`` and the vanishing point, and every
    ``X = const`` edge is parallel to the dock line.  A homography, written out so tests can check the
    properties without inverting a matrix.
    """
    _p10, s_anchor, A = projective_anchor(p00, vp, px_per_ft_x, anchor_ft)

    def at(X: float, Y: float) -> np.ndarray:
        s = A * float(X) / (1.0 + A * float(X))
        return (np.asarray(p00, float) + s * (np.asarray(vp, float) - np.asarray(p00, float))
                + (1.0 - s) * (float(Y) / float(width_ft)) * (np.asarray(p01, float)
                                                              - np.asarray(p00, float)))
    return at, s_anchor


def projective_anchor(p00: np.ndarray, vp: np.ndarray, px_per_ft: float,
                      anchor_ft: float) -> tuple[np.ndarray, float, float]:
    """The anchor corner: where ``X = anchor_ft`` lands along ``p00 -> vp``.

    ``s(X) = A·X / (1 + A·X)`` is the projective fraction of the way to the vanishing point, and
    ``|vp - p00| · A`` is the px-per-foot at the dock edge — so fixing that scale fixes ``A``, and the
    decay towards ``vp`` is then real perspective rather than a guessed image fraction.

    Returns ``(point, s, A)``.  ``s >= 1`` means the anchor lies at or past the vanishing point,
    which the caller must refuse.
    """
    span = float(np.linalg.norm(np.asarray(vp, float) - np.asarray(p00, float)))
    if span < 1e-6:
        raise ValueError("the vanishing point coincides with the metric origin")
    A = float(px_per_ft) / span
    s = A * float(anchor_ft) / (1.0 + A * float(anchor_ft))
    p = np.asarray(p00, float) + s * (np.asarray(vp, float) - np.asarray(p00, float))
    return p, float(s), float(A)


def px_per_ft_ratio(session: plane_mod.PlaneSession | None) -> tuple[float, str]:
    """px/ft along X divided by px/ft along Y, at the dock corner of the session's own quad.

    Under a projective map these two differ (that *is* the foreshortening across the pool), and neither
    the waterlines nor the frame can measure the ratio — only a camera model or an aligned quad can.
    So the seed inherits it from whatever quad is on screen.  The absolute scale still comes from the
    detected width, so a stale quad can only tilt the anchor, not misplace the origin.
    """
    if session is None or not session.placed:
        return 1.0, "no placed quad — px/ft assumed equal along X and Y"
    try:
        H = np.asarray(session.homography(), dtype=float)
        # Analytic Jacobian at (X, Y) = (0, 0): p = (h0·[X,Y,1], h1·[X,Y,1]) / (h2·[X,Y,1]), so
        # d(px)/dX = (h00·h22 - h02·h20, h10·h22 - h12·h20) / h22² (and likewise for Y).  A finite
        # difference over 0.25 ft left a 0.3 % scale error, which is 1.4 px on the anchor corner.
        w0 = float(H[2, 2])
        if abs(w0) < 1e-12:
            raise ValueError("degenerate homography at the origin")
        jx = float(np.hypot(H[0, 0] * w0 - H[0, 2] * H[2, 0],
                            H[1, 0] * w0 - H[1, 2] * H[2, 0]) / (w0 * w0))
        jy = float(np.hypot(H[0, 1] * w0 - H[0, 2] * H[2, 1],
                            H[1, 1] * w0 - H[1, 2] * H[2, 1]) / (w0 * w0))
    except Exception:                                                        # noqa: BLE001
        return 1.0, "the current quad has no usable homography — px/ft assumed equal"
    if jy < 1e-9 or not np.isfinite(jx / jy):
        return 1.0, "degenerate current quad — px/ft assumed equal"
    rho = jx / jy
    lo, hi = RHO_BAND
    if not (lo <= rho <= hi):
        return 1.0, (f"current quad's px/ft ratio {rho:.2f} is implausible "
                     f"(outside {lo:g}-{hi:g}) — assuming 1.0")
    return float(rho), f"px/ft along X is {rho:.2f}x the along-Y scale, inherited from the current quad"


def quad_area(pts: Sequence[Sequence[float]] | dict[str, Any]) -> float:
    """Absolute area of a 4-point polygon, in px²."""
    arr = (np.asarray([pts[t] for t in CORNER_TAGS], dtype=float) if isinstance(pts, dict)
           else np.asarray(pts, dtype=float))
    return float(abs(cv2.contourArea(arr.astype(np.float32))))




def _far_side_ok(p00: np.ndarray, vp: np.ndarray, dock_side: str, frame_w: int) -> tuple[bool, str]:
    """X must increase *away* from the dock's side of the frame.

    The vanishing point of the pool's long direction is the far end of the pool, so it has to lie on
    the opposite side of the origin from the dock.  Getting this backwards mirrors the plane: every
    distance would still be positive and plausible, and every one would be measured towards the dock.
    """
    if dock_side == "left":
        ok = float(vp[0]) > float(p00[0])
    elif dock_side == "right":
        ok = float(vp[0]) < float(p00[0])
    else:
        return True, "dock side unknown — direction not checked"
    return ok, (f"vanishing point x={vp[0]:.0f} vs origin x={p00[0]:.0f} with the dock on the "
                f"{dock_side} (frame width {frame_w})")


# ---------------------------------------------------------------------------
# The result
# ---------------------------------------------------------------------------
@dataclass
class SeedResult:
    level: str = "none"
    corners: dict[str, tuple[float, float]] | None = None
    gates: dict[str, dict[str, Any]] = field(default_factory=dict)
    refused: bool = True
    reason: str = ""
    notes: list[str] = field(default_factory=list)
    anchor_ft: float = 0.0
    width_ft: float = 0.0
    dock_side: str = "unknown"
    px_per_ft_x: float = float("nan")
    px_per_ft_y: float = float("nan")
    rho: float = 1.0
    geometry: dict[str, Any] = field(default_factory=dict)

    @property
    def placed(self) -> bool:
        return self.corners is not None and not self.refused

    def failing(self) -> list[str]:
        return [k for k, v in self.gates.items() if not v.get("pass") and not v.get("warn")]

    def as_dict(self) -> dict[str, Any]:
        return {"level": self.level, "refused": bool(self.refused), "reason": self.reason,
                "notes": list(self.notes), "anchor_ft": float(self.anchor_ft),
                "width_ft": float(self.width_ft), "dock_side": self.dock_side,
                "px_per_ft_x": round(float(self.px_per_ft_x), 3),
                "px_per_ft_y": round(float(self.px_per_ft_y), 3), "px_per_ft_ratio": round(self.rho, 3),
                "corners": (None if self.corners is None else
                            {k: [round(float(v[0]), 2), round(float(v[1]), 2)]
                             for k, v in self.corners.items()}),
                "gates": self.gates, "geometry": self.geometry}


# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------
def _gate(pass_: bool, detail: str, *, warn: bool = False) -> dict[str, Any]:
    return {"pass": bool(pass_), "warn": bool(warn), "detail": detail}


def gate_mask(edges: Any) -> dict[str, Any]:
    """S2 — is the mask plausibly a pool at all?

    Not "one component": glare, a reflected banner and a shaded half of the pool routinely split the
    water into several regions, and upstream's own report flags those rather than rejecting them. What
    *is* disqualifying is a mask that is mostly not water (a few % of the frame, or nearly all of it) or
    a collection of comparably sized blobs, where "the water" is whichever one won the score.
    """
    frac = float(getattr(edges, "water_fraction", float("nan")))
    ncomp = int(getattr(edges, "n_components", 0))
    kept = int(getattr(edges, "components_kept", 0))
    share = float(getattr(edges, "dominant_share", float("nan")))
    ok = (np.isfinite(frac) and 0.05 <= frac <= 0.95
          and (not np.isfinite(share) or share >= 0.5))
    detail = (f"water covers {frac:.1%} of the ROI; {kept} region(s) kept of {ncomp} above 50 px; "
              f"the largest is {share:.0%} of that area"
              + ("" if ok else " — expected water in 5-95% of the frame in one dominant region"))
    return _gate(ok, detail)


def gate_geometry(p00: np.ndarray | None, p01: np.ndarray | None, vp: np.ndarray | None,
                  s: float, frame_shape: tuple[int, int], dock_side: str) -> dict[str, Any]:
    """S3 — do the three lines actually form a pool seen down its length?"""
    if p00 is None or p01 is None or vp is None:
        return _gate(False, "the three lines do not intersect pairwise (parallel pair, or a missed "
                            "edge)")
    h, w = int(frame_shape[0]), int(frame_shape[1])
    width_px = float(np.linalg.norm(p01 - p00))
    far_ok, far_why = _far_side_ok(p00, vp, dock_side, w)
    ok = (width_px >= MIN_WIDTH_FRAC * h) and (0.0 < s < MAX_ANCHOR_PROJECTIVE) and far_ok
    detail = (f"origin {p00.round(1).tolist()}, width {width_px:.0f} px "
              f"({width_px / max(h, 1):.2f} of frame height), anchor at {s:.3f} of the way to the "
              f"vanishing point; {far_why}")
    return _gate(ok, detail)


def gate_quad(corners: dict[str, np.ndarray], *, length_ft: float, width_ft: float,
              frame_shape: tuple[int, int]) -> dict[str, Any]:
    """S4 — P1-P4 on the seeded quad, through the app's own validator."""
    s = plane_mod.PlaneSession(frame_shape=(int(frame_shape[0]), int(frame_shape[1])),
                               length_ft=float(length_ft), width_ft=float(width_ft))
    for tag in CORNER_TAGS:
        s.corners[tag] = (float(corners[tag][0]), float(corners[tag][1]))
    try:
        H = np.asarray(s.homography(), dtype=float)
        gates = plane_mod.validate_plane({"homography": H.tolist()}, s)
    except Exception as exc:                                                  # noqa: BLE001
        return _gate(False, f"the seeded quad has no usable homography: {exc}")
    bad = [k for k, v in gates.items() if not v.get("pass") and not v.get("warn")]
    detail = "; ".join(f"{k} {'ok' if v.get('pass') else 'FAIL'} ({v.get('detail', '')})"
                       for k, v in sorted(gates.items()))
    return {"pass": not bad, "warn": False, "detail": detail,
            "sub": {k: bool(v.get("pass")) for k, v in gates.items()}}


def gate_area(corners: dict[str, np.ndarray], reference_area: float) -> dict[str, Any]:
    """S5 — a mask that latched onto glare or a banner produces a wildly sized quad."""
    area = quad_area(corners)
    lo, hi = AREA_RATIO_BAND
    ratio = area / max(float(reference_area), 1e-6)
    ok = lo <= ratio <= hi
    return _gate(ok, f"seeded area {area:.0f} px² = {ratio:.2f}x the default quad "
                     f"(band {lo:g}-{hi:g}x)")


def gate_edges_found(edges: Any) -> dict[str, Any]:
    """S1 — at least the far edge, and every edge that *is* used passed its own thresholds."""
    found = list(getattr(edges, "found", ()))
    parts = []
    for e in getattr(edges, "edges", ()):
        if e.found:
            parts.append(f"{e.name} rms {e.rms:.1f} inl {e.inlier_fraction:.2f}"
                         + ("" if e.stability_px is None
                            else f" stab {e.stability_px:.1f}"))
        else:
            parts.append(f"{e.name} not found ({e.reason})")
    ok = "far" in found
    return _gate(ok, ("; ".join(parts) or "no edges") +
                 ("" if ok else " — the far waterline is required; place the plane by hand"))


# ---------------------------------------------------------------------------
# Building the quad
# ---------------------------------------------------------------------------
def _default_quad(frame_shape: tuple[int, int], length_ft: float,
                  width_ft: float) -> plane_mod.PlaneSession:
    s = plane_mod.PlaneSession(frame_shape=(int(frame_shape[0]), int(frame_shape[1])),
                               length_ft=float(length_ft), width_ft=float(width_ft))
    s.fit_to_frame()
    return s


def mirror_session(session: plane_mod.PlaneSession) -> plane_mod.PlaneSession:
    """A copy of the quad mirrored left-right in the image, with the corner tags kept **metric**.

    ``fit_to_frame`` hard-codes the dock on the left (X=0 at 6 % of the width).  On a clip whose dock
    is on the right that default is mirrored, and at L1/L0 — where X=0 is the quad's own edge rather
    than a detected line — the mirror *is* the seed: every distance would come out measured towards
    the dock.  No gate can see that (P4 still passes: the foreshortening is symmetric), so the flip has
    to be caught before the quad is used.
    """
    w = float(int(session.frame_shape[1]) - 1)
    out = plane_mod.PlaneSession(frame_shape=(int(session.frame_shape[0]),
                                              int(session.frame_shape[1])),
                                 length_ft=float(session.length_ft),
                                 width_ft=float(session.width_ft))
    for tag in CORNER_TAGS:
        p = session.corners.get(tag)
        # The tags do NOT swap: corner (0,0) stays the point that means (0,0) — it has simply moved to
        # the other side of the image, which is the whole point.  Swapping them as well would leave the
        # mirrored quad's origin back where it started (measured: X=0 at x~345 on a 960-px frame).
        out.corners[tag] = None if p is None else (float(w - float(p[0])), float(p[1]))
    return out


def x_scale_from_extent(edge: Any, p00: np.ndarray, vp: np.ndarray, *, anchor_ft: float,
                        frame_shape: tuple[int, int], margin: float = 0.12) -> tuple[float, str] | None:
    """px per foot along X at the dock edge, from where the detected far waterline runs out of shot.

    The X scale is the one quantity the three lines cannot measure: the two long waterlines fix the
    width and the vanishing point, not how many pixels a foot along the pool subtends.  What *is*
    observable is how far the far waterline extends, and when that extent runs off the side of the
    frame the water continues past the shot — so the operator's anchor should sit at that end, which is
    exactly where they would drag the far edge.  Measured on video004 seg01: the session's anisotropy
    prior put the far edge 237 px from the dock while the detected far waterline runs 872 px, i.e. a
    quad covering a quarter of the visible pool.

    Returns ``(px_per_ft_x, why)``, or ``None`` when the extent does not reach a frame edge (a water
    mask that simply faded out mid-pool says nothing about where the pool ends).
    """
    if edge is None or not getattr(edge, "found", False) or edge.fit is None:
        return None
    ends = [np.asarray(p, dtype=float) for p in edge.fit["endpoints"]]
    h, w = float(int(frame_shape[0])), float(int(frame_shape[1]))
    near_border = any(min(e[0], w - 1 - e[0]) <= margin * w or min(e[1], h - 1 - e[1]) <= margin * h
                      for e in ends)
    if not near_border:
        return None
    far_end = max(ends, key=lambda p: float(np.linalg.norm(p - p00)))
    span = float(np.linalg.norm(np.asarray(vp, float) - np.asarray(p00, float)))
    if span < 1e-6:
        return None
    s_ext = float((far_end - p00) @ (np.asarray(vp, float) - p00)) / (span * span)
    if not (1e-3 < s_ext < 0.9) or anchor_ft <= 0:
        return None
    px_per_ft = s_ext * span / (float(anchor_ft) * (1.0 - s_ext))
    return px_per_ft, (f"the far waterline runs out of shot at {far_end.round(0).tolist()}, so the "
                       f"anchor sits there ({s_ext:.3f} of the way to the vanishing point) — "
                       f"{px_per_ft:.1f} px/ft along X at the dock edge")


def _ground_truth_line(edges: Any, name: str):
    """The fitted line for one end — or ``None`` when it is a FRAME CUT, not a waterline.

    A ``frame_cut`` is the honest boundary of what is visible: a vertical at the picture's edge,
    invented so a quad could still be closed.  It must never enter the FIT.  Measured on
    video004 seg01: with the cut (a fabricated 90 deg line at x=0) in the homography, the
    extrapolated X=const edges came out at 16.8 and 20.7 deg while the detected dock waterline is
    **12.2 deg** — the mapping was sweeping from the real edge towards the invented one.  Three real
    waterlines fix the geometry; the fourth edge is then extrapolated from it (order 2026-10-27).
    """
    e = getattr(edges, name, None)
    if e is None or not getattr(e, "found", False):
        return None
    if getattr(e, "status", "") == "frame_cut":
        return None
    return line_of(e)


#: How narrow the constructed opposite short edge may get before it is widened (px).  A vertical
#: bridging the two long edges needs the far waterline to actually END somewhere: when its visible end
#: sits on top of the dock corner, the quad would collapse and the homography with it.
MIN_OPP_WIDTH_PX = 30.0


def seed_from_edges(edges: Any, *, length_ft: float, width_ft: float,
                    frame_shape: tuple[int, int],
                    current: plane_mod.PlaneSession | None = None,
                    dock_side: str | None = None,
                    dock_hint_x: float | None = None) -> SeedResult:
    """Place the quad from the detected edges, or refuse and say why.

    Falls back along the ladder of ``docs/PLAN_WATERLINE_SEED.md`` §9 and records which level it used.
    A refused result never carries corners, so a caller cannot accidentally apply a half-decided
    plane.

    ``dock_hint_x`` is the image x where the clip's dog *starts* (the median of the first accepted
    tail-set detections).  It is used for one thing: when X=0 has to come from the quad rather than
    from a detected line, a mirror-flipped quad is the worst possible default, and the hint is what
    catches it.
    """
    res = SeedResult(anchor_ft=float(length_ft), width_ft=float(width_ft))
    far = _ground_truth_line(edges, "far")
    near = _ground_truth_line(edges, "near")
    near_raw = near          # the DETECTED near waterline, kept for anything that reasons about the
                             # vanishing point: the wall drop must not move the far edge
    dock = _ground_truth_line(edges, "dock")
    near_dropped = False
    if near is not None and NEAR_WALL_DROP_PX:
        # The near long edge the detector returns is the coping (the top of the wall); the water
        # surface meets that wall below it, so the quad's near edge is dropped by that much.
        near = (near[0] + np.array([0.0, float(NEAR_WALL_DROP_PX)]), near[1])
        near_dropped = True
    side = dock_side or getattr(edges, "dock_side", "unknown")
    res.dock_side = side
    # The orientation control beats the detector's vote (operator order 2026-10-21): *"+x = 0 snaps to
    # the right or left waterline depending on the orientation selected by the user"*.  Both short ends
    # are detected now, so honouring the choice costs nothing and a clip the detector got backwards
    # stops needing a hand-drag.
    if dock_side in ("left", "right"):
        chosen = line_of(getattr(edges, dock_side, None))
        if chosen is None:
            res.notes.append(f"the {dock_side} short-end waterline was not found — X=0 falls back to "
                             f"whichever end the detector chose")
        else:
            dock = chosen
            res.dock_side = dock_side
            if getattr(edges, "dock_side", "unknown") not in (dock_side, "unknown"):
                res.notes.append(f"X=0 snapped to the {dock_side} waterline because that is the "
                                 f"orientation selected; the detector had the dock on the "
                                 f"{edges.dock_side}")
    # --- the pool's OTHER short end: a VERTICAL line bridging the two long waterlines ----------
    # Order 2026-10-28: *"simplify the opposite short edge rule immensely: the opposite short edge of
    # the pool should be placed as a vertical line bridging the near and far long edges."*  So it is
    # one x and two intersections — no vanishing point, no direction borrowed from the dock edge, no
    # detector line whose status might be wrong (a border line reported `observed` was making this
    # side a vertical at x = 0, which is the picture, not the pool).
    opp_side = {"left": "right", "right": "left"}.get(side)
    cut = [n for n in ("left", "right")
           if getattr(getattr(edges, n, None), "status", "") == "frame_cut"]
    opp = None
    e_opp = getattr(edges, opp_side, None) if opp_side else None
    p_dock = intersect(dock, far) if (dock is not None and far is not None) else None
    far_ends = [np.asarray(p, float) for p in (getattr(edges.far, "fit", None) or {})
                .get("endpoints", [])]
    if e_opp is not None and p_dock is not None and far_ends:
        # The x is where the far waterline ends away from the dock: the far end of the water we can
        # see.  (Not a detected short-side line — the vertical is constructed, so its status cannot
        # matter.)
        far_end = max(far_ends, key=lambda p: float(np.linalg.norm(p - p_dock)))
        cand = float(far_end[0])
        if abs(cand - float(p_dock[0])) >= MIN_OPP_WIDTH_PX:
            opp_x = cand
            opp = (np.array([cand, 0.0]), np.array([0.0, 1.0]))      # vertical: x = cand
            res.notes.append(
                f"the opposite short edge is the VERTICAL x = {cand:.0f} bridging the far and near "
                f"waterlines (the far end of the visible water, {abs(float(p_dock[0]) - cand):.0f} px "
                f"from the dock corner)")
        else:
            res.notes.append(
                f"the far waterline ends only {abs(cand - float(p_dock[0])):.0f} px from the dock "
                f"corner — too narrow to bridge, so X = 35 is placed at the minimum span")
            opp_x = float(p_dock[0]) + (-MIN_OPP_WIDTH_PX if cand < float(p_dock[0])
                                        else MIN_OPP_WIDTH_PX)
            opp = (np.array([opp_x, 0.0]), np.array([0.0, 1.0]))
    if cut:
        res.notes.append(
            f"the {' and '.join(cut)} end is a FRAME CUT (the water leaves the picture there), not a "
            f"waterline — excluded from the fit; the opposite side is the constructed vertical")

    if near_dropped:
        res.notes.append(
            f"the quad's near edge sits {NEAR_WALL_DROP_PX:g} px BELOW the detected near waterline: "
            f"that line is the top of the pool wall, and the water meets the wall below it")

    res.gates["S1"] = gate_edges_found(edges)
    res.gates["S2"] = gate_mask(edges)

    if far is None:
        res.reason = "no far waterline detected"
        res.notes.append("nothing was written — the plane was left exactly as it was")
        return res

    cur = current
    if cur is not None and cur.placed and dock is None and dock_hint_x is not None:
        cx = 0.5 * (float(frame_shape[1]) - 1.0)
        if (float(dock_hint_x) < cx) != (float(cur.corners["00"][0]) < cx):
            cur = mirror_session(cur)
            res.notes.append(
                f"mirrored the current quad: the clip's first detections are on the "
                f"{'left' if float(dock_hint_x) < cx else 'right'} of the frame while X=0 sat on the "
                f"other side — without the dock waterline there is nothing else to catch this")
    cur_c = ({t: np.asarray(cur.corners[t], float) for t in CORNER_TAGS}
             if cur is not None and cur.placed else None)

    if far is not None and near is not None and dock is not None and opp is not None:
        res.level = "L4"
    elif far is not None and near is not None and dock is not None:
        res.level = "L3"
    elif far is not None and dock is not None:
        res.level = "L2"
    elif far is not None and near is not None:
        res.level = "L1"
    else:
        res.level = "L0"

    rho, rho_why = px_per_ft_ratio(cur)
    res.rho = rho
    res.notes.append(rho_why)

    p00 = p01 = vp = None
    s_anchor = 0.0
    px_per_ft_y = float("nan")
    px_per_ft_x = float("nan")

    if res.level == "L4":
        # Four detected lines -> four corners.  No vanishing point, no anchor, no px/ft: the quad is
        # the visible water's own quadrilateral, and the metric layer stays untouched.
        p00, p01 = intersect(dock, far), intersect(dock, near)
        p10, p11 = intersect(opp, far), intersect(opp, near)
        missing = [n for n, p in (("00", p00), ("01", p01), ("10", p10), ("11", p11))
                   if p is None]
        if missing:
            res.reason = (f"the four detected lines do not meet pairwise at corner(s) {missing} "
                          f"— a short edge is parallel to a long one")
            return res
        corners = {"00": p00, "01": p01, "10": p10, "11": p11}
        stat = {e.name: e.status for e in getattr(edges, "all_edges", ())}
        res.notes.append(
            f"L4: the quad's four edges ARE the four detected waterlines — far/near long, "
            f"dock side {side} ({stat.get(side, '?')}), opposite side {opp_side} "
            f"({stat.get(opp_side, '?')})")
        cut = [n for n in (side, opp_side) if stat.get(n) == "frame_cut"]
        if cut:
            res.notes.append(
                f"the {' and '.join(cut)} edge is CUT AT THE FRAME: the water leaves the picture there, "
                f"so that side is the boundary of the visible water, not a measured pool wall")
        res.gates["S3"] = _gate(True, "not used at L4 — four lines determine the quad; "
                                     "no vanishing point or px/ft is involved")
        res.geometry = {"route": "four_lines", "dock": side, "opposite": opp_side,
                        "corner_source": "line intersections",
                        "corners": {k: [round(float(v), 1) for v in p] for k, p in corners.items()}}

    elif res.level == "L3":
        p00 = intersect(dock, far)
        p01 = intersect(dock, near)
        vp = intersect(far, near_raw)
        if p00 is None or p01 is None or vp is None:
            res.gates["S3"] = gate_geometry(p00, p01, vp, 0.0, frame_shape, side)
            res.reason = "S3 failed: the three detected lines are not pairwise concurrent"
            return res
        # The X scale comes from the DETECTED width, not the wall-dropped one: the drop is a
        # placement of the near edge, and letting it rescale the pool would move the far edge too.
        p01_raw = intersect(dock, near_raw)
        if p01_raw is None:
            p01_raw = p01
        px_per_ft_y = float(np.linalg.norm(p01_raw - p00)) / max(float(width_ft), 1e-6)
        # X scale: the detected far extent when the water runs out of shot, otherwise the session's
        # anisotropy prior.  See x_scale_from_extent for why the extent wins when it applies.
        _ext = x_scale_from_extent(edges.far, p00, vp, anchor_ft=float(length_ft),
                                   frame_shape=frame_shape)
        if _ext is not None:
            px_per_ft_x, _why_x = _ext
            res.notes.append(_why_x)
        else:
            px_per_ft_x = rho * px_per_ft_y
            res.notes.append(f"px/ft along X from the session's anisotropy prior "
                             f"({rho:.2f}x the measured {px_per_ft_y:.1f} px/ft across)")
        anchor_px, s_anchor, _A = projective_anchor(p00, vp, px_per_ft_x, float(length_ft))
        res.gates["S3"] = gate_geometry(p00, p01, vp, s_anchor, frame_shape, side)
        if not res.gates["S3"]["pass"]:
            res.reason = "geometry gate S3 failed: " + res.gates["S3"]["detail"]
            return res
        dock_dir = np.asarray(dock[1], float)
        p11 = intersect(near, (anchor_px, dock_dir))
        if p11 is None:
            res.reason = "the anchor corner could not be projected onto the near waterline"
            return res
        corners = {"00": p00, "10": anchor_px, "11": p11, "01": p01}
        res.geometry = {"vanishing_point": vp.round(2).tolist(), "anchor_fraction": s_anchor}

    elif res.level == "L2":
        p00 = intersect(dock, far)
        if p00 is None:
            res.reason = "the dock and far lines are parallel"
            return res
        if cur_c is None:
            res.reason = ("the near waterline was not found and there is no placed quad to take the "
                          "width and the X-direction convergence from — find the near edge or place "
                          "the plane by hand")
            return res
        p01 = p00 + (cur_c["01"] - cur_c["00"])
        # The near line is the current quad's near edge moved onto p01, so the X-direction vanishing
        # point is where that line meets the DETECTED far line.  (Using the dock's direction here
        # would be the perpendicular axis and would put the vanishing point in the wrong place.)
        vp = intersect(far, (p01, np.asarray(cur_c["11"] - cur_c["01"], float)))
        if vp is None:
            res.reason = ("no X-direction vanishing point: the current quad's near edge is parallel "
                          "to the detected far waterline")
            return res
        px_per_ft_y = float(np.linalg.norm(p01 - p00)) / max(float(width_ft), 1e-6)
        # X scale: the detected far extent when the water runs out of shot, otherwise the session's
        # anisotropy prior.  See x_scale_from_extent for why the extent wins when it applies.
        _ext = x_scale_from_extent(edges.far, p00, vp, anchor_ft=float(length_ft),
                                   frame_shape=frame_shape)
        if _ext is not None:
            px_per_ft_x, _why_x = _ext
            res.notes.append(_why_x)
        else:
            px_per_ft_x = rho * px_per_ft_y
            res.notes.append(f"px/ft along X from the session's anisotropy prior "
                             f"({rho:.2f}x the measured {px_per_ft_y:.1f} px/ft across)")
        anchor_px, s_anchor, _A = projective_anchor(p00, vp, px_per_ft_x, float(length_ft))
        res.gates["S3"] = gate_geometry(p00, p01, vp, s_anchor, frame_shape, side)
        if not res.gates["S3"]["pass"]:
            res.reason = "geometry gate S3 failed: " + res.gates["S3"]["detail"]
            return res
        p11 = anchor_px + (1.0 - s_anchor) * (p01 - p00)
        res.notes.append("near waterline not found: the near edge is the current quad's width, "
                         "translated onto the detected dock line")
        corners = {"00": p00, "10": anchor_px, "11": p11, "01": p01}
        res.geometry = {"vanishing_point": vp.round(2).tolist(), "anchor_fraction": s_anchor}

    elif res.level == "L1":
        if cur_c is None:
            res.reason = ("the dock edge was not found and there is no placed quad to inherit X=0 "
                          "from")
            return res
        vp = intersect(far, near_raw)
        if vp is None:
            res.reason = "the far and near lines are parallel: no vanishing point"
            return res
        p00 = foot_on_line(cur_c["00"], far)
        dock_dir = np.asarray(cur_c["01"] - cur_c["00"], float)
        # Width from the DETECTED near waterline, not from the quad: it is the only metric fact in the
        # frame.  (Measured on video004 seg01: inheriting the default quad's width instead put the
        # near edge at y=691 on a 600-px frame, i.e. 250 px below the near waterline the detector had
        # already found.)
        p01 = intersect(near, (p00, dock_dir))
        if p01 is None:
            p01 = p00 + dock_dir
            res.notes.append("near waterline is parallel to the dock edge — width taken from the quad")
        # Scale from the DETECTED near waterline (the wall drop is placement, not a rescale).
        p01_raw = intersect(near_raw, (p00, dock_dir))
        px_per_ft_y = (float(np.linalg.norm((p01_raw if p01_raw is not None else p01) - p00))
                       / max(float(width_ft), 1e-6))
        # X scale: the detected far extent when the water runs out of shot, otherwise the session's
        # anisotropy prior.  See x_scale_from_extent for why the extent wins when it applies.
        _ext = x_scale_from_extent(edges.far, p00, vp, anchor_ft=float(length_ft),
                                   frame_shape=frame_shape)
        if _ext is not None:
            px_per_ft_x, _why_x = _ext
            res.notes.append(_why_x)
        else:
            px_per_ft_x = rho * px_per_ft_y
            res.notes.append(f"px/ft along X from the session's anisotropy prior "
                             f"({rho:.2f}x the measured {px_per_ft_y:.1f} px/ft across)")
        anchor_px, s_anchor, _A = projective_anchor(p00, vp, px_per_ft_x, float(length_ft))
        res.gates["S3"] = gate_geometry(p00, p01, vp, s_anchor, frame_shape, side)
        if not res.gates["S3"]["pass"]:
            res.reason = "geometry gate S3 failed: " + res.gates["S3"]["detail"]
            return res
        p11 = intersect(near, (anchor_px, dock_dir))
        if p11 is None:
            p11 = anchor_px + dock_dir
        res.notes.append("dock edge not found: X=0 is the current quad's dock edge moved onto the "
                         "far waterline — treat the origin as a default, not a measurement")
        corners = {"00": p00, "10": anchor_px, "11": p11, "01": p01}
        res.geometry = {"vanishing_point": vp.round(2).tolist(), "anchor_fraction": s_anchor}

    else:  # L0 — only the far waterline
        if cur_c is None:
            res.reason = ("only the far waterline was found and there is no placed quad: a far line "
                          "alone cannot place a rectangle")
            return res
        d = np.asarray(far[1], float)
        cur_d = cur_c["10"] - cur_c["00"]
        cur_n = float(np.linalg.norm(cur_d))
        if cur_n < 1e-6:
            res.reason = "the current quad has a degenerate far edge"
            return res
        ang_d = float(np.degrees(np.arctan2(d[1], d[0])))
        ang_c = float(np.degrees(np.arctan2(cur_d[1], cur_d[0])))
        R = _rotation(ang_d - ang_c)
        origin = foot_on_line(cur_c["00"], far)
        corners = {t: origin + R @ (cur_c[t] - cur_c["00"]) for t in CORNER_TAGS}
        p00 = corners["00"]
        res.notes.append("only the far waterline was found: the current quad was rotated and moved "
                         "onto it — X=0, the width and the anchor are all unchanged defaults")
        res.gates["S3"] = _gate(True, "not applicable at L0 (one line, no intersections)")
        res.geometry = {}

    # --- quad-level gates, identical at every level -------------------------
    res.gates["S4"] = gate_quad(corners, length_ft=float(length_ft), width_ft=float(width_ft),
                                frame_shape=frame_shape)
    if res.level == "L4":
        # S5 compares the quad's area with the frame-filling default quad's.  At L4 the quad is the
        # visible water, which on a side-on view is a wide thin band — a legitimate shape that the
        # default's area band was never written for, so it is reported and not enforced.
        res.gates["S5"] = _gate(True, "not applied at L4 — the quad is the visible water, not a fit "
                                      "to the frame-filling default", warn=True)
        # P1-P4 encode a down-the-pool geometry (far edge shorter than the near one).  On a side-on
        # view they can fire on a correct quad, so at L4 they are advisory rather than a refusal.
        if not res.gates["S4"]["pass"]:
            res.gates["S4"]["warn"] = True
            res.notes.append("the P-gates are advisory at L4: " + res.gates["S4"]["detail"][:160])
    else:
        ref = _default_quad(frame_shape, float(length_ft), float(width_ft))
        res.gates["S5"] = gate_area(corners, quad_area({t: np.asarray(ref.corners[t], float)
                                                        for t in CORNER_TAGS}))
    # S6 is a WARNING, not a refusal: at L1/L0 the plane is still usable, but its origin is inherited
    # rather than measured, and the operator has to drag the dock-side edge onto the dock lip.
    res.gates["S6"] = (_gate(True, "X=0 is a measured intersection (dock waterline detected)")
                       if dock is not None else
                       _gate(True, "X=0 is the quad's own edge, NOT a measurement — drag the "
                                   "dock-side edge onto the dock lip", warn=True))
    bad = res.failing()
    if bad:
        res.reason = "refused: " + ", ".join(f"{k} — {res.gates[k]['detail']}" for k in bad)
        res.notes.append("nothing was written — the plane was left exactly as it was")
        return res

    res.corners = {t: (float(corners[t][0]), float(corners[t][1])) for t in CORNER_TAGS}
    res.refused = False
    res.px_per_ft_x = px_per_ft_x
    res.px_per_ft_y = px_per_ft_y
    return res


def build_block(result: SeedResult, edges: Any, *, upstream: str, file_sha256: str) -> dict[str, Any]:
    """The ``waterline_seed`` block stored in the pool document.

    Enough to answer "how was this plane placed?" years later without the clip: which algorithm and
    which bytes of it, the window, the thresholds, every edge's fit quality, the inherited px/ft ratio
    and the gates.  ``placement`` is flipped to ``auto+edited`` by the first drag after a seed.
    """
    block = {"upstream": upstream, "file_sha256": file_sha256,
             "placement": "auto", "seed": result.as_dict(),
             "detection": edges.as_dict() if hasattr(edges, "as_dict") else {}}
    return block
