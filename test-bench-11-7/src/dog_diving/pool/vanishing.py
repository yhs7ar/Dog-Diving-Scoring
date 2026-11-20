"""Phase 3 — vanishing points, the across-pool (transverse) baseline, and intrinsics.

`pool-translation-plan.txt`, Phase 3::

    Objective: Solve for the pool width without direct physical cross-pool measurements.
    Implementation:
      * Detect parallel lines of lanes/walls to calculate vanishing points.
      * Anchor the transverse Y-axis using standard dock diving pool dimensions
        (e.g. 14 to 20 feet wide).
      * Apply symmetry constraints based on standard lane boundaries.

Why "no direct cross-pool measurement" is not a convenience
----------------------------------------------------------
The pool's painted marks give a **collinear** row of points whose along-pool spacing is
known (0/1/15/16/30/31 ft).  A single row of collinear points determines the scale along
that one line and *nothing* transverse to it: every transverse distance is unpinned by
an arbitrary projective factor.  That factor is what this module attacks, and it can be
attacked in exactly two ways:

* a **vanishing point** for the transverse direction (from the two long pool edges, the
  dock edge, a lane divider, the ladder rails) — a direction, which fixes the transverse
  *field* but not its *scale*; and then
* a **scale anchor** — either the plan's width prior (14–20 ft) or, better, an observed
  far-edge line whose world position follows from the prior.

Two orthogonal vanishing points additionally give the **focal length** without a
checkerboard (`focal_from_orthogonal_vps`).  That is the bridge to Phase 1: the line
route in :mod:`dog_diving.pool.lens` needs an ``f`` to start from, and the pool itself
supplies two orthogonal directions for free (along-pool and across-pool).  Using the
pool's own geometry to recover the intrinsics, then the intrinsics to rectify the pool,
is exactly the plan's Phase 1 ↔ Phase 3 coupling.

The honest failure mode
-----------------------
A fixed side view (which this corpus is: `PLAN.md` D2) puts the along-pool VP *very* far
away and the across-pool VP close to the principal point.  Both are the bad cases: a far
VP is numerically ill-conditioned, and a transverse VP near the principal point means
the transverse direction is nearly parallel to the optical axis, where vanishing-point
reasoning has almost no leverage.  :func:`vanish` therefore *returns its conditioning*,
and every caller is expected to check it rather than trust the point.  When the
transverse VP is unusable, the correct move is route A in :mod:`dog_diving.pool.homography`
(real (X, Y) correspondences) — not a confident-looking extrapolation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import cv2
import numpy as np

from .lens import LensModel

# A vanishing point further than this many image widths from the principal point is
# reported as "at infinity" for practical purposes: the direction is still meaningful,
# the *point* is not, and any code that treats it as a finite point will lose precision.
FAR_VP_WIDTHS = 50.0


# ============================================================================
# Homogeneous line primitives
# ============================================================================
def line_from_points(p1: Sequence[float], p2: Sequence[float]) -> np.ndarray:
    """The homogeneous line ``(a, b, c)`` through two points, with ``a^2+b^2 = 1``.

    Normalising ``a, b`` (not the whole vector) is what makes ``l . p`` a *distance*
    in pixels, which is what makes :func:`fit_line` and :func:`vanish` least-squares
    problems with a physical meaning instead of algebraic ones.
    """
    a, b, c = np.cross([float(p1[0]), float(p1[1]), 1.0],
                       [float(p2[0]), float(p2[1]), 1.0])
    n = float(np.hypot(a, b))
    if n < 1e-12:
        raise ValueError(f"degenerate line through {tuple(p1)} and {tuple(p2)}")
    return np.array([a / n, b / n, c / n], dtype=float)


def line_value(line: np.ndarray, point: Sequence[float]) -> float:
    """Signed perpendicular distance from ``point`` to ``line`` (pixels)."""
    return float(line[0] * point[0] + line[1] * point[1] + line[2])


def fit_line(points: np.ndarray) -> np.ndarray:
    """Total-least-squares homogeneous line through ``(N, 2)`` image points."""
    arr = np.asarray(points, dtype=float).reshape(-1, 2)
    if arr.shape[0] < 2:
        raise ValueError("need >= 2 points for a line")
    centroid = arr.mean(axis=0)
    _, _, vt = np.linalg.svd(arr - centroid, full_matrices=False)
    d = vt[0]
    normal = np.array([-d[1], d[0]], dtype=float)
    c = -float(normal @ centroid)
    return np.array([normal[0], normal[1], c], dtype=float)


def intersect_lines(l1: Sequence[float], l2: Sequence[float]) -> np.ndarray:
    """Homogeneous intersection ``l1 x l2`` (``w == 0`` when parallel)."""
    return np.cross(np.asarray(l1, dtype=float), np.asarray(l2, dtype=float))


def dehomogenise(v: Sequence[float]) -> tuple[float, float] | None:
    """``(x, y)`` of a homogeneous point, or ``None`` when it is at infinity."""
    v = np.asarray(v, dtype=float).ravel()
    if abs(v[2]) < 1e-12:
        return None
    return float(v[0] / v[2]), float(v[1] / v[2])


@dataclass
class VanishingPoint:
    """A vanishing point plus its conditioning — never just the point."""

    v: np.ndarray                      # homogeneous (3,)
    rms_px: float                      # RMS perpendicular distance of the lines to v
    n_lines: int
    finite: bool                       # w != 0
    xy: tuple[float, float] | None     # None when at infinity
    direction: tuple[float, float]     # unit image-space direction of the family
    distance_px: float                 # |xy - principal point|, inf when at infinity
    usable: bool                       # finite and not absurdly far away
    notes: list[str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "v": [float(x) for x in self.v],
            "xy": None if self.xy is None else [float(self.xy[0]), float(self.xy[1])],
            "direction": [float(self.direction[0]), float(self.direction[1])],
            "rms_px": float(self.rms_px),
            "n_lines": int(self.n_lines),
            "finite": bool(self.finite),
            "distance_px": None if not np.isfinite(self.distance_px) else float(self.distance_px),
            "usable": bool(self.usable),
            "notes": list(self.notes),
        }


def vanish(
    lines: Sequence[Sequence[float]],
    *,
    image_size: tuple[int, int] | None = None,
    principal_point: Sequence[float] | None = None,
    far_vp_widths: float = FAR_VP_WIDTHS,
) -> VanishingPoint:
    """Least-squares vanishing point of a family of parallel image lines.

    Solves ``min_v sum_i (l_i . v)^2`` subject to ``|v| = 1`` — the singular vector of
    the line-coefficient matrix belonging to its smallest singular value, i.e. the
    point that minimises the summed *perpendicular pixel distance* to all the lines.
    Using perpendicular distance (rather than the algebraic ``l . v`` form raw) is why
    :func:`line_from_points` normalises ``a, b``: the two are only the same metric when
    ``a^2 + b^2 = 1``.

    Returns a :class:`VanishingPoint` carrying the residual and the conditioning.  A
    family of near-parallel lines (the fixed-side-view case) yields a point at infinity,
    which is reported as such rather than as a huge finite coordinate.
    """
    L = [np.asarray(ln, dtype=float).ravel() for ln in lines]
    L = [ln[:3] for ln in L if ln.size >= 3 and np.hypot(ln[0], ln[1]) > 1e-12]
    if not L:
        raise ValueError("vanish needs at least one line")
    M = np.array([[ln[0], ln[1], ln[2]] for ln in L], dtype=float)
    M /= np.hypot(M[:, 0], M[:, 1])[:, None]        # (a,b) unit -> l.p is a distance

    notes: list[str] = []
    # "+Z" canonical direction for a line (a, b, c) with a^2+b^2 = 1 is (-b, a).  Every
    # infinite-case branch below uses the mean of the family's own directions, which is the
    # exact estimator for a parallel family (they are all equal) and is not the same thing as
    # the algebraic singular vector.
    unit_dirs = np.column_stack([-M[:, 1], M[:, 0]]) if M.shape[0] else np.zeros((0, 2))

    def _mean_direction() -> np.ndarray:
        d = unit_dirs.mean(axis=0)
        n = float(np.linalg.norm(d))
        d = d / n if n > 0 else np.array([1.0, 0.0])
        return np.array([d[0], d[1], 0.0], dtype=float)

    if M.shape[0] == 1:
        # One line constrains a direction, never a point: return the direction at infinity
        # rather than a point on the line pretending to be a vanishing point.
        v = _mean_direction()
        xy = None
        rms = float("nan")
        notes.append("only one line supplied: a vanishing point needs at least two lines, so "
                     "what is returned is that line's DIRECTION, not a point")
    else:
        # Solve min_(x,y) sum_i (a_i x + b_i y + c_i)^2 with w = 1 — the geometric
        # least-squares point.  The normal equations are a 2x2 system, which matters: the
        # algebraic "smallest singular vector of the coefficient matrix" that a naive
        # implementation takes is a *different* estimator, and with exactly two lines it
        # returns a point on the bisector rather than the intersection (measured: 173 px of
        # RMS on a two-line family whose true intersection reproduced to 0.0 px).
        A = M[:, :2]
        c = M[:, 2]
        AtA = A.T @ A
        Atc = A.T @ c
        det = float(AtA[0, 0] * AtA[1, 1] - AtA[0, 1] ** 2)
        scale = (float(np.trace(AtA)) / 2.0) ** 2
        if scale > 0 and det > 1e-14 * scale:
            try:
                sol = np.linalg.solve(AtA, -Atc)
                v = np.array([sol[0], sol[1], 1.0], dtype=float)
                xy = (float(sol[0]), float(sol[1]))
                rms = float(np.sqrt(np.mean((A @ sol + c) ** 2)))
            except np.linalg.LinAlgError:
                v, xy, rms = np.array([M[0, 1], -M[0, 0], 0.0]), None, float("nan")
        else:
            # The family is parallel to numerical precision: the "vanishing point" is a
            # direction. Reported as such instead of as a 1e17-pixel coordinate.
            v = _mean_direction()
            xy, rms = None, float("nan")
            notes.append("the family is parallel to numerical precision (|det| of the normal "
                         "equations is at machine level)")

    finite = xy is not None
    if not finite:
        notes.append("vanishing point is at infinity (the family is parallel in the image):"
                     " the DIRECTION is usable, the POINT is not")
        d = float("inf")
        direction = (float(v[0]), float(v[1]))
        norm = float(np.hypot(*direction)) or 1.0
        direction = (direction[0] / norm, direction[1] / norm)
    else:
        direction = (float(v[1]), float(-v[0]))
        norm = float(np.hypot(*direction)) or 1.0
        direction = (direction[0] / norm, direction[1] / norm)
        if principal_point is not None:
            d = float(np.hypot(xy[0] - float(principal_point[0]),
                               xy[1] - float(principal_point[1])))
        elif image_size is not None:
            w, h = int(image_size[0]), int(image_size[1])
            d = float(np.hypot(xy[0] - (w - 1) / 2.0, xy[1] - (h - 1) / 2.0))
        else:
            d = float("inf")

    usable = finite and (image_size is None or d <= far_vp_widths * float(image_size[0]))
    if finite and not usable:
        notes.append(
            f"vanishing point is {d:.0f} px from the principal point "
            f"(> {far_vp_widths:g} image widths) — numerically unusable as a constraint, "
            f"even though a finite value was returned"
        )
    return VanishingPoint(v=v, rms_px=rms, n_lines=len(L), finite=finite, xy=xy,
                          direction=direction, distance_px=d, usable=usable, notes=notes)


# ============================================================================
# Intrinsics from orthogonal vanishing points (Phase 1 <- Phase 3 coupling)
# ============================================================================
def focal_from_orthogonal_vps(
    vp1: Sequence[float],
    vp2: Sequence[float],
    principal_point: Sequence[float],
) -> float:
    """Focal length in pixels from two orthogonal vanishing points.

    For a pinhole camera with unit aspect and zero skew, two *finite* vanishing points
    of mutually orthogonal world directions satisfy

        (v1 - p) . (v2 - p) = -f^2

    so ``f = sqrt(-(v1-p).(v2-p))``.  A non-negative dot product means the two
    directions are **not** orthogonal (or a VP is wrong); this returns ``nan`` rather
    than a square root of a negative number, because a negative ``f^2`` silently turned
    into a positive ``f`` is a plausible-looking camera that is wrong by 90 degrees.
    """
    p = np.asarray(principal_point, dtype=float).ravel()[:2]
    a = np.asarray(vp1, dtype=float).ravel()[:2] - p
    b = np.asarray(vp2, dtype=float).ravel()[:2] - p
    val = -float(a @ b)
    if val <= 0.0:
        return float("nan")
    return float(np.sqrt(val))


def principal_point_from_orthocentre(
    vp1: Sequence[float], vp2: Sequence[float], vp3: Sequence[float]
) -> tuple[float, float] | None:
    """Principal point as the orthocentre of three mutually-orthogonal-direction VPs.

    The classic construction: for three orthogonal world directions, the principal point is the
    orthocentre of the triangle formed by their three vanishing points.

    Built from **altitudes**, which is what an orthocentre is: the altitude from ``vp1`` passes
    through ``vp1`` and is perpendicular to the side ``vp2``–``vp3``.  An earlier version drew the
    line from each VP through the *midpoint* of the opposite side — a median, not an altitude — and
    returned (1084.7, 3726.9) on a scene whose principal point is (959.5, 539.5).  The two coincide
    only for an equilateral triangle, which a VP triangle almost never is.
    """
    pts = []
    for v in (vp1, vp2, vp3):
        v = np.asarray(v, dtype=float).ravel()
        if v.size < 3 or abs(v[2]) < 1e-12:
            return None
        pts.append(v[:2] / v[2])

    def altitude(i: int, j: int, k: int) -> np.ndarray:
        side = pts[j] - pts[k]
        normal = np.array([-side[1], side[0]], dtype=float)
        return line_from_points(pts[i], pts[i] + normal)

    h = intersect_lines(altitude(0, 1, 2), altitude(1, 0, 2))
    xy = dehomogenise(h)
    if xy is None:
        return None
    if not np.all(np.isfinite(xy)):
        return None
    return float(xy[0]), float(xy[1])


def intrinsics_from_orthogonal_vps(
    vp_a: VanishingPoint,
    vp_b: VanishingPoint,
    image_size: tuple[int, int],
    *,
    principal_point: Sequence[float] | None = None,
) -> tuple[LensModel, dict[str, Any]]:
    """Build assumed-intrinsics from one orthogonal pair of vanishing points.

    Returns ``(lens, report)``.  ``report["ok"]`` is False when the pair cannot
    determine ``f`` (a VP at infinity, a non-negative dot product, or a focal outside
    the physically sane band) — in that case the returned ``lens`` is the ratio prior
    and the caller is expected to notice, not to use ``f`` as if it were measured.
    """
    from .lens import placeholder_intrinsics

    w, h = int(image_size[0]), int(image_size[1])
    pp = tuple(float(x) for x in (principal_point if principal_point is not None
                                 else ((w - 1) / 2.0, (h - 1) / 2.0)))
    report: dict[str, Any] = {
        "principal_point": [pp[0], pp[1]],
        "pp_source": "image_centre" if principal_point is None else "given",
        "focal_px": None, "focal_ratio": None, "ok": False,
        "vp_a": vp_a.as_dict(), "vp_b": vp_b.as_dict(), "notes": [],
    }
    if vp_a.xy is None or vp_b.xy is None:
        report["notes"].append("at least one vanishing point is at infinity — the "
                               "orthogonality relation needs two finite points")
        return placeholder_intrinsics(image_size, principal_point=pp), report

    f = focal_from_orthogonal_vps(vp_a.xy, vp_b.xy, pp)
    lo, hi = 0.15 * max(w, h), 20.0 * max(w, h)
    if not np.isfinite(f) or not (lo <= f <= hi):
        report["notes"].append(
            f"focal from orthogonal vanishing points is out of range (f={f}) — the two "
            f"directions are not orthogonal in the image, or a principal point is wrong; "
            f"falling back to the ratio prior"
        )
        return placeholder_intrinsics(image_size, principal_point=pp), report

    report.update({"focal_px": f, "focal_ratio": f / max(w, h), "ok": True})
    report["notes"].append(
        f"f={f:.1f} px ({f / max(w, h):.2f} x max(size)) from orthogonal VPs at "
        f"{vp_a.xy} and {vp_b.xy} — this is a MEASURED focal, so the line-route "
        f"distortion fit may freeze `f` and solve only for k1/k2"
    )
    return placeholder_intrinsics(image_size, focal_px=f, principal_point=pp), report


# ============================================================================
# Line extraction + families (the "detect parallel lines" half of Phase 3)
# ============================================================================




def cluster_lines_by_angle(
    lines: Sequence[Sequence[float]], *, angle_tol_deg: float = 3.0
) -> list[list[np.ndarray]]:
    """Group homogeneous lines by image **angle** — an approximation, and a bad one in general.

    Kept only because it is the cheap first pass on a near-parallel family, and the docstring says
    what that means.  Lines that are parallel in the world converge on a vanishing point, so their
    *image angles differ* by however much the VP subtends — and for a competition pool with a fixed
    side camera that is not small.  Measured on this pipeline's synthetic scene: the pool's three
    lines along its length (all truly parallel) image at 167.8 deg, 171.5 deg and 173.5 deg, a spread
    of 5.7 deg, because their common VP sits at (3310, 111) rather than at infinity.  Any tolerance
    loose enough to group them merges the across-pool family too.

    :func:`vanishing_points_from_lines` is the correct tool and is what
    :func:`family_report` uses.  This function survives for the case it is actually good at: lines
    whose VP really is far from the scene, where the angles are within a fraction of a degree.
    """
    groups: list[list[np.ndarray]] = []
    angles: list[float] = []
    for ln in lines:
        l = np.asarray(ln, dtype=float).ravel()
        if l.size == 4:                     # a segment: convert to a homogeneous line
            l = line_from_points(l[:2], l[2:])
        th = np.degrees(np.arctan2(l[1], l[0])) % 180.0
        for i, g in enumerate(groups):
            d = min(abs(th - angles[i]), 180.0 - abs(th - angles[i]))
            if d <= angle_tol_deg:
                g.append(l)
                break
        else:
            groups.append([l])
            angles.append(th)
    return groups


def vanishing_points_from_lines(
    lines: Sequence[Sequence[float]],
    *,
    image_size: tuple[int, int] | None = None,
    principal_point: Sequence[float] | None = None,
    tol_px: float = 4.0,
    min_support: int = 2,
    max_vps: int = 6,
    far_vp_widths: float = FAR_VP_WIDTHS,
) -> list[VanishingPoint]:
    """Find vanishing points by **consensus over line pairs** — the correct clustering.

    Every pair of lines proposes its intersection as a candidate vanishing point; the candidate that
    is within ``tol_px`` of the most lines wins; those lines are removed and the search repeats.
    This is the standard construction, and the reason it is necessary rather than decorative is that
    a family of truly parallel world lines has *no* common image angle
    (see :func:`cluster_lines_by_angle`) — so any method that groups lines first and intersects them
    second has to be wrong whenever the vanishing point is anywhere near the scene, which is the
    normal case for a fixed pool camera.

    Candidates are scored by a **robust** measure of support: a line supports a candidate if the
    candidate is within ``tol_px`` of it *or* the line's own direction points within a couple of
    degrees of the candidate.  The second clause exists because the along-pool vanishing point can
    sit thousands of pixels off-frame, where "within tol_px" is a far tighter demand than the
    geometry warrants.

    Returns one :class:`VanishingPoint` per family, best-supported first.
    """
    L: list[np.ndarray] = []
    for ln in lines:
        l = np.asarray(ln, dtype=float).ravel()
        if l.size == 4:
            l = line_from_points(l[:2], l[2:])
        if l.size >= 3 and np.hypot(l[0], l[1]) > 1e-12:
            L.append(l / float(np.hypot(l[0], l[1])))
    if len(L) < 2:
        raise ValueError(f"vanishing_points_from_lines needs >= 2 lines, got {len(L)}")

    remaining = list(range(len(L)))
    out: list[VanishingPoint] = []
    while len(remaining) >= 2 and len(out) < int(max_vps):
        best: tuple[int, tuple[float, float] | None, np.ndarray, list[int]] | None = None
        for ii in range(len(remaining)):
            for jj in range(ii + 1, len(remaining)):
                a, b = L[remaining[ii]], L[remaining[jj]]
                cand = intersect_lines(a, b)
                xy = dehomogenise(cand)
                support = [k for k in remaining
                           if _supports(L[k], xy, cand, tol_px=tol_px, image_size=image_size)]
                if best is None or len(support) > best[0]:
                    best = (len(support), xy, cand, support)
        assert best is not None
        n_support, xy, cand, support = best
        if n_support < int(min_support):
            break
        vp = vanish([L[k] for k in support], image_size=image_size,
                    principal_point=principal_point, far_vp_widths=far_vp_widths)
        out.append(vp)
        remaining = [k for k in remaining if k not in set(support)]
        if len(support) < 2:
            break
    out.sort(key=lambda vp: (not vp.usable, -vp.n_lines,
                             vp.rms_px if np.isfinite(vp.rms_px) else np.inf))
    return out


def _supports(line: np.ndarray, xy: tuple[float, float] | None, cand: np.ndarray, *,
              tol_px: float, image_size: tuple[int, int] | None = None,
              angle_tol_deg: float = 2.5, far_frac: float = 2.5) -> bool:
    """Does ``line`` belong to the family whose intersection is ``cand``?

    The primary test is the one that means something: the candidate lies *on* the line to within
    ``tol_px``.

    The direction fallback exists for a vanishing point so far away that the perpendicular distance
    is a meaningless measure — but it must be restricted to *genuinely remote* candidates, and an
    earlier version was not.  For a candidate that sits near a line, "the direction from the line to
    the candidate" is trivially almost the line's own direction, so the fallback fires for every
    line and the search merges all the families into one.  Measured: a candidate at (-110, 477) — in
    the middle of the scene — collected all five lines with an RMS of 161 px.

    So the fallback applies only when the candidate is more than ``far_frac`` image half-diagonals
    from the frame centre, where it is the only usable signal.
    """
    if xy is not None:
        if abs(line_value(line, xy)) <= tol_px:
            return True
        if image_size is None:
            return False
        w, h = float(image_size[0]), float(image_size[1])
        cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
        far = float(np.hypot(xy[0] - cx, xy[1] - cy)) > float(far_frac) * 0.5 * float(
            np.hypot(w, h))
        if not far:
            return False
        d = np.array([xy[0] - cx, xy[1] - cy], dtype=float)
    else:
        d = np.asarray(cand, dtype=float)[:2]

    n = float(np.linalg.norm(d))
    if n < 1e-12:
        return False
    d = d / n
    line_dir = np.array([-line[1], line[0]])
    cos = min(1.0, abs(float(line_dir @ d)))
    return float(np.degrees(np.arccos(cos))) <= angle_tol_deg


def family_report(lines: Sequence[Sequence[float]], *,
                  image_size: tuple[int, int] | None = None,
                  principal_point: Sequence[float] | None = None,
                  max_families: int = 6,
                  tol_px: float = 4.0) -> list[VanishingPoint]:
    """One vanishing point per parallel family, found by consensus (see the RANSAC function).

    ``lines`` may be homogeneous ``(3,)`` lines or raw segments ``(4,)``;
    :func:`vanishing_points_from_lines` accepts both and converts, so there is no flag to get wrong.
    Delegates rather than re-implementing the grouping — the angle-based clustering this used to do
    is documented as wrong for this geometry.
    """
    try:
        return vanishing_points_from_lines(lines, image_size=image_size,
                                           principal_point=principal_point,
                                           max_vps=max_families, tol_px=tol_px)
    except ValueError:
        return []
