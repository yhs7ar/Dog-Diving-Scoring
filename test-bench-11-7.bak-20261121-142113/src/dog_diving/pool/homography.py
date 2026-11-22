"""Phase 4 — planar homography: undistorted pixels <-> real-world foot increments.

`pool-translation-plan.txt`, Phase 4::

    Objective: Map 2D pixel space to real-world metric foot increments.
    Implementation:
      * Match undistorted pixel coordinates of pool features to known real-world
        measurements.
      * Compute the 2D homography transformation matrix to build a uniform metric grid
        extending outward from the dock.

The world frame (one convention, used by every module in this package)
---------------------------------------------------------------------
    origin  = the 0 ft point of the dock edge, on the marked plane
    +X      = the jump direction, in feet, 0 ft at the dock and increasing into the pool
    +Y      = transverse (across the pool), in feet
    +Z      = up, away from the water; the marked plane is Z = 0 and the water is at
              Z = -h with h the freeboard in feet

A "homography" here is always the 3x3 ``H`` with ``[u, v, 1]^T ~ H [X, Y, 1]^T`` on the
**marked plane**, in **undistorted** pixels.  Two things about that sentence are load
bearing and are therefore stated once, here:

1. *Undistorted* pixels — ``H`` is fitted downstream of :mod:`dog_diving.pool.lens`.  A
   homography fitted on raw pixels while a distortion model exists elsewhere in the
   document is a silent inconsistency that shows up as a position-dependent scale error.
2. *The marked plane* — the pool's painted marks are one **collinear row**, and the
   "wall markers" the plan's Phase 2 mentions live on a **vertical** wall.  A row of
   collinear points cannot determine a 2-D homography (8 dof from 1-D data), and a
   vertical surface is not coplanar with the landing plane.  So the module offers three
   genuinely different routes and says which one it used:

   * ``route A`` — :func:`homography_from_correspondences`: >= 4 points with known
     (X, Y).  Metric, best conditioned, needs data that this corpus does not yet have.
   * ``route A'`` — :func:`pose_from_vps_and_points`: two orthogonal vanishing points
     *give the two world axis directions* (``v = K r``), so two known points suffice.
     This is the plan's Phase 3 supplying Phase 4.
   * ``route B`` — :func:`homography_from_row_and_edge`: the marked row + the transverse
     vanishing point + the far pool edge + the 14–20 ft width prior.  This **is** the
     plan's Phase 3/4 as written, and it is the route the current corpus can actually
     feed.

The width prior is a *prior*, so its cost is measured rather than asserted:
:func:`width_sensitivity` re-solves the calibration at ``W +- sigma`` and reports how
many inches the reported distance moves.  A prior that moves the answer by more than the
accuracy target is reported as such instead of being buried in a footnote.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np

from .lens import LensModel, _least_squares, undistort_points
from .vanishing import dehomogenise, line_from_points

# ============================================================================
# Homography primitives
# ============================================================================
def normalise_homography(H: np.ndarray) -> np.ndarray:
    """Scale ``H`` to a canonical form: ``H[2,2] = 1`` when possible, else unit norm.

    Canonicalising matters because every function that compares two homographies
    (agreement checks, re-runs, serialisation round-trips) would otherwise compare
    scales, not geometry.
    """
    H = np.asarray(H, dtype=float).reshape(3, 3)
    if abs(H[2, 2]) > 1e-12:
        return H / H[2, 2]
    n = float(np.linalg.norm(H))
    return H / n if n > 0 else H


def apply_homography(H: np.ndarray, pts: Any) -> np.ndarray:
    """``(X, Y)`` world points (feet) -> ``(u, v)`` undistorted pixels."""
    arr = np.asarray(pts, dtype=float).reshape(-1, 2)
    homog = np.hstack([arr, np.ones((arr.shape[0], 1))])
    out = (np.asarray(H, dtype=float) @ homog.T).T
    w = out[:, 2:3]
    w = np.where(np.abs(w) < 1e-15, np.nan, w)
    return out[:, :2] / w


def apply_homography_inv(H: np.ndarray, pts: Any) -> np.ndarray:
    """``(u, v)`` undistorted pixels -> ``(X, Y)`` feet on the plane ``H`` belongs to."""
    return apply_homography(np.linalg.inv(np.asarray(H, dtype=float)), pts)


def homography_residual_ft(
    H: np.ndarray, world_xy: Any, image_xy: Any
) -> dict[str, Any]:
    """Round-trip error of a homography, in **feet** — the unit the score is in.

    Reporting the residual in pixels is not enough for this pipeline: a 1 px error at
    30 ft is several inches, and a 1 px error at 2 ft is a fraction of an inch.  So the
    fit is judged by pushing the observed image points back to the plane and comparing
    feet to feet.  Both numbers are returned because the pixel figure is what the
    clicking precision can be reasoned about in.
    """
    W = np.asarray(world_xy, dtype=float).reshape(-1, 2)
    I = np.asarray(image_xy, dtype=float).reshape(-1, 2)
    pred_world = apply_homography_inv(H, I)
    pred_img = apply_homography(H, W)
    err_ft = np.linalg.norm(pred_world - W, axis=1)
    err_px = np.linalg.norm(pred_img - I, axis=1)
    return {
        "n": int(W.shape[0]),
        "rms_ft": float(np.sqrt(np.mean(err_ft ** 2))) if W.size else 0.0,
        "max_ft": float(np.max(err_ft)) if W.size else 0.0,
        "rms_px": float(np.sqrt(np.mean(err_px ** 2))) if W.size else 0.0,
        "max_px": float(np.max(err_px)) if W.size else 0.0,
        "per_point_ft": [float(v) for v in err_ft],
        "per_point_px": [float(v) for v in err_px],
    }


# ============================================================================
# Route A — direct correspondences
# ============================================================================
def _normalise_pts(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Hartley normalisation: centroid at the origin, mean distance ``sqrt(2)``."""
    c = pts.mean(axis=0)
    d = np.linalg.norm(pts - c, axis=1).mean()
    s = float(np.sqrt(2.0) / d) if d > 1e-12 else 1.0
    T = np.array([[s, 0.0, -s * c[0]], [0.0, s, -s * c[1]], [0.0, 0.0, 1.0]])
    return (T @ np.hstack([pts, np.ones((pts.shape[0], 1))]).T).T, T


def _collinearity(pts: np.ndarray) -> tuple[float, float]:
    """``(spread, second_singular_value/sqrt(n))`` of a centred point set.

    ``spread`` is the RMS distance from the centroid to the farthest principal direction;
    the second value is the RMS extent *perpendicular* to the dominant direction, i.e. how
    far the set departs from a straight line.  Returned as a pair so the caller can gate on
    an absolute tolerance in the point set's own unit rather than on a relative one — a
    relative test cannot distinguish "collinear" from "collinear and far apart".
    """
    arr = np.asarray(pts, dtype=float).reshape(-1, 2)
    if arr.shape[0] < 3:
        return 0.0, 0.0
    c = arr - arr.mean(axis=0)
    s = np.linalg.svd(c, compute_uv=False)
    n = float(arr.shape[0])
    return float(s[0] / np.sqrt(n)), float(s[1] / np.sqrt(n)) if s.size > 1 else 0.0


def assert_not_collinear(points: Any, *, role: str, min_perp: float) -> dict[str, float]:
    """Refuse a point set that cannot determine a homography, *by name*.

    This guard exists because of a measured failure, not for tidiness.  A planar homography
    has 8 degrees of freedom; a set of **collinear** points supplies only the 1-D-projective
    subset of them, and the remaining degrees are entirely unconstrained.  Fed such a set, a
    DLT does not fail — it returns a matrix that reproduces the input points perfectly and is
    meaningless everywhere else.  Measured on the synthetic scene's six collinear pool marks:
    the fit reported a healthy-looking pixel residual on its own points while the
    reconstructed world coordinates were wrong by **12.7 ft RMS, up to 18.8 ft**.  A silent
    18-foot error is worse than a crash, so this raises instead.
    """
    arr = np.asarray(points, dtype=float).reshape(-1, 2)
    spread, perp = _collinearity(arr)
    detail = {"n": float(arr.shape[0]), "spread": spread, "perp": perp,
              "min_perp_required": float(min_perp)}
    if perp < float(min_perp):
        raise ValueError(
            f"the {role} points are collinear to within {perp:.4g} (need >= {min_perp:g} "
            f"perpendicular spread): {arr.shape[0]} collinear points cannot determine a planar "
            f"homography — the fit would reproduce them exactly and be wrong everywhere else. "
            f"Supply points that are not on one line (e.g. two of the pool's long edges), or use "
            f"homography_from_row_and_edge (route B), which is built for a single row."
        )
    return detail


def dlt_homography(world_xy: Any, image_xy: Any) -> np.ndarray:
    """Normalised direct linear transform from >= 4 correspondences.

    Hartley normalisation is not optional here, and the reason is measurable: the world
    coordinates are in **feet** (single digits) while the image coordinates are in
    **pixels** (hundreds to thousands).  A raw DLT on those two scales produces a design
    matrix whose singular values span ~10^6 and the smallest singular vector is then
    dominated by floating-point noise.  Normalising both sides to unit scale before the
    SVD makes the fit exact on exact data and well behaved on noisy data.

    Both point sets are checked for collinearity first (:func:`assert_not_collinear`): a
    collinear set is the one input where the DLT returns a confident, exact-on-its-own-points
    and completely wrong homography.
    """
    W = np.asarray(world_xy, dtype=float).reshape(-1, 2)
    I = np.asarray(image_xy, dtype=float).reshape(-1, 2)
    if W.shape[0] != I.shape[0]:
        raise ValueError(f"correspondence count mismatch: {W.shape[0]} world vs {I.shape[0]} image")
    if W.shape[0] < 4:
        raise ValueError(
            f"a planar homography needs >= 4 correspondences with known (X, Y); got "
            f"{W.shape[0]}. A single collinear row of pool marks cannot determine one — "
            f"use homography_from_row_and_edge (route B) instead."
        )
    assert_not_collinear(W, role="world (X, Y) correspondence", min_perp=0.25)
    assert_not_collinear(I, role="image correspondence", min_perp=2.0)
    Wn, Tw = _normalise_pts(W)
    In, Ti = _normalise_pts(I)
    # Rows are written in the natural column order h11 h12 h13 h21 h22 h23 h31 h32 h33 so
    # that the smallest right-singular vector reshapes directly into H.  Writing the two
    # 3-blocks in the other order is an easy and *silent* mistake: the 9-vector is then a
    # permutation, `reshape(3, 3)` scrambles the matrix, and the fit still reproduces its
    # own input points to a fraction of a pixel while being wrong by ~15 ft elsewhere.
    A = []
    for (X, Y, w), (u, v, wi) in zip(Wn, In):
        A.append([X, Y, w, 0, 0, 0, -u * X, -u * Y, -u * w])
        A.append([0, 0, 0, X, Y, w, -v * X, -v * Y, -v * w])
    _, _, vt = np.linalg.svd(np.asarray(A, dtype=float))
    Hn = vt[-1].reshape(3, 3)
    return normalise_homography(np.linalg.inv(Ti) @ Hn @ Tw)


def homography_from_correspondences(
    world_xy: Any,
    image_xy: Any,
    *,
    weights: Sequence[float] | None = None,
    robust: bool = False,
    huber_k: float = 2.0,
    max_iter: int = 40,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Route A — fit ``H`` from points whose real-world (X, Y) in feet are known.

    ``robust=True`` runs iteratively-reweighted least squares with a Huber weight on the
    *image* residual: a homography fit is dragged by a single mis-clicked correspondence
    more than by any other error in this pipeline, and there is no RANSAC here because
    the point count is tiny (4–10), where RANSAC's own randomness costs more
    reproducibility than the outliers it removes.  The reweighting is deterministic.

    Returns ``(H, info)`` where ``info`` carries the residual table, the weights used and
    the inlier count — never just a matrix.
    """
    W = np.asarray(world_xy, dtype=float).reshape(-1, 2)
    I = np.asarray(image_xy, dtype=float).reshape(-1, 2)
    w0 = np.ones(W.shape[0], dtype=float) if weights is None else \
        np.asarray(weights, dtype=float).reshape(-1)
    if np.any(w0 <= 0):
        raise ValueError("weights must be positive")

    H = dlt_homography(W, I)
    w = w0.copy()
    iters = 0
    for iters in range(int(max_iter) if robust else 1):
        pred = apply_homography(H, W)
        r = np.linalg.norm(pred - I, axis=1)
        if not robust:
            break
        med = float(np.median(r)) or 1e-9
        s = 1.4826 * med
        u = r / max(s, 1e-9)
        hw = np.where(u <= huber_k, 1.0, huber_k / np.maximum(u, 1e-12))
        w_new = w0 * hw
        if float(np.max(np.abs(w_new - w))) < 1e-6:
            break
        w = w_new
        H = _weighted_dlt(W, I, w)

    res = homography_residual_ft(H, W, I)
    info = {
        "route": "A_correspondences",
        "robust": bool(robust),
        "iterations": int(iters + 1),
        "residual": res,
        "weights": [float(v) for v in w],
        "inliers": int(np.sum(w >= 0.5 * max(float(np.max(w)), 1e-12))),
        "n_points": int(W.shape[0]),
    }
    return H, info


def _weighted_dlt(W: np.ndarray, I: np.ndarray, w: np.ndarray) -> np.ndarray:
    Wn, Tw = _normalise_pts(W)
    In, Ti = _normalise_pts(I)
    A = []
    for (X, Y, ww), (u, v, wi), weight in zip(Wn, In, w):
        r1 = [X, Y, ww, 0, 0, 0, -u * X, -u * Y, -u * ww]
        r2 = [0, 0, 0, X, Y, ww, -v * X, -v * Y, -v * ww]
        A.append([c * float(weight) for c in r1])
        A.append([c * float(weight) for c in r2])
    _, _, vt = np.linalg.svd(np.asarray(A, dtype=float))
    return normalise_homography(np.linalg.inv(Ti) @ vt[-1].reshape(3, 3) @ Tw)


# ============================================================================
# Route A' — pose from two orthogonal vanishing points + two known points
# ============================================================================
def pose_from_vps_and_points(
    K: np.ndarray,
    vp_x: Sequence[float],
    vp_y: Sequence[float],
    world_xy: Any,
    image_xy: Any,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Two orthogonal vanishing points + >= 2 known (X, Y) points -> full camera pose.

    The identity that makes this cheap: a point at infinity in world direction ``d``
    projects to ``K r_d``, so

        rx ~ K^-1 vp_x,   ry ~ K^-1 vp_y,   rz = rx x ry

    i.e. **two vanishing points hand over two columns of the rotation for free**, and the
    third follows.  What remains is the translation ``t`` (3 dof) — and each known
    correspondence supplies 2 equations, so **two points determine the pose**, which is
    exactly what a single collinear row of pool marks can provide.  This is the route
    that turns the plan's Phase 3 into a Phase 4 homography without needing a fourth
    non-collinear point the corpus does not have.

    Requires the two VPs to be genuine vanishing points of *orthogonal* world directions
    (here: along-pool and across-pool).  Returns ``(R, t, info)``; ``info["orthogonality"]``
    is the |cos| between the recovered axis directions before orthonormalisation — if that
    is not near 0, the two "directions" were not orthogonal and the pose is refused.
    """
    K = np.asarray(K, dtype=float).reshape(3, 3)
    K_inv = np.linalg.inv(K)

    def _dir(v: Sequence[float]) -> np.ndarray:
        """Vanishing point -> unit world direction.

        A 2-vector is read as a **finite** point, ``(x, y, 1)``.  An infinite vanishing point
        (a direction, ``w = 0``) must be passed as a 3-vector — which is what
        :class:`dog_diving.pool.vanishing.VanishingPoint` already carries in ``.v`` — because a
        direction and a point cannot be told apart from two numbers alone, and guessing
        ``w = 1`` for a far-away direction is a silent, large error.
        """
        v = np.asarray(v, dtype=float).ravel()
        if v.size == 2:
            v = np.array([v[0], v[1], 1.0])
        d = K_inv @ v[:3]
        n = float(np.linalg.norm(d))
        return d / n if n > 0 else d

    rx = _dir(vp_x)
    ry = _dir(vp_y)
    cos_orth = float(abs(rx @ ry))
    info: dict[str, Any] = {"route": "A_prime_vps_points", "orthogonality_cos": cos_orth,
                            "notes": []}
    if cos_orth > 0.15:
        info["notes"].append(
            f"the two vanishing points imply axis directions {np.degrees(np.arccos(min(1.0, cos_orth))):.1f} deg "
            f"apart, not 90 — the along-pool and across-pool families were not the two "
            f"orthogonal world directions (a wrong line family, or a mis-clicked edge)"
        )
        return np.eye(3), np.zeros(3), info

    rz = np.cross(rx, ry)
    R0 = np.column_stack([rx, ry, rz])

    W = np.asarray(world_xy, dtype=float).reshape(-1, 2)
    I = np.asarray(image_xy, dtype=float).reshape(-1, 2)
    if W.shape[0] < 2:
        raise ValueError("pose_from_vps_and_points needs >= 2 known (X, Y) points "
                         "(with the two VPs they pin the remaining 3 dof of t)")

    def _solve_t(rx_s: np.ndarray, ry_s: np.ndarray) -> tuple[np.ndarray, float]:
        """Solve for ``t`` in closed form, then return the reprojection RMS in pixels.

        The projection condition ``K (rx X + ry Y + t) ~ (u, v, 1)`` is **linear in ``t``**
        once ``R`` is fixed, because it only says the two vectors are parallel:

            (u * K[2] - K[0]) . (rx X + ry Y + t) = 0
            (v * K[2] - K[1]) . (rx X + ry Y + t) = 0

        That is two linear equations per correspondence, so ``t`` comes from least squares in
        one step.  Fitting ``t`` by iterative nonlinear least squares instead — as the first
        version of this function did — is not merely slower: it failed to converge from a
        plausible initial guess and returned a pose 124 ft off with a 915 px reprojection RMS,
        because the objective is badly scaled in a translation parameterisation.  Linear is both
        correct and exact.
        """
        rows: list[np.ndarray] = []
        rhs: list[np.ndarray] = []
        for (X, Y), (u, v) in zip(W, I):
            q = rx_s * float(X) + ry_s * float(Y)
            A = np.vstack([float(u) * K[2] - K[0], float(v) * K[2] - K[1]])
            rows.append(A)
            rhs.append(-(A @ q))
        A_all = np.vstack(rows)
        b_all = np.concatenate(rhs)
        t_sol, *_ = np.linalg.lstsq(A_all, b_all, rcond=None)
        pred = np.asarray([_project(K, rx_s, ry_s, t_sol, X, Y) for X, Y in W], dtype=float)
        good = np.all(np.isfinite(pred), axis=1)
        rms = float(np.sqrt(np.mean(np.linalg.norm(pred[good] - I[good], axis=1) ** 2))) \
            if np.any(good) else float("inf")
        return t_sol, rms

    best: tuple[float, np.ndarray, np.ndarray, np.ndarray] | None = None
    for sx in (1.0, -1.0):
        for sy in (1.0, -1.0):
            rx_s = sx * rx
            ry_s = sy * ry
            t_sol, rms = _solve_t(rx_s, ry_s)
            if best is None or rms < best[0]:
                best = (rms, rx_s, ry_s, t_sol)
    assert best is not None
    rms_px, rx_s, ry_s, t = best
    R = _nearest_rotation(np.column_stack([rx_s, ry_s, np.cross(rx_s, ry_s)]))

    H = homography_from_pose(K, R, t)
    info.update({
        "R": R.tolist(), "t": [float(v) for v in t],
        "axis_signs": [float(rx_s[0] / rx[0]) if rx[0] != 0 else 1.0,
                       float(ry_s[0] / ry[0]) if ry[0] != 0 else 1.0],
        "residual": homography_residual_ft(H, W, I),
        "rms_px": rms_px,
        "n_points": int(W.shape[0]),
    })
    if rms_px > 2.0:
        info["notes"].append(
            f"the recovered pose reprojects the known points with {rms_px:.1f} px RMS — the two "
            f"vanishing points and the known points disagree, so either a VP is ill-conditioned "
            f"or the correspondences are wrong"
        )
    else:
        info["notes"].append(
            f"pose recovered from two vanishing points + {W.shape[0]} known point(s); "
            f"reprojection RMS {rms_px:.4f} px"
        )
    return R, t, info


def _project(K: np.ndarray, rx: np.ndarray, ry: np.ndarray, t: np.ndarray,
             X: float, Y: float) -> np.ndarray:
    """Project a plane point with an explicit ``(rx, ry, t)`` — used by the A' solve."""
    p = K @ (rx * float(X) + ry * float(Y) + np.asarray(t, dtype=float))
    if abs(float(p[2])) < 1e-12:
        return np.array([np.nan, np.nan])
    return np.array([p[0] / p[2], p[1] / p[2]], dtype=float)


def homography_from_pose(K: np.ndarray, R: np.ndarray, t: Sequence[float]) -> np.ndarray:
    """``H = K [r1 r2 t]`` — the plane-induced homography of the world plane Z = 0."""
    R = np.asarray(R, dtype=float).reshape(3, 3)
    t = np.asarray(t, dtype=float).ravel()
    return normalise_homography(np.asarray(K, dtype=float) @ np.column_stack([R[:, 0], R[:, 1], t]))


def _nearest_rotation(M: np.ndarray) -> np.ndarray:
    """Closest rotation matrix to ``M`` in the Frobenius sense (SVD projection)."""
    U, _, Vt = np.linalg.svd(np.asarray(M, dtype=float))
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1.0
        R = U @ Vt
    return R


# ============================================================================
# Route B — the marked row + the transverse VP + the far edge + the width prior
# ============================================================================
WIDTH_PRIOR_RANGE_FT = (14.0, 20.0)     # plan's Phase 3: "standard dock diving pool dimensions"


def homography_from_row_and_edge(
    marks: Sequence[dict],
    *,
    vp_transverse: Sequence[float] | None,
    edge_image_line: Sequence[float] | None,
    width_ft: float,
    marks_on_edge: str = "near",
    vp_transverse_finite: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Route B — the plan's Phase 3/4 exactly: row + transverse VP + far edge + width prior.

    ``marks`` are ``{"ft": X, "x_px": u, "y_px": v}`` (undistorted pixels); the world frame
    puts the marked row at ``Y = 0`` when ``marks_on_edge="near"`` and at ``Y = width_ft``
    when it is ``"far"``.  ``edge_image_line`` is the observed homogeneous image line of the
    **other** long pool edge (the one the marks are not painted on).  ``vp_transverse`` is
    the transverse vanishing point as ``(x, y)`` (finite) or ``(dx, dy)`` (a direction, when
    the edges are parallel in the image); pass ``vp_transverse_finite=True`` for the former.

    The parameterisation is what makes this well posed.  The transverse direction is a
    *hard* constraint (``H[:, 1] ~ vp``), so it is built into the model rather than fitted:

        H = [ h1 , alpha * vp , h3 ]          (h3 normalised so H[2, 2] = 1)

    leaving 6 unknowns (h1 three, h3 two, alpha one) for ``2n + 2`` residuals from ``n``
    marks and the far-edge line.  **Three marks are enough**; the corpus has four or six.

    What it will not do: infer the transverse scale from the width prior **alone**.  A
    width is a *length*, and a single view of one direction cannot convert a length into a
    transverse coordinate without observing a transverse extent — there is nothing to
    measure it against.  Pass ``edge_image_line=None`` and this raises, naming the missing
    input, rather than returning a homography that looks fine and is off by the width.
    """
    ms = [m for m in marks if m.get("x_px") is not None and m.get("y_px") is not None]
    if len(ms) < 3:
        raise ValueError(
            f"route B needs >= 3 marks on one row (got {len(ms)}); with 2 the system is "
            f"underdetermined and the fit would absorb the missing constraint into the "
            f"transverse scale"
        )
    if edge_image_line is None:
        raise ValueError(
            "route B cannot fix the transverse scale from a width prior alone: supply "
            "edge_image_line (the observed image line of the other long pool edge) so "
            "there is a transverse extent to measure the width against. Without it use "
            "route A (>= 4 known (X, Y) points)."
        )
    if vp_transverse is None:
        raise ValueError(
            "route B needs the transverse vanishing point (a direction or a finite point) "
            "— without it the transverse field is a free projective family and the width "
            "prior cannot be applied to it"
        )
    if marks_on_edge not in ("near", "far"):
        raise ValueError(f"marks_on_edge must be 'near' or 'far', got {marks_on_edge!r}")

    Y_row = 0.0 if marks_on_edge == "near" else float(width_ft)
    Y_other = float(width_ft) if marks_on_edge == "near" else 0.0

    vp = np.asarray(vp_transverse, dtype=float).ravel()
    if vp.size == 2:
        vp = np.array([vp[0], vp[1], 1.0 if vp_transverse_finite else 0.0])
    if np.linalg.norm(vp) < 1e-12:
        raise ValueError("vp_transverse is degenerate (zero)")
    vp = vp / np.linalg.norm(vp)

    lo = np.asarray(edge_image_line, dtype=float).ravel()
    if lo.size != 3:
        raise ValueError("edge_image_line must be a homogeneous line (a, b, c)")
    lo = lo / max(float(np.hypot(lo[0], lo[1])), 1e-12)

    W = np.array([[float(m["ft"]), Y_row] for m in ms], dtype=float)
    I = np.array([[float(m["x_px"]), float(m["y_px"])] for m in ms], dtype=float)

    # sample points of the observed far/near edge, to convert the world line constraint
    # into pixel residuals (uniform with the mark residuals, so the weights are honest)
    edge_pts = _sample_line(lo, I, n=5)

    def _H_from_x(x: np.ndarray) -> np.ndarray:
        h1 = x[0:3]
        alpha = x[3]
        h3 = np.array([x[4], x[5], 1.0])
        return np.column_stack([h1, alpha * vp, h3])

    def _residuals(x: np.ndarray) -> np.ndarray:
        H = _H_from_x(x)
        out = list((apply_homography(H, W) - I).ravel())
        # the world line Y = Y_other maps to image line  H^-T (0, 1, -Y_other)
        try:
            Hinv_T = np.linalg.inv(H).T
        except np.linalg.LinAlgError:
            return np.concatenate([out, np.full(edge_pts.shape[0], 1e6)])
        try:
            l_img = Hinv_T @ np.array([0.0, 1.0, -Y_other], dtype=float)
        except np.linalg.LinAlgError:
            return np.concatenate([out, np.full(edge_pts.shape[0], 1e6)])
        if np.hypot(l_img[0], l_img[1]) < 1e-12:
            return np.concatenate([out, np.full(edge_pts.shape[0], 1e6)])
        l_img = l_img / np.hypot(l_img[0], l_img[1])
        out.extend((edge_pts @ l_img[:2] + l_img[2]).tolist())
        return np.asarray(out, dtype=float)

    # --- initial guess: h3 ~ the lowest mark's ray, h1 from the two extreme marks -----
    order = np.argsort(W[:, 0])
    i_lo_, i_hi_ = int(order[0]), int(order[-1])
    span = float(W[i_hi_, 0] - W[i_lo_, 0]) or 1.0
    p_lo = np.array([I[i_lo_, 0], I[i_lo_, 1], 1.0])
    p_hi = np.array([I[i_hi_, 0], I[i_hi_, 1], 1.0])
    h1_0 = (p_hi - p_lo) / span          # linear between the extreme marks: plenty for LS
    h3_0 = p_lo
    alpha_0 = 1.0
    x0 = np.concatenate([h1_0, [alpha_0], h3_0[:2]])
    res = _least_squares(_residuals, x0, max_iter=200)
    H = normalise_homography(_H_from_x(np.asarray(res.x, dtype=float)))

    resinfo = homography_residual_ft(H, W, I)
    info = {
        "route": "B_row_vp_edge_width",
        "width_ft": float(width_ft),
        "width_source": "prior",
        "marks_on_edge": marks_on_edge,
        "Y_row_ft": float(Y_row),
        "Y_other_ft": float(Y_other),
        "n_marks": len(ms),
        "vp_transverse": [float(v) for v in vp],
        "vp_transverse_finite": bool(vp_transverse_finite),
        "edge_line": [float(v) for v in lo],
        "residual": resinfo,
        "mark_residual_ft": resinfo["per_point_ft"],
        "condition": _homography_condition(H),
        "notes": [
            "the transverse scale is fixed by the observed far edge against the "
            f"{width_ft:g} ft width prior — it is a PRIOR, and width_sensitivity() "
            "reports how many inches of distance ride on it"
        ],
    }
    return H, info


def _sample_line(line: np.ndarray, near: np.ndarray, n: int = 5) -> np.ndarray:
    """Sample ``n`` points spread along an image line, centred on the ``near`` points.

    A line constraint evaluated at one point is a single equation; evaluated *along a span*
    it also constrains the line's orientation — which is what the far pool edge is actually
    contributing, and what stops the fit from satisfying the constraint by rotating a line
    about a single sample.
    """
    line = np.asarray(line, dtype=float).ravel()
    line = line / max(float(np.hypot(line[0], line[1])), 1e-12)
    d = np.array([-line[1], line[0]], dtype=float)      # unit direction along the line
    foot = -line[2] * np.array([line[0], line[1]])      # point on the line nearest the origin
    centre = np.asarray(near, dtype=float).reshape(-1, 2).mean(axis=0)
    t0 = float((centre - foot) @ d)
    span = max(float(np.ptp(np.asarray(near, dtype=float).reshape(-1, 2)[:, 0])), 50.0)
    ts = np.linspace(t0 - span, t0 + span, int(n))
    return foot + np.outer(ts, d)


def width_sensitivity(
    marks: Sequence[dict],
    *,
    vp_transverse: Sequence[float] | None,
    edge_image_line: Sequence[float] | None,
    width_ft: float,
    probe_world_xy: Any,
    deltas: Sequence[float] = (-1.0, -0.5, 0.5, 1.0),
    marks_on_edge: str = "near",
    vp_transverse_finite: bool = False,
) -> dict[str, Any]:
    """How many inches of distance ride on the pool-width prior.

    Re-solves route B at ``width_ft + d`` for each ``d`` in ``deltas`` (feet) and reports
    the induced movement of a probe point's world coordinates, in inches.  This is the
    number that decides whether the plan's "14 to 20 ft wide" allowance is affordable: the
    full prior band is 6 ft, so the reported figure is scaled to the band and compared with
    the project's ±2 in accuracy criterion.  A prior that cannot pay for itself is
    reported, not hidden.
    """
    probe = np.asarray(probe_world_xy, dtype=float).reshape(-1, 2)
    out: dict[str, Any] = {"width_ft": float(width_ft), "probes": [], "notes": []}
    for d in deltas:
        try:
            H, _ = homography_from_row_and_edge(
                marks, vp_transverse=vp_transverse, edge_image_line=edge_image_line,
                width_ft=float(width_ft) + float(d), marks_on_edge=marks_on_edge,
                vp_transverse_finite=vp_transverse_finite)
        except Exception as exc:  # noqa: BLE001 - a prior sensitivity is best-effort
            out["notes"].append(f"re-solve at width{float(d):+.2f} ft failed: {exc}")
            continue
        # push the probe through the width-shifted homography by matching its own pixel
        img = apply_homography(
            homography_from_row_and_edge(
                marks, vp_transverse=vp_transverse, edge_image_line=edge_image_line,
                width_ft=float(width_ft), marks_on_edge=marks_on_edge,
                vp_transverse_finite=vp_transverse_finite)[0], probe)
        got = apply_homography_inv(H, img)
        shift_in = np.linalg.norm(got - probe, axis=1) * 12.0
        out["probes"].append({"delta_ft": float(d),
                              "shift_in": [float(v) for v in shift_in]})
    shifts = [max(p["shift_in"]) for p in out["probes"]] if out["probes"] else []
    if shifts:
        deltas_abs = [abs(p["delta_ft"]) for p in out["probes"]]
        per_ft = float(np.mean([s / max(a, 1e-9) for s, a in zip(shifts, deltas_abs)]))
        out["in_per_ft_of_width"] = per_ft
        out["in_across_prior_band"] = per_ft * (WIDTH_PRIOR_RANGE_FT[1] - WIDTH_PRIOR_RANGE_FT[0])
        out["notes"].append(
            f"{per_ft:.2f} in of distance per foot of pool width; across the plan's full "
            f"{WIDTH_PRIOR_RANGE_FT[0]:g}-{WIDTH_PRIOR_RANGE_FT[1]:g} ft prior band that is "
            f"{out['in_across_prior_band']:.1f} in — compare with the +/-2 in accuracy "
            f"criterion before quoting a score from route B"
        )
    return out


# ============================================================================
# Condition number: was this homography actually determined by the data?
# ============================================================================
def _homography_condition(H: np.ndarray) -> dict[str, Any]:
    """Singular-value spread of ``H`` plus the local anisotropy of the px/ft mapping.

    Two different failure modes are worth separating:

    * a near-singular ``H`` (the plane is edge-on) — reported by the singular values;
    * a ``H`` that is well conditioned overall but *anisotropic*, i.e. one foot is
      thousands of pixels along x and a handful along y.  That is not an error, it is
      foreshortening, but it is exactly what makes a transverse error cheap and a
      longitudinal error expensive, so the ratio is reported next to every distance.
    """
    H = np.asarray(H, dtype=float)
    sv = np.linalg.svd(H, compute_uv=False)
    J = np.linalg.inv(H)          # d(world)/d(pixel) up to the projective factor
    s = np.linalg.svd(J[:, :2], compute_uv=False)
    px_per_ft_x = 1.0 / max(float(s[1]), 1e-12) if s[1] > 0 else float("inf")
    px_per_ft_y = 1.0 / max(float(s[0]), 1e-12) if s[0] > 0 else float("inf")
    return {
        "singular_values": [float(v) for v in sv],
        "cond": float(sv[0] / sv[-1]) if sv[-1] > 1e-300 else float("inf"),
        "px_per_ft_along": px_per_ft_x,
        "px_per_ft_across": px_per_ft_y,
        "anisotropy": float(px_per_ft_x / px_per_ft_y) if px_per_ft_y > 0 else float("inf"),
    }


# ============================================================================
# The plane pose — this is what makes the water plane reachable
# ============================================================================
@dataclass
class PlanePose:
    """Camera pose induced by a plane homography, with an explicit up-axis convention.

    ``R``, ``t`` are the world->camera rotation and translation with the plane at Z = 0,
    and the world's +Z axis is ``z_sign * R[:, 2]``.  The extra ``z_sign`` exists because
    the homography decomposition cannot know which side of the plane is "up": both
    ``(R, t)`` and the same ``R`` with a flipped Z axis describe the pixels identically.
    Rather than guess, :meth:`resolve_up` decides it from an observed vertical in the
    image — the rim/water pair the plan's Phase 5 already asks the operator to click.

    Why it matters: the water plane is the marked plane *translated down by the freeboard*,
    and "down" is exactly this sign.  Getting it backwards puts the water surface above the
    dock and every distance on the wrong side of the pool.
    """

    K: np.ndarray
    R: np.ndarray
    t: np.ndarray
    H: np.ndarray
    z_sign: float = 1.0
    info: dict[str, Any] = field(default_factory=dict)

    @property
    def z_axis(self) -> np.ndarray:
        """World +Z in camera coordinates (the plane normal, pointing up)."""
        return self.z_sign * self.R[:, 2]

    def vertical_vp(self, z_sign: float | None = None) -> np.ndarray:
        """Image of the world **+Z** direction's point at infinity: ``K z_axis`` (homogeneous).

        The image direction toward "up" at any pixel is the direction toward this vanishing
        point.  That single fact is what :meth:`resolve_up` uses to fix the up-axis sign
        without knowing where anything is in world coordinates.
        """
        s = self.z_sign if z_sign is None else float(z_sign)
        return np.asarray(self.K, dtype=float) @ (s * np.asarray(self.R, dtype=float)[:, 2])

    @property
    def camera_centre(self) -> np.ndarray:
        """Camera centre in world coordinates (feet), plane at Z = 0."""
        return -self.R.T @ self.t

    def plane_homography(self, z_ft: float) -> np.ndarray:
        """Homography of the parallel plane at world height ``z_ft`` (water: ``-h``).

        This is the whole of Phase 5 in one line.  A world point is ``(X, Y, z)``, so

            (u, v) ~ K (r1 X + r2 Y + rz z + t) = K [r1, r2, t + z * rz] (X, Y, 1)

        i.e. the plane's homography differs from ``H`` **only in its third column**.
        Phase 5 therefore is not a new fit — it is an exact re-parameterisation of the
        Phase 4 homography at a different height, which is why the water plane can be
        reached where it is invisible: nothing about it needs to be seen.
        """
        rz = self.z_axis
        return normalise_homography(
            np.asarray(self.K, dtype=float) @ np.column_stack([self.R[:, 0], self.R[:, 1],
                                                               self.t + float(z_ft) * rz]))


    def height_at_image_point(self, pixel: Sequence[float],
                              world_xy: Sequence[float]) -> float:
        """World Z (feet) of the point on the vertical through ``world_xy`` that images
        to ``pixel``.

        Used to turn a clicked image pair *up a vertical* into a real height: the two
        clicks need not be plotted anywhere, they only need to be on the same vertical,
        which
        is what a pool wall's corner is.
        """
        K = np.asarray(self.K, dtype=float)
        R = np.asarray(self.R, dtype=float)
        d_cam = K @ np.array([float(pixel[0]), float(pixel[1]), 1.0])
        d_cam = d_cam / np.linalg.norm(d_cam)
        d_world = R.T @ d_cam                    # ray direction in world coordinates
        C = self.camera_centre
        p = np.array([float(world_xy[0]), float(world_xy[1]), 0.0])
        # Solve X(s) = Xp, Y(s) = Yp for the single unknown s (least squares, because the
        # ray only passes exactly through the vertical when the clicks were exact), then
        # read Z off the same parameter.
        A = d_world[:2]
        b = np.array([p[0] - C[0], p[1] - C[1]])
        s = float((A @ b) / max(float(A @ A), 1e-15))
        return float(C[2] + s * d_world[2])


    def resolve_up(self, lower_pixel: Sequence[float], upper_pixel: Sequence[float]) -> tuple[float, dict[str, Any]]:
        """Decide ``z_sign`` from an observed vertical in the image — no world data needed.

        The plan's Phase 5 supplies exactly the right observation: a rim top and the water
        contact **below** it.  The test is the vertical vanishing point.  Points at greater world
        height project further along the image line toward the vanishing point of the +Z
        direction, so if the clicked "upper" point is *away* from that VP relative to the clicked
        "lower" one, the naive ``z_sign = +1`` has the axis upside down and must be flipped.

        An earlier version of this method asked for the vertical's world ``(X, Y)`` and
        compared reconstructed heights instead.  That was wrong in a way worth recording: it
        is circular (the height is what is being solved for), and with the default ``(0, 0)``
        it silently measured a completely different vertical — it reported a rim at 19.4 ft on
        a scene where the rim is at 0.33 ft.  The vanishing-point test needs no such input.

        Returns ``(z_sign, detail)``.  When the two clicks are indistinguishable along the
        vertical's image direction the answer is genuinely undetermined, and the detail says so
        rather than picking one.
        """
        lower = np.asarray(lower_pixel, dtype=float).ravel()[:2]
        upper = np.asarray(upper_pixel, dtype=float).ravel()[:2]
        d = upper - lower
        v = self.vertical_vp(z_sign=1.0)
        # The direction of +Z *motion* in the image is toward the VP only when the camera-frame
        # z of the +Z axis is positive.  When it is negative (the camera sits above the scene and
        # world-up points partially back toward and behind it), points at infinity along +Z are
        # behind the camera, and the image therefore runs AWAY from the VP as the world runs up.
        # Omitting this factor is a coin flip on every water level the pipeline reports.
        if abs(float(v[2])) > 1e-9:
            toward = (np.asarray(v[:2], dtype=float) / float(v[2])) - lower
            mode = "finite"
        else:
            toward = np.asarray(v[:2], dtype=float)
            mode = "at_infinity"
        motion_dir = toward if float(v[2]) >= 0.0 else -toward
        proj = float(d @ motion_dir)
        if abs(proj) < 1e-9:
            return 1.0, {
                "z_sign": 1.0,
                "rule": "undetermined: the two clicks are perpendicular to (or degenerate "
                        "along) the image direction of the vertical, so no sign can be "
                        "distinguished from this pair",
                "vp_z": [float(x) for x in v], "vp_mode": mode, "projection": proj,
            }
        sign = 1.0 if proj > 0.0 else -1.0
        return sign, {
            "z_sign": float(sign),
            "rule": "chosen so the clicked upper point lies further along the image direction "
                    "of +Z motion (toward the vertical vanishing point once the sign of its "
                    "depth is accounted for) than the lower one",
            "vp_z": [float(x) for x in v], "vp_mode": mode, "projection": proj,
            "step_px": float(np.linalg.norm(d)),
        }


def decompose_plane_homography(H: np.ndarray, K: np.ndarray, *,
                               z_sign: float = 1.0) -> PlanePose:
    """``H`` and ``K`` -> the plane's pose ``(R, t)``.

    Standard plane-induced-homography decomposition: ``K^-1 H = s [r1 r2 t]``, so the first
    two columns — once scaled to unit length — *are* the first two rotation columns and the
    third column is the translation.  The scale is taken from Zhang's symmetric estimate
    ``lambda = 2 / (|m1| + |m2|)`` (better conditioned than dividing both, and it fails
    gracefully when one column is nearly zero, i.e. a plane seen almost edge-on), and the
    sign is fixed by requiring the plane to be in **front** of the camera — the origin's
    camera-frame depth ``t_z`` must be positive.

    Any residual non-orthonormality from noisy ``H`` is projected away by the nearest-
    rotation SVD, and the size of that projection is reported as ``info["orthonormality"]``
    — the honest measure of how well the two columns agreed with being perpendicular
    unit vectors, and hence of how much the pose can be trusted.
    """
    H = normalise_homography(H)
    K = np.asarray(K, dtype=float).reshape(3, 3)
    M = np.linalg.inv(K) @ H
    m1, m2, m3 = M[:, 0], M[:, 1], M[:, 2]
    n1, n2 = float(np.linalg.norm(m1)), float(np.linalg.norm(m2))
    if n1 < 1e-15 or n2 < 1e-15:
        raise ValueError("degenerate homography: a column of K^-1 H is zero "
                         "(the plane is edge-on to the camera)")
    lam = 2.0 / (n1 + n2)
    if float(m3[2]) * lam < 0.0:
        lam = -lam
    r1, r2, t = lam * m1, lam * m2, lam * m3
    rz = np.cross(r1, r2)
    ortho = float(np.linalg.norm(np.array([r1 @ r2, r1 @ r1 - 1.0, r2 @ r2 - 1.0])))
    R = _nearest_rotation(np.column_stack([r1, r2, rz]))
    info = {
        "lambda": float(lam),
        "orthonormality": ortho,
        "t": [float(v) for v in t],
        "camera_centre_ft": (np.linalg.inv(R) @ -t).tolist(),
        "notes": [],
    }
    if ortho > 0.05:
        info["notes"].append(
            f"orthonormality residual {ortho:.3f} is large: the fitted homography is not "
            f"consistent with a rigid plane (usually few, poorly spread or mis-clicked "
            f"correspondences), so the decomposed pose is unreliable"
        )
    return PlanePose(K=K, R=R, t=t, H=H, z_sign=float(z_sign), info=info)


def plane_from_three_points(p0: Sequence[float], p1: Sequence[float],
                            p2: Sequence[float]) -> tuple[np.ndarray, float]:
    """Plane ``(n, d)`` with ``n . x = d`` through three points — the literal anchor set.

    Kept because Phase 5 says "anchor the plane using the visible top rim and vertical side
    walls": if an operator *does* have three non-collinear points on the water, this is the
    one call that turns them into a plane.  Everywhere else the plane comes from gravity
    (a normal) plus one observed height, which is 1 dof instead of 3 and is the reason the
    rim is enough.
    """
    a = np.asarray(p0, dtype=float).ravel()
    b = np.asarray(p1, dtype=float).ravel()
    c = np.asarray(p2, dtype=float).ravel()
    n = np.cross(b - a, c - a)
    nn = float(np.linalg.norm(n))
    if nn < 1e-12:
        raise ValueError("the three points are collinear — they do not define a plane")
    n = n / nn
    return n, float(n @ a)
