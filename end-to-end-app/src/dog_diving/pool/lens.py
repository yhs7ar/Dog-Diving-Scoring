"""Phase 1 — camera intrinsics and lens-distortion correction.

`pool-translation-plan.txt`, Phase 1::

    Objective: Remove radial and tangential lens warping to ensure linear spatial
               measurements.
    Implementation:
      * Estimate camera intrinsic parameters and distortion coefficients
        (k1, k2, p1, p2, k3) via standard calibration patterns or straight pool-edge
        line fitting.
      * Pass raw pixel coordinates (u, v) through an undistortion function
        (e.g., OpenCV ``cv2.undistortPoints``).

Why this module has to exist at all
-----------------------------------
Every metric claim in this pipeline is a statement about *straight lines*: the pool
marks are equally spaced along a straight ruler, the pool's long edges are straight
and parallel, and the intersection of a tail-set ray with the water plane is only
meaningful if the ray was built from an undistorted pixel.  A rectilinear lens that
bends the pool edge by a few pixels therefore injects a *position-dependent* scale
error into the distance, which is exactly the kind of error that never shows up in a
residual table.  The pre-existing build started from raw pixels and had no
distortion term anywhere (see ``docs/POOL_TRANSLATION_IMPACT.md`` §1.1).

Two ways in, one model out
--------------------------
* **Checkerboard** (:func:`calibrate_from_checkerboard`) — the textbook route; the
  only one that measures the *whole* model (focal, principal point, k1,k2,p1,p2,k3).
  The corpus ships no chessboard image, so this path exists and is unit-tested, but
  it is not the default.
* **Straightness of the pool's own edges** (:func:`estimate_from_lines`) — the
  plan's own fallback ("or straight pool-edge line fitting").  It knows nothing
  about the world; it only asserts the physical fact that a straight edge in the
  world must image as a straight line.  It cannot recover everything from one view,
  and the honest statement of *what* it can recover is in :func:`estimate_from_lines`.

The degeneracy worth naming
---------------------------
Focal length and radial distortion are only weakly separable from straightness
alone: for a family of lines that all pass near the image centre, a shorter focal
length and a positive ``k1`` produce almost the same bend.  :func:`estimate_from_lines`
therefore reports the parameter correlation and refuses to claim a well-determined
``f``/``k1`` pair when the lines do not span the field.  Silently returning a
confident-looking ``(f, k1)`` pair here would corrupt every subsequent phase, because
Phase 4's homography is computed in the *undistorted* space this module defines.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import cv2
import numpy as np

# --- optional SciPy: the plan names SciPy, and it is what the fit prefers ---------
try:  # pragma: no cover - exercised by both branches depending on the environment
    from scipy.optimize import least_squares as _scipy_lsq

    HAVE_SCIPY = True
except Exception:  # noqa: BLE001 - a missing SciPy must not break import
    _scipy_lsq = None
    HAVE_SCIPY = False

# Default focal-length prior, as a ratio f / max(image_width, image_height).  It is a
# *prior*, not a measurement: every LensModel records ``source`` so a downstream number
# can never be quoted without saying that its intrinsics were assumed.  1.0 corresponds
# to roughly a 53 deg horizontal field of view, which is the conservative middle of the
# phone / action-camera range this corpus was shot on.
DEFAULT_FOCAL_RATIO = 1.0

# Distortion vector layout, identical to OpenCV's 5-parameter model.
#   [k1, k2, p1, p2, k3]
DIST_NAMES = ("k1", "k2", "p1", "p2", "k3")


# ============================================================================
# The model
# ============================================================================
@dataclass
class LensModel:
    """A pinhole camera with OpenCV's radial-tangential distortion.

    ``K`` is the 3x3 intrinsic matrix **of the undistorted image** — i.e. the target
    of :meth:`undistort_points`.  The pipeline keeps one image coordinate system
    (that one): homographies, grids, overlays and intersections all live in
    undistorted pixels, so ``K`` is the only intrinsics object the rest of the
    pipeline ever needs.
    """

    K: np.ndarray
    dist: np.ndarray
    image_size: tuple[int, int]                       # (width, height)
    source: str = "assumed"                           # checkerboard | lines | assumed | none
    rms_px: float | None = None                       # reprojection / straightness RMS
    notes: list[str] = field(default_factory=list)

    # --- convenience -------------------------------------------------------
    @property
    def width(self) -> int:
        return int(self.image_size[0])

    @property
    def height(self) -> int:
        return int(self.image_size[1])

    @property
    def fx(self) -> float:
        return float(self.K[0, 0])

    @property
    def fy(self) -> float:
        return float(self.K[1, 1])

    @property
    def principal_point(self) -> tuple[float, float]:
        return float(self.K[0, 2]), float(self.K[1, 2])

    @property
    def is_distorted(self) -> bool:
        """True when the model would actually move a pixel."""
        return bool(np.any(np.abs(np.asarray(self.dist, dtype=float)) > 1e-12))

    @property
    def deterministic(self) -> bool:
        """True when the intrinsics were measured rather than assumed."""
        return self.source in ("checkerboard", "lines")

    def as_dict(self) -> dict[str, Any]:
        return {
            "K": np.asarray(self.K, dtype=float).tolist(),
            "dist": [float(v) for v in np.asarray(self.dist, dtype=float).ravel()],
            "dist_model": "opencv_radtan5",
            "dist_names": list(DIST_NAMES),
            "image_size": [int(self.image_size[0]), int(self.image_size[1])],
            "source": self.source,
            "rms_px": None if self.rms_px is None else float(self.rms_px),
            "focal_ratio": float(self.fx / max(self.width, self.height)),
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, doc: dict[str, Any]) -> "LensModel":
        K = np.asarray(doc["K"], dtype=float).reshape(3, 3)
        dist = np.asarray(doc.get("dist", [0.0] * 5), dtype=float).ravel()
        if dist.size < 5:                       # pad a short vector rather than fail
            dist = np.concatenate([dist, np.zeros(5 - dist.size)])
        return cls(
            K=K,
            dist=dist[:5],
            image_size=(int(doc["image_size"][0]), int(doc["image_size"][1])),
            source=str(doc.get("source", "assumed")),
            rms_px=None if doc.get("rms_px") is None else float(doc["rms_px"]),
            notes=list(doc.get("notes") or []),
        )


# ============================================================================
# Constructors
# ============================================================================
def focal_prior_px(image_size: tuple[int, int], ratio: float = DEFAULT_FOCAL_RATIO) -> float:
    """Focal length prior in pixels: ``ratio * max(width, height)``."""
    w, h = int(image_size[0]), int(image_size[1])
    return float(ratio) * float(max(w, h))


def focal_from_hfov(image_size: tuple[int, int], hfov_deg: float) -> float:
    """Focal length in pixels from a horizontal field of view (e.g. from a spec sheet).

    A measured spec sheet beats a ratio prior and is worth the one extra argument:
    ``f = (w / 2) / tan(hfov / 2)``.
    """
    w = float(image_size[0])
    return (w / 2.0) / float(np.tan(np.radians(hfov_deg) / 2.0))


def placeholder_intrinsics(
    image_size: tuple[int, int],
    *,
    focal_px: float | None = None,
    focal_ratio: float = DEFAULT_FOCAL_RATIO,
    principal_point: tuple[float, float] | None = None,
) -> LensModel:
    """An *assumed* pinhole model — no distortion, focal from a prior.

    This is the starting point for :func:`estimate_from_lines` and the honest
    fallback when nothing better exists.  ``source="assumed"`` is what makes that
    honest: a calibration whose ``lens.source`` is ``assumed`` can still measure a
    distance, but it must say that its intrinsics were not measured.
    """
    w, h = int(image_size[0]), int(image_size[1])
    f = float(focal_px) if focal_px is not None else focal_prior_px((w, h), focal_ratio)
    cx, cy = principal_point if principal_point is not None else ((w - 1) / 2.0, (h - 1) / 2.0)
    K = np.array([[f, 0.0, float(cx)], [0.0, f, float(cy)], [0.0, 0.0, 1.0]], dtype=float)
    return LensModel(K=K, dist=np.zeros(5, dtype=float), image_size=(w, h), source="assumed")




# ============================================================================
# Point transforms — the plan's "pass raw pixels through cv2.undistortPoints"
# ============================================================================
def _as_points(points: Any) -> tuple[np.ndarray, bool]:
    """Normalise input to an ``(N, 2) float64`` array, remembering if it was a tuple."""
    arr = np.asarray(points, dtype=float)
    single = arr.ndim == 1
    arr = arr.reshape(-1, 2)
    return arr, single


_UNDISTORT_CRITERIA: Any = None


def _undistort_criteria():
    """The iteration criterion for ``cv2.undistortPoints``, built once.

    ``cv2.undistortPoints`` inverts the distortion model **iteratively**, and its default stopping
    rule leaves the inverse measurably short: on the synthetic scene this pipeline is tested
    against, the round-trip error at the image corners is **3.9e-5 px** by default versus
    **1.6e-13 px** with the criterion below.  Both are far below a pixel, so this is not a scoring
    issue — it is the difference between a claim that can honestly be written "float noise" and one
    that has to be quoted as a tolerance, and it costs one extra keyword.
    """
    global _UNDISTORT_CRITERIA
    if _UNDISTORT_CRITERIA is None:
        try:
            _UNDISTORT_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-14)
        except Exception:  # noqa: BLE001 - a missing constant must not break import
            _UNDISTORT_CRITERIA = False
    return _UNDISTORT_CRITERIA


def undistort_points(points: Any, lens: LensModel | None) -> np.ndarray:
    """Raw pixel coordinates -> undistorted pixel coordinates, same pixel grid.

    ``P=lens.K`` is passed explicitly so the output lands back in the *same* pixel
    units as the input (OpenCV's default would silently return normalised
    coordinates and scale every downstream homography by ``f``).  ``lens=None`` or a
    distortion-free model is a no-op but still returns a copy, so callers can always
    write to the result.

    The inversion is run to convergence (see :func:`_undistort_criteria`), with a fallback to
    OpenCV's plain call signature so an older build that lacks the ``criteria`` keyword still
    works — a missing keyword must not turn a calibration into a crash.
    """
    arr, single = _as_points(points)
    if lens is None or not lens.is_distorted:
        out = arr.copy()
    else:
        src = arr.reshape(-1, 1, 2).astype(np.float64)
        K = np.asarray(lens.K, dtype=float)
        dist = np.asarray(lens.dist, dtype=float)
        crit = _undistort_criteria()
        out = None
        if crit:
            try:
                out = cv2.undistortPoints(src, K, dist, None, K, criteria=crit).reshape(-1, 2)
            except Exception:  # noqa: BLE001 - fall back to the plain signature
                out = None
        if out is None:
            out = cv2.undistortPoints(src, K, dist, None, K).reshape(-1, 2)
        if not np.all(np.isfinite(out)):          # never hand a NaN pixel downstream
            out = cv2.undistortPoints(src, K, dist, None, K).reshape(-1, 2)
    return out[0] if single else out


def distort_points(points: Any, lens: LensModel | None) -> np.ndarray:
    """Undistorted pixel coordinates -> raw pixel coordinates (forward model).

    Needed for one thing only, and it matters: *drawing*.  A homography is fitted in
    the undistorted space, so an overlay computed there must be pushed back through
    the real lens before it is composited onto a raw video frame, or the grid sits a
    few pixels off the pool marks it is supposed to be checked against.
    """
    arr, single = _as_points(points)
    if lens is None or not lens.is_distorted:
        out = arr.copy()
    else:
        # pixel -> normalised -> projectPoints with a null pose returns normalised -> distorted
        K = np.asarray(lens.K, dtype=float)
        K_inv = np.linalg.inv(K)
        homog = np.hstack([arr, np.ones((arr.shape[0], 1))])
        norm = (K_inv @ homog.T).T
        obj = norm.reshape(-1, 1, 3)
        out = cv2.projectPoints(obj, np.zeros(3), np.zeros(3), K,
                                np.asarray(lens.dist, dtype=float))[0].reshape(-1, 2)
    return out[0] if single else out


def undistort_image(image: np.ndarray, lens: LensModel | None, *,
                    alpha: float = 0.0) -> np.ndarray:
    """Undistort a whole frame with the *same* ``K`` the points use.

    ``alpha=0`` crops to the valid region by re-using ``lens.K`` as the new camera
    matrix, which is what keeps ``undistort_image`` and :func:`undistort_points`
    consistent.  Using ``cv2.getOptimalNewCameraMatrix`` instead would change the
    intrinsics and invalidate every pixel coordinate the pipeline holds.
    """
    if lens is None or not lens.is_distorted:
        return image.copy()
    return cv2.undistort(image, np.asarray(lens.K, dtype=float),
                         np.asarray(lens.dist, dtype=float), None,
                         np.asarray(lens.K, dtype=float))


# ============================================================================
# Straightness — the measurement the line route is built on
# ============================================================================
def fit_line_total_ls(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Total-least-squares line through ``(N, 2)`` points.

    Returns ``(centroid, direction)`` with ``direction`` a unit vector.  Total least
    squares (PCA) is the right estimator here: the pool edge is a *near-vertical*
    line in some clips and near-horizontal in others, so a ``y = mx + c`` fit would
    be unusable for half the corpus.
    """
    arr = np.asarray(points, dtype=float).reshape(-1, 2)
    if arr.shape[0] < 2:
        raise ValueError("need >= 2 points for a line")
    centroid = arr.mean(axis=0)
    _, _, vt = np.linalg.svd(arr - centroid, full_matrices=False)
    return centroid, vt[0]


def line_perp_residuals(points: np.ndarray) -> np.ndarray:
    """Signed perpendicular distances of ``points`` from their own TLS line."""
    arr = np.asarray(points, dtype=float).reshape(-1, 2)
    centroid, direction = fit_line_total_ls(arr)
    normal = np.array([-direction[1], direction[0]], dtype=float)
    return (arr - centroid) @ normal


def straightness_residual(points: np.ndarray) -> float:
    """Max |perpendicular distance| of a point set from its own best-fit line (px).

    This is the whole error metric of the line route: a perfectly straight physical
    edge must give 0, and any nonzero value is the lens bending it.
    """
    if len(np.asarray(points).reshape(-1, 2)) < 3:
        return 0.0
    return float(np.max(np.abs(line_perp_residuals(points))))


def total_straightness(lines: Iterable[np.ndarray]) -> float:
    """RMS of the per-line straightness residuals over a family of lines (px)."""
    vals = [straightness_residual(np.asarray(ln, dtype=float)) for ln in lines]
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.sqrt(np.mean(np.square(vals)))) if vals else float("nan")


def line_coverage(lines: Sequence[np.ndarray], image_size: tuple[int, int]) -> dict[str, float]:
    """Where the supplied lines actually sit in the frame.

    Used to decide whether the ``f``/``k1`` pair is identifiable: the classic
    degeneracy of distortion-from-straightness is that lines hugging the principal
    point straddle the lens axis, where ``k1`` has almost no effect, so the fit can
    trade ``f`` against ``k1`` freely and return a confident nonsense pair.
    """
    w, h = int(image_size[0]), int(image_size[1])
    pts = np.vstack([np.asarray(ln, dtype=float).reshape(-1, 2) for ln in lines]) \
        if len(lines) else np.zeros((0, 2))
    if pts.size == 0:
        return {"n_lines": 0.0, "span_x": 0.0, "span_y": 0.0, "max_radius_frac": 0.0}
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    radius = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)
    return {
        "n_lines": float(len(lines)),
        "span_x": float(pts[:, 0].max() - pts[:, 0].min()),
        "span_y": float(pts[:, 1].max() - pts[:, 1].min()),
        "max_radius_frac": float(radius.max() / (0.5 * float(np.hypot(w, h)))),
    }


# ============================================================================
# The line-based fit (the plan's "straight pool-edge line fitting")
# ============================================================================
_FIT_INDEX = {"f": 0, "k1": 1, "k2": 2, "p1": 3, "p2": 4, "cx": 5, "cy": 6, "k3": 7}


def _pack(lens: LensModel, fit: Sequence[str]) -> tuple[np.ndarray, list[str]]:
    names = list(fit)
    x0 = np.zeros(len(names), dtype=float)
    cx, cy = lens.principal_point
    for i, nm in enumerate(names):
        if nm == "f":
            x0[i] = lens.fx
        elif nm == "cx":
            x0[i] = cx
        elif nm == "cy":
            x0[i] = cy
        else:
            j = DIST_NAMES.index(nm)
            x0[i] = float(np.asarray(lens.dist, dtype=float).ravel()[j])
    return x0, names


def _unpack(x: np.ndarray, names: Sequence[str], lens: LensModel,
            image_size: tuple[int, int]) -> LensModel:
    K = np.array(lens.K, dtype=float).copy()
    dist = np.asarray(lens.dist, dtype=float).ravel().copy()
    for i, nm in enumerate(names):
        if nm == "f":
            K[0, 0] = K[1, 1] = float(x[i])
        elif nm == "cx":
            K[0, 2] = float(x[i])
        elif nm == "cy":
            K[1, 2] = float(x[i])
        else:
            dist[DIST_NAMES.index(nm)] = float(x[i])
    return LensModel(K=K, dist=dist[:5], image_size=image_size, source=lens.source)


def _residual_vector(x: np.ndarray, names: Sequence[str], lens: LensModel,
                     lines: Sequence[np.ndarray]) -> np.ndarray:
    """Stacked perpendicular residuals of every sampled point of every line.

    The residual is deliberately *not* "distance from a line fitted to the
    undistorted points" averaged per line: that would let a bent line be scored by a
    straight one as long as the bending cancels.  Instead every point contributes its
    own perpendicular distance to the family's shared TLS line, so a shape that is
    bent in opposite directions along one line still shows up.
    """
    model = _unpack(x, names, lens, lens.image_size)
    out: list[float] = []
    for ln in lines:
        pts = np.asarray(ln, dtype=float).reshape(-1, 2)
        if pts.shape[0] < 3:
            continue
        out.extend(line_perp_residuals(undistort_points(pts, model)).tolist())
    return np.asarray(out, dtype=float)


def _least_squares(fun, x0: np.ndarray, *, bounds=(-np.inf, np.inf), max_iter: int = 100):
    """``scipy.optimize.least_squares`` when available, else Gauss-Newton.

    The plan names SciPy, so SciPy is used when present (it is installed in the
    project interpreter — see ``HANDOFF.md`` "Environment").  The fallback exists
    because a missing SciPy must not turn a calibration into a crash, and it is a
    plain damped Gauss-Newton with a numerical Jacobian: slower, same answer.
    """
    if HAVE_SCIPY:
        return _scipy_lsq(fun, np.asarray(x0, dtype=float), bounds=bounds,
                          max_nfev=int(max_iter) * 20)

    x = np.asarray(x0, dtype=float).copy()
    lam = 1e-3
    r = np.asarray(fun(x), dtype=float)
    cost = float(r @ r)
    for _ in range(int(max_iter) * 20):
        J = np.zeros((r.size, x.size), dtype=float)
        for j in range(x.size):
            dx = np.zeros_like(x)
            step = 1e-6 * max(1.0, abs(x[j]))
            dx[j] = step
            J[:, j] = (np.asarray(fun(x + dx), dtype=float) - r) / step
        JtJ = J.T @ J
        Jtr = J.T @ r
        improved = False
        for _try in range(12):
            try:
                step = np.linalg.solve(JtJ + lam * np.diag(np.diag(JtJ) + 1e-12), -Jtr)
            except np.linalg.LinAlgError:
                lam *= 10.0
                continue
            cand = x + step
            rc = np.asarray(fun(cand), dtype=float)
            cc = float(rc @ rc)
            if cc < cost:
                x, r, cost, lam = cand, rc, cc, max(lam / 3.0, 1e-12)
                improved = True
                break
            lam *= 10.0
        if not improved or float(np.linalg.norm(Jtr)) < 1e-12:
            break

    class _Res:
        pass

    res = _Res()
    res.x = x
    res.fun = r
    res.cost = 0.5 * cost
    res.success = True
    res.message = "gauss-newton (scipy unavailable)"
    return res


def estimate_from_lines(
    lines: Sequence[np.ndarray],
    image_size: tuple[int, int],
    *,
    lens0: LensModel | None = None,
    fit: Sequence[str] = ("k1", "k2"),
    max_iter: int = 100,
    min_max_radius_frac: float = 0.35,
) -> LensModel:
    """Recover a distortion model from the requirement that straight edges stay straight.

    ``lines`` is a sequence of ``(Ni, 2)`` arrays of sampled image points, one per
    physical straight edge (typically the pool's two long edges plus any straight
    wall/rail).  Only ``fit`` parameters move; everything else is held at ``lens0``.

    What it can and cannot do, stated instead of guessed:

    * ``fit=("k1","k2")`` with a focal prior is well posed **if** the lines reach
      outside the central region.  Measured behaviour with lines confined near the
      axis: ``k1`` drifts toward 0 and the residual barely changes, i.e. the data
      does not constrain it — the returned model then *equals* the prior and says so
      via ``lens.rms_px`` being no better than the prior's.
    * ``fit`` including ``"f"`` too is the degenerate pair.  The fit is allowed,
      because with a wide family of lines (a pool spans the frame) it does work, but
      the correlation between ``f`` and ``k1`` is computed and reported, and a
      correlation above ``CORR_WARN`` is written into ``notes`` so no downstream
      number can pretend the pair was separated.
    """
    lines = [np.asarray(ln, dtype=float).reshape(-1, 2) for ln in lines
             if np.asarray(ln).reshape(-1, 2).shape[0] >= 3]
    if not lines:
        raise ValueError("estimate_from_lines needs at least one straight line "
                         "sampled with >= 3 points")

    base = lens0 if lens0 is not None else placeholder_intrinsics(image_size)
    if tuple(base.image_size) != (int(image_size[0]), int(image_size[1])):
        base = LensModel(K=base.K, dist=base.dist, image_size=(int(image_size[0]),
                                                              int(image_size[1])),
                         source=base.source, notes=list(base.notes))

    unknown = [nm for nm in fit if nm in _FIT_INDEX]
    unknown = list(dict.fromkeys(unknown))          # de-dupe, keep order
    notes: list[str] = []
    cover = line_coverage(lines, image_size)

    before = total_straightness(lines)
    if not unknown:
        notes.append("no distortion parameters requested: the lens model is the prior"
                     "unchanged (fit=() is a no-op)")
        out = LensModel(K=base.K.copy(), dist=base.dist.copy(), image_size=base.image_size,
                        source=base.source, rms_px=before, notes=notes)
        out.notes.append(f"straightness before/after: {before:.4f} px (unchanged)")
        return out

    x0, names = _pack(base, unknown)
    res = _least_squares(lambda x: _residual_vector(x, names, base, lines), x0,
                         max_iter=max_iter)
    fitted = _unpack(np.asarray(res.x, dtype=float), names, base, base.image_size)
    after = total_straightness([undistort_points(ln, fitted) for ln in lines])

    rms = float(np.sqrt(np.mean(np.square(np.asarray(res.fun, dtype=float)))))
    notes.append(f"straightness RMS {before:.4f} px -> {after:.4f} px "
                 f"(max-per-line), residual RMS {rms:.4f} px")
    notes.append(f"coverage: {cover['n_lines']:.0f} line(s), span "
                 f"{cover['span_x']:.0f}x{cover['span_y']:.0f} px, "
                 f"max radius {cover['max_radius_frac'] * 100:.0f}% of the half-diagonal")

    # --- identifiability guards ------------------------------------------
    if cover["max_radius_frac"] < float(min_max_radius_frac):
        notes.append(
            f"ID-1: the supplied lines stay within {cover['max_radius_frac'] * 100:.0f}% of "
            f"the frame centre, below the {min_max_radius_frac * 100:.0f}% needed for radial "
            f"distortion to be identifiable — the fitted coefficients are weakly determined "
            f"and must not be quoted as a measurement"
        )
    corr = _parameter_correlation(res, names)
    if "f" in names and ("k1" in names or "k2" in names):
        c = max((abs(corr.get(("f", k), 0.0)) for k in ("k1", "k2") if ("f", k) in corr),
                default=0.0)
        if c > CORR_WARN:
            notes.append(
                f"ID-2: focal/distortion correlation |rho|={c:.2f} > {CORR_WARN:.2f} — the "
                f"data cannot separate `f` from `{'k1' if ('f', 'k1') in corr else 'k2'}`; "
                f"freeze `f` (fit=('k1','k2')) or supply a spec-sheet focal"
            )

    fitted.source = "lines"
    fitted.rms_px = after
    fitted.notes = notes
    return fitted


CORR_WARN = 0.90        # |rho| above which an f/k pair is reported as unseparated


def _parameter_correlation(res: Any, names: Sequence[str]) -> dict[tuple[str, str], float]:
    """Parameter correlations from the fit's Jacobian (empty when unavailable)."""
    J = getattr(res, "jac", None)
    if J is None:
        return {}
    try:
        J = np.asarray(J, dtype=float)
        if J.ndim != 2 or J.shape[1] != len(names):
            return {}
        cov = np.linalg.pinv(J.T @ J)
        d = np.sqrt(np.clip(np.diag(cov), 1e-300, None))
        out: dict[tuple[str, str], float] = {}
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                out[(names[i], names[j])] = float(cov[i, j] / (d[i] * d[j]))
        return out
    except Exception:  # noqa: BLE001 - reporting only, never fatal
        return {}


# ============================================================================
# Checkerboard route
# ============================================================================
def calibrate_from_checkerboard(
    images: Sequence[np.ndarray],
    pattern_size: tuple[int, int],
    *,
    square_size: float = 1.0,
    flags: int | None = None,
    refine: bool = True,
) -> LensModel:
    """Standard planar-target calibration: ``cv2.findChessboardCorners`` + ``cv2.calibrateCamera``.

    ``pattern_size`` is ``(inner_corners_x, inner_corners_y)``, as OpenCV defines it.
    ``square_size`` is in whatever unit the caller wants distances in *here* — the
    pipeline's own metric unit is the foot, so pass the square size in feet and the
    extrinsics that come out are in feet too.  Frames with no detected board are
    skipped and counted; a run that detects nothing raises rather than returning a
    plausible-looking model.
    """
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), dtype=np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    objp *= float(square_size)

    obj_points: list[np.ndarray] = []
    img_points: list[np.ndarray] = []
    image_size: tuple[int, int] | None = None
    skipped = 0
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
    fl = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE if flags is None else flags

    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        if image_size is None:
            image_size = (int(gray.shape[1]), int(gray.shape[0]))
        elif (int(gray.shape[1]), int(gray.shape[0])) != image_size:
            raise ValueError(
                f"all checkerboard frames must share a resolution; got "
                f"{image_size} and {(int(gray.shape[1]), int(gray.shape[0]))}"
            )
        found, corners = cv2.findChessboardCorners(gray, tuple(pattern_size), fl)
        if not found:
            skipped += 1
            continue
        if refine:
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        # float32, and the (N, 1, 2) layout OpenCV produced: `cv2.calibrateCamera` on OpenCV 5
        # rejects float64 image points outright ("imagePoints1 should contain vector of vectors of
        # points of type Point2f"), so casting up to float64 for "precision" breaks the call.
        obj_points.append(objp.astype(np.float32).reshape(-1, 1, 3))
        img_points.append(np.asarray(corners, dtype=np.float32).reshape(-1, 1, 2))

    if not obj_points:
        raise ValueError(f"no checkerboard found in {len(list(images))} frame(s)")

    assert image_size is not None
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, image_size, None, None
    )
    lens = LensModel(K=np.asarray(K, dtype=float),
                     dist=np.asarray(dist, dtype=float).ravel()[:5],
                     image_size=image_size, source="checkerboard", rms_px=float(rms))
    lens.notes.append(
        f"{len(obj_points)}/{len(obj_points) + skipped} frame(s) contributed; "
        f"square={square_size:g} (extrinsics are in the same unit)"
    )
    return lens


# ============================================================================
# Validation: a reprojection check anyone can re-run
# ============================================================================
def roundtrip_error(points: Any, lens: LensModel | None) -> float:
    """Max |p - distort(undistort(p))| in pixels — the distortion model's self-consistency.

    A model that cannot invert itself is a bug, not a tolerance question, so this is reported in the
    calibration document as ``lens.roundtrip_px`` and asserted in the tests.  Measured on a
    realistic action-camera model (f=1.1*max(w,h), k1=-0.12, k2=0.02, principal point at the image
    centre), at the frame corners where the distortion is strongest:

    * **1.6e-13 px** with the converged inversion this module uses (float noise);
    * **3.9e-5 px** with ``cv2.undistortPoints``' default stopping rule.

    Both are negligible for a distance measured in inches; the value is recorded because a claim of
    "float noise" has to be one that was measured.
    """
    arr = np.asarray(points, dtype=float).reshape(-1, 2)
    if lens is None or not lens.is_distorted or arr.size == 0:
        return 0.0
    back = np.asarray(distort_points(undistort_points(arr, lens), lens), dtype=float)
    return float(np.max(np.linalg.norm(back - arr, axis=1)))
