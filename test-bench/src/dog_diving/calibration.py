"""Calibration — mark schema, px→inches fit, validation gates, 1-ft grid.

M2 deliverable (PLAN.md §5).  This is the **metric layer**: it converts an image
x-coordinate into a distance in inches from the dock edge.  It is deliberately
independent of *how* the marks are obtained, which matters because the settled
user constraint (see the `dock-diving-scoring` skill, 2026-10-06) rejects a human
drawing the dock edge: the mark source is expected to become a seconds-long
gesture plus CV snapping, or a learned detector.  Whatever produces the marks,
this module consumes them.

Geometry (PLAN §4)
------------------
Side view, distance-away-from-dock maps to image x.  A least-squares fit
``d(x_px) -> inches`` is built through the origin of the first mark, so the fit
is a *difference* mapping and needs no separate intercept.  The reported
measurement is PLAN §4.2's formula verbatim::

    D_inches(x) = d(x) - d(O.x)          O = dock-edge line ∩ waterline

``d(O.x)`` is the offset between the dock edge and the 0-ft mark.  On a good
calibration it is ~0, and gate G2 checks exactly that (PLAN §4.3) — which is why
the dock edge and the 0-ft mark are carried separately instead of being assumed
equal.

Fixed 2026-10-07: ``list_setups()`` referenced a ``CALIB_DIR`` that was never
imported (plain ``NameError`` on any call, and ``dir.glob`` on ``None``), and the
persisted schema was missing ``created`` / ``model_sha`` / ``notes`` required by
PLAN §5.  Both are repaired here and pinned by ``test_m2_calib.py``.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

from .config import CALIB_DIR

FitType = Literal["linear", "quadratic"]
Mark = dict

REQUIRED_MARK_KEYS = ("label", "ft", "inch", "x_px", "y_px")


# --- fitting ----------------------------------------------------------------
def mark_inches(mark: Mark) -> float:
    """Nominal real-world distance of a mark from the dock, in inches."""
    return float(mark["ft"]) * 12.0 + float(mark["inch"])


def choose_fit_type(n_marks: int, fit_type: FitType | None = None) -> FitType:
    """PLAN §4.2: linear with 3-4 marks, quadratic with 5-6."""
    if fit_type is not None:
        return fit_type
    return "quadratic" if n_marks >= 5 else "linear"


def fit_from_marks(marks: list[Mark], fit_type: FitType | None = None) -> dict:
    """Least-squares fit ``d(x_px) -> inches``, constrained to pass through the
    first mark's x (the 0-ft mark), so distances come out as pure differences."""
    if len(marks) < 2:
        raise ValueError(f"need at least 2 marks to fit, got {len(marks)}")
    for m in marks:
        missing = [k for k in REQUIRED_MARK_KEYS if k not in m]
        if missing:
            raise ValueError(f"mark {m.get('label', m)!r} missing keys {missing}")

    xs = np.array([m["x_px"] for m in marks], dtype=float)
    ds = np.array([mark_inches(m) for m in marks], dtype=float)
    if np.any(np.diff(xs) <= 0):
        raise ValueError(f"mark x_px must be strictly increasing, got {xs.tolist()}")

    x0 = float(xs[0])
    x = xs - x0
    kind = choose_fit_type(len(marks), fit_type)

    if kind == "quadratic":
        A = np.vstack([x ** 2, x]).T
        (a, b), *_ = np.linalg.lstsq(A, ds, rcond=None)
        pred = a * x ** 2 + b * x
    else:
        A = x.reshape(-1, 1)
        (a,), *_ = np.linalg.lstsq(A, ds, rcond=None)
        b = 0.0
        pred = a * x

    resid = ds - pred
    return {
        "type": kind,
        "a": float(a),
        "b": float(b),
        "x0": x0,
        "residual_rms_in": float(np.sqrt(np.mean(resid ** 2))),
        "residual_max_in": float(np.max(np.abs(resid))),
        "per_mark_error_in": [float(r) for r in resid],
    }


# --- mapping ----------------------------------------------------------------
def map_inches(x_px: float, calib: dict) -> float:
    """Raw fitted mapping ``d(x)``: inches from the 0-ft mark."""
    f = calib.get("fit", calib)
    x = float(x_px) - f["x0"]
    return f["a"] * x * x + f["b"] * x if f["type"] == "quadratic" else f["a"] * x


def dock_zero_x(calib: dict) -> float:
    """``O.x`` — the x of the dock edge, i.e. the measurement origin (distance 0)."""
    return float(calib["dock_line"][0][0])


def distance_inches(x_px: float, calib: dict) -> float:
    """PLAN §4.2: ``D(x) = d(x) - d(O.x)`` — inches from the dock edge."""
    return map_inches(x_px, calib) - map_inches(dock_zero_x(calib), calib)


def inverse_ft(inches: float, calib: dict) -> float:
    """Invert the fit: inches -> x_px (used to draw the 1-ft grid)."""
    f = calib["fit"]
    if f["type"] == "quadratic":
        a, b, x0 = f["a"], f["b"], f["x0"]
        c = -float(inches)
        disc = b * b - 4 * a * c
        if a == 0 or disc < 0:
            return float("nan")
        roots = [(-b + np.sqrt(disc)) / (2 * a), (-b - np.sqrt(disc)) / (2 * a)]
        # pick the root on the measured side (x >= 0 in fit space)
        valid = [r for r in roots if r >= -1e-9]
        return x0 + (min(valid) if valid else max(roots))
    return f["x0"] + float(inches) / f["a"]


def grid_points(n_ft: int = 31, step_ft: int = 1) -> list[dict]:
    """0…(n_ft-1) ft ticks with their nominal real-world distance."""
    return [
        {"ft": ft, "inch": 0, "nominal_in": ft * 12}
        for ft in range(0, n_ft, step_ft)
    ]


# --- assembly + persistence -------------------------------------------------
def build(
    setup_id: str,
    image_size: tuple[int, int],
    dock_line: list,
    waterline: list,
    marks: list[Mark],
    *,
    model_sha: str = "",
    notes: str = "",
    fit_type: FitType | None = None,
    created: str | None = None,
) -> dict:
    """Full calibration document in the PLAN §5 schema."""
    return {
        "setup_id": setup_id,
        "created": created or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "image_size": list(image_size),
        "dock_line": dock_line,
        "waterline": waterline,
        "marks": marks,
        "fit": fit_from_marks(marks, fit_type),
        "model_sha": model_sha,
        "notes": notes,
    }


def save(calib: dict, path: str | Path) -> None:
    """Atomic write — a half-written calibration is worse than none."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.json")
    tmp.write_text(json.dumps(calib, indent=2))
    shutil.move(str(tmp), str(path))


def load(path: str | Path) -> dict:
    path = Path(path)
    with path.open() as f:
        calib = json.load(f)
    validate_schema(calib)
    return calib


def list_setups(dir: str | Path | None = None) -> list[Path]:
    d = Path(dir) if dir is not None else CALIB_DIR
    if not d.exists():
        return []
    return sorted(d.glob("*.json"))


def validate_schema(calib: dict) -> None:
    required = ("setup_id", "created", "image_size", "dock_line", "waterline",
                "marks", "fit", "model_sha", "notes")
    missing = [k for k in required if k not in calib]
    if missing:
        raise ValueError(f"calibration is missing required keys: {missing}")
    if len(calib["marks"]) < 2:
        raise ValueError("calibration needs at least 2 marks")


def setup_path(setup_id: str, dir: str | Path | None = None) -> Path:
    d = Path(dir) if dir is not None else CALIB_DIR
    return d / f"{setup_id}.json"


# --- validation gates (PLAN §4.3) -------------------------------------------
def validate_gates(calib: dict) -> dict[str, dict]:
    """G1-G5.  Returns ``{gate: {"pass": bool, "warn": bool, "detail": str}}``.

    G3 is tiered per PLAN §4.3: warn above 2 in, *block* above 3 in — so the gate
    passes at <=3 in and raises ``warn`` between 2 and 3.
    """
    marks = calib["marks"]
    xs = [float(m["x_px"]) for m in marks]
    nominal = [mark_inches(m) for m in marks]
    gates: dict[str, dict] = {}

    # G1: strictly increasing x, and each consecutive pair spans its nominal gap
    increasing = all(xs[i] < xs[i + 1] for i in range(len(xs) - 1))
    gaps = [nominal[i + 1] - nominal[i] for i in range(len(nominal) - 1)]
    gates["G1"] = {
        "pass": increasing and all(g > 0 for g in gaps),
        "warn": False,
        "detail": f"x increasing={increasing}; nominal gaps (in)={[round(g, 1) for g in gaps]}",
    }

    # G2: dock edge vs 0-ft mark agreement — the strongest single sanity check
    d0 = map_inches(dock_zero_x(calib), calib)
    gates["G2"] = {
        "pass": abs(d0) <= 3.0,
        "warn": 3.0 < abs(d0) <= 6.0,
        "detail": f"|d(O.x)|={abs(d0):.2f} in (warn >3, block >6)",
    }

    # G3: fit residual RMS — warn >2, block >3
    rms = float(calib["fit"]["residual_rms_in"])
    gates["G3"] = {
        "pass": rms <= 3.0,
        "warn": rms > 2.0,
        "detail": f"RMS={rms:.2f} in (warn >2, block >3)",
    }

    # G4: visual only — projecting the grid needs a frame, so it cannot be judged here
    gates["G4"] = {
        "pass": True,
        "warn": False,
        "detail": "visual-only: user compares the projected 1-ft grid against the painted markings",
    }

    # G5: monotonic over the measurement span (no fold-back)
    lo, hi = min(xs), max(xs)
    n = 50
    vals = [map_inches(lo + (hi - lo) * i / (n - 1), calib) for i in range(n)]
    monotonic = all(vals[i] <= vals[i + 1] + 1e-9 for i in range(n - 1))
    gates["G5"] = {
        "pass": monotonic,
        "warn": False,
        "detail": f"monotonic over [{lo:.0f}, {hi:.0f}] px = {monotonic}",
    }
    return gates


def gates_pass(gates: dict[str, dict]) -> bool:
    return all(g["pass"] for g in gates.values())


# --- line / mark snapping (PLAN §5 calibration.py) ---------------------------
# "Line snapping helpers (refine_dock_edge, refine_waterline) using gradient/Hough
# search near the clicked line — keeps calibration reproducible without demanding
# pixel-perfect clicking."
#
# This is what reconciles the click tool with the M2 measurement: a click can only
# land within a pixel or so, and the V2 budget needs marks localised to ~0.2 px.
# The click is a *seed*; the sub-pixel location comes from the image.
#
# Every refiner shares one primitive: find the strongest 1-D response near a seed,
# then interpolate its peak with a parabola fitted to the three samples around it.


def _gray(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 3:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float64)
    return frame.astype(np.float64)


def _parabolic_peak(vals: np.ndarray, k: int) -> float:
    """Sub-pixel offset of the peak of ``vals`` around index ``k`` (in samples)."""
    if 0 < k < len(vals) - 1:
        y0, y1, y2 = float(vals[k - 1]), float(vals[k]), float(vals[k + 1])
        denom = y0 - 2.0 * y1 + y2
        if denom != 0.0:
            return float(np.clip(0.5 * (y0 - y2) / denom, -1.0, 1.0))
    return 0.0


def _tick_response(profile: np.ndarray, bg_radius: int) -> np.ndarray:
    """Top-hat response: local background minus the profile, clipped at 0.

    A tick is a *narrow dark line*, so the signal is its contrast against the local
    neighbourhood — not its absolute darkness.  Searching for the globally darkest
    sample instead is what made the first version snap a mark onto the dock edge
    18 px away: the dock's shadowed water is darker than any painted tick.  The box
    background is wider than a tick and narrower than the tick spacing.
    """
    k = 2 * int(bg_radius) + 1
    row = profile.astype(np.float32).reshape(1, -1)
    bg = cv2.boxFilter(row, -1, (k, 1), borderType=cv2.BORDER_REPLICATE).ravel()
    return np.clip(bg - profile, 0.0, None)


def _peak_near(profile: np.ndarray, seed: float, radius: int) -> float | None:
    """Sub-pixel index of the strongest |gradient| within ``radius`` of ``seed``.

    Returns ``None`` when the winner sits on the *window boundary*: that means the
    true peak is probably outside the window, and reporting the boundary position
    would silently track the seed instead of the feature.
    """
    seed_i = int(round(seed))
    lo = max(1, seed_i - radius)
    hi = min(len(profile) - 1, seed_i + radius + 1)
    if hi - lo < 3:
        return None
    g = np.abs(np.diff(profile[lo:hi].astype(np.float64)))
    k = int(np.argmax(g))
    if k == 0 or k == len(g) - 1:
        return None
    # g[i] is centred at lo + i + 0.5 in profile-index space
    return lo + k + 0.5 + _parabolic_peak(g, k)


def _robust_line(pts: list[tuple[float, float]], tol: float = 3.0) -> tuple[float, float] | None:
    """Fit ``y = m*x + c`` (or ``x = m*y + c``) with one outlier-rejection pass.

    Returns ``(m, c)`` or ``None`` when too few points survive.
    """
    if len(pts) < 3:
        return None
    arr = np.asarray(pts, dtype=float)
    t, s = arr[:, 0], arr[:, 1]
    for _ in range(2):
        if len(t) < 3:
            return None
        m, c = np.polyfit(t, s, 1)
        resid = np.abs(s - (m * t + c))
        med = float(np.median(resid))
        keep = resid <= max(tol, 3.0 * med)
        if keep.all():
            break
        t, s = t[keep], s[keep]
    m, c = np.polyfit(t, s, 1)
    return float(m), float(c)


def refine_line(
    frame: np.ndarray,
    p1: tuple[float, float],
    p2: tuple[float, float],
    *,
    radius: int = 25,
    axis: str = "x",
    band: int = 2,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Snap a clicked near-straight line to the image, sub-pixel.

    ``axis="x"``: the line is near-vertical; scan rows and locate the x of the
    strongest horizontal gradient.  ``axis="y"``: near-horizontal; scan columns and
    locate the y of the strongest vertical gradient.  The two endpoints are
    re-expressed on the robustly fitted line so the caller keeps their extent.
    """
    gray = _gray(frame)
    h, w = gray.shape
    x1, y1 = float(p1[0]), float(p1[1])
    x2, y2 = float(p2[0]), float(p2[1])
    pts: list[tuple[float, float]] = []

    if axis == "x":
        n = max(int(round(abs(y2 - y1))), 8)
        for i in range(n + 1):
            y = int(round(y1 + (y2 - y1) * i / n))
            if not (0 <= y < h):
                continue
            seed_x = x1 + (x2 - x1) * i / n
            row = gray[max(0, y - band):min(h, y + band + 1), :].mean(axis=0)
            xs = _peak_near(row, seed_x, radius)
            if xs is not None and 1 < xs < w - 1:
                pts.append((float(y), xs))
    elif axis == "y":
        n = max(int(round(abs(x2 - x1))), 8)
        for i in range(n + 1):
            x = int(round(x1 + (x2 - x1) * i / n))
            if not (0 <= x < w):
                continue
            seed_y = y1 + (y2 - y1) * i / n
            col = gray[:, max(0, x - band):min(w, x + band + 1)].mean(axis=1)
            ys = _peak_near(col, seed_y, radius)
            if ys is not None and 1 < ys < h - 1:
                pts.append((float(x), ys))
    else:
        raise ValueError(f"axis must be 'x' or 'y', got {axis!r}")

    # NOTE: a Hough/intercept-vote pre-filter was tried here (2026-10-07) to keep
    # per-row estimates from splitting across two nearby edges. It measured *worse*
    # on a clean synthetic line (spread 0.008 -> 0.682 px) because the voted bin
    # shifts as rows jitter, and it did not help the real frame either (30.8 -> 31.4
    # px). Reverted. The practical defence is a SMALL radius: with the search narrow
    # each row locks onto the nearest edge rather than the globally strongest one.
    fit = _robust_line(pts)
    if fit is None:
        return (x1, y1), (x2, y2)
    m, c = fit
    if axis == "x":
        return (m * y1 + c, y1), (m * y2 + c, y2)
    return (x1, m * x1 + c), (x2, m * x2 + c)


def refine_dock_edge(frame, p1, p2, *, radius: int = 25):
    """Snap the dock-edge line (near-vertical) to the strongest edge."""
    return refine_line(frame, p1, p2, radius=radius, axis="x")


def refine_waterline(frame, p1, p2, *, radius: int = 25):
    """Snap the waterline (near-horizontal) to the strongest edge."""
    return refine_line(frame, p1, p2, radius=radius, axis="y")


def refine_mark_x(
    frame: np.ndarray,
    y: float,
    x_seed: float,
    *,
    radius: int = 12,
    band: int = 12,
    bg_radius: int = 25,
) -> float:
    """Snap a foot-marking tick to sub-pixel x.

    Response is local contrast (``_tick_response``), then the sub-pixel centre is the
    centroid of a narrow window around the peak.

    ``radius`` MUST stay below the tick spacing: adjacent 1-ft ticks are only ~10-18 px
    apart at these pool distances, and a wider search can silently snap a mark onto
    its *neighbouring* tick — a full foot of error that looks like a clean result.
    ``snap_marks`` derives a safe radius automatically for this reason.
    """
    gray = _gray(frame)
    h, w = gray.shape
    y0, y1_ = max(0, int(round(y)) - band), min(h, int(round(y)) + band + 1)
    if y1_ - y0 < 1:
        return float(x_seed)
    profile = gray[y0:y1_, :].mean(axis=0)
    resp = _tick_response(profile, bg_radius)

    seed_i = int(round(x_seed))
    lo, hi = max(1, seed_i - radius), min(w - 1, seed_i + radius + 1)
    if hi - lo < 3:
        return float(x_seed)
    seg = resp[lo:hi]
    peak = float(seg.max())
    if peak <= 0.0:
        return float(x_seed)              # no tick above the local background

    # Sub-pixel centre = centroid of the response over the window, masked to the
    # dominant peak. A bare argmax picks one sample and jumps between them on a
    # blurred, flat-topped tick: measured spread 27 px at 15/16 ft versus 1.0/0.0 px
    # for the masked centroid (seed jittered +/-4 px).
    wts = np.where(seg >= 0.5 * peak, seg, 0.0)
    total = float(wts.sum())
    if total <= 0.0:
        return float(x_seed)
    return float((np.arange(lo, hi, dtype=np.float64) * wts).sum() / total)


def safe_mark_radius(marks: list[Mark], requested: int) -> int:
    """Cap the search radius at just under half the closest *nominal* mark spacing.

    Prevents the neighbouring-tick failure mode in ``refine_mark_x``: adjacent 1-ft
    ticks are only ~10-18 px apart, so a wide search can pull a mark onto its
    neighbour — a silent full-foot error that still looks like a clean result.

    The spacing is computed from the marks' *nominal* ft values scaled by a px-per-ft
    estimated across the widest-separated pair, NOT from the clicked x values. Using
    the clicks is circular: a 15 ft click nudged toward 16 ft shrinks the apparent gap,
    which shrinks the radius, which then excludes the very tick being looked for
    (measured: radius collapsed to 4 px and the mark was left unsnapped).
    """
    if len(marks) < 2:
        return requested
    pairs = sorted((float(m["ft"]) + float(m["inch"]) / 12.0, float(m["x_px"])) for m in marks)
    (f_lo, x_lo), (f_hi, x_hi) = pairs[0], pairs[-1]
    if f_hi - f_lo <= 1e-6:
        return requested
    px_per_ft = (x_hi - x_lo) / (f_hi - f_lo)
    nominal_gaps = [b[0] - a[0] for a, b in zip(pairs, pairs[1:]) if (b[0] - a[0]) > 1e-6]
    if not nominal_gaps or px_per_ft <= 0:
        return requested
    return int(max(3, min(requested, 0.45 * px_per_ft * min(nominal_gaps))))


def snap_marks(
    frame: np.ndarray,
    marks: list[Mark],
    *,
    radius: int = 12,
    band: int = 12,
    bg_radius: int = 25,
) -> tuple[list[Mark], list[float]]:
    """Snap every mark's x to the image. Returns ``(marks, dx_per_mark)``.

    ``dx_per_mark`` is how far snapping moved each click, in pixels — the operator's
    click error, which is exactly the quantity the V2 budget is spent on. A shift
    approaching the safe radius means the click was at the tolerance limit, and a
    shift of exactly 0.0 means no tick was found (the mark keeps its raw click), so
    each mark carries a ``snapped`` flag rather than leaving that indistinguishable
    from a perfectly-placed click.
    """
    r = safe_mark_radius(marks, radius)
    out, shifts = [], []
    for m in marks:
        x_snapped = refine_mark_x(frame, m["y_px"], m["x_px"],
                                  radius=r, band=band, bg_radius=bg_radius)
        shifts.append(x_snapped - float(m["x_px"]))
        out.append({**m, "x_px": x_snapped, "snapped": x_snapped != float(m["x_px"])})
    return out, shifts
