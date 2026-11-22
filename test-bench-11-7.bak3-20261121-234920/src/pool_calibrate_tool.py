"""Pool-translation calibration tool — Phase 2's click collection, and the document builder's shell.

Collects the clicks `pool-translation-plan.txt` needs and hands them to
:func:`dog_diving.pool.calib.build`, which does all the geometry.  This file deliberately contains no
maths of its own: everything it does is either a prompt, a window, or an argument, so the document a
reviewer reads and the document the GUI produces cannot diverge.

The click sequence (PLAN.md §12 item 4 is still open on exactly where pools paint their marks, so
the marks are the operator's to name):

1. ``--marks`` ft values — the pool's distance marks, all on one row.  These are the **metric
   anchor**: without at least one real length nothing downstream has a scale.
2. The **across-pool** line family — the two end walls, a lane divider, the ladder rails.  Two lines
   minimum; their vanishing point is Phase 3's transverse direction.
3. The **along-pool** line family — the pool's two long edges, a third parallel feature if visible.
   Their vanishing point, together with the transverse one, also recovers the focal length.
4. The far long edge, as one line — route B's transverse extent.
5. The **rim pair**: a coping/wall corner where the top edge and the water's contact are both
   visible, clicked top-then-contact, on the *same vertical*.  This is what makes the occluded water
   level observable (Phase 5).

Lens handling, and why "lines" is a first-class option
-----------------------------------------------------
Phase 1 says intrinsics may come from "standard calibration patterns **or straight pool-edge line
fitting**".  Both are offered:

``--lens-source prior``
    no distortion correction; the focal comes from ``--focal-px``/``--focal-ratio`` (default 1.0 x
    the long edge).  Fast, honest, and labelled ``assumed`` in the document.
``--lens-source lines``
    fit ``k1``/``k2`` from the requirement that the **clicked** pool edges be straight — the plan's
    own fallback, and free once the operator has already clicked those edges.  The fit runs on the
    *raw* clicks, so there is no circularity with the undistortion that follows.
``--lens-source checkerboard --checkerboard DIR``
    the textbook route; measures the whole model.  Needs images the corpus does not ship.

Whatever is chosen is recorded in the document, with the fit's own residual, so a distance can never
be quoted without its lens provenance.

Usage
-----
Interactive (needs a display)::

    python src/pool_calibrate_tool.py --clip data/clips/18-07.mp4 --frame-index 120 \\
        --setup-id poolB --marks 0,1,15,16,30,31 --lens-source lines

Headless (scripted / CI) — ``--clicks`` is a JSON file in exactly
:func:`dog_diving.pool.calib.build`'s ``clicks`` shape::

    python src/pool_calibrate_tool.py --frame still.png --setup-id t \\
        --clicks clicks.json --print-gates

Keys in the interactive window: ``z`` undo, ``k`` skip an optional group, ``g`` toggle the metric
grid, ``w`` cycle the water method, ``s`` save, ``q`` quit.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dog_diving import video as video_mod  # noqa: E402
from dog_diving.config import (  # noqa: E402
    MODEL_PATH,
    POOL_FOCAL_RATIO,
    POOL_FREEBOARD_IN,
    POOL_MARKS_ON_EDGE,
    POOL_WALL_AXIS,
    POOL_WALL_VALUE,
    POOL_WATER_METHOD,
    POOL_WIDTH_FT,
)
from dog_diving.pool import calib as C  # noqa: E402
from dog_diving.pool import grid as GR  # noqa: E402
from dog_diving.pool import lens as L  # noqa: E402

# The venue's declared defaults live in ``config.py`` (the POOL_* block) so there is one place to
# change them for a different pool; these are the CLI's names for the same values.
DEFAULT_MARKS_FT = "0,1,15,16,30,31"
DEFAULT_WIDTH_FT = float(POOL_WIDTH_FT)
DEFAULT_FREEBOARD_IN = float(POOL_FREEBOARD_IN)


# ============================================================================
# The request sequence
# ============================================================================
def build_requests(
    marks_ft: Sequence[float],
    *,
    n_transverse_lines: int = 2,
    n_along_lines: int = 3,
    want_edge: bool = True,
    want_rim: bool = True,
) -> list[dict[str, Any]]:
    """The ordered click requests.  One list, so the prompt, the undo stack and the document agree."""
    reqs: list[dict[str, Any]] = []
    for ft in marks_ft:
        reqs.append({"kind": "mark", "ft": float(ft), "point": 0,
                     "prompt": f"click the {float(ft):g} ft mark"})
    for i in range(int(n_transverse_lines)):
        for pt in (0, 1):
            reqs.append({"kind": "line", "group": "vp_transverse", "index": i, "point": pt,
                         "optional": i >= 2,
                         "prompt": f"ACROSS-pool line {i + 1}, end {pt + 1}/2 "
                                   f"(pool end wall / lane / rail)"})
    for i in range(int(n_along_lines)):
        for pt in (0, 1):
            reqs.append({"kind": "line", "group": "vp_along", "index": i, "point": pt,
                         "optional": i >= 2,
                         "prompt": f"ALONG-pool line {i + 1}, end {pt + 1}/2 "
                                   f"(long pool edge)"})
    if want_edge:
        for pt in (0, 1):
            reqs.append({"kind": "edge", "point": pt,
                         "prompt": f"the FAR long edge (the one the marks are NOT on), "
                                   f"end {pt + 1}/2"})
    if want_rim:
        reqs.append({"kind": "rim", "role": "top", "point": 0,
                     "prompt": "click the TOP of a coping/wall corner (rim top)"})
        reqs.append({"kind": "rim", "role": "contact", "point": 0,
                     "prompt": "now click where the WATER meets that same corner "
                               "(same vertical)"})
    return reqs


@dataclass
class PoolCalibrationSession:
    """Clicks in, clicks out.  No cv2 calls, so it is drivable headlessly and testable."""

    frame_shape: tuple[int, int]                    # (height, width)
    requests: list[dict[str, Any]]
    setup_id: str = "pool"
    lens: L.LensModel | None = None
    width_ft: float = DEFAULT_WIDTH_FT
    freeboard_in: float = DEFAULT_FREEBOARD_IN
    wall: dict[str, Any] | None = None
    marks_on_edge: str = "near"
    water_method: str = "rim_pair"
    notes: str = ""
    model_sha: str = ""
    clicks: dict[str, Any] = field(default_factory=dict)
    history: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.clicks.setdefault("marks", [])
        self.clicks.setdefault("vp_transverse", {"lines": []})
        self.clicks.setdefault("vp_along", {"lines": []})
        self.clicks.setdefault("rim", {})

    # --- prompt state ----------------------------------------------------
    @property
    def done(self) -> bool:
        return len(self.history) >= len(self.requests)

    @property
    def prompt(self) -> str:
        if self.done:
            return ("complete - 'g' to check the grid against the marks, "
                    "'w' to change the water method, 's' to save")
        return self.requests[len(self.history)]["prompt"]

    # --- input -----------------------------------------------------------
    def click(self, x: float, y: float) -> str:
        if self.done:
            return "already complete"
        req = self.requests[len(self.history)]
        self._apply(req, float(x), float(y))
        self.history.append(req)
        return self.prompt

    def _apply(self, req: dict[str, Any], x: float, y: float) -> None:
        kind = req["kind"]
        if kind == "mark":
            self.clicks["marks"].append({"ft": float(req["ft"]), "inch": 0.0,
                                         "label": f"{float(req['ft']):g}'",
                                         "x_px": x, "y_px": y})
        elif kind == "line":
            lines = self.clicks[req["group"]]["lines"]
            if req["point"] == 0:
                lines.append([x, y, float("nan"), float("nan")])
            else:
                if len(lines) <= req["index"]:
                    lines.append([float("nan"), float("nan"), x, y])
                else:
                    lines[req["index"]][2] = x
                    lines[req["index"]][3] = y
        elif kind == "edge":
            seg = self.clicks.get("edge_line") or [float("nan")] * 4
            seg[0 if req["point"] == 0 else 2] = x
            seg[1 if req["point"] == 0 else 3] = y
            self.clicks["edge_line"] = seg
        elif kind == "rim":
            self.clicks["rim"][req["role"]] = [x, y]
        else:
            raise ValueError(f"unknown request kind {kind!r}")

    def undo(self) -> str:
        if not self.history:
            return self.prompt
        req = self.history.pop()
        kind = req["kind"]
        if kind == "mark":
            self.clicks["marks"].pop()
        elif kind == "line":
            lines = self.clicks[req["group"]]["lines"]
            if req["point"] == 0 and lines:
                lines.pop()
            elif lines:
                lines[req["index"]][2] = float("nan")
                lines[req["index"]][3] = float("nan")
        elif kind == "edge":
            self.clicks.pop("edge_line", None)
        elif kind == "rim":
            self.clicks["rim"].pop(req["role"], None)
        return self.prompt

    def skip_optional(self) -> str:
        """Drop every unclicked optional request (the second/third lines are nice-to-have)."""
        while not self.done and self.requests[len(self.history)].get("optional"):
            self.history.append(self.requests[len(self.history)])
        return self.prompt

    # --- output ----------------------------------------------------------

    def build(self) -> dict[str, Any]:
        if not self.done:
            raise ValueError(f"the click sequence is incomplete — {self.prompt}")
        if len(self.clicks["marks"]) < 2:
            raise ValueError("at least two distance marks are required as the metric anchor")
        clicks = {k: v for k, v in self.clicks.items() if v not in (None, {}, [])}
        clicks["wall"] = self.wall or {"axis": "y", "value": 0.0}
        clicks["water"] = self._water_spec()
        doc = C.build(self.setup_id, clicks=clicks, lens=self.lens,
                      width_ft=self.width_ft, marks_on_edge=self.marks_on_edge,
                      model_sha=self.model_sha, notes=self.notes,
                      image_size=(self.frame_shape[1], self.frame_shape[0]))
        return doc

    def _water_spec(self) -> dict[str, Any]:
        if self.water_method == "known_freeboard":
            return {"method": "known_freeboard", "freeboard_in": self.freeboard_in,
                    "source": "operator"}
        return {"method": self.water_method}

    def gates(self) -> dict[str, Any]:
        return self.build()["gates"]


# ============================================================================
# Overlay
# ============================================================================
def draw_overlay(frame: np.ndarray, session: PoolCalibrationSession, *,
                 lens: L.LensModel | None = None, show_grid: bool = False,
                 grid_upto_ft: float = 31.0) -> np.ndarray:
    """Everything clicked so far, plus optionally the finished metric lattice.

    The overlay draws **raw** pixels (it is composited on a raw frame), so any geometry that came
    out of the undistorted space is pushed back through the lens by the caller or by
    :func:`dog_diving.pool.grid.draw_grid`, which takes ``lens`` for exactly that.
    """
    out = frame.copy()
    h, w = out.shape[:2]
    cl = session.clicks

    for m in cl["marks"]:
        p = (int(round(m["x_px"])), int(round(m["y_px"])))
        cv2.drawMarker(out, p, (0, 0, 255), cv2.MARKER_TILTED_CROSS, 18, 2)
        cv2.putText(out, f"{m['ft']:g}'", (p[0] + 8, p[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 0, 255), 2)

    for key, colour in (("vp_transverse", (255, 0, 255)), ("vp_along", (0, 255, 0))):
        for seg in cl[key]["lines"]:
            if not np.all(np.isfinite(np.asarray(seg, dtype=float))):
                continue
            cv2.line(out, (int(seg[0]), int(seg[1])), (int(seg[2]), int(seg[3])), colour, 2)

    seg = cl.get("edge_line")
    if seg is not None and np.all(np.isfinite(np.asarray(seg, dtype=float))):
        cv2.line(out, (int(seg[0]), int(seg[1])), (int(seg[2]), int(seg[3])), (255, 255, 0), 2)

    rim = cl.get("rim") or {}
    if rim.get("top"):
        cv2.drawMarker(out, (int(rim["top"][0]), int(rim["top"][1])), (0, 165, 255),
                       cv2.MARKER_DIAMOND, 20, 2)
    if rim.get("contact"):
        cv2.drawMarker(out, (int(rim["contact"][0]), int(rim["contact"][1])), (0, 165, 255),
                       cv2.MARKER_STAR, 20, 2)
    if rim.get("top") and rim.get("contact"):
        cv2.line(out, (int(rim["top"][0]), int(rim["top"][1])),
                 (int(rim["contact"][0]), int(rim["contact"][1])), (0, 165, 255), 1)

    if show_grid and session.done:
        try:
            doc = session.build()
            pose, water = C.pose_from_document(doc), C.water_from_document(doc)
            out = GR.draw_grid(out, pose, water, lens=lens, along_ft=grid_upto_ft,
                               width_ft=session.width_ft)
        except Exception as exc:  # noqa: BLE001 - the overlay must never crash the loop
            cv2.putText(out, f"grid unavailable: {exc}", (12, h - 16), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 0, 255), 2)

    cv2.rectangle(out, (0, 0), (w, 34), (0, 0, 0), -1)
    cv2.putText(out, session.prompt, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    if session.done:
        try:
            line = C.gate_summary(session.gates())
            cv2.putText(out, line, (10, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 0) if C.gates_pass(session.gates()) else (0, 0, 255), 2)
        except Exception:  # noqa: BLE001
            pass
    return out


# ============================================================================
# Lens sources
# ============================================================================
def lens_from_lines(session: PoolCalibrationSession, base: L.LensModel,
                    *, frame: np.ndarray | None = None, samples: int = 60,
                    radius: int = 10, fit: Sequence[str] = ("k1", "k2")) -> L.LensModel:
    """Phase 1's "straight pool-edge line fitting", driven by the edges already clicked.

    The fit runs on the **raw** clicks — straightness in the raw frame is exactly the observation —
    so there is no circularity with the undistortion that consumes the result.

    A two-point click is a **chord**, and a chord is straight by construction: fitting straightness to
    endpoint pairs alone is a no-op that always returns ``k1 = 0`` (measured — the first version of
    this function did exactly that and reported ``k1 = 0.0000`` against a true ``-0.12``).  So when a
    ``frame`` is available the edge is **traced** between the two clicks: at each of ``samples``
    positions along the chord the strongest gradient within ``radius`` pixels perpendicular to it is
    found and taken as the edge's true sub-pixel position.  Those traced points carry the curvature.
    Without a frame the function says the fit is unidentifiable instead of returning a confident zero.
    """
    segs = [s for key in ("vp_transverse", "vp_along") for s in session.clicks[key]["lines"]
            if np.all(np.isfinite(np.asarray(s, dtype=float)))]
    edges = session.clicks.get("edge_line")
    if edges is not None and np.all(np.isfinite(np.asarray(edges, dtype=float))):
        segs.append(edges)
    if not segs:
        raise ValueError("--lens-source lines needs the line clicks first")

    lines: list[np.ndarray] = []
    traced = 0
    for s in segs:
        p0, p1 = np.array([s[0], s[1]], dtype=float), np.array([s[2], s[3]], dtype=float)
        ts = np.linspace(0.0, 1.0, int(samples))
        chord = np.array([p0 + t * (p1 - p0) for t in ts])
        if frame is None:
            lines.append(chord)
            continue
        traced_pts = _trace_edge(frame, p0, p1, radius=int(radius))
        if traced_pts is not None and traced_pts.shape[0] >= 10:
            lines.append(traced_pts)
            traced += 1
        else:
            lines.append(chord)

    h, w = session.frame_shape
    fitted = L.estimate_from_lines(lines, (w, h), lens0=base, fit=tuple(fit))

    if frame is None or traced == 0:
        fitted.notes.append(
            "IDENTIFIABILITY: the edges were supplied as two-point clicks only, so the sampled "
            "points lie on a straight chord and straightness carries no information about radial "
            "distortion. The coefficients returned are the prior unchanged; pass a frame so the "
            "edges can be traced, or use a checkerboard."
        )
        return _lens_unchanged(base, fitted.rms_px, fitted.notes)

    # --- self-check: did the fit actually improve the thing it optimises? ----------------
    # This is the guard that stops the route reporting a number it cannot support.  A traced edge
    # that wandered onto neighbouring structure, or a frame whose edges are too blurred to locate,
    # both produce a fit that moves k1 substantially while leaving the straightness residual where it
    # was — and a moved k1 with an unmoved residual is a fit that has absorbed noise, not lens.
    # Measured on the polyline-rendered fixture: the fit reported k1 = -0.074 against a true -0.12
    # while the residual went *up*.  So: if the residual is not materially better than the prior's,
    # refuse and keep the prior.
    before = L.total_straightness([L.undistort_points(ln, base) for ln in lines])
    after = L.total_straightness([L.undistort_points(ln, fitted) for ln in lines])
    improvement = 1.0 - (after / before) if before > 0 else 0.0
    fitted.notes.append(
        f"straightness after undistortion: {before:.3f} px with the prior, {after:.3f} px with the "
        f"fit ({improvement * 100:+.1f}%); {traced}/{len(segs)} clicked edge(s) traced in the frame"
    )
    if improvement < 0.20:
        fitted.notes.append(
            f"REFUSED: the fit does not improve the straightness residual by a material amount "
            f"({improvement * 100:+.1f}%, threshold +20%). A k1 that moves while the residual does "
            f"not is the fit absorbing tracing error rather than lens distortion, so the prior is "
            f"returned instead of the coefficients. Trace with a cleaner/higher-contrast frame, click "
            f"more edges, or use a checkerboard."
        )
        return _lens_unchanged(base, after, fitted.notes)
    return fitted


def _lens_unchanged(base: L.LensModel, rms_px: float | None, notes: list[str]) -> L.LensModel:
    """The prior, with the reason attached — the honest answer when a fit cannot be supported."""
    return L.LensModel(K=np.asarray(base.K, dtype=float).copy(),
                       dist=np.asarray(base.dist, dtype=float).copy(),
                       image_size=base.image_size, source="assumed", rms_px=rms_px,
                       notes=list(notes))


def _perp_offset(gray: np.ndarray, base: np.ndarray, perp: np.ndarray, *, radius: float,
                 min_response: float = 4.0) -> float | None:
    """Sub-pixel perpendicular offset of the edge nearest ``base``, or ``None`` if not found.

    The perpendicular gradient is sampled on a fine grid and the edge position taken as the
    **masked centroid** of that response rather than its argmax.  Both choices are forced by
    measurement: a real edge is a *thick* line, so its perpendicular gradient has **two** peaks (one
    per border), and an argmax locks onto whichever border is marginally stronger — which put the
    traced points a measured ~3 px off the centreline, with a *varying* offset along the line.  That
    variation is precisely the shape signal a distortion fit reads, so the estimator error was being
    measured as lens error (``k1 = +0.24`` against a true ``-0.12``).  A response-weighted centroid
    over both peaks recovers the centreline, the same lesson the ``dock-diving-scoring`` skill
    records for mark snapping.
    """
    h, w = gray.shape

    def sample(q: np.ndarray) -> float:
        x, y = float(q[0]), float(q[1])
        x0, y0 = int(np.floor(x)), int(np.floor(y))
        if not (0 <= x0 < w - 1 and 0 <= y0 < h - 1):
            return float("nan")
        fx, fy = x - x0, y - y0
        return float((1 - fx) * (1 - fy) * gray[y0, x0] + fx * (1 - fy) * gray[y0, x0 + 1]
                     + (1 - fx) * fy * gray[y0 + 1, x0] + fx * fy * gray[y0 + 1, x0 + 1])

    offsets = np.linspace(-float(radius), float(radius), 4 * int(max(radius, 1)) + 1)
    resp = np.array([abs(sample(base + (o + 1.0) * perp) - sample(base + (o - 1.0) * perp)) / 2.0
                     for o in offsets])
    if not np.any(np.isfinite(resp)):
        return None
    peak = float(np.nanmax(resp))
    if peak < float(min_response):
        return None
    wts = np.where(resp >= 0.5 * peak, resp, 0.0)
    total = float(np.nansum(wts))
    if total <= 0.0:
        return None
    off = float(np.nansum(wts * offsets) / total)
    return None if abs(off) >= float(radius) else off


def _trace_edge(frame: np.ndarray, p0: np.ndarray, p1: np.ndarray, *,
                radius: int = 8, samples: int = 60, min_response: float = 4.0,
                step: float = 5.0) -> np.ndarray | None:
    """March along the image edge between two clicked endpoints, following it where it bends.

    A **marching** trace, not a search perpendicular to the chord, and the difference is not
    cosmetic.  With a real lens the image of a straight world edge departs from the chord between its
    endpoints by the distortion itself — measured on this pipeline's synthetic scene, up to **160 px**
    near the frame corners (``k1 = -0.12`` at ``f = 2112``).  A fixed perpendicular window therefore
    cannot find the edge at all along most of its length, and widening the window until it can means
    catching neighbouring structure instead.

    So the trace advances in small steps, re-estimating the local direction from the points it has
    already found, and only ever searches a few pixels to either side.  It stops when it runs out of
    response, leaves the frame, or reaches the far end.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float64) if frame.ndim == 3 \
        else frame.astype(np.float64)
    h, w = gray.shape
    span = float(np.linalg.norm(np.asarray(p1, dtype=float) - np.asarray(p0, dtype=float)))
    if span < 8.0:
        return None
    chord = (np.asarray(p1, dtype=float) - np.asarray(p0, dtype=float)) / span

    pts: list[np.ndarray] = []
    cur = np.asarray(p0, dtype=float).copy()
    off0 = _perp_offset(gray, cur, np.array([-chord[1], chord[0]]), radius=float(radius),
                        min_response=min_response)
    if off0 is not None:
        cur = cur + off0 * np.array([-chord[1], chord[0]])
    pts.append(cur.copy())
    direction = chord.copy()

    travelled = 0.0
    while travelled < span * 0.90:
        perp = np.array([-direction[1], direction[0]])
        nxt = cur + direction * float(step)
        if not (2 <= nxt[0] < w - 2 and 2 <= nxt[1] < h - 2):
            break
        off = _perp_offset(gray, nxt, perp, radius=float(radius), min_response=min_response)
        if off is None:
            break
        nxt = nxt + off * perp
        seg = nxt - cur
        n = float(np.linalg.norm(seg))
        if n < 1e-6:
            break
        # smooth the direction so one noisy sample cannot steer the march off the edge
        direction = 0.6 * direction + 0.4 * (seg / n)
        direction = direction / float(np.linalg.norm(direction))
        cur = nxt
        travelled += n
        pts.append(cur.copy())

    if len(pts) < 10:
        return None
    traced = np.asarray(pts, dtype=float)
    # Plausibility guards.  A march that has wandered onto a different edge still produces a smooth
    # point set, so geometry has to reject it: the path must be about as long as the chord (a line
    # that hits a tangent and runs off grows without bound) and must not depart from the chord by
    # more than a distortion could (measured on this scene the true curve departs by up to 160 px on
    # a ~1700 px chord, so 30% is a generous ceiling).
    chord_len = span
    path_len = float(np.sum(np.linalg.norm(np.diff(traced, axis=0), axis=1)))
    if not (0.6 * chord_len <= path_len <= 1.4 * chord_len):
        return None
    a, b = np.asarray(p0, dtype=float), np.asarray(p1, dtype=float)
    ab = b - a
    t = np.clip(((traced - a) @ ab) / max(float(ab @ ab), 1e-12), 0.0, 1.0)
    dev = float(np.max(np.linalg.norm(traced - (a + t[:, None] * ab), axis=1)))
    if dev > 0.30 * chord_len:
        return None
    return traced




def lens_from_checkerboard_dir(directory: str | Path, pattern: tuple[int, int],
                               square: float) -> L.LensModel:
    d = Path(directory)
    files = sorted([p for p in d.iterdir()
                    if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")])
    if not files:
        raise ValueError(f"no images under {d}")
    imgs = []
    for p in files:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            imgs.append(img)
    return L.calibrate_from_checkerboard(imgs, pattern, square_size=square)


# ============================================================================
# Frame + window
# ============================================================================
def load_frame(path: Path, frame_index: int | None = None) -> np.ndarray:
    path = Path(path)
    if path.suffix.lower() in (".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"):
        idx = 0 if frame_index is None else int(frame_index)
        for _i, frame in video_mod.iter_frames(path, start=idx, stop=idx + 1):
            return frame
        raise RuntimeError(f"could not read frame {idx} of {path}")
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"could not read image {path}")
    return img


def run_gui(frame: np.ndarray, session: PoolCalibrationSession, out_path: Path) -> int:
    window = ("pool calibrate - click order shown in the banner "
              "(z undo, k skip optional, g grid, w water, s save, q quit)")
    state = {"grid": False, "saved": False}
    scale = 1.0
    h, w = frame.shape[:2]
    max_w, max_h = 1500, 850
    if w > max_w or h > max_h:
        scale = min(max_w / w, max_h / h)

    def render() -> np.ndarray:
        vis = draw_overlay(frame, session, lens=session.lens, show_grid=state["grid"])
        return cv2.resize(vis, None, fx=scale, fy=scale) if scale != 1.0 else vis

    def on_mouse(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            session.click(x / scale, y / scale)
            cv2.imshow(window, render())

    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window, on_mouse)
    cv2.imshow(window, render())
    while True:
        key = cv2.waitKey(20) & 0xFF
        if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
            break
        if key == ord("q"):
            break
        if key == ord("z"):
            session.undo()
        elif key == ord("k"):
            session.skip_optional()
        elif key == ord("g"):
            state["grid"] = not state["grid"]
        elif key == ord("w"):
            order = ["rim_pair", "wall_contact", "known_freeboard"]
            session.water_method = order[(order.index(session.water_method) + 1) % len(order)]
            print(f"water method -> {session.water_method}")
        elif key == ord("s"):
            if not session.done:
                print(f"cannot save yet: {session.prompt}")
                continue
            doc = session.build()
            C.save(doc, out_path)
            state["saved"] = True
            print(f"saved {out_path}")
            for name, gate in doc["gates"].items():
                print(f"  {name}: {'PASS' if gate['pass'] else 'FAIL'}  {gate['detail']}")
            break
        else:
            continue
        cv2.imshow(window, render())
    cv2.destroyAllWindows()
    return 0 if state["saved"] else 1


# ============================================================================
# CLI
# ============================================================================
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Pool-surface translation calibration (Phase 2).")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--frame", type=Path, help="still image")
    src.add_argument("--clip", type=Path, help="video file (use --frame-index)")
    ap.add_argument("--frame-index", type=int, default=0)
    ap.add_argument("--setup-id", required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--marks", default=DEFAULT_MARKS_FT,
                    help=f"comma-separated ft values to click, in x order (default {DEFAULT_MARKS_FT})")
    ap.add_argument("--width-ft", type=float, default=DEFAULT_WIDTH_FT)
    ap.add_argument("--freeboard-in", type=float, default=DEFAULT_FREEBOARD_IN)
    ap.add_argument("--marks-on-edge", choices=("near", "far"), default=str(POOL_MARKS_ON_EDGE))
    ap.add_argument("--water", choices=("rim_pair", "wall_contact", "known_freeboard"),
                    default=str(POOL_WATER_METHOD))
    ap.add_argument("--wall-axis", choices=("x", "y"), default=str(POOL_WALL_AXIS))
    ap.add_argument("--wall-value", type=float, default=float(POOL_WALL_VALUE))
    ap.add_argument("--transverse-lines", type=int, default=2)
    ap.add_argument("--along-lines", type=int, default=3)
    ap.add_argument("--no-edge", action="store_true")
    ap.add_argument("--no-rim", action="store_true")
    ap.add_argument("--lens-source", choices=("prior", "lines", "checkerboard"), default="prior")
    ap.add_argument("--focal-px", type=float, default=None)
    ap.add_argument("--focal-ratio", type=float, default=float(POOL_FOCAL_RATIO))
    ap.add_argument("--checkerboard", type=Path, default=None)
    ap.add_argument("--checkerboard-pattern", default="9x6")
    ap.add_argument("--checkerboard-square", type=float, default=1.0)
    ap.add_argument("--clicks", type=Path, default=None,
                    help="headless: a JSON file in calib.build()'s clicks shape (no window)")
    ap.add_argument("--lens-json", type=Path, default=None,
                    help="headless: a saved LensModel JSON")
    ap.add_argument("--model-sha", default=None)
    ap.add_argument("--notes", default="")
    ap.add_argument("--print-gates", action="store_true")
    ap.add_argument("--seed", default=None,
                    help="headless shorthand: 'x,y;x,y;...' consumed in prompt order with --marks")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    path = args.frame or args.clip
    if not path.is_file():
        print(f"no such file: {path}", file=sys.stderr)
        return 2
    frame = load_frame(path, args.frame_index if args.clip else None)
    h, w = frame.shape[:2]
    out_path = args.out or C.setup_path(args.setup_id)

    lens = None
    if args.lens_json is not None:
        lens = L.LensModel.from_dict(json.loads(Path(args.lens_json).read_text(encoding="utf-8")))
    elif args.lens_source == "checkerboard":
        if args.checkerboard is None:
            print("--lens-source checkerboard needs --checkerboard DIR", file=sys.stderr)
            return 2
        pat = tuple(int(v) for v in str(args.checkerboard_pattern).lower().split("x"))
        lens = lens_from_checkerboard_dir(args.checkerboard, pat, args.checkerboard_square)
    elif args.lens_source == "prior":
        lens = L.placeholder_intrinsics((w, h), focal_px=args.focal_px,
                                        focal_ratio=args.focal_ratio)

    model_sha = args.model_sha
    if model_sha is None and Path(MODEL_PATH).is_file():
        try:
            model_sha = video_mod.sha256(MODEL_PATH)
        except OSError:
            model_sha = ""

    marks_ft = [float(v) for v in str(args.marks).split(",") if v.strip()]
    requests = build_requests(marks_ft, n_transverse_lines=args.transverse_lines,
                             n_along_lines=args.along_lines, want_edge=not args.no_edge,
                             want_rim=not args.no_rim)
    session = PoolCalibrationSession(
        frame_shape=(h, w), requests=requests, setup_id=args.setup_id, lens=lens,
        width_ft=args.width_ft, freeboard_in=args.freeboard_in,
        wall={"axis": args.wall_axis, "value": args.wall_value},
        marks_on_edge=args.marks_on_edge, water_method=args.water, model_sha=model_sha,
        notes=args.notes)

    # --- headless paths ------------------------------------------------------
    if args.clicks is not None or args.seed is not None:
        clicks: dict[str, Any] = {}
        if args.clicks is not None:
            clicks = json.loads(Path(args.clicks).read_text(encoding="utf-8"))
        else:
            for pair in str(args.seed).split(";"):
                pair = pair.strip()
                if not pair:
                    continue
                x, y = (float(v) for v in pair.split(","))
                session.click(x, y)
            if not session.done:
                print(f"seed did not complete the sequence — {session.prompt}", file=sys.stderr)
                return 2
            clicks = session.clicks

        if args.lens_source == "lines":
            base = lens or L.placeholder_intrinsics(
                (w, h), focal_px=args.focal_px, focal_ratio=args.focal_ratio)
            session.clicks = clicks
            # The frame is always available on the CLI path, so the edges can be traced and the fit
            # is identifiable.  (Passing no frame would return the prior with the reason, because a
            # two-point click is a straight chord and cannot constrain distortion.)
            lens = lens_from_lines(session, base, frame=frame)
            print(f"lens from clicked edges: source={lens.source} dist={lens.dist[:2]}")
            for note in lens.notes:
                print(f"  {note}")

        clicks = dict(clicks)
        clicks.setdefault("wall", {"axis": args.wall_axis, "value": args.wall_value})
        clicks.setdefault("water", {"method": args.water, "freeboard_in": args.freeboard_in,
                                    "source": "operator"})
        doc = C.build(args.setup_id, clicks=clicks, lens=lens, width_ft=args.width_ft,
                      marks_on_edge=args.marks_on_edge, model_sha=model_sha, notes=args.notes,
                      image_size=(w, h))
        C.save(doc, out_path)
        print(f"frame      : {path.name} {w}x{h}")
        print(f"route      : {doc['route']}")
        print(f"lens       : source={doc['lens']['source']} roundtrip="
              f"{doc['lens']['roundtrip_px']:.3g} px")
        print(f"water      : {doc['water']['method']} z={doc['water']['z_ft']:.4f} ft "
              f"({doc['water']['offset_in']:+.1f} in)")
        print(f"pose       : camera height {doc['pose']['camera_height_ft']:.2f} ft, "
              f"orthonormality {doc['pose']['orthonormality']:.2g}")
        print(f"saved      : {out_path}")
        if args.print_gates:
            for name, gate in doc["gates"].items():
                print(f"  {name}: {'PASS' if gate['pass'] else 'FAIL'}"
                      f"{' (warn)' if gate.get('warn') else ''}  {gate['detail']}")
            print(f"  overall: {'PASS' if C.gates_pass(doc['gates']) else 'FAIL'}")
        return 0

    if lens is None:
        lens = L.placeholder_intrinsics((w, h), focal_px=args.focal_px,
                                        focal_ratio=args.focal_ratio)
        session.lens = lens
    return run_gui(frame, session, out_path)


if __name__ == "__main__":
    raise SystemExit(main())
