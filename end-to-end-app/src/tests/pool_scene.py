"""Synthetic ground truth for the pool-translation pipeline — a shared test fixture.

Not a test module (pytest will not collect it): it is the *scene generator* the phase tests
build on.  Every phase of ``pool-translation-plan.txt`` is verified against this scene's
known numbers rather than against the pipeline's own output agreeing with itself, because a
pipeline that is self-consistently wrong is the failure mode that matters here.

The scene is deliberately a *hard* one for the geometry it is testing:

* a **fixed, oblique side camera** — the corpus's real configuration (`PLAN.md` D2) — which
  puts the across-pool vanishing point close to the principal point and the along-pool one
  far away, i.e. the ill-conditioned end of both;
* **real radial distortion** (``k1 = -0.12``), so the lens phase has something to remove and
  a phase that forgets to undistort fails loudly;
* the marks on the **deck** (``Z = 0``) and the water **below** them (19.5 in of Dog Diving
  freeboard), so Phase 5's "extend the plane downward" is exercised rather than skipped.

World frame, identical to :mod:`dog_diving.pool.homography`: origin at the 0 ft dock edge mark,
+X along the jump direction, +Y across the pool, +Z up, marked plane at Z = 0.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import cv2
import numpy as np

# Dog Diving big-air freeboard: the dock top sits ~19.5 in above the water.
DEFAULT_FREEBOARD_IN = 19.5
# A competition pool's stated length, and the 14-20 ft width band the plan's Phase 3 names.
DEFAULT_LENGTH_FT = 31.0
DEFAULT_WIDTH_FT = 20.0
DEFAULT_IMAGE = (1920, 1080)


def checkerboard_texture(pattern: tuple[int, int], *, px_per_square: int = 40,
                         border_px: int = 80, border_value: int = 128) -> np.ndarray:
    """A crisp chessboard image: ``pattern`` is the count of **inner corners**."""
    cols, rows = int(pattern[0]) + 1, int(pattern[1]) + 1      # squares
    board = np.full((rows * px_per_square, cols * px_per_square), 0, np.uint8)
    for r in range(rows):
        for c in range(cols):
            if (r + c) % 2 == 0:
                board[r * px_per_square:(r + 1) * px_per_square,
                      c * px_per_square:(c + 1) * px_per_square] = 255
    return cv2.copyMakeBorder(board, border_px, border_px, border_px, border_px,
                              cv2.BORDER_CONSTANT, value=border_value)


def render_checkerboard(
    image_size: tuple[int, int],
    K: np.ndarray,
    dist: np.ndarray,
    rvec: Sequence[float],
    tvec: Sequence[float],
    pattern: tuple[int, int],
    *,
    square: float = 1.0,
    px_per_square: int = 40,
    border_px: int = 80,
) -> np.ndarray:
    """Render a planar target **exactly**: per-pixel inverse mapping, distortion included.

    The naive way to build a synthetic calibration image is to project the board's corner lattice and
    warp a straight-edged board through the resulting homography.  That is wrong, and wrong in a way
    that silently destroys the calibration it is meant to validate: with real lens distortion, the
    projected corners are *not* the image of a lattice under any homography, so the rendered board
    and the corner positions disagree by the distortion itself.  Feeding that to
    ``cv2.calibrateCamera`` recovers a focal length that is 24% wrong — and looks like the
    calibrator's fault.

    So this renders by inverse mapping: for every output pixel, undo the distortion, cast the ray,
    intersect the target's plane, and sample the texture there.  That is exact (to the undistortion
    inversion's 1e-13 px) and hole-free.

    The board's world coordinates carry an arbitrary constant offset relative to the object points a
    caller passes to ``calibrateCamera``; a constant translation is absorbed into the extrinsics and
    leaves ``K`` and the distortion untouched, so only the *spacing* has to agree — which
    ``square = px_per_square * world_units_per_pixel`` guarantees.
    """
    W, H = int(image_size[0]), int(image_size[1])
    K = np.asarray(K, dtype=float)
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=float).reshape(3, 1))
    t = np.asarray(tvec, dtype=float).reshape(3)
    board = checkerboard_texture(pattern, px_per_square=px_per_square, border_px=border_px)
    bh, bw = board.shape[:2]
    scale = float(square) / float(px_per_square)          # world units per board pixel

    uu, vv = np.meshgrid(np.arange(W), np.arange(H))
    pts = np.stack([uu.ravel(), vv.ravel()], axis=1).astype(np.float64)
    und = cv2.undistortPoints(pts.reshape(-1, 1, 2), K, np.asarray(dist, dtype=float),
                             None, K, criteria=(cv2.TERM_CRITERIA_EPS
                                                + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-14))
    und = und.reshape(-1, 2)
    rays = (np.linalg.inv(K) @ np.hstack([und, np.ones((und.shape[0], 1))]).T).T
    d_world = (R.T @ rays.T).T
    C = -R.T @ t
    dz = d_world[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        s = np.where(np.abs(dz) > 1e-12, -C[2] / dz, np.nan)
    Xw = C[0] + s * d_world[:, 0]
    Yw = C[1] + s * d_world[:, 1]
    bx = np.round(Xw / scale).astype(np.int64)
    by = np.round(Yw / scale).astype(np.int64)
    ok = np.isfinite(bx) & np.isfinite(by) & (bx >= 0) & (bx < bw) & (by >= 0) & (by < bh)
    out = np.full((H, W), 30, np.uint8)
    flat = out.reshape(-1)
    flat[ok] = board[by[ok], bx[ok]]
    return out.reshape(H, W)


def look_at(eye: Sequence[float], target: Sequence[float],
            up: Sequence[float] = (0.0, 0.0, 1.0)) -> tuple[np.ndarray, np.ndarray]:
    """Rotation/translation for ``world -> camera`` in **OpenCV** convention.

    OpenCV's camera frame looks along **+Z** (image-right = +X, image-down = +Y), so
    ``z_cam`` is built from ``target - eye`` — *not* ``eye - target``, which is the OpenGL
    convention and puts the whole scene behind the camera (every projection then returns
    ``nan`` and the scene looks broken rather than mis-oriented).  Returns ``(R, t)`` with
    ``p_cam = R p_world + t``.
    """
    eye = np.asarray(eye, dtype=float)
    target = np.asarray(target, dtype=float)
    up = np.asarray(up, dtype=float)
    z = target - eye                       # optical axis, pointing into the scene
    z = z / np.linalg.norm(z)
    x = np.cross(z, up)                    # image-right
    nx = np.linalg.norm(x)
    if nx < 1e-9:
        raise ValueError("the up vector is parallel to the view direction")
    x = x / nx
    y = np.cross(z, x)                     # image-down
    R = np.vstack([x, y, z])
    return R, -R @ eye


@dataclass
class Scene:
    """A synthetic pool with known camera, known geometry and known water level."""

    K: np.ndarray
    R: np.ndarray
    t: np.ndarray
    dist: np.ndarray
    image_size: tuple[int, int]
    water_z_ft: float
    width_ft: float = DEFAULT_WIDTH_FT
    length_ft: float = DEFAULT_LENGTH_FT
    mark_row_y_ft: float = 0.0
    marks_ft: tuple[float, ...] = (0.0, 1.0, 15.0, 16.0, 30.0, 31.0)
    rim_z_ft: float = 0.33                       # a ~4 in coping lip above the marked plane
    name: str = "deck_marks"
    extra: dict[str, Any] = field(default_factory=dict)

    # --- geometry ---------------------------------------------------------
    @property
    def freeboard_in(self) -> float:
        """How far the water sits below the marked plane, in inches (positive = below)."""
        return -float(self.water_z_ft) * 12.0

    @property
    def eye(self) -> np.ndarray:
        return -self.R.T @ self.t

    def camera_point(self, X: float, Y: float, Z: float) -> np.ndarray:
        return self.R @ np.array([float(X), float(Y), float(Z)]) + self.t

    def project(self, X: float, Y: float, Z: float) -> np.ndarray:
        """World ``(X, Y, Z)`` -> **raw (distorted)** pixel, or ``(nan, nan)`` behind the camera."""
        p = self.camera_point(X, Y, Z)
        if p[2] <= 1e-9:
            return np.array([np.nan, np.nan])
        uv = cv2.projectPoints(p.reshape(1, 1, 3), np.zeros(3), np.zeros(3),
                               self.K, self.dist)[0].reshape(2)
        return uv

    def project_many(self, pts: Sequence[Sequence[float]]) -> np.ndarray:
        return np.asarray([self.project(*p) for p in pts], dtype=float)

    def project_clean(self, X: float, Y: float, Z: float) -> np.ndarray:
        """Undistorted-pixel projection (no distortion applied) — the pipeline's target space."""
        p = self.camera_point(X, Y, Z)
        if p[2] <= 1e-9:
            return np.array([np.nan, np.nan])
        uv = self.K @ p
        return uv[:2] / uv[2]

    def project_many_clean(self, pts: Sequence[Sequence[float]]) -> np.ndarray:
        return np.asarray([self.project_clean(*p) for p in pts], dtype=float)

    # --- the pieces each phase consumes -----------------------------------
    def marks(self) -> list[dict[str, Any]]:
        """The distance/class marks, as the calibration document carries them."""
        out = []
        for ft in self.marks_ft:
            uv = self.project(float(ft), self.mark_row_y_ft, 0.0)
            out.append({"ft": float(ft), "inch": 0.0, "label": f"{int(ft)}'00\"",
                        "x_px": float(uv[0]), "y_px": float(uv[1])})
        return out

    def true_world_xy(self) -> np.ndarray:
        return np.array([[float(f), self.mark_row_y_ft] for f in self.marks_ft], dtype=float)

    def vanishing_point(self, direction: Sequence[float]) -> np.ndarray:
        """Homogeneous vanishing point of a world direction: ``K R d``."""
        d = np.asarray(direction, dtype=float)
        return self.K @ (self.R @ d)

    def vp_x(self) -> np.ndarray:
        return self.vanishing_point((1.0, 0.0, 0.0))

    def vp_y(self) -> np.ndarray:
        return self.vanishing_point((0.0, 1.0, 0.0))

    def edge_line(self, y_ft: float) -> np.ndarray:
        """The image line of the world line ``Y = y_ft`` (the far pool edge), homogeneous."""
        p1 = self.project_clean(0.0, y_ft, 0.0)
        p2 = self.project_clean(self.length_ft, y_ft, 0.0)
        if not (np.all(np.isfinite(p1)) and np.all(np.isfinite(p2))):
            raise RuntimeError("the edge line has a point behind the camera")
        x1, y1 = float(p1[0]), float(p1[1])
        x2, y2 = float(p2[0]), float(p2[1])
        a, b, c = np.cross([x1, y1, 1.0], [x2, y2, 1.0])
        n = float(np.hypot(a, b))
        return np.array([a / n, b / n, c / n])

    # --- the water entry the pipeline must reproduce ----------------------

    def entry_pixel(self, X: float, Y: float, Z: float | None = None) -> np.ndarray:
        """The raw pixel of a tail-set at height ``Z`` (default: at the water surface)."""
        z = self.water_z_ft if Z is None else float(Z)
        return self.project(X, Y, z)

    def rim_pair(self, X: float = 4.0, Y: float | None = None) -> tuple[np.ndarray, np.ndarray]:
        """A rim-top / water-contact click pair on one vertical wall, as the operator gives it.

        Both clicks are on the same world vertical through ``(X, Y)``: the top at the coping's
        height and the contact at the water level.  This is exactly the pair the plan's Phase 5
        asks for, and it is what makes the water level observable without seeing the water
        plane broadside.
        """
        yy = self.mark_row_y_ft if Y is None else float(Y)
        top = self.project(X, yy, self.rim_z_ft)
        con = self.project(X, yy, self.water_z_ft)
        return top, con

    # --- reports ----------------------------------------------------------
    def summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "image_size": list(self.image_size),
            "eye_ft": [float(v) for v in self.eye],
            "water_z_ft": float(self.water_z_ft),
            "freeboard_in": float(-self.water_z_ft * 12.0),
            "width_ft": float(self.width_ft),
            "length_ft": float(self.length_ft),
            "vp_x": [float(v) for v in self.vp_x()],
            "vp_y": [float(v) for v in self.vp_y()],
            "marks_ft": [float(v) for v in self.marks_ft],
        }


# ============================================================================
# The scenes used by the tests
# ============================================================================
def deck_marks_scene(
    *,
    eye: Sequence[float] = (-7.0, -17.0, 5.6),
    target: Sequence[float] = (15.0, 7.0, -1.0),
    image_size: tuple[int, int] = DEFAULT_IMAGE,
    focal_ratio: float = 1.1,
    dist: Sequence[float] = (-0.12, 0.02, 0.0, 0.0, 0.0),
    water_z_ft: float = -DEFAULT_FREEBOARD_IN / 12.0,
    width_ft: float = DEFAULT_WIDTH_FT,
    length_ft: float = DEFAULT_LENGTH_FT,
) -> Scene:
    """The primary scene: marks on the deck (Z = 0), water 19.5 in below.

    The camera is oblique — to the near side and slightly behind the dock edge — which is what
    a real fixed setup looks like and what keeps the along-pool vanishing point finite enough
    to be worth having.
    """
    w, h = int(image_size[0]), int(image_size[1])
    f = float(focal_ratio) * max(w, h)
    K = np.array([[f, 0.0, (w - 1) / 2.0], [0.0, f, (h - 1) / 2.0], [0.0, 0.0, 1.0]],
                 dtype=float)
    R, t = look_at(eye, target)
    d = np.asarray(list(dist) + [0.0] * (5 - len(dist)), dtype=float)[:5]
    return Scene(K=K, R=R, t=t, dist=d, image_size=(w, h), water_z_ft=float(water_z_ft),
                 width_ft=float(width_ft), length_ft=float(length_ft), name="deck_marks")


def waterline_marks_scene(**kw: Any) -> Scene:
    """The other venue arrangement: marks painted on the wall **at the waterline**.

    Then the marked plane *is* the water plane (offset ~0) and Phase 5 has nothing to extend.
    Kept as a separate scene because the plan's Phase 5 assumes the downward-extrapolation
    case, and a pipeline that only works in that case would be silently wrong here.
    """
    kw.setdefault("water_z_ft", 0.0)
    scene = deck_marks_scene(**kw)
    scene.name = "waterline_marks"
    return scene


def _inverse_world_map(image_size, K, dist, R, t, *, roi) -> tuple[np.ndarray, np.ndarray]:
    """Inverse-map a sub-image of raw pixels onto the plane ``Z = 0``.

    Returns ``(flat_pixel_indices, world_xy)``: for each pixel in ``roi`` ``(y0, y0h, x0, x0h)`` the
    corresponding world ``(X, Y)`` on the marked plane.  Undo the distortion, cast the ray, intersect
    the plane.  This is the exact inverse of :meth:`Scene.project`, which is what lets a fixture
    render a *geometrically exact* thick line instead of a rasterised approximation.
    """
    W, H = int(image_size[0]), int(image_size[1])
    y0, y1, x0, x1 = roi
    y0, y1 = max(0, int(y0)), min(H, int(y1))
    x0, x1 = max(0, int(x0)), min(W, int(x1))
    if y1 <= y0 or x1 <= x0:
        return np.zeros((0,), dtype=np.int64), np.zeros((0, 2))
    yy, xx = np.mgrid[y0:y1, x0:x1]
    px = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float64)
    und = cv2.undistortPoints(px.reshape(-1, 1, 2), np.asarray(K, dtype=float),
                             np.asarray(dist, dtype=float), None, np.asarray(K, dtype=float),
                             criteria=(cv2.TERM_CRITERIA_EPS
                                       + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-14)).reshape(-1, 2)
    rays = (np.linalg.inv(np.asarray(K, dtype=float))
            @ np.hstack([und, np.ones((und.shape[0], 1))]).T).T
    dw = (np.asarray(R, dtype=float).T @ rays.T).T
    C = -np.asarray(R, dtype=float).T @ np.asarray(t, dtype=float).reshape(3)
    with np.errstate(divide="ignore", invalid="ignore"):
        s = np.where(np.abs(dw[:, 2]) > 1e-12, -C[2] / dw[:, 2], np.nan)
    Xw = C[0] + s * dw[:, 0]
    Yw = C[1] + s * dw[:, 1]
    ok = np.isfinite(Xw) & np.isfinite(Yw)
    idx = (yy.ravel() * W + xx.ravel())[ok]
    return idx, np.stack([Xw[ok], Yw[ok]], axis=1)


def render_world_lines_exact(
    scene: Scene,
    world_lines: Sequence[tuple[Sequence[float], Sequence[float]]],
    *,
    half_width_ft: float = 0.035,
    colour: int = 245,
    background: int = 15,
    feather_ft: float = 0.012,
) -> np.ndarray:
    """A clean frame of exact, geometrically rendered world lines — no rasterisation bias.

    ``cv2.polylines`` draws a line of integer thickness whose anti-aliasing is **not symmetric**, so
    an edge traced from such a frame is offset from the true centreline by a sub-pixel amount that
    *varies along the line*.  That variation is exactly the shape signal the distortion fit reads, so
    on a polyline-rendered frame the fit measures the rasteriser rather than the lens (measured:
    traced mean offset 1.2 px, and a resulting ``k1`` of -0.074 against a true -0.12).

    This renders each line as the inverse-mapped set of pixels whose world position lies within
    ``half_width_ft`` of the segment, with a soft edge over ``feather_ft``.  The profile is then
    symmetric about the true centreline by construction, so a traced edge carries the distortion and
    nothing else.
    """
    W, H = int(scene.image_size[0]), int(scene.image_size[1])
    out = np.full((H, W), float(background), dtype=np.float64)

    # region of interest: the projected endpoints, dilated generously
    pts = []
    for p0, p1 in world_lines:
        for p in (p0, p1):
            q = scene.project(*p)
            if np.all(np.isfinite(q)):
                pts.append(q)
    if not pts:
        return cv2.cvtColor(out.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    arr = np.asarray(pts, dtype=float)
    pad = 24
    roi = (int(arr[:, 1].min()) - pad, int(arr[:, 1].max()) + pad + 1,
           int(arr[:, 0].min()) - pad, int(arr[:, 0].max()) + pad + 1)
    idx, world = _inverse_world_map((W, H), scene.K, scene.dist, scene.R, scene.t, roi=roi)
    if idx.size == 0:
        return cv2.cvtColor(out.astype(np.uint8), cv2.COLOR_GRAY2BGR)

    field = np.full(idx.shape[0], background, dtype=np.float64)
    for p0, p1 in world_lines:
        a = np.asarray(p0, dtype=float)[:2]
        b = np.asarray(p1, dtype=float)[:2]
        ab = b - a
        denom = float(ab @ ab) or 1e-12
        t = np.clip(((world - a) @ ab) / denom, 0.0, 1.0)
        d = np.linalg.norm(world - (a + t[:, None] * ab), axis=1)
        soft = np.clip((half_width_ft + feather_ft - d) / max(2.0 * feather_ft, 1e-9), 0.0, 1.0)
        field = np.maximum(field, background + soft * (float(colour) - background))
    flat = out.reshape(-1)
    flat[idx] = field
    return cv2.cvtColor(out.reshape(H, W).astype(np.uint8), cv2.COLOR_GRAY2BGR)


def render_world_lines(scene: Scene, world_lines: Sequence[tuple[Sequence[float], Sequence[float]]],
                       *, thickness: int = 3, colour: int = 245,
                       background: int = 15) -> np.ndarray:
    """A clean frame containing only the given world lines, projected **through the lens**.

    For exercising Phase 1's line-fitting route without the clutter of a full scene: the lines are
    drawn as projected polylines, so they carry the real curvature the distortion fit has to remove,
    and there is nothing else in the frame for an edge search to lock onto by mistake.  A full
    ``frame_from_scene`` render does not serve here — measured, an edge trace on it landed 4-11 px
    off the true curve because the frame also contains the mark band, ticks and labels.
    """
    img = np.full((scene.image_size[1], scene.image_size[0]), int(background), np.uint8)
    n = 80
    for p0, p1 in world_lines:
        pts = []
        for t in np.linspace(0.0, 1.0, n):
            q = scene.project(*(np.asarray(p0, dtype=float)
                                + t * (np.asarray(p1, dtype=float) - np.asarray(p0, dtype=float))))
            if np.all(np.isfinite(q)):
                pts.append((int(round(q[0])), int(round(q[1]))))
        if len(pts) >= 2:
            cv2.polylines(img, [np.asarray(pts, np.int32).reshape(-1, 1, 2)], False,
                          int(colour), int(thickness), cv2.LINE_AA)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def clean_edge_lines_for(scene: Scene) -> list[tuple[tuple[float, float, float],
                                                    tuple[float, float, float]]]:
    """The pool's own straight edges, as the world segment list :func:`render_world_lines` takes."""
    w, L = scene.width_ft, scene.length_ft
    return [
        ((0.0, 0.0, 0.0), (L, 0.0, 0.0)),
        ((0.0, w, 0.0), (L, w, 0.0)),
        ((0.0, w / 2.0, 0.0), (L, w / 2.0, 0.0)),
        ((0.0, 0.0, 0.0), (0.0, w, 0.0)),
        ((L, 0.0, 0.0), (L, w, 0.0)),
        ((L / 2.0, 0.0, 0.0), (L / 2.0, w, 0.0)),
    ]


def raw_clicks(sc: "Scene", *, route: str = "auto", water: dict | None = None,
               wall: dict | None = None) -> dict:
    """A complete click set in :func:`dog_diving.pool.calib.build`'s shape, from **raw** pixels.

    Raw on purpose: the document build is responsible for undistorting every click, so a fixture that
    pre-undistorted them would leave that path untested — and raw is how an operator's input arrives.
    Lives here, next to the scene, so the tool test and the document test drive identical clicks.
    """
    def seg(p, q):
        return [*sc.project(*p)[:2], *sc.project(*q)[:2]]

    clicks: dict = {
        "marks": [{"ft": m["ft"], "x_px": m["x_px"], "y_px": m["y_px"]} for m in sc.marks()],
        "vp_transverse": {"lines": [seg((0.0, 0.0, 0.0), (0.0, sc.width_ft, 0.0)),
                                    seg((31.0, 0.0, 0.0), (31.0, sc.width_ft, 0.0))]},
        "vp_along": {"lines": [seg((0.0, 0.0, 0.0), (31.0, 0.0, 0.0)),
                               seg((0.0, sc.width_ft, 0.0), (31.0, sc.width_ft, 0.0)),
                               seg((0.0, sc.width_ft / 2, 0.0), (31.0, sc.width_ft / 2, 0.0))]},
        "edge_line": seg((0.0, sc.width_ft, 0.0), (31.0, sc.width_ft, 0.0)),
        "rim": {"top": list(sc.project(4.0, 0.0, sc.rim_z_ft)),
                "contact": list(sc.project(4.0, 0.0, sc.water_z_ft))},
        "wall": wall or {"axis": "y", "value": 0.0},
    }
    if route == "A":
        clicks["correspondences"] = [
            {"X_ft": x, "Y_ft": y, "x_px": float(sc.project(x, y, 0.0)[0]),
             "y_px": float(sc.project(x, y, 0.0)[1])}
            for x, y in [(0.0, 0.0), (31.0, 0.0), (0.0, 16.0), (31.0, 16.0), (15.0, 8.0)]]
    if water is not None:
        clicks["water"] = water
    return clicks


def frame_from_scene(scene: Scene, *, background: int = 40) -> np.ndarray:
    """Render a crude but *geometrically correct* frame: the pool and marks, drawn distorted.

    Not a photo — a projection check.  Everything is drawn by projecting the true world
    geometry, so a phase that gets the geometry wrong cannot accidentally be right, and the
    line-extraction and edge-snapping code in the pipeline has something real to find.
    """
    img = np.full((scene.image_size[1], scene.image_size[0], 3), background, dtype=np.uint8)

    def draw_world_polyline(pts: Sequence[Sequence[float]], colour, thickness: int = 1) -> None:
        uv = []
        for p in pts:
            q = scene.project(*p)
            if np.all(np.isfinite(q)):
                uv.append((int(round(q[0])), int(round(q[1]))))
        if len(uv) >= 2:
            cv2.polylines(img, [np.asarray(uv, dtype=np.int32).reshape(-1, 1, 2)], False,
                          colour, thickness, cv2.LINE_AA)

    # the pool: water surface quad, then the near/far edges and the end wall
    n = 40
    xs = np.linspace(0.0, scene.length_ft, n)
    draw_world_polyline([(x, 0.0, scene.water_z_ft) for x in xs], (90, 70, 40), 2)
    draw_world_polyline([(x, scene.width_ft, scene.water_z_ft) for x in xs], (90, 70, 40), 2)
    for y in np.linspace(0.0, scene.width_ft, 9):
        draw_world_polyline([(0.0, y, scene.water_z_ft), (scene.length_ft, y, scene.water_z_ft)],
                            (70, 55, 32), 1)
    for x in np.linspace(0.0, scene.length_ft, 12):
        draw_world_polyline([(x, 0.0, scene.water_z_ft), (x, scene.width_ft, scene.water_z_ft)],
                            (70, 55, 32), 1)

    # the marked row: a light band on the deck plus a dark tick at each mark
    draw_world_polyline([(x, scene.mark_row_y_ft, 0.0) for x in xs], (140, 140, 130), 3)
    for m in scene.marks():
        p = scene.project_clean(float(m["ft"]), scene.mark_row_y_ft, 0.0)
        cv2.circle(img, (int(round(p[0])), int(round(p[1]))), 6, (10, 10, 10), -1)
        cv2.putText(img, f"{int(m['ft'])}", (int(round(p[0])) + 8, int(round(p[1])) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (240, 240, 240), 2)
    return img
