"""The pool plane — the distance mapping, from one aligned grid.

Replaces the four bars (PLAN §15, 2026-10-19).  The operator aligns **one plane** on the water
surface: a quad whose corners are the four corners of a metric rectangle on the pool, ``X = 0``
along the dock lip and ``+X`` away down the pool.  Corner drags give any homography, side drags
translate an edge, and the editor is allowed to extend **beyond the frame** where the pool is not
visible (the pool's near edge and far end usually are not).

What it produces
----------------
Four world -> image correspondences and therefore a homography ``H`` on the **water surface**.  For
any pixel on the water, ``H^-1`` gives ``(X, Y)`` in feet, and ``X`` *is* the distance from the dock
edge — which is the whole measurement.  No heights are inferred: the waterline, the foreground
waterline and the dock freeboard are **not** reported by this route (operator decision, 2026-10-19).
The Dog Diving freeboard (19.5 in) is recorded in the document as a specification for the record and
is *not* used by the mapping, because the aligned plane already *is* the water surface.

Why this is allowed to be simple
--------------------------------
The 3-D route's expensive parts (lens, vanishing points, pose decomposition, water-plane offset)
exist to reconstruct a plane that the operator cannot see or cannot identify.  Here the operator
identifies the plane directly, so the mapping is a homography and nothing else.  ``calib.build`` is
still used to *assemble* the document (route ``A_correspondences``), so ``pose_from_document`` /
``water_from_document`` / ``translate_track`` keep working — but the pose it decomposes is a
by-product, and with an assumed lens its orthonormality gate can fail while the mapping stays exact.
The app therefore shows the plane gates (P1-P4) as the delivery gate and the W1-W7 block as
informational, each labelled with what it actually grades.

Grid
----
Minor lines every **1 ft** (unlabelled), major every **5 ft**, labelled along the pool length only
(feet from the dock edge).  Across-pool lines get no numbers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import cv2
import numpy as np

from ..config import POOL_FREEBOARD_IN, POOL_LENGTH_FT, POOL_WIDTH_FT
from . import calib as _calib
from . import homography as _homography
from .lens import LensModel, placeholder_intrinsics

SCHEMA_BLOCK = "dog_diving_pool_plane/v1"

#: ``length_ft`` is the operator's +x ANCHOR (where the quad's far edge sits), not the pool's
#: length.  The metric axis itself is fixed and always runs 0 -> ``METRIC_LENGTH_FT`` feet, so a
#: distance can be reported past the far edge of the quad by extrapolation through the same
#: homography (2026-10-20 operator order; config.POOL_LENGTH_FT).
METRIC_LENGTH_FT = float(POOL_LENGTH_FT)

DEFAULT_LENGTH_FT = 25.0
DEFAULT_WIDTH_FT = float(POOL_WIDTH_FT)
MINOR_FT = 1.0
MAJOR_FT = 5.0

#: The canvas extends this fraction of the frame beyond each edge, so a corner can be dragged to
#: where the pool continues out of shot (the near waterline and the far end are normally off-frame).
PAD_FRAC = 0.35

#: Handle grab radius, in **display** px (the padded canvas), and side/corner minimum sizes.
GRAB_PX = 16.0
MIN_SIDE_PX = 12.0
MIN_AREA_PX = 400.0

#: Corner tags, in the order the quad is drawn.  ``Y`` is across the pool: 0 = back (far) edge.
CORNER_TAGS: tuple[str, ...] = ("00", "10", "11", "01")
SIDE_TAGS: tuple[str, ...] = ("back", "right", "front", "left")
_SIDE_OF = {"back": ("00", "10"), "right": ("10", "11"), "front": ("11", "01"),
            "left": ("01", "00")}


def corner_world(tag: str, length_ft: float, width_ft: float) -> tuple[float, float]:
    """The metric coordinate of a corner: X from the dock edge, Y from the back edge."""
    xi, yi = int(tag[0]), int(tag[1])
    return (float(xi) * float(length_ft), float(yi) * float(width_ft))


# ============================================================================
# The plane
# ============================================================================
@dataclass
class PlaneSession:
    """A quad on the water surface, edited by its corners and sides.

    Corners are stored in **image** coordinates and are *not* clamped to the frame: a corner may sit
    outside it, which is the point of the padded canvas.  ``pad`` is where the padded display canvas
    puts the image origin, so ``from_display``/``to_display`` are the only conversions the view needs.
    """

    frame_shape: tuple[int, int]                                   # (height, width)
    corners: dict[str, tuple[float, float] | None] = field(default_factory=dict)
    length_ft: float = DEFAULT_LENGTH_FT
    width_ft: float = DEFAULT_WIDTH_FT
    #: X (ft) of the quad's dock-side edge.  ``0`` means that edge IS the dock lip.  A non-zero
    #: value is the second "+x anchor" (operator order 2026-10-21): the near edge is placed on a
    #: visible pool marker — "type 6 ft as the lower anchor, then put the two dock-side corners on
    #: the far and near 6 ft markers" — instead of on a pool corner that may be outside the frame.
    #: The metric axis still runs to 0 ft, and reported X adds this offset, so distances stay
    #: measured from the dock.
    dock_offset_ft: float = 0.0
    pad: tuple[float, float] = (0.0, 0.0)
    dragging: str | None = None                                    # corner tag or side name
    #: The handle the arrow pad / arrow keys move.  Unlike ``dragging`` it survives a release, so a
    #: corner can be nudged one image pixel at a time without the mouse being on it.
    selected: str | None = None
    notes: list[str] = field(default_factory=list)
    #: Provenance of an **automatic** placement (the waterline seed, PLAN §17): which upstream
    #: detector and byte hash, the calm window, every edge's fit quality, the inherited px/ft ratio
    #: and the S-gates.  ``None`` means the plane was placed by hand.  The first drag after a seed
    #: flips ``placement`` from ``auto`` to ``auto+edited``, so a reviewer can tell a seeded plane
    #: from one the operator moved.  Written into the document as ``waterline_seed``.
    seed: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        for tag in CORNER_TAGS:
            self.corners.setdefault(tag, None)
        if self.pad == (0.0, 0.0):
            self.pad = default_pad(self.frame_shape)

    # --- state ------------------------------------------------------------
    @property
    def placed(self) -> bool:
        return all(self.corners.get(t) is not None for t in CORNER_TAGS)

    @property
    def prompt(self) -> str:
        if not self.placed:
            return "plane: click the four corners (back-left, dock-side, far, near) or press Fit"
        return ("plane: drag corners to fit the pool, drag a side to move that edge, then Build "
                "document")

    def to_display(self, x: float, y: float) -> tuple[float, float]:
        return (float(x) + self.pad[0], float(y) + self.pad[1])

    def from_display(self, x: float, y: float) -> tuple[float, float]:
        return (float(x) - self.pad[0], float(y) - self.pad[1])

    # --- construction / editing -------------------------------------------
    def fit_to_frame(self) -> str:
        """A pool-shaped default quad: narrower and higher at the back, wider at the near edge."""
        h, w = self.frame_shape
        # X = 0 on the left (the dock lip), +X to the right, and the far end (X = L) SHORTER in the
        # image than the dock end — a pool seen down its length foreshortens, and a default without
        # that fails the P4 orientation gate (measured: a symmetric trapezoid gives ratio 1.00).
        self.corners = {
            "00": (0.06 * w, 0.30 * h),          # X=0, Y=0  (dock lip, back edge)
            "10": (0.64 * w, 0.42 * h),          # X=L, Y=0  (far end, back edge)
            "11": (0.64 * w, 0.80 * h),          # X=L, Y=W  (far end, near edge)
            "01": (0.06 * w, 0.95 * h),          # X=0, Y=W  (dock lip, near edge)
        }
        self.dragging = None
        return self.prompt

    def clear(self) -> None:
        for tag in CORNER_TAGS:
            self.corners[tag] = None
        self.dragging = None
        self.selected = None

    def select(self, handle: str | None) -> str | None:
        """Make ``handle`` the target of the arrow pad / arrow keys (``None`` clears it)."""
        self.selected = handle if handle in CORNER_TAGS or handle in SIDE_TAGS else None
        return self.selected

    def nudge(self, dx: float, dy: float, handle: str | None = None) -> bool:
        """Move the selected handle by ``(dx, dy)`` **image** px.  Sub-pixel steps are kept as floats.

        The arrow pad sends ±1 image px (Shift ±10, Ctrl ±0.1); a side moves both of its corners so
        the edge stays straight, exactly like a drag.  Nothing is clamped to the frame — a corner
        may sit outside it, which is what the padded canvas is for.
        """
        which = self.selected if handle is None else handle
        if which is None:
            return False
        if which in CORNER_TAGS:
            p = self.corners.get(which)
            if p is None:
                return False
            self.corners[which] = (float(p[0]) + float(dx), float(p[1]) + float(dy))
            self._touch()
            return True
        if which in SIDE_TAGS:
            a, b = _SIDE_OF[which]
            pa, pb = self.corners.get(a), self.corners.get(b)
            if pa is None or pb is None:
                return False
            self.corners[a] = (pa[0] + float(dx), pa[1] + float(dy))
            self.corners[b] = (pb[0] + float(dx), pb[1] + float(dy))
            self._touch()
            return True
        return False

    def _touch(self) -> None:
        """Mark an automatic placement as edited (no-op for a hand-placed plane)."""
        if self.seed and self.seed.get("placement") == "auto":
            self.seed["placement"] = "auto+edited"

    def apply_seed(self, corners: dict[str, tuple[float, float]],
                   block: dict[str, Any] | None = None) -> str:
        """Write an automatic placement (four image-px corners) into the session.

        The caller owns the decision *whether* to apply; this only guarantees the session is left in
        a coherent state: all four corners set, no half-finished drag, the anchor edge selected so the
        arrow pad is pointing at the handle the operator will most likely want, and the provenance
        block recorded.
        """
        missing = [t for t in CORNER_TAGS if t not in corners]
        if missing:
            raise ValueError(f"seed is missing corners {missing}")
        for tag in CORNER_TAGS:
            self.set_corner(tag, float(corners[tag][0]), float(corners[tag][1]))
        self.dragging = None
        self.selected = "right"          # the X = anchor edge, in tag terms
        self.seed = dict(block or {})
        return self.prompt

    def flip_x(self) -> bool:
        """Reverse the X **numbering** in place: not one pixel of the plane moves.

        Operator order (2026-10-21): *"Flip Plane (left-right) should keep the plane orientation
        completely and only change the number scale used, this is the best solution to avoid
        disrupting how the user placed the plane."*  So the four corners keep their exact image
        positions and only the tags move: ``00``<->``10`` and ``01``<->``11``.  The quad on screen is
        untouched (identical points, identical handles), ``Y`` is untouched (each pair swapped shares
        its row), and the X axis now runs the other way — which is the whole point: a plane whose X=0
        end sits on the dock's far side measures every distance towards the dock, and no gate can see
        it because P4's foreshortening is symmetric under the mirror.

        This replaces the older behaviour of mirroring the corners about the quad's centre: that moved
        a hand-placed quad away from where the operator had put it, and a flip is a decision about
        *which end is zero*, not about the shape.
        """
        if not self.placed:
            return False
        for a, b in (("00", "10"), ("01", "11")):
            self.corners[a], self.corners[b] = self.corners[b], self.corners[a]
        self.dragging = None
        self._touch()
        return True

    def ensure_far_edge_short(self) -> bool:
        """Make the far edge the SHORT one in the image, swapping the Y ends if it is not.

        A pool seen down its length foreshortens, so the far edge (``Y = 0``) must subtend fewer
        pixels than the near edge.  Operator order (2026-10-21): *"snap the plane to the detected
        waterlines with the far edge short and the near edge long."*  If the mapping came out the
        other way round, the two Y ends are exchanged — corners keep their image positions and only
        their tags move, exactly like :meth:`flip_x`, so nothing the operator sees is disturbed.
        Returns True when the ends were swapped.
        """
        if not self.placed:
            return False
        p00, p10 = self.corners["00"], self.corners["10"]
        p01, p11 = self.corners["01"], self.corners["11"]
        # The two long waterlines: the Y = 0 edge (the far one, up the image) against the Y = W edge
        # (the near one).  Both span the pool's length, so the comparison is like for like.
        far_px = float(np.hypot(p10[0] - p00[0], p10[1] - p00[1]))
        near_px = float(np.hypot(p11[0] - p01[0], p11[1] - p01[1]))
        if near_px <= 0.0 or far_px <= near_px * (1.0 + 1e-6):
            return False
        for a, b in (("00", "01"), ("10", "11")):
            self.corners[a], self.corners[b] = self.corners[b], self.corners[a]
        self._touch()
        return True

    def set_corner(self, tag: str, x: float, y: float) -> None:
        if tag not in CORNER_TAGS:
            raise ValueError(f"unknown corner {tag!r}: expected one of {CORNER_TAGS}")
        self.corners[tag] = (float(x), float(y))

    # --- placement aids: parallelogram, near anchor, waterline snap --------
    def make_parallelogram(self) -> str:
        """Rewrite the quad as the parallelogram with the same centre and edge directions.

        Operator order (2026-10-21): *"have the plane start as a parallelogram"* — the starting shape
        for a hand placement.  It is a placement aid, NOT a constraint: the two X-direction edges are
        averaged into one vector, the two Y-direction edges into another, and the four corners are
        rebuilt from the centroid.  Dragging afterwards is free, so the perspective skew a homography
        needs is still available (a plane forced to stay affine would lose the foreshortening that
        makes the far end of a pool shorter in the image).
        """
        if not self.placed:
            return self.prompt
        p00, p10 = self._pt("00"), self._pt("10")
        p11, p01 = self._pt("11"), self._pt("01")
        cx = 0.25 * (p00[0] + p10[0] + p11[0] + p01[0])
        cy = 0.25 * (p00[1] + p10[1] + p11[1] + p01[1])
        ux = 0.5 * ((p10[0] - p00[0]) + (p11[0] - p01[0]))
        uy = 0.5 * ((p10[1] - p00[1]) + (p11[1] - p01[1]))
        vx = 0.5 * ((p01[0] - p00[0]) + (p11[0] - p10[0]))
        vy = 0.5 * ((p01[1] - p00[1]) + (p11[1] - p10[1]))
        self.corners["00"] = (cx - 0.5 * (ux + vx), cy - 0.5 * (uy + vy))
        self.corners["10"] = (cx + 0.5 * (ux - vx), cy + 0.5 * (uy - vy))
        self.corners["11"] = (cx + 0.5 * (ux + vx), cy + 0.5 * (uy + vy))
        self.corners["01"] = (cx - 0.5 * (ux - vx), cy - 0.5 * (uy - vy))
        self.dragging = None
        self._touch()
        return self.prompt



    def snap_side_to_line(self, side: str, point: Sequence[float],
                          direction: Sequence[float]) -> bool:
        """Slide one side of the quad onto the given image-space line.

        ``side`` is a :data:`SIDE_TAGS` name — ``back`` (the far edge, Y = 0), ``front`` (the near
        edge, Y = W), ``left`` (X = 0) or ``right`` (X = L).  Both corners of that side are projected
        perpendicularly onto the line, so the edge keeps its own length and direction as far as the
        line allows — the operator is saying *"the water is here"*, not *"reshoot the quad"*.  The
        opposite side is untouched.  Returns True when a corner actually moved.
        """
        pair = _SIDE_OF.get(str(side))
        if pair is None:
            return False
        a, b = (self.corners.get(t) for t in pair)
        if a is None or b is None:
            return False
        p = np.asarray(point, float).reshape(2)
        d = np.asarray(direction, float).reshape(2)
        n = float(np.hypot(*d))
        if not np.isfinite(n) or n < 1e-9:
            return False
        d = d / n
        moved = False
        for tag in pair:
            q = np.asarray(self.corners[tag], float).reshape(2)
            foot = p + d * float(np.dot(q - p, d))
            if float(np.hypot(*(foot - q))) > 1e-6:
                self.corners[tag] = (float(foot[0]), float(foot[1]))
                moved = True
        if moved:
            self._touch()
        return moved

    def snap_far_edge_to_line(self, point: Sequence[float],
                              direction: Sequence[float]) -> bool:
        """Slide the far edge (corners ``10``/``11``) onto the given image-space line.

        Kept as the name the seed and the first snap control were written against; it is
        :meth:`snap_side_to_line` with ``side="back"``.
        """
        return self.snap_side_to_line("back", point, direction)

    def _pt(self, tag: str) -> tuple[float, float]:
        p = self.corners.get(tag)
        if p is None:
            raise ValueError(f"corner {tag!r} is not placed")
        return (float(p[0]), float(p[1]))

    def click(self, x: float, y: float, *, zoom: float = 1.0) -> bool:
        """Grab a handle (corner preferred over side), or place the next missing corner.

        ``zoom`` is the pane's zoom, so the grab radius stays ~``GRAB_PX`` *screen* px instead of
        shrinking to nothing when the operator zooms in to do detail work.
        """
        if not self.placed:
            for tag in CORNER_TAGS:
                if self.corners[tag] is None:
                    # x/y arrive in DISPLAY px (the padded canvas); corners live in image px
                    self.corners[tag] = self.from_display(x, y)
                    self.selected = tag
                    return True
            return False
        hit = self.handle_at(x, y, tol=GRAB_PX / max(float(zoom), 1e-6))
        if hit is None:
            return False
        self.dragging = hit
        self.selected = hit
        return True

    def handle_at(self, x: float, y: float, tol: float = GRAB_PX) -> str | None:
        """The nearest handle within ``tol`` display px — a corner, else a side's midpoint."""
        best: tuple[float, str] | None = None
        for tag in CORNER_TAGS:
            p = self.corners.get(tag)
            if p is None:
                continue
            d = float(np.hypot(self.pad[0] + p[0] - x, self.pad[1] + p[1] - y))
            if d <= tol and (best is None or d < best[0]):
                best = (d, tag)
        if best is not None:
            return best[1]
        for name in SIDE_TAGS:
            m = self.side_midpoint(name)
            if m is None:
                continue
            d = float(np.hypot(self.pad[0] + m[0] - x, self.pad[1] + m[1] - y))
            if d <= tol and (best is None or d < best[0]):
                best = (d, name)
        return None if best is None else best[1]

    def side_midpoint(self, name: str) -> tuple[float, float] | None:
        a, b = (_SIDE_OF[name][0], _SIDE_OF[name][1])
        pa, pb = self.corners.get(a), self.corners.get(b)
        if pa is None or pb is None:
            return None
        return (0.5 * (pa[0] + pb[0]), 0.5 * (pa[1] + pb[1]))

    def drag_to(self, x: float, y: float) -> bool:
        """Move the grabbed handle to a display point.  A corner moves freely (giving perspective);
        a side translates both of its corners, so the edge stays straight and parallel to itself."""
        if self.dragging is None:
            return False
        ix, iy = self.from_display(x, y)
        if self.dragging in CORNER_TAGS:
            self.corners[self.dragging] = (ix, iy)
            self._touch()
            return True
        name = self.dragging
        a, b = _SIDE_OF[name]
        pa, pb = self.corners.get(a), self.corners.get(b)
        if pa is None or pb is None:
            return False
        mx, my = 0.5 * (pa[0] + pb[0]), 0.5 * (pa[1] + pb[1])
        dx, dy = ix - mx, iy - my
        self.corners[a] = (pa[0] + dx, pa[1] + dy)
        self.corners[b] = (pb[0] + dx, pb[1] + dy)
        self._touch()
        return True

    def release(self) -> str | None:
        which, self.dragging = self.dragging, None
        return which

    # --- geometry ---------------------------------------------------------
    def correspondences(self) -> list[dict[str, float]]:
        """The four corners as ``(X, Y)`` feet against image px, in **absolute** X.

        X is measured from the dock lip, so the quad's dock-side edge sits at ``dock_offset_ft``
        and its far edge at ``dock_offset_ft + length_ft``.  Building the world side of the
        homography this way (rather than mapping to 0..length and adding the offset downstream)
        means every consumer — the measurement, the grid, the P-gates — reads the same absolute
        axis, and a non-zero anchor cannot be quietly dropped by one of them.
        """
        out = []
        off = float(self.dock_offset_ft)
        for tag in CORNER_TAGS:
            p = self.corners.get(tag)
            if p is None:
                continue
            X, Y = corner_world(tag, self.length_ft, self.width_ft)
            out.append({"X_ft": off + X, "Y_ft": Y,
                        "x_px": float(p[0]), "y_px": float(p[1])})
        return out

    def homography(self) -> np.ndarray:
        if not self.placed:
            raise ValueError(f"the plane is not placed — {self.prompt}")
        W = np.asarray([[c["X_ft"], c["Y_ft"]] for c in self.correspondences()], dtype=float)
        I = np.asarray([[c["x_px"], c["y_px"]] for c in self.correspondences()], dtype=float)
        H, _info = _homography.homography_from_correspondences(W, I, robust=False)
        return H

    def world_at(self, x: float, y: float) -> tuple[float, float]:
        """Image px -> ``(X, Y)`` feet on the water surface."""
        H = self.homography()
        return tuple(float(v) for v in _homography.apply_homography_inv(H, np.array([[x, y]]))[0])

    def pixel_of(self, X: float, Y: float) -> tuple[float, float]:
        H = self.homography()
        return tuple(float(v) for v in _homography.apply_homography(H, np.array([[X, Y]]))[0])

    def extent(self) -> dict[str, float]:
        """The metric bounding box: **always** 0 -> ``METRIC_LENGTH_FT`` along +X.

        ``length_ft`` is only the anchor that positions the quad's far edge; it must never
        shorten the axis a distance is reported on (2026-10-20 operator order).
        """
        return {"x_min": 0.0, "x_max": float(METRIC_LENGTH_FT),
                "y_min": 0.0, "y_max": float(self.width_ft)}

    # --- output -----------------------------------------------------------
    def build(self, setup_id: str, *, lens: LensModel | None = None,
              image_size: tuple[int, int] | None = None, notes: str = "") -> dict[str, Any]:
        """Assemble a pool-calibration document whose plane *is* the water surface."""
        if not self.placed:
            raise ValueError(f"the plane is not placed — {self.prompt}")
        h, w = int(self.frame_shape[0]), int(self.frame_shape[1])
        lens = lens or placeholder_intrinsics(image_size or (w, h))
        marks = [{"ft": float(c["X_ft"]), "inch": 0.0, "label": f"{c['X_ft']:g}'",
                  "x_px": c["x_px"], "y_px": c["y_px"]}
                 for c in self.correspondences() if float(c["Y_ft"]) == 0.0]
        doc = _calib.build(
            setup_id,
            clicks={"marks": marks, "correspondences": self.correspondences(),
                    # The aligned plane IS the water, so the water offset is zero BY
                    # CONSTRUCTION.  Passing freeboard 0 through the known-freeboard path keeps
                    # calib.build's own machinery (and W5) honest instead of special-casing it.
                    "water": {"method": "known_freeboard", "freeboard_in": 0.0,
                              "source": "plane_is_the_water_surface"}},
            lens=lens, route="A_correspondences", width_ft=float(self.width_ft),
            marks_on_edge="far", extent=self.extent(), image_size=(w, h),
            notes=(notes or "water-surface plane (app_gui)"),
        )
        doc["plane"] = self.as_dict()
        doc["plane_gates"] = validate_plane(doc, self)
        doc["mapping"] = mapping_summary(self)
        return doc

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"schema": SCHEMA_BLOCK,
                "frame_shape": [int(self.frame_shape[0]), int(self.frame_shape[1])],
                "corners": {k: (None if v is None else [float(v[0]), float(v[1])])
                            for k, v in self.corners.items()},
                "length_ft": float(self.length_ft), "width_ft": float(self.width_ft),
                "dock_offset_ft": float(self.dock_offset_ft),
                # The axis, as distinct from the anchor: X is reported 0 -> 35 ft whatever the
                # operator typed as the far edge, and the mapping extrapolates past the quad.
                "metric_length_ft": float(METRIC_LENGTH_FT),
                "length_note": ("length_ft is the +x ANCHOR — where the operator's far edge sits, "
                                "used to line the grid up against visible pool detail. "
                                "dock_offset_ft is the near-edge anchor: the X of the quad's "
                                "DOCK-SIDE edge, 0 when that edge is the dock lip. Quad world X "
                                "runs dock_offset_ft -> dock_offset_ft + length_ft; the metric "
                                "axis always runs 0 -> metric_length_ft."),
                "pad": [float(self.pad[0]), float(self.pad[1])],
                "kind": "water_surface", "mapping": "homography",
                "freeboard_in_spec": float(POOL_FREEBOARD_IN),
                "freeboard_note": ("Dog Diving' 19.5 in recorded as a specification for the record. "
                                   "NOT used by the mapping: the aligned plane IS the water "
                                   "surface, so the water offset is zero by construction")}
        if self.seed:
            # Present only for an automatically placed plane (PLAN §17), so a hand-placed plane's
            # document stays byte-identical to what this app wrote before the feature existed.
            out["waterline_seed"] = dict(self.seed)
        return out

    @classmethod
    def from_doc(cls, doc: dict[str, Any]) -> "PlaneSession":
        block = doc.get("plane") or {}
        if not block:
            raise ValueError("this document has no 'plane' block — it was not made by the plane "
                             "editor")
        if block.get("schema") != SCHEMA_BLOCK:
            raise ValueError(f"unsupported plane schema {block.get('schema')!r}")
        fh = tuple(int(v) for v in block.get("frame_shape") or (0, 0))
        s = cls(frame_shape=(fh[0], fh[1]), length_ft=float(block.get("length_ft", DEFAULT_LENGTH_FT)),
                width_ft=float(block.get("width_ft", DEFAULT_WIDTH_FT)),
                # Documents written before the near anchor existed have no key: 0 keeps their
                # quad exactly where it was, so an old floor a run used still measures the same.
                dock_offset_ft=float(block.get("dock_offset_ft", 0.0) or 0.0))
        for tag, p in (block.get("corners") or {}).items():
            if tag in CORNER_TAGS:
                s.corners[tag] = None if p is None else (float(p[0]), float(p[1]))
        seed = block.get("waterline_seed")
        s.seed = dict(seed) if seed else None
        return s


def default_pad(frame_shape: tuple[int, int], frac: float = PAD_FRAC) -> tuple[float, float]:
    """How far the display canvas extends past the image, in px (both sides, both axes)."""
    h, w = int(frame_shape[0]), int(frame_shape[1])
    m = float(frac) * max(w, h)
    return (m, m)


def padded_canvas(frame: np.ndarray, pad: tuple[float, float]) -> np.ndarray:
    """The frame inset into a larger dark canvas, so off-frame handles have somewhere to live."""
    mx, my = int(round(pad[0])), int(round(pad[1]))
    return cv2.copyMakeBorder(frame, my, my, mx, mx, cv2.BORDER_CONSTANT, value=(28, 28, 28))


# ============================================================================
# The mapping (what the tool exists to produce)
# ============================================================================
def mapping_summary(plane: PlaneSession, *, xs: Sequence[float] | None = None) -> dict[str, Any]:
    """Where the metric lines land, and how finely the water resolves distance.

    Probes the whole metric axis (0 -> ``METRIC_LENGTH_FT``), not just the operator's quad, because
    that is the range a distance may be reported over.
    """
    H = plane.homography()
    y_mid = plane.width_ft / 2.0
    probes = list(xs if xs is not None else np.arange(0.0, METRIC_LENGTH_FT + 1e-9, 5.0))
    rows = []
    for X in probes:
        p0 = _homography.apply_homography(H, np.array([[float(X), y_mid]]))[0]
        p1 = _homography.apply_homography(H, np.array([[float(X) + 1.0, y_mid]]))[0]
        back = _homography.apply_homography_inv(H, np.array([p0]))[0]
        rows.append({"X_ft": float(X), "pixel": [float(p0[0]), float(p0[1])],
                     "px_per_ft_along": float(np.linalg.norm(p1 - p0)),
                     "roundtrip_error_ft": float(np.linalg.norm(back - [float(X), y_mid]))})
    return {"y_probe_ft": float(y_mid), "probe": rows,
            "px_per_ft_along_at_0": rows[0]["px_per_ft_along"] if rows else None,
            "px_per_ft_along_at_far": rows[-1]["px_per_ft_along"] if rows else None,
            "note": ("X is the distance from the dock edge by construction: the plane's X = 0 edge "
                     "is the dock lip. No height is used — the plane IS the water surface.")}


# ============================================================================
# P1-P4 — what the plane itself can fail on
# ============================================================================
def validate_plane(doc: dict[str, Any], plane: PlaneSession | None = None) -> dict[str, dict[str, Any]]:
    """P1-P4.  These are the gates that matter for this route; W1-W7 grade the reconstructed pose,
    which the mapping does not use (see the module docstring)."""
    g: dict[str, dict[str, Any]] = {}
    block = (plane.as_dict() if plane is not None else (doc.get("plane") or {}))
    corners = block.get("corners") or {}
    pts = [corners.get(t) for t in CORNER_TAGS]
    L = float(block.get("length_ft") or DEFAULT_LENGTH_FT)
    Wd = float(block.get("width_ft") or DEFAULT_WIDTH_FT)

    # --- P1 the quad is a quad -------------------------------------------------
    problems: list[str] = [f"corner {t} missing" for t, p in zip(CORNER_TAGS, pts) if p is None]
    area = 0.0
    if not problems:
        arr = np.asarray(pts, dtype=float)
        area = float(abs(cv2.contourArea(arr.astype(np.float32))))
        sides = [float(np.linalg.norm(arr[i] - arr[(i + 1) % 4])) for i in range(4)]
        if min(sides) < MIN_SIDE_PX:
            problems.append(f"shortest side {min(sides):.0f} px")
        if area < MIN_AREA_PX:
            problems.append(f"area {area:.0f} px")
        if _self_intersects(arr):
            problems.append("the quad crosses itself")
    g["P1"] = {"pass": not problems, "warn": False,
               "detail": f"quad area {area:.0f} px², {L:g} x {Wd:g} ft"
                         + ("" if not problems else "; " + "; ".join(problems))}

    # --- P2 the homography is usable and orientation-preserving ----------------
    try:
        H = np.asarray(doc["homography"], dtype=float).reshape(3, 3)
        cond = _homography._homography_condition(H)
        det = float(np.linalg.det(H[:2, :2]))
        ok = np.isfinite(cond["cond"]) and cond["cond"] < 1e12 and abs(det) > 1e-12
        g["P2"] = {"pass": bool(ok), "warn": False,
                   "detail": f"px/ft along {cond['px_per_ft_along']:.1f}, across "
                             f"{cond['px_per_ft_across']:.1f}, cond {cond['cond']:.3g}"
                             + (" — the transverse axis is coarsely resolved" if
                                cond["anisotropy"] > 20 else "")}
    except Exception as exc:                                     # noqa: BLE001
        g["P2"] = {"pass": False, "warn": False, "detail": f"degenerate homography: {exc}"}

    # --- P3 the mapping round-trips -------------------------------------------
    try:
        worst = 0.0
        off = float(block.get("dock_offset_ft") or 0.0)
        for tag in CORNER_TAGS:
            X, Y = corner_world(tag, L, Wd)
            X += off
            merged = (plane if plane is not None
                      else PlaneSession.from_doc(doc))
            px = merged.pixel_of(X, Y)
            back = merged.world_at(*px)
            worst = max(worst, float(np.hypot(back[0] - X, back[1] - Y)))
        g["P3"] = {"pass": worst < 0.01, "warn": False,
                   "detail": f"corner round-trip worst {worst:.2e} ft (block > 0.01 ft)"}
    except Exception as exc:                                     # noqa: BLE001
        g["P3"] = {"pass": False, "warn": False, "detail": f"round-trip not computable: {exc}"}

    # --- P4 the plane is not reversed end-for-end -----------------------------
    # The physical test, not a geometric one: a foot must subtend FEWER pixels at the far end than
    # at the dock end.  A plane aligned with X running towards the camera inverts that, and an
    # earlier version of this gate (distance from the dock corner) could not see it at all —
    # measured: a reversed plane passed with ratio 1.0x, i.e. the gate could not fail.
    try:
        b = PlaneSession.from_doc(doc) if plane is None else plane
        y_mid = Wd / 2.0
        p0, p1 = np.asarray(b.pixel_of(0.0, y_mid)), np.asarray(b.pixel_of(1.0, y_mid))
        q0, q1 = np.asarray(b.pixel_of(L, y_mid)), np.asarray(b.pixel_of(L + 1.0, y_mid))
        near = float(np.linalg.norm(p1 - p0))
        far = float(np.linalg.norm(q1 - q0))
        ratio = far / max(near, 1e-9)
        ok = ratio < 1.0
        g["P4"] = {"pass": bool(ok), "warn": bool(0.9 <= ratio < 1.0),
                   "detail": f"1 ft subtends {near:.1f} px at the dock edge and {far:.1f} px at "
                             f"the far end (ratio {ratio:.2f}x) — a reversed plane scores > 1"}
    except Exception as exc:                                     # noqa: BLE001
        g["P4"] = {"pass": False, "warn": False, "detail": f"orientation not checkable: {exc}"}
    return g


def _self_intersects(pts: np.ndarray) -> bool:
    """Bow-tie test for a 4-point polygon (its contour area is then meaningless and H folds)."""
    def seg_hits(p1, p2, p3, p4) -> bool:
        def orient(a, b, c):
            return np.sign((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
        return (orient(p1, p2, p3) * orient(p1, p2, p4) < 0
                and orient(p3, p4, p1) * orient(p3, p4, p2) < 0)
    return seg_hits(pts[0], pts[1], pts[2], pts[3]) or seg_hits(pts[1], pts[2], pts[3], pts[0])


# ============================================================================
# Drawing — the Review-pane grid, on a padded canvas
# ============================================================================
def draw_plane(frame: np.ndarray, plane: PlaneSession, *, grid: bool = True,
               minor: float = MINOR_FT, major: float = MAJOR_FT) -> np.ndarray:
    """Draw the padded canvas, the plane and its grid.

    Grid style follows the Review pane: minor lines every foot, major every 5 ft with a label along
    the pool length only.  Lines are drawn in world space through the homography, so they are
    straight in the image and converge correctly.
    """
    vis = padded_canvas(frame, plane.pad)
    ox, oy = float(plane.pad[0]), float(plane.pad[1])
    put = lambda p: (int(round(p[0] + ox)), int(round(p[1] + oy)))       # noqa: E731

    if not plane.placed:
        for tag in CORNER_TAGS:
            p = plane.corners.get(tag)
            if p is not None:
                cv2.circle(vis, put(p), 6, (0, 220, 255), -1, cv2.LINE_AA)
        _banner(vis, plane.prompt)
        return vis

    H = plane.homography()
    L, Wd = float(plane.length_ft), float(plane.width_ft)

    def line(X0, Y0, X1, Y1, colour, thick):
        a = _homography.apply_homography(H, np.array([[X0, Y0]]))[0]
        b = _homography.apply_homography(H, np.array([[X1, Y1]]))[0]
        if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
            return
        cv2.line(vis, put(a), put(b), colour, thick, cv2.LINE_AA)

    # fill first so the grid reads on dark water
    quad = np.asarray([put(plane.corners[t]) for t in CORNER_TAGS], dtype=np.int32)
    overlay = vis.copy()
    cv2.fillConvexPoly(overlay, quad, (60, 45, 30))
    vis = cv2.addWeighted(overlay, 0.35, vis, 0.65, 0)

    if grid:
        # The grid spans the METRIC axis (0 -> 35 ft), not the operator's quad: the far edge only
        # anchors where X = length_ft sits, and the lines past it are the same homography
        # extrapolated.  Thin lines deliberately — this render is inspected at 1:1 and at high
        # zoom, where a 2-3 px line hides the very pixels the operator is aligning to.
        n_minor = int(round(METRIC_LENGTH_FT / minor))
        for i in range(n_minor + 1):
            X = i * minor
            is_major = abs(X / major - round(X / major)) < 1e-9
            line(X, 0.0, X, Wd, (120, 210, 120) if is_major else (90, 110, 90), 1)
            if is_major:
                p = _homography.apply_homography(H, np.array([[X, 0.0]]))[0]
                if np.all(np.isfinite(p)):
                    cv2.putText(vis, f"{X:g}'", (int(p[0] + ox + 3), int(p[1] + oy - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 240, 150), 1, cv2.LINE_AA)
        n_across = int(round(Wd / minor))
        for j in range(n_across + 1):
            Y = j * minor
            is_major = abs(Y / major - round(Y / major)) < 1e-9
            line(0.0, Y, METRIC_LENGTH_FT, Y, (110, 170, 110) if is_major else (80, 105, 80), 1)

    # the datum edge (X = 0), the quad's anchored far edge, and the quad outline
    line(0.0, 0.0, 0.0, Wd, (60, 120, 255), 2)
    line(L, 0.0, L, Wd, (150, 150, 150), 1)
    line(0.0, 0.0, L, 0.0, (0, 200, 255), 1)
    line(0.0, Wd, L, Wd, (0, 200, 255), 1)

    # handles: corners (solid) and side midpoints (hollow).  Small on purpose — a handle is a
    # target, not an object, and a fat disc covers the pixels being aligned.
    for tag in CORNER_TAGS:
        p = put(plane.corners[tag])
        cv2.circle(vis, p, 4, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(vis, p, 4, (60, 120, 255), 1, cv2.LINE_AA)
    for name in SIDE_TAGS:
        m = plane.side_midpoint(name)
        if m is not None:
            cv2.circle(vis, put(m), 3, (200, 200, 200), 1, cv2.LINE_AA)
    if plane.selected in CORNER_TAGS or plane.selected in SIDE_TAGS:
        sp = (plane.corners.get(plane.selected) if plane.selected in CORNER_TAGS
              else plane.side_midpoint(plane.selected))
        if sp is not None:
            cv2.circle(vis, put(sp), 8, (255, 255, 255), 1, cv2.LINE_AA)
    if plane.dragging:
        p = (plane.corners.get(plane.dragging) if plane.dragging in CORNER_TAGS
             else plane.side_midpoint(plane.dragging))
        if p is not None:
            cv2.circle(vis, put(p), 10, (255, 255, 255), 1, cv2.LINE_AA)
    return vis


def draw_grid_on(frame: np.ndarray, homography: Any, *, length_ft: float, width_ft: float,
                 lens: Any | None = None, minor: float = MINOR_FT, major: float = MAJOR_FT,
                 labels: bool = True, fade_above_y: float | None = None,
                 fade_alpha: float = 0.35) -> np.ndarray:
    """Draw only the metric grid, on a **raw** frame, from a document's own homography.

    The Review pane needs exactly this and nothing else: the grid the operator aligned has to be the
    grid the reviewer sees.  Passing ``homography`` (``doc["homography"]``) rather than rebuilding
    one from ``pose_from_document`` is deliberate — the pose-derived matrix is a *different* map
    whenever the lens is assumed (measured 9.8 in apart on the synthetic scene), and drawing that
    one would put the review grid somewhere the calibration never claimed.

    ``fade_above_y`` splits every line at that image row and blends the part **above** it
    (``y < fade_above_y``) at ``fade_alpha``, leaving the part below at full strength.  Review passes
    the landing point's own row, so the grid behind the dog is see-through and the grid in front of it
    is not — the dog stays legible on its final frame without losing the lattice in front of it.
    """
    from .lens import distort_points

    H = np.asarray(homography, dtype=float).reshape(3, 3)
    out = frame.copy()          # never mutate the caller's frame: the overlay is one of several

    def px(X: float, Y: float) -> tuple[int, int] | None:
        p = _homography.apply_homography(H, np.array([[float(X), float(Y)]]))[0]
        if not np.all(np.isfinite(p)):
            return None
        if lens is not None:
            p = distort_points(np.asarray([p], dtype=float), lens)[0]
        return int(round(float(p[0]))), int(round(float(p[1])))

    y_split = None if fade_above_y is None else float(fade_above_y)
    layer = out.copy() if y_split is not None else None
    mask = np.zeros(out.shape[:2], np.uint8) if layer is not None else None

    def _seg(a, b, colour, thick: int, faded: bool) -> None:
        if faded and layer is not None:
            cv2.line(layer, a, b, colour, thick, cv2.LINE_AA)
            cv2.line(mask, a, b, 255, max(1, thick), cv2.LINE_AA)
        else:
            cv2.line(out, a, b, colour, thick, cv2.LINE_AA)

    def line(a: tuple[int, int] | None, b: tuple[int, int] | None,
             colour, thick: int) -> None:
        """One lattice line, split at the fade row if it crosses it."""
        if a is None or b is None:
            return
        if y_split is None or a[1] == b[1]:
            _seg(a, b, colour, thick, bool(y_split is not None and a[1] < y_split))
            return
        a_above, b_above = a[1] < y_split, b[1] < y_split
        if a_above == b_above:
            _seg(a, b, colour, thick, a_above)
            return
        t = (y_split - a[1]) / float(b[1] - a[1])
        cross = (int(round(a[0] + t * (b[0] - a[0]))), int(round(y_split)))
        if a_above:
            _seg(a, cross, colour, thick, True)
            _seg(cross, b, colour, thick, False)
        else:
            _seg(a, cross, colour, thick, False)
            _seg(cross, b, colour, thick, True)

    def label(pos: tuple[int, int] | None, text: str, colour) -> None:
        if pos is None:
            return
        faded = y_split is not None and pos[1] < y_split
        target = layer if (faded and layer is not None) else out
        cv2.putText(target, text, (pos[0] + 3, pos[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    colour, 1, cv2.LINE_AA)
        if faded and mask is not None:
            cv2.putText(mask, text, (pos[0] + 3, pos[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        255, 1, cv2.LINE_AA)

    for i in range(int(round(length_ft / minor)) + 1):
        X = i * minor
        is_major = abs(X / major - round(X / major)) < 1e-9
        line(px(X, 0.0), px(X, width_ft),
             (120, 210, 120) if is_major else (85, 115, 85), 2 if is_major else 1)
        if is_major and labels:
            label(px(X, 0.0), f"{X:g}'", (150, 240, 150))
    for j in range(int(round(width_ft / minor)) + 1):
        Y = j * minor
        is_major = abs(Y / major - round(Y / major)) < 1e-9
        line(px(0.0, Y), px(length_ft, Y),
             (110, 170, 110) if is_major else (75, 105, 75), 2 if is_major else 1)
    line(px(0.0, 0.0), px(0.0, width_ft), (60, 120, 255), 3)      # the dock-edge datum

    if layer is not None and mask is not None and mask.any():
        blended = cv2.addWeighted(layer, float(fade_alpha), out, 1.0 - float(fade_alpha), 0.0)
        out = np.where(mask[..., None] > 0, blended, out).astype(np.uint8)
    return out


def project(H: Any, X: float, Y: float) -> tuple[float, float]:
    """World ``(X, Y)`` feet -> image px through a homography.  ``nan`` when the map folds.

    A one-shot helper so an overlay can compute the matrix **once** and project ~50 grid points
    with it, instead of re-solving for every line.
    """
    p = _homography.apply_homography(np.asarray(H, dtype=float).reshape(3, 3),
                                     np.array([[float(X), float(Y)]]))[0]
    return (float(p[0]), float(p[1]))


def grid_spec(homography: Any, *, length_ft: float, width_ft: float, lens: Any | None = None,
              minor: float = MINOR_FT, major: float = MAJOR_FT) -> dict[str, Any]:
    """The metric lattice as **image-space segments**, for a caller that draws it itself.

    :func:`draw_grid_on` bakes the lattice into a frame; Review needs the same lattice as vector
    items so the operator's opacity slider and the zoom can reach it (operator order 2026-10-21:
    *"the grid does not adhere to the opacity rule"*).  Both come from here, so they cannot drift:
    same homography, same ``minor``/``major`` steps, same lens distortion, same datum line.

    Returns ``{"lines": [{"a": (x, y), "b": (x, y), "major": bool}], "labels": [{"at", "text"}],
    "datum": {"a", "b"}}``, all in ORIGINAL image px.  Lines whose projection folds (past the
    vanishing point) are dropped rather than drawn across the frame.
    """
    from .lens import distort_points

    H = np.asarray(homography, dtype=float).reshape(3, 3)

    def px(X: float, Y: float) -> tuple[float, float] | None:
        p = _homography.apply_homography(H, np.array([[float(X), float(Y)]]))[0]
        if not np.all(np.isfinite(p)):
            return None
        if lens is not None:
            p = distort_points(np.asarray([p], dtype=float), lens)[0]
        if not np.all(np.isfinite(p)):
            return None
        return float(p[0]), float(p[1])

    lines: list[dict[str, Any]] = []
    labels: list[dict[str, Any]] = []
    for i in range(int(round(length_ft / minor)) + 1):
        X = i * minor
        is_major = abs(X / major - round(X / major)) < 1e-9
        a, b = px(X, 0.0), px(X, width_ft)
        if a is not None and b is not None:
            lines.append({"a": a, "b": b, "major": is_major})
        if is_major:
            at = px(X, 0.0)
            if at is not None:
                labels.append({"at": at, "text": f"{X:g}'"})
    for j in range(int(round(width_ft / minor)) + 1):
        Y = j * minor
        is_major = abs(Y / major - round(Y / major)) < 1e-9
        a, b = px(0.0, Y), px(length_ft, Y)
        if a is not None and b is not None:
            lines.append({"a": a, "b": b, "major": is_major})
    out: dict[str, Any] = {"lines": lines, "labels": labels}
    d0, d1 = px(0.0, 0.0), px(0.0, width_ft)
    if d0 is not None and d1 is not None:
        out["datum"] = {"a": d0, "b": d1}
    return out


def _banner(vis: np.ndarray, text: str) -> None:
    cv2.rectangle(vis, (0, 0), (vis.shape[1], 30), (20, 20, 20), -1)
    cv2.putText(vis, text, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
