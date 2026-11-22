"""pair_homography — map the water plane from the loaded camera's image into the paired camera's.

The operator fits ONE plane, on the loaded clip.  For the paired camera nothing is stored: its plane
is *derived* every time it is drawn, from

    H_world->cam2 = H12 @ H_world->cam1

where ``H_world->cam1`` is the loaded clip's plane homography (feet -> px) and ``H12`` maps a point
of the WATER SURFACE in camera 1's image to the same point in camera 2's image.  ``H12`` is only
valid for points on that plane (the dog in the air, the dock, the trees do not map through it).

Where ``H12`` comes from is a swappable *source*: any callable
``(frame1, frame2, mask1) -> H12Result``.  The default, :func:`estimate_h12_from_images`, matches
image features between the two first frames, restricted (in camera 1) to the plane's own footprint
so the trees/sky — which sit at a different depth — cannot win the fit.  A point-cloud / Fusion-JSON
source can replace it later without touching the display code.

Pure: numpy + OpenCV, no Tk.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import cv2
import numpy as np

#: Fewer RANSAC inliers than this and the alignment is reported as failed rather than drawn.
MIN_INLIERS = 12
#: RANSAC reprojection threshold, image px.
RANSAC_PX = 4.0
#: Lowe ratio test for descriptor matches.
RATIO = 0.75
#: How far (px) the matching mask reaches beyond the plane's footprint in camera 1 — the pool
#: walls and coping sit at the waterline and are the most reliable features.
MASK_MARGIN_PX = 40


@dataclass
class H12Result:
    """Outcome of one alignment attempt.  ``H`` is ``None`` when it failed; ``message`` says why."""

    H: np.ndarray | None
    message: str
    inliers: int = 0
    matches: int = 0
    source: str = "image features"
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.H is not None


H12Source = Callable[[np.ndarray, np.ndarray, "np.ndarray | None"], H12Result]


def plane_mask(frame_shape: tuple[int, int], H_world_to_img: np.ndarray, *, length_ft: float,
               width_ft: float, margin_px: int = MASK_MARGIN_PX) -> np.ndarray | None:
    """uint8 mask (255 = use) of the plane's ``0..length x 0..width`` footprint, grown by a margin.

    ``None`` when the footprint does not project to a usable polygon inside the frame.
    """
    h, w = int(frame_shape[0]), int(frame_shape[1])
    world = np.array([[0.0, 0.0], [length_ft, 0.0], [length_ft, width_ft], [0.0, width_ft]],
                     dtype=np.float64).reshape(-1, 1, 2)
    img = cv2.perspectiveTransform(world, np.asarray(H_world_to_img, dtype=np.float64))
    if not np.all(np.isfinite(img)):
        return None
    pts = np.clip(img.reshape(-1, 2), -4 * max(h, w), 5 * max(h, w)).astype(np.int32)
    mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    if margin_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * margin_px + 1, 2 * margin_px + 1))
        mask = cv2.dilate(mask, k)
    return mask if int(mask.sum()) > 0 else None


def _detector():
    if hasattr(cv2, "SIFT_create"):
        return cv2.SIFT_create(nfeatures=4000), cv2.NORM_L2, "SIFT"
    return cv2.ORB_create(nfeatures=4000), cv2.NORM_HAMMING, "ORB"   # pragma: no cover


def estimate_h12_from_images(frame1: np.ndarray, frame2: np.ndarray,
                             mask1: np.ndarray | None = None) -> H12Result:
    """Feature-match camera 1 -> camera 2 and fit a RANSAC homography (the default source)."""
    det, norm, name = _detector()
    g1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY) if frame1.ndim == 3 else frame1
    g2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY) if frame2.ndim == 3 else frame2
    k1, d1 = det.detectAndCompute(g1, mask1)
    k2, d2 = det.detectAndCompute(g2, None)
    if d1 is None or d2 is None or len(k1) < 4 or len(k2) < 4:
        return H12Result(None, f"too few {name} features to align the cameras "
                               f"({len(k1 or [])} / {len(k2 or [])})", source=f"{name} features")
    pairs = cv2.BFMatcher(norm).knnMatch(d1, d2, k=2)
    good = [m for m, *rest in pairs if rest and m.distance < RATIO * rest[0].distance]
    if len(good) < MIN_INLIERS:
        return H12Result(None, f"only {len(good)} feature matches between the cameras",
                         matches=len(good), source=f"{name} features")
    p1 = np.float32([k1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    p2 = np.float32([k2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, inl = cv2.findHomography(p1, p2, cv2.RANSAC, RANSAC_PX)
    n_in = int(inl.sum()) if inl is not None else 0
    if H is None or n_in < MIN_INLIERS or not is_sane(H, frame1.shape[:2]):
        return H12Result(None, f"camera alignment rejected ({n_in} of {len(good)} matches agree)",
                         inliers=n_in, matches=len(good), source=f"{name} features")
    return H12Result(H / H[2, 2], f"aligned on {n_in} of {len(good)} feature matches",
                     inliers=n_in, matches=len(good), source=f"{name} features")


def is_sane(H: np.ndarray, frame_shape: tuple[int, int]) -> bool:
    """Reject folds and absurd scalings: the frame's corners must stay a convex, same-orientation
    quad of a plausible size."""
    h, w = frame_shape
    c = np.float64([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    hom = np.c_[c.reshape(-1, 2), np.ones(4)] @ np.asarray(H, dtype=np.float64).T
    if np.any(hom[:, 2] <= 0):                       # a corner crosses the horizon of camera 2
        return False
    q = (hom[:, :2] / hom[:, 2:]).astype(np.float32)
    if not cv2.isContourConvex(q.reshape(-1, 1, 2)):
        return False
    area = abs(cv2.contourArea(q))
    return 0.02 * w * h < area < 50.0 * w * h


def compose(H12: np.ndarray, H_world_to_img1: np.ndarray) -> np.ndarray:
    """``H_world->cam2``: the loaded plane carried into the paired camera."""
    H = np.asarray(H12, dtype=np.float64) @ np.asarray(H_world_to_img1, dtype=np.float64)
    return H / H[2, 2] if abs(H[2, 2]) > 1e-12 else H


def map_points(H: np.ndarray, pts) -> np.ndarray:
    """Apply a homography to ``(N, 2)`` points."""
    a = np.asarray(pts, dtype=np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(a, np.asarray(H, dtype=np.float64)).reshape(-1, 2)
