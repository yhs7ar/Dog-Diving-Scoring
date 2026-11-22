"""Standalone pool-coverage estimate — how much of the pool the plane calibration covers.

Separate from the GUI/CLI.  For every clip in ``analyzed-clips/`` that has an
assigned plane, load one calm frame, find the *saturated-blue* water pixels
(the pool), and report two numbers against the plane's image-space quad:

  * ``pool_in_quad``   — % of the mapped plane's pixels that are pool water
    (is the calibration sitting on the water?).
  * ``quad_of_pool``   — % of the frame's pool water that falls inside the quad
    (does the calibration capture the whole pool, or only part of it?).

"Water" is a fixed HSV threshold (saturated teal→blue water — the pools read as
a saturated blue, median hue ~105); results are an approximation and vary with
lighting.  Run: ``python src/plane_coverage.py``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from test_bench import manifest as M  # noqa: E402

ANALYZED_ROOT = SRC.parent / "analyzed-clips"

# Saturated water (OpenCV HSV: H 0-179, S/V 0-255): teal→blue pool water.
BLUE_H_LO, BLUE_H_HI = 80, 140
BLUE_S_MIN, BLUE_V_MIN = 120, 60


def pool_mask(frame_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV).astype(np.int32)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    return (h >= BLUE_H_LO) & (h <= BLUE_H_HI) & (s > BLUE_S_MIN) & (v > BLUE_V_MIN)


def quad_mask(shape: tuple[int, int], corners: dict) -> np.ndarray:
    pts = np.array([corners["00"], corners["10"], corners["11"], corners["01"]],
                   dtype=np.int32)
    m = np.zeros(shape, np.uint8)
    cv2.fillPoly(m, [pts], 255)
    return m > 0


def coverage(clip_dir: Path, plane_doc: dict) -> dict | None:
    man = M.read_manifest(clip_dir)
    clip = M.clip_video_path(man, clip_dir)
    if clip is None:
        return None
    cap = cv2.VideoCapture(str(clip))
    if not cap.isOpened():
        cap.release()
        return None
    ok, frame = cap.read()  # frame 0: calm water, dog not yet in
    cap.release()
    if not ok:
        return None

    blue = pool_mask(frame)
    quad = quad_mask(frame.shape[:2], plane_doc["plane"]["corners"])

    quad_area = int(quad.sum())
    blue_total = int(blue.sum())
    blue_in_quad = int((blue & quad).sum())

    return {
        "clip_id": man["clip_id"],
        "frame_w": frame.shape[1],
        "frame_h": frame.shape[0],
        "quad_area_px": quad_area,
        "pool_px_total": blue_total,
        "pool_px_in_quad": blue_in_quad,
        "pool_in_quad_pct": 100.0 * blue_in_quad / quad_area if quad_area else None,
        "quad_of_pool_pct": 100.0 * blue_in_quad / blue_total if blue_total else None,
    }


def main() -> int:
    rows = []
    for clip_dir in sorted(p for p in ANALYZED_ROOT.iterdir() if p.is_dir()):
        man_path = clip_dir / "manifest.json"
        if not man_path.is_file():
            continue
        man = M.read_manifest(clip_dir)
        plane = M.clip_plane(man)
        if plane is None:
            continue
        r = coverage(clip_dir, plane)
        if r is None:
            print(f"{man['clip_id']}: clip unreadable")
            continue
        rows.append(r)
        print(
            f"{r['clip_id']:<24} quad={r['quad_area_px']:>6}px  "
            f"pool-in-quad={r['pool_in_quad_pct']:>5.1f}%  "
            f"quad-of-pool={r['quad_of_pool_pct']:>5.1f}%"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
