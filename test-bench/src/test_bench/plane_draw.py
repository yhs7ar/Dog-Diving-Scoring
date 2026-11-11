"""Plane re-draw — recompute the homography from re-placed pool corners.

The technician re-places the four pool corners (``00, 10, 11, 01``) on a frame;
the module refits the plane homography (route A: direct correspondences) and
returns a new document.  The world ``(X, Y)`` of each corner is taken from the
*existing* homography applied to the existing corners — the pool is the same
physical rectangle, only the corner placement moved — so the fit is ``(world,
new_pixel)``.  Pose / water / lens / extent / dimensions carry over unchanged;
``measure.plane_distance`` answers from the homography alone, so those fields
staying put is sound.
"""
from __future__ import annotations

import copy
from datetime import datetime, timezone
from typing import Any

import numpy as np

from dog_diving.pool import homography as _homog

CORNER_ORDER = ("00", "10", "11", "01")


def redraw_plane(existing_doc: dict[str, Any], pixel_corners: dict[str, Any],
                 *, setup_id: str | None = None, updated: str | None = None) -> dict[str, Any]:
    """Return a new plane doc with the homography + corners replaced.

    ``pixel_corners`` is ``{corner: [x, y]}`` for the four pool corners.
    """
    if not existing_doc or not (existing_doc.get("plane") or {}):
        raise ValueError("a plane document with a plane block is required")
    old_corners = existing_doc["plane"].get("corners") or {}
    missing = set(CORNER_ORDER) - set(old_corners)
    if missing:
        raise ValueError(f"existing doc has no full 4-corner set; missing {sorted(missing)}")
    for k in CORNER_ORDER:
        v = pixel_corners.get(k)
        if not isinstance(v, (list, tuple)) or len(v) != 2:
            raise ValueError(f"pixel_corners[{k!r}] must be [x, y]")

    H_old = np.asarray(existing_doc["homography"], dtype=float).reshape(3, 3)
    old_px = np.asarray([old_corners[k] for k in CORNER_ORDER], dtype=float)
    world = _homog.apply_homography_inv(H_old, old_px)          # (X, Y) feet
    new_px = np.asarray([pixel_corners[k] for k in CORNER_ORDER], dtype=float)
    H_new, info = _homog.homography_from_correspondences(world, new_px, robust=True)

    doc = copy.deepcopy(existing_doc)
    doc["homography"] = H_new.tolist()
    doc["plane"]["corners"] = {k: [float(v) for v in pixel_corners[k]] for k in CORNER_ORDER}
    doc["setup_id"] = setup_id or f"{doc.get('setup_id', 'plane')}-redrawn"
    doc["updated"] = updated or datetime.now(timezone.utc).isoformat()
    doc["notes"] = ((doc.get("notes") or "").rstrip() +
                    f"\nredrawn: homography refit from {len(world)} corners").strip()
    return doc
