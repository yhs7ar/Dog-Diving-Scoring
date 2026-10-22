"""``dog_diving.pool`` — the 3-D pool-surface translation pipeline.

`pool-translation-plan.txt` implemented as six phases, one module each:

======================  ============================================================
phase                   module
======================  ============================================================
1  lens / distortion    :mod:`dog_diving.pool.lens`
2  origin + features    `calibrate_tool` (clicks) + :mod:`dog_diving.pool.calib` (schema)
3  transverse scaling   :mod:`dog_diving.pool.vanishing`
4  planar homography    :mod:`dog_diving.pool.homography`
5  water-plane extrap.  :mod:`dog_diving.pool.surface`
6  trajectory intersect :mod:`dog_diving.pool.intersection`
—  shared ray/plane     :mod:`dog_diving.pool.geometry`
—  the metric grid      :mod:`dog_diving.pool.grid`
—  document + gates     :mod:`dog_diving.pool.calib`
—  end to end           :mod:`dog_diving.pool.pipeline`
======================  ============================================================

The world frame every module shares: origin at the 0 ft mark of the dock edge, ``+X`` along the
jump direction in feet, ``+Y`` across the pool in feet, ``+Z`` up with the marks' plane at Z = 0.
Pixel coordinates are always **undistorted**.
"""

from __future__ import annotations

from . import calib, geometry, grid, homography, intersection, lens, pipeline, surface, vanishing

__all__ = [
    "calib", "geometry", "grid", "homography", "intersection", "lens", "pipeline",
    "surface", "vanishing",
]
