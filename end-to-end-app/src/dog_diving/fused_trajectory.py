"""Fused-cloud trajectory — the stereo-anchored 3-D trajectory (step 4a).

Re-points the mono bootstrap at the fused two-camera cloud
-----------------------------------------------------------
:func:`dog_diving.pool.pipeline.track_to_world` builds a 3-D trajectory from a **mono**
camera: it unprojects each detection pixel to the water height, fits one vertical plane
through the smeared ``(X, Y)``, then re-intersects every ray with that plane.  That
single-pass bootstrap is measurably biased — 110 in on the project's shallow synthetic
view and 73 in on a steep one (``test_m30``) — because a monocular pixel carries no depth.

This module is the **fused-cloud replacement**: it builds per-camera point clouds through
the :class:`dog_diving.fusion.PointCloudSource` seam, merges them through a
:class:`dog_diving.fusion.FusionConfig` into one cloud in the reference camera's frame, and
hands that merged cloud to the existing world-frame trajectory/distance chain
(:func:`dog_diving.pc_trajectory.trajectory_from_cloud` /
:func:`dog_diving.pc_distance.pc_distance`, which already do the vertical-plane + 3-D-arc +
water-crossing).

What the fused cloud contributes, precisely
-------------------------------------------
* The **water plane** — fitted on the *fused* cloud (two cameras of water coverage), which
  is what anchors the world frame and the crossing ``Z``.  This is the part the mono
  bootstrap could only *unproject* to (and got wrong).
* The dog's per-detection 3-D points still come from the **reference camera's own
  per-frame clouds** (``frame_{i:06d}.npy``) — real NEURAL depth, no unprojection.  A
  detection is a pixel in the reference image, and that camera's depth at that pixel is the
  honest answer; the second camera's points do not help locate a dog that was only detected
  in the first image, and inventing a cross-camera match would be a guess, so the fused path
  does not do it.

Both together are the "stereo depth rather than a single view" the operator asked for: the
water plane and world frame are stereo-anchored, and the dog points are real depth rather
than mono unprojection.

The mono path stays as the fallback — :func:`dog_diving.pool.pipeline.track_to_world` is
untouched.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from . import fusion as _fusion
from . import pc_distance as _pc_distance
from . import pc_trajectory as _pc
from .fusion import FusionConfig, PointCloudSource


def fused_water_plane(
    source: PointCloudSource,
    config: FusionConfig,
    *,
    to_world: _pc.ToWorld = _pc.zed_to_world,
    **fit_kwargs: Any,
) -> dict[str, Any]:
    """Build the fused cloud via the source seam and fit the water plane on it.

    ``source.load()`` returns per-camera clouds; :func:`dog_diving.fusion.fuse_point_clouds`
    merges them into the reference camera's frame (metres); then
    :func:`dog_diving.pc_trajectory.fit_water_plane` fits the dominant horizontal plane and
    reports ``water_z_ft`` in world feet.

    Returns ``{status, water_z_ft, water_fit, reference_serial, counts, n_points}``.
    ``status`` / ``water_z_ft`` / ``water_fit`` mirror ``fit_water_plane``'s; a refused fit
    (``no_points`` / ``degenerate`` / ``few_inliers`` / ``not_horizontal``) carries no
    ``water_z_ft``.
    """
    fused = _fusion.fuse_from_source(source, config)
    fit = _pc.fit_water_plane(fused.points, to_world=to_world, **fit_kwargs)
    return {
        "status": fit["status"],
        "water_z_ft": fit.get("water_z_ft"),
        "water_fit": fit,
        "reference_serial": fused.reference_serial,
        "counts": dict(fused.counts),
        "n_points": int(len(fused)),
    }


def _fused_provenance(fused: _fusion.FusedCloud) -> dict[str, Any]:
    return {
        "reference_serial": fused.reference_serial,
        "counts": dict(fused.counts),
        "n_points": int(len(fused)),
    }


def trajectory_from_fused_cloud(
    frames_dir: str | Path,
    detections: Iterable[Any],
    source: PointCloudSource,
    config: FusionConfig,
    *,
    to_world: _pc.ToWorld = _pc.zed_to_world,
    min_points: int = _pc.MIN_POINTS,
    water_z_ft: float | None = None,
) -> dict[str, Any]:
    """The fused-cloud path beside :func:`dog_diving.pool.pipeline.track_to_world`.

    Fuses the per-camera clouds into one reference-frame cloud and hands it to
    :func:`dog_diving.pc_trajectory.trajectory_from_cloud` as the water cloud, so the water
    plane (and therefore the crossing ``Z`` and world frame) is stereo-anchored.  The dog's
    points come from the per-frame reference clouds in ``frames_dir`` — real depth, no
    unprojection.

    When ``water_z_ft`` is ``None`` the fused cloud is built and fitted (``water_z_source``
    comes back ``"cloud_fit"`` with ``water_fit_source == "fused_cloud"`` and a ``fused``
    provenance block).  When ``water_z_ft`` is supplied explicitly the fusion is **skipped**
    entirely — the fused cloud is only needed for the water plane, so a CPU host can still
    score a run from an operator-supplied water level.
    """
    if water_z_ft is None:
        fused = _fusion.fuse_from_source(source, config)
        r = _pc.trajectory_from_cloud(
            frames_dir, detections, water_z_ft=None, water_cloud=fused.points,
            min_points=min_points, to_world=to_world,
        )
        r["fused"] = _fused_provenance(fused)
        r["water_fit_source"] = "fused_cloud"
        return r
    return _pc.trajectory_from_cloud(
        frames_dir, detections, water_z_ft=water_z_ft,
        min_points=min_points, to_world=to_world,
    )


def pc_distance_fused(
    frames_dir: str | Path,
    detections: Iterable[Any],
    source: PointCloudSource,
    config: FusionConfig,
    *,
    lip: dict[str, Any] | None = None,
    to_world: _pc.ToWorld = _pc.zed_to_world,
    min_points: int = _pc.MIN_POINTS,
    water_z_ft: float | None = None,
) -> dict[str, Any]:
    """Route ``"pc"`` with the water plane anchored in the fused cloud.

    Mirrors :func:`trajectory_from_fused_cloud` but returns the
    :func:`dog_diving.pc_distance.pc_distance` result shape (``route`` / ``total_inches`` /
    ``display`` / ...) with the same ``fused`` provenance and ``water_fit_source`` fields.
    """
    if water_z_ft is None:
        fused = _fusion.fuse_from_source(source, config)
        r = _pc_distance.pc_distance(
            frames_dir, detections, lip=lip, water_z_ft=None, water_cloud=fused.points,
            min_points=min_points, to_world=to_world,
        )
        r["fused"] = _fused_provenance(fused)
        r["water_fit_source"] = "fused_cloud"
        return r
    return _pc_distance.pc_distance(
        frames_dir, detections, lip=lip, water_z_ft=water_z_ft,
        min_points=min_points, to_world=to_world,
    )
