"""Annotate — the per-frame overlay renderer (M7 deliverable, PLAN.md §5).

PLAN §5 ``annotate.py``, verbatim:

    Per-frame overlay: dock-edge line, waterline, calibration marks + 1-ft grid ticks,
    trajectory polyline (yellow), last point (red), the measurement point with a vertical
    drop line to the waterline, and a banner ``distance 22' 7" \u00b1 4" | frame 138 |
    calib RMS 1.3"``.  VideoWriter with the same fps/resolution as source.

Layering (bottom to top): calibration (grid, dock, water, marks) -> trajectory
(polyline, fitted arc, inliers, last point) -> contact candidates -> the measurement
point + its drop line -> the banner.  The two middle layers are *not* re-implemented
here: they are :func:`dog_diving.trajectory.draw_overlay` and
:func:`dog_diving.contact.draw_candidates`, the same functions the M4/M5 gates already
verified, so the reference-compatible yellow polyline and the candidate list cannot
drift between the gate and the shipped overlay.

Every number in the banner is read from the measurement dict, never recomputed
--------------------------------------------------------------------------
A banner that re-derives the distance is a second implementation of the measurement
and will eventually disagree with ``result.json``.  :func:`banner_text` formats
``measurement["display"]`` / ``measurement["sigma_in"]`` and nothing else; when the
measurement has no inches (``total_inches is None`` — the current state of every clip,
since ``data/calib/`` is empty) the banner says so in words rather than drawing a
plausible-looking number.

Measured: the codec is not a free choice (2026-10-08, this host, 60 frames of 21-01.mp4)
-----------------------------------------------------------------------------------
PLAN §11 M7 requires the batch artifacts to be *"unchanged by re-runs"*.  The overlay
frames are deterministic by construction; the container is not necessarily:

| fourcc | frame | size | byte-identical across two runs? |
|---|---|---|---|
| ``mp4v`` | 14.9 ms | 1.61 MB | **yes** (same sha256, same size) |
| ``avc1`` | 8.1 ms | 15.55 MB | **no** (differing sha256 at identical size) |

``avc1``/H.264 additionally cannot be used on this host: OpenCV reports
*"Failed to load OpenH264 library: openh264-2.5.0-win64.dll"* and the writer produces
a 10x-larger, non-deterministic file.  The default is therefore ``mp4v``, and
``test_m7_annotate_report_cli.py`` pins the byte-determinism so a codec change that
breaks the §11 gate fails loudly instead of silently.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Sequence

import cv2
import numpy as np

from . import calibration as _calibration
from . import contact as _contact
from . import trajectory as _trajectory
from . import video as _video

# --- codec ------------------------------------------------------------------
# mp4v is the only writer measured byte-deterministic on this host (see module docstring).
DEFAULT_FOURCC = "mp4v"

# --- colours (BGR; the trajectory/contact layers carry their own) -----------
C_DOCK = (0, 255, 0)          # dock edge — same green as the calibration tool
C_WATER = (255, 255, 0)       # waterline — same yellow-cyan as the calibration tool
C_MARK = (0, 0, 255)          # foot marks, red
C_GRID = (0, 200, 255)        # 1-ft ticks, orange
C_GRID_MINOR = (0, 120, 160)
C_POINT = (0, 0, 255)         # measurement point + drop line, red
C_BANNER_BG = (0, 0, 0)
C_BANNER_FG = (255, 255, 255)

FONT = cv2.FONT_HERSHEY_SIMPLEX
BANNER_H = 46


# --- the banner -------------------------------------------------------------
def banner_text(measurement: dict | None, calib: dict | None, frame: int | None) -> str:
    """PLAN §5's banner line.  ASCII only — ``cv2.putText`` cannot render ``\u00b1``.

    Reads the measurement, never recomputes it.  ``sigma_in`` is labelled a lower
    bound when the budget is incomplete (``sigma_complete`` False), which is the
    normal case whenever the frame-quantisation term was refused (M6 finding 5).
    """
    m = measurement or {}
    if m.get("display"):
        txt = f"distance {m['display']}"
        if m.get("sigma_in") is not None:
            txt += f" +/- {float(m['sigma_in']):.1f}\""
            if not m.get("sigma_complete", False):
                txt += " (lower bound)"
    elif m.get("total_inches") is not None:
        txt = f"distance {float(m['total_inches']):.1f} in"
    else:
        txt = "distance n/a (no calibration)"

    rms = (calib or {}).get("fit", {}).get("residual_rms_in")
    calib_txt = (f"calib RMS {float(rms):.1f}\"" if rms is not None else "calib none")
    return f"{txt} | frame {frame} | {calib_txt}"


# --- calibration layer ------------------------------------------------------
def draw_calibration(
    frame: np.ndarray,
    calib: dict | None,
    *,
    show_grid: bool = True,
    upto_ft: int = 31,
) -> np.ndarray:
    """Dock edge, waterline, foot marks and PLAN §4.3 G4's 1-ft grid ticks.

    Mirrors ``calibrate_tool.draw_grid`` (same ticks, same colours) so the overlay a
    reviewer sees in the batch video is the same one the G4 visual check uses.  A
    calibration-free clip draws nothing here rather than inventing a datum.
    """
    out = frame
    if not calib:
        return out

    water = calib.get("waterline")
    if water and len(water) == 2:
        (x1, y1), (x2, y2) = water[0], water[1]
        cv2.line(out, (int(x1), int(y1)), (int(x2), int(y2)), C_WATER, 2)

    dock = calib.get("dock_line")
    if dock and len(dock) == 2:
        (x1, y1), (x2, y2) = dock[0], dock[1]
        cv2.line(out, (int(x1), int(y1)), (int(x2), int(y2)), C_DOCK, 2)

    for mk in calib.get("marks", []) or []:
        mx, my = int(mk["x_px"]), int(mk["y_px"])
        cv2.circle(out, (mx, my), 5, C_MARK, -1)
        cv2.putText(out, str(mk.get("label", "")), (mx + 6, my - 6), FONT, 0.6, C_MARK, 2)

    if show_grid and water and len(water) == 2:
        y_top = int(min(water[0][1], water[1][1]))
        y_bot = int(max(water[0][1], water[1][1]))
        w = out.shape[1]
        for pt in _calibration.grid_points(upto_ft + 1):
            x = _calibration.inverse_ft(pt["nominal_in"], calib)
            if not np.isfinite(x) or not (0 <= x < w):
                continue
            major = pt["ft"] % 5 == 0
            cv2.line(out, (int(x), y_top), (int(x), y_bot),
                     C_GRID if major else C_GRID_MINOR, 2 if major else 1)
            if major:
                cv2.putText(out, str(pt["ft"]), (int(x) + 3, y_top - 6), FONT, 0.6, C_GRID, 2)
    return out


# --- the measurement point --------------------------------------------------
def draw_measurement(frame: np.ndarray, measurement: dict | None) -> np.ndarray:
    """The measurement point, its vertical drop line to the waterline, and its (x, y)."""
    out = frame
    m = measurement or {}
    x, y = m.get("x_px"), m.get("y_px")
    if x is None or y is None:
        return out
    xi, yi = int(round(float(x))), int(round(float(y)))
    wy = m.get("waterline_y")
    if wy is not None:
        cv2.line(out, (xi, yi), (xi, int(round(float(wy)))), C_POINT, 2)
    # The SAME mark Review draws — a tilted X, not a plus (operator order 2026-10-21: the X for the
    # last detection did not match the pane's).  One quantity, one mark, in the pane and in the still.
    cv2.drawMarker(out, (xi, yi), C_POINT, cv2.MARKER_TILTED_CROSS, 26, 3)
    cv2.putText(out, f"({float(x):.0f}, {float(y):.0f})", (xi + 10, yi + 22), FONT, 0.6, C_POINT, 2)
    return out


def draw_pool_surface(
    frame: np.ndarray,
    pool_doc: dict | None,
    *,
    along_ft: float | None = None,
    width_ft: float | None = None,
    step_ft: float = 1.0,
    labels: bool = True,
    fade_above_y: float | None = None,
) -> np.ndarray:
    """The 3-D route's overlay: the metric lattice on the **water** surface.

    The 1-D calibration draws ticks along the marked plane, where the marks physically are.  This
    draws the grid where the *measurement* is taken — the true water plane, which phase 5
    reconstructs around the occlusion — so the overlay and the reported distance agree by
    construction.  It is a fixed camera and a fixed lattice, so the lattice could be pre-rendered
    once per clip; it is drawn per frame for now because it is cheap relative to decode.

    ``pool_doc`` is a loaded ``dog_diving_pool_calibration/v1`` document.  An unloadable document
    raises rather than silently drawing nothing: a reviewer seeing no grid cannot tell "no
    calibration" from "calibration ignored".
    """
    from dog_diving.pool import calib as _pcalib
    from dog_diving.pool import grid as _pgrid
    from dog_diving.pool import plane as _pplane

    if pool_doc is None:
        return frame
    _pcalib.require_schema(pool_doc)

    # A plane calibration (PLAN §15) draws from ITS OWN matrix.  Rebuilding a homography from the
    # decomposed pose would draw a different lattice — the two disagree by 9.8 in under an assumed
    # lens — and the reviewer would be looking at a grid the calibration never claimed.
    plane_block = pool_doc.get("plane") or {}
    if plane_block:
        return _pplane.draw_grid_on(
            frame, pool_doc["homography"],
            length_ft=float(plane_block.get("length_ft", _pplane.DEFAULT_LENGTH_FT)
                            if along_ft is None else along_ft),
            width_ft=float(plane_block.get("width_ft", _pplane.DEFAULT_WIDTH_FT)
                           if width_ft is None else width_ft),
            lens=_pcalib.lens_from_document(pool_doc), labels=bool(labels),
            fade_above_y=fade_above_y)

    pose = _pcalib.pose_from_document(pool_doc)
    water = _pcalib.water_from_document(pool_doc)
    lens = _pcalib.lens_from_document(pool_doc)
    from dog_diving import config as _cfg

    pool = pool_doc.get("pool") or {}
    return _pgrid.draw_grid(
        frame, pose, water, lens=lens,
        along_ft=float(pool.get("along_ft", _cfg.POOL_GRID_UPTO_FT)
                       if along_ft is None else along_ft),
        width_ft=float(pool.get("width_ft", _cfg.POOL_WIDTH_FT)
                       if width_ft is None else width_ft),
        y_min_ft=float(pool.get("y_min_ft", 0.0)),
        step_ft=float(step_ft), labels=bool(labels))


# --- the composed frame -----------------------------------------------------
def draw_frame(
    frame: np.ndarray,
    *,
    frame_idx: int | None = None,
    traj: Any | None = None,
    contact: Any | None = None,
    calib: dict | None = None,
    measurement: dict | None = None,
    show_grid: bool = True,
    show_candidates: bool = False,
    show_traj_conf: bool = True,
    banner: bool = True,
    pool_doc: dict | None = None,
    pool_result: dict | None = None,
    entry_mark: bool = True,
    fade_above_y: float | None = None,
    arc: dict | None = None,
    excluded: set[int] | tuple[int, ...] = (),
) -> np.ndarray:
    """One annotated frame.  Pure: ``frame`` is copied, never drawn onto in place.

    ``pool_doc`` switches the overlay from the 1-D tick strip to the 3-D route's metric lattice on
    the water plane (see :func:`draw_pool_surface`).  Supplying it *replaces* the tick strip rather
    than adding to it, because the two draw different quantities in the same pixels — the ticks mark
    the graded plane, the lattice marks the measurement plane — and a frame carrying both invites
    reading the wrong one as the score.

    ``show_candidates`` is **off by default**, and that is a measured decision rather than
    a preference.  PLAN §5's ``annotate.py`` layer list is *"dock-edge line, waterline,
    calibration marks + 1-ft grid ticks, trajectory polyline, last point, the measurement
    point with a vertical drop line, and a banner"* — the ranked candidate list is not in
    it (``contact.draw_candidates`` documents itself as *"the GUI's frame ticks"*, and the
    candidates are fully recorded in ``result.json``).  Drawn anyway, it collides: the
    banner rectangle occupies rows 0..``BANNER_H`` and :func:`dog_diving.contact.draw_candidates`
    starts its list at y=30, so on 18-07 the banner hides candidate ``#1``'s line outright
    (verified by eye on ``out/18-07/final_frame.png``).  A reviewer's video that silently
    drops the preselected candidate is worse than one without the list; the flag remains
    for QA use.
    """
    out = frame.copy()
    if pool_doc is not None:
        out = draw_pool_surface(out, pool_doc, labels=show_grid, fade_above_y=fade_above_y)
    else:
        out = draw_calibration(out, calib, show_grid=show_grid)
    if traj is not None:
        out = _trajectory.draw_overlay(out, traj, show_conf=show_traj_conf, arc=arc,
                                       excluded=excluded)
    if show_candidates and contact is not None:
        out = _contact.draw_candidates(out, contact)
    if pool_result is not None and entry_mark:
        out = _draw_pool_entry(out, pool_result)
    out = draw_measurement(out, measurement)
    if banner:
        h, w = out.shape[:2]
        cv2.rectangle(out, (0, 0), (w, BANNER_H), C_BANNER_BG, -1)
        text = (banner_text(measurement, calib, frame_idx) if pool_result is None
                else pool_banner_text(pool_result, measurement, frame_idx))
        cv2.putText(out, text, (10, 31), FONT, 0.8, C_BANNER_FG, 2)
    return out


def _draw_pool_entry(frame: np.ndarray, pool_result: dict) -> np.ndarray:
    """Mark the 3-D route's ``(X, Y)`` entry point, labelled in feet.

    Distinct from :func:`draw_measurement`'s 1-D point on purpose: that one is a *pixel* on the
    graded row with a distance read off a fit, this one is a *world coordinate on the water plane*
    recovered by intersecting the tail-set ray with the plane.  They can disagree by design (the
    measured 186 in on a row 12 ft off the marked plane), so they must not be drawn as if they were
    the same quantity.
    """
    out = frame
    entry = (pool_result or {}).get("entry") or {}
    if entry.get("X_ft") is None or entry.get("Y_ft") is None:
        return out
    if entry.get("x_px") is None or entry.get("y_px") is None:
        return out
    xi, yi = int(round(float(entry["x_px"]))), int(round(float(entry["y_px"])))
    h, w = out.shape[:2]
    if not (0 <= xi < w and 0 <= yi < h):
        return out
    cv2.drawMarker(out, (xi, yi), C_POINT, cv2.MARKER_TILTED_CROSS, 26, 3)
    cv2.putText(out, f"({float(entry['X_ft']):.1f}, {float(entry['Y_ft']):.1f}) ft", (xi + 12, yi + 26),
                FONT, 0.6, C_POINT, 2)
    return out


def pool_banner_text(pool_result: dict, measurement: dict | None, frame: int | None) -> str:
    """The banner when the 3-D route supplied the number: the along-pool distance in feet.

    ``distance_ft`` **is** the along-pool (plank) length the plan asks for — it is ``X_ft``, the
    coordinate along the measurement axis, which is what "the dog jumped 24 ft" means.  The 1-D
    route's inches are shown beside it because the cross-check between the two is the project's
    correctness proof, not a detail.
    """
    entry = (pool_result or {}).get("entry") or {}
    parts: list[str] = []
    if frame is not None:
        parts.append(f"f{int(frame)}")
    if entry.get("X_ft") is not None and entry.get("Y_ft") is not None:
        parts.append(f"entry=({float(entry['X_ft']):.2f}, {float(entry['Y_ft']):.2f}) ft")
        parts.append(f"plank={float(entry['distance_ft']):.2f} ft")
    if (measurement or {}).get("distance_in") is not None:
        parts.append(f"1-D={float(measurement['distance_in']):.1f} in")
    if entry.get("height_assumed_ft") is not None:
        parts.append(f"h={float(entry['height_assumed_ft']):.2f} ft")
    if not (pool_result or {}).get("pose_ok", True):
        parts.append("POSE-GATE-FAILED")
    return "   ".join(parts) or "(no entry)"


# --- video / still writers --------------------------------------------------
def write_video(
    clip: str | Path,
    out_path: str | Path,
    *,
    traj: Any | None = None,
    contact: Any | None = None,
    calib: dict | None = None,
    measurement: dict | None = None,
    frame_range: tuple[int, int] | None = None,
    fourcc: str = DEFAULT_FOURCC,
    show_grid: bool = True,
    show_candidates: bool = False,
    show_traj_conf: bool = True,
    pool_doc: dict | None = None,
    pool_result: dict | None = None,
    arc: dict | None = None,
    excluded: set[int] | tuple[int, ...] = (),
    progress_cb: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """Write PLAN §7's ``annotated.mp4`` at the source resolution and fps.

    Iterates the clip through :func:`dog_diving.video.iter_frames` (the M1-verified
    reader), so the annotated video has exactly the decoded frame set — no VFR
    over-declaration from the container.  Raises if the writer cannot be created:
    a silently empty overlay video is worse than a failed run.

    ``arc``/``excluded`` carry the Review step's deselections into the video, so an export taken
    after the operator clicked the other dog's detections out shows the arc they kept.
    """
    clip = Path(clip)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    info = _video.probe(clip, count_frames=False)
    vw = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*fourcc),
                         float(info.fps), (int(info.width), int(info.height)))
    if not vw.isOpened():
        raise RuntimeError(f"cv2.VideoWriter could not open {out_path} with fourcc={fourcc!r}")

    start, stop = (0, None) if frame_range is None else (int(frame_range[0]), int(frame_range[1]))
    n = 0
    try:
        for idx, frame in _video.iter_frames(clip, start=start, stop=stop):
            vw.write(draw_frame(frame, frame_idx=idx, traj=traj, contact=contact, calib=calib,
                                measurement=measurement, show_grid=show_grid,
                                show_candidates=show_candidates,
                                show_traj_conf=show_traj_conf,
                                pool_doc=pool_doc, pool_result=pool_result,
                                arc=arc, excluded=excluded))
            n += 1
            if progress_cb is not None:
                progress_cb(n, (stop - start) if stop is not None else info.n_frames)
    finally:
        vw.release()
    return {"path": str(out_path), "n_frames": n, "fourcc": fourcc,
            "fps": float(info.fps), "width": int(info.width), "height": int(info.height)}


def write_still(
    clip: str | Path,
    out_path: str | Path,
    frame_index: int,
    *,
    traj: Any | None = None,
    contact: Any | None = None,
    calib: dict | None = None,
    measurement: dict | None = None,
    show_grid: bool = True,
    show_candidates: bool = False,
    show_traj_conf: bool = True,
    pool_doc: dict | None = None,
    pool_result: dict | None = None,
    arc: dict | None = None,
    excluded: set[int] | tuple[int, ...] = (),
) -> dict[str, Any]:
    """PLAN §7's ``final_frame.png`` — the chosen contact frame, with callouts.

    The single frame is read through :func:`dog_diving.video.iter_frames` (never
    ``CAP_PROP_POS_FRAMES`` seeking, which is unreliable on the VFR MOVs — M1).
    """
    clip = Path(clip)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    idx = max(0, int(frame_index))
    frame = None
    for i, f in _video.iter_frames(clip, start=idx, stop=idx + 1):
        frame = f
        break
    if frame is None:
        raise RuntimeError(f"could not read frame {idx} of {clip}")
    drawn = draw_frame(frame, frame_idx=idx, traj=traj, contact=contact, calib=calib,
                       measurement=measurement, show_grid=show_grid,
                       show_candidates=show_candidates, show_traj_conf=show_traj_conf,
                       pool_doc=pool_doc, pool_result=pool_result,
                       arc=arc, excluded=excluded)
    if not cv2.imwrite(str(out_path), drawn):
        raise RuntimeError(f"cv2.imwrite failed for {out_path}")
    return {"path": str(out_path), "frame": idx, "width": drawn.shape[1], "height": drawn.shape[0]}
