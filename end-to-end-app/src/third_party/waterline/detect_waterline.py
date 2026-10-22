#!/usr/bin/env python3
"""
Waterline detection from temporal noise in a stationary-camera video.

Core idea
---------
With a fixed camera, pixels on rigid structure (dock, pool wall, deck) barely
change intensity from frame to frame -- only sensor noise and compression
artifacts. Water pixels keep changing (ripples, reflections, flicker) even
when the surface looks "calm". A per-pixel temporal statistic computed over a
calm window of frames therefore separates water (high temporal noise) from
static structure (low temporal noise) without relying on colour or edges.

Pipeline (each stage is a standalone, testable function):

    1. load_frames()          -> (N, H, W) or (N, H, W, 3) uint8 stack
    2. compute_noise_maps()   -> per-pixel temporal variance / std /
                                 detrended variance / mean |f[t]-f[t-k]|
       normalize_noise_map()  -> 8-bit map for display and thresholding
    3. threshold_noise_map()  -> binary mask (Otsu or manual)
       clean_mask()           -> morphological open/close + keep largest blob
    4. extract_waterline()    -> ordered (x, y) points on the water boundary
    5. waterline_stability()  -> per-column std of waterline over sub-windows
       draw_waterline(), write_overlay_clip() -> visual validation
    6. save_outputs()         -> JSON / .npy / PNG / MP4 on disk

IMPORTANT: run this on a *calm* segment (before any dog jumps in). A dog or
splash in frame produces huge temporal variance wherever it moves, which
corrupts the statistics and will drag the detected "water" region onto the
dog's path. The output of this script is a *baseline* waterline position,
not a tracker that works through an active splash.

Run `python detect_waterline.py --help` for CLI usage; see README.md.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

NOISE_METHODS = ("variance", "std", "detrended", "diff")
WATER_SIDES = ("auto", "below", "above", "left", "right")


# ---------------------------------------------------------------------------
# 1. Frame extraction
# ---------------------------------------------------------------------------

@dataclass
class VideoInfo:
    path: str
    fps: float
    frame_count: int
    width: int
    height: int


def probe_video(video_path: str | Path) -> VideoInfo:
    """Read basic metadata from a video file."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")
    try:
        info = VideoInfo(
            path=str(video_path),
            fps=cap.get(cv2.CAP_PROP_FPS) or 30.0,
            frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
    finally:
        cap.release()
    return info


def resolve_frame_range(
    info: VideoInfo,
    start_frame: int | None = None,
    end_frame: int | None = None,
    start_time: float | None = None,
    end_time: float | None = None,
) -> tuple[int, int]:
    """Turn frame-index and/or time (seconds) bounds into [start, end) frame indices.

    Frame indices take precedence over times if both are given.
    `end` is exclusive. Missing bounds default to the start/end of the video.
    """
    if start_frame is None:
        start_frame = int(round(start_time * info.fps)) if start_time is not None else 0
    if end_frame is None:
        end_frame = int(round(end_time * info.fps)) if end_time is not None else info.frame_count
    start_frame = max(0, start_frame)
    if info.frame_count > 0:
        end_frame = min(end_frame, info.frame_count)
    if end_frame <= start_frame:
        raise ValueError(f"Empty frame range: start={start_frame}, end={end_frame}")
    return start_frame, end_frame


def load_frames(
    video_path: str | Path,
    start_frame: int = 0,
    end_frame: int | None = None,
    color: bool = False,
    step: int = 1,
    scale: float = 1.0,
) -> np.ndarray:
    """Load frames [start_frame, end_frame) from a video as a uint8 stack.

    Args:
        video_path: path to the .mp4 (or anything OpenCV can decode).
        start_frame: first frame index (inclusive).
        end_frame: last frame index (exclusive); None = until end of video.
        color: if True return (N, H, W, 3) BGR, else (N, H, W) grayscale.
        step: keep every `step`-th frame (temporal subsampling).
        scale: spatial resize factor (e.g. 0.5 halves width/height) to save
            memory/time on high-resolution footage.

    Returns:
        uint8 array of shape (N, H, W) or (N, H, W, 3).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")
    frames = []
    try:
        # Seeking is not always frame-accurate for compressed video, so we seek
        # and then verify the reported position; fall back to reading forward.
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if idx != start_frame:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            idx = 0
        while end_frame is None or idx < end_frame:
            ok, frame = cap.read()
            if not ok:
                break
            if idx >= start_frame and (idx - start_frame) % step == 0:
                if scale != 1.0:
                    frame = cv2.resize(frame, None, fx=scale, fy=scale,
                                       interpolation=cv2.INTER_AREA)
                if not color:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frames.append(frame)
            idx += 1
    finally:
        cap.release()
    if not frames:
        raise ValueError(f"No frames read from {video_path} in range [{start_frame}, {end_frame})")
    return np.stack(frames)


# ---------------------------------------------------------------------------
# 2. Temporal noise map
# ---------------------------------------------------------------------------

def _as_gray_float(frame: np.ndarray, pre_blur: int) -> np.ndarray:
    if frame.ndim == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    f = frame.astype(np.float32)
    if pre_blur and pre_blur > 1:
        k = pre_blur | 1  # kernel must be odd
        f = cv2.GaussianBlur(f, (k, k), 0)
    return f


def compute_noise_maps(frames: np.ndarray, pre_blur: int = 3, diff_lag: int = 1) -> dict[str, np.ndarray]:
    """Compute all temporal noise statistics in one pass over the frame stack.

    Args:
        frames: (N, H, W) or (N, H, W, 3) uint8 stack, N >= 2. Colour frames
            are converted to grayscale (intensity flicker is what we measure).
        pre_blur: Gaussian kernel applied to each frame first. Suppresses
            per-pixel sensor/compression noise (which is spatially white) while
            keeping ripple flicker (which is spatially correlated). 0 disables.
        diff_lag: frame gap k for the differencing statistic,
            mean |frame[t] - frame[t-k]|. k=1 is plain consecutive-frame
            differencing. Larger k picks up slower ripples (see notes).

    Returns:
        dict of float32 (H, W) maps:
          "variance":  per-pixel temporal variance across the window.
          "std":       sqrt(variance).
          "detrended": variance of each pixel's residual after removing a
                       per-pixel linear trend in time.
          "diff":      mean |frame[t] - frame[t-k]| across the window.

    Notes on lighting drift and method choice:
        * A slow, linear brightness change of total size D over the window
          adds about D^2/12 to *every* pixel's variance, so static structure
          starts to look "noisy" and the water/non-water contrast collapses.
        * Frame differencing only sees k*D/(N-1) per pair, so it is largely
          insensitive to slow drift. BUT it also only sees *fast* change. In
          testing on synthetic video with slow, smooth ripples, consecutive
          differencing (k=1) left the water barely above sensor noise and
          mp4 block artifacts, while variance separated it cleanly. A lag of
          3-5 frames fixes most of that.
        * "detrended" subtracts the best-fit line per pixel before measuring
          variance. This removes linear drift exactly and keeps sensitivity
          to slow ripples. It is the default because it behaved best on both
          the drift and no-drift test videos. It does not remove non-linear
          lighting changes (e.g. a cloud passing and leaving); for those, use
          diff with a lag, or a shorter calm window.
    """
    n = len(frames)
    if n < 2:
        raise ValueError("Need at least 2 frames to compute temporal noise")
    diff_lag = int(np.clip(diff_lag, 1, n - 1))

    # Accumulate shifted sums (relative to the first frame) in float64 so the
    # E[x^2] - E[x]^2 formula stays numerically stable, without materialising
    # an (N, H, W) float copy of the whole stack.
    ref = _as_gray_float(frames[0], pre_blur).astype(np.float64)
    s1 = np.zeros_like(ref)
    s2 = np.zeros_like(ref)
    stx = np.zeros_like(ref)
    sdiff = np.zeros_like(ref)
    history = []  # last `diff_lag` frames, for lagged differencing
    for i in range(n):
        cur = ref if i == 0 else _as_gray_float(frames[i], pre_blur).astype(np.float64)
        d = cur - ref
        s1 += d
        s2 += d * d
        stx += i * d
        if len(history) == diff_lag:
            sdiff += np.abs(cur - history.pop(0))
        history.append(cur)

    mean = s1 / n
    variance = np.maximum(s2 / n - mean * mean, 0.0)
    # Linear detrend: residual var = var(x) - cov(t, x)^2 / var(t).
    t_mean, t_var = (n - 1) / 2.0, (n * n - 1) / 12.0
    cov = stx / n - t_mean * mean
    detrended = np.maximum(variance - cov * cov / t_var, 0.0)
    return {
        "variance": variance.astype(np.float32),
        "std": np.sqrt(variance).astype(np.float32),
        "detrended": detrended.astype(np.float32),
        "diff": (sdiff / (n - diff_lag)).astype(np.float32),
    }


def compute_noise_map(frames: np.ndarray, method: str = "detrended", pre_blur: int = 3,
                      diff_lag: int = 1) -> np.ndarray:
    """Convenience wrapper returning a single noise map (see compute_noise_maps)."""
    if method not in NOISE_METHODS:
        raise ValueError(f"method must be one of {NOISE_METHODS}")
    return compute_noise_maps(frames, pre_blur, diff_lag)[method]


@dataclass
class NormParams:
    """Parameters that map a raw noise map onto 0..255. Re-using the same
    params on another window keeps thresholds comparable across windows."""
    log: bool
    lo: float
    hi: float


def fit_normalization(noise: np.ndarray, roi_mask: np.ndarray | None = None,
                      log: bool = True, pct: tuple[float, float] = (1.0, 99.5)) -> NormParams:
    """Choose a robust stretch for a noise map (percentiles inside the ROI).

    Temporal variance is heavy-tailed (a few glare pixels can be 100x the
    median), so we stretch log1p(noise) between low/high percentiles rather
    than min/max.
    """
    vals = np.log1p(noise) if log else noise
    if roi_mask is not None:
        vals = vals[roi_mask > 0]
    lo, hi = np.percentile(vals, pct)
    if hi <= lo:
        hi = lo + 1e-6
    return NormParams(log=log, lo=float(lo), hi=float(hi))


def normalize_noise_map(noise: np.ndarray, params: NormParams | None = None,
                        post_blur: int = 5) -> np.ndarray:
    """Map a raw noise map to a uint8 image (0 = static, 255 = most dynamic).

    Args:
        noise: raw float noise map.
        params: stretch parameters (fit_normalization); fitted on `noise` if None.
        post_blur: Gaussian kernel applied to the map before thresholding.
            Water flicker is intermittent per pixel; smoothing the map
            spatially makes the water region more uniform. 0 disables.
    """
    if params is None:
        params = fit_normalization(noise)
    vals = np.log1p(noise) if params.log else noise.astype(np.float32)
    out = (vals - params.lo) / (params.hi - params.lo)
    out = np.clip(out * 255.0, 0, 255).astype(np.float32)
    if post_blur and post_blur > 1:
        k = post_blur | 1
        out = cv2.GaussianBlur(out, (k, k), 0)
    return out.astype(np.uint8)


def separability(norm_map: np.ndarray, roi_mask: np.ndarray | None = None) -> dict:
    """Score how cleanly a normalized noise map splits into two classes.

    Uses Otsu's split, then reports the Fisher ratio
    (mu_hi - mu_lo)^2 / (var_hi + var_lo). Higher = water and static structure
    are more distinct. Used to compare the variance vs diff methods.
    """
    vals = norm_map[roi_mask > 0] if roi_mask is not None else norm_map.ravel()
    t, _ = cv2.threshold(vals.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    lo, hi = vals[vals <= t].astype(np.float64), vals[vals > t].astype(np.float64)
    if lo.size == 0 or hi.size == 0:
        return {"otsu_threshold": float(t), "fisher_ratio": 0.0, "high_fraction": float(hi.size / max(vals.size, 1))}
    fisher = (hi.mean() - lo.mean()) ** 2 / (hi.var() + lo.var() + 1e-9)
    return {"otsu_threshold": float(t), "fisher_ratio": float(fisher),
            "high_fraction": float(hi.size / vals.size)}


def colorize_noise_map(norm_map: np.ndarray) -> np.ndarray:
    """False-colour version of the normalized noise map for inspection."""
    return cv2.applyColorMap(norm_map, cv2.COLORMAP_INFERNO)


# ---------------------------------------------------------------------------
# ROI handling
# ---------------------------------------------------------------------------

def parse_roi(text: str | None) -> np.ndarray | None:
    """Parse an ROI string into an (M, 2) int polygon.

    Accepted forms:
        "x1,y1,x2,y2"                 -> axis-aligned bounding box
        "x1,y1;x2,y2;x3,y3;..."       -> polygon (>= 3 vertices)
    """
    if not text:
        return None
    if ";" in text:
        pts = [tuple(float(v) for v in p.split(",")) for p in text.split(";") if p.strip()]
        if len(pts) < 3 or any(len(p) != 2 for p in pts):
            raise ValueError("Polygon ROI needs >= 3 'x,y' vertices separated by ';'")
        return np.array(pts, dtype=np.float64)
    vals = [float(v) for v in text.split(",")]
    if len(vals) != 4:
        raise ValueError("Bounding-box ROI must be 'x1,y1,x2,y2'")
    x1, y1, x2, y2 = vals
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float64)


def roi_to_mask(roi_poly: np.ndarray | None, shape: tuple[int, int], scale: float = 1.0) -> np.ndarray:
    """Rasterise an ROI polygon (in original-video pixel coords) to a uint8 mask
    at the working resolution. No ROI = whole frame."""
    h, w = shape
    if roi_poly is None:
        return np.full((h, w), 255, np.uint8)
    mask = np.zeros((h, w), np.uint8)
    pts = np.round(roi_poly * scale).astype(np.int32)
    cv2.fillPoly(mask, [pts], 255)
    if not mask.any():
        raise ValueError("ROI does not overlap the frame")
    return mask


# ---------------------------------------------------------------------------
# 3. Segmentation
# ---------------------------------------------------------------------------

def threshold_noise_map(norm_map: np.ndarray, mode: str = "otsu", value: float | None = None,
                        roi_mask: np.ndarray | None = None) -> tuple[np.ndarray, float]:
    """Binarise a normalized (uint8) noise map: 255 = water (high noise).

    Args:
        norm_map: output of normalize_noise_map (0..255).
        mode: "otsu" (automatic, computed from ROI pixels only) or "manual".
        value: threshold on the 0..255 normalized scale, required for "manual".
            Use it when glare/reflections pull the Otsu split to the wrong
            place: the CLI prints the Otsu value so you can start near it.
        roi_mask: pixels outside are forced to 0 and excluded from Otsu.

    Returns:
        (mask uint8 {0,255}, threshold used)
    """
    roi = roi_mask if roi_mask is not None else np.full(norm_map.shape, 255, np.uint8)
    if mode == "otsu":
        vals = norm_map[roi > 0].reshape(-1, 1)
        t, _ = cv2.threshold(vals, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif mode == "manual":
        if value is None:
            raise ValueError("manual threshold mode requires a threshold value")
        t = float(value)
    else:
        raise ValueError("mode must be 'otsu' or 'manual'")
    mask = np.where((norm_map > t) & (roi > 0), 255, 0).astype(np.uint8)
    return mask, float(t)


def clean_mask(mask: np.ndarray, open_size: int = 5, close_size: int = 15,
               keep_largest: bool = True, min_area: int = 0) -> np.ndarray:
    """Morphological cleanup of the water mask.

    Opening (erode->dilate) deletes small isolated "noisy" specks on static
    structure (e.g. a twinkling highlight). Closing (dilate->erode) fills small
    holes inside the water where the surface happened to be still. Then keep
    only the largest connected component (the pool), or all components larger
    than `min_area` pixels if keep_largest is False.
    """
    out = mask.copy()
    if open_size > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_size, open_size))
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, k)
    if close_size > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, k)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(out, connectivity=8)
    if n <= 1:
        return out
    areas = stats[1:, cv2.CC_STAT_AREA]
    if keep_largest:
        keep = [1 + int(np.argmax(areas))]
    else:
        keep = [1 + i for i, a in enumerate(areas) if a >= min_area]
    out = np.isin(labels, keep).astype(np.uint8) * 255
    # Fill fully enclosed holes (e.g. a floating toy) so they don't create
    # spurious internal boundaries.
    contours, _ = cv2.findContours(out, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(out)
    cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
    return filled


def component_report(raw_mask: np.ndarray, frames_max: np.ndarray, min_area: int = 50,
                     sat_level: int = 250, sat_frac: float = 0.05) -> dict:
    """Diagnostics used to flag likely glare / false-positive regions.

    Every high-noise blob larger than `min_area` is listed. A blob is flagged
    as likely sun glare / specular reflection when more than `sat_frac` of its
    pixels hit saturation (>= sat_level) at some point in the window
    (`frames_max` = per-pixel temporal max). Glare flickers, so it often never
    saturates in the *mean* image. Ordinary ripple flicker rarely saturates.
    Sparkling sun glints on the water itself can, though, so a flag on the
    main water blob is a prompt to check mask_raw.png, not proof of an error.
    """
    n, labels, stats, cents = cv2.connectedComponentsWithStats(raw_mask, connectivity=8)
    comps, glare = [], []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        sat = float(np.mean(frames_max[labels == i] >= sat_level))
        c = {"area": area, "centroid_xy": [round(float(v), 1) for v in cents[i]],
             "saturated_fraction": round(sat, 3)}
        comps.append(c)
        if sat > sat_frac:
            glare.append(c)
    comps.sort(key=lambda c: -c["area"])
    return {"n_components": len(comps), "components": comps[:10], "glare_suspects": glare}


# ---------------------------------------------------------------------------
# 4. Waterline extraction
# ---------------------------------------------------------------------------

def infer_water_side(mask: np.ndarray) -> str:
    """Guess which side of the waterline the water is on from where the water
    mask's centroid sits relative to the frame centre."""
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return "below"
    h, w = mask.shape
    dy = ys.mean() / h - 0.5
    dx = xs.mean() / w - 0.5
    # A typical dock camera looks across the water, so the waterline is
    # usually roughly horizontal; bias toward vertical sides.
    if abs(dy) >= 0.5 * abs(dx):
        return "below" if dy > 0 else "above"
    return "right" if dx > 0 else "left"


def _orient(a: np.ndarray, side: str) -> np.ndarray:
    """Transform an image so that water is 'below' the waterline."""
    if side == "below":
        return a
    if side == "above":
        return a[::-1, :]
    if side == "right":
        return a.T
    if side == "left":
        return a.T[::-1, :]
    raise ValueError(side)


def _unorient_points(scan: np.ndarray, pos: np.ndarray, side: str, shape: tuple[int, int]) -> np.ndarray:
    """Map (scan index, boundary position) from oriented space back to (x, y)."""
    h, w = shape
    if side == "below":
        x, y = scan, pos
    elif side == "above":
        x, y = scan, (h - 1) - pos
    elif side == "right":
        x, y = pos, scan
    else:  # left
        x, y = (w - 1) - pos, scan
    return np.stack([x, y], axis=1).astype(np.float64)


def _median_smooth(v: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or v.size < window:
        return v
    k = window | 1
    pad = k // 2
    padded = np.pad(v, pad, mode="edge")
    return np.median(np.lib.stride_tricks.sliding_window_view(padded, k), axis=1)


def scan_boundary(mask: np.ndarray, roi_mask: np.ndarray, side: str) -> tuple[np.ndarray, np.ndarray]:
    """Column-wise (or row-wise) scan for the water's leading edge.

    In the oriented frame (water below), for every column take the first water
    pixel from the top. The column is rejected when the pixel just above it is
    outside the ROI or the image -- then the "edge" is just where the ROI cuts
    the water, not a real water/structure boundary.

    Returns (scan_indices, boundary_positions) in the oriented frame; a
    position of NaN means no valid boundary in that column.
    """
    m = _orient(mask > 0, side)
    r = _orient(roi_mask > 0, side)
    n_rows, n_cols = m.shape
    has = m.any(axis=0)
    first = np.argmax(m, axis=0)
    pos = np.full(n_cols, np.nan)
    cols = np.arange(n_cols)
    valid = has & (first > 0)
    above_in_roi = np.zeros(n_cols, bool)
    above_in_roi[valid] = r[first[valid] - 1, cols[valid]]
    valid &= above_in_roi
    pos[valid] = first[valid]
    return cols, pos


def extract_waterline(mask: np.ndarray, roi_mask: np.ndarray | None = None,
                      method: str = "scan", water_side: str = "auto",
                      smooth: int = 9) -> tuple[np.ndarray, str]:
    """Extract the waterline as an ordered sequence of (x, y) pixel coordinates.

    Args:
        mask: cleaned binary water mask (255 = water).
        roi_mask: region of interest; boundaries that coincide with the ROI
            edge (or image border) are discarded because they are artifacts of
            cropping, not real water edges.
        method:
            "scan"    - one point per column (water above/below) or per row
                        (water left/right). Best when the camera looks across
                        the water at the dock edge so the waterline is a
                        single-valued curve. Robust and easy to compare
                        between windows.
            "contour" - cv2.findContours on the water blob, keeping the
                        boundary pixels not on the ROI/image border. Use it
                        for curved/closed shorelines (e.g. top-down view of a
                        pool corner) where the line is not single-valued.
        water_side: "auto" or which side of the line the water is on.
        smooth: median-filter window (in points) applied to the scan result
            to remove 1-2 px jaggedness. 0/1 disables. Ignored for contour.

    Returns:
        (points (K, 2) float array of x, y; resolved water_side)
    """
    if roi_mask is None:
        roi_mask = np.full(mask.shape, 255, np.uint8)
    if water_side == "auto":
        water_side = infer_water_side(mask)

    if method == "scan":
        scan, pos = scan_boundary(mask, roi_mask, water_side)
        ok = ~np.isnan(pos)
        scan, pos = scan[ok], pos[ok]
        pos = _median_smooth(pos, smooth)
        pts = _unorient_points(scan, pos, water_side, mask.shape)
        return pts, water_side

    if method == "contour":
        contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_NONE)
        if not contours:
            return np.empty((0, 2)), water_side
        c = max(contours, key=cv2.contourArea).reshape(-1, 2)
        # Pixels on the boundary of the ROI (or image) are crop edges, not
        # waterline. Erode the ROI with a 1-pixel image border removed.
        roi_inner = roi_mask.copy()
        roi_inner[0, :] = roi_inner[-1, :] = 0
        roi_inner[:, 0] = roi_inner[:, -1] = 0
        roi_inner = cv2.erode(roi_inner, np.ones((5, 5), np.uint8))
        keep = roi_inner[c[:, 1], c[:, 0]] > 0
        if not keep.any():
            return np.empty((0, 2)), water_side
        # Rotate so the longest kept run starts at index 0 (keeps the line
        # ordered/contiguous rather than wrapping around the contour start).
        if not keep.all():
            start = int(np.argmax(~keep))  # first dropped point
            c, keep = np.roll(c, -start, axis=0), np.roll(keep, -start)
        return c[keep].astype(np.float64), water_side

    raise ValueError("method must be 'scan' or 'contour'")


def waterline_profile(points: np.ndarray, side: str, shape: tuple[int, int]) -> np.ndarray:
    """Resample waterline points to one position per scan index (column for
    above/below, row for left/right); NaN where there is no point. Used to
    compare waterlines from different windows position-by-position."""
    h, w = shape
    n = w if side in ("below", "above") else h
    prof = np.full(n, np.nan)
    if points.size == 0:
        return prof
    if side in ("below", "above"):
        idx, val = points[:, 0], points[:, 1]
    else:
        idx, val = points[:, 1], points[:, 0]
    idx = np.round(idx).astype(int)
    ok = (idx >= 0) & (idx < n)
    # If several points share an index (contour method), average them.
    sums = np.bincount(idx[ok], weights=val[ok], minlength=n)
    cnts = np.bincount(idx[ok], minlength=n)
    has = cnts > 0
    prof[has] = sums[has] / cnts[has]
    return prof


# ---------------------------------------------------------------------------
# 5. Stability / validation
# ---------------------------------------------------------------------------

@dataclass
class PipelineParams:
    method: str = "detrended"
    diff_lag: int = 1
    pre_blur: int = 3
    post_blur: int = 5
    threshold_mode: str = "otsu"
    threshold_value: float | None = None
    open_size: int = 5
    close_size: int = 15
    line_method: str = "scan"
    water_side: str = "auto"
    smooth: int = 9


def detect_in_window(frames: np.ndarray, roi_mask: np.ndarray, p: PipelineParams,
                     norm: NormParams | None = None) -> dict:
    """Run noise map -> mask -> waterline on one stack of frames."""
    maps = compute_noise_maps(frames, p.pre_blur, p.diff_lag)
    raw = maps[p.method]
    if norm is None:
        norm = fit_normalization(raw, roi_mask)
    norm_map = normalize_noise_map(raw, norm, p.post_blur)
    raw_mask, t = threshold_noise_map(norm_map, p.threshold_mode, p.threshold_value, roi_mask)
    mask = clean_mask(raw_mask, p.open_size, p.close_size)
    pts, side = extract_waterline(mask, roi_mask, p.line_method, p.water_side, p.smooth)
    return {"maps": maps, "norm": norm, "norm_map": norm_map, "raw_mask": raw_mask,
            "mask": mask, "threshold": t, "points": pts, "side": side}


def auto_stability_window(n_frames: int, fps: float) -> int:
    """Default sub-window length: at least ~2 s and a third of the segment.

    A sub-window must span at least one full ripple/swell period. Otherwise a
    pixel's variance depends on which part of the wave cycle the window caught,
    the water mask goes patchy, and the line jitters for reasons that have
    nothing to do with the waterline. In synthetic tests with a ~3 s ripple
    period, 30-frame (1 s) windows gave ~2.5-5 px of jitter, and 90-frame
    windows gave ~0.5 px.
    """
    return int(min(max(2 * fps, n_frames // 3), n_frames // 2))


def waterline_stability(frames: np.ndarray, roi_mask: np.ndarray, p: PipelineParams,
                        norm: NormParams, side: str, window: int = 60,
                        stride: int | None = None) -> dict:
    """How much does the detected waterline move across the calm segment?

    A temporal noise map needs several frames, so a single frame cannot yield
    a waterline on its own. Instead the calm segment is split into sliding
    sub-windows of `window` frames, the full pipeline runs on each one with
    the baseline's water side, and we measure the spread of the line position.

    Normalization: in Otsu mode each window re-fits its own normalization.
    Shorter windows see less drift and less of the slow ripple cycle, so their
    raw noise level differs from the full segment's. Re-using the baseline
    scale made whole windows fall below threshold in drift tests. In manual
    mode the baseline scale is re-used so the threshold value means the same
    thing in every window.

    Reported stats (pixels, at working resolution):
        per_position_std_*: std of the line position at each column (or row)
            across windows, summarised by median / mean / p95 / max. On a
            calm segment with a long enough window this should be ~0.5-2 px.
        mean_position_std:  std of the line's average height across windows
            (global up/down jitter).
        coverage:           fraction of columns where every window found the
            line (low coverage = line is intermittently lost).
    """
    n = len(frames)
    window = int(min(window, n))
    stride = stride or max(1, window // 2)
    starts = list(range(0, n - window + 1, stride))
    shape = frames.shape[1:3]
    profiles, lines = [], []
    q = PipelineParams(**{**p.__dict__, "water_side": side, "line_method": "scan"})
    fixed_norm = norm if p.threshold_mode == "manual" else None
    for s in starts:
        r = detect_in_window(frames[s:s + window], roi_mask, q, fixed_norm)
        lines.append((s, s + window, r["points"]))
        profiles.append(waterline_profile(r["points"], side, shape))
    out: dict = {"window_frames": int(window), "stride_frames": int(stride), "n_windows": len(starts)}
    if len(starts) < 2:
        out["warning"] = "segment too short for >= 2 sub-windows; lower --stability-window"
        return {"stats": out, "lines": lines}
    P = np.array(profiles)
    full = ~np.isnan(P).any(axis=0)
    out["coverage"] = float(full.sum() / max(1, (~np.isnan(P)).any(axis=0).sum()))
    if not full.any():
        out["warning"] = "no column had a waterline in every sub-window"
        return {"stats": out, "lines": lines}
    per_pos = P[:, full].std(axis=0)
    means = P[:, full].mean(axis=1)
    out.update({
        "per_position_std_median": float(np.median(per_pos)),
        "per_position_std_mean": float(per_pos.mean()),
        "per_position_std_p95": float(np.percentile(per_pos, 95)),
        "per_position_std_max": float(per_pos.max()),
        "mean_position_std": float(means.std()),
        "mean_position_range": float(means.max() - means.min()),
    })
    return {"stats": out, "lines": lines}


def lighting_drift(frames: np.ndarray, static_mask: np.ndarray) -> dict:
    """Measure slow brightness change over the segment on static pixels.

    Fits a line to the mean intensity of the static (non-water) region versus
    frame index. A total change of more than a few gray levels inflates raw
    variance everywhere; frame differencing is then the safer method.
    """
    sel = static_mask > 0
    if not sel.any():
        return {"total_change": 0.0}
    means = np.array([(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if f.ndim == 3 else f)[sel].mean()
                      for f in frames])
    t = np.arange(len(means))
    slope = float(np.polyfit(t, means, 1)[0]) if len(means) > 1 else 0.0
    return {"slope_per_frame": slope, "total_change": slope * (len(means) - 1),
            "start_mean": float(means[0]), "end_mean": float(means[-1])}


def fit_straight_line(points: np.ndarray, water_side: str = "below") -> dict:
    """Fit one straight line to the detected waterline points.

    Uses cv2.fitLine with a Huber loss: points close to the line count like
    ordinary least squares, but far-off points (a glare blob, a dip where a
    reflection fooled the mask) count much less, so a few bad points don't
    tilt the line.

    Args:
        points: (K, 2) x, y waterline points.
        water_side: decides how the line is written out. For "below"/"above"
            it is y = slope * x + intercept; for "left"/"right" (a roughly
            vertical line) it is x = slope * y + intercept.

    Returns a dict with:
        form, slope, intercept: the line equation (see above).
        point, direction:   a point on the line and its unit direction.
        angle_deg:          angle from the +x axis in image coordinates
                            (y down), so positive = line falls to the right.
        endpoints:          the line clipped to the extent of the points.
        rms_residual, p95_residual, max_residual: perpendicular distances
                            of the points from the line, in pixels.
        rms_center, rms_ends: RMS residual of the middle 60% vs the outer
                            20% at each end. Ends much worse than the centre
                            is the signature of lens (barrel) distortion
                            bending a straight edge.
    """
    if len(points) < 2:
        raise ValueError("Need at least 2 points to fit a line")
    pts = np.asarray(points, dtype=np.float32)
    vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_HUBER, 0, 0.01, 0.01).ravel().astype(np.float64)
    vertical = water_side in ("left", "right")
    # Point the direction along +x (or +y for a vertical line) so the signs of
    # slope/angle/offsets are consistent between fits.
    if (vy if vertical else vx) < 0:
        vx, vy = -vx, -vy
    p0 = np.array([x0, y0])
    d = np.array([vx, vy])
    n = np.array([-vy, vx])
    rel = pts.astype(np.float64) - p0
    t = rel @ d
    resid = rel @ n
    if vertical:
        slope = vx / vy if abs(vy) > 1e-9 else float("inf")
        form, intercept = "x = slope * y + intercept", x0 - slope * y0
    else:
        slope = vy / vx if abs(vx) > 1e-9 else float("inf")
        form, intercept = "y = slope * x + intercept", y0 - slope * x0
    lo, hi = np.percentile(t, [20, 80])
    center, ends = resid[(t >= lo) & (t <= hi)], resid[(t < lo) | (t > hi)]

    def rms(r: np.ndarray) -> float:
        return float(np.sqrt(np.mean(r ** 2))) if r.size else float("nan")

    return {
        "form": form,
        "slope": float(slope),
        "intercept": float(intercept),
        "point": [float(x0), float(y0)],
        "direction": [float(vx), float(vy)],
        "angle_deg": float(np.degrees(np.arctan2(vy, vx))),
        "endpoints": [(p0 + t.min() * d).tolist(), (p0 + t.max() * d).tolist()],
        "rms_residual": rms(resid),
        "p95_residual": float(np.percentile(np.abs(resid), 95)),
        "max_residual": float(np.abs(resid).max()),
        "rms_center": rms(center),
        "rms_ends": rms(ends),
    }


def line_fit_stability(window_lines: list, baseline_fit: dict, water_side: str,
                       scale: float = 1.0) -> dict:
    """Fit a straight line to each stability sub-window's waterline and report
    how much the fitted line tilts and shifts between windows.

    angle_std_deg: spread of the line's angle across windows.
    offset_std_px: spread of each window line's perpendicular distance from
        the baseline line's midpoint (i.e. how much it moves up/down).
    """
    mid = np.mean(np.array(baseline_fit["endpoints"]), axis=0)
    angles, offsets = [], []
    for _, _, lp in window_lines:
        if len(lp) < 2:
            continue
        f = fit_straight_line(lp / scale, water_side)
        vx, vy = f["direction"]
        angles.append(f["angle_deg"])
        offsets.append(float((mid - np.array(f["point"])) @ np.array([-vy, vx])))
    if len(angles) < 2:
        return {"warning": "fewer than 2 sub-window fits"}
    return {"n_windows": len(angles), "angle_std_deg": float(np.std(angles)),
            "offset_std_px": float(np.std(offsets)),
            "offset_range_px": float(np.max(offsets) - np.min(offsets))}


def draw_fitted_line(image: np.ndarray, fit: dict, color=(255, 0, 255), thickness: int = 1) -> np.ndarray:
    """Draw a fitted straight line (between its endpoints) on a copy of image."""
    out = image.copy()
    if out.ndim == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    a, b = (tuple(int(round(v)) for v in p) for p in fit["endpoints"])
    cv2.line(out, a, b, color, thickness, cv2.LINE_AA)
    return out


def draw_waterline(image: np.ndarray, points: np.ndarray, color=(0, 255, 0), thickness: int = 2,
                   roi_poly: np.ndarray | None = None, closed: bool = False) -> np.ndarray:
    """Draw a waterline polyline (and optional ROI outline) on a copy of image.
    Breaks the polyline wherever consecutive points are > 3 px apart so gaps
    (columns with no detection) are not bridged by straight segments."""
    out = image.copy()
    if out.ndim == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    if roi_poly is not None:
        cv2.polylines(out, [np.round(roi_poly).astype(np.int32)], True, (255, 200, 0), 1, cv2.LINE_AA)
    if len(points) >= 2:
        p = np.round(points).astype(np.int32)
        gaps = np.where(np.abs(np.diff(p, axis=0)).max(axis=1) > 3)[0] + 1
        for seg in np.split(p, gaps):
            if len(seg) >= 2:
                cv2.polylines(out, [seg.reshape(-1, 1, 2)], closed, color, thickness, cv2.LINE_AA)
    elif len(points) == 1:
        cv2.circle(out, tuple(np.round(points[0]).astype(int)), thickness, color, -1)
    return out


def write_overlay_clip(video_path: str | Path, out_path: str | Path, start: int, end: int,
                       baseline: np.ndarray, roi_poly: np.ndarray | None,
                       window_lines: list | None = None, scale: float = 1.0,
                       fps: float = 30.0, line_fit: dict | None = None) -> None:
    """Write an MP4 with the baseline waterline (green) drawn on each frame of
    [start, end), plus the sub-window waterline covering that frame (yellow)
    when stability windows are given, and the fitted straight line (magenta)
    when `line_fit` is given. Coordinates are in original-video pixels.

    Frames past the calm segment can be included (pass a larger `end`) to see
    where the baseline sits once the dog is in the water.
    """
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    writer = None
    idx = start
    try:
        while idx < end:
            ok, frame = cap.read()
            if not ok:
                break
            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            vis = frame
            if window_lines:
                for s, e, pts in window_lines:
                    if s <= idx < e:
                        vis = draw_waterline(vis, pts / scale, (0, 255, 255), 1)
                        break
            vis = draw_waterline(vis, baseline, (0, 255, 0), 2, roi_poly)
            if line_fit is not None:
                vis = draw_fitted_line(vis, line_fit)
            cv2.putText(vis, f"frame {idx}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 2, cv2.LINE_AA)
            writer.write(vis)
            idx += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()


# ---------------------------------------------------------------------------
# 6. Output
# ---------------------------------------------------------------------------

def _jsonable(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if hasattr(o, "__dict__"):
        return o.__dict__
    raise TypeError(type(o))


def save_outputs(out_dir: Path, points: np.ndarray, meta: dict) -> None:
    """Save waterline coordinates (JSON + .npy) and run metadata/stats."""
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "waterline.npy", points.astype(np.float32))
    payload = {"points_xy": np.round(points, 2).tolist(), **meta}
    with open(out_dir / "waterline.json", "w") as f:
        json.dump(payload, f, indent=2, default=_jsonable)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Detect the waterline from per-pixel temporal noise in a stationary-camera video.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--video", required=True, help="input video (.mp4)")
    p.add_argument("--output-dir", default="out", help="directory for results")

    g = p.add_argument_group("frame range (use a calm segment, before any jump)")
    g.add_argument("--start-frame", type=int, help="first frame (inclusive)")
    g.add_argument("--end-frame", type=int, help="last frame (exclusive)")
    g.add_argument("--start-time", type=float, help="start in seconds (if --start-frame not set)")
    g.add_argument("--end-time", type=float, help="end in seconds (if --end-frame not set)")
    g.add_argument("--step", type=int, default=1, help="use every Nth frame")
    g.add_argument("--scale", type=float, default=1.0,
                   help="resize factor for processing (output coords are in original pixels)")
    g.add_argument("--color", action="store_true",
                   help="load colour frames (noise is still measured on intensity; "
                        "colour is used for overlays)")

    g = p.add_argument_group("noise map")
    g.add_argument("--method", choices=NOISE_METHODS, default="detrended",
                   help="temporal statistic used for segmentation (all are saved for comparison)")
    g.add_argument("--diff-lag", type=int, default=1,
                   help="frame gap k for --method diff: mean |f[t] - f[t-k]|")
    g.add_argument("--pre-blur", type=int, default=3, help="per-frame Gaussian blur kernel (0=off)")
    g.add_argument("--post-blur", type=int, default=5, help="noise-map Gaussian blur kernel (0=off)")

    g = p.add_argument_group("segmentation")
    g.add_argument("--roi", help="'x1,y1,x2,y2' box or 'x1,y1;x2,y2;x3,y3;...' polygon, "
                                 "in original video pixels")
    g.add_argument("--threshold-mode", choices=("otsu", "manual"), default="otsu")
    g.add_argument("--threshold-value", type=float,
                   help="manual threshold on the 0-255 normalized noise map")
    g.add_argument("--open-size", type=int, default=5, help="morphological opening kernel")
    g.add_argument("--close-size", type=int, default=15, help="morphological closing kernel")

    g = p.add_argument_group("waterline")
    g.add_argument("--line-method", choices=("scan", "contour"), default="scan")
    g.add_argument("--water-side", choices=WATER_SIDES, default="auto",
                   help="side of the waterline the water is on")
    g.add_argument("--smooth", type=int, default=9, help="median smoothing window (scan method)")
    g.add_argument("--fit", choices=("none", "line"), default="none",
                   help="also fit a single straight line to the detected waterline")

    g = p.add_argument_group("validation")
    g.add_argument("--stability-window", type=int,
                   help="frames per sub-window for the stability check; should span a full "
                        "ripple period (default: max(2 s, segment/3)); 0 = skip")
    g.add_argument("--stability-stride", type=int, help="sub-window stride (default window/2)")
    g.add_argument("--clip", action="store_true", help="write an overlay MP4 of the segment")
    g.add_argument("--clip-extra", type=int, default=0,
                   help="also render this many frames after the segment in the clip "
                        "(to see the baseline once motion starts)")
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.threshold_mode == "manual" and args.threshold_value is None:
        parser.error("--threshold-mode manual requires --threshold-value")
    t0 = time.time()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    info = probe_video(args.video)
    start, end = resolve_frame_range(info, args.start_frame, args.end_frame,
                                     args.start_time, args.end_time)
    print(f"Video: {info.width}x{info.height} @ {info.fps:.2f} fps, {info.frame_count} frames")
    print(f"Calm segment: frames [{start}, {end}) = {(end - start) / info.fps:.2f} s, "
          f"step={args.step}, scale={args.scale}")

    frames = load_frames(args.video, start, end, args.color, args.step, args.scale)
    if len(frames) < 10:
        print(f"WARNING: only {len(frames)} frames; temporal statistics will be noisy.")
    shape = frames.shape[1:3]
    roi_poly = parse_roi(args.roi)
    roi_mask = roi_to_mask(roi_poly, shape, args.scale)

    params = PipelineParams(
        method=args.method, diff_lag=args.diff_lag, pre_blur=args.pre_blur, post_blur=args.post_blur,
        threshold_mode=args.threshold_mode, threshold_value=args.threshold_value,
        open_size=args.open_size, close_size=args.close_size,
        line_method=args.line_method, water_side=args.water_side, smooth=args.smooth)

    # --- 2/3/4: baseline detection on the full calm segment ---------------
    res = detect_in_window(frames, roi_mask, params)
    maps = res["maps"]
    print(f"Threshold ({args.threshold_mode}): {res['threshold']:.1f} / 255 on '{args.method}' map")

    # Compare all methods: save each normalized map + separability score.
    comparison = {}
    panels = []
    for m in NOISE_METHODS:
        nm = normalize_noise_map(maps[m], fit_normalization(maps[m], roi_mask), args.post_blur)
        cv2.imwrite(str(out_dir / f"noise_map_{m}.png"), nm)
        cv2.imwrite(str(out_dir / f"noise_map_{m}_color.png"), colorize_noise_map(nm))
        comparison[m] = separability(nm, roi_mask)
        panel = colorize_noise_map(nm)
        cv2.putText(panel, f"{m}  fisher={comparison[m]['fisher_ratio']:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        panels.append(panel)
    grid = np.vstack([np.hstack(panels[0:2]), np.hstack(panels[2:4])])
    cv2.imwrite(str(out_dir / "noise_map_comparison.png"), grid)
    print("Method separability (Fisher ratio, higher = cleaner water/static split):")
    for m, c in comparison.items():
        print(f"  {m:9s} fisher={c['fisher_ratio']:6.2f}  otsu={c['otsu_threshold']:.0f}  "
              f"water_fraction={c['high_fraction']:.2f}"
              + ("   <- split isolated a small region (glare?), not the water" if c["high_fraction"] < 0.1 else ""))

    cv2.imwrite(str(out_dir / "mask_raw.png"), res["raw_mask"])
    cv2.imwrite(str(out_dir / "mask_clean.png"), res["mask"])

    pts = res["points"]
    side = res["side"]
    if len(pts) == 0:
        print("ERROR: no waterline found. Check the ROI, try --threshold-mode manual, "
              "or inspect noise_map_*.png / mask_*.png.", file=sys.stderr)
        return 1
    pts_orig = pts / args.scale
    print(f"Waterline: {len(pts)} points, water side = {side}, method = {args.line_method}")

    # --- Diagnostics: glare and lighting drift ------------------------------
    gray_max = frames.max(axis=0)
    if gray_max.ndim == 3:
        gray_max = gray_max.max(axis=2)
    comp = component_report(res["raw_mask"], gray_max)
    warnings = []
    if comp["n_components"] > 1:
        warnings.append(
            f"{comp['n_components']} separate high-noise regions in the raw mask "
            f"(areas {[c['area'] for c in comp['components'][:5]]}). Only the largest is kept; the "
            "others may be glare, foliage or reflections -- tighten --roi if they touch the waterline.")
    for c in comp["glare_suspects"]:
        warnings.append(
            f"High-noise region at x,y={c['centroid_xy']} (area {c['area']} px) is "
            f"{c['saturated_fraction']:.0%} saturated at times: likely sun glare / specular reflection, "
            "not water. Exclude it with --roi or raise --threshold-value if it affects the line.")
    static = cv2.bitwise_and(cv2.bitwise_not(res["mask"]), roi_mask)
    drift = lighting_drift(frames, static)
    if abs(drift["total_change"]) > 3.0:
        warnings.append(
            f"Lighting drift of {drift['total_change']:+.1f} gray levels on static pixels over the "
            "segment. Raw variance is inflated by drift (~D^2/12); use --method detrended or diff.")
    print(f"Lighting drift on static pixels: {drift['total_change']:+.2f} gray levels over segment")

    # --- Optional straight-line fit -----------------------------------------
    line_fit = None
    if args.fit == "line":
        line_fit = fit_straight_line(pts_orig, side)
        lhs, rhs = ("y", "x") if side in ("below", "above") else ("x", "y")
        print(f"Line fit: {lhs} = {line_fit['slope']:.4f} * {rhs} + {line_fit['intercept']:.2f} "
              f"(angle {line_fit['angle_deg']:+.2f} deg); residual RMS={line_fit['rms_residual']:.2f} px, "
              f"p95={line_fit['p95_residual']:.2f}, max={line_fit['max_residual']:.2f} "
              f"[centre {line_fit['rms_center']:.2f} / ends {line_fit['rms_ends']:.2f}]")
        if line_fit["rms_residual"] > 3.0:
            if line_fit["rms_ends"] > 2 * line_fit["rms_center"]:
                hint = ("Residuals are much larger near the ends than in the centre, which suggests "
                        "lens distortion bending a straight edge; use a narrower --roi or undistort first.")
            else:
                hint = ("The detected edge is not straight (curved shoreline or detection errors); "
                        "check overlay.png before trusting the fitted line.")
            warnings.append(f"Straight-line fit is poor (RMS {line_fit['rms_residual']:.1f} px). {hint}")

    # --- 5: stability across the calm segment --------------------------------
    stab = {"stats": {}, "lines": []}
    win = args.stability_window
    if win is None:
        win = auto_stability_window(len(frames), info.fps / args.step)
    if win > 1:
        stab = waterline_stability(frames, roi_mask, params, res["norm"], side,
                                   win, args.stability_stride)
        s = stab["stats"]
        if "per_position_std_median" in s:
            k = 1.0 / args.scale
            print(f"Stability over {s['n_windows']} windows of {s['window_frames']} frames "
                  f"(original px): per-column std median={s['per_position_std_median'] * k:.2f}, "
                  f"p95={s['per_position_std_p95'] * k:.2f}, max={s['per_position_std_max'] * k:.2f}; "
                  f"mean-height std={s['mean_position_std'] * k:.2f}; coverage={s['coverage']:.0%}")
            if s["per_position_std_median"] * k > 3.0:
                warnings.append("Waterline moves > 3 px between sub-windows: the segment may not be "
                                "calm, the sub-window may be shorter than the ripple period (try a "
                                "larger --stability-window), or the threshold is unstable "
                                "(see stability_lines.png).")
        else:
            print(f"Stability: {s.get('warning')}")
        if line_fit is not None and stab["lines"]:
            lf = line_fit_stability(stab["lines"], line_fit, side, args.scale)
            stab["stats"]["line_fit"] = lf
            if "angle_std_deg" in lf:
                print(f"Line-fit stability over {lf['n_windows']} windows: angle std="
                      f"{lf['angle_std_deg']:.3f} deg, offset std={lf['offset_std_px']:.2f} px")

    # --- Visualisation ------------------------------------------------------
    sample_idx = (end - start) // 2 + start
    sample = load_frames(args.video, sample_idx, sample_idx + 1, color=True)[0]
    overlay = draw_waterline(sample, pts_orig, (0, 255, 0), 2, roi_poly)
    if line_fit is not None:
        overlay = draw_fitted_line(overlay, line_fit)
    cv2.imwrite(str(out_dir / "overlay.png"), overlay)
    if stab["lines"]:
        colors = [tuple(int(c) for c in cv2.applyColorMap(np.uint8([[int(255 * i / max(1, len(stab['lines']) - 1))]]),
                                                         cv2.COLORMAP_JET)[0, 0])
                  for i in range(len(stab["lines"]))]
        sl = sample.copy()
        for (s0, e0, lp), c in zip(stab["lines"], colors):
            sl = draw_waterline(sl, lp / args.scale, c, 1)
        cv2.imwrite(str(out_dir / "stability_lines.png"), sl)
    if args.clip:
        write_overlay_clip(args.video, out_dir / "overlay_clip.mp4", start, end + args.clip_extra,
                           pts_orig, roi_poly,
                           [(start + s0 * args.step, start + e0 * args.step, lp)
                            for s0, e0, lp in stab["lines"]],
                           args.scale, info.fps, line_fit)

    for w in warnings:
        print("WARNING:", w)

    # --- 6: save -------------------------------------------------------------
    meta = {
        "video": info.__dict__,
        "frame_range": [start, end],
        "params": params.__dict__ | {"scale": args.scale, "step": args.step, "roi": args.roi},
        "water_side": side,
        "threshold_used": res["threshold"],
        "method_comparison": comparison,
        "line_fit": line_fit,
        "stability": stab["stats"],
        "lighting_drift": drift,
        "glare_check": comp,
        "warnings": warnings,
        "coordinate_space": "original video pixels (x right, y down)",
    }
    save_outputs(out_dir, pts_orig, meta)
    print(f"Saved results to {out_dir.resolve()} in {time.time() - t0:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
