#!/usr/bin/env python3
"""
Lightweight waterline detector for a stationary camera.

Same idea as detect_waterline.py, with only the essentials kept:
water flickers over time, static structure doesn't. We measure how much each
pixel's brightness wobbles over a short calm clip, threshold that "noise map"
into water / not-water, take the top edge of the water in each column, and
fit a straight line to it.

Assumes the water is BELOW the waterline in the image (camera looking across
the pool at the far wall). Use a calm clip with nothing moving in front of
the waterline.

Usage:
    python detect_waterline_lite.py --video clip.mp4 --end-frame 30 \
        --roi 0,200,1920,800 --output-dir out_lite/
    python detect_waterline_lite.py --video clip.mp4 --start-time 12.5 --end-time 13.5 \
        --roi 0,200,1920,800 --output-dir out_lite/

Outputs (in --output-dir):
    waterline.json  - detected points, inlier flags, fitted line, parameters
    overlay.png     - points (green = used, red = rejected) + fitted line (magenta)
    noise_map.png   - bright = more flicker over time
    mask.png        - pixels classed as water
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

# Tuning constants. These rarely need changing; edit here if they do.
PRE_BLUR = 3        # blur each frame first to suppress per-pixel sensor noise
POST_BLUR = 5       # blur the noise map so the water region is smooth
OPEN_SIZE = 5       # remove bright specks smaller than this (px)
CLOSE_SIZE = 15     # fill dark holes in the water smaller than this (px)
SMOOTH = 9          # median filter over neighbouring columns (px)
OUTLIER_PX = 3.0    # minimum distance (px) from the line before a point can be rejected


def load_frames(video_path, start_frame=0, end_frame=None):
    """Read frames [start_frame, end_frame) as an (N, H, W) uint8 grayscale stack."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frames, idx = [], start_frame
    while end_frame is None or idx < end_frame:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        idx += 1
    cap.release()
    if len(frames) < 2:
        raise ValueError(f"Need at least 2 frames, got {len(frames)}")
    return np.stack(frames)


def noise_map(frames):
    """Per-pixel temporal variance after removing each pixel's linear trend.

    Removing the trend means slow lighting changes (clouds, sun) are ignored;
    only the back-and-forth flicker of the water counts.
    Returns a float32 (H, W) map.
    """
    n = len(frames)
    t = np.arange(n, dtype=np.float64) - (n - 1) / 2.0     # centred time
    s1 = s2 = st = 0.0
    for i, f in enumerate(frames):
        x = cv2.GaussianBlur(f.astype(np.float32), (PRE_BLUR, PRE_BLUR), 0).astype(np.float64)
        s1 = s1 + x
        s2 = s2 + x * x
        st = st + t[i] * x
    mean = s1 / n
    var = s2 / n - mean * mean
    slope = st / np.sum(t * t)                              # per-pixel brightness trend
    detrended = var - slope * slope * np.mean(t * t)
    return np.maximum(detrended, 0).astype(np.float32)


def to_uint8(noise, roi):
    """Stretch the noise map to 0-255 using percentiles inside the ROI (log scale,
    because a few very noisy pixels would otherwise squash everything else)."""
    v = np.log1p(noise)
    lo, hi = np.percentile(v[roi > 0], (1, 99.5))
    out = np.clip((v - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
    return cv2.GaussianBlur(out, (POST_BLUR, POST_BLUR), 0)


def roi_mask(shape, roi):
    """uint8 mask, 255 inside the 'x1,y1,x2,y2' box (whole frame if roi is None)."""
    m = np.zeros(shape, np.uint8)
    if roi is None:
        m[:] = 255
        return m
    x1, y1, x2, y2 = (int(v) for v in roi.split(","))
    m[min(y1, y2):max(y1, y2), min(x1, x2):max(x1, x2)] = 255
    return m


def water_mask(norm, roi, threshold=None):
    """Threshold (Otsu unless given), clean up, keep the largest water blob."""
    if threshold is None:
        threshold, _ = cv2.threshold(norm[roi > 0].reshape(-1, 1), 0, 255,
                                     cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = ((norm > threshold) & (roi > 0)).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (OPEN_SIZE, OPEN_SIZE))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (CLOSE_SIZE, CLOSE_SIZE))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if n > 1:
        mask = (labels == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])).astype(np.uint8) * 255
    return mask, float(threshold)


def scan_waterline(mask, roi):
    """For each column, the first water pixel from the top = the waterline.

    Columns where the pixel just above is outside the ROI are skipped: there
    the 'edge' is just the ROI box cutting the water, not a real waterline.
    Returns (K, 2) float array of (x, y).
    """
    water = mask > 0
    has = water.any(axis=0)
    y = np.argmax(water, axis=0)
    x = np.arange(mask.shape[1])
    ok = has & (y > 0)
    ok[ok] = roi[y[ok] - 1, x[ok]] > 0
    x, y = x[ok], y[ok].astype(np.float64)
    if len(y) >= SMOOTH:
        pad = np.pad(y, SMOOTH // 2, mode="edge")
        y = np.median(np.lib.stride_tricks.sliding_window_view(pad, SMOOTH), axis=1)
    return np.stack([x, y], axis=1).astype(np.float64)


def fit_line(points, iterations=3):
    """Robust straight-line fit y = slope * x + intercept.

    Fit, reject points far from the line, refit. "Far" is
    max(OUTLIER_PX, 3 x the typical residual), so the cut-off adapts to how
    noisy the edge is. Handles stretches where something moving behind the
    wall (people, a dog) got merged into the water and pulled the edge up.
    Returns (fit dict, inlier boolean array).
    """
    inlier = np.ones(len(points), bool)
    for _ in range(iterations):
        vx, vy, x0, y0 = cv2.fitLine(points[inlier].astype(np.float32),
                                     cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
        slope = vy / vx
        intercept = y0 - slope * x0
        resid = np.abs(points[:, 1] - (slope * points[:, 0] + intercept))
        mad = np.median(resid[inlier]) * 1.4826           # robust std estimate
        new_inlier = resid <= max(OUTLIER_PX, 3 * mad)
        if new_inlier.sum() < 2:
            break
        inlier = new_inlier
    r = resid[inlier]
    x_min, x_max = points[inlier, 0].min(), points[inlier, 0].max()
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "angle_deg": float(np.degrees(np.arctan(slope))),
        "endpoints": [[float(x_min), float(slope * x_min + intercept)],
                      [float(x_max), float(slope * x_max + intercept)]],
        "rms_residual_px": float(np.sqrt(np.mean(r ** 2))),
        "inlier_fraction": float(inlier.mean()),
    }, inlier


def draw(image, points, inlier, fit, roi):
    out = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image.copy()
    if roi is not None:
        x1, y1, x2, y2 = (int(v) for v in roi.split(","))
        cv2.rectangle(out, (x1, y1), (x2, y2), (255, 200, 0), 1)
    for (x, y), ok in zip(points.astype(int), inlier):
        out[max(y - 1, 0):y + 2, x] = (0, 255, 0) if ok else (0, 0, 255)
    a, b = (tuple(int(round(v)) for v in p) for p in fit["endpoints"])
    cv2.line(out, a, b, (255, 0, 255), 2, cv2.LINE_AA)
    return out


def detect(video, start_frame=0, end_frame=None, roi=None, threshold=None):
    """Run the whole pipeline. Returns a dict with points, inliers, fit and images."""
    frames = load_frames(video, start_frame, end_frame)
    roi_m = roi_mask(frames.shape[1:], roi)
    norm = to_uint8(noise_map(frames), roi_m)
    mask, used_threshold = water_mask(norm, roi_m, threshold)
    points = scan_waterline(mask, roi_m)
    if len(points) < 2:
        raise RuntimeError("No waterline found: check --roi, or try a --threshold value "
                           "(see noise_map.png and mask.png).")
    fit, inlier = fit_line(points)
    return {"frames": frames, "noise_map": norm, "mask": mask, "threshold": used_threshold,
            "points": points, "inlier": inlier, "fit": fit}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--video", required=True, help="input video")
    start = p.add_mutually_exclusive_group()
    start.add_argument("--start-frame", type=int, help="first frame (inclusive); default = 0")
    start.add_argument("--start-time", type=float, help="start in seconds (instead of --start-frame)")
    end = p.add_mutually_exclusive_group()
    end.add_argument("--end-frame", type=int, help="last frame (exclusive); default = end of video")
    end.add_argument("--end-time", type=float, help="end in seconds (instead of --end-frame)")
    p.add_argument("--roi", help="only search inside this box: x1,y1,x2,y2")
    p.add_argument("--threshold", type=float,
                   help="water cut-off on the 0-255 noise map (default: automatic Otsu)")
    p.add_argument("--output-dir", default="out_lite")
    a = p.parse_args(argv)

    # Convert seconds to frame numbers using the video's frame rate.
    fps = cv2.VideoCapture(str(a.video)).get(cv2.CAP_PROP_FPS) or 30.0
    if a.start_time is not None:
        a.start_frame = int(round(a.start_time * fps))
    if a.end_time is not None:
        a.end_frame = int(round(a.end_time * fps))
    a.start_frame = a.start_frame or 0

    r = detect(a.video, a.start_frame, a.end_frame, a.roi, a.threshold)
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    mid = r["frames"][len(r["frames"]) // 2]
    cv2.imwrite(str(out / "overlay.png"), draw(mid, r["points"], r["inlier"], r["fit"], a.roi))
    cv2.imwrite(str(out / "noise_map.png"), r["noise_map"])
    cv2.imwrite(str(out / "mask.png"), r["mask"])
    with open(out / "waterline.json", "w") as f:
        json.dump({
            "video": str(a.video),
            "frame_range": [a.start_frame, a.start_frame + len(r["frames"])],
            "time_range_s": [round(a.start_frame / fps, 3),
                             round((a.start_frame + len(r["frames"])) / fps, 3)],
            "fps": fps,
            "image_size": [int(r["frames"].shape[2]), int(r["frames"].shape[1])],  # [width, height]
            "roi": a.roi,
            "threshold": r["threshold"],
            "line": r["fit"],
            "points_xy": np.round(r["points"], 2).tolist(),
            "inlier": r["inlier"].tolist(),
        }, f, indent=1)

    fit = r["fit"]
    n_end = a.start_frame + len(r["frames"])
    print(f"Frames {a.start_frame}-{n_end} ({a.start_frame / fps:.2f}-{n_end / fps:.2f} s at "
          f"{fps:.2f} fps), threshold {r['threshold']:.0f}/255")
    print(f"Waterline: {len(r['points'])} points, {fit['inlier_fraction']:.0%} used for the line")
    print(f"Line: y = {fit['slope']:.5f} * x + {fit['intercept']:.2f}  "
          f"(angle {fit['angle_deg']:+.2f} deg, RMS {fit['rms_residual_px']:.2f} px)")
    if fit["inlier_fraction"] < 0.7:
        print("WARNING: over 30% of points were rejected; check overlay.png "
              "(red = rejected). Tighten --roi or pick a calmer clip.")
    print(f"Saved to {out.resolve()}")


if __name__ == "__main__":
    main()
