#!/usr/bin/env python3
"""
Estimate the water plane in the left camera's coordinates from a detected
waterline (waterline.json from detect_waterline_lite.py) and the ZED
recording it came from.

The plane is written as   n . X = d
    n  unit vector pointing DOWN (gravity), in the camera frame
    d  camera height above the water, metres
    X  any 3D point in the camera frame (x right, y down, z forward, metres)

Method:
  1. Pick sample columns along the waterline (inlier points only).
  2. In each column, sample the wall N pixels ABOVE the line, where stereo
     depth is reliable (b/c depth on the water itself is not).
  3. Read the 3D point there in every frame of the clip; the SDK's confidence
     threshold turns unreliable pixels into NaN, which are skipped. Take the
     per-pixel median over frames.
  4. Move each point down onto the waterline: keep the wall depth Z, but use
     the waterline pixel's own ray (X = (u-cx)Z/fx, Y = (v-cy)Z/fy). This
     removes the "sampled point is N px above the water" error.
  5. Gravity: average the accelerometer, flip it (it reads up at rest),
     rotate into the camera frame -> n.
  6. d = median over points of n . p. The spread of n . p is the quality check.

Optional --origin-pixel u,v: a marked point (e.g. on the pool wall in line with
the dock end). Its 3D position is read the same way, and the output adds its
height above the water and the point directly below it on the water.

Until zed_real.py exists, run with --stub (uses zed_stub.StubCamera):
    python zed_stub.py --write-waterline out_stub/waterline.json
    python estimate_water_plane.py --waterline out_stub/waterline.json --stub
"""

import argparse             # CLI library
import json                 # JSON library
import sys                  # 
import warnings             #
from pathlib import Path    #

import numpy as np          # it's numpy

# returns dictionary from waterline json
def load_waterline(path) -> dict:
    with open(path) as f:
        return json.load(f)

# Given the json pixels of interest, desired number of samples, and a pixel offset
# this function will return a set of points evenly spaced throughout the samples of interest
# the points will be generated using the line best fit gained from the detect_waterline.py script
# it also returns a second set of points which assumes a pixel offset from the detected waterline
def sample_pixels(waterline: dict, n_samples: int, pixels_above: int):
    """Evenly spaced columns over the inlier waterline points.

    Returns (line_px, wall_px): (M, 2) float pixels on the fitted line, and
    (M, 2) int pixels N rows above them (on the wall)."""
    line = waterline["line"]                            # get line best fit from json (dict)
    pts = np.asarray(waterline["points_xy"], float)     # construct array from waterline points detected (from json)
    inlier = np.asarray(waterline["inlier"], bool)      # True/False mask if point was inlier of line best fit waterline
    us = np.unique(pts[inlier, 0])                      # find which columns (px) are going to be used (get columns from points via mask)
    if len(us) == 0:
        raise ValueError("waterline.json has no inlier points")
    # len(us)      --> how many inlier points we have
    # n_samples    --> how many desired samples we want (# samples is min of len(us) and n_samples)
    # linspace(start, stop, count) --> gives count evenly spaced numbers from start to stop
    # linespace    --> min() samples from 0 to last of valid
    # np.round()   --> positions must be whole numbers (linespace can return evenly spaced points which are not whole)
    # .astype(int) --> converts decimal type to integer type
    # np.unique()  --> theoretically shouldnt matter if you have significant number of column samples where col >> n_samples
    #                  ^^^Just checks for unique points in case of weird case of too many samples whith not enough columns available
    # us = us[...] --> replace array with all column nums with evenly spaced sample locations
    us = us[np.unique(np.round(np.linspace(0, len(us) - 1, min(n_samples, len(us)))).astype(int))]
    vs = line["slope"] * us + line["intercept"] # get veritcal positions (y) for each column sample using line best fit
    line_px = np.stack([us, vs], axis=1) # combine points into pairs together into 1 array
    wall_px = np.stack([us, np.round(vs - pixels_above)], axis=1).astype(int) # move vert. position up in pixels then round to int type
    return line_px, wall_px # returns the line pixels as well as the pixels on the wall to be sampled

# cam      --> the camera object
# pixels   --> table of [u,v] pixels to measure
# frames   --> which frames to read from (e.g. range(0,60) --> frames 0 to 59)
# this function gathers 3d points for each pixel of interest, and also samples IMU from camera for gravity vector
def read_points(cam, pixels: np.ndarray, frames) -> tuple[np.ndarray, np.ndarray]:
    """3D points at `pixels` for every frame, plus one accelerometer sample per
    frame. Returns ((F, K, 3) points with NaN for invalid, (F, 3) accel)."""
    pts, accel = [], [] # lists for points and acceleromneter readings for each frame
    for i in frames:
        cam.set_frame(i)                        # jump to frame i in video (ZED SDK - set_svo_position)
        if not cam.grab():                      # was frame grabbed?
            break
        pts.append(cam.get_xyz(pixels))         # get 3d position from pixel locations (camera object has reference to its own current frame)
        accel.append(cam.get_accelerometer())   # get accelerometer reading for frame (during frame)
    if not pts: #if no frames were gathered (ERROR)
        raise RuntimeError("No frames could be read from the recording")
    return np.array(pts), np.array(accel)       # return the array of 3d points, as well as array of gravity vector samples

# med -->   array with median measurements (x,y,z) for each pixel
# min_valid --> threshold for amount of valid measurements per pixel
def median_over_frames(stack: np.ndarray, min_valid: float) -> tuple[np.ndarray, np.ndarray]:
    """Per-pixel median over frames, ignoring NaN. Pixels valid in fewer than
    `min_valid` (fraction) of frames become NaN. Returns (points, valid counts)."""
    valid = ~np.isnan(stack).any(axis=2)    # marks each number as True where NaN, converts each pixel measurement as T/F, flips with ~ (now a 2d mask for each frames pixel measurement on whether valid or not)
    counts = valid.sum(axis=0)              # count number of measurements for a pixel per frame (e.g. pixel 0 has 4 valid frames) e.g. counts [3, 1] <-- pixel 0 valid in 3 frames, pixel 1 in only 1
    # if pixel is NaN for every frame, nanmedian returns NaN. this block silence warnings (keep an eye on this though, could lead to silent failures?)
    with warnings.catch_warnings():         
        warnings.simplefilter("ignore", RuntimeWarning)   # all-NaN pixels -> NaN, expected
        med = np.nanmedian(stack, axis=0)   # for each pixel and each x,y,z take median across frames (ignoring NaN)
    med[counts < max(1, min_valid * len(stack))] = np.nan # marks each pixel with not enough points as invalud (NaN)
    return med, counts

# Notes on above
# len(stack) --> number of frames (1st dimmensions)
# min_valid * len(stack) --> min number of valid frames
# max(1, ...) makes sure min is at least 1 frame
# counts < ... --> ex. [False, True]: pixel 1 doesn't have enough
# med[[False, True]] = np.nan --> if pixel doesnt have enough samples then set median coordinates to NaN

# Need to verify this section
# this accounts for the fact that the original pixels were measured some distance above the actual waterline
# so it attempts to reduce the introduced error
# water_pts --> table of measured wall pts [x,y,z]
# line_px --> table of matching waterline pixels (row i gies with row i of wall_pts)
# cam_info --> cam_info.cx reads its cx value
def project_to_waterline(wall_pts: np.ndarray, line_px: np.ndarray, cam_info) -> np.ndarray:
    """Place each point on its waterline pixel's ray at the wall's depth Z."""
    z = wall_pts[:, 2]                                      # Z = depth: distance straight ahdead in meters
    x = (line_px[:, 0] - cam_info.cx) * z / cam_info.fx     # u, v = pixel (column, row)
    y = (line_px[:, 1] - cam_info.cy) * z / cam_info.fy     # cx, cy = image center in pixels | fx, fy = focal length, in pixels (how zoomed in)
    return np.stack([x, y, z], axis=1)

# accel --> all the accelerometer readings, one per frame. ex. shape (60, 3) for 60 rows of 3 numbers
# imu_to_cam --> 3x3 grid of numbers (rotation matrix) that converts from IMU to camera axis. (obtained from camera)
# down - is gravity vector in terms of camera coordinate space
# spread - is percieved noise/varance in each "down" measurement
def gravity_down(accel: np.ndarray, imu_to_cam: np.ndarray) -> tuple[np.ndarray, float]:
    """Unit 'down' vector in the camera frame from accelerometer samples (IMU
    frame, reading UP at rest). Also returns the samples' angular spread (deg)."""
    up_imu = accel.mean(axis=0)     # averages down rows, giving 3 numbers
    down = -(imu_to_cam @ up_imu)   # rotate the direction from IMU axes to camera axes (@ = dot product), then flip
    down /= np.linalg.norm(down)    # linalg gives length of direction, /= --> down = down / length (i.e. get unit vector)
    # next 3 lines only measure how noisy the sensor was (for report/debug)
    dirs = -(accel @ imu_to_cam.T) 
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    spread = np.degrees(np.arccos(np.clip(dirs @ down, -1, 1))).std() if len(accel) > 1 else 0.0
    return down, float(spread)

# down - "down" vector from gravity_down(), 3 numbers of length 1.
# points - waterline points from project_to_waterline, of shape (M,3) with one [x,y,z] per row
def fit_plane(down: np.ndarray, points: np.ndarray) -> dict:
    """Plane with normal `down` through the points: d = median(down . p)."""
    h = points @ down       # how far below the camera the waterline is along gravity vector (dot product @)
    d = float(np.median(h)) # find median of h
    mad = float(np.median(np.abs(h - d)) * 1.4826)
    return {"d": d, "heights": h, "height_std_m": float(h.std()), "height_mad_m": mad}


def camera_tilt(down: np.ndarray) -> dict:
    """Camera pitch (+ = looking down) and roll implied by gravity, degrees."""
    return {"pitch_deg": float(np.degrees(np.arcsin(np.clip(down[2], -1, 1)))),
            "roll_deg": float(np.degrees(np.arctan2(-down[0], down[1])))}


def open_camera(use_stub: bool, svo_path, confidence_threshold: float):
    if use_stub:
        from zed_stub import StubCamera
        cam = StubCamera(confidence_threshold=confidence_threshold)
    else:
        try:
            from zed_real import ZedCamera  # not written yet: needs the ZED SDK
        except ImportError:
            sys.exit("zed_real.py (real ZED SDK wrapper) is not available yet; run with --stub.")
        cam = ZedCamera(confidence_threshold=confidence_threshold)
    return cam, cam.open(svo_path)


def estimate(cam, info, waterline: dict, pixels_above=10, n_samples=50, min_valid=0.3,
             frame_offset=0, origin_pixel=None, project=True) -> dict:
    """Run the whole method. Returns the result dict written to plane.json."""
    size = waterline.get("image_size")
    if size and list(size) != [info.width, info.height]:
        raise ValueError(f"waterline.json image size {size} does not match the recording "
                         f"({info.width}x{info.height}); use the same resolution.")
    f0, f1 = waterline["frame_range"]
    frames = range(f0 + frame_offset, f1 + frame_offset)

    line_px, wall_px = sample_pixels(waterline, n_samples, pixels_above)
    pixels = wall_px if origin_pixel is None else np.vstack([wall_px, [origin_pixel]])
    stack, accel = read_points(cam, pixels, frames)
    med, counts = median_over_frames(stack, min_valid)
    wall_pts, origin_pt = med[:len(wall_px)], (None if origin_pixel is None else med[-1])

    ok = ~np.isnan(wall_pts).any(axis=1)
    if ok.sum() < 3:
        raise RuntimeError(f"Only {ok.sum()} sample pixels had valid depth; try a larger "
                           "--pixels-above, a looser confidence threshold, or check the waterline.")
    pts = project_to_waterline(wall_pts[ok], line_px[ok], info) if project else wall_pts[ok]

    down, g_spread = gravity_down(accel, cam.imu_to_camera_rotation())
    plane = fit_plane(down, pts)
    result = {
        "plane": {"normal": down.tolist(), "d": plane["d"],
                  "equation": "normal . X = d  (X in camera frame; normal points down; d = camera height above water)",
                  "frame": "left camera: x right, y down, z forward", "units": "metres"},
        "quality": {"points_used": int(ok.sum()), "points_sampled": len(wall_px),
                    "height_spread_std_m": plane["height_std_m"],
                    "height_spread_mad_m": plane["height_mad_m"],
                    "valid_depth_fraction": float(counts[:len(wall_px)].sum() / (len(frames) * len(wall_px))),
                    "gravity_sample_spread_deg": g_spread,
                    "camera_tilt": camera_tilt(down)},
        "inputs": {"frames": [frames.start, frames.stop], "pixels_above": pixels_above,
                   "samples": n_samples, "min_valid": min_valid,
                   "confidence_threshold": getattr(cam, "confidence_threshold", None),
                   "projected_to_waterline": project},
        "samples": [{"line_px": lp.tolist(), "wall_px": wp.tolist(), "point_m": p.tolist(),
                     "height_m": float(h), "valid_frames": int(c)}
                    for lp, wp, p, h, c in zip(line_px[ok], wall_px[ok], pts, plane["heights"],
                                               counts[:len(wall_px)][ok])],
    }
    if origin_pixel is not None:
        if np.isnan(origin_pt).any():
            raise RuntimeError("No valid depth at --origin-pixel; pick a textured, visible spot.")
        h = plane["d"] - origin_pt @ down
        result["origin"] = {"pixel": list(origin_pixel), "point_m": origin_pt.tolist(),
                            "height_above_water_m": float(h),
                            "on_water_m": (origin_pt + h * down).tolist()}
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description="Estimate the water plane from a waterline and ZED depth.")
    p.add_argument("--waterline", required=True, help="waterline.json from detect_waterline_lite.py")
    p.add_argument("--svo", help="ZED .svo2 recording (ignored with --stub)")
    p.add_argument("--stub", action="store_true", help="use the fake camera in zed_stub.py")
    p.add_argument("--pixels-above", type=int, default=10, help="sample the wall this many px above the line")
    p.add_argument("--samples", type=int, default=50, help="number of columns sampled along the line")
    p.add_argument("--confidence-threshold", type=float, default=50,
                   help="SDK depth confidence threshold (1-100, lower = stricter)")
    p.add_argument("--min-valid", type=float, default=0.3,
                   help="drop pixels with valid depth in fewer than this fraction of frames")
    p.add_argument("--frame-offset", type=int, default=0,
                   help="add to waterline.json frame numbers if the mp4 and SVO2 frames differ")
    p.add_argument("--origin-pixel", help="u,v of a marked reference point (optional)")
    p.add_argument("--output", default="plane.json")
    a = p.parse_args(argv)
    if not a.stub and not a.svo:
        p.error("--svo is required unless --stub is used")
    origin = [int(v) for v in a.origin_pixel.split(",")] if a.origin_pixel else None

    cam, info = open_camera(a.stub, a.svo, a.confidence_threshold)
    try:
        res = estimate(cam, info, load_waterline(a.waterline), a.pixels_above, a.samples,
                       a.min_valid, a.frame_offset, origin)
    finally:
        cam.close()
    res["inputs"].update({"waterline": a.waterline, "svo": "stub" if a.stub else a.svo})
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    with open(a.output, "w") as f:
        json.dump(res, f, indent=1)

    q, pl = res["quality"], res["plane"]
    print(f"Plane: normal = {np.round(pl['normal'], 5).tolist()}, d = {pl['d']:.4f} m "
          f"(camera height above water)")
    print(f"Camera tilt: pitch {q['camera_tilt']['pitch_deg']:+.2f} deg, roll {q['camera_tilt']['roll_deg']:+.2f} deg")
    print(f"Quality: {q['points_used']}/{q['points_sampled']} points, height spread "
          f"{q['height_spread_mad_m'] * 100:.2f} cm (MAD), {q['valid_depth_fraction']:.0%} valid depth, "
          f"gravity spread {q['gravity_sample_spread_deg']:.3f} deg")
    if "origin" in res:
        o = res["origin"]
        print(f"Origin: {o['height_above_water_m']:.4f} m above water; on water at "
              f"{np.round(o['on_water_m'], 4).tolist()}")
    print(f"Saved {a.output}")


if __name__ == "__main__":
    main()
