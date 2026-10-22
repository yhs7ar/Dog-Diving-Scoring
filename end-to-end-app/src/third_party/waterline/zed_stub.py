#!/usr/bin/env python3
"""
Stand-in for the ZED SDK, for testing the water-plane pipeline without a camera.

StubCamera has the same small interface that the real SDK wrapper (a future
zed_real.py) will provide, so estimate_water_plane.py can switch between them
by changing one import. Instead of reading an SVO2 file, it simulates a known
scene, so tests can compare the recovered plane with the true one:

    * a camera a known height above flat water, slightly tilted
    * a vertical far wall at a known distance
    * stereo-like depth noise that grows with distance squared, including a
      fixed per-pixel bias (real stereo errors repeat from frame to frame)
    * no depth on the water or sky (NaN), noisier depth just above the
      waterline, occasional wild outliers
    * SDK-style confidence filtering: pixels whose confidence value is above
      the threshold come back as NaN (in the ZED SDK, lower value = more
      confident; verify for your SDK version)
    * an accelerometer that reads UP (+9.81 m/s^2 against gravity) in its own
      slightly rotated IMU frame

Coordinates: camera frame, x right, y down, z forward, metres. This matches
the SDK's COORDINATE_SYSTEM.IMAGE with UNIT.METER.

Real SDK equivalents (from the ZED SDK Python API as I know it; check the
names against the docs for your SDK version before writing zed_real.py):
    open()                     sl.Camera.open(InitParameters with set_from_svo_file,
                               coordinate_system=IMAGE, coordinate_units=METER)
    CameraInfo                 get_camera_information().camera_configuration
                               (.resolution, .calibration_parameters.left_cam.fx/fy/cx/cy)
    set_frame()                set_svo_position()
    grab()                     grab(RuntimeParameters(confidence_threshold=...))
    get_xyz()                  retrieve_measure(mat, MEASURE.XYZ), then mat.get_data()[v, u, :3]
    get_accelerometer()        get_sensors_data(data, TIME_REFERENCE.IMAGE)
                               .get_imu_data().get_linear_acceleration()
    imu_to_camera_rotation()   get_camera_information().sensors_configuration
                               .camera_imu_transform (check which direction it maps!)

Run as a script to write the waterline.json this fake scene would produce:
    python zed_stub.py --write-waterline out_stub/waterline.json
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

GRAVITY = 9.81


def rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


@dataclass
class CameraInfo:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    fps: float
    frame_count: int


@dataclass
class StubScene:
    """The fake world. All distances in metres, angles in degrees."""
    width: int = 1920
    height: int = 1080
    fx: float = 1100.0
    fy: float = 1100.0
    cx: float = 960.0
    cy: float = 540.0
    fps: float = 30.0
    frame_count: int = 60
    camera_height: float = 0.5        # camera above the water
    pitch_deg: float = 2.0            # positive = camera looking down
    roll_deg: float = 1.0
    wall_distance: float = 8.0        # horizontal distance to the far wall
    wall_top: float = 0.6             # wall height above the water
    depth_noise: float = 0.001        # per-frame noise std = depth_noise * Z^2
    depth_bias: float = 0.0005        # fixed per-pixel bias std = depth_bias * Z^2
    edge_px: int = 4                  # band above the waterline with poor depth
    outlier_fraction: float = 0.01    # wild depth values that still pass confidence
    accel_noise: float = 0.02         # m/s^2 per sample
    accel_bias: tuple = (0.0, 0.0, 0.0)  # constant accelerometer bias, IMU frame
    imu_rotation_deg: tuple = (0.5, -0.3, 0.2)  # IMU axes vs camera axes (x, y, z)

    # ---- ground truth --------------------------------------------------
    def down(self) -> np.ndarray:
        """True gravity direction (unit, pointing down) in the camera frame."""
        return rot_z(np.radians(self.roll_deg)) @ rot_x(np.radians(self.pitch_deg)) @ np.array([0.0, 1.0, 0.0])

    def forward(self) -> np.ndarray:
        """Horizontal direction towards the wall (camera z flattened)."""
        z, d = np.array([0.0, 0.0, 1.0]), self.down()
        f = z - (z @ d) * d
        return f / np.linalg.norm(f)

    def imu_to_camera(self) -> np.ndarray:
        rx, ry, rz = np.radians(self.imu_rotation_deg)
        c, s = np.cos(ry), np.sin(ry)
        rot_y = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        return rot_z(rz) @ rot_y @ rot_x(rx)

    def rays(self, u, v) -> np.ndarray:
        u, v = np.asarray(u, float), np.asarray(v, float)
        return np.stack([(u - self.cx) / self.fx, (v - self.cy) / self.fy, np.ones_like(u)], axis=-1)

    def waterline_v(self, u) -> np.ndarray:
        """Image row of the true waterline (wall meets water) in column u.
        Solves (D*down - H*forward) . ray(u, v) = 0, which is linear in v."""
        a = self.wall_distance * self.down() - self.camera_height * self.forward()
        u = np.asarray(u, float)
        return self.cy - self.fy * (a[2] + a[0] * (u - self.cx) / self.fx) / a[1]

    def true_xyz(self, u, v) -> np.ndarray:
        """Noise-free 3D point seen at pixel (u, v); NaN for water or sky."""
        r = self.rays(u, v)
        t = self.wall_distance / (r @ self.forward())
        p = t[..., None] * r
        height = self.camera_height - p @ self.down()     # above the water
        visible = (t > 0) & (height >= 0) & (height <= self.wall_top)
        p[~visible] = np.nan
        return p


class StubCamera:
    """Fake ZED camera. See the module docstring for the real SDK equivalents."""

    def __init__(self, scene: StubScene | None = None, confidence_threshold: float = 50, seed: int = 0):
        self.scene = scene or StubScene()
        self.confidence_threshold = confidence_threshold
        self.seed = seed
        self.frame = 0
        self._rng = None

    def open(self, svo_path=None) -> CameraInfo:
        """svo_path is ignored; the stub always 'plays' its fake scene."""
        s = self.scene
        return CameraInfo(s.width, s.height, s.fx, s.fy, s.cx, s.cy, s.fps, s.frame_count)

    def set_frame(self, index: int) -> None:
        self.frame = int(index)

    def grab(self) -> bool:
        """Advance to the current frame. False past the end of the recording."""
        if not 0 <= self.frame < self.scene.frame_count:
            return False
        self._rng = np.random.default_rng((self.seed, self.frame))
        return True

    def get_xyz(self, pixels: np.ndarray) -> np.ndarray:
        """(K, 2) integer pixels (u, v) -> (K, 3) points in metres, NaN where
        depth is invalid or rejected by the confidence threshold."""
        s, rng = self.scene, self._rng
        px = np.asarray(pixels, float).reshape(-1, 2)
        u, v = px[:, 0], px[:, 1]
        p = s.true_xyz(u, v)
        r = s.rays(u, v)
        z = p[:, 2]

        # Fixed per-pixel bias (same every frame) + per-frame noise, along the ray.
        unit_bias = np.array([np.random.default_rng((self.seed, 999, int(a), int(b))).normal()
                              for a, b in zip(u, v)])
        bias = unit_bias * s.depth_bias * z ** 2
        near_edge = (s.waterline_v(u) - v >= 0) & (s.waterline_v(u) - v < s.edge_px)
        noise = rng.normal(0, 1, len(u)) * s.depth_noise * z ** 2 * np.where(near_edge, 3.0, 1.0)
        p = p + ((bias + noise) / r[:, 2])[:, None] * r

        wild = rng.random(len(u)) < s.outlier_fraction
        p[wild] *= rng.uniform(0.5, 1.5, (wild.sum(), 1))

        # Confidence: lower = better (as in the ZED SDK). Edge pixels score badly.
        conf = 15 + rng.uniform(0, 40, len(u)) + np.where(near_edge, 50, 0)
        p[conf > self.confidence_threshold] = np.nan
        return p

    def get_accelerometer(self) -> np.ndarray:
        """One accelerometer sample (m/s^2) in the IMU frame for the current frame.
        At rest it reads UP: the opposite of gravity."""
        s, rng = self.scene, self._rng
        up_cam = -GRAVITY * s.down()
        a_imu = s.imu_to_camera().T @ up_cam
        return a_imu + np.asarray(s.accel_bias) + rng.normal(0, s.accel_noise, 3)

    def imu_to_camera_rotation(self) -> np.ndarray:
        """3x3 rotation taking IMU-frame vectors into the left camera frame."""
        return self.scene.imu_to_camera()

    def close(self) -> None:
        pass


def write_waterline(path, scene: StubScene | None = None, margin: int = 100) -> dict:
    """Write the waterline.json (lite format) that this scene would produce,
    so the plane estimator can be tested on pixels that match the 3D scene."""
    s = scene or StubScene()
    u = np.arange(margin, s.width - margin, dtype=float)
    v = s.waterline_v(u)
    slope, intercept = np.polyfit(u, v, 1)
    data = {
        "video": "stub",
        "frame_range": [0, s.frame_count],
        "time_range_s": [0.0, s.frame_count / s.fps],
        "fps": s.fps,
        "image_size": [s.width, s.height],
        "roi": None,
        "threshold": None,
        "line": {"slope": float(slope), "intercept": float(intercept),
                 "angle_deg": float(np.degrees(np.arctan(slope))),
                 "endpoints": [[float(u[0]), float(v[0])], [float(u[-1]), float(v[-1])]],
                 "rms_residual_px": 0.0, "inlier_fraction": 1.0},
        "points_xy": np.round(np.stack([u, v], axis=1), 2).tolist(),
        "inlier": [True] * len(u),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=1)
    return data


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Write the waterline.json for the stub scene.")
    ap.add_argument("--write-waterline", default="out_stub/waterline.json")
    a = ap.parse_args()
    s = StubScene()
    write_waterline(a.write_waterline, s)
    print(f"Wrote {a.write_waterline}. True plane: down = {np.round(s.down(), 5).tolist()}, "
          f"camera height = {s.camera_height} m")
