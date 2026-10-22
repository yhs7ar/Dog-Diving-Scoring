#!/usr/bin/env python3
"""
Real ZED SDK camera for estimate_water_plane.py.

ZedCamera has exactly the same methods as StubCamera in zed_stub.py, so the
plane script works with either one. This file needs the ZED SDK and its
Python package (pyzed), so it only runs on a machine with the SDK installed.

Conventions (must match the stub; estimate_water_plane.py relies on them):
    * 3D points in the LEFT camera frame: x right, y down, z forward, metres
    * invalid depth comes back as NaN
    * accelerometer in m/s^2, reading UP at rest (+9.8 away from the Earth)
    * imu_to_camera_rotation() converts IMU-axis vectors into camera axes

Check on the real camera before trusting results (see check_imu() below):
the SDK docs don't say whether IMU readings are already in camera axes. If
they are, pass imu_in_camera_frame=True so the rotation isn't applied twice.

Quick test on a recording:
    python zed_real.py recording.svo2
"""

import sys

import numpy as np
import pyzed.sl as sl

from zed_stub import CameraInfo   # same simple info container as the stub


class ZedCamera:
    """Plays back an SVO/SVO2 recording through the ZED SDK."""

    # ---- Step 1: create the object (nothing is opened yet) -----------------
    def __init__(self, confidence_threshold: float = 50, texture_confidence_threshold: float = 100,
                 depth_mode=sl.DEPTH_MODE.NEURAL, imu_in_camera_frame: bool = False):
        self.confidence_threshold = confidence_threshold
        self.zed = sl.Camera()
        self.depth_mode = depth_mode
        self.imu_in_camera_frame = imu_in_camera_frame

        # Settings used on every grab(). Lower thresholds = stricter: pixels
        # the SDK isn't confident about come back as NaN.
        self.runtime = sl.RuntimeParameters()
        self.runtime.confidence_threshold = int(confidence_threshold)
        self.runtime.texture_confidence_threshold = int(texture_confidence_threshold)

        # Reusable containers the SDK fills in (created once, not every frame).
        self.xyz = sl.Mat()
        self.sensors = sl.SensorsData()

    # ---- Step 2: open the recording and read the calibration ---------------
    def open(self, svo_path) -> CameraInfo:
        init = sl.InitParameters()
        init.set_from_svo_file(str(svo_path))
        init.coordinate_system = sl.COORDINATE_SYSTEM.IMAGE   # x right, y down, z forward
        init.coordinate_units = sl.UNIT.METER                 # SDK default is millimetres!
        init.depth_mode = self.depth_mode
        init.svo_real_time_mode = False                       # read frames as fast as we ask

        err = self.zed.open(init)
        if err != sl.ERROR_CODE.SUCCESS:
            raise IOError(f"Could not open {svo_path}: {err}")

        config = self.zed.get_camera_information().camera_configuration
        left = config.calibration_parameters.left_cam
        return CameraInfo(
            width=config.resolution.width,
            height=config.resolution.height,
            fx=left.fx, fy=left.fy, cx=left.cx, cy=left.cy,
            fps=config.fps,
            frame_count=self.zed.get_svo_number_of_frames(),
        )

    # ---- Step 3: move through the recording ---------------------------------
    def set_frame(self, index: int) -> None:
        self.zed.set_svo_position(int(index))

    def grab(self) -> bool:
        """Load the current frame and compute its depth. False at the end of
        the recording or on any error."""
        return self.zed.grab(self.runtime) == sl.ERROR_CODE.SUCCESS

    # ---- Step 4: 3D points at chosen pixels ----------------------------------
    def get_xyz(self, pixels: np.ndarray) -> np.ndarray:
        """(K, 2) integer pixels [u, v] -> (K, 3) points in metres, NaN if invalid.

        retrieve_measure fills a whole-image table (height x width x 4: X, Y,
        Z and an unused 4th value) for the last grabbed frame. We then pick out
        just the pixels we need. Note the order: the table is indexed
        [row, column] = [v, u], not [u, v].
        """
        self.zed.retrieve_measure(self.xyz, sl.MEASURE.XYZ)
        cloud = self.xyz.get_data()                    # NumPy array (H, W, 4)
        px = np.asarray(pixels, int).reshape(-1, 2)
        u, v = px[:, 0], px[:, 1]
        pts = cloud[v, u, :3].astype(np.float64)       # (K, 3)
        # The SDK marks invalid depth with NaN, and too-far / too-close with
        # +inf / -inf. Treat all of them as "no data".
        pts[~np.isfinite(pts).all(axis=1)] = np.nan
        return pts

    # ---- Step 5: the accelerometer -------------------------------------------
    def get_accelerometer(self) -> np.ndarray:
        """One accelerometer reading (m/s^2) for the current frame. Includes
        gravity and points UP (away from the Earth) when the camera is still."""
        self.zed.get_sensors_data(self.sensors, sl.TIME_REFERENCE.IMAGE)
        acc = self.sensors.get_imu_data().get_linear_acceleration()
        return np.asarray(acc, dtype=np.float64)

    def imu_to_camera_rotation(self) -> np.ndarray:
        """3x3 rotation from IMU axes to left-camera axes.

        The SDK documents camera_imu_transform as the "IMU to left camera"
        transform; only its rotation part matters for a direction like gravity.
        If the SDK already reports IMU data in camera axes, return identity.
        """
        if self.imu_in_camera_frame:
            return np.eye(3)
        t = self.zed.get_camera_information().sensors_configuration.camera_imu_transform
        return np.array(t.m)[:3, :3]                   # top-left 3x3 of the 4x4 = rotation

    # ---- Step 6: clean up -------------------------------------------------------
    def close(self) -> None:
        self.zed.close()


def check_imu(svo_path, frames: int = 30) -> None:
    """Print what the camera thinks 'down' is, to check the IMU conventions.

    Hold or mount the camera LEVEL for this recording. In camera axes, "up"
    is -y, so the rotated reading should be close to [0, -9.8, 0]:
        * close to [0, -9.8, 0]  -> conventions are right
        * close to [0, +9.8, 0]  -> the sign is flipped
        * 9.8 on x or z instead  -> the axes don't match; try
                                     imu_in_camera_frame=True (or False)
    """
    cam = ZedCamera()
    info = cam.open(svo_path)
    readings = []
    for i in range(min(frames, info.frame_count)):
        cam.set_frame(i)
        if cam.grab():
            readings.append(cam.get_accelerometer())
    raw = np.mean(readings, axis=0)
    for in_cam in (False, True):
        cam.imu_in_camera_frame = in_cam
        rotated = cam.imu_to_camera_rotation() @ raw
        print(f"imu_in_camera_frame={in_cam!s:5s}: camera-axis reading = {np.round(rotated, 3)}")
    print(f"raw IMU reading = {np.round(raw, 3)}   (|a| = {np.linalg.norm(raw):.3f} m/s^2, expect ~9.8)")
    print(f"resolution {info.width}x{info.height}, fx={info.fx:.1f} fy={info.fy:.1f} "
          f"cx={info.cx:.1f} cy={info.cy:.1f}, {info.frame_count} frames")
    cam.close()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python zed_real.py recording.svo2")
    check_imu(sys.argv[1])
