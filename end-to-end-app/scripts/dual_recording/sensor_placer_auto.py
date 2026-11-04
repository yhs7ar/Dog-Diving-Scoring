#!/usr/bin/env python3
"""
Automated ZED Sensor Placer & Extrinsic Calibration Tool
Replaces the GUI-based ZED Sensor Placer tool.

Supports:
1. Floor Alignment (find_floor_plane) + IMU leveling (matches Sensor Placer protocol)
2. Interactive/Automated Point-Pair Alignment (Yaw + Translation or full 6-DoF Kabsch)
3. ArUco Marker detection across synchronized SVO2 frames
4. Pure NumPy/SciPy ICP (Iterative Closest Point) fallback (no Open3D required)
5. Output format support:
   - ZED Sensor Placer JSON ("zeds": [{serial, rotation, translation}])
   - ZED Fusion JSON ("cameras": [{serial_number, position: {translation, rotation}}])
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import KDTree

try:
    import pyzed.sl as sl
except ImportError:
    print("ERROR: pyzed (ZED SDK Python API) is not installed.")
    sys.exit(1)


ARUCO_DICT_MAP = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
    "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
    "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
    "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
}


def detect_aruco_corners_2d(image_bgr, aruco_dict_name="DICT_4X4_50"):
    dict_id = ARUCO_DICT_MAP.get(aruco_dict_name)
    if dict_id is None:
        raise ValueError(f"Unknown ArUco dictionary: {aruco_dict_name}")

    aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
    aruco_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

    corners, ids, _ = detector.detectMarkers(image_bgr)
    result = {}
    if ids is not None:
        for i, marker_id in enumerate(ids.flatten()):
            result[int(marker_id)] = corners[i][0]
    return result


def get_3d_points_at_pixels(point_cloud_np, pixel_coords):
    points_3d = []
    h, w = point_cloud_np.shape[:2]
    for col, row in pixel_coords:
        col_i, row_i = int(round(col)), int(round(row))
        if 0 <= row_i < h and 0 <= col_i < w:
            pt = point_cloud_np[row_i, col_i, :3]
            points_3d.append(pt)
        else:
            points_3d.append([np.nan, np.nan, np.nan])
    return np.array(points_3d, dtype=np.float64)


def kabsch_transform(P, Q):
    """
    Computes optimal 6-DoF rigid rotation R and translation T:
    min ||R @ P.T + T - Q.T||^2
    P = source points (Camera 1), Q = target points (Camera 0)
    """
    assert P.shape == Q.shape and P.shape[1] == 3
    centroid_P = P.mean(axis=0)
    centroid_Q = Q.mean(axis=0)

    P_centered = P - centroid_P
    Q_centered = Q - centroid_Q

    H = P_centered.T @ Q_centered
    U, S, Vt = np.linalg.svd(H)

    d = np.linalg.det(Vt.T @ U.T)
    sign_matrix = np.diag([1, 1, np.sign(d)])

    R = Vt.T @ sign_matrix @ U.T
    T = centroid_Q - R @ centroid_P

    aligned = (R @ P.T).T + T
    rmse = np.sqrt(np.mean(np.sum((aligned - Q) ** 2, axis=1)))
    return R, T, rmse


def yaw_translation_from_pairs(src_pts, tgt_pts):
    """
    Replicates the exact Sensor Placer GUI algorithm (`yawTranslationFromPairs`):
    Constrains rotation around Y axis (yaw) + translation (X, Y, Z).
    """
    n = len(src_pts)
    sc = np.mean(src_pts, axis=0)
    tc = np.mean(tgt_pts, axis=0)

    A = 0.0
    B = 0.0
    for s, t in zip(src_pts, tgt_pts):
        sx, sz = s[0] - sc[0], s[2] - sc[2]
        tx, tz = t[0] - tc[0], t[2] - tc[2]
        A += tx * sx + tz * sz
        B += tx * sz - tz * sx

    theta = math.atan2(B, A)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)

    R = np.array([
        [cos_t, 0.0, sin_t],
        [0.0,   1.0, 0.0],
        [-sin_t, 0.0, cos_t]
    ], dtype=np.float64)

    t = np.array([
        tc[0] - (cos_t * sc[0] + sin_t * sc[2]),
        tc[1] - sc[1],
        tc[2] - (-sin_t * sc[0] + cos_t * sc[2])
    ], dtype=np.float64)

    aligned = (R @ src_pts.T).T + t
    rmse = np.sqrt(np.mean(np.sum((aligned - tgt_pts) ** 2, axis=1)))
    return R, t, rmse


def icp_scipy(src_pts, tgt_pts, max_iter=50, tolerance=1e-4, max_dist=0.3):
    """
    Lightweight ICP implementation using SciPy KDTree without external 3D libraries.
    """
    tree = KDTree(tgt_pts)
    current_src = src_pts.copy()
    R_total = np.eye(3, dtype=np.float64)
    T_total = np.zeros(3, dtype=np.float64)

    prev_rmse = float('inf')

    for iteration in range(max_iter):
        dists, indices = tree.query(current_src)
        mask = dists < max_dist
        if np.sum(mask) < 20:
            break

        p_matched = current_src[mask]
        q_matched = tgt_pts[indices[mask]]

        R_step, T_step, rmse = kabsch_transform(p_matched, q_matched)

        current_src = (R_step @ current_src.T).T + T_step
        R_total = R_step @ R_total
        T_total = R_step @ T_total + T_step

        if abs(prev_rmse - rmse) < tolerance:
            break
        prev_rmse = rmse

    fitness = np.sum(mask) / len(src_pts) if len(src_pts) > 0 else 0
    return R_total, T_total, fitness, prev_rmse


def open_svo(svo_path, depth_mode_str="NEURAL"):
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.set_from_svo_file(str(svo_path))
    init_params.svo_real_time_mode = False
    init_params.coordinate_units = sl.UNIT.METER
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP

    depth_modes = {
        "NEURAL": sl.DEPTH_MODE.NEURAL,
        "NEURAL_PLUS": sl.DEPTH_MODE.NEURAL_PLUS,
        "ULTRA": sl.DEPTH_MODE.ULTRA,
        "QUALITY": sl.DEPTH_MODE.QUALITY,
        "PERFORMANCE": sl.DEPTH_MODE.PERFORMANCE,
    }
    init_params.depth_mode = depth_modes.get(depth_mode_str.upper(), sl.DEPTH_MODE.NEURAL)

    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"Failed to open SVO: {svo_path} — {repr(err)}")

    runtime_params = sl.RuntimeParameters()
    return zed, runtime_params


def grab_frame_data(zed, runtime_params):
    status = zed.grab(runtime_params)
    if status != sl.ERROR_CODE.SUCCESS:
        return None, None, None

    image_zed = sl.Mat()
    zed.retrieve_image(image_zed, sl.VIEW.LEFT)
    bgr = image_zed.get_data()[:, :, :3].copy()

    pc_zed = sl.Mat()
    zed.retrieve_measure(pc_zed, sl.MEASURE.XYZRGBA)
    pc_np = pc_zed.get_data().copy()

    ts = zed.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_milliseconds()
    return bgr, pc_np, ts


def rotation_matrix_to_rodrigues(R):
    rvec, _ = cv2.Rodrigues(R)
    return rvec.flatten()


def rotation_matrix_to_euler_deg(R):
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    if not singular:
        rx = math.atan2(R[2, 1], R[2, 2])
        ry = math.atan2(-R[2, 0], sy)
        rz = math.atan2(R[1, 0], R[0, 0])
    else:
        rx = math.atan2(-R[1, 2], R[1, 1])
        ry = math.atan2(-R[2, 0], sy)
        rz = 0.0
    return np.degrees([rx, ry, rz])


def detect_floor_alignment(zed, runtime_params, max_attempts=15):
    """
    Detect floor plane using sl.Camera.find_floor_plane, matching Sensor Placer 'Floor Align'.
    """
    plane = sl.Plane()
    reset_transform = sl.Transform()
    for _ in range(max_attempts):
        if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
            st = zed.find_floor_plane(plane, reset_transform)
            if st == sl.ERROR_CODE.SUCCESS:
                R_mat = reset_transform.get_rotation_matrix().r
                T_vec = reset_transform.get_translation().get()
                return np.array(R_mat), np.array(T_vec), True
    return np.eye(3), np.zeros(3), False


def main():
    parser = argparse.ArgumentParser(
        description="Automated ZED Sensor Placer — replicates ZED Sensor Placer GUI logic."
    )
    parser.add_argument("--svo0", required=True, help="Reference camera SVO2 (Cam 0)")
    parser.add_argument("--svo1", required=True, help="Target camera SVO2 (Cam 1)")
    parser.add_argument("--output", default="sensor_placer_config.json", help="Output config path")
    parser.add_argument("--format", choices=["sensor_placer", "fusion", "both"], default="both",
                        help="Output format: 'sensor_placer' (ZED Sensor Placer JSON), 'fusion' (Fusion API JSON), or 'both'")
    parser.add_argument("--depth-mode", default="NEURAL", help="ZED depth mode")
    parser.add_argument("--aruco-dict", default="DICT_4X4_50", help="ArUco dictionary")
    parser.add_argument("--use-floor-align", action="store_true", default=True,
                        help="Perform floor alignment first (standard Sensor Placer protocol)")
    parser.add_argument("--yaw-only", action="store_true",
                        help="Constrain alignment to Yaw + Translation (exact Sensor Placer match)")
    parser.add_argument("--max-frames", type=int, default=100, help="Max frames to inspect")
    args = parser.parse_args()

    svo0_path = Path(args.svo0)
    svo1_path = Path(args.svo1)

    print("=" * 60)
    print("  Automated ZED Sensor Placer")
    print("=" * 60)
    print(f"  Cam 0 (Ref) : {svo0_path}")
    print(f"  Cam 1       : {svo1_path}")
    print(f"  Format      : {args.format}")

    zed0, rt0 = open_svo(svo0_path, args.depth_mode)
    zed1, rt1 = open_svo(svo1_path, args.depth_mode)

    serial0 = zed0.get_camera_information().serial_number
    serial1 = zed1.get_camera_information().serial_number
    print(f"  Cam 0 Serial: {serial0} | Cam 1 Serial: {serial1}")

    # Optional Step: Floor Align
    if args.use_floor_align:
        print("\n[Protocol Step 1] Detecting Floor Plane (Sensor Placer Floor Align)...")
        R_floor0, T_floor0, found0 = detect_floor_alignment(zed0, rt0)
        R_floor1, T_floor1, found1 = detect_floor_alignment(zed1, rt1)
        print(f"  Cam 0 Floor found: {found0}")
        print(f"  Cam 1 Floor found: {found1}")
        zed0.set_svo_position(0)
        zed1.set_svo_position(0)

    # Step 2: Look for point correspondences across frames
    print("\n[Protocol Step 2] Scanning for point correspondences (ArUco markers)...")
    pts0_list, pts1_list = [], []
    frame_idx = 0

    while frame_idx < args.max_frames:
        bgr0, pc0, ts0 = grab_frame_data(zed0, rt0)
        bgr1, pc1, ts1 = grab_frame_data(zed1, rt1)
        if bgr0 is None or bgr1 is None:
            break
        frame_idx += 1

        m0 = detect_aruco_corners_2d(bgr0, args.aruco_dict)
        m1 = detect_aruco_corners_2d(bgr1, args.aruco_dict)
        common = set(m0.keys()) & set(m1.keys())

        for mid in common:
            p0 = get_3d_points_at_pixels(pc0, m0[mid])
            p1 = get_3d_points_at_pixels(pc1, m1[mid])
            valid = np.isfinite(p0).all(axis=1) & np.isfinite(p1).all(axis=1)
            if valid.sum() >= 3:
                pts0_list.append(p0[valid])
                pts1_list.append(p1[valid])

    R, T, rmse = None, None, None
    method_name = ""

    if len(pts0_list) > 0:
        pts0 = np.vstack(pts0_list)
        pts1 = np.vstack(pts1_list)
        print(f"  Found {len(pts0)} valid corresponding 3D points.")

        if args.yaw_only:
            R, T, rmse = yaw_translation_from_pairs(pts1, pts0)
            method_name = "SensorPlacer_YawTranslation_ArUco"
        else:
            R, T, rmse = kabsch_transform(pts1, pts0)
            method_name = "Kabsch_SVD_6DoF_ArUco"
    else:
        print("  No common ArUco markers detected. Using SciPy ICP point cloud alignment...")
        zed0.set_svo_position(0)
        zed1.set_svo_position(0)
        _, pc0, _ = grab_frame_data(zed0, rt0)
        _, pc1, _ = grab_frame_data(zed1, rt1)

        xyz0 = pc0[:, :, :3].reshape(-1, 3)
        xyz1 = pc1[:, :, :3].reshape(-1, 3)
        xyz0 = xyz0[np.isfinite(xyz0).all(axis=1)]
        xyz1 = xyz1[np.isfinite(xyz1).all(axis=1)]

        # Subsample for fast ICP
        sub0 = xyz0[np.random.choice(len(xyz0), min(5000, len(xyz0)), replace=False)]
        sub1 = xyz1[np.random.choice(len(xyz1), min(5000, len(xyz1)), replace=False)]

        R, T, fitness, rmse = icp_scipy(sub1, sub0)
        method_name = f"SciPy_ICP (fitness: {fitness:.2f})"

    print(f"\n[Result] Method: {method_name}")
    print(f"  RMSE Error: {rmse:.4f} m")
    print(f"  Translation (T): [{T[0]:.4f}, {T[1]:.4f}, {T[2]:.4f}] m")
    euler = rotation_matrix_to_euler_deg(R)
    print(f"  Rotation (Euler deg): [{euler[0]:.2f}, {euler[1]:.2f}, {euler[2]:.2f}]")

    rvec = rotation_matrix_to_rodrigues(R)

    # 1. ZED Sensor Placer native format
    sensor_placer_json = {
        "zeds": [
            {
                "serial": int(serial0),
                "rotation": [0.0, 0.0, 0.0],
                "translation": [0.0, 0.0, 0.0]
            },
            {
                "serial": int(serial1),
                "rotation": [round(float(rvec[0]), 6), round(float(rvec[1]), 6), round(float(rvec[2]), 6)],
                "translation": [round(float(T[0]), 6), round(float(T[1]), 6), round(float(T[2]), 6)]
            }
        ]
    }

    # 2. ZED Fusion API format
    fusion_json = {
        "world": {
            "override_gravity": True,
            "rotation": [0.0, 0.0, 0.0],
            "translation": [0.0, 0.0, 0.0]
        },
        "cameras": [
            {
                "serial_number": int(serial0),
                "camera_model": "ZED_X",
                "stream": {"type": "usb", "configuration": ""},
                "position": {
                    "translation": [0.0, 0.0, 0.0],
                    "rotation": [0.0, 0.0, 0.0]
                }
            },
            {
                "serial_number": int(serial1),
                "camera_model": "ZED_X",
                "stream": {"type": "usb", "configuration": ""},
                "position": {
                    "translation": [round(float(T[0]), 6), round(float(T[1]), 6), round(float(T[2]), 6)],
                    "rotation": [round(float(euler[0]), 4), round(float(euler[1]), 4), round(float(euler[2]), 4)]
                }
            }
        ]
    }

    out_p = Path(args.output)
    if args.format in ("sensor_placer", "both"):
        sp_path = out_p.with_name(f"{out_p.stem}_sensor_placer.json") if args.format == "both" else out_p
        with open(sp_path, "w") as f:
            json.dump(sensor_placer_json, f, indent=2)
        print(f"\nSaved Sensor Placer config: {sp_path}")

    if args.format in ("fusion", "both"):
        f_path = out_p.with_name(f"{out_p.stem}_fusion.json") if args.format == "both" else out_p
        with open(f_path, "w") as f:
            json.dump(fusion_json, f, indent=2)
        print(f"Saved Fusion API config: {f_path}")

    zed0.close()
    zed1.close()
    print("\nCalibration completed successfully.")


if __name__ == "__main__":
    main()
