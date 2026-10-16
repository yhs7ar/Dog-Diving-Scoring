import os
import re
import sys
import json
import time
import math
import cv2
import numpy as np
import torch
from pathlib import Path

# Force immediate terminal printouts
sys.stdout.reconfigure(line_buffering=True)

# AMD Windows fix
torch.backends.cudnn.enabled = False

from ultralytics import YOLO

# ----------------- CONFIGURATION -----------------
# Path to the trained model (.pt or Jetson .engine)
MODEL_PATH = Path(r"trained_models\dog_pose_s_2kpt_clahe_model\weights\best.pt")

# Path to the unseen test video
VIDEO_PATH = Path(r"I:\Capstone\Dog-Diving-Scoring\Our_dog_diving_videos\test-clip-video004_right.mp4")

# Path to the Test.txt file
TEST_TXT_PATH = Path(r"I:\Capstone\Dog-Diving-Scoring\Our_dog_diving_videos\test-clip-video004_right_pose\Test.txt")

# Output directory
OUTPUT_DIR = Path(r"I:\Capstone\Dog-Diving-Scoring\test_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Keypoints: 0 = Nose, 1 = Tailset
TAILSET_INDEX = 1
CONF_THRESHOLD = 0.10       # Low threshold for small/dark dogs
TAIL_CONF_THRESHOLD = 0.30

# Horizontal Corridor Crop (Cuts out sky and bottom pool liner)
CORRIDOR_Y_START_RATIO = 0.08  # ~8% down (just above dock)
CORRIDOR_Y_END_RATIO = 0.75    # ~75% down (just below water surface)
# --------------------------------------------------

def compute_iou(box1, box2):
    """Computes IoU between two boxes [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0.0

def apply_clahe(frame, clip_limit=2.5, tile_grid_size=(8, 8)):
    """Enhances contrast for dark dogs against bright sky/water."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_enhanced = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l_enhanced, a, b)), cv2.COLOR_LAB2BGR)

def fit_parabolic_trajectory(points, frame_width, frame_height, min_points=6):
    """Fits quadratic curve y = Ax^2 + Bx + C to extrapolate splashdown point."""
    if len(points) < min_points:
        return None, None

    x = np.array([p[0] for p in points], dtype=np.float64)
    y = np.array([p[1] for p in points], dtype=np.float64)

    if np.max(x) - np.min(x) < 30:
        return None, None

    try:
        poly = np.poly1d(np.polyfit(x, y, 2))
    except Exception:
        return None, None

    dx = x[-1] - x[0]
    step = 3 if dx >= 0 else -3

    curve_points = []
    contact_point = None
    curr_x = int(x[0])

    while 0 <= curr_x < frame_width:
        curr_y = poly(curr_x)
        if curr_y >= frame_height - 1:
            contact_point = (int(curr_x), int(frame_height - 1))
            curve_points.append(contact_point)
            break
        if curr_y >= 0:
            curve_points.append((int(curr_x), int(curr_y)))
        curr_x += step

    return curve_points, contact_point

def load_ground_truth(test_txt_path):
    """Reads Test.txt and maps each annotated frame to its label file."""
    gt_map = {}
    if not test_txt_path.exists():
        print(f"Warning: {test_txt_path} not found. Running inference without GT metrics.")
        return gt_map

    base_dir = test_txt_path.parent
    with open(test_txt_path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    print(f"Found {len(lines)} ground-truth entries in {test_txt_path.name}")

    for img_rel_path in lines:
        img_p = Path(img_rel_path)
        if not img_p.is_absolute():
            img_p = base_dir / img_p

        lbl_p = img_p.with_suffix(".txt")
        if not lbl_p.exists():
            parts = list(img_p.parts)
            if "images" in parts:
                parts[parts.index("images")] = "labels"
                alt_lbl = Path(*parts).with_suffix(".txt")
                if alt_lbl.exists():
                    lbl_p = alt_lbl

        if lbl_p.exists():
            match = re.search(r"(\d+)", img_p.stem)
            if match:
                frame_num = int(match.group(1))
                gt_map[frame_num] = lbl_p
            gt_map[img_p.stem] = lbl_p

    print(f"Successfully indexed {len(gt_map)} ground-truth labels.")
    return gt_map

def main():
    print(f"Loading model from: {MODEL_PATH}")
    model = YOLO(str(MODEL_PATH))

    # Index Ground Truth
    gt_labels = load_ground_truth(TEST_TXT_PATH)

    # Open Video
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        print(f"Error: Could not open video: {VIDEO_PATH}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_video_path = OUTPUT_DIR / f"annotated_{VIDEO_PATH.name}"
    out_json_path = OUTPUT_DIR / f"{VIDEO_PATH.stem}_tail_data.json"

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(out_video_path), fourcc, fps, (width, height))

    print(f"Processing '{VIDEO_PATH.name}' ({total_frames} frames @ {fps:.1f} FPS)...")

    # Tracking & Metrics Buffers
    tail_json_data = []
    raw_tail_trajectory = []
    
    dog_detected_count = 0
    tail_detected_count = 0
    inference_times = []

    # Ground-truth metric accumulators
    gt_evaluated_frames = 0
    gt_tracked_frames = 0
    gt_valid_tail_frames = 0
    tail_pixel_errors = []
    box_center_errors = []
    box_ious = []

    # Calculate Corridor Y Bounds
    y_start = int(height * CORRIDOR_Y_START_RATIO)
    y_end = int(height * CORRIDOR_Y_END_RATIO)

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # Check if ground truth exists for this frame
        gt_file = gt_labels.get(frame_idx) or gt_labels.get(f"frame_{frame_idx:04d}")

        # Slice Horizontal Corridor (Magnifies dog by removing sky/pool floor)
        corridor_frame = frame[y_start:y_end, :]

        # CLAHE Contrast Enhancement
        enhanced_corridor = apply_clahe(corridor_frame)

        # Benchmark Inference & Tracking on Corridor
        t0 = time.perf_counter()
        results = model.track(
            source=enhanced_corridor,
            imgsz=1920,
            conf=CONF_THRESHOLD,
            max_det=1,
            tracker="bytetrack.yaml",
            persist=True,
            device=0,
            verbose=False
        )
        t1 = time.perf_counter()
        inference_times.append(t1 - t0)

        r = results[0]

        pred_box_xyxy = None
        pred_box_center = None
        pred_tail_pt = None

        # Check Detection & REMAP Coordinates to Full Frame
        if r.boxes is not None and len(r.boxes) > 0:
            dog_detected_count += 1
            track_id = int(r.boxes.id[0].item()) if r.boxes.id is not None else 1

            # Restore Bounding Box to Full Frame (add y_start offset)
            box_corr = r.boxes.xyxy[0].cpu().numpy()
            pred_box_xyxy = [
                float(box_corr[0]),
                float(box_corr[1] + y_start),
                float(box_corr[2]),
                float(box_corr[3] + y_start)
            ]
            pred_box_center = (
                (pred_box_xyxy[0] + pred_box_xyxy[2]) / 2.0,
                (pred_box_xyxy[1] + pred_box_xyxy[3]) / 2.0
            )

            # Draw restored box on original frame
            cv2.rectangle(
                frame,
                (int(pred_box_xyxy[0]), int(pred_box_xyxy[1])),
                (int(pred_box_xyxy[2]), int(pred_box_xyxy[3])),
                (0, 255, 0), 2
            )

            # Restore Tail Keypoint
            if r.keypoints is not None and len(r.keypoints.data) > 0:
                kpts = r.keypoints.data[0].cpu().numpy()
                tail = kpts[TAILSET_INDEX]

                if tail[2] >= TAIL_CONF_THRESHOLD:
                    tail_detected_count += 1
                    # X is untouched; Y gets y_start added
                    pred_tail_pt = (float(tail[0]), float(tail[1] + y_start))
                    raw_tail_trajectory.append((int(pred_tail_pt[0]), int(pred_tail_pt[1])))

                    tail_entry = {
                        "frame_number": frame_idx,
                        "timestamp_seconds": round(frame_idx / fps, 3),
                        "tail_x_px": round(pred_tail_pt[0], 2),
                        "tail_y_px": round(pred_tail_pt[1], 2),
                        "confidence": round(float(tail[2]), 3),
                        "track_id": track_id
                    }

                    # If Ground Truth exists, compute pixel error for this frame
                    if gt_file:
                        with open(gt_file, "r") as gf:
                            gt_parts = gf.readline().strip().split()
                        if len(gt_parts) >= 11:
                            gt_tx = float(gt_parts[8]) * width
                            gt_ty = float(gt_parts[9]) * height
                            error_px = math.sqrt((pred_tail_pt[0] - gt_tx)**2 + (pred_tail_pt[1] - gt_ty)**2)
                            tail_entry["gt_tail_x_px"] = round(gt_tx, 2)
                            tail_entry["gt_tail_y_px"] = round(gt_ty, 2)
                            tail_entry["tail_error_px"] = round(error_px, 2)

                    tail_json_data.append(tail_entry)

        # Evaluate Against Ground Truth (Both in 1920x1080 space)
        if gt_file:
            with open(gt_file, "r") as gf:
                gt_parts = gf.readline().strip().split()
            if len(gt_parts) >= 11:
                gt_evaluated_frames += 1

                if pred_box_xyxy is not None:
                    gt_tracked_frames += 1

                # GT Box
                gt_cx = float(gt_parts[1]) * width
                gt_cy = float(gt_parts[2]) * height
                gt_bw = float(gt_parts[3]) * width
                gt_bh = float(gt_parts[4]) * height
                gt_box_xyxy = [gt_cx - gt_bw / 2, gt_cy - gt_bh / 2, gt_cx + gt_bw / 2, gt_cy + gt_bh / 2]

                if pred_box_xyxy is not None:
                    box_ious.append(compute_iou(pred_box_xyxy, gt_box_xyxy))
                    box_center_errors.append(math.sqrt((pred_box_center[0] - gt_cx)**2 + (pred_box_center[1] - gt_cy)**2))

                # GT Tail
                gt_tx = float(gt_parts[8]) * width
                gt_ty = float(gt_parts[9]) * height
                gt_t_vis = float(gt_parts[10])

                if pred_tail_pt is not None and gt_t_vis > 0:
                    gt_valid_tail_frames += 1
                    tail_pixel_errors.append(math.sqrt((pred_tail_pt[0] - gt_tx)**2 + (pred_tail_pt[1] - gt_ty)**2))

        # Draw Full-Frame Ballistic Flight Arc & Contact Point
        parabola_curve, contact_point = fit_parabolic_trajectory(raw_tail_trajectory, width, height, min_points=6)

        if parabola_curve and len(parabola_curve) > 1:
            curve_arr = np.array(parabola_curve, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [curve_arr], isClosed=False, color=(0, 255, 0), thickness=3)

        if len(raw_tail_trajectory) > 1:
            raw_arr = np.array(raw_tail_trajectory, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [raw_arr], isClosed=False, color=(0, 255, 255), thickness=4)

        if len(raw_tail_trajectory) > 0:
            cv2.circle(frame, raw_tail_trajectory[-1], 7, (0, 0, 255), -1)

        if contact_point:
            cx, cy = contact_point
            cv2.circle(frame, (cx, cy), 12, (0, 0, 255), 2)
            cv2.drawMarker(frame, (cx, cy), (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=18, thickness=2)

        # Write Original 1080p Frame to Output Video
        writer.write(frame)

        if frame_idx % 30 == 0:
            print(f"  Processed {frame_idx}/{total_frames} frames...", end="\r")

    cap.release()
    writer.release()

    # ----------------- SUMMARY METRICS -----------------
    avg_fps = len(inference_times) / sum(inference_times) if inference_times else 0.0
    avg_latency = (sum(inference_times) / len(inference_times)) * 1000 if inference_times else 0.0

    tracking_continuity_pct = (gt_tracked_frames / gt_evaluated_frames * 100) if gt_evaluated_frames else (dog_detected_count / total_frames * 100)
    valid_tail_pct = (gt_valid_tail_frames / gt_evaluated_frames * 100) if gt_evaluated_frames else (tail_detected_count / total_frames * 100)

    mean_tail_mae = np.mean(tail_pixel_errors) if tail_pixel_errors else 0.0
    mean_box_iou = np.mean(box_ious) * 100 if box_ious else 0.0
    mean_box_mae = np.mean(box_center_errors) if box_center_errors else 0.0

    # ----------------- JSON EXPORT -----------------
    export_payload = {
        "video_file": VIDEO_PATH.name,
        "video_fps": round(fps, 2),
        "resolution_px": [width, height],
        "metrics_summary": {
            "total_video_frames": total_frames,
            "ground_truth_annotated_frames": gt_evaluated_frames,
            "airborne_tracking_continuity_pct": round(tracking_continuity_pct, 2),
            "valid_tail_base_detection_pct": round(valid_tail_pct, 2),
            "tail_base_mae_px": round(mean_tail_mae, 2),
            "bounding_box_mean_iou_pct": round(mean_box_iou, 2),
            "box_center_mae_px": round(mean_box_mae, 2),
            "average_fps": round(avg_fps, 2),
            "average_latency_ms": round(avg_latency, 2)
        },
        "tail_detections": tail_json_data
    }

    with open(out_json_path, "w") as f:
        json.dump(export_payload, f, indent=4)

    # ----------------- CONSOLE REPORT -----------------
    print("\n" + "="*62)
    print("="*62)
    print(f"Video File Evaluated          : {VIDEO_PATH.name}")
    print(f"Ground Truth Frames Checked   : {gt_evaluated_frames}")
    print("-" * 62)
    print(f"   Tracking Continuity        : {tracking_continuity_pct:.2f}%")
    print(f"   Valid Tail-Base Detections : {valid_tail_pct:.2f}%")
    print(f"   Tail-Base Localization MAE : {mean_tail_mae:.2f} px")
    print(f"   Bounding Box Mean IoU      : {mean_box_iou:.2f}%")
    print(f"   Box Center Location Error  : {mean_box_mae:.2f} px")
    print(f"   Model Inference Throughput : {avg_fps:.2f} FPS")
    print(f"   Average Latency per Frame  : {avg_latency:.2f} ms")
    print("-" * 62)
    print(f"Output Video saved to      : {out_video_path}")
    print(f"Tail Position JSON saved to: {out_json_path}")
    print("="*62)

if __name__ == "__main__":
    main()