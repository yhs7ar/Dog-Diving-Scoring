import os
import sys
import math
import cv2
import numpy as np
import torch
from pathlib import Path

# Force immediate console output
sys.stdout.reconfigure(line_buffering=True)

print("[1/5] Initializing script and checking environment...", flush=True)

# AMD Windows fix
torch.backends.cudnn.enabled = False

from ultralytics import YOLO

# ----------------- ABSOLUTE PATHS -----------------
BASE_DIR = Path(r"I:\Capstone\Dog-Diving-Scoring")
VAL_IMG_DIR = BASE_DIR / "dog-pose-2kpt" / "images" / "val"
VAL_LBL_DIR = BASE_DIR / "dog-pose-2kpt" / "labels" / "val"

# Check if model exists in finetune folder, otherwise fallback to base
MODEL_PATH = BASE_DIR / "trained_models" / "dog_pose_s_2kpt_clahe_model" / "weights" / "best.pt"
if not MODEL_PATH.exists():
    MODEL_PATH = BASE_DIR / "trained_models" / "dog_pose_2kpt_finetuned" / "weights" / "best.pt"

PIXELS_PER_INCH = 20.0
TAILSET_INDEX = 1       # 0 = Nose, 1 = Tailset
# --------------------------------------------------

def compute_iou(box1, box2):
    """
    Computes Intersection over Union (IoU) between two boxes [x1, y1, x2, y2].
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0

def main():
    print(f"[2/5] Checking file paths...", flush=True)
    print(f"  - Images Path: {VAL_IMG_DIR}")
    print(f"  - Labels Path: {VAL_LBL_DIR}")
    print(f"  - Model Path : {MODEL_PATH}")

    if not VAL_IMG_DIR.exists():
        print(f"\nERROR: Image directory does not exist: {VAL_IMG_DIR}")
        return
    if not VAL_LBL_DIR.exists():
        print(f"\nERROR: Label directory does not exist: {VAL_LBL_DIR}")
        return
    if not MODEL_PATH.exists():
        print(f"\nERROR: Model weights file not found at: {MODEL_PATH}")
        return

    image_extensions = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG")
    images = []
    for ext in image_extensions:
        images.extend(list(VAL_IMG_DIR.glob(ext)))

    print(f"Found {len(images)} validation images.", flush=True)
    if len(images) == 0:
        print("ERROR: No images found in the validation folder!")
        return

    print(f"[3/5] Loading YOLO model into GPU...", flush=True)
    model = YOLO(str(MODEL_PATH))
    print("Model loaded successfully!", flush=True)

    print(f"[4/5] Calculating Bounding Box & Tailset MAE across validation frames...\n", flush=True)

    tail_pixel_errors = []
    box_center_errors = []
    box_ious = []
    evaluated_count = 0

    for idx, img_path in enumerate(images):
        lbl_path = VAL_LBL_DIR / f"{img_path.stem}.txt"
        if not lbl_path.exists():
            continue

        with open(lbl_path, "r") as f:
            lines = f.readlines()
        if not lines:
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        parts = lines[0].strip().split()
        if len(parts) < 11:
            continue

        # Ground Truth Bounding Box (Normalized -> Pixels)
        gt_cx = float(parts[1]) * w
        gt_cy = float(parts[2]) * h
        gt_bw = float(parts[3]) * w
        gt_bh = float(parts[4]) * h
        gt_box_xyxy = [gt_cx - gt_bw / 2, gt_cy - gt_bh / 2, gt_cx + gt_bw / 2, gt_cy + gt_bh / 2]

        # Ground Truth Tailset
        gt_tail_x = float(parts[8]) * w
        gt_tail_y = float(parts[9]) * h
        gt_tail_vis = float(parts[10])

        # Run Prediction
        results = model.predict(source=img, imgsz=1280, conf=0.15, max_det=1, device=0, verbose=False)
        r = results[0]

        # Evaluate Bounding Box
        if r.boxes is not None and len(r.boxes) > 0:
            pred_xyxy = r.boxes.xyxy[0].cpu().numpy()
            pred_xywh = r.boxes.xywh[0].cpu().numpy()

            # Bounding Box IoU
            iou = compute_iou(pred_xyxy, gt_box_xyxy)
            box_ious.append(iou)

            # Bounding Box Center Error
            center_err = math.sqrt((pred_xywh[0] - gt_cx)**2 + (pred_xywh[1] - gt_cy)**2)
            box_center_errors.append(center_err)

        # Evaluate Tailset Keypoint
        if r.keypoints is not None and len(r.keypoints.data) > 0:
            kpts = r.keypoints.data[0].cpu().numpy()
            pred_tail = kpts[TAILSET_INDEX]

            if pred_tail[2] > 0.2 and gt_tail_vis > 0:
                err_px = math.sqrt((pred_tail[0] - gt_tail_x)**2 + (pred_tail[1] - gt_tail_y)**2)
                tail_pixel_errors.append(err_px)

        evaluated_count += 1

        if (idx + 1) % 100 == 0 or idx == len(images) - 1:
            avg_iou = np.mean(box_ious) * 100 if box_ious else 0.0
            avg_tail = np.mean(tail_pixel_errors) if tail_pixel_errors else 0.0
            print(f"  Progress: [{idx + 1}/{len(images)}] | Mean IoU: {avg_iou:.1f}% | Tail MAE: {avg_tail:.2f} px", flush=True)

    # Final Metrics
    mean_iou = np.mean(box_ious) * 100 if box_ious else 0.0
    box_center_mae_px = np.mean(box_center_errors) if box_center_errors else 0.0
    box_center_mae_in = box_center_mae_px / PIXELS_PER_INCH

    tail_mae_px = np.mean(tail_pixel_errors) if tail_pixel_errors else 0.0
    tail_mae_in = tail_mae_px / PIXELS_PER_INCH

    print("\n" + "="*58)
    print("      MILESTONE 1: VALIDATION QUANTITATIVE RESULTS        ")
    print("="*58)
    print(f"Total Annotated Frames Evaluated : {evaluated_count}")
    print("-" * 58)
    print("BOUNDING BOX LOCALIZATION (DOG DETECTION):")
    print(f"  - Mean Intersection over Union (IoU) : {mean_iou:.2f}%")
    print(f"  - Box Center Error (MAE)            : {box_center_mae_px:.2f} px ({box_center_mae_in:.2f} inches)")
    print("-" * 58)
    print("TAIL-BASE KEYPOINT LOCALIZATION:")
    print(f"  - Tail-Base Error (MAE)             : {tail_mae_px:.2f} pixels")
    print(f"  - Physical Real-World Error         : {tail_mae_in:.2f} inches")
    print("="*58)

if __name__ == "__main__":
    main()