import os
import cv2
import numpy as np
import torch
from pathlib import Path
from ultralytics import YOLO

# Bypass MIOpen issue on AMD Windows
torch.backends.cudnn.enabled = False

def apply_clahe(frame, clip_limit=2.5, tile_grid_size=(8, 8)):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced_lab = cv2.merge((clahe.apply(l), a, b))
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

def fit_parabolic_trajectory(points, frame_width, frame_height, min_points=6):
    """
    Fits a quadratic curve y = A*x^2 + B*x + C and extrapolates 
    ahead to the edge of the video (water contact).
    """
    if len(points) < min_points:
        return None, None

    x = np.array([p[0] for p in points], dtype=np.float64)
    y = np.array([p[1] for p in points], dtype=np.float64)

    # Need sufficient horizontal motion to fit a stable curve
    if np.max(x) - np.min(x) < 40:
        return None, None

    try:
        # Fit 2nd-degree polynomial (y = Ax^2 + Bx + C)
        poly_coeffs = np.polyfit(x, y, 2)
        parabola = np.poly1d(poly_coeffs)
    except Exception:
        return None, None

    # Determine horizontal flight direction (left-to-right vs right-to-left)
    dx = x[-1] - x[0]
    step = 3 if dx >= 0 else -3

    # Extrapolate forward from the launch point to the edge of the video
    start_x = int(x[0])
    curve_points = []
    contact_point = None

    curr_x = start_x
    while 0 <= curr_x < frame_width:
        curr_y = parabola(curr_x)

        # Stop when the trajectory reaches the bottom edge of the frame (water surface)
        if curr_y >= frame_height - 1:
            contact_point = (int(curr_x), int(frame_height - 1))
            curve_points.append(contact_point)
            break

        # Record valid in-frame points
        if curr_y >= 0:
            curve_points.append((int(curr_x), int(curr_y)))

        curr_x += step

    if len(curve_points) > 1 and contact_point is None:
        contact_point = curve_points[-1]

    return curve_points, contact_point

def process_video(video_path, output_path, model):
    print(f"\nProcessing ballistic trajectory for: {video_path.name}")
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    TAILSET_INDEX = 1
    raw_tail_points = []
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # CLAHE contrast enhancement for dark dogs
        enhanced_frame = apply_clahe(frame, clip_limit=2.5)

        # Tracking
        results = model.track(
            source=enhanced_frame,
            imgsz=1280,
            conf=0.15,
            max_det=1,
            tracker="bytetrack.yaml",
            persist=True,
            device=0,
            verbose=False
        )

        r = results[0]
        annotated_frame = r.plot(img=frame)

        # Extract Tailset
        if r.keypoints is not None and len(r.keypoints.data) > 0:
            kpts = r.keypoints.data[0].cpu().numpy()
            tail = kpts[TAILSET_INDEX]

            if tail[2] > 0.3:
                raw_tail_points.append((int(tail[0]), int(tail[1])))

        # 4Fit & Extrapolate Quadratic Parabola
        # Uses points to calculate flight arc to the edge
        parabola_curve, contact_point = fit_parabolic_trajectory(
            raw_tail_points, width, height, min_points=4
        )

        # Draw the predicted parabolic trajectory (Neon Green arc)
        if parabola_curve and len(parabola_curve) > 1:
            curve_arr = np.array(parabola_curve, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated_frame, [curve_arr], isClosed=False, color=(0, 255, 0), thickness=3)

        # Draw the real tracked historical points (Yellow line)
        if len(raw_tail_points) > 1:
            raw_arr = np.array(raw_tail_points, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated_frame, [raw_arr], isClosed=False, color=(0, 255, 255), thickness=4)

        # Draw current tail point (Red circle)
        if len(raw_tail_points) > 0:
            cv2.circle(annotated_frame, raw_tail_points[-1], 7, (0, 0, 255), -1)

        # Highlight Predicted Water Contact Point (Target Crosshair)
        if contact_point:
            cx, cy = contact_point
            cv2.circle(annotated_frame, (cx, cy), 12, (0, 0, 255), 2)  # Red outer ring
            cv2.drawMarker(annotated_frame, (cx, cy), (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=18, thickness=2)
            cv2.putText(annotated_frame, f"Predicted Entry: X={cx}px", (cx - 80, cy - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        writer.write(annotated_frame)

        if frame_idx % 60 == 0:
            print(f"Processed {frame_idx}/{total_frames} frames...", end="\r")

    cap.release()
    writer.release()
    print(f"\nSaved ballistic trajectory to: {output_path}")

def main():
    torch.backends.cudnn.enabled = False

    model_path = r"trained_models\dog_pose_s_2kpt_clahe_model\weights\best.pt"
    input_folder = Path(r"I:\Capstone\Dog-Diving-Scoring\Initial Dog diving videos")
    output_folder = Path(r"I:\Capstone\Dog-Diving-Scoring\output_videos\parabolic_trajectories_clahe")
    output_folder.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {model_path}...")
    model = YOLO(model_path)

    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    video_files = [f for f in input_folder.iterdir() if f.suffix.lower() in video_extensions]

    for vid in video_files:
        out_file = output_folder / f"ballistic_{vid.name}"
        process_video(vid, out_file, model)

    print("\nAll ballistic trajectories generated successfully!")

if __name__ == "__main__":
    main()