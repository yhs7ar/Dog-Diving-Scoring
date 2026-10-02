import os
import cv2
import numpy as np
import torch
from pathlib import Path
from ultralytics import YOLO

# Bypass MIOpen issue on AMD Windows
torch.backends.cudnn.enabled = False

def apply_clahe(frame, clip_limit=2.5, tile_grid_size=(8, 8)):
    """
    Locally enhances contrast on dark subjects without blowing out bright backgrounds.
    """
    # Convert BGR to LAB color space to isolate Luminance (L) from color channels
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Apply CLAHE to the Luminance channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_enhanced = clahe.apply(l)

    # Merge channels back and convert to BGR
    enhanced_lab = cv2.merge((l_enhanced, a, b))
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

def process_video(video_path, output_path, model):
    print(f"\nProcessing trajectory (with CLAHE) for: {video_path.name}")
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    # In the 2-keypoint model: 0 = Nose, 1 = Tailset
    TAILSET_INDEX = 1
    trajectory_points = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        # Apply CLAHE contrast enhancement
        enhanced_frame = apply_clahe(frame, clip_limit=2.5)

        # Run tracking on the enhanced frame
        # 'persist=True' maintains ByteTrack IDs across frame-by-frame loop
        results = model.track(
            source=enhanced_frame,
            imgsz=1920,
            conf=0.12,          # Kept low to catch dark silhouettes
            max_det=1,
            tracker="bytetrack.yaml",
            persist=True,       # Keeps tracker state active between frames
            device=0,
            verbose=False
        )

        r = results[0]

        # Draw annotations
        # Passing img=frame draws boxes & keypoints on the NATURAL-colored frame
        annotated_frame = r.plot(img=frame)

        # Extract tailset keypoint
        if r.keypoints is not None and len(r.keypoints.data) > 0:
            kpts = r.keypoints.data[0].cpu().numpy()
            tail = kpts[TAILSET_INDEX]  # [x, y, confidence]

            if tail[2] > 0.3:
                pt = (int(tail[0]), int(tail[1]))
                trajectory_points.append(pt)

        # Draw trajectory flight arc
        if len(trajectory_points) > 1:
            pts_array = np.array(trajectory_points, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated_frame, [pts_array], isClosed=False, color=(0, 255, 255), thickness=4)

        if len(trajectory_points) > 0:
            cv2.circle(annotated_frame, trajectory_points[-1], 7, (0, 0, 255), -1)

        writer.write(annotated_frame)

        if frame_count % 60 == 0:
            print(f"Processed {frame_count}/{total_frames} frames...", end="\r")

    cap.release()
    writer.release()
    print(f"\nSaved trajectory video to: {output_path}")

def main():
    torch.backends.cudnn.enabled = False

    # Paths
    model_path = r"trained_models\dog_pose_s_2kpt_50epochs_720p_finetune\weights\best.pt"
    input_folder = Path(r"I:\Capstone\Dog-Diving-Scoring\Initial Dog diving videos")
    output_folder = Path(r"I:\Capstone\Dog-Diving-Scoring\output_videos\tail_trajectories_clahe")
    output_folder.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {model_path}...")
    model = YOLO(model_path)

    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    video_files = [f for f in input_folder.iterdir() if f.suffix.lower() in video_extensions]

    print(f"Found {len(video_files)} videos to process.")
    for vid in video_files:
        out_file = output_folder / f"trajectory_{vid.name}"
        process_video(vid, out_file, model)

    print("\nAll trajectories with CLAHE generated successfully!")

if __name__ == "__main__":
    main()