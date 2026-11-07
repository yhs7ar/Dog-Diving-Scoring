import os
import torch

# Bypass MIOpen issue on AMD Windows
torch.backends.cudnn.enabled = False

from ultralytics import YOLO

def main():
    torch.backends.cudnn.enabled = False

    # Start from fresh base weights
    model = YOLO("yolo26s-pose.pt")

    results = model.train(
        data=r"I:\Capstone\Dog-Diving-Scoring\dog-pose-2kpt-clahe\dog-pose-2kpt-clahe.yaml",
        epochs=50,
        patience=12,
        close_mosaic=10,
        imgsz=1280,
        batch=12,
        workers=2,
        cache=False,
        device=0,
        project=r"I:\Capstone\Dog-Diving-Scoring\trained_models",
        name="dog_pose_s_2kpt_inflated_clahe_50epochs_720p",
        exist_ok=True
    )

if __name__ == '__main__':
    main()