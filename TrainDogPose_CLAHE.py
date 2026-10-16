import os
import torch

# Prevent memory fragmentation on AMD ROCm / Windows
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Bypass MIOpen issue on AMD Windows
torch.backends.cudnn.enabled = False

from ultralytics import YOLO

def main():
    torch.backends.cudnn.enabled = False

    # Start from fresh base weights
    model = YOLO("yolo26s-pose.pt")

    # Train on the CLAHE dataset
    results = model.train(
        data=r"I:\Capstone\Dog-Diving-Scoring\dog-pose-2kpt-clahe\dog-pose-2kpt-clahe.yaml",
        epochs=30,           # 30-35 epochs is optimal
        imgsz=1280,          # 1280 for fast training & great accuracy
        batch=8,
        workers=2,
        device=0,
        project=r"I:\Capstone\Dog-Diving-Scoring\trained_models",
        name="dog_pose_s_2kpt_clahe_model",
        exist_ok=True
    )

if __name__ == '__main__':
    main()