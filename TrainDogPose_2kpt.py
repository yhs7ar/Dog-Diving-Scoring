import os
import torch

# Prevent VRAM fragmentation
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

# Bypass MIOpen runtime kernel compilation on AMD Windows
torch.backends.cudnn.enabled = False

from ultralytics import YOLO

def main():
    torch.backends.cudnn.enabled = False

    # Load base pretrained model
    model = YOLO("yolo26s-pose.pt")

    # Train on your 2-keypoint dataset
    results = model.train(
        data=r"I:\Capstone\Dog-Diving-Scoring\dog-pose-2kpt\dog-pose-2kpt.yaml",
        epochs=30,
        imgsz=1280,
        batch=8,
        workers=2,
        device=0,
        project=r"I:\Capstone\Dog-Diving-Scoring\trained_models",
        name="dog_pose_s_2kpt_30epochs_720p_basedata",
        exist_ok=True,
        cache=True
    )

if __name__ == '__main__':
    main()