import os
import torch

# Prevent memory fragmentation on AMD / Windows
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Bypass MIOpen issue on AMD Windows
torch.backends.cudnn.enabled = False

from ultralytics import YOLO

def main():
    torch.backends.cudnn.enabled = False

    # Point to the existing trained model
    base_trained_model = r"trained_models\dog_pose_s_2kpt_30epochs_720p_basedata\weights\best.pt"

    print(f"Loading existing model from: {base_trained_model}")
    model = YOLO(base_trained_model)

    # Fine-tune on the merged dataset for 15-20 epochs
    results = model.train(
        data=r"I:\Capstone\Dog-Diving-Scoring\dog-pose-2kpt\dog-pose-2kpt.yaml",
        epochs=20,
        imgsz=1280,
        batch=8,
        workers=2,
        device=0,
        project=r"I:\Capstone\Dog-Diving-Scoring\trained_models",
        name="dog_pose_s_2kpt_50epochs_720p_finetune",
        exist_ok=True
    )

if __name__ == '__main__':
    main()