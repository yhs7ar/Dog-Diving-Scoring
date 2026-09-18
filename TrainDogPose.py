import torch

# Bypass MIOpen runtime kernel compilation on AMD Windows
torch.backends.cudnn.enabled = False

from ultralytics import YOLO

def main():
    # Ensure it is disabled in the main worker thread
    torch.backends.cudnn.enabled = False

    # Load model
    model = YOLO("yolo26n-pose.pt")

    # Train model on your AMD GPU
    results = model.train(
        data="dog-pose.yaml",
        epochs=50,
        imgsz=1280,
        device="0",
        
        project=r"I:\Capstone\Dog-Diving-Scoring\trained_models",
        name="dog_pose_720p_50epochs",
        exist_ok=True
    )

if __name__ == '__main__':
    main()