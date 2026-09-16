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
        epochs=20,
        imgsz=640,
        device=0
    )

if __name__ == '__main__':
    main()