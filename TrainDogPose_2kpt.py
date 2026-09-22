import torch

# Keep this for AMD Windows GPU
torch.backends.cudnn.enabled = False

from ultralytics import YOLO

def main():
    torch.backends.cudnn.enabled = False

    # Load base pretrained model
    model = YOLO("yolo26s-pose.pt")

    # Train on your new 2-keypoint dataset
    results = model.train(
        data=r"I:\Capstone\Dog-Diving-Scoring\dog-pose-2kpt\dog-pose-2kpt.yaml",
        epochs=50,
        imgsz=1920,
        device=0,
        batch=4,
        project=r"I:\Capstone\Dog-Diving-Scoring\trained_models",
        name="dog_pose_2kpt_50epochs_1080p_basedata",
        exist_ok=True
    )

if __name__ == '__main__':
    main()