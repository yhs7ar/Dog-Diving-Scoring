import torch

# Bypass MIOpen issue on AMD Windows
torch.backends.cudnn.enabled = False

from ultralytics import YOLO

def main():
    torch.backends.cudnn.enabled = False

    # Load 'last.pt' from your interrupted run
    checkpoint_path = r"I:\Capstone\Dog-Diving-Scoring\trained_models\dog_pose_s_2kpt_inflated_clahe_50epochs_720p\weights\last.pt"
    model = YOLO(checkpoint_path)

    # Resume training
    # Note: You do not need to pass data, imgsz, or batch size again! 
    # Ultralytics reads all original settings directly from last.pt.
    results = model.train(resume=True)

if __name__ == '__main__':
    main()