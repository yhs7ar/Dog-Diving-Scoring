from ultralytics import YOLO

# 1. Load a pre-trained YOLO pose model to start from
model = YOLO("yolo26n-pose.pt")  

# 2. Train the model on the Stanford Dog Pose dataset
# Ultralytics will automatically download the dataset when you specify 'dog-pose.yaml'
results = model.train(
    data="dog-pose/dog-pose.yaml", 
    epochs=100,            # Adjust based on your time/compute
    imgsz=640,             # Image size
    batch=16,              # Adjust based on your GPU VRAM
    device='cpu'               # Use device='cpu' if you don't have a GPU
)