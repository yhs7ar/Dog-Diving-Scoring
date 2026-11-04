from ultralytics import YOLO
from dog_diving.config import MODEL_PATH, DEFAULT_IMGSZ
from pathlib import Path

def export():
    # Ensure the model file exists
    path = Path(MODEL_PATH)
    if not path.is_file():
        print(f"Checkpoint not found at: {path}")
        return

    print(f"Exporting {path} to TensorRT engine...")
    # The export creates the .engine file in the same directory as the .pt model
    model = YOLO(str(path))
    model.export(format='engine', device='0', imgsz=DEFAULT_IMGSZ)
    print("Export complete.")

if __name__ == "__main__":
    export()
