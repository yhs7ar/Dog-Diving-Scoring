import os
import shutil
from pathlib import Path

SOURCE_DATASET = Path(r"I:\Capstone\Dog-Diving-Scoring\dog-pose")
TARGET_DATASET = Path(r"I:\Capstone\Dog-Diving-Scoring\dog-pose-2kpt")

TAIL_START_IDX = 12
NOSE_IDX = 16

def convert_label_file(src_path, dst_path):
    with open(src_path, "r") as f:
        lines = f.readlines()
    
    new_lines = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        
        cls = parts[0]
        bbox = parts[1:5]  # [x_center, y_center, width, height]
        kpts = parts[5:]   # 24 * 3 coordinates (x, y, visibility)
        
        if len(kpts) < 24 * 3:
            continue
        
        # Extract Nose (Point 0 in new model)
        nose = kpts[NOSE_IDX * 3 : NOSE_IDX * 3 + 3]
        
        # Extract Tail-Start (Point 1 in new model)
        tail = kpts[TAIL_START_IDX * 3 : TAIL_START_IDX * 3 + 3]
        
        # Assemble new 2-keypoint line: [class, bbox(4), nose(3), tail(3)]
        new_line = f"{cls} " + " ".join(bbox) + " " + " ".join(nose) + " " + " ".join(tail) + "\n"
        new_lines.append(new_line)
        
    with open(dst_path, "w") as f:
        f.writelines(new_lines)

def main():
    print("Creating 2-Keypoint Dataset (Nose & Tailset)...")
    
    for split in ["train", "val"]:
        src_label_dir = SOURCE_DATASET / "labels" / split
        dst_label_dir = TARGET_DATASET / "labels" / split
        dst_label_dir.mkdir(parents=True, exist_ok=True)
        
        # Link or copy images
        src_img_dir = SOURCE_DATASET / "images" / split
        dst_img_dir = TARGET_DATASET / "images" / split
        dst_img_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert label files
        label_files = list(src_label_dir.glob("*.txt"))
        for file in label_files:
            convert_label_file(file, dst_label_dir / file.name)
            
        print(f"Processed {len(label_files)} labels for {split} split.")

    # Create the custom dataset YAML configuration
    yaml_content = f"""path: {TARGET_DATASET.as_posix()}
train: images/train
val: images/val

# Keypoints: [Nose, Tailset]
kpt_shape: [2, 3]  # 2 keypoints, 3 values each (x, y, visibility)

names:
  0: dog
"""
    yaml_path = TARGET_DATASET / "dog-pose-2kpt.yaml"
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
        
    print(f"\nFinished! YAML config created at: {yaml_path}")

if __name__ == "__main__":
    main()