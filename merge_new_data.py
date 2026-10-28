import os
import shutil
from pathlib import Path
import random

# Path to the NEW folder from CVAT / Roboflow
NEW_DATA_DIR = Path(r"I:\Capstone\Dog-Diving-Scoring\paws_to_swim_Pose_10-2-2026")

# Path to your master 2-keypoint dataset
TARGET_DATASET = Path(r"I:\Capstone\Dog-Diving-Scoring\dog-pose-2kpt")

# Unique prefix to prevent overwriting existing merged images (e.g., custom_frame_001.jpg)
DATASET_PREFIX = "paws"

def merge():
    print(f"Starting dataset merge from: {NEW_DATA_DIR.name}")
    
    if not NEW_DATA_DIR.exists():
        print(f"Error: Directory not found: {NEW_DATA_DIR}")
        return

    # Index all .txt label files by filename (ignoring list files like train.txt)
    ignore_files = {"train.txt", "val.txt", "test.txt", "README.txt"}
    label_map = {}
    for txt_file in NEW_DATA_DIR.rglob("*.txt"):
        if txt_file.name not in ignore_files:
            label_map[txt_file.stem] = txt_file
            
    print(f"Found {len(label_map)} label (.txt) files.")

    # Find all images
    supported_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    all_images = []
    for ext in supported_extensions:
        all_images.extend(list(NEW_DATA_DIR.rglob(f"*{ext}")))
    
    print(f"Found {len(all_images)} image files.")

    # Match images with their corresponding label file
    matched_pairs = []
    for img_path in all_images:
        if img_path.stem in label_map:
            matched_pairs.append((img_path, label_map[img_path.stem]))

    print(f"Successfully matched {len(matched_pairs)} image-label pairs!")

    if len(matched_pairs) == 0:
        print("\nError: Could not match image filenames to .txt filenames.")
        if all_images:
            print(f"  Sample image path: {all_images[0]}")
        if label_map:
            sample_stem = next(iter(label_map.keys()))
            print(f"  Sample label path: {label_map[sample_stem]}")
        return

    # Shuffle and split 85% train / 15% val
    random.seed(42)
    random.shuffle(matched_pairs)
    
    split_index = int(len(matched_pairs) * 0.85)
    train_pairs = matched_pairs[:split_index]
    val_pairs = matched_pairs[split_index:]

    for split_name, pairs in [("train", train_pairs), ("val", val_pairs)]:
        dst_img_dir = TARGET_DATASET / "images" / split_name
        dst_lbl_dir = TARGET_DATASET / "labels" / split_name
        
        dst_img_dir.mkdir(parents=True, exist_ok=True)
        dst_lbl_dir.mkdir(parents=True, exist_ok=True)
        
        copied_count = 0
        for img_path, lbl_path in pairs:
            # Use unique prefix to guarantee no overwrites
            unique_name = f"{DATASET_PREFIX}_{img_path.stem}"
            
            # Copy Image
            shutil.copy2(img_path, dst_img_dir / f"{unique_name}{img_path.suffix}")
            # Copy Label
            shutil.copy2(lbl_path, dst_lbl_dir / f"{unique_name}.txt")
            copied_count += 1
            
        print(f"Added {copied_count} samples to {split_name} split.")

    # Clear old cache files so YOLO rescans
    print("\nClearing old cache files...")
    for cache_file in [TARGET_DATASET / "labels" / "train.cache", TARGET_DATASET / "labels" / "val.cache"]:
        if cache_file.exists():
            os.remove(cache_file)
            print(f"  Removed: {cache_file}")

    print("\nMerge complete! Your dataset is updated and ready.")

if __name__ == "__main__":
    merge()