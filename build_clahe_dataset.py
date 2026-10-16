import os
import cv2
import shutil
from pathlib import Path

# Paths
SRC_DATASET = Path(r"I:\Capstone\Dog-Diving-Scoring\dog-pose-2kpt")
DST_DATASET = Path(r"I:\Capstone\Dog-Diving-Scoring\dog-pose-2kpt-clahe")

def apply_clahe(img, clip_limit=2.5, tile_grid_size=(8, 8)):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_enhanced = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l_enhanced, a, b)), cv2.COLOR_LAB2BGR)

def process_split(split):
    src_img_dir = SRC_DATASET / "images" / split
    dst_img_dir = DST_DATASET / "images" / split
    dst_img_dir.mkdir(parents=True, exist_ok=True)

    img_files = list(src_img_dir.glob("*.jpg")) + list(src_img_dir.glob("*.png"))
    total = len(img_files)
    print(f"[*] Processing CLAHE for {split} split ({total} images)...")

    for idx, img_path in enumerate(img_files):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        
        # Apply CLAHE
        enhanced = apply_clahe(img)
        
        # Save to new dataset
        cv2.imwrite(str(dst_img_dir / img_path.name), enhanced)

        if (idx + 1) % 500 == 0 or idx == total - 1:
            print(f"  Processed {idx + 1}/{total} images...", end="\r")
    print(f"\n[+] Finished {split} images.")

def main():
    print("[*] Creating CLAHE-Enhanced Dataset...")

    # Process images for train and val
    for split in ["train", "val"]:
        process_split(split)

    # Copy labels directly (coordinates do not change)
    print("[*] Copying label text files...")
    src_lbl_dir = SRC_DATASET / "labels"
    dst_lbl_dir = DST_DATASET / "labels"
    
    if dst_lbl_dir.exists():
        shutil.rmtree(dst_lbl_dir)
    shutil.copytree(src_lbl_dir, dst_lbl_dir)

    # Remove old .cache files so YOLO generates fresh ones
    for cache in dst_lbl_dir.glob("*.cache"):
        os.remove(cache)
        print(f"[+] Removed old cache: {cache.name}")

    # Generate new dataset YAML
    yaml_content = f"""path: {DST_DATASET.as_posix()}
train: images/train
val: images/val

kpt_shape: [2, 3]

names:
  0: dog
"""
    yaml_path = DST_DATASET / "dog-pose-2kpt-clahe.yaml"
    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    print(f"\n[+] CLAHE Dataset created successfully at: {DST_DATASET}")
    print(f"[+] Config YAML written to: {yaml_path}")

if __name__ == "__main__":
    main()