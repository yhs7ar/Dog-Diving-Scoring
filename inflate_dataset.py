import os
import cv2
import numpy as np
from pathlib import Path

# Path to master dataset
DATASET_DIR = Path(r"I:\Capstone\Dog-Diving-Scoring\dog-pose-2kpt")
TRAIN_IMG_DIR = DATASET_DIR / "images" / "train"
TRAIN_LBL_DIR = DATASET_DIR / "labels" / "train"

# Prefixes of images to inflate (targets the diving dogs)
TARGET_PREFIXES = ("custom_", "paws_")

def flip_label_line(line):
    """
    Horizontally mirrors a YOLO pose annotation line.
    Format: cls cx cy w h nose_x nose_y nose_v tail_x tail_y tail_v
    """
    parts = line.strip().split()
    if len(parts) < 11:
        return line

    cls = parts[0]
    cx = 1.0 - float(parts[1])  # Mirror bbox center X
    cy = float(parts[2])
    w = float(parts[3])
    h = float(parts[4])

    # Nose (Index 0)
    nx = 1.0 - float(parts[5]) if float(parts[7]) > 0 else float(parts[5])
    ny = float(parts[6])
    nv = parts[7]

    # Tailset (Index 1)
    tx = 1.0 - float(parts[8]) if float(parts[10]) > 0 else float(parts[8])
    ty = float(parts[9])
    tv = parts[10]

    return f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {nx:.6f} {ny:.6f} {nv} {tx:.6f} {ty:.6f} {tv}\n"

def adjust_hsv(img, h_shift=10, s_scale=1.1, v_scale=1.0):
    """Adjusts color hue, saturation, and exposure."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 0] = (hsv[:, :, 0] + h_shift) % 180
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * s_scale, 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * v_scale, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

def main():
    print(f"Starting dataset inflation in: {TRAIN_IMG_DIR}")

    # Gather images to augment
    all_imgs = [
        p for p in TRAIN_IMG_DIR.glob("*.*")
        if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
        and p.name.startswith(TARGET_PREFIXES)
        and not any(tag in p.stem for tag in ["_flip", "_dark", "_bright", "_contrast"])
    ]

    print(f"Found {len(all_imgs)} custom diving images eligible for inflation.")
    if len(all_imgs) == 0:
        print("No eligible images found. Check folder paths or prefix filters.")
        return

    created_count = 0

    for idx, img_path in enumerate(all_imgs):
        lbl_path = TRAIN_LBL_DIR / f"{img_path.stem}.txt"
        if not lbl_path.exists():
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        with open(lbl_path, "r") as f:
            orig_labels = f.readlines()

        flipped_labels = [flip_label_line(line) for line in orig_labels]

        # ---------------- VARIATION 1: Horizontal Flip (Mirror Jump Direction) ----------------
        flip_name = f"{img_path.stem}_flip"
        cv2.imwrite(str(TRAIN_IMG_DIR / f"{flip_name}{img_path.suffix}"), cv2.flip(img, 1))
        with open(TRAIN_LBL_DIR / f"{flip_name}.txt", "w") as f:
            f.writelines(flipped_labels)
        created_count += 1

        # ---------------- VARIATION 2: Shadow / Dark Coat Exposure ----------------
        # Simulates shadowed docks & underexposed dark fur
        dark_img = cv2.convertScaleAbs(img, alpha=0.82, beta=-35)
        dark_name = f"{img_path.stem}_dark"
        cv2.imwrite(str(TRAIN_IMG_DIR / f"{dark_name}{img_path.suffix}"), dark_img)
        with open(TRAIN_LBL_DIR / f"{dark_name}.txt", "w") as f:
            f.writelines(orig_labels)
        created_count += 1

        # ---------------- VARIATION 3: High Sunlight / Water Glare ----------------
        # Simulates midday harsh lighting & reflection wash-out
        bright_img = cv2.convertScaleAbs(img, alpha=1.18, beta=35)
        bright_name = f"{img_path.stem}_bright"
        cv2.imwrite(str(TRAIN_IMG_DIR / f"{bright_name}{img_path.suffix}"), bright_img)
        with open(TRAIN_LBL_DIR / f"{bright_name}.txt", "w") as f:
            f.writelines(orig_labels)
        created_count += 1

        # ---------------- VARIATION 4: Flipped + Contrast / Murky Water Shift ----------------
        # Combines mirror jump direction with contrast and color temperature variance
        contrast_img = cv2.convertScaleAbs(cv2.flip(img, 1), alpha=1.35, beta=-15)
        color_shifted = adjust_hsv(contrast_img, h_shift=8, s_scale=1.15)
        contrast_name = f"{img_path.stem}_flip_contrast"
        cv2.imwrite(str(TRAIN_IMG_DIR / f"{contrast_name}{img_path.suffix}"), color_shifted)
        with open(TRAIN_LBL_DIR / f"{contrast_name}.txt", "w") as f:
            f.writelines(flipped_labels)
        created_count += 1

        if (idx + 1) % 50 == 0 or idx == len(all_imgs) - 1:
            print(f"  Augmented [{idx + 1}/{len(all_imgs)}] base frames ({created_count} new images created)...")

    # Wipe cache files so YOLO rescans the enlarged training set
    print("\nClearing old YOLO cache files...")
    for cache_file in [DATASET_DIR / "labels" / "train.cache", DATASET_DIR / "labels" / "val.cache"]:
        if cache_file.exists():
            os.remove(cache_file)
            print(f"  Removed: {cache_file.name}")

    print("\n" + "="*50)
    print("           DATASET INFLATION COMPLETE             ")
    print("="*50)
    print(f"Original Diving Images Processed : {len(all_imgs)}")
    print(f"New Augmented Images Created     : {created_count}")
    print(f"Total Added to Training Split    : {created_count} frames (~5x multiplier)")
    print("="*50)

if __name__ == "__main__":
    main()