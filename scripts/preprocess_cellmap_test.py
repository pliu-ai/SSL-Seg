"""Preprocess CellMap test data (imagesTs/labelsTs) to match nnUNet 3d_lowres_large_patch npy format.

Replicates the nnUNet preprocessing pipeline:
  1. Read nii.gz
  2. Resample to target spacing (4.0, 4.0, 4.0)
  3. ZScore normalization (global foreground stats)
  4. Save as .npy + .pkl (metadata for reverse resampling at inference)

Usage:
    python scripts/preprocess_cellmap_test.py
"""

import os
import glob
import pickle
import numpy as np
import SimpleITK as sitk
from nnunetv2.preprocessing.resampling.default_resampling import resample_data_or_seg_to_shape

# ---------- config ----------
RAW_ROOT = "/projects/weilab/liupeng/projects/frameworks/nnUNet/DATASET/nnUNet_raw/Dataset200_Dataset101_CellMap_model_high_res"
IMAGES_TS = os.path.join(RAW_ROOT, "imagesTs")
LABELS_TS = os.path.join(RAW_ROOT, "labelsTs")

OUTPUT_DIR = "/projects/weilab/liupeng/projects/frameworks/nnUNet/DATASET/nnUNet_preprocessed/Dataset200_Dataset101_CellMap_model_high_res/nnUNetPlans_3d_lowres_large_patch_test"

TARGET_SPACING = np.array([4.0, 4.0, 4.0])

# Global foreground intensity stats from dataset_fingerprint.json (channel 0)
GLOBAL_MEAN = 162.47509765625
GLOBAL_STD = 42.84706115722656
# ---------- end config ----------


def compute_new_shape(old_shape, old_spacing, new_spacing):
    """Compute the target shape after resampling to new_spacing."""
    new_shape = np.round(
        (np.array(old_shape) * np.array(old_spacing)) / np.array(new_spacing)
    ).astype(int)
    return tuple(new_shape.tolist())


def preprocess_case(img_path, lbl_path, out_dir):
    case_name = os.path.basename(img_path).replace("_0000.nii.gz", "")
    print(f"Processing {case_name} ...")

    img_sitk = sitk.ReadImage(img_path)
    img_arr = sitk.GetArrayFromImage(img_sitk).astype(np.float32)
    original_spacing = np.array(img_sitk.GetSpacing())[::-1]  # sitk is xyz, we need zyx
    original_shape = img_arr.shape

    lbl_sitk = sitk.ReadImage(lbl_path)
    lbl_arr = sitk.GetArrayFromImage(lbl_sitk).astype(np.int8)

    # Compute target shape
    new_shape = compute_new_shape(original_shape, original_spacing, TARGET_SPACING)
    print(f"  original: shape={original_shape}, spacing={original_spacing}")
    print(f"  target:   shape={new_shape}, spacing={TARGET_SPACING}")

    # Resample image (add channel dim: (1, D, W, H))
    img_4d = img_arr[np.newaxis]
    img_resampled = resample_data_or_seg_to_shape(
        data=img_4d,
        new_shape=new_shape,
        current_spacing=original_spacing,
        new_spacing=TARGET_SPACING,
        is_seg=False,
        order=3,
        order_z=0,
        force_separate_z=None,
    )

    # Resample label
    lbl_4d = lbl_arr[np.newaxis].astype(np.float32)
    lbl_resampled = resample_data_or_seg_to_shape(
        data=lbl_4d,
        new_shape=new_shape,
        current_spacing=original_spacing,
        new_spacing=TARGET_SPACING,
        is_seg=True,
        order=1,
        order_z=0,
        force_separate_z=None,
    )
    lbl_resampled = lbl_resampled.astype(np.int8)

    # ZScore normalization
    img_resampled = (img_resampled - GLOBAL_MEAN) / max(GLOBAL_STD, 1e-8)
    img_resampled = img_resampled.astype(np.float32)

    print(f"  resampled: img={img_resampled.shape}, lbl={lbl_resampled.shape}, "
          f"lbl_unique={np.unique(lbl_resampled)}")

    # Save .npy
    np.save(os.path.join(out_dir, f"{case_name}.npy"), img_resampled)
    np.save(os.path.join(out_dir, f"{case_name}_seg.npy"), lbl_resampled)

    # Save .pkl metadata (for reverse-resampling predictions back to original space)
    props = {
        "sitk_stuff": {
            "spacing": tuple(img_sitk.GetSpacing()),
            "origin": tuple(img_sitk.GetOrigin()),
            "direction": tuple(img_sitk.GetDirection()),
        },
        "spacing": list(original_spacing),
        "shape_before_cropping": original_shape,
        "bbox_used_for_cropping": [[0, s] for s in original_shape],
        "shape_after_cropping_and_before_resampling": original_shape,
    }
    with open(os.path.join(out_dir, f"{case_name}.pkl"), "wb") as f:
        pickle.dump(props, f)

    return case_name


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    img_files = sorted(glob.glob(os.path.join(IMAGES_TS, "*_0000.nii.gz")))
    print(f"Found {len(img_files)} test images")

    processed = []
    for img_path in img_files:
        case_name = os.path.basename(img_path).replace("_0000.nii.gz", "")
        lbl_path = os.path.join(LABELS_TS, f"{case_name}.nii.gz")
        if not os.path.exists(lbl_path):
            print(f"  WARNING: no label for {case_name}, skipping")
            continue
        cn = preprocess_case(img_path, lbl_path, OUTPUT_DIR)
        processed.append(cn)

    # Write test.txt (two-column: img_path seg_path)
    test_txt = os.path.join(OUTPUT_DIR, "test.txt")
    with open(test_txt, "w") as f:
        for cn in processed:
            img_npy = os.path.join(OUTPUT_DIR, f"{cn}.npy")
            seg_npy = os.path.join(OUTPUT_DIR, f"{cn}_seg.npy")
            f.write(f"{img_npy} {seg_npy}\n")
    print(f"\nWrote {test_txt} ({len(processed)} cases)")

    # Also regenerate train.txt to use ALL preprocessed training data
    train_dir = os.path.join(
        os.path.dirname(OUTPUT_DIR),
        "nnUNetPlans_3d_lowres_large_patch"
    )
    train_npys = sorted(
        [f for f in glob.glob(os.path.join(train_dir, "*.npy"))
         if not f.endswith("_seg.npy")]
    )
    train_txt = os.path.join(train_dir, "train_all.txt")
    with open(train_txt, "w") as f:
        for npy_path in train_npys:
            seg_path = npy_path.replace(".npy", "_seg.npy")
            f.write(f"{npy_path} {seg_path}\n")
    print(f"Wrote {train_txt} ({len(train_npys)} cases, all for training)")
    print("\nDone!")


if __name__ == "__main__":
    main()
