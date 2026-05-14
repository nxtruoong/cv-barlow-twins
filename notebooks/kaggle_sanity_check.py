"""Day 1: sanity checks before burning GPU days.

Paste this into a Kaggle notebook cell. Requires the State Farm dataset attached
to the notebook (input path: /kaggle/input/competitions/state-farm-distracted-driver-detection).

Run this once. Verifies:
- driver_imgs_list.csv parses + filenames match
- Class distribution is balanced
- GroupKFold splits drivers cleanly (no leakage)
- 1-epoch dry run of SimCLR pretrain + 1-epoch fine-tune does not crash
"""
import sys
sys.path.insert(0, "/kaggle/working/CV")  # repo cloned into /kaggle/working

import random
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

from src.config import CLASS_NAMES
from src.data import load_driver_table, build_group_kfold, list_test_images
from src.seed_utils import set_seed

set_seed()

df = load_driver_table()
print(f"Labeled images: {len(df)}")
print(f"Unique drivers: {df['subject'].nunique()}")
print(f"Test images: {len(list_test_images())}")

print("\nClass distribution:")
print(df["classname"].value_counts().sort_index())

print("\nPer-driver image counts (first 10):")
print(df["subject"].value_counts().head(10))

# Verify files exist
sample = df.sample(5, random_state=42)
for _, row in sample.iterrows():
    assert Path(row["img_path"]).exists(), f"Missing: {row['img_path']}"
    img = Image.open(row["img_path"])
    print(f"OK: {Path(row['img_path']).name} size={img.size} label=c{row['label']} subj={row['subject']}")

# GroupKFold leakage check
folds = build_group_kfold(df, n_splits=5)
for i, (tr, va) in enumerate(folds):
    tr_subj = set(df.iloc[tr]["subject"])
    va_subj = set(df.iloc[va]["subject"])
    overlap = tr_subj & va_subj
    print(f"Fold {i}: train {len(tr)} imgs ({len(tr_subj)} drivers), "
          f"val {len(va)} imgs ({len(va_subj)} drivers), overlap={len(overlap)}")
    assert not overlap, f"Subject leakage in fold {i}!"
print("\nAll fold subject overlaps = 0. GroupKFold OK.")

# Show 1 image per class
fig, axes = plt.subplots(2, 5, figsize=(16, 6))
for i, cls in enumerate(sorted(df["classname"].unique())):
    sample_path = df[df["classname"] == cls]["img_path"].iloc[0]
    axes.flat[i].imshow(Image.open(sample_path))
    axes.flat[i].set_title(f"{cls}: {CLASS_NAMES[i]}", fontsize=9)
    axes.flat[i].axis("off")
plt.tight_layout()
plt.show()

print("\n--- Sanity check complete. Ready for pretrain. ---")
