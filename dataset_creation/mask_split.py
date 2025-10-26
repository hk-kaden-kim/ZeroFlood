import os
import numpy as np
import pandas as pd
import shutil
import tifffile
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# ============================
# Configuration
# ============================
MASK_DIR = "./target"
OUTPUT_DIR = "./target"
NUM_BINS = 10
SPLIT_RATIOS = (0.6, 0.2, 0.2)  # train, val, test
BIN_EDGES = np.linspace(0, 1, 10 + 1)

# Create output folders
for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(OUTPUT_DIR, split), exist_ok=True)


# ============================
# Step A. EDA
# ============================
def count_mask_values(mask_path):
    """Read mask and count 0s and 1s."""
    mask = tifffile.imread(mask_path)
    mask = np.array(mask)
    zeros = np.sum(mask == 0)
    ones = np.sum(mask == 1)
    return zeros, ones

mask_files = [
    (os.path.join(MASK_DIR, f), os.path.join(MASK_DIR, f.replace("Flood","WaterBody")))
    for f in os.listdir(MASK_DIR)
    if f.lower().endswith(".tif") and f.lower().startswith("flood")
]

records = []
print(f"Found {len(mask_files)} mask files.")

for mask_path, water_path in tqdm(mask_files, desc="Analyzing masks"):
    zeros, ones = count_mask_values(mask_path)
    _, w_ones = count_mask_values(water_path)

    ratio = w_ones/ones

    records.append({
        "f_mask": mask_path,
        "w_mask": water_path,
        "ratio": ratio
    })

df = pd.DataFrame(records)

# ============================
# Step B. Grouping and Splitting
# ============================

# Create bins
df["bin"] = pd.cut(df["ratio"], bins=BIN_EDGES, labels=False)

train_files, val_files, test_files = [], [], []

for bin_idx, group in df.groupby("bin"):
    if len(group) < 3:
        # Small bins may go entirely to train
        train_files.extend(group["f_mask"].tolist())
        continue

    train_ratio, val_ratio, test_ratio = SPLIT_RATIOS

    # First split train+temp
    train, temp = train_test_split(
        group,
        test_size=(1 - train_ratio),
        random_state=42,
        shuffle=True
    )

    # Split temp into val and test (val:test = 1:2 here)
    rel_test_size = test_ratio / (val_ratio + test_ratio)
    val, test = train_test_split(
        temp,
        test_size=rel_test_size,
        random_state=42,
        shuffle=True
    )

    train_files.extend(train["f_mask"].tolist())
    val_files.extend(val["f_mask"].tolist())
    test_files.extend(test["f_mask"].tolist())


# ============================
# Step C. Move files to folders
# ============================
def move_files(file_list, split_name):
    dest_dir = os.path.join(OUTPUT_DIR, split_name)
    for src in tqdm(file_list, desc=f"Moving {split_name} files"):
        dst = os.path.join(dest_dir, os.path.basename(src))
        shutil.move(src, dst)
        src = src.replace("Flood","WaterBody")
        dst = os.path.join(dest_dir, os.path.basename(src))
        shutil.move(src, dst)

move_files(train_files, "train")
move_files(val_files, "val")
move_files(test_files, "test")

# ============================
# Step C. Save splits
# ============================
# pd.DataFrame({"f_mask": train_files}).to_csv(os.path.join(OUTPUT_DIR, "train.csv"), index=False)
# pd.DataFrame({"f_mask": val_files}).to_csv(os.path.join(OUTPUT_DIR, "val.csv"), index=False)
# pd.DataFrame({"f_mask": test_files}).to_csv(os.path.join(OUTPUT_DIR, "test.csv"), index=False)

print("\n=== Split Summary ===")
print(f"Train: {len(train_files)} files")
print(f"Val:   {len(val_files)} files")
print(f"Test:  {len(test_files)} files")
print(f"CSV files saved to: {OUTPUT_DIR}")
