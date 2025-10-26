import os
import numpy as np
import tifffile
from tqdm import tqdm
import matplotlib.pyplot as plt

# Directory containing all mask .tif files
MASK_DIR = "./target"
# Output path for histogram image
OUTPUT_HIST_PATH = "mask_foreground_ratio_hist.png"
WF_OUTPUT_HIST_PATH = "mask_water_flood_ratio_hist.png"
BIN_EDGES = np.linspace(0, 1, 10 + 1)

def count_mask_values(mask_path):
    """Read mask and count 0s and 1s."""
    mask = tifffile.imread(mask_path)
    mask = np.array(mask)
    zeros = np.sum(mask == 0)
    ones = np.sum(mask == 1)
    return zeros, ones

def main():
    mask_files = [
        (os.path.join(MASK_DIR, f), os.path.join(MASK_DIR, f.replace("Flood","WaterBody")))
        for f in os.listdir(MASK_DIR)
        if f.lower().endswith(".tif") and f.lower().startswith("flood")
    ]

    total_zeros = 0
    total_ones = 0
    zero_counts = []
    one_counts = []
    ratio = []
    print(f"Found {len(mask_files)} mask files. {mask_files[0][0]}, {mask_files[0][1]}")

    for mask_path, water_path in tqdm(mask_files, desc="Analyzing masks"):
        zeros, ones = count_mask_values(mask_path)
        _, w_ones = count_mask_values(water_path)
        total_zeros += zeros
        total_ones += ones
        zero_counts.append(zeros)
        one_counts.append(ones)
        ratio.append(w_ones/ones)

    print("\n=== Mask EDA Summary ===")
    print(f"Total masks: {len(mask_files)}")
    
    # Save histogram instead of showing
    plt.figure(figsize=(8, 5))
    plt.hist(
        np.array(one_counts) / (np.array(one_counts) + np.array(zero_counts)),
        bins=BIN_EDGES, edgecolor='k'
    )
    plt.title("Distribution of Flood Foreground Pixel Ratio (per mask)")
    plt.xlabel("Foreground ratio")
    plt.ylabel("Number of masks")
    plt.tight_layout()
    plt.savefig(OUTPUT_HIST_PATH)
    plt.close()

    # Save histogram instead of showing
    plt.figure(figsize=(8, 5))
    plt.hist(ratio, bins=BIN_EDGES, edgecolor='k')
    plt.title("Distribution of Water/Flood Pixel Ratio (per mask)")
    plt.xlabel("Water/Flood ratio")
    plt.ylabel("Number of masks")
    plt.tight_layout()
    plt.savefig(WF_OUTPUT_HIST_PATH)
    plt.close()

    print(f"\nHistogram saved to: {OUTPUT_HIST_PATH}")

if __name__ == "__main__":
    main()
