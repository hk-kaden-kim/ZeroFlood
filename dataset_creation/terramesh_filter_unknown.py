import os
import pandas as pd
import webdataset as wds
from PIL import Image
from tqdm import tqdm
from terramesh import build_terramesh_dataset, Transpose, MultimodalTransforms, MultimodalNormalize, statistics
from albumentations.pytorch import ToTensorV2
import albumentations as A
from torch.utils.data import DataLoader
import torchvision
import tifffile
import torch
import time

def save_to_tif(t: torch.Tensor, path: str):
    arr = t.detach().cpu().numpy()
    # If shape is (C, H, W), transpose to (H, W, C)
    if arr.ndim == 3 and arr.shape[0] <= 20:
        arr = arr.transpose(1, 2, 0)
    # Save multi-band TIFF
    tifffile.imwrite(path, arr.astype("float32"))

# ========================================
# Configuration
# ========================================
modalities = ["S1RTC",  "S2L1C",  "S2L2A",  "S2RGB"]
terramesh_folder = "/dss/dsstbyfs02/scratch/07/di54rur/terramesh"  # your WebDataset source
output_dir = "/dss/dsstbyfs02/scratch/07/di54rur/terramesh_tiny"

MASK_DIR = "/dss/dsstbyfs02/scratch/07/di54rur/terramesh_tiny/target/test_unknown"
mask_files = set([
    f.split("_", 1)[-1][:-4]
    for f in os.listdir(MASK_DIR)
    if f.lower().endswith(".tif") and f.lower().startswith("flood")
])

# ========================================
# Step 2. Load TerraMesh
# ========================================
modalities = ["S1RTC",  "S2L1C",  "S2L2A",  "S2RGB"]

# If you pass multiple modalities, the modalities are returned using the modality names as keys
dataset = build_terramesh_dataset(
    path="/dss/dsstbyfs02/scratch/07/di54rur/terramesh",  # Streaming or local path
    modalities=modalities, 
    shuffle=False,  # Set false for split="val"
    split="val",
    batch_size=1
)

# ========================================
# Step 2. Filter out and Save
# ========================================
iter, start_time = 0, time.time()
cnt = 0
for b_idx, item in enumerate(dataset):
    key = item["__key__"][0]
    if "S1RTC" not in item.keys():
         continue
    S1RTC, S2L1C, S2L2A, S2RGB = item["S1RTC"][0], item["S2L1C"][0], item["S2L2A"][0], item["S2RGB"][0]
    # print(key, mask_files)
    # break
    if key in mask_files:
        cnt += 1
        save_to_tif(S1RTC, os.path.join(output_dir, "test_unknown", "S1RTC",f"{key}.tif"))
        save_to_tif(S2L1C, os.path.join(output_dir, "test_unknown", "S2L1C",f"{key}.tif"))
        save_to_tif(S2L2A, os.path.join(output_dir, "test_unknown", "S2L2A",f"{key}.tif"))
        save_to_tif(S2RGB, os.path.join(output_dir, "test_unknown", "S2RGB",f"{key}.tif"))
        mask_files.discard(key)
        print(f"Found {key}! Remain {len(mask_files)}")
    
    iter += 1
    if (iter+1) % 1000 == 0:
         print(f"~{iter+1} : {time.time() - start_time} sec")
         start_time = time.time()
    if len(mask_files) == 0:
            break

print(f"DONE! {iter} {cnt}")
