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

train_filename_csv = "/dss/dsshome1/07/di54rur/zeroflood/target/train.csv"    # CSV file containing filenames
val_filename_csv = "/dss/dsshome1/07/di54rur/zeroflood/target/val.csv"    # CSV file containing filenames
test_filename_csv = "/dss/dsshome1/07/di54rur/zeroflood/target/test.csv"    # CSV file containing filenames

output_dir = "/dss/dsstbyfs02/scratch/07/di54rur/terramesh_filtered"         # where to save filtered images

os.makedirs(output_dir, exist_ok=True)
for m in modalities:
    os.makedirs(os.path.join(output_dir,"train",m), exist_ok=True)
    os.makedirs(os.path.join(output_dir,"val",m), exist_ok=True)
    os.makedirs(os.path.join(output_dir,"test",m), exist_ok=True)

# ========================================
# Step 1. Load filename list
# ========================================
train_df = pd.read_csv(train_filename_csv)
train_keys = set(os.path.splitext(os.path.basename(f))[0].split("_", 1)[-1] for f in train_df["f_mask"])
print(f"Train: Loaded {len(train_keys)} filenames to keep.")

val_df = pd.read_csv(val_filename_csv)
val_keys = set(os.path.splitext(os.path.basename(f))[0].split("_", 1)[-1] for f in val_df["f_mask"])
print(f"Val: Loaded {len(val_keys)} filenames to keep.")

test_df = pd.read_csv(test_filename_csv)
test_keys = set(os.path.splitext(os.path.basename(f))[0].split("_", 1)[-1] for f in test_df["f_mask"])
print(f"Test: Loaded {len(test_keys)} filenames to keep.")


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
train_cnt, val_cnt, test_cnt = 0, 0, 0
for b_idx, item in enumerate(dataset):
    key = item["__key__"][0]
    if "S1RTC" not in item.keys():
         continue
    S1RTC, S2L1C, S2L2A, S2RGB = item["S1RTC"][0], item["S2L1C"][0], item["S2L2A"][0], item["S2RGB"][0]
    if key in train_keys:
        train_cnt += 1
        save_to_tif(S1RTC, os.path.join(output_dir, "train", "S1RTC",f"{key}.tif"))
        save_to_tif(S2L1C, os.path.join(output_dir, "train", "S2L1C",f"{key}.tif"))
        save_to_tif(S2L2A, os.path.join(output_dir, "train", "S2L2A",f"{key}.tif"))
        save_to_tif(S2RGB, os.path.join(output_dir, "train", "S2RGB",f"{key}.tif"))
        train_keys.discard(key)
    
    elif key in val_keys:
        val_cnt += 1
        save_to_tif(S1RTC, os.path.join(output_dir, "val", "S1RTC",f"{key}.tif"))
        save_to_tif(S2L1C, os.path.join(output_dir, "val", "S2L1C",f"{key}.tif"))
        save_to_tif(S2L2A, os.path.join(output_dir, "val", "S2L2A",f"{key}.tif"))
        save_to_tif(S2RGB, os.path.join(output_dir, "val", "S2RGB",f"{key}.tif"))
        val_keys.discard(key)

    elif key in test_keys:
        test_cnt += 1
        save_to_tif(S1RTC, os.path.join(output_dir, "test", "S1RTC",f"{key}.tif"))
        save_to_tif(S2L1C, os.path.join(output_dir, "test", "S2L1C",f"{key}.tif"))
        save_to_tif(S2L2A, os.path.join(output_dir, "test", "S2L2A",f"{key}.tif"))
        save_to_tif(S2RGB, os.path.join(output_dir, "test", "S2RGB",f"{key}.tif"))
        test_keys.discard(key)
    # else:
    #     save_to_tif(S1RTC, os.path.join(output_dir,f"{key}_1.tif"))
    #     save_to_tif(S2L1C, os.path.join(output_dir,f"{key}_2.tif"))
    #     save_to_tif(S2L2A, os.path.join(output_dir,f"{key}_3.tif"))
    #     save_to_tif(S2RGB, os.path.join(output_dir,f"{key}_4.tif"))
    #     break
    
    iter += 1
    if (iter+1) % 1000 == 0:
         print(f"~{iter+1} : {time.time() - start_time} sec")
         start_time = time.time()
    if len(train_keys)+len(val_keys)+len(test_keys) == 0:
            break

print(f"DONE! {iter} {(train_cnt, val_cnt, test_cnt)}")
