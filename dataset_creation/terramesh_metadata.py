import pandas as pd
from terramesh import build_terramesh_dataset
import time
from pathlib import Path


# ========================================
# Configuration
# ========================================
modalities = ["S1RTC",  "S2L1C",  "S2L2A",  "S2RGB"]
terramesh_folder = "/dss/dsstbyfs02/scratch/07/di54rur/terramesh"  # your WebDataset source

train_filename_csv = Path("/dss/dsstbyfs02/scratch/07/di54rur/terramesh_tiny/train.csv")    # CSV file containing filenames
val_filename_csv = Path("/dss/dsstbyfs02/scratch/07/di54rur/terramesh_tiny/val.csv")    # CSV file containing filenames
test_filename_csv = Path("/dss/dsstbyfs02/scratch/07/di54rur/terramesh_tiny/test.csv")    # CSV file containing filenames

# ========================================
# Step 1. Load filename list
# ========================================
train_df = pd.read_csv(train_filename_csv)
train_df['lon'] = None
train_df['lat'] = None
print(f"Train: Loaded {len(train_df)} filenames to keep.")

val_df = pd.read_csv(val_filename_csv)
val_df['lon'] = None
val_df['lat'] = None
print(f"Val: Loaded {len(val_df)} filenames to keep.")

test_df = pd.read_csv(test_filename_csv)
test_df['lon'] = None
test_df['lat'] = None
print(f"Test: Loaded {len(test_df)} filenames to keep.")


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
    batch_size=1,
    return_metadata=True,
    time_dim=True,
)

# ========================================
# Step 2. Filter out and Save
# ========================================
iter, start_time = 0, time.time()
train_cnt, val_cnt, test_cnt = 0, 0, 0
for b_idx, item in enumerate(dataset):
    key = item["__key__"][0]
    lon, lat = item["center_lon"][0], item["center_lat"][0]
    if "S1RTC" not in item.keys():
         continue
    
    # S1RTC, S2L1C, S2L2A, S2RGB = item["S1RTC"][0], item["S2L1C"][0], item["S2L2A"][0], item["S2RGB"][0]
    mask = train_df['f_mask'].str.contains(key, case=False, na=False)
    if mask.any():
        train_df.loc[mask, 'lon'] = lon
        train_df.loc[mask, 'lat'] = lat
        continue

    mask = val_df['f_mask'].str.contains(key, case=False, na=False)
    if mask.any():
        val_df.loc[mask, 'lon'] = lon
        val_df.loc[mask, 'lat'] = lat
        continue

    mask = test_df['f_mask'].str.contains(key, case=False, na=False)
    if mask.any():
        test_df.loc[mask, 'lon'] = lon
        test_df.loc[mask, 'lat'] = lat
        continue

    iter += 1
    if (iter+1) % 1000 == 0:
         print(f"~{iter+1} : {time.time() - start_time} sec")
         start_time = time.time()
    if train_df['lon'].notna().all() and val_df['lon'].notna().all() and test_df['lon'].notna().all():
        break


train_df.to_csv(train_filename_csv.with_name(train_filename_csv.stem + "_new" + train_filename_csv.suffix), index=False)
val_df.to_csv(val_filename_csv.with_name(val_filename_csv.stem + "_new" + val_filename_csv.suffix), index=False)
test_df.to_csv(test_filename_csv.with_name(test_filename_csv.stem + "_new" + test_filename_csv.suffix), index=False)

print(f"PROBE: {key} {item.keys()} {train_df} {lon} {lat} {mask} {type(mask)}")
print(f"DONE! {iter} {(train_cnt, val_cnt, test_cnt)}")