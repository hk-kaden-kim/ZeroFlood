
import torch
from model import TerraMindModel, CustomSemanticSegmentationTask
from datamodule import statistics

import os
import tifffile
import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import random
from skimage.exposure import rescale_intensity
from terratorch.datasets.utils import clip_image

# -------------------------------------------------------
# 1️⃣ Helper: find checkpoint
# -------------------------------------------------------
def find_epoch_checkpoint(folder_path: str) -> str:
    for f in os.listdir(folder_path):
        if f.endswith(".ckpt") and "epoch" in f:
            return os.path.join(folder_path, f)
    raise FileNotFoundError(f"No .ckpt file with 'epoch' found in {folder_path}")

def viz_S1RTC(input):
    if input.shape[-1] !=2:
        msg = f"Check the image shape is (H, W, C)... {input.shape}"
        raise ValueError(msg)
    
    VV_lin = input[:,:,0]
    VH_lin = input[:,:,1]

    # Construct RGB components
    R = rescale_intensity(VH_lin, out_range=(0, 127))
    G = rescale_intensity(VV_lin, out_range=(0, 127))
    B = rescale_intensity(VV_lin / (VH_lin + 1e-6), out_range=(0, 127))  # avoid divide-by-zero

    # Stack and normalize for display
    input = np.stack([R, G, B], axis=-1)
    input = (input - input.min()) / (input.max() - input.min())
    input = (input * 255).astype(np.uint8)
    return input

def viz_S2L2A(input):
    if input.shape[-1] not in [6,12,13]:
        msg = f"Check the image shape is (H, W, C)... {input.shape}"
        raise ValueError(msg)
    
    # Normalize to [0,255]
    input = input[:, :, [3, 2, 1]]  # B04(R), B03(G), B02(B)
    p2, p98 = np.percentile(input, (2, 98))
    input = np.clip((input - p2) / (p98 - p2), 0, 1)
    return input


# -------------------------------------------------------
# 2️⃣ Settings
# -------------------------------------------------------
is_large_model = 1
modality = "S2L2A"  # S2L2A     S1RTC
tim = ["S1RTC", "DEM", "LULC"]       # ["S2L2A","DEM", "LULC"]

data_root = "/dss/dsstbyfs02/scratch/07/di54rur/zeroflood_output/lightning_logs"
ckpt_path = f"early_stopping_version/terramind/large/{modality[:2].lower()}/tim-lv{len(tim)}"
if len(tim) > 0 and len(tim) < 3:
    _tim = [m[:2].lower() if m[0].lower() == 's' else m for m in tim]
    ckpt_path = ckpt_path + f"/{'_'.join(_tim).lower()}"
# input_folder = f"/dss/dsstbyfs02/scratch/07/di54rur/terramesh_tiny/test_unknown/{modality}"
# gt_folder    = "/dss/dsstbyfs02/scratch/07/di54rur/terramesh_tiny/target/test_unknown"
input_folder = f"/dss/dsstbyfs02/scratch/07/di54rur/terramesh_tiny/test/{modality}"
gt_folder    = "/dss/dsstbyfs02/scratch/07/di54rur/terramesh_tiny/target/test"

# -------------------------------------------------------
# 3️⃣ Load model + checkpoint
# -------------------------------------------------------
model = TerraMindModel(
    size='base' if not is_large_model else 'large',
    modality=[modality],
    tim=tim,
)

task = CustomSemanticSegmentationTask(
    model_factory="EncoderDecoderFactory",
    model_args=model.get_args(),
    ignore_index=-1,
    class_names=['0', 'Risk']
)

full_ckpt_path = find_epoch_checkpoint(os.path.join(data_root, ckpt_path))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

state = torch.load(full_ckpt_path, map_location=device)
task.load_state_dict(state['state_dict'])
task.eval()
task.freeze()
task.to(device)

# -------------------------------------------------------
# 4️⃣ Define preprocessing transform
# -------------------------------------------------------
predict_transform = A.Compose([
    A.Normalize(mean=statistics['mean'][modality],
                std=statistics['std'][modality],
                normalization="image_per_channel"),
    A.pytorch.transforms.ToTensorV2(),
])

# -------------------------------------------------------
# 5️⃣ Collect sample file paths
# -------------------------------------------------------
tif_files = sorted([f for f in os.listdir(input_folder) if f.endswith(".tif")])
# tif_files = [

#     # Testset Best samples
#     "majortom_val_0010901.tif",
#     "majortom_val_0015796.tif",
#     "majortom_val_0027730.tif",
#     "majortom_val_0031271.tif",
#     "majortom_val_0037906.tif",
#     "majortom_val_0077680.tif",
#     "majortom_val_0078482.tif",

#     # Testset GT example
#     # "majortom_val_0027700.tif",
#     # "majortom_val_0079399.tif",

#     # Trainset GT example
#     # "majortom_val_0023503.tif",
#     # "majortom_val_0035084.tif",

#     # Valset GT example
#     # "majortom_val_0023971.tif",
#     # "majortom_val_0021863.tif",
# ]

print(f" Found {len(tif_files)} samples.")

# -------------------------------------------------------
# 6️⃣ Run inference on each sample
# -------------------------------------------------------
results = []

for fname in tif_files:
    input_path = os.path.join(input_folder, fname)
    gt_path    = os.path.join(gt_folder, 'Flood_'+fname)

    # Load input and GT
    image = tifffile.imread(input_path)
    gt = tifffile.imread(gt_path) if os.path.exists(gt_path) else None

    # Transform
    transformed = predict_transform(image=image)
    tensor = transformed["image"].unsqueeze(0).to(device)

    # Model prediction
    with torch.no_grad():
        pred = task(tensor).output  # expected shape (B, num_classes, H, W)
        pred_label = torch.argmax(pred, dim=1).squeeze(0).cpu().numpy()

    results.append({
        "filename": fname,
        "image": image,
        "prediction": pred_label,
        "gt": gt,
    })

print(" Inference completed for all samples.")

# -------------------------------------------------------
# 7️⃣ Visualization: Individually
# -------------------------------------------------------
for result in results:
    fname = result["filename"]
    image = result["image"]
    pred = result["prediction"]
    gt = result["gt"]

    # Define folder structure
    base_folder = os.path.join("predictions", fname, modality, f"tim_lv{len(tim)}")
    os.makedirs(base_folder, exist_ok=True)

    # Ensure image has 3 channels
    img_show = image if image.ndim == 3 else np.stack([image] * 3, axis=-1)
    if modality == "S2L2A":
        img_show = viz_S2L2A(image)
    elif modality == "S1RTC":
        img_show = viz_S1RTC(image)

    # --- Save image ---
    plt.figure(figsize=(4, 4))
    plt.imshow(img_show)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(base_folder, "image.pdf"), bbox_inches="tight")
    plt.close()

    # --- Save prediction ---
    plt.figure(figsize=(4, 4))
    plt.imshow(pred, cmap="copper")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(base_folder, "pred.pdf"), bbox_inches="tight")
    plt.close()

    # --- Save ground truth (if available) ---
    if gt is not None:
        plt.figure(figsize=(4, 4))
        plt.imshow(gt, cmap="copper")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(base_folder, "gt.pdf"), bbox_inches="tight")
        plt.close()

    print(f"✅ Saved PDFs for {base_folder}")

print("\n🎉 All visualizations have been saved successfully!")