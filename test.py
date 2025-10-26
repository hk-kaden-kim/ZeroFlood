import os
import albumentations as A
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, RichProgressBar

from datamodule import FloodRiskMapNonGeoDataModule, statistics
from model import TerraMindModel, PrithviModel, CustomSemanticSegmentationTask

import torch
import datetime
import argparse

pl.seed_everything(0)
TARGET_SIZE = (264, 264) # Example input height, width

def find_epoch_checkpoint(folder_path: str) -> str:
    """Return the full path of the .ckpt file containing 'epoch' in its name."""
    for f in os.listdir(folder_path):
        if f.endswith(".ckpt") and "epoch" in f:
            return os.path.join(folder_path, f)
    raise FileNotFoundError(f"No .ckpt file with 'epoch' found in {folder_path}")

def main(data_root, ckpt_path, batch_size, modality, tim, is_prithvi, is_large_model):

    # Initialize dataset
    datamodule = FloodRiskMapNonGeoDataModule(
        data_root = data_root,
        batch_size = batch_size,
        num_workers = 4,
        modality = modality,
        test_transform = A.Compose([
            A.Normalize(mean=statistics['mean'][modality], 
                        std=statistics['std'][modality], 
                        normalization="image_per_channel"),
            A.ToTensorV2(),
        ]),
        for_prithvi = is_prithvi,
    )
    datamodule.setup("test")
    print("Done | Generate datamodule")

    # Initialize model and task
    if not is_prithvi:
        model = TerraMindModel(
            size='base' if not is_large_model else 'large',
            modality=[modality], 
            tim=tim,
            )
    else:
        model = PrithviModel(
            size='300' if not is_large_model else '600',
        )
    task = CustomSemanticSegmentationTask(
        model_factory="EncoderDecoderFactory",
        model_args=model.get_args(),
        ignore_index=-1,
        class_names=['0', 'Risk']  # optionally define class names
        )
    
    # NOTE:
    # def load_from_checkpoint() in  class TerraTorchTask() gave an error
    # TypeError: SemanticSegmentationTask.__init__() got an unexpected keyword argument 'task'
    full_ckpt_path = find_epoch_checkpoint(os.path.join(data_root, ckpt_path))
    if torch.cuda.is_available():
        state = torch.load(full_ckpt_path)
    else:
        state = torch.load(full_ckpt_path, map_location=torch.device('cpu'))   
    task.load_state_dict(state['state_dict'])
    task.eval()
    task.freeze()
    print(f"Done | Generate model and task")

    # Setup pl.LightningModule
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision='32'if (not is_prithvi) and (is_large_model) else '16-mixed',  # '16-mixed' for Speed up training.
        num_nodes=1,
        logger=False,
    )
    print(f"Done | Setup LightningModule {timestamp}")

    # Start Testing
    test_results = trainer.test(task, datamodule=datamodule)
    print("Done | Testing completed!")

    # Evaluation metrics
    print(f"\nModel: {full_ckpt_path[len(data_root):]}")
    if isinstance(test_results, list):
        test_results = test_results[0]  # Lightning returns a list of dicts
    eval_metrics = {
        'HitRate' : test_results['test/Recall_Risk'] * 100,
        'FalseAlarmRate' : (1-test_results['test/Precision_Risk']) * 100,
        'CriticalSuccessRate' : test_results['test/IoU_Risk'] * 100
    }
    for metric_name, value in eval_metrics.items():
        print(f"{metric_name:30s}: {value:.2f}")

if __name__ == '__main__': 

    parser = argparse.ArgumentParser(description="Run segmentation training")

    parser.add_argument("--data_root", type=str, default="/dss/dsstbyfs02/scratch/07/di54rur",
                        help="Path to dataset root")
    parser.add_argument("--ckpt_path", type=str, default="Folder path where ***epoch=**.ckpt is located")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--modality", type=str, default="S1RTC", help="Modality to use (e.g., S1RTC, S2L2A)")
    parser.add_argument("--tim", nargs="*", default=[], help="Other modalities to be generated (e.g., S1RTC S2L2A DEM LULC)")
    parser.add_argument("--is_prithvi", type=int, default=0, help="Exception! (1:Yes, 0:No)")
    parser.add_argument("--is_large_model", type=int, default=0, help="TerraMind Large / Prithvi v2 600! (1:Yes, 0:No)")

    args = parser.parse_args()

    if torch.cuda.is_available():
        print("GPU is available")
    else:
        print("GPU is not available")

    main(
        args.data_root, args.ckpt_path,
        args.batch_size,
        args.modality, args.tim, 
        args.is_prithvi, args.is_large_model
        )