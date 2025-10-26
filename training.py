import albumentations as A
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, RichProgressBar, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger

from datamodule import FloodRiskMapNonGeoDataModule, statistics
from model import TerraMindModel, PrithviModel, CustomSemanticSegmentationTask

import torch
import datetime
import argparse

pl.seed_everything(0)
TARGET_SIZE = (264, 264) # Example input height, width

def main(data_root, batch_size, epochs, modality, tim, is_prithvi, is_large_model, early_stopping):

    # Initialize dataset
    datamodule = FloodRiskMapNonGeoDataModule(
        data_root = data_root,
        batch_size = batch_size,
        num_workers = 4,
        modality = modality,
        train_transform = A.Compose([
            A.RandomCrop(height=TARGET_SIZE[0], width=TARGET_SIZE[1], p=1.0),
            A.SquareSymmetry(p=1.0),
            A.Normalize(mean=statistics['mean'][modality], 
                        std=statistics['std'][modality], 
                        normalization="image_per_channel"),
            A.ToTensorV2(),
        ]),
        val_transform = A.Compose([
            A.Normalize(mean=statistics['mean'][modality], 
                        std=statistics['std'][modality], 
                        normalization="image_per_channel"),
            A.ToTensorV2(),
        ]),
        test_transform = A.Compose([
            A.Normalize(mean=statistics['mean'][modality], 
                        std=statistics['std'][modality], 
                        normalization="image_per_channel"),
            A.ToTensorV2(),
        ]),
        for_prithvi = is_prithvi,
    )
    datamodule.setup("fit")
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
        loss="focal",
        optimizer="AdamW",
        lr=1e-4,
        ignore_index=-1,
        freeze_backbone=True, # Speeds up fine-tuning
        freeze_decoder=False,
        plot_on_val=True,
        class_names=['0', 'Risk']  # optionally define class names
    )
    print("Done | Generate model and task")

    # Setup Trainer
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_callback = ModelCheckpoint(
        filename=f"{timestamp}" + "-{epoch:02d}",
        monitor=task.monitor,
        mode="min",
        save_last=True
    )
    early_stop_callback = EarlyStopping(
        monitor=task.monitor,     # Metric to monitor
        mode="min",             # Lower is better
        patience=50,            # Stop if no improvement for 10 epochs
        min_delta=1e-8,          # Minimum change to qualify as improvement
        verbose=True
    )
    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        strategy="auto",
        devices=1, # Deactivate multi-gpu because it often fails in notebooks
        precision='32'if (not is_prithvi) and (is_large_model) else '16-mixed',  # '16-mixed' for Speed up training.
        num_nodes=1,
        logger=True,
        max_epochs=epochs, # For demos
        log_every_n_steps=1,
        enable_checkpointing=True,
        callbacks=[checkpoint_callback, RichProgressBar()] + [early_stop_callback] if early_stopping else [],
        default_root_dir=f"{data_root}/zeroflood_output",
    )
    print(f"Done | Setup trainer {timestamp}")

    # Start training
    trainer.fit(task, datamodule=datamodule)
    print("Done | Training completed!")

if __name__ == '__main__': 

    parser = argparse.ArgumentParser(description="Run segmentation training")

    parser.add_argument("--data_root", type=str, default="/dss/dsstbyfs02/scratch/07/di54rur",
                        help="Path to dataset root")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--max_epochs", type=int, default=100, help="Maximum number of epochs")
    parser.add_argument("--modality", type=str, default="S1RTC", help="Modality to use (e.g., S1RTC, S2L2A)")
    parser.add_argument("--tim", nargs="*", default=[], help="Other modalities to be generated (e.g., S1RTC S2L2A DEM LULC)")
    parser.add_argument("--is_prithvi", type=int, default=0, help="Exception! (1:Yes, 0:No)")
    parser.add_argument("--is_large_model", type=int, default=0, help="TerraMind Large / Prithvi v2 600! (1:Yes, 0:No)")
    parser.add_argument("--early_stopping", type=int, default=1, help="Early Stopping ON:1(Default) / OFF:0)")

    args = parser.parse_args()

    print("+++++++++++++++++++++++++++++++++++++")
    print(f"{args.batch_size} {args.max_epochs} {args.modality} {args.tim} {args.is_prithvi} {args.is_large_model} {args.early_stopping}")

    if torch.cuda.is_available():
        print("GPU is available")
    else:
        print("GPU is not available")

    main(
        args.data_root, 
        args.batch_size, args.max_epochs, 
        args.modality, args.tim, 
        args.is_prithvi, args.is_large_model, args.early_stopping
        )
