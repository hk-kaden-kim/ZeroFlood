import os
import glob
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from torch import Tensor
from torch.utils.data import DataLoader

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

import albumentations as A
from skimage.exposure import rescale_intensity
import kornia.augmentation as K  # noqa: N812
from kornia.augmentation import AugmentationSequential

import tifffile
from torchgeo.datasets import NonGeoDataset
from torchgeo.datamodules import NonGeoDataModule
from terratorch.datasets.utils import default_transform, validate_bands, clip_image
from terratorch.datamodules.utils import wrap_in_compose_is_list

statistics = {
    "mean": {
        "S2L1C": [2357.090, 2137.398, 2018.799, 2082.998, 2295.663, 2854.548, 3122.860, 3040.571, 3306.491, 1473.849,
                506.072, 2472.840, 1838.943],
        "S2L2A": [1390.461, 1503.332, 1718.211, 1853.926, 2199.116, 2779.989, 2987.025, 3083.248, 3132.235, 3162.989,
                2424.902, 1857.665],
        "S2RGB": [110.349, 99.507, 75.843],
        "S1RTC": [-10.93, -17.329],
        "PRITHVI_V2": [1087.0, 1342.0, 1433.0, 2734.0, 1958.0, 1363.0]
    },
    "std": {
        "S2L1C": [1673.639, 1722.641, 1602.205, 1873.138, 1866.055, 1779.839, 1776.496, 1724.114, 1771.041, 1079.786,
                512.404, 1340.879, 1172.435],
        "S2L2A": [2131.157, 2163.666, 2059.311, 2152.477, 2105.179, 1912.773, 1842.326, 1893.568, 1775.656, 1814.907,
                1436.282, 1336.155],
        "S2RGB": [69.905, 53.708, 53.378],
        "S1RTC": [4.391, 4.459],
        "PRITHVI_V2": [2248.0, 2179.0, 2178.0, 1850.0, 1242.0, 1049.0]
    }
}

class FloodRiskMapNonGeo(NonGeoDataset):

    all_modality_names = (
        "S1RTC",
        "S2L1C", "S2L2A", "S2RGB"
    )
    rgb_bands = ("RED", "GREEN", "BLUE")
    MODALITY_SETS = {"all": all_modality_names, "rgb": rgb_bands}
    plot_legend = ["-","Risk"]
    num_classes = 2
    data_dir = "terramesh_tiny"
    target_dir = "terramesh_tiny/target"
    splits = {"train": "train", "val": "valid", "test": "test"}

    def __init__(
            self,
            data_root: str = './',
            split: str = "train",
            modality: str = "S1RTC",
            transform: A.Compose | None = None,
            constant_scale: float = 0.0001,
            no_data_replace: float | None = 0,
            no_target_replace: int | None = -1,
            for_prithvi: int = 0,
    ) -> None:
        super().__init__()
        if split not in self.splits:
            msg = f"Incorrect split '{split}', please choose one of {self.splits}."
            raise ValueError(msg)
        self.split = split

        validate_bands([modality], self.all_modality_names)
        self.modality = modality
        self.modality_index = self.all_modality_names.index(modality)
        self.constant_scale = constant_scale
        self.data_root = Path(data_root)

        data_dir = self.data_root / self.data_dir
        target_dir = self.data_root / self.target_dir

        self.image_files = sorted(glob.glob(os.path.join(data_dir, self.split, self.modality, "*.tif")))
        self.target_files = sorted(glob.glob(os.path.join(target_dir, self.split, "Flood_*.tif")))
        self.valid_files = pd.read_csv(data_dir / f"{self.split}.csv", header=0).iloc[:,0].to_list()

        if not(len(self.image_files) == len(self.target_files) == len(self.valid_files)):
            assert False, f"{len(self.image_files)} / {len(self.target_files)} / {len(self.valid_files)}"

        self.no_data_replace = no_data_replace
        self.no_target_replace = no_target_replace

        self.transform = transform if transform else default_transform

        self.for_prithvi = for_prithvi
        if for_prithvi:
            if modality != 'S2L2A':
                assert False, f"Only S2L2A (not {modality}) is used for Prithvi model."

    def __len__(self) -> int:
        return len(self.image_files)
    
    def __getitem__(self, index: int) -> dict[str, Any]:

        image = self._load_file(self.image_files[index], nan_replace=self.no_data_replace) # >> HWC
        # image = np.moveaxis(image, -1, 0) # >> CHW
        mask = self._load_file(self.target_files[index], nan_replace=self.no_target_replace) # HW
        # to channels last
        # image = np.moveaxis(image, 0, -1) # >> HWC

        if self.for_prithvi:
            image = image[..., [1,2,3,8,10,11]]

        output = {
            "image": image.astype(np.float32) * self.constant_scale,
            "mask": mask,
            }
        if not self.for_prithvi:
            output["metadata"] = self.image_files[index]

        if self.transform:
            output = self.transform(**output)
        
        output["mask"] = output["mask"].long()

        return output
    
    def _load_file(self, path:Path, nan_replace: int | float | None = None):
        data = tifffile.imread(path)
        if nan_replace is not None:
            data = np.nan_to_num(data, nan = nan_replace)
        return data
    
    def plot(self, 
             sample: dict[str, Tensor], 
             suptitle: str | None = None
             ) -> Figure:

        if self.modality == "S1RTC":
            num_images, image, mask, prediction = self._S1RTC_plot(sample)
        elif self.modality == "S2L1C" or self.modality == "S2L2A":
            num_images, image, mask, prediction = self._S2L1C_S2L2A_plot(sample)
        elif self.modality == "S2RGB":
            num_images, image, mask, prediction = self._S2RGB_plot(sample)
        else:
            if self.for_prithvi:
                if sample['image'].shape[0] == 6:
                    new_sample = sample
                else:
                    assert False, f"Check the data shape: {sample['image'].shape}"
                num_images, image, mask, prediction = self._S2RGB_plot(new_sample)
            else:
                assert False, f"{self.modality} is not supported."

        fig, ax = plt.subplots(1, num_images, figsize=(12, 5), layout="compressed")

        ax[0].axis("off")

        norm = mpl.colors.Normalize(vmin=0, vmax=self.num_classes - 1)
        ax[1].axis("off")
        ax[1].title.set_text("Image")
        ax[1].imshow(image)

        ax[2].axis("off")
        ax[2].title.set_text("Ground Truth Mask")
        ax[2].imshow(mask, cmap="jet", norm=norm)

        ax[3].axis("off")
        ax[3].title.set_text("GT Mask on Image")
        ax[3].imshow(image)
        ax[3].imshow(mask, cmap="jet", alpha=0.3, norm=norm)

        if "prediction" in sample:
            ax[4].title.set_text("Predicted Mask")
            ax[4].imshow(prediction, cmap="jet", norm=norm)

        cmap = plt.get_cmap("jet")
        legend_data = [[i, cmap(norm(i)), str(i)] for i in range(self.num_classes)]
        handles = [Rectangle((0, 0), 1, 1, color=tuple(v for v in c)) for k, c, n in legend_data]
        labels = [self.plot_legend[k] for k, c, n in legend_data]
        ax[0].legend(handles, labels, loc="center")
        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig

    def _S1RTC_plot(self, sample):
        num_images = 4

        image = sample["image"].permute(1,2,0).numpy()  # CHW >> HWC
        mask = sample["mask"]
        if len(sample['mask'].shape) == 3:
            mask = sample["mask"].permute(1,2,0).numpy()  # 1HW >> HW1  
        
        if image.shape[-1] !=2:
            msg = f"Check the image shape is (H, W, C)... {image.shape}"
            raise ValueError(msg)
        
        VV_lin = image[:,:,0]
        VH_lin = image[:,:,1]

        # Construct RGB components
        R = rescale_intensity(VH_lin, out_range=(0, 127))
        G = rescale_intensity(VV_lin, out_range=(0, 127))
        B = rescale_intensity(VV_lin / (VH_lin + 1e-6), out_range=(0, 127))  # avoid divide-by-zero

        # Stack and normalize for display
        image = np.stack([R, G, B], axis=-1)
        image = (image - image.min()) / (image.max() - image.min())
        image = (image * 255).astype(np.uint8)

        if "prediction" in sample:
            prediction = sample["prediction"]
            num_images += 1
        else:
            prediction = None
        
        return num_images, image, mask, prediction

    def _S2RGB_plot(self, sample):
        num_images = 4

        image = sample["image"].permute(1,2,0).numpy()  # CHW >> HWC
        mask = sample["mask"]
        if len(sample['mask'].shape) == 3:
            mask = sample["mask"].permute(1,2,0).numpy()    # 1HW >> HW1
            
        if image.shape[-1] !=3:
            msg = f"Check the image shape is (H, W, C)... {image.shape}"
            raise ValueError(msg)
        
        # Normalize to [0,255]
        # image = (image - image.min()) / (image.max() - image.min())
        # image = (image * 255).astype(np.uint8)
        p2, p98 = np.percentile(image, (2, 98))
        image = np.clip((image - p2) / (p98 - p2), 0, 1)
        image = clip_image(image)

        if "prediction" in sample:
            prediction = sample["prediction"]
            num_images += 1
        else:
            prediction = None
        
        return num_images, image, mask, prediction
    
    def _S2L1C_S2L2A_plot(self, sample):
        num_images = 4

        image = sample["image"].permute(1,2,0).numpy()      # CHW >> HWC
        mask = sample["mask"]
        if len(sample['mask'].shape) == 3:
            mask = sample["mask"].permute(1,2,0).numpy()    # 1HW >> HW1

        if image.shape[-1] not in [6,12,13]:
            msg = f"Check the image shape is (H, W, C)... {image.shape}"
            raise ValueError(msg)
        
        # Normalize to [0,255]
        if self.for_prithvi:
            image = image[:, :, [2, 1, 0]] 
        else:
            image = image[:, :, [3, 2, 1]]  # B04(R), B03(G), B02(B)
        # image = (image - image.min()) / (image.max() - image.min())
        # image = (image * 255).astype(np.uint8)
        p2, p98 = np.percentile(image, (2, 98))
        image = np.clip((image - p2) / (p98 - p2), 0, 1)
        # image = clip_image(image)

        if "prediction" in sample:
            prediction = sample["prediction"]
            num_images += 1
        else:
            prediction = None
        
        return num_images, image, mask, prediction

class FloodRiskMapNonGeoDataModule(NonGeoDataModule):

    def __init__(
        self,

        data_root: str,
        batch_size: int = 4,
        num_workers: int = 4,

        modality: str = FloodRiskMapNonGeo.all_modality_names[0],

        train_transform: A.Compose | None | list[A.BasicTransform] = None,
        val_transform: A.Compose | None | list[A.BasicTransform] = None,
        test_transform: A.Compose | None | list[A.BasicTransform] = None,
        predict_transform: A.Compose | None | list[A.BasicTransform] = None,

        drop_last: bool = True,
        constant_scale: float = 0.0001,

        no_data_replace: float | None = 0,
        no_target_replace: int | None = -1,

        for_prithvi: int=0,

        **kwargs: Any,
    ) -> None:

        super().__init__(FloodRiskMapNonGeo, batch_size, num_workers, **kwargs)
        self.data_root = data_root
        self.for_prithvi = for_prithvi

        if self.for_prithvi:
            means = statistics['mean']['PRITHVI_V2']
            stds = statistics['std']['PRITHVI_V2']
        else:
            means = statistics['mean'][modality]
            stds = statistics['std'][modality]

        self.modality = modality

        self.train_transform = wrap_in_compose_is_list(train_transform)
        self.val_transform = wrap_in_compose_is_list(val_transform)
        self.test_transform = wrap_in_compose_is_list(test_transform)
        self.predict_transform = wrap_in_compose_is_list(predict_transform)

        self.aug = AugmentationSequential(K.Normalize(means, stds), data_keys=None)
        self.drop_last = drop_last
        self.constant_scale = constant_scale

        self.no_data_replace = no_data_replace
        self.no_target_replace = no_target_replace

    def setup(self, stage: str) -> None:
        if stage in ["fit"]:
            self.train_dataset = self.dataset_class(
                split="train",
                data_root=self.data_root,
                transform=self.train_transform,
                modality=self.modality,
                constant_scale=self.constant_scale,
                no_data_replace=self.no_data_replace,
                no_target_replace=self.no_target_replace,
                for_prithvi = self.for_prithvi,
            )
        if stage in ["fit", "validate"]:
            self.val_dataset = self.dataset_class(
                split="val",
                data_root=self.data_root,
                transform=self.val_transform,
                modality=self.modality,
                constant_scale=self.constant_scale,
                no_data_replace=self.no_data_replace,
                no_target_replace=self.no_target_replace,
                for_prithvi = self.for_prithvi,
            )
        if stage in ["test"]:
            self.test_dataset = self.dataset_class(
                split="test",
                data_root=self.data_root,
                transform=self.test_transform,
                modality=self.modality,
                constant_scale=self.constant_scale,
                no_data_replace=self.no_data_replace,
                no_target_replace=self.no_target_replace,
                for_prithvi = self.for_prithvi,
            )
        if stage in ["predict"]:
            self.predict_dataset = self.dataset_class(
                split="test",
                data_root=self.data_root,
                transform=self.predict_transform,
                modality=self.modality,
                constant_scale=self.constant_scale,
                no_data_replace=self.no_data_replace,
                no_target_replace=self.no_target_replace,
                for_prithvi = self.for_prithvi,
            )

    def _dataloader_factory(self, split: str) -> DataLoader[dict[str, Tensor]]:
        """Implement one or more PyTorch DataLoaders.

        Args:
            split: Either 'train', 'val', 'test', or 'predict'.

        Returns:
            A collection of data loaders specifying samples.

        Raises:
            MisconfigurationException: If :meth:`setup` does not define a
                dataset or sampler, or if the dataset or sampler has length 0.
        """
        dataset = self._valid_attribute(f"{split}_dataset", "dataset")
        batch_size = self._valid_attribute(f"{split}_batch_size", "batch_size")
        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=split == "train",
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            drop_last=split == "train" and self.drop_last,
        )

        

