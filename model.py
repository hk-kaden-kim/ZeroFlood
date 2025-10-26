from terratorch.tasks import SemanticSegmentationTask
from torch import nn
from torchmetrics import ClasswiseWrapper, MetricCollection
from torchmetrics.classification import MulticlassAccuracy, MulticlassPrecision, MulticlassRecall, MulticlassF1Score, MulticlassJaccardIndex
from terratorch.tasks import SemanticSegmentationTask


class CustomSemanticSegmentationTask(SemanticSegmentationTask):
    """Customize only metrics configuration."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def configure_metrics(self) -> None:
        """Initialize the performance metrics."""
        num_classes: int = self.hparams["model_args"]["num_classes"]
        ignore_index: int = self.hparams["ignore_index"]
        class_names = self.hparams["class_names"]
        metrics = MetricCollection(
            {
                "mIoU": MulticlassJaccardIndex(
                    num_classes=num_classes,
                    ignore_index=ignore_index,
                    average="macro",
                ),
                "mIoU_Micro": MulticlassJaccardIndex(
                    num_classes=num_classes,
                    ignore_index=ignore_index,
                    average="micro",
                ),
                "F1_Score": MulticlassF1Score(
                    num_classes=num_classes,
                    ignore_index=ignore_index,
                    average="macro",
                ),
                "Accuracy": MulticlassAccuracy(
                    num_classes=num_classes,
                    ignore_index=ignore_index,
                    average="macro",
                ),
                "Pixel_Accuracy": MulticlassAccuracy(
                    num_classes=num_classes,
                    ignore_index=ignore_index,
                    average="micro",
                ),
                "Precision": ClasswiseWrapper(      # Added instead of FAR (False Alarm Rate)
                    MulticlassPrecision(num_classes=num_classes, ignore_index=ignore_index, average=None),
                    labels=class_names,
                    prefix="Precision_",
                ),
                "Recall": ClasswiseWrapper(          # Added instead of HR (Hit Rate)
                    MulticlassRecall(num_classes=num_classes, ignore_index=ignore_index, average=None),
                    labels=class_names,
                    prefix="Recall_",
                ),
                "IoU": ClasswiseWrapper(
                    MulticlassJaccardIndex(num_classes=num_classes, ignore_index=ignore_index, average=None),
                    labels=class_names,
                    prefix="IoU_",
                ),
                "Class_Accuracy": ClasswiseWrapper(
                    MulticlassAccuracy(
                        num_classes=num_classes,
                        ignore_index=ignore_index,
                        average=None,
                    ),
                    labels=class_names,
                    prefix="Class_Accuracy_",
                ),
            }
        )
        self.train_metrics = metrics.clone(prefix="train/")
        self.val_metrics = metrics.clone(prefix="val/")
        if self.hparams["test_dataloaders_names"] is not None:
            self.test_metrics = nn.ModuleList(
                [metrics.clone(prefix=f"test/{dl_name}/") for dl_name in self.hparams["test_dataloaders_names"]]
            )
        else:
            self.test_metrics = nn.ModuleList([metrics.clone(prefix="test/")])

class PrithviModel():
    def __init__(self, size, num_classes=2):

        model_args = {
            "backbone_pretrained": True,
            "backbone_bands": [
                "BLUE", "GREEN", "RED",     # index: 1, 2, 3
                "NIR_NARROW",               # index: 8
                "SWIR_1", "SWIR_2"],        # index: 10, 11
        }

        if size in ['300', '600']:
            backbone = f"prithvi_eo_v2_{size}"
            necks = [
                {"name": "ReshapeTokensToImage"},
                {"name": "SelectIndices", "indices": [5, 11, 17, 23] if size == '300' else [7, 15, 23, 31]},
                {"name": "LearnedInterpolateToPyramidal"}
                ]
        else:
            assert False, f"{size} is not supported... choose one of 'base' 'large' "
        
        model_args['backbone'] = backbone
        model_args['necks'] = necks

        model_args['num_classes'] = num_classes
        model_args['decoder'] = 'UNetDecoder'
        model_args['decoder_channels'] = [512, 256, 128, 64]

        self.model_args = model_args

    def get_args(self):
        return self.model_args

class TerraMindModel():
    def __init__(self, size, modality, tim=[], num_classes=2):

        model_args = {
            "backbone_modalities": modality,
            "backbone_pretrained": True,
        }

        if size in ['base', 'large']:
            backbone = f"terramind_v1_{size}"
            necks = [
                {"name": "ReshapeTokensToImage", "remove_cls_token": False},
                {"name": "SelectIndices", "indices": [2, 5, 8, 11] if size == 'base' else [5, 11, 17, 23]},
                {"name": "LearnedInterpolateToPyramidal"}
                ]
        else:
            assert False, f"{size} is not supported... choose one of 'base' 'large' "
        
        if len(tim) != 0:
            backbone += "_tim"
            model_args['backbone_tim_modalities'] = tim

        model_args['backbone'] = backbone
        model_args['necks'] = necks

        model_args['num_classes'] = num_classes
        model_args['decoder'] = 'UNetDecoder'
        model_args['decoder_channels'] = [512, 256, 128, 64]

        self.model_args = model_args

    def get_args(self):
        return self.model_args
