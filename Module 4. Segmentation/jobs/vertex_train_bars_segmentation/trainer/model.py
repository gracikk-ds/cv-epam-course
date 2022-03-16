import torch
import skimage
import numpy as np
from typing import Tuple
from torchmetrics import IoU
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
from torchmetrics import MeanAbsolutePercentageError


BATCH_SIZE = 32


def BarsSegmentation(
    classes=None,
    model="FPN",
    encoder_name: str = "efficientnet-b3",
    encoder_weights: str = "imagenet",
    activation: str = "sigmoid",
    decoder_attention_type: str = "scse",
    model_weights_path=None,
):
    if classes is None:
        classes = ["bars"]
    if hasattr(smp, model):
        model_class = getattr(smp, model)
        if isinstance(model_class, type):
            kwargs = {}
            if model == "UnetPlusPlus":
                kwargs = {"decoder_attention_type": decoder_attention_type}
            model = model_class(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                classes=len(classes),
                activation=activation,
                **kwargs,
            )
    else:
        raise ValueError(f"Unsupported model name {model}")

    if model_weights_path is not None:
        model.load_state_dict(torch.load(model_weights_path))

    preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder_name, encoder_weights)

    return model, preprocessing_fn


class Runner(pl.LightningModule):
    def __init__(self, model, classes, lr: float = 1e-2, scheduler_T=75) -> None:

        super().__init__()
        self.model = model
        self.classes = classes
        self.lr = lr
        self.scheduler_T = scheduler_T
        self.criterion = smp.utils.losses.DiceLoss()

        # define metric
        self.metrics = torch.nn.ModuleDict(
            {
                "IOUScore": IoU(num_classes=2),
                "mape": MeanAbsolutePercentageError(compute_on_step=True),
            }
        )

    def unorm(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        for t, m, s in zip(tensor, mean, std):
            t.mul_(s).add_(m)
        return tensor

    def visualization(self, images, target_masks, predicted_masks, batch_idx):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes = axes.ravel()

        image = np.moveaxis(self.unorm(images[0].cpu()).numpy(), 0, -1)
        axes[0].imshow(target_masks[0].detach().squeeze().cpu().numpy())
        axes[1].imshow(predicted_masks)
        axes[2].imshow(image, vmin=0, vmax=255)
        axes[0].set_title(f"target_masks, batch={batch_idx}")
        axes[1].set_title(f"predicted_masks, batch={batch_idx}")
        axes[2].set_title(f"image, batch={batch_idx}")
        plt.show()

    def count_blobs(self, predicted_masks):
        counts_predicted = []
        for i, mask in enumerate(predicted_masks.detach()):
            mask = (mask.squeeze().cpu().numpy() * 255).astype("uint8")
            mask = np.stack((mask,) * 3, axis=-1)
            mask[mask < 30] = 0
            mask[mask > 0] = 255

            blobs = skimage.feature.blob_log(
                mask, min_sigma=3, max_sigma=4, num_sigma=1, threshold=0.02
            )
            count_predicted = len(blobs)
            counts_predicted.append(count_predicted)

        return counts_predicted

    def forward(self, batch, *args, **kwargs):
        images = batch
        predicted_masks = self.model(images.float())
        return predicted_masks

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx
    ) -> torch.Tensor:
        images, target_masks, counts = batch
        predicted_masks = self.model(images.float())
        loss = self.criterion(predicted_masks, target_masks // 255)

        for i, metric in enumerate(self.metrics.values()):
            if i == 0:
                metric.update(predicted_masks, target_masks // 255)

        self.log("Train/Loss", loss.item(), on_step=True, batch_size=BATCH_SIZE)
        self.log(
            "Train/LR",
            self.lr_schedulers().get_last_lr()[0],
            on_step=True,
            batch_size=BATCH_SIZE,
        )

        return loss

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx
    ) -> torch.Tensor:

        images, target_masks, counts = batch

        predicted_masks = self.model(images.float())
        loss = self.criterion(predicted_masks, target_masks // 255)

        counts_predicted = torch.tensor(self.count_blobs(predicted_masks))
        counts_predicted[counts_predicted == 0] = 1
        counts = counts.cpu().reshape(-1)
        counts[counts == 0] = 1

        for i, metric in enumerate(self.metrics.values()):
            if i == 0:
                metric.update(predicted_masks, target_masks // 255)
            else:
                metric.update(counts_predicted, counts)

        self.log(
            "Validation/Classification Loss",
            loss.item(),
            on_step=True,
            batch_size=BATCH_SIZE,
        )

        return loss

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx
    ) -> torch.Tensor:

        images, target_masks, counts = batch

        predicted_masks = self.model(images.float())
        loss = self.criterion(predicted_masks, target_masks // 255)

        counts_predicted = torch.tensor(self.count_blobs(predicted_masks))
        counts_predicted[counts_predicted == 0] = 1
        counts = counts.cpu().reshape(-1)
        counts[counts == 0] = 1

        for i, metric in enumerate(self.metrics.values()):
            if i == 0:
                metric.update(predicted_masks, target_masks // 255)
            else:
                metric.update(counts_predicted, counts)

        return loss

    def validation_epoch_end(self, something) -> None:
        for name, metric in self.metrics.items():
            metric_val = metric.compute()
            self.log(f"Validation/{name}", metric_val, on_step=False, on_epoch=True)
            metric.reset()

    def configure_optimizers(self):
        params = list(filter(lambda p: p.requires_grad, self.model.parameters()))
        optimizer = torch.optim.Adam(params, lr=self.lr)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer, T_max=self.scheduler_T, eta_min=1e-5
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler},
        }
