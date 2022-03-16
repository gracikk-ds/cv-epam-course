import cv2
import torch
import numpy as np
from typing import Tuple
from torchmetrics import IoU
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
from torchmetrics.detection.map import MeanAveragePrecision


BATCH_SIZE = 32


def BarcodeSegmentation(
    classes=None,
    model="FPN",
    encoder_name: str = "efficientnet-b3",
    encoder_weights: str = "imagenet",
    activation: str = "sigmoid",
    decoder_attention_type: str = "scse",
    model_weights_path=None,
):
    if classes is None:
        classes = ["barcode"]
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


def process_cnts(cnts):
    if cnts:
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        for i, cnt in enumerate(cnts):
            area = cv2.contourArea(cnt)
            if area > 50:
                continue
            else:
                cnts = cnts[:i]
                break
    return cnts


def detect_barcode(masks, target):
    masks = masks.squeeze().cpu().numpy()

    detections = []

    for i, mask in enumerate(masks):
        mask = np.uint8(mask.squeeze() * 255)
        (cnts, _) = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = process_cnts(cnts)

        # find all bboxes
        if cnts:
            barcodes = []
            for cnt in cnts:
                coords = list(cv2.boundingRect(cnt))
                coords[2] += coords[0]
                coords[3] += coords[1]
                barcodes.append(coords)

            # append them to list
            if target:
                dict_ = dict(
                    boxes=torch.Tensor(barcodes), labels=torch.zeros(len(barcodes))
                )
            else:
                dict_ = dict(
                    boxes=torch.Tensor(barcodes),
                    scores=torch.ones(len(barcodes)) * 0.9,
                    labels=torch.zeros(len(barcodes)),
                )

            detections.append(dict_)
        else:
            # create empy predictions list
            if target:
                dict_ = dict(boxes=torch.Tensor([]), labels=torch.Tensor([]))

            else:
                dict_ = dict(
                    boxes=torch.Tensor([]),
                    scores=torch.Tensor([]),
                    labels=torch.Tensor([]),
                )

            detections.append(dict_)

    return detections


class Runner(pl.LightningModule):
    def __init__(self, model, classes, lr: float = 1e-3, scheduler_T=1000) -> None:

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
                "mAP": MeanAveragePrecision(box_format="xyxy"),
            }
        )

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx
    ) -> torch.Tensor:
        images, target_masks = batch
        predicted_masks = self.model(images.float())
        loss = self.criterion(predicted_masks, target_masks // 255)
        bboxes_target = detect_barcode(target_masks // 255, target=True)
        bboxes_predicted = detect_barcode(predicted_masks, target=False)

        # if batch_idx % 100 == 0:
        #     fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        #     axes = axes.ravel()
        #
        #     image = np.moveaxis(unorm(images[0].cpu()).numpy(), 0, -1)
        #     axes[0].imshow(target_masks[0].detach().squeeze().cpu().numpy())
        #     axes[1].imshow(predicted_masks[0].detach().squeeze().cpu().numpy())
        #     axes[2].imshow(image, vmin=0, vmax=255)
        #     axes[0].set_title(f"target_masks, batch={batch_idx}")
        #     axes[1].set_title(f"predicted_masks, batch={batch_idx}")
        #     axes[2].set_title(f"image, batch={batch_idx}")
        #     plt.show()

        for i, metric in enumerate(self.metrics.values()):
            if i == 1:
                metric.update(bboxes_predicted, bboxes_target)
            else:
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
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx
    ) -> torch.Tensor:

        images, target_masks = batch

        predicted_masks = self.model(images.float())
        loss = self.criterion(predicted_masks, target_masks // 255)

        bboxes_target = detect_barcode(target_masks // 255, target=True)
        bboxes_predicted = detect_barcode(predicted_masks, target=False)

        # if batch_idx in [1]:
        #     print("\nValidation Start \n", "******" * 10, )
        #
        # if batch_idx in [1, 2]:
        #     fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        #     axes = axes.ravel()
        #
        #     image = np.moveaxis(unorm(images[0].cpu()).numpy(), 0, -1)  # ().astype(np.uint8)
        #     target_mask = (target_masks[0].squeeze().cpu().numpy()).astype(np.uint8)
        #     predicted_mask = (predicted_masks[0].squeeze().cpu().numpy()).astype(np.uint8)
        #
        #     axes[0].imshow(target_mask)
        #     axes[1].imshow(predicted_mask)
        #     axes[2].imshow(image)
        #
        #     axes[0].set_title("target_masks")
        #     axes[1].set_title("predicted_masks")
        #     axes[2].set_title("image")
        #     plt.show()

        for i, metric in enumerate(self.metrics.values()):
            if i == 1:
                print(bboxes_predicted)
                print(bboxes_target)
                metric.update(bboxes_predicted, bboxes_target)
            else:
                metric.update(predicted_masks, target_masks // 255)

        self.log(
            "Validation/Classification Loss",
            loss.item(),
            on_step=True,
            batch_size=BATCH_SIZE,
        )

        return loss

    def validation_epoch_end(self, something) -> None:
        for name, metric in self.metrics.items():
            metric_val = metric.compute()
            self.log(f"Validation/{name}", metric_val, on_step=False, on_epoch=True)
            metric.reset()

    def test_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx
    ) -> torch.Tensor:

        images, target_masks = batch

        predicted_masks = self.model(images.float())
        loss = self.criterion(predicted_masks, target_masks // 255)

        bboxes_target = detect_barcode(target_masks // 255, target=True)
        bboxes_predicted = detect_barcode(predicted_masks, target=False)

        # for i, (bbox_t, bbox_r, img, msk) in enumerate(zip(bboxes_target, bboxes_predicted, images, predicted_masks)):
        #     msk = (msk.squeeze().cpu().numpy())
        #     msk = cv2.cvtColor((msk * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        #     img = unorm(img.cpu()).numpy()
        #     img = np.moveaxis(img, 0, -1)
        #     img = np.ascontiguousarray(img * 255, dtype=np.uint8)
        #     img = cv2.addWeighted(img, 0.3, msk, 0.7, 0)
        #     bboxes_t = bbox_t["boxes"]
        #     bboxes_r = bbox_r["boxes"]
        #     for bbox_ in bboxes_t:
        #         bbox_ = bbox_.numpy().astype(int)
        #         img = cv2.rectangle(img, (bbox_[0], bbox_[1]), (bbox_[2], bbox_[3]), (0, 0, 255), 1)
        #     for bbox_ in bboxes_r:
        #         bbox_ = bbox_.numpy().astype(int)
        #         img = cv2.rectangle(img, (bbox_[0], bbox_[1]), (bbox_[2], bbox_[3]), (255, 0, 0), 3)
        #     cv2.imwrite("./results/img_" + str(batch_idx) + "_" + str(i) + ".png", img)
        #     # plt.imshow(img)
        #     # plt.show()

        for i, metric in enumerate(self.metrics.values()):

            if i == 1:
                metric.update(bboxes_predicted, bboxes_target)
            else:
                metric.update(predicted_masks, target_masks // 255)

        self.log(
            "Test/Classification Loss",
            loss.item(),
            on_step=True,
            batch_size=BATCH_SIZE,
        )

        return loss

    def test_epoch_end(self, something) -> None:
        for name, metric in self.metrics.items():
            metric_val = metric.compute()
            self.log(f"Test/{name}", metric_val, on_step=False, on_epoch=True)
            metric.reset()

    def configure_optimizers(self):
        params = list(filter(lambda p: p.requires_grad, self.model.parameters()))
        optimizer = torch.optim.Adam(params, lr=self.lr)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer, T_max=self.scheduler_T, eta_min=1e-8
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler},
        }
