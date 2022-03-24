import io
from typing import Tuple

import numpy as np
import pytorch_lightning as pl
import timm
import torch
import torch.nn as nn
from PIL import Image
from matplotlib import pyplot as plt
import seaborn as sns
from pytorch_metric_learning import losses, miners  # , testers#, samplers  # , reducers

# from pytorch_metric_learning.utils.accuracy_calculator import (
#     AccuracyCalculator,
#     precision_at_k,
# )
from torchmetrics import Accuracy, ConfusionMatrix

from .loss import FocalLoss

BATCH_SIZE = 48


class EmbeddingsModel(nn.Module):
    def __init__(
        self,
        num_classes: int,
        embedding_size: int = 512,
        backbone: str = "resnext101_32x8d",
    ):
        super().__init__()
        self.backbone = backbone

        self.trunk = timm.create_model(backbone, pretrained=True)
        self.trunk_output_size = self.trunk.fc.in_features
        self.embedding_size = embedding_size
        self.num_classes = num_classes

        self.trunk.fc = nn.Linear(
            in_features=self.trunk.fc.in_features,
            out_features=embedding_size,
            bias=False,
        )
        self.classifier = torch.nn.Sequential(
            nn.Linear(embedding_size, num_classes, bias=True),
        )

    #         for p in self.trunk.parameters():
    #             p.requires_grad = False
    #         for p in self.trunk.layer4.parameters():
    #             p.requires_grad = True
    #         for p in self.trunk.fc.parameters():
    #             p.requires_grad = True

    def forward(self, inpt):
        emb = self.trunk(inpt)
        logits = self.classifier(emb)

        return logits, emb


class Runner(pl.LightningModule):
    def __init__(
        self,
        model,
        classes,
        class_mapping,
        #                  ignore_index,
        lr: float = 1e-3,
        scheduler_T=1000,
        metric_coeff: float = 0.2,
    ) -> None:
        super().__init__()
        self.model = model
        self.classes = classes
        self.class_mapping = class_mapping
        self.mapped_classes = sorted(set(class_mapping.values()))
        assert model.num_classes == len(self.mapped_classes)
        self.lr = lr
        self.scheduler_T = scheduler_T
        self.criterion = FocalLoss()
        self.metric_coeff = metric_coeff
        #         self.ignore_index = ignore_index

        num_classes = len(self.mapped_classes)
        class_mapper = np.zeros((len(classes), num_classes))
        mapped_classes_to_idx = {clz: i for i, clz in enumerate(self.mapped_classes)}
        self.target_mapping = {
            i: mapped_classes_to_idx[class_mapping[clz]]
            for i, clz in enumerate(classes)
        }
        for i, j in self.target_mapping.items():
            class_mapper[i][j] = 1

        self.class_mapper = torch.from_numpy(class_mapper)

        self.miner = miners.MultiSimilarityMiner(epsilon=0.1)
        self.metric_loss = losses.CosFaceLoss(
            num_classes=len(classes), embedding_size=model.embedding_size
        )

        self.metrics = torch.nn.ModuleDict(
            {
                "accuracy": Accuracy(
                    num_classes=num_classes, compute_on_step=False, average="macro"
                ),
                "confusion_matrix": ConfusionMatrix(
                    num_classes=num_classes, normalize="true", compute_on_step=False
                ),
            }
        )


    def transform_targets(self, targets):
        mapped_targets = torch.tensor(
            [self.target_mapping[t.item()] for t in targets], device=targets.device
        )

        return mapped_targets

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], *args: list
    ) -> torch.Tensor:
        images, targets = batch[0], batch[1]

        logits, embeddings = self.model(images)

        clf_loss = self.criterion(logits, self.transform_targets(targets))

        hard_pairs = self.miner(embeddings, targets)
        m_loss = self.metric_loss(embeddings, targets, hard_pairs)

        loss = clf_loss + self.metric_coeff * m_loss

        self.log("Train/Loss", loss.item(), on_step=True, batch_size=BATCH_SIZE)
        self.log(
            "Train/Metric Loss", m_loss.item(), on_step=True, batch_size=BATCH_SIZE
        )
        self.log(
            "Train/Classification Loss",
            clf_loss.item(),
            on_step=True,
            batch_size=BATCH_SIZE,
        )
        self.log(
            "Train/LR",
            self.lr_schedulers().get_last_lr()[0],
            on_step=True,
            batch_size=BATCH_SIZE,
        )

        return loss

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], *args: list
    ) -> torch.Tensor:
        images, targets = batch[0], batch[1]
        logits, embeddings = self.model(images)

        targets = self.transform_targets(targets)

        clf_loss = self.criterion(logits, targets)

        preds = logits.softmax(axis=1)
        for metric in self.metrics.values():
            metric(preds=preds, target=targets)

        self.log(
            "Validation/Classification Loss",
            clf_loss.item(),
            on_step=True,
            batch_size=BATCH_SIZE,
        )

        return clf_loss

    def log_cm(self, confusion_matrix):
        plt.figure(figsize=(50, 50))
        sns.heatmap(
            np.around(confusion_matrix.cpu().numpy(), 3),
            annot=True,
            cmap="YlGnBu",
            xticklabels=self.mapped_classes,
            yticklabels=self.mapped_classes,
        )
        buf = io.BytesIO()
        plt.savefig(buf)
        buf.seek(0)
        image = np.array(Image.open(buf))[:, :, :3]
        buf.close()
        plt.clf()
        self.logger.experiment.add_image(
            "conf_matr", image, self.current_epoch, dataformats="HWC"
        )

    def validation_epoch_end(self, outputs) -> None:
        for name, metric in self.metrics.items():
            metric_val = metric.compute()
            self.log(f"Validation/{name}", metric_val, on_step=False, on_epoch=True)
            metric.reset()
            if name != "confusion_matrix":
                print(f"Validation {name} = {metric_val}")
            else:
                self.log_cm(metric_val)

    def configure_optimizers(self):
        params = list(filter(lambda p: p.requires_grad, self.model.parameters()))
        optimizer = torch.optim.Adam(params, lr=self.lr)
        if len([p for p in self.metric_loss.parameters()]) > 0:
            optimizer.add_param_group({"params": self.metric_loss.parameters()})
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer, T_max=self.scheduler_T, eta_min=1e-8
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler},
        }
