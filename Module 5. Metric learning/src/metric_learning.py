import io
import timm
import umap
import torch
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
import torch.nn as nn
from typing import Tuple
import pytorch_lightning as pl
from matplotlib import pyplot as plt
from torch.nn import CrossEntropyLoss
from sklearn.preprocessing import StandardScaler
from torchmetrics import Accuracy, ConfusionMatrix
from pytorch_metric_learning import losses, miners
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator


BATCH_SIZE = 32


class EmbeddingsModel(nn.Module):
    def __init__(
        self,
        num_classes: int,
        embedding_size: int = 512,
        backbone: str = "resnext101_32x8d",
    ):
        super().__init__()
        self.trunk = timm.create_model(backbone, pretrained=True)
        self.embedding_size = embedding_size
        self.trunk.fc = nn.Linear(
            in_features=self.trunk.fc.in_features,
            out_features=embedding_size,
            bias=False,
        )

        self.classifier = torch.nn.Sequential(
            nn.Linear(embedding_size, num_classes, bias=True),
        )

    def forward(self, inpt):
        # get embeddings
        emb = self.trunk(inpt)

        # get logits
        logits = self.classifier(emb)

        return logits, emb


def get_embeddings(trainer, loader):
    embeddings = trainer.predict(dataloaders=loader, ckpt_path="best")

    embs = []
    targets = []
    for element in embeddings:
        emb, target = element
        embs.append(emb)
        targets.append(target)

    embeddings, targets = torch.concat(embs), torch.concat(targets)
    return embeddings.cpu(), targets.cpu()


def sampling(embeddings, targets, N):
    """
    stratified sampling to speed up validation
    :param embeddings: full embeddings
    :param targets: full targets
    :param N: number of samples to save per class
    :return: subsample of embeddings and targets
    """
    embeddings = embeddings.numpy()
    targets = targets.numpy()
    df = pd.DataFrame.from_records(data=embeddings)
    df["target"] = targets
    df = df.groupby("target", group_keys=False).apply(
        lambda x: x.sample(min(len(x), N), random_state=42)
    )
    targets = df["target"].values
    df.drop(columns=["target"], inplace=True)
    embeddings = df.values

    return torch.tensor(embeddings).contiguous(), torch.tensor(targets).contiguous()


def calculate_accuracy(trainer, train_dl, val_dl):
    embeddings_train, targets_train = get_embeddings(trainer, train_dl)
    embeddings_val, targets_val = get_embeddings(trainer, val_dl)

    embeddings_train, targets_train = sampling(embeddings_train, targets_train, 300)
    embeddings_val, targets_val = sampling(embeddings_val, targets_val, 300)

    accuracy_calculator = AccuracyCalculator(device=torch.device("cpu"))

    accuracies = accuracy_calculator.get_accuracy(
        embeddings_train, embeddings_val, targets_train, targets_val, False
    )

    return accuracies


class Runner(pl.LightningModule):
    def __init__(
        self,
        model,
        classes,
        mapper,
        lr: float = 1e-3,
        scheduler_T=1000,
        metric_coeff: float = 0.3,
    ) -> None:

        super().__init__()

        self.model = model
        self.classes = classes
        self.lr = lr
        self.scheduler_T = scheduler_T
        self.criterion = CrossEntropyLoss()
        self.metric_coeff = metric_coeff
        self.miner = miners.MultiSimilarityMiner(epsilon=0.1)
        self.metric_loss = losses.SubCenterArcFaceLoss(  # ArcFaceLoss
            num_classes=len(classes), embedding_size=model.embedding_size
        ).to(torch.device("cuda"))

        num_classes = len(self.classes)
        self.mapper = mapper

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
        self.accuracy_calculator = AccuracyCalculator(device=torch.device("cpu"))

        self.embeddings_train = []
        self.embeddings_val = []
        self.targets_train = []
        self.targets_val = []

    def predict_step(self, batch, batch_idx, **kwargs):
        images, targets = batch
        logits, embeddings = self.model(images)
        return embeddings, targets

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx
    ) -> torch.Tensor:

        images, targets = batch

        logits, embeddings = self.model(images)

        self.embeddings_train.append(embeddings.detach().cpu())
        self.targets_train.append(targets.detach().cpu())

        # calculating classification loss
        clf_loss = self.criterion(logits, targets)

        # calculating metric loss
        hard_pairs = self.miner(embeddings, targets)
        m_loss = self.metric_loss(embeddings, targets, hard_pairs)

        loss = self.metric_coeff * clf_loss + (1 - self.metric_coeff) * m_loss

        # calculating metrics
        for i, metric in enumerate(self.metrics.values()):
            metric.update(logits.softmax(axis=1), targets)

        self.log(
            "Train/Metric Loss",
            m_loss.item(),
            on_step=True,
            batch_size=BATCH_SIZE,
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
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx
    ) -> torch.Tensor:

        images, targets = batch
        logits, embeddings = self.model(images)
        self.embeddings_val.append(embeddings.detach().cpu())
        self.targets_val.append(targets.detach().cpu())

        clf_loss = self.criterion(logits, targets)

        # calculating metrics
        for i, metric in enumerate(self.metrics.values()):
            metric.update(logits.softmax(axis=1), targets)

        self.log(
            "Validation/Classification Loss",
            clf_loss.item(),
            on_step=True,
            batch_size=BATCH_SIZE,
        )
        return clf_loss

    def log_cm(self, confusion_matrix):
        print("start drawing conf matrix")
        plt.figure(figsize=(50, 50))
        sns.heatmap(
            np.around(confusion_matrix.cpu().numpy(), 3),
            annot=True,
            cmap="YlGnBu",
            xticklabels=self.classes,
            yticklabels=self.classes,
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
        print("done!")

    def log_umap(self, embeddings, targets):
        print("run umap logger")
        sns.set(style="whitegrid", font_scale=1.3)

        print("     run StandardScaler")
        scaler = StandardScaler()
        embeddings_scaled = scaler.fit_transform(embeddings)
        print("     done!")

        print("     sampling")
        embeddings_scaled, targets = sampling(
            torch.tensor(embeddings_scaled), torch.tensor(targets), 50
        )
        print("     done!")

        print("     starting umap transforms")
        umap_obj = umap.UMAP(n_neighbors=20, min_dist=0.15)
        embedding_2d = umap_obj.fit_transform(embeddings_scaled.numpy())
        print("     done!")

        plot_df = pd.DataFrame.from_records(data=embedding_2d, columns=["x", "y"])
        plot_df["target"] = targets.numpy()
        plot_df["target"] = plot_df["target"].apply(lambda x: self.mapper[x])

        plt.figure(figsize=(14, 10))
        plt.title("UMAP")
        sns.scatterplot(x="x", y="y", data=plot_df, hue="target", palette="Paired")

        buf = io.BytesIO()
        plt.savefig(buf)
        buf.seek(0)
        image = np.array(Image.open(buf))[:, :, :3]
        buf.close()
        plt.clf()
        self.logger.experiment.add_image(
            "umap", image, self.current_epoch, dataformats="HWC"
        )
        print("done!")

    def validation_epoch_end(self, outputs) -> None:

        if len(self.embeddings_train) != 0:
            self.embeddings_train = torch.concat(self.embeddings_train)
            self.targets_train = torch.concat(self.targets_train)
            self.embeddings_val = torch.concat(self.embeddings_val)
            self.targets_val = torch.concat(self.targets_val)
            print("embedding example", self.embeddings_train[0][:10])
            print("embeddings train shape: ", self.embeddings_train.shape)
            print("embeddings val shape: ", self.embeddings_val.shape)

            self.embeddings_train, self.targets_train = sampling(
                self.embeddings_train, self.targets_train, 300
            )
            self.embeddings_val, self.targets_val = sampling(
                self.embeddings_val, self.targets_val, 300
            )

            # embeddings metrices
            print("accuracy_calculator start")
            accuracies = self.accuracy_calculator.get_accuracy(
                self.embeddings_train,
                self.embeddings_val,
                self.targets_train,
                self.targets_val,
                False,
            )
            print("done!")

            for name in accuracies:
                self.log(
                    f"Validation/{name}", accuracies[name], on_step=False, on_epoch=True
                )

            self.log_umap(
                embeddings=self.embeddings_val.numpy(),
                targets=self.targets_val.numpy(),
            )

            self.embeddings_train = []
            self.embeddings_val = []
            self.targets_train = []
            self.targets_val = []

        # classification metrices
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
            optimizer=optimizer, T_max=self.scheduler_T, eta_min=1e-7
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler},
        }
