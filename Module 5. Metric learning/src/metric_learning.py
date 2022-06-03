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
from sklearn.preprocessing import StandardScaler
from pytorch_metric_learning import losses, miners
from sklearn.model_selection import train_test_split
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator


BATCH_SIZE = 32


class EmbeddingsModel(nn.Module):
    def __init__(
        self,
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

    def forward(self, inpt):
        # get embeddings
        emb = self.trunk(inpt)
        return emb


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


def sampling(embeddings, targets, N, umap=True):
    """
    stratified sampling to speed up validation
    :param embeddings: full embeddings
    :param targets: full targets
    :param N: number of samples to save per class
    :param umap: sampling for umap or not
    :return: subsample of embeddings and targets
    """
    embeddings = embeddings.numpy()
    targets = targets.numpy()
    df = pd.DataFrame.from_records(data=embeddings)
    df["target"] = targets

    if umap:
        df = df.groupby("target", group_keys=False).apply(
            lambda x: x.sample(min(len(x), N), random_state=42)
        )
        targets = df["target"].values
        df.drop(columns=["target"], inplace=True)
        embeddings = df.values

        return torch.tensor(embeddings).contiguous(), torch.tensor(targets).contiguous()
    else:
        frame = pd.DataFrame(df.loc[:, ["target"]].value_counts().reset_index())
        frame.columns = ["target", "count"]
        classes_to_test = frame.loc[frame["count"] >= 4].target.values[:N]

        df = df.loc[df["target"].isin(classes_to_test)]

        (
            embeddings_train,
            embeddings_test,
            targets_train,
            targets_test,
        ) = train_test_split(
            df.drop(columns=["target"]).values,
            df["target"].values,
            test_size=0.5,
            stratify=df["target"].values,
        )

        embeddings_gallery = torch.tensor(embeddings_train).contiguous()
        embeddings_query = torch.tensor(embeddings_test).contiguous()
        targets_gallery = torch.tensor(targets_train).contiguous()
        targets_query = torch.tensor(targets_test).contiguous()

        return embeddings_gallery, embeddings_query, targets_gallery, targets_query


def calculate_accuracy(trainer, train_dl, val_dl):
    embeddings_train, targets_train = get_embeddings(trainer, train_dl)
    embeddings_val, targets_val = get_embeddings(trainer, val_dl)

    (
        embeddings_gallery_train,
        embeddings_query_train,
        targets_gallery_train,
        targets_query_train,
    ) = sampling(embeddings_train, targets_train, 300, umap=False)

    (
        embeddings_gallery_val,
        embeddings_query_val,
        targets_gallery_val,
        targets_query_val,
    ) = sampling(embeddings_val, targets_val, 300, umap=False)

    accuracy_calculator_train = AccuracyCalculator(device=torch.device("cpu"))

    accuracies_train = accuracy_calculator_train.get_accuracy(
        embeddings_query_train,
        embeddings_gallery_train,
        targets_query_train,
        targets_gallery_train,
        False,
    )

    accuracy_calculator_test = AccuracyCalculator(device=torch.device("cpu"))

    accuracies_test = accuracy_calculator_test.get_accuracy(
        embeddings_query_val,
        embeddings_gallery_val,
        targets_query_val,
        targets_gallery_val,
        False,
    )

    return accuracies_train, accuracies_test


class Runner(pl.LightningModule):
    def __init__(
        self,
        model,
        classes,
        mapper,
        lr: float = 1e-3,
        scheduler_T=1000,
    ) -> None:

        super().__init__()

        self.model = model
        self.classes = classes
        self.lr = lr
        self.scheduler_T = scheduler_T
        self.miner = miners.MultiSimilarityMiner(epsilon=0.1)
        self.metric_loss = losses.SubCenterArcFaceLoss(
            num_classes=len(classes), embedding_size=model.embedding_size
        ).to(torch.device("cuda"))

        self.mapper = mapper

        self.accuracy_calculator = AccuracyCalculator(device=torch.device("cpu"))

    def predict_step(self, batch, batch_idx, **kwargs):
        images, targets = batch
        embeddings = self.model(images)
        return embeddings, targets

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx):

        images, targets = batch

        embeddings = self.model(images)

        # calculating metric loss
        hard_pairs = self.miner(embeddings, targets)
        m_loss = self.metric_loss(embeddings, targets, hard_pairs)

        self.log(
            "Train/Metric Loss",
            m_loss.item(),
            on_step=True,
            batch_size=BATCH_SIZE,
        )

        self.log(
            "Train/LR",
            self.lr_schedulers().get_last_lr()[0],
            on_step=True,
            batch_size=BATCH_SIZE,
        )

        return {"loss": m_loss, "embeddings": embeddings, "targets": targets}

    def training_epoch_end(self, training_epoch_outputs):

        print(training_epoch_outputs.keys())
        print(training_epoch_outputs["embeddings"].shape)

        embeddings_train = training_epoch_outputs["embeddings"]
        targets_train = training_epoch_outputs["targets"]

        # embeddings_train = torch.concat(embeddings_train)
        # targets_val = torch.concat(targets_val)

        (
            embeddings_gallery_train,
            embeddings_query_train,
            targets_gallery_train,
            targets_query_train,
        ) = sampling(embeddings_train, targets_train, 300, umap=False)

        accuracy_calculator_train = AccuracyCalculator(device=torch.device("cpu"))

        accuracies_train = accuracy_calculator_train.get_accuracy(
            embeddings_query_train,
            embeddings_gallery_train,
            targets_query_train,
            targets_gallery_train,
            False,
        )

        for name in accuracies_train:
            self.log(
                f"Train/{name}", accuracies_train[name], on_step=False, on_epoch=True
            )

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx):

        images, targets = batch
        embeddings = self.model(images)

        hard_pairs = self.miner(embeddings, targets)
        m_loss = self.metric_loss(embeddings, targets, hard_pairs)

        self.log(
            "Validation/Metric learning Loss",
            m_loss.item(),
            on_step=True,
            batch_size=BATCH_SIZE,
        )
        return {"loss": m_loss, "embeddings": embeddings, "targets": targets}

    def validation_epoch_end(self, validation_epoch_outputs) -> None:

        print(validation_epoch_outputs.keys())
        print(validation_epoch_outputs["embeddings"].shape)

        embeddings_val = validation_epoch_outputs["embeddings"]
        targets_val = validation_epoch_outputs["targets"]

        if len(embeddings_val) != 0:
            # embeddings_val = torch.concat(embeddings_val)
            # targets_val = torch.concat(targets_val)

            (
                embeddings_gallery_val,
                embeddings_query_val,
                targets_gallery_val,
                targets_query_val,
            ) = sampling(embeddings_val, targets_val, 300, umap=False)

            accuracy_calculator_test = AccuracyCalculator(device=torch.device("cpu"))

            accuracies_test = accuracy_calculator_test.get_accuracy(
                embeddings_query_val,
                embeddings_gallery_val,
                targets_query_val,
                targets_gallery_val,
                False,
            )

            for name in accuracies_test:
                self.log(
                    f"Validation/{name}",
                    accuracies_test[name],
                    on_step=False,
                    on_epoch=True,
                )

            self.log_umap(
                embeddings=embeddings_val.numpy(),
                targets=targets_val.numpy(),
            )

    def log_umap(self, embeddings, targets):
        targets = [x.split("_")[-1] for x in targets]
        print(targets)
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
