import os
import click
import pickle
import logging
import contextlib
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

import pytorch_lightning as pl
from pytorch_metric_learning.samplers import MPerClassSampler
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from transformations import Transforms
from metric_learning import (
    EmbeddingsModel,
    Runner,
    BATCH_SIZE,
    calculate_accuracy,
)


SIZE = 224
BACKBONE = "resnext101_32x8d"


@click.command()
@click.option(
    "--dataset_folder",
    help="path to processed.",
    default="/training/data/processed/dataset",
)
@click.option("--tb_log_dir", help="GCS path to tb_dir.", default="/training/logs/")
@click.option("--model_dir", help="GCS path to model_dir.", default="/training/models/")
@click.option("--max_epochs", default=30)
def main(
    dataset_folder: str,
    tb_log_dir: str,
    model_dir: str,
    max_epochs: int,
):
    os.makedirs(tb_log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    tb_log_dir_to_use = Path(tb_log_dir)
    model_dir_to_use = Path(model_dir)
    dataset_folder_to_use = Path(dataset_folder)
    path = tb_log_dir_to_use / "stdout.txt"
    path_err = tb_log_dir_to_use / "stderr.txt"

    # setting up logger
    logging.basicConfig(
        filename=str(tb_log_dir_to_use / "logs.log"),
        filemode="a",
        format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        level=logging.INFO,
    )

    logger = logging.getLogger("metric_learning_logs")
    logger.info("Running metric learning task!")

    with open(path, "w") as f, open(path_err, "w") as e:
        with contextlib.redirect_stdout(f), contextlib.redirect_stderr(e):
            classes_train = set(
                [p.name for p in (dataset_folder_to_use / "train").glob("*")]
            )
            classes_val = set(
                [p.name for p in (dataset_folder_to_use / "test").glob("*")]
            )

            logger.info(f"Number of classes in train {len(classes_train)}")
            logger.info(f"Number of classes in val {len(classes_val)}")
            logger.info(
                f"Number of classes in train & val {len(classes_train & classes_val)}"
            )
            logger.info(
                f"Number of classes in train - val {len(classes_train - classes_val)}"
            )

            logger.info("creating datasets")

            train_dataset = ImageFolder(
                root=str(dataset_folder_to_use / "train"),
                transform=Transforms(),
            )

            val_dataset = ImageFolder(
                root=str(dataset_folder_to_use / "test"),
                transform=Transforms(segment="val"),
            )

            mapper = {
                train_dataset.class_to_idx[i]: i for i in train_dataset.class_to_idx
            }

            logger.info("datasets were created")

            logger.info("creating data loaders")
            sampler = MPerClassSampler(
                train_dataset.targets,
                m=3,
                length_before_new_iter=len(train_dataset),
            )

            train_dl = DataLoader(
                train_dataset,
                BATCH_SIZE,
                pin_memory=False,
                sampler=sampler,
                num_workers=12,
                drop_last=True,
            )

            val_dl = DataLoader(
                val_dataset,
                BATCH_SIZE,
                pin_memory=False,
                shuffle=False,
                num_workers=12,
                drop_last=False,
            )
            logger.info("data loaders were created")

            assert val_dataset.classes == train_dataset.classes

            logger.info("creating runner")
            runner = Runner(
                model=EmbeddingsModel(
                    num_classes=len(classes_train), backbone=BACKBONE
                ),
                classes=train_dataset.classes,
                lr=1e-3,
                scheduler_T=max_epochs,  # * len(train_dl),
                mapper=mapper,
            )
            logger.info("runner was created")

            logger.info("creating trainer!")
            trainer = pl.Trainer(
                max_epochs=max_epochs,
                gpus=-1,
                logger=pl.loggers.tensorboard.TensorBoardLogger(tb_log_dir_to_use),
                callbacks=[
                    ModelCheckpoint(
                        dirpath=model_dir,
                        save_top_k=1,
                        verbose=True,
                        filename="checkpoint-{epoch:02d}",
                    ),
                    EarlyStopping(
                        patience=10, monitor="Validation/accuracy", mode="max"
                    ),
                ],
            )
            logger.info("trainer was created!")

            # find learning rate
            logger.info("Run learning rate finder")
            lr_finder = trainer.tuner.lr_find(runner, train_dl)

            # Pick point based on plot, or get suggestion
            new_lr = lr_finder.suggestion()

            # update hparams of the model
            runner.hparams.lr = new_lr
            logger.info("Done!")

            logger.info("run training pipeline")
            trainer.fit(runner, train_dl, val_dl)
            logger.info("done!")

    train_dataset_clean = ImageFolder(
        root=str(dataset_folder_to_use / "train"),
        transform=Transforms(segment="test"),
    )

    train_dl_clean = DataLoader(
        train_dataset_clean,
        BATCH_SIZE,
        shuffle=False,
        pin_memory=False,
        num_workers=1,
        drop_last=False,
    )

    accuracy = calculate_accuracy(
        trainer=trainer, train_dl=train_dl_clean, val_dl=val_dl
    )

    with open(str(tb_log_dir_to_use / "accuracy.pickle"), "wb") as handle:
        pickle.dump(accuracy, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # save the model
    runner.model.eval()
    b = next(iter(val_dl))
    traced_model = torch.jit.trace(runner.model, b[0])
    meta = {
        "inference_params": {
            "image_height": SIZE,
            "image_width": SIZE,
        },
    }
    traced_model.save(
        str(model_dir_to_use / "torchscript.pt"),
        _extra_files={f"{k}.txt": str(v) for k, v in meta.items()},
    )


if __name__ == "__main__":
    main()
