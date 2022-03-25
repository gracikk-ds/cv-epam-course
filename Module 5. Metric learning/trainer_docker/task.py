import click
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_metric_learning.samplers import MPerClassSampler

from .data import Transforms
from .metric_learning import EmbeddingsModel, Runner, BATCH_SIZE


SIZE = 224
BACKBONE = "resnext101_32x8d"


@click.command()
@click.option("--dataset_folder", help="GCS path to dataset.")
@click.option("--tb_log_dir", help="GCS path to tb_dir.")
@click.option("--model_dir", help="GCS path to model_dir.")
@click.option("--max_epochs", default=1)
def main(
        dataset_folder: str,
        tb_log_dir: str,
        model_dir: str,
        max_epochs: int,
):
    tb_log_dir_to_use = Path(tb_log_dir)
    model_dir_to_use = Path(model_dir)
    dataset_folder_to_use = Path(dataset_folder)

    # setting up logger
    logging.basicConfig(
        filename=str(tb_log_dir_to_use / "logs.log"),
        filemode="a",
        format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        level=logging.DEBUG,
    )

    classes_train = set(
        [p.name for p in (dataset_folder_to_use / "train").glob("*")]
    )
    classes_val = set(
        [p.name for p in (dataset_folder_to_use / "val").glob("*")]
    )

    print(
        f"Number of classes in train {len(classes_train)}",
        f"Number of classes in val {len(classes_val)}",
        f"Number of classes in train & val {len(classes_train & classes_val)}",
        f"Number of classes in train - val {len(classes_train - classes_val)}",
    )

    train_dataset = ImageFolder(
        root=str(dataset_folder_to_use / "train"),
        transform=Transforms(),
    )

    val_dataset = ImageFolder(
        root=str(dataset_folder_to_use / "test"),
        transform=Transforms(segment="val"),
    )

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
        num_workers=8,
        drop_last=True,
    )

    val_dl = DataLoader(
        val_dataset,
        BATCH_SIZE,
        pin_memory=False,
        shuffle=False,
        num_workers=4,
        drop_last=False,
    )

    assert val_dataset.classes == train_dataset.classes

    runner = Runner(
        model=EmbeddingsModel(
            num_classes=len(classes_train),
            backbone=BACKBONE
        ),
        classes=train_dataset.classes,
        lr=1e-3,
        scheduler_T=max_epochs * len(train_dl),
    )

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        gpus=-1,
        logger=pl.loggers.TensorBoardLogger(tb_log_dir_to_use),
        callbacks=[
            ModelCheckpoint(
                dirpath=model_dir,
                save_top_k=-1,
                verbose=True,
                filename="checkpoint-{epoch:02d}",
            ),
        ],
    )

    trainer.fit(runner, train_dl, val_dl)

    # save the model
    runner.model.eval()
    b = next(iter(val_dl))
    traced_model = torch.jit.trace(runner.model, b[0])
    meta = {
        "class_names": runner.mapped_classes,
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
