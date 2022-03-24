import os
import sys
import click
import tempfile
import traceback
import contextlib
import subprocess
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_metric_learning.samplers import MPerClassSampler

from .data import Transforms, FilteredImageFolder
from .metric_learning import EmbeddingsModel, Runner, BATCH_SIZE


SIZE = 224

SELECTED_BRANDS = (
    "Sochnyj",
    "Rich",
    "Schaste",
    "Agusha",
    "Sadochok",
)
BACKBONE = "resnext101_32x8d"


def target_pretransform_fn(target: str, selected_brands: Tuple[str] = SELECTED_BRANDS):
    if any([target.lower().startswith(prefix.lower()) for prefix in selected_brands]):
        return target
    return "other"


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

    path = tb_log_dir / "stdout.txt"
    path_err = tb_log_dir / "stderr.txt"

    with open(path, "w") as f, open(path_err, "w") as e:
        with contextlib.redirect_stdout(f), contextlib.redirect_stderr(e):

            classes_train = set(
                [p.name for p in (dataset_folder / "train").glob("*")]
            )
            classes_val = set(
                [p.name for p in (dataset_folder / "val").glob("*")]
            )

            print(
                f"Number of classes in train {len(classes_train)}",
                f"Number of classes in val {len(classes_val)}",
                f"Number of classes in train & val {len(classes_train & classes_val)}",
                f"Number of classes in train - val {len(classes_train - classes_val)}",
            )

            classes_superset = sorted(classes_train)

            train_dataset = FilteredImageFolder(
                str(dataset_folder / "train"),
                transform=Transforms(),
                classes_superset=classes_superset,
            )
            val_dataset = FilteredImageFolder(
                str(dataset_folder / "val"),
                classes_superset=classes_superset,
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

            class_mapping = {
                clz: target_pretransform_fn(clz)
                if clz in val_dataset.existing_classes
                else "other"
                for clz in val_dataset.classes
            }

            runner = Runner(
                model=EmbeddingsModel(
                    num_classes=len(set(class_mapping.values())), backbone=BACKBONE
                ),
                classes=train_dataset.classes,
                class_mapping=class_mapping,
                lr=1e-4,
                scheduler_T=max_epochs * len(train_dl),
            )

            trainer = pl.Trainer(
                max_epochs=max_epochs,
                gpus=-1,
                # gradient_clip_val=1.0,
                logger=pl.loggers.TensorBoardLogger(tb_log_dir),
                callbacks=[
                    ModelCheckpoint(
                        dirpath=model_dir,
                        save_top_k=-1,
                        verbose=True,
                        filename="checkpoint-{epoch:02d}",
                    ),
                ],
            )

            try:
                trainer.fit(runner, train_dl, val_dl)

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
                    str(model_dir / "torchscript.pt"),
                    _extra_files={f"{k}.txt": str(v) for k, v in meta.items()},
                )

            except Exception as ex:
                print(ex)
                print(traceback.format_exc(), flush=True)
                print(traceback.print_stack(), flush=True)


if __name__ == "__main__":
    main()
