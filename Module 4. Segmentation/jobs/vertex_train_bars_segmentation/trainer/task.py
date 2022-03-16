import os
import click
import traceback
import contextlib
import pandas as pd
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from .data import Transforms, SegmentationDataSet, get_preprocessing
from .model import BarsSegmentation, Runner, BATCH_SIZE

GS_PREFIX = "gs://"
GCSFUSE_PREFIX = "/gcs/"
SIZE = 224


def to_fuse(url: str):
    if url is not None and url.startswith(GS_PREFIX):
        return url.replace(GS_PREFIX, GCSFUSE_PREFIX)
    return url


@click.command()
@click.option(
    "--dataset_folder",
    default="gs://lidless-eye/Barcodes/bars_data",  # my_backet/Barcodes/dataset
    help="GCS path to dataset.",
)
@click.option("--max_epochs", default=75)
@click.option("--model_type", default="FPN")
def main(dataset_folder: str, max_epochs: int, model_type: str):

    model_dir = os.getenv("AIP_MODEL_DIR", "model")
    tb_log_dir = os.getenv("AIP_TENSORBOARD_LOG_DIR", "tensorboard")

    model_dir_to_use = Path(to_fuse(model_dir))
    tb_log_dir_to_use = Path(to_fuse(tb_log_dir))
    dataset_folder_to_use = Path(to_fuse(dataset_folder))

    model_dir_to_use.mkdir(exist_ok=True, parents=True)
    tb_log_dir_to_use.mkdir(exist_ok=True, parents=True)
    path = tb_log_dir_to_use / "stdout.txt"
    path_err = tb_log_dir_to_use / "stderr.txt"

    print(f"model_dir_to_use: {model_dir_to_use}")
    print(f"tb_log_dir_to_use: {tb_log_dir_to_use}")
    print(f"path: {path}")
    print(f"path_err: {path_err}")

    with open(path, "w") as f, open(path_err, "w") as e:
        with contextlib.redirect_stdout(f), contextlib.redirect_stderr(e):

            print("Defining the model...")
            model, preprocessing_fn = BarsSegmentation(model=model_type)
            preprocessing = get_preprocessing(preprocessing_fn)
            print("Done!\n", "---" * 10, "\n")

            df_count_train = pd.read_csv(dataset_folder_to_use / "blobs_df_train.csv")
            df_count_val = pd.read_csv(dataset_folder_to_use / "blobs_df_val.csv")

            # gather paths to images
            images_train = sorted(
                [
                    file
                    for file in (dataset_folder_to_use / "train" / "Images").glob("*")
                ]
            )
            images_val = sorted(
                [file for file in (dataset_folder_to_use / "val" / "Images").glob("*")]
            )
            # images_test = sorted([
            #     file for file in (dataset_folder_to_use / "test" / "Images").glob("*")
            # ])
            print("images_train: ", images_train[:1])

            masks_train = sorted(
                [file for file in (dataset_folder_to_use / "train" / "Masks").glob("*")]
            )
            masks_val = sorted(
                [file for file in (dataset_folder_to_use / "val" / "Masks").glob("*")]
            )
            # masks_test = sorted([
            #     file for file in (dataset_folder_to_use / "test" / "Mask").glob("*")
            # ])
            print("masks_train: ", masks_train[:1])

            check_trian = [x.stem for x in masks_train]
            check_val = [x.stem for x in masks_val]
            # check_test = [x.stem for x in masks_test]

            images_train = [x for x in images_train if x.stem in check_trian]
            images_val = [x for x in images_val if x.stem in check_val]
            # images_test = [x for x in images_test if x.stem in check_test]

            for img, msk in zip(images_train, masks_train):
                if Path(img).stem != Path(msk).stem:
                    print("Error!")
                    raise AssertionError()

            print("Creaing Dataset...")
            train_dataset = SegmentationDataSet(
                images=images_train,
                masks=masks_train,
                transform=Transforms(),
                preprocessing=preprocessing,
                df_count=df_count_train,
            )
            val_dataset = SegmentationDataSet(
                images=images_val,
                masks=masks_val,
                transform=Transforms(segment="val"),
                preprocessing=preprocessing,
                df_count=df_count_val,
            )
            # test_dataset = SegmentationDataSet(
            #     images=images_test,
            #     masks=masks_test,
            #     transform=Transforms(segment="val"),
            #     preprocessing=preprocessing,
            # )
            print("Datasets were created\n", "---" * 10, "\n")

            print("Creaing dataloaders...")
            train_dl = DataLoader(
                train_dataset,
                BATCH_SIZE,
                pin_memory=False,
                num_workers=4,
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

            # test_dl = DataLoader(
            #     test_dataset,
            #     BATCH_SIZE,
            #     pin_memory=False,
            #     shuffle=False,
            #     num_workers=4,
            #     drop_last=False,
            # )
            print("Dataloaders were created\n", "---" * 10, "\n")

            print("Defining Runner...")
            runner = Runner(
                model=model,
                classes=["bar"],
                lr=1e-2,
                scheduler_T=max_epochs,
            )
            print("Done!\n", "---" * 10, "\n")

            print("Defining trainer...")
            trainer = pl.Trainer(
                max_epochs=max_epochs,
                gpus=-1,
                logger=pl.loggers.TensorBoardLogger(tb_log_dir_to_use, name=model_type),
                callbacks=[
                    ModelCheckpoint(
                        monitor="Validation/IOUScore",
                        dirpath=model_dir_to_use,
                        save_top_k=2,
                        verbose=True,
                        filename="checkpoint-" + model_type + "-{epoch:02d}",
                        mode="max",
                    ),
                    EarlyStopping(
                        patience=10, monitor="Validation/IOUScore", mode="max"
                    ),
                ],
            )
            print("Done!\n", "---" * 10, "\n")

            # # find learning rate
            # print("Run learning rate finder")
            # lr_finder = trainer.tuner.lr_find(runner, train_dl)
            #
            # # Pick point based on plot, or get suggestion
            # new_lr = lr_finder.suggestion()
            #
            # # update hparams of the model
            # runner.hparams.lr = new_lr
            # print("Done!\n")

            try:
                print("fitting the model...\n")
                trainer.fit(runner, train_dl, val_dl)
                print("\nDone!\n", "---" * 10, "\n")

                # print("Testing the model")
                # runner.model.eval()
                # score = trainer.test(runner, dataloaders=test_dl)
                # print(score)
                # print("\nDone!\n", "---" * 10, "\n")

                print("Creating traced model...")
                runner.model.encoder.set_swish(memory_efficient=False)
                b = next(iter(val_dl))
                traced_model = torch.jit.trace(runner.model, b[0].float())
                meta = {
                    "class_names": ["barcode"],
                    "inference_params": {
                        "image_height": SIZE,
                        "image_width": SIZE,
                    },
                }
                print("Done!\n", "---" * 10, "\n")

                print("Saving the model..")
                traced_model.save(
                    str(model_dir_to_use / "torchscript.pt"),
                    _extra_files={f"{k}.txt": str(v) for k, v in meta.items()},
                )
                print("Done\n", "---" * 10, "\n")

            except Exception as ex:
                print(ex)
                print(traceback.format_exc(), flush=True)
                print(traceback.print_stack(), flush=True)


if __name__ == "__main__":
    main()
