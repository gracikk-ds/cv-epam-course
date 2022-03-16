import os
import click
import traceback
import contextlib
from pathlib import Path

os.environ["LOCAL_RANK"] = "-1"
os.environ["RANK"] = "-1"
os.environ["WORLD_SIZE"] = "1"

from yolov5 import train, val, detect
from .cfgs import dataset_yaml_file, hyp_yaml_file

GS_PREFIX = "gs://"
GCSFUSE_PREFIX = "/gcs/"

OPTMIZER = "Adam"
MODEL = "yolov5s"
SIZE = 544
NAME = f"{MODEL}-dim{SIZE}"
BATCH = -1


def to_fuse(url: str):
    if url is not None and url.startswith(GS_PREFIX):
        return url.replace(GS_PREFIX, GCSFUSE_PREFIX)
    return url


@click.command()
@click.option(
    "--dataset_folder",
    default="gs://my_bucket/Barcodes/dataset_yolo",
    help="GCS path to dataset.",
)
@click.option("--max_epochs", default=100)
@click.option("--model", default="yolov5s")
def main(dataset_folder: str, max_epochs: int, model: str):

    model_dir = os.getenv("AIP_MODEL_DIR", "model")
    tb_log_dir = os.getenv("AIP_TENSORBOARD_LOG_DIR", "tensorboard")

    model_dir_to_use = Path(to_fuse(model_dir))
    tb_log_dir_to_use = Path(to_fuse(tb_log_dir))
    dataset_folder_to_use = Path(to_fuse(dataset_folder))

    model_dir_to_use.mkdir(exist_ok=True, parents=True)
    tb_log_dir_to_use.mkdir(exist_ok=True, parents=True)
    path = tb_log_dir_to_use / "stdout.txt"
    path_err = tb_log_dir_to_use / "stderr.txt"

    path_to_data_yaml = dataset_yaml_file(
        root_dir=str(model_dir_to_use),
        path_to_train_images=str(dataset_folder_to_use / "images" / "train"),
        path_to_val_images=str(dataset_folder_to_use / "images" / "val"),
        path_to_test_images=str(dataset_folder_to_use / "images" / "test"),
    )

    path_to_hyper_yaml = hyp_yaml_file(root_dir=str(model_dir_to_use))

    with open(path, "w") as f, open(path_err, "w") as e:
        with contextlib.redirect_stdout(f), contextlib.redirect_stderr(e):

            print(f"path to model dir: {model_dir_to_use}")
            print(f"path to tb log dir: {tb_log_dir_to_use}")
            print(f"path to dataset dir: {dataset_folder_to_use}")
            print(f"path to stdout.txt file: {path}")
            print(f"path to stderr.txt file: {path_err}")
            print(f"path to hyper.yaml file: {path_to_hyper_yaml}")
            print(f"path to data.yaml file: {path_to_data_yaml}")

            try:
                print("\n", "*" * 50, "\n Start train.run process:")
                train.run(
                    weights=f"./{model}.pt",  # path to initial weights,
                    data=path_to_data_yaml,  # dataset.yaml path
                    hyp=path_to_hyper_yaml,  # hyperparameters path
                    epochs=max_epochs,  # number of epochs
                    batch=BATCH,  # total batch size for all GPUs, -1 for autobatch
                    imgsz=SIZE,  # train, val image size (pixels)
                    cache=True,  # cache images in "ram" or "disk"
                    multi_scale=True,  # vary img-size +/- 50%
                    optimizer=OPTMIZER,  # optimizer
                    project=str(tb_log_dir_to_use),  # save to project/name
                    name=NAME,  # save to project/name
                    exist_ok=True,  # existing project/name ok, do not increment
                    label_smoothing=0,  # Label smoothing epsilon
                    patience=25,  # EarlyStopping patience (epochs without improvement)
                    save_period=20  # Save checkpoint every x epochs
                    # evolve=300 - for hyp optimization
                )

                print("train.run process has finished")

                # evaluate the model
                print("\n", "*" * 50, "\n Start val.run process:")
                val.run(
                    imgsz=SIZE,
                    data=path_to_data_yaml,
                    weights=f"{str(tb_log_dir_to_use)}/{NAME}/weights/best.pt",
                    task="test",
                    project=str(tb_log_dir_to_use),  # save to project/name
                    name=NAME + "/val",
                    exist_ok=True,
                )

                print("val.run process has finished")

                # run detection for visualization
                print("\n", "*" * 50, "\n Start detect.run process:")
                detect.run(
                    imgsz=SIZE,
                    source=str(dataset_folder_to_use) + "/images/test",
                    weights=f"{str(tb_log_dir_to_use)}/{NAME}/weights/best.pt",
                    project=str(tb_log_dir_to_use),  # save to project/name
                    name=NAME + "/results_visualization",
                    exist_ok=True,
                )

                print("detect.run process has finished")
            except Exception as ex:
                print(ex)
                print(traceback.format_exc(), flush=True)
                print(traceback.print_stack(), flush=True)


if __name__ == "__main__":
    main()
