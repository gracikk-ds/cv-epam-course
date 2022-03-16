import os
import json
import click
import traceback
import contextlib
from pathlib import Path
from typing import Optional, List

import mmcv
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmdet.datasets.builder import build_dataset, build_dataloader
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector
from mmdet.apis import train_detector, single_gpu_test


GS_PREFIX = "gs://"
GCSFUSE_PREFIX = "/gcs/"


def to_fuse(url: str):
    if url is not None and url.startswith(GS_PREFIX):
        return url.replace(GS_PREFIX, GCSFUSE_PREFIX)
    return url


@click.command()
@click.option("--dataset_folder", help="GCS path to dataset.", multiple=True)
@click.option("--max_epochs", default=5)
@click.option(
    "--train_ann_file", default="annotations/annotations_train.json", multiple=True
)
@click.option(
    "--val_ann_file", default="annotations/annotations_val.json", multiple=True
)
@click.option(
    "--test_ann_file", default="annotations/annotations_test.json", multiple=True
)
@click.option("--train_img_prefix", default="train images prefix")
@click.option("--val_img_prefix", default="val images prefix")
@click.option("--test_img_prefix", default="test images prefix")
@click.option("--load_from", required=False)
@click.option("--gsfuse", type=bool, default=True)
def main(
    dataset_folder: List[str],
    max_epochs: int,
    train_ann_file: List[str],
    val_ann_file: List[str],
    test_ann_file: List[str],
    train_img_prefix: str,
    val_img_prefix: str,
    test_img_prefix: str,
    load_from: Optional[str],
    gsfuse: bool = False,
):
    model_dir = os.getenv("AIP_MODEL_DIR", "model")
    tb_log_dir = os.getenv("AIP_TENSORBOARD_LOG_DIR", "logs")

    dataset_folder_to_use = [Path(to_fuse(d)) for d in dataset_folder]
    model_dir_to_use = Path(to_fuse(model_dir))
    tb_log_dir_to_use = Path(to_fuse(tb_log_dir))
    load_from = to_fuse(load_from)

    print("Stdout logging check")
    model_dir_to_use.mkdir(exist_ok=True, parents=True)
    tb_log_dir_to_use.mkdir(exist_ok=True, parents=True)
    path = tb_log_dir_to_use / "stdout.txt"
    path_err = tb_log_dir_to_use / "stderr.txt"

    with open(path, "w") as f, open(path_err, "w") as e:
        with contextlib.redirect_stdout(f), contextlib.redirect_stderr(e):
            # Stdout is not shown in logs as is used to  be in AI Platform, so this is a dirty way to get logs.
            # When "Stdout logging check" appear in logs (hope it will), remove the redirect.
            print("HELLO WORLD VERTEX TRAIN!")
            print(f"gsfuse={gsfuse}")
            print(f"\nmodel_dir_to_use: {model_dir_to_use}")
            print(f"tb_log_dir_to_use: {tb_log_dir_to_use}\n")

            # TODO: multi-folder case
            path_to_anno_file = "/".join(
                [str(dataset_folder_to_use[0]), train_ann_file[0]]
            )
            print(f"path_to_anno_file: {path_to_anno_file}")
            with open(path_to_anno_file) as json_file:
                annotations_train = json.load(json_file)
            classes = [element["name"] for element in annotations_train["categories"]]
            print(f"number of classes {len(classes)}")

            cfg = Config.fromfile(str(Path(__file__).parent.resolve() / "yolox_small_8x8_300e_coco.py"))
            cfg.work_dir = str(model_dir_to_use)
            cfg.classes = classes
            cfg.seed = 42

            # train data config
            cfg.train_dataset.dataset.ann_file = [
                str(folder / file)
                for file, folder in zip(train_ann_file, dataset_folder_to_use)
            ]
            cfg.train_dataset.dataset.img_prefix = [
                str(folder / train_img_prefix) for folder in dataset_folder_to_use
            ]

            # val data config
            cfg.data.val.ann_file = [
                str(folder / file)
                for file, folder in zip(val_ann_file, dataset_folder_to_use)
            ]
            cfg.data.val.img_prefix = [
                str(folder / val_img_prefix) for folder in dataset_folder_to_use
            ]

            # test data config
            cfg.data.test.ann_file = [
                str(folder / file)
                for file, folder in zip(test_ann_file, dataset_folder_to_use)
            ]
            cfg.data.test.img_prefix = [
                str(folder / test_img_prefix) for folder in dataset_folder_to_use
            ]
            cfg.load_from = load_from
            cfg.max_epochs = max_epochs
            cfg.num_last_epochs = 5
            cfg.gpu_ids = [0]  # TODO

            cfg.log_config = dict(
                interval=5,
                hooks=[
                    dict(type="TextLoggerHook", out_dir=str(tb_log_dir_to_use)),
                    dict(type="TensorboardLoggerHook", log_dir=str(tb_log_dir_to_use)),
                ],
            )
            print("log config is ready to use")

            print(f"=======\nConfig:\n{cfg.pretty_text}")
            cfg.dump(str(model_dir_to_use / "yolox_small_8x8_300e_coco.py"))
            print("config was saved to model_dir_to_use")

            try:
                print("build_dataset", flush=True)
                datasets = [build_dataset(cfg.data.train)]
                print("\n Train Dataset: \n", datasets[0])

                print("build_detector", flush=True)
                model = build_detector(cfg.model)

                print("train_detector", flush=True)
                train_detector(model, datasets, cfg, distributed=False, validate=True)

                # Test
                print("Build test dataset", flush=True)
                cfg.data.test.test_mode = True
                test_dataset = build_dataset(cfg.data.test)
                print("\n Test Dataset: \n", test_dataset)

                data_loader = build_dataloader(
                    test_dataset,
                    samples_per_gpu=1,
                    workers_per_gpu=cfg.data.workers_per_gpu,
                    shuffle=False,
                )
                cfg.model.train_cfg = None
                model = build_detector(cfg.model)
                load_checkpoint(
                    model, str(Path(cfg.work_dir) / "latest.pth"), map_location="cpu"
                )

                print("Run evaluation")
                model = MMDataParallel(model, device_ids=[0])
                outputs = single_gpu_test(model, data_loader)
                metric = test_dataset.evaluate(
                    outputs,
                    metric=["bbox", "proposal"],
                    iou_thrs=[0.50, 0.75],
                    proposal_nums=[300, 1000, 3000],
                    metric_items=[
                        "mAP",
                        "mAP_50",
                        "mAP_75",
                        "mAP_s",
                        "mAP_m",
                        "mAP_l",
                        "AR@300",
                        "AR@1000",
                        "AR_s@1000",
                        "AR_m@1000",
                        "AR_l@1000",
                    ],
                    classwise=True,
                    jsonfile_prefix=str(model_dir_to_use / "test"),
                )
                metric_dict = dict(metric=metric)
                mmcv.dump(metric_dict, str(model_dir_to_use / "test_metrics.json"))

            except Exception as ex:
                print(ex)
                print(traceback.format_exc(), flush=True)
                print(traceback.print_stack(), flush=True)


if __name__ == "__main__":
    main()
