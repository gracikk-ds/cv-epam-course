from jobs.vertex_train_bars_segmentation.trainer.data import (
    SegmentationTestDataSet,
    Transforms,
    get_preprocessing,
)
from jobs.vertex_train_bars_segmentation.trainer.model import BarsSegmentation, Runner
from pathlib import Path
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import cv2
from PIL import Image


def prediction(images_inference):
    model, preprocessing_fn = BarsSegmentation(model="FPN")
    chkpt = torch.load(
        "../checkpoint-FPN-epoch=63.ckpt", map_location=torch.device("cpu")
    )

    prefix = "model."
    n_clip = len(prefix)
    adapted_chkpt = {
        k[n_clip:]: v for k, v in chkpt["state_dict"].items() if k.startswith(prefix)
    }
    model.load_state_dict(adapted_chkpt)

    runner = Runner(model=model, classes=["bars"])

    preprocessing = get_preprocessing(preprocessing_fn)

    inference_dataset = SegmentationTestDataSet(
        images=images_inference,
        transform=Transforms(segment="test"),
        preprocessing=preprocessing,
    )

    test_dl = DataLoader(
        inference_dataset,
        1,
        pin_memory=False,
        shuffle=False,
        num_workers=4,
        drop_last=False,
    )

    trainer = pl.Trainer()
    results = trainer.predict(model=runner, dataloaders=test_dl)

    for tensor, path in zip(results, images_inference):
        img = tensor.squeeze().numpy() * 255
        img[img < 30] = 0
        img[img > 0] = 255
        Image.fromarray(img.astype("uint8")).save(
            "../test_data/predictions_croped/" + path.name
        )


if __name__ == "__main__":
    images_inference = sorted(
        [file for file in Path("../test_data/Images_crops/").glob("*")]
    )
    prediction(images_inference)
