import numpy as np
import albumentations as albu
from albumentations.pytorch import ToTensorV2
from albumentations.augmentations import CoarseDropout


class Transforms:
    def __init__(self, segment="train"):
        if segment == "train":
            transforms = [
                albu.LongestMaxSize(max_size=224 + 5, always_apply=True, p=1),
                albu.RandomBrightnessContrast(p=0.3),
                albu.ColorJitter(hue=0.01, saturation=0.02, p=0.3),
                # geometric transformations
                albu.GridDistortion(distort_limit=0.6, p=0.3),
                albu.ShiftScaleRotate(border_mode=1, rotate_limit=3, p=0.3),
                albu.PadIfNeeded(
                    min_height=224 + 5,
                    min_width=224 + 5,
                    always_apply=True,
                    border_mode=0,
                    value=(255, 255, 255),
                ),
                albu.RandomCrop(width=224, height=224),
                albu.HorizontalFlip(p=0.5),
            ]
        else:
            transforms = [
                albu.LongestMaxSize(max_size=224, always_apply=True, p=1),
                albu.PadIfNeeded(
                    min_height=224,
                    min_width=224,
                    always_apply=True,
                    border_mode=0,
                    value=(255, 255, 255),
                ),
            ]
        transforms.extend(
            [
                albu.Normalize(),
                ToTensorV2(),
            ]
        )

        self.transforms = albu.Compose(transforms)

    def __call__(self, img, *args, **kwargs):
        return self.transforms(image=np.array(img))["image"]
