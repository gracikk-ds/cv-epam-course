from pathlib import Path
from typing import Optional, Callable, List

import albumentations as albu
import numpy as np
from albumentations.pytorch import ToTensorV2
from torchvision.datasets import ImageFolder


class Transforms:
    def __init__(self, segment="train"):
        if segment == "train":
            transforms = [
                albu.LongestMaxSize(max_size=224 + 5, always_apply=True, p=1),
                albu.GridDistortion(distort_limit=0.6, p=0.8),
                albu.RandomRotate90(p=0.1),
                albu.ColorJitter(hue=0.01, saturation=0.02, p=1),
                albu.ShiftScaleRotate(border_mode=1, rotate_limit=3, p=0.3),
                albu.OneOf(
                    [
                        albu.Blur(blur_limit=3),
                        albu.Downscale(scale_min=0.7, scale_max=0.9),
                    ],
                    p=0.3,
                ),
                albu.PadIfNeeded(
                    min_height=224 + 5,
                    min_width=224 + 5,
                    always_apply=True,
                    border_mode=0,
                    value=(255, 255, 255),
                ),
                albu.RandomCrop(width=224, height=224),
                albu.HorizontalFlip(p=0.5),
                albu.RandomBrightnessContrast(p=0.2),
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


class FilteredImageFolder(ImageFolder):
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
        classes_superset: Optional[List[str]] = None,
    ):
        super().__init__(
            root,
            transform=transform,
            target_transform=target_transform,
            is_valid_file=is_valid_file,
        )

        self.existing_classes = self.classes
        if classes_superset is None:
            self.classes = sorted(self.classes)
        else:
            self.classes = classes_superset

        self.class_to_idx = {clz: i for i, clz in enumerate(self.classes)}
        self.samples = [
            (s[0], Path(s[0]).parent.name)
            for s in self.samples
            if Path(s[0]).parent.name in self.classes
        ]
        self.imgs = self.samples
        self.targets = [s[1] for s in self.samples]
        self.samples = [(s[0], self.class_to_idx[s[1]]) for s in self.samples]
