import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import albumentations as albu
from torch.utils.data import Dataset
from albumentations.pytorch.transforms import ToTensorV2
from typing import Optional, Callable


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callbale): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose

    """

    _transform = [
        albu.Lambda(image=preprocessing_fn),
        ToTensorV2(transpose_mask=True),
    ]
    return albu.Compose(_transform)


class Transforms:
    def __init__(self, segment="train"):
        if segment == "train":
            transforms = [
                albu.LongestMaxSize(max_size=230, always_apply=True, p=1),
                albu.OneOf(
                    [
                        albu.ColorJitter(hue=0.015, saturation=0.3),
                        albu.RandomBrightnessContrast(
                            brightness_limit=0.05, contrast_limit=0.05
                        ),
                    ],
                    p=0.3,
                ),
                albu.ShiftScaleRotate(
                    border_mode=1, rotate_limit=25, scale_limit=0.3, p=0.3
                ),
                albu.PadIfNeeded(
                    min_height=230,
                    min_width=230,
                    always_apply=True,
                    border_mode=0,
                    value=(255, 255, 255),
                ),
                albu.OneOf(
                    [
                        albu.RandomCrop(width=224, height=224),
                        albu.Resize(width=224, height=224),
                    ],
                    p=1,
                ),
                albu.HorizontalFlip(p=0.5),
                albu.VerticalFlip(p=0.5),
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

        self.transforms = albu.Compose(transforms)

    def __call__(self, img, *args, **kwargs):  # msk,
        return self.transforms(image=np.array(img))  # , mask=np.array(msk)


class SegmentationDataSet(Dataset):
    def __init__(
        self,
        images: list,
        masks: list,
        df_count,
        transform: Optional[Callable] = None,
        preprocessing: Optional[Callable] = None,
    ):
        super().__init__()

        # determine path lists
        self.images = sorted(images)
        self.masks = sorted(masks)
        self.df_count = df_count

        # transformation
        self.transform = transform

        # preprocessing
        self.preprocessing = preprocessing

        # getting len info
        self.len_ = len(self.images)

    def __getitem__(self, index):
        # read data
        count = self.df_count.loc[
            self.df_count["filename"] == Path(str(self.images[index])).name, "count"
        ].values
        image = Image.open(str(self.images[index]))
        image = np.array(image)
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            if image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        mask = Image.open(str(self.masks[index])).convert("L")
        mask = np.array(mask)
        # print(str(self.images[index]), f": shape {image.shape}, {mask.shape} \n")

        # Preprocessing
        if self.transform is not None:
            try:
                sample = self.transform(img=image, msk=mask)
                image, mask = sample["image"], sample["mask"]
            except:
                print(str(self.images[index]))
                print(image.shape)
                print(mask.shape)
                raise AssertionError()

        # apply preprocessing
        if self.preprocessing:
            try:
                mask = mask[..., np.newaxis]
                sample = self.preprocessing(image=image, mask=mask)
                image, mask = sample["image"], sample["mask"]
            except Exception:
                print(str(self.images[index]))
                print(image.shape)
                print(str(self.masks[index]))
                print(mask.shape)

        return image, mask, count

    def __len__(self):
        return self.len_


class SegmentationTestDataSet(Dataset):
    def __init__(
        self,
        images: list,
        transform: Optional[Callable] = None,
        preprocessing: Optional[Callable] = None,
    ):
        super().__init__()

        # determine path lists
        self.images = sorted(images)

        # transformation
        self.transform = transform

        # preprocessing
        self.preprocessing = preprocessing

        # getting len info
        self.len_ = len(self.images)

    def __getitem__(self, index):
        # read data
        image = Image.open(str(self.images[index]))
        image = np.array(image)
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            if image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

        # Preprocessing
        if self.transform is not None:
            sample = self.transform(img=image)
            image = sample["image"]

        # apply preprocessing
        if self.preprocessing is not None:
            sample = self.preprocessing(image=image)
            image = sample["image"]

        return image

    def __len__(self):
        return self.len_
