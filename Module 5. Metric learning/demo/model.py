import timm
import torch
import numpy as np
import torch.nn as nn
import albumentations as albu
from albumentations.pytorch import ToTensorV2


def transformation(img):
    transforms = [
        albu.LongestMaxSize(max_size=224, always_apply=True, p=1),
        albu.PadIfNeeded(
            min_height=224,
            min_width=224,
            always_apply=True,
            border_mode=0,
            value=(255, 255, 255),
        ),
        albu.Normalize(),
        ToTensorV2(),
    ]

    transforms = albu.Compose(transforms)

    img_tr = transforms(image=np.array(img))["image"]

    return img_tr


class EmbeddingsModel(nn.Module):
    def __init__(
        self,
        num_classes: int,
        embedding_size: int = 512,
        backbone: str = "resnext101_32x8d",
    ):
        super().__init__()
        self.trunk = timm.create_model(backbone, pretrained=False)
        self.embedding_size = embedding_size
        self.trunk.fc = nn.Linear(
            in_features=self.trunk.fc.in_features,
            out_features=embedding_size,
            bias=False,
        )

        self.classifier = torch.nn.Sequential(
            nn.Linear(embedding_size, num_classes, bias=True),
        )

    def forward(self, inpt):
        # get embeddings
        emb = self.trunk(inpt)

        # get logits
        logits = self.classifier(emb)

        return logits, emb
