from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = [
    "torch==1.11.0",
    "torchvision==0.12.0",
    "torchmetrics==0.7.3",
    "pytorch_lightning==1.5.10",
    "pytorch_metric_learning==1.2.1",
    "timm",
    "tqdm",
    "click",
    "future==0.18.2",
    "seaborn",
    "tensorboard",
    "record-keeper",
    "albumentations==1.1.0",
]

setup(
    name="trainer",
    version="0.1",
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description="Pytorch metric learning training on custom docker container",
)
