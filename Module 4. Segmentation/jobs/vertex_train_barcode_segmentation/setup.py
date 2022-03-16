from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = [
    "numpy",
    "torch",
    "torchvision==0.8.0",
    "pytorch_lightning",
    "segmentation_models_pytorch",
    "torchmetrics==0.7.1",
    "torchserve",
    "torch-model-archiver",
    "tensorboard",
    "albumentations",
    "faiss-gpu >= 1.6.3",
    "tqdm",
    "cloudml-hypertune",
    "click",
    "record-keeper",
    "future",
    "opencv-python",
    "seaborn",
]

setup(
    name="trainer",
    version="0.1",
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description="Pytorch barcodes segmentation model on Vertex AI",
)
