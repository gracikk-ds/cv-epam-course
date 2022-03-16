from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = [
    "mmdet==2.21.0",
    "mmcv-full==1.4.5",
    "tqdm",
    "cloudml-hypertune",
    "click==7.1.2",
    "future==0.18.2",
    "tensorboard",
    "albumentations"
]

setup(
    name="trainer",
    version="0.1",
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description="MMDet training on Vertex AI",
    author="epam",
    author_email="epam@epam.com",
    url="https://www.epam.com/"
)
