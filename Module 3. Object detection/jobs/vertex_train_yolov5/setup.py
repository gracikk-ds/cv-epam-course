from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = [
    "matplotlib >= 3.2.2",
    "numpy >= 1.18.5",
    "opencv-python >= 4.1.2",
    "Pillow >= 7.1.2",
    "PyYAML >= 5.3.1",
    "requests >= 2.23.0",
    "scipy >= 1.4.1",
    "torch >= 1.7.0",
    "torchvision >= 0.8.1",
    "tqdm >= 4.41.0",
    "tensorboard>=2.4.1",
    "pandas>=1.1.4",
    "seaborn>=0.11.0",
    "albumentations>=1.0.3",
    "yolov5 == 6.0.6",
]

setup(
    name="trainer",
    version="0.1",
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description="Yolo v5 detector",
)
