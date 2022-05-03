import math
import click
import numpy as np
import pandas as pd
from PIL import Image
import seaborn as sns
from pathlib import Path
from typing import List
from matplotlib import pyplot as plt


def labels_paths_dict(dataset_folder: str) -> List[dict]:
    """
    Creates two dicts. The first one with paths to images.
    The second one with number of images per class
    :param dataset_folder: path to dataset_full folder
    :return: dicts
    """
    dataset_folder_to_use = Path(dataset_folder)

    # pathes to dirs
    TRAIN_DIR = dataset_folder_to_use / "train"
    TEST_DIR = dataset_folder_to_use / "test"

    # pathes to images
    train_files: list = sorted(list(TRAIN_DIR.rglob("*.JPG")))
    test_files: list = sorted(list(TEST_DIR.rglob("*.JPG")))

    # labels
    train_labels: list = [path.parent.name for path in train_files]
    n_classes = len(np.unique(train_labels))

    print(f"number of train images: {len(train_files)}")
    print(f"number of test images: {len(test_files)}")
    print(f"number of classes: {n_classes}")

    dict_path: dict = {}
    dict_length: dict = {}
    for label in list(np.unique(train_labels)):
        dict_path[label] = []
        dict_length[label] = {}

    for path, label in zip(train_files, train_labels):
        dict_path[label].append(path)

    for label in dict_path:
        dict_length[label] = len(dict_path[label])

    return [dict_path, dict_length]


@click.command()
@click.option(
    "--dataset_folder",
    type=str,
    help="path to data.",
    default="../data/interim/dataset_full",
)
def per_class_visualization(dataset_folder: str) -> None:
    """
    Save plt figure with visualization of images per class
    :param dataset_folder: path to dataset_full folder
    :return: None
    """

    dict_path, dict_length = labels_paths_dict(dataset_folder)

    # fig 1: hist of class distribution
    df = pd.DataFrame.from_dict(dict_length, orient="index", columns=["count"])
    print(df.head())

    plt.figure(figsize=(12, 5))
    sns.barplot(x="index", y="count", data=df.reset_index())
    plt.title("Distribution of classes")
    # plt.show()
    plt.savefig("../reports/figures/class_distribution.png")

    nrows = math.ceil(len(dict_path) / 4)

    # fig 2: per class visualization
    fig, axes = plt.subplots(nrows=nrows, ncols=4, figsize=(20, 6 * nrows))
    axes = axes.flatten()

    for i, label in enumerate(list(dict_path.keys())):
        random_object = int(np.random.uniform(0, len(dict_path[label])))
        im_path, img_label = dict_path[label][random_object], label
        image = np.array(Image.open(im_path))
        axes[i].imshow(image)
        axes[i].set_title(img_label)
        axes[i].grid(False)
    plt.tight_layout()
    # plt.show()
    plt.savefig("../reports/figures/per_class_visualization.png")


if __name__ == "__main__":
    per_class_visualization()
