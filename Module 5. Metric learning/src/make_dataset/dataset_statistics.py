import math
import numpy as np
import pandas as pd
from PIL import Image
import seaborn as sns
from pathlib import Path
from matplotlib import pyplot as plt


# Main folder
dataset_folder_to_use = Path("../../dataset")

# pathes to dirs
TRAIN_DIR = dataset_folder_to_use / 'train'
TEST_DIR = dataset_folder_to_use / 'test'

# pathes to images
train_files = sorted(list(TRAIN_DIR.rglob('*.JPG')))
test_files = sorted(list(TEST_DIR.rglob('*.JPG')))

# labels
train_labels = [path.parent.name for path in train_files]
n_classes = len(np.unique(train_labels))

print(f"number of train images: {len(train_files)}")
print(f"number of test images: {len(test_files)}")


def labels_pathes_dict(train_files, train_labels):
    dict_path = {}
    dict_length = {}
    for label in np.unique(train_labels).tolist():
        dict_path[label] = []
        dict_length[label] = {}

    for path, label in zip(train_files, train_labels):
        dict_path[label].append(path)

    for label in dict_path:
        dict_length[label] = len(dict_path[label])
    df = pd.DataFrame.from_dict(dict_length, orient="index", columns=["count"])
    fig = plt.figure(figsize=(12, 5))
    sns.barplot(x="index", y="count", data=df.reset_index())
    plt.title("Distribution of classes")
    plt.show()
    return dict_path


def per_class_visualization(dict_path):
    nrows = math.ceil(len(dict_path)/4)

    fig, axes = plt.subplots(
        nrows=nrows, ncols=4,
        figsize=(20, 6 * nrows)
    )
    axes = axes.flatten()

    for i, label in enumerate(list(dict_path.keys())):
        random_object = int(np.random.uniform(0, len(dict_path[label])))
        im_path, img_label = dict_path[label][random_object], label
        image = np.array(Image.open(im_path))
        axes[i].imshow(image)
        axes[i].set_title(img_label)
        axes[i].grid(False)
    plt.tight_layout()
    plt.show()


dict_path = labels_pathes_dict(train_files, train_labels)
per_class_visualization(dict_path)
