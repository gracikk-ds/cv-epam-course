import os
import shutil
from tqdm import tqdm
from pathlib import Path

test_paths = [x for x in Path("../../data/interim/dataset_part/test").rglob("*.JPG")]
print(test_paths)
train_paths = [x for x in Path("../../data/interim/dataset_part/train").rglob("*.JPG")]


def make_dataset(paths: list, output_folder: str):
    output_folder = Path(output_folder)
    folders = []
    for path in tqdm(paths, total=len(paths)):
        main_category_name = path.parent.name.split("_")[0]
        folder, fname = path.stem.split("_")
        folder += "_" + main_category_name
        fname += ".jpg"
        folders.append(folder)
        os.makedirs(output_folder / folder, exist_ok=True)
        shutil.copy(path, output_folder / folder / fname)
    return folders


if __name__ == "__main__":
    folders_test = make_dataset(
        paths=test_paths, output_folder="../../data/processed/dataset_part/test"
    )

    folders_train = make_dataset(
        paths=train_paths, output_folder="../../data/processed/dataset_part/train"
    )

    intersection = [x for x in folders_test if x in folders_train]

    print(
        f"share of classes in test set that presents in train "
        f"{len(intersection) / len(folders_test)}"
    )
    print(folders_test[:10])
