import os
import click
import shutil
from tqdm import tqdm
from pathlib import Path


@click.command()
@click.option("--flag", type=str, default="train", required=True,)
def make_dataset(flag: str):
    """
    Create Imagefolder dataset from .txt files
    :param flag: one of "train"/"test"
    :return: None
    """
    root_base = Path("./Stanford_Online_Products")

    # determine info file
    if flag == "train":
        destinations_store = "Ebay_train.txt"
    else:
        destinations_store = "Ebay_test.txt"

    # collect paths
    with open(root_base / destinations_store) as file_txt:
        lines = file_txt.readlines()
        paths = [root_base / x.split()[-1] for x in lines[1:]]

    # make dirs for each category
    categories = set([x.parts[-2] for x in paths])
    for category in categories:
        os.makedirs("./dataset/" + flag + "/" + category, exist_ok=True)

    # copy images
    for path in tqdm(paths):
        shutil.copy(path, "./dataset/" + flag + "/" + path.parts[-2] + "/" + Path(path).name)


if __name__ == "__main__":
    make_dataset()
