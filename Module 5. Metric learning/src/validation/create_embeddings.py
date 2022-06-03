import click
import torch
import pickle
import numpy as np
from tqdm import tqdm
from pathlib import Path
from transformations import Transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from sklearn.preprocessing import StandardScaler


BATCH_SIZE = 32


def get_embeddings(model, dataloader):
    classes = []
    embeddings = []
    model.eval()
    with torch.no_grad():
        for i, (x, y) in enumerate(tqdm(dataloader)):
            classes.extend(y.numpy())
            _, embeddings_tmp = model(x)
            embeddings_tmp = list(embeddings_tmp)
            embeddings.extend(embeddings_tmp)
    classes = np.array(classes)
    embeddings = np.array([x.numpy() for x in embeddings])
    print(f"len embeddings: {len(embeddings)}, len classes {len(classes)}")
    return embeddings, classes


@click.command()
@click.option(
    "--dataset_folder",
    help="path to processed dataset folder.",
    default="../data/interim/dataset_part",
)
def create_embeddings(dataset_folder: str):
    dataset_folder_to_use = Path(dataset_folder)

    # initialize the model
    model = torch.jit.load("../models/torchscript.pt")

    train_dataset = ImageFolder(
        root=str(dataset_folder_to_use / "train"),
        transform=Transforms(segment="val"),
    )

    val_dataset = ImageFolder(
        root=str(dataset_folder_to_use / "test"),
        transform=Transforms(segment="val"),
    )

    train_dl = DataLoader(
        train_dataset,
        BATCH_SIZE,
        pin_memory=False,
        shuffle=False,
        num_workers=4,
        drop_last=False,
    )

    val_dl = DataLoader(
        val_dataset,
        BATCH_SIZE,
        pin_memory=False,
        shuffle=False,
        num_workers=4,
        drop_last=False,
    )

    embeddings_train, classes_train = get_embeddings(model, train_dl)
    embeddings_val, classes_val = get_embeddings(model, val_dl)

    scaler = StandardScaler()
    embeddings_train = scaler.fit_transform(embeddings_train)
    embeddings_val = scaler.transform(embeddings_val)

    data_train = [
        [i for i in range(len(classes_train))],
        classes_train.tolist(),
        embeddings_train.tolist(),
    ]

    data_test = [
        [i for i in range(len(classes_val))],
        classes_val.tolist(),
        embeddings_val.tolist(),
    ]

    with open("../data/results/embeddings_train.pickle", "wb") as handle:
        pickle.dump(data_train, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open("../data/results/embeddings_test.pickle", "wb") as handle:
        pickle.dump(data_test, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    create_embeddings()
