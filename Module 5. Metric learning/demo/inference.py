import pickle
import torch
import numpy as np
import pandas as pd
from typing import List
from dynaconf import settings
from PIL import Image, ImageOps
from model import transformation
from pymilvus import connections


HOST_MILVUS = settings["milvus"]["host"]
PORT_MILVUS = settings["milvus"]["port"]


def remap_it(idx_class, mapper, decode=True):
    if decode:
        result = list(mapper.keys())[list(mapper.values()).index(idx_class)]
    else:
        result = list(mapper.values())[list(mapper.keys()).index(idx_class)]
    return result


def process_the_image(image_loader):
    """Preprocessing uploaded image"""

    image_name = image_loader.name
    image = Image.open(image_loader).convert("RGB")
    image = ImageOps.exif_transpose(image)
    image = np.array(image)

    return image, image_name


def predict(image):
    model = torch.jit.load("../models/torchscript.pt")

    image = transformation(img=image)
    image = image[np.newaxis, ...]

    model.eval()
    with torch.no_grad():
        embedding = model(image.float())
        embedding = embedding.squeeze().numpy()

    with open("./pickles/normalizer.pickle", "rb") as handle:
        scaler = pickle.load(handle)

    embedding = scaler.transform([embedding])

    return embedding[0]


def milvus_search(
    embeddings: List[List[float]],
    host: str = HOST_MILVUS,
    port: str = PORT_MILVUS,
    collection_name: str = "demo_metric",
):
    connection = connections.connect(host=host, port=port)

    ids_range = list(
        range(connection.get_collection_stats(collection_name)["row_count"])
    )

    maping_df = connection.query(
        collection_name=collection_name,
        expr=f"embedding_id in {ids_range}",
        output_fields=["embedding_id", "label_id"],
    )

    maping_df = (
        pd.DataFrame.from_dict(maping_df)
        .sort_values(by="embedding_id")
        .reset_index(drop=True)
    )

    search_params = {"metric_type": "L2", "params": {"nprobe": 1024}}

    results = connection.search(
        collection_name=collection_name,
        data=embeddings,
        anns_field="embeddings",
        param=search_params,
        limit=3,
        expression=None,
    )

    result_ids = [element.ids for element in results]
    result_dsts_top = [element.distances for element in results]

    pathes = pd.read_csv("./pickles/data_pathes.csv")
    result_ids[0] = [int(x) for x in result_ids[0]]
    pathes = pathes.iloc[result_ids[0]]

    concat_df = pd.DataFrame(index=range(len(np.array(result_ids).T[0])))
    for i in range(len(np.array(result_ids).T)):
        df = (
            maping_df.iloc[np.array(result_ids).T[i]].reset_index().loc[:, ["label_id"]]
        )
        concat_df = pd.concat([concat_df, df], axis=1)

    print(concat_df)
    # concat_df["result"] = concat_df.mode(axis=1)[0].values.astype(int)
    predictions_label_id = concat_df.iloc[0, :].values.astype(int)

    with open("./pickles/mapper_faces.pickle", "rb") as handle:
        mapper_dict = pickle.load(handle)

    predictions = [
        remap_it(class_id, mapper_dict, decode=True)
        for class_id in predictions_label_id
    ]

    pathes = pathes.loc[pathes["classes_names"].isin(predictions), "paths"].values

    print(predictions)
    print(result_dsts_top[0])
    print(pathes)

    return predictions, pathes
