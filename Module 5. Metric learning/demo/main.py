import time
import pickle
import numpy as np
from PIL import Image
import streamlit as st
from pathlib import Path
from dynaconf import settings
from visualization import draw_objects
from inference import process_the_image, predict, milvus_search
from pymilvus import (
    connections,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    utility,
    drop_collection,
)


HOST_MILVUS = settings["milvus"]["host"]
PORT_MILVUS = settings["milvus"]["port"]


def milvus_add_collection(
    host: str = HOST_MILVUS,
    port: str = PORT_MILVUS,
    collection_name: str = "demo_metric",
):
    connections.connect(host=host, port=port)
    drop_collection(collection_name="demo_metric")
    # Does collection demo_metric exist in Milvus?
    has = utility.has_collection(collection_name)
    print("Collection status: ", has)
    if not has:

        # create new collection
        schema = CollectionSchema(
            [
                FieldSchema("embedding_id", DataType.INT64, is_primary=True),
                FieldSchema("label_id", DataType.INT64),
                FieldSchema("embeddings", dtype=DataType.FLOAT_VECTOR, dim=512),
            ]
        )

        collection = Collection(
            name=collection_name, schema=schema, using="default", shards_num=2
        )

        # upload embeddings
        with open("./pickles/embeddings.pickle", "rb") as handle:
            embeddings = pickle.load(handle)
        # fill collection with embeddings
        collection.insert(embeddings)

        # create index for fast search
        index_params = {
            "metric_type": "L2",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 1024},
        }

        collection.create_index("embeddings", index_params=index_params)
        collection.load()


def config():
    """Core demo configuration"""
    milvus_add_collection()
    logo = Image.open("branded/flower-logo.png")
    st.set_page_config(
        page_title="Metric Learning Demo",
        page_icon=logo,
        layout="wide",
    )

    st.title("Welcome to Metric Learning Demo")
    st.write(" ------ ")
    st.sidebar.title("Explore the Following")


def run():
    """main run demo script"""
    # Create image uploader button
    image_loader = st.sidebar.file_uploader(
        label="Upload image", type=["png", "jpg", "JPG", "jpeg"]
    )
    image = None

    # Create run detection button
    run_button = st.sidebar.button(label="Run")

    # Image preprocessing
    if image_loader is not None:
        image, image_name = process_the_image(image_loader)

    # Running the model
    if run_button and image is not None:
        start = time.process_time()

        embedding = predict(image)
        results, paths = milvus_search([embedding])
        for result, path in zip(results, paths):
            img_path = Path("../data/processed/dataset_part/train/") / result / path
            img = np.array(Image.open(img_path))
            fig = draw_objects(image, img, result)
            st.pyplot(fig=fig)

            end = time.process_time()
            st.text(f"Model running time: {end - start: 2f} s")

    elif run_button and image is None:
        st.write("Image is not selected.")


if __name__ == "__main__":
    config()
    run()
