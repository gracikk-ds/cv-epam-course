import time
import numpy as np
from PIL import Image
import streamlit as st
from pathlib import Path
from visualization import draw_objects
from inference import process_the_image, predict, milvus_search


def config():
    """Core demo configuration"""
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
        result = milvus_search(embedding)
        img_path = [x for x in Path("../examples/" + result[0]).glob("*")][0]
        img = np.array(Image.open(img_path))
        fig = draw_objects(image, img, result[0])
        st.pyplot(fig=fig)

        end = time.process_time()
        st.text(f"Model running time: {end - start: 2f} s")

    elif run_button and image is None:
        st.write("Image is not selected.")


if __name__ == "__main__":
    config()
    run()
