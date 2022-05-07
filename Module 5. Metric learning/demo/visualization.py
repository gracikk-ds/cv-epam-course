from matplotlib import pyplot as plt
import numpy as np
from PIL import Image


def draw_objects(image_input, image_example, title_text):
    image_example = np.array(
        Image.fromarray(image_example).resize(
            (image_input.shape[1], image_input.shape[0])
        )
    )
    # create figure
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    fig.suptitle(f"Predicted class: {title_text}", fontsize=20)
    axes[0].imshow(image_input)
    axes[0].set_title("Input image")
    axes[0].set_axis_off()
    axes[1].imshow(image_example)
    axes[1].set_title("The most similar one")
    axes[1].set_axis_off()
    return fig
