import skimage
from pathlib import Path
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt

paths = [x for x in Path("./masks").glob("*")]

for path in paths:
    img = np.array(Image.open(path))
    print(img.dtype)
    print(img.shape)
    print(img.max())
    # detect blobs

    blobs = skimage.feature.blob_log(
        img, min_sigma=3, max_sigma=4, num_sigma=1, threshold=0.02
    )
    for blob in blobs:
        img = cv2.circle(img, (int(blob[1]), int(blob[0])), 8, (0, 0, 255), 2)

    plt.imshow(img)
    plt.show()
    print(f"filename: {path.name}, len blobs: {len(blobs)}")
