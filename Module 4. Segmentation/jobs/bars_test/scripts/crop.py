import numpy as np
from PIL import Image
from pathlib import Path


# TODO: write function description


def crop(img_pth, msk_pth):
    """function description here"""

    img = np.array(Image.open(img_pth))
    msk = np.array(Image.open(msk_pth))

    y, x = msk.shape[:2]
    coef_y = y // 255
    coef_x = x // 255

    resolution_y = np.ceil(y / coef_y)
    resolution_x = np.ceil(x / coef_x)

    print(f"coef_y: {coef_y}, coef_x: {coef_x}")
    print(f"resolution_y: {resolution_y}, resolution_x: {resolution_x}")

    for i in range(coef_y):
        for j in range(coef_x):
            img_crop = Image.fromarray(
                img[
                    int(i * resolution_y) : int((i + 1) * resolution_y),
                    int(j * resolution_x) : int((j + 1) * resolution_x),
                ]
            )

            msk_crop = Image.fromarray(
                msk[
                    int(i * resolution_y) : int((i + 1) * resolution_y),
                    int(j * resolution_x) : int((j + 1) * resolution_x),
                ]
            )

            img_crop.save(
                "./test/Images_crops/"
                + img_pth.stem
                + "-"
                + str(i)
                + "-"
                + str(j)
                + ".png"
            )
            msk_crop.save(
                "./test/Masks_crops/"
                + msk_pth.stem
                + "-"
                + str(i)
                + "-"
                + str(j)
                + ".png"
            )


if __name__ == "__main__":
    paths_img = [x for x in Path("./test/Images").glob("*")]
    paths_msk = [x for x in Path("./test/Masks").glob("*")]

    for img_pth, msk_pth in zip(paths_img, paths_msk):
        crop(img_pth, msk_pth)
