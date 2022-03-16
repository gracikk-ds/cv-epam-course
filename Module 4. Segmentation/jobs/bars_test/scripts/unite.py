import re
import numpy as np
from PIL import Image
from pathlib import Path


def concatenation(paths, name):
    for i, list_of_paths in enumerate(paths):
        for j, path in enumerate(list_of_paths):
            msk = np.array(Image.open(path))
            if j == 0:
                levels_x = msk
            else:
                levels_x = np.concatenate([levels_x, msk], axis=1)
        if i == 0:
            levels_y = levels_x
        else:
            levels_y = np.concatenate([levels_y, levels_x], axis=0)

    Image.fromarray(levels_y).save(f"../test_data/predictions/{name}.png")


if __name__ == "__main__":
    paths = np.array(
        [x for x in Path("../test_data/predictions_croped").glob("*")]
    ).reshape(-1, 12)

    for img_paths in paths:
        result = re.search("(.*)-", img_paths[0].name)
        name = result.group(1)[:-2]
        concatenation(img_paths.reshape(-1, 4), name)
