import cv2
import torch
import pandas as pd
import skimage
import numpy as np
from PIL import Image
from pathlib import Path
from torchmetrics import MeanAbsolutePercentageError


def count_blobs(msk):

    blobs = skimage.feature.blob_log(
        msk, min_sigma=3, max_sigma=7, num_sigma=1, threshold=0.02
    )

    return len(blobs)


if __name__ == "__main__":
    mape = MeanAbsolutePercentageError()
    paths_img = [x for x in Path("../test_data/predictions").glob("*")]

    df = pd.read_csv("../test_data/blobs_df_test.csv")

    for i, path_img in enumerate(paths_img):
        msk = np.array(Image.open(path_img))
        count_predicted = count_blobs(msk)
        df.loc[df["new_filename"] == path_img.name, "predicted"] = count_predicted

    df.to_csv("../test_data/blobs_df_test.csv")
    print(
        f"MAPE: {mape(torch.tensor(df['count'].values), torch.tensor(df['predicted'].values)):0.2f}"
    )
