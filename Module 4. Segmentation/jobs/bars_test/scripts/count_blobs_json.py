import json
import string
import pandas as pd
from pathlib import Path


# TODO: write function description


def count_blobs(paths_images_old, paths_images_new, data_json, df):
    """function description here"""
    for i, (path_old, path_new) in enumerate(zip(paths_images_old, paths_images_new)):
        df.loc[i, "filename_old"] = path_old.name
        df.loc[i, "new_filename"] = path_new.name
        df.loc[i, "count"] = len(data_json[path_old.name]["regions"].values())
    return df


if __name__ == "__main__":
    with open("../test_data/test_labels.json") as json_file:
        data_json_test = json.load(json_file)

    # remove scale from key
    nonalpha = string.digits + string.punctuation + string.whitespace

    for old_name, new_name in [
        (x, x.rstrip(nonalpha)) for x in list(data_json_test.keys())
    ]:
        data_json_test[new_name] = data_json_test.pop(old_name)

    paths_images_test_old = sorted(
        [x for x in Path("../test_data/bars_test").glob("*")]
    )
    paths_images_test_new = [
        Path("../test_data/Images/image_" + str(i + 1) + ".png")
        for i in range(len(paths_images_test_old))
    ]
    blobs_df_test = pd.DataFrame(
        columns=["filename_old", "new_filename", "count"],
        index=range(len(paths_images_test_new)),
    )

    blobs_df_test = count_blobs(
        paths_images_test_old, paths_images_test_new, data_json_test, blobs_df_test
    )

    blobs_df_test.to_csv("../test_data/blobs_df_test.csv")
