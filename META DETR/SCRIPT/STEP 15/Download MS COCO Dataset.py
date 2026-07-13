# ==========================================================
# STEP 15 : Download MS COCO Dataset
# ==========================================================

import os
import urllib.request

os.makedirs("/content/datasets/coco", exist_ok=True)

downloads = {
    "train2017.zip":
        "http://images.cocodataset.org/zips/train2017.zip",

    "val2017.zip":
        "http://images.cocodataset.org/zips/val2017.zip",

    "annotations_trainval2017.zip":
        "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
}

for filename, url in downloads.items():

    filepath = os.path.join("/content/datasets/coco", filename)

    if os.path.exists(filepath):

        print(f"{filename} already exists.")

    else:

        print(f"Downloading {filename} ...")

        urllib.request.urlretrieve(url, filepath)

print("\nDownload Finished.")
