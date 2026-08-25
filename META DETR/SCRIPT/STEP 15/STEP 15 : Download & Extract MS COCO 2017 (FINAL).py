# ==========================================================
# STEP 15 : Download & Extract MS COCO 2017 (FINAL)
# ==========================================================

import os
import urllib.request
import zipfile


COCO_ROOT = COCO_CONFIG["root"]

os.makedirs(
    COCO_ROOT,
    exist_ok=True
)


downloads = {

    "train2017.zip":
        "http://images.cocodataset.org/zips/train2017.zip",

    "val2017.zip":
        "http://images.cocodataset.org/zips/val2017.zip",

    "annotations_trainval2017.zip":
        "http://images.cocodataset.org/annotations/"
        "annotations_trainval2017.zip"
}


markers = {

    "train2017.zip":
        os.path.join(
            COCO_ROOT,
            "train2017"
        ),

    "val2017.zip":
        os.path.join(
            COCO_ROOT,
            "val2017"
        ),

    "annotations_trainval2017.zip":
        os.path.join(
            COCO_ROOT,
            "annotations",
            "instances_train2017.json"
        )
}


for filename, url in downloads.items():

    zip_path = os.path.join(
        COCO_ROOT,
        filename
    )

    marker = markers[filename]

    print("-" * 70)
    print(filename)

    # Dataset already extracted
    if os.path.exists(marker):

        print("✓ Already extracted")

        continue

    # Download if zip unavailable
    if not os.path.exists(zip_path):

        print("Downloading...")

        urllib.request.urlretrieve(
            url,
            zip_path
        )

        print("✓ Download finished")

    else:

        print("✓ ZIP already exists")

    # Extract
    print("Extracting...")

    with zipfile.ZipFile(
        zip_path,
        "r"
    ) as zip_ref:

        zip_ref.extractall(
            COCO_ROOT
        )

    print("✓ Extraction finished")


print("=" * 70)
print("STEP 15 COMPLETED")
print("=" * 70)
