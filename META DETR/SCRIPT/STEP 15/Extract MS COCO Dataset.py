# ==========================================================
# STEP 15 : Extract MS COCO Dataset
# ==========================================================

import zipfile

zip_files = [

    "train2017.zip",

    "val2017.zip",

    "annotations_trainval2017.zip"

]

for zip_name in zip_files:

    zip_path = f"/content/datasets/coco/{zip_name}"

    print(f"Extracting {zip_name} ...")

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:

        zip_ref.extractall("/content/datasets/coco")

print("\nExtraction Finished.")
