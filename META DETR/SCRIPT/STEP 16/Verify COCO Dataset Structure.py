# ==========================================================
# STEP 16 : Verify COCO Dataset Structure
# ==========================================================

import os

root = "/content/datasets/coco"

folders = [
    "train2017",
    "val2017",
    "annotations"
]

print("=" * 60)
print("COCO Dataset Structure")
print("=" * 60)

for folder in folders:

    path = os.path.join(root, folder)

    if os.path.exists(path):

        total_files = len(os.listdir(path))

        print(f"{folder:15} : OK ({total_files} files)")

    else:

        print(f"{folder:15} : NOT FOUND")
