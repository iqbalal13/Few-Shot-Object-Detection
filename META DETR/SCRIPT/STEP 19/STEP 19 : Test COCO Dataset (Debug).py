# ==========================================================
# STEP 19 : Test COCO Dataset (Debug)
# ==========================================================

image_id = train_dataset.image_ids[0]

print("Image ID :", image_id)

image_info = coco.loadImgs(image_id)[0]

print("File Name :", image_info["file_name"])

import os

image_path = os.path.join(
    train_dataset.image_dir,
    image_info["file_name"]
)

print("Image Path :", image_path)

print("File Exists :", os.path.exists(image_path)
