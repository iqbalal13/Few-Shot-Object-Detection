# ============================================================
# STEP 9 — RAW COCO ANNOTATION SANITY CHECK
# ============================================================

from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import random
import os

print("=" * 70)
print("STEP 9 — RAW COCO ANNOTATION CHECK")
print("=" * 70)

rng = random.Random(SEED)

train_image_ids = coco_train_api.getImgIds()

# Cari image yang punya minimal satu annotation non-crowd
valid_image_ids = []

for image_id in rng.sample(train_image_ids, min(500, len(train_image_ids))):

    ann_ids = coco_train_api.getAnnIds(
        imgIds=[image_id],
        iscrowd=False
    )

    if len(ann_ids) > 0:
        valid_image_ids.append(image_id)

    if len(valid_image_ids) >= 5:
        break

assert len(valid_image_ids) > 0, "No valid annotated image found."

image_id = valid_image_ids[0]

image_info = coco_train_api.loadImgs([image_id])[0]

image_path = os.path.join(
    CONFIG["train_images"],
    image_info["file_name"]
)

image = Image.open(image_path).convert("RGB")

ann_ids = coco_train_api.getAnnIds(
    imgIds=[image_id],
    iscrowd=False
)

annotations = coco_train_api.loadAnns(ann_ids)

draw = ImageDraw.Draw(image)

valid_boxes = 0

for ann in annotations:

    x, y, w, h = ann["bbox"]

    if w <= 0 or h <= 0:
        continue

    cat_id = ann["category_id"]

    if cat_id not in CAT_ID_TO_CONTIGUOUS:
        continue

    name = CATEGORY_INFO[cat_id]["name"]

    draw.rectangle(
        [x, y, x + w, y + h],
        outline="red",
        width=3
    )

    draw.text(
        (x, max(0, y - 12)),
        name,
        fill="red"
    )

    valid_boxes += 1

print("Image ID       :", image_id)
print("Filename       :", image_info["file_name"])
print("Original size  :", image.size)
print("Valid objects  :", valid_boxes)

plt.figure(figsize=(12, 8))
plt.imshow(image)
plt.axis("off")
plt.title("Raw COCO Annotation Sanity Check")
plt.show()

assert valid_boxes > 0

print("\nSTEP 9 PASSED")
