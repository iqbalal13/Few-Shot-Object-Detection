# ==========================================================
# STEP 19 : Initialize COCO Dataset (8K Balanced Subset)
# ==========================================================

from torch.utils.data import Subset
import random

SEED = 42
SUBSET_SIZE = 8000
MIN_IMAGES_PER_CLASS = 25

random.seed(SEED)

# ----------------------------------------------------------
# Full Episode Dataset
# ----------------------------------------------------------
full_dataset = COCOEpisodeDataset(
    coco=coco,
    image_dir="/content/datasets/coco/train2017",
    transform=transform
)

print("=" * 60)
print("Creating Balanced COCO Episodic Subset...")
print("=" * 60)

selected_img_ids = set()

# ----------------------------------------------------------
# Step 1 : Ensure every class is represented
# ----------------------------------------------------------
for cat_id in coco.getCatIds():

    img_ids = coco.getImgIds(catIds=[cat_id])
    random.shuffle(img_ids)

    count = 0

    for img_id in img_ids:

        if img_id not in selected_img_ids:
            selected_img_ids.add(img_id)
            count += 1

        if count >= MIN_IMAGES_PER_CLASS:
            break

# ----------------------------------------------------------
# Step 2 : Fill until exactly 8000 images
# ----------------------------------------------------------
remaining = list(
    set(coco.getImgIds()) - selected_img_ids
)

random.shuffle(remaining)

need = SUBSET_SIZE - len(selected_img_ids)

if need > 0:
    selected_img_ids.update(remaining[:need])

# ----------------------------------------------------------
# Convert Image ID -> Dataset Index
# ----------------------------------------------------------
id_to_index = {
    img_id: idx
    for idx, img_id in enumerate(full_dataset.image_ids)
}

subset_indices = [
    id_to_index[img_id]
    for img_id in selected_img_ids
]

subset_indices.sort()

# ----------------------------------------------------------
# Create Episodic Training Dataset
# ----------------------------------------------------------
train_dataset = Subset(
    full_dataset,
    subset_indices
)

print("=" * 60)
print("Balanced COCO Episodic Subset Ready")
print("=" * 60)
print(f"Original Dataset : {len(full_dataset)}")
print(f"Subset Dataset   : {len(train_dataset)}")
print(f"Classes          : {len(coco.getCatIds())}")
print(f"Min/Class Target : {MIN_IMAGES_PER_CLASS}")
