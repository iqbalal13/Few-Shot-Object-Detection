# ==========================================================
# STEP 19 : Initialize COCO Dataset
# ==========================================================

train_dataset = COCOBaseDataset(
    coco=coco,
    image_dir="/content/datasets/coco/train2017",
    transform=transform
)

print("=" * 60)
print("COCO Dataset Ready")
print("=" * 60)
print("Dataset Size :", len(train_dataset))
