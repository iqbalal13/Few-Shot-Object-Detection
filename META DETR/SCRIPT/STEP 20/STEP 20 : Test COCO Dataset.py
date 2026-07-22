# ==========================================================
# STEP 20 : Test COCO Dataset
# ==========================================================

image, target = train_dataset[0]

print("=" * 60)
print("Image Shape :", image.shape)
print("Number of Boxes :", len(target["boxes"]))
print("Number of Labels :", len(target["labels"]))
print("=" * 60)
