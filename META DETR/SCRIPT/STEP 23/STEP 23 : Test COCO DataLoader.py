# ==========================================================
# STEP 23 : Test COCO DataLoader
# ==========================================================

images, targets = next(iter(train_loader))

print("=" * 60)
print("Batch Size :", len(images))
print("Image Shape :", images[0].shape)
print("Boxes in First Image :", len(targets[0]["boxes"]))
print("Labels in First Image :", len(targets[0]["labels"]))
print("=" * 60)
