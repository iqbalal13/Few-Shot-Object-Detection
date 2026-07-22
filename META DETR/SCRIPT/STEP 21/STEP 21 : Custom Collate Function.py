# ==========================================================
# STEP 21 : Custom Collate Function
# ==========================================================

def collate_fn(batch):
    images = []
    targets = []

    for image, target in batch:
        images.append(image)
        targets.append(target)

    return images, targets

print("=" * 60)
print("Custom Collate Function Ready")
print("=" * 60)
