# ==========================================================
# STEP 23 : Test COCO Episodic DataLoader
# ==========================================================

batch = next(iter(train_loader))

support_images = batch["support_images"]
support_targets = batch["support_targets"]

query_images = batch["query_images"]
query_targets = batch["query_targets"]

print("=" * 60)
print("Batch Size :", len(support_images))
print("-" * 60)
print("Support Image Shape :", support_images[0].shape)
print("Support Boxes       :", len(support_targets[0]["boxes"]))
print("Support Labels      :", len(support_targets[0]["labels"]))
print("-" * 60)
print("Query Image Shape   :", query_images[0].shape)
print("Query Boxes         :", len(query_targets[0]["boxes"]))
print("Query Labels        :", len(query_targets[0]["labels"]))
print("=" * 60)
