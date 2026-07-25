# ==========================================================
# STEP 20 : Test COCO Episodic Dataset
# ==========================================================

episode = train_dataset[0]

support_image = episode["support_image"]
support_target = episode["support_target"]

query_image = episode["query_image"]
query_target = episode["query_target"]

print("=" * 60)
print("Support Image Shape :", support_image.shape)
print("Support Boxes       :", len(support_target["boxes"]))
print("Support Labels      :", len(support_target["labels"]))
print("-" * 60)
print("Query Image Shape   :", query_image.shape)
print("Query Boxes         :", len(query_target["boxes"]))
print("Query Labels        :", len(query_target["labels"]))
print("=" * 60)
