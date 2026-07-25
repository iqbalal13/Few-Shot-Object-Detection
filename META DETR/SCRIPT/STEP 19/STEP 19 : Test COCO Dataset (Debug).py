# ==========================================================
# STEP 19 : Test COCO Episodic Dataset (Debug)
# ==========================================================

import torch

print("=" * 60)
print("Testing COCO Episodic Dataset...")
print("=" * 60)

# ----------------------------------------------------------
# Get One Episode
# ----------------------------------------------------------
episode = train_dataset[0]

support_image = episode["support_image"]
support_target = episode["support_target"]

query_image = episode["query_image"]
query_target = episode["query_target"]

# ----------------------------------------------------------
# Support Information
# ----------------------------------------------------------
print("\nSUPPORT")
print("-" * 40)

print("Image Shape :", tuple(support_image.shape))
print("Image ID    :", support_target["image_id"].item())
print("Boxes       :", support_target["boxes"].shape)
print("Labels      :", support_target["labels"].shape)

if len(support_target["labels"]) > 0:
    print("Classes     :", support_target["labels"].tolist())
else:
    print("Classes     : []")

# ----------------------------------------------------------
# Query Information
# ----------------------------------------------------------
print("\nQUERY")
print("-" * 40)

print("Image Shape :", tuple(query_image.shape))
print("Image ID    :", query_target["image_id"].item())
print("Boxes       :", query_target["boxes"].shape)
print("Labels      :", query_target["labels"].shape)

if len(query_target["labels"]) > 0:
    print("Classes     :", query_target["labels"].tolist())
else:
    print("Classes     : []")

# ----------------------------------------------------------
# Episode Summary
# ----------------------------------------------------------
print("\nEPISODE SUMMARY")
print("-" * 40)

print(
    "Support Boxes :",
    len(support_target["boxes"])
)

print(
    "Query Boxes   :",
    len(query_target["boxes"])
)

support_classes = set(
    support_target["labels"].tolist()
)

query_classes = set(
    query_target["labels"].tolist()
)

common_classes = support_classes.intersection(query_classes)

print("Shared Classes :", sorted(list(common_classes)))

if len(common_classes) > 0:
    print("\n✅ Episode Created Successfully")
else:
    print("\n⚠️ Warning: No Shared Class Found")
