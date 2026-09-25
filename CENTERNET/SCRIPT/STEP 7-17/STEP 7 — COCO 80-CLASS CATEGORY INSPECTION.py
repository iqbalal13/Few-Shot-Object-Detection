# ============================================================
# STEP 7 — COCO 80-CLASS CATEGORY INSPECTION
# ============================================================

from pycocotools.coco import COCO

print("=" * 70)
print("STEP 7 — COCO CATEGORY INSPECTION")
print("=" * 70)

coco_train_api = COCO(CONFIG["train_annotations"])
coco_val_api   = COCO(CONFIG["val_annotations"])

train_cat_ids = sorted(coco_train_api.getCatIds())
val_cat_ids   = sorted(coco_val_api.getCatIds())

train_categories = coco_train_api.loadCats(train_cat_ids)
val_categories   = coco_val_api.loadCats(val_cat_ids)

assert len(train_cat_ids) == 80, \
    f"Expected 80 COCO train categories, got {len(train_cat_ids)}"

assert len(val_cat_ids) == 80, \
    f"Expected 80 COCO val categories, got {len(val_cat_ids)}"

assert train_cat_ids == val_cat_ids, \
    "Train and validation category IDs do not match."

print(f"\nNumber of train categories : {len(train_cat_ids)}")
print(f"Number of val categories   : {len(val_cat_ids)}")

print("\nCOCO 80 categories:")
print("-" * 70)

for i, category in enumerate(train_categories):
    print(
        f"{i:02d} | "
        f"COCO ID: {category['id']:02d} | "
        f"{category['name']}"
    )

print("\nSTEP 7 PASSED")
