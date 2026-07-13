# ==========================================================
# STEP 17 : Show COCO Categories
# ==========================================================

categories = coco.loadCats(coco.getCatIds())

print("=" * 60)
print("COCO Categories")
print("=" * 60)

for category in categories:
    print(category["id"], "-", category["name"])
