# ============================================================
# STEP 8 — COCO CATEGORY ID → CONTIGUOUS 0–79
# ============================================================

print("=" * 70)
print("STEP 8 — CATEGORY ID MAPPING")
print("=" * 70)

COCO_CAT_IDS = sorted(coco_train_api.getCatIds())

CAT_ID_TO_CONTIGUOUS = {
    cat_id: idx
    for idx, cat_id in enumerate(COCO_CAT_IDS)
}

CONTIGUOUS_TO_CAT_ID = {
    idx: cat_id
    for cat_id, idx in CAT_ID_TO_CONTIGUOUS.items()
}

CATEGORY_INFO = {
    cat["id"]: cat
    for cat in coco_train_api.loadCats(COCO_CAT_IDS)
}

CONTIGUOUS_TO_NAME = {
    CAT_ID_TO_CONTIGUOUS[cat_id]: CATEGORY_INFO[cat_id]["name"]
    for cat_id in COCO_CAT_IDS
}

NAME_TO_CONTIGUOUS = {
    name: idx
    for idx, name in CONTIGUOUS_TO_NAME.items()
}

PERSON_COCO_ID = coco_train_api.getCatIds(catNms=["person"])[0]
PERSON_CLASS_INDEX = CAT_ID_TO_CONTIGUOUS[PERSON_COCO_ID]

assert len(CAT_ID_TO_CONTIGUOUS) == 80
assert set(CAT_ID_TO_CONTIGUOUS.values()) == set(range(80))

print(f"COCO categories      : {len(CAT_ID_TO_CONTIGUOUS)}")
print(f"Contiguous range     : 0 - {max(CAT_ID_TO_CONTIGUOUS.values())}")
print(f"Person COCO ID       : {PERSON_COCO_ID}")
print(f"Person class index   : {PERSON_CLASS_INDEX}")

print("\nFirst 10 mappings:")
print("-" * 70)

for cat_id in COCO_CAT_IDS[:10]:
    idx = CAT_ID_TO_CONTIGUOUS[cat_id]
    name = CATEGORY_INFO[cat_id]["name"]

    print(
        f"COCO ID {cat_id:02d}"
        f" → class index {idx:02d}"
        f" → {name}"
    )

CONFIG["person_class_index"] = PERSON_CLASS_INDEX

print("\nSTEP 8 PASSED")
