# ==========================================================
# STEP 14: Person-Only Category Mapping
# ==========================================================

coco_train = COCO(TRAIN_ANN_PATH)
coco_val = COCO(VAL_ANN_PATH)

COCO_CAT_IDS = sorted(
    coco_train.getCatIds(
        catNms=["person"]
    )
)

assert len(COCO_CAT_IDS) == 1

assert sorted(
    coco_val.getCatIds(
        catNms=["person"]
    )
) == COCO_CAT_IDS

PERSON_CAT_ID = COCO_CAT_IDS[0]

TRAIN_CAT_IDS = list(COCO_CAT_IDS)
VAL_CAT_IDS = list(COCO_CAT_IDS)

CAT_ID_TO_LABEL = {
    PERSON_CAT_ID: 0
}

LABEL_TO_CAT_ID = {
    0: PERSON_CAT_ID
}

CATEGORY_NAMES = {
    0: "person"
}

print("=" * 70)
print("STEP 14: PERSON MAPPING READY")
print("=" * 70)

print("COCO person category ID:", PERSON_CAT_ID)
print("Internal person label  : 0")

print(
    "Person train images:",
    len(
        coco_train.getImgIds(
            catIds=COCO_CAT_IDS
        )
    )
)

print(
    "Person val images:",
    len(
        coco_val.getImgIds(
            catIds=COCO_CAT_IDS
        )
    )
)

print("=" * 70)
