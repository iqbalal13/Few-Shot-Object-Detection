# ==========================================================
# STEP 14 : COCO Annotation & Class Mapping
# ==========================================================

assert "COCO_ROOT" in globals(), (
    "Run STEP 13 first."
)

assert "TRAIN_ANN_PATH" in globals()
assert "VAL_ANN_PATH" in globals()


print("=" * 70)
print("STEP 14 : LOADING COCO ANNOTATIONS")
print("=" * 70)


coco_train = COCO(
    TRAIN_ANN_PATH
)

coco_val = COCO(
    VAL_ANN_PATH
)


TRAIN_CAT_IDS = sorted(
    coco_train.getCatIds()
)

VAL_CAT_IDS = sorted(
    coco_val.getCatIds()
)


assert (
    TRAIN_CAT_IDS
    ==
    VAL_CAT_IDS
), (
    "COCO Train/Val category IDs differ."
)


assert (
    len(TRAIN_CAT_IDS)
    ==
    COCO_CONFIG["num_classes"]
), (
    "Expected 80 COCO classes."
)


COCO_CAT_IDS = TRAIN_CAT_IDS


CAT_ID_TO_LABEL = {

    cat_id:
        label

    for label, cat_id
    in enumerate(
        COCO_CAT_IDS
    )
}


LABEL_TO_CAT_ID = {

    label:
        cat_id

    for cat_id, label
    in CAT_ID_TO_LABEL.items()
}


CATEGORY_NAMES = {}


for cat_id in COCO_CAT_IDS:

    category = (
        coco_train.loadCats(
            [cat_id]
        )[0]
    )

    label = (
        CAT_ID_TO_LABEL[
            cat_id
        ]
    )

    CATEGORY_NAMES[
        label
    ] = category["name"]


assert (
    set(
        CAT_ID_TO_LABEL.values()
    )
    ==
    set(range(80))
)


assert (
    CATEGORY_NAMES[0]
    ==
    "person"
)


print("-" * 70)

print(
    "Train Images      :",
    len(coco_train.imgs)
)

print(
    "Val Images        :",
    len(coco_val.imgs)
)

print(
    "Train Annotations :",
    len(coco_train.anns)
)

print(
    "Val Annotations   :",
    len(coco_val.anns)
)

print(
    "COCO Classes      :",
    len(COCO_CAT_IDS)
)

print("-" * 70)

print(
    "Label 0           :",
    CATEGORY_NAMES[0]
)

print(
    "Label 79          :",
    CATEGORY_NAMES[79]
)

print("=" * 70)
print("✓ STEP 14 PASSED")
print("=" * 70)
