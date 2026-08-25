# ==========================================================
# STEP 17 : Load COCO Annotations & Class Mapping (FINAL)
# ==========================================================

!pip install -q pycocotools


import os

from pycocotools.coco import COCO


# ==========================================================
# Annotation Paths
# ==========================================================

TRAIN_ANN_PATH = os.path.join(
    COCO_ROOT,
    COCO_CONFIG["train_annotation"]
)

VAL_ANN_PATH = os.path.join(
    COCO_ROOT,
    COCO_CONFIG["val_annotation"]
)


# ==========================================================
# Verify Annotation Files
# ==========================================================

assert os.path.exists(
    TRAIN_ANN_PATH
), (
    f"Training annotation not found: "
    f"{TRAIN_ANN_PATH}"
)


assert os.path.exists(
    VAL_ANN_PATH
), (
    f"Validation annotation not found: "
    f"{VAL_ANN_PATH}"
)


# ==========================================================
# Load COCO API
# ==========================================================

print("=" * 70)
print("Loading MS COCO annotations...")
print("=" * 70)


coco_train = COCO(
    TRAIN_ANN_PATH
)


coco_val = COCO(
    VAL_ANN_PATH
)


# ==========================================================
# Category IDs
#
# Important:
# Original COCO IDs are NOT contiguous.
#
# Example:
# 1, 2, 3, ..., 11, 13, ...
#
# Model labels must be:
# 0 ... 79
# ==========================================================

TRAIN_CAT_IDS = sorted(
    coco_train.getCatIds()
)


VAL_CAT_IDS = sorted(
    coco_val.getCatIds()
)


# ==========================================================
# Train / Val Category Consistency
# ==========================================================

assert (
    TRAIN_CAT_IDS
    ==
    VAL_CAT_IDS
), (
    "COCO Train and Val category IDs do not match."
)


assert (
    len(TRAIN_CAT_IDS)
    ==
    CONFIG["num_classes"]
), (
    f"Expected {CONFIG['num_classes']} COCO classes, "
    f"found {len(TRAIN_CAT_IDS)}."
)


COCO_CAT_IDS = TRAIN_CAT_IDS


# ==========================================================
# COCO Category ID -> Model Label
#
# Example:
#
# COCO cat_id = 1
# ->
# model label = 0
# ==========================================================

CAT_ID_TO_LABEL = {

    cat_id: label

    for label, cat_id
    in enumerate(
        COCO_CAT_IDS
    )
}


# ==========================================================
# Model Label -> COCO Category ID
# ==========================================================

LABEL_TO_CAT_ID = {

    label: cat_id

    for cat_id, label
    in CAT_ID_TO_LABEL.items()
}


# ==========================================================
# Model Label -> Human-readable Category Name
# ==========================================================

CATEGORY_NAMES = {}


for cat_id in COCO_CAT_IDS:

    category = coco_train.loadCats(
        [cat_id]
    )[0]

    label = CAT_ID_TO_LABEL[
        cat_id
    ]

    CATEGORY_NAMES[
        label
    ] = category[
        "name"
    ]


# ==========================================================
# Sanity Checks
# ==========================================================

assert len(
    CAT_ID_TO_LABEL
) == CONFIG["num_classes"]


assert len(
    LABEL_TO_CAT_ID
) == CONFIG["num_classes"]


assert len(
    CATEGORY_NAMES
) == CONFIG["num_classes"]


assert set(
    CAT_ID_TO_LABEL.values()
) == set(
    range(
        CONFIG["num_classes"]
    )
)


# ==========================================================
# Summary
# ==========================================================

print("=" * 70)
print("STEP 17 : COCO ANNOTATIONS READY")
print("=" * 70)

print(
    "Training Images      :",
    len(coco_train.imgs)
)

print(
    "Validation Images    :",
    len(coco_val.imgs)
)

print(
    "Training Annotations :",
    len(coco_train.anns)
)

print(
    "Validation Annotations:",
    len(coco_val.anns)
)

print(
    "Foreground Classes   :",
    len(COCO_CAT_IDS)
)

print(
    "Model Label Range    :",
    f"0 - {CONFIG['num_classes'] - 1}"
)

print(
    "No-object Index      :",
    CONFIG["no_object_index"]
)

print("-" * 70)


# Display mapping
for label in range(
    CONFIG["num_classes"]
):

    cat_id = LABEL_TO_CAT_ID[
        label
    ]

    class_name = CATEGORY_NAMES[
        label
    ]

    print(
        f"{label:2d} -> "
        f"COCO ID {cat_id:2d} -> "
        f"{class_name}"
    )


print("=" * 70)
print("✓ STEP 17 PASSED")
print("=" * 70)
