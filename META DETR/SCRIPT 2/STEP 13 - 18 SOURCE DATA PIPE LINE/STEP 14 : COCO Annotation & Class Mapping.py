# ==========================================================
# STEP 14 : COCO-80 Category Mapping
#
# COCO category IDs are NOT contiguous:
# e.g. 1, 2, ..., gaps, ..., 90.
#
# We therefore create:
#
# COCO category ID
#       ↕
# internal label 0..79
#
# Stage 1 uses all labels.
# Stage 2 later selects person only.
# ==========================================================

coco_train = COCO(
    TRAIN_ANN_PATH
)

coco_val = COCO(
    VAL_ANN_PATH
)


# ==========================================================
# CATEGORY IDs
# ==========================================================

TRAIN_CAT_IDS = sorted(
    coco_train.getCatIds()
)

VAL_CAT_IDS = sorted(
    coco_val.getCatIds()
)


assert (
    len(TRAIN_CAT_IDS)
    == 80
), (
    "COCO train must contain "
    "80 instance categories."
)

assert (
    len(VAL_CAT_IDS)
    == 80
), (
    "COCO val must contain "
    "80 instance categories."
)

assert (
    TRAIN_CAT_IDS
    ==
    VAL_CAT_IDS
), (
    "Train/val category IDs differ."
)


COCO_CAT_IDS = list(
    TRAIN_CAT_IDS
)


# ==========================================================
# LOAD CATEGORY METADATA
# ==========================================================

category_metadata = (
    coco_train.loadCats(
        COCO_CAT_IDS
    )
)

category_metadata = sorted(
    category_metadata,
    key=lambda item: item["id"]
)


CAT_ID_TO_LABEL = {

    cat_id: label

    for label, cat_id
    in enumerate(
        COCO_CAT_IDS
    )
}


LABEL_TO_CAT_ID = {

    label: cat_id

    for cat_id, label
    in CAT_ID_TO_LABEL.items()
}


CAT_ID_TO_NAME = {

    int(item["id"]):
        str(item["name"])

    for item in category_metadata
}


CATEGORY_NAMES = {

    CAT_ID_TO_LABEL[
        cat_id
    ]:
        CAT_ID_TO_NAME[
            cat_id
        ]

    for cat_id
    in COCO_CAT_IDS
}


# ==========================================================
# PERSON IDENTIFIERS
#
# Retained for Stage 2 person specialization.
# ==========================================================

person_ids = (
    coco_train.getCatIds(
        catNms=[
            "person"
        ]
    )
)

assert (
    len(person_ids)
    == 1
)

PERSON_CAT_ID = int(
    person_ids[0]
)

PERSON_LABEL = int(
    CAT_ID_TO_LABEL[
        PERSON_CAT_ID
    ]
)


assert (
    CATEGORY_NAMES[
        PERSON_LABEL
    ]
    ==
    "person"
)


# ==========================================================
# CROSS-CHECK CATEGORY NAMES WITH VALIDATION
# ==========================================================

val_metadata = sorted(
    coco_val.loadCats(
        VAL_CAT_IDS
    ),
    key=lambda item: item["id"]
)

val_names = {

    int(item["id"]):
        str(item["name"])

    for item in val_metadata
}

assert (
    val_names
    ==
    CAT_ID_TO_NAME
)


# ==========================================================
# BASIC SOURCE DATA STATISTICS
# ==========================================================

TRAIN_IMAGES_PER_CATEGORY = {

    cat_id:
        len(
            coco_train.getImgIds(
                catIds=[
                    cat_id
                ]
            )
        )

    for cat_id
    in COCO_CAT_IDS
}


VAL_IMAGES_PER_CATEGORY = {

    cat_id:
        len(
            coco_val.getImgIds(
                catIds=[
                    cat_id
                ]
            )
        )

    for cat_id
    in COCO_CAT_IDS
}


assert all(
    count > 0
    for count
    in TRAIN_IMAGES_PER_CATEGORY.values()
)

assert all(
    count > 0
    for count
    in VAL_IMAGES_PER_CATEGORY.values()
)


print("=" * 70)
print("STEP 14 : COCO-80 CATEGORY MAPPING READY")
print("=" * 70)

print(
    "Number of categories :",
    len(
        COCO_CAT_IDS
    )
)

print(
    "Internal labels      :",
    (
        min(
            LABEL_TO_CAT_ID
        ),
        max(
            LABEL_TO_CAT_ID
        )
    )
)

print(
    "Person COCO ID       :",
    PERSON_CAT_ID
)

print(
    "Person internal label:",
    PERSON_LABEL
)

print(
    "Person train images  :",
    TRAIN_IMAGES_PER_CATEGORY[
        PERSON_CAT_ID
    ]
)

print(
    "Person val images    :",
    VAL_IMAGES_PER_CATEGORY[
        PERSON_CAT_ID
    ]
)

print()
print("First 10 internal categories:")

for label in range(10):

    print(
        f"{label:02d}"
        " -> COCO "
        f"{LABEL_TO_CAT_ID[label]:02d}"
        " -> "
        f"{CATEGORY_NAMES[label]}"
    )

print("=" * 70)
