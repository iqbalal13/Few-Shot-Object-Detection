# ==========================================================
# STEP 16 : Verify COCO Dataset Structure (FINAL)
# ==========================================================

import os


# ==========================================================
# Required COCO Paths
# ==========================================================

required_paths = {

    "Train Images":
        os.path.join(
            COCO_ROOT,
            COCO_CONFIG["train_images"]
        ),

    "Validation Images":
        os.path.join(
            COCO_ROOT,
            COCO_CONFIG["val_images"]
        ),

    "Train Annotation":
        os.path.join(
            COCO_ROOT,
            COCO_CONFIG["train_annotation"]
        ),

    "Validation Annotation":
        os.path.join(
            COCO_ROOT,
            COCO_CONFIG["val_annotation"]
        )
}


# ==========================================================
# Verify Required Files / Directories
# ==========================================================

print("=" * 70)
print("STEP 16 : VERIFY COCO DATASET")
print("=" * 70)


for name, path in required_paths.items():

    exists = os.path.exists(
        path
    )

    print(
        f"{name:25s}: "
        f"{'✓ FOUND' if exists else '✗ MISSING'}"
    )

    print(
        f"  {path}"
    )

    assert exists, (
        f"Missing required COCO data: {path}"
    )


# ==========================================================
# Count Images
# ==========================================================

train_image_dir = required_paths[
    "Train Images"
]

val_image_dir = required_paths[
    "Validation Images"
]


train_images = [

    filename

    for filename
    in os.listdir(
        train_image_dir
    )

    if filename.lower().endswith(
        (
            ".jpg",
            ".jpeg",
            ".png"
        )
    )
]


val_images = [

    filename

    for filename
    in os.listdir(
        val_image_dir
    )

    if filename.lower().endswith(
        (
            ".jpg",
            ".jpeg",
            ".png"
        )
    )
]


train_count = len(
    train_images
)

val_count = len(
    val_images
)


# ==========================================================
# Basic Dataset Sanity Check
# ==========================================================

assert (
    train_count > 0
), "COCO train2017 directory is empty."


assert (
    val_count > 0
), "COCO val2017 directory is empty."


# COCO 2017 standard expected counts
EXPECTED_TRAIN_IMAGES = 118287
EXPECTED_VAL_IMAGES = 5000


print("-" * 70)

print(
    "Train Images :",
    train_count
)

print(
    "Val Images   :",
    val_count
)


# ==========================================================
# Expected COCO Size Check
# ==========================================================

if (
    train_count
    ==
    EXPECTED_TRAIN_IMAGES
):

    print(
        "Train Count  : ✓ Standard COCO 2017"
    )

else:

    print(
        "Train Count  : WARNING - "
        f"expected {EXPECTED_TRAIN_IMAGES}, "
        f"found {train_count}"
    )


if (
    val_count
    ==
    EXPECTED_VAL_IMAGES
):

    print(
        "Val Count    : ✓ Standard COCO 2017"
    )

else:

    print(
        "Val Count    : WARNING - "
        f"expected {EXPECTED_VAL_IMAGES}, "
        f"found {val_count}"
    )


# ==========================================================
# Annotation File Size Check
# ==========================================================

train_ann_size_mb = (

    os.path.getsize(
        required_paths[
            "Train Annotation"
        ]
    )

    /

    (1024 ** 2)
)


val_ann_size_mb = (

    os.path.getsize(
        required_paths[
            "Validation Annotation"
        ]
    )

    /

    (1024 ** 2)
)


print("-" * 70)

print(
    "Train Annotation Size :",
    f"{train_ann_size_mb:.2f} MB"
)

print(
    "Val Annotation Size   :",
    f"{val_ann_size_mb:.2f} MB"
)


assert (
    train_ann_size_mb > 0
), "Training annotation file is empty."


assert (
    val_ann_size_mb > 0
), "Validation annotation file is empty."


print("=" * 70)
print("✓ STEP 16 PASSED")
print("=" * 70)
