# ==========================================================
# STEP 15 : Generic Image Transformations
#
# SUPPORT
#   object crop from the episode category
#
# QUERY
#   full image
#
# No geometry-changing random augmentation is introduced
# here because query bounding boxes must stay aligned.
# ==========================================================

from torchvision import transforms


IMAGENET_MEAN = [
    0.485,
    0.456,
    0.406,
]

IMAGENET_STD = [
    0.229,
    0.224,
    0.225,
]


# ==========================================================
# SUPPORT TRANSFORM
#
# Input is already a crop around one object instance.
# Category may be ANY of the 80 COCO categories.
# ==========================================================

support_transform = (
    transforms.Compose([

        transforms.Resize(
            (
                CONFIG[
                    "image_size"
                ],
                CONFIG[
                    "image_size"
                ],
            ),
            antialias=True
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD
        ),
    ])
)


# ==========================================================
# QUERY TRANSFORM
#
# Full detection image.
#
# Bounding boxes will be normalized in STEP 16 using
# original image dimensions, so fixed resizing to
# [image_size, image_size] remains geometrically consistent
# in normalized coordinates.
# ==========================================================

query_transform = (
    transforms.Compose([

        transforms.Resize(
            (
                CONFIG[
                    "image_size"
                ],
                CONFIG[
                    "image_size"
                ],
            ),
            antialias=True
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD
        ),
    ])
)


# ==========================================================
# TRANSFORM SANITY
# ==========================================================

assert (
    len(IMAGENET_MEAN)
    == 3
)

assert (
    len(IMAGENET_STD)
    == 3
)


print("=" * 70)
print("STEP 15 : GENERIC SUPPORT/QUERY TRANSFORMS READY")
print("=" * 70)

print(
    "Image size       :",
    CONFIG[
        "image_size"
    ]
)

print(
    "Support input    :",
    "object crop from any COCO category"
)

print(
    "Query input      :",
    "full COCO image"
)

print(
    "Normalization    :",
    "ImageNet"
)

print(
    "Geometry aug.    :",
    "disabled for now"
)

print("=" * 70)
