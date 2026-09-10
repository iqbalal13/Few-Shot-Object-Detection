# ==========================================================
# STEP 15 : Image Transformations
# ==========================================================

from torchvision import transforms


IMAGENET_MEAN = [
    0.485,
    0.456,
    0.406
]

IMAGENET_STD = [
    0.229,
    0.224,
    0.225
]


# ==========================================================
# SUPPORT
#
# Input is already an object crop.
# ==========================================================

support_transform = (
    transforms.Compose([

        transforms.Resize(

            (
                CONFIG["image_size"],
                CONFIG["image_size"]
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
# QUERY
#
# Full detection image.
# ==========================================================

query_transform = (
    transforms.Compose([

        transforms.Resize(

            (
                CONFIG["image_size"],
                CONFIG["image_size"]
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


print("=" * 70)
print("STEP 15 : IMAGE TRANSFORMS READY")
print("=" * 70)

print(
    "Image Size:",
    CONFIG["image_size"]
)

print(
    "Normalization: ImageNet"
)

print("=" * 70)
