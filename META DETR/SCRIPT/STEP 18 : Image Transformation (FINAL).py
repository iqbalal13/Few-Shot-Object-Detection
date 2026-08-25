# ==========================================================
# STEP 18 : Image Transformation (FINAL)
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
# Support
#
# Support sudah berupa object crop.
# ==========================================================

support_transform = transforms.Compose([

    transforms.Resize(
        (
            CONFIG["image_size"],
            CONFIG["image_size"]
        )
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD
    )

])


# ==========================================================
# Query
#
# Query adalah full detection image.
# ==========================================================

query_transform = transforms.Compose([

    transforms.Resize(
        (
            CONFIG["image_size"],
            CONFIG["image_size"]
        )
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD
    )

])


print("=" * 70)
print("STEP 18 : TRANSFORMS READY")
print("=" * 70)

print("Support Transform")
print(support_transform)

print("-" * 70)

print("Query Transform")
print(query_transform)

print("=" * 70)
