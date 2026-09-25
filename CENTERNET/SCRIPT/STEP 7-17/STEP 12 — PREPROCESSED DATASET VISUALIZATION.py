# ============================================================
# STEP 12 — PREPROCESSED DATASET VISUALIZATION
# ============================================================

import matplotlib.pyplot as plt
import matplotlib.patches as patches

print("=" * 70)
print("STEP 12 — DATASET VISUALIZATION")
print("=" * 70)

rng = random.Random(SEED)

sample_index = rng.randrange(
    len(train_dataset)
)

image_tensor, target = train_dataset[
    sample_index
]

image_np = denormalize_image(
    image_tensor
)

fig, ax = plt.subplots(
    figsize=(10, 10)
)

ax.imshow(image_np)

boxes = target["boxes"].numpy()
labels = target["labels"].numpy()

for box, label in zip(
    boxes,
    labels
):

    x1, y1, x2, y2 = box

    width = x2 - x1
    height = y2 - y1

    rect = patches.Rectangle(
        (x1, y1),
        width,
        height,
        linewidth=2,
        edgecolor="red",
        facecolor="none"
    )

    ax.add_patch(rect)

    class_name = CONTIGUOUS_TO_NAME[
        int(label)
    ]

    ax.text(
        x1,
        max(0, y1 - 3),
        class_name,
        fontsize=9,
        bbox=dict(
            facecolor="white",
            alpha=0.7
        )
    )

ax.set_title(
    f"Preprocessed COCO Sample — "
    f"{len(boxes)} objects"
)

ax.axis("off")
plt.show()

print("Tensor shape :", image_tensor.shape)
print("Boxes        :", target["boxes"].shape)
print("Labels       :", target["labels"].shape)
print("Image ID     :", target["image_id"].item())

assert image_tensor.shape == (
    3,
    CONFIG["input_size"],
    CONFIG["input_size"]
)

print("\nSTEP 12 PASSED")
