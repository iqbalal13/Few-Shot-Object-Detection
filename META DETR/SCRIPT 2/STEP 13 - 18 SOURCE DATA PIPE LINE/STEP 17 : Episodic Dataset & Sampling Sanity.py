# ==========================================================
# STEP 17: Person Dataset Sanity and Visual Check
# ==========================================================

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


train_dataset.set_epoch(0)

for i in range(10):
    episode = train_dataset[i]
    target = episode["query_target"]

    assert episode["episode_class"].item() == 0

    assert (
        target["labels"] == 0
    ).all()

    assert len(target["boxes"]) > 0

    assert (
        episode["support_target"]["image_id"].item()
        != target["image_id"].item()
    )

    boxes = target["boxes"]

    assert torch.isfinite(boxes).all()

    assert (
        boxes[:, 2:] > 0
    ).all()

    assert (
        boxes[:, :2] - boxes[:, 2:] / 2 >= -1e-6
    ).all()

    assert (
        boxes[:, :2] + boxes[:, 2:] / 2 <= 1 + 1e-6
    ).all()


def display_tensor_image(tensor):
    array = (
        tensor
        .detach()
        .cpu()
        .permute(1, 2, 0)
        .numpy()
    )

    array = (
        array * np.array(IMAGENET_STD)
        + np.array(IMAGENET_MEAN)
    )

    return np.clip(array, 0, 1)


episode = train_dataset[0]

fig, axes = plt.subplots(
    1,
    2,
    figsize=(12, 6)
)

axes[0].imshow(
    display_tensor_image(
        episode["support_image"]
    )
)

axes[0].set_title(
    "Support crop: person"
)

axes[1].imshow(
    display_tensor_image(
        episode["query_image"]
    )
)

height, width = episode[
    "query_image"
].shape[-2:]

for cx, cy, w, h in episode[
    "query_target"
]["boxes"].tolist():
    axes[1].add_patch(
        Rectangle(
            (
                (cx - w / 2) * width,
                (cy - h / 2) * height
            ),
            w * width,
            h * height,
            fill=False,
            edgecolor="lime",
            linewidth=2,
        )
    )

axes[1].set_title(
    "Different query image: all eligible person GT"
)

for axis in axes:
    axis.axis("off")

plt.tight_layout()
plt.show()
plt.close(fig)

print("STEP 17 PASS: person-only targets.")
print("Inspect displayed boxes before training.")
