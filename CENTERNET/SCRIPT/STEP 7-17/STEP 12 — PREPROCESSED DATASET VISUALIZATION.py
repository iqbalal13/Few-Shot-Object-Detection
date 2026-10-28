
# ============================================================
# STEP 12 — TRAIN AUGMENTATION VISUAL SANITY
# ============================================================

import random
import numpy as np
import torch

import matplotlib.pyplot as plt
import matplotlib.patches as patches

index = random.Random(SEED).randrange(
    len(train_dataset)
)

img, target = train_dataset[index]

assert torch.isfinite(img).all()
assert target["boxes"].shape[1] == 4

assert (
    len(target["boxes"])
    == len(target["labels"])
)

if len(target["boxes"]):
    assert torch.all(
        target["boxes"] >= 0
    )

    assert torch.all(
        target["boxes"] <= CONFIG["input_size"] - 1
    )

    assert torch.all(
        target["boxes"][:, 2] > target["boxes"][:, 0]
    )

    assert torch.all(
        target["boxes"][:, 3] > target["boxes"][:, 1]
    )

fig, ax = plt.subplots(
    figsize=(8, 8)
)

ax.imshow(
    denormalize_image(img)
)

for b, lbl in zip(
    target["boxes"],
    target["labels"]
):
    x1, y1, x2, y2 = b.tolist()

    ax.add_patch(
        patches.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            fill=False,
            edgecolor="red",
            linewidth=1.5
        )
    )

ax.set_title(
    "A3-v2 train augmentation — "
    "jointly transformed GT"
)

ax.axis("off")
plt.show()

print(
    "STEP 12 PASS, image",
    int(target["image_id"]),
    "GT",
    len(target["boxes"])
)
