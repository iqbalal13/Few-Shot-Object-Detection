# ============================================================
# STEP 17 — CENTERNET TARGET / HEATMAP SANITY CHECK
# ============================================================

import torch.nn.functional as F
import matplotlib.pyplot as plt

print("=" * 70)
print("STEP 17 — TARGET / HEATMAP SANITY CHECK")
print("=" * 70)

# Cari contoh yang punya object
sample_index = None

for idx in range(
    min(100, len(train_dataset))
):

    img, tgt = train_dataset[idx]

    if len(tgt["boxes"]) > 0:
        sample_index = idx
        break

assert sample_index is not None

image_tensor, target = train_dataset[
    sample_index
]

centernet_target = build_centernet_target(
    target
)

image_np = denormalize_image(
    image_tensor
)

heatmap = centernet_target[
    "heatmap"
]

# Max response seluruh 80 kelas
combined_heatmap = heatmap.max(
    dim=0
).values

combined_heatmap_up = F.interpolate(
    combined_heatmap[
        None,
        None
    ],
    size=(
        CONFIG["input_size"],
        CONFIG["input_size"]
    ),
    mode="bilinear",
    align_corners=False
)[0, 0]

person_index = CONFIG[
    "person_class_index"
]

person_heatmap = heatmap[
    person_index
]

person_heatmap_up = F.interpolate(
    person_heatmap[
        None,
        None
    ],
    size=(
        CONFIG["input_size"],
        CONFIG["input_size"]
    ),
    mode="bilinear",
    align_corners=False
)[0, 0]


fig, axes = plt.subplots(
    1,
    3,
    figsize=(20, 7)
)

# ------------------------------------------------------------
# Original preprocessed image
# ------------------------------------------------------------

axes[0].imshow(
    image_np
)

for box, label in zip(
    target["boxes"],
    target["labels"]
):

    x1, y1, x2, y2 = box.tolist()

    rect = patches.Rectangle(
        (x1, y1),
        x2 - x1,
        y2 - y1,
        linewidth=2,
        edgecolor="red",
        facecolor="none"
    )

    axes[0].add_patch(rect)

    axes[0].text(
        x1,
        max(0, y1 - 3),
        CONTIGUOUS_TO_NAME[
            int(label)
        ],
        fontsize=8,
        bbox=dict(
            facecolor="white",
            alpha=0.7
        )
    )

axes[0].set_title(
    "Preprocessed Image + GT Boxes"
)

axes[0].axis("off")


# ------------------------------------------------------------
# All-class center heatmap
# ------------------------------------------------------------

axes[1].imshow(
    image_np
)

axes[1].imshow(
    combined_heatmap_up.numpy(),
    alpha=0.55,
    cmap="jet"
)

axes[1].set_title(
    "CenterNet Target Heatmap — All Classes"
)

axes[1].axis("off")


# ------------------------------------------------------------
# Person-only heatmap
# ------------------------------------------------------------

axes[2].imshow(
    image_np
)

axes[2].imshow(
    person_heatmap_up.numpy(),
    alpha=0.55,
    cmap="jet"
)

axes[2].set_title(
    "Person Channel Heatmap"
)

axes[2].axis("off")

plt.tight_layout()
plt.show()


active_objects = int(
    centernet_target[
        "mask"
    ].sum().item()
)

heatmap_peaks = int(
    (
        heatmap == 1.0
    ).sum().item()
)

print(
    "GT boxes             :",
    len(target["boxes"])
)

print(
    "Encoded objects      :",
    active_objects
)

print(
    "Exact heatmap peaks  :",
    heatmap_peaks
)

print(
    "Heatmap range        :",
    float(heatmap.min()),
    "→",
    float(heatmap.max())
)

assert active_objects > 0
assert heatmap.max() <= 1.0
assert heatmap.min() >= 0.0

print("\nSTEP 17 PASSED")
