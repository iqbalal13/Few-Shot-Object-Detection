# ============================================================
# STEP 15 — CENTERNET GAUSSIAN HEATMAP UTILITIES
# ============================================================

import math
import torch

print("=" * 70)
print("STEP 15 — GAUSSIAN HEATMAP UTILITIES")
print("=" * 70)


def gaussian_radius(
    height,
    width,
    min_overlap=0.7
):

    height = float(height)
    width = float(width)

    a1 = 1
    b1 = height + width
    c1 = (
        width * height
        * (1 - min_overlap)
        / (1 + min_overlap)
    )

    sq1 = math.sqrt(
        max(0.0, b1 ** 2 - 4 * a1 * c1)
    )

    r1 = (
        b1 + sq1
    ) / 2


    a2 = 4
    b2 = 2 * (
        height + width
    )

    c2 = (
        (1 - min_overlap)
        * width
        * height
    )

    sq2 = math.sqrt(
        max(0.0, b2 ** 2 - 4 * a2 * c2)
    )

    r2 = (
        b2 + sq2
    ) / 2


    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (
        height + width
    )

    c3 = (
        min_overlap - 1
    ) * width * height

    sq3 = math.sqrt(
        max(0.0, b3 ** 2 - 4 * a3 * c3)
    )

    r3 = (
        b3 + sq3
    ) / (
        2 * a3
    )

    return max(
        0.0,
        min(r1, r2, r3)
    )


def gaussian2d(
    shape,
    sigma=1.0
):

    height, width = shape

    y = torch.arange(
        height,
        dtype=torch.float32
    ) - (height - 1) / 2

    x = torch.arange(
        width,
        dtype=torch.float32
    ) - (width - 1) / 2

    yy, xx = torch.meshgrid(
        y,
        x,
        indexing="ij"
    )

    gaussian = torch.exp(
        -(xx ** 2 + yy ** 2)
        / (2 * sigma ** 2)
    )

    return gaussian


def draw_gaussian(
    heatmap,
    center,
    radius
):

    diameter = (
        2 * radius + 1
    )

    gaussian = gaussian2d(
        (diameter, diameter),
        sigma=diameter / 6
    )

    x, y = (
        int(center[0]),
        int(center[1])
    )

    height, width = heatmap.shape

    left = min(
        x,
        radius
    )

    right = min(
        width - x - 1,
        radius
    )

    top = min(
        y,
        radius
    )

    bottom = min(
        height - y - 1,
        radius
    )

    if (
        left < 0
        or right < 0
        or top < 0
        or bottom < 0
    ):
        return heatmap

    masked_heatmap = heatmap[
        y - top:y + bottom + 1,
        x - left:x + right + 1
    ]

    masked_gaussian = gaussian[
        radius - top:
        radius + bottom + 1,

        radius - left:
        radius + right + 1
    ]

    torch.maximum(
        masked_heatmap,
        masked_gaussian,
        out=masked_heatmap
    )

    return heatmap


# Sanity test
test_heatmap = torch.zeros(
    (160, 160),
    dtype=torch.float32
)

test_heatmap = draw_gaussian(
    test_heatmap,
    center=(80, 80),
    radius=5
)

print("Heatmap shape :", test_heatmap.shape)
print("Heatmap max   :", test_heatmap.max().item())
print("Heatmap min   :", test_heatmap.min().item())

assert test_heatmap.max().item() == 1.0
assert test_heatmap.min().item() >= 0.0

print("\nSTEP 15 PASSED")
