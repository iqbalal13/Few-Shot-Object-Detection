
# ============================================================
# STEP 15 — CENTERNET GAUSSIAN HEATMAP UTILITIES
# CORRECTED QUADRATIC ROOT + CACHED GAUSSIAN
# EXPERIMENTAL CHANGE: Requires source retraining, no retroactive effect.
# ============================================================

import math
import torch
from functools import lru_cache

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
        b1 - sq1
    ) / (2 * a1)

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
        b2 - sq2
    ) / (2 * a2)

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
    ) / (2 * a3)

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
