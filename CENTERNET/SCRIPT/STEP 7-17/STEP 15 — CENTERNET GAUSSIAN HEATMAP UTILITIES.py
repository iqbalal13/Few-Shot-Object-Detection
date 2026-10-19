
# ============================================================
# STEP 15 — CENTERNET GAUSSIAN HEATMAP
# B1: V1.3 CORRECTED RADIUS
# SUPPORTS V1.2 FOR CONTROL TESTS
# NO TRAINING
# ============================================================

import math
import torch
from functools import lru_cache


def gaussian_radius(
    height,
    width,
    min_overlap=0.7,
    mode=None
):
    if mode is None:
        mode = CONFIG["gaussian_radius_mode"]

    h = float(height)
    w = float(width)
    o = float(min_overlap)

    assert h > 0 and w > 0
    assert 0 < o < 1

    # Case 1
    a1 = 1.0
    b1 = h + w
    c1 = h * w * (1.0 - o) / (1.0 + o)

    d1 = math.sqrt(
        max(0.0, b1 * b1 - 4.0 * a1 * c1)
    )

    # Case 2
    a2 = 4.0
    b2 = 2.0 * (h + w)
    c2 = (1.0 - o) * h * w

    d2 = math.sqrt(
        max(0.0, b2 * b2 - 4.0 * a2 * c2)
    )

    # Case 3
    a3 = 4.0 * o
    b3 = -2.0 * o * (h + w)
    c3 = (o - 1.0) * h * w

    d3 = math.sqrt(
        max(0.0, b3 * b3 - 4.0 * a3 * c3)
    )

    if mode == "v1_2_original":
        r1 = (b1 + d1) / (2.0 * a1)
        r2 = (b2 + d2) / (2.0 * a2)

    elif mode == "v1_3_corrected":
        r1 = (b1 - d1) / (2.0 * a1)
        r2 = (b2 - d2) / (2.0 * a2)

    else:
        raise ValueError(
            f"Unsupported Gaussian mode: {mode}"
        )

    r3 = (b3 + d3) / (2.0 * a3)

    return max(0.0, min(r1, r2, r3))


def gaussian2d(shape, sigma=1.0):
    h, w = shape

    y = (
        torch.arange(h, dtype=torch.float32)
        - (h - 1) / 2.0
    )

    x = (
        torch.arange(w, dtype=torch.float32)
        - (w - 1) / 2.0
    )

    yy, xx = torch.meshgrid(
        y, x, indexing="ij"
    )

    return torch.exp(
        -(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2)
    )


@lru_cache(maxsize=256)
def _gaussian_for_radius(radius):
    diameter = 2 * int(radius) + 1

    return gaussian2d(
        (diameter, diameter),
        sigma=diameter / 6.0
    )


def draw_gaussian(heatmap, center, radius):
    radius = int(radius)

    x = int(center[0])
    y = int(center[1])

    height, width = heatmap.shape

    left = min(x, radius)
    right = min(width - x - 1, radius)
    top = min(y, radius)
    bottom = min(height - y - 1, radius)

    if min(left, right, top, bottom) < 0:
        return heatmap

    gaussian = _gaussian_for_radius(radius)

    region = heatmap[
        y - top:y + bottom + 1,
        x - left:x + right + 1
    ]

    gaussian_region = gaussian[
        radius - top:radius + bottom + 1,
        radius - left:radius + right + 1
    ]

    torch.maximum(
        region,
        gaussian_region,
        out=region
    )

    return heatmap


# ============================================================
# B1 FORMULA SANITY TEST
# ============================================================

assert CONFIG["gaussian_radius_mode"] == "v1_3_corrected"

sizes = (4, 8, 12, 20, 40, 80)

radius_table = {}

for n in sizes:
    r12 = int(gaussian_radius(
        n, n,
        mode="v1_2_original"
    ))

    r13 = int(gaussian_radius(
        n, n,
        mode="v1_3_corrected"
    ))

    assert r13 <= r12

    radius_table[n] = {
        "v12": r12,
        "v13": r13
    }

test_heatmap = torch.zeros(
    (160, 160),
    dtype=torch.float32
)

draw_gaussian(
    test_heatmap,
    center=(80, 80),
    radius=5
)

assert test_heatmap.max().item() == 1.0
assert test_heatmap.min().item() >= 0.0
assert torch.isfinite(test_heatmap).all()

# Confirm the active default is V1.3.
for n in sizes:
    assert int(gaussian_radius(n, n)) == (
        radius_table[n]["v13"]
    )

print("=" * 70)
print("STEP 15 — B1 GAUSSIAN V1.3")
print("=" * 70)

for n, values in radius_table.items():
    print(
        f"Size {n:3d}: "
        f"V1.2={values['v12']:3d} "
        f"V1.3={values['v13']:3d}"
    )

print("Active mode:", CONFIG["gaussian_radius_mode"])
print("STEP 15 PASSED")
