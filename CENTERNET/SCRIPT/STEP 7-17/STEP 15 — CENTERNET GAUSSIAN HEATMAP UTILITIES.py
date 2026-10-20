
# ============================================================
# STEP 15 — B1 GAUSSIAN HEATMAP
# V1.2 / V1.3 SELECTABLE RADIUS
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

    # Case 2
    a2 = 4.0
    b2 = 2.0 * (h + w)
    c2 = (1.0 - o) * w * h

    # Case 3
    a3 = 4.0 * o
    b3 = -2.0 * o * (h + w)
    c3 = (o - 1.0) * w * h

    d1 = math.sqrt(
        max(0.0, b1*b1 - 4*a1*c1)
    )
    d2 = math.sqrt(
        max(0.0, b2*b2 - 4*a2*c2)
    )
    d3 = math.sqrt(
        max(0.0, b3*b3 - 4*a3*c3)
    )

    if mode == "v1_2_original":
        r1 = (b1 + d1) / (2.0*a1)
        r2 = (b2 + d2) / (2.0*a2)

    elif mode == "v1_3_corrected":
        r1 = (b1 - d1) / (2.0*a1)
        r2 = (b2 - d2) / (2.0*a2)

    else:
        raise ValueError(
            f"Unsupported gaussian mode: {mode}"
        )

    r3 = (b3 + d3) / (2.0*a3)

    return max(0.0, min(r1, r2, r3))


def gaussian2d(shape, sigma=1.0):
    h, w = shape

    y = torch.arange(
        h, dtype=torch.float32
    ) - (h - 1)/2

    x = torch.arange(
        w, dtype=torch.float32
    ) - (w - 1)/2

    yy, xx = torch.meshgrid(
        y, x, indexing="ij"
    )

    return torch.exp(
        -(xx**2 + yy**2)/(2*sigma**2)
    )


@lru_cache(maxsize=256)
def _gaussian_for_radius(radius):
    diameter = 2*int(radius) + 1

    return gaussian2d(
        (diameter, diameter),
        sigma=diameter/6
    )


def draw_gaussian(heatmap, center, radius):
    radius = int(radius)

    x, y = int(center[0]), int(center[1])
    height, width = heatmap.shape

    left = min(x, radius)
    right = min(width-x-1, radius)
    top = min(y, radius)
    bottom = min(height-y-1, radius)

    if min(left, right, top, bottom) < 0:
        return heatmap

    gaussian = _gaussian_for_radius(radius)

    region = heatmap[
        y-top:y+bottom+1,
        x-left:x+right+1
    ]

    gaussian_region = gaussian[
        radius-top:radius+bottom+1,
        radius-left:radius+right+1
    ]

    torch.maximum(
        region,
        gaussian_region,
        out=region
    )

    return heatmap


# ============================================================
# RADIUS SANITY TEST
# ============================================================

assert CONFIG["gaussian_radius_mode"] == "v1_3_corrected"

test_heatmap = torch.zeros(
    (160, 160), dtype=torch.float32
)

draw_gaussian(
    test_heatmap,
    center=(80, 80),
    radius=5
)

assert test_heatmap.max().item() == 1.0
assert test_heatmap.min().item() >= 0.0

assert gaussian_radius(20, 20) > 0
assert gaussian_radius(40, 40) > gaussian_radius(20, 20)

for size in (4, 8, 12, 20, 40, 80):
    r12 = int(gaussian_radius(
        size, size, mode="v1_2_original"
    ))
    r13 = int(gaussian_radius(
        size, size, mode="v1_3_corrected"
    ))

    assert r13 <= r12

    print(
        f"Size {size:3d}x{size:3d} "
        f"| V1.2={r12:2d} "
        f"| V1.3={r13:2d}"
    )

assert int(gaussian_radius(
    12, 12, mode="v1_2_original"
)) == 1

assert int(gaussian_radius(
    12, 12, mode="v1_3_corrected"
)) == 0

print("STEP 15 PASSED — B1 GAUSSIAN V1.3")
