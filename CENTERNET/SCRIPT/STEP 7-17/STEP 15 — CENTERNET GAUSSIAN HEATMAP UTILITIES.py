# ============================================================
# STEP 15 — CENTERNET GAUSSIAN HEATMAP
# ORIGINAL V1.2 RADIUS
# ============================================================

import math
import torch
from functools import lru_cache


def gaussian_radius(height, width, min_overlap=0.7):
    h = float(height)
    w = float(width)
    o = float(min_overlap)

    a1 = 1.0
    b1 = h + w
    c1 = h * w * (1 - o) / (1 + o)

    r1 = (
        b1 + math.sqrt(
            max(0.0, b1*b1 - 4*a1*c1)
        )
    ) / (2*a1)

    a2 = 4.0
    b2 = 2*(h + w)
    c2 = (1 - o)*w*h

    r2 = (
        b2 + math.sqrt(
            max(0.0, b2*b2 - 4*a2*c2)
        )
    ) / (2*a2)

    a3 = 4*o
    b3 = -2*o*(h + w)
    c3 = (o - 1)*w*h

    r3 = (
        b3 + math.sqrt(
            max(0.0, b3*b3 - 4*a3*c3)
        )
    ) / (2*a3)

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
        region, gaussian_region, out=region
    )

    return heatmap


test_heatmap = torch.zeros(
    (160, 160), dtype=torch.float32
)

draw_gaussian(
    test_heatmap, center=(80, 80), radius=5
)

assert test_heatmap.max().item() == 1.0
assert test_heatmap.min().item() >= 0.0

assert gaussian_radius(20, 20) > 0
assert gaussian_radius(40, 40) > gaussian_radius(20, 20)

print("STEP 15 PASSED")
print("Gaussian:", CONFIG["gaussian_radius_mode"])
print("Radii:", {
    n: int(gaussian_radius(n, n))
    for n in (4, 8, 12, 20, 40, 80)
})
