
# ============================================================
# STEP 35 QUALITY CHECK 2/4
# GAUSSIAN RADIUS AUDIT — V1.2 VS V1.3
# NO TRAINING / NO WEIGHT MODIFICATION
# ============================================================

import math
import json
import os
from collections import Counter

import numpy as np

assert len(val_dataset) == 5000


# ============================================================
# 1. ORIGINAL V1.2 FORMULA
# ============================================================

def q2_v12_radius(h, w, o=0.7):
    h, w = float(h), float(w)

    a1 = 1.0
    b1 = h + w
    c1 = w * h * (1 - o) / (1 + o)

    a2 = 4.0
    b2 = 2 * (h + w)
    c2 = (1 - o) * w * h

    a3 = 4 * o
    b3 = -2 * o * (h + w)
    c3 = (o - 1) * w * h

    r1 = (
        b1 + math.sqrt(max(0, b1*b1 - 4*a1*c1))
    ) / 2

    r2 = (
        b2 + math.sqrt(max(0, b2*b2 - 4*a2*c2))
    ) / 2

    r3 = (
        b3 + math.sqrt(max(0, b3*b3 - 4*a3*c3))
    ) / (2 * a3)

    return max(0.0, min(r1, r2, r3))


# ============================================================
# 2. V1.3 FORMULA
# ============================================================

def q2_v13_radius(h, w, o=0.7):
    h, w = float(h), float(w)

    a1 = 1.0
    b1 = h + w
    c1 = w * h * (1 - o) / (1 + o)

    a2 = 4.0
    b2 = 2 * (h + w)
    c2 = (1 - o) * w * h

    a3 = 4 * o
    b3 = -2 * o * (h + w)
    c3 = (o - 1) * w * h

    r1 = (
        b1 - math.sqrt(max(0, b1*b1 - 4*a1*c1))
    ) / (2 * a1)

    r2 = (
        b2 - math.sqrt(max(0, b2*b2 - 4*a2*c2))
    ) / (2 * a2)

    r3 = (
        b3 + math.sqrt(max(0, b3*b3 - 4*a3*c3))
    ) / (2 * a3)

    return max(0.0, min(r1, r2, r3))


# ============================================================
# 3. SCAN COCO-VAL ANNOTATIONS
# ============================================================

q2_coco = val_dataset.coco
q2_rows = []
q2_overflow = 0

for q2_id in val_dataset.image_ids:

    im = q2_coco.imgs[int(q2_id)]

    scale = min(
        640 / im["width"],
        640 / im["height"]
    )

    pad_x = (
        640 - round(im["width"] * scale)
    ) // 2

    pad_y = (
        640 - round(im["height"] * scale)
    ) // 2

    ann_ids = q2_coco.getAnnIds(
        imgIds=[int(q2_id)],
        iscrowd=False
    )

    annotations = q2_coco.loadAnns(ann_ids)

    accepted = 0

    for ann in annotations:

        x, y, w, h = ann["bbox"]

        if w <= 1 or h <= 1:
            continue

        if ann["category_id"] not in (
            val_dataset.cat_id_to_contiguous
        ):
            continue

        x1 = max(
            0.0,
            min(639.0, x * scale + pad_x)
        )

        y1 = max(
            0.0,
            min(639.0, y * scale + pad_y)
        )

        x2 = max(
            0.0,
            min(639.0, (x+w)*scale + pad_x)
        )

        y2 = max(
            0.0,
            min(639.0, (y+h)*scale + pad_y)
        )

        W = x2 - x1
        H = y2 - y1

        if W <= 1 or H <= 1:
            continue

        cx = (x1 + x2) / 8.0
        cy = (y1 + y2) / 8.0

        if not (
            0 <= cx < 160
            and 0 <= cy < 160
        ):
            continue

        if accepted >= 128:
            q2_overflow += 1
            continue

        accepted += 1

        # Bounding box dimensions in feature-map cells
        feature_w = W / 4.0
        feature_h = H / 4.0

        radius12 = int(
            q2_v12_radius(feature_h, feature_w)
        )

        radius13 = int(
            q2_v13_radius(feature_h, feature_w)
        )

        area = float(
            ann.get("area", w*h)
        )

        if area < 32**2:
            size = "small"
        elif area < 96**2:
            size = "medium"
        else:
            size = "large"

        q2_rows.append(
            (radius12, radius13, size)
        )


# ============================================================
# 4. PRINT DISTRIBUTION
# ============================================================

assert q2_rows

print("=" * 75)
print("QUALITY CHECK 2 — GAUSSIAN RADIUS DISTRIBUTION")
print("=" * 75)

print("Objects evaluated:", len(q2_rows))
print("Overflow excluded:", q2_overflow)

for size in [
    "all", "small", "medium", "large"
]:

    rows = [
        r for r in q2_rows
        if size == "all" or r[2] == size
    ]

    if not rows:
        continue

    values = np.array(
        [[r[0], r[1]] for r in rows],
        dtype=np.float64
    )

    v12 = values[:, 0]
    v13 = values[:, 1]

    print(f"\nObject size: {size}")
    print("Count:", len(rows))

    print(
        "V1.2 radius zero: "
        f"{100 * np.mean(v12 == 0):.3f}%"
    )

    print(
        "V1.3 radius zero: "
        f"{100 * np.mean(v13 == 0):.3f}%"
    )

    print(
        "Changed radius: "
        f"{100 * np.mean(v12 != v13):.3f}%"
    )

    print(
        f"Mean radius V1.2: {v12.mean():.3f}"
    )

    print(
        f"Mean radius V1.3: {v13.mean():.3f}"
    )


# ============================================================
# 5. SAVE DIAGNOSTIC
# ============================================================

q2_save_dir = (
    "/content/drive/MyDrive/"
    "CenterNet_Quality_Diagnostics"
)

os.makedirs(q2_save_dir, exist_ok=True)

q2_result = {
    "objects": len(q2_rows),
    "overflow_excluded": q2_overflow,
    "radius_v12": dict(
        Counter(r[0] for r in q2_rows)
    ),
    "radius_v13": dict(
        Counter(r[1] for r in q2_rows)
    ),
    "changed_pct": float(
        100 * sum(
            a != b for a, b, _ in q2_rows
        ) / len(q2_rows)
    )
}

q2_save_file = os.path.join(
    q2_save_dir,
    "quality_check_2_gaussian.json"
)

with open(
    q2_save_file,
    "w",
    encoding="utf-8"
) as f:
    json.dump(q2_result, f, indent=2)

print("\nQUALITY CHECK 2 COMPLETE")
print("Results:", q2_save_file)
print("No model weights modified.")
