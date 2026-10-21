
# ============================================================
# STEP 35 — QC2: GAUSSIAN RADIUS V1.2 VS V1.3
# B1 COMPATIBLE / DATA AUDIT ONLY / NO TRAINING
# ============================================================

import os
import json
import math
import numpy as np

# ============================================================
# 1. VERIFY B1 CONFIG
# ============================================================

assert CONFIG["experiment_name"] == (
    "CenterNet_COCO80_B1_V13_ON_A2_FRESH_160K_BS8"
)

assert CONFIG["gaussian_radius_mode"] == "v1_3_corrected"
assert CONFIG["gaussian_min_overlap"] == 0.7
assert CONFIG["wh_loss_type"] == "hybrid_abs_relative"

assert len(val_dataset) == 5000
assert not val_dataset.augment

print("=" * 70)
print("STEP 35 — QC2 GAUSSIAN V1.2 VS V1.3")
print("Active mode:", CONFIG["gaussian_radius_mode"])
print("=" * 70)


# ============================================================
# 2. INDEPENDENT GAUSSIAN RADIUS CALCULATION
# ============================================================

def q2_radius(h, w, version, overlap=0.7):

    h = float(h)
    w = float(w)
    o = float(overlap)

    assert h > 0 and w > 0
    assert 0 < o < 1

    a1 = 1.0
    b1 = h + w
    c1 = h*w*(1-o)/(1+o)

    a2 = 4.0
    b2 = 2*(h+w)
    c2 = (1-o)*h*w

    a3 = 4*o
    b3 = -2*o*(h+w)
    c3 = (o-1)*h*w

    d1 = math.sqrt(max(0., b1*b1-4*a1*c1))
    d2 = math.sqrt(max(0., b2*b2-4*a2*c2))
    d3 = math.sqrt(max(0., b3*b3-4*a3*c3))

    if version == "v12":

        r1 = (b1+d1)/(2*a1)
        r2 = (b2+d2)/(2*a2)
        mode = "v1_2_original"

    elif version == "v13":

        r1 = (b1-d1)/(2*a1)
        r2 = (b2-d2)/(2*a2)
        mode = "v1_3_corrected"

    else:
        raise ValueError(version)

    r3 = (b3+d3)/(2*a3)

    radius = max(0, int(min(r1, r2, r3)))

    # Cross-check against Step 15 implementation
    actual = int(gaussian_radius(
        h,
        w,
        min_overlap=o,
        mode=mode
    ))

    assert radius == actual, (
        f"Step 15 radius mismatch: "
        f"{version}, QC2={radius}, Step15={actual}"
    )

    return radius


assert q2_radius(12, 12, "v12") == 1
assert q2_radius(12, 12, "v13") == 0


# ============================================================
# 3. AUDIT FULL COCO-VAL
# ============================================================

coco35 = val_dataset.coco

r35 = []
overflow35 = 0
invalid35 = 0
images35 = 0

INPUT35 = CONFIG["input_size"]
STRIDE35 = CONFIG["output_stride"]
MAX35 = CONFIG["max_objects"]

assert INPUT35 == 640
assert STRIDE35 == 4
assert MAX35 == 128

for image_id in val_dataset.image_ids:

    image_id = int(image_id)
    meta = coco35.imgs[image_id]

    original_w = meta["width"]
    original_h = meta["height"]

    scale = min(
        INPUT35 / original_w,
        INPUT35 / original_h
    )

    resized_w = int(round(original_w * scale))
    resized_h = int(round(original_h * scale))

    px = (INPUT35 - resized_w) // 2
    py = (INPUT35 - resized_h) // 2

    accepted = 0

    ann_ids = coco35.getAnnIds(
        imgIds=[image_id],
        iscrowd=False
    )

    annotations = coco35.loadAnns(ann_ids)

    for ann in annotations:

        x, y, w, h = ann["bbox"]

        if w <= 1 or h <= 1:
            invalid35 += 1
            continue

        if ann["category_id"] not in (
            val_dataset.cat_id_to_contiguous
        ):
            invalid35 += 1
            continue

        x1 = max(0., min(639., x*scale+px))
        x2 = max(0., min(639., (x+w)*scale+px))

        y1 = max(0., min(639., y*scale+py))
        y2 = max(0., min(639., (y+h)*scale+py))

        W = x2-x1
        H = y2-y1

        if W <= 1 or H <= 1:
            invalid35 += 1
            continue

        cx = (x1+x2) / (2*STRIDE35)
        cy = (y1+y2) / (2*STRIDE35)

        if not (
            0 <= cx < 160
            and 0 <= cy < 160
        ):
            invalid35 += 1
            continue

        if accepted >= MAX35:
            overflow35 += 1
            continue

        accepted += 1

        # Original COCO area categories
        area = ann.get("area", w*h)

        if area < 32**2:
            group = "small"
        elif area < 96**2:
            group = "medium"
        else:
            group = "large"

        r12 = q2_radius(
            H/STRIDE35,
            W/STRIDE35,
            "v12",
            CONFIG["gaussian_min_overlap"]
        )

        r13 = q2_radius(
            H/STRIDE35,
            W/STRIDE35,
            "v13",
            CONFIG["gaussian_min_overlap"]
        )

        assert r13 <= r12

        r35.append((r12, r13, group))

    images35 += 1


# ============================================================
# 4. STATISTICS
# ============================================================

assert images35 == 5000
assert len(r35) > 0

result35 = {
    "experiment_name": CONFIG["experiment_name"],
    "active_radius_mode": CONFIG["gaussian_radius_mode"],
    "images": images35,
    "n_objects": len(r35),
    "overflow_excluded": overflow35,
    "invalid_excluded": invalid35,
    "groups": {}
}

for group in ("all", "small", "medium", "large"):

    sample = [
        (a, b)
        for a, b, size_group in r35
        if group == "all" or size_group == group
    ]

    if not sample:
        continue

    values = np.asarray(sample, dtype=np.int32)

    v12 = values[:, 0]
    v13 = values[:, 1]

    stats = {
        "n": len(sample),
        "v12_radius0_pct":
            float(100*np.mean(v12 == 0)),
        "v13_radius0_pct":
            float(100*np.mean(v13 == 0)),
        "changed_pct":
            float(100*np.mean(v12 != v13)),
        "v12_mean_radius":
            float(np.mean(v12)),
        "v13_mean_radius":
            float(np.mean(v13))
    }

    result35["groups"][group] = stats

    print(
        f"{group:7s} | "
        f"GT={stats['n']:6d} | "
        f"changed={stats['changed_pct']:.2f}% | "
        f"radius {stats['v12_mean_radius']:.3f}"
        f" -> {stats['v13_mean_radius']:.3f} | "
        f"V1.3 R0={stats['v13_radius0_pct']:.2f}%"
    )


# ============================================================
# 5. SAVE — SEPARATE B1 AUDIT FILE
# ============================================================

QC35_OUT = os.path.join(
    FULL_CHECKPOINT_DIR,
    "quality_checks_B1"
)

os.makedirs(QC35_OUT, exist_ok=True)

QC35_FILE = os.path.join(
    QC35_OUT,
    "qc2_gaussian_v12_vs_v13_data_audit.json"
)

with open(QC35_FILE, "w", encoding="utf-8") as f:
    json.dump(
        result35,
        f,
        indent=2,
        allow_nan=False
    )

print("=" * 70)
print("STEP 35 PASSED")
print("Images:", images35)
print("Objects:", len(r35))
print("Overflow excluded:", overflow35)
print("Invalid excluded:", invalid35)
print("Saved:", QC35_FILE)
print("NO TRAINING / NO MODEL WEIGHT UPDATE")
