
# ============================================================
# STEP 33 — A3-v2 QC SETUP AND CHECKPOINT INTEGRITY
# A3 176K vs A3-v2 SELECTED
# NO TRAINING
# ============================================================

import os
import json
import gc
import numpy as np
import torch

from torch.utils.data import DataLoader, Subset

assert "results31" in globals()
assert "SELECTED_A3_V2" in globals()

assert "A3_176K_reference" in results31
assert "A3_v2_selected" in results31, (
    "A3-v2 selected checkpoint is not available."
)

assert os.path.isfile(A3_CKPT)
assert os.path.isfile(SELECTED_A3_V2)

assert os.path.abspath(A3_CKPT) != os.path.abspath(
    SELECTED_A3_V2
)

assert not val_dataset.augment
assert CONFIG["a3_neighbor_lambda"] == 0.25
assert CONFIG["gaussian_radius_mode"] == "v1_2_original"

QC_REFERENCE = "A3_reference"
QC_CANDIDATE = "A3_v2"

QC_PATHS = {
    QC_REFERENCE: A3_CKPT,
    QC_CANDIDATE: SELECTED_A3_V2
}

QC_DIR = os.path.join(
    V2_ROOT,
    "quality_checks_A3_v2"
)

os.makedirs(QC_DIR, exist_ok=True)

QC_N = min(1000, len(val_dataset))

QC_GROUPS = (
    "all", "small", "medium", "large", "person"
)

QC_DIRECTIONS = [
    (dx, dy)
    for dy in (-1, 0, 1)
    for dx in (-1, 0, 1)
    if dx != 0 or dy != 0
]


def qc_save(filename, obj):
    path = os.path.join(QC_DIR, filename)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            obj,
            f,
            indent=2,
            allow_nan=False,
            default=lambda x: (
                x.item()
                if isinstance(x, np.generic)
                else str(x)
            )
        )

    return path


def qc_loader():
    return DataLoader(
        Subset(val_dataset, range(QC_N)),
        batch_size=8,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        collate_fn=centernet_collate_fn
    )


def qc_model(arm):
    ck = torch.load(
        QC_PATHS[arm],
        map_location="cpu",
        weights_only=False
    )

    expected = (
        "A3_neighbor_R1"
        if arm == QC_REFERENCE
        else "A3_v2"
    )

    assert ck["arm"] == expected
    assert ck["gaussian_radius_mode"] == "v1_2_original"

    assert abs(
        float(ck["neighbor_lambda"]) - 0.25
    ) < 1e-12

    net = CenterNetResNet101(
        num_classes=CONFIG["num_classes"],
        pretrained_backbone=False
    ).to(device)

    net.load_state_dict(
        ck["model_state_dict"],
        strict=True
    )

    assert not any(
        "gate_c" in k or "lateral_c" in k
        for k in net.state_dict()
    )

    del ck
    return net.eval()


def qc_wh_iou(pred, truth):
    p = np.maximum(
        np.asarray(pred, dtype=float), 0.
    )
    g = np.maximum(
        np.asarray(truth, dtype=float), 0.
    )

    inter = (
        min(p[0], g[0])
        * min(p[1], g[1])
    )

    union = (
        p[0] * p[1]
        + g[0] * g[1]
        - inter
    )

    return (
        float(inter / union)
        if union > 0
        else 0.0
    )


def qc_box(cx, cy, w, h):
    w = max(float(w), 0.)
    h = max(float(h), 0.)
    cx, cy = float(cx), float(cy)

    return (
        cx - w / 2,
        cy - h / 2,
        cx + w / 2,
        cy + h / 2
    )


def qc_box_iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    inter = (
        max(0., x2 - x1)
        * max(0., y2 - y1)
    )

    area_a = (
        max(0., a[2] - a[0])
        * max(0., a[3] - a[1])
    )

    area_b = (
        max(0., b[2] - b[0])
        * max(0., b[3] - b[1])
    )

    union = area_a + area_b - inter

    return (
        float(inter / union)
        if union > 0
        else 0.0
    )


# Verify checkpoint metadata and state-dict keys.

report33 = {}
key_sets = {}

for arm, path in QC_PATHS.items():
    ck = torch.load(
        path,
        map_location="cpu",
        weights_only=False
    )

    expected = (
        "A3_neighbor_R1"
        if arm == QC_REFERENCE
        else "A3_v2"
    )

    assert ck["arm"] == expected
    assert ck["gaussian_radius_mode"] == "v1_2_original"
    assert abs(float(ck["neighbor_lambda"]) - 0.25) < 1e-12

    if arm == QC_CANDIDATE:
        assert ck["base_checkpoint"] == A3_CKPT
        assert int(ck["additional_exposures"]) > 0

    keys = set(ck["model_state_dict"].keys())

    assert not any(
        "gate_c" in k or "lateral_c" in k
        for k in keys
    )

    key_sets[arm] = keys

    report33[arm] = {
        "checkpoint": path,
        "additional_exposures_recorded": int(
            ck["additional_exposures"]
        ),
        "epoch_recorded": int(ck["epoch"]),
        "state_entries": len(keys),
        "neighbor_lambda": float(ck["neighbor_lambda"])
    }

    del ck

assert (
    key_sets[QC_REFERENCE]
    == key_sets[QC_CANDIDATE]
)

print(json.dumps(report33, indent=2))

print(
    "Saved:",
    qc_save(
        "qc33_checkpoint_integrity.json",
        report33
    )
)

print("STEP 33 PASS — A3 vs A3-v2")
