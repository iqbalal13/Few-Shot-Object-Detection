
# STEP 32 — B2 QC setup
import os
import json
import gc
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

QC_DIR = os.path.join(
    PILOT_ROOT,
    "quality_checks_B2"
)
os.makedirs(QC_DIR, exist_ok=True)

assert set(paths) == {
    "A2_control", B2_ARM
}
assert len(val_dataset) == 5000


def qc_save(filename, obj):
    p = os.path.join(QC_DIR, filename)

    with open(p, "w", encoding="utf8") as f:
        json.dump(
            obj, f,
            indent=2,
            allow_nan=False
        )

    return p


def qc_loader():
    return DataLoader(
        Subset(val_dataset, range(1000)),
        batch_size=8,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        collate_fn=centernet_collate_fn
    )


def qc_model(arm):
    ck = torch.load(
        paths[arm],
        map_location="cpu",
        weights_only=False
    )

    m = CenterNetResNet101(
        num_classes=80,
        pretrained_backbone=False
    ).to(device)

    if arm == "A2_control":
        load_a2_into_b2(
            m, ck["model_state_dict"]
        )
    else:
        m.load_state_dict(
            ck["model_state_dict"],
            strict=True
        )

    m.eval()
    return m


def qc_wh_iou(pred, truth):
    p = np.maximum(
        np.asarray(pred, dtype=float), 0.0
    )
    g = np.maximum(
        np.asarray(truth, dtype=float), 0.0
    )

    inter = np.minimum(p, g).prod()

    return float(
        inter / max(
            1e-9,
            p.prod() + g.prod() - inter
        )
    )


def qc_box_iou(pred, truth):
    p = np.asarray(pred, dtype=float)
    g = np.asarray(truth, dtype=float)

    inter = np.maximum(
        0,
        np.minimum(p[2:], g[2:])
        - np.maximum(p[:2], g[:2])
    ).prod()

    a = np.maximum(0, p[2:] - p[:2]).prod()
    b = np.maximum(0, g[2:] - g[:2]).prod()

    return float(
        inter / max(1e-9, a + b - inter)
    )


def qc_box(cx, cy, w, h):
    w = max(0, float(w))
    h = max(0, float(h))

    return [
        cx - w / 2,
        cy - h / 2,
        cx + w / 2,
        cy + h / 2
    ]


print("STEP 32 PASS — B2 QC setup")
