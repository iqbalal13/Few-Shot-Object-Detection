
# ============================================================
# STEP 34 — QC1 / WH + OFFSET + ORACLE BOX
# BASELINE VS A1 VS A2 — ALL AT 160K
# INFERENCE ONLY
# ============================================================

import os
import json
import gc
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

QC_BASE160 = os.path.join(
    "/content/drive/MyDrive",
    "CenterNet_COCO80_V12_CONTINUE_160K",
    "epoch20_160k.pth"
)

QC_A1_160 = os.path.join(
    "/content/drive/MyDrive",
    "CenterNet_COCO80_V12_A_RELWH_FRESH_160K_BS8",
    "epoch20_160k.pth"
)

# Historical A2 is separate from the current B1 run.
QC_A2_DIR = os.path.join(
    "/content/drive/MyDrive",
    "CenterNet_COCO80_V12_A2_HYBRIDWH_FRESH_160K_BS8"
)
QC_A2_160 = os.path.join(
    QC_A2_DIR, "epoch20_160k.pth"
)

QC_B1_160 = milestone30(20)

assert (
    os.path.abspath(QC_A2_160)
    != os.path.abspath(QC_B1_160)
)

# Historical comparison always uses 3 original models.
QC_PATHS = {
    "Baseline_V12_160K": QC_BASE160,
    "Experiment_A1_160K": QC_A1_160,
    "Experiment_A2_160K": QC_A2_160
}

# Keep corrected reports separate from old mislabeled data.
QC_OUT = os.path.join(
    FULL_CHECKPOINT_DIR,
    "quality_checks_verified_historical"
)
os.makedirs(QC_OUT, exist_ok=True)

assert CONFIG["source_images_total"] == 160000
assert CONFIG["wh_loss_type"] == "hybrid_abs_relative"
assert len(val_dataset) == 5000
assert not val_dataset.augment

for name, path in QC_PATHS.items():
    assert os.path.isfile(path), f"{name} missing: {path}"

# Verify A2 is actually V1.2 and hybrid WH.
a2_ckpt34 = torch.load(
    QC_A2_160,
    map_location="cpu",
    weights_only=False
)
assert int(a2_ckpt34["epoch"]) == 20
assert (
    a2_ckpt34["config"]["gaussian_radius_mode"]
    == "v1_2_original"
)
assert (
    a2_ckpt34["config"]["wh_loss_type"]
    == "hybrid_abs_relative"
)
del a2_ckpt34

assert len(QC_PATHS) == 3

print("QC1 historical checkpoints:")
for label, path in QC_PATHS.items():
    print(f"{label}: {path}")


# ============================================================
# SHARED QC HELPERS
# ============================================================

def qc_load_model(path):
    ckpt = torch.load(
        path,
        map_location="cpu",
        weights_only=False
    )

    assert int(ckpt["epoch"]) == 20
    assert "model_state_dict" in ckpt

    net = CenterNetResNet101(
        num_classes=80,
        pretrained_backbone=False
    ).to(device)

    net.load_state_dict(
        ckpt["model_state_dict"],
        strict=True
    )
    net.eval()

    del ckpt
    return net


def qc_loader():
    return DataLoader(
        Subset(val_dataset, range(1000)),
        batch_size=8,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        collate_fn=centernet_collate_fn
    )


def qc_valid_gt(t, height, width):
    boxes = t["boxes"].to(device).float()
    labels = t["labels"].to(device).long()

    stride = CONFIG["output_stride"]

    wh = (
        boxes[:, 2:] - boxes[:, :2]
    ) / stride

    ctr = (
        boxes[:, :2] + boxes[:, 2:]
    ) / (2 * stride)

    ij = ctr.floor().long()

    ok = (
        (wh > 1 / stride).all(1)
        & (ij[:, 0] >= 0)
        & (ij[:, 0] < width)
        & (ij[:, 1] >= 0)
        & (ij[:, 1] < height)
        & (labels >= 0)
        & (labels < 80)
    )

    return (
        boxes[ok],
        labels[ok],
        wh[ok],
        ctr[ok],
        ij[ok]
    )


def qc_iou(pred_boxes, gt_boxes):
    lt = torch.maximum(
        pred_boxes[:, :2],
        gt_boxes[:, :2]
    )
    rb = torch.minimum(
        pred_boxes[:, 2:],
        gt_boxes[:, 2:]
    )

    intersection = (
        rb - lt
    ).clamp_min(0).prod(1)

    pred_area = (
        pred_boxes[:, 2:] - pred_boxes[:, :2]
    ).clamp_min(0).prod(1)

    gt_area = (
        gt_boxes[:, 2:] - gt_boxes[:, :2]
    ).clamp_min(0).prod(1)

    return intersection / (
        pred_area + gt_area - intersection
    ).clamp_min(1e-8)


def qc_box(center_cells, wh_cells):
    stride = CONFIG["output_stride"]
    center_px = center_cells * stride
    half = wh_cells.clamp_min(0) * stride / 2

    return torch.cat(
        (
            center_px - half,
            center_px + half
        ),
        dim=-1
    )


# ============================================================
# EVALUATE ALL THREE CHECKPOINTS
# ============================================================

qc1 = {}

for name, path in QC_PATHS.items():
    model = qc_load_model(path)

    stat = {
        key: []
        for key in (
            "wh_abs",
            "off_abs",
            "off05_abs",
            "iou",
            "wh_bad_gt",
            "gt_confidence",
            "top_confidence",
            "wh_bad_top",
            "off_bad_top"
        )
    }

    image_count = 0

    with torch.inference_mode():
        for images, targets in qc_loader():

            with torch.amp.autocast(
                "cuda",
                dtype=torch.bfloat16
            ):
                raw = model(images.to(device))

            raw = {
                key: value.float()
                for key, value in raw.items()
            }

            probability = raw["heatmap"].sigmoid()
            peaks = heatmap_local_nms(probability)

            scores, indices, _, _, _ = topk_centers(
                peaks, K=100
            )

            top_wh = transpose_and_gather_feature(
                raw["wh"], indices
            )
            top_off = transpose_and_gather_feature(
                raw["offset"], indices
            )

            stat["top_confidence"].extend(
                scores.flatten().cpu().tolist()
            )

            stat["wh_bad_top"].extend(
                (top_wh <= 0).any(-1).flatten().cpu().tolist()
            )

            stat["off_bad_top"].extend(
                (
                    (top_off < 0)
                    | (top_off >= 1)
                ).any(-1).flatten().cpu().tolist()
            )

            H, W = probability.shape[-2:]

            for b, target in enumerate(targets):
                boxes, labels, gt_wh, center, ij = (
                    qc_valid_gt(target, H, W)
                )

                if not len(labels):
                    continue

                x = ij[:, 0]
                y = ij[:, 1]

                pred_wh = raw["wh"][b, :, y, x].T
                pred_off = raw["offset"][b, :, y, x].T
                gt_off = center - ij.float()

                oracle_iou = qc_iou(
                    qc_box(
                        ij.float() + pred_off,
                        pred_wh
                    ),
                    boxes
                )

                stat["wh_abs"].extend(
                    (pred_wh - gt_wh)
                    .abs()
                    .flatten()
                    .cpu()
                    .tolist()
                )

                stat["off_abs"].extend(
                    (pred_off - gt_off)
                    .abs()
                    .flatten()
                    .cpu()
                    .tolist()
                )

                stat["off05_abs"].extend(
                    (gt_off - 0.5)
                    .abs()
                    .flatten()
                    .cpu()
                    .tolist()
                )

                stat["iou"].extend(
                    oracle_iou.cpu().tolist()
                )

                stat["wh_bad_gt"].extend(
                    (pred_wh <= 0)
                    .any(-1)
                    .cpu()
                    .tolist()
                )

                stat["gt_confidence"].extend(
                    probability[b, labels, y, x]
                    .cpu()
                    .tolist()
                )

            image_count += len(targets)

    assert image_count == 1000

    oracle = np.asarray(stat["iou"])
    assert len(oracle) > 0

    qc1[name] = {
        "images": image_count,
        "gt_count": int(len(oracle)),
        "wh_mae_cells": float(
            np.mean(stat["wh_abs"])
        ),
        "offset_mae_cells": float(
            np.mean(stat["off_abs"])
        ),
        "offset_constant_05_mae": float(
            np.mean(stat["off05_abs"])
        ),
        "oracle_mean_iou": float(oracle.mean()),
        "oracle_median_iou": float(np.median(oracle)),
        "oracle_iou50_pct": float(
            100 * np.mean(oracle >= .50)
        ),
        "oracle_iou75_pct": float(
            100 * np.mean(oracle >= .75)
        ),
        "wh_nonpositive_gt_pct": float(
            100 * np.mean(stat["wh_bad_gt"])
        ),
        "wh_nonpositive_top100_pct": float(
            100 * np.mean(stat["wh_bad_top"])
        ),
        "offset_outside_top100_pct": float(
            100 * np.mean(stat["off_bad_top"])
        ),
        "gt_class_score_median": float(
            np.median(stat["gt_confidence"])
        ),
        "top100_score_median": float(
            np.median(stat["top_confidence"])
        )
    }

    print("\nQC1:", name)
    for key, value in qc1[name].items():
        print(f"  {key}: {value}")

    del model
    gc.collect()
    torch.cuda.empty_cache()


# ============================================================
# VERIFY + SAVE
# ============================================================

assert len({
    result["gt_count"]
    for result in qc1.values()
}) == 1

with open(
    os.path.join(
        QC_OUT,
        "qc1_baseline_A1_A2_160k.json"
    ),
    "w",
    encoding="utf-8"
) as file:
    json.dump(
        qc1,
        file,
        indent=2,
        allow_nan=False
    )

print("STEP 34 COMPLETE — NO TRAINING")
print("Saved:", QC_OUT)
