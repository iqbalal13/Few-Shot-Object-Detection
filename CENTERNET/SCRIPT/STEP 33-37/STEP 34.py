# ============================================================
# STEP 34 / QUALITY CHECK 1
# WH / OFFSET / ORACLE IoU
# 80K VS 160K — NO TRAINING
# ============================================================

import os
import json
import numpy as np
import torch

from torch.utils.data import DataLoader

assert len(val_dataset) == 5000
assert not val_dataset.augment

qc1_paths = {
    "V1.2_80K": BASE_BEST30,
    "V1.2_160K": MILESTONE20_30
}

qc1_loader = DataLoader(
    val_dataset,
    batch_size=8,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
    collate_fn=centernet_collate_fn
)

qc1_results = {}

for label, path in qc1_paths.items():

    ckpt = torch.load(
        path,
        map_location="cpu",
        weights_only=False
    )

    net = CenterNetResNet101(
        num_classes=80,
        pretrained_backbone=False
    ).to(device)

    net.load_state_dict(
        ckpt["model_state_dict"],
        strict=True
    )

    net.eval()

    counters = {
        "gt": 0,
        "gt_nonpositive_wh": 0,
        "topk": 0,
        "topk_nonpositive_wh": 0,
        "topk_bad_offset": 0
    }

    vals = {
        "wh_abs": [],
        "offset_abs": [],
        "oracle_iou": [],
        "topk_score": [],
        "gt_class_score": []
    }

    images_seen = 0

    with torch.inference_mode():

        for images, targets in qc1_loader:

            if images_seen >= 1000:
                break

            images = images.to(device)

            with torch.amp.autocast(
                "cuda",
                dtype=torch.bfloat16
            ):
                raw = net(images)

            out = {
                k: v.float()
                for k, v in raw.items()
            }

            scores, idx, cls, ys, xs = topk_centers(
                heatmap_local_nms(
                    out["heatmap"].sigmoid()
                ),
                K=100
            )

            wh_top = transpose_and_gather_feature(
                out["wh"], idx
            )

            off_top = transpose_and_gather_feature(
                out["offset"], idx
            )

            counters["topk"] += int(
                wh_top.shape[0] * wh_top.shape[1]
            )

            counters["topk_nonpositive_wh"] += int(
                (wh_top <= 0).any(-1).sum()
            )

            counters["topk_bad_offset"] += int(
                (
                    (off_top < 0)
                    | (off_top >= 1)
                ).any(-1).sum()
            )

            vals["topk_score"].extend(
                scores.cpu().flatten().tolist()
            )

            H, W = out["heatmap"].shape[-2:]

            for b, t in enumerate(targets):

                boxes = t["boxes"].to(device).float()
                labels = t["labels"].to(device).long()

                if len(boxes) == 0:
                    continue

                gt_wh = (
                    boxes[:, 2:] - boxes[:, :2]
                ) / 4

                centers = (
                    boxes[:, :2] + boxes[:, 2:]
                ) / 8

                ij = centers.floor().long()

                good = (
                    (gt_wh > 0.25).all(1)
                    & (ij[:, 0] >= 0)
                    & (ij[:, 0] < W)
                    & (ij[:, 1] >= 0)
                    & (ij[:, 1] < H)
                    & (labels >= 0)
                    & (labels < 80)
                )

                boxes, labels, gt_wh, centers, ij = (
                    x[good]
                    for x in (
                        boxes, labels, gt_wh,
                        centers, ij
                    )
                )

                if len(boxes) == 0:
                    continue

                x, y = ij[:, 0], ij[:, 1]

                pred_wh = out["wh"][b, :, y, x].T
                pred_off = out["offset"][b, :, y, x].T

                gt_off = centers - ij.float()

                counters["gt"] += len(boxes)

                counters["gt_nonpositive_wh"] += int(
                    (pred_wh <= 0).any(1).sum()
                )

                vals["wh_abs"].extend(
                    (pred_wh - gt_wh)
                    .abs()
                    .cpu()
                    .flatten()
                    .tolist()
                )

                vals["offset_abs"].extend(
                    (pred_off - gt_off)
                    .abs()
                    .cpu()
                    .flatten()
                    .tolist()
                )

                vals["gt_class_score"].extend(
                    out["heatmap"][
                        b, labels, y, x
                    ].sigmoid().cpu().tolist()
                )

                pc = (ij.float() + pred_off) * 4
                ph = pred_wh.clamp(min=0) * 2

                pred_boxes = torch.cat(
                    (pc - ph, pc + ph),
                    dim=1
                )

                lt = torch.maximum(
                    pred_boxes[:, :2],
                    boxes[:, :2]
                )

                rb = torch.minimum(
                    pred_boxes[:, 2:],
                    boxes[:, 2:]
                )

                intersection = (
                    rb - lt
                ).clamp(min=0).prod(1)

                pa = (
                    pred_boxes[:, 2:]
                    - pred_boxes[:, :2]
                ).clamp(min=0).prod(1)

                ga = (
                    boxes[:, 2:]
                    - boxes[:, :2]
                ).clamp(min=0).prod(1)

                iou = intersection / (
                    pa + ga - intersection
                ).clamp(min=1e-8)

                vals["oracle_iou"].extend(
                    iou.cpu().tolist()
                )

            images_seen += len(targets)

    assert images_seen == 1000
    assert counters["gt"] > 0

    a = np.asarray(vals["oracle_iou"])

    result = {
        "images": images_seen,
        "ground_truth": counters["gt"],
        "topk_candidates": counters["topk"],

        "wh_nonpositive_gt_pct": (
            100 * counters["gt_nonpositive_wh"]
            / counters["gt"]
        ),

        "wh_nonpositive_topk_pct": (
            100 * counters["topk_nonpositive_wh"]
            / counters["topk"]
        ),

        "offset_outside_topk_pct": (
            100 * counters["topk_bad_offset"]
            / counters["topk"]
        ),

        "wh_mae_cells": float(
            np.mean(vals["wh_abs"])
        ),

        "offset_mae_cells": float(
            np.mean(vals["offset_abs"])
        ),

        "oracle_iou_mean": float(a.mean()),

        "oracle_iou_median": float(
            np.median(a)
        ),

        "oracle_iou_ge50_pct": float(
            100 * np.mean(a >= 0.50)
        ),

        "oracle_iou_ge75_pct": float(
            100 * np.mean(a >= 0.75)
        ),

        "topk_confidence_median": float(
            np.median(vals["topk_score"])
        ),

        "gt_class_confidence_median": float(
            np.median(vals["gt_class_score"])
        )
    }

    qc1_results[label] = result

    print("\nQC1:", label)

    for k, v in result.items():
        print(k, ":", v)

    del net, ckpt
    torch.cuda.empty_cache()


qc1_dir = os.path.join(
    ROOT30, "CenterNet_Quality_Diagnostics"
)

os.makedirs(qc1_dir, exist_ok=True)

with open(
    os.path.join(
        qc1_dir,
        "qc1_v12_80k_vs_160k.json"
    ),
    "w"
) as f:
    json.dump(
        qc1_results,
        f,
        indent=2,
        allow_nan=False
    )

print("QC1 COMPLETE — NO TRAINING")
