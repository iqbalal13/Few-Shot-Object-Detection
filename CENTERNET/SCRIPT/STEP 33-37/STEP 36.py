# ============================================================
# STEP 36 / QUALITY CHECK 3
# HEATMAP & TOP-K DIAGNOSTIC
# 80K VS 160K — NO TRAINING
# ============================================================

import os
import json
import numpy as np
import torch

from torch.utils.data import DataLoader

qc3_paths = {
    "V1.2_80K": BASE_BEST30,
    "V1.2_160K": MILESTONE20_30
}

qc3_loader = DataLoader(
    val_dataset,
    batch_size=8,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
    collate_fn=centernet_collate_fn
)

qc3_results = {}

for name, path in qc3_paths.items():

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

    counts = {
        "gt": 0,
        "center_correct_class": 0,
        "center_is_local_peak": 0,
        "top100_exact_spatial": 0,
        "top100_exact_class": 0,
        "top100_near1_spatial": 0,
        "top100_near1_class": 0
    }

    true_scores = []
    top_scores = []
    max_class_scores = []

    seen = 0

    with torch.inference_mode():

        for images, targets in qc3_loader:

            if seen >= 1000:
                break

            with torch.amp.autocast(
                "cuda",
                dtype=torch.bfloat16
            ):
                out = net(
                    images.to(device)
                )

            probs = out["heatmap"].float().sigmoid()

            peaks = heatmap_local_nms(probs)

            (
                scores,
                indices,
                classes,
                ys,
                xs
            ) = topk_centers(
                peaks,
                K=100
            )

            H, W = probs.shape[-2:]

            top_scores.extend(
                scores.cpu().flatten().tolist()
            )

            for i, t in enumerate(targets):

                boxes = t["boxes"].to(device).float()
                labels = t["labels"].to(device).long()

                if len(boxes) == 0:
                    continue

                wh = (
                    boxes[:, 2:] - boxes[:, :2]
                )

                centers = (
                    boxes[:, :2] + boxes[:, 2:]
                ) / 8

                xy = centers.floor().long()

                ok = (
                    (wh > 1).all(1)
                    & (xy[:, 0] >= 0)
                    & (xy[:, 0] < W)
                    & (xy[:, 1] >= 0)
                    & (xy[:, 1] < H)
                    & (labels >= 0)
                    & (labels < 80)
                )

                xy = xy[ok]
                labels = labels[ok]

                if len(labels) == 0:
                    continue

                xx = xy[:, 0]
                yy = xy[:, 1]

                true = probs[
                    i, labels, yy, xx
                ]

                best_class = probs[
                    i, :, yy, xx
                ].argmax(0)

                true_scores.extend(
                    true.cpu().tolist()
                )

                max_class_scores.extend(
                    probs[
                        i, :, yy, xx
                    ].amax(0).cpu().tolist()
                )

                counts["gt"] += len(labels)

                counts["center_correct_class"] += int(
                    (best_class == labels).sum()
                )

                counts["center_is_local_peak"] += int(
                    (
                        peaks[
                            i, labels, yy, xx
                        ] > 0
                    ).sum()
                )

                for j in range(len(labels)):

                    dx = (
                        xs[i] - xx[j]
                    ).abs()

                    dy = (
                        ys[i] - yy[j]
                    ).abs()

                    exact = (
                        (dx == 0)
                        & (dy == 0)
                    )

                    near = (
                        (dx <= 1)
                        & (dy <= 1)
                    )

                    same = (
                        classes[i] == labels[j]
                    )

                    counts["top100_exact_spatial"] += int(
                        exact.any()
                    )

                    counts["top100_exact_class"] += int(
                        (exact & same).any()
                    )

                    counts["top100_near1_spatial"] += int(
                        near.any()
                    )

                    counts["top100_near1_class"] += int(
                        (near & same).any()
                    )

            seen += len(targets)

    assert seen == 1000
    assert counts["gt"] > 0

    den = counts["gt"]

    qc3_results[name] = {
        "images": seen,
        "gt": den,

        "gt_center_argmax_class_accuracy_pct": (
            100 * counts["center_correct_class"] / den
        ),

        "gt_center_is_local_peak_pct": (
            100 * counts["center_is_local_peak"] / den
        ),

        "top100_exact_spatial_recall_pct": (
            100 * counts["top100_exact_spatial"] / den
        ),

        "top100_exact_class_recall_pct": (
            100 * counts["top100_exact_class"] / den
        ),

        "top100_near1_spatial_recall_pct": (
            100 * counts["top100_near1_spatial"] / den
        ),

        "top100_near1_class_recall_pct": (
            100 * counts["top100_near1_class"] / den
        ),

        "gt_correct_class_score_median": float(
            np.median(true_scores)
        ),

        "gt_max_class_score_median": float(
            np.median(max_class_scores)
        ),

        "top100_score_median": float(
            np.median(top_scores)
        )
    }

    print("\nQC3:", name)

    for k, v in qc3_results[name].items():
        print(k, ":", v)

    del net, ckpt
    torch.cuda.empty_cache()


qc3_dir = os.path.join(
    ROOT30,
    "CenterNet_Quality_Diagnostics"
)

os.makedirs(qc3_dir, exist_ok=True)

with open(
    os.path.join(
        qc3_dir,
        "qc3_heatmap_topk_80k_vs_160k.json"
    ),
    "w"
) as f:
    json.dump(
        qc3_results,
        f,
        indent=2,
        allow_nan=False
    )

print("QC3 COMPLETE")
print("PURE INFERENCE — NO TRAINING")
