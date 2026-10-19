
# ============================================================
# STEP 41 — B1 POST-TRAINING QUALITY CONTROL
# BASELINE V1.2 vs A2 vs B1
# FIXED 160K / FIRST 1000 COCO-VAL IMAGES
# NO TRAINING
# ============================================================

import os
import json
import gc

from collections import defaultdict

import numpy as np
import torch

from torch.utils.data import DataLoader, Subset


# ============================================================
# 1. CHECKPOINTS
# ============================================================

QC41_PATHS = {
    "Baseline_V12_160K": (
        "/content/drive/MyDrive/"
        "CenterNet_COCO80_V12_CONTINUE_160K/"
        "epoch20_160k.pth"
    ),

    "A2_V12_160K": (
        "/content/drive/MyDrive/"
        "CenterNet_COCO80_V12_A2_HYBRIDWH_FRESH_160K_BS8/"
        "epoch20_160k.pth"
    ),

    "B1_V13_160K": milestone30(20)
}

QC41_SCORE = 0.05
QC41_IMAGE_COUNT = 1000

QC41_GROUPS = (
    "all",
    "small",
    "medium",
    "large",
    "person"
)

PERSON41 = int(
    NAME_TO_CONTIGUOUS["person"]
)


# ============================================================
# 2. SIZE-ONLY IoU
# PREDICTED AND GT BOX SHARE SAME CENTER
# ============================================================

def qc41_size_iou(pred_wh, gt_wh):
    pw, ph = [
        max(0.0, float(v))
        for v in pred_wh
    ]

    gw, gh = [
        float(v)
        for v in gt_wh
    ]

    iw = min(pw, gw)
    ih = min(ph, gh)

    intersection = iw * ih

    union = (
        pw * ph
        + gw * gh
        - intersection
    )

    return intersection / max(union, 1e-9)


# ============================================================
# 3. FIXED 1000-IMAGE LOADER
# ============================================================

def qc41_loader():
    return DataLoader(
        Subset(
            val_dataset,
            range(QC41_IMAGE_COUNT)
        ),
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        collate_fn=centernet_collate_fn
    )


# ============================================================
# 4. EVALUATE ONE CHECKPOINT
# ============================================================

def qc41_run_model(path, name):
    ckpt = torch.load(
        path,
        map_location="cpu",
        weights_only=False
    )

    assert int(ckpt["epoch"]) == 20, (
        name, ckpt["epoch"]
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
    del ckpt

    sums = {
        group: defaultdict(float)
        for group in QC41_GROUPS
    }

    images_seen = 0

    with torch.inference_mode():
        for images, raw_targets in qc41_loader():
            images = images.to(
                device,
                non_blocking=True
            )

            with torch.amp.autocast(
                "cuda",
                dtype=torch.bfloat16
            ):
                out = net(images)

            logits = out["heatmap"].float()

            local = heatmap_local_nms(
                logits.sigmoid()
            )

            scores, _, classes, py, px = (
                topk_centers(
                    local,
                    K=100
                )
            )

            scores = scores.cpu().numpy()
            classes = classes.cpu().numpy()
            py = py.cpu().numpy()
            px = px.cpu().numpy()

            for j, target in enumerate(raw_targets):
                # WH/indices are unchanged by Gaussian mode.
                enc = build_centernet_target(
                    target
                )

                valid = torch.nonzero(
                    enc["mask"],
                    as_tuple=False
                ).flatten().tolist()

                scale = float(
                    target["scale"].item()
                )

                peak_cache = {}
                this_image_gt = []

                for k in valid:
                    cls = int(
                        enc["labels"][k]
                    )

                    idx = int(
                        enc["indices"][k]
                    )

                    cy, cx = divmod(
                        idx,
                        OUTPUT_WIDTH
                    )

                    gt_wh = (
                        enc["wh"][k].tolist()
                    )

                    # Approximate original bbox area.
                    area = (
                        gt_wh[0]
                        * gt_wh[1]
                        * (OUTPUT_STRIDE / scale) ** 2
                    )

                    if area < 32 ** 2:
                        size = "small"
                    elif area < 96 ** 2:
                        size = "medium"
                    else:
                        size = "large"

                    groups = ["all", size]

                    if cls == PERSON41:
                        groups.append("person")

                    # ----------------------------------------
                    # CLASS-CORRECT LOCAL PEAKS
                    # ----------------------------------------

                    if cls not in peak_cache:
                        coords = torch.nonzero(
                            local[j, cls] >= QC41_SCORE,
                            as_tuple=False
                        ).cpu().numpy()

                        peak_cache[cls] = coords

                    coords = peak_cache[cls]

                    if len(coords):
                        distances = np.maximum(
                            abs(coords[:, 0] - cy),
                            abs(coords[:, 1] - cx)
                        )

                        closest = int(
                            np.argmin(distances)
                        )

                        nearest = float(
                            distances[closest]
                        )

                        peak_y, peak_x = map(
                            int,
                            coords[closest]
                        )

                    else:
                        nearest = float("inf")
                        peak_y = -1
                        peak_x = -1

                    # ----------------------------------------
                    # CLASS AT GT CENTER
                    # ----------------------------------------

                    center_correct = int(
                        int(
                            logits[
                                j, :, cy, cx
                            ].argmax().item()
                        ) == cls
                    )

                    center_peak = int(
                        float(
                            local[
                                j, cls, cy, cx
                            ]
                        ) >= QC41_SCORE
                    )

                    # ----------------------------------------
                    # WH AT GT CENTER
                    # ----------------------------------------

                    wh_center = (
                        out["wh"][
                            j, :, cy, cx
                        ].float().tolist()
                    )

                    center_iou = qc41_size_iou(
                        wh_center,
                        gt_wh
                    )

                    # ----------------------------------------
                    # TOP-100 CLASS-CORRECT PEAK ±1
                    # ----------------------------------------

                    topmask = (
                        (classes[j] == cls)
                        & (scores[j] >= QC41_SCORE)
                        & (
                            np.maximum(
                                abs(py[j] - cy),
                                abs(px[j] - cx)
                            ) <= 1
                        )
                    )

                    top100_near1 = int(
                        bool(topmask.any())
                    )

                    # ----------------------------------------
                    # COLLECT GROUP STATS
                    # ----------------------------------------

                    for group in groups:
                        s = sums[group]

                        s["n"] += 1

                        s["center_class_correct"] += (
                            center_correct
                        )

                        s["exact_peak"] += center_peak

                        s["peak1"] += int(
                            nearest <= 1
                        )

                        s["peak2"] += int(
                            nearest <= 2
                        )

                        s["peak4"] += int(
                            nearest <= 4
                        )

                        s["top100_near1"] += (
                            top100_near1
                        )

                        s["center_wh_iou_sum"] += (
                            center_iou
                        )

                        # Paired analysis:
                        # Same object, GT center vs
                        # nearest class-correct peak.
                        if nearest <= 4:
                            wh_peak = (
                                out["wh"][
                                    j,
                                    :,
                                    peak_y,
                                    peak_x
                                ].float().tolist()
                            )

                            s["paired_center_iou_sum"] += (
                                center_iou
                            )

                            s["paired_peak_iou_sum"] += (
                                qc41_size_iou(
                                    wh_peak,
                                    gt_wh
                                )
                            )

                            s["paired_n"] += 1

                    this_image_gt.append(
                        (cls, cy, cx, groups)
                    )

                # ============================================
                # ONE-TO-ONE TOP-100 MATCHING ±1
                # ============================================

                candidates = []

                for gi, (cls, cy, cx, _) in enumerate(
                    this_image_gt
                ):
                    for pi in range(
                        len(scores[j])
                    ):
                        if (
                            int(classes[j, pi]) != cls
                            or scores[j, pi] < QC41_SCORE
                        ):
                            continue

                        distance = max(
                            abs(py[j, pi] - cy),
                            abs(px[j, pi] - cx)
                        )

                        if distance <= 1:
                            candidates.append((
                                -float(scores[j, pi]),
                                float(distance),
                                gi,
                                pi
                            ))

                candidates.sort()

                assigned_gt = set()
                assigned_pred = set()

                for _, _, gi, pi in candidates:
                    if (
                        gi in assigned_gt
                        or pi in assigned_pred
                    ):
                        continue

                    assigned_gt.add(gi)
                    assigned_pred.add(pi)

                    for group in this_image_gt[gi][3]:
                        sums[group][
                            "top100_one_to_one_near1"
                        ] += 1

            images_seen += len(raw_targets)

    assert images_seen == QC41_IMAGE_COUNT

    # ========================================================
    # 5. NORMALIZE METRICS
    # ========================================================

    result = {
        "model": name,
        "images": images_seen,
        "groups": {}
    }

    for group, s in sums.items():
        n = max(s["n"], 1)
        m = max(s["paired_n"], 1)

        result["groups"][group] = {
            "gt": int(s["n"]),

            "class_correct_center_pct": (
                100 * s["center_class_correct"] / n
            ),

            "exact_peak_pct": (
                100 * s["exact_peak"] / n
            ),

            "peak1_pct": (
                100 * s["peak1"] / n
            ),

            "peak2_pct": (
                100 * s["peak2"] / n
            ),

            "peak4_pct": (
                100 * s["peak4"] / n
            ),

            "top100_near1_pct": (
                100 * s["top100_near1"] / n
            ),

            "top100_one_to_one_near1_pct": (
                100 * s["top100_one_to_one_near1"] / n
            ),

            "wh_center_iou": (
                s["center_wh_iou_sum"] / n
            ),

            "matched_peak4_n": int(
                s["paired_n"]
            ),

            "wh_center_iou_matched_peak4": (
                s["paired_center_iou_sum"] / m
            ),

            "wh_peak_iou_matched_peak4": (
                s["paired_peak_iou_sum"] / m
            )
        }

    del net

    gc.collect()
    torch.cuda.empty_cache()

    return result


# ============================================================
# 6. RUN ALL AVAILABLE 160K CHECKPOINTS
# ============================================================

print("=" * 70)
print("STEP 41 — B1 QUALITY CONTROL")
print("Baseline V1.2 vs A2 vs B1")
print("=" * 70)

qc41_results = {}

for label, path in QC41_PATHS.items():
    if not os.path.isfile(path):
        if label == "B1_V13_160K":
            raise FileNotFoundError(path)

        print(
            "SKIP — checkpoint not found:",
            label,
            path
        )
        continue

    print("\nEvaluating:", label)

    qc41_results[label] = qc41_run_model(
        path,
        label
    )


# ============================================================
# 7. PRINT GROUP-WISE RESULTS
# ============================================================

for group in QC41_GROUPS:
    print("\nGROUP:", group)

    print(
        "Model                GT  "
        "Peak±1  Peak±2  Peak±4 "
        "Top100±1  One-to-one±1  WH@GT"
    )

    for name, result in qc41_results.items():
        s = result["groups"][group]

        print(
            f"{name:20s} "
            f"{s['gt']:5d} "
            f"{s['peak1_pct']:7.2f} "
            f"{s['peak2_pct']:7.2f} "
            f"{s['peak4_pct']:7.2f} "
            f"{s['top100_near1_pct']:9.2f} "
            f"{s['top100_one_to_one_near1_pct']:13.2f} "
            f"{s['wh_center_iou']:.4f}"
        )


# ============================================================
# 8. PAIRED WH READOUT ANALYSIS
# ============================================================

print(
    "\nPaired WH readout:"
    "\nNearest class-correct peak within ±4."
)

for name, result in qc41_results.items():
    s = result["groups"]["all"]

    print(
        name,
        "matched=", s["matched_peak4_n"],
        "center_iou=",
        round(
            s["wh_center_iou_matched_peak4"],
            4
        ),
        "peak_iou=",
        round(
            s["wh_peak_iou_matched_peak4"],
            4
        )
    )


# ============================================================
# 9. SAVE DIAGNOSTICS
# ============================================================

qc41_save_path = os.path.join(
    FULL_CHECKPOINT_DIR,
    "b1_qc41_comparison.json"
)

with open(
    qc41_save_path,
    "w",
    encoding="utf-8"
) as file:
    json.dump(
        qc41_results,
        file,
        indent=2,
        allow_nan=False
    )

print("\nB1 official COCOeval at fixed 160K:")

print(
    step33_160k_result["metrics"]
)

print("\nResults saved:")
print(qc41_save_path)

print(
    "\nNote: all Step 41 peak metrics "
    "use local NMS and score >= 0.05."
)

print(
    "Historical Step 36 metrics without "
    "a score threshold are not directly identical."
)

print("STEP 41 PASSED — NO TRAINING")
