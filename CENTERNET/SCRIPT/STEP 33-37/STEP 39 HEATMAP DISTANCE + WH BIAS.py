
# ============================================================
# STEP 39 — QC6 / HEATMAP DISTANCE + WH BIAS
# BASELINE VS A1 VS A2 — 160K
# FIXED V1.2 RADIUS GROUPS
# INFERENCE ONLY — NO TRAINING
# ============================================================

import os
import json
import gc

import numpy as np
import torch

PERSON39 = int(val_dataset.cat_id_to_contiguous[1])

# FIX: Historical grouping always uses V1.2.
QC39_RADIUS_GROUP_MODE = "v1_2_original"

assert CONFIG["gaussian_radius_mode"] == "v1_3_corrected"

assert set(QC_PATHS) == {
    "Baseline_V12_160K",
    "Experiment_A1_160K",
    "Experiment_A2_160K"
}

assert len(val_dataset) == 5000

print(
    "QC6 historical radius grouping:",
    QC39_RADIUS_GROUP_MODE
)

R39 = 8
RADII39 = (0, 1, 2, 4, 8)
THRESH39 = (0.0, 0.05, 0.10, 0.20)

offsets39 = torch.arange(
    -R39, R39 + 1,
    device=device
)

dy39, dx39 = torch.meshgrid(
    offsets39,
    offsets39,
    indexing="ij"
)

distance_grid39 = torch.maximum(
    dy39.abs(),
    dx39.abs()
)[None, :, :]


# ============================================================
# 1. STATISTICAL HELPERS
# ============================================================

def stats39(values):
    array = np.asarray(
        values,
        dtype=np.float64
    )

    if len(array) == 0:
        return None

    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p10": float(np.percentile(array, 10)),
        "p90": float(np.percentile(array, 90))
    }


def summarize39(rows):
    if not rows:
        return None

    result = {
        "gt_count": len(rows),

        "class_at_gt_accuracy_pct": float(
            100 * np.mean([
                row["class_correct"]
                for row in rows
            ])
        ),

        "gaussian_radius_zero_pct": float(
            100 * np.mean([
                row["radius_zero"]
                for row in rows
            ])
        ),

        "center_score": stats39([
            row["gt_score"]
            for row in rows
        ]),

        "strongest_peak_score_r4": stats39([
            row["max_peak_r4"]
            for row in rows
        ])
    }

    result["peak_coverage"] = {}

    for threshold in THRESH39:

        distances = np.asarray([
            row["distances"][str(threshold)]
            for row in rows
        ])

        result["peak_coverage"][
            f"{threshold:.2f}"
        ] = {
            f"within_{radius}_pct": float(
                100 * np.mean(
                    distances <= radius
                )
            )
            for radius in RADII39
        }

        result["peak_coverage"][
            f"{threshold:.2f}"
        ]["not_found_within_8_pct"] = float(
            100 * np.mean(distances > 8)
        )

    result["wh"] = {}

    for dim in ("w", "h"):

        signed = np.asarray([
            row[f"signed_{dim}"]
            for row in rows
        ])

        relative = np.asarray([
            row[f"rel_{dim}"]
            for row in rows
        ])

        result["wh"][
            f"signed_{dim}_cells"
        ] = stats39(signed)

        result["wh"][
            f"rel_{dim}"
        ] = stats39(relative)

        result["wh"][
            f"abs_rel_{dim}"
        ] = stats39(
            np.abs(relative)
        )

        result["wh"][
            f"{dim}_underestimated_pct"
        ] = float(
            100 * np.mean(relative < 0)
        )

        result["wh"][
            f"{dim}_overestimated_20pct_pct"
        ] = float(
            100 * np.mean(relative > 0.20)
        )

        result["wh"][
            f"{dim}_underestimated_20pct_pct"
        ] = float(
            100 * np.mean(relative < -0.20)
        )

    return result


# ============================================================
# 2. DIAGNOSE EACH MODEL
# ============================================================

qc6 = {}

for name, path in QC_PATHS.items():

    model = qc_load_model(path)

    records = []
    images_seen = 0

    with torch.inference_mode():

        for images, targets in qc_loader():

            with torch.amp.autocast(
                "cuda",
                dtype=torch.bfloat16
            ):
                output = model(
                    images.to(device)
                )

            probability = (
                output["heatmap"].float().sigmoid()
            )

            peaks = heatmap_local_nms(
                probability
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

                pred_wh = output["wh"][
                    b, :, y, x
                ].T.float()

                gt_score = probability[
                    b, labels, y, x
                ]

                class_correct = (
                    probability[b, :, y, x].argmax(0)
                    == labels
                )

                # --------------------------------------------
                # Extract 17x17 window around GT center
                # --------------------------------------------

                patch_x = (
                    x[:, None, None]
                    + dx39[None, :, :]
                )

                patch_y = (
                    y[:, None, None]
                    + dy39[None, :, :]
                )

                within_image = (
                    (patch_x >= 0)
                    & (patch_x < W)
                    & (patch_y >= 0)
                    & (patch_y < H)
                )

                peak_patch = peaks[
                    b,
                    labels[:, None, None],
                    patch_y.clamp(0, H - 1),
                    patch_x.clamp(0, W - 1)
                ]

                peak_patch = (
                    peak_patch.masked_fill(
                        ~within_image,
                        0.0
                    )
                )

                within4 = (
                    (distance_grid39 <= 4)
                    & within_image
                )

                max_peak_r4 = (
                    peak_patch
                    .masked_fill(
                        ~within4,
                        0.0
                    )
                    .flatten(1)
                    .amax(1)
                )

                # --------------------------------------------
                # Nearest local peak by threshold
                # --------------------------------------------

                nearest = {}

                for threshold in THRESH39:

                    if threshold == 0.0:
                        active = (
                            (peak_patch > 0)
                            & within_image
                        )

                    else:
                        active = (
                            (peak_patch >= threshold)
                            & within_image
                        )

                    distances = torch.where(
                        active,
                        distance_grid39,
                        torch.full_like(
                            distance_grid39,
                            999
                        )
                    )

                    nearest[str(threshold)] = (
                        distances
                        .flatten(1)
                        .amin(1)
                        .cpu()
                        .numpy()
                    )

                # --------------------------------------------
                # Group objects by original bbox area
                # --------------------------------------------

                scale = float(
                    target["scale"].item()
                )

                area = (
                    (boxes[:, 2] - boxes[:, 0])
                    * (boxes[:, 3] - boxes[:, 1])
                    / scale ** 2
                )

                gtwh_np = gt_wh.cpu().numpy()
                predwh_np = pred_wh.cpu().numpy()
                labels_np = labels.cpu().numpy()
                areas_np = area.cpu().numpy()
                scores_np = gt_score.cpu().numpy()

                maxpeak_np = (
                    max_peak_r4.cpu().numpy()
                )

                correct_np = (
                    class_correct.cpu().numpy()
                )

                for j in range(len(labels_np)):

                    gt_w, gt_h = map(
                        float,
                        gtwh_np[j]
                    )

                    pred_w, pred_h = map(
                        float,
                        predwh_np[j]
                    )

                    if areas_np[j] < 32 ** 2:
                        size_group = "small"

                    elif areas_np[j] < 96 ** 2:
                        size_group = "medium"

                    else:
                        size_group = "large"

                    # ========================================
                    # CRITICAL FIX
                    # The historical radius groups use V1.2,
                    # regardless of current B1 CONFIG.
                    # ========================================

                    radius = int(
                        gaussian_radius(
                            gt_h,
                            gt_w,
                            min_overlap=0.7,
                            mode=QC39_RADIUS_GROUP_MODE
                        )
                    )

                    records.append({
                        "class_id": int(labels_np[j]),
                        "size": size_group,

                        "radius_zero": radius == 0,

                        "class_correct": bool(
                            correct_np[j]
                        ),

                        "gt_score": float(
                            scores_np[j]
                        ),

                        "max_peak_r4": float(
                            maxpeak_np[j]
                        ),

                        "gt_w": gt_w,
                        "gt_h": gt_h,
                        "pred_w": pred_w,
                        "pred_h": pred_h,

                        "signed_w": pred_w - gt_w,
                        "signed_h": pred_h - gt_h,

                        "rel_w": (
                            (pred_w - gt_w) / gt_w
                        ),

                        "rel_h": (
                            (pred_h - gt_h) / gt_h
                        ),

                        "distances": {
                            str(threshold): int(
                                nearest[
                                    str(threshold)
                                ][j]
                            )
                            for threshold in THRESH39
                        }
                    })

            images_seen += len(targets)

    assert images_seen == 1000
    assert len(records) == 7117

    # ========================================================
    # 3. GROUPS
    # ========================================================

    groups = {
        "all": records,

        "small": [
            row for row in records
            if row["size"] == "small"
        ],

        "medium": [
            row for row in records
            if row["size"] == "medium"
        ],

        "large": [
            row for row in records
            if row["size"] == "large"
        ],

        "person": [
            row for row in records
            if row["class_id"] == PERSON39
        ],

        "radius_0": [
            row for row in records
            if row["radius_zero"]
        ],

        "radius_positive": [
            row for row in records
            if not row["radius_zero"]
        ]
    }

    # Expected counts from original A2 QC6.
    assert len(groups["radius_0"]) == 2815, (
        f"Unexpected radius_0 count for {name}: "
        f"{len(groups['radius_0'])}"
    )

    assert len(groups["radius_positive"]) == 4302

    qc6[name] = {
        group: summarize39(rows)
        for group, rows in groups.items()
        if rows
    }

    print(
        f"\nQC6 — {name} | GT={len(records)}"
    )

    for group, summary in qc6[name].items():

        peak = summary[
            "peak_coverage"
        ]["0.05"]

        bias_w = (
            100
            * summary["wh"]["rel_w"]["mean"]
        )

        bias_h = (
            100
            * summary["wh"]["rel_h"]["mean"]
        )

        print(
            f"{group:<17} "
            f"N={summary['gt_count']:5d} "
            f"peak±1={peak['within_1_pct']:6.2f}% "
            f"peak±4={peak['within_4_pct']:6.2f}% "
            f"WH bias W/H="
            f"{bias_w:+7.2f}%/{bias_h:+7.2f}%"
        )

    del model
    gc.collect()
    torch.cuda.empty_cache()


# ============================================================
# 4. VERIFY SAME GT POPULATION
# ============================================================

assert len({
    model["all"]["gt_count"]
    for model in qc6.values()
}) == 1

assert len({
    model["radius_0"]["gt_count"]
    for model in qc6.values()
}) == 1


# ============================================================
# 5. COMPARISON BETWEEN HISTORICAL MODELS
# ============================================================

pairs = (
    (
        "Baseline_V12_160K",
        "Experiment_A1_160K"
    ),

    (
        "Baseline_V12_160K",
        "Experiment_A2_160K"
    ),

    (
        "Experiment_A1_160K",
        "Experiment_A2_160K"
    )
)

for left_name, right_name in pairs:

    print(
        f"\nDELTA {right_name} MINUS {left_name}"
    )

    for group in (
        "all",
        "small",
        "medium",
        "large",
        "person"
    ):

        left = qc6[left_name][group]
        right = qc6[right_name][group]

        for key in (
            "rel_w",
            "rel_h",
            "abs_rel_w",
            "abs_rel_h"
        ):

            delta = (
                right["wh"][key]["mean"]
                - left["wh"][key]["mean"]
            )

            print(
                f"{group:<8} "
                f"{key:<12} "
                f"{delta:+.6f}"
            )


# ============================================================
# 6. SAVE VERIFIED QC6 RESULTS
# ============================================================

QC39_FILE = os.path.join(
    QC_OUT,
    "qc6_heatmap_wh_baseline_A1_A2_160k_fixed_v12_groups.json"
)

with open(
    QC39_FILE,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        qc6,
        file,
        indent=2,
        allow_nan=False
    )

print("=" * 70)
print("STEP 39 COMPLETE — NO TRAINING")
print("Radius group definition:", QC39_RADIUS_GROUP_MODE)
print("Expected radius_0:", 2815)
print("Expected radius_positive:", 4302)
print("Saved:", QC39_FILE)
print("=" * 70)
