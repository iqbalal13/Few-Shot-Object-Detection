
# ================================================================
# QUALITY CHECK 1/4
# CENTERNET BALANCED80K V1.2
#
# DIAGNOSIS:
# 1. WH regression
# 2. Offset regression
# 3. Heatmap confidence
# 4. Oracle-center bounding box IoU
# 5. Top-100 candidate statistics
#
# NO TRAINING
# NO F1
# NO CHECKPOINT MODIFICATION
# ================================================================

import os
import json
import math
import numpy as np
import torch

from torch.utils.data import DataLoader
from google.colab import drive

print("=" * 70)
print("QUALITY CHECK 1/4 — WH / OFFSET / HEATMAP / IoU")
print("=" * 70)


# ================================================================
# 1. CONFIGURATION
# ================================================================

assert torch.cuda.is_available(), (
    "Please enable a GPU runtime."
)

assert len(val_dataset) == 5000

assert not val_dataset.augment, (
    "Validation augmentation must be disabled."
)

assert callable(centernet_collate_fn)
assert callable(heatmap_local_nms)
assert callable(topk_centers)
assert callable(transpose_and_gather_feature)

DEVICE_Q1 = torch.device("cuda")

STRIDE_Q1 = int(
    CONFIG["output_stride"]
)

MAX_IMAGES_Q1 = 1000

TOPK_Q1 = 100

print("Device:", DEVICE_Q1)
print("Validation images:", MAX_IMAGES_Q1)
print("Top-K:", TOPK_Q1)
print("Output stride:", STRIDE_Q1)


# ================================================================
# 2. LOAD EXISTING V1.2 CHECKPOINT
# ================================================================

drive.mount(
    "/content/drive",
    force_remount=False
)

V12_DIR_Q1 = (
    "/content/drive/MyDrive/"
    "CenterNet_COCO80_Source_BALANCED80K_V1_2_BS8"
)

V12_BEST_Q1 = os.path.join(
    V12_DIR_Q1,
    "centernet_coco80_best.pth"
)

assert os.path.isfile(V12_BEST_Q1), (
    "V1.2 checkpoint not found: "
    + V12_BEST_Q1
)

v12_ckpt_q1 = torch.load(
    V12_BEST_Q1,
    map_location="cpu",
    weights_only=False
)

assert v12_ckpt_q1.get(
    "experiment_name"
) == "CenterNet_COCO80_FSOD_BALANCED80K_V1_2_BS8"

assert int(
    v12_ckpt_q1.get("source_images_total", -1)
) == 80000

assert int(
    v12_ckpt_q1.get("epoch", -1)
) == 10

print("Checkpoint verified.")
print("Experiment: Balanced80K V1.2")
print("Best epoch:", v12_ckpt_q1["epoch"])


# ================================================================
# 3. BUILD MODEL AND RESTORE WEIGHTS
# ================================================================

v12_model_q1 = CenterNetResNet101(
    num_classes=80,
    pretrained_backbone=False
).to(DEVICE_Q1)

v12_model_q1.load_state_dict(
    v12_ckpt_q1["model_state_dict"],
    strict=True
)

v12_model_q1.eval()

del v12_ckpt_q1

print("V1.2 model loaded successfully.")


# ================================================================
# 4. VALIDATION DATALOADER
# ================================================================

loader_q1 = DataLoader(
    val_dataset,
    batch_size=8,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
    drop_last=False,
    collate_fn=centernet_collate_fn
)

print(
    "Validation loader batches:",
    len(loader_q1)
)


# ================================================================
# 5. DIAGNOSTIC STORAGE
# ================================================================

wh_abs_q1 = []
off_abs_q1 = []
oracle_iou_q1 = []
true_center_scores_q1 = []

valid_gt_q1 = 0
invalid_gt_wh_q1 = 0

topk_candidates_q1 = 0
topk_nonpos_wh_q1 = 0
topk_bad_offset_q1 = 0
topk_scores_q1 = []

images_seen_q1 = 0


# ================================================================
# 6. INFERENCE AND DIAGNOSTICS
# ================================================================

with torch.inference_mode():

    for images_q1, raw_targets_q1 in loader_q1:

        remaining_q1 = (
            MAX_IMAGES_Q1 - images_seen_q1
        )

        if remaining_q1 <= 0:
            break

        if len(raw_targets_q1) > remaining_q1:

            images_q1 = images_q1[:remaining_q1]

            raw_targets_q1 = raw_targets_q1[
                :remaining_q1
            ]

        images_q1 = images_q1.to(
            DEVICE_Q1,
            non_blocking=True
        )

        # --------------------------------------------------------
        # FORWARD PASS
        # --------------------------------------------------------

        with torch.amp.autocast(
            device_type="cuda",
            dtype=torch.bfloat16
        ):

            out_q1 = v12_model_q1(
                images_q1
            )

        out_q1 = {
            key: value.float()
            for key, value in out_q1.items()
        }

        # --------------------------------------------------------
        # HEATMAP LOCAL MAXIMUM
        # --------------------------------------------------------

        peak_q1 = heatmap_local_nms(
            out_q1["heatmap"].sigmoid()
        )

        (
            score_q1,
            ind_q1,
            _,
            _,
            _
        ) = topk_centers(
            peak_q1,
            K=TOPK_Q1
        )

        # --------------------------------------------------------
        # RAW WH AND OFFSET AT PREDICTED CENTERS
        # --------------------------------------------------------

        wh_top_q1 = transpose_and_gather_feature(
            out_q1["wh"],
            ind_q1
        )

        off_top_q1 = transpose_and_gather_feature(
            out_q1["offset"],
            ind_q1
        )

        topk_candidates_q1 += int(
            wh_top_q1.shape[0] *
            wh_top_q1.shape[1]
        )

        topk_nonpos_wh_q1 += int(
            (wh_top_q1 <= 0)
            .any(dim=-1)
            .sum()
            .item()
        )

        topk_bad_offset_q1 += int(
            (
                (off_top_q1 < 0)
                |
                (off_top_q1 >= 1)
            )
            .any(dim=-1)
            .sum()
            .item()
        )

        topk_scores_q1.extend(
            score_q1
            .detach()
            .cpu()
            .reshape(-1)
            .tolist()
        )

        H_q1, W_q1 = out_q1[
            "heatmap"
        ].shape[-2:]

        # --------------------------------------------------------
        # DIAGNOSTIC USING GROUND-TRUTH CENTERS
        # --------------------------------------------------------

        for batch_idx_q1, target_q1 in enumerate(
            raw_targets_q1
        ):

            box_q1 = target_q1[
                "boxes"
            ].to(DEVICE_Q1).float()

            lab_q1 = target_q1[
                "labels"
            ].to(DEVICE_Q1).long()

            if len(box_q1) == 0:
                continue

            # Ground truth width / height in feature cells
            gt_wh_q1 = (
                box_q1[:, 2:4]
                - box_q1[:, 0:2]
            ) / STRIDE_Q1

            # Ground truth center in feature coordinates
            gt_center_q1 = (
                box_q1[:, 0:2]
                + box_q1[:, 2:4]
            ) / (2 * STRIDE_Q1)

            ij_q1 = torch.floor(
                gt_center_q1
            ).long()

            good_q1 = (
                (gt_wh_q1[:, 0] > 0.25)
                &
                (gt_wh_q1[:, 1] > 0.25)
                &
                (ij_q1[:, 0] >= 0)
                &
                (ij_q1[:, 0] < W_q1)
                &
                (ij_q1[:, 1] >= 0)
                &
                (ij_q1[:, 1] < H_q1)
                &
                (lab_q1 >= 0)
                &
                (lab_q1 < 80)
            )

            if not bool(
                good_q1.any().item()
            ):
                continue

            gt_wh_q1 = gt_wh_q1[
                good_q1
            ]

            gt_center_q1 = gt_center_q1[
                good_q1
            ]

            ij_q1 = ij_q1[
                good_q1
            ]

            lab_q1 = lab_q1[
                good_q1
            ]

            gt_box_q1 = box_q1[
                good_q1
            ]

            xx_q1 = ij_q1[:, 0]
            yy_q1 = ij_q1[:, 1]

            # ----------------------------------------------------
            # PREDICTED WH AT GT CENTER
            # ----------------------------------------------------

            pred_wh_q1 = out_q1[
                "wh"
            ][
                batch_idx_q1,
                :,
                yy_q1,
                xx_q1
            ].T

            # ----------------------------------------------------
            # PREDICTED OFFSET AT GT CENTER
            # ----------------------------------------------------

            pred_off_q1 = out_q1[
                "offset"
            ][
                batch_idx_q1,
                :,
                yy_q1,
                xx_q1
            ].T

            truth_off_q1 = (
                gt_center_q1
                - ij_q1.float()
            )

            # ----------------------------------------------------
            # HEATMAP CONFIDENCE AT GT CENTER
            # ----------------------------------------------------

            gt_score_q1 = out_q1[
                "heatmap"
            ][
                batch_idx_q1,
                lab_q1,
                yy_q1,
                xx_q1
            ].sigmoid()

            # ----------------------------------------------------
            # REGRESSION ERRORS
            # ----------------------------------------------------

            invalid_gt_wh_q1 += int(
                (pred_wh_q1 <= 0)
                .any(dim=1)
                .sum()
                .item()
            )

            valid_gt_q1 += int(
                len(gt_wh_q1)
            )

            wh_abs_q1.extend(
                (
                    pred_wh_q1 - gt_wh_q1
                )
                .abs()
                .reshape(-1)
                .cpu()
                .tolist()
            )

            off_abs_q1.extend(
                (
                    pred_off_q1 - truth_off_q1
                )
                .abs()
                .reshape(-1)
                .cpu()
                .tolist()
            )

            true_center_scores_q1.extend(
                gt_score_q1
                .cpu()
                .tolist()
            )

            # ----------------------------------------------------
            # ORACLE-CENTER BOUNDING BOX
            #
            # Uses known GT center cells, then predicted WH and
            # offset. This measures regression quality only.
            # It is NOT ordinary object detection.
            # ----------------------------------------------------

            center_q1 = (
                ij_q1.float()
                + pred_off_q1
            ) * STRIDE_Q1

            half_q1 = (
                pred_wh_q1.clamp(min=0.0)
                * (STRIDE_Q1 / 2.0)
            )

            pred_box_q1 = torch.cat(
                [
                    center_q1 - half_q1,
                    center_q1 + half_q1
                ],
                dim=1
            )

            # ----------------------------------------------------
            # PAIRWISE ALIGNED IoU
            # ----------------------------------------------------

            lt_q1 = torch.maximum(
                pred_box_q1[:, :2],
                gt_box_q1[:, :2]
            )

            rb_q1 = torch.minimum(
                pred_box_q1[:, 2:],
                gt_box_q1[:, 2:]
            )

            inter_q1 = (
                rb_q1 - lt_q1
            ).clamp(
                min=0
            ).prod(dim=1)

            area_p_q1 = (
                pred_box_q1[:, 2:]
                - pred_box_q1[:, :2]
            ).clamp(
                min=0
            ).prod(dim=1)

            area_g_q1 = (
                gt_box_q1[:, 2:]
                - gt_box_q1[:, :2]
            ).clamp(
                min=0
            ).prod(dim=1)

            union_q1 = (
                area_p_q1
                + area_g_q1
                - inter_q1
            ).clamp(
                min=1e-8
            )

            iou_q1 = (
                inter_q1 / union_q1
            )

            oracle_iou_q1.extend(
                iou_q1.cpu().tolist()
            )

        images_seen_q1 += len(
            raw_targets_q1
        )

        if images_seen_q1 % 200 == 0:

            print(
                "Diagnostic images processed:",
                images_seen_q1
            )


# ================================================================
# 7. VALIDATE DIAGNOSTIC COMPLETION
# ================================================================

assert images_seen_q1 == MAX_IMAGES_Q1

assert valid_gt_q1 > 0

assert topk_candidates_q1 > 0

oracle_np_q1 = np.asarray(
    oracle_iou_q1,
    dtype=np.float64
)

score_np_q1 = np.asarray(
    topk_scores_q1,
    dtype=np.float64
)

gt_score_np_q1 = np.asarray(
    true_center_scores_q1,
    dtype=np.float64
)


# ================================================================
# 8. COMPUTE SUMMARY METRICS
# ================================================================

metrics_q1 = {

    "checkpoint":
        "Balanced80K V1.2 best epoch 10",

    "images_evaluated":
        int(images_seen_q1),

    "valid_gt_objects":
        int(valid_gt_q1),

    "topk_candidates":
        int(topk_candidates_q1),

    # WH statistics
    "pred_wh_nonpositive_at_gt_pct":
        float(
            100.0
            * invalid_gt_wh_q1
            / valid_gt_q1
        ),

    "pred_wh_nonpositive_at_topk_pct":
        float(
            100.0
            * topk_nonpos_wh_q1
            / topk_candidates_q1
        ),

    # Offset statistics
    "topk_offset_outside_0_1_pct":
        float(
            100.0
            * topk_bad_offset_q1
            / topk_candidates_q1
        ),

    # Regression errors in feature-map cells
    "gt_center_wh_mae_feature_cells":
        float(
            np.mean(wh_abs_q1)
        ),

    "gt_center_offset_mae_feature_cells":
        float(
            np.mean(off_abs_q1)
        ),

    # Oracle IoU
    "oracle_gt_center_box_iou_mean":
        float(
            oracle_np_q1.mean()
        ),

    "oracle_gt_center_box_iou_median":
        float(
            np.median(oracle_np_q1)
        ),

    "oracle_gt_center_box_iou_ge_050_pct":
        float(
            100.0
            * np.mean(
                oracle_np_q1 >= 0.50
            )
        ),

    "oracle_gt_center_box_iou_ge_075_pct":
        float(
            100.0
            * np.mean(
                oracle_np_q1 >= 0.75
            )
        ),

    # Heatmap confidence
    "topk_confidence_median":
        float(
            np.median(score_np_q1)
        ),

    "gt_center_class_confidence_median":
        float(
            np.median(gt_score_np_q1)
        )
}


# ================================================================
# 9. PRINT RESULTS
# ================================================================

print("\n" + "=" * 70)
print("QUALITY CHECK 1/4 — FINAL RESULTS")
print("=" * 70)

print(
    "Model: CenterNet Balanced80K V1.2"
)

print(
    "COCO-Val images:",
    images_seen_q1
)

print(
    "Valid GT objects:",
    valid_gt_q1
)

print("-" * 70)

for key_q1, value_q1 in metrics_q1.items():

    if isinstance(value_q1, float):

        print(
            f"{key_q1:<43} "
            f"{value_q1:>10.5f}"
        )

    else:

        print(
            f"{key_q1:<43} "
            f"{value_q1}"
        )


# ================================================================
# 10. SAVE RESULTS
# ================================================================

SAVE_DIR_Q1 = (
    "/content/drive/MyDrive/"
    "CenterNet_Quality_Diagnostics"
)

os.makedirs(
    SAVE_DIR_Q1,
    exist_ok=True
)

SAVE_FILE_Q1 = os.path.join(
    SAVE_DIR_Q1,
    "quality_check_1_v12_wh_offset.json"
)

with open(
    SAVE_FILE_Q1,
    "w",
    encoding="utf-8"
) as f_q1:

    json.dump(
        metrics_q1,
        f_q1,
        indent=2,
        allow_nan=False
    )

print("\nResults saved:")
print(SAVE_FILE_Q1)

print("\nQUALITY CHECK 1/4 COMPLETE")
print("No training performed.")
print("No checkpoint weights modified.")


# ================================================================
# 11. CLEANUP GPU MEMORY
# ================================================================

del v12_model_q1

if torch.cuda.is_available():
    torch.cuda.empty_cache()

print("GPU diagnostic model released.")
