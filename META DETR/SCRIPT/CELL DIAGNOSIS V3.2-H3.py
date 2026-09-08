# ==========================================================
# CELL DIAGNOSIS V3.2-H3
# ORACLE LOCALIZATION ON BEST V3.2 MODEL
#
# PURPOSE
# ----------------------------------------------------------
# H1 V3.2 is still not resolved.
# H2 V3.2 meaningfully improved.
#
# Now test localization independently from confidence/ranking.
#
# For EACH GT object:
#   search ALL 100 predicted boxes
#   take the maximum possible IoU
#
# Model:
#   v32_model = BEST V3.2 short epoch
#
# IMPORTANT:
# - NO training
# - NO gradients
# - correct support only
# - confidence is COMPLETELY ignored
# - 160 fixed unseen COCO-Val episodes
# ==========================================================

import numpy as np
import torch

from tqdm.auto import tqdm


print("=" * 70)
print("CELL DIAGNOSIS V3.2-H3 : ORACLE LOCALIZATION")
print("=" * 70)


# ==========================================================
# SAFETY
# ==========================================================

assert (
    "v32_model"
    in globals()
), (
    "v32_model does not exist in current runtime."
)


assert (
    "val_dataset"
    in globals()
), (
    "val_dataset does not exist."
)


assert (
    "box_cxcywh_to_xyxy"
    in globals()
), (
    "box_cxcywh_to_xyxy() is not defined."
)


assert (
    "box_iou"
    in globals()
), (
    "box_iou() is not defined."
)


v32_model.eval()


NUM_EPISODES = min(
    160,
    len(val_dataset)
)


best_ious_all = []


# ==========================================================
# EVALUATE
# ==========================================================

with torch.inference_mode():

    for episode_id in tqdm(

        range(
            NUM_EPISODES
        ),

        desc=
            "V3.2 H3 Oracle Localization"
    ):

        # ==================================================
        # EPISODE
        # ==================================================

        episode = val_dataset[
            episode_id
        ]


        # ==================================================
        # CORRECT SUPPORT
        # ==================================================

        support_image = (

            episode[
                "support_image"
            ]

            .unsqueeze(0)

            .to(
                CONFIG["device"]
            )
        )


        # ==================================================
        # QUERY
        # ==================================================

        query_image = (

            episode[
                "query_image"
            ]

            .unsqueeze(0)

            .to(
                CONFIG["device"]
            )
        )


        # ==================================================
        # GT BOXES
        #
        # normalized cxcywh
        # ==================================================

        gt_boxes = (

            episode[
                "query_target"
            ][
                "boxes"
            ]

            .to(
                CONFIG["device"]
            )
        )


        if (
            gt_boxes.shape[0]
            ==
            0
        ):

            continue


        # ==================================================
        # V3.2 FORWARD
        #
        # CORRECT support only
        # ==================================================

        outputs = (

            v32_model(

                support_image,

                query_image
            )
        )


        pred_boxes = (

            outputs[
                "pred_boxes"
            ][0]
        )


        # ==================================================
        # normalized cxcywh -> normalized xyxy
        # ==================================================

        pred_xyxy = (

            box_cxcywh_to_xyxy(
                pred_boxes
            )
        )


        gt_xyxy = (

            box_cxcywh_to_xyxy(
                gt_boxes
            )
        )


        # ==================================================
        # PAIRWISE IOU
        #
        # Shape:
        # [100 predicted queries, num_gt]
        # ==================================================

        ious, _ = (

            box_iou(

                pred_xyxy,

                gt_xyxy
            )
        )


        # ==================================================
        # ORACLE:
        #
        # For EACH GT object, choose the best predicted box
        # from ALL object queries.
        #
        # Confidence/ranking is ignored completely.
        # ==================================================

        best_per_gt = (

            ious

            .max(
                dim=0
            )

            .values
        )


        best_ious_all.extend(

            best_per_gt

            .detach()

            .cpu()

            .tolist()
        )


# ==========================================================
# ARRAY
# ==========================================================

best_ious_all = np.asarray(

    best_ious_all,

    dtype=np.float64
)


assert (
    len(
        best_ious_all
    )
    >
    0
), (
    "No GT boxes were evaluated."
)


# ==========================================================
# METRICS
# ==========================================================

mean_best_iou = float(

    best_ious_all.mean()
)


median_best_iou = float(

    np.median(
        best_ious_all
    )
)


q25 = float(

    np.quantile(

        best_ious_all,

        0.25
    )
)


q75 = float(

    np.quantile(

        best_ious_all,

        0.75
    )
)


oracle_recall_30 = float(

    (
        best_ious_all
        >=
        0.30
    )

    .mean()
)


oracle_recall_50 = float(

    (
        best_ious_all
        >=
        0.50
    )

    .mean()
)


oracle_recall_75 = float(

    (
        best_ious_all
        >=
        0.75
    )

    .mean()
)


# ==========================================================
# RESULT
# ==========================================================

print()
print("=" * 70)
print("V3.2-H3 ORACLE LOCALIZATION RESULT")
print("=" * 70)


print(
    "Episodes tested        :",
    NUM_EPISODES
)


print(
    "Total GT objects       :",
    len(
        best_ious_all
    )
)


print("-" * 70)


print(
    f"Mean Best IoU          : "
    f"{mean_best_iou:.6f}"
)


print(
    f"Median Best IoU        : "
    f"{median_best_iou:.6f}"
)


print(
    f"IoU Q25                : "
    f"{q25:.6f}"
)


print(
    f"IoU Q75                : "
    f"{q75:.6f}"
)


print("-" * 70)


print(
    f"Oracle Recall@0.30     : "
    f"{oracle_recall_30:.6f}"
)


print(
    f"Oracle Recall@0.50     : "
    f"{oracle_recall_50:.6f}"
)


print(
    f"Oracle Recall@0.75     : "
    f"{oracle_recall_75:.6f}"
)


print("=" * 70)


# ==========================================================
# REFERENCES
# ==========================================================

print()
print("REFERENCE — PREVIOUS H3")
print("-" * 70)


print("V2:")
print(
    "Mean Best IoU      = 0.025763"
)
print(
    "Oracle Recall@0.30 = 0.022663"
)
print(
    "Oracle Recall@0.50 = 0.002833"
)
print(
    "Oracle Recall@0.75 = 0.000000"
)


print("-" * 70)


print("V3:")
print(
    "Mean Best IoU      = 0.014592"
)
print(
    "Oracle Recall@0.30 = 0.005666"
)
print(
    "Oracle Recall@0.50 = 0.002833"
)
print(
    "Oracle Recall@0.75 = 0.000000"
)


print("-" * 70)


print(
    "Question:"
)

print(
    "Did V3.2 representation improvement "
    "translate into better localization?"
)


print("=" * 70)
