# ==========================================================
# CELL DIAGNOSIS V3-H3
# ORACLE LOCALIZATION AFTER SHORT TRAINING
#
# Question:
# Even if confidence/ranking is ignored,
# does V3 generate ANY boxes near unseen COCO GT?
# ==========================================================

import numpy as np
import torch

from tqdm.auto import tqdm


print("=" * 70)
print("CELL DIAGNOSIS V3-H3 : ORACLE LOCALIZATION")
print("=" * 70)


short_model.eval()


NUM_EPISODES = min(
    160,
    len(val_dataset)
)


best_ious_all = []


# ==========================================================
# Evaluate
# ==========================================================

with torch.inference_mode():

    for episode_id in tqdm(
        range(NUM_EPISODES),
        desc="V3 H3 Oracle Localization"
    ):

        episode = val_dataset[
            episode_id
        ]


        support_image = (
            episode[
                "support_image"
            ]
            .unsqueeze(0)
            .to(
                CONFIG["device"]
            )
        )


        query_image = (
            episode[
                "query_image"
            ]
            .unsqueeze(0)
            .to(
                CONFIG["device"]
            )
        )


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


        if gt_boxes.shape[0] == 0:
            continue


        # ==================================================
        # Forward with CORRECT support
        # ==================================================

        outputs = short_model(

            support_image,

            query_image
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
        # Pairwise IoU:
        #
        # [100 predicted queries, num_gt]
        # ==================================================

        ious, _ = box_iou(

            pred_xyxy,

            gt_xyxy
        )


        # ==================================================
        # For EACH GT:
        # choose the best possible predicted box
        #
        # Confidence is COMPLETELY ignored.
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
# Array
# ==========================================================

best_ious_all = np.asarray(

    best_ious_all,

    dtype=np.float64
)


assert (
    len(best_ious_all) > 0
), "No GT boxes were evaluated."


# ==========================================================
# Metrics
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
    ).mean()
)


oracle_recall_50 = float(

    (
        best_ious_all
        >=
        0.50
    ).mean()
)


oracle_recall_75 = float(

    (
        best_ious_all
        >=
        0.75
    ).mean()
)


# ==========================================================
# Result
# ==========================================================

print()
print("=" * 70)
print("V3-H3 ORACLE LOCALIZATION RESULT")
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
