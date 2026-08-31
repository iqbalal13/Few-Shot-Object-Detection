# ==========================================================
# CELL DIAGNOSIS V3-H1
# SUPPORT SWAP AFTER SHORT TRAINING
#
# Model tested:
# short_model = V3 after short 5-epoch training
#
# Question:
# Does trained V3 actually USE support?
# ==========================================================

import numpy as np
import torch

from tqdm.auto import tqdm


print("=" * 70)
print("CELL DIAGNOSIS V3-H1 : SUPPORT SWAP")
print("=" * 70)


short_model.eval()

NUM_EPISODES = min(
    160,
    len(val_dataset)
)


# ==========================================================
# Find an episode belonging to another semantic class
# ==========================================================

def find_wrong_support_episode(
    dataset,
    current_index,
    current_class
):

    for offset in range(
        1,
        len(dataset)
    ):

        candidate_index = (
            current_index + offset
        ) % len(dataset)

        candidate = dataset[
            candidate_index
        ]

        candidate_class = int(
            candidate[
                "episode_class"
            ].item()
        )

        if (
            candidate_class
            !=
            current_class
        ):

            return candidate

    raise RuntimeError(
        "Could not find wrong-class support."
    )


# ==========================================================
# Containers
# ==========================================================

correct_predictions = []
wrong_predictions = []

episode_gt_boxes = {}

total_gt = 0

logit_deltas = []
bbox_deltas = []

max_score_correct = []
max_score_wrong = []

top10_jaccards = []


# ==========================================================
# Support Swap
# ==========================================================

with torch.inference_mode():

    for episode_id in tqdm(
        range(NUM_EPISODES),
        desc="V3 H1 Support Swap"
    ):

        episode = val_dataset[
            episode_id
        ]


        episode_class = int(
            episode[
                "episode_class"
            ].item()
        )


        # --------------------------------------------------
        # Correct support
        # --------------------------------------------------

        correct_support = (
            episode[
                "support_image"
            ]
            .unsqueeze(0)
            .to(
                CONFIG["device"]
            )
        )


        # --------------------------------------------------
        # Same fixed query
        # --------------------------------------------------

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
            .detach()
            .cpu()
        )


        episode_gt_boxes[
            episode_id
        ] = gt_boxes


        total_gt += int(
            gt_boxes.shape[0]
        )


        # --------------------------------------------------
        # Wrong-class support
        # --------------------------------------------------

        wrong_episode = (
            find_wrong_support_episode(

                val_dataset,

                episode_id,

                episode_class
            )
        )


        wrong_support = (
            wrong_episode[
                "support_image"
            ]
            .unsqueeze(0)
            .to(
                CONFIG["device"]
            )
        )


        # --------------------------------------------------
        # Forward:
        # same query, different support
        # --------------------------------------------------

        correct_output = short_model(

            correct_support,

            query_image
        )


        wrong_output = short_model(

            wrong_support,

            query_image
        )


        correct_logits = (
            correct_output[
                "pred_logits"
            ][0, :, 0]
        )

        wrong_logits = (
            wrong_output[
                "pred_logits"
            ][0, :, 0]
        )


        correct_boxes = (
            correct_output[
                "pred_boxes"
            ][0]
        )

        wrong_boxes = (
            wrong_output[
                "pred_boxes"
            ][0]
        )


        correct_scores = torch.sigmoid(
            correct_logits
        )

        wrong_scores = torch.sigmoid(
            wrong_logits
        )


        # ==================================================
        # Raw output sensitivity
        # ==================================================

        logit_deltas.append(

            (
                correct_logits
                -
                wrong_logits
            )
            .abs()
            .mean()
            .item()
        )


        bbox_deltas.append(

            (
                correct_boxes
                -
                wrong_boxes
            )
            .abs()
            .mean()
            .item()
        )


        max_score_correct.append(
            correct_scores.max().item()
        )

        max_score_wrong.append(
            wrong_scores.max().item()
        )


        # ==================================================
        # Top-10 query identity overlap
        # ==================================================

        k = min(
            10,
            correct_scores.numel()
        )


        correct_top = set(

            torch.topk(
                correct_scores,
                k=k
            )
            .indices
            .detach()
            .cpu()
            .tolist()
        )


        wrong_top = set(

            torch.topk(
                wrong_scores,
                k=k
            )
            .indices
            .detach()
            .cpu()
            .tolist()
        )


        union = (
            correct_top
            |
            wrong_top
        )


        intersection = (
            correct_top
            &
            wrong_top
        )


        if len(union) > 0:

            top10_jaccards.append(

                len(intersection)
                /
                len(union)
            )


        # ==================================================
        # AP records
        #
        # Store ALL 100 predictions just like ranking AP.
        # ==================================================

        for query_idx in range(
            correct_scores.shape[0]
        ):

            correct_predictions.append({

                "episode_id":
                    episode_id,

                "score":
                    float(
                        correct_scores[
                            query_idx
                        ].item()
                    ),

                "box":
                    correct_boxes[
                        query_idx
                    ]
                    .detach()
                    .cpu()
            })


            wrong_predictions.append({

                "episode_id":
                    episode_id,

                "score":
                    float(
                        wrong_scores[
                            query_idx
                        ].item()
                    ),

                "box":
                    wrong_boxes[
                        query_idx
                    ]
                    .detach()
                    .cpu()
            })


# ==========================================================
# AP50
# ==========================================================

correct_map50 = (
    compute_single_class_ap50(

        correct_predictions,

        episode_gt_boxes,

        total_gt,

        iou_threshold=0.50
    )
)


wrong_map50 = (
    compute_single_class_ap50(

        wrong_predictions,

        episode_gt_boxes,

        total_gt,

        iou_threshold=0.50
    )
)


# ==========================================================
# Results
# ==========================================================

print()
print("=" * 70)
print("V3-H1 SUPPORT SWAP RESULT")
print("=" * 70)

print(
    "Episodes              :",
    NUM_EPISODES
)

print(
    "GT objects            :",
    total_gt
)

print("-" * 70)

print(
    f"Correct Support mAP50 : "
    f"{correct_map50:.8f}"
)

print(
    f"Wrong Support mAP50   : "
    f"{wrong_map50:.8f}"
)

print("-" * 70)

print(
    f"Mean |Δ logit|        : "
    f"{np.mean(logit_deltas):.8f}"
)

print(
    f"Mean |Δ bbox|         : "
    f"{np.mean(bbox_deltas):.8f}"
)

print(
    f"Max score correct     : "
    f"{np.mean(max_score_correct):.8f}"
)

print(
    f"Max score wrong       : "
    f"{np.mean(max_score_wrong):.8f}"
)

print(
    f"Top-10 Jaccard        : "
    f"{np.mean(top10_jaccards):.8f}"
)

print("=" * 70)
