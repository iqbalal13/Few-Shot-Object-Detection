# ==========================================================
# CELL DIAGNOSIS V3.2-H1
# SUPPORT SWAP ON BEST V3.2 MODEL
#
# PURPOSE
# ----------------------------------------------------------
# H2 V3.2 has improved on unseen COCO-Val.
#
# Now test whether detector OUTPUT actually depends
# on support identity.
#
# SAME protocol as:
#   V3-H1
#   V3.1-H1
#
# Model:
#   v32_model = BEST V3.2 short epoch
#
# NO TRAINING.
# NO GRADIENT.
# ==========================================================

import numpy as np
import torch

from tqdm.auto import tqdm


print("=" * 70)
print("CELL DIAGNOSIS V3.2-H1 : SUPPORT SWAP")
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


v32_model.eval()


NUM_EPISODES = min(
    160,
    len(val_dataset)
)


# ==========================================================
# WRONG SUPPORT
#
# Keep EXACT SAME protocol as previous H1 diagnostics:
# choose support from another semantic class.
# ==========================================================

def v32_h1_find_wrong_support_episode(
    dataset,
    current_index,
    current_class
):

    for offset in range(
        1,
        len(dataset)
    ):

        candidate_index = (
            current_index
            +
            offset
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
# STORAGE
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
# SUPPORT SWAP
# ==========================================================

with torch.inference_mode():

    for episode_id in tqdm(

        range(
            NUM_EPISODES
        ),

        desc=
            "V3.2 H1 Support Swap"
    ):

        # ==================================================
        # EPISODE
        # ==================================================

        episode = val_dataset[
            episode_id
        ]


        episode_class = int(

            episode[
                "episode_class"
            ].item()
        )


        # ==================================================
        # CORRECT SUPPORT
        # ==================================================

        correct_support = (

            episode[
                "support_image"
            ]

            .unsqueeze(0)

            .to(
                CONFIG["device"]
            )
        )


        # ==================================================
        # QUERY — SAME FOR BOTH CONDITIONS
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
        # GT
        # ==================================================

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


        # ==================================================
        # WRONG SUPPORT
        # ==================================================

        wrong_episode = (

            v32_h1_find_wrong_support_episode(

                dataset=
                    val_dataset,

                current_index=
                    episode_id,

                current_class=
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


        # ==================================================
        # CORRECT SUPPORT FORWARD
        # ==================================================

        correct_output = (

            v32_model(

                correct_support,

                query_image
            )
        )


        # ==================================================
        # WRONG SUPPORT FORWARD
        # ==================================================

        wrong_output = (

            v32_model(

                wrong_support,

                query_image
            )
        )


        # ==================================================
        # LOGITS / BOXES
        # ==================================================

        correct_logits = (

            correct_output[
                "pred_logits"
            ][
                0,
                :,
                0
            ]
        )


        wrong_logits = (

            wrong_output[
                "pred_logits"
            ][
                0,
                :,
                0
            ]
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


        correct_scores = (

            torch.sigmoid(
                correct_logits
            )
        )


        wrong_scores = (

            torch.sigmoid(
                wrong_logits
            )
        )


        # ==================================================
        # OUTPUT SENSITIVITY
        # ==================================================

        logit_delta = (

            (
                correct_logits
                -
                wrong_logits
            )

            .abs()

            .mean()

            .item()
        )


        bbox_delta = (

            (
                correct_boxes
                -
                wrong_boxes
            )

            .abs()

            .mean()

            .item()
        )


        logit_deltas.append(
            logit_delta
        )


        bbox_deltas.append(
            bbox_delta
        )


        # ==================================================
        # MAX CONFIDENCE
        # ==================================================

        max_score_correct.append(

            correct_scores
            .max()
            .item()
        )


        max_score_wrong.append(

            wrong_scores
            .max()
            .item()
        )


        # ==================================================
        # TOP-10 QUERY OVERLAP
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


        if (
            len(union)
            >
            0
        ):

            top10_jaccards.append(

                len(intersection)
                /
                len(union)
            )


        # ==================================================
        # AP50 RECORDS
        #
        # ALL object queries retained.
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
# CORRECT SUPPORT AP50
# ==========================================================

correct_map50 = (

    compute_single_class_ap50(

        prediction_records=
            correct_predictions,

        episode_gt_boxes=
            episode_gt_boxes,

        total_gt=
            total_gt,

        iou_threshold=
            0.50
    )
)


# ==========================================================
# WRONG SUPPORT AP50
# ==========================================================

wrong_map50 = (

    compute_single_class_ap50(

        prediction_records=
            wrong_predictions,

        episode_gt_boxes=
            episode_gt_boxes,

        total_gt=
            total_gt,

        iou_threshold=
            0.50
    )
)


# ==========================================================
# ARRAYS
# ==========================================================

logit_deltas = np.asarray(
    logit_deltas,
    dtype=np.float64
)


bbox_deltas = np.asarray(
    bbox_deltas,
    dtype=np.float64
)


max_score_correct = np.asarray(
    max_score_correct,
    dtype=np.float64
)


max_score_wrong = np.asarray(
    max_score_wrong,
    dtype=np.float64
)


top10_jaccards = np.asarray(
    top10_jaccards,
    dtype=np.float64
)


# ==========================================================
# RESULT
# ==========================================================

print()
print("=" * 70)
print("V3.2-H1 SUPPORT SWAP RESULT")
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
    f"{logit_deltas.mean():.8f}"
)


print(
    f"Mean |Δ bbox|         : "
    f"{bbox_deltas.mean():.8f}"
)


print(
    f"Max score correct     : "
    f"{max_score_correct.mean():.8f}"
)


print(
    f"Max score wrong       : "
    f"{max_score_wrong.mean():.8f}"
)


print(
    f"Score gap correct-wrong: "
    f"{(
        max_score_correct.mean()
        -
        max_score_wrong.mean()
    ):.8f}"
)


print(
    f"Top-10 Jaccard        : "
    f"{top10_jaccards.mean():.8f}"
)


print("=" * 70)


# ==========================================================
# REFERENCES
# ==========================================================

print()
print("REFERENCE — PREVIOUS H1")
print("-" * 70)


print("V3:")
print(
    "Mean |Δ logit| = 0.00001367"
)
print(
    "Mean |Δ bbox|  = 0.00000211"
)
print(
    "Top-10 Jaccard = 0.86742424"
)


print("-" * 70)


print("V3.1:")
print(
    "Mean |Δ logit| = 0.00003026"
)
print(
    "Mean |Δ bbox|  = 0.00000307"
)
print(
    "Top-10 Jaccard = 0.88750000"
)


print("-" * 70)


print(
    "V3.2-H2 unseen mean margin "
    "= +0.047036"
)


print(
    "Question: did improved H2 representation "
    "translate into meaningful detector support dependence?"
)


print("=" * 70)
