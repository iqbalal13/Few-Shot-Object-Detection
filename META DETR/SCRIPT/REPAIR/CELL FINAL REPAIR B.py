# ==========================================================
# CELL FINAL REPAIR B
# SUPPORT-DEPENDENCE SANITY
#
# Question:
# Does explicit final classifier now react materially
# when support identity changes?
#
# NO TRAINING.
# NO GRADIENT.
#
# Only 32 unseen COCO-Val episodes.
# ==========================================================

import numpy as np
import torch

from tqdm.auto import tqdm


print("=" * 70)
print("CELL FINAL REPAIR B : SUPPORT-DEPENDENCE SANITY")
print("=" * 70)


# ==========================================================
# SAFETY
# ==========================================================

assert (
    "final_model"
    in globals()
), (
    "Run FINAL REPAIR A first."
)


assert (
    "val_dataset"
    in globals()
), (
    "val_dataset is missing."
)


final_model.eval()


NUM_SANITY_EPISODES = min(
    32,
    len(val_dataset)
)


# ==========================================================
# WRONG-CLASS SUPPORT
#
# Same semantic-class swap logic used by previous H1.
# ==========================================================

def final_find_wrong_support(
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

logit_deltas = []
bbox_deltas = []
top10_jaccards = []

correct_max_scores = []
wrong_max_scores = []


# ==========================================================
# SUPPORT SWAP
# ==========================================================

with torch.inference_mode():

    for episode_id in tqdm(

        range(
            NUM_SANITY_EPISODES
        ),

        desc=
            "Final Support Sanity"
    ):

        episode = val_dataset[
            episode_id
        ]


        episode_class = int(

            episode[
                "episode_class"
            ].item()
        )


        correct_support = (

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


        wrong_episode = (

            final_find_wrong_support(

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


        # --------------------------------------------------
        # Same query, correct support
        # --------------------------------------------------

        correct_output = final_model(

            correct_support,

            query_image
        )


        # --------------------------------------------------
        # Same query, wrong support
        # --------------------------------------------------

        wrong_output = final_model(

            wrong_support,

            query_image
        )


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


        # --------------------------------------------------
        # Raw sensitivity
        # --------------------------------------------------

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


        # --------------------------------------------------
        # Scores
        # --------------------------------------------------

        correct_scores = torch.sigmoid(
            correct_logits
        )


        wrong_scores = torch.sigmoid(
            wrong_logits
        )


        correct_max_scores.append(

            correct_scores
            .max()
            .item()
        )


        wrong_max_scores.append(

            wrong_scores
            .max()
            .item()
        )


        # --------------------------------------------------
        # Top-10 ranking overlap
        # --------------------------------------------------

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

            .cpu()

            .tolist()
        )


        wrong_top = set(

            torch.topk(
                wrong_scores,
                k=k
            )

            .indices

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


        top10_jaccards.append(

            len(intersection)
            /
            max(
                len(union),
                1
            )
        )


# ==========================================================
# RESULTS
# ==========================================================

logit_deltas = np.asarray(
    logit_deltas,
    dtype=np.float64
)

bbox_deltas = np.asarray(
    bbox_deltas,
    dtype=np.float64
)

top10_jaccards = np.asarray(
    top10_jaccards,
    dtype=np.float64
)

correct_max_scores = np.asarray(
    correct_max_scores,
    dtype=np.float64
)

wrong_max_scores = np.asarray(
    wrong_max_scores,
    dtype=np.float64
)


mean_logit_delta = float(
    logit_deltas.mean()
)


# Previous locked V3.2 H1 value
V32_H1_LOGIT_DELTA = 0.00001440


improvement_ratio = (

    mean_logit_delta
    /
    V32_H1_LOGIT_DELTA
)


print()
print("=" * 70)
print("FINAL REPAIR SUPPORT SANITY RESULT")
print("=" * 70)


print(
    "Episodes              :",
    NUM_SANITY_EPISODES
)


print("-" * 70)


print(
    f"Mean |Δ logit|        : "
    f"{mean_logit_delta:.8f}"
)


print(
    f"Median |Δ logit|      : "
    f"{np.median(logit_deltas):.8f}"
)


print(
    f"Mean |Δ bbox|         : "
    f"{bbox_deltas.mean():.8f}"
)


print(
    f"Top-10 Jaccard        : "
    f"{top10_jaccards.mean():.8f}"
)


print("-" * 70)


print(
    f"Max score correct     : "
    f"{correct_max_scores.mean():.8f}"
)


print(
    f"Max score wrong       : "
    f"{wrong_max_scores.mean():.8f}"
)


print(
    f"Score gap             : "
    f"{(
        correct_max_scores.mean()
        -
        wrong_max_scores.mean()
    ):.8f}"
)


print("-" * 70)


print(
    f"Previous V3.2 |Δlogit|: "
    f"{V32_H1_LOGIT_DELTA:.8f}"
)


print(
    f"Sensitivity ratio     : "
    f"{improvement_ratio:.2f}x"
)


print("-" * 70)


# ==========================================================
# SANITY DECISION
#
# This is only an architecture sanity,
# NOT evidence of semantic generalization.
# ==========================================================

if (
    improvement_ratio
    >=
    10.0
):

    print(
        "✓ PASS: support identity now materially affects logits."
    )

else:

    print(
        "✗ FAIL: support dependence is still too weak."
    )


print("=" * 70)
print(
    "NOTE: correct support does NOT need to outperform "
    "wrong support yet — model has not been trained."
)
print("=" * 70)
