# ==========================================================
# CELL DIAGNOSIS H1
# SUPPORT SWAP TEST
# ==========================================================

import numpy as np
import torch

from collections import defaultdict
from tqdm.auto import tqdm


print("=" * 70)
print("CELL DIAGNOSIS H1 : SUPPORT SWAP")
print("=" * 70)

model.eval()

NUM_EPISODES = min(160, len(val_dataset))
TOP_K = 10


# ----------------------------------------------------------
# Find deterministic wrong-class support
# ----------------------------------------------------------

def find_wrong_support_index(
    dataset,
    current_index,
    current_class
):
    for offset in range(1, len(dataset)):

        candidate_index = (
            current_index + offset
        ) % len(dataset)

        candidate = dataset[candidate_index]

        candidate_class = int(
            candidate["episode_class"].item()
        )

        if candidate_class != current_class:
            return candidate_index

    raise RuntimeError(
        "No different-class support found."
    )


# ----------------------------------------------------------
# Containers
# ----------------------------------------------------------

logit_deltas = []
bbox_deltas = []
max_score_correct = []
max_score_wrong = []
topk_jaccards = []

correct_records = defaultdict(list)
wrong_records = defaultdict(list)

gt_by_class = defaultdict(dict)
gt_count_by_class = defaultdict(int)


# ----------------------------------------------------------
# Diagnostic
# ----------------------------------------------------------

with torch.inference_mode():

    for episode_id in tqdm(
        range(NUM_EPISODES),
        desc="H1 Support Swap"
    ):

        episode = val_dataset[episode_id]

        episode_class = int(
            episode["episode_class"].item()
        )

        query_image = (
            episode["query_image"]
            .unsqueeze(0)
            .to(CONFIG["device"])
        )

        correct_support = (
            episode["support_image"]
            .unsqueeze(0)
            .to(CONFIG["device"])
        )

        gt_boxes = (
            episode["query_target"]["boxes"]
            .detach()
            .cpu()
        )

        # ----------------------------------------------
        # Wrong-class support
        # ----------------------------------------------

        wrong_idx = find_wrong_support_index(
            val_dataset,
            episode_id,
            episode_class
        )

        wrong_episode = val_dataset[wrong_idx]

        wrong_support = (
            wrong_episode["support_image"]
            .unsqueeze(0)
            .to(CONFIG["device"])
        )

        wrong_class = int(
            wrong_episode[
                "episode_class"
            ].item()
        )

        assert wrong_class != episode_class


        # ----------------------------------------------
        # Same query, different support
        # ----------------------------------------------

        out_correct = model(
            correct_support,
            query_image
        )

        out_wrong = model(
            wrong_support,
            query_image
        )


        # ----------------------------------------------
        # Logits / scores
        # ----------------------------------------------

        logits_correct = (
            out_correct["pred_logits"][0, :, 0]
        )

        logits_wrong = (
            out_wrong["pred_logits"][0, :, 0]
        )

        scores_correct = (
            logits_correct.sigmoid()
        )

        scores_wrong = (
            logits_wrong.sigmoid()
        )


        # ----------------------------------------------
        # Boxes
        # ----------------------------------------------

        boxes_correct = (
            out_correct["pred_boxes"][0]
        )

        boxes_wrong = (
            out_wrong["pred_boxes"][0]
        )


        # ----------------------------------------------
        # Output sensitivity
        # ----------------------------------------------

        logit_deltas.append(
            (
                logits_correct
                -
                logits_wrong
            )
            .abs()
            .mean()
            .item()
        )

        bbox_deltas.append(
            (
                boxes_correct
                -
                boxes_wrong
            )
            .abs()
            .mean()
            .item()
        )

        max_score_correct.append(
            scores_correct.max().item()
        )

        max_score_wrong.append(
            scores_wrong.max().item()
        )


        # ----------------------------------------------
        # Top-K query ranking overlap
        # ----------------------------------------------

        k = min(
            TOP_K,
            scores_correct.numel()
        )

        top_correct = set(
            torch.topk(
                scores_correct,
                k=k
            )
            .indices
            .cpu()
            .tolist()
        )

        top_wrong = set(
            torch.topk(
                scores_wrong,
                k=k
            )
            .indices
            .cpu()
            .tolist()
        )

        union = top_correct | top_wrong
        intersection = top_correct & top_wrong

        topk_jaccards.append(
            len(intersection)
            /
            max(len(union), 1)
        )


        # ----------------------------------------------
        # GT
        # ----------------------------------------------

        gt_by_class[
            episode_class
        ][
            episode_id
        ] = gt_boxes

        gt_count_by_class[
            episode_class
        ] += len(gt_boxes)


        # ----------------------------------------------
        # Correct-support predictions
        # ----------------------------------------------

        scores_correct_cpu = (
            scores_correct
            .detach()
            .cpu()
        )

        boxes_correct_cpu = (
            boxes_correct
            .detach()
            .cpu()
        )

        for q in range(
            boxes_correct_cpu.shape[0]
        ):

            correct_records[
                episode_class
            ].append({

                "score":
                    float(
                        scores_correct_cpu[q].item()
                    ),

                "box":
                    boxes_correct_cpu[q],

                "episode_id":
                    episode_id
            })


        # ----------------------------------------------
        # Wrong-support predictions
        # ----------------------------------------------

        scores_wrong_cpu = (
            scores_wrong
            .detach()
            .cpu()
        )

        boxes_wrong_cpu = (
            boxes_wrong
            .detach()
            .cpu()
        )

        for q in range(
            boxes_wrong_cpu.shape[0]
        ):

            wrong_records[
                episode_class
            ].append({

                "score":
                    float(
                        scores_wrong_cpu[q].item()
                    ),

                "box":
                    boxes_wrong_cpu[q],

                "episode_id":
                    episode_id
            })


# ==========================================================
# Compute AP50
# ==========================================================

correct_ap = {}
wrong_ap = {}

for class_label in sorted(
    gt_by_class.keys()
):

    correct_ap[class_label] = (
        compute_single_class_ap50(

            correct_records[
                class_label
            ],

            gt_by_class[
                class_label
            ],

            gt_count_by_class[
                class_label
            ],

            iou_threshold=0.50
        )
    )

    wrong_ap[class_label] = (
        compute_single_class_ap50(

            wrong_records[
                class_label
            ],

            gt_by_class[
                class_label
            ],

            gt_count_by_class[
                class_label
            ],

            iou_threshold=0.50
        )
    )


correct_map50 = float(
    np.mean(
        list(correct_ap.values())
    )
)

wrong_map50 = float(
    np.mean(
        list(wrong_ap.values())
    )
)


# ==========================================================
# Results
# ==========================================================

print()
print("=" * 70)
print("H1 SUPPORT SWAP RESULT")
print("=" * 70)

print(
    "Episodes              :",
    NUM_EPISODES
)

print(
    "Classes               :",
    len(correct_ap)
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
    f"Top-{TOP_K} Jaccard        : "
    f"{np.mean(topk_jaccards):.8f}"
)

print("=" * 70)
