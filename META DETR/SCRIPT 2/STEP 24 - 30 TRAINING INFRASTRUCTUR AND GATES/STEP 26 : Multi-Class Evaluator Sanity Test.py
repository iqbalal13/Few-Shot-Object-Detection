# ==========================================================
# STEP 26 : Multi-Class Evaluator Sanity Test
#
# TEST A:
# Explicitly prove that:
#
#   per-class AP = 1.0
#   class-aware mAP = 1.0
#
# while incorrect GLOBAL pooled AP < 1.0
#
# TEST B:
# End-to-end two-class episodic evaluator
# with perfect detections:
#
#   mAP50     = 1.0
#   Precision = 1.0
#   Recall    = 1.0
#   Mean IoU  = 1.0
#
# This test would catch the old STEP 25 bug.
# ==========================================================

import math
import torch
import torch.nn as nn


print("=" * 70)
print("STEP 26 : MULTI-CLASS EVALUATOR SANITY")
print("=" * 70)


# ==========================================================
# TEST A
#
# CLASS-WISE AP MUST BE INDEPENDENT OF SCORE SCALE
# ACROSS DIFFERENT SUPPORT CLASSES.
#
# Class 0:
#   TP = 0.95
#   FP = 0.90
#
# Class 1:
#   TP = 0.60
#   FP = 0.10
#
# Each class individually:
# TP ranks above FP => AP = 1
#
# If globally pooled:
# 0.95 TP(class0)
# 0.90 FP(class0)
# 0.60 TP(class1)
#
# => incorrect pooled AP < 1.
# ==========================================================

perfect_gt_box = torch.tensor(

    [[
        0.50,
        0.50,
        0.20,
        0.20
    ]],

    dtype=torch.float32
)


false_box = torch.tensor(

    [
        0.10,
        0.10,
        0.10,
        0.10
    ],

    dtype=torch.float32
)


prediction_records_by_class = {

    0: [

        {
            "episode_id":
                0,

            "score":
                0.95,

            "box":
                perfect_gt_box[
                    0
                ]
        },

        {
            "episode_id":
                0,

            "score":
                0.90,

            "box":
                false_box
        }
    ],


    1: [

        {
            "episode_id":
                1,

            "score":
                0.60,

            "box":
                perfect_gt_box[
                    0
                ]
        },

        {
            "episode_id":
                1,

            "score":
                0.10,

            "box":
                false_box
        }
    ]
}


gt_boxes_by_class = {

    0: {
        0:
            perfect_gt_box.clone()
    },

    1: {
        1:
            perfect_gt_box.clone()
    }
}


total_gt_by_class = {

    0: 1,
    1: 1
}


(
    correct_classwise_map,
    correct_per_class_ap
) = compute_classwise_map50(

    prediction_records_by_class=
        prediction_records_by_class,

    gt_boxes_by_class=
        gt_boxes_by_class,

    total_gt_by_class=
        total_gt_by_class,

    iou_threshold=
        0.50
)


# ==========================================================
# Simulate OLD BUG:
# merge all predictions from both semantic classes.
# ==========================================================

wrong_global_records = (

    prediction_records_by_class[0]

    +

    prediction_records_by_class[1]
)


wrong_global_gt = {

    0:
        perfect_gt_box.clone(),

    1:
        perfect_gt_box.clone()
}


wrong_global_ap = (
    compute_single_class_ap50(

        prediction_records=
            wrong_global_records,

        episode_gt_boxes=
            wrong_global_gt,

        total_gt=
            2,

        iou_threshold=
            0.50
    )
)


print()
print("-" * 70)
print("TEST A — CROSS-CLASS SCORE POOLING TRAP")
print("-" * 70)

print(
    f"AP50 Class 0       : "
    f"{correct_per_class_ap[0]:.6f}"
)

print(
    f"AP50 Class 1       : "
    f"{correct_per_class_ap[1]:.6f}"
)

print(
    f"Correct class mAP  : "
    f"{correct_classwise_map:.6f}"
)

print(
    f"Wrong pooled AP    : "
    f"{wrong_global_ap:.6f}"
)


assert math.isclose(

    correct_per_class_ap[0],

    1.0,

    abs_tol=1e-6
)


assert math.isclose(

    correct_per_class_ap[1],

    1.0,

    abs_tol=1e-6
)


assert math.isclose(

    correct_classwise_map,

    1.0,

    abs_tol=1e-6
)


assert (
    wrong_global_ap
    <
    1.0
), (
    "Synthetic trap failed: "
    "global pooled AP should be below 1."
)


print(
    "✓ TEST A PASSED:"
)

print(
    "  Semantic classes must NOT share one AP ranking."
)


# ==========================================================
# TEST B
#
# END-TO-END evaluate_episodic_model()
# with TWO semantic classes.
#
# Class identity is encoded in support tensor:
#
# class 0 -> support all zeros
# class 1 -> support all ones
# ==========================================================

def probability_to_logit(
    probability
):

    probability = float(
        probability
    )


    return math.log(

        probability

        /

        (
            1.0
            -
            probability
        )
    )


class PerfectTwoClassDummyDetector(
    nn.Module
):

    def forward(
        self,
        support_images,
        query_images
    ):

        B = (
            query_images.shape[0]
        )


        device = (
            query_images.device
        )


        # Four queries.

        logits = torch.full(

            (
                B,
                4,
                1
            ),

            probability_to_logit(
                0.01
            ),

            dtype=torch.float32,

            device=device
        )


        boxes = torch.tensor(

            [

                [
                    0.50,
                    0.50,
                    0.20,
                    0.20
                ],

                [
                    0.10,
                    0.10,
                    0.10,
                    0.10
                ],

                [
                    0.90,
                    0.10,
                    0.10,
                    0.10
                ],

                [
                    0.10,
                    0.90,
                    0.10,
                    0.10
                ]
            ],

            dtype=torch.float32,

            device=device

        ).unsqueeze(0).expand(

            B,
            -1,
            -1

        ).clone()


        # --------------------------------------------------
        # True prediction always query 0.
        #
        # Different absolute confidence scale per class:
        #
        # class0 -> 0.95
        # class1 -> 0.60
        #
        # This verifies classes do not need cross-class
        # score calibration for class-wise AP.
        # --------------------------------------------------

        for b in range(B):

            support_mean = float(

                support_images[
                    b
                ]
                .mean()
                .item()
            )


            if support_mean < 0.5:

                true_prob = 0.95

            else:

                true_prob = 0.60


            logits[
                b,
                0,
                0
            ] = probability_to_logit(
                true_prob
            )


        return {

            "pred_logits":
                logits,

            "pred_boxes":
                boxes
        }


dummy_model = (
    PerfectTwoClassDummyDetector()

    .to(
        CONFIG["device"]
    )
)


# ==========================================================
# TWO EPISODES
# ==========================================================

dummy_batch_class0 = {

    "support_images":
        torch.zeros(

            1,
            3,
            64,
            64
        ),

    "query_images":
        torch.zeros(

            1,
            3,
            64,
            64
        ),

    "episode_classes":
        torch.tensor(

            [0],

            dtype=torch.long
        ),

    "support_targets":
        [{}],

    "query_targets":
        [{

            "boxes":
                perfect_gt_box.clone(),

            "labels":
                torch.tensor(

                    [0],

                    dtype=torch.long
                ),

            "image_id":
                torch.tensor(

                    100,

                    dtype=torch.long
                )
        }]
}


dummy_batch_class1 = {

    "support_images":
        torch.ones(

            1,
            3,
            64,
            64
        ),

    "query_images":
        torch.zeros(

            1,
            3,
            64,
            64
        ),

    "episode_classes":
        torch.tensor(

            [1],

            dtype=torch.long
        ),

    "support_targets":
        [{}],

    "query_targets":
        [{

            "boxes":
                perfect_gt_box.clone(),

            "labels":
                torch.tensor(

                    [1],

                    dtype=torch.long
                ),

            "image_id":
                torch.tensor(

                    101,

                    dtype=torch.long
                )
        }]
}


dummy_loader = [

    dummy_batch_class0,

    dummy_batch_class1
]


dummy_metrics = (
    evaluate_episodic_model(

        model=
            dummy_model,

        data_loader=
            dummy_loader,

        criterion=
            criterion,

        device=
            CONFIG["device"],

        score_threshold=
            0.50,

        iou_threshold=
            0.50,

        show_progress=
            False
    )
)


print()
print("-" * 70)
print("TEST B — END-TO-END TWO-CLASS EVALUATOR")
print("-" * 70)

print(
    "Evaluated Classes :",
    dummy_metrics[
        "evaluated_classes"
    ]
)

print(
    "Per-Class AP50    :",
    dummy_metrics[
        "per_class_ap50"
    ]
)

print(
    "mAP50             :",
    dummy_metrics[
        "episodic_map50"
    ]
)

print(
    "Micro Precision   :",
    dummy_metrics[
        "precision50"
    ]
)

print(
    "Micro Recall      :",
    dummy_metrics[
        "recall50"
    ]
)

print(
    "Macro Precision   :",
    dummy_metrics[
        "macro_precision50"
    ]
)

print(
    "Macro Recall      :",
    dummy_metrics[
        "macro_recall50"
    ]
)

print(
    "Mean Best IoU     :",
    dummy_metrics[
        "mean_best_iou"
    ]
)


# ==========================================================
# ASSERTIONS
# ==========================================================

assert (
    dummy_metrics[
        "num_eval_classes"
    ]
    ==
    2
)


assert set(

    dummy_metrics[
        "evaluated_classes"
    ]

) == {
    0,
    1
}


assert math.isclose(

    dummy_metrics[
        "per_class_ap50"
    ][0],

    1.0,

    abs_tol=1e-6
)


assert math.isclose(

    dummy_metrics[
        "per_class_ap50"
    ][1],

    1.0,

    abs_tol=1e-6
)


assert math.isclose(

    dummy_metrics[
        "episodic_map50"
    ],

    1.0,

    abs_tol=1e-6
)


assert math.isclose(

    dummy_metrics[
        "precision50"
    ],

    1.0,

    abs_tol=1e-6
)


assert math.isclose(

    dummy_metrics[
        "recall50"
    ],

    1.0,

    abs_tol=1e-6
)


assert math.isclose(

    dummy_metrics[
        "mean_best_iou"
    ],

    1.0,

    abs_tol=1e-6
)


del dummy_model


if torch.cuda.is_available():

    torch.cuda.empty_cache()


print()
print("=" * 70)
print("✓ STEP 26 MULTI-CLASS EVALUATOR SANITY PASSED")
print("=" * 70)

print(
    "✓ Per-class AP calculation validated."
)

print(
    "✓ Cross-class score pooling bug is detected."
)

print(
    "✓ End-to-end episodic evaluator validated."
)

print("=" * 70)
