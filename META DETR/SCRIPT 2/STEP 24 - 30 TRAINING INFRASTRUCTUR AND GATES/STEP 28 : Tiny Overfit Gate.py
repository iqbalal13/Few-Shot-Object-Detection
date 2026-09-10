# ==========================================================
# STEP 28 : Tiny Overfit Gate
#
# FINAL LOCKED GATE:
# - 10 FIXED training episodes
# - 100 tiny epochs
# - FULL objective:
#
#   Ltotal = Ldet + lambda_support * Lsupport
#
# - Correct class-aware evaluator from STEP 25
#
# Official `model` remains untouched.
# ==========================================================

import copy
import torch

from torch.utils.data import (
    Subset,
    DataLoader
)


print("=" * 70)
print("STEP 28 : TINY OVERFIT GATE")
print("FULL DETECTION + SUPPORT-RANKING OBJECTIVE")
print("=" * 70)


# ==========================================================
# DEPENDENCIES
# ==========================================================

assert "model" in globals()
assert "train_dataset" in globals()
assert "criterion" in globals()
assert "support_rank_criterion" in globals()
assert "compute_combined_training_loss" in globals()
assert "evaluate_episodic_model" in globals()
assert "TRAIN_CONFIG" in globals()


# ==========================================================
# FIX EXACT SAME 10 EPISODES
# ==========================================================

train_dataset.set_epoch(0)


tiny_indices = list(
    range(
        TRAIN_CONFIG[
            "tiny_episodes"
        ]
    )
)


tiny_subset = Subset(
    train_dataset,
    tiny_indices
)


tiny_loader = DataLoader(

    tiny_subset,

    batch_size=1,

    shuffle=False,

    num_workers=0,

    pin_memory=
        torch.cuda.is_available(),

    collate_fn=
        episodic_collate_fn,

    drop_last=False
)


# ==========================================================
# FRESH MODEL COPY
#
# Official source model is NEVER modified.
# ==========================================================

tiny_model = copy.deepcopy(
    model
).to(
    CONFIG["device"]
)


(
    tiny_optimizer,
    _
) = build_optimizer_and_scheduler(
    tiny_model
)


# ==========================================================
# INITIAL METRICS
# ==========================================================

initial_metrics = evaluate_episodic_model(

    model=
        tiny_model,

    data_loader=
        tiny_loader,

    criterion=
        criterion,

    device=
        CONFIG["device"],

    score_threshold=
        TRAIN_CONFIG[
            "score_threshold"
        ],

    iou_threshold=
        TRAIN_CONFIG[
            "iou_threshold"
        ],

    show_progress=False
)


print("-" * 70)

print(
    "Fixed Episodes       :",
    len(tiny_subset)
)

print(
    "Evaluated Classes    :",
    initial_metrics[
        "num_eval_classes"
    ]
)

print(
    "Tiny Epochs          :",
    TRAIN_CONFIG[
        "tiny_epochs"
    ]
)

print(
    "Support Rank Weight  :",
    CONFIG[
        "support_rank_weight"
    ]
)

print(
    "Support Rank Margin  :",
    CONFIG[
        "support_rank_margin"
    ]
)

print("-" * 70)

print(
    f"Initial Loss         : "
    f"{initial_metrics['loss']:.6f}"
)

print(
    f"Initial mAP50        : "
    f"{initial_metrics['episodic_map50']:.6f}"
)

print(
    f"Initial Mean Best IoU: "
    f"{initial_metrics['mean_best_iou']:.6f}"
)

print("=" * 70)


# ==========================================================
# FULL-OBJECTIVE TINY TRAINING
#
# IMPORTANT:
#
# epoch=0 and local_step fixed
# -> same positive episodes
# -> same deterministic absent wrong supports
#
# across all tiny epochs.
# ==========================================================

for tiny_epoch in range(

    TRAIN_CONFIG[
        "tiny_epochs"
    ]
):

    tiny_model.train()


    freeze_backbone_bn_statistics(
        tiny_model.backbone
    )


    running_combined = 0.0
    running_detection = 0.0
    running_rank = 0.0

    running_correct_sim = 0.0
    running_wrong_sim = 0.0
    running_margin = 0.0

    actual_steps = 0


    for local_step, batch in enumerate(
        tiny_loader
    ):

        result = compute_combined_training_loss(

            target_model=
                tiny_model,

            batch=
                batch,

            dataset=
                train_dataset,

            # Keep wrong-support sampling fixed.
            epoch=0,

            step=
                local_step,

            device=
                CONFIG["device"]
        )


        combined_loss = (
            result[
                "combined_loss"
            ]
        )


        tiny_optimizer.zero_grad(
            set_to_none=True
        )


        combined_loss.backward()


        torch.nn.utils.clip_grad_norm_(

            tiny_model.parameters(),

            max_norm=
                TRAIN_CONFIG[
                    "gradient_clip"
                ]
        )


        tiny_optimizer.step()


        running_combined += float(
            combined_loss.item()
        )


        running_detection += float(

            result[
                "detection_losses"
            ][
                "loss_total"
            ].item()
        )


        running_rank += float(

            result[
                "support_rank_loss"
            ].item()
        )


        stats = result[
            "support_rank_stats"
        ]


        running_correct_sim += float(

            stats[
                "correct_similarity"
            ]
        )


        running_wrong_sim += float(

            stats[
                "wrong_similarity"
            ]
        )


        running_margin += float(

            stats[
                "observed_margin"
            ]
        )


        actual_steps += 1


    if actual_steps == 0:

        raise RuntimeError(
            "Tiny training completed zero steps."
        )


    # ======================================================
    # REPORT
    # ======================================================

    should_report = (

        tiny_epoch == 0

        or

        (
            tiny_epoch + 1
        )
        % 20
        == 0

        or

        (
            tiny_epoch + 1
            ==
            TRAIN_CONFIG[
                "tiny_epochs"
            ]
        )
    )


    if should_report:

        current_metrics = (
            evaluate_episodic_model(

                model=
                    tiny_model,

                data_loader=
                    tiny_loader,

                criterion=
                    criterion,

                device=
                    CONFIG["device"],

                score_threshold=
                    TRAIN_CONFIG[
                        "score_threshold"
                    ],

                iou_threshold=
                    TRAIN_CONFIG[
                        "iou_threshold"
                    ],

                show_progress=False
            )
        )


        print(

            f"Epoch "
            f"{tiny_epoch+1:3d}/"
            f"{TRAIN_CONFIG['tiny_epochs']} "

            f"| Combined="
            f"{running_combined/actual_steps:.4f} "

            f"| Det="
            f"{running_detection/actual_steps:.4f} "

            f"| Rank="
            f"{running_rank/actual_steps:.4f} "

            f"| Margin="
            f"{running_margin/actual_steps:.4f} "

            f"| Eval="
            f"{current_metrics['loss']:.4f} "

            f"| mAP50="
            f"{current_metrics['episodic_map50']:.4f} "

            f"| MeanIoU="
            f"{current_metrics['mean_best_iou']:.4f} "

            f"| LocR50="
            f"{current_metrics['localization_recall50']:.4f}"
        )


# ==========================================================
# FINAL EVALUATION
# ==========================================================

final_metrics = evaluate_episodic_model(

    model=
        tiny_model,

    data_loader=
        tiny_loader,

    criterion=
        criterion,

    device=
        CONFIG["device"],

    score_threshold=
        TRAIN_CONFIG[
            "score_threshold"
        ],

    iou_threshold=
        TRAIN_CONFIG[
            "iou_threshold"
        ],

    show_progress=False
)


# ==========================================================
# LOCKED TINY GATE
#
# Goal is NOT final-quality AP.
# Goal is simply to prove that the complete objective can
# memorize / learn on the fixed tiny training subset.
# ==========================================================

loss_improved = (

    final_metrics[
        "loss"
    ]

    <

    initial_metrics[
        "loss"
    ]
)


detection_alive = (

    final_metrics[
        "episodic_map50"
    ]
    >
    0.05

    or

    final_metrics[
        "recall50"
    ]
    >
    0.05
)


TINY_GATE_PASSED = bool(

    loss_improved

    and

    detection_alive
)


# ==========================================================
# REPORT
# ==========================================================

print()
print("=" * 70)
print("STEP 28 : TINY OVERFIT RESULT")
print("=" * 70)

print(
    f"Initial Loss        : "
    f"{initial_metrics['loss']:.6f}"
)

print(
    f"Final Loss          : "
    f"{final_metrics['loss']:.6f}"
)

print("-" * 70)

print(
    f"Final mAP50         : "
    f"{final_metrics['episodic_map50']:.6f}"
)

print(
    f"Final Precision@.50 : "
    f"{final_metrics['precision50']:.6f}"
)

print(
    f"Final Recall@.50    : "
    f"{final_metrics['recall50']:.6f}"
)

print("-" * 70)

print(
    f"Mean Best IoU       : "
    f"{final_metrics['mean_best_iou']:.6f}"
)

print(
    f"Localization R@.30 : "
    f"{final_metrics['localization_recall30']:.6f}"
)

print(
    f"Localization R@.50 : "
    f"{final_metrics['localization_recall50']:.6f}"
)

print(
    f"Localization R@.75 : "
    f"{final_metrics['localization_recall75']:.6f}"
)

print("-" * 70)


if TINY_GATE_PASSED:

    print(
        "✓ STEP 28 TINY OVERFIT GATE PASSED"
    )

    print(
        "✓ Full detection + support-ranking objective can learn."
    )

    print(
        "NEXT: STEP 29 Short COCO Generalization Gate."
    )


else:

    print(
        "✗ STEP 28 TINY OVERFIT GATE FAILED"
    )

    print(
        "STOP: Do NOT run STEP 29."
    )


print(
    "✓ official model remains untouched"
)

print("=" * 70)


# ==========================================================
# RELEASE COPY
# ==========================================================

del tiny_model
del tiny_optimizer


if torch.cuda.is_available():

    torch.cuda.empty_cache()


if not TINY_GATE_PASSED:

    raise RuntimeError(
        "STOP: Full-objective tiny overfit failed. "
        "Do NOT run STEP 29."
    )
