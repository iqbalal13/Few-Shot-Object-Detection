# ==========================================================
# CELL FINAL REPAIR C
# TINY OVERFIT GATE
#
# PURPOSE:
# Verify that FINAL REPAIR can actually learn detection
# on a tiny fixed episodic subset.
#
# - 10 fixed COCO training episodes
# - 100 tiny epochs
# - ~1000 optimizer steps
# - trains a deepcopy of final_model
# - final_model remains untouched
# ==========================================================

import copy
import torch

from torch.utils.data import (
    DataLoader,
    Subset
)


print("=" * 70)
print("CELL FINAL REPAIR C : TINY OVERFIT GATE")
print("=" * 70)


# ==========================================================
# SAFETY
# ==========================================================

assert (
    "final_model" in globals()
), "Run FINAL REPAIR A first."

assert (
    "train_dataset" in globals()
), "train_dataset missing."

assert (
    "criterion" in globals()
), "criterion missing."

assert (
    "evaluate_episodic_model" in globals()
), "Evaluator missing."


FINAL_TINY_CONFIG = {

    "num_episodes": 10,

    "epochs": 100,

    "eval_every": 20,
}


# ==========================================================
# FIXED EPISODES
# ==========================================================

train_dataset.set_epoch(0)


tiny_indices = list(

    range(

        min(

            FINAL_TINY_CONFIG[
                "num_episodes"
            ],

            len(
                train_dataset
            )
        )
    )
)


final_tiny_subset = Subset(

    train_dataset,

    tiny_indices
)


final_tiny_loader = DataLoader(

    final_tiny_subset,

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
# COPY FINAL MODEL
#
# final_model itself remains untouched
# ==========================================================

final_tiny_model = copy.deepcopy(

    final_model

).to(
    CONFIG["device"]
)


final_tiny_optimizer, _ = (

    build_optimizer_and_scheduler(
        final_tiny_model
    )
)


# ==========================================================
# INITIAL METRICS
# ==========================================================

initial_metrics = evaluate_episodic_model(

    model=
        final_tiny_model,

    data_loader=
        final_tiny_loader,

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

    show_progress=
        False
)


print("-" * 70)

print(
    "Fixed Episodes :",
    FINAL_TINY_CONFIG[
        "num_episodes"
    ]
)

print(
    "Tiny Epochs    :",
    FINAL_TINY_CONFIG[
        "epochs"
    ]
)

print("-" * 70)

print(
    "Initial Loss   :",
    f"{initial_metrics['loss']:.6f}"
)

print(
    "Initial mAP50  :",
    f"{initial_metrics['episodic_map50']:.6f}"
)

print(
    "Initial Recall :",
    f"{initial_metrics['recall50']:.6f}"
)


# ==========================================================
# TRAIN
# ==========================================================

for epoch in range(

    FINAL_TINY_CONFIG[
        "epochs"
    ]
):

    final_tiny_model.train()


    freeze_backbone_bn_statistics(

        final_tiny_model.backbone
    )


    running_loss = 0.0


    for batch in final_tiny_loader:

        support_images = (

            batch[
                "support_images"
            ]

            .to(
                CONFIG["device"],
                non_blocking=True
            )
        )


        query_images = (

            batch[
                "query_images"
            ]

            .to(
                CONFIG["device"],
                non_blocking=True
            )
        )


        targets = move_targets_to_device(

            batch[
                "query_targets"
            ],

            CONFIG["device"]
        )


        outputs = final_tiny_model(

            support_images,

            query_images
        )


        loss_dict = criterion(

            outputs,

            targets
        )


        loss = loss_dict[
            "loss_total"
        ]


        if not torch.isfinite(
            loss
        ):

            raise RuntimeError(
                "Final tiny loss became NaN/Inf."
            )


        final_tiny_optimizer.zero_grad(
            set_to_none=True
        )


        loss.backward()


        torch.nn.utils.clip_grad_norm_(

            final_tiny_model.parameters(),

            max_norm=
                TRAIN_CONFIG[
                    "gradient_clip"
                ]
        )


        final_tiny_optimizer.step()


        running_loss += (
            loss.item()
        )


    # ======================================================
    # REPORT
    # ======================================================

    should_report = (

        (
            epoch + 1
        )
        %
        FINAL_TINY_CONFIG[
            "eval_every"
        ]
        ==
        0

        or
        epoch == 0

        or
        epoch + 1
        ==
        FINAL_TINY_CONFIG[
            "epochs"
        ]
    )


    if should_report:

        current_metrics = evaluate_episodic_model(

            model=
                final_tiny_model,

            data_loader=
                final_tiny_loader,

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

            show_progress=
                False
        )


        print(

            f"Epoch {epoch+1:3d}/"
            f"{FINAL_TINY_CONFIG['epochs']} "

            f"| TrainLoss="
            f"{running_loss/len(final_tiny_loader):.4f} "

            f"| EvalLoss="
            f"{current_metrics['loss']:.4f} "

            f"| mAP50="
            f"{current_metrics['episodic_map50']:.4f} "

            f"| P="
            f"{current_metrics['precision50']:.4f} "

            f"| R="
            f"{current_metrics['recall50']:.4f}"
        )


# ==========================================================
# FINAL METRICS
# ==========================================================

final_metrics = evaluate_episodic_model(

    model=
        final_tiny_model,

    data_loader=
        final_tiny_loader,

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

    show_progress=
        False
)


print()
print("=" * 70)
print("FINAL REPAIR TINY OVERFIT RESULT")
print("=" * 70)

print(
    "Initial Loss :",
    f"{initial_metrics['loss']:.6f}"
)

print(
    "Final Loss   :",
    f"{final_metrics['loss']:.6f}"
)

print(
    "Final mAP50  :",
    f"{final_metrics['episodic_map50']:.6f}"
)

print(
    "Final P@0.50 :",
    f"{final_metrics['precision50']:.6f}"
)

print(
    "Final R@0.50 :",
    f"{final_metrics['recall50']:.6f}"
)

print("=" * 70)


# ==========================================================
# GATE
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


if (
    loss_improved
    and
    detection_alive
):

    print(
        "✓ FINAL REPAIR TINY OVERFIT GATE PASSED"
    )

else:

    raise RuntimeError(

        "FINAL REPAIR TINY OVERFIT GATE FAILED. "
        "STOP and inspect output."
    )


print("=" * 70)
print(
    "final_model remains untouched."
)
print("=" * 70)
