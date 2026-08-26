# ==========================================================
# STEP 31B : TINY OVERFIT GATE
#            MANDATORY BEFORE STEP 32
# ==========================================================

import copy
import torch

from torch.utils.data import (
    DataLoader,
    Subset
)


TINY_OVERFIT_CONFIG = {

    # Fixed episodes
    "num_episodes":
        10,

    # 100 * 10 =
    # ~1000 optimizer steps
    "epochs":
        100,

    "eval_every":
        20,
}


print("=" * 70)
print(
    "STEP 31B : TINY OVERFIT GATE"
)
print("=" * 70)


print(
    "Fixed Episodes :",
    TINY_OVERFIT_CONFIG[
        "num_episodes"
    ]
)


print(
    "Tiny Epochs    :",
    TINY_OVERFIT_CONFIG[
        "epochs"
    ]
)


# ==========================================================
# Episodes fixed
# ==========================================================

train_dataset.set_epoch(
    0
)


tiny_indices = list(

    range(

        min(

            TINY_OVERFIT_CONFIG[
                "num_episodes"
            ],

            len(
                train_dataset
            )
        )
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
# COPY MODEL
#
# official model TIDAK di-training oleh tiny test
# ==========================================================

tiny_model = copy.deepcopy(
    model
).to(
    CONFIG["device"]
)


tiny_optimizer, _ = (
    build_optimizer_and_scheduler(
        tiny_model
    )
)


# ==========================================================
# Initial metrics
# ==========================================================

initial_tiny_metrics = (
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

        show_progress=
            False
    )
)


print("-" * 70)


print(
    "Initial Loss   :",
    f"{initial_tiny_metrics['loss']:.6f}"
)


print(
    "Initial mAP50  :",
    f"{initial_tiny_metrics['episodic_map50']:.6f}"
)


print(
    "Initial Recall :",
    f"{initial_tiny_metrics['recall50']:.6f}"
)


# ==========================================================
# Tiny Training
# ==========================================================

for tiny_epoch in range(

    TINY_OVERFIT_CONFIG[
        "epochs"
    ]
):

    tiny_model.train()


    freeze_backbone_bn_statistics(
        tiny_model.backbone
    )


    running_loss = 0.0


    for batch in tiny_loader:

        support_images = (

            batch[
                "support_images"
            ].to(
                CONFIG["device"],
                non_blocking=True
            )
        )


        query_images = (

            batch[
                "query_images"
            ].to(
                CONFIG["device"],
                non_blocking=True
            )
        )


        targets = (
            move_targets_to_device(

                batch[
                    "query_targets"
                ],

                CONFIG["device"]
            )
        )


        outputs = tiny_model(

            support_images,

            query_images
        )


        loss_dict = criterion(

            outputs,

            targets
        )


        loss = (
            loss_dict[
                "loss_total"
            ]
        )


        if not torch.isfinite(
            loss
        ):

            raise RuntimeError(
                "Tiny-overfit loss "
                "became NaN/Inf."
            )


        tiny_optimizer.zero_grad(
            set_to_none=True
        )


        loss.backward()


        torch.nn.utils.clip_grad_norm_(

            tiny_model.parameters(),

            max_norm=
                TRAIN_CONFIG[
                    "gradient_clip"
                ]
        )


        tiny_optimizer.step()


        running_loss += (
            loss.item()
        )


    should_report = (

        (
            tiny_epoch + 1
        )
        %
        TINY_OVERFIT_CONFIG[
            "eval_every"
        ]
        ==
        0

        or

        tiny_epoch
        ==
        0

        or

        tiny_epoch + 1
        ==
        TINY_OVERFIT_CONFIG[
            "epochs"
        ]
    )


    if should_report:

        tiny_metrics = (
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

                show_progress=
                    False
            )
        )


        print(

            f"Tiny Epoch "
            f"{tiny_epoch+1:3d}/"
            f"{TINY_OVERFIT_CONFIG['epochs']} "

            f"| TrainLoss="
            f"{running_loss/len(tiny_loader):.4f} "

            f"| EvalLoss="
            f"{tiny_metrics['loss']:.4f} "

            f"| mAP50="
            f"{tiny_metrics['episodic_map50']:.4f} "

            f"| P="
            f"{tiny_metrics['precision50']:.4f} "

            f"| R="
            f"{tiny_metrics['recall50']:.4f}"
        )


# ==========================================================
# Final Tiny Metrics
# ==========================================================

final_tiny_metrics = (
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

        show_progress=
            False
    )
)


print(
    "\n"
    +
    "=" * 70
)


print(
    "TINY OVERFIT FINAL RESULT"
)


print("=" * 70)


print(
    "Initial Loss :",
    f"{initial_tiny_metrics['loss']:.6f}"
)


print(
    "Final Loss   :",
    f"{final_tiny_metrics['loss']:.6f}"
)


print(
    "Final mAP50  :",
    f"{final_tiny_metrics['episodic_map50']:.6f}"
)


print(
    "Final P@0.50 :",
    f"{final_tiny_metrics['precision50']:.6f}"
)


print(
    "Final R@0.50 :",
    f"{final_tiny_metrics['recall50']:.6f}"
)


print("=" * 70)


# ==========================================================
# Gate
# ==========================================================

loss_improved = (

    final_tiny_metrics[
        "loss"
    ]

    <

    initial_tiny_metrics[
        "loss"
    ]
)


detection_alive = (

    final_tiny_metrics[
        "episodic_map50"
    ]
    >
    0.05

    or

    final_tiny_metrics[
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
        "✓ TINY OVERFIT GATE PASSED"
    )

    print(
        "You may proceed to STEP 32."
    )


else:

    raise RuntimeError(

        "TINY OVERFIT GATE FAILED. "
        "DO NOT run STEP 32. "
        "Send the tiny-overfit "
        "output for diagnosis."
    )


# ==========================================================
# Free copied model
# ==========================================================

del tiny_model
del tiny_optimizer


if torch.cuda.is_available():

    torch.cuda.empty_cache()
