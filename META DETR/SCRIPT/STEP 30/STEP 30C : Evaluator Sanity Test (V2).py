# ==========================================================
# STEP 30C : Evaluator Sanity Test (V2)
# ==========================================================

import math

from torch.utils.data import (
    DataLoader,
    Subset
)


EVAL_SANITY_EPISODES = (
    16
)


eval_sanity_subset = Subset(

    val_dataset,

    list(
        range(
            min(
                EVAL_SANITY_EPISODES,
                len(
                    val_dataset
                )
            )
        )
    )
)


eval_sanity_loader = DataLoader(

    eval_sanity_subset,

    batch_size=1,

    shuffle=False,

    num_workers=0,

    pin_memory=
        torch.cuda.is_available(),

    collate_fn=
        episodic_collate_fn,

    drop_last=False
)


eval_sanity_metrics = (
    evaluate_episodic_model(

        model=
            model,

        data_loader=
            eval_sanity_loader,

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


print("=" * 70)
print(
    "STEP 30C : EVALUATOR SANITY TEST"
)
print("=" * 70)


for key in [

    "loss",
    "loss_cls",
    "loss_bbox",
    "loss_giou",
    "episodic_map50",
    "precision50",
    "recall50"
]:

    value = float(

        eval_sanity_metrics[
            key
        ]
    )


    print(
        f"{key:20s}: "
        f"{value:.6f}"
    )


    assert math.isfinite(
        value
    )


for key in [

    "episodic_map50",
    "precision50",
    "recall50"
]:

    value = (

        eval_sanity_metrics[
            key
        ]
    )


    assert (
        0.0
        <=
        value
        <=
        1.0
    )


assert (
    eval_sanity_metrics[
        "num_evaluated_classes"
    ]
    >
    0
)


print(
    "Evaluated Classes   :",
    eval_sanity_metrics[
        "num_evaluated_classes"
    ]
)


print("=" * 70)
print("✓ STEP 30C PASSED")
print("=" * 70)
