# ==========================================================
# STEP 24 : Training Configuration
# FINAL CLEAN PROTOCOL
# ==========================================================

import copy
import os

import torch
import torch.nn as nn
import torch.optim as optim

from torch.optim.lr_scheduler import (
    MultiStepLR
)


# ==========================================================
# DEPENDENCY CHECK
# ==========================================================

assert "model" in globals(), (
    "Run STEP 11 first."
)

assert "criterion" in globals(), (
    "Run STEP 21 first."
)

assert "support_rank_criterion" in globals(), (
    "Run STEP 22 first."
)

assert "train_loader" in globals(), (
    "Run STEP 18 first."
)


# ==========================================================
# LOCKED TRAINING CONFIG
# ==========================================================

TRAIN_CONFIG = {

    "epochs":
        25,

    "steps_per_epoch":
        800,

    "learning_rate":
        1e-4,

    "backbone_learning_rate":
        1e-5,

    "weight_decay":
        1e-4,

    "gradient_clip":
        0.1,

    "milestones":
        [15, 20],

    "gamma":
        0.1,

    "validation_episodes":
        800,

    "score_threshold":
        0.50,

    "iou_threshold":
        0.50,

    # ----------------------------------------------
    # Pre-training gates
    # ----------------------------------------------

    "tiny_episodes":
        10,

    "tiny_epochs":
        100,

    "short_epochs":
        5,

    "short_steps_per_epoch":
        800,

    # Minimum short-generalization requirement.
    #
    # mAP must:
    # 1. reach at least 0.01
    # 2. reach at least 1.25× its own baseline
    #
    # This is only the GO/NO-GO gate,
    # not the final performance target.
    # ----------------------------------------------

    "short_map50_absolute_floor":
        0.01,

    "short_map50_relative_factor":
        1.25,
}


# ==========================================================
# BACKBONE BN
#
# Parameters remain trainable.
# Only running statistics are frozen.
# ==========================================================

def freeze_backbone_bn_statistics(
    backbone
):

    for module in backbone.modules():

        if isinstance(
            module,
            nn.BatchNorm2d
        ):

            module.eval()


# ==========================================================
# OPTIMIZER / SCHEDULER BUILDER
# ==========================================================

def build_optimizer_and_scheduler(
    target_model
):

    backbone_parameters = [

        parameter

        for parameter
        in target_model.backbone.parameters()

        if parameter.requires_grad
    ]


    backbone_ids = {

        id(parameter)

        for parameter
        in backbone_parameters
    }


    main_parameters = [

        parameter

        for parameter
        in target_model.parameters()

        if (
            parameter.requires_grad
            and
            id(parameter)
            not in backbone_ids
        )
    ]


    optimizer = optim.AdamW(

        [

            {
                "params":
                    main_parameters,

                "lr":
                    TRAIN_CONFIG[
                        "learning_rate"
                    ],

                "name":
                    "main"
            },

            {
                "params":
                    backbone_parameters,

                "lr":
                    TRAIN_CONFIG[
                        "backbone_learning_rate"
                    ],

                "name":
                    "backbone"
            },
        ],

        weight_decay=
            TRAIN_CONFIG[
                "weight_decay"
            ]
    )


    scheduler = MultiStepLR(

        optimizer,

        milestones=
            TRAIN_CONFIG[
                "milestones"
            ],

        gamma=
            TRAIN_CONFIG[
                "gamma"
            ]
    )


    return (
        optimizer,
        scheduler
    )


# ==========================================================
# ONE COMBINED TRAINING FORWARD
#
# IMPORTANT:
# Correct detection + absent wrong support
# use the SAME decoder objects.
# ==========================================================

def compute_combined_training_loss(
    target_model,
    batch,
    dataset,
    epoch,
    step,
    device
):

    support_images = (

        batch[
            "support_images"
        ]
        .to(
            device,
            non_blocking=True
        )
    )


    query_images = (

        batch[
            "query_images"
        ]
        .to(
            device,
            non_blocking=True
        )
    )


    query_targets = (
        move_targets_to_device(

            batch[
                "query_targets"
            ],

            device
        )
    )


    episode_classes = (

        batch[
            "episode_classes"
        ]
    )


    # ======================================================
    # CORRECT SUPPORT FORWARD
    # ======================================================

    (
        outputs,
        extras
    ) = target_model.forward_with_features(

        support_image=
            support_images,

        query_image=
            query_images
    )


    # ======================================================
    # MATCH ON CORRECT-SUPPORT PREDICTIONS
    # ======================================================

    indices = matcher(

        outputs,

        query_targets
    )


    # ======================================================
    # DETECTION LOSS
    # ======================================================

    detection_losses = criterion(

        outputs,

        query_targets,

        indices=
            indices
    )


    # ======================================================
    # ABSENT WRONG SUPPORT FOR EACH BATCH ITEM
    #
    # Current protocol locks B=1,
    # but this remains batch-safe.
    # ======================================================

    wrong_support_list = []
    wrong_classes = []


    for batch_idx in range(
        support_images.shape[0]
    ):

        current_class = int(

            episode_classes[
                batch_idx
            ].item()
        )


        query_image_id = int(

            batch[
                "query_targets"
            ][
                batch_idx
            ][
                "image_id"
            ].item()
        )


        wrong_episode = (
            sample_absent_wrong_support(

                dataset=
                    dataset,

                query_image_id=
                    query_image_id,

                current_class=
                    current_class,

                epoch=
                    epoch,

                step=
                    (
                        int(step)
                        +
                        batch_idx
                    )
            )
        )


        wrong_support_list.append(

            wrong_episode[
                "image"
            ]
        )


        wrong_classes.append(

            wrong_episode[
                "class"
            ]
        )


    wrong_support_images = (

        torch.stack(
            wrong_support_list
        )

        .to(
            device,
            non_blocking=True
        )
    )


    # ======================================================
    # MATCHED-QUERY SUPPORT RANKING
    # ======================================================

    (
        support_rank_loss,
        support_rank_stats
    ) = support_rank_criterion(

        model=
            target_model,

        decoder_objects=
            extras[
                "decoder_objects"
            ],

        correct_similarity=
            extras[
                "support_similarity"
            ],

        wrong_support_images=
            wrong_support_images,

        indices=
            indices
    )


    combined_loss = (

        detection_losses[
            "loss_total"
        ]

        +

        CONFIG[
            "support_rank_weight"
        ]

        *
        support_rank_loss
    )


    if not torch.isfinite(
        combined_loss
    ):

        raise RuntimeError(
            "Combined training loss became NaN/Inf."
        )


    return {

        "outputs":
            outputs,

        "extras":
            extras,

        "targets":
            query_targets,

        "indices":
            indices,

        "detection_losses":
            detection_losses,

        "support_rank_loss":
            support_rank_loss,

        "support_rank_stats":
            support_rank_stats,

        "combined_loss":
            combined_loss,

        "wrong_classes":
            wrong_classes,
    }


# ==========================================================
# CHECKPOINT PATHS
# ==========================================================

os.makedirs(
    CHECKPOINT_DIR,
    exist_ok=True
)


BEST_SOURCE_CHECKPOINT_PATH = (
    os.path.join(

        CHECKPOINT_DIR,

        "meta_detr_final_clean_source_best.pth"
    )
)


LATEST_SOURCE_CHECKPOINT_PATH = (
    os.path.join(

        CHECKPOINT_DIR,

        "meta_detr_final_clean_source_latest.pth"
    )
)


print("=" * 70)
print("STEP 24 : TRAINING CONFIGURATION READY")
print("=" * 70)

print(
    "Official Epochs     :",
    TRAIN_CONFIG[
        "epochs"
    ]
)

print(
    "Steps / Epoch       :",
    TRAIN_CONFIG[
        "steps_per_epoch"
    ]
)

print(
    "Main LR             :",
    TRAIN_CONFIG[
        "learning_rate"
    ]
)

print(
    "Backbone LR         :",
    TRAIN_CONFIG[
        "backbone_learning_rate"
    ]
)

print(
    "Weight Decay        :",
    TRAIN_CONFIG[
        "weight_decay"
    ]
)

print(
    "Gradient Clip       :",
    TRAIN_CONFIG[
        "gradient_clip"
    ]
)

print(
    "Support Rank Weight :",
    CONFIG[
        "support_rank_weight"
    ]
)

print(
    "Support Rank Margin :",
    CONFIG[
        "support_rank_margin"
    ]
)

print("=" * 70)
