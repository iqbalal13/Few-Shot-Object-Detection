# ==========================================================
# STEP 29 : Official Training Configuration (V2 LOCKED)
# ==========================================================

import os
import torch
import torch.nn as nn
import torch.optim as optim

from torch.optim.lr_scheduler import (
    MultiStepLR
)


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
        [
            15,
            20
        ],

    "gamma":
        0.1,

    "validation_episodes":
        800,

    "score_threshold":
        0.50,

    "iou_threshold":
        0.50
}


# ==========================================================
# Google Drive
# ==========================================================

try:

    from google.colab import drive


    if not os.path.exists(
        "/content/drive/MyDrive"
    ):

        print(
            "Mounting Google Drive..."
        )


        drive.mount(
            "/content/drive"
        )


except Exception as exc:

    print(
        "Google Drive mount skipped:"
    )

    print(
        exc
    )


# ==========================================================
# SAVE DIR
# ==========================================================

if os.path.exists(
    "/content/drive/MyDrive"
):

    SAVE_DIR = (

        "/content/drive/MyDrive/"
        "META_DETR_CHECKPOINT_"
        "V2_EPISODIC_BINARY"
    )

    CHECKPOINT_STORAGE = (
        "Google Drive"
    )


else:

    SAVE_DIR = (
        "/content/checkpoints"
    )

    CHECKPOINT_STORAGE = (
        "LOCAL COLAB RUNTIME"
    )


    print(
        "WARNING: Google Drive "
        "is not mounted."
    )

    print(
        "Checkpoint will disappear "
        "if runtime is deleted."
    )


os.makedirs(
    SAVE_DIR,
    exist_ok=True
)


# ==========================================================
# Paths
# ==========================================================

BEST_CHECKPOINT_PATH = os.path.join(

    SAVE_DIR,

    "meta_detr_base_v2_"
    "episodic_binary_best.pth"
)


LATEST_CHECKPOINT_PATH = os.path.join(

    SAVE_DIR,

    "meta_detr_base_v2_"
    "episodic_binary_latest.pth"
)


BASE_FINAL_PATH = os.path.join(

    SAVE_DIR,

    "meta_detr_base_v2_"
    "episodic_binary_final.pth"
)


# ==========================================================
# Optimizer Builder
# ==========================================================

def build_optimizer_and_scheduler(
    model
):

    backbone_parameters = [

        parameter

        for parameter
        in model.backbone.parameters()

        if parameter.requires_grad
    ]


    backbone_ids = {

        id(
            parameter
        )

        for parameter
        in backbone_parameters
    }


    other_parameters = [

        parameter

        for parameter
        in model.parameters()

        if (
            parameter.requires_grad

            and

            id(
                parameter
            )
            not in
            backbone_ids
        )
    ]


    optimizer = optim.AdamW(

        [

            {

                "params":
                    backbone_parameters,

                "lr":
                    TRAIN_CONFIG[
                        "backbone_learning_rate"
                    ]
            },


            {

                "params":
                    other_parameters,

                "lr":
                    TRAIN_CONFIG[
                        "learning_rate"
                    ]
            }
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
# Freeze backbone BN running statistics
# ==========================================================

def freeze_backbone_bn_statistics(
    backbone
):

    for module in (
        backbone.modules()
    ):

        if isinstance(

            module,

            nn.modules.batchnorm._BatchNorm
        ):

            module.eval()


# ==========================================================
# Parameter Sanity
# ==========================================================

backbone_parameter_count = sum(

    parameter.numel()

    for parameter
    in model.backbone.parameters()

    if parameter.requires_grad
)


other_parameter_count = sum(

    parameter.numel()

    for name, parameter
    in model.named_parameters()

    if (
        parameter.requires_grad

        and

        not name.startswith(
            "backbone."
        )
    )
)


assert (
    backbone_parameter_count
    >
    0
)


assert (
    other_parameter_count
    >
    0
)


# ==========================================================
# Summary
# ==========================================================

print("=" * 70)

print(
    "STEP 29 : V2 OFFICIAL "
    "TRAINING CONFIGURATION"
)

print("=" * 70)


for key, value in (
    TRAIN_CONFIG.items()
):

    print(
        f"{key:25s}: "
        f"{value}"
    )


print("-" * 70)


print(
    "Checkpoint Storage  :",
    CHECKPOINT_STORAGE
)


print(
    "Checkpoint Directory:",
    SAVE_DIR
)


print("-" * 70)


print(
    "Backbone Parameters :",
    f"{backbone_parameter_count:,}"
)


print(
    "Other Parameters    :",
    f"{other_parameter_count:,}"
)


print("=" * 70)
print("✓ STEP 29 READY")
print("=" * 70)
