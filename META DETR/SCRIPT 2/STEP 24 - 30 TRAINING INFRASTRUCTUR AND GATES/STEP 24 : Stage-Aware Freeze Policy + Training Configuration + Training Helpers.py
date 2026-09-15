# ==========================================================
# STEP 24 : Stage-Aware Freeze Policy
#           + Training Configuration
#           + Training Helpers
# ==========================================================

import gc
import copy

from itertools import (
    islice
)

from torch.optim.lr_scheduler import (
    MultiStepLR
)

from tqdm.auto import tqdm


# ==========================================================
# TRAINING CONFIGURATION
# ==========================================================

TRAIN_CONFIG = {

    # ------------------------------------------------------
    # STAGE 1 — COCO 80
    # ------------------------------------------------------

    "stage1": {

        "epochs": 25,

        "steps_per_epoch": 800,

        "main_lr": 1e-4,

        "backbone_lr": 1e-5,

        "weight_decay": 1e-4,

        "gradient_clip": 0.1,

        "milestones": [
            15,
            20,
        ],

        "gamma": 0.1,
    },

    # ------------------------------------------------------
    # STAGE 2 defaults
    # Actual Stage-2 training appears after STEP 30.
    # ------------------------------------------------------

    "stage2": {

        "main_lr": 5e-5,

        "backbone_lr": 5e-6,

        "weight_decay": 1e-4,

        "gradient_clip": 0.1,
    },

    # ------------------------------------------------------
    # STAGE 3 defaults
    # ------------------------------------------------------

    "stage3": {

        "main_lr": 1e-4,

        "weight_decay": 1e-4,

        "gradient_clip": 0.1,
    },

    # ------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------

    "score_threshold": 0.50,

    "primary_iou_threshold": 0.50,

    # ------------------------------------------------------
    # Tiny multi-class learning test
    # ------------------------------------------------------

    "tiny": {

        "num_classes": 8,

        "episodes_per_class": 2,

        "epochs": 50,

        "min_map50_improvement": 0.05,

        "min_geometry_improvement": 0.10,
    },

    # ------------------------------------------------------
    # Short unseen COCO-Val generalization test
    # ------------------------------------------------------

    "short": {

        "epochs": 5,

        "steps_per_epoch": 800,

        # mAP values are [0,1].
        "absolute_map50_floor": 0.01,

        "relative_factor": 1.25,

        "minimum_map5095_improvement": 1e-4,
    },
}


# ==========================================================
# INITIAL MODEL SNAPSHOT
#
# Every diagnostic experiment starts from EXACT SAME
# initial model weights.
# ==========================================================

model.cpu()

gc.collect()

if torch.cuda.is_available():
    torch.cuda.empty_cache()


MODEL_TEMPLATE = (
    copy.deepcopy(
        model
    )
    .cpu()
)


INITIAL_MODEL_STATE = {

    key:
        value.detach()
        .cpu()
        .clone()

    for key, value
    in model.state_dict().items()
}


def make_trial_model():

    reset_model = (
        copy.deepcopy(
            MODEL_TEMPLATE
        )
    )

    reset_model.load_state_dict(
        INITIAL_MODEL_STATE,
        strict=True
    )

    return reset_model.to(
        CONFIG[
            "device"
        ]
    )


# ==========================================================
# TRAINABLE HELPERS
# ==========================================================

def set_module_trainable(
    module,
    trainable
):

    for parameter in (
        module.parameters()
    ):

        parameter.requires_grad = bool(
            trainable
        )


def freeze_backbone_bn_statistics(
    backbone
):

    # Running statistics frozen.
    #
    # Affine parameters may remain trainable
    # depending on freeze policy.

    for module in (
        backbone.modules()
    ):

        if isinstance(
            module,
            nn.BatchNorm2d
        ):

            module.eval()


def apply_freeze_policy(
    target_model,
    stage
):

    stage = str(
        stage
    ).lower()

    # Start from fully trainable.
    for parameter in (
        target_model.parameters()
    ):

        parameter.requires_grad = True

    # ------------------------------------------------------
    # STAGE 1:
    # Full model trainable.
    # Backbone receives smaller LR.
    # ------------------------------------------------------

    if stage == "stage1":

        pass

    # ------------------------------------------------------
    # STAGE 2:
    #
    # backbone.body indexes:
    #
    # 0 conv1
    # 1 bn1
    # 2 relu
    # 3 maxpool
    # 4 layer1
    # 5 layer2
    # 6 layer3
    # 7 layer4
    # ------------------------------------------------------

    elif stage == "stage2":

        set_module_trainable(
            target_model.backbone,
            False
        )

        set_module_trainable(
            target_model
            .backbone
            .body[6],
            True
        )

        set_module_trainable(
            target_model
            .backbone
            .body[7],
            True
        )

    # ------------------------------------------------------
    # STAGE 3:
    #
    # CCTV few-shot:
    #
    # frozen:
    # backbone
    # query encoder
    # transformer encoder
    # transformer decoder
    #
    # train:
    # support encoder
    # relation module
    # detection head
    # ------------------------------------------------------

    elif stage == "stage3":

        for parameter in (
            target_model.parameters()
        ):

            parameter.requires_grad = False

        set_module_trainable(
            target_model
            .support_encoder,
            True
        )

        set_module_trainable(
            target_model
            .relation_module,
            True
        )

        set_module_trainable(
            target_model
            .detection_head,
            True
        )

    else:

        raise ValueError(
            "Unknown stage: "
            f"{stage}"
        )

    freeze_backbone_bn_statistics(
        target_model.backbone
    )


def prepare_model_for_training(
    target_model
):

    target_model.train()

    # model.train() switches BN back to train,
    # so freeze BN statistics again.
    freeze_backbone_bn_statistics(
        target_model.backbone
    )


def count_trainable_parameters(
    target_model
):

    return sum(
        parameter.numel()
        for parameter
        in target_model.parameters()
        if parameter.requires_grad
    )


# ==========================================================
# OPTIMIZER
# ==========================================================

def build_optimizer_and_scheduler(
    target_model,
    stage,
    use_scheduler=True
):

    stage = str(
        stage
    ).lower()

    apply_freeze_policy(
        target_model,
        stage
    )

    if stage == "stage1":

        stage_config = (
            TRAIN_CONFIG[
                "stage1"
            ]
        )

        main_lr = (
            stage_config[
                "main_lr"
            ]
        )

        backbone_lr = (
            stage_config[
                "backbone_lr"
            ]
        )

        milestones = (
            stage_config[
                "milestones"
            ]
        )

        gamma = (
            stage_config[
                "gamma"
            ]
        )

    elif stage == "stage2":

        stage_config = (
            TRAIN_CONFIG[
                "stage2"
            ]
        )

        main_lr = (
            stage_config[
                "main_lr"
            ]
        )

        backbone_lr = (
            stage_config[
                "backbone_lr"
            ]
        )

        milestones = []

        gamma = 1.0

    elif stage == "stage3":

        stage_config = (
            TRAIN_CONFIG[
                "stage3"
            ]
        )

        main_lr = (
            stage_config[
                "main_lr"
            ]
        )

        backbone_lr = None

        milestones = []

        gamma = 1.0

    else:

        raise ValueError(
            stage
        )

    backbone_parameters = [

        parameter

        for parameter
        in target_model
        .backbone
        .parameters()

        if parameter.requires_grad
    ]

    backbone_ids = {
        id(
            parameter
        )
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
            id(
                parameter
            )
            not in backbone_ids
        )
    ]

    parameter_groups = []

    if main_parameters:

        parameter_groups.append(
            {
                "params":
                    main_parameters,

                "lr":
                    main_lr,

                "name":
                    "main",
            }
        )

    if backbone_parameters:

        parameter_groups.append(
            {
                "params":
                    backbone_parameters,

                "lr":
                    backbone_lr,

                "name":
                    "backbone",
            }
        )

    if not parameter_groups:

        raise RuntimeError(
            "No trainable parameters."
        )

    optimizer = (
        torch.optim.AdamW(

            parameter_groups,

            weight_decay=
                stage_config[
                    "weight_decay"
                ],
        )
    )

    scheduler = None

    if (
        use_scheduler
        and
        milestones
    ):

        scheduler = (
            MultiStepLR(

                optimizer,

                milestones=
                    milestones,

                gamma=
                    gamma,
            )
        )

    return (
        optimizer,
        scheduler
    )


# ==========================================================
# DETECTION TRAINING LOSS
# ==========================================================

def compute_detection_training_loss(
    target_model,
    batch,
    device,
):

    validate_episodic_batch(
        batch
    )

    targets = (
        move_targets_to_device(
            batch[
                "query_targets"
            ],
            device
        )
    )

    support_images = (
        batch[
            "support_images"
        ].to(
            device,
            non_blocking=True
        )
    )

    query_images = (
        batch[
            "query_images"
        ].to(
            device,
            non_blocking=True
        )
    )

    outputs = target_model(
        support_images,
        query_images,
    )

    losses = criterion(
        outputs,
        targets,
    )

    if not all(
        torch.isfinite(
            value
        ).all()
        for value
        in losses.values()
    ):

        raise RuntimeError(
            "Detection loss contains "
            "NaN/Inf."
        )

    return losses


# ==========================================================
# TRAIN ONE EPOCH
# ==========================================================

def train_detection_epoch(
    target_model,
    loader,
    optimizer,
    max_steps,
    stage,
    description,
    show_progress=True,
):

    max_steps = int(
        max_steps
    )

    if max_steps < 1:

        raise ValueError(
            "max_steps must be >= 1."
        )

    prepare_model_for_training(
        target_model
    )

    totals = {

        "loss_cls": 0.0,

        "loss_bbox": 0.0,

        "loss_giou": 0.0,

        "loss_total": 0.0,
    }

    count = 0

    iterator = islice(
        loader,
        max_steps
    )

    if show_progress:

        iterator = tqdm(

            iterator,

            total=min(
                max_steps,
                len(
                    loader
                )
            ),

            desc=description,
        )

    trainable_parameters = [

        parameter

        for parameter
        in target_model.parameters()

        if parameter.requires_grad
    ]

    for batch in iterator:

        optimizer.zero_grad(
            set_to_none=True
        )

        losses = (
            compute_detection_training_loss(

                target_model=
                    target_model,

                batch=
                    batch,

                device=
                    CONFIG[
                        "device"
                    ],
            )
        )

        losses[
            "loss_total"
        ].backward()

        stage_gradient_clip = (
            TRAIN_CONFIG[
                stage
            ][
                "gradient_clip"
            ]
        )

        torch.nn.utils.clip_grad_norm_(

            trainable_parameters,

            max_norm=
                stage_gradient_clip,

            error_if_nonfinite=True,
        )

        optimizer.step()

        for key in totals:

            totals[
                key
            ] += (
                losses[
                    key
                ]
                .detach()
                .item()
            )

        count += 1

    if count == 0:

        raise RuntimeError(
            "No training batches "
            "were processed."
        )

    return {

        **{
            key:
                value
                /
                count

            for key, value
            in totals.items()
        },

        "updates":
            count,
    }


# ==========================================================
# CHECKPOINT PATHS
# ==========================================================

COCO80_BEST_CHECKPOINT_PATH = (
    os.path.join(
        COCO80_CHECKPOINT_DIR,
        "coco80_meta_best.pth"
    )
)


COCO80_LATEST_CHECKPOINT_PATH = (
    os.path.join(
        COCO80_CHECKPOINT_DIR,
        "coco80_meta_latest.pth"
    )
)


print("=" * 70)
print("STEP 24 : STAGE-AWARE TRAINING HELPERS READY")
print("=" * 70)

print(
    "Stage 1 backbone : trainable, LR=1e-5"
)

print(
    "Stage 1 main     : trainable, LR=1e-4"
)

print(
    "Stage 2 backbone : layer3+layer4 only"
)

print(
    "Stage 3 backbone : frozen"
)

print(
    "Stage 3 train    : support/relation/head"
)

print("=" * 70)
