# ==========================================================
# STEP 24 — FULL REPLACEMENT
#
# Stabilized Stage-1:
# - main LR       = 5e-5
# - backbone LR   = 5e-6
# - gradient accumulation = 4
# - warmup = 500 optimizer updates
# - cosine decay
# ==========================================================

import gc
import copy

from itertools import (
    islice
)

from tqdm.auto import tqdm


TRAIN_CONFIG = {

    'stage1': {

        'epochs':
            25,

        # Number of optimizer updates, not micro-batches.
        'steps_per_epoch':
            800,

        'main_lr':
            5e-5,

        'backbone_lr':
            5e-6,

        'weight_decay':
            1e-4,

        'gradient_clip':
            0.1,

        'accumulation_steps':
            4,

        'warmup_updates':
            500,

        'warmup_start_factor':
            0.10,

        'min_lr_factor':
            0.10,
    },

    # Fallback only
    'stage2': {

        'main_lr':
            5e-5,

        'backbone_lr':
            5e-6,

        'weight_decay':
            1e-4,

        'gradient_clip':
            0.1,

        'accumulation_steps':
            1,
    },

    'stage3': {

        'main_lr':
            1e-4,

        'weight_decay':
            1e-4,

        'gradient_clip':
            0.1,

        'accumulation_steps':
            1,
    },

    # Locked evaluation
    'score_threshold':
        0.50,

    'primary_iou_threshold':
        0.50,

    'tiny': {

        'num_classes':
            8,

        'episodes_per_class':
            2,

        'epochs':
            50,

        'min_map50_improvement':
            0.05,

        'min_geometry_improvement':
            0.10,
    },

    'short': {

        'epochs':
            5,

        'steps_per_epoch':
            800,

        'absolute_map50_floor':
            0.01,

        'relative_factor':
            1.25,

        'stability_fraction':
            0.80,

        'stability_window':
            3,
    },
}


# ==========================================================
# Initial model snapshot
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
        value
        .detach()
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
        strict=True,
    )

    return reset_model.to(
        CONFIG[
            'device'
        ]
    )


# ==========================================================
# Freeze helpers
# ==========================================================

def set_module_trainable(
    module,
    trainable,
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

    for module in (
        backbone.modules()
    ):

        if isinstance(
            module,
            nn.BatchNorm2d,
        ):
            module.eval()


def apply_freeze_policy(
    target_model,
    stage,
):

    stage = str(
        stage
    ).lower()

    # Default = all trainable.
    for parameter in (
        target_model.parameters()
    ):

        parameter.requires_grad = True

    # ------------------------------------------------------
    # Stage 1
    # ------------------------------------------------------
    if stage == 'stage1':

        pass

    # ------------------------------------------------------
    # Stage 2 fallback
    # ------------------------------------------------------
    elif stage == 'stage2':

        set_module_trainable(
            target_model.backbone,
            False,
        )

        # ResNet layer3
        set_module_trainable(
            target_model
            .backbone
            .body[6],
            True,
        )

        # ResNet layer4
        set_module_trainable(
            target_model
            .backbone
            .body[7],
            True,
        )

    # ------------------------------------------------------
    # Stage 3 CCTV
    # ------------------------------------------------------
    elif stage == 'stage3':

        for parameter in (
            target_model.parameters()
        ):

            parameter.requires_grad = False

        set_module_trainable(
            target_model
            .support_encoder,
            True,
        )

        set_module_trainable(
            target_model
            .relation_module,
            True,
        )

        set_module_trainable(
            target_model
            .detection_head,
            True,
        )

    else:
        raise ValueError(
            f'Unknown stage: {stage}'
        )

    freeze_backbone_bn_statistics(
        target_model.backbone
    )


def prepare_model_for_training(
    target_model
):

    target_model.train()

    # model.train() re-enables BN training mode.
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
# Warmup + cosine scheduler
# ==========================================================

class WarmupCosineScheduler:

    def __init__(
        self,
        optimizer,
        total_updates,
        warmup_updates,
        warmup_start_factor=0.10,
        min_lr_factor=0.10,
    ):

        self.optimizer = optimizer

        self.total_updates = int(
            total_updates
        )

        self.warmup_updates = min(
            int(
                warmup_updates
            ),
            self.total_updates,
        )

        self.warmup_start_factor = float(
            warmup_start_factor
        )

        self.min_lr_factor = float(
            min_lr_factor
        )

        if self.total_updates < 1:
            raise ValueError(
                'total_updates must be >= 1.'
            )

        if not (
            0.0
            <
            self.warmup_start_factor
            <=
            1.0
        ):
            raise ValueError(
                'warmup_start_factor must be in (0,1].'
            )

        if not (
            0.0
            <=
            self.min_lr_factor
            <=
            1.0
        ):
            raise ValueError(
                'min_lr_factor must be in [0,1].'
            )

        self.base_lrs = [
            float(
                group[
                    'lr'
                ]
            )
            for group
            in optimizer.param_groups
        ]

        self.num_updates = 0

        # Start at warmup LR.
        self._apply_factor(
            self._factor_for_update(
                0
            )
        )

    def _factor_for_update(
        self,
        update_index,
    ):

        update_index = int(
            update_index
        )

        # Linear warmup
        if (
            self.warmup_updates > 0
            and
            update_index
            <
            self.warmup_updates
        ):

            progress = (
                update_index
                /
                float(
                    max(
                        self.warmup_updates,
                        1,
                    )
                )
            )

            return (
                self.warmup_start_factor
                +
                (
                    1.0
                    -
                    self.warmup_start_factor
                )
                *
                progress
            )

        # Cosine decay
        decay_updates = max(
            self.total_updates
            -
            self.warmup_updates,
            1,
        )

        decay_progress = (
            update_index
            -
            self.warmup_updates
        ) / float(
            decay_updates
        )

        decay_progress = min(
            max(
                decay_progress,
                0.0,
            ),
            1.0,
        )

        cosine = (
            0.5
            *
            (
                1.0
                +
                math.cos(
                    math.pi
                    *
                    decay_progress
                )
            )
        )

        return (
            self.min_lr_factor
            +
            (
                1.0
                -
                self.min_lr_factor
            )
            *
            cosine
        )

    def _apply_factor(
        self,
        factor,
    ):

        for (
            base_lr,
            group,

        ) in zip(
            self.base_lrs,
            self.optimizer.param_groups,
        ):

            group[
                'lr'
            ] = (
                base_lr
                *
                float(
                    factor
                )
            )

    def step(
        self
    ):

        self.num_updates += 1

        next_update = min(
            self.num_updates,
            self.total_updates,
        )

        self._apply_factor(
            self._factor_for_update(
                next_update
            )
        )

    def state_dict(
        self
    ):

        return {

            'total_updates':
                self.total_updates,

            'warmup_updates':
                self.warmup_updates,

            'warmup_start_factor':
                self.warmup_start_factor,

            'min_lr_factor':
                self.min_lr_factor,

            'base_lrs':
                copy.deepcopy(
                    self.base_lrs
                ),

            'num_updates':
                self.num_updates,
        }

    def load_state_dict(
        self,
        state_dict,
    ):

        self.total_updates = int(
            state_dict[
                'total_updates'
            ]
        )

        self.warmup_updates = int(
            state_dict[
                'warmup_updates'
            ]
        )

        self.warmup_start_factor = float(
            state_dict[
                'warmup_start_factor'
            ]
        )

        self.min_lr_factor = float(
            state_dict[
                'min_lr_factor'
            ]
        )

        self.base_lrs = list(
            state_dict[
                'base_lrs'
            ]
        )

        self.num_updates = int(
            state_dict[
                'num_updates'
            ]
        )

        self._apply_factor(
            self._factor_for_update(
                min(
                    self.num_updates,
                    self.total_updates,
                )
            )
        )


# ==========================================================
# Optimizer
# ==========================================================

def build_optimizer_and_scheduler(
    target_model,
    stage,
    use_scheduler=True,
    total_updates=None,
):

    stage = str(
        stage
    ).lower()

    apply_freeze_policy(
        target_model,
        stage,
    )

    if stage == 'stage1':

        stage_config = (
            TRAIN_CONFIG[
                'stage1'
            ]
        )

        main_lr = (
            stage_config[
                'main_lr'
            ]
        )

        backbone_lr = (
            stage_config[
                'backbone_lr'
            ]
        )

    elif stage == 'stage2':

        stage_config = (
            TRAIN_CONFIG[
                'stage2'
            ]
        )

        main_lr = (
            stage_config[
                'main_lr'
            ]
        )

        backbone_lr = (
            stage_config[
                'backbone_lr'
            ]
        )

    elif stage == 'stage3':

        stage_config = (
            TRAIN_CONFIG[
                'stage3'
            ]
        )

        main_lr = (
            stage_config[
                'main_lr'
            ]
        )

        backbone_lr = None

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
            not in
            backbone_ids
        )
    ]

    parameter_groups = []

    if main_parameters:

        parameter_groups.append(
            {
                'params':
                    main_parameters,

                'lr':
                    main_lr,

                'name':
                    'main',
            }
        )

    if backbone_parameters:

        parameter_groups.append(
            {
                'params':
                    backbone_parameters,

                'lr':
                    backbone_lr,

                'name':
                    'backbone',
            }
        )

    if not parameter_groups:
        raise RuntimeError(
            'No trainable parameters.'
        )

    optimizer = (
        torch.optim.AdamW(

            parameter_groups,

            weight_decay=
                stage_config[
                    'weight_decay'
                ],
        )
    )

    scheduler = None

    if (
        use_scheduler
        and
        stage == 'stage1'
    ):

        if total_updates is None:
            raise ValueError(
                'Stage1 scheduler requires total_updates.'
            )

        scheduler = (
            WarmupCosineScheduler(

                optimizer=
                    optimizer,

                total_updates=
                    int(
                        total_updates
                    ),

                warmup_updates=
                    stage_config[
                        'warmup_updates'
                    ],

                warmup_start_factor=
                    stage_config[
                        'warmup_start_factor'
                    ],

                min_lr_factor=
                    stage_config[
                        'min_lr_factor'
                    ],
            )
        )

    return (
        optimizer,
        scheduler,
    )


# ==========================================================
# Detection loss
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
                'query_targets'
            ],
            device,
        )
    )

    support_images = (
        batch[
            'support_images'
        ]
        .to(
            device,
            non_blocking=True,
        )
    )

    support_padding_masks = (
        batch[
            'support_padding_masks'
        ]
        .to(
            device,
            non_blocking=True,
        )
    )

    query_images = (
        batch[
            'query_images'
        ]
        .to(
            device,
            non_blocking=True,
        )
    )

    query_padding_masks = (
        batch[
            'query_padding_masks'
        ]
        .to(
            device,
            non_blocking=True,
        )
    )

    outputs = (
        target_model(

            support_images,

            query_images,

            support_padding_mask=
                support_padding_masks,

            query_padding_mask=
                query_padding_masks,
        )
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
            'Detection loss contains NaN/Inf.'
        )

    return losses


# ==========================================================
# Train one epoch
#
# max_steps = optimizer updates.
# ==========================================================

def train_detection_epoch(
    target_model,
    loader,
    optimizer,
    max_steps,
    stage,
    description,
    show_progress=True,
    scheduler=None,
    accumulation_steps=None,
):

    max_steps = int(
        max_steps
    )

    if max_steps < 1:
        raise ValueError(
            'max_steps must be >= 1.'
        )

    stage = str(
        stage
    ).lower()

    if accumulation_steps is None:

        accumulation_steps = int(
            TRAIN_CONFIG[
                stage
            ].get(
                'accumulation_steps',
                1,
            )
        )

    accumulation_steps = int(
        accumulation_steps
    )

    if accumulation_steps < 1:
        raise ValueError(
            'accumulation_steps must be >= 1.'
        )

    micro_batches_needed = (
        max_steps
        *
        accumulation_steps
    )

    if (
        len(
            loader
        )
        <
        micro_batches_needed
    ):
        raise RuntimeError(
            'Loader is too short for requested optimizer '
            'updates: need '
            f'{micro_batches_needed} micro-batches, '
            f'have {len(loader)}.'
        )

    prepare_model_for_training(
        target_model
    )

    totals = {
        'loss_cls':
            0.0,

        'loss_bbox':
            0.0,

        'loss_giou':
            0.0,

        'loss_total':
            0.0,
    }

    micro_count = 0
    update_count = 0

    iterator = islice(
        loader,
        micro_batches_needed,
    )

    if show_progress:

        iterator = tqdm(
            iterator,
            total=
                micro_batches_needed,
            desc=
                description,
        )

    trainable_parameters = [

        parameter

        for parameter
        in target_model.parameters()

        if parameter.requires_grad
    ]

    optimizer.zero_grad(
        set_to_none=True
    )

    for batch in iterator:

        losses = (
            compute_detection_training_loss(

                target_model=
                    target_model,

                batch=
                    batch,

                device=
                    CONFIG[
                        'device'
                    ],
            )
        )

        # Mean gradient over the effective episodic batch.
        (
            losses[
                'loss_total'
            ]
            /
            float(
                accumulation_steps
            )
        ).backward()

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

        micro_count += 1

        # Wait until all micro-batches are accumulated.
        if (
            micro_count
            %
            accumulation_steps
            !=
            0
        ):
            continue

        stage_gradient_clip = (
            TRAIN_CONFIG[
                stage
            ][
                'gradient_clip'
            ]
        )

        torch.nn.utils.clip_grad_norm_(

            trainable_parameters,

            max_norm=
                stage_gradient_clip,

            error_if_nonfinite=True,
        )

        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        optimizer.zero_grad(
            set_to_none=True
        )

        update_count += 1

    if micro_count == 0:
        raise RuntimeError(
            'No training micro-batches were processed.'
        )

    if (
        update_count
        !=
        max_steps
    ):
        raise RuntimeError(
            'Unexpected optimizer update count: '
            f'{update_count} != {max_steps}'
        )

    physical_batch_size = int(
        loader.batch_size
        if loader.batch_size
        is not None
        else
        1
    )

    return {

        **{
            key:
                value
                /
                micro_count

            for key, value
            in totals.items()
        },

        'updates':
            update_count,

        'micro_batches':
            micro_count,

        'accumulation_steps':
            accumulation_steps,

        'effective_batch_size':
            physical_batch_size
            *
            accumulation_steps,
    }


# ==========================================================
# Checkpoint paths
# ==========================================================

COCO80_BEST_CHECKPOINT_PATH = (
    os.path.join(
        COCO80_CHECKPOINT_DIR,
        'coco80_meta_best.pth',
    )
)

COCO80_LATEST_CHECKPOINT_PATH = (
    os.path.join(
        COCO80_CHECKPOINT_DIR,
        'coco80_meta_latest.pth',
    )
)


print('=' * 70)
print('STEP 24 : STABILIZED TRAINING HELPERS READY')
print('=' * 70)

print(
    'Stage1 main LR      :',
    TRAIN_CONFIG[
        'stage1'
    ][
        'main_lr'
    ],
)

print(
    'Stage1 backbone LR  :',
    TRAIN_CONFIG[
        'stage1'
    ][
        'backbone_lr'
    ],
)

print(
    'Accumulation steps  :',
    TRAIN_CONFIG[
        'stage1'
    ][
        'accumulation_steps'
    ],
)

print(
    'Effective batch     :',
    TRAIN_CONFIG[
        'stage1'
    ][
        'accumulation_steps'
    ],
)

print(
    'Warmup updates      :',
    TRAIN_CONFIG[
        'stage1'
    ][
        'warmup_updates'
    ],
)

print(
    'Scheduler            : warmup + cosine'
)

print(
    'Stage3 backbone      : frozen'
)

print(
    'Stage3 train         : support/relation/head'
)

print('=' * 70)
