# ==========================================================
# STEP 24 — FULL REPLACEMENT
#
# LOCKED TRAINING SEMANTICS
#
# Stage-1:
#   800 EPISODES / epoch
#   physical batch = 1
#   accumulation   = 4
#
# therefore:
#   800 episodes / 4 = 200 optimizer updates / epoch
#
# Full:
#   25 epochs
#   20,000 episodes
#   5,000 optimizer updates
#
# Short:
#   5 epochs
#   4,000 episodes
#   1,000 optimizer updates
# ==========================================================

import gc
import copy
import math

from itertools import islice
from tqdm.auto import tqdm


TRAIN_CONFIG = {

    'stage1': {

        'epochs':
            25,

        # IMPORTANT:
        # EPISODES / MICRO-BATCHES, not optimizer updates.
        'episodes_per_epoch':
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

    # ======================================================
    # COCO-person fallback only
    # ======================================================

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

    # ======================================================
    # CCTV adaptation
    # ======================================================

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

    # ======================================================
    # FINAL P/R thresholds
    # ======================================================

    'score_threshold':
        0.50,

    'primary_iou_threshold':
        0.50,

    # ======================================================
    # Tiny learnability test
    # ======================================================

    'tiny': {

        'num_classes':
            8,

        'episodes_per_class':
            2,

        'epochs':
            50,
    },

    # ======================================================
    # Short source gate
    # ======================================================

    'short': {

        'epochs':
            5,

        # Same number of EPISODES per epoch as full Stage-1.
        'episodes_per_epoch':
            800,

        'stability_fraction':
            0.80,

        'stability_window':
            3,
    },
}


# ==========================================================
# Budget helper
# ==========================================================

def optimizer_updates_for_episodes(
    num_episodes,
    accumulation_steps,
):

    num_episodes = int(
        num_episodes
    )

    accumulation_steps = int(
        accumulation_steps
    )

    if num_episodes < 1:
        raise ValueError(
            'num_episodes must be >= 1.'
        )

    if accumulation_steps < 1:
        raise ValueError(
            'accumulation_steps must be >= 1.'
        )

    return int(
        math.ceil(
            num_episodes
            /
            accumulation_steps
        )
    )


STAGE1_UPDATES_PER_EPOCH = (
    optimizer_updates_for_episodes(

        TRAIN_CONFIG[
            'stage1'
        ][
            'episodes_per_epoch'
        ],

        TRAIN_CONFIG[
            'stage1'
        ][
            'accumulation_steps'
        ],
    )
)


STAGE1_TOTAL_UPDATES = (

    TRAIN_CONFIG[
        'stage1'
    ][
        'epochs'
    ]

    *

    STAGE1_UPDATES_PER_EPOCH
)


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

    # Default: trainable.
    for parameter in (
        target_model.parameters()
    ):
        parameter.requires_grad = True


    if stage == 'stage1':

        # All trainable.
        pass


    elif stage == 'stage2':

        # Fallback policy.
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


    elif stage == 'stage3':

        # CCTV:
        # backbone/query encoder/transformer frozen.
        for parameter in (
            target_model.parameters()
        ):
            parameter.requires_grad = False

        set_module_trainable(
            target_model.support_encoder,
            True,
        )

        set_module_trainable(
            target_model.relation_module,
            True,
        )

        set_module_trainable(
            target_model.detection_head,
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

    # model.train() would reactivate BN.
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
                'warmup_start_factor '
                'must be in (0,1].'
            )


        if not (
            0.0
            <=
            self.min_lr_factor
            <=
            1.0
        ):

            raise ValueError(
                'min_lr_factor '
                'must be in [0,1].'
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


        # Linear warmup.
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


        # Cosine decay.
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
            f'Unknown stage: {stage}'
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


    optimizer = torch.optim.AdamW(

        parameter_groups,

        weight_decay=
            stage_config[
                'weight_decay'
            ],
    )


    scheduler = None


    if (
        use_scheduler
        and
        stage == 'stage1'
    ):

        if total_updates is None:

            raise ValueError(
                'Stage1 scheduler requires '
                'total_updates.'
            )


        scheduler = WarmupCosineScheduler(

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


    outputs = target_model(

        support_images,
        query_images,

        support_padding_mask=
            support_padding_masks,

        query_padding_mask=
            query_padding_masks,
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
# max_episodes = number of episodic micro-batches processed.
#
# Scheduler steps ONCE PER OPTIMIZER UPDATE.
# ==========================================================

def train_detection_epoch(
    target_model,
    loader,
    optimizer,
    max_episodes,
    stage,
    description,
    show_progress=True,
    scheduler=None,
    accumulation_steps=None,
):

    max_episodes = int(
        max_episodes
    )


    if max_episodes < 1:

        raise ValueError(
            'max_episodes must be >= 1.'
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


    if (
        len(
            loader
        )
        <
        max_episodes
    ):

        raise RuntimeError(
            'Loader is too short for requested '
            f'episodes: need {max_episodes}, '
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


    trainable_parameters = [

        parameter

        for parameter
        in target_model.parameters()

        if parameter.requires_grad
    ]


    iterator = iter(
        islice(
            loader,
            max_episodes,
        )
    )


    progress = None

    if show_progress:

        progress = tqdm(
            total=max_episodes,
            desc=description,
        )


    processed_episodes = 0
    update_count = 0


    while (
        processed_episodes
        <
        max_episodes
    ):

        remaining = (
            max_episodes
            -
            processed_episodes
        )


        group_size = min(
            accumulation_steps,
            remaining,
        )


        optimizer.zero_grad(
            set_to_none=True
        )


        for _ in range(
            group_size
        ):

            try:

                batch = next(
                    iterator
                )

            except StopIteration as exc:

                raise RuntimeError(
                    'Training loader ended before '
                    'max_episodes was reached.'
                ) from exc


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


            # Average gradient over THIS accumulation group.
            (
                losses[
                    'loss_total'
                ]
                /
                float(
                    group_size
                )
            ).backward()


            for key in totals:

                totals[
                    key
                ] += float(
                    losses[
                        key
                    ]
                    .detach()
                    .item()
                )


            processed_episodes += 1


            if progress is not None:
                progress.update(
                    1
                )


        torch.nn.utils.clip_grad_norm_(

            trainable_parameters,

            max_norm=
                TRAIN_CONFIG[
                    stage
                ][
                    'gradient_clip'
                ],

            error_if_nonfinite=True,
        )


        optimizer.step()


        if scheduler is not None:
            scheduler.step()


        update_count += 1


    if progress is not None:
        progress.close()


    expected_updates = (
        optimizer_updates_for_episodes(
            max_episodes,
            accumulation_steps,
        )
    )


    if (
        update_count
        !=
        expected_updates
    ):

        raise RuntimeError(
            'Unexpected optimizer update count: '
            f'{update_count} != {expected_updates}'
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
                processed_episodes

            for key, value
            in totals.items()
        },

        'episodes':
            processed_episodes,

        'updates':
            update_count,

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

COCO80_BEST_CHECKPOINT_PATH = os.path.join(
    COCO80_CHECKPOINT_DIR,
    'coco80_meta_best.pth',
)

COCO80_LATEST_CHECKPOINT_PATH = os.path.join(
    COCO80_CHECKPOINT_DIR,
    'coco80_meta_latest.pth',
)


# ==========================================================
# Summary
# ==========================================================

print('=' * 70)
print('STEP 24 : STABILIZED TRAINING HELPERS READY')
print('=' * 70)

print(
    'Stage1 main LR          :',
    TRAIN_CONFIG[
        'stage1'
    ][
        'main_lr'
    ],
)

print(
    'Stage1 backbone LR      :',
    TRAIN_CONFIG[
        'stage1'
    ][
        'backbone_lr'
    ],
)

print(
    'Episodes / epoch        :',
    TRAIN_CONFIG[
        'stage1'
    ][
        'episodes_per_epoch'
    ],
)

print(
    'Accumulation steps      :',
    TRAIN_CONFIG[
        'stage1'
    ][
        'accumulation_steps'
    ],
)

print(
    'Optimizer updates/epoch :',
    STAGE1_UPDATES_PER_EPOCH,
)

print(
    'Total Stage1 updates    :',
    STAGE1_TOTAL_UPDATES,
)

print(
    'Total Stage1 episodes   :',
    TRAIN_CONFIG[
        'stage1'
    ][
        'epochs'
    ]
    *
    TRAIN_CONFIG[
        'stage1'
    ][
        'episodes_per_epoch'
    ],
)

print(
    'Warmup updates          :',
    TRAIN_CONFIG[
        'stage1'
    ][
        'warmup_updates'
    ],
)

print(
    'Scheduler               : warmup + cosine'
)

print(
    'Stage3 backbone         : frozen'
)

print(
    'Stage3 train            : support/relation/head'
)

print('=' * 70)
