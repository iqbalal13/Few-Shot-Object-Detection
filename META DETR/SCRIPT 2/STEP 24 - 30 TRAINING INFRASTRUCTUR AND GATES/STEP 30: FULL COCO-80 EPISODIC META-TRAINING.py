# ==========================================================
# STEP 30 — FULL REPLACEMENT
# FULL COCO-80 EPISODIC META-TRAINING
#
# 25 × 800 episodes = 20,000 episodes
# accumulation 4
# 200 updates/epoch
# 5,000 total optimizer updates
#
# Best checkpoint:
#   Geometry50 primary INTERNAL criterion
#   validation loss tie-breaker
#
# Final source gate:
#   validation loss improved
#   Geometry improved
#   tail Geometry >= 80% best
# ==========================================================


if not COCO80_GENERALIZATION_GATE_PASSED:

    raise RuntimeError(
        'STEP 29 source stability gate failed. '
        'Do not start full source training.'
    )


source_model = (
    make_trial_model()
)


stage1_epochs = int(
    TRAIN_CONFIG[
        'stage1'
    ][
        'epochs'
    ]
)


stage1_episodes_per_epoch = int(
    TRAIN_CONFIG[
        'stage1'
    ][
        'episodes_per_epoch'
    ]
)


stage1_accumulation = int(
    TRAIN_CONFIG[
        'stage1'
    ][
        'accumulation_steps'
    ]
)


source_updates_per_epoch = (
    optimizer_updates_for_episodes(

        stage1_episodes_per_epoch,

        stage1_accumulation,
    )
)


source_total_updates = (
    stage1_epochs
    *
    source_updates_per_epoch
)


assert source_updates_per_epoch == 200
assert source_total_updates == 5000


source_optimizer, source_scheduler = (
    build_optimizer_and_scheduler(

        source_model,

        stage='stage1',

        use_scheduler=True,

        total_updates=
            source_total_updates,
    )
)


# ==========================================================
# INITIAL VALIDATION
# ==========================================================

val_dataset.set_epoch(
    0
)


source_initial_report = (
    evaluate_episodic_model(

        source_model,

        val_loader,

        show_progress=True,
    )
)


if (
    source_initial_report[
        'metrics'
    ][
        'classes_evaluated'
    ]
    !=
    80
):

    raise RuntimeError(
        'Initial full validation does not '
        'cover all 80 classes.'
    )


source_initial_loss = float(

    source_initial_report[
        'mean_loss'
    ][
        'loss_total'
    ]
)


source_initial_geometry = float(

    source_initial_report[
        'metrics'
    ][
        'geometry_recall50'
    ]
)


source_history = []


source_best_report = (
    copy.deepcopy(
        source_initial_report
    )
)


source_best_epoch = 0


# ==========================================================
# CHECKPOINT HELPERS
# ==========================================================

def checkpoint_config_snapshot():

    return {

        key:
            (
                str(
                    value
                )

                if isinstance(
                    value,
                    torch.device,
                )

                else

                copy.deepcopy(
                    value
                )
            )

        for key, value
        in CONFIG.items()
    }


def save_source_checkpoint(
    path,
    epoch,
    target_model,
    optimizer,
    scheduler,
    val_report,
):

    payload = {

        'stage':
            'COCO80_META_TRAINING',

        'epoch':
            int(
                epoch
            ),

        'model_state':
            target_model.state_dict(),

        'optimizer_state':
            optimizer.state_dict(),

        'scheduler_state':
            (
                scheduler.state_dict()

                if scheduler is not None

                else None
            ),

        'val_report':
            copy.deepcopy(
                val_report
            ),

        'model_config':
            checkpoint_config_snapshot(),

        'train_config':
            copy.deepcopy(
                TRAIN_CONFIG
            ),

        'research_protocol':
            copy.deepcopy(
                RESEARCH_PROTOCOL
            ),

        'freeze_policy':
            (
                'Stage1 all model parameters trainable; '
                'ResNet-101 uses lower LR; BN running '
                'statistics frozen; effective episodic '
                'batch=4 by gradient accumulation.'
            ),

        'training_budget':
            {
                'episodes_per_epoch':
                    stage1_episodes_per_epoch,

                'updates_per_epoch':
                    source_updates_per_epoch,

                'epochs':
                    stage1_epochs,

                'total_episodes':
                    (
                        stage1_epochs
                        *
                        stage1_episodes_per_epoch
                    ),

                'total_updates':
                    source_total_updates,
            },

        'preprocessing':
            (
                'aspect-ratio-preserving letterbox '
                '640x640 + padding masks'
            ),
    }


    temporary_path = (
        path
        +
        '.tmp'
    )


    torch.save(
        payload,
        temporary_path,
    )


    os.replace(
        temporary_path,
        path,
    )


# Epoch-0 diagnostic checkpoint.
save_source_checkpoint(

    os.path.join(
        COCO80_CHECKPOINT_DIR,
        'coco80_meta_initial.pth',
    ),

    0,

    source_model,

    source_optimizer,

    source_scheduler,

    source_initial_report,
)


print('=' * 70)
print('STEP 30 : FULL COCO-80 META-TRAINING')
print('=' * 70)

print(
    'Initial val loss :',
    source_initial_loss,
)

print(
    'Initial Geometry :',
    source_initial_geometry,
)

print(
    'Initial P        :',
    source_initial_report[
        'metrics'
    ][
        'precision50'
    ],
)

print(
    'Initial R        :',
    source_initial_report[
        'metrics'
    ][
        'recall50'
    ],
)

print(
    'Episodes/epoch   :',
    stage1_episodes_per_epoch,
)

print(
    'Updates/epoch    :',
    source_updates_per_epoch,
)

print(
    'Total updates    :',
    source_total_updates,
)

print('=' * 70)


# ==========================================================
# FULL TRAINING
# ==========================================================

for epoch in range(
    1,
    stage1_epochs + 1,
):

    train_dataset.set_epoch(
        epoch
    )


    train_stats = (
        train_detection_epoch(

            target_model=
                source_model,

            loader=
                train_loader,

            optimizer=
                source_optimizer,

            max_episodes=
                stage1_episodes_per_epoch,

            stage=
                'stage1',

            description=(
                f'COCO80 '
                f'{epoch}/{stage1_epochs}'
            ),

            show_progress=True,

            scheduler=
                source_scheduler,
        )
    )


    val_dataset.set_epoch(
        0
    )


    val_report = (
        evaluate_episodic_model(

            source_model,

            val_loader,

            show_progress=True,
        )
    )


    if (
        val_report[
            'metrics'
        ][
            'classes_evaluated'
        ]
        !=
        80
    ):

        raise RuntimeError(
            'Full validation does not '
            'cover all 80 classes.'
        )


    current_lrs = {

        group.get(
            'name',
            str(
                index
            ),
        ):
            float(
                group[
                    'lr'
                ]
            )

        for index, group
        in enumerate(
            source_optimizer
            .param_groups
        )
    }


    source_history.append(
        {

            'epoch':
                epoch,

            'train':
                copy.deepcopy(
                    train_stats
                ),

            'val':
                copy.deepcopy(
                    val_report
                ),

            'learning_rates':
                copy.deepcopy(
                    current_lrs
                ),

            'scheduler_updates':
                int(
                    source_scheduler
                    .num_updates
                ),
        }
    )


    # INTERNAL checkpoint selection:
    # 1. Geometry50 highest
    # 2. lower validation loss
    current_rank = (

        val_report[
            'metrics'
        ][
            'geometry_recall50'
        ],

        -val_report[
            'mean_loss'
        ][
            'loss_total'
        ],
    )


    best_rank = (

        source_best_report[
            'metrics'
        ][
            'geometry_recall50'
        ],

        -source_best_report[
            'mean_loss'
        ][
            'loss_total'
        ],
    )


    improved = bool(
        current_rank
        >
        best_rank
    )


    if improved:

        source_best_report = (
            copy.deepcopy(
                val_report
            )
        )

        source_best_epoch = (
            epoch
        )


        save_source_checkpoint(

            COCO80_BEST_CHECKPOINT_PATH,

            epoch,

            source_model,

            source_optimizer,

            source_scheduler,

            val_report,
        )


    # Always save latest.
    save_source_checkpoint(

        COCO80_LATEST_CHECKPOINT_PATH,

        epoch,

        source_model,

        source_optimizer,

        source_scheduler,

        val_report,
    )


    print(

        f'Epoch {epoch:02d}',

        '| train',
        round(
            train_stats[
                'loss_total'
            ],
            4,
        ),

        '| val-loss',
        round(
            val_report[
                'mean_loss'
            ][
                'loss_total'
            ],
            4,
        ),

        '| P',
        round(
            val_report[
                'metrics'
            ][
                'precision50'
            ],
            4,
        ),

        '| R',
        round(
            val_report[
                'metrics'
            ][
                'recall50'
            ],
            4,
        ),

        '| geo',
        round(
            val_report[
                'metrics'
            ][
                'geometry_recall50'
            ],
            4,
        ),

        '| update',
        source_scheduler
        .num_updates,

        '| lr',
        f"{current_lrs.get('main', float('nan')):.2e}",

        '| BEST'
        if improved
        else
        '',
    )


# ==========================================================
# FINAL BUDGET SANITY
# ==========================================================

if (
    source_scheduler
    .num_updates
    !=
    source_total_updates
):

    raise RuntimeError(
        'Full scheduler update count mismatch: '
        f'{source_scheduler.num_updates} '
        f'!= {source_total_updates}'
    )


# ==========================================================
# FINAL SOURCE STABILITY
# ==========================================================

if source_best_epoch <= 0:

    raise RuntimeError(
        'Full COCO-80 training did not improve '
        'over initialization.'
    )


if not os.path.isfile(
    COCO80_BEST_CHECKPOINT_PATH
):

    raise RuntimeError(
        'Best COCO-80 checkpoint '
        'was not created.'
    )


source_min_val_loss = min(

    item[
        'val'
    ][
        'mean_loss'
    ][
        'loss_total'
    ]

    for item
    in source_history
)


source_loss_improved = bool(
    source_min_val_loss
    <
    source_initial_loss
)


source_best_geometry = max(

    item[
        'val'
    ][
        'metrics'
    ][
        'geometry_recall50'
    ]

    for item
    in source_history
)


source_geometry_improved = bool(
    source_best_geometry
    >
    source_initial_geometry
)


window = min(

    TRAIN_CONFIG[
        'short'
    ][
        'stability_window'
    ],

    len(
        source_history
    ),
)


if window < 1:

    raise RuntimeError(
        'No full-training history.'
    )


last_window = (
    source_history[
        -window:
    ]
)


full_tail_geometry = np.asarray(
    [

        item[
            'val'
        ][
            'metrics'
        ][
            'geometry_recall50'
        ]

        for item
        in last_window
    ],

    dtype=np.float64,
)


source_tail_geometry_mean = float(
    full_tail_geometry.mean()
)


source_geometry_stability_ratio = (

    source_tail_geometry_mean

    /

    max(
        source_best_geometry,
        1e-12,
    )
)


source_stable_tail = bool(

    source_geometry_stability_ratio

    >=

    TRAIN_CONFIG[
        'short'
    ][
        'stability_fraction'
    ]
)


source_pr_alive = bool(

    source_best_report[
        'metrics'
    ][
        'precision50'
    ]
    >
    0.0

    and

    source_best_report[
        'metrics'
    ][
        'recall50'
    ]
    >
    0.0
)


COCO80_META_TRAINING_COMPLETE = bool(

    source_best_epoch
    >
    0

    and

    source_loss_improved

    and

    source_geometry_improved

    and

    source_stable_tail
)


print('=' * 70)
print('STEP 30 COMPLETE : COCO-80 META-TRAINING')
print('=' * 70)

print(
    'Initial loss       :',
    source_initial_loss,
)

print(
    'Best observed loss :',
    source_min_val_loss,
)

print(
    'Loss improved      :',
    source_loss_improved,
)

print(
    'Initial Geometry   :',
    source_initial_geometry,
)

print(
    'Best Geometry      :',
    source_best_geometry,
)

print(
    'Geometry improved  :',
    source_geometry_improved,
)

print(
    'Tail Geometry      :',
    full_tail_geometry.tolist(),
)

print(
    'Tail Geo mean      :',
    source_tail_geometry_mean,
)

print(
    'Geo stability ratio:',
    source_geometry_stability_ratio,
)

print(
    'Stable >=80% best  :',
    source_stable_tail,
)

print(
    'Best Precision     :',
    source_best_report[
        'metrics'
    ][
        'precision50'
    ],
)

print(
    'Best Recall        :',
    source_best_report[
        'metrics'
    ][
        'recall50'
    ],
)

print(
    'P/R non-zero       :',
    source_pr_alive,
    '(diagnostic at generic source gate)',
)

print(
    'SOURCE STABLE      :',
    COCO80_META_TRAINING_COMPLETE,
)

print(
    'Best epoch         :',
    source_best_epoch,
)

print(
    'Best checkpoint    :',
    COCO80_BEST_CHECKPOINT_PATH,
)

print('=' * 70)


if not COCO80_META_TRAINING_COMPLETE:

    raise RuntimeError(
        'Full COCO-80 run is still unstable. '
        'Do NOT proceed to person/CCTV yet.'
    )


print(
    'NEXT: COCO-Val PERSON-only readiness gate.'
)

print(
    'COCO-person specialization remains FALLBACK ONLY.'
)
