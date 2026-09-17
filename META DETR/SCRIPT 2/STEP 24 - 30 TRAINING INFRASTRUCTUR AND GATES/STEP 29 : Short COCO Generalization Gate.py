# ==========================================================
# STEP 29 — FULL REPLACEMENT
#
# SHORT COCO-80 STABILITY / GENERALIZATION GATE
#
# 5 epochs
# 800 episodes / epoch
# accumulation = 4
#
# 200 optimizer updates / epoch
# 1,000 optimizer updates executed
#
# Scheduler horizon = FULL 5,000 updates.
#
# HARD:
#   validation loss improves
#   Geometry50 improves
#   last-3 Geometry mean >= 80% best Geometry
#
# P/R = diagnostic here.
# ==========================================================


if not TINY_GATE_PASSED:

    raise RuntimeError(
        'STEP 28 tiny learning gate failed. '
        'Do not run source stability gate.'
    )


short_epochs = int(
    TRAIN_CONFIG[
        'short'
    ][
        'epochs'
    ]
)


short_episodes_per_epoch = int(
    TRAIN_CONFIG[
        'short'
    ][
        'episodes_per_epoch'
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


if (
    short_episodes_per_epoch
    !=
    stage1_episodes_per_epoch
):

    raise RuntimeError(
        'Short and full Stage-1 must use '
        'the same episodes_per_epoch.'
    )


short_updates_per_epoch = (
    optimizer_updates_for_episodes(

        short_episodes_per_epoch,

        stage1_accumulation,
    )
)


short_run_updates = (
    short_epochs
    *
    short_updates_per_epoch
)


full_stage1_total_updates = (
    TRAIN_CONFIG[
        'stage1'
    ][
        'epochs'
    ]
    *
    optimizer_updates_for_episodes(

        stage1_episodes_per_epoch,

        stage1_accumulation,
    )
)


assert short_updates_per_epoch == 200
assert short_run_updates == 1000
assert full_stage1_total_updates == 5000


# ==========================================================
# FIXED VALIDATION
# ==========================================================

val_dataset.set_epoch(
    0
)


short_model = (
    make_trial_model()
)


short_optimizer, short_scheduler = (
    build_optimizer_and_scheduler(

        short_model,

        stage='stage1',

        use_scheduler=True,

        total_updates=
            full_stage1_total_updates,
    )
)


short_initial_report = (
    evaluate_episodic_model(

        short_model,

        val_loader,

        show_progress=True,
    )
)


if (
    short_initial_report[
        'metrics'
    ][
        'classes_evaluated'
    ]
    !=
    80
):

    raise RuntimeError(
        'Validation episodes do not cover '
        'all 80 source classes.'
    )


initial_loss = float(
    short_initial_report[
        'mean_loss'
    ][
        'loss_total'
    ]
)


initial_geometry = float(
    short_initial_report[
        'metrics'
    ][
        'geometry_recall50'
    ]
)


print('=' * 70)
print('STEP 29 : SHORT COCO-80 STABILITY GATE')
print('=' * 70)

print(
    'Short epochs            :',
    short_epochs,
)

print(
    'Episodes / epoch        :',
    short_episodes_per_epoch,
)

print(
    'Optimizer updates/epoch :',
    short_updates_per_epoch,
)

print(
    'Short updates executed  :',
    short_run_updates,
)

print(
    'Scheduler horizon       :',
    full_stage1_total_updates,
)

print(
    'Initial loss            :',
    initial_loss,
)

print(
    'Initial Geometry50      :',
    initial_geometry,
)

print(
    'Initial Precision       :',
    short_initial_report[
        'metrics'
    ][
        'precision50'
    ],
)

print(
    'Initial Recall          :',
    short_initial_report[
        'metrics'
    ][
        'recall50'
    ],
)

print('=' * 70)


short_history = []


short_best_report = (
    copy.deepcopy(
        short_initial_report
    )
)


short_best_epoch = 0


# ==========================================================
# TRAIN
# ==========================================================

for epoch in range(
    1,
    short_epochs + 1,
):

    train_dataset.set_epoch(
        epoch
    )


    train_stats = (
        train_detection_epoch(

            target_model=
                short_model,

            loader=
                train_loader,

            optimizer=
                short_optimizer,

            max_episodes=
                short_episodes_per_epoch,

            stage=
                'stage1',

            description=(
                f'Short COCO80 '
                f'{epoch}/{short_epochs}'
            ),

            show_progress=True,

            scheduler=
                short_scheduler,
        )
    )


    val_dataset.set_epoch(
        0
    )


    val_report = (
        evaluate_episodic_model(

            short_model,

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
            'Validation coverage changed.'
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
            short_optimizer
            .param_groups
        )
    }


    short_history.append(
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
                    short_scheduler
                    .num_updates
                ),
        }
    )


    # INTERNAL checkpoint selection:
    # Geometry primary, lower val loss tie-break.
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

        short_best_report[
            'metrics'
        ][
            'geometry_recall50'
        ],

        -short_best_report[
            'mean_loss'
        ][
            'loss_total'
        ],
    )


    if (
        current_rank
        >
        best_rank
    ):

        short_best_report = (
            copy.deepcopy(
                val_report
            )
        )

        short_best_epoch = epoch


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
        short_scheduler
        .num_updates,

        '| lr(main)',
        f"{current_lrs.get('main', float('nan')):.2e}",
    )


# ==========================================================
# SCHEDULER SANITY
# ==========================================================

if (
    short_scheduler
    .num_updates
    !=
    short_run_updates
):

    raise RuntimeError(
        'Unexpected scheduler update count: '
        f'{short_scheduler.num_updates} '
        f'!= {short_run_updates}'
    )


if (
    short_scheduler
    .total_updates
    !=
    full_stage1_total_updates
):

    raise RuntimeError(
        'Scheduler horizon mismatch.'
    )


# ==========================================================
# LOSS IMPROVEMENT
# ==========================================================

best_observed_loss = min(

    item[
        'val'
    ][
        'mean_loss'
    ][
        'loss_total'
    ]

    for item
    in short_history
)


loss_improved = bool(
    best_observed_loss
    <
    initial_loss
)


# ==========================================================
# GEOMETRY IMPROVEMENT
# ==========================================================

best_geometry = max(

    item[
        'val'
    ][
        'metrics'
    ][
        'geometry_recall50'
    ]

    for item
    in short_history
)


geometry_improved = bool(
    best_geometry
    >
    initial_geometry
)


# ==========================================================
# TAIL GEOMETRY STABILITY
# ==========================================================

window = min(

    TRAIN_CONFIG[
        'short'
    ][
        'stability_window'
    ],

    len(
        short_history
    ),
)


if window < 1:

    raise RuntimeError(
        'No short-training history.'
    )


last_window = (
    short_history[
        -window:
    ]
)


last_geometry = np.asarray(
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


last_precision = np.asarray(
    [

        item[
            'val'
        ][
            'metrics'
        ][
            'precision50'
        ]

        for item
        in last_window
    ],

    dtype=np.float64,
)


last_recall = np.asarray(
    [

        item[
            'val'
        ][
            'metrics'
        ][
            'recall50'
        ]

        for item
        in last_window
    ],

    dtype=np.float64,
)


last_geometry_mean = float(
    last_geometry.mean()
)


geometry_stability_ratio = (

    last_geometry_mean

    /

    max(
        best_geometry,
        1e-12,
    )
)


geometry_stable = bool(

    geometry_stability_ratio

    >=

    TRAIN_CONFIG[
        'short'
    ][
        'stability_fraction'
    ]
)


precision_recall_alive = bool(

    float(
        last_precision.mean()
    )
    >
    0.0

    and

    float(
        last_recall.mean()
    )
    >
    0.0
)


# ==========================================================
# FINAL GATE
# ==========================================================

COCO80_GENERALIZATION_GATE_PASSED = bool(

    short_best_epoch
    >
    0

    and

    loss_improved

    and

    geometry_improved

    and

    geometry_stable
)


print('=' * 70)
print('STEP 29 : SHORT COCO-80 GATE RESULT')
print('=' * 70)

print(
    'Initial loss          :',
    initial_loss,
)

print(
    'Best observed loss    :',
    best_observed_loss,
)

print(
    'Loss improved         :',
    loss_improved,
)

print(
    'Initial Geometry50    :',
    initial_geometry,
)

print(
    'Best Geometry50       :',
    best_geometry,
)

print(
    'Geometry improved     :',
    geometry_improved,
)

print(
    'Last-window Geometry  :',
    last_geometry.tolist(),
)

print(
    'Tail Geometry mean    :',
    last_geometry_mean,
)

print(
    'Geometry stability    :',
    geometry_stability_ratio,
)

print(
    'Required stability    :',
    TRAIN_CONFIG[
        'short'
    ][
        'stability_fraction'
    ],
)

print(
    'P mean (diagnostic)   :',
    float(
        last_precision.mean()
    ),
)

print(
    'R mean (diagnostic)   :',
    float(
        last_recall.mean()
    ),
)

print(
    'P/R alive diagnostic  :',
    precision_recall_alive,
)

print(
    'GATE PASSED           :',
    COCO80_GENERALIZATION_GATE_PASSED,
)

print('=' * 70)


short_model.cpu()

del short_model
del short_optimizer
del short_scheduler

gc.collect()

if torch.cuda.is_available():
    torch.cuda.empty_cache()
