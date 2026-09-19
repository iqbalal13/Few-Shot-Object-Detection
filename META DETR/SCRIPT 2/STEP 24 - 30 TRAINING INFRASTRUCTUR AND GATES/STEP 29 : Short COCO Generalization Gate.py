# ==========================================================
# STEP 29 — FULL REPLACEMENT
#
# SHORT COCO-80 LEARNING / READINESS GATE
#
# PURPOSE:
# ----------------------------------------------------------
# This short run is NOT the final stability certificate.
# Its purpose is to verify that the model:
#
#   1. learns on unseen COCO-Val80 episodes,
#   2. improves localization geometry,
#   3. retains that improvement near the end of the run.
#
# True long-horizon stability is checked in STEP 30.
#
#
# TRAINING BUDGET:
# ----------------------------------------------------------
# 5 epochs
# 800 episodes / epoch
# accumulation = 4
#
# 200 optimizer updates / epoch
# 1,000 optimizer updates executed
#
# Scheduler horizon = FULL Stage-1 = 5,000 updates.
#
#
# HARD GATE:
# ----------------------------------------------------------
#   1. validation loss improves
#   2. Geometry50 improves
#   3. last-window mean Geometry50 remains above initialization
#
#
# DIAGNOSTIC ONLY:
# ----------------------------------------------------------
#   tail Geometry / best Geometry stability ratio
#   old reference threshold = 0.80
#   Precision@0.50
#   Recall@0.50
#
# The 0.80 stability ratio is NOT a hard blocker here.
#
# ==========================================================


# ==========================================================
# REQUIRE TINY LEARNING PASS
# ==========================================================

if not TINY_GATE_PASSED:

    raise RuntimeError(
        'STEP 28 tiny learning gate failed. '
        'Do not run source readiness gate.'
    )


# ==========================================================
# CONFIGURATION
# ==========================================================

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


# ==========================================================
# SHORT/FULL CONSISTENCY
# ==========================================================

if (
    short_episodes_per_epoch
    !=
    stage1_episodes_per_epoch
):

    raise RuntimeError(
        'Short and full Stage-1 must use '
        'the same episodes_per_epoch.'
    )


# ==========================================================
# UPDATE BUDGET
# ==========================================================

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


# Locked protocol sanity.
assert short_updates_per_epoch == 200
assert short_run_updates == 1000
assert full_stage1_total_updates == 5000


# ==========================================================
# FIXED VALIDATION EPISODES
# ==========================================================

val_dataset.set_epoch(
    0
)


# ==========================================================
# FRESH MODEL
#
# Tiny-model weights are NOT reused.
# ==========================================================

short_model = (
    make_trial_model()
)


# ==========================================================
# OPTIMIZER + FULL-HORIZON SCHEDULER
#
# Short run executes 1,000 updates,
# but follows first 1,000 updates of the
# full 5,000-update Stage-1 schedule.
# ==========================================================

short_optimizer, short_scheduler = (
    build_optimizer_and_scheduler(

        short_model,

        stage='stage1',

        use_scheduler=True,

        total_updates=
            full_stage1_total_updates,
    )
)


# ==========================================================
# INITIAL UNSEEN COCO-VAL80 EVALUATION
# ==========================================================

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


# ==========================================================
# INITIAL VALUES
# ==========================================================

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


initial_precision = float(

    short_initial_report[
        'metrics'
    ][
        'precision50'
    ]
)


initial_recall = float(

    short_initial_report[
        'metrics'
    ][
        'recall50'
    ]
)


# ==========================================================
# INITIAL REPORT
# ==========================================================

print('=' * 70)

print(
    'STEP 29 : SHORT COCO-80 '
    'LEARNING / READINESS GATE'
)

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
    'Accumulation steps      :',
    stage1_accumulation,
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
    'Full scheduler horizon  :',
    full_stage1_total_updates,
)


print(
    'Warmup updates          :',
    TRAIN_CONFIG[
        'stage1'
    ][
        'warmup_updates'
    ],
)


print('-' * 70)


print(
    'Initial loss            :',
    initial_loss,
)


print(
    'Initial Geometry50      :',
    initial_geometry,
)


print(
    'Initial Precision@.50   :',
    initial_precision,
)


print(
    'Initial Recall@.50      :',
    initial_recall,
)


print('=' * 70)


# ==========================================================
# HISTORY
# ==========================================================

short_history = []


short_best_report = (
    copy.deepcopy(
        short_initial_report
    )
)


short_best_epoch = 0


# ==========================================================
# SHORT SOURCE TRAINING
# ==========================================================

for epoch in range(
    1,
    short_epochs + 1,
):

    # ------------------------------------------------------
    # Deterministic episodic schedule for this epoch.
    # ------------------------------------------------------

    train_dataset.set_epoch(
        epoch
    )


    # ------------------------------------------------------
    # 800 EPISODES, not 800 optimizer updates.
    # ------------------------------------------------------

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


    # ======================================================
    # FIXED UNSEEN VALIDATION
    # ======================================================

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
            'Validation coverage changed unexpectedly.'
        )


    # ======================================================
    # CURRENT LEARNING RATES
    # ======================================================

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


    # ======================================================
    # HISTORY
    # ======================================================

    short_history.append(
        {

            'epoch':
                int(
                    epoch
                ),

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


    # ======================================================
    # BEST CHECKPOINT DIAGNOSTIC
    #
    # INTERNAL ranking:
    #
    #   1. higher Geometry50
    #   2. lower validation loss
    #
    # P/R are deliberately NOT used for generic
    # source checkpoint selection.
    # ======================================================

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

        short_best_epoch = int(
            epoch
        )


    # ======================================================
    # EPOCH LOG
    # ======================================================

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
# HISTORY SANITY
# ==========================================================

if (
    len(
        short_history
    )
    !=
    short_epochs
):

    raise RuntimeError(
        'Unexpected short-history length: '
        f'{len(short_history)} '
        f'!= {short_epochs}.'
    )


# ==========================================================
# LOSS IMPROVEMENT
#
# HARD GATE COMPONENT 1
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
#
# HARD GATE COMPONENT 2
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
# LAST-WINDOW STATISTICS
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
        'No short-training history was created.'
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


last_loss = np.asarray(
    [

        item[
            'val'
        ][
            'mean_loss'
        ][
            'loss_total'
        ]

        for item
        in last_window
    ],

    dtype=np.float64,
)


# ==========================================================
# TAIL VALUES
# ==========================================================

last_geometry_mean = float(
    last_geometry.mean()
)


last_loss_mean = float(
    last_loss.mean()
)


last_precision_mean = float(
    last_precision.mean()
)


last_recall_mean = float(
    last_recall.mean()
)


final_geometry = float(
    short_history[
        -1
    ][
        'val'
    ][
        'metrics'
    ][
        'geometry_recall50'
    ]
)


# ==========================================================
# RETAINED LEARNING
#
# HARD GATE COMPONENT 3
#
# The short run must finish with a LAST-WINDOW mean
# still above the untrained initialization.
#
# This prevents a model from passing merely because of
# one transient early Geometry peak.
# ==========================================================

tail_above_initial = bool(
    last_geometry_mean
    >
    initial_geometry
)


# ==========================================================
# STABILITY DIAGNOSTIC
#
# NOT A HARD GATE IN STEP 29.
#
# True long-horizon stability belongs to STEP 30.
# ==========================================================

geometry_stability_ratio = (

    last_geometry_mean

    /

    max(
        best_geometry,
        1e-12,
    )
)


geometry_stable_diagnostic = bool(

    geometry_stability_ratio

    >=

    TRAIN_CONFIG[
        'short'
    ][
        'stability_fraction'
    ]
)


# ==========================================================
# DROP FROM PEAK
#
# Pure diagnostic information.
# ==========================================================

peak_to_final_drop_fraction = max(

    0.0,

    (
        best_geometry
        -
        final_geometry
    )

    /

    max(
        best_geometry,
        1e-12,
    )
)


peak_to_tail_mean_drop_fraction = max(

    0.0,

    (
        best_geometry
        -
        last_geometry_mean
    )

    /

    max(
        best_geometry,
        1e-12,
    )
)


# ==========================================================
# PRECISION / RECALL DIAGNOSTIC
#
# Not a hard gate at generic COCO80 source stage.
#
# Fixed score threshold can remain zero while
# localization already learns.
# ==========================================================

precision_recall_alive = bool(

    last_precision_mean
    >
    0.0

    and

    last_recall_mean
    >
    0.0
)


# ==========================================================
# FINAL STEP-29 READINESS GATE
#
# HARD CONDITIONS:
#
#   1. model produced a better checkpoint
#   2. validation loss improved
#   3. Geometry50 improved
#   4. tail mean Geometry remains above initialization
#
# NOTE:
# 80% peak stability ratio is diagnostic only here.
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

    tail_above_initial
)


# ==========================================================
# FINAL REPORT
# ==========================================================

print('=' * 70)

print(
    'STEP 29 : SHORT COCO-80 '
    'LEARNING / READINESS RESULT'
)

print('=' * 70)


print(
    'Initial loss              :',
    initial_loss,
)


print(
    'Best observed loss        :',
    best_observed_loss,
)


print(
    'Loss improved             :',
    loss_improved,
)


print(
    'Tail mean loss            :',
    last_loss_mean,
)


print('-' * 70)


print(
    'Initial Geometry50        :',
    initial_geometry,
)


print(
    'Best Geometry50           :',
    best_geometry,
)


print(
    'Final Geometry50          :',
    final_geometry,
)


print(
    'Geometry improved         :',
    geometry_improved,
)


print(
    'Last-window Geometry      :',
    last_geometry.tolist(),
)


print(
    'Tail Geometry mean        :',
    last_geometry_mean,
)


print(
    'Tail > initialization     :',
    tail_above_initial,
)


print('-' * 70)


print(
    'Geometry stability ratio  :',
    geometry_stability_ratio,
)


print(
    'Reference ratio           :',
    TRAIN_CONFIG[
        'short'
    ][
        'stability_fraction'
    ],
)


print(
    'Stable >= reference       :',
    geometry_stable_diagnostic,
    '(diagnostic only)',
)


print(
    'Peak -> final drop        :',
    f'{peak_to_final_drop_fraction * 100.0:.2f}%',
)


print(
    'Peak -> tail-mean drop    :',
    f'{peak_to_tail_mean_drop_fraction * 100.0:.2f}%',
)


print('-' * 70)


print(
    'Tail Precision mean       :',
    last_precision_mean,
)


print(
    'Tail Recall mean          :',
    last_recall_mean,
)


print(
    'P/R alive                 :',
    precision_recall_alive,
    '(diagnostic only)',
)


print('-' * 70)


print(
    'Best epoch                :',
    short_best_epoch,
)


print(
    'Scheduler updates         :',
    short_scheduler
    .num_updates,
)


print(
    'Scheduler horizon         :',
    short_scheduler
    .total_updates,
)


print(
    'READINESS GATE PASSED     :',
    COCO80_GENERALIZATION_GATE_PASSED,
)


print('=' * 70)


# ==========================================================
# INTERPRETATION
# ==========================================================

if COCO80_GENERALIZATION_GATE_PASSED:

    print(
        'STEP 29 PASS: short-run learning is confirmed.'
    )

    print(
        'Proceed to STEP 30 for full Stage-1 training '
        'and long-horizon stability evaluation.'
    )

else:

    print(
        'STEP 29 FAIL: short-run learning/readiness '
        'criteria were not satisfied.'
    )

    print(
        'Do not start full Stage-1 training yet.'
    )


print('=' * 70)


# ==========================================================
# CLEANUP
# ==========================================================

short_model.cpu()


del short_model
del short_optimizer
del short_scheduler


gc.collect()


if torch.cuda.is_available():

    torch.cuda.empty_cache()
