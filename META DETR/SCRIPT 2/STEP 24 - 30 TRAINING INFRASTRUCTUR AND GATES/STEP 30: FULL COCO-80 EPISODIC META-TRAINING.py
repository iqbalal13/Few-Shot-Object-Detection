# ==========================================================
# STEP 30 — FULL REPLACEMENT
# Full stabilized COCO-80 episodic meta-training
# ==========================================================

if not COCO80_GENERALIZATION_GATE_PASSED:

    raise RuntimeError(
        'STEP 29 COCO-80 stability/generalization '
        'gate failed. Do not start full source training.'
    )


# Fresh initialization.
source_model = (
    make_trial_model()
)


source_total_updates = (
    TRAIN_CONFIG[
        'stage1'
    ][
        'epochs'
    ]
    *
    TRAIN_CONFIG[
        'stage1'
    ][
        'steps_per_epoch'
    ]
)


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
# Initial validation
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
        'Initial full validation does not cover '
        'all 80 classes.'
    )


source_history = []


source_best_report = (
    copy.deepcopy(
        source_initial_report
    )
)


source_best_epoch = 0


source_initial_map50 = (
    source_initial_report[
        'metrics'
    ][
        'mAP50'
    ]
)


source_required_map50 = max(

    TRAIN_CONFIG[
        'short'
    ][
        'absolute_map50_floor'
    ],

    source_initial_map50
    *
    TRAIN_CONFIG[
        'short'
    ][
        'relative_factor'
    ],
)


# ==========================================================
# Checkpoint helpers
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
                if scheduler
                is not None
                else
                None
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
                'Stage1: all model parameters trainable; '
                'ResNet-101 lower LR; BN running statistics '
                'frozen; effective episodic batch=4 by '
                'gradient accumulation.'
            ),

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
print('STEP 30 : FULL STABILIZED COCO-80 META-TRAINING')
print('=' * 70)

print(
    'Initial AP50      :',
    source_initial_report[
        'metrics'
    ][
        'mAP50'
    ],
)

print(
    'Required AP50     :',
    source_required_map50,
)

print(
    'Initial Precision :',
    source_initial_report[
        'metrics'
    ][
        'precision50'
    ],
)

print(
    'Initial Recall    :',
    source_initial_report[
        'metrics'
    ][
        'recall50'
    ],
)

print(
    'Initial Geometry  :',
    source_initial_report[
        'metrics'
    ][
        'geometry_recall50'
    ],
)

print(
    'Total updates     :',
    source_total_updates,
)

print('=' * 70)


# ==========================================================
# Full training
# ==========================================================

for epoch in range(
    1,
    TRAIN_CONFIG[
        'stage1'
    ][
        'epochs'
    ]
    +
    1,
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

            max_steps=
                TRAIN_CONFIG[
                    'stage1'
                ][
                    'steps_per_epoch'
                ],

            stage=
                'stage1',

            description=(
                f'COCO80 '
                f'{epoch}/'
                f"{TRAIN_CONFIG['stage1']['epochs']}"
            ),

            show_progress=True,

            scheduler=
                source_scheduler,
        )
    )

    # Fixed unseen validation.
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
            'Full validation does not cover '
            'all 80 classes.'
        )

    current_lrs = {

        group.get(
            'name',
            str(
                index
            ),
        ):
            group[
                'lr'
            ]

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
                current_lrs,
        }
    )

    # AP50 = primary.
    # Geometry first tie-breaker.
    # P/R fixed threshold later tie-breakers only.
    current_rank = (

        val_report[
            'metrics'
        ][
            'mAP50'
        ],

        val_report[
            'metrics'
        ][
            'geometry_recall50'
        ],

        val_report[
            'metrics'
        ][
            'precision50'
        ],

        val_report[
            'metrics'
        ][
            'recall50'
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
            'mAP50'
        ],

        source_best_report[
            'metrics'
        ][
            'geometry_recall50'
        ],

        source_best_report[
            'metrics'
        ][
            'precision50'
        ],

        source_best_report[
            'metrics'
        ][
            'recall50'
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
        '| val',
        round(
            val_report[
                'mean_loss'
            ][
                'loss_total'
            ],
            4,
        ),
        '| AP50',
        round(
            val_report[
                'metrics'
            ][
                'mAP50'
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
        '| AP75',
        round(
            val_report[
                'metrics'
            ][
                'mAP75'
            ],
            4,
        ),
        '| lr',
        f"{current_lrs.get('main', float('nan')):.2e}",
        '| BEST'
        if improved
        else
        '',
    )


# ==========================================================
# Final source stability gate
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
        'Best COCO-80 checkpoint was not created.'
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
        'No full-training history was created.'
    )


last_window = (
    source_history[
        -window:
    ]
)


full_tail_map50 = np.asarray(
    [
        item[
            'val'
        ][
            'metrics'
        ][
            'mAP50'
        ]

        for item
        in last_window
    ],
    dtype=np.float64,
)


source_best_map50 = (
    source_best_report[
        'metrics'
    ][
        'mAP50'
    ]
)


source_stable_tail = bool(

    float(
        full_tail_map50.mean()
    )

    >=

    TRAIN_CONFIG[
        'short'
    ][
        'stability_fraction'
    ]

    *

    max(
        source_best_map50,
        1e-12,
    )
)


# Diagnostic only.
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


# Hard gate:
# 1. learned checkpoint
# 2. AP50 floor
# 3. stable tail
COCO80_META_TRAINING_COMPLETE = bool(

    source_best_epoch
    >
    0

    and

    source_best_map50
    >=
    source_required_map50

    and

    source_stable_tail
)


print('=' * 70)
print('STEP 30 COMPLETE : COCO-80 META-TRAINING')
print('=' * 70)

print(
    'Initial AP50     :',
    source_initial_map50,
)

print(
    'Required AP50    :',
    source_required_map50,
)

print(
    'Best epoch       :',
    source_best_epoch,
)

print(
    'Best AP50        :',
    source_best_report[
        'metrics'
    ][
        'mAP50'
    ],
)

print(
    'Best Precision   :',
    source_best_report[
        'metrics'
    ][
        'precision50'
    ],
)

print(
    'Best Recall      :',
    source_best_report[
        'metrics'
    ][
        'recall50'
    ],
)

print(
    'Best Geometry50  :',
    source_best_report[
        'metrics'
    ][
        'geometry_recall50'
    ],
)

print(
    'Diagnostic AP75  :',
    source_best_report[
        'metrics'
    ][
        'mAP75'
    ],
)

print(
    'Diagnostic 50:95 :',
    source_best_report[
        'metrics'
    ][
        'mAP50_95'
    ],
)

print(
    'Tail AP50        :',
    full_tail_map50.tolist(),
)

print(
    'Tail mean AP50   :',
    float(
        full_tail_map50.mean()
    ),
)

print(
    'Stable >=80% best:',
    source_stable_tail,
)

print(
    'P/R non-zero     :',
    source_pr_alive,
    '(diagnostic only)',
)

print(
    'SOURCE STABLE    :',
    COCO80_META_TRAINING_COMPLETE,
)

print(
    'Best checkpoint  :',
    COCO80_BEST_CHECKPOINT_PATH,
)

print()


if not COCO80_META_TRAINING_COMPLETE:

    raise RuntimeError(
        'Full COCO-80 run is still unstable. '
        'Do NOT proceed to CCTV yet.'
    )


print(
    'NEXT: COCO-Val PERSON-only readiness gate.'
)

print(
    'COCO-person specialization is FALLBACK ONLY '
    'if that gate fails.'
)

print('=' * 70)
