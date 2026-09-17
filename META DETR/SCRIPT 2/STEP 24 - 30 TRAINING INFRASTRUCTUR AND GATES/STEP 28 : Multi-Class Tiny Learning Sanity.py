# ==========================================================
# STEP 28 — FULL REPLACEMENT
# Multi-class tiny learnability sanity
#
# HARD GATE:
#   validation loss improves
#   AND
#   Geometry50 improves
#
# P/R are diagnostic only.
# ==========================================================


train_dataset.set_epoch(
    0
)


tiny_num_classes = int(
    TRAIN_CONFIG[
        'tiny'
    ][
        'num_classes'
    ]
)


tiny_episodes_per_class = int(
    TRAIN_CONFIG[
        'tiny'
    ][
        'episodes_per_class'
    ]
)


TINY_CLASS_LABELS = (

    np.linspace(

        0,

        CONFIG[
            'source_num_categories'
        ]
        -
        1,

        tiny_num_classes,

        dtype=int,
    )
    .tolist()
)


TINY_CLASS_LABELS = list(
    dict.fromkeys(
        TINY_CLASS_LABELS
    )
)


if (
    len(
        TINY_CLASS_LABELS
    )
    !=
    tiny_num_classes
):

    raise RuntimeError(
        'Tiny class selection contains duplicates.'
    )


tiny_indices = []


for semantic_label in (
    TINY_CLASS_LABELS
):

    matching_indices = [

        index

        for index, label
        in enumerate(
            train_dataset
            .episode_labels
        )

        if int(
            label
        )
        ==
        int(
            semantic_label
        )
    ]


    if (
        len(
            matching_indices
        )
        <
        tiny_episodes_per_class
    ):

        raise RuntimeError(
            'Not enough fixed episodes '
            f'for tiny class {semantic_label}'
        )


    tiny_indices.extend(
        matching_indices[
            :tiny_episodes_per_class
        ]
    )


tiny_subset = Subset(
    train_dataset,
    tiny_indices,
)


tiny_loader = (
    make_episode_loader(

        tiny_subset,

        batch_size=1,

        num_workers=0,
    )
)


tiny_model = (
    make_trial_model()
)


tiny_optimizer, _ = (
    build_optimizer_and_scheduler(

        tiny_model,

        stage='stage1',

        use_scheduler=False,
    )
)


# ==========================================================
# INITIAL EVALUATION
# ==========================================================

tiny_initial_report = (
    evaluate_episodic_model(

        tiny_model,

        tiny_loader,

        show_progress=False,
    )
)


tiny_history = [

    {
        'epoch':
            0,

        'report':
            copy.deepcopy(
                tiny_initial_report
            ),
    }
]


tiny_best_report = (
    copy.deepcopy(
        tiny_initial_report
    )
)


tiny_best_epoch = 0


print(
    'Tiny initial:',
    {

        'loss':
            tiny_initial_report[
                'mean_loss'
            ][
                'loss_total'
            ],

        'precision50':
            tiny_initial_report[
                'metrics'
            ][
                'precision50'
            ],

        'recall50':
            tiny_initial_report[
                'metrics'
            ][
                'recall50'
            ],

        'geometry50':
            tiny_initial_report[
                'metrics'
            ][
                'geometry_recall50'
            ],
    }
)


# ==========================================================
# FIXED-SET TRAINING
#
# Accumulation = 1 intentionally for tiny sanity.
# ==========================================================

for epoch in range(

    1,

    TRAIN_CONFIG[
        'tiny'
    ][
        'epochs'
    ]
    +
    1,
):

    # Same exact episodes every epoch.
    train_dataset.set_epoch(
        0
    )


    train_stats = (
        train_detection_epoch(

            target_model=
                tiny_model,

            loader=
                tiny_loader,

            optimizer=
                tiny_optimizer,

            max_episodes=
                len(
                    tiny_loader
                ),

            stage=
                'stage1',

            description=
                f'Tiny {epoch}',

            show_progress=False,

            accumulation_steps=1,
        )
    )


    if (
        epoch == 1
        or
        epoch % 10 == 0
        or
        epoch
        ==
        TRAIN_CONFIG[
            'tiny'
        ][
            'epochs'
        ]
    ):

        report = (
            evaluate_episodic_model(

                tiny_model,

                tiny_loader,

                show_progress=False,
            )
        )


        tiny_history.append(
            {
                'epoch':
                    epoch,

                'train':
                    copy.deepcopy(
                        train_stats
                    ),

                'report':
                    copy.deepcopy(
                        report
                    ),
            }
        )


        current_rank = (

            report[
                'metrics'
            ][
                'geometry_recall50'
            ],

            -report[
                'mean_loss'
            ][
                'loss_total'
            ],
        )


        best_rank = (

            tiny_best_report[
                'metrics'
            ][
                'geometry_recall50'
            ],

            -tiny_best_report[
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

            tiny_best_report = (
                copy.deepcopy(
                    report
                )
            )

            tiny_best_epoch = (
                epoch
            )


        print(

            f'Tiny epoch {epoch:03d}',

            '| loss',
            round(
                report[
                    'mean_loss'
                ][
                    'loss_total'
                ],
                4,
            ),

            '| P',
            round(
                report[
                    'metrics'
                ][
                    'precision50'
                ],
                4,
            ),

            '| R',
            round(
                report[
                    'metrics'
                ][
                    'recall50'
                ],
                4,
            ),

            '| geometry50',
            round(
                report[
                    'metrics'
                ][
                    'geometry_recall50'
                ],
                4,
            ),
        )


tiny_initial_loss = float(
    tiny_initial_report[
        'mean_loss'
    ][
        'loss_total'
    ]
)


tiny_best_loss = float(
    tiny_best_report[
        'mean_loss'
    ][
        'loss_total'
    ]
)


tiny_initial_geometry = float(
    tiny_initial_report[
        'metrics'
    ][
        'geometry_recall50'
    ]
)


tiny_best_geometry = float(
    tiny_best_report[
        'metrics'
    ][
        'geometry_recall50'
    ]
)


tiny_loss_improved = bool(
    tiny_best_loss
    <
    tiny_initial_loss
)


tiny_geometry_improved = bool(
    tiny_best_geometry
    >
    tiny_initial_geometry
)


TINY_GATE_PASSED = bool(

    tiny_best_epoch
    >
    0

    and

    tiny_loss_improved

    and

    tiny_geometry_improved
)


print('=' * 70)
print('STEP 28 RESULT')
print('=' * 70)

print(
    'Classes          :',
    [
        CATEGORY_NAMES[
            label
        ]
        for label
        in TINY_CLASS_LABELS
    ],
)

print(
    'Best epoch       :',
    tiny_best_epoch,
)

print(
    'Initial loss     :',
    tiny_initial_loss,
)

print(
    'Best loss        :',
    tiny_best_loss,
)

print(
    'Loss improved    :',
    tiny_loss_improved,
)

print(
    'Initial Geometry :',
    tiny_initial_geometry,
)

print(
    'Best Geometry    :',
    tiny_best_geometry,
)

print(
    'Geometry improved:',
    tiny_geometry_improved,
)

print(
    'Best Precision   :',
    tiny_best_report[
        'metrics'
    ][
        'precision50'
    ],
)

print(
    'Best Recall      :',
    tiny_best_report[
        'metrics'
    ][
        'recall50'
    ],
)

print(
    'TINY GATE        :',
    TINY_GATE_PASSED,
)

print('=' * 70)


tiny_model.cpu()

del tiny_model
del tiny_optimizer

gc.collect()

if torch.cuda.is_available():
    torch.cuda.empty_cache()
