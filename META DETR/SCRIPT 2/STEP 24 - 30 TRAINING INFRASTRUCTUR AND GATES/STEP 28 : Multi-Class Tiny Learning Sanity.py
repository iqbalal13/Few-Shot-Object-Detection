# ==========================================================
# STEP 28 : Multi-Class Tiny Learning Sanity
#
# Fixed tiny set:
# 8 semantic classes
# 2 episodes / class
#
# This is an OVERFIT / LEARNING-CAPABILITY test.
# NOT a generalization test.
# ==========================================================

train_dataset.set_epoch(
    0
)


tiny_num_classes = (
    TRAIN_CONFIG[
        "tiny"
    ][
        "num_classes"
    ]
)


tiny_episodes_per_class = (
    TRAIN_CONFIG[
        "tiny"
    ][
        "episodes_per_class"
    ]
)


# Spread selected labels across 0..79.
TINY_CLASS_LABELS = (
    np.linspace(
        0,
        CONFIG[
            "source_num_categories"
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


if len(
    TINY_CLASS_LABELS
) != tiny_num_classes:

    raise RuntimeError(
        "Tiny class selection "
        "contains duplicates."
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

    if len(
        matching_indices
    ) < tiny_episodes_per_class:

        raise RuntimeError(
            "Not enough fixed episodes "
            "for tiny class "
            f"{semantic_label}"
        )

    tiny_indices.extend(
        matching_indices[
            :tiny_episodes_per_class
        ]
    )


tiny_subset = Subset(
    train_dataset,
    tiny_indices
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

        stage="stage1",

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
        "epoch": 0,

        "report":
            tiny_initial_report,
    }
]


tiny_best_report = (
    copy.deepcopy(
        tiny_initial_report
    )
)


tiny_best_epoch = 0


print(
    "Tiny initial:",
    {
        "loss":
            tiny_initial_report[
                "mean_loss"
            ][
                "loss_total"
            ],

        "mAP50":
            tiny_initial_report[
                "metrics"
            ][
                "mAP50"
            ],

        "geometry50":
            tiny_initial_report[
                "metrics"
            ][
                "geometry_recall50"
            ],
    }
)


# ==========================================================
# FIXED-SET TRAINING
# ==========================================================

for epoch in range(
    1,
    TRAIN_CONFIG[
        "tiny"
    ][
        "epochs"
    ]
    +
    1
):

    # Keep same exact episodes.
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

            max_steps=
                len(
                    tiny_loader
                ),

            stage=
                "stage1",

            description=
                f"Tiny {epoch}",

            show_progress=False,
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
            "tiny"
        ][
            "epochs"
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
                "epoch":
                    epoch,

                "train":
                    train_stats,

                "report":
                    report,
            }
        )

        print(
            f"Tiny epoch {epoch:03d}",
            "| loss",
            round(
                report[
                    "mean_loss"
                ][
                    "loss_total"
                ],
                4
            ),
            "| mAP50",
            round(
                report[
                    "metrics"
                ][
                    "mAP50"
                ],
                4
            ),
            "| geometry50",
            round(
                report[
                    "metrics"
                ][
                    "geometry_recall50"
                ],
                4
            ),
        )

        current_rank = (

            report[
                "metrics"
            ][
                "mAP50"
            ],

            report[
                "metrics"
            ][
                "geometry_recall50"
            ],

            -report[
                "mean_loss"
            ][
                "loss_total"
            ],
        )


        best_rank = (

            tiny_best_report[
                "metrics"
            ][
                "mAP50"
            ],

            tiny_best_report[
                "metrics"
            ][
                "geometry_recall50"
            ],

            -tiny_best_report[
                "mean_loss"
            ][
                "loss_total"
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


# ==========================================================
# TINY GATE
# ==========================================================

tiny_map_improvement = (

    tiny_best_report[
        "metrics"
    ][
        "mAP50"
    ]

    -

    tiny_initial_report[
        "metrics"
    ][
        "mAP50"
    ]
)


tiny_geometry_improvement = (

    tiny_best_report[
        "metrics"
    ][
        "geometry_recall50"
    ]

    -

    tiny_initial_report[
        "metrics"
    ][
        "geometry_recall50"
    ]
)


TINY_GATE_PASSED = bool(

    tiny_best_report[
        "mean_loss"
    ][
        "loss_total"
    ]

    <

    tiny_initial_report[
        "mean_loss"
    ][
        "loss_total"
    ]

    and

    tiny_map_improvement
    >=
    TRAIN_CONFIG[
        "tiny"
    ][
        "min_map50_improvement"
    ]

    and

    tiny_geometry_improvement
    >=
    TRAIN_CONFIG[
        "tiny"
    ][
        "min_geometry_improvement"
    ]
)


print("=" * 70)
print("STEP 28 RESULT")
print("=" * 70)

print(
    "Classes      :",
    [
        CATEGORY_NAMES[
            label
        ]
        for label
        in TINY_CLASS_LABELS
    ]
)

print(
    "Best epoch   :",
    tiny_best_epoch
)

print(
    "Initial mAP50:",
    tiny_initial_report[
        "metrics"
    ][
        "mAP50"
    ]
)

print(
    "Best mAP50   :",
    tiny_best_report[
        "metrics"
    ][
        "mAP50"
    ]
)

print(
    "mAP Δ        :",
    tiny_map_improvement
)

print(
    "Geometry Δ   :",
    tiny_geometry_improvement
)

print(
    "TINY GATE    :",
    TINY_GATE_PASSED
)

print("=" * 70)


tiny_model.cpu()

del tiny_model
del tiny_optimizer

gc.collect()

if torch.cuda.is_available():
    torch.cuda.empty_cache()
