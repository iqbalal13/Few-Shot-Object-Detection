# ==========================================================
# STEP 29 : Short COCO-80 Generalization Gate
#
# Fresh model from original initialization.
#
# Train:
#   COCO-Train episodic
#
# Evaluate:
#   fixed COCO-Val episodic
#
# Tiny weights are NOT reused.
# ==========================================================

if not TINY_GATE_PASSED:

    raise RuntimeError(
        "STEP 28 tiny multi-class "
        "learning gate failed. "
        "Do not run generalization yet."
    )


# ==========================================================
# FIX VALIDATION EPISODES
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

        stage="stage1",

        use_scheduler=False,
    )
)


# ==========================================================
# BEFORE TRAINING
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
        "metrics"
    ][
        "classes_evaluated"
    ]
    !=
    80
):

    raise RuntimeError(
        "Validation episodes do not "
        "cover all 80 source classes."
    )


print("=" * 70)
print("SHORT GENERALIZATION — INITIAL")
print("=" * 70)

print(
    "mAP50:",
    short_initial_report[
        "metrics"
    ][
        "mAP50"
    ]
)

print(
    "mAP50:95:",
    short_initial_report[
        "metrics"
    ][
        "mAP50_95"
    ]
)

print(
    "Geometry50:",
    short_initial_report[
        "metrics"
    ][
        "geometry_recall50"
    ]
)


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
    TRAIN_CONFIG[
        "short"
    ][
        "epochs"
    ]
    +
    1
):

    # New balanced training episodes each epoch.
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

            max_steps=
                TRAIN_CONFIG[
                    "short"
                ][
                    "steps_per_epoch"
                ],

            stage=
                "stage1",

            description=
                (
                    f"Short COCO80 "
                    f"{epoch}/"
                    f"{TRAIN_CONFIG['short']['epochs']}"
                ),
        )
    )

    # Validation stays FIXED.
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
            "metrics"
        ][
            "classes_evaluated"
        ]
        !=
        80
    ):

        raise RuntimeError(
            "Validation coverage "
            "changed unexpectedly."
        )

    short_history.append(
        {
            "epoch":
                epoch,

            "train":
                train_stats,

            "val":
                val_report,
        }
    )

    print(
        f"Epoch {epoch}",
        "| train loss",
        round(
            train_stats[
                "loss_total"
            ],
            4
        ),
        "| val mAP50",
        round(
            val_report[
                "metrics"
            ][
                "mAP50"
            ],
            4
        ),
        "| val mAP50:95",
        round(
            val_report[
                "metrics"
            ][
                "mAP50_95"
            ],
            4
        ),
        "| geometry",
        round(
            val_report[
                "metrics"
            ][
                "geometry_recall50"
            ],
            4
        ),
    )

    current_rank = (

        val_report[
            "metrics"
        ][
            "mAP50_95"
        ],

        val_report[
            "metrics"
        ][
            "mAP50"
        ],

        val_report[
            "metrics"
        ][
            "geometry_recall50"
        ],
    )

    best_rank = (

        short_best_report[
            "metrics"
        ][
            "mAP50_95"
        ],

        short_best_report[
            "metrics"
        ][
            "mAP50"
        ],

        short_best_report[
            "metrics"
        ][
            "geometry_recall50"
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

        short_best_epoch = (
            epoch
        )


# ==========================================================
# GENERALIZATION GATE
# ==========================================================

initial_map50 = (
    short_initial_report[
        "metrics"
    ][
        "mAP50"
    ]
)


required_map50 = max(

    TRAIN_CONFIG[
        "short"
    ][
        "absolute_map50_floor"
    ],

    initial_map50
    *
    TRAIN_CONFIG[
        "short"
    ][
        "relative_factor"
    ],
)


map5095_improvement = (

    short_best_report[
        "metrics"
    ][
        "mAP50_95"
    ]

    -

    short_initial_report[
        "metrics"
    ][
        "mAP50_95"
    ]
)


geometry_improved = (

    short_best_report[
        "metrics"
    ][
        "geometry_recall50"
    ]

    >

    short_initial_report[
        "metrics"
    ][
        "geometry_recall50"
    ]
)


COCO80_GENERALIZATION_GATE_PASSED = bool(

    short_best_epoch
    >
    0

    and

    short_best_report[
        "metrics"
    ][
        "mAP50"
    ]
    >=
    required_map50

    and

    map5095_improvement
    >=
    TRAIN_CONFIG[
        "short"
    ][
        "minimum_map5095_improvement"
    ]

    and

    geometry_improved

    and

    short_best_report[
        "metrics"
    ][
        "classes_evaluated"
    ]
    ==
    80
)


print("=" * 70)
print("STEP 29 : COCO-80 GENERALIZATION RESULT")
print("=" * 70)

print(
    "Initial mAP50  :",
    initial_map50
)

print(
    "Required mAP50 :",
    required_map50
)

print(
    "Best mAP50     :",
    short_best_report[
        "metrics"
    ][
        "mAP50"
    ]
)

print(
    "Initial 50:95  :",
    short_initial_report[
        "metrics"
    ][
        "mAP50_95"
    ]
)

print(
    "Best 50:95     :",
    short_best_report[
        "metrics"
    ][
        "mAP50_95"
    ]
)

print(
    "Best epoch     :",
    short_best_epoch
)

print(
    "GENERALIZATION:",
    COCO80_GENERALIZATION_GATE_PASSED
)

print("=" * 70)


short_model.cpu()

del short_model
del short_optimizer
del short_scheduler

gc.collect()

if torch.cuda.is_available():
    torch.cuda.empty_cache()
