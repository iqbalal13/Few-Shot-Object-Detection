# ==========================================================
# STEP 30 : FULL COCO-80 EPISODIC META-TRAINING
#
# STAGE 1
#
# Fresh original initialization.
#
# Train:
#   full balanced COCO-80 episodic source training
#
# Validation:
#   fixed unseen COCO-Val episodes
#
# Best checkpoint selected by:
# 1. mAP50:95
# 2. mAP50
# 3. geometry recall
#
# NEXT after this:
# STEP 31 = COCO-person specialization.
# ==========================================================

if not COCO80_GENERALIZATION_GATE_PASSED:

    raise RuntimeError(
        "STEP 29 COCO-80 generalization "
        "gate failed. "
        "Do not start full source training."
    )


source_model = (
    make_trial_model()
)


source_optimizer, source_scheduler = (
    build_optimizer_and_scheduler(

        source_model,

        stage="stage1",

        use_scheduler=True,
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


source_history = []


source_best_report = (
    copy.deepcopy(
        source_initial_report
    )
)


source_best_epoch = 0


def checkpoint_config_snapshot():

    return {

        key:
            (
                str(
                    value
                )
                if isinstance(
                    value,
                    torch.device
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

        "stage":
            "COCO80_META_TRAINING",

        "epoch":
            int(
                epoch
            ),

        "model_state":
            target_model.state_dict(),

        "optimizer_state":
            optimizer.state_dict(),

        "scheduler_state":
            (
                scheduler.state_dict()
                if scheduler
                is not None
                else None
            ),

        "val_report":
            copy.deepcopy(
                val_report
            ),

        "model_config":
            checkpoint_config_snapshot(),

        "train_config":
            copy.deepcopy(
                TRAIN_CONFIG
            ),

        "research_protocol":
            copy.deepcopy(
                RESEARCH_PROTOCOL
            ),

        "freeze_policy": (
            "Stage1: all model parameters trainable; "
            "ResNet-101 uses lower LR; "
            "BatchNorm running statistics frozen."
        ),
    }

    temporary_path = (
        path
        +
        ".tmp"
    )

    torch.save(
        payload,
        temporary_path
    )

    os.replace(
        temporary_path,
        path
    )


# Save epoch-0 checkpoint only as diagnostic.
save_source_checkpoint(

    os.path.join(
        COCO80_CHECKPOINT_DIR,
        "coco80_meta_initial.pth"
    ),

    0,

    source_model,

    source_optimizer,

    source_scheduler,

    source_initial_report,
)


print("=" * 70)
print("STEP 30 : FULL COCO-80 META-TRAINING")
print("=" * 70)

print(
    "Initial val mAP50:",
    source_initial_report[
        "metrics"
    ][
        "mAP50"
    ]
)

print(
    "Initial val mAP50:95:",
    source_initial_report[
        "metrics"
    ][
        "mAP50_95"
    ]
)


# ==========================================================
# FULL TRAINING
# ==========================================================

for epoch in range(
    1,
    TRAIN_CONFIG[
        "stage1"
    ][
        "epochs"
    ]
    +
    1
):

    # Balanced but newly shuffled source episodes.
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
                    "stage1"
                ][
                    "steps_per_epoch"
                ],

            stage=
                "stage1",

            description=
                (
                    f"COCO80 "
                    f"{epoch}/"
                    f"{TRAIN_CONFIG['stage1']['epochs']}"
                ),

            show_progress=True,
        )
    )

    # Fixed unseen validation episodes.
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
            "metrics"
        ][
            "classes_evaluated"
        ]
        !=
        80
    ):

        raise RuntimeError(
            "Full validation does not "
            "cover all 80 classes."
        )

    source_history.append(
        {
            "epoch":
                epoch,

            "train":
                train_stats,

            "val":
                val_report,

            "learning_rates":
                {
                    group.get(
                        "name",
                        str(
                            index
                        )
                    ):
                        group[
                            "lr"
                        ]

                    for index, group
                    in enumerate(
                        source_optimizer
                        .param_groups
                    )
                },
        }
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

        -val_report[
            "mean_loss"
        ][
            "loss_total"
        ],
    )

    best_rank = (

        source_best_report[
            "metrics"
        ][
            "mAP50_95"
        ],

        source_best_report[
            "metrics"
        ][
            "mAP50"
        ],

        source_best_report[
            "metrics"
        ][
            "geometry_recall50"
        ],

        -source_best_report[
            "mean_loss"
        ][
            "loss_total"
        ],
    )

    improved = (
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
        f"Epoch {epoch:02d}",
        "| train",
        round(
            train_stats[
                "loss_total"
            ],
            4
        ),
        "| val loss",
        round(
            val_report[
                "mean_loss"
            ][
                "loss_total"
            ],
            4
        ),
        "| mAP50",
        round(
            val_report[
                "metrics"
            ][
                "mAP50"
            ],
            4
        ),
        "| mAP75",
        round(
            val_report[
                "metrics"
            ][
                "mAP75"
            ],
            4
        ),
        "| mAP50:95",
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
        "| BEST"
        if improved
        else "",
    )

    if source_scheduler is not None:

        source_scheduler.step()


# ==========================================================
# FINAL SOURCE-STAGE CHECK
# ==========================================================

if source_best_epoch <= 0:

    raise RuntimeError(
        "Full COCO-80 training did not "
        "improve over initialization."
    )


if not os.path.isfile(
    COCO80_BEST_CHECKPOINT_PATH
):

    raise RuntimeError(
        "Best COCO-80 checkpoint "
        "was not created."
    )


COCO80_META_TRAINING_COMPLETE = True


print("=" * 70)
print("STEP 30 COMPLETE : COCO-80 META-TRAINING")
print("=" * 70)

print(
    "Best epoch     :",
    source_best_epoch
)

print(
    "Best mAP50     :",
    source_best_report[
        "metrics"
    ][
        "mAP50"
    ]
)

print(
    "Best mAP75     :",
    source_best_report[
        "metrics"
    ][
        "mAP75"
    ]
)

print(
    "Best mAP95     :",
    source_best_report[
        "metrics"
    ][
        "mAP95"
    ]
)

print(
    "Best mAP50:95  :",
    source_best_report[
        "metrics"
    ][
        "mAP50_95"
    ]
)

print(
    "Geometry50     :",
    source_best_report[
        "metrics"
    ][
        "geometry_recall50"
    ]
)

print(
    "Best checkpoint:",
    COCO80_BEST_CHECKPOINT_PATH
)

print()
print(
    "IMPORTANT: model is NOT yet the final CCTV source checkpoint."
)

print(
    "NEXT: COCO-person specialization + "
    "COCO-Val-person gate."
)

print("=" * 70)
