# ==========================================================
# STEP 29 : Short COCO-80 Stability + Generalization Gate
#
# Fresh model from original initialization.
# Tiny weights are NOT reused.
#
# PASS requires:
# - genuine unseen AP50 improvement
# - Precision/Recall become non-zero
# - geometry improves
# - last validation window retains >=80% of best AP50
# ==========================================================

if not TINY_GATE_PASSED:
    raise RuntimeError(
        "STEP 28 tiny multi-class learning gate failed. "
        "Do not run generalization yet."
    )

val_dataset.set_epoch(0)

short_model = make_trial_model()

short_total_updates = (
    TRAIN_CONFIG["short"]["epochs"]
    *
    TRAIN_CONFIG["short"]["steps_per_epoch"]
)

short_optimizer, short_scheduler = (
    build_optimizer_and_scheduler(
        short_model,
        stage="stage1",
        use_scheduler=True,
        total_updates=short_total_updates,
    )
)


# ==========================================================
# BEFORE TRAINING
# ==========================================================

short_initial_report = evaluate_episodic_model(
    short_model,
    val_loader,
    show_progress=True,
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
        "Validation episodes do not cover all 80 source classes."
    )

print("=" * 70)
print("SHORT GENERALIZATION — INITIAL")
print("=" * 70)
print("AP50      :", short_initial_report["metrics"]["mAP50"])
print("Precision :", short_initial_report["metrics"]["precision50"])
print("Recall    :", short_initial_report["metrics"]["recall50"])
print("Geometry50:", short_initial_report["metrics"]["geometry_recall50"])
print("=" * 70)

short_history = []
short_best_report = copy.deepcopy(
    short_initial_report
)
short_best_epoch = 0


# ==========================================================
# SHORT SOURCE TRAINING
# ==========================================================

for epoch in range(
    1,
    TRAIN_CONFIG["short"]["epochs"] + 1
):

    train_dataset.set_epoch(epoch)

    train_stats = train_detection_epoch(
        target_model=short_model,
        loader=train_loader,
        optimizer=short_optimizer,
        max_steps=(
            TRAIN_CONFIG[
                "short"
            ][
                "steps_per_epoch"
            ]
        ),
        stage="stage1",
        description=(
            f"Short COCO80 {epoch}/"
            f"{TRAIN_CONFIG['short']['epochs']}"
        ),
        show_progress=True,
        scheduler=short_scheduler,
    )

    val_dataset.set_epoch(0)

    val_report = evaluate_episodic_model(
        short_model,
        val_loader,
        show_progress=True,
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
            "Validation coverage changed unexpectedly."
        )

    current_lrs = {
        group.get(
            "name",
            str(index)
        ):
            group[
                "lr"
            ]
        for index, group
        in enumerate(
            short_optimizer.param_groups
        )
    }

    short_history.append(
        {
            "epoch": epoch,
            "train": copy.deepcopy(
                train_stats
            ),
            "val": copy.deepcopy(
                val_report
            ),
            "learning_rates": current_lrs,
        }
    )

    # Final thesis accuracy priority:
    # AP50 first, then P/R, then geometry/loss diagnostics.
    current_rank = (
        val_report["metrics"]["mAP50"],
        val_report["metrics"]["precision50"],
        val_report["metrics"]["recall50"],
        val_report["metrics"]["geometry_recall50"],
        -val_report["mean_loss"]["loss_total"],
    )

    best_rank = (
        short_best_report["metrics"]["mAP50"],
        short_best_report["metrics"]["precision50"],
        short_best_report["metrics"]["recall50"],
        short_best_report["metrics"]["geometry_recall50"],
        -short_best_report["mean_loss"]["loss_total"],
    )

    if current_rank > best_rank:
        short_best_report = copy.deepcopy(
            val_report
        )
        short_best_epoch = epoch

    print(
        f"Epoch {epoch:02d}",
        "| train",
        round(
            train_stats[
                "loss_total"
            ],
            4
        ),
        "| AP50",
        round(
            val_report[
                "metrics"
            ][
                "mAP50"
            ],
            4
        ),
        "| P",
        round(
            val_report[
                "metrics"
            ][
                "precision50"
            ],
            4
        ),
        "| R",
        round(
            val_report[
                "metrics"
            ][
                "recall50"
            ],
            4
        ),
        "| geo",
        round(
            val_report[
                "metrics"
            ][
                "geometry_recall50"
            ],
            4
        ),
        "| lr(main)",
        f"{current_lrs.get('main', float('nan')):.2e}",
    )


# ==========================================================
# STABILITY-AWARE GATE
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

window = min(
    TRAIN_CONFIG[
        "short"
    ][
        "stability_window"
    ],
    len(
        short_history
    )
)

if window < 1:
    raise RuntimeError(
        "No short-training history was created."
    )

last_window = short_history[
    -window:
]

last_map50 = np.asarray(
    [
        item[
            "val"
        ][
            "metrics"
        ][
            "mAP50"
        ]
        for item
        in last_window
    ],
    dtype=np.float64,
)

last_precision = np.asarray(
    [
        item[
            "val"
        ][
            "metrics"
        ][
            "precision50"
        ]
        for item
        in last_window
    ],
    dtype=np.float64,
)

last_recall = np.asarray(
    [
        item[
            "val"
        ][
            "metrics"
        ][
            "recall50"
        ]
        for item
        in last_window
    ],
    dtype=np.float64,
)

best_map50 = (
    short_best_report[
        "metrics"
    ][
        "mAP50"
    ]
)

stable_tail = bool(
    float(
        last_map50.mean()
    )
    >=
    TRAIN_CONFIG[
        "short"
    ][
        "stability_fraction"
    ]
    *
    max(
        best_map50,
        1e-12
    )
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

geometry_improved = bool(
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
    short_best_epoch > 0
    and
    best_map50 >= required_map50
    and
    geometry_improved
    and
    precision_recall_alive
    and
    stable_tail
)

print("=" * 70)
print("STEP 29 : SHORT COCO-80 STABILITY GATE")
print("=" * 70)
print("Initial AP50       :", initial_map50)
print("Required AP50      :", required_map50)
print("Best epoch         :", short_best_epoch)
print("Best AP50          :", best_map50)
print("Best Precision     :", short_best_report["metrics"]["precision50"])
print("Best Recall        :", short_best_report["metrics"]["recall50"])
print("Best Geometry50    :", short_best_report["metrics"]["geometry_recall50"])
print("Last-window AP50   :", last_map50.tolist())
print("Last-window mean   :", float(last_map50.mean()))
print("Stable >=80% best  :", stable_tail)
print("P/R alive          :", precision_recall_alive)
print("Geometry improved  :", geometry_improved)
print("GATE PASSED        :", COCO80_GENERALIZATION_GATE_PASSED)
print("=" * 70)

short_model.cpu()
del short_model
del short_optimizer
del short_scheduler

gc.collect()

if torch.cuda.is_available():
    torch.cuda.empty_cache()
