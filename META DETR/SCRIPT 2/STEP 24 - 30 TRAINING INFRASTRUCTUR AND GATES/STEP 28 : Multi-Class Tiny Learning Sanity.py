# ==========================================================
# STEP 28 : Multi-Class Tiny Learning Sanity
#
# PURPOSE
# ----------------------------------------------------------
# Fixed tiny set:
#   - 8 semantic classes
#   - 2 episodes / class
#   - total = 16 fixed episodes
#
# This is an OVERFIT / LEARNING-CAPABILITY test.
# This is NOT a generalization test.
#
# IMPORTANT:
# Tiny test intentionally uses accumulation_steps = 1.
# Stage-1 real COCO training still uses accumulation_steps = 4.
# ==========================================================


# ==========================================================
# FIX EPISODE GENERATION
# ==========================================================

train_dataset.set_epoch(0)


tiny_num_classes = (
    TRAIN_CONFIG["tiny"]["num_classes"]
)

tiny_episodes_per_class = (
    TRAIN_CONFIG["tiny"]["episodes_per_class"]
)


# ==========================================================
# SELECT 8 CLASSES SPREAD ACROSS COCO-80
# ==========================================================

TINY_CLASS_LABELS = (
    np.linspace(
        0,
        CONFIG["source_num_categories"] - 1,
        tiny_num_classes,
        dtype=int,
    )
    .tolist()
)

# Remove accidental duplicates while preserving order.
TINY_CLASS_LABELS = list(
    dict.fromkeys(
        TINY_CLASS_LABELS
    )
)

if len(TINY_CLASS_LABELS) != tiny_num_classes:
    raise RuntimeError(
        "Tiny class selection contains duplicates."
    )


# ==========================================================
# BUILD FIXED TINY EPISODE SET
# ==========================================================

tiny_indices = []

for semantic_label in TINY_CLASS_LABELS:

    matching_indices = [
        index
        for index, label
        in enumerate(
            train_dataset.episode_labels
        )
        if int(label) == int(semantic_label)
    ]

    if (
        len(matching_indices)
        <
        tiny_episodes_per_class
    ):
        raise RuntimeError(
            "Not enough fixed episodes "
            f"for tiny class {semantic_label}."
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


tiny_loader = make_episode_loader(
    tiny_subset,
    batch_size=1,
    num_workers=0,
)


expected_tiny_episodes = (
    tiny_num_classes
    *
    tiny_episodes_per_class
)

if len(tiny_loader) != expected_tiny_episodes:
    raise RuntimeError(
        "Unexpected tiny loader size: "
        f"expected {expected_tiny_episodes}, "
        f"got {len(tiny_loader)}."
    )


print("=" * 70)
print("STEP 28 : TINY FIXED-SET SETUP")
print("=" * 70)
print(
    "Classes:",
    [
        CATEGORY_NAMES[label]
        for label in TINY_CLASS_LABELS
    ],
)
print(
    "Episodes/class :",
    tiny_episodes_per_class,
)
print(
    "Total episodes :",
    len(tiny_loader),
)
print(
    "Tiny accumulation steps : 1"
)
print("=" * 70)


# ==========================================================
# FRESH MODEL
#
# Tiny experiment must start from the same original
# initialization used by diagnostic trials.
# ==========================================================

tiny_model = make_trial_model()


# ==========================================================
# OPTIMIZER
#
# Build normal Stage-1 optimizer groups,
# but DO NOT use scheduler for this fixed overfit test.
# ==========================================================

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
        "report": copy.deepcopy(
            tiny_initial_report
        ),
    }
]


tiny_best_report = copy.deepcopy(
    tiny_initial_report
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

        "AP50":
            tiny_initial_report[
                "metrics"
            ][
                "mAP50"
            ],

        "precision":
            tiny_initial_report[
                "metrics"
            ][
                "precision50"
            ],

        "recall":
            tiny_initial_report[
                "metrics"
            ][
                "recall50"
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
#
# CRITICAL:
# accumulation_steps = 1 here.
#
# 16 episodes therefore produce:
# 16 optimizer updates / epoch.
#
# This preserves the original purpose of the tiny overfit
# sanity test.
# ==========================================================

for epoch in range(
    1,
    TRAIN_CONFIG["tiny"]["epochs"] + 1,
):

    # Keep EXACTLY the same episode construction.
    train_dataset.set_epoch(0)

    train_stats = train_detection_epoch(
        target_model=tiny_model,
        loader=tiny_loader,
        optimizer=tiny_optimizer,

        # Here max_steps means optimizer updates.
        # With accumulation_steps=1:
        # 16 micro-batches = 16 optimizer updates.
        max_steps=len(tiny_loader),

        stage="stage1",

        description=(
            f"Tiny {epoch}"
        ),

        show_progress=False,

        # --------------------------------------------------
        # IMPORTANT FIX
        # --------------------------------------------------
        accumulation_steps=1,
    )


    # ======================================================
    # PERIODIC EVALUATION
    # ======================================================

    should_evaluate = (
        epoch == 1
        or
        epoch % 10 == 0
        or
        epoch
        ==
        TRAIN_CONFIG["tiny"]["epochs"]
    )

    if not should_evaluate:
        continue


    # ------------------------------------------------------
    # IMPORTANT:
    # evaluate_episodic_model does NOT accept
    # accumulation_steps.
    # ------------------------------------------------------

    report = evaluate_episodic_model(
        tiny_model,
        tiny_loader,
        show_progress=False,
    )


    tiny_history.append(
        {
            "epoch": epoch,
            "train": copy.deepcopy(
                train_stats
            ),
            "report": copy.deepcopy(
                report
            ),
        }
    )


    # ======================================================
    # READ METRICS
    # ======================================================

    current_loss = (
        report[
            "mean_loss"
        ][
            "loss_total"
        ]
    )

    current_ap50 = (
        report[
            "metrics"
        ][
            "mAP50"
        ]
    )

    current_precision = (
        report[
            "metrics"
        ][
            "precision50"
        ]
    )

    current_recall = (
        report[
            "metrics"
        ][
            "recall50"
        ]
    )

    current_geometry = (
        report[
            "metrics"
        ][
            "geometry_recall50"
        ]
    )


    print(
        f"Tiny epoch {epoch:03d}",
        "| loss",
        round(
            current_loss,
            4,
        ),
        "| AP50",
        round(
            current_ap50,
            4,
        ),
        "| P",
        round(
            current_precision,
            4,
        ),
        "| R",
        round(
            current_recall,
            4,
        ),
        "| geometry50",
        round(
            current_geometry,
            4,
        ),
    )


    # ======================================================
    # BEST TINY CHECKPOINT/RANK
    #
    # Tiny test remains primarily an AP50 / geometry
    # learnability test.
    #
    # Precision and Recall are included as additional
    # tie-breakers / diagnostics.
    # ======================================================

    current_rank = (
        current_ap50,
        current_geometry,
        current_precision,
        current_recall,
        -current_loss,
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

        tiny_best_report[
            "metrics"
        ][
            "precision50"
        ],

        tiny_best_report[
            "metrics"
        ][
            "recall50"
        ],

        -tiny_best_report[
            "mean_loss"
        ][
            "loss_total"
        ],
    )


    if current_rank > best_rank:

        tiny_best_report = (
            copy.deepcopy(
                report
            )
        )

        tiny_best_epoch = epoch


# ==========================================================
# TINY LEARNING IMPROVEMENTS
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


tiny_precision_improvement = (

    tiny_best_report[
        "metrics"
    ][
        "precision50"
    ]

    -

    tiny_initial_report[
        "metrics"
    ][
        "precision50"
    ]
)


tiny_recall_improvement = (

    tiny_best_report[
        "metrics"
    ][
        "recall50"
    ]

    -

    tiny_initial_report[
        "metrics"
    ][
        "recall50"
    ]
)


# ==========================================================
# TINY GATE
#
# Locked gate remains:
#
# 1. Best loss < initial loss
# 2. AP50 improvement >= configured threshold
# 3. Geometry improvement >= configured threshold
#
# Precision / Recall are reported for diagnosis but are
# NOT added as hard tiny-gate requirements.
# ==========================================================

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


# ==========================================================
# FINAL STEP-28 REPORT
# ==========================================================

print("=" * 70)
print("STEP 28 RESULT")
print("=" * 70)

print(
    "Classes       :",
    [
        CATEGORY_NAMES[label]
        for label
        in TINY_CLASS_LABELS
    ],
)

print(
    "Best epoch    :",
    tiny_best_epoch,
)

print("-" * 70)

print(
    "Initial loss  :",
    tiny_initial_report[
        "mean_loss"
    ][
        "loss_total"
    ],
)

print(
    "Best loss     :",
    tiny_best_report[
        "mean_loss"
    ][
        "loss_total"
    ],
)

print("-" * 70)

print(
    "Initial AP50  :",
    tiny_initial_report[
        "metrics"
    ][
        "mAP50"
    ],
)

print(
    "Best AP50     :",
    tiny_best_report[
        "metrics"
    ][
        "mAP50"
    ],
)

print(
    "AP50 Δ        :",
    tiny_map_improvement,
)

print("-" * 70)

print(
    "Initial P     :",
    tiny_initial_report[
        "metrics"
    ][
        "precision50"
    ],
)

print(
    "Best P        :",
    tiny_best_report[
        "metrics"
    ][
        "precision50"
    ],
)

print(
    "Precision Δ   :",
    tiny_precision_improvement,
)

print("-" * 70)

print(
    "Initial R     :",
    tiny_initial_report[
        "metrics"
    ][
        "recall50"
    ],
)

print(
    "Best R        :",
    tiny_best_report[
        "metrics"
    ][
        "recall50"
    ],
)

print(
    "Recall Δ      :",
    tiny_recall_improvement,
)

print("-" * 70)

print(
    "Initial Geo50 :",
    tiny_initial_report[
        "metrics"
    ][
        "geometry_recall50"
    ],
)

print(
    "Best Geo50    :",
    tiny_best_report[
        "metrics"
    ][
        "geometry_recall50"
    ],
)

print(
    "Geometry Δ    :",
    tiny_geometry_improvement,
)

print("-" * 70)

print(
    "TINY GATE     :",
    TINY_GATE_PASSED,
)

print("=" * 70)


# ==========================================================
# HARD STOP IF TINY LEARNABILITY FAILS
#
# We deliberately do NOT raise here because STEP 29 already
# checks TINY_GATE_PASSED and provides the controlled stop.
# ==========================================================


# ==========================================================
# CLEANUP
# ==========================================================

tiny_model.cpu()

del tiny_model
del tiny_optimizer

gc.collect()

if torch.cuda.is_available():
    torch.cuda.empty_cache()
