# ============================================================
# STEP 30 — FULL COCO-80 SOURCE TRAINING
# Train COCO-Train
# Validate COCO-Val every epoch
# Save BEST by Geometry50
# ============================================================

import os
import torch

from google.colab import drive

print("=" * 70)
print("STEP 30 — FULL COCO-80 SOURCE TRAINING")
print("=" * 70)


# ------------------------------------------------------------
# Persistent Google Drive storage
# ------------------------------------------------------------

drive.mount(
    "/content/drive",
    force_remount=False
)

FULL_CHECKPOINT_DIR = (
    "/content/drive/MyDrive/"
    "CenterNet_COCO80_Source"
)

os.makedirs(
    FULL_CHECKPOINT_DIR,
    exist_ok=True
)

BEST_CHECKPOINT = os.path.join(
    FULL_CHECKPOINT_DIR,
    "centernet_coco80_best.pth"
)

LAST_CHECKPOINT = os.path.join(
    FULL_CHECKPOINT_DIR,
    "centernet_coco80_last.pth"
)


# ------------------------------------------------------------
# Training configuration
# ------------------------------------------------------------

FULL_EPOCHS = 25

print("Epochs          :", FULL_EPOCHS)
print("Train images    :", len(train_dataset))
print("Val images      :", len(val_dataset))
print("Batch size      :", CONFIG["batch_size"])
print("Score threshold :", CONFIG["score_threshold"])
print("IoU threshold   :", CONFIG["iou_threshold"])

print(
    "Best criterion  : "
    "highest Geometry50 "
    "(val-loss tie-break)"
)


# ------------------------------------------------------------
# Fresh source model
# ------------------------------------------------------------

model = build_fresh_source_model()

optimizer = build_source_optimizer(
    model
)

scaler = torch.amp.GradScaler(
    "cuda",
    enabled=True
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=FULL_EPOCHS
)


start_epoch = 0

best_geometry = -1.0
best_val_loss = float(
    "inf"
)

history = []


# ------------------------------------------------------------
# Resume support
# ------------------------------------------------------------

if os.path.exists(
    LAST_CHECKPOINT
):

    print(
        "\nPrevious LAST checkpoint detected."
    )

    checkpoint = torch.load(
        LAST_CHECKPOINT,
        map_location=device,
        weights_only=False
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    optimizer.load_state_dict(
        checkpoint[
            "optimizer_state_dict"
        ]
    )

    scheduler.load_state_dict(
        checkpoint[
            "scheduler_state_dict"
        ]
    )

    if (
        "scaler_state_dict"
        in checkpoint
    ):

        scaler.load_state_dict(
            checkpoint[
                "scaler_state_dict"
            ]
        )


    start_epoch = checkpoint[
        "epoch"
    ]

    best_geometry = checkpoint.get(
        "best_geometry",
        -1.0
    )

    best_val_loss = checkpoint.get(
        "best_val_loss",
        float("inf")
    )

    history = checkpoint.get(
        "history",
        []
    )


    print(
        "Resume from epoch :",
        start_epoch + 1
    )

    print(
        "Best Geometry50    :",
        best_geometry
    )


else:

    print(
        "\nStarting CLEAN COCO-80 "
        "source training."
    )


# ------------------------------------------------------------
# Full training
# ------------------------------------------------------------

for epoch_index in range(
    start_epoch,
    FULL_EPOCHS
):

    epoch_number = (
        epoch_index + 1
    )


    print(
        "\n"
        + "=" * 70
    )

    print(
        f"FULL EPOCH "
        f"{epoch_number}/{FULL_EPOCHS}"
    )

    print(
        "=" * 70
    )


    # --------------------------------------------------------
    # TRAIN — COCO-TRAIN
    # --------------------------------------------------------

    train_stats = \
        train_centernet_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            max_batches=None
        )


    # --------------------------------------------------------
    # VALIDATE — FULL COCO-VAL
    # No gradients / no optimizer update
    # --------------------------------------------------------

    print(
        "\nRunning full COCO-Val..."
    )

    val_stats = evaluate_centernet(
        model,
        val_loader,
        max_batches=None,
        score_threshold=CONFIG[
            "score_threshold"
        ],
        iou_threshold=CONFIG[
            "iou_threshold"
        ]
    )


    print(
        "\n"
        + "-" * 70
    )

    print(
        f"EPOCH {epoch_number} RESULT"
    )

    print(
        "-" * 70
    )

    print(
        f"Train loss       : "
        f"{train_stats['loss']:.4f}"
    )

    print(
        f"Val loss         : "
        f"{val_stats['loss']:.4f}"
    )

    print(
        f"Geometry50       : "
        f"{val_stats['geometry50']:.4f}"
    )

    print(
        f"Precision@0.50   : "
        f"{val_stats['precision']:.4f}"
    )

    print(
        f"Recall@0.50      : "
        f"{val_stats['recall']:.4f}"
    )

    print(
        f"Learning rate    : "
        f"{optimizer.param_groups[0]['lr']:.8f}"
    )


    epoch_record = {

        "epoch":
            epoch_number,

        "train_loss":
            train_stats[
                "loss"
            ],

        "val_loss":
            val_stats[
                "loss"
            ],

        "geometry50":
            val_stats[
                "geometry50"
            ],

        "precision":
            val_stats[
                "precision"
            ],

        "recall":
            val_stats[
                "recall"
            ]
    }


    history.append(
        epoch_record
    )


    # --------------------------------------------------------
    # BEST CHECKPOINT
    #
    # Primary:
    # highest Geometry50
    #
    # Tie-break:
    # lower validation loss
    # --------------------------------------------------------

    geometry_improved = (
        val_stats[
            "geometry50"
        ]
        > best_geometry
    )

    geometry_tied = math.isclose(
        val_stats[
            "geometry50"
        ],
        best_geometry,
        rel_tol=0.0,
        abs_tol=1e-12
    )

    lower_loss_on_tie = (
        geometry_tied
        and
        val_stats[
            "loss"
        ]
        < best_val_loss
    )


    if (
        geometry_improved
        or lower_loss_on_tie
    ):

        best_geometry = \
            val_stats[
                "geometry50"
            ]

        best_val_loss = \
            val_stats[
                "loss"
            ]


        torch.save(
            {
                "model_state_dict":
                    model.state_dict(),

                "epoch":
                    epoch_number,

                "val_stats":
                    val_stats,

                "train_stats":
                    train_stats,

                "config":
                    CONFIG
            },
            BEST_CHECKPOINT
        )


        print(
            "\n★ NEW BEST SOURCE CHECKPOINT"
        )

        print(
            "Geometry50 :",
            best_geometry
        )

        print(
            "Saved to   :",
            BEST_CHECKPOINT
        )


    # Scheduler after completed epoch
    scheduler.step()


    # --------------------------------------------------------
    # LAST CHECKPOINT
    # Used only for training resume
    # --------------------------------------------------------

    torch.save(
        {
            "model_state_dict":
                model.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "scheduler_state_dict":
                scheduler.state_dict(),

            "scaler_state_dict":
                scaler.state_dict(),

            "epoch":
                epoch_number,

            "best_geometry":
                best_geometry,

            "best_val_loss":
                best_val_loss,

            "history":
                history,

            "config":
                CONFIG
        },
        LAST_CHECKPOINT
    )


    print(
        "\nLast checkpoint saved."
    )


print(
    "\n"
    + "=" * 70
)

print(
    "FULL COCO-80 TRAINING COMPLETE"
)

print(
    "=" * 70
)

print(
    "Best Geometry50 :",
    best_geometry
)

print(
    "Best checkpoint :",
    BEST_CHECKPOINT
)

print(
    "\nSTEP 30 PASSED"
)
