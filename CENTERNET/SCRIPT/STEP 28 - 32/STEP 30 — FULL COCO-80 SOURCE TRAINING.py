# ============================================================
# STEP 30 — FULL COCO-80 SOURCE TRAINING
# Resume from completed-epoch checkpoint
# Full COCO-Val every epoch
# Best checkpoint by Geometry50
# Early stopping
# ============================================================

import os
import math
import torch

from google.colab import drive


print("=" * 70)
print("STEP 30 — FULL COCO-80 SOURCE TRAINING")
print("=" * 70)


assert torch.cuda.is_available(), (
    "CUDA GPU tidak aktif. "
    "Gunakan GPU runtime sebelum Step 30."
)


# ------------------------------------------------------------
# Persistent Google Drive
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

# Kita evaluasi source model sampai epoch 5 dulu.
MAX_SOURCE_EPOCHS = 5


# JANGAN ubah ke 5.
# Checkpoint lama dibuat dengan scheduler T_max=25,
# jadi kita pertahankan horizon scheduler ini agar resume
# konsisten.
SCHEDULER_TMAX = 25


EARLY_STOPPING_PATIENCE = 2

MIN_GEOMETRY_DELTA = 1e-4


print(
    "Max source epochs       :",
    MAX_SOURCE_EPOCHS
)

print(
    "Scheduler T_max         :",
    SCHEDULER_TMAX
)

print(
    "Early stopping patience :",
    EARLY_STOPPING_PATIENCE
)

print(
    "Train images            :",
    len(train_dataset)
)

print(
    "Val images              :",
    len(val_dataset)
)

print(
    "Batch size              :",
    CONFIG["batch_size"]
)

print(
    "Score threshold         :",
    CONFIG[
        "score_threshold"
    ]
)

print(
    "IoU threshold           :",
    CONFIG[
        "iou_threshold"
    ]
)

print(
    "Best criterion          : "
    "highest Geometry50 "
    "(val-loss tie-break)"
)


# ------------------------------------------------------------
# Build architecture
# ------------------------------------------------------------

model = build_fresh_source_model()


optimizer = \
    build_source_optimizer(
        model
    )


scaler = \
    torch.amp.GradScaler(
        "cuda",
        enabled=True
    )


scheduler = \
    torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=SCHEDULER_TMAX
    )


start_epoch = 0

best_geometry = -1.0

best_val_loss = float(
    "inf"
)

history = []

epochs_without_improvement = 0


# ------------------------------------------------------------
# Check model parameters are finite
# ------------------------------------------------------------

def assert_model_finite(
    model
):

    for name, parameter \
        in model.named_parameters():

        if not torch.isfinite(
            parameter
        ).all():

            raise FloatingPointError(
                "Non-finite parameter "
                f"detected: {name}"
            )


# ------------------------------------------------------------
# Resume from LAST completed epoch
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


    # Ensure epoch-2 checkpoint itself is healthy
    assert_model_finite(
        model
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


    start_epoch = int(
        checkpoint[
            "epoch"
        ]
    )


    best_geometry = float(
        checkpoint.get(
            "best_geometry",
            -1.0
        )
    )


    best_val_loss = float(
        checkpoint.get(
            "best_val_loss",
            float("inf")
        )
    )


    history = checkpoint.get(
        "history",
        []
    )


    epochs_without_improvement = \
        int(
            checkpoint.get(
                "epochs_without_improvement",
                0
            )
        )


    print(
        "Resume from epoch       :",
        start_epoch + 1
    )

    print(
        "Best Geometry50         :",
        best_geometry
    )

    print(
        "Best val loss           :",
        best_val_loss
    )

    print(
        "Current learning rate   :",
        optimizer.param_groups[
            0
        ]["lr"]
    )


else:

    print(
        "\nStarting CLEAN COCO-80 "
        "source training."
    )


# ------------------------------------------------------------
# Check whether requested training already completed
# ------------------------------------------------------------

if (
    start_epoch
    >= MAX_SOURCE_EPOCHS
):

    print(
        f"\nCheckpoint already "
        f"completed epoch "
        f"{start_epoch}."
    )

    print(
        "No additional training required."
    )


else:

    # ========================================================
    # FULL TRAINING LOOP
    # ========================================================

    for epoch_index in range(
        start_epoch,
        MAX_SOURCE_EPOCHS
    ):

        epoch_number = (
            epoch_index
            + 1
        )


        print(
            "\n"
            + "=" * 70
        )

        print(
            f"FULL EPOCH "
            f"{epoch_number}/"
            f"{MAX_SOURCE_EPOCHS}"
        )

        print(
            "=" * 70
        )


        # ----------------------------------------------------
        # COCO-TRAIN
        # ----------------------------------------------------

        train_stats = \
            train_centernet_epoch(
                model,
                train_loader,
                optimizer,
                scaler,
                max_batches=None
            )


        print(
            f"\nTraining epoch "
            f"{epoch_number} complete."
        )

        print(
            "Skipped gradient batches :",
            train_stats.get(
                "skipped_batches",
                0
            )
        )


        # ----------------------------------------------------
        # FULL COCO-VAL
        # ----------------------------------------------------

        print(
            "\nRunning full COCO-Val..."
        )


        val_stats = \
            evaluate_centernet(
                model,
                val_loader,
                max_batches=None,

                score_threshold=
                    CONFIG[
                        "score_threshold"
                    ],

                iou_threshold=
                    CONFIG[
                        "iou_threshold"
                    ]
            )


        # ----------------------------------------------------
        # EPOCH RESULT
        # ----------------------------------------------------

        print(
            "\n"
            + "-" * 70
        )

        print(
            f"EPOCH "
            f"{epoch_number} RESULT"
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
                ],

            "skipped_batches":
                train_stats.get(
                    "skipped_batches",
                    0
                )
        }


        history.append(
            epoch_record
        )


        # ----------------------------------------------------
        # BEST CHECKPOINT
        # ----------------------------------------------------

        current_geometry = float(
            val_stats[
                "geometry50"
            ]
        )


        current_val_loss = float(
            val_stats[
                "loss"
            ]
        )


        geometry_improved = (
            current_geometry
            >
            best_geometry
            + MIN_GEOMETRY_DELTA
        )


        geometry_tied = \
            math.isclose(
                current_geometry,
                best_geometry,
                rel_tol=0.0,
                abs_tol=
                    MIN_GEOMETRY_DELTA
            )


        lower_loss_on_tie = (
            geometry_tied
            and current_val_loss
            < best_val_loss
        )


        is_best = (
            geometry_improved
            or lower_loss_on_tie
        )


        if is_best:

            best_geometry = \
                current_geometry

            best_val_loss = \
                current_val_loss

            epochs_without_improvement = 0


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
                "Val loss   :",
                best_val_loss
            )

            print(
                "Saved to   :",
                BEST_CHECKPOINT
            )


        else:

            epochs_without_improvement += 1


            print(
                "\nNo new best checkpoint."
            )

            print(
                "Patience : "
                f"{epochs_without_improvement}/"
                f"{EARLY_STOPPING_PATIENCE}"
            )


        # ----------------------------------------------------
        # Scheduler after completed epoch
        # ----------------------------------------------------

        scheduler.step()


        # ----------------------------------------------------
        # LAST CHECKPOINT
        #
        # This is always a COMPLETED epoch.
        # ----------------------------------------------------

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

                "epochs_without_improvement":
                    epochs_without_improvement,

                "config":
                    CONFIG
            },
            LAST_CHECKPOINT
        )


        print(
            "\nLast completed-epoch "
            "checkpoint saved."
        )


        # ----------------------------------------------------
        # EARLY STOPPING
        # ----------------------------------------------------

        if (
            epochs_without_improvement
            >= EARLY_STOPPING_PATIENCE
        ):

            print(
                "\n"
                + "=" * 70
            )

            print(
                "EARLY STOPPING TRIGGERED"
            )

            print(
                "=" * 70
            )

            print(
                "Geometry50 did not "
                "improve for "
                f"{EARLY_STOPPING_PATIENCE} "
                "consecutive epochs."
            )

            break


# ------------------------------------------------------------
# FINAL STATUS
# ------------------------------------------------------------

print(
    "\n"
    + "=" * 70
)

print(
    "FULL COCO-80 TRAINING FINISHED"
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
    "Last checkpoint :",
    LAST_CHECKPOINT
)

print(
    "\nSTEP 30 PASSED"
)
