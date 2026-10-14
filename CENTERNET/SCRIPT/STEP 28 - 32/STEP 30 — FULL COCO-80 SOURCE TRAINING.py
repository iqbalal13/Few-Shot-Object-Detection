# ============================================================
# STEP 30 — FULL COCO-80 SOURCE TRAINING
# FAST-REPRO V1 — LOCKED
#
# CLEAN experiment namespace
#
# - BF16 forward
# - FP32 stable loss
# - no GradScaler
# - deterministic epoch-specific shuffle
# - full COCO-Val every epoch
# - best checkpoint by Geometry50
# - validation-loss tie break
# - early stopping patience = 2
# - maximum 5 source epochs
# ============================================================

import os
import gc
import math
import torch

from google.colab import drive


print("=" * 70)
print("STEP 30 — FULL COCO-80 SOURCE TRAINING")
print("FAST-REPRO V1")
print("=" * 70)


# ============================================================
# HARDWARE
# ============================================================

assert torch.cuda.is_available(), (
    "CUDA GPU tidak aktif."
)


assert torch.cuda.is_bf16_supported(), (
    "Current GPU does not support BF16."
)


print(
    "GPU            :",
    torch.cuda.get_device_name(0)
)


print(
    "Forward dtype  : BF16"
)


print(
    "Loss dtype     : FP32"
)


print(
    "GradScaler     : OFF"
)


# ============================================================
# GOOGLE DRIVE
# ============================================================

drive.mount(
    "/content/drive",
    force_remount=False
)


# ============================================================
# NEW CLEAN EXPERIMENT DIRECTORY
#
# Intentionally different from old baseline run.
# ============================================================

FULL_CHECKPOINT_DIR = (

    "/content/drive/"
    "MyDrive/"
    "CenterNet_COCO80_Source_"
    "FAST_REPRO_V1"
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


# ============================================================
# LOCKED SOURCE TRAINING CONFIG
# ============================================================

MAX_SOURCE_EPOCHS = 5


SCHEDULER_TMAX = 25


EARLY_STOPPING_PATIENCE = 2


MIN_GEOMETRY_DELTA = 1e-4


FULL_TRAIN_SEED_BASE = (
    SEED
    + 3000
)


print(
    "Experiment               :",
    CONFIG[
        "experiment_name"
    ]
)


print(
    "Checkpoint directory     :",
    FULL_CHECKPOINT_DIR
)


print(
    "Max source epochs        :",
    MAX_SOURCE_EPOCHS
)


print(
    "Scheduler T_max          :",
    SCHEDULER_TMAX
)


print(
    "Early stopping patience  :",
    EARLY_STOPPING_PATIENCE
)


print(
    "Train images             :",
    len(train_dataset)
)


print(
    "Val images               :",
    len(val_dataset)
)


print(
    "Batch size               :",
    CONFIG[
        "batch_size"
    ]
)


print(
    "Workers                  :",
    CONFIG[
        "num_workers"
    ]
)


print(
    "Prefetch factor          :",
    CONFIG[
        "prefetch_factor"
    ]
)


print(
    "Score threshold          :",
    CONFIG[
        "score_threshold"
    ]
)


print(
    "IoU threshold            :",
    CONFIG[
        "iou_threshold"
    ]
)


print(
    "Best criterion           : "
    "highest Geometry50 "
    "(val-loss tie-break)"
)


# ============================================================
# MODEL FINITE CHECK
# ============================================================

def assert_model_finite(
    model,
    label="model"
):


    bad_parameters = []


    for name, parameter \
        in model.named_parameters():


        if not torch.isfinite(
            parameter
        ).all():

            bad_parameters.append(
                name
            )


    if bad_parameters:

        raise FloatingPointError(

            f"{label} contains "
            f"non-finite parameters: "

            f"{bad_parameters[:10]}"
        )


# ============================================================
# CHECK RESUME STATUS FIRST
# ============================================================

resume_checkpoint = None


if os.path.exists(
    LAST_CHECKPOINT
):


    print(
        "\nFAST-REPRO LAST checkpoint detected."
    )


    resume_checkpoint = torch.load(

        LAST_CHECKPOINT,

        map_location=device,

        weights_only=False
    )


# ============================================================
# BUILD MODEL
#
# When resuming, ImageNet download is unnecessary because
# every parameter will immediately be replaced by checkpoint.
# ============================================================

if resume_checkpoint is None:


    model = \
        CenterNetResNet101(

            num_classes=
                CONFIG[
                    "num_classes"
                ],

            pretrained_backbone=True

        ).to(
            device
        )


else:


    model = \
        CenterNetResNet101(

            num_classes=
                CONFIG[
                    "num_classes"
                ],

            pretrained_backbone=False

        ).to(
            device
        )


# ============================================================
# OPTIMIZER / SCHEDULER
# ============================================================

optimizer = \
    build_source_optimizer(
        model
    )


scheduler = \
    torch.optim.lr_scheduler.CosineAnnealingLR(

        optimizer,

        T_max=
            SCHEDULER_TMAX
    )


start_epoch = 0


best_geometry = -1.0


best_val_loss = float(
    "inf"
)


history = []


epochs_without_improvement = 0


# ============================================================
# LOAD FAST-REPRO CHECKPOINT
# ============================================================

if resume_checkpoint is not None:


    checkpoint = \
        resume_checkpoint


    checkpoint_experiment = \
        checkpoint.get(
            "experiment_name",
            None
        )


    if checkpoint_experiment is not None:

        assert (
            checkpoint_experiment
            == CONFIG[
                "experiment_name"
            ]
        ), (
            "Checkpoint berasal dari "
            "experiment berbeda."
        )


    model.load_state_dict(

        checkpoint[
            "model_state_dict"
        ]
    )


    assert_model_finite(

        model,

        label=
            "Loaded FAST-REPRO checkpoint"
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
        "Loaded completed epoch   :",
        start_epoch
    )


    print(
        "Resume from epoch        :",
        start_epoch + 1
    )


    print(
        "Best Geometry50          :",
        best_geometry
    )


    print(
        "Best val loss            :",
        best_val_loss
    )


    print(
        "Current learning rate    :",
        optimizer.param_groups[
            0
        ][
            "lr"
        ]
    )


else:


    print(
        "\nStarting CLEAN "
        "FAST-REPRO source training."
    )


# ============================================================
# RELEASE OLD GLOBAL TRAIN LOADER
#
# From this point Step 30 owns an independent train loader.
# This avoids training-order dependence on Steps 25/29.
# ============================================================

if "train_loader" in globals():

    try:
        del train_loader
    except Exception:
        pass


gc.collect()


# ============================================================
# INDEPENDENT FULL SOURCE TRAIN LOADER
# ============================================================

full_train_loader, \
full_train_generator = \
    build_train_loader(
        FULL_TRAIN_SEED_BASE
    )


print(
    "\nFull source train batches :",
    len(
        full_train_loader
    )
)


# ============================================================
# BF16 MODEL HEALTH CHECK
#
# Use validation loader, NOT full_train_loader.
# Therefore train shuffle generator remains untouched.
# ============================================================

print(
    "\nRunning BF16 "
    "checkpoint/model health check..."
)


health_images, \
health_targets = next(
    iter(
        val_loader
    )
)


health_images = \
    health_images[
        :1
    ].to(
        device,
        non_blocking=True
    )


with torch.no_grad():


    model.eval()


    with torch.amp.autocast(

        device_type="cuda",

        dtype=
            torch.bfloat16
    ):


        health_outputs = \
            model(
                health_images
            )


assert_outputs_finite(

    health_outputs,

    context=
        "during Step-30 health check."
)


health_encoded = \
    build_batch_centernet_targets(

        health_targets[
            :1
        ],

        device=device
    )


health_losses = \
    criterion(

        health_outputs,

        health_encoded
    )


if not torch.isfinite(

    health_losses[
        "loss_total"
    ]

):

    raise FloatingPointError(
        "Health-check loss "
        "is NaN/Inf."
    )


print(
    "Health check : PASSED"
)


print(
    "Health loss  :",
    float(
        health_losses[
            "loss_total"
        ].detach()
    )
)


del health_images
del health_targets
del health_outputs
del health_encoded
del health_losses


torch.cuda.empty_cache()


# ============================================================
# ALREADY COMPLETE?
# ============================================================

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
    # FULL SOURCE TRAINING
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
        # DETERMINISTIC EPOCH-SPECIFIC SHUFFLE
        #
        # Epoch N always receives the same shuffle,
        # regardless of interruption/resume.
        # ----------------------------------------------------

        epoch_seed = (

            FULL_TRAIN_SEED_BASE

            + epoch_number
        )


        full_train_generator.manual_seed(
            epoch_seed
        )


        print(
            "Epoch shuffle seed :",
            epoch_seed
        )


        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------

        train_stats = \
            train_centernet_epoch(

                model,

                full_train_loader,

                optimizer,

                max_batches=None
            )


        assert_model_finite(

            model,

            label=(

                f"Model after "
                f"epoch "
                f"{epoch_number}"
            )
        )


        print(

            f"\nTraining epoch "
            f"{epoch_number} complete."
        )


        # ----------------------------------------------------
        # FULL COCO-VAL
        # ----------------------------------------------------

        print(
            "\nRunning FULL COCO-Val..."
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
        # RESULTS
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

            "epoch_seed":
                epoch_seed,

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

            "precision_mode":
                "bfloat16"
        }


        history.append(
            epoch_record
        )


        # ----------------------------------------------------
        # BEST CHECKPOINT LOGIC
        # ----------------------------------------------------

        current_geometry = \
            float(

                val_stats[
                    "geometry50"
                ]
            )


        current_val_loss = \
            float(

                val_stats[
                    "loss"
                ]
            )


        geometry_improved = (

            current_geometry

            > best_geometry
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


        # ----------------------------------------------------
        # BEST
        # ----------------------------------------------------

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

                    "epoch_seed":
                        epoch_seed,

                    "val_stats":
                        val_stats,

                    "train_stats":
                        train_stats,

                    "config":
                        CONFIG,

                    "experiment_name":
                        CONFIG[
                            "experiment_name"
                        ],

                    "precision_mode":
                        "bfloat16"

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
        # SCHEDULER
        # ----------------------------------------------------

        scheduler.step()


        # ----------------------------------------------------
        # LAST COMPLETED-EPOCH CHECKPOINT
        # ----------------------------------------------------

        torch.save(

            {

                "model_state_dict":
                    model.state_dict(),

                "optimizer_state_dict":
                    optimizer.state_dict(),

                "scheduler_state_dict":
                    scheduler.state_dict(),

                "epoch":
                    epoch_number,

                "epoch_seed":
                    epoch_seed,

                "best_geometry":
                    best_geometry,

                "best_val_loss":
                    best_val_loss,

                "history":
                    history,

                "epochs_without_improvement":
                    epochs_without_improvement,

                "config":
                    CONFIG,

                "experiment_name":
                    CONFIG[
                        "experiment_name"
                    ],

                "precision_mode":
                    "bfloat16"

            },

            LAST_CHECKPOINT
        )


        print(
            "\nLAST completed-epoch "
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

                "Geometry50 did not improve "
                f"for "
                f"{EARLY_STOPPING_PATIENCE} "
                "consecutive epochs."
            )


            break


# ============================================================
# FINAL STATUS
# ============================================================

print(
    "\n"
    + "=" * 70
)


print(
    "FULL COCO-80 SOURCE TRAINING FINISHED"
)


print(
    "FAST-REPRO V1"
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
    "Forward dtype   : BF16"
)


print(
    "Loss dtype      : FP32"
)


print(
    "GradScaler      : OFF"
)


print(
    "\nSTEP 30 PASSED"
)
