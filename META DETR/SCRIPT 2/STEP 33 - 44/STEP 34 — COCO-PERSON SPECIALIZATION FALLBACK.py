# ==========================================================
# STEP 34 — COCO-PERSON SPECIALIZATION FALLBACK
# ==========================================================
# LOCK:
#
# FROZEN:
#   - backbone
#   - query_encoder
#   - transformer_encoder
#   - transformer_decoder
#
# TRAINABLE:
#   - support_encoder
#   - relation_module
#   - detection_head
#
# SOURCE:
#   best stable COCO80 checkpoint
#
# TRAIN:
#   COCO-Train PERSON-only episodic
#
# VALIDATION:
#   fixed COCO-Val PERSON-only episodes from STEP 31
#
# IMPORTANT:
#   - no architecture change
#   - no head reinitialization
#   - foreground prior remains unchanged
#   - final score threshold remains 0.50
#   - IoU threshold remains 0.50
#   - do not lower threshold to force P/R alive
# ==========================================================


import os
import copy
import gc

import numpy as np
import torch


# ==========================================================
# PREREQUISITES
# ==========================================================

if not COCO80_META_TRAINING_COMPLETE:

    raise RuntimeError(
        "STEP 30 COCO-80 meta-training is not complete."
    )


if not os.path.isfile(
    COCO80_BEST_CHECKPOINT_PATH
):

    raise RuntimeError(
        "Missing best COCO-80 checkpoint: "
        f"{COCO80_BEST_CHECKPOINT_PATH}"
    )


if "person_val_loader" not in globals():

    raise RuntimeError(
        "person_val_loader is missing. "
        "Run STEP 31 first."
    )


if not globals().get(
    "STEP33_CONFIDENCE_DIAGNOSTIC_COMPLETE",
    False,
):

    raise RuntimeError(
        "Run STEP 33 confidence diagnostic first."
    )


# ==========================================================
# LOCKED FALLBACK BUDGET
# ==========================================================

PERSON_SPECIALIZATION_EPOCHS = 10

PERSON_SPECIALIZATION_EPISODES_PER_EPOCH = 800

PERSON_SPECIALIZATION_ACCUMULATION = int(
    TRAIN_CONFIG[
        "stage2"
    ][
        "accumulation_steps"
    ]
)

PERSON_SPECIALIZATION_LR = float(
    TRAIN_CONFIG[
        "stage2"
    ][
        "main_lr"
    ]
)

PERSON_SPECIALIZATION_WEIGHT_DECAY = float(
    TRAIN_CONFIG[
        "stage2"
    ][
        "weight_decay"
    ]
)


# Geometry is not allowed to collapse.
#
# Retain at least 80% of the original
# source-person Geometry50 baseline.
PERSON_GEOMETRY_RETENTION_FRACTION = 0.80


# ==========================================================
# PROTOCOL SANITY
# ==========================================================

if (
    PERSON_SPECIALIZATION_ACCUMULATION
    !=
    1
):

    raise RuntimeError(
        "Locked Stage-2 accumulation must be 1."
    )


if (
    TRAIN_CONFIG[
        "score_threshold"
    ]
    !=
    0.50
):

    raise RuntimeError(
        "Final score threshold must remain 0.50."
    )


if (
    TRAIN_CONFIG[
        "primary_iou_threshold"
    ]
    !=
    0.50
):

    raise RuntimeError(
        "Primary IoU threshold must remain 0.50."
    )


# ==========================================================
# CHECKPOINT PATHS
# ==========================================================

PERSON_SPECIALIZATION_DIR = os.path.join(

    os.path.dirname(
        COCO80_CHECKPOINT_DIR
    ),

    "coco_person_specialization",
)


os.makedirs(
    PERSON_SPECIALIZATION_DIR,
    exist_ok=True,
)


PERSON_SPECIALIZATION_BEST_CHECKPOINT_PATH = os.path.join(

    PERSON_SPECIALIZATION_DIR,

    "coco_person_specialized_best.pth",
)


PERSON_SPECIALIZATION_LATEST_CHECKPOINT_PATH = os.path.join(

    PERSON_SPECIALIZATION_DIR,

    "coco_person_specialized_latest.pth",
)


# ==========================================================
# PERSON-ONLY COCO TRAIN DATASET
# ==========================================================

person_train_dataset = COCOEpisodicDataset(

    coco=
        coco_train,

    image_dir=
        TRAIN_IMAGE_DIR,

    support_transform=
        support_transform,

    query_transform=
        query_transform,

    num_episodes=
        PERSON_SPECIALIZATION_EPISODES_PER_EPOCH,

    seed=
        COCO_CONFIG[
            "seed"
        ]
        +
        300_000,

    cat_id_to_label=
        CAT_ID_TO_LABEL,

    category_names=
        CATEGORY_NAMES,

    min_bbox_size=
        COCO_CONFIG[
            "min_bbox_size"
        ],

    allowed_labels=[
        PERSON_LABEL
    ],
)


person_train_dataset.set_epoch(
    0
)


person_train_loader = (
    make_episode_loader(

        person_train_dataset,

        batch_size=1,

        num_workers=
            COCO_CONFIG[
                "num_workers"
            ],
    )
)


if (
    len(
        person_train_loader
    )
    <
    PERSON_SPECIALIZATION_EPISODES_PER_EPOCH
):

    raise RuntimeError(
        "Person train loader is shorter than "
        "the requested specialization budget."
    )


# ==========================================================
# LOAD BEST COCO80 CHECKPOINT
# ==========================================================

person_specialization_model = (
    make_trial_model()
)


_source_ckpt = torch.load(

    COCO80_BEST_CHECKPOINT_PATH,

    map_location=
        CONFIG[
            "device"
        ],

    weights_only=False,
)


person_specialization_model.load_state_dict(

    _source_ckpt[
        "model_state"
    ],

    strict=True,
)


person_specialization_model = (
    person_specialization_model.to(
        CONFIG[
            "device"
        ]
    )
)


# ==========================================================
# EXACT LOCKED FREEZE POLICY
# ==========================================================

# Freeze EVERYTHING first.
for parameter in (
    person_specialization_model.parameters()
):

    parameter.requires_grad = False


# Then unlock ONLY these modules.

set_module_trainable(

    person_specialization_model
    .support_encoder,

    True,
)


set_module_trainable(

    person_specialization_model
    .relation_module,

    True,
)


set_module_trainable(

    person_specialization_model
    .detection_head,

    True,
)


# Backbone BatchNorm statistics stay frozen.
freeze_backbone_bn_statistics(

    person_specialization_model
    .backbone
)


# ==========================================================
# FREEZE SANITY
# ==========================================================

def _has_trainable(
    module
):

    return any(

        parameter.requires_grad

        for parameter
        in module.parameters()
    )


if _has_trainable(
    person_specialization_model
    .backbone
):

    raise RuntimeError(
        "LOCK VIOLATION: backbone must be frozen."
    )


if _has_trainable(
    person_specialization_model
    .query_encoder
):

    raise RuntimeError(
        "LOCK VIOLATION: query encoder must be frozen."
    )


if _has_trainable(
    person_specialization_model
    .transformer_encoder
):

    raise RuntimeError(
        "LOCK VIOLATION: transformer encoder must be frozen."
    )


if _has_trainable(
    person_specialization_model
    .transformer_decoder
):

    raise RuntimeError(
        "LOCK VIOLATION: transformer decoder must be frozen."
    )


if not _has_trainable(
    person_specialization_model
    .support_encoder
):

    raise RuntimeError(
        "LOCK VIOLATION: support encoder must be trainable."
    )


if not _has_trainable(
    person_specialization_model
    .relation_module
):

    raise RuntimeError(
        "LOCK VIOLATION: relation module must be trainable."
    )


if not _has_trainable(
    person_specialization_model
    .detection_head
):

    raise RuntimeError(
        "LOCK VIOLATION: detection head must be trainable."
    )


# ==========================================================
# TRAINABLE PARAMETER LIST
# ==========================================================

trainable_parameters = [

    parameter

    for parameter
    in person_specialization_model.parameters()

    if parameter.requires_grad
]


if not trainable_parameters:

    raise RuntimeError(
        "No trainable parameters."
    )


# ==========================================================
# OPTIMIZER
#
# IMPORTANT:
# Do NOT call:
#
#   build_optimizer_and_scheduler(stage="stage2")
#
# We want the exact locked freeze policy above.
#
# Fixed AdamW is used here.
# ==========================================================

person_specialization_optimizer = (
    torch.optim.AdamW(

        trainable_parameters,

        lr=
            PERSON_SPECIALIZATION_LR,

        weight_decay=
            PERSON_SPECIALIZATION_WEIGHT_DECAY,
    )
)


# ==========================================================
# FIXED PERSON BASELINE
#
# This should reproduce the source-person behavior
# before specialization.
# ==========================================================

person_val_dataset.set_epoch(
    0
)


initial_report = (
    evaluate_episodic_model(

        person_specialization_model,

        person_val_loader,

        show_progress=True,
    )
)


baseline_precision = float(

    initial_report[
        "metrics"
    ][
        "precision50"
    ]
)


baseline_recall = float(

    initial_report[
        "metrics"
    ][
        "recall50"
    ]
)


baseline_geometry = float(

    initial_report[
        "metrics"
    ][
        "geometry_recall50"
    ]
)


baseline_loss = float(

    initial_report[
        "mean_loss"
    ][
        "loss_total"
    ]
)


geometry_floor = (

    PERSON_GEOMETRY_RETENTION_FRACTION

    *

    baseline_geometry
)


# ==========================================================
# CHECKPOINT SAVE HELPER
# ==========================================================

def _save_person_specialization(
    path,
    epoch,
    model,
    optimizer,
    val_report,
    metrics,
):

    torch.save(

        {

            "stage":
                "COCO_PERSON_SPECIALIZATION",

            "epoch":
                int(
                    epoch
                ),

            "source_checkpoint":
                COCO80_BEST_CHECKPOINT_PATH,

            "model_state":
                model.state_dict(),

            "optimizer_state":
                optimizer.state_dict(),

            "val_report":
                copy.deepcopy(
                    val_report
                ),

            "metrics":
                copy.deepcopy(
                    metrics
                ),

            "freeze_policy": {

                "backbone":
                    "frozen",

                "query_encoder":
                    "frozen",

                "transformer_encoder":
                    "frozen",

                "transformer_decoder":
                    "frozen",

                "support_encoder":
                    "trainable",

                "relation_module":
                    "trainable",

                "detection_head":
                    "trainable",
            },

            "hyperparameters": {

                "epochs":
                    PERSON_SPECIALIZATION_EPOCHS,

                "episodes_per_epoch":
                    PERSON_SPECIALIZATION_EPISODES_PER_EPOCH,

                "accumulation_steps":
                    PERSON_SPECIALIZATION_ACCUMULATION,

                "lr":
                    PERSON_SPECIALIZATION_LR,

                "weight_decay":
                    PERSON_SPECIALIZATION_WEIGHT_DECAY,

                "score_threshold":
                    0.50,

                "iou_threshold":
                    0.50,

                "geometry_retention_fraction":
                    PERSON_GEOMETRY_RETENTION_FRACTION,
            },
        },

        path,
    )


# ==========================================================
# INITIAL REPORT
# ==========================================================

print("=" * 78)

print(
    "STEP 34 : COCO-PERSON SPECIALIZATION FALLBACK"
)

print("=" * 78)


print(
    "Source checkpoint        :",
    COCO80_BEST_CHECKPOINT_PATH,
)


print(
    "Epochs                   :",
    PERSON_SPECIALIZATION_EPOCHS,
)


print(
    "Episodes / epoch         :",
    PERSON_SPECIALIZATION_EPISODES_PER_EPOCH,
)


print(
    "Accumulation             :",
    PERSON_SPECIALIZATION_ACCUMULATION,
)


print(
    "LR                       :",
    PERSON_SPECIALIZATION_LR,
)


print(
    "Weight decay             :",
    PERSON_SPECIALIZATION_WEIGHT_DECAY,
)


print(
    "Trainable parameters     :",
    sum(
        parameter.numel()

        for parameter
        in trainable_parameters
    ),
)


print(
    "Frozen                   : "
    "backbone/query encoder/transformer enc+dec"
)


print(
    "Trainable                : "
    "support encoder/relation/detection head"
)


print("-" * 78)


print(
    "Baseline P@0.50          :",
    baseline_precision,
)


print(
    "Baseline R@0.50          :",
    baseline_recall,
)


print(
    "Baseline Geometry50      :",
    baseline_geometry,
)


print(
    "Geometry retention floor :",
    geometry_floor,
)


print(
    "Baseline val loss        :",
    baseline_loss,
)


print("=" * 78)


# ==========================================================
# TRAIN
# ==========================================================

person_specialization_history = []


best_rank = None

person_specialization_best_epoch = None

person_specialization_best_report = None


for epoch in range(
    1,
    PERSON_SPECIALIZATION_EPOCHS + 1,
):

    # ------------------------------------------------------
    # New deterministic person episodes each epoch.
    # ------------------------------------------------------

    person_train_dataset.set_epoch(
        epoch
    )


    # ------------------------------------------------------
    # Train 800 episodes.
    #
    # train_detection_epoch does NOT change requires_grad.
    #
    # We explicitly supply accumulation=1.
    # ------------------------------------------------------

    train_stats = (
        train_detection_epoch(

            target_model=
                person_specialization_model,

            loader=
                person_train_loader,

            optimizer=
                person_specialization_optimizer,

            max_episodes=
                PERSON_SPECIALIZATION_EPISODES_PER_EPOCH,

            stage=
                "stage2",

            description=(
                f"COCO-person specialization "
                f"{epoch}/{PERSON_SPECIALIZATION_EPOCHS}"
            ),

            show_progress=True,

            scheduler=None,

            accumulation_steps=
                PERSON_SPECIALIZATION_ACCUMULATION,
        )
    )


    # ======================================================
    # FIXED VALIDATION
    # ======================================================

    person_val_dataset.set_epoch(
        0
    )


    val_report = (
        evaluate_episodic_model(

            person_specialization_model,

            person_val_loader,

            show_progress=True,
        )
    )


    p50 = float(

        val_report[
            "metrics"
        ][
            "precision50"
        ]
    )


    r50 = float(

        val_report[
            "metrics"
        ][
            "recall50"
        ]
    )


    g50 = float(

        val_report[
            "metrics"
        ][
            "geometry_recall50"
        ]
    )


    val_loss = float(

        val_report[
            "mean_loss"
        ][
            "loss_total"
        ]
    )


    # ======================================================
    # F1 DIAGNOSTIC
    # ======================================================

    f1_50 = 0.0


    if (
        p50
        +
        r50
    ) > 0.0:

        f1_50 = (

            2.0

            *

            p50

            *

            r50

            /

            max(
                p50
                +
                r50,
                1e-12,
            )
        )


    # ======================================================
    # PRELIMINARY READINESS CANDIDATE
    #
    # This is NOT the final Step-35 gate.
    # ======================================================

    pr_alive = bool(

        p50 > 0.0

        and

        r50 > 0.0
    )


    geometry_retained = bool(

        g50

        >=

        geometry_floor
    )


    readiness_candidate = bool(

        pr_alive

        and

        geometry_retained
    )


    # ======================================================
    # CHECKPOINT RANK
    #
    # Priority:
    #
    # 1. readiness candidate
    # 2. F1@0.50
    # 3. Geometry50
    # 4. lower validation loss
    #
    # Confidence separation is checked in STEP 35.
    # ======================================================

    current_rank = (

        int(
            readiness_candidate
        ),

        float(
            f1_50
        ),

        float(
            g50
        ),

        -float(
            val_loss
        ),
    )


    epoch_metrics = {

        "precision50":
            p50,

        "recall50":
            r50,

        "f1_50":
            f1_50,

        "geometry50":
            g50,

        "geometry_retained":
            geometry_retained,

        "readiness_candidate":
            readiness_candidate,

        "val_loss":
            val_loss,

        "train_loss":
            float(
                train_stats[
                    "loss_total"
                ]
            ),
    }


    person_specialization_history.append(

        {

            "epoch":
                int(
                    epoch
                ),

            "train":
                copy.deepcopy(
                    train_stats
                ),

            "val":
                copy.deepcopy(
                    val_report
                ),

            "metrics":
                copy.deepcopy(
                    epoch_metrics
                ),
        }
    )


    # ======================================================
    # SAVE LATEST
    # ======================================================

    _save_person_specialization(

        PERSON_SPECIALIZATION_LATEST_CHECKPOINT_PATH,

        epoch,

        person_specialization_model,

        person_specialization_optimizer,

        val_report,

        epoch_metrics,
    )


    # ======================================================
    # SAVE BEST
    # ======================================================

    if (
        best_rank is None

        or

        current_rank > best_rank
    ):

        best_rank = current_rank


        person_specialization_best_epoch = (
            int(
                epoch
            )
        )


        person_specialization_best_report = (
            copy.deepcopy(
                val_report
            )
        )


        _save_person_specialization(

            PERSON_SPECIALIZATION_BEST_CHECKPOINT_PATH,

            epoch,

            person_specialization_model,

            person_specialization_optimizer,

            val_report,

            epoch_metrics,
        )


    # ======================================================
    # EPOCH LOG
    # ======================================================

    print(

        f"Epoch {epoch:02d}"

        f" | train={train_stats['loss_total']:.4f}"

        f" | val={val_loss:.4f}"

        f" | P={p50:.6f}"

        f" | R={r50:.6f}"

        f" | F1={f1_50:.6f}"

        f" | Geo={g50:.6f}"

        f" | GeoRetained={geometry_retained}"

        f" | Candidate={readiness_candidate}"
    )


# ==========================================================
# FINAL SANITY
# ==========================================================

if (
    person_specialization_best_epoch
    is None
):

    raise RuntimeError(
        "No best specialization checkpoint selected."
    )


if not os.path.isfile(
    PERSON_SPECIALIZATION_BEST_CHECKPOINT_PATH
):

    raise RuntimeError(
        "Best specialization checkpoint was not saved."
    )


PERSON_SPECIALIZATION_COMPLETE = True


# ==========================================================
# SUMMARY
# ==========================================================

print()

print("=" * 78)

print(
    "STEP 34 COMPLETE"
)

print("=" * 78)


print(
    "Best epoch       :",
    person_specialization_best_epoch,
)


print(
    "Best P@0.50      :",
    person_specialization_best_report[
        "metrics"
    ][
        "precision50"
    ],
)


print(
    "Best R@0.50      :",
    person_specialization_best_report[
        "metrics"
    ][
        "recall50"
    ],
)


print(
    "Best Geometry50  :",
    person_specialization_best_report[
        "metrics"
    ][
        "geometry_recall50"
    ],
)


print(
    "Best val loss    :",
    person_specialization_best_report[
        "mean_loss"
    ][
        "loss_total"
    ],
)


print(
    "Best checkpoint  :",
    PERSON_SPECIALIZATION_BEST_CHECKPOINT_PATH,
)


print(
    "NEXT             : "
    "STEP 35 re-gate + confidence separation"
)


print("=" * 78)


# ==========================================================
# CLEANUP
# ==========================================================

person_specialization_model.cpu()


del person_specialization_model
del person_specialization_optimizer
del _source_ckpt


gc.collect()


if torch.cuda.is_available():

    torch.cuda.empty_cache()
