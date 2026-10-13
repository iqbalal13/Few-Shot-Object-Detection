# ============================================================
# STEP 29 — SHORT COCO-80 TRAINING + VALIDATION
# AMP + stable FP32 loss + gradient clipping
# ============================================================

import os
import math
import torch
import numpy as np

from torchvision.ops import box_iou

print("=" * 70)
print("STEP 29 — SHORT COCO-80 TRAINING + VALIDATION")
print("=" * 70)

GRAD_CLIP_NORM = 10.0


# ============================================================
# DETECTION MATCHING
# ============================================================

def greedy_detection_counts(
    prediction,
    target,
    score_threshold=0.50,
    iou_threshold=0.50,
    class_filter=None
):

    pred_boxes = prediction[
        "boxes"
    ].detach().cpu()

    pred_scores = prediction[
        "scores"
    ].detach().cpu()

    pred_labels = prediction[
        "labels"
    ].detach().cpu()

    gt_boxes = target[
        "boxes"
    ].detach().cpu()

    gt_labels = target[
        "labels"
    ].detach().cpu()


    # --------------------------------------------------------
    # Optional class-only evaluation
    # --------------------------------------------------------

    if class_filter is not None:

        pred_keep = (
            pred_labels
            == class_filter
        )

        pred_boxes = pred_boxes[
            pred_keep
        ]

        pred_scores = pred_scores[
            pred_keep
        ]

        pred_labels = pred_labels[
            pred_keep
        ]


        gt_keep = (
            gt_labels
            == class_filter
        )

        gt_boxes = gt_boxes[
            gt_keep
        ]

        gt_labels = gt_labels[
            gt_keep
        ]


    # --------------------------------------------------------
    # Confidence threshold
    # --------------------------------------------------------

    keep = (
        pred_scores
        >= score_threshold
    )

    pred_boxes = pred_boxes[
        keep
    ]

    pred_scores = pred_scores[
        keep
    ]

    pred_labels = pred_labels[
        keep
    ]


    if len(pred_scores) > 0:

        order = torch.argsort(
            pred_scores,
            descending=True
        )

        pred_boxes = pred_boxes[
            order
        ]

        pred_scores = pred_scores[
            order
        ]

        pred_labels = pred_labels[
            order
        ]


    matched_gt = set()

    true_positive = 0
    false_positive = 0


    for pred_index in range(
        len(pred_boxes)
    ):

        same_class_indices = torch.where(
            gt_labels
            == pred_labels[pred_index]
        )[0]

        available_indices = [
            int(idx)
            for idx in same_class_indices
            if int(idx) not in matched_gt
        ]

        if len(
            available_indices
        ) == 0:

            false_positive += 1
            continue


        candidate_boxes = gt_boxes[
            available_indices
        ]

        ious = box_iou(
            pred_boxes[
                pred_index:
                pred_index + 1
            ],
            candidate_boxes
        )[0]

        (
            best_value,
            best_local_index
        ) = torch.max(
            ious,
            dim=0
        )

        if (
            float(best_value)
            >= iou_threshold
        ):

            matched_global_index = \
                available_indices[
                    int(
                        best_local_index
                    )
                ]

            matched_gt.add(
                matched_global_index
            )

            true_positive += 1

        else:

            false_positive += 1


    false_negative = (
        len(gt_boxes)
        - true_positive
    )

    return {
        "tp": true_positive,
        "fp": false_positive,
        "fn": false_negative,
        "gt": len(gt_boxes)
    }


# ============================================================
# VALIDATION FUNCTION
# ============================================================

def evaluate_centernet(
    model,
    loader,
    max_batches=None,
    score_threshold=0.50,
    iou_threshold=0.50,
    class_filter=None,
    compute_loss=True
):

    model.eval()

    total_loss = 0.0
    loss_batches = 0

    total_tp = 0
    total_fp = 0
    total_fn = 0

    geometry_matches = 0
    geometry_gt = 0


    with torch.no_grad():

        for batch_index, (
            images,
            targets
        ) in enumerate(loader):


            if (
                max_batches is not None
                and batch_index
                >= max_batches
            ):
                break


            images = images.to(
                device,
                non_blocking=True
            )


            # Model forward stays AMP
            with torch.amp.autocast(
                device_type="cuda",
                dtype=torch.float16
            ):

                outputs = model(
                    images
                )


            # Loss itself is FP32 inside Step 24
            if compute_loss:

                encoded_targets = \
                    build_batch_centernet_targets(
                        targets,
                        device=device
                    )

                losses = criterion(
                    outputs,
                    encoded_targets
                )

                loss_value = losses[
                    "loss_total"
                ]

                if not torch.isfinite(
                    loss_value
                ):

                    raise FloatingPointError(
                        f"Non-finite validation loss "
                        f"at batch {batch_index + 1}. "
                        f"hm={float(losses['loss_heatmap'].detach()):.6f}, "
                        f"wh={float(losses['loss_wh'].detach()):.6f}, "
                        f"off={float(losses['loss_offset'].detach()):.6f}"
                    )

                total_loss += float(
                    loss_value.detach()
                )

                loss_batches += 1


            # Decode with confidence ignored
            # for Geometry50.
            predictions = decode_centernet(
                outputs,
                K=100,
                score_threshold=0.0
            )


            for prediction, target in zip(
                predictions,
                targets
            ):

                # --------------------------------------------
                # Primary Precision / Recall
                # score >= 0.50
                # IoU >= 0.50
                # --------------------------------------------

                primary = \
                    greedy_detection_counts(
                        prediction,
                        target,
                        score_threshold=
                            score_threshold,
                        iou_threshold=
                            iou_threshold,
                        class_filter=
                            class_filter
                    )

                total_tp += primary[
                    "tp"
                ]

                total_fp += primary[
                    "fp"
                ]

                total_fn += primary[
                    "fn"
                ]


                # --------------------------------------------
                # Geometry50
                # confidence ignored
                # --------------------------------------------

                geometry = \
                    greedy_detection_counts(
                        prediction,
                        target,
                        score_threshold=0.0,
                        iou_threshold=
                            iou_threshold,
                        class_filter=
                            class_filter
                    )

                geometry_matches += \
                    geometry["tp"]

                geometry_gt += \
                    geometry["gt"]


    precision = (
        total_tp
        / max(
            total_tp + total_fp,
            1
        )
    )

    recall = (
        total_tp
        / max(
            total_tp + total_fn,
            1
        )
    )

    geometry50 = (
        geometry_matches
        / max(
            geometry_gt,
            1
        )
    )

    mean_loss = (
        total_loss
        / max(
            loss_batches,
            1
        )
        if compute_loss
        else None
    )


    return {
        "loss":
            mean_loss,

        "precision":
            precision,

        "recall":
            recall,

        "geometry50":
            geometry50,

        "tp":
            total_tp,

        "fp":
            total_fp,

        "fn":
            total_fn,

        "gt":
            geometry_gt
    }


# ============================================================
# GENERIC TRAIN FUNCTION
# ============================================================

def train_centernet_epoch(
    model,
    loader,
    optimizer,
    scaler,
    max_batches=None
):

    model.train()

    set_backbone_bn_eval(
        model
    )

    total_loss = 0.0
    total_hm = 0.0
    total_wh = 0.0
    total_offset = 0.0

    successful_batches = 0
    skipped_batches = 0


    for batch_index, (
        images,
        targets
    ) in enumerate(loader):


        if (
            max_batches is not None
            and batch_index
            >= max_batches
        ):
            break


        images = images.to(
            device,
            non_blocking=True
        )


        encoded_targets = \
            build_batch_centernet_targets(
                targets,
                device=device
            )


        optimizer.zero_grad(
            set_to_none=True
        )


        # ----------------------------------------------------
        # Forward stays FP16 AMP
        # ----------------------------------------------------

        with torch.amp.autocast(
            device_type="cuda",
            dtype=torch.float16
        ):

            outputs = model(
                images
            )


        # ----------------------------------------------------
        # Step-24 criterion calculates losses in FP32
        # ----------------------------------------------------

        losses = criterion(
            outputs,
            encoded_targets
        )

        total = losses[
            "loss_total"
        ]


        # ----------------------------------------------------
        # Catch loss problem with component information
        # ----------------------------------------------------

        if not torch.isfinite(
            total
        ):

            raise FloatingPointError(
                f"Non-finite training loss "
                f"at batch {batch_index + 1}. "
                f"hm={float(losses['loss_heatmap'].detach()):.6f}, "
                f"wh={float(losses['loss_wh'].detach()):.6f}, "
                f"off={float(losses['loss_offset'].detach()):.6f}"
            )


        # ----------------------------------------------------
        # Backward with GradScaler
        # ----------------------------------------------------

        scaler.scale(
            total
        ).backward()


        # ----------------------------------------------------
        # Unscale BEFORE clipping
        # ----------------------------------------------------

        scaler.unscale_(
            optimizer
        )


        grad_norm = \
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=GRAD_CLIP_NORM,
                error_if_nonfinite=False
            )


        # ----------------------------------------------------
        # Rare AMP overflow protection
        # ----------------------------------------------------

        if not torch.isfinite(
            grad_norm
        ):

            skipped_batches += 1

            optimizer.zero_grad(
                set_to_none=True
            )

            scaler.update()

            print(
                f"  WARNING batch "
                f"{batch_index + 1}: "
                f"non-finite gradient; "
                f"optimizer step skipped."
            )

            continue


        scaler.step(
            optimizer
        )

        scaler.update()


        total_loss += float(
            total.detach()
        )

        total_hm += float(
            losses[
                "loss_heatmap"
            ].detach()
        )

        total_wh += float(
            losses[
                "loss_wh"
            ].detach()
        )

        total_offset += float(
            losses[
                "loss_offset"
            ].detach()
        )

        successful_batches += 1


        if (
            batch_index == 0
            or (
                batch_index + 1
            ) % 50 == 0
        ):

            print(
                f"  batch "
                f"{batch_index + 1:04d} | "
                f"loss "
                f"{float(total.detach()):.4f} | "
                f"grad_norm "
                f"{float(grad_norm.detach()):.4f}"
            )


    if successful_batches == 0:

        raise RuntimeError(
            "No successful optimizer "
            "update in this epoch."
        )


    return {
        "loss":
            total_loss
            / successful_batches,

        "heatmap":
            total_hm
            / successful_batches,

        "wh":
            total_wh
            / successful_batches,

        "offset":
            total_offset
            / successful_batches,

        "successful_batches":
            successful_batches,

        "skipped_batches":
            skipped_batches
    }


# ============================================================
# SHORT TRAINING CONFIG
# ============================================================

SHORT_EPOCHS = 2
SHORT_TRAIN_BATCHES = 300
SHORT_VAL_BATCHES = 100


# Fresh source model
short_model = \
    build_fresh_source_model()

short_optimizer = \
    build_source_optimizer(
        short_model
    )

short_scaler = \
    torch.amp.GradScaler(
        "cuda",
        enabled=True
    )


# ------------------------------------------------------------
# Initial validation
# ------------------------------------------------------------

print(
    "\nInitial short validation..."
)

initial_val = evaluate_centernet(
    short_model,
    val_loader,
    max_batches=
        SHORT_VAL_BATCHES,
    score_threshold=
        CONFIG[
            "score_threshold"
        ],
    iou_threshold=
        CONFIG[
            "iou_threshold"
        ]
)


print(
    f"Initial val loss   : "
    f"{initial_val['loss']:.4f}"
)

print(
    f"Initial Geometry50 : "
    f"{initial_val['geometry50']:.4f}"
)


# ------------------------------------------------------------
# Short training
# ------------------------------------------------------------

best_short_geometry = -1.0


SHORT_CHECKPOINT = \
    os.path.join(
        CONFIG[
            "checkpoint_root"
        ],
        "centernet_coco80_short_best.pth"
    )


for epoch in range(
    1,
    SHORT_EPOCHS + 1
):

    print(
        "\n"
        + "=" * 70
    )

    print(
        f"SHORT EPOCH "
        f"{epoch}/"
        f"{SHORT_EPOCHS}"
    )

    print(
        "=" * 70
    )


    train_stats = \
        train_centernet_epoch(
            short_model,
            train_loader,
            short_optimizer,
            short_scaler,
            max_batches=
                SHORT_TRAIN_BATCHES
        )


    val_stats = \
        evaluate_centernet(
            short_model,
            val_loader,
            max_batches=
                SHORT_VAL_BATCHES,
            score_threshold=
                CONFIG[
                    "score_threshold"
                ],
            iou_threshold=
                CONFIG[
                    "iou_threshold"
                ]
        )


    print("\nTRAIN")

    print(
        f"loss      : "
        f"{train_stats['loss']:.4f}"
    )

    print(
        f"heatmap   : "
        f"{train_stats['heatmap']:.4f}"
    )

    print(
        f"wh        : "
        f"{train_stats['wh']:.4f}"
    )

    print(
        f"offset    : "
        f"{train_stats['offset']:.4f}"
    )

    print(
        f"skipped   : "
        f"{train_stats['skipped_batches']}"
    )


    print("\nVALIDATION")

    print(
        f"loss      : "
        f"{val_stats['loss']:.4f}"
    )

    print(
        f"Geometry50: "
        f"{val_stats['geometry50']:.4f}"
    )

    print(
        f"Precision : "
        f"{val_stats['precision']:.4f}"
    )

    print(
        f"Recall    : "
        f"{val_stats['recall']:.4f}"
    )


    if (
        val_stats[
            "geometry50"
        ]
        > best_short_geometry
    ):

        best_short_geometry = \
            val_stats[
                "geometry50"
            ]

        torch.save(
            {
                "model_state_dict":
                    short_model.state_dict(),

                "epoch":
                    epoch,

                "val_stats":
                    val_stats
            },
            SHORT_CHECKPOINT
        )

        print(
            "\nSHORT BEST CHECKPOINT SAVED"
        )


print(
    "\n"
    + "=" * 70
)

print(
    "STEP 29 COMPLETE"
)

print(
    "=" * 70
)

print(
    "Best short Geometry50 :",
    best_short_geometry
)

print(
    "Checkpoint            :",
    SHORT_CHECKPOINT
)

print(
    "Gradient clip norm    :",
    GRAD_CLIP_NORM
)

print(
    "\nSTEP 29 PASSED"
)
