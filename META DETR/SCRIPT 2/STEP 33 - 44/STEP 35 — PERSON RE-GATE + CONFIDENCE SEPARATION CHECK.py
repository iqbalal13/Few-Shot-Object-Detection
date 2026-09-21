# ==========================================================
# STEP 35 — PERSON RE-GATE + CONFIDENCE SEPARATION CHECK
# ==========================================================
#
# Compare:
#
# A. best COCO80 source checkpoint
# B. best COCO-person-specialized checkpoint
#
# using EXACTLY the same fixed COCO-Val person episodes.
#
#
# HARD PASS CONDITIONS:
#
# 1. Precision@0.50 > 0
# 2. Recall@0.50    > 0
# 3. Geometry50 retains >= 80% source-person Geometry50
# 4. foreground/background confidence separation improves
#
#
# Final thesis threshold remains:
#
# score >= 0.50
# IoU   >= 0.50
#
# ==========================================================


import os
import gc

import numpy as np
import torch


# ==========================================================
# PREREQUISITES
# ==========================================================

if not globals().get(
    "PERSON_SPECIALIZATION_COMPLETE",
    False,
):

    raise RuntimeError(
        "Run STEP 34 first."
    )


if not os.path.isfile(
    PERSON_SPECIALIZATION_BEST_CHECKPOINT_PATH
):

    raise RuntimeError(
        "Missing specialized checkpoint."
    )


if not os.path.isfile(
    COCO80_BEST_CHECKPOINT_PATH
):

    raise RuntimeError(
        "Missing source checkpoint."
    )


# ==========================================================
# LOCKED RE-GATE THRESHOLDS
# ==========================================================

RE_GATE_SCORE_THRESHOLD = 0.50

RE_GATE_IOU_THRESHOLD = 0.50

RE_GATE_GEOMETRY_RETENTION = 0.80


DIAGNOSTIC_SCORE_THRESHOLDS = (

    0.05,

    0.10,

    0.20,

    0.30,

    0.40,

    0.50,
)


# ==========================================================
# DISTRIBUTION HELPER
# ==========================================================

def _stats(
    values
):

    values = np.asarray(

        values,

        dtype=np.float64,
    )


    if values.size == 0:

        return {

            "count":
                0,

            "mean":
                None,

            "median":
                None,

            "p90":
                None,

            "p95":
                None,

            "max":
                None,
        }


    return {

        "count":
            int(
                values.size
            ),

        "mean":
            float(
                values.mean()
            ),

        "median":
            float(
                np.median(
                    values
                )
            ),

        "p90":
            float(
                np.percentile(
                    values,
                    90
                )
            ),

        "p95":
            float(
                np.percentile(
                    values,
                    95
                )
            ),

        "max":
            float(
                values.max()
            ),
    }


# ==========================================================
# DETAILED PERSON CHECKPOINT EVALUATOR
# ==========================================================

@torch.inference_mode()
def evaluate_person_checkpoint_detailed(
    checkpoint_path,
    description,
):

    # ======================================================
    # LOAD MODEL
    # ======================================================

    target_model = (
        make_trial_model()
    )


    checkpoint = torch.load(

        checkpoint_path,

        map_location=
            CONFIG[
                "device"
            ],

        weights_only=False,
    )


    target_model.load_state_dict(

        checkpoint[
            "model_state"
        ],

        strict=True,
    )


    target_model = (
        target_model.to(
            CONFIG[
                "device"
            ]
        )
    )


    target_model.eval()


    # Fixed validation episodes.
    person_val_dataset.set_epoch(
        0
    )


    # ======================================================
    # STORAGE
    # ======================================================

    records = []


    loss_totals = {

        "loss_cls":
            0.0,

        "loss_bbox":
            0.0,

        "loss_giou":
            0.0,

        "loss_total":
            0.0,
    }


    batch_count = 0


    all_query_scores = []


    # Prediction overlaps at least one GT at IoU >= .50.
    positive_query_scores = []


    # Prediction overlaps every GT at less than IoU .10.
    background_query_scores = []


    # For each geometrically-detectable GT:
    # maximum confidence among boxes IoU >= .50.
    valid_gt_max_scores = []


    # ======================================================
    # EVALUATION LOOP
    # ======================================================

    iterator = tqdm(

        person_val_loader,

        total=
            len(
                person_val_loader
            ),

        desc=
            description,
    )


    for batch in iterator:

        validate_episodic_batch(
            batch
        )


        support_images = (

            batch[
                "support_images"
            ]

            .to(
                CONFIG[
                    "device"
                ],
                non_blocking=True,
            )
        )


        support_padding_masks = (

            batch[
                "support_padding_masks"
            ]

            .to(
                CONFIG[
                    "device"
                ],
                non_blocking=True,
            )
        )


        query_images = (

            batch[
                "query_images"
            ]

            .to(
                CONFIG[
                    "device"
                ],
                non_blocking=True,
            )
        )


        query_padding_masks = (

            batch[
                "query_padding_masks"
            ]

            .to(
                CONFIG[
                    "device"
                ],
                non_blocking=True,
            )
        )


        targets = (
            move_targets_to_device(

                batch[
                    "query_targets"
                ],

                CONFIG[
                    "device"
                ],
            )
        )


        # ==================================================
        # FORWARD
        # ==================================================

        outputs = target_model(

            support_images,

            query_images,

            support_padding_mask=
                support_padding_masks,

            query_padding_mask=
                query_padding_masks,
        )


        # ==================================================
        # LOSS
        # ==================================================

        losses = criterion(

            outputs,

            targets,
        )


        for key in loss_totals:

            loss_totals[
                key
            ] += float(

                losses[
                    key
                ]

                .detach()

                .item()
            )


        batch_count += 1


        # ==================================================
        # SCORES / BOXES
        # ==================================================

        scores_tensor = (

            outputs[
                "pred_logits"
            ]

            .sigmoid()

            .squeeze(
                -1
            )
        )


        pred_boxes_tensor = (

            outputs[
                "pred_boxes"
            ]
        )


        # ==================================================
        # PER BATCH ITEM
        # ==================================================

        for batch_index in range(

            scores_tensor.shape[
                0
            ]
        ):

            semantic_label = int(

                batch[
                    "episode_classes"
                ][
                    batch_index
                ]

                .item()
            )


            if (
                semantic_label
                !=
                int(
                    PERSON_LABEL
                )
            ):

                raise RuntimeError(
                    "Non-person episode found in STEP 35."
                )


            scores = (

                scores_tensor[
                    batch_index
                ]

                .detach()

                .cpu()

                .numpy()

                .astype(
                    np.float64
                )

                .reshape(
                    -1
                )
            )


            pred_boxes = (

                pred_boxes_tensor[
                    batch_index
                ]

                .detach()

                .cpu()

                .numpy()

                .astype(
                    np.float64
                )

                .reshape(
                    -1,
                    4,
                )
            )


            gt_boxes = (

                batch[
                    "query_targets"
                ][
                    batch_index
                ][
                    "boxes"
                ]

                .detach()

                .cpu()

                .numpy()

                .astype(
                    np.float64
                )

                .reshape(
                    -1,
                    4,
                )
            )


            # ==================================================
            # RECORD FOR EXACT P/R EVALUATOR
            # ==================================================

            records.append(

                {

                    "semantic_label":
                        semantic_label,

                    "scores":
                        scores.copy(),

                    "pred_boxes":
                        pred_boxes.copy(),

                    "gt_boxes":
                        gt_boxes.copy(),
                }
            )


            all_query_scores.extend(
                scores.tolist()
            )


            if (
                len(
                    pred_boxes
                )
                ==
                0
            ):

                continue


            if (
                len(
                    gt_boxes
                )
                ==
                0
            ):

                background_query_scores.extend(
                    scores.tolist()
                )

                continue


            # ==================================================
            # IoU
            # ==================================================

            ious = numpy_box_iou(

                pred_boxes,

                gt_boxes,
            )


            # --------------------------------------------------
            # Prediction-level proxy labels.
            # --------------------------------------------------

            max_iou_per_prediction = (

                ious.max(
                    axis=1
                )
            )


            positive_mask = (

                max_iou_per_prediction

                >=

                RE_GATE_IOU_THRESHOLD
            )


            background_mask = (

                max_iou_per_prediction

                <

                0.10
            )


            positive_query_scores.extend(

                scores[
                    positive_mask
                ]

                .tolist()
            )


            background_query_scores.extend(

                scores[
                    background_mask
                ]

                .tolist()
            )


            # --------------------------------------------------
            # GT-level geometrically valid score.
            # --------------------------------------------------

            for gt_index in range(

                len(
                    gt_boxes
                )
            ):

                valid_mask = (

                    ious[
                        :,
                        gt_index
                    ]

                    >=

                    RE_GATE_IOU_THRESHOLD
                )


                if bool(
                    valid_mask.any()
                ):

                    valid_gt_max_scores.append(

                        float(

                            scores[
                                valid_mask
                            ]

                            .max()
                        )
                    )


    # ======================================================
    # SANITY
    # ======================================================

    if batch_count == 0:

        raise RuntimeError(
            "STEP 35 processed zero validation batches."
        )


    # ======================================================
    # MEAN LOSS
    # ======================================================

    mean_loss = {

        key:

            value
            /
            batch_count

        for key, value
        in loss_totals.items()
    }


    # ======================================================
    # FINAL METRICS @ SCORE .50 / IoU .50
    # ======================================================

    final_metrics = (

        compute_episodic_metrics(

            records,

            score_threshold=
                RE_GATE_SCORE_THRESHOLD,

            primary_iou_threshold=
                RE_GATE_IOU_THRESHOLD,
        )
    )


    # ======================================================
    # INTERNAL THRESHOLD SWEEP
    # ======================================================

    threshold_results = {

        threshold:

            compute_episodic_metrics(

                records,

                score_threshold=
                    threshold,

                primary_iou_threshold=
                    RE_GATE_IOU_THRESHOLD,
            )

        for threshold
        in DIAGNOSTIC_SCORE_THRESHOLDS
    }


    # ======================================================
    # SCORE DISTRIBUTIONS
    # ======================================================

    all_query_stats = (
        _stats(
            all_query_scores
        )
    )


    positive_stats = (
        _stats(
            positive_query_scores
        )
    )


    background_stats = (
        _stats(
            background_query_scores
        )
    )


    valid_gt_stats = (
        _stats(
            valid_gt_max_scores
        )
    )


    # ======================================================
    # FOREGROUND / BACKGROUND SEPARATION
    # ======================================================

    separation_gap = None


    if (

        positive_stats[
            "mean"
        ]
        is not None

        and

        background_stats[
            "mean"
        ]
        is not None
    ):

        separation_gap = float(

            positive_stats[
                "mean"
            ]

            -

            background_stats[
                "mean"
            ]
        )


    # ======================================================
    # LEARNED HEAD PARAMETERS
    # ======================================================

    class_bias = float(

        target_model
        .detection_head
        .class_bias

        .detach()

        .cpu()

        .item()
    )


    logit_scale = float(

        target_model
        .detection_head
        .get_logit_scale()

        .detach()

        .cpu()

        .item()
    )


    # ======================================================
    # RESULT
    # ======================================================

    result = {

        "checkpoint_path":
            checkpoint_path,

        "checkpoint_epoch":
            int(
                checkpoint.get(
                    "epoch",
                    -1,
                )
            ),

        "mean_loss":
            mean_loss,

        "metrics":
            final_metrics,

        "threshold_results":
            threshold_results,

        "all_query_score_stats":
            all_query_stats,

        "geometry_positive_query_score_stats":
            positive_stats,

        "background_query_score_stats":
            background_stats,

        "geometry_valid_gt_score_stats":
            valid_gt_stats,

        "separation_gap":
            separation_gap,

        "class_bias":
            class_bias,

        "logit_scale":
            logit_scale,
    }


    # ======================================================
    # CLEANUP MODEL
    # ======================================================

    target_model.cpu()


    del target_model
    del checkpoint


    gc.collect()


    if torch.cuda.is_available():

        torch.cuda.empty_cache()


    return result


# ==========================================================
# EVALUATE BOTH CHECKPOINTS ON IDENTICAL PERSON VALIDATION
# ==========================================================

source_person_detailed = (
    evaluate_person_checkpoint_detailed(

        COCO80_BEST_CHECKPOINT_PATH,

        "STEP 35 source-person diagnostic",
    )
)


specialized_person_detailed = (
    evaluate_person_checkpoint_detailed(

        PERSON_SPECIALIZATION_BEST_CHECKPOINT_PATH,

        "STEP 35 specialized-person diagnostic",
    )
)


# ==========================================================
# SOURCE VALUES
# ==========================================================

source_precision = float(

    source_person_detailed[
        "metrics"
    ][
        "precision50"
    ]
)


source_recall = float(

    source_person_detailed[
        "metrics"
    ][
        "recall50"
    ]
)


source_geometry = float(

    source_person_detailed[
        "metrics"
    ][
        "geometry_recall50"
    ]
)


source_gap = (

    source_person_detailed[
        "separation_gap"
    ]
)


# ==========================================================
# SPECIALIZED VALUES
# ==========================================================

specialized_precision = float(

    specialized_person_detailed[
        "metrics"
    ][
        "precision50"
    ]
)


specialized_recall = float(

    specialized_person_detailed[
        "metrics"
    ][
        "recall50"
    ]
)


specialized_geometry = float(

    specialized_person_detailed[
        "metrics"
    ][
        "geometry_recall50"
    ]
)


specialized_gap = (

    specialized_person_detailed[
        "separation_gap"
    ]
)


# ==========================================================
# HARD CONDITION 1 + 2:
# P/R @ .50 MUST BE ALIVE
# ==========================================================

specialized_pr_alive = bool(

    specialized_precision
    >
    0.0

    and

    specialized_recall
    >
    0.0
)


# ==========================================================
# HARD CONDITION 3:
# GEOMETRY MUST NOT COLLAPSE
# ==========================================================

geometry_retention_ratio = (

    specialized_geometry

    /

    max(
        source_geometry,
        1e-12,
    )
)


geometry_retained = bool(

    geometry_retention_ratio

    >=

    RE_GATE_GEOMETRY_RETENTION
)


# ==========================================================
# HARD CONDITION 4:
# FOREGROUND/BACKGROUND SEPARATION MUST IMPROVE
# ==========================================================

separation_improved = bool(

    source_gap
    is not None

    and

    specialized_gap
    is not None

    and

    specialized_gap
    >
    source_gap
)


# ==========================================================
# FINAL PERSON RE-GATE
# ==========================================================

COCO_PERSON_RE_GATE_PASSED = bool(

    specialized_pr_alive

    and

    geometry_retained

    and

    separation_improved
)


# ==========================================================
# CURRENT PIPELINE READINESS
# ==========================================================

COCO_PERSON_GATE_PASSED = (
    COCO_PERSON_RE_GATE_PASSED
)


PERSON_SPECIALIZATION_REQUIRED = (
    not COCO_PERSON_RE_GATE_PASSED
)


SOURCE_READY_FOR_CCTV = (
    COCO_PERSON_RE_GATE_PASSED
)


SOURCE_CHECKPOINT_FOR_CCTV = (

    PERSON_SPECIALIZATION_BEST_CHECKPOINT_PATH

    if SOURCE_READY_FOR_CCTV

    else None
)


# ==========================================================
# SAVE DIAGNOSTIC IN MEMORY
# ==========================================================

PERSON_RE_GATE_DIAGNOSTIC = {

    "source":
        source_person_detailed,

    "specialized":
        specialized_person_detailed,

    "geometry_retention_ratio":
        float(
            geometry_retention_ratio
        ),

    "specialized_pr_alive":
        specialized_pr_alive,

    "geometry_retained":
        geometry_retained,

    "separation_improved":
        separation_improved,

    "gate_passed":
        COCO_PERSON_RE_GATE_PASSED,

    "source_checkpoint_for_cctv":
        SOURCE_CHECKPOINT_FOR_CCTV,
}


# ==========================================================
# FINAL REPORT
# ==========================================================

print()

print("=" * 88)

print(
    "STEP 35 : PERSON RE-GATE + CONFIDENCE SEPARATION"
)

print("=" * 88)


print(
    "Source checkpoint              :",
    COCO80_BEST_CHECKPOINT_PATH,
)


print(
    "Specialized checkpoint         :",
    PERSON_SPECIALIZATION_BEST_CHECKPOINT_PATH,
)


print("-" * 88)


print(
    "Source P@0.50                  :",
    source_precision,
)


print(
    "Specialized P@0.50             :",
    specialized_precision,
)


print(
    "Source R@0.50                  :",
    source_recall,
)


print(
    "Specialized R@0.50             :",
    specialized_recall,
)


print(
    "P/R alive after specialization:",
    specialized_pr_alive,
)


print("-" * 88)


print(
    "Source Geometry50              :",
    source_geometry,
)


print(
    "Specialized Geometry50         :",
    specialized_geometry,
)


print(
    "Geometry retention ratio       :",
    geometry_retention_ratio,
)


print(
    "Required retention             :",
    RE_GATE_GEOMETRY_RETENTION,
)


print(
    "Geometry retained              :",
    geometry_retained,
)


print("-" * 88)


print(
    "Source positive score stats    :",
    source_person_detailed[
        "geometry_positive_query_score_stats"
    ],
)


print(
    "Specialized positive stats     :",
    specialized_person_detailed[
        "geometry_positive_query_score_stats"
    ],
)


print(
    "Source background score stats  :",
    source_person_detailed[
        "background_query_score_stats"
    ],
)


print(
    "Specialized background stats   :",
    specialized_person_detailed[
        "background_query_score_stats"
    ],
)


print(
    "Source separation gap          :",
    source_gap,
)


print(
    "Specialized separation gap     :",
    specialized_gap,
)


print(
    "Separation improved            :",
    separation_improved,
)


print("-" * 88)


print(
    "Source valid-GT score stats    :",
    source_person_detailed[
        "geometry_valid_gt_score_stats"
    ],
)


print(
    "Specialized valid-GT stats     :",
    specialized_person_detailed[
        "geometry_valid_gt_score_stats"
    ],
)


print("-" * 88)


print(
    "Source class bias              :",
    source_person_detailed[
        "class_bias"
    ],
)


print(
    "Specialized class bias         :",
    specialized_person_detailed[
        "class_bias"
    ],
)


print(
    "Source logit scale             :",
    source_person_detailed[
        "logit_scale"
    ],
)


print(
    "Specialized logit scale        :",
    specialized_person_detailed[
        "logit_scale"
    ],
)


# ==========================================================
# THRESHOLD SWEEP — SOURCE
# ==========================================================

print("-" * 88)

print(
    "THRESHOLD SWEEP — SOURCE"
)


for threshold in (
    DIAGNOSTIC_SCORE_THRESHOLDS
):

    metrics = (

        source_person_detailed[
            "threshold_results"
        ][
            threshold
        ]
    )


    print(

        f"{threshold:.2f}"

        f" | P={metrics['precision50']:.6f}"

        f" | R={metrics['recall50']:.6f}"

        f" | TP={metrics['tp']}"

        f" | FP={metrics['fp']}"

        f" | FN={metrics['fn']}"
    )


# ==========================================================
# THRESHOLD SWEEP — SPECIALIZED
# ==========================================================

print("-" * 88)

print(
    "THRESHOLD SWEEP — SPECIALIZED"
)


for threshold in (
    DIAGNOSTIC_SCORE_THRESHOLDS
):

    metrics = (

        specialized_person_detailed[
            "threshold_results"
        ][
            threshold
        ]
    )


    print(

        f"{threshold:.2f}"

        f" | P={metrics['precision50']:.6f}"

        f" | R={metrics['recall50']:.6f}"

        f" | TP={metrics['tp']}"

        f" | FP={metrics['fp']}"

        f" | FN={metrics['fn']}"
    )


# ==========================================================
# FINAL STATE
# ==========================================================

print("=" * 88)


print(
    "PERSON RE-GATE PASSED          :",
    COCO_PERSON_RE_GATE_PASSED,
)


print(
    "SOURCE READY FOR CCTV          :",
    SOURCE_READY_FOR_CCTV,
)


print(
    "SOURCE CHECKPOINT FOR CCTV     :",
    SOURCE_CHECKPOINT_FOR_CCTV,
)


print("=" * 88)


if SOURCE_READY_FOR_CCTV:

    print(
        "PASS: use the specialized checkpoint "
        "as the common source initialization "
        "for CCTV 1/3/5-shot."
    )

else:

    print(
        "FAIL: do not start CCTV yet. "
        "Inspect the specialization diagnostics "
        "before changing the classification objective."
    )


print("=" * 88)
