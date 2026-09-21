# ==========================================================
# STEP 33 — PERSON CONFIDENCE DIAGNOSTIC
#
# PURPOSE
# ----------------------------------------------------------
# Diagnose why COCO-Val PERSON has:
#
#   Geometry50 > 0
#   but
#   Precision@0.50 = 0
#   Recall@0.50    = 0
#
#
# THIS STEP DOES NOT TRAIN THE MODEL.
# THIS STEP DOES NOT MODIFY THE CHECKPOINT.
#
#
# DIAGNOSTICS
# ----------------------------------------------------------
# 1. Precision / Recall threshold sweep:
#
#       0.05
#       0.10
#       0.20
#       0.30
#       0.40
#       0.50
#
# 2. Distribution of all query confidence scores.
#
# 3. Distribution of scores attached to geometrically
#    valid predictions (IoU >= 0.50).
#
# 4. Detection-head calibration:
#
#       logit = scale * cosine_similarity + class_bias
#
#    Since cosine similarity is bounded [-1, 1],
#    we can calculate whether score >= 0.50 is even
#    mathematically reachable with the learned head.
#
# ==========================================================


import os
import math
import numpy as np
import torch


# ==========================================================
# REQUIRE COMPLETED SOURCE TRAINING
# ==========================================================

if not COCO80_META_TRAINING_COMPLETE:

    raise RuntimeError(
        'STEP 30 COCO-80 meta-training is not complete.'
    )


if not os.path.isfile(
    COCO80_BEST_CHECKPOINT_PATH
):

    raise RuntimeError(
        'Best COCO-80 checkpoint does not exist: '
        f'{COCO80_BEST_CHECKPOINT_PATH}'
    )


if 'person_val_loader' not in globals():

    raise RuntimeError(
        'person_val_loader is missing. '
        'Run STEP 31 first.'
    )


# ==========================================================
# LOCKED DIAGNOSTIC SETTINGS
# ==========================================================

PERSON_DIAGNOSTIC_THRESHOLDS = (
    0.05,
    0.10,
    0.20,
    0.30,
    0.40,
    0.50,
)

PERSON_DIAGNOSTIC_IOU = 0.50


# ==========================================================
# FIX PERSON VALIDATION EPISODES
# ==========================================================

person_val_dataset.set_epoch(
    0
)


# ==========================================================
# LOAD BEST STAGE-1 CHECKPOINT
# ==========================================================

diagnostic_model = (
    make_trial_model()
)


diagnostic_checkpoint = torch.load(

    COCO80_BEST_CHECKPOINT_PATH,

    map_location=
        CONFIG[
            'device'
        ],

    weights_only=False,
)


diagnostic_model.load_state_dict(

    diagnostic_checkpoint[
        'model_state'
    ],

    strict=True,
)


diagnostic_model = (
    diagnostic_model.to(
        CONFIG[
            'device'
        ]
    )
)


diagnostic_model.eval()


# ==========================================================
# READ LEARNED DETECTION-HEAD CALIBRATION
#
# Current head:
#
# logit =
#     learned_scale * cosine_similarity
#     + learned_class_bias
#
# cosine_similarity in [-1, 1].
# ==========================================================

with torch.inference_mode():

    learned_class_bias = float(

        diagnostic_model
        .detection_head
        .class_bias
        .detach()
        .cpu()
        .item()
    )


    learned_logit_scale = float(

        diagnostic_model
        .detection_head
        .get_logit_scale()
        .detach()
        .cpu()
        .item()
    )


# ----------------------------------------------------------
# score = 0.50 corresponds to logit = 0.
#
# therefore:
#
# 0 = scale * similarity + bias
#
# similarity_required =
#     -bias / scale
# ----------------------------------------------------------

similarity_required_for_050 = (

    -learned_class_bias

    /

    max(
        learned_logit_scale,
        1e-12,
    )
)


# ----------------------------------------------------------
# Maximum possible score under cosine similarity = +1.
# ----------------------------------------------------------

maximum_possible_logit = (

    learned_logit_scale
    +
    learned_class_bias
)


maximum_possible_score = (

    1.0

    /

    (
        1.0
        +
        math.exp(
            -maximum_possible_logit
        )
    )
)


score_050_mathematically_reachable = bool(

    similarity_required_for_050
    <=
    1.0
)


# ==========================================================
# STORAGE
# ==========================================================

all_query_scores = []

all_query_logits = []

image_max_scores = []


# ----------------------------------------------------------
# For every GT:
#
# best_iou_per_gt:
#     maximum IoU obtained by any prediction.
#
# best_iou_query_scores:
#     score belonging to the prediction with maximum IoU.
#
# geometry_valid_max_scores:
#     when at least one prediction has IoU >= 0.50,
#     take the HIGHEST confidence among those valid boxes.
#
# The last one is especially important:
# it tells us whether geometrically correct predictions
# are simply below the score threshold.
# ----------------------------------------------------------

best_iou_per_gt = []

best_iou_query_scores = []

geometry_valid_max_scores = []


total_gt = 0

geometry_gt_hits = 0


# ==========================================================
# THRESHOLD COUNTERS
# ==========================================================

threshold_counts = {

    threshold: {
        'tp': 0,
        'fp': 0,
        'fn': 0,
    }

    for threshold
    in PERSON_DIAGNOSTIC_THRESHOLDS
}


# ==========================================================
# RUN PERSON-ONLY DIAGNOSTIC
# ==========================================================

device = CONFIG[
    'device'
]


@torch.inference_mode()
def run_person_confidence_diagnostic():

    global total_gt
    global geometry_gt_hits


    iterator = tqdm(

        person_val_loader,

        total=len(
            person_val_loader
        ),

        desc='STEP 33 Person Confidence Diagnostic',
    )


    for batch in iterator:

        # --------------------------------------------------
        # Validate batch using existing notebook validator.
        # --------------------------------------------------

        validate_episodic_batch(
            batch
        )


        # --------------------------------------------------
        # Person-only sanity.
        # --------------------------------------------------

        episode_classes = (
            batch[
                'episode_classes'
            ]
        )


        for label in (
            episode_classes
            .detach()
            .cpu()
            .reshape(-1)
            .tolist()
        ):

            if int(label) != int(
                PERSON_LABEL
            ):

                raise RuntimeError(
                    'STEP 33 received a non-person episode: '
                    f'label={label}'
                )


        # --------------------------------------------------
        # Device transfer
        # --------------------------------------------------

        support_images = (

            batch[
                'support_images'
            ]

            .to(
                device,
                non_blocking=True,
            )
        )


        support_padding_masks = (

            batch[
                'support_padding_masks'
            ]

            .to(
                device,
                non_blocking=True,
            )
        )


        query_images = (

            batch[
                'query_images'
            ]

            .to(
                device,
                non_blocking=True,
            )
        )


        query_padding_masks = (

            batch[
                'query_padding_masks'
            ]

            .to(
                device,
                non_blocking=True,
            )
        )


        # --------------------------------------------------
        # Forward
        # --------------------------------------------------

        outputs = diagnostic_model(

            support_images,

            query_images,

            support_padding_mask=
                support_padding_masks,

            query_padding_mask=
                query_padding_masks,
        )


        logits_tensor = (

            outputs[
                'pred_logits'
            ]

            .squeeze(
                -1
            )
        )


        scores_tensor = (
            logits_tensor.sigmoid()
        )


        pred_boxes_tensor = (

            outputs[
                'pred_boxes'
            ]
        )


        # --------------------------------------------------
        # Physical batch is locked to 1 here.
        # --------------------------------------------------

        if (
            scores_tensor.shape[0]
            !=
            1
        ):

            raise RuntimeError(
                'STEP 33 expects evaluation batch size 1.'
            )


        scores = (

            scores_tensor[
                0
            ]

            .detach()
            .cpu()
            .numpy()
            .astype(
                np.float64
            )
        )


        logits = (

            logits_tensor[
                0
            ]

            .detach()
            .cpu()
            .numpy()
            .astype(
                np.float64
            )
        )


        pred_boxes = (

            pred_boxes_tensor[
                0
            ]

            .detach()
            .cpu()
            .numpy()
            .astype(
                np.float64
            )
        )


        gt_boxes = (

            batch[
                'query_targets'
            ][
                0
            ][
                'boxes'
            ]

            .detach()
            .cpu()
            .numpy()
            .astype(
                np.float64
            )
        )


        gt_boxes = gt_boxes.reshape(
            -1,
            4,
        )


        pred_boxes = pred_boxes.reshape(
            -1,
            4,
        )


        scores = scores.reshape(
            -1
        )


        logits = logits.reshape(
            -1
        )


        # --------------------------------------------------
        # General score distributions
        # --------------------------------------------------

        all_query_scores.extend(
            scores.tolist()
        )


        all_query_logits.extend(
            logits.tolist()
        )


        if len(
            scores
        ) > 0:

            image_max_scores.append(
                float(
                    scores.max()
                )
            )


        total_gt += int(
            len(
                gt_boxes
            )
        )


        # ==================================================
        # GEOMETRY-CONFIDENCE RELATION
        # ==================================================

        if (
            len(
                pred_boxes
            )
            >
            0

            and

            len(
                gt_boxes
            )
            >
            0
        ):

            ious = numpy_box_iou(

                pred_boxes,

                gt_boxes,
            )


            # shape:
            # [num_predictions, num_gt]

            for gt_index in range(
                len(
                    gt_boxes
                )
            ):

                gt_ious = (

                    ious[
                        :,
                        gt_index
                    ]
                )


                best_prediction_index = int(

                    np.argmax(
                        gt_ious
                    )
                )


                best_iou = float(

                    gt_ious[
                        best_prediction_index
                    ]
                )


                score_of_best_iou_query = float(

                    scores[
                        best_prediction_index
                    ]
                )


                best_iou_per_gt.append(
                    best_iou
                )


                best_iou_query_scores.append(
                    score_of_best_iou_query
                )


                # ------------------------------------------
                # All queries whose box is geometrically
                # valid for this GT.
                # ------------------------------------------

                valid_geometry_mask = (

                    gt_ious
                    >=
                    PERSON_DIAGNOSTIC_IOU
                )


                if bool(
                    valid_geometry_mask.any()
                ):

                    geometry_gt_hits += 1


                    max_valid_score = float(

                        scores[
                            valid_geometry_mask
                        ].max()
                    )


                    geometry_valid_max_scores.append(
                        max_valid_score
                    )


        # ==================================================
        # EXACT P/R THRESHOLD SWEEP
        #
        # Uses the SAME greedy evaluator logic as STEP 25.
        # ==================================================

        for threshold in (
            PERSON_DIAGNOSTIC_THRESHOLDS
        ):

            keep = (

                scores
                >=
                threshold
            )


            kept_scores = (

                scores[
                    keep
                ]
            )


            kept_boxes = (

                pred_boxes[
                    keep
                ]
            )


            detection_flags = (

                greedy_detection_flags(

                    kept_scores,

                    kept_boxes,

                    gt_boxes,

                    PERSON_DIAGNOSTIC_IOU,
                )
            )


            episode_tp = int(

                detection_flags.sum()
            )


            episode_fp = int(

                len(
                    kept_scores
                )
                -
                episode_tp
            )


            episode_fn = int(

                len(
                    gt_boxes
                )
                -
                episode_tp
            )


            threshold_counts[
                threshold
            ][
                'tp'
            ] += episode_tp


            threshold_counts[
                threshold
            ][
                'fp'
            ] += episode_fp


            threshold_counts[
                threshold
            ][
                'fn'
            ] += episode_fn


run_person_confidence_diagnostic()


# ==========================================================
# CONVERT ARRAYS
# ==========================================================

all_query_scores = np.asarray(
    all_query_scores,
    dtype=np.float64,
)


all_query_logits = np.asarray(
    all_query_logits,
    dtype=np.float64,
)


image_max_scores = np.asarray(
    image_max_scores,
    dtype=np.float64,
)


best_iou_per_gt = np.asarray(
    best_iou_per_gt,
    dtype=np.float64,
)


best_iou_query_scores = np.asarray(
    best_iou_query_scores,
    dtype=np.float64,
)


geometry_valid_max_scores = np.asarray(
    geometry_valid_max_scores,
    dtype=np.float64,
)


# ==========================================================
# BASIC SANITY
# ==========================================================

if total_gt < 1:

    raise RuntimeError(
        'STEP 33 found no person ground-truth boxes.'
    )


if all_query_scores.size < 1:

    raise RuntimeError(
        'STEP 33 produced no prediction scores.'
    )


# ==========================================================
# THRESHOLD METRICS
# ==========================================================

threshold_results = {}


for threshold in (
    PERSON_DIAGNOSTIC_THRESHOLDS
):

    tp = int(
        threshold_counts[
            threshold
        ][
            'tp'
        ]
    )


    fp = int(
        threshold_counts[
            threshold
        ][
            'fp'
        ]
    )


    fn = int(
        threshold_counts[
            threshold
        ][
            'fn'
        ]
    )


    precision = (

        tp

        /

        max(
            tp + fp,
            1,
        )
    )


    recall = (

        tp

        /

        max(
            total_gt,
            1,
        )
    )


    threshold_results[
        threshold
    ] = {

        'precision':
            float(
                precision
            ),

        'recall':
            float(
                recall
            ),

        'tp':
            tp,

        'fp':
            fp,

        'fn':
            fn,
    }


# ==========================================================
# HELPER FOR DISTRIBUTION STATS
# ==========================================================

def diagnostic_distribution(
    values
):

    values = np.asarray(
        values,
        dtype=np.float64,
    )


    if values.size == 0:

        return {
            'count': 0,
            'min': None,
            'mean': None,
            'median': None,
            'p75': None,
            'p90': None,
            'p95': None,
            'p99': None,
            'max': None,
        }


    return {

        'count':
            int(
                values.size
            ),

        'min':
            float(
                values.min()
            ),

        'mean':
            float(
                values.mean()
            ),

        'median':
            float(
                np.median(
                    values
                )
            ),

        'p75':
            float(
                np.percentile(
                    values,
                    75
                )
            ),

        'p90':
            float(
                np.percentile(
                    values,
                    90
                )
            ),

        'p95':
            float(
                np.percentile(
                    values,
                    95
                )
            ),

        'p99':
            float(
                np.percentile(
                    values,
                    99
                )
            ),

        'max':
            float(
                values.max()
            ),
    }


all_score_stats = (
    diagnostic_distribution(
        all_query_scores
    )
)


image_max_score_stats = (
    diagnostic_distribution(
        image_max_scores
    )
)


best_iou_score_stats = (
    diagnostic_distribution(
        best_iou_query_scores
    )
)


geometry_valid_score_stats = (
    diagnostic_distribution(
        geometry_valid_max_scores
    )
)


best_iou_stats = (
    diagnostic_distribution(
        best_iou_per_gt
    )
)


# ==========================================================
# GEOMETRY RECALL RECONSTRUCTION
# ==========================================================

diagnostic_geometry50 = (

    geometry_gt_hits

    /

    max(
        total_gt,
        1,
    )
)


# ==========================================================
# SCORE COVERAGE AMONG GEOMETRICALLY VALID GTs
#
# For each GT that has at least one box IoU >= 0.50,
# ask whether at least one such valid box also exceeds
# the confidence threshold.
# ==========================================================

geometry_score_coverage = {}


for threshold in (
    PERSON_DIAGNOSTIC_THRESHOLDS
):

    if (
        geometry_valid_max_scores.size
        >
        0
    ):

        coverage = float(

            (
                geometry_valid_max_scores
                >=
                threshold
            )
            .mean()
        )

    else:

        coverage = 0.0


    geometry_score_coverage[
        threshold
    ] = coverage


# ==========================================================
# AUTOMATIC DIAGNOSIS
# ==========================================================

pr_at_050 = (

    threshold_results[
        0.50
    ][
        'precision'
    ],
    threshold_results[
        0.50
    ][
        'recall'
    ],
)


lower_threshold_recall_alive = bool(

    any(

        threshold_results[
            threshold
        ][
            'recall'
        ]
        >
        0.0

        for threshold
        in PERSON_DIAGNOSTIC_THRESHOLDS

        if threshold < 0.50
    )
)


if (
    diagnostic_geometry50
    <=
    0.0
):

    CONFIDENCE_DIAGNOSIS = (
        'LOCALIZATION BOTTLENECK: '
        'no GT reaches IoU >= 0.50.'
    )


elif (
    not score_050_mathematically_reachable
):

    CONFIDENCE_DIAGNOSIS = (
        'HEAD CALIBRATION BOTTLENECK: '
        'with the learned class bias and logit scale, '
        'score 0.50 is mathematically unreachable even '
        'at cosine similarity = 1.'
    )


elif (
    pr_at_050[
        1
    ]
    ==
    0.0

    and

    lower_threshold_recall_alive
):

    CONFIDENCE_DIAGNOSIS = (
        'UNDER-CONFIDENCE / CALIBRATION BOTTLENECK: '
        'geometrically valid detections exist and become '
        'detectable at lower score thresholds, but not '
        'at the locked final threshold 0.50.'
    )


elif (
    pr_at_050[
        1
    ]
    ==
    0.0

    and

    diagnostic_geometry50
    >
    0.0
):

    CONFIDENCE_DIAGNOSIS = (
        'SEVERE UNDER-CONFIDENCE: '
        'localization exists, but foreground confidence '
        'is too low even across the diagnostic thresholds.'
    )


else:

    CONFIDENCE_DIAGNOSIS = (
        'P/R@0.50 IS ALIVE: '
        'the previous person-gate failure should be '
        'rechecked for run/checkpoint consistency.'
    )


# ==========================================================
# SAVE DIAGNOSTIC RESULT FOR STEP 34
# ==========================================================

PERSON_CONFIDENCE_DIAGNOSTIC = {

    'checkpoint':
        COCO80_BEST_CHECKPOINT_PATH,

    'episodes':
        int(
            len(
                person_val_loader
            )
        ),

    'total_gt':
        int(
            total_gt
        ),

    'iou_threshold':
        PERSON_DIAGNOSTIC_IOU,

    'threshold_results':
        threshold_results,

    'geometry50':
        float(
            diagnostic_geometry50
        ),

    'geometry_score_coverage':
        geometry_score_coverage,

    'all_score_stats':
        all_score_stats,

    'image_max_score_stats':
        image_max_score_stats,

    'best_iou_stats':
        best_iou_stats,

    'best_iou_query_score_stats':
        best_iou_score_stats,

    'geometry_valid_score_stats':
        geometry_valid_score_stats,

    'learned_class_bias':
        learned_class_bias,

    'learned_logit_scale':
        learned_logit_scale,

    'similarity_required_for_050':
        float(
            similarity_required_for_050
        ),

    'maximum_possible_logit':
        float(
            maximum_possible_logit
        ),

    'maximum_possible_score':
        float(
            maximum_possible_score
        ),

    'score_050_mathematically_reachable':
        bool(
            score_050_mathematically_reachable
        ),

    'diagnosis':
        CONFIDENCE_DIAGNOSIS,
}


STEP33_CONFIDENCE_DIAGNOSTIC_COMPLETE = True


# ==========================================================
# REPORT
# ==========================================================

print()
print('=' * 78)
print('STEP 33 : PERSON CONFIDENCE DIAGNOSTIC')
print('=' * 78)

print(
    'Checkpoint                 :',
    COCO80_BEST_CHECKPOINT_PATH,
)

print(
    'Episodes                   :',
    len(
        person_val_loader
    ),
)

print(
    'Total person GT            :',
    total_gt,
)

print(
    'Diagnostic Geometry50      :',
    diagnostic_geometry50,
)


print()
print('-' * 78)
print('DETECTION HEAD CALIBRATION')
print('-' * 78)

print(
    'Learned class bias         :',
    learned_class_bias,
)

print(
    'Learned logit scale        :',
    learned_logit_scale,
)

print(
    'Similarity needed @0.50    :',
    similarity_required_for_050,
)

print(
    'Cosine similarity maximum :',
    1.0,
)

print(
    'Max theoretical logit      :',
    maximum_possible_logit,
)

print(
    'Max theoretical score      :',
    maximum_possible_score,
)

print(
    'Score 0.50 reachable       :',
    score_050_mathematically_reachable,
)


print()
print('-' * 78)
print('PRECISION / RECALL THRESHOLD SWEEP')
print('-' * 78)

for threshold in (
    PERSON_DIAGNOSTIC_THRESHOLDS
):

    result = (
        threshold_results[
            threshold
        ]
    )

    print(
        f'score >= {threshold:.2f}'
        f' | P={result["precision"]:.6f}'
        f' | R={result["recall"]:.6f}'
        f' | TP={result["tp"]}'
        f' | FP={result["fp"]}'
        f' | FN={result["fn"]}'
    )


print()
print('-' * 78)
print('GEOMETRICALLY VALID DETECTION SCORE COVERAGE')
print('-' * 78)

print(
    'GT with IoU>=0.50          :',
    geometry_gt_hits,
    '/',
    total_gt,
)

for threshold in (
    PERSON_DIAGNOSTIC_THRESHOLDS
):

    print(
        f'valid-box score >= {threshold:.2f}'
        f' : '
        f'{geometry_score_coverage[threshold]:.6f}'
    )


print()
print('-' * 78)
print('SCORE DISTRIBUTIONS')
print('-' * 78)

print(
    'All query scores           :',
    all_score_stats,
)

print(
    'Per-image max scores       :',
    image_max_score_stats,
)

print(
    'Best-IoU query scores      :',
    best_iou_score_stats,
)

print(
    'IoU>=0.50 max scores       :',
    geometry_valid_score_stats,
)

print(
    'Best IoU per GT            :',
    best_iou_stats,
)


print()
print('-' * 78)
print('DIAGNOSIS')
print('-' * 78)

print(
    CONFIDENCE_DIAGNOSIS
)


print('=' * 78)


# ==========================================================
# CLEANUP MODEL ONLY
#
# Keep PERSON_CONFIDENCE_DIAGNOSTIC in memory
# for STEP 34.
# ==========================================================

diagnostic_model.cpu()

del diagnostic_model
del diagnostic_checkpoint


gc.collect()


if torch.cuda.is_available():

    torch.cuda.empty_cache()
