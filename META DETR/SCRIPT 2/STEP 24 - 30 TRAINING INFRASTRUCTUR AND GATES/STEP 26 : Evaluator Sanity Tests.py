# ==========================================================
# STEP 26 — FULL REPLACEMENT
# Evaluator sanity tests
# ==========================================================


# ==========================================================
# PERFECT CASE
# ==========================================================

perfect_records = [

    {
        'semantic_label':
            0,

        'scores':
            np.array(
                [
                    0.95,
                    0.05,
                ]
            ),

        'pred_boxes':
            np.array(
                [
                    [
                        0.50,
                        0.50,
                        0.20,
                        0.20,
                    ],
                    [
                        0.10,
                        0.10,
                        0.10,
                        0.10,
                    ],
                ]
            ),

        'gt_boxes':
            np.array(
                [
                    [
                        0.50,
                        0.50,
                        0.20,
                        0.20,
                    ]
                ]
            ),
    },

    {
        'semantic_label':
            1,

        'scores':
            np.array(
                [
                    0.90,
                    0.10,
                ]
            ),

        'pred_boxes':
            np.array(
                [
                    [
                        0.70,
                        0.70,
                        0.15,
                        0.15,
                    ],
                    [
                        0.20,
                        0.20,
                        0.10,
                        0.10,
                    ],
                ]
            ),

        'gt_boxes':
            np.array(
                [
                    [
                        0.70,
                        0.70,
                        0.15,
                        0.15,
                    ]
                ]
            ),
    },
]


perfect_metrics = (
    compute_episodic_metrics(

        perfect_records,

        score_threshold=0.50,
        primary_iou_threshold=0.50,
    )
)


assert np.isclose(
    perfect_metrics[
        'precision50'
    ],
    1.0,
)


assert np.isclose(
    perfect_metrics[
        'recall50'
    ],
    1.0,
)


assert np.isclose(
    perfect_metrics[
        'geometry_recall50'
    ],
    1.0,
)


# ==========================================================
# BAD LOCALIZATION
# ==========================================================

bad_records = [

    {
        'semantic_label':
            0,

        'scores':
            np.array(
                [
                    0.99
                ]
            ),

        'pred_boxes':
            np.array(
                [
                    [
                        0.10,
                        0.10,
                        0.10,
                        0.10,
                    ]
                ]
            ),

        'gt_boxes':
            np.array(
                [
                    [
                        0.90,
                        0.90,
                        0.10,
                        0.10,
                    ]
                ]
            ),
    },
]


bad_metrics = (
    compute_episodic_metrics(

        bad_records,

        score_threshold=0.50,
        primary_iou_threshold=0.50,
    )
)


assert np.isclose(
    bad_metrics[
        'precision50'
    ],
    0.0,
)


assert np.isclose(
    bad_metrics[
        'recall50'
    ],
    0.0,
)


assert np.isclose(
    bad_metrics[
        'geometry_recall50'
    ],
    0.0,
)


# ==========================================================
# GOOD GEOMETRY, LOW CONFIDENCE
#
# Proves Geometry is independent from score threshold.
# ==========================================================

low_confidence_records = [

    {
        'semantic_label':
            0,

        'scores':
            np.array(
                [
                    0.10
                ]
            ),

        'pred_boxes':
            np.array(
                [
                    [
                        0.50,
                        0.50,
                        0.20,
                        0.20,
                    ]
                ]
            ),

        'gt_boxes':
            np.array(
                [
                    [
                        0.50,
                        0.50,
                        0.20,
                        0.20,
                    ]
                ]
            ),
    },
]


low_confidence_metrics = (
    compute_episodic_metrics(

        low_confidence_records,

        score_threshold=0.50,
        primary_iou_threshold=0.50,
    )
)


assert np.isclose(
    low_confidence_metrics[
        'precision50'
    ],
    0.0,
)


assert np.isclose(
    low_confidence_metrics[
        'recall50'
    ],
    0.0,
)


assert np.isclose(
    low_confidence_metrics[
        'geometry_recall50'
    ],
    1.0,
)


print('=' * 70)
print('STEP 26 PASS : EVALUATOR SANITY VALID')
print('=' * 70)

print(
    'Perfect P/R/Geo :',
    perfect_metrics[
        'precision50'
    ],
    perfect_metrics[
        'recall50'
    ],
    perfect_metrics[
        'geometry_recall50'
    ],
)

print(
    'Bad P/R/Geo     :',
    bad_metrics[
        'precision50'
    ],
    bad_metrics[
        'recall50'
    ],
    bad_metrics[
        'geometry_recall50'
    ],
)

print(
    'LowConf P/R/Geo :',
    low_confidence_metrics[
        'precision50'
    ],
    low_confidence_metrics[
        'recall50'
    ],
    low_confidence_metrics[
        'geometry_recall50'
    ],
)

print('=' * 70)


del perfect_records
del perfect_metrics
del bad_records
del bad_metrics
del low_confidence_records
del low_confidence_metrics
