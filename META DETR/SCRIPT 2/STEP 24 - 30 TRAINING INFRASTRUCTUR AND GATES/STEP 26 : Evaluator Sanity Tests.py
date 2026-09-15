# ==========================================================
# STEP 26 : Evaluator Sanity Tests
# ==========================================================

perfect_records = [

    {
        "semantic_label": 0,

        "scores":
            np.array(
                [
                    0.95,
                    0.05,
                ]
            ),

        "pred_boxes":
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

        "gt_boxes":
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
        "semantic_label": 1,

        "scores":
            np.array(
                [
                    0.90,
                    0.10,
                ]
            ),

        "pred_boxes":
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

        "gt_boxes":
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
    )
)


assert np.isclose(
    perfect_metrics[
        "mAP50"
    ],
    1.0,
)

assert np.isclose(
    perfect_metrics[
        "mAP75"
    ],
    1.0,
)

assert np.isclose(
    perfect_metrics[
        "mAP95"
    ],
    1.0,
)

assert np.isclose(
    perfect_metrics[
        "precision50"
    ],
    1.0,
)

assert np.isclose(
    perfect_metrics[
        "recall50"
    ],
    1.0,
)

assert np.isclose(
    perfect_metrics[
        "geometry_recall50"
    ],
    1.0,
)


bad_records = [

    {
        "semantic_label": 0,

        "scores":
            np.array(
                [
                    0.99
                ]
            ),

        "pred_boxes":
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

        "gt_boxes":
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
    )
)


assert np.isclose(
    bad_metrics[
        "mAP50"
    ],
    0.0,
)

assert np.isclose(
    bad_metrics[
        "recall50"
    ],
    0.0,
)


print("=" * 70)
print("STEP 26 PASS : EVALUATOR SANITY VALID")
print("=" * 70)

print(
    "Perfect mAP50:",
    perfect_metrics[
        "mAP50"
    ]
)

print(
    "Bad mAP50    :",
    bad_metrics[
        "mAP50"
    ]
)

print("=" * 70)


del perfect_records
del bad_records
del perfect_metrics
del bad_metrics
