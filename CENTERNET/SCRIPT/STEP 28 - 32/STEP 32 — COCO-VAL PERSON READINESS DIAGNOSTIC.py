# ============================================================
# STEP 32 — COCO-VAL PERSON READINESS DIAGNOSTIC
# Primary score threshold = 0.50
# Primary IoU threshold   = 0.50
# ============================================================

print("=" * 70)
print("STEP 32 — PERSON READINESS DIAGNOSTIC")
print("=" * 70)

PERSON_CLASS = CONFIG[
    "person_class_index"
]

IOU_THRESHOLD = CONFIG[
    "iou_threshold"
]

SCORE_THRESHOLDS = [
    0.05,
    0.10,
    0.20,
    0.30,
    0.40,
    0.50
]


threshold_counts = {

    threshold: {
        "tp": 0,
        "fp": 0,
        "fn": 0
    }

    for threshold
    in SCORE_THRESHOLDS
}


geometry_matches = 0
geometry_gt = 0


best_model.eval()


with torch.no_grad():

    for batch_index, (
        images,
        targets
    ) in enumerate(
        val_loader
    ):


        images = images.to(
            device,
            non_blocking=True
        )


        with torch.amp.autocast(
            device_type="cuda",
            dtype=torch.float16
        ):

            outputs = best_model(
                images
            )


        # Decode once at threshold zero.
        predictions = decode_centernet(
            outputs,
            K=100,
            score_threshold=0.0
        )


        for prediction, target in zip(
            predictions,
            targets
        ):


            # ------------------------------------------------
            # Geometry50:
            # same class + IoU >= 0.50
            # confidence ignored
            # ------------------------------------------------

            geometry = \
                greedy_detection_counts(
                    prediction,
                    target,
                    score_threshold=0.0,
                    iou_threshold=IOU_THRESHOLD,
                    class_filter=PERSON_CLASS
                )

            geometry_matches += \
                geometry[
                    "tp"
                ]

            geometry_gt += \
                geometry[
                    "gt"
                ]


            # ------------------------------------------------
            # Confidence threshold sweep
            # ------------------------------------------------

            for threshold in \
                SCORE_THRESHOLDS:

                counts = \
                    greedy_detection_counts(
                        prediction,
                        target,
                        score_threshold=threshold,
                        iou_threshold=IOU_THRESHOLD,
                        class_filter=PERSON_CLASS
                    )


                threshold_counts[
                    threshold
                ][
                    "tp"
                ] += counts[
                    "tp"
                ]

                threshold_counts[
                    threshold
                ][
                    "fp"
                ] += counts[
                    "fp"
                ]

                threshold_counts[
                    threshold
                ][
                    "fn"
                ] += counts[
                    "fn"
                ]


        if (
            batch_index == 0
            or (
                batch_index + 1
            ) % 250 == 0
        ):

            print(
                "Processed val batches:",
                batch_index + 1
            )


# ============================================================
# RESULTS
# ============================================================

geometry50 = (
    geometry_matches
    / max(
        geometry_gt,
        1
    )
)


print(
    "\n"
    + "=" * 70
)

print(
    "PERSON READINESS — COCO-VAL"
)

print(
    "=" * 70
)

print(
    "Person class index :",
    PERSON_CLASS
)

print(
    "Person GT objects  :",
    geometry_gt
)

print(
    f"Geometry50         : "
    f"{geometry50:.4f}"
)


print(
    "\n"
    + "-" * 70
)

print(
    "CONFIDENCE SWEEP — IoU >= 0.50"
)

print(
    "-" * 70
)

print(
    f"{'Score':<10}"
    f"{'Precision':<15}"
    f"{'Recall':<15}"
    f"{'TP':<10}"
    f"{'FP':<10}"
    f"{'FN':<10}"
)

print(
    "-" * 70
)


person_results = {}


for threshold in \
    SCORE_THRESHOLDS:

    counts = threshold_counts[
        threshold
    ]

    tp = counts[
        "tp"
    ]

    fp = counts[
        "fp"
    ]

    fn = counts[
        "fn"
    ]


    precision = (
        tp
        / max(
            tp + fp,
            1
        )
    )

    recall = (
        tp
        / max(
            tp + fn,
            1
        )
    )


    person_results[
        threshold
    ] = {
        "precision":
            precision,

        "recall":
            recall,

        "tp":
            tp,

        "fp":
            fp,

        "fn":
            fn
    }


    print(
        f"{threshold:<10.2f}"
        f"{precision:<15.4f}"
        f"{recall:<15.4f}"
        f"{tp:<10d}"
        f"{fp:<10d}"
        f"{fn:<10d}"
    )


# ============================================================
# PRIMARY LOCKED RESULT @ SCORE 0.50
# ============================================================

primary = person_results[
    0.50
]


print(
    "\n"
    + "=" * 70
)

print(
    "PRIMARY PERSON DIAGNOSTIC"
)

print(
    "=" * 70
)

print(
    "Score threshold : 0.50"
)

print(
    "IoU threshold   : 0.50"
)

print(
    f"Geometry50      : "
    f"{geometry50:.4f}"
)

print(
    f"Precision       : "
    f"{primary['precision']:.4f}"
)

print(
    f"Recall          : "
    f"{primary['recall']:.4f}"
)

print(
    "TP             :",
    primary[
        "tp"
    ]
)

print(
    "FP             :",
    primary[
        "fp"
    ]
)

print(
    "FN             :",
    primary[
        "fn"
    ]
)


print(
    "\nNO TRAINING OR WEIGHT UPDATE "
    "WAS PERFORMED IN STEP 32."
)

print(
    "\nSTEP 32 COMPLETE"
)

print(
    "\nBASE / SOURCE PHASE COMPLETE"
)
