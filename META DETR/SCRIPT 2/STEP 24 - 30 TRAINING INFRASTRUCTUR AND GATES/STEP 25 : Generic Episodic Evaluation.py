# ==========================================================
# STEP 25 — FULL REPLACEMENT
# Generic episodic evaluator
# ==========================================================

def as_numpy(
    value
):

    if torch.is_tensor(
        value
    ):

        value = (
            value
            .detach()
            .cpu()
            .numpy()
        )

    return np.asarray(
        value,
        dtype=np.float64,
    )


def numpy_box_iou(
    boxes_a,
    boxes_b,
):

    a = (
        as_numpy(
            boxes_a
        )
        .reshape(
            -1,
            4,
        )
    )

    b = (
        as_numpy(
            boxes_b
        )
        .reshape(
            -1,
            4,
        )
    )

    if (
        len(
            a
        ) == 0
        or
        len(
            b
        ) == 0
    ):

        return np.zeros(
            (
                len(
                    a
                ),
                len(
                    b
                ),
            ),
            dtype=np.float64,
        )

    # cxcywh -> xyxy
    a = np.concatenate(
        (
            a[
                :,
                :2
            ]
            -
            a[
                :,
                2:
            ]
            /
            2,

            a[
                :,
                :2
            ]
            +
            a[
                :,
                2:
            ]
            /
            2,
        ),
        axis=1,
    )

    b = np.concatenate(
        (
            b[
                :,
                :2
            ]
            -
            b[
                :,
                2:
            ]
            /
            2,

            b[
                :,
                :2
            ]
            +
            b[
                :,
                2:
            ]
            /
            2,
        ),
        axis=1,
    )

    top_left = np.maximum(
        a[
            :,
            None,
            :2
        ],
        b[
            None,
            :,
            :2
        ],
    )

    bottom_right = np.minimum(
        a[
            :,
            None,
            2:
        ],
        b[
            None,
            :,
            2:
        ],
    )

    wh = np.maximum(
        0.0,
        bottom_right
        -
        top_left,
    )

    intersection = (
        wh[
            ...,
            0
        ]
        *
        wh[
            ...,
            1
        ]
    )

    area_a = (
        np.maximum(
            0.0,
            a[
                :,
                2:
            ]
            -
            a[
                :,
                :2
            ],
        )
        .prod(
            axis=1
        )
    )

    area_b = (
        np.maximum(
            0.0,
            b[
                :,
                2:
            ]
            -
            b[
                :,
                :2
            ],
        )
        .prod(
            axis=1
        )
    )

    union = (
        area_a[
            :,
            None
        ]
        +
        area_b[
            None,
            :
        ]
        -
        intersection
    )

    return (
        intersection
        /
        np.maximum(
            union,
            1e-12,
        )
    )


def greedy_detection_flags(
    scores,
    pred_boxes,
    gt_boxes,
    iou_threshold,
):

    scores = (
        as_numpy(
            scores
        )
        .reshape(
            -1
        )
    )

    pred_boxes = (
        as_numpy(
            pred_boxes
        )
        .reshape(
            -1,
            4,
        )
    )

    gt_boxes = (
        as_numpy(
            gt_boxes
        )
        .reshape(
            -1,
            4,
        )
    )

    flags = np.zeros(
        len(
            scores
        ),
        dtype=np.float64,
    )

    if (
        len(
            scores
        ) == 0
        or
        len(
            gt_boxes
        ) == 0
    ):
        return flags

    ious = numpy_box_iou(
        pred_boxes,
        gt_boxes,
    )

    used_gt = set()

    prediction_order = np.argsort(
        -scores,
        kind='stable',
    )

    for pred_index in (
        prediction_order
    ):

        gt_order = np.argsort(
            -ious[
                pred_index
            ],
            kind='stable',
        )

        for gt_index in (
            gt_order
        ):

            if (
                int(
                    gt_index
                )
                in
                used_gt
            ):
                continue

            if (
                ious[
                    pred_index,
                    gt_index
                ]
                <
                iou_threshold
            ):
                break

            used_gt.add(
                int(
                    gt_index
                )
            )

            flags[
                pred_index
            ] = 1.0

            break

    return flags


def average_precision_from_ranked(
    scores,
    flags,
    total_gt,
):

    scores = (
        as_numpy(
            scores
        )
        .reshape(
            -1
        )
    )

    flags = (
        as_numpy(
            flags
        )
        .reshape(
            -1
        )
    )

    total_gt = int(
        total_gt
    )

    if total_gt <= 0:
        return float(
            'nan'
        )

    if len(
        scores
    ) == 0:
        return 0.0

    order = np.argsort(
        -scores,
        kind='stable',
    )

    tp = np.cumsum(
        flags[
            order
        ]
    )

    fp = np.cumsum(
        1.0
        -
        flags[
            order
        ]
    )

    recall = (
        tp
        /
        float(
            total_gt
        )
    )

    precision = (
        tp
        /
        np.maximum(
            tp
            +
            fp,
            1e-12,
        )
    )

    mrec = np.concatenate(
        (
            [
                0.0
            ],
            recall,
            [
                1.0
            ],
        )
    )

    mpre = np.concatenate(
        (
            [
                0.0
            ],
            precision,
            [
                0.0
            ],
        )
    )

    for index in range(
        len(
            mpre
        ) - 2,
        -1,
        -1,
    ):

        mpre[
            index
        ] = max(
            mpre[
                index
            ],
            mpre[
                index + 1
            ],
        )

    changes = np.where(
        mrec[
            1:
        ]
        !=
        mrec[
            :-1
        ]
    )[0]

    ap = np.sum(
        (
            mrec[
                changes + 1
            ]
            -
            mrec[
                changes
            ]
        )
        *
        mpre[
            changes + 1
        ]
    )

    return float(
        ap
    )


def compute_episodic_metrics(
    records,
    score_threshold=0.50,
    primary_iou_threshold=0.50,
):

    if not records:
        raise ValueError(
            'No evaluation records.'
        )

    iou_thresholds = tuple(
        round(
            float(
                value
            ),
            2,
        )
        for value
        in np.arange(
            0.50,
            0.951,
            0.05,
        )
    )

    labels_present = sorted({
        int(
            record[
                'semantic_label'
            ]
        )
        for record
        in records
    })

    # ======================================================
    # AP per IoU / semantic episode class
    # ======================================================

    ap_per_iou = {}
    per_class_ap = {}

    for threshold in (
        iou_thresholds
    ):

        class_aps = {}

        for semantic_label in (
            labels_present
        ):

            class_records = [

                record

                for record
                in records

                if int(
                    record[
                        'semantic_label'
                    ]
                )
                ==
                semantic_label
            ]

            total_gt = sum(
                len(
                    record[
                        'gt_boxes'
                    ]
                )
                for record
                in class_records
            )

            all_scores = []
            all_flags = []

            for record in (
                class_records
            ):

                scores = (
                    as_numpy(
                        record[
                            'scores'
                        ]
                    )
                    .reshape(
                        -1
                    )
                )

                flags = (
                    greedy_detection_flags(
                        scores,
                        record[
                            'pred_boxes'
                        ],
                        record[
                            'gt_boxes'
                        ],
                        threshold,
                    )
                )

                all_scores.extend(
                    scores.tolist()
                )

                all_flags.extend(
                    flags.tolist()
                )

            class_ap = (
                average_precision_from_ranked(
                    all_scores,
                    all_flags,
                    total_gt,
                )
            )

            class_aps[
                semantic_label
            ] = class_ap

        ap_per_iou[
            threshold
        ] = float(
            np.mean(
                list(
                    class_aps.values()
                )
            )
        )

        per_class_ap[
            threshold
        ] = class_aps

    # ======================================================
    # Fixed-threshold P/R
    # ======================================================

    total_gt = 0
    tp = 0
    fp = 0
    fn = 0

    geometry_hits50 = 0

    best_ious_all_gt = []
    maximum_scores = []

    for record in records:

        scores = (
            as_numpy(
                record[
                    'scores'
                ]
            )
            .reshape(
                -1
            )
        )

        pred_boxes = (
            as_numpy(
                record[
                    'pred_boxes'
                ]
            )
            .reshape(
                -1,
                4,
            )
        )

        gt_boxes = (
            as_numpy(
                record[
                    'gt_boxes'
                ]
            )
            .reshape(
                -1,
                4,
            )
        )

        total_gt += len(
            gt_boxes
        )

        if len(
            scores
        ):

            maximum_scores.append(
                float(
                    scores.max()
                )
            )

        keep = (
            scores
            >=
            score_threshold
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

        threshold_flags = (
            greedy_detection_flags(
                kept_scores,
                kept_boxes,
                gt_boxes,
                primary_iou_threshold,
            )
        )

        episode_tp = int(
            threshold_flags.sum()
        )

        tp += episode_tp

        fp += (
            len(
                kept_scores
            )
            -
            episode_tp
        )

        fn += (
            len(
                gt_boxes
            )
            -
            episode_tp
        )

        # Geometry independent of confidence.
        if (
            len(
                pred_boxes
            ) > 0
            and
            len(
                gt_boxes
            ) > 0
        ):

            ious = numpy_box_iou(
                pred_boxes,
                gt_boxes,
            )

            best_per_gt = (
                ious.max(
                    axis=0
                )
            )

            best_ious_all_gt.extend(
                best_per_gt.tolist()
            )

            rows, cols = (
                linear_sum_assignment(
                    -ious
                )
            )

            geometry_hits50 += int(
                (
                    ious[
                        rows,
                        cols
                    ]
                    >=
                    primary_iou_threshold
                ).sum()
            )

        elif len(
            gt_boxes
        ) > 0:

            best_ious_all_gt.extend(
                [
                    0.0
                ]
                *
                len(
                    gt_boxes
                )
            )

    precision50 = (
        tp
        /
        max(
            tp + fp,
            1,
        )
    )

    recall50 = (
        tp
        /
        max(
            total_gt,
            1,
        )
    )

    geometry_recall50 = (
        geometry_hits50
        /
        max(
            total_gt,
            1,
        )
    )

    mAP50 = (
        ap_per_iou[
            0.50
        ]
    )

    mAP75 = (
        ap_per_iou[
            0.75
        ]
    )

    mAP95 = (
        ap_per_iou[
            0.95
        ]
    )

    mAP50_95 = float(
        np.mean(
            list(
                ap_per_iou.values()
            )
        )
    )

    per_class_ap50 = {

        CATEGORY_NAMES.get(
            label,
            str(
                label
            ),
        ):
            float(
                value
            )

        for label, value
        in per_class_ap[
            0.50
        ].items()
    }

    return {

        'mAP50':
            float(
                mAP50
            ),

        'mAP75':
            float(
                mAP75
            ),

        'mAP95':
            float(
                mAP95
            ),

        'mAP50_95':
            float(
                mAP50_95
            ),

        'precision50':
            float(
                precision50
            ),

        'recall50':
            float(
                recall50
            ),

        'geometry_recall50':
            float(
                geometry_recall50
            ),

        'mean_best_iou':
            float(
                np.mean(
                    best_ious_all_gt
                )
                if
                best_ious_all_gt
                else
                0.0
            ),

        'mean_max_score':
            float(
                np.mean(
                    maximum_scores
                )
                if
                maximum_scores
                else
                0.0
            ),

        'tp':
            int(
                tp
            ),

        'fp':
            int(
                fp
            ),

        'fn':
            int(
                fn
            ),

        'total_gt':
            int(
                total_gt
            ),

        'classes_evaluated':
            len(
                labels_present
            ),

        'mAP_by_iou': {

            f'{threshold:.2f}':
                float(
                    value
                )

            for threshold, value
            in ap_per_iou.items()
        },

        'AP50_per_class':
            per_class_ap50,
    }


# ==========================================================
# Evaluator
# ==========================================================

@torch.no_grad()
def evaluate_episodic_model(
    target_model,
    data_loader,
    max_batches=None,
    show_progress=False,
):

    target_model.eval()

    records = []

    loss_totals = {
        'loss_cls':
            0.0,

        'loss_bbox':
            0.0,

        'loss_giou':
            0.0,

        'loss_total':
            0.0,
    }

    batch_count = 0

    iterator = data_loader

    if max_batches is not None:

        iterator = islice(
            iterator,
            int(
                max_batches
            ),
        )

    if show_progress:

        total = (
            len(
                data_loader
            )
            if
            max_batches is None
            else
            min(
                len(
                    data_loader
                ),
                int(
                    max_batches
                ),
            )
        )

        iterator = tqdm(
            iterator,
            total=total,
            desc='Evaluation',
        )

    for batch in iterator:

        validate_episodic_batch(
            batch
        )

        support_images = (
            batch[
                'support_images'
            ]
            .to(
                CONFIG[
                    'device'
                ],
                non_blocking=True,
            )
        )

        support_padding_masks = (
            batch[
                'support_padding_masks'
            ]
            .to(
                CONFIG[
                    'device'
                ],
                non_blocking=True,
            )
        )

        query_images = (
            batch[
                'query_images'
            ]
            .to(
                CONFIG[
                    'device'
                ],
                non_blocking=True,
            )
        )

        query_padding_masks = (
            batch[
                'query_padding_masks'
            ]
            .to(
                CONFIG[
                    'device'
                ],
                non_blocking=True,
            )
        )

        targets = (
            move_targets_to_device(
                batch[
                    'query_targets'
                ],
                CONFIG[
                    'device'
                ],
            )
        )

        outputs = (
            target_model(
                support_images,
                query_images,

                support_padding_mask=
                    support_padding_masks,

                query_padding_mask=
                    query_padding_masks,
            )
        )

        losses = criterion(
            outputs,
            targets,
        )

        for key in (
            loss_totals
        ):

            loss_totals[
                key
            ] += (
                losses[
                    key
                ]
                .item()
            )

        scores = (
            outputs[
                'pred_logits'
            ]
            .sigmoid()
            .squeeze(
                -1
            )
        )

        pred_boxes = (
            outputs[
                'pred_boxes'
            ]
        )

        for batch_index in range(
            scores.shape[0]
        ):

            semantic_label = int(
                batch[
                    'episode_classes'
                ][
                    batch_index
                ].item()
            )

            records.append(
                {
                    'semantic_label':
                        semantic_label,

                    'scores':
                        scores[
                            batch_index
                        ]
                        .detach()
                        .cpu(),

                    'pred_boxes':
                        pred_boxes[
                            batch_index
                        ]
                        .detach()
                        .cpu(),

                    'gt_boxes':
                        targets[
                            batch_index
                        ][
                            'boxes'
                        ]
                        .detach()
                        .cpu(),
                }
            )

        batch_count += 1

    if batch_count == 0:
        raise RuntimeError(
            'Evaluation loader produced no batches.'
        )

    metrics = (
        compute_episodic_metrics(

            records,

            score_threshold=
                TRAIN_CONFIG[
                    'score_threshold'
                ],

            primary_iou_threshold=
                TRAIN_CONFIG[
                    'primary_iou_threshold'
                ],
        )
    )

    return {

        'mean_loss': {

            key:
                value
                /
                batch_count

            for key, value
            in loss_totals.items()
        },

        'metrics':
            metrics,

        'episodes':
            len(
                records
            ),
    }


print('=' * 70)
print('STEP 25 : GENERIC EPISODIC EVALUATOR READY')
print('=' * 70)

print(
    'Primary: AP50 + '
    'Precision@0.50 + Recall@0.50'
)

print(
    'Diagnostics: mAP75 / mAP95 / '
    'mAP50:95 / Geometry50'
)

print('=' * 70)
