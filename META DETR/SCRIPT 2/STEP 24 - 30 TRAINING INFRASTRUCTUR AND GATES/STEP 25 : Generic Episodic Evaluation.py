# ==========================================================
# STEP 25 — FULL REPLACEMENT
#
# FINAL:
#   Precision@0.50
#   Recall@0.50
#
# INTERNAL:
#   Geometry50
#   validation loss
#
# NO AP / mAP.
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


# ==========================================================
# IoU for normalized cxcywh boxes
# ==========================================================

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
        len(a) == 0
        or
        len(b) == 0
    ):

        return np.zeros(
            (
                len(a),
                len(b),
            ),
            dtype=np.float64,
        )


    # cxcywh -> xyxy
    a_xyxy = np.concatenate(
        (
            a[:, :2]
            -
            a[:, 2:] / 2.0,

            a[:, :2]
            +
            a[:, 2:] / 2.0,
        ),
        axis=1,
    )


    b_xyxy = np.concatenate(
        (
            b[:, :2]
            -
            b[:, 2:] / 2.0,

            b[:, :2]
            +
            b[:, 2:] / 2.0,
        ),
        axis=1,
    )


    top_left = np.maximum(
        a_xyxy[:, None, :2],
        b_xyxy[None, :, :2],
    )


    bottom_right = np.minimum(
        a_xyxy[:, None, 2:],
        b_xyxy[None, :, 2:],
    )


    wh = np.maximum(
        0.0,
        bottom_right
        -
        top_left,
    )


    intersection = (
        wh[..., 0]
        *
        wh[..., 1]
    )


    area_a = (
        np.maximum(
            0.0,
            a_xyxy[:, 2:]
            -
            a_xyxy[:, :2],
        )
        .prod(
            axis=1
        )
    )


    area_b = (
        np.maximum(
            0.0,
            b_xyxy[:, 2:]
            -
            b_xyxy[:, :2],
        )
        .prod(
            axis=1
        )
    )


    union = (
        area_a[:, None]
        +
        area_b[None, :]
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


# ==========================================================
# Greedy 1-to-1 matching for final Precision / Recall
#
# Predictions are processed highest confidence first.
# ==========================================================

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
        len(scores),
        dtype=np.float64,
    )


    if (
        len(scores) == 0
        or
        len(gt_boxes) == 0
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

            gt_index = int(
                gt_index
            )


            if gt_index in used_gt:
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
                gt_index
            )


            flags[
                pred_index
            ] = 1.0

            break


    return flags


# ==========================================================
# Final P/R + internal Geometry
# ==========================================================

def compute_episodic_metrics(
    records,
    score_threshold=0.50,
    primary_iou_threshold=0.50,
):

    if not records:

        raise ValueError(
            'No evaluation records.'
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


    total_gt = 0

    tp = 0
    fp = 0
    fn = 0

    # INTERNAL diagnostic.
    geometry_hits50 = 0


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


        # ==================================================
        # FINAL Precision / Recall
        #
        # score >= 0.50
        # IoU   >= 0.50
        # ==================================================

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


        episode_fp = (
            len(
                kept_scores
            )
            -
            episode_tp
        )


        episode_fn = (
            len(
                gt_boxes
            )
            -
            episode_tp
        )


        tp += episode_tp
        fp += episode_fp
        fn += episode_fn


        # ==================================================
        # INTERNAL Geometry50
        #
        # Confidence is intentionally ignored.
        #
        # For every GT:
        # does AT LEAST ONE predicted box reach IoU >= .50?
        # ==================================================

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


            geometry_hits50 += int(
                (
                    best_per_gt
                    >=
                    primary_iou_threshold
                ).sum()
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


    return {

        # FINAL thesis accuracy metrics
        'precision50':
            float(
                precision50
            ),

        'recall50':
            float(
                recall50
            ),

        # INTERNAL ONLY
        'geometry_recall50':
            float(
                geometry_recall50
            ),

        # Bookkeeping
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
    }


# ==========================================================
# Generic episodic evaluator
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


    iterator = (
        data_loader
    )


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

            if max_batches is None

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


        outputs = target_model(

            support_images,
            query_images,

            support_padding_mask=
                support_padding_masks,

            query_padding_mask=
                query_padding_masks,
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
            ] += float(

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
    'FINAL     : Precision@0.50 + Recall@0.50'
)

print(
    'INTERNAL  : Geometry50 + validation loss'
)

print(
    'AP / mAP  : REMOVED'
)

print('=' * 70)
