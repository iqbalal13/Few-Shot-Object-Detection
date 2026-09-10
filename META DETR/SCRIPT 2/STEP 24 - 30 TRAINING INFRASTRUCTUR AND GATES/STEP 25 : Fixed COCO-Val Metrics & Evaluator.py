# ==========================================================
# STEP 25 : Fixed COCO-Val Metrics & Evaluator
# CLASS-AWARE EPISODIC EVALUATION
#
# IMPORTANT FIX:
#
# OLD:
#   all episodic predictions from all 80 COCO classes
#   -> one global binary AP
#
# NEW:
#   class 0 episodes -> AP50_0
#   class 1 episodes -> AP50_1
#   ...
#   class 79 episodes -> AP50_79
#
#   mAP50 = mean(AP50 over classes PRESENT in evaluation)
#
# For full fixed COCO-Val:
#   800 episodes = 10 episodes × 80 classes
#   => mean over all 80 classes
#
# For tiny subsets:
#   mean only over classes actually present
#
# P/R remain MICRO across all evaluated episodes.
# ==========================================================

import numpy as np
import torch

from collections import defaultdict
from tqdm.auto import tqdm


assert "val_loader" in globals(), (
    "Run STEP 18 first."
)

assert "box_iou" in globals(), (
    "Run STEP 19 first."
)

assert "box_cxcywh_to_xyxy" in globals(), (
    "Run STEP 19 first."
)


# ==========================================================
# SINGLE SEMANTIC CLASS AP50
#
# prediction_records:
# [
#     {
#         "episode_id": int,
#         "score": float,
#         "box": Tensor[4]
#     },
#     ...
# ]
#
# episode_gt_boxes:
# {
#     episode_id: Tensor[N,4]
# }
# ==========================================================

def compute_single_class_ap50(
    prediction_records,
    episode_gt_boxes,
    total_gt,
    iou_threshold=0.50
):

    if total_gt <= 0:

        return 0.0


    if len(prediction_records) == 0:

        return 0.0


    # ------------------------------------------------------
    # Rank predictions ONLY WITHIN THIS semantic class.
    # ------------------------------------------------------

    prediction_records = sorted(

        prediction_records,

        key=lambda record:
            record["score"],

        reverse=True
    )


    matched_gt = {

        episode_id:
            set()

        for episode_id
        in episode_gt_boxes.keys()
    }


    tp = []
    fp = []


    for record in prediction_records:

        episode_id = int(
            record["episode_id"]
        )


        pred_box = (

            record["box"]

            .unsqueeze(0)
        )


        gt_boxes = (
            episode_gt_boxes[
                episode_id
            ]
        )


        # --------------------------------------------------
        # No GT for this episode/class.
        # Normally impossible in our episodic dataset,
        # but remain defensive.
        # --------------------------------------------------

        if len(gt_boxes) == 0:

            tp.append(0.0)
            fp.append(1.0)

            continue


        pred_xyxy = (
            box_cxcywh_to_xyxy(
                pred_box
            )
        )


        gt_xyxy = (
            box_cxcywh_to_xyxy(
                gt_boxes
            )
        )


        ious, _ = box_iou(

            pred_xyxy,

            gt_xyxy
        )


        # Highest-IoU GT first.

        candidate_order = torch.argsort(

            ious[0],

            descending=True
        )


        matched = False


        for gt_idx in candidate_order:

            gt_idx_int = int(
                gt_idx.item()
            )


            if (
                gt_idx_int
                in matched_gt[
                    episode_id
                ]
            ):

                continue


            if (
                ious[
                    0,
                    gt_idx_int
                ].item()
                >=
                iou_threshold
            ):

                matched_gt[
                    episode_id
                ].add(
                    gt_idx_int
                )

                matched = True

                break


        if matched:

            tp.append(1.0)
            fp.append(0.0)

        else:

            tp.append(0.0)
            fp.append(1.0)


    tp = np.asarray(
        tp,
        dtype=np.float64
    )


    fp = np.asarray(
        fp,
        dtype=np.float64
    )


    cumulative_tp = np.cumsum(
        tp
    )


    cumulative_fp = np.cumsum(
        fp
    )


    recall = (

        cumulative_tp

        /

        float(
            max(
                total_gt,
                1
            )
        )
    )


    precision = (

        cumulative_tp

        /

        np.maximum(

            cumulative_tp
            +
            cumulative_fp,

            1e-12
        )
    )


    # ------------------------------------------------------
    # Precision envelope / all-point AP.
    #
    # Same integration rule as previous notebook;
    # only semantic-class aggregation is corrected.
    # ------------------------------------------------------

    mrec = np.concatenate(

        (
            [0.0],
            recall,
            [1.0]
        )
    )


    mpre = np.concatenate(

        (
            [0.0],
            precision,
            [0.0]
        )
    )


    for i in range(

        len(mpre) - 2,
        -1,
        -1
    ):

        mpre[i] = max(

            mpre[i],

            mpre[i + 1]
        )


    change_points = np.where(

        mrec[1:]
        !=
        mrec[:-1]

    )[0]


    ap = np.sum(

        (
            mrec[
                change_points + 1
            ]

            -

            mrec[
                change_points
            ]
        )

        *

        mpre[
            change_points + 1
        ]
    )


    return float(ap)


# ==========================================================
# CLASS-AWARE mAP50
# ==========================================================

def compute_classwise_map50(
    prediction_records_by_class,
    gt_boxes_by_class,
    total_gt_by_class,
    iou_threshold=0.50
):

    class_ap50 = {}


    # ------------------------------------------------------
    # Evaluate ONLY classes present in current loader.
    #
    # Full COCO-Val -> all 80
    # Tiny set     -> only classes represented by tiny set
    # ------------------------------------------------------

    evaluated_classes = sorted(

        class_label

        for class_label
        in total_gt_by_class.keys()

        if (
            total_gt_by_class[
                class_label
            ]
            >
            0
        )
    )


    if len(evaluated_classes) == 0:

        return (
            0.0,
            {}
        )


    for class_label in (
        evaluated_classes
    ):

        class_ap50[
            int(class_label)
        ] = compute_single_class_ap50(

            prediction_records=
                prediction_records_by_class[
                    class_label
                ],

            episode_gt_boxes=
                gt_boxes_by_class[
                    class_label
                ],

            total_gt=
                total_gt_by_class[
                    class_label
                ],

            iou_threshold=
                iou_threshold
        )


    map50 = float(

        np.mean(

            list(
                class_ap50.values()
            )
        )
    )


    return (
        map50,
        class_ap50
    )


# ==========================================================
# THRESHOLD P/R FOR ONE EPISODE
# ==========================================================

def match_episode_predictions(
    scores,
    pred_boxes,
    gt_boxes,
    score_threshold=0.50,
    iou_threshold=0.50
):

    keep = torch.where(

        scores
        >=
        score_threshold

    )[0]


    if len(keep) == 0:

        return (
            0,
            0,
            len(gt_boxes)
        )


    # Highest-confidence predictions first.

    kept_scores = (
        scores[
            keep
        ]
    )


    order = torch.argsort(

        kept_scores,

        descending=True
    )


    keep = (
        keep[
            order
        ]
    )


    matched_gt = set()

    tp = 0
    fp = 0


    if len(gt_boxes) > 0:

        gt_xyxy = (
            box_cxcywh_to_xyxy(
                gt_boxes
            )
        )

    else:

        gt_xyxy = None


    for pred_idx in keep:

        if len(gt_boxes) == 0:

            fp += 1

            continue


        pred_xyxy = (
            box_cxcywh_to_xyxy(

                pred_boxes[
                    pred_idx
                ]

                .unsqueeze(0)
            )
        )


        ious, _ = box_iou(

            pred_xyxy,

            gt_xyxy
        )


        candidate_order = torch.argsort(

            ious[0],

            descending=True
        )


        matched = False


        for gt_idx in (
            candidate_order
        ):

            gt_idx_int = int(
                gt_idx.item()
            )


            if (
                gt_idx_int
                in matched_gt
            ):

                continue


            if (
                ious[
                    0,
                    gt_idx_int
                ].item()
                >=
                iou_threshold
            ):

                matched_gt.add(
                    gt_idx_int
                )

                tp += 1

                matched = True

                break


        if not matched:

            fp += 1


    fn = (

        len(gt_boxes)

        -
        len(matched_gt)
    )


    return (
        tp,
        fp,
        fn
    )


# ==========================================================
# FULL CLASS-AWARE EPISODIC EVALUATOR
# ==========================================================

def evaluate_episodic_model(
    model,
    data_loader,
    criterion,
    device,
    score_threshold=0.50,
    iou_threshold=0.50,
    show_progress=True
):

    model.eval()


    # ------------------------------------------------------
    # LOSS
    # ------------------------------------------------------

    total_loss = 0.0
    num_batches = 0


    # ------------------------------------------------------
    # CLASS-AWARE AP STORAGE
    #
    # prediction_records_by_class[class] -> list
    #
    # gt_boxes_by_class[class][episode_id] -> GT
    #
    # total_gt_by_class[class] -> count
    # ------------------------------------------------------

    prediction_records_by_class = (
        defaultdict(list)
    )


    gt_boxes_by_class = (
        defaultdict(dict)
    )


    total_gt_by_class = (
        defaultdict(int)
    )


    # ------------------------------------------------------
    # MICRO P/R
    # ------------------------------------------------------

    total_tp = 0
    total_fp = 0
    total_fn = 0


    # ------------------------------------------------------
    # OPTIONAL CLASS-SPECIFIC P/R
    # ------------------------------------------------------

    tp_by_class = (
        defaultdict(int)
    )

    fp_by_class = (
        defaultdict(int)
    )

    fn_by_class = (
        defaultdict(int)
    )


    # ------------------------------------------------------
    # LOCALIZATION DIAGNOSTIC
    # score-independent
    # ------------------------------------------------------

    best_ious_all_gt = []


    episode_counter = 0


    iterator = (
        data_loader
    )


    if show_progress:

        iterator = tqdm(

            data_loader,

            desc="COCO Validation"
        )


    with torch.inference_mode():

        for batch in iterator:

            support_images = (

                batch[
                    "support_images"
                ]

                .to(
                    device,
                    non_blocking=True
                )
            )


            query_images = (

                batch[
                    "query_images"
                ]

                .to(
                    device,
                    non_blocking=True
                )
            )


            targets = move_targets_to_device(

                batch[
                    "query_targets"
                ],

                device
            )


            # ------------------------------------------------
            # Semantic episode classes.
            # THIS WAS MISSING FROM OLD EVALUATOR.
            # ------------------------------------------------

            episode_classes = (

                batch[
                    "episode_classes"
                ]

                .detach()
                .cpu()
                .long()
            )


            outputs = model(

                support_images,

                query_images
            )


            loss_dict = criterion(

                outputs,

                targets
            )


            total_loss += float(

                loss_dict[
                    "loss_total"
                ].item()
            )


            num_batches += 1


            batch_size = (

                outputs[
                    "pred_logits"
                ].shape[0]
            )


            assert (
                len(
                    episode_classes
                )
                ==
                batch_size
            )


            for b in range(
                batch_size
            ):

                class_label = int(

                    episode_classes[
                        b
                    ].item()
                )


                scores = (

                    outputs[
                        "pred_logits"
                    ][
                        b,
                        :,
                        0
                    ]

                    .sigmoid()

                    .detach()
                    .cpu()
                )


                pred_boxes = (

                    outputs[
                        "pred_boxes"
                    ][b]

                    .detach()
                    .cpu()
                )


                gt_boxes = (

                    targets[b][
                        "boxes"
                    ]

                    .detach()
                    .cpu()
                )


                episode_id = (
                    episode_counter
                )


                episode_counter += 1


                # =================================================
                # CLASS-AWARE GT
                # =================================================

                gt_boxes_by_class[
                    class_label
                ][
                    episode_id
                ] = gt_boxes


                total_gt_by_class[
                    class_label
                ] += len(
                    gt_boxes
                )


                # =================================================
                # CLASS-AWARE AP RECORDS
                #
                # All queries in an episode are predictions
                # of that episode's support semantic class.
                # =================================================

                for q in range(
                    len(scores)
                ):

                    prediction_records_by_class[
                        class_label
                    ].append({

                        "episode_id":
                            episode_id,

                        "score":
                            float(
                                scores[q].item()
                            ),

                        "box":
                            pred_boxes[q],
                    })


                # =================================================
                # MICRO P/R
                # =================================================

                (
                    tp,
                    fp,
                    fn
                ) = match_episode_predictions(

                    scores=
                        scores,

                    pred_boxes=
                        pred_boxes,

                    gt_boxes=
                        gt_boxes,

                    score_threshold=
                        score_threshold,

                    iou_threshold=
                        iou_threshold
                )


                total_tp += tp
                total_fp += fp
                total_fn += fn


                tp_by_class[
                    class_label
                ] += tp


                fp_by_class[
                    class_label
                ] += fp


                fn_by_class[
                    class_label
                ] += fn


                # =================================================
                # LOCALIZATION ONLY
                #
                # Best IoU among ALL object queries,
                # independent of classification score.
                # =================================================

                if len(gt_boxes) > 0:

                    pred_xyxy = (
                        box_cxcywh_to_xyxy(
                            pred_boxes
                        )
                    )


                    gt_xyxy = (
                        box_cxcywh_to_xyxy(
                            gt_boxes
                        )
                    )


                    iou_matrix, _ = box_iou(

                        pred_xyxy,

                        gt_xyxy
                    )


                    best_per_gt = (

                        iou_matrix

                        .max(
                            dim=0
                        )

                        .values
                    )


                    best_ious_all_gt.extend(

                        best_per_gt
                        .tolist()
                    )


    # ==========================================================
    # CLASS-WISE AP -> mAP
    # ==========================================================

    (
        map50,
        per_class_ap50
    ) = compute_classwise_map50(

        prediction_records_by_class=
            prediction_records_by_class,

        gt_boxes_by_class=
            gt_boxes_by_class,

        total_gt_by_class=
            total_gt_by_class,

        iou_threshold=
            iou_threshold
    )


    evaluated_classes = sorted(

        per_class_ap50.keys()
    )


    # ==========================================================
    # MICRO PRECISION / RECALL
    # ==========================================================

    precision50 = (

        total_tp

        /

        max(
            total_tp
            +
            total_fp,

            1
        )
    )


    recall50 = (

        total_tp

        /

        max(
            total_tp
            +
            total_fn,

            1
        )
    )


    # ==========================================================
    # OPTIONAL MACRO CLASS PRECISION / RECALL
    # ==========================================================

    per_class_precision50 = {}
    per_class_recall50 = {}


    for class_label in (
        evaluated_classes
    ):

        class_tp = (
            tp_by_class[
                class_label
            ]
        )

        class_fp = (
            fp_by_class[
                class_label
            ]
        )

        class_fn = (
            fn_by_class[
                class_label
            ]
        )


        per_class_precision50[
            class_label
        ] = (

            class_tp

            /

            max(
                class_tp
                +
                class_fp,

                1
            )
        )


        per_class_recall50[
            class_label
        ] = (

            class_tp

            /

            max(
                class_tp
                +
                class_fn,

                1
            )
        )


    macro_precision50 = (

        float(

            np.mean(

                list(
                    per_class_precision50
                    .values()
                )
            )
        )

        if
        len(
            per_class_precision50
        )
        >
        0

        else
        0.0
    )


    macro_recall50 = (

        float(

            np.mean(

                list(
                    per_class_recall50
                    .values()
                )
            )
        )

        if
        len(
            per_class_recall50
        )
        >
        0

        else
        0.0
    )


    # ==========================================================
    # LOCALIZATION METRICS
    # ==========================================================

    best_ious_np = np.asarray(

        best_ious_all_gt,

        dtype=np.float64
    )


    if len(best_ious_np) > 0:

        mean_best_iou = float(
            best_ious_np.mean()
        )


        localization_recall30 = float(

            np.mean(
                best_ious_np
                >=
                0.30
            )
        )


        localization_recall50 = float(

            np.mean(
                best_ious_np
                >=
                0.50
            )
        )


        localization_recall75 = float(

            np.mean(
                best_ious_np
                >=
                0.75
            )
        )


    else:

        mean_best_iou = 0.0

        localization_recall30 = 0.0

        localization_recall50 = 0.0

        localization_recall75 = 0.0


    total_gt = int(

        sum(
            total_gt_by_class.values()
        )
    )


    return {

        # --------------------------------------------------
        # Existing keys preserved
        # so Steps 28/29 do not break.
        # --------------------------------------------------

        "loss":
            (
                total_loss

                /
                max(
                    num_batches,
                    1
                )
            ),

        "episodic_map50":
            float(
                map50
            ),

        # MICRO P/R
        "precision50":
            float(
                precision50
            ),

        "recall50":
            float(
                recall50
            ),

        # Additional macro P/R
        "macro_precision50":
            float(
                macro_precision50
            ),

        "macro_recall50":
            float(
                macro_recall50
            ),

        # Localization
        "mean_best_iou":
            float(
                mean_best_iou
            ),

        "localization_recall30":
            float(
                localization_recall30
            ),

        "localization_recall50":
            float(
                localization_recall50
            ),

        "localization_recall75":
            float(
                localization_recall75
            ),

        # Diagnostics
        "total_gt":
            total_gt,

        "num_eval_classes":
            int(
                len(
                    evaluated_classes
                )
            ),

        "evaluated_classes":
            evaluated_classes,

        "per_class_ap50":
            per_class_ap50,

        "per_class_precision50":
            per_class_precision50,

        "per_class_recall50":
            per_class_recall50,
    }


print("=" * 70)
print("STEP 25 : CLASS-AWARE COCO EVALUATOR READY")
print("=" * 70)

print(
    "AP protocol:"
)

print(
    "  AP50 computed independently for each "
    "semantic COCO episode class."
)

print(
    "  mAP50 = mean of per-class AP50."
)

print(
    "  Full fixed COCO-Val should evaluate 80 classes."
)

print(
    "  Tiny subsets average only classes present."
)

print(
    "Precision/Recall:"
)

print(
    "  precision50 / recall50 = MICRO across episodes."
)

print(
    "  macro_precision50 / macro_recall50 also returned."
)

print("=" * 70)
