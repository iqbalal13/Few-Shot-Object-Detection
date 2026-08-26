# ==========================================================
# STEP 30B : Episodic Binary Detection Metrics
#            (V2 LOCKED)
# ==========================================================

import numpy as np
import torch

from tqdm import tqdm


def move_targets_to_device(
    targets,
    device
):

    return [

        {

            "boxes":
                target[
                    "boxes"
                ].to(
                    device
                ),

            "labels":
                target[
                    "labels"
                ].to(
                    device
                )
        }

        for target
        in targets
    ]


# ==========================================================
# AP50 one semantic class
# ==========================================================

def compute_single_class_ap50(

    prediction_records,

    episode_gt_boxes,

    total_gt,

    iou_threshold=0.5
):

    if (
        total_gt == 0

        or

        len(
            prediction_records
        )
        ==
        0
    ):

        return 0.0


    prediction_records = sorted(

        prediction_records,

        key=lambda x:
            x["score"],

        reverse=True
    )


    matched_gt = {

        episode_id:
            set()

        for episode_id
        in episode_gt_boxes
    }


    tp = []

    fp = []


    for record in (
        prediction_records
    ):

        episode_id = (
            record[
                "episode_id"
            ]
        )


        pred_box = (
            record[
                "box"
            ].unsqueeze(
                0
            )
        )


        gt_boxes = (
            episode_gt_boxes[
                episode_id
            ]
        )


        if (
            len(
                gt_boxes
            )
            ==
            0
        ):

            tp.append(
                0.0
            )

            fp.append(
                1.0
            )

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


        best_iou, best_gt = (
            ious[
                0
            ].max(
                dim=0
            )
        )


        best_gt = int(
            best_gt.item()
        )


        if (

            best_iou.item()
            >=
            iou_threshold

            and

            best_gt
            not in
            matched_gt[
                episode_id
            ]
        ):

            matched_gt[
                episode_id
            ].add(
                best_gt
            )


            tp.append(
                1.0
            )

            fp.append(
                0.0
            )


        else:

            tp.append(
                0.0
            )

            fp.append(
                1.0
            )


    tp = np.asarray(
        tp,
        dtype=np.float64
    )


    fp = np.asarray(
        fp,
        dtype=np.float64
    )


    cumulative_tp = (
        np.cumsum(
            tp
        )
    )


    cumulative_fp = (
        np.cumsum(
            fp
        )
    )


    recall = (

        cumulative_tp

        /

        max(
            total_gt,
            1
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


    # ======================================================
    # COCO-style 101-point interpolation @ IoU 0.50
    # ======================================================

    recall_thresholds = np.linspace(
        0.0,
        1.0,
        101
    )


    interpolated_precision = []


    for threshold in (
        recall_thresholds
    ):

        valid = (
            recall
            >=
            threshold
        )


        if (
            np.any(
                valid
            )
        ):

            interpolated_precision.append(

                float(
                    np.max(
                        precision[
                            valid
                        ]
                    )
                )
            )


        else:

            interpolated_precision.append(
                0.0
            )


    return float(
        np.mean(
            interpolated_precision
        )
    )


# ==========================================================
# Full episodic evaluator
# ==========================================================

@torch.no_grad()
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


    running_total = 0.0
    running_cls = 0.0
    running_bbox = 0.0
    running_giou = 0.0


    num_batches = 0


    prediction_records = []


    episode_gt_boxes = {}

    episode_class_map = {}


    total_gt = 0

    episode_counter = 0


    iterator = (
        tqdm(
            data_loader,
            desc="COCO Validation"
        )
        if show_progress
        else data_loader
    )


    for batch in iterator:

        support_images = (

            batch[
                "support_images"
            ].to(
                device,
                non_blocking=True
            )
        )


        query_images = (

            batch[
                "query_images"
            ].to(
                device,
                non_blocking=True
            )
        )


        episode_classes = (

            batch[
                "episode_classes"
            ]
        )


        targets = (
            move_targets_to_device(

                batch[
                    "query_targets"
                ],

                device
            )
        )


        outputs = model(

            support_images,

            query_images
        )


        if (
            outputs[
                "pred_logits"
            ].shape[-1]
            !=
            1
        ):

            raise RuntimeError(
                "V2 evaluator expects "
                "one foreground logit "
                "per query."
            )


        loss_dict = criterion(

            outputs,

            targets
        )


        running_total += (
            loss_dict[
                "loss_total"
            ].item()
        )


        running_cls += (
            loss_dict[
                "loss_cls"
            ].item()
        )


        running_bbox += (
            loss_dict[
                "loss_bbox"
            ].item()
        )


        running_giou += (
            loss_dict[
                "loss_giou"
            ].item()
        )


        # ==================================================
        # FOREGROUND SCORES
        #
        # semantic class BUKAN logit index
        # ==================================================

        foreground_scores = (

            outputs[
                "pred_logits"
            ]
            .sigmoid()
            .squeeze(-1)
        )


        for batch_idx in range(
            query_images.shape[0]
        ):

            class_label = int(

                episode_classes[
                    batch_idx
                ].item()
            )


            class_scores = (

                foreground_scores[
                    batch_idx
                ]
                .detach()
                .cpu()
            )


            boxes = (

                outputs[
                    "pred_boxes"
                ][
                    batch_idx
                ]
                .detach()
                .cpu()
            )


            gt_boxes = (

                targets[
                    batch_idx
                ][
                    "boxes"
                ]
                .detach()
                .cpu()
            )


            episode_id = (
                episode_counter
            )


            episode_gt_boxes[
                episode_id
            ] = gt_boxes


            episode_class_map[
                episode_id
            ] = (
                class_label
            )


            total_gt += (
                len(
                    gt_boxes
                )
            )


            for query_idx in range(
                boxes.shape[0]
            ):

                prediction_records.append({

                    "score":
                        float(
                            class_scores[
                                query_idx
                            ].item()
                        ),

                    "box":
                        boxes[
                            query_idx
                        ],

                    "episode_id":
                        episode_id,

                    "class_label":
                        class_label
                })


            episode_counter += 1


        num_batches += 1


    # ======================================================
    # Per Semantic-Class AP50
    # ======================================================

    evaluated_classes = sorted(

        set(
            episode_class_map.values()
        )
    )


    class_ap50 = {}


    for class_label in (
        evaluated_classes
    ):

        class_episode_ids = [

            episode_id

            for episode_id, label
            in episode_class_map.items()

            if (
                label
                ==
                class_label
            )
        ]


        class_gt = {

            episode_id:
                episode_gt_boxes[
                    episode_id
                ]

            for episode_id
            in class_episode_ids
        }


        class_total_gt = sum(

            len(
                boxes
            )

            for boxes
            in class_gt.values()
        )


        class_predictions = [

            record

            for record
            in prediction_records

            if (
                record[
                    "class_label"
                ]
                ==
                class_label
            )
        ]


        class_ap50[
            class_label
        ] = (
            compute_single_class_ap50(

                class_predictions,

                class_gt,

                class_total_gt,

                iou_threshold=
                    iou_threshold
            )
        )


    if (
        len(
            class_ap50
        )
        >
        0
    ):

        episodic_map50 = float(

            np.mean(

                list(
                    class_ap50.values()
                )
            )
        )


    else:

        episodic_map50 = (
            0.0
        )


    # ======================================================
    # P / R @ score threshold
    # ======================================================

    matched_gt = {

        episode_id:
            set()

        for episode_id
        in episode_gt_boxes
    }


    filtered_predictions = [

        record

        for record
        in prediction_records

        if (
            record[
                "score"
            ]
            >=
            score_threshold
        )
    ]


    filtered_predictions.sort(

        key=lambda x:
            x["score"],

        reverse=True
    )


    tp = 0
    fp = 0


    for record in (
        filtered_predictions
    ):

        episode_id = (
            record[
                "episode_id"
            ]
        )


        gt_boxes = (
            episode_gt_boxes[
                episode_id
            ]
        )


        if (
            len(
                gt_boxes
            )
            ==
            0
        ):

            fp += 1
            continue


        pred_xyxy = (
            box_cxcywh_to_xyxy(

                record[
                    "box"
                ].unsqueeze(
                    0
                )
            )
        )


        gt_xyxy = (
            box_cxcywh_to_xyxy(
                gt_boxes
            )
        )


        ious, _ = (
            box_iou(

                pred_xyxy,

                gt_xyxy
            )
        )


        best_iou, best_gt = (
            ious[
                0
            ].max(
                dim=0
            )
        )


        best_gt = int(
            best_gt.item()
        )


        if (

            best_iou.item()
            >=
            iou_threshold

            and

            best_gt
            not in
            matched_gt[
                episode_id
            ]
        ):

            matched_gt[
                episode_id
            ].add(
                best_gt
            )

            tp += 1


        else:

            fp += 1


    precision50 = (

        tp

        /

        max(
            tp + fp,
            1
        )
    )


    recall50 = (

        tp

        /

        max(
            total_gt,
            1
        )
    )


    return {

        "loss":

            running_total

            /
            max(
                num_batches,
                1
            ),


        "loss_cls":

            running_cls

            /
            max(
                num_batches,
                1
            ),


        "loss_bbox":

            running_bbox

            /
            max(
                num_batches,
                1
            ),


        "loss_giou":

            running_giou

            /
            max(
                num_batches,
                1
            ),


        "episodic_map50":
            episodic_map50,


        "precision50":
            float(
                precision50
            ),


        "recall50":
            float(
                recall50
            ),


        "class_ap50":
            class_ap50,


        "num_evaluated_classes":
            len(
                evaluated_classes
            )
    }


print("=" * 70)
print(
    "STEP 30B : V2 EPISODIC "
    "EVALUATOR READY"
)
print("=" * 70)
