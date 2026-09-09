# ==========================================================
# FINAL REPAIR HELPER SETUP
# Runtime-reset recovery only
#
# Defines ONLY helpers required by FINAL REPAIR D.
# NO training.
# ==========================================================

import random
import numpy as np
import torch
import torch.nn.functional as F


def v32_get_present_classes(
    dataset,
    image_id
):
    ann_ids = dataset.coco.getAnnIds(
        imgIds=[int(image_id)]
    )

    annotations = dataset.coco.loadAnns(
        ann_ids
    )

    present_labels = set()

    for ann in annotations:

        category_id = ann.get(
            "category_id",
            None
        )

        if category_id in dataset.cat2label:

            present_labels.add(
                int(
                    dataset.cat2label[
                        category_id
                    ]
                )
            )

    return present_labels


def v32_make_absent_wrong_support(
    dataset,
    query_image_id,
    current_class,
    epoch,
    step
):

    present_labels = (
        v32_get_present_classes(
            dataset=dataset,
            image_id=query_image_id
        )
    )

    present_labels.add(
        int(current_class)
    )

    candidate_labels = [

        int(label)

        for label in dataset.valid_labels

        if int(label)
        not in present_labels
    ]

    if len(candidate_labels) == 0:

        raise RuntimeError(
            "No absent wrong-support class "
            "available for this query image."
        )

    rng = random.Random(

        int(CONFIG["seed"])
        +
        2_000_000
        +
        int(epoch) * 100_000
        +
        int(step)
    )

    wrong_class = int(
        rng.choice(candidate_labels)
    )

    wrong_ann_id = rng.choice(
        dataset.class_to_ann_ids[
            wrong_class
        ]
    )

    wrong_support_image, _ = (
        dataset._load_support(
            wrong_ann_id,
            wrong_class
        )
    )

    assert (
        wrong_class
        not in present_labels
    )

    return (
        wrong_support_image,
        wrong_class
    )


def v32_compute_contrastive_loss(
    model,
    correct_prototype,
    wrong_prototype,
    query_feature_map,
    gt_boxes,
    required_margin
):

    _, _, feature_h, feature_w = (
        query_feature_map.shape
    )

    metric_losses = []
    correct_similarities = []
    wrong_similarities = []
    observed_margins = []

    for gt_box in gt_boxes:

        cx = float(
            gt_box[0].detach().item()
        )

        cy = float(
            gt_box[1].detach().item()
        )

        bw = float(
            gt_box[2].detach().item()
        )

        bh = float(
            gt_box[3].detach().item()
        )

        x1_float = (
            cx - bw / 2.0
        ) * feature_w

        y1_float = (
            cy - bh / 2.0
        ) * feature_h

        x2_float = (
            cx + bw / 2.0
        ) * feature_w

        y2_float = (
            cy + bh / 2.0
        ) * feature_h

        x1 = max(
            0,
            min(
                feature_w - 1,
                int(np.floor(x1_float))
            )
        )

        y1 = max(
            0,
            min(
                feature_h - 1,
                int(np.floor(y1_float))
            )
        )

        x2 = max(
            x1 + 1,
            min(
                feature_w,
                int(np.ceil(x2_float))
            )
        )

        y2 = max(
            y1 + 1,
            min(
                feature_h,
                int(np.ceil(y2_float))
            )
        )

        roi_feature = query_feature_map[
            :,
            :,
            y1:y2,
            x1:x2
        ]

        if (
            roi_feature.shape[-2] <= 0
            or
            roi_feature.shape[-1] <= 0
        ):
            raise RuntimeError(
                "Empty query ROI encountered."
            )

        roi_prototype = (
            model.support_encoder(
                roi_feature
            )
        )

        sim_correct = (
            F.cosine_similarity(
                correct_prototype,
                roi_prototype,
                dim=-1
            )
        )

        sim_wrong = (
            F.cosine_similarity(
                wrong_prototype,
                roi_prototype,
                dim=-1
            )
        )

        observed_margin = (
            sim_correct
            -
            sim_wrong
        )

        metric_loss = F.relu(
            required_margin
            -
            observed_margin
        )

        metric_losses.append(
            metric_loss
        )

        correct_similarities.append(
            sim_correct.detach()
        )

        wrong_similarities.append(
            sim_wrong.detach()
        )

        observed_margins.append(
            observed_margin.detach()
        )

    if len(metric_losses) == 0:

        raise RuntimeError(
            "No GT ROI available "
            "for contrastive loss."
        )

    contrastive_loss = (
        torch.stack(
            metric_losses
        )
        .mean()
    )

    mean_correct_similarity = (
        torch.stack(
            correct_similarities
        )
        .mean()
        .item()
    )

    mean_wrong_similarity = (
        torch.stack(
            wrong_similarities
        )
        .mean()
        .item()
    )

    mean_observed_margin = (
        torch.stack(
            observed_margins
        )
        .mean()
        .item()
    )

    return (
        contrastive_loss,
        mean_correct_similarity,
        mean_wrong_similarity,
        mean_observed_margin
    )


print("=" * 70)
print("✓ FINAL REPAIR HELPERS READY")
print("=" * 70)
