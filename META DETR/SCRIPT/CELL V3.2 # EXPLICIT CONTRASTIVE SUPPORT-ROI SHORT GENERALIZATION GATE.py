# ==========================================================
# CELL V3.2
# EXPLICIT CONTRASTIVE SUPPORT-ROI SHORT GENERALIZATION GATE
#
# PURPOSE
# ----------------------------------------------------------
# Keep EXACT SAME V3 architecture.
#
# Add explicit representation objective:
#
#   sim(correct support, target ROI)
#       >
#   sim(wrong support, target ROI) + margin
#
# This directly targets remaining H2 weakness and tests
# whether stronger support discrimination can improve
# unseen episodic generalization.
#
# IMPORTANT
# ----------------------------------------------------------
# - Starts from fresh official V3 `model`
# - Does NOT continue from short_model
# - Does NOT continue from v31_model
# - Official V3 weights remain untouched
# - 5 short epochs only
# - 800 steps / epoch
# - fixed COCO-Val
# - restores BEST V3.2 short epoch at end
# - DO NOT run STEP 32 afterward
# ==========================================================

import copy
import random
import numpy as np

import torch
import torch.nn.functional as F

from tqdm.auto import tqdm


print("=" * 70)
print("V3.2 CONTRASTIVE SUPPORT-ROI SHORT GENERALIZATION GATE")
print("=" * 70)


# ==========================================================
# CONFIG
# ==========================================================

V32_CONFIG = {

    "epochs":
        5,

    "steps_per_epoch":
        800,

    # Previous V3-H2 mean margin was only about +0.028.
    # We explicitly ask for clearer separation.
    "contrastive_margin":
        0.10,

    # Auxiliary representation loss weight.
    # Keep moderate so normal detection remains primary.
    "contrastive_weight":
        2.0
}


# ==========================================================
# PRE-FLIGHT SAFETY
# ==========================================================

assert (
    model.__class__.__name__
    ==
    "SimplifiedMetaDETRV3"
), (
    "Current `model` is not SimplifiedMetaDETRV3."
)


assert (
    CONFIG[
        "support_conditioning"
    ]
    ==
    "prototype_relation"
), (
    "V3.2 requires V3 prototype_relation conditioning."
)


assert (
    COCO_CONFIG[
        "batch_size"
    ]
    ==
    1
), (
    "V3.2 short gate currently expects batch_size=1."
)


assert (
    COCO_CONFIG[
        "shuffle"
    ]
    is False
), (
    "Expected deterministic COCO training loader "
    "with shuffle=False."
)


# ==========================================================
# OPTIONAL GPU MEMORY CLEANUP
#
# Old experimental models are no longer required for this
# training run. Move them to CPU, but KEEP variables/results.
# ==========================================================

for old_model_name in [
    "tiny_model",
    "short_model",
    "v31_model"
]:

    if old_model_name in globals():

        old_model = globals()[
            old_model_name
        ]

        if hasattr(
            old_model,
            "to"
        ):

            old_model.to(
                "cpu"
            )


if torch.cuda.is_available():

    torch.cuda.empty_cache()


# ==========================================================
# FRESH V3 COPY
#
# model = original untouched V3
# v32_model = new independent experiment
# ==========================================================

v32_model = copy.deepcopy(
    model
).to(
    CONFIG["device"]
)


v32_optimizer, v32_scheduler = (
    build_optimizer_and_scheduler(
        v32_model
    )
)


print(
    "Fresh source model :",
    model.__class__.__name__
)

print(
    "Experiment model   :",
    v32_model.__class__.__name__
)

print(
    "Contrastive margin :",
    V32_CONFIG[
        "contrastive_margin"
    ]
)

print(
    "Contrastive weight :",
    V32_CONFIG[
        "contrastive_weight"
    ]
)

print("=" * 70)


# ==========================================================
# HELPER 1
# Get ALL semantic classes appearing in a COCO query image.
#
# Conservative:
# if a class appears anywhere in COCO annotations,
# we do NOT use it as wrong support.
# ==========================================================

def v32_get_present_classes(
    dataset,
    image_id
):

    ann_ids = (
        dataset.coco.getAnnIds(

            imgIds=[
                int(
                    image_id
                )
            ]
        )
    )


    annotations = (
        dataset.coco.loadAnns(
            ann_ids
        )
    )


    present_labels = set()


    for ann in annotations:

        category_id = ann.get(
            "category_id",
            None
        )


        if (
            category_id
            in
            dataset.cat2label
        ):

            present_labels.add(

                int(
                    dataset.cat2label[
                        category_id
                    ]
                )
            )


    return present_labels


# ==========================================================
# HELPER 2
# Pick a deterministic WRONG support whose class is
# ABSENT from the current query image.
#
# This avoids accidental false-negative supervision.
# ==========================================================

def v32_make_absent_wrong_support(
    dataset,
    query_image_id,
    current_class,
    epoch,
    step
):

    present_labels = (
        v32_get_present_classes(

            dataset=
                dataset,

            image_id=
                query_image_id
        )
    )


    # Safety: positive episodic class is never eligible.
    present_labels.add(
        int(
            current_class
        )
    )


    candidate_labels = [

        int(
            label
        )

        for label
        in dataset.valid_labels

        if (
            int(label)
            not in
            present_labels
        )
    ]


    if (
        len(
            candidate_labels
        )
        ==
        0
    ):

        raise RuntimeError(
            "No absent wrong-support class "
            "available for this query image."
        )


    # ----------------------------------------------
    # Deterministic RNG
    # ----------------------------------------------

    rng = random.Random(

        int(
            CONFIG["seed"]
        )

        +

        2_000_000

        +

        int(epoch)
        *
        100_000

        +

        int(step)
    )


    wrong_class = int(
        rng.choice(
            candidate_labels
        )
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


    # ----------------------------------------------
    # Critical safety check
    # ----------------------------------------------

    assert (
        wrong_class
        not in
        present_labels
    )


    return (
        wrong_support_image,
        wrong_class
    )


# ==========================================================
# HELPER 3
# Reproduce EXACT V3 forward while exposing:
#
#   support prototype
#   query feature map
#
# Architecture is NOT modified.
# ==========================================================

def v32_forward_with_features(
    model,
    support_image,
    query_image
):

    # ======================================================
    # Shared Backbone
    # ======================================================

    support_feature_map = (
        model.backbone(
            support_image
        )
    )


    query_feature_map = (
        model.backbone(
            query_image
        )
    )


    # ======================================================
    # V3 Support Prototype
    # ======================================================

    prototype = (
        model.support_encoder(
            support_feature_map
        )
    )


    # ======================================================
    # Query Spatial Tokens
    # ======================================================

    (
        query_tokens,
        spatial_shape

    ) = model.query_encoder(
        query_feature_map
    )


    # ======================================================
    # Support / Query Batch Compatibility
    # ======================================================

    if (
        prototype.shape[0]
        ==
        1

        and

        query_tokens.shape[0]
        >
        1
    ):

        prototype = (
            prototype.expand(

                query_tokens.shape[0],

                -1
            )
        )


    if (
        prototype.shape[0]
        !=
        query_tokens.shape[0]
    ):

        raise RuntimeError(

            "V3.2 prototype/query batch mismatch: "

            f"{prototype.shape[0]} vs "

            f"{query_tokens.shape[0]}"
        )


    # ======================================================
    # V3 Prototype Relation Conditioning
    # ======================================================

    guided_query = (
        model.prototype_conditioner(

            query_tokens=
                query_tokens,

            prototype=
                prototype
        )
    )


    # ======================================================
    # Position Encoding
    # ======================================================

    query_position = (
        model.position_encoding(
            query_feature_map
        )
    )


    assert (
        query_position.shape
        ==
        guided_query.shape
    )


    # ======================================================
    # Transformer Encoder
    # ======================================================

    encoder_input = (

        guided_query

        +

        query_position
    )


    memory = (
        model.transformer_encoder(
            encoder_input
        )
    )


    # ======================================================
    # Prototype-conditioned Decoder
    # ======================================================

    decoder_output = (
        model.transformer_decoder(

            memory=
                memory,

            prototype=
                prototype
        )
    )


    # ======================================================
    # Detection Head
    # ======================================================

    (
        pred_logits,
        pred_boxes

    ) = model.detection_head(
        decoder_output
    )


    outputs = {

        "pred_logits":
            pred_logits,

        "pred_boxes":
            pred_boxes
    }


    return (
        outputs,
        prototype,
        query_feature_map
    )


# ==========================================================
# HELPER 4
# Explicit Correct-vs-Wrong Support Contrastive Loss
#
# For every GT object:
#
#   observed_margin =
#       sim(correct support, GT ROI)
#       -
#       sim(wrong support, GT ROI)
#
# Loss:
#
#   ReLU(required_margin - observed_margin)
#
# So:
#
#   correct_sim >= wrong_sim + required_margin
#
# is rewarded.
# ==========================================================

def v32_compute_contrastive_loss(
    model,
    correct_prototype,
    wrong_prototype,
    query_feature_map,
    gt_boxes,
    required_margin
):

    (
        _,
        _,
        feature_h,
        feature_w

    ) = query_feature_map.shape


    metric_losses = []

    correct_similarities = []

    wrong_similarities = []

    observed_margins = []


    # ======================================================
    # Every GT object in the query
    # ======================================================

    for gt_box in gt_boxes:

        # --------------------------------------------------
        # GT format:
        # normalized cx,cy,w,h
        # --------------------------------------------------

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


        # --------------------------------------------------
        # Convert normalized box to feature-map coordinates
        # --------------------------------------------------

        x1_float = (
            cx
            -
            bw / 2.0
        ) * feature_w


        y1_float = (
            cy
            -
            bh / 2.0
        ) * feature_h


        x2_float = (
            cx
            +
            bw / 2.0
        ) * feature_w


        y2_float = (
            cy
            +
            bh / 2.0
        ) * feature_h


        x1 = max(

            0,

            min(

                feature_w - 1,

                int(
                    np.floor(
                        x1_float
                    )
                )
            )
        )


        y1 = max(

            0,

            min(

                feature_h - 1,

                int(
                    np.floor(
                        y1_float
                    )
                )
            )
        )


        x2 = max(

            x1 + 1,

            min(

                feature_w,

                int(
                    np.ceil(
                        x2_float
                    )
                )
            )
        )


        y2 = max(

            y1 + 1,

            min(

                feature_h,

                int(
                    np.ceil(
                        y2_float
                    )
                )
            )
        )


        # --------------------------------------------------
        # Extract target ROI from QUERY backbone feature map
        #
        # Gradient still flows through this ROI tensor.
        # --------------------------------------------------

        roi_feature = (
            query_feature_map[
                :,
                :,
                y1:y2,
                x1:x2
            ]
        )


        if (
            roi_feature.shape[-2]
            <=
            0

            or

            roi_feature.shape[-1]
            <=
            0
        ):

            raise RuntimeError(
                "Empty query ROI encountered."
            )


        # --------------------------------------------------
        # SAME V3 prototype encoder represents GT ROI
        # --------------------------------------------------

        roi_prototype = (
            model.support_encoder(
                roi_feature
            )
        )


        # --------------------------------------------------
        # Correct support vs target ROI
        # --------------------------------------------------

        sim_correct = (
            F.cosine_similarity(

                correct_prototype,

                roi_prototype,

                dim=-1
            )
        )


        # --------------------------------------------------
        # Wrong support vs SAME target ROI
        # --------------------------------------------------

        sim_wrong = (
            F.cosine_similarity(

                wrong_prototype,

                roi_prototype,

                dim=-1
            )
        )


        # --------------------------------------------------
        # Relative discrimination margin
        # --------------------------------------------------

        observed_margin = (

            sim_correct

            -

            sim_wrong
        )


        # --------------------------------------------------
        # Margin ranking / triplet-like objective
        # --------------------------------------------------

        metric_loss = (
            F.relu(

                required_margin

                -

                observed_margin
            )
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


    if (
        len(
            metric_losses
        )
        ==
        0
    ):

        raise RuntimeError(
            "No GT ROI available for V3.2 contrastive loss."
        )


    # ======================================================
    # Aggregate all GT objects from this episode
    # ======================================================

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


# ==========================================================
# BASELINE
#
# Fresh V3 before any V3.2 optimization.
# ==========================================================

print()
print("Running fresh V3 baseline COCO-Val...")
print()


v32_baseline_metrics = (
    evaluate_episodic_model(

        model=
            v32_model,

        data_loader=
            val_loader,

        criterion=
            criterion,

        device=
            CONFIG["device"],

        score_threshold=
            TRAIN_CONFIG[
                "score_threshold"
            ],

        iou_threshold=
            TRAIN_CONFIG[
                "iou_threshold"
            ],

        show_progress=
            True
    )
)


print()
print("-" * 70)
print("V3.2 BASELINE")
print("-" * 70)


print(
    f"Val Loss : "
    f"{v32_baseline_metrics['loss']:.6f}"
)


print(
    f"mAP50    : "
    f"{v32_baseline_metrics['episodic_map50']:.8f}"
)


print(
    f"P@0.50   : "
    f"{v32_baseline_metrics['precision50']:.6f}"
)


print(
    f"R@0.50   : "
    f"{v32_baseline_metrics['recall50']:.6f}"
)


print("-" * 70)


# ==========================================================
# HISTORY
# ==========================================================

v32_history = []


v32_best_state = None

v32_best_epoch = -1

v32_best_map50 = -1.0

v32_best_val_loss = float(
    "inf"
)


# ==========================================================
# TRAIN V3.2
# ==========================================================

for v32_epoch in range(
    V32_CONFIG[
        "epochs"
    ]
):

    # ------------------------------------------------------
    # New deterministic episodic samples each epoch
    # ------------------------------------------------------

    train_dataset.set_epoch(
        v32_epoch
    )


    v32_model.train()


    # ------------------------------------------------------
    # Same BN policy as existing training
    # ------------------------------------------------------

    freeze_backbone_bn_statistics(
        v32_model.backbone
    )


    # ======================================================
    # Running statistics
    # ======================================================

    running_detection = 0.0

    running_cls = 0.0

    running_bbox = 0.0

    running_giou = 0.0

    running_contrastive = 0.0

    running_total = 0.0

    running_correct_sim = 0.0

    running_wrong_sim = 0.0

    running_margin = 0.0

    actual_steps = 0


    progress_bar = tqdm(

        train_loader,

        desc=(

            f"V3.2 "

            f"[{v32_epoch + 1}/"

            f"{V32_CONFIG['epochs']}]"
        )
    )


    # ======================================================
    # MINI-BATCH LOOP
    # ======================================================

    for batch in progress_bar:

        if (
            actual_steps
            >=
            V32_CONFIG[
                "steps_per_epoch"
            ]
        ):

            break


        # ==================================================
        # CORRECT SUPPORT
        # ==================================================

        support_images = (

            batch[
                "support_images"
            ]

            .to(

                CONFIG["device"],

                non_blocking=True
            )
        )


        # ==================================================
        # QUERY
        # ==================================================

        query_images = (

            batch[
                "query_images"
            ]

            .to(

                CONFIG["device"],

                non_blocking=True
            )
        )


        # ==================================================
        # TARGETS
        # ==================================================

        targets = (
            move_targets_to_device(

                batch[
                    "query_targets"
                ],

                CONFIG["device"]
            )
        )


        # ==================================================
        # CURRENT EPISODIC CLASS
        # ==================================================

        current_class = int(

            batch[
                "episode_classes"
            ][0]
            .item()
        )


        # ==================================================
        # QUERY IMAGE ID
        # ==================================================

        query_image_id = int(

            batch[
                "query_targets"
            ][0][
                "image_id"
            ]
            .item()
        )


        # ==================================================
        # ABSENT WRONG SUPPORT
        # ==================================================

        (
            wrong_support_image,
            wrong_class

        ) = v32_make_absent_wrong_support(

            dataset=
                train_dataset,

            query_image_id=
                query_image_id,

            current_class=
                current_class,

            epoch=
                v32_epoch,

            step=
                actual_steps
        )


        wrong_support_images = (

            wrong_support_image

            .unsqueeze(0)

            .to(

                CONFIG["device"],

                non_blocking=True
            )
        )


        # ==================================================
        # MAIN V3 FORWARD
        #
        # Correct support + query
        # ==================================================

        (
            outputs,
            correct_prototype,
            query_feature_map

        ) = v32_forward_with_features(

            model=
                v32_model,

            support_image=
                support_images,

            query_image=
                query_images
        )


        # ==================================================
        # STANDARD DETECTION LOSS
        # ==================================================

        detection_loss_dict = (
            criterion(

                outputs,

                targets
            )
        )


        detection_loss = (

            detection_loss_dict[
                "loss_total"
            ]
        )


        # ==================================================
        # WRONG SUPPORT PROTOTYPE
        #
        # No detector forward is needed.
        # We only need its representation.
        # ==================================================

        wrong_support_feature_map = (
            v32_model.backbone(
                wrong_support_images
            )
        )


        wrong_prototype = (
            v32_model.support_encoder(
                wrong_support_feature_map
            )
        )


        # ==================================================
        # EXPLICIT SUPPORT-ROI CONTRASTIVE LOSS
        # ==================================================

        (
            contrastive_loss,
            mean_correct_sim,
            mean_wrong_sim,
            mean_observed_margin

        ) = v32_compute_contrastive_loss(

            model=
                v32_model,

            correct_prototype=
                correct_prototype,

            wrong_prototype=
                wrong_prototype,

            query_feature_map=
                query_feature_map,

            gt_boxes=
                targets[0][
                    "boxes"
                ],

            required_margin=
                V32_CONFIG[
                    "contrastive_margin"
                ]
        )


        # ==================================================
        # TOTAL V3.2 OBJECTIVE
        #
        # Normal detector objective remains primary.
        # ==================================================

        total_loss = (

            detection_loss

            +

            V32_CONFIG[
                "contrastive_weight"
            ]

            *

            contrastive_loss
        )


        # ==================================================
        # NUMERICAL SAFETY
        # ==================================================

        if not torch.isfinite(
            total_loss
        ):

            raise RuntimeError(
                "V3.2 total loss became NaN/Inf."
            )


        if not torch.isfinite(
            contrastive_loss
        ):

            raise RuntimeError(
                "V3.2 contrastive loss became NaN/Inf."
            )


        # ==================================================
        # BACKPROP
        # ==================================================

        v32_optimizer.zero_grad(
            set_to_none=True
        )


        total_loss.backward()


        torch.nn.utils.clip_grad_norm_(

            v32_model.parameters(),

            max_norm=
                TRAIN_CONFIG[
                    "gradient_clip"
                ]
        )


        v32_optimizer.step()


        # ==================================================
        # STATISTICS
        # ==================================================

        running_detection += (
            detection_loss.item()
        )


        running_cls += (

            detection_loss_dict[
                "loss_cls"
            ]
            .item()
        )


        running_bbox += (

            detection_loss_dict[
                "loss_bbox"
            ]
            .item()
        )


        running_giou += (

            detection_loss_dict[
                "loss_giou"
            ]
            .item()
        )


        running_contrastive += (
            contrastive_loss.item()
        )


        running_total += (
            total_loss.item()
        )


        running_correct_sim += (
            mean_correct_sim
        )


        running_wrong_sim += (
            mean_wrong_sim
        )


        running_margin += (
            mean_observed_margin
        )


        actual_steps += 1


        # ==================================================
        # PROGRESS
        # ==================================================

        progress_bar.set_postfix({

            "Det":
                f"{detection_loss.item():.3f}",

            "Ctr":
                f"{contrastive_loss.item():.3f}",

            "Margin":
                f"{mean_observed_margin:.3f}",

            "Step":
                (
                    f"{actual_steps}/"
                    f"{V32_CONFIG['steps_per_epoch']}"
                )
        })


    if (
        actual_steps
        ==
        0
    ):

        raise RuntimeError(
            "V3.2 completed zero training steps."
        )


    # ======================================================
    # FIXED COCO-VAL
    #
    # Evaluation remains STANDARD correct-support episodic
    # detection. Contrastive loss is NOT added to Val Loss.
    # ======================================================

    print()
    print("Running fixed COCO-Val...")
    print()


    v32_val_metrics = (
        evaluate_episodic_model(

            model=
                v32_model,

            data_loader=
                val_loader,

            criterion=
                criterion,

            device=
                CONFIG["device"],

            score_threshold=
                TRAIN_CONFIG[
                    "score_threshold"
                ],

            iou_threshold=
                TRAIN_CONFIG[
                    "iou_threshold"
                ],

            show_progress=
                True
        )
    )


    # ======================================================
    # EPOCH RECORD
    # ======================================================

    record = {

        "epoch":
            v32_epoch + 1,

        "detection_loss":
            running_detection
            /
            actual_steps,

        "cls_loss":
            running_cls
            /
            actual_steps,

        "bbox_loss":
            running_bbox
            /
            actual_steps,

        "giou_loss":
            running_giou
            /
            actual_steps,

        "contrastive_loss":
            running_contrastive
            /
            actual_steps,

        "total_loss":
            running_total
            /
            actual_steps,

        "correct_sim":
            running_correct_sim
            /
            actual_steps,

        "wrong_sim":
            running_wrong_sim
            /
            actual_steps,

        "train_margin":
            running_margin
            /
            actual_steps,

        "val_loss":
            v32_val_metrics[
                "loss"
            ],

        "map50":
            v32_val_metrics[
                "episodic_map50"
            ],

        "precision50":
            v32_val_metrics[
                "precision50"
            ],

        "recall50":
            v32_val_metrics[
                "recall50"
            ]
    }


    v32_history.append(
        record
    )


    # ======================================================
    # BEST MODEL SELECTION
    #
    # Primary   : mAP50
    # Tie-break : Val Loss
    # ======================================================

    is_better = (

        record[
            "map50"
        ]

        >

        v32_best_map50

        or

        (
            abs(

                record[
                    "map50"
                ]

                -

                v32_best_map50

            )
            <
            1e-12

            and

            record[
                "val_loss"
            ]

            <

            v32_best_val_loss
        )
    )


    if is_better:

        v32_best_epoch = (
            v32_epoch
            +
            1
        )


        v32_best_map50 = float(
            record[
                "map50"
            ]
        )


        v32_best_val_loss = float(
            record[
                "val_loss"
            ]
        )


        # ----------------------------------------------
        # Save best state on CPU.
        # No official checkpoint file is written.
        # ----------------------------------------------

        v32_best_state = {

            name:
                tensor
                .detach()
                .cpu()
                .clone()

            for (
                name,
                tensor
            )

            in

            v32_model
            .state_dict()
            .items()
        }


    # ======================================================
    # REPORT
    # ======================================================

    print()
    print("=" * 70)

    print(

        f"V3.2 SHORT EPOCH "
        f"{v32_epoch + 1}/"
        f"{V32_CONFIG['epochs']}"
    )

    print("-" * 70)


    print(
        f"Detection Loss    : "
        f"{record['detection_loss']:.4f}"
    )


    print(
        f"CLS Loss          : "
        f"{record['cls_loss']:.4f}"
    )


    print(
        f"BBox Loss         : "
        f"{record['bbox_loss']:.4f}"
    )


    print(
        f"GIoU Loss         : "
        f"{record['giou_loss']:.4f}"
    )


    print(
        f"Contrastive Loss  : "
        f"{record['contrastive_loss']:.6f}"
    )


    print(
        f"Combined Loss     : "
        f"{record['total_loss']:.4f}"
    )


    print("-" * 70)


    print(
        f"Correct Sim       : "
        f"{record['correct_sim']:.6f}"
    )


    print(
        f"Wrong Sim         : "
        f"{record['wrong_sim']:.6f}"
    )


    print(
        f"Train Margin      : "
        f"{record['train_margin']:.6f}"
    )


    print(
        f"Required Margin   : "
        f"{V32_CONFIG['contrastive_margin']:.6f}"
    )


    print("-" * 70)


    print(
        f"Val Loss          : "
        f"{record['val_loss']:.4f}"
    )


    print(
        f"Val mAP50         : "
        f"{record['map50']:.8f}"
    )


    print(
        f"Val P@0.50        : "
        f"{record['precision50']:.6f}"
    )


    print(
        f"Val R@0.50        : "
        f"{record['recall50']:.6f}"
    )


    print("=" * 70)


    # ======================================================
    # SCHEDULER
    # ======================================================

    v32_scheduler.step()


# ==========================================================
# RESTORE BEST V3.2 SHORT EPOCH
# ==========================================================

assert (
    v32_best_state
    is not None
), (
    "No V3.2 best state was recorded."
)


v32_model.load_state_dict(
    v32_best_state
)


v32_model.to(
    CONFIG["device"]
)


v32_model.eval()


# ==========================================================
# FINAL SUMMARY
# ==========================================================

print()
print("=" * 70)
print("V3.2 SHORT GENERALIZATION FINAL")
print("=" * 70)


print(
    f"Baseline mAP50     : "
    f"{v32_baseline_metrics['episodic_map50']:.8f}"
)


print(
    f"Best Epoch         : "
    f"{v32_best_epoch}"
)


print(
    f"Best Val Loss      : "
    f"{v32_best_val_loss:.6f}"
)


print(
    f"Best mAP50         : "
    f"{v32_best_map50:.8f}"
)


print(
    f"Contrastive Margin : "
    f"{V32_CONFIG['contrastive_margin']:.3f}"
)


print(
    f"Contrastive Weight : "
    f"{V32_CONFIG['contrastive_weight']:.3f}"
)


print("-" * 70)


print(
    "✓ v32_model restored to BEST V3.2 short epoch."
)


print(
    "✓ Official V3 `model` weights remain untouched."
)


print(
    "✓ V3 architecture itself was NOT changed."
)


print(
    "STOP HERE. DO NOT RUN STEP 32."
)


print("=" * 70)
