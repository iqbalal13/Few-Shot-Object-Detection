# ==========================================================
# CELL FINAL REPAIR D
# SHORT COCO GENERALIZATION GATE
#
# 5 epochs x 800 steps
# Fixed unseen COCO-Val
#
# Objective:
#   detection loss
#   +
#   2.0 * support-ROI contrastive loss
#
# IMPORTANT:
# - starts fresh from final_model
# - NOT from final_tiny_model
# - final_model remains untouched
# - restores best short epoch at end
# ==========================================================

import copy
import numpy as np
import torch
import torch.nn.functional as F

from tqdm.auto import tqdm


print("=" * 70)
print("CELL FINAL REPAIR D : SHORT COCO GENERALIZATION")
print("=" * 70)


# ==========================================================
# SAFETY
# ==========================================================

assert "final_model" in globals()
assert "train_dataset" in globals()
assert "train_loader" in globals()
assert "val_loader" in globals()
assert "criterion" in globals()

# These are the already-tested V3.2 helpers.
assert "v32_make_absent_wrong_support" in globals(), (
    "V3.2 wrong-support helper missing."
)

assert "v32_compute_contrastive_loss" in globals(), (
    "V3.2 contrastive helper missing."
)


FINAL_SHORT_CONFIG = {

    "epochs":
        5,

    "steps_per_epoch":
        800,

    "contrastive_margin":
        0.10,

    "contrastive_weight":
        2.0
}


# ==========================================================
# FINAL FORWARD WITH FEATURES
#
# Same final_model forward,
# but expose:
#   prototype
#   query backbone feature map
# ==========================================================

def final_forward_with_features(
    model,
    support_image,
    query_image
):

    support_feature_map = model.backbone(
        support_image
    )

    query_feature_map = model.backbone(
        query_image
    )


    prototype = model.support_encoder(
        support_feature_map
    )


    (
        query_tokens,
        spatial_shape

    ) = model.query_encoder(
        query_feature_map
    )


    if (
        prototype.shape[0] == 1
        and
        query_tokens.shape[0] > 1
    ):

        prototype = prototype.expand(
            query_tokens.shape[0],
            -1
        )


    if (
        prototype.shape[0]
        !=
        query_tokens.shape[0]
    ):

        raise RuntimeError(
            "Final prototype/query batch mismatch."
        )


    guided_query = model.prototype_conditioner(

        query_tokens=
            query_tokens,

        prototype=
            prototype
    )


    query_position = model.position_encoding(
        query_feature_map
    )


    memory = model.transformer_encoder(

        guided_query
        +
        query_position
    )


    decoder_output = model.transformer_decoder(

        memory=
            memory,

        prototype=
            prototype
    )


    (
        pred_logits,
        pred_boxes

    ) = model.detection_head(

        decoder_output,

        prototype
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
# FRESH SHORT MODEL
# ==========================================================

final_short_model = copy.deepcopy(

    final_model

).to(
    CONFIG["device"]
)


final_short_optimizer, final_short_scheduler = (

    build_optimizer_and_scheduler(
        final_short_model
    )
)


print(
    "Model source       : fresh final_model"
)

print(
    "Epochs             :",
    FINAL_SHORT_CONFIG["epochs"]
)

print(
    "Steps / epoch      :",
    FINAL_SHORT_CONFIG[
        "steps_per_epoch"
    ]
)

print(
    "Contrastive margin :",
    FINAL_SHORT_CONFIG[
        "contrastive_margin"
    ]
)

print(
    "Contrastive weight :",
    FINAL_SHORT_CONFIG[
        "contrastive_weight"
    ]
)

print("=" * 70)


# ==========================================================
# BASELINE
# ==========================================================

print()
print("Running FINAL baseline COCO-Val...")
print()


final_baseline_metrics = evaluate_episodic_model(

    model=
        final_short_model,

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


print()
print("-" * 70)
print("FINAL REPAIR BASELINE")
print("-" * 70)

print(
    f"Val Loss : "
    f"{final_baseline_metrics['loss']:.6f}"
)

print(
    f"mAP50    : "
    f"{final_baseline_metrics['episodic_map50']:.8f}"
)

print(
    f"P@0.50   : "
    f"{final_baseline_metrics['precision50']:.6f}"
)

print(
    f"R@0.50   : "
    f"{final_baseline_metrics['recall50']:.6f}"
)

print("-" * 70)


# ==========================================================
# BEST STATE
# ==========================================================

final_short_history = []

final_best_state = None

final_best_epoch = -1

final_best_map50 = -1.0

final_best_val_loss = float(
    "inf"
)


# ==========================================================
# TRAIN
# ==========================================================

for epoch in range(

    FINAL_SHORT_CONFIG[
        "epochs"
    ]
):

    train_dataset.set_epoch(
        epoch
    )


    final_short_model.train()


    freeze_backbone_bn_statistics(
        final_short_model.backbone
    )


    running_detection = 0.0
    running_cls = 0.0
    running_bbox = 0.0
    running_giou = 0.0
    running_contrastive = 0.0
    running_combined = 0.0

    running_correct_sim = 0.0
    running_wrong_sim = 0.0
    running_margin = 0.0

    actual_steps = 0


    progress_bar = tqdm(

        train_loader,

        desc=(
            f"FINAL SHORT "
            f"[{epoch+1}/"
            f"{FINAL_SHORT_CONFIG['epochs']}]"
        )
    )


    for batch in progress_bar:

        if (
            actual_steps
            >=
            FINAL_SHORT_CONFIG[
                "steps_per_epoch"
            ]
        ):

            break


        # ==================================================
        # DATA
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


        query_images = (

            batch[
                "query_images"
            ]

            .to(
                CONFIG["device"],
                non_blocking=True
            )
        )


        targets = move_targets_to_device(

            batch[
                "query_targets"
            ],

            CONFIG["device"]
        )


        current_class = int(

            batch[
                "episode_classes"
            ][0].item()
        )


        query_image_id = int(

            batch[
                "query_targets"
            ][0][
                "image_id"
            ].item()
        )


        # ==================================================
        # WRONG SUPPORT — class absent from query image
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
                epoch,

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
        # FINAL DETECTOR FORWARD
        # ==================================================

        (
            outputs,
            correct_prototype,
            query_feature_map

        ) = final_forward_with_features(

            model=
                final_short_model,

            support_image=
                support_images,

            query_image=
                query_images
        )


        # ==================================================
        # DETECTION LOSS
        # ==================================================

        detection_loss_dict = criterion(

            outputs,

            targets
        )


        detection_loss = detection_loss_dict[
            "loss_total"
        ]


        # ==================================================
        # WRONG SUPPORT PROTOTYPE
        # ==================================================

        wrong_support_feature_map = (

            final_short_model.backbone(
                wrong_support_images
            )
        )


        wrong_prototype = (

            final_short_model.support_encoder(
                wrong_support_feature_map
            )
        )


        # ==================================================
        # SAME LOCKED V3.2 CONTRASTIVE OBJECTIVE
        # ==================================================

        (
            contrastive_loss,
            mean_correct_sim,
            mean_wrong_sim,
            mean_observed_margin

        ) = v32_compute_contrastive_loss(

            model=
                final_short_model,

            correct_prototype=
                correct_prototype,

            wrong_prototype=
                wrong_prototype,

            query_feature_map=
                query_feature_map,

            gt_boxes=
                targets[0]["boxes"],

            required_margin=
                FINAL_SHORT_CONFIG[
                    "contrastive_margin"
                ]
        )


        combined_loss = (

            detection_loss

            +

            FINAL_SHORT_CONFIG[
                "contrastive_weight"
            ]

            *
            contrastive_loss
        )


        if not torch.isfinite(
            combined_loss
        ):

            raise RuntimeError(
                "Final short loss became NaN/Inf."
            )


        # ==================================================
        # BACKPROP
        # ==================================================

        final_short_optimizer.zero_grad(
            set_to_none=True
        )


        combined_loss.backward()


        torch.nn.utils.clip_grad_norm_(

            final_short_model.parameters(),

            max_norm=
                TRAIN_CONFIG[
                    "gradient_clip"
                ]
        )


        final_short_optimizer.step()


        # ==================================================
        # STATS
        # ==================================================

        running_detection += (
            detection_loss.item()
        )

        running_cls += (

            detection_loss_dict[
                "loss_cls"
            ].item()
        )

        running_bbox += (

            detection_loss_dict[
                "loss_bbox"
            ].item()
        )

        running_giou += (

            detection_loss_dict[
                "loss_giou"
            ].item()
        )

        running_contrastive += (
            contrastive_loss.item()
        )

        running_combined += (
            combined_loss.item()
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
                    f"{FINAL_SHORT_CONFIG['steps_per_epoch']}"
                )
        })


    if actual_steps == 0:

        raise RuntimeError(
            "Final short completed zero steps."
        )


    # ======================================================
    # FIXED COCO-VAL
    # ======================================================

    print()
    print("Running fixed unseen COCO-Val...")
    print()


    val_metrics = evaluate_episodic_model(

        model=
            final_short_model,

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


    record = {

        "epoch":
            epoch + 1,

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

        "combined_loss":
            running_combined
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
            val_metrics[
                "loss"
            ],

        "map50":
            val_metrics[
                "episodic_map50"
            ],

        "precision50":
            val_metrics[
                "precision50"
            ],

        "recall50":
            val_metrics[
                "recall50"
            ]
    }


    final_short_history.append(
        record
    )


    # ======================================================
    # BEST MODEL
    # primary mAP50, tie-break val loss
    # ======================================================

    is_better = (

        record["map50"]
        >
        final_best_map50

        or

        (
            abs(
                record["map50"]
                -
                final_best_map50
            )
            <
            1e-12

            and

            record["val_loss"]
            <
            final_best_val_loss
        )
    )


    if is_better:

        final_best_epoch = (
            epoch + 1
        )

        final_best_map50 = float(
            record["map50"]
        )

        final_best_val_loss = float(
            record["val_loss"]
        )


        final_best_state = {

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
            final_short_model
            .state_dict()
            .items()
        }


    # ======================================================
    # REPORT
    # ======================================================

    print()
    print("=" * 70)

    print(
        f"FINAL SHORT EPOCH "
        f"{epoch+1}/"
        f"{FINAL_SHORT_CONFIG['epochs']}"
    )

    print("-" * 70)

    print(
        f"Detection Loss   : "
        f"{record['detection_loss']:.4f}"
    )

    print(
        f"CLS Loss         : "
        f"{record['cls_loss']:.4f}"
    )

    print(
        f"BBox Loss        : "
        f"{record['bbox_loss']:.4f}"
    )

    print(
        f"GIoU Loss        : "
        f"{record['giou_loss']:.4f}"
    )

    print(
        f"Contrastive Loss : "
        f"{record['contrastive_loss']:.6f}"
    )

    print(
        f"Combined Loss    : "
        f"{record['combined_loss']:.4f}"
    )

    print("-" * 70)

    print(
        f"Correct Sim      : "
        f"{record['correct_sim']:.6f}"
    )

    print(
        f"Wrong Sim        : "
        f"{record['wrong_sim']:.6f}"
    )

    print(
        f"Train Margin     : "
        f"{record['train_margin']:.6f}"
    )

    print(
        f"Required Margin  : "
        f"{FINAL_SHORT_CONFIG['contrastive_margin']:.6f}"
    )

    print("-" * 70)

    print(
        f"Val Loss         : "
        f"{record['val_loss']:.4f}"
    )

    print(
        f"Val mAP50        : "
        f"{record['map50']:.8f}"
    )

    print(
        f"Val P@0.50       : "
        f"{record['precision50']:.6f}"
    )

    print(
        f"Val R@0.50       : "
        f"{record['recall50']:.6f}"
    )

    print("=" * 70)


    final_short_scheduler.step()


# ==========================================================
# RESTORE BEST
# ==========================================================

assert final_best_state is not None


final_short_model.load_state_dict(
    final_best_state
)


final_short_model.to(
    CONFIG["device"]
)


final_short_model.eval()


# ==========================================================
# FINAL SUMMARY
# ==========================================================

print()
print("=" * 70)
print("FINAL REPAIR SHORT GENERALIZATION RESULT")
print("=" * 70)

print(
    f"Baseline mAP50 : "
    f"{final_baseline_metrics['episodic_map50']:.8f}"
)

print(
    f"Best Epoch     : "
    f"{final_best_epoch}"
)

print(
    f"Best Val Loss  : "
    f"{final_best_val_loss:.6f}"
)

print(
    f"Best mAP50     : "
    f"{final_best_map50:.8f}"
)

print(
    f"Margin         : "
    f"{FINAL_SHORT_CONFIG['contrastive_margin']:.3f}"
)

print(
    f"Weight         : "
    f"{FINAL_SHORT_CONFIG['contrastive_weight']:.3f}"
)

print("-" * 70)

print(
    "✓ final_short_model restored to BEST short epoch."
)

print(
    "✓ final_model remains untouched."
)

print("=" * 70)
print("STOP HERE AND SEND THE OUTPUT.")
print("=" * 70)
