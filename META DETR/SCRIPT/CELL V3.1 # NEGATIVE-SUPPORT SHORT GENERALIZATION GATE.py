# ==========================================================
# CELL V3.1
# NEGATIVE-SUPPORT SHORT GENERALIZATION GATE
#
# PURPOSE:
# Force the model to distinguish:
#
#   correct support + query
#       -> detect matching objects
#
#   absent wrong-class support + SAME query
#       -> ALL queries must be background
#
# IMPORTANT:
# - Architecture V3 is NOT changed.
# - Starts from fresh official V3 `model`.
# - Does NOT use short_model.
# - Official V3 `model` remains untouched.
# - Wrong support class is guaranteed absent
#   from the query image according to COCO annotations.
# - Best V3.1 state is restored at the end.
# ==========================================================

import copy
import random
import numpy as np
import torch

from tqdm.auto import tqdm


print("=" * 70)
print("V3.1 NEGATIVE-SUPPORT SHORT GENERALIZATION GATE")
print("=" * 70)


# ==========================================================
# CONFIG
# ==========================================================

V31_CONFIG = {

    "epochs":
        5,

    "steps_per_epoch":
        800,

    # Negative branch uses only classification/background loss.
    #
    # Focal loss for easy all-background predictions is naturally
    # small, so give this branch explicit weight.
    #
    # This is a SHORT DIAGNOSTIC value, not yet final thesis HP.
    "negative_support_weight":
        5.0
}


# ==========================================================
# SAFETY / PROTOCOL ASSERTIONS
# ==========================================================

assert (
    CONFIG["support_conditioning"]
    ==
    "prototype_relation"
), (
    "Current model is not configured as V3 "
    "prototype_relation."
)


assert (
    COCO_CONFIG["batch_size"]
    ==
    1
), (
    "This V3.1 gate currently expects batch_size=1."
)


assert (
    COCO_CONFIG["shuffle"]
    is False
), (
    "This V3.1 gate expects deterministic shuffle=False."
)


# ==========================================================
# FRESH COPY OF OFFICIAL V3
#
# DO NOT start from short_model.
# We want an apples-to-apples comparison with V3.
# ==========================================================

v31_model = copy.deepcopy(
    model
).to(
    CONFIG["device"]
)


v31_optimizer, v31_scheduler = (
    build_optimizer_and_scheduler(
        v31_model
    )
)


# ==========================================================
# HELPER:
# Classes actually present in a COCO query image
# ==========================================================

def v31_get_present_classes(
    dataset,
    image_id
):

    annotation_ids = (
        dataset.coco.getAnnIds(
            imgIds=[
                int(image_id)
            ]
        )
    )


    annotations = (
        dataset.coco.loadAnns(
            annotation_ids
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

                dataset.cat2label[
                    category_id
                ]
            )


    return present_labels


# ==========================================================
# HELPER:
# Build wrong support whose semantic class is ABSENT
# from the current query image.
# ==========================================================

def v31_make_absent_wrong_support(
    dataset,
    query_image_id,
    current_class,
    epoch,
    step
):

    present_labels = (
        v31_get_present_classes(
            dataset,
            query_image_id
        )
    )


    # Current episodic class must obviously be considered present.
    present_labels.add(
        int(current_class)
    )


    candidate_labels = [

        label

        for label
        in dataset.valid_labels

        if label
        not in present_labels
    ]


    if len(candidate_labels) == 0:

        raise RuntimeError(
            "No absent wrong-support class "
            "available for this query image."
        )


    # Deterministic choice
    rng = random.Random(

        CONFIG["seed"]
        +
        1_000_000
        +
        epoch * 100_000
        +
        step
    )


    wrong_label = rng.choice(
        candidate_labels
    )


    support_ann_id = rng.choice(

        dataset.class_to_ann_ids[
            wrong_label
        ]
    )


    wrong_support_image, _ = (
        dataset._load_support(

            support_ann_id,

            wrong_label
        )
    )


    return (
        wrong_support_image,
        int(wrong_label),
        present_labels
    )


# ==========================================================
# HELPER:
# Empty targets for WRONG-support branch.
#
# This is valid with the existing matcher/criterion:
# no matching object relative to an absent support class.
# ==========================================================

def v31_make_empty_targets(
    batch_size,
    device
):

    empty_targets = []


    for _ in range(
        batch_size
    ):

        empty_targets.append({

            "boxes":
                torch.empty(
                    (0, 4),
                    dtype=torch.float32,
                    device=device
                ),

            "labels":
                torch.empty(
                    (0,),
                    dtype=torch.long,
                    device=device
                )
        })


    return empty_targets


# ==========================================================
# BASELINE
# ==========================================================

print()
print("Running fresh V3 baseline on fixed COCO-Val...")
print()


v31_baseline_metrics = (
    evaluate_episodic_model(

        model=
            v31_model,

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
print("V3.1 BASELINE")
print("-" * 70)

print(
    f"Val Loss : "
    f"{v31_baseline_metrics['loss']:.6f}"
)

print(
    f"mAP50    : "
    f"{v31_baseline_metrics['episodic_map50']:.8f}"
)

print(
    f"P@0.50   : "
    f"{v31_baseline_metrics['precision50']:.6f}"
)

print(
    f"R@0.50   : "
    f"{v31_baseline_metrics['recall50']:.6f}"
)

print("-" * 70)


# ==========================================================
# HISTORY / BEST STATE
# ==========================================================

v31_history = []

v31_best_state = None

v31_best_epoch = -1

v31_best_map50 = -1.0

v31_best_val_loss = float(
    "inf"
)


# ==========================================================
# TRAINING
# ==========================================================

for v31_epoch in range(
    V31_CONFIG["epochs"]
):

    # New deterministic episodic samples each epoch
    train_dataset.set_epoch(
        v31_epoch
    )


    v31_model.train()


    freeze_backbone_bn_statistics(
        v31_model.backbone
    )


    running_positive_loss = 0.0

    running_negative_loss = 0.0

    running_total_loss = 0.0

    running_positive_cls = 0.0

    running_bbox = 0.0

    running_giou = 0.0

    running_wrong_max_score = 0.0

    running_correct_max_score = 0.0

    actual_steps = 0


    progress_bar = tqdm(

        train_loader,

        desc=(
            f"V3.1 "
            f"[{v31_epoch + 1}/"
            f"{V31_CONFIG['epochs']}]"
        )
    )


    for batch in progress_bar:

        if (
            actual_steps
            >=
            V31_CONFIG[
                "steps_per_epoch"
            ]
        ):

            break


        # ==================================================
        # CURRENT POSITIVE EPISODE
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

                CONFIG["device"]
            )
        )


        current_class = int(
            episode_classes[
                0
            ].item()
        )


        query_image_id = int(

            batch[
                "query_targets"
            ][0][
                "image_id"
            ].item()
        )


        # ==================================================
        # BUILD ABSENT WRONG SUPPORT
        # ==================================================

        (
            wrong_support_image,
            wrong_class,
            present_labels

        ) = v31_make_absent_wrong_support(

            dataset=
                train_dataset,

            query_image_id=
                query_image_id,

            current_class=
                current_class,

            epoch=
                v31_epoch,

            step=
                actual_steps
        )


        # Critical safety assertion:
        # wrong semantic class MUST NOT exist in query.
        assert (
            wrong_class
            not in
            present_labels
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
        # POSITIVE FORWARD
        #
        # correct support + query
        # ==================================================

        positive_outputs = (
            v31_model(

                support_images,

                query_images
            )
        )


        positive_loss_dict = (
            criterion(

                positive_outputs,

                targets
            )
        )


        positive_loss = (
            positive_loss_dict[
                "loss_total"
            ]
        )


        # ==================================================
        # NEGATIVE FORWARD
        #
        # absent wrong-class support + SAME query
        #
        # Expected:
        # ALL detector queries = background.
        # ==================================================

        negative_outputs = (
            v31_model(

                wrong_support_images,

                query_images
            )
        )


        negative_targets = (
            v31_make_empty_targets(

                batch_size=
                    query_images.shape[0],

                device=
                    CONFIG["device"]
            )
        )


        negative_loss_dict = (
            criterion(

                negative_outputs,

                negative_targets
            )
        )


        # Only classification pressure matters.
        # bbox / GIoU are zero for empty targets.
        negative_support_loss = (

            negative_loss_dict[
                "loss_cls"
            ]
        )


        # ==================================================
        # COMBINED OBJECTIVE
        # ==================================================

        total_loss = (

            positive_loss

            +

            V31_CONFIG[
                "negative_support_weight"
            ]
            *
            negative_support_loss
        )


        if not torch.isfinite(
            total_loss
        ):

            raise RuntimeError(
                "V3.1 loss became NaN/Inf."
            )


        # ==================================================
        # BACKPROP
        # ==================================================

        v31_optimizer.zero_grad(
            set_to_none=True
        )


        total_loss.backward()


        torch.nn.utils.clip_grad_norm_(

            v31_model.parameters(),

            max_norm=
                TRAIN_CONFIG[
                    "gradient_clip"
                ]
        )


        v31_optimizer.step()


        # ==================================================
        # MONITOR SUPPORT BEHAVIOR
        # ==================================================

        with torch.no_grad():

            correct_max_score = (
                torch.sigmoid(

                    positive_outputs[
                        "pred_logits"
                    ]

                )
                .max()
                .item()
            )


            wrong_max_score = (
                torch.sigmoid(

                    negative_outputs[
                        "pred_logits"
                    ]

                )
                .max()
                .item()
            )


        # ==================================================
        # STATISTICS
        # ==================================================

        running_positive_loss += (
            positive_loss.item()
        )


        running_negative_loss += (
            negative_support_loss.item()
        )


        running_total_loss += (
            total_loss.item()
        )


        running_positive_cls += (

            positive_loss_dict[
                "loss_cls"
            ].item()
        )


        running_bbox += (

            positive_loss_dict[
                "loss_bbox"
            ].item()
        )


        running_giou += (

            positive_loss_dict[
                "loss_giou"
            ].item()
        )


        running_correct_max_score += (
            correct_max_score
        )


        running_wrong_max_score += (
            wrong_max_score
        )


        actual_steps += 1


        progress_bar.set_postfix({

            "Pos":
                f"{positive_loss.item():.3f}",

            "Neg":
                f"{negative_support_loss.item():.3f}",

            "WrongP":
                f"{wrong_max_score:.3f}",

            "Step":
                (
                    f"{actual_steps}/"
                    f"{V31_CONFIG['steps_per_epoch']}"
                )
        })


    # ======================================================
    # FIXED COCO-VAL
    #
    # Normal correct-support evaluation.
    # ======================================================

    print()
    print(
        "Running fixed COCO-Val..."
    )


    v31_val_metrics = (
        evaluate_episodic_model(

            model=
                v31_model,

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


    record = {

        "epoch":
            v31_epoch + 1,

        "positive_loss":
            running_positive_loss
            /
            actual_steps,

        "negative_loss":
            running_negative_loss
            /
            actual_steps,

        "total_loss":
            running_total_loss
            /
            actual_steps,

        "positive_cls":
            running_positive_cls
            /
            actual_steps,

        "bbox":
            running_bbox
            /
            actual_steps,

        "giou":
            running_giou
            /
            actual_steps,

        "correct_max_score":
            running_correct_max_score
            /
            actual_steps,

        "wrong_max_score":
            running_wrong_max_score
            /
            actual_steps,

        "val_loss":
            v31_val_metrics[
                "loss"
            ],

        "map50":
            v31_val_metrics[
                "episodic_map50"
            ],

        "precision50":
            v31_val_metrics[
                "precision50"
            ],

        "recall50":
            v31_val_metrics[
                "recall50"
            ]
    }


    v31_history.append(
        record
    )


    # ======================================================
    # BEST MODEL
    # ======================================================

    is_better = (

        record["map50"]
        >
        v31_best_map50

        or

        (
            abs(
                record["map50"]
                -
                v31_best_map50
            )
            <
            1e-12

            and

            record["val_loss"]
            <
            v31_best_val_loss
        )
    )


    if is_better:

        v31_best_epoch = (
            v31_epoch + 1
        )

        v31_best_map50 = (
            record["map50"]
        )

        v31_best_val_loss = (
            record["val_loss"]
        )


        # Store best weights on CPU,
        # avoiding another full GPU copy.
        v31_best_state = {

            name:
                tensor.detach()
                .cpu()
                .clone()

            for name, tensor
            in v31_model
            .state_dict()
            .items()
        }


    # ======================================================
    # REPORT
    # ======================================================

    print()
    print("=" * 70)

    print(
        f"V3.1 SHORT EPOCH "
        f"{v31_epoch + 1}/"
        f"{V31_CONFIG['epochs']}"
    )

    print("-" * 70)

    print(
        f"Positive Loss    : "
        f"{record['positive_loss']:.4f}"
    )

    print(
        f"Negative Loss    : "
        f"{record['negative_loss']:.6f}"
    )

    print(
        f"Combined Loss    : "
        f"{record['total_loss']:.4f}"
    )

    print(
        f"Positive CLS     : "
        f"{record['positive_cls']:.4f}"
    )

    print(
        f"BBox             : "
        f"{record['bbox']:.4f}"
    )

    print(
        f"GIoU             : "
        f"{record['giou']:.4f}"
    )

    print("-" * 70)

    print(
        f"Train Correct MaxP : "
        f"{record['correct_max_score']:.6f}"
    )

    print(
        f"Train Wrong MaxP   : "
        f"{record['wrong_max_score']:.6f}"
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


    v31_scheduler.step()


# ==========================================================
# RESTORE BEST V3.1 STATE
# ==========================================================

assert (
    v31_best_state
    is not None
)


v31_model.load_state_dict(
    v31_best_state
)


v31_model.to(
    CONFIG["device"]
)


v31_model.eval()


# ==========================================================
# FINAL SUMMARY
# ==========================================================

print()
print("=" * 70)
print("V3.1 SHORT GENERALIZATION FINAL")
print("=" * 70)

print(
    f"Baseline mAP50       : "
    f"{v31_baseline_metrics['episodic_map50']:.8f}"
)

print(
    f"Best Epoch           : "
    f"{v31_best_epoch}"
)

print(
    f"Best Val Loss        : "
    f"{v31_best_val_loss:.6f}"
)

print(
    f"Best mAP50           : "
    f"{v31_best_map50:.8f}"
)

print(
    f"Negative Weight      : "
    f"{V31_CONFIG['negative_support_weight']:.2f}"
)

print("-" * 70)

print(
    "✓ v31_model restored to BEST short-training epoch."
)

print(
    "✓ Official V3 `model` remains untouched."
)

print(
    "STOP HERE. DO NOT RUN STEP 32."
)

print("=" * 70)
