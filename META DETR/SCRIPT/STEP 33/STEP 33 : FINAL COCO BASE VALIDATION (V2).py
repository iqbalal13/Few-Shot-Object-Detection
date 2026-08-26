# ==========================================================
# STEP 33 : FINAL COCO BASE VALIDATION (V2)
# ==========================================================

import os
import torch


assert os.path.exists(
    BEST_CHECKPOINT_PATH
), (
    "Best checkpoint "
    "was not found."
)


print("=" * 70)
print(
    "STEP 33 : LOAD BEST "
    "COCO BASE MODEL"
)
print("=" * 70)


checkpoint = torch.load(

    BEST_CHECKPOINT_PATH,

    map_location=
        CONFIG["device"],

    weights_only=
        False
)


model.load_state_dict(

    checkpoint[
        "model_state_dict"
    ]
)


print(
    "Best Epoch :",
    checkpoint[
        "epoch"
    ]
)


print(
    "Stored mAP50:",
    checkpoint[
        "validation_metrics"
    ][
        "episodic_map50"
    ]
)


print(
    "\nRunning final fixed validation..."
)


final_val_metrics = (
    evaluate_episodic_model(

        model=
            model,

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


print(
    "\n"
    +
    "=" * 70
)


print(
    "FINAL COCO BASE "
    "VALIDATION RESULT"
)


print("=" * 70)


print(
    f"Validation Loss  : "
    f"{final_val_metrics['loss']:.6f}"
)


print(
    f"Classification   : "
    f"{final_val_metrics['loss_cls']:.6f}"
)


print(
    f"BBox Loss        : "
    f"{final_val_metrics['loss_bbox']:.6f}"
)


print(
    f"GIoU Loss        : "
    f"{final_val_metrics['loss_giou']:.6f}"
)


print("-" * 70)


print(
    f"Episodic mAP50   : "
    f"{final_val_metrics['episodic_map50']:.6f}"
)


print(
    f"Precision @ 0.50 : "
    f"{final_val_metrics['precision50']:.6f}"
)


print(
    f"Recall @ 0.50    : "
    f"{final_val_metrics['recall50']:.6f}"
)


print("=" * 70)
