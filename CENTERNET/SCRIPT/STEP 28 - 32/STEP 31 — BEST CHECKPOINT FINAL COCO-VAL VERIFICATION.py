# ============================================================
# STEP 31 — BEST CHECKPOINT FINAL COCO-VAL VERIFICATION
# ============================================================

import os
import torch

print("=" * 70)
print("STEP 31 — FINAL SOURCE VALIDATION")
print("=" * 70)


assert os.path.exists(
    BEST_CHECKPOINT
), (
    "Best checkpoint tidak ditemukan. "
    "Step 30 harus selesai terlebih dahulu."
)


# ------------------------------------------------------------
# Build architecture WITHOUT downloading pretrained weights
# because checkpoint will replace all parameters.
# ------------------------------------------------------------

best_model = CenterNetResNet101(
    num_classes=CONFIG["num_classes"],
    pretrained_backbone=False
).to(device)


checkpoint = torch.load(
    BEST_CHECKPOINT,
    map_location=device,
    weights_only=False
)


best_model.load_state_dict(
    checkpoint[
        "model_state_dict"
    ]
)

best_model.eval()


print(
    "Loaded checkpoint epoch :",
    checkpoint[
        "epoch"
    ]
)

print(
    "Running FULL COCO-Val..."
)


final_source_metrics = \
    evaluate_centernet(
        best_model,
        val_loader,
        max_batches=None,
        score_threshold=CONFIG[
            "score_threshold"
        ],
        iou_threshold=CONFIG[
            "iou_threshold"
        ]
    )


print(
    "\n"
    + "=" * 70
)

print(
    "FINAL COCO-80 SOURCE RESULT"
)

print(
    "=" * 70
)


print(
    f"Validation loss : "
    f"{final_source_metrics['loss']:.4f}"
)

print(
    f"Geometry50      : "
    f"{final_source_metrics['geometry50']:.4f}"
)

print(
    f"Precision@0.50  : "
    f"{final_source_metrics['precision']:.4f}"
)

print(
    f"Recall@0.50     : "
    f"{final_source_metrics['recall']:.4f}"
)

print(
    "TP             :",
    final_source_metrics[
        "tp"
    ]
)

print(
    "FP             :",
    final_source_metrics[
        "fp"
    ]
)

print(
    "FN             :",
    final_source_metrics[
        "fn"
    ]
)


print(
    "\nCheckpoint verification PASSED"
)

print(
    "Weights were NOT updated."
)

print(
    "\nSTEP 31 PASSED"
)
