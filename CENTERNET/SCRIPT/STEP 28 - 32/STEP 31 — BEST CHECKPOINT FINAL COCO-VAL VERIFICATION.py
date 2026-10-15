
# ============================================================
# STEP 31 — VERIFY AND EVALUATE 80K BEST CHECKPOINT
# No training or weight updates are performed here.
# ============================================================

import os
import torch

print("=" * 70)
print("STEP 31 — BALANCED80K FINAL SOURCE VERIFICATION")
print("=" * 70)

assert os.path.isfile(BEST_CHECKPOINT), (
    "Best checkpoint Step 30 tidak ada."
)

assert os.path.isfile(LAST_CHECKPOINT), (
    "Last checkpoint Step 30 tidak ada."
)

# Verify all ten training epochs completed
last_ckpt = torch.load(
    LAST_CHECKPOINT,
    map_location="cpu",
    weights_only=False
)

assert last_ckpt.get(
    "experiment_name"
) == CONFIG["experiment_name"]

assert last_ckpt.get(
    "source_manifest_sha256"
) == source_manifest_sha256

assert last_ckpt.get(
    "source_images_total"
) == 80000

assert int(
    last_ckpt.get("epoch", -1)
) == 10, "Training belum selesai 10 epoch."

assert len(
    last_ckpt.get("history", [])
) == 10

del last_ckpt


# Load best checkpoint
checkpoint = torch.load(
    BEST_CHECKPOINT,
    map_location="cpu",
    weights_only=False
)

assert checkpoint.get(
    "experiment_name"
) == CONFIG["experiment_name"]

assert checkpoint.get(
    "source_manifest_sha256"
) == source_manifest_sha256

assert checkpoint.get(
    "source_images_total"
) == 80000

assert 1 <= int(
    checkpoint["epoch"]
) <= 10

assert (
    checkpoint.get(
        "actual_images_seen_at_checkpoint"
    )
    == int(checkpoint["epoch"]) * 8000
)


# Build model without pretrained download
best_model = CenterNetResNet101(
    num_classes=CONFIG["num_classes"],
    pretrained_backbone=False
).to(device)

best_model.load_state_dict(
    checkpoint["model_state_dict"],
    strict=True
)

best_model.eval()


print(
    "Total completed epochs       : 10"
)

print(
    "Best checkpoint epoch       :",
    checkpoint["epoch"]
)

print(
    "Exposures at best checkpoint:",
    checkpoint["actual_images_seen_at_checkpoint"]
)

print(
    "Full training budget         : 80,000 exposures"
)


# Full validation
final_source_metrics = evaluate_centernet(
    best_model,
    source_val_loader,
    max_batches=None,
    score_threshold=CONFIG["score_threshold"],
    iou_threshold=CONFIG["iou_threshold"]
)


print("\nFULL COCO-VAL, best checkpoint")

for name in (
    "loss",
    "geometry50",
    "precision",
    "recall",
    "tp",
    "fp",
    "fn",
    "gt"
):
    print(
        f"{name:12s}: {final_source_metrics[name]}"
    )


print("\nCheckpoint identity OK.")
print("No weights were updated.")
print("STEP 31 PASSED")
