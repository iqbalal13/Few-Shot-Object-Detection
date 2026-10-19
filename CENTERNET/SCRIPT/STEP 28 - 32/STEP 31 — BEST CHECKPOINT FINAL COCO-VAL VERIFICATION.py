# ============================================================
# STEP 31 — VERIFY SELECTED SOURCE CHECKPOINT
# NO TRAINING
# ============================================================

import os
import torch

assert os.path.isfile(BEST_CHECKPOINT)
assert os.path.isfile(LAST_CHECKPOINT)

last31 = torch.load(
    LAST_CHECKPOINT,
    map_location="cpu",
    weights_only=False
)

assert last31["source_base"] == "V1.2"
assert int(last31["epoch"]) == 20

assert (
    last31["manifest_sha256"]
    == source_manifest_sha256
)

checkpoint = torch.load(
    BEST_CHECKPOINT,
    map_location="cpu",
    weights_only=False
)

assert checkpoint["source_base"] == "V1.2"

assert (
    checkpoint["manifest_sha256"]
    == source_manifest_sha256
)

assert int(checkpoint["epoch"]) in (15, 20)

best_model = CenterNetResNet101(
    num_classes=80,
    pretrained_backbone=False
).to(device)

best_model.load_state_dict(
    checkpoint["model_state_dict"],
    strict=True
)

best_model.eval()

if "source_val_loader" not in globals():
    source_val_loader, _ = build_val_loader(
        SEED + 3301,
        encoded=True
    )

final_source_metrics = evaluate_centernet(
    best_model,
    source_val_loader,
    max_batches=None,
    score_threshold=CONFIG["score_threshold"],
    iou_threshold=CONFIG["iou_threshold"],
    compute_official_ap=False
)

print("STEP 31 — SOURCE VERIFICATION")
print("Checkpoint:", BEST_CHECKPOINT)
print("Checkpoint epoch:", checkpoint["epoch"])
print("Exposures:", checkpoint["epoch"] * 8000)

for key, value in final_source_metrics.items():
    if value is not None:
        print(key, ":", value)

print("STEP 31 PASSED — NO TRAINING")
