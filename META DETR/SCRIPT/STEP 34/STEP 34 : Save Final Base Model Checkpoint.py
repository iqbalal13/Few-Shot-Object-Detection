# ==========================================================
# STEP 34 : Save Final Base Model Checkpoint
# ==========================================================

import os
import torch

CHECKPOINT_DIR = "/content/checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

checkpoint_path = os.path.join(
    CHECKPOINT_DIR,
    "meta_detr_base_coco_final.pth"
)

torch.save({
    "epoch": NUM_EPOCHS,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "config": CONFIG
}, checkpoint_path)

print("=" * 60)
print("Checkpoint Saved Successfully")
print(f"Location : {checkpoint_path}")
print("=" * 60)
