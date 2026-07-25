# ==========================================================
# STEP 32 : Save Base Model Checkpoint
# ==========================================================

import os
import torch

SAVE_DIR = "/content/checkpoints"
os.makedirs(SAVE_DIR, exist_ok=True)

CHECKPOINT_PATH = os.path.join(
    SAVE_DIR,
    "meta_detr_base_coco.pth"
)

torch.save({
    "epoch": NUM_EPOCHS,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
}, CHECKPOINT_PATH)

print("="*60)
print("Checkpoint saved!")
print(CHECKPOINT_PATH)
print("="*60)
