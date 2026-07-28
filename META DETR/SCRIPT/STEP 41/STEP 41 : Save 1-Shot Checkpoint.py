# ==========================================================
# STEP 41 : Save 1-Shot Checkpoint
# ==========================================================

import os
import torch

print("=" * 70)
print("STEP 41 : SAVE 1-SHOT CHECKPOINT")
print("=" * 70)

# ----------------------------------------------------------
# Save Directory
# ----------------------------------------------------------

SAVE_DIR = "/content/drive/MyDrive/META_DETR_CHECKPOINT"

os.makedirs(SAVE_DIR, exist_ok=True)

CHECKPOINT_NAME = "meta_detr_cctv_1shot.pth"

SAVE_PATH = os.path.join(
    SAVE_DIR,
    CHECKPOINT_NAME
)

# ----------------------------------------------------------
# Save Checkpoint
# ----------------------------------------------------------

checkpoint = {

    "experiment": "1-shot",

    "epoch": FS_EPOCHS,

    "model_state_dict": model.state_dict(),

    "optimizer_state_dict": optimizer.state_dict(),

    "num_shots": NUM_SHOTS,

    "num_classes": len(class_to_indices),

    "batch_size": QUERY_BATCH_SIZE,

    "learning_rate": optimizer.param_groups[0]["lr"],

    "seed": SEED

}

torch.save(

    checkpoint,

    SAVE_PATH

)

# ----------------------------------------------------------
# Information
# ----------------------------------------------------------

print(f"Checkpoint Saved : {SAVE_PATH}")

print(f"Experiment       : 1-Shot")

print(f"Epoch            : {FS_EPOCHS}")

print(f"Support Images   : {len(support_dataset)}")

print(f"Query Images     : {len(query_dataset)}")

print("=" * 70)
print("STEP 41 COMPLETED")
print("=" * 70)
