# ==========================================================
# VERIFY & LOAD COCO BASE CHECKPOINT
# ==========================================================

import os
import torch

CHECKPOINT_PATH = "/content/meta_detr_base_coco_final.pth"

print("=" * 60)
print("VERIFYING CHECKPOINT")
print("=" * 60)

# ----------------------------------------------------------
# 1. Check file existence
# ----------------------------------------------------------
assert os.path.exists(CHECKPOINT_PATH), \
    f"Checkpoint not found:\n{CHECKPOINT_PATH}"

print(f"✓ File Found")
print(f"Location : {CHECKPOINT_PATH}")
print(f"Size     : {os.path.getsize(CHECKPOINT_PATH)/1024/1024:.2f} MB")

# ----------------------------------------------------------
# 2. Load checkpoint
# ----------------------------------------------------------
checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=CONFIG["device"]
)

print("\n✓ Checkpoint Loaded")

# ----------------------------------------------------------
# 3. Display checkpoint information
# ----------------------------------------------------------
print("\nCheckpoint Keys :")
print(list(checkpoint.keys()))

print(f"\nSaved Epoch : {checkpoint['epoch']}")

# ----------------------------------------------------------
# 4. Restore model & optimizer
# ----------------------------------------------------------
model.load_state_dict(checkpoint["model_state_dict"])
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

print("\n✓ Model State Loaded")
print("✓ Optimizer State Loaded")

# ----------------------------------------------------------
# 5. Display config (optional)
# ----------------------------------------------------------
if "config" in checkpoint:
    print("\nConfiguration Found")
    print(checkpoint["config"])

print("=" * 60)
print("COCO BASE CHECKPOINT VERIFIED SUCCESSFULLY")
print("=" * 60)
