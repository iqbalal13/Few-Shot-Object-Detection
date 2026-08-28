# ==========================================================
# V3 CHECKPOINT PATH
# KEEP V2 CHECKPOINT SAFE
# ==========================================================

import os


SAVE_DIR = (
    "/content/drive/MyDrive/"
    "META_DETR_CHECKPOINT_V3_PROTO_RELATION"
)

os.makedirs(
    SAVE_DIR,
    exist_ok=True
)


BEST_CHECKPOINT_PATH = os.path.join(
    SAVE_DIR,
    "meta_detr_base_v3_best.pth"
)

LATEST_CHECKPOINT_PATH = os.path.join(
    SAVE_DIR,
    "meta_detr_base_v3_latest.pth"
)

BASE_FINAL_PATH = os.path.join(
    SAVE_DIR,
    "meta_detr_base_v3_final.pth"
)


print("=" * 70)
print("V3 CHECKPOINT PATH")
print("=" * 70)

print(
    "Best   :",
    BEST_CHECKPOINT_PATH
)

print(
    "Latest :",
    LATEST_CHECKPOINT_PATH
)

print(
    "Final  :",
    BASE_FINAL_PATH
)

print("=" * 70)
