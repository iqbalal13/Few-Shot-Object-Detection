# ==========================================================
# STEP 36 : Mount Google Drive & Load CCTV Dataset
# ==========================================================

import os
from google.colab import drive

# ----------------------------------------------------------
# Mount Google Drive
# ----------------------------------------------------------
try:
    drive.mount('/content/drive', force_remount=True)
except Exception as e:
    print("Mount gagal!")
    print(e)
    raise

# ----------------------------------------------------------
# Dataset Path
# ----------------------------------------------------------
DATASET_ROOT = "/content/drive/MyDrive/DATASET/ORIGINAL DATASET"

TRAIN_DIR = os.path.join(DATASET_ROOT, "TRAIN")
VAL_DIR   = os.path.join(DATASET_ROOT, "VALIDATION")
TEST_DIR  = os.path.join(DATASET_ROOT, "TEST")

# ----------------------------------------------------------
# Verify Dataset
# ----------------------------------------------------------
print("=" * 60)
print("VERIFY CCTV DATASET")
print("=" * 60)

for name, path in [
    ("TRAIN", TRAIN_DIR),
    ("VALIDATION", VAL_DIR),
    ("TEST", TEST_DIR)
]:
    print(f"{name}: {path}")
    print("Exists :", os.path.exists(path))

print("=" * 60)
