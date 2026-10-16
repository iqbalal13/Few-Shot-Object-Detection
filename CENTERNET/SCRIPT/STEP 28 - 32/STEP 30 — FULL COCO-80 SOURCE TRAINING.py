
# ============================================================
# STEP 30 — CENTERNET-FSOD BALANCED80K V1.3 AUG+GAUSS
# FRESH SOURCE TRAINING: 10 epochs x 8000 images
# 100 selections/class/epoch, batch=8
# 1000 updates/epoch = 10000 updates total
# ============================================================

import os
import gc
import json
import math
import time
import random
import hashlib
from collections import Counter

import numpy as np
import torch
from torch.utils.data import Sampler
from google.colab import drive

# ============================================================
# 1. LOCKED PROTOCOL CHECKS
# ============================================================

assert torch.cuda.is_available(), "Select GPU in Colab Runtime."
assert torch.cuda.is_bf16_supported(), "BF16 GPU required."

assert CONFIG["experiment_name"] == (
    "CenterNet_COCO80_FSOD_BALANCED80K_V1_3_AUG_GAUSS_BS8"
)

assert CONFIG["source_epochs"] == 10
assert CONFIG["source_images_per_epoch"] == 8000
assert CONFIG["source_images_total"] == 80000
assert CONFIG["source_images_per_class_per_epoch"] == 100
assert CONFIG["num_classes"] == 80
assert CONFIG["batch_size"] == 8
assert CONFIG["max_objects"] == 128

assert len(train_dataset) == 118287
assert len(val_dataset) == 5000

EPOCHS = CONFIG["source_epochs"]
IMAGES_PER_EPOCH = CONFIG["source_images_per_epoch"]

STEPS_PER_EPOCH = (
    IMAGES_PER_EPOCH // CONFIG["batch_size"]
)

SCHEDULER_TMAX = 25  # held constant to compare against V1.2
MIN_GEOMETRY_DELTA = 1e-4

MANIFEST_SEED = (
    int(SEED)
    + int(CONFIG["source_manifest_seed_offset"])
)

# ============================================================
# 2. INDEPENDENT 80K EXPERIMENT DIRECTORY
# ============================================================

drive.mount(
    "/content/drive",
    force_remount=False
)

FULL_CHECKPOINT_DIR = (
    "/content/drive/MyDrive/"
    "CenterNet_COCO80_Source_BALANCED80K_V1_3_AUG_GAUSS_BS8"
)

os.makedirs(
    FULL_CHECKPOINT_DIR,
    exist_ok=True
)

BEST_CHECKPOINT = os.path.join(
    FULL_CHECKPOINT_DIR,
    "centernet_coco80_best.pth"
)

LAST_CHECKPOINT = os.path.join(
    FULL_CHECKPOINT_DIR,
    "centernet_coco80_last.pth"
)

MANIFEST_FILE = os.path.join(
    FULL_CHECKPOINT_DIR,
    "balanced80k_v13_manifest.jsonl"
)

HISTORY_FILE = os.path.join(
    FULL_CHECKPOINT_DIR,
    "training_history.json"
