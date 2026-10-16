
# ============================================================
# STEP 33 — OFFICIAL COCO mAP / IoU EVALUATION
# ALL 80 CLASSES — BALANCED80K V1.3
# NO TRAINING — NO WEIGHT UPDATES
# ============================================================

import os
import json
import time
import math
import numpy as np
import torch

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

print("=" * 72)
print("STEP 33 — OFFICIAL COCO BBOX mAP / IoU")
print("=" * 72)

# ============================================================
# 1. CHECK CONFIGURATION AND CHECKPOINT
# ============================================================

assert CONFIG["experiment_name"] == (
    "CenterNet_COCO80_FSOD_BALANCED80K_V1_3_AUG_GAUSS_BS8"
)

assert CONFIG["num_classes"] == 80
assert len(val_dataset) == 5000
assert len(val_dataset.cat_ids) == 80
assert torch.cuda.is_available()
assert os.path.isfile(BEST_CHECKPOINT)

coco33_ckpt = torch.load(
    BEST_CHECKPOINT,
    map_location="cpu",
    weights_only=False
)

assert coco33_ckpt["experiment_name"] == (
    CONFIG["experiment_name"]
)

assert coco33_ckpt["source_images_total"] == 80000

if "source_manifest_sha256" in globals():
    assert (
        coco33_ckpt["source_manifest_sha256"]
        == source_manifest_sha256
    )

coco33_epoch = int(coco33_ckpt["epoch"])

# Always reload the saved best model
coco33_model = CenterNetResNet101(
    num_classes=80,
    pretrained_backbone=False
).to(device)

coco33_model.load_state_dict(
    coco33_ckpt["model_state_dict"],
    strict=True
)

coco33_model.eval()

del coco33_ckpt

# Use original COCO annotations, not letterboxed GT
coco33_gt = val_dataset.coco

coco33_label_to_catid = tuple(
    int(x) for x in val_dataset.cat_ids
)

coco33_image_ids = list(
    map(int, val_dataset.image_ids)
)

assert set(coco33_gt.getImgIds()) == set(
    coco33_image_ids
)

# ============================================================
# 2. EVALUATION CONFIGURATION
# ============================================================

COCO33_TOPK = 100

# Important: do not use confidence 0.50 for AP.
# AP evaluates the ranking across confidence scores.
COCO33_MIN_SCORE = 0.0

COCO33_SAVE_PREDICTIONS = True
