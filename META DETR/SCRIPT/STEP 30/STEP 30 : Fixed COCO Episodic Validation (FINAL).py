# ==========================================================
# STEP 30 : Fixed COCO Episodic Validation (FINAL)
# ==========================================================

import os
import numpy as np
import torch

from torch.utils.data import DataLoader
from tqdm import tqdm


VAL_IMAGE_DIR = os.path.join(

    COCO_ROOT,

    COCO_CONFIG[
        "val_images"
    ]
)


# ==========================================================
# Fixed Validation Dataset
# ==========================================================

val_dataset = COCOEpisodeDataset(

    coco=coco_val,

    image_dir=
        VAL_IMAGE_DIR,

    support_transform=
        support_transform,

    query_transform=
        query_transform,

    num_episodes=
        TRAIN_CONFIG[
            "validation_episodes"
        ],

    # Independent deterministic seed
    seed=
        CONFIG["seed"]
        +
        10000,

    min_bbox_size=
        COCO_CONFIG[
            "min_bbox_size"
        ]
)


# Always fixed
val_dataset.set_epoch(
    0
)


val_loader = DataLoader(

    dataset=
        val_dataset,

    batch_size=
        COCO_CONFIG[
            "batch_size"
        ],

    shuffle=False,

    num_workers=
        COCO_CONFIG[
            "num_workers"
        ],

    pin_memory=
        COCO_CONFIG[
            "pin_memory"
        ],

    collate_fn=
        episodic_collate_fn,

    drop_last=False
)


print("=" * 70)
print("COCO FIXED VALIDATION SET")
print("=" * 70)

print(
    "Validation Episodes :",
    len(val_dataset)
)

print(
    "COCO Classes        :",
    len(
        val_dataset.valid_labels
    )
)

print(
    "Episodes / Class    :",
    len(val_dataset)
    //
    len(
        val_dataset.valid_labels
    )
)

print("=" * 70)
