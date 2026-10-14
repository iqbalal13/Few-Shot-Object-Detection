# ============================================================
# STEP 4 — GLOBAL CONFIGURATION
# FAST-REPRO V1 — LOCKED
# ============================================================

import os

CONFIG = {

    # --------------------------------------------------------
    # Experiment identity
    # --------------------------------------------------------
    "experiment_name":
        "CenterNet_COCO80_FAST_REPRO_V1",

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------
    "backbone":
        "resnet101",

    "backbone_pretrained":
        True,

    "num_classes":
        80,

    # --------------------------------------------------------
    # Image
    # --------------------------------------------------------
    "input_size":
        640,

    "output_stride":
        4,

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------
    "batch_size":
        4,

    "num_workers":
        8,

    "persistent_workers":
        True,

    "prefetch_factor":
        4,

    "learning_rate":
        1e-4,

    "weight_decay":
        1e-4,

    # --------------------------------------------------------
    # Precision
    # --------------------------------------------------------
    "precision":
        "bfloat16",

    "grad_clip_norm":
        10.0,

    # --------------------------------------------------------
    # Primary detection thresholds
    # --------------------------------------------------------
    "score_threshold":
        0.50,

    "iou_threshold":
        0.50,

    # --------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------
    "seed":
        SEED,

    # --------------------------------------------------------
    # Dataset root
    # --------------------------------------------------------
    "data_root":
        "/content/coco",

    # --------------------------------------------------------
    # Local checkpoints
    # --------------------------------------------------------
    "checkpoint_root":
        "/content/checkpoints_centernet",
}


# ============================================================
# DIRECTORIES
# ============================================================

os.makedirs(
    CONFIG["data_root"],
    exist_ok=True
)

os.makedirs(
    CONFIG["checkpoint_root"],
    exist_ok=True
)


CONFIG["train_images"] = os.path.join(
    CONFIG["data_root"],
    "train2017"
)


CONFIG["val_images"] = os.path.join(
    CONFIG["data_root"],
    "val2017"
)


CONFIG["train_annotations"] = os.path.join(
    CONFIG["data_root"],
    "annotations",
    "instances_train2017.json"
)


CONFIG["val_annotations"] = os.path.join(
    CONFIG["data_root"],
    "annotations",
    "instances_val2017.json"
)


# ============================================================
# PRINT CONFIG
# ============================================================

print("=" * 70)
print("CENTERNET GLOBAL CONFIG — FAST-REPRO V1")
print("=" * 70)

for key, value in CONFIG.items():

    print(
        f"{key:24s}: {value}"
    )


# ============================================================
# LOCKED CONFIG SANITY
# ============================================================

assert CONFIG["backbone"] == "resnet101"
assert CONFIG["num_classes"] == 80

assert CONFIG["input_size"] == 640
assert CONFIG["output_stride"] == 4

assert CONFIG["batch_size"] == 4
assert CONFIG["num_workers"] == 8

assert CONFIG["learning_rate"] == 1e-4
assert CONFIG["weight_decay"] == 1e-4

assert CONFIG["precision"] == "bfloat16"
assert CONFIG["grad_clip_norm"] == 10.0

assert CONFIG["score_threshold"] == 0.50
assert CONFIG["iou_threshold"] == 0.50

assert CONFIG["seed"] == 42


print("\nFAST-REPRO V1 CONFIG LOCKED")
print("STEP 4 PASSED")
