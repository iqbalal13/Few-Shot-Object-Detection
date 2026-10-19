
# ============================================================
# STEP 4 — EXPERIMENT B1 CONFIGURATION
# GAUSSIAN RADIUS V1.3 / BASELINE MASKED L1 / FRESH 160K
# ============================================================

import os

CONFIG = {
    # --------------------------------------------------------
    # 1. EXPERIMENT IDENTITY
    # --------------------------------------------------------
    "experiment_name": (
        "CenterNet_COCO80_V13_B1_MASKEDL1_FRESH_160K_BS8"
    ),

    # --------------------------------------------------------
    # 2. MODEL ARCHITECTURE — UNCHANGED
    # --------------------------------------------------------
    "backbone": "resnet101",
    "backbone_pretrained": True,
    "num_classes": 80,
    "input_size": 640,
    "output_stride": 4,

    # --------------------------------------------------------
    # 3. DATALOADER — UNCHANGED
    # --------------------------------------------------------
    "batch_size": 8,
    "num_workers": 8,
    "persistent_workers": True,
    "prefetch_factor": 4,

    # --------------------------------------------------------
    # 4. OPTIMIZER — UNCHANGED
    # --------------------------------------------------------
    "learning_rate": 1e-4,
    "weight_decay": 1e-4,
    "grad_clip_norm": 10.0,
    "precision": "bfloat16",

    # --------------------------------------------------------
    # 5. SOURCE TRAINING — UNCHANGED
    # --------------------------------------------------------
    "source_epochs": 20,
    "source_images_total": 160000,
    "source_images_per_epoch": 8000,
    "source_images_per_class_per_epoch": 100,
    "source_sampling_policy": "class-balanced-multiclass",
    "source_manifest_seed_offset": 3000,

    # --------------------------------------------------------
    # 6. EXPERIMENT B1 — ONLY GAUSSIAN RADIUS CHANGES
    # --------------------------------------------------------
    "train_horizontal_flip_p": 0.0,

    "gaussian_radius_mode": "v1_3_corrected",
    "gaussian_min_overlap": 0.7,

    "best_checkpoint_metric": "official_COCO_mAP_50_95",

    # --------------------------------------------------------
    # 7. WH LOSS — ORIGINAL BASELINE
    # --------------------------------------------------------
    "wh_loss_type": "masked_l1",

    # Retained for compatibility with the existing loss API.
    # These parameters are INACTIVE when wh_loss_type
    # is "masked_l1".
    "wh_absolute_fraction": 0.75,
    "wh_relative_fraction": 0.25,
    "wh_relative_eps": 2.0,
    "wh_relative_beta": 0.20,
    "wh_relative_scale": 10.0,

    # --------------------------------------------------------
    # 8. ENGINEERING — UNCHANGED
    # --------------------------------------------------------
    "encoded_num_workers": 4,
    "encoded_prefetch_factor": 2,
    "use_fused_adamw": False,
    "use_channels_last": False,

    "run_engineering_sanity": False,
    "run_tiny_overfit_diagnostic": False,
    "debug_finite_checks": False,
    "progress_every": 100,

    # --------------------------------------------------------
    # 9. EVALUATION — UNCHANGED
    # --------------------------------------------------------
    "score_threshold": 0.50,
    "iou_threshold": 0.50,
    "seed": SEED,

    # --------------------------------------------------------
    # 10. PATHS
    # --------------------------------------------------------
    "data_root": "/content/coco",
    "checkpoint_root": "/content/checkpoints_centernet",
}


# ============================================================
# DATASET PATHS
# ============================================================

CONFIG["train_images"] = os.path.join(
    CONFIG["data_root"], "train2017"
)

CONFIG["val_images"] = os.path.join(
    CONFIG["data_root"], "val2017"
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


os.makedirs(
    CONFIG["data_root"],
    exist_ok=True
)

os.makedirs(
    CONFIG["checkpoint_root"],
    exist_ok=True
)


# ============================================================
# CONFIGURATION VALIDATION
# ============================================================

# Architecture
assert CONFIG["backbone"] == "resnet101"
assert CONFIG["num_classes"] == 80
assert CONFIG["input_size"] == 640
assert CONFIG["output_stride"] == 4
assert CONFIG["batch_size"] == 8

# Training budget
assert CONFIG["source_epochs"] == 20
assert CONFIG["source_images_total"] == 160000
assert CONFIG["source_images_per_epoch"] == 8000
assert CONFIG["source_images_per_class_per_epoch"] == 100

assert (
    CONFIG["source_epochs"]
    * CONFIG["source_images_per_epoch"]
    == CONFIG["source_images_total"]
)

assert (
    CONFIG["source_images_per_class_per_epoch"]
    * CONFIG["num_classes"]
    == CONFIG["source_images_per_epoch"]
)

# Source sampling and augmentation
assert (
    CONFIG["source_sampling_policy"]
    == "class-balanced-multiclass"
)

assert CONFIG["source_manifest_seed_offset"] == 3000
assert CONFIG["train_horizontal_flip_p"] == 0.0

# B1 experiment integrity
assert CONFIG["gaussian_radius_mode"] == "v1_3_corrected"
assert CONFIG["gaussian_min_overlap"] == 0.7

# WH must remain the original baseline loss.
assert CONFIG["wh_loss_type"] == "masked_l1"

# Retained inactive parameters
assert CONFIG["wh_relative_eps"] > 0
assert CONFIG["wh_relative_beta"] > 0
assert CONFIG["wh_relative_scale"] > 0

# Engineering flags
assert CONFIG["run_tiny_overfit_diagnostic"] is False
assert CONFIG["run_engineering_sanity"] is False

# Experiment identity
assert "B1" in CONFIG["experiment_name"]
assert "V13" in CONFIG["experiment_name"]
assert "MASKEDL1" in CONFIG["experiment_name"]
assert "FRESH_160K" in CONFIG["experiment_name"]


# ============================================================
# PRINT EXPERIMENT CONFIGURATION
# ============================================================

print("=" * 70)
print("STEP 4 — EXPERIMENT B1 / GAUSSIAN V1.3 / FRESH 160K")
print("=" * 70)

for key, value in CONFIG.items():
    print(f"{key:34s}: {value}")

print()
print("=" * 70)
print("B1 EXPERIMENT LOCK")
print("=" * 70)

print("Experiment       :", CONFIG["experiment_name"])
print("Gaussian radius  :", CONFIG["gaussian_radius_mode"])
print("Min overlap      :", CONFIG["gaussian_min_overlap"])
print("WH loss          :", CONFIG["wh_loss_type"])
print("Backbone         :", CONFIG["backbone"])
print("Input size       :", CONFIG["input_size"])
print("Output stride    :", CONFIG["output_stride"])
print("Training images  :", CONFIG["source_images_total"])
print("Source epochs    :", CONFIG["source_epochs"])
print("Batch size       :", CONFIG["batch_size"])
print("Seed             :", CONFIG["seed"])

print()
print("STEP 4 PASSED — B1 CONFIGURATION READY")
print("NO TRAINING EXECUTED")
