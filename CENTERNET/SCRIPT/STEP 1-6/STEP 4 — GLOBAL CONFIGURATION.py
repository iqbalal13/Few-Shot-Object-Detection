# ============================================================
# STEP 4 — EXPERIMENT B1 CONFIGURATION
# GAUSSIAN RADIUS V1.3 / MASKED L1 WH / FRESH 160K
# ============================================================

import os

CONFIG = {
    "experiment_name": (
        "CenterNet_COCO80_B1_GAUSSIAN_V13_FRESH_160K_BS8"
    ),

    # Model
    "backbone": "resnet101",
    "backbone_pretrained": True,
    "num_classes": 80,
    "input_size": 640,
    "output_stride": 4,

    # DataLoader
    "batch_size": 8,
    "num_workers": 8,
    "persistent_workers": True,
    "prefetch_factor": 4,
    "encoded_num_workers": 4,
    "encoded_prefetch_factor": 2,

    # Optimizer
    "learning_rate": 1e-4,
    "weight_decay": 1e-4,
    "grad_clip_norm": 10.0,
    "precision": "bfloat16",

    # Source training
    "source_epochs": 20,
    "source_images_total": 160000,
    "source_images_per_epoch": 8000,
    "source_images_per_class_per_epoch": 100,
    "source_sampling_policy": "class-balanced-multiclass",
    "source_manifest_seed_offset": 3000,

    # Experiment B1
    "train_horizontal_flip_p": 0.0,
    "gaussian_radius_mode": "v1_3_corrected",
    "gaussian_min_overlap": 0.7,

    # Baseline WH loss, NOT A2 hybrid
    "wh_loss_type": "masked_l1",

    # Keep A2 parameters for metadata compatibility;
    # they are NOT used when wh_loss_type is masked_l1.
    "wh_absolute_fraction": 0.75,
    "wh_relative_fraction": 0.25,
    "wh_relative_eps": 2.0,
    "wh_relative_beta": 0.20,
    "wh_relative_scale": 10.0,

    # Original loss weights
    "hm_loss_weight": 1.0,
    "wh_loss_weight": 0.1,
    "offset_loss_weight": 1.0,

    # Checkpoint selection
    "best_checkpoint_metric": "official_COCO_mAP_50_95",

    # Engineering
    "use_fused_adamw": False,
    "use_channels_last": False,
    "run_engineering_sanity": False,
    "run_tiny_overfit_diagnostic": False,
    "debug_finite_checks": False,
    "progress_every": 100,

    # Evaluation
    "score_threshold": 0.50,
    "iou_threshold": 0.50,

    # Reproducibility
    "seed": SEED,

    # Paths
    "data_root": "/content/coco",
    "checkpoint_root": "/content/checkpoints_centernet",
}

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

os.makedirs(CONFIG["data_root"], exist_ok=True)
os.makedirs(CONFIG["checkpoint_root"], exist_ok=True)

# ============================================================
# EXPERIMENT LOCK
# ============================================================

assert CONFIG["num_classes"] == 80
assert CONFIG["input_size"] == 640
assert CONFIG["output_stride"] == 4
assert CONFIG["batch_size"] == 8

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
    CONFIG["num_classes"]
    * CONFIG["source_images_per_class_per_epoch"]
    == CONFIG["source_images_per_epoch"]
)

assert CONFIG["train_horizontal_flip_p"] == 0.0

assert (
    CONFIG["gaussian_radius_mode"]
    == "v1_3_corrected"
)

assert CONFIG["gaussian_min_overlap"] == 0.7

assert CONFIG["wh_loss_type"] == "masked_l1"

assert CONFIG["hm_loss_weight"] == 1.0
assert CONFIG["wh_loss_weight"] == 0.1
assert CONFIG["offset_loss_weight"] == 1.0

assert CONFIG["run_engineering_sanity"] is False
assert CONFIG["run_tiny_overfit_diagnostic"] is False

# ============================================================
# SUMMARY
# ============================================================

print("=" * 70)
print("STEP 4 — EXPERIMENT B1 CONFIGURATION")
print("=" * 70)

for key, value in CONFIG.items():
    print(f"{key:36s}: {value}")

print("=" * 70)
print("Experiment : B1 Gaussian Radius V1.3")
print("WH loss    : Original masked L1")
print("Training   : Fresh 160K")
print("Neck       : Original deconvolution-only")
print("STEP 4 PASSED")
