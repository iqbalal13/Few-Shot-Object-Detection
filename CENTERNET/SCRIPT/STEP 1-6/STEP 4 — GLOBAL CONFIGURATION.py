
# ============================================================
# STEP 4 — GLOBAL CONFIGURATION
# BALANCED80K V1.2 — LOCKED
# ============================================================

import os

CONFIG = {
    # Experiment identity
    "experiment_name": "CenterNet_COCO80_FSOD_BALANCED80K_V1_2_BS8",

    # Model
    "backbone": "resnet101",
    "backbone_pretrained": True,
    "num_classes": 80,

    # Image
    "input_size": 640,
    "output_stride": 4,

    # Training
    "batch_size": 8,
    "num_workers": 8,
    "persistent_workers": True,
    "prefetch_factor": 4,
    "learning_rate": 1e-4,
    "weight_decay": 1e-4,

    # Balanced80K source training
    "source_epochs": 10,
    "source_images_total": 80000,
    "source_images_per_epoch": 8000,
    "source_images_per_class_per_epoch": 100,
    "source_sampling_policy": "class-balanced-multiclass",
    "source_manifest_seed_offset": 3000,

    # CPU target encoding workers
    "encoded_num_workers": 4,
    "encoded_prefetch_factor": 2,

    # Numerical kernels unchanged
    "use_fused_adamw": False,
    "use_channels_last": False,

    # Precision / stability
    "precision": "bfloat16",
    "grad_clip_norm": 10.0,

    # Engineering switches
    "run_engineering_sanity": False,
    "debug_finite_checks": False,
    "progress_every": 100,

    # Detection thresholds
    "score_threshold": 0.50,
    "iou_threshold": 0.50,

    # Reproducibility
    "seed": SEED,

    # Dataset root
    "data_root": "/content/coco",

    # Checkpoints
    "checkpoint_root": "/content/checkpoints_centernet",
}

os.makedirs(CONFIG["data_root"], exist_ok=True)
os.makedirs(CONFIG["checkpoint_root"], exist_ok=True)

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

print("=" * 70)
print("CENTERNET GLOBAL CONFIG — BALANCED80K V1.2")
print("=" * 70)

for key, value in CONFIG.items():
    print(f"{key:26s}: {value}")

# Original invariants
assert CONFIG["backbone"] == "resnet101"
assert CONFIG["num_classes"] == 80
assert CONFIG["input_size"] == 640
assert CONFIG["output_stride"] == 4
assert CONFIG["batch_size"] == 8
assert CONFIG["num_workers"] == 8
assert CONFIG["persistent_workers"] is True
assert CONFIG["prefetch_factor"] == 4
assert CONFIG["learning_rate"] == 1e-4
assert CONFIG["weight_decay"] == 1e-4
assert CONFIG["precision"] == "bfloat16"
assert CONFIG["grad_clip_norm"] == 10.0
assert CONFIG["score_threshold"] == 0.50
assert CONFIG["iou_threshold"] == 0.50
assert CONFIG["seed"] == 42

# Balanced80K invariants
assert CONFIG["source_epochs"] == 10
assert CONFIG["source_images_per_epoch"] == 8000
assert CONFIG["source_images_total"] == 80000
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

assert (
    CONFIG["source_images_per_epoch"]
    % CONFIG["batch_size"] == 0
)

assert CONFIG["use_fused_adamw"] is False
assert CONFIG["use_channels_last"] is False

print("BALANCED80K V1.2 CONFIG LOCKED")
print("STEP 4 PASSED")
