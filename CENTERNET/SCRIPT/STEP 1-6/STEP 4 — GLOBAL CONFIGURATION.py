# ============================================================
# STEP 4 — GLOBAL CONFIGURATION
# CENTERNET COCO-80 V1.2 / 160K
# ============================================================

import os

CONFIG = {
    "experiment_name": "CenterNet_COCO80_FSOD_V1_2_160K_BS8",

    "backbone": "resnet101",
    "backbone_pretrained": True,
    "num_classes": 80,

    "input_size": 640,
    "output_stride": 4,

    "batch_size": 8,
    "num_workers": 8,
    "persistent_workers": True,
    "prefetch_factor": 4,
    "learning_rate": 1e-4,
    "weight_decay": 1e-4,

    "source_epochs": 20,
    "source_images_total": 160000,
    "source_images_per_epoch": 8000,
    "source_images_per_class_per_epoch": 100,
    "source_sampling_policy": "class-balanced-multiclass",
    "source_manifest_seed_offset": 3000,

    "train_horizontal_flip_p": 0.0,
    "gaussian_radius_mode": "v1_2_original",
    "best_checkpoint_metric": "official_COCO_mAP_50_95",

    "encoded_num_workers": 4,
    "encoded_prefetch_factor": 2,

    "use_fused_adamw": False,
    "use_channels_last": False,

    "precision": "bfloat16",
    "grad_clip_norm": 10.0,

    "run_engineering_sanity": False,
    "run_tiny_overfit_diagnostic": False,
    "debug_finite_checks": False,
    "progress_every": 100,

    "score_threshold": 0.50,
    "iou_threshold": 0.50,

    "seed": SEED,

    "data_root": "/content/coco",
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
    CONFIG["data_root"], "annotations",
    "instances_train2017.json"
)
CONFIG["val_annotations"] = os.path.join(
    CONFIG["data_root"], "annotations",
    "instances_val2017.json"
)

assert CONFIG["num_classes"] == 80
assert CONFIG["input_size"] == 640
assert CONFIG["output_stride"] == 4
assert CONFIG["batch_size"] == 8
assert CONFIG["source_epochs"] == 20
assert CONFIG["source_images_total"] == 160000
assert CONFIG["source_images_per_epoch"] == 8000
assert CONFIG["train_horizontal_flip_p"] == 0.0
assert CONFIG["run_tiny_overfit_diagnostic"] is False
assert CONFIG["run_engineering_sanity"] is False

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

print("=" * 65)
print("STEP 4 — V1.2 / 160K CONFIGURATION")
print("=" * 65)

for key, value in CONFIG.items():
    print(f"{key:30s}: {value}")

print("STEP 4 PASSED")
