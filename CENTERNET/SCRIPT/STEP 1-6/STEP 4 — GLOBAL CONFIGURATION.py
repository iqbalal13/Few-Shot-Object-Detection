
# ============================================================
# STEP 4 — EXPERIMENT B1 CONFIGURATION
# A2 HYBRID WH + CORRECTED GAUSSIAN RADIUS V1.3
# FRESH 160K / CONTROL = A2 V1.2
# ============================================================

import os

CONFIG = {
    "experiment_name":
        "CenterNet_COCO80_B1_V13_ON_A2_FRESH_160K_BS8",

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
    "grad_clip_norm": 10.0,
    "precision": "bfloat16",

    "source_epochs": 20,
    "source_images_total": 160000,
    "source_images_per_epoch": 8000,
    "source_images_per_class_per_epoch": 100,
    "source_sampling_policy": "class-balanced-multiclass",
    "source_manifest_seed_offset": 3000,

    "train_horizontal_flip_p": 0.0,

    # B1 — only the heatmap target radius changes.
    "gaussian_radius_mode": "v1_3_corrected",
    "gaussian_min_overlap": 0.7,

    "best_checkpoint_metric": "official_COCO_mAP_50_95",

    # UNCHANGED FROM A2
    "wh_loss_type": "hybrid_abs_relative",
    "wh_absolute_fraction": 0.75,
    "wh_relative_fraction": 0.25,
    "wh_relative_eps": 2.0,
    "wh_relative_beta": 0.20,
    "wh_relative_scale": 10.0,

    "encoded_num_workers": 4,
    "encoded_prefetch_factor": 2,
    "use_fused_adamw": False,
    "use_channels_last": False,

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

os.makedirs(CONFIG["data_root"], exist_ok=True)
os.makedirs(CONFIG["checkpoint_root"], exist_ok=True)

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
    CONFIG["source_images_per_class_per_epoch"]
    * CONFIG["num_classes"]
    == CONFIG["source_images_per_epoch"]
)

assert CONFIG["train_horizontal_flip_p"] == 0.0
assert CONFIG["gaussian_radius_mode"] == "v1_3_corrected"
assert CONFIG["gaussian_min_overlap"] == 0.7

assert CONFIG["wh_loss_type"] == "hybrid_abs_relative"
assert CONFIG["wh_absolute_fraction"] == 0.75
assert CONFIG["wh_relative_fraction"] == 0.25
assert abs(
    CONFIG["wh_absolute_fraction"]
    + CONFIG["wh_relative_fraction"] - 1.0
) < 1e-12

assert CONFIG["wh_relative_eps"] == 2.0
assert CONFIG["wh_relative_beta"] == 0.20
assert CONFIG["wh_relative_scale"] == 10.0

assert not CONFIG["run_tiny_overfit_diagnostic"]
assert not CONFIG["run_engineering_sanity"]

print("=" * 65)
print("STEP 4 — EXPERIMENT B1 / GAUSSIAN V1.3")
print("=" * 65)

for key, value in CONFIG.items():
    print(f"{key:34s}: {value}")

print("STEP 4 PASSED")
