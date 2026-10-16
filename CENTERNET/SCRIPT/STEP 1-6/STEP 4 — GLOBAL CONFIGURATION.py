
# ============================================================
# STEP 4 — GLOBAL CONFIGURATION
# BALANCED80K V1.3 — LOCKED
# ============================================================

import os

CONFIG = {
    # Experiment identity
    "experiment_name": "CenterNet_COCO80_FSOD_BALANCED80K_V1_3_AUG_GAUSS_BS8",

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
    "train_horizontal_flip_p": 0.5,
    "gaussian_radius_mode": "quadratic_small_root",
    "best_checkpoint_metric": "official_COCO_AP50",
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
print("CENTERNET GLOBAL CONFIG — BALANCED80K V1.3")
