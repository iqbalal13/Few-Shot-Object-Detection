
# ============================================================
# STEP 4 — A3-v2: STRONGER SOURCE-STAGE CONTINUATION
# A2 Hybrid WH + A3 Neighbor R1
# ============================================================

import os

CONFIG = {
    "experiment_name": "CenterNet_COCO80_A3_V2_AUGMENTED_MIXED_96K",

    # Model
    "backbone": "resnet101",
    "backbone_pretrained": True,
    "num_classes": 80,
    "input_size": 640,
    "output_stride": 4,

    # DataLoader
    "batch_size": 8,
    "num_workers": 8,
    "persistent_workers": False,
    "prefetch_factor": 4,
    "encoded_num_workers": 4,
    "encoded_prefetch_factor": 2,

    # Common training parameters
    "learning_rate": 2e-5,
    "weight_decay": 1e-4,
    "grad_clip_norm": 10.0,
    "precision": "bfloat16",

    # Historical source parameters
    "source_epochs": 20,
    "source_images_total": 160000,
    "source_images_per_epoch": 8000,
    "source_images_per_class_per_epoch": 100,
    "source_sampling_policy": "class-balanced-multiclass",
    "source_manifest_seed_offset": 3000,

    # A3-v2 train augmentation
    "train_horizontal_flip_p": 0.5,
    "train_scale_min": 0.85,
    "train_scale_max": 1.15,
    "train_translate_frac": 0.04,
    "train_visible_box_min": 0.30,
    "train_color_jitter": 0.15,

    # Preserve Gaussian V1.2
    "gaussian_radius_mode": "v1_2_original",
    "gaussian_min_overlap": 0.7,

    # Preserve A2 Hybrid WH
    "wh_loss_type": "hybrid_abs_relative",
    "wh_absolute_fraction": 0.75,
    "wh_relative_fraction": 0.25,
    "wh_relative_eps": 2.0,
    "wh_relative_beta": 0.20,
    "wh_relative_scale": 10.0,

    # Preserve A3
    "a3_neighbor_radius": 1,
    "a3_neighbor_lambda": 0.25,

    # A3-v2 continuation: 12 x 8000 = +96K
    "v2_epochs": 12,
    "v2_images_per_epoch": 8000,
    "v2_uniform_fraction": 0.5,

    # Differential learning rates
    "v2_backbone_lr": 1e-5,
    "v2_neck_head_lr": 3e-5,
    "v2_eta_min": 2e-6,

    # Evaluation and model selection
    "v2_eval_interval": 2,
    "v2_early_stop_evals": 3,
    "v2_min_improvement": 0.00005,
    "v2_starting_map": 0.152170,
    "best_checkpoint_metric": "official_COCO_mAP_50_95",

    # Engineering
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
    "max_objects": 128,
}

for key, relative in {
    "train_images": "train2017",
    "val_images": "val2017",
    "train_annotations": "annotations/instances_train2017.json",
    "val_annotations": "annotations/instances_val2017.json",
}.items():
    CONFIG[key] = os.path.join(
        CONFIG["data_root"],
        relative
    )

os.makedirs(CONFIG["data_root"], exist_ok=True)
os.makedirs(CONFIG["checkpoint_root"], exist_ok=True)

assert CONFIG["gaussian_radius_mode"] == "v1_2_original"
assert CONFIG["a3_neighbor_lambda"] == 0.25
assert CONFIG["wh_loss_type"] == "hybrid_abs_relative"

assert (
    CONFIG["v2_images_per_epoch"]
    % CONFIG["batch_size"] == 0
)

print(
    "STEP 4 PASS — A3-v2: "
    "+96K maximum, COCOeval + early stopping"
)
