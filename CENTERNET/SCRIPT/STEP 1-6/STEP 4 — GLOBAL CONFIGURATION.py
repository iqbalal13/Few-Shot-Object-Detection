
# STEP 4 — A3 PILOT CONFIG
import os

CONFIG = {
    "experiment_name": "CenterNet_COCO80_A3_R1_PILOT_FROM_A2_160K",
    "backbone": "resnet101",
    "backbone_pretrained": True,
    "num_classes": 80,
    "input_size": 640,
    "output_stride": 4,
    "batch_size": 8,
    "num_workers": 8,
    "persistent_workers": True,
    "prefetch_factor": 4,
    "encoded_num_workers": 4,
    "encoded_prefetch_factor": 2,

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
    "gaussian_radius_mode": "v1_2_original",
    "gaussian_min_overlap": 0.7,

    "best_checkpoint_metric": "official_COCO_mAP_50_95",

    "wh_loss_type": "hybrid_abs_relative",
    "wh_absolute_fraction": 0.75,
    "wh_relative_fraction": 0.25,
    "wh_relative_eps": 2.0,
    "wh_relative_beta": 0.20,
    "wh_relative_scale": 10.0,

    # A3
    "a3_neighbor_radius": 1,
    "a3_neighbor_lambda": 0.25,
    "a3_pilot_lr": 2e-5,
    "a3_pilot_epochs": 2,

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
    CONFIG[key] = os.path.join(CONFIG["data_root"], relative)

os.makedirs(CONFIG["data_root"], exist_ok=True)
os.makedirs(CONFIG["checkpoint_root"], exist_ok=True)

assert CONFIG["gaussian_radius_mode"] == "v1_2_original"
assert CONFIG["wh_loss_type"] == "hybrid_abs_relative"
assert CONFIG["wh_absolute_fraction"] == 0.75
assert CONFIG["wh_relative_fraction"] == 0.25
assert CONFIG["a3_neighbor_radius"] == 1
assert CONFIG["a3_neighbor_lambda"] == 0.25
assert CONFIG["source_images_total"] == 160000

print("STEP 4 PASS — A3 PILOT CONFIG")
for key, value in CONFIG.items():
    print(f"{key}: {value}")
