# ============================================================
# STEP 4 — GLOBAL CONFIGURATION
# ============================================================

CONFIG = {
    # Model
    "backbone": "resnet101",
    "backbone_pretrained": True,
    "num_classes": 80,

    # Image
    "input_size": 640,
    "output_stride": 4,

    # Training
    "batch_size": 4,
    "num_workers": 2,
    "learning_rate": 1e-4,
    "weight_decay": 1e-4,

    # Primary detection thresholds
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

print("=" * 60)
print("CENTERNET GLOBAL CONFIG")
print("=" * 60)

for key, value in CONFIG.items():
    print(f"{key:22s}: {value}")
