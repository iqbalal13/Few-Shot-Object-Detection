# ==========================================================
# STEP 14 : COCO Base Meta-Training Configuration (FINAL)
# ==========================================================

COCO_CONFIG = {

    # ======================================================
    # Dataset Root
    # ======================================================

    "root":
        "/content/datasets/coco",

    "train_images":
        "train2017",

    "val_images":
        "val2017",

    "train_annotation":
        "annotations/instances_train2017.json",

    "val_annotation":
        "annotations/instances_val2017.json",


    # ======================================================
    # Episodic Meta-Training
    # ======================================================

    # 8000 generated meta-learning episodes
    "num_train_episodes": 8000,

    # 1-way episodic learning
    "episode_way":
        CONFIG["episode_way"],

    "support_shot":
        CONFIG["base_support_shot"],


    # ======================================================
    # Image
    # ======================================================

    "image_size":
        CONFIG["image_size"],


    # ======================================================
    # Bounding Box Filtering
    # ======================================================

    "min_bbox_size": 2.0,


    # ======================================================
    # DataLoader
    # ======================================================

    "batch_size": 1,

    "shuffle": True,

    "num_workers": 2,

    "pin_memory":
        torch.cuda.is_available(),


    # ======================================================
    # Reproducibility
    # ======================================================

    "seed":
        CONFIG["seed"]
}


print("=" * 70)
print("COCO Base Meta-Training Configuration")
print("=" * 70)

for key, value in COCO_CONFIG.items():
    print(f"{key:25s}: {value}")

print("=" * 70)
