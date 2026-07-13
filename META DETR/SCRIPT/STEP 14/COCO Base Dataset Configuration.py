# ==========================================================
# STEP 14 : COCO Base Dataset Configuration
# ==========================================================

COCO_CONFIG = {

    # Root Dataset
    "root": "/content/MetaDETR_Thesis/datasets/coco",

    # Images
    "train_images": "train2017",
    "val_images": "val2017",

    # Annotation
    "train_annotation": "annotations/instances_train2017.json",
    "val_annotation": "annotations/instances_val2017.json",

    # Image Size
    "image_size": 800,

    # Batch Size
    "batch_size": 2,

    # Workers
    "num_workers": 2,

    # Shuffle
    "shuffle": True,

    # Pin Memory
    "pin_memory": True

}

print("="*60)
print("COCO Base Dataset Configuration")
print("="*60)

for k,v in COCO_CONFIG.items():
    print(f"{k:25}: {v}")
