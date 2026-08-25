# ==========================================================
# STEP 22 : Create COCO Episodic DataLoader (FINAL)
# ==========================================================

from torch.utils.data import DataLoader
import torch


# ==========================================================
# Reproducible DataLoader Generator
# ==========================================================

train_generator = torch.Generator()

train_generator.manual_seed(
    CONFIG["seed"]
)


# ==========================================================
# DataLoader
# ==========================================================

train_loader = DataLoader(

    dataset=train_dataset,

    batch_size=
        COCO_CONFIG["batch_size"],

    shuffle=
        COCO_CONFIG["shuffle"],

    num_workers=
        COCO_CONFIG["num_workers"],

    pin_memory=
        COCO_CONFIG["pin_memory"],

    collate_fn=
        episodic_collate_fn,

    drop_last=False,

    generator=train_generator
)


print("=" * 70)
print("STEP 22 : COCO EPISODIC DATALOADER READY")
print("=" * 70)

print(
    "Episodes in Dataset :",
    len(train_dataset)
)

print(
    "Total Batches       :",
    len(train_loader)
)

print(
    "Batch Size          :",
    COCO_CONFIG["batch_size"]
)

print("=" * 70)
