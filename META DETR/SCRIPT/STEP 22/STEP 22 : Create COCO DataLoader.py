# ==========================================================
# STEP 22 : Create COCO DataLoader
# ==========================================================

from torch.utils.data import DataLoader

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=COCO_CONFIG["batch_size"],
    shuffle=COCO_CONFIG["shuffle"],
    num_workers=COCO_CONFIG["num_workers"],
    pin_memory=COCO_CONFIG["pin_memory"],
    collate_fn=collate_fn
)

print("=" * 60)
print("COCO DataLoader Ready")
print("Number of Batches :", len(train_loader))
print("=" * 60)
