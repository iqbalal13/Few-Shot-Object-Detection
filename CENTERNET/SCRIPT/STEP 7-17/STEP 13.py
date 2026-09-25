# ============================================================
# STEP 13 — TRAIN / VALIDATION DATALOADER
# ============================================================

from torch.utils.data import DataLoader

print("=" * 70)
print("STEP 13 — BUILD DATALOADERS")
print("=" * 70)


def centernet_collate_fn(batch):

    images = torch.stack(
        [item[0] for item in batch],
        dim=0
    )

    targets = [
        item[1]
        for item in batch
    ]

    return images, targets


train_loader = DataLoader(
    train_dataset,
    batch_size=CONFIG["batch_size"],
    shuffle=True,
    num_workers=CONFIG["num_workers"],
    pin_memory=torch.cuda.is_available(),
    drop_last=True,
    collate_fn=centernet_collate_fn
)

val_loader = DataLoader(
    val_dataset,
    batch_size=CONFIG["batch_size"],
    shuffle=False,
    num_workers=CONFIG["num_workers"],
    pin_memory=torch.cuda.is_available(),
    drop_last=False,
    collate_fn=centernet_collate_fn
)

images, targets = next(
    iter(train_loader)
)

print("Train batches :", len(train_loader))
print("Val batches   :", len(val_loader))

print("\nSingle train batch:")
print("Images shape  :", images.shape)
print("Batch targets :", len(targets))

for i, target in enumerate(targets):
    print(
        f"Sample {i}: "
        f"{len(target['boxes'])} objects"
    )

assert images.ndim == 4
assert images.shape[1] == 3
assert images.shape[2] == CONFIG["input_size"]
assert images.shape[3] == CONFIG["input_size"]

print("\nSTEP 13 PASSED")
