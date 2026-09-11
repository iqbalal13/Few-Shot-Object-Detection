# ==========================================================
# STEP 18: Person-Only DataLoaders
# ==========================================================

from torch.utils.data import DataLoader, Subset


def episodic_collate_fn(batch):
    return {
        "support_images": torch.stack([
            item["support_image"]
            for item in batch
        ]),

        "query_images": torch.stack([
            item["query_image"]
            for item in batch
        ]),

        "episode_classes": torch.stack([
            item["episode_class"]
            for item in batch
        ]),

        "support_targets": [
            item["support_target"]
            for item in batch
        ],

        "query_targets": [
            item["query_target"]
            for item in batch
        ],
    }


def make_episode_loader(
    dataset,
    num_workers=None,
):
    workers = (
        COCO_CONFIG["num_workers"]
        if num_workers is None
        else num_workers
    )

    return DataLoader(
        dataset,
        batch_size=COCO_CONFIG["batch_size"],
        shuffle=False,
        num_workers=workers,
        pin_memory=COCO_CONFIG["pin_memory"],
        collate_fn=episodic_collate_fn,
        persistent_workers=False,
    )


train_loader = make_episode_loader(
    train_dataset
)

val_dataset = COCOEpisodeDataset(
    coco=coco_val,
    image_dir=VAL_IMAGE_DIR,
    support_transform=support_transform,
    query_transform=query_transform,
    num_episodes=COCO_CONFIG["num_val_episodes"],
    seed=CONFIG["seed"] + 10000,
    min_bbox_size=COCO_CONFIG["min_bbox_size"],
)

# Tetap sama pada seluruh epoch validasi.
val_dataset.set_epoch(0)

val_loader = make_episode_loader(
    val_dataset
)

assert (
    train_dataset.valid_labels
    == val_dataset.valid_labels
    == [0]
)

assert set(
    train_dataset.class_to_img_ids[0]
).isdisjoint(
    val_dataset.class_to_img_ids[0]
)

print("=" * 70)
print("STEP 18: PERSON LOADERS READY")
print("=" * 70)

print("Train episodes:", len(train_dataset))
print("Val episodes  :", len(val_dataset))
print("Classes       : person only")

print(
    "Validation uses fixed person support-query episodes "
    "from COCO-Val."
)

print("=" * 70)
