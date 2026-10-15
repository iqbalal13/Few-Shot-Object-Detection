
# ============================================================
# STEP 13 — DATALOADERS: BASELINE + WORKER-ENCODED VARIANTS
# Keep baseline loader interface for Steps 17/28/29 sanity.
# Step 30 uses an independent balanced sampler + encoded loaders.
# ============================================================

import random
import numpy as np
import torch
from torch.utils.data import DataLoader


def centernet_collate_fn(batch):
    """
    Original contract:
        (images, list_of_raw_targets)
    """
    images = torch.stack(
        [item[0] for item in batch],
        dim=0
    )

    targets = [
        item[1]
        for item in batch
    ]

    return images, targets


def centernet_encoded_collate_fn(batch):
    """
    Runs inside DataLoader workers after Step 16.

    Returns:
        images
        raw_targets
        encoded_cpu_targets
    """
    images, targets = centernet_collate_fn(batch)

    encoded_per_image = [
        build_centernet_target(target)
        for target in targets
    ]

    names = (
        "heatmap",
        "wh",
        "offset",
        "indices",
        "mask"
    )

    encoded = {
        name: torch.stack([
            target[name]
            for target in encoded_per_image
        ])
        for name in names
    }

    return images, targets, encoded


def seed_worker(worker_id):
    worker_seed = (
        torch.initial_seed() % (2 ** 32)
    )

    random.seed(worker_seed)
    np.random.seed(worker_seed)


def make_loader_generator(seed):
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return generator


def _loader(
    dataset,
    seed,
    shuffle,
    sampler=None,
    encoded=False,
    drop_last=False
):
    generator = make_loader_generator(seed)

    if encoded:
        workers = CONFIG["encoded_num_workers"]
        prefetch = CONFIG["encoded_prefetch_factor"]
    else:
        workers = CONFIG["num_workers"]
        prefetch = CONFIG["prefetch_factor"]

    kwargs = dict(
        dataset=dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
        drop_last=drop_last,
        collate_fn=(
            centernet_encoded_collate_fn
            if encoded
            else centernet_collate_fn
        ),
    )

    if workers > 0:
        kwargs.update(
            persistent_workers=bool(
                CONFIG["persistent_workers"]
            ),
            prefetch_factor=prefetch,
            worker_init_fn=seed_worker,
        )

    loader = DataLoader(**kwargs)

    return loader, generator


def build_train_loader(
    seed,
    sampler=None,
    encoded=False
):
    return _loader(
        train_dataset,
        seed,
        shuffle=(sampler is None),
        sampler=sampler,
        encoded=encoded,
        drop_last=True
    )


def build_val_loader(
    seed,
    encoded=False
):
    return _loader(
        val_dataset,
        seed,
        shuffle=False,
        encoded=encoded,
        drop_last=False
    )


# Original-format loaders for sanity checks
train_loader, train_generator = build_train_loader(
    SEED + 1300
)

val_loader, val_generator = build_val_loader(
    SEED + 1301
)


# Sanity check without consuming shuffled train loader
san_loader = DataLoader(
    train_dataset,
    batch_size=CONFIG["batch_size"],
    shuffle=False,
    num_workers=0,
    collate_fn=centernet_collate_fn
)

images, targets = next(iter(san_loader))

assert images.shape == (
    CONFIG["batch_size"],
    3,
    CONFIG["input_size"],
    CONFIG["input_size"]
)

assert len(targets) == CONFIG["batch_size"]

print(
    "STEP 13 PASSED | COCO train:",
    len(train_dataset),
    "| COCO val:",
    len(val_dataset),
    "| global train batches:",
    len(train_loader)
)

del san_loader, images, targets
