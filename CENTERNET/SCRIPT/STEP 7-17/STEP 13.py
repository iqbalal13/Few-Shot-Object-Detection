
# STEP 13 — DATA LOADERS (A3 ENCODED)
import random
import numpy as np
import torch
from torch.utils.data import DataLoader

def centernet_collate_fn(batch):
    images = torch.stack([v[0] for v in batch])
    targets = [v[1] for v in batch]
    return images, targets

def centernet_encoded_collate_fn(batch):
    images, raw_targets = centernet_collate_fn(batch)
    items = [
        build_centernet_target(t)
        for t in raw_targets
    ]

    fields = (
        "heatmap", "wh", "offset",
        "indices", "mask",
        "neighbor_indices", "neighbor_mask"
    )

    encoded = {
        k: torch.stack([v[k] for v in items])
        for k in fields
    }

    return images, raw_targets, encoded

def seed_worker(worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)

def make_loader_generator(seed):
    g = torch.Generator()
    g.manual_seed(int(seed))
    return g

def _loader(
    dataset, seed, shuffle,
    sampler=None, encoded=False,
    drop_last=False
):
    g = make_loader_generator(seed)

    workers = (
        CONFIG["encoded_num_workers"]
        if encoded else CONFIG["num_workers"]
    )

    kwargs = dict(
        dataset=dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        generator=g,
        drop_last=drop_last,
        worker_init_fn=seed_worker,
        collate_fn=(
            centernet_encoded_collate_fn
            if encoded else centernet_collate_fn
        )
    )

    if workers > 0:
        kwargs["persistent_workers"] = CONFIG["persistent_workers"]
        kwargs["prefetch_factor"] = (
            CONFIG["encoded_prefetch_factor"]
            if encoded else CONFIG["prefetch_factor"]
        )

    return DataLoader(**kwargs), g

def build_train_loader(seed, sampler=None, encoded=False):
    return _loader(
        train_dataset, seed,
        shuffle=sampler is None,
        sampler=sampler,
        encoded=encoded,
        drop_last=True
    )

def build_val_loader(seed, encoded=False):
    return _loader(
        val_dataset, seed,
        shuffle=False,
        encoded=encoded
    )

train_loader, train_generator = build_train_loader(SEED + 1300)
val_loader, val_generator = build_val_loader(SEED + 1301)

images, targets = next(iter(DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=False,
    num_workers=0,
    collate_fn=centernet_collate_fn
)))

assert images.shape == (8, 3, 640, 640)
assert len(targets) == 8

print("STEP 13 PASS — A3 DATALOADER READY")
del images, targets
