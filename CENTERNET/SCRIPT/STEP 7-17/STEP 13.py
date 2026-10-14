# ============================================================
# STEP 13 — TRAIN / VALIDATION DATALOADER
# FAST-REPRO V1
#
# - 8 workers
# - pinned memory
# - persistent workers
# - prefetch
# - deterministic worker seeding
# - deterministic DataLoader generator
# ============================================================

import random
import numpy as np
import torch

from torch.utils.data import DataLoader


print("=" * 70)
print("STEP 13 — BUILD FAST-REPRO DATALOADERS")
print("=" * 70)


# ============================================================
# COLLATE FUNCTION
# ============================================================

def centernet_collate_fn(
    batch
):

    images = torch.stack(
        [
            item[0]
            for item in batch
        ],
        dim=0
    )


    targets = [
        item[1]
        for item in batch
    ]


    return images, targets


# ============================================================
# DATALOADER WORKER SEEDING
# ============================================================

def seed_worker(
    worker_id
):

    worker_seed = (
        torch.initial_seed()
        % (2 ** 32)
    )

    np.random.seed(
        worker_seed
    )

    random.seed(
        worker_seed
    )


# ============================================================
# GENERATOR BUILDER
# ============================================================

def make_loader_generator(
    seed
):

    generator = torch.Generator()

    generator.manual_seed(
        int(seed)
    )

    return generator


# ============================================================
# TRAIN LOADER BUILDER
#
# A builder is useful because Step 30 will create its own
# independent deterministic source-training loader.
# ============================================================

def build_train_loader(
    seed
):

    generator = \
        make_loader_generator(
            seed
        )


    loader = DataLoader(

        train_dataset,

        batch_size=
            CONFIG["batch_size"],

        shuffle=True,

        num_workers=
            CONFIG["num_workers"],

        pin_memory=
            torch.cuda.is_available(),

        persistent_workers=
            CONFIG[
                "persistent_workers"
            ],

        prefetch_factor=
            CONFIG[
                "prefetch_factor"
            ],

        worker_init_fn=
            seed_worker,

        generator=
            generator,

        drop_last=True,

        collate_fn=
            centernet_collate_fn
    )


    return loader, generator


# ============================================================
# VALIDATION LOADER BUILDER
# ============================================================

def build_val_loader(
    seed
):

    generator = \
        make_loader_generator(
            seed
        )


    loader = DataLoader(

        val_dataset,

        batch_size=
            CONFIG["batch_size"],

        shuffle=False,

        num_workers=
            CONFIG["num_workers"],

        pin_memory=
            torch.cuda.is_available(),

        persistent_workers=
            CONFIG[
                "persistent_workers"
            ],

        prefetch_factor=
            CONFIG[
                "prefetch_factor"
            ],

        worker_init_fn=
            seed_worker,

        generator=
            generator,

        drop_last=False,

        collate_fn=
            centernet_collate_fn
    )


    return loader, generator


# ============================================================
# GLOBAL LOADERS
#
# These are used by sanity checks / short training.
#
# Full source training in Step 30 will build its own
# independent train loader so its shuffle sequence does not
# depend on whether earlier engineering tests were run.
# ============================================================

train_loader, train_generator = \
    build_train_loader(
        SEED + 1300
    )


val_loader, val_generator = \
    build_val_loader(
        SEED + 1301
    )


# ============================================================
# SANITY LOADER
#
# IMPORTANT:
# do NOT consume shuffled train_loader here.
# This keeps training RNG untouched.
# ============================================================

sanity_loader = DataLoader(

    train_dataset,

    batch_size=
        CONFIG["batch_size"],

    shuffle=False,

    num_workers=0,

    pin_memory=
        torch.cuda.is_available(),

    drop_last=False,

    collate_fn=
        centernet_collate_fn
)


images, targets = next(
    iter(
        sanity_loader
    )
)


# ============================================================
# OUTPUT
# ============================================================

print(
    "Train images  :",
    len(train_dataset)
)

print(
    "Val images    :",
    len(val_dataset)
)

print(
    "Train batches :",
    len(train_loader)
)

print(
    "Val batches   :",
    len(val_loader)
)

print(
    "Batch size    :",
    CONFIG["batch_size"]
)

print(
    "Workers       :",
    CONFIG["num_workers"]
)

print(
    "Persistent    :",
    CONFIG["persistent_workers"]
)

print(
    "Prefetch      :",
    CONFIG["prefetch_factor"]
)


print(
    "\nSingle deterministic sanity batch:"
)

print(
    "Images shape  :",
    images.shape
)

print(
    "Batch targets :",
    len(targets)
)


for i, target in enumerate(
    targets
):

    print(
        f"Sample {i}: "
        f"{len(target['boxes'])} objects"
    )


# ============================================================
# SANITY ASSERTIONS
# ============================================================

assert images.ndim == 4

assert (
    images.shape[0]
    == CONFIG["batch_size"]
)

assert images.shape[1] == 3

assert (
    images.shape[2]
    == CONFIG["input_size"]
)

assert (
    images.shape[3]
    == CONFIG["input_size"]
)


del sanity_loader
del images
del targets


print(
    "\nDataLoader sequence : SEEDED"
)

print(
    "Training generator : INDEPENDENT"
)

print(
    "STEP 13 PASSED"
)
