# ==========================================================
# STEP 18 : Generic Episodic DataLoaders
# ==========================================================

from torch.utils.data import (
    DataLoader,
    Subset,
)


def episodic_collate_fn(
    batch
):

    return {

        "support_images":
            torch.stack([
                item[
                    "support_image"
                ]
                for item in batch
            ]),

        "query_images":
            torch.stack([
                item[
                    "query_image"
                ]
                for item in batch
            ]),

        "episode_classes":
            torch.stack([
                item[
                    "episode_class"
                ]
                for item in batch
            ]),

        "support_targets":
            [
                item[
                    "support_target"
                ]
                for item in batch
            ],

        "query_targets":
            [
                item[
                    "query_target"
                ]
                for item in batch
            ],
    }


def make_episode_loader(
    dataset,
    batch_size=None,
    num_workers=None,
):

    if batch_size is None:

        batch_size = (
            COCO_CONFIG[
                "batch_size"
            ]
        )

    if num_workers is None:

        num_workers = (
            COCO_CONFIG[
                "num_workers"
            ]
        )

    return DataLoader(

        dataset,

        batch_size=
            int(
                batch_size
            ),

        # Episode class ordering is already
        # deterministically shuffled inside dataset.
        shuffle=False,

        num_workers=
            int(
                num_workers
            ),

        pin_memory=
            COCO_CONFIG[
                "pin_memory"
            ],

        persistent_workers=False,

        collate_fn=
            episodic_collate_fn,
    )


train_loader = (
    make_episode_loader(
        train_dataset
    )
)


val_loader = (
    make_episode_loader(
        val_dataset
    )
)


# ==========================================================
# LOADER SANITY
# ==========================================================

_loader_batch = next(
    iter(
        train_loader
    )
)

assert (
    _loader_batch[
        "support_images"
    ].ndim
    ==
    4
)

assert (
    _loader_batch[
        "query_images"
    ].ndim
    ==
    4
)

assert (
    len(
        _loader_batch[
            "query_targets"
        ]
    )
    ==
    _loader_batch[
        "query_images"
    ].shape[0]
)


print("=" * 70)
print("STEP 18 : EPISODIC DATALOADERS READY")
print("=" * 70)

print(
    "Train batches :",
    len(
        train_loader
    )
)

print(
    "Val batches   :",
    len(
        val_loader
    )
)

print(
    "Batch size    :",
    COCO_CONFIG[
        "batch_size"
    ]
)

print("=" * 70)


del _loader_batch
