# ==========================================================
# STEP 18 : Collate Function & Source DataLoaders
# ==========================================================

from torch.utils.data import (
    DataLoader
)


# ==========================================================
# COLLATE
# ==========================================================

def episodic_collate_fn(
    batch
):

    support_images = torch.stack([

        episode[
            "support_image"
        ]

        for episode in batch
    ])


    query_images = torch.stack([

        episode[
            "query_image"
        ]

        for episode in batch
    ])


    episode_classes = torch.stack([

        episode[
            "episode_class"
        ]

        for episode in batch
    ]).long()


    support_targets = [

        episode[
            "support_target"
        ]

        for episode in batch
    ]


    query_targets = [

        episode[
            "query_target"
        ]

        for episode in batch
    ]


    return {

        "episode_classes":
            episode_classes,

        "support_images":
            support_images,

        "support_targets":
            support_targets,

        "query_images":
            query_images,

        "query_targets":
            query_targets
    }


# ==========================================================
# TRAIN LOADER
# ==========================================================

train_generator = (
    torch.Generator()
)

train_generator.manual_seed(
    CONFIG["seed"]
)


train_loader = DataLoader(

    dataset=
        train_dataset,

    batch_size=
        COCO_CONFIG[
            "batch_size"
        ],

    shuffle=
        False,

    num_workers=
        COCO_CONFIG[
            "num_workers"
        ],

    pin_memory=
        COCO_CONFIG[
            "pin_memory"
        ],

    collate_fn=
        episodic_collate_fn,

    drop_last=
        False,

    generator=
        train_generator
)


# ==========================================================
# FIXED COCO-VAL
# ==========================================================

val_dataset = COCOEpisodeDataset(

    coco=
        coco_val,

    image_dir=
        VAL_IMAGE_DIR,

    support_transform=
        support_transform,

    query_transform=
        query_transform,

    num_episodes=
        COCO_CONFIG[
            "num_val_episodes"
        ],

    seed=
        CONFIG["seed"]
        +
        10000,

    min_bbox_size=
        COCO_CONFIG[
            "min_bbox_size"
        ]
)


# Permanently fixed validation episodes.

val_dataset.set_epoch(0)


val_loader = DataLoader(

    dataset=
        val_dataset,

    batch_size=
        COCO_CONFIG[
            "batch_size"
        ],

    shuffle=
        False,

    num_workers=
        COCO_CONFIG[
            "num_workers"
        ],

    pin_memory=
        COCO_CONFIG[
            "pin_memory"
        ],

    collate_fn=
        episodic_collate_fn,

    drop_last=
        False
)


# ==========================================================
# LOADER SANITY
# ==========================================================

batch = next(
    iter(train_loader)
)


assert (
    batch[
        "support_images"
    ].shape
    ==
    (
        1,
        3,
        CONFIG["image_size"],
        CONFIG["image_size"]
    )
)


assert (
    batch[
        "query_images"
    ].shape
    ==
    (
        1,
        3,
        CONFIG["image_size"],
        CONFIG["image_size"]
    )
)


assert (
    len(val_dataset)
    ==
    800
)


assert (
    len(
        val_dataset.valid_labels
    )
    ==
    80
)


print("=" * 70)
print("STEP 18 : SOURCE DATALOADERS READY")
print("=" * 70)

print(
    "Train Episodes :",
    len(train_dataset)
)

print(
    "Train Batches  :",
    len(train_loader)
)

print(
    "Val Episodes   :",
    len(val_dataset)
)

print(
    "Val Classes    :",
    len(
        val_dataset.valid_labels
    )
)

print(
    "Val/Class      :",
    len(val_dataset)
    //
    len(
        val_dataset.valid_labels
    )
)

print("=" * 70)
