# ==========================================================
# STEP 17 : Episodic Dataset & Sampling Sanity
# ==========================================================

assert "train_dataset" in globals(), (
    "Run STEP 16 first."
)


train_dataset.set_epoch(0)


# ==========================================================
# BALANCED 80-CLASS CYCLE
# ==========================================================

first_cycle_classes = [

    int(
        train_dataset[i][
            "episode_class"
        ].item()
    )

    for i in range(80)
]


assert (
    len(
        set(
            first_cycle_classes
        )
    )
    ==
    80
), (
    "First 80 episodes must cover "
    "all 80 COCO classes."
)


# ==========================================================
# EPISODE SANITY
# ==========================================================

print("=" * 70)
print("STEP 17 : EPISODIC SAMPLING SANITY")
print("=" * 70)


for i in range(5):

    episode = train_dataset[i]


    episode_class = int(

        episode[
            "episode_class"
        ].item()
    )


    support_image = (
        episode[
            "support_image"
        ]
    )

    query_image = (
        episode[
            "query_image"
        ]
    )

    support_target = (
        episode[
            "support_target"
        ]
    )

    query_target = (
        episode[
            "query_target"
        ]
    )


    assert (
        support_image.shape
        ==
        (
            3,
            CONFIG["image_size"],
            CONFIG["image_size"]
        )
    )


    assert (
        query_image.shape
        ==
        (
            3,
            CONFIG["image_size"],
            CONFIG["image_size"]
        )
    )


    assert (
        support_target[
            "labels"
        ].numel()
        ==
        1
    )


    assert (
        support_target[
            "labels"
        ][0].item()
        ==
        episode_class
    )


    assert (

        query_target[
            "labels"
        ]
        ==
        episode_class

    ).all()


    assert (
        support_target[
            "image_id"
        ].item()
        !=
        query_target[
            "image_id"
        ].item()
    )


    boxes = (
        query_target[
            "boxes"
        ]
    )


    assert (
        boxes.ndim == 2
        and
        boxes.shape[-1] == 4
    )


    assert torch.all(
        boxes >= 0.0
    )

    assert torch.all(
        boxes <= 1.0
    )

    assert torch.all(
        boxes[:, 2:] > 0.0
    )


    print(

        f"Episode {i+1} | "
        f"Class={episode_class:2d} "
        f"({CATEGORY_NAMES[episode_class]}) | "
        f"GT={len(boxes)}"
    )


print("-" * 70)

print(
    "First 80 episodes : "
    "✓ exactly 80 unique COCO classes"
)

print("=" * 70)
print("✓ STEP 17 PASSED")
print("=" * 70)
