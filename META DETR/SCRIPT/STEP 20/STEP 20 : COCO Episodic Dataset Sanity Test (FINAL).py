# ==========================================================
# STEP 20 : COCO Episodic Dataset Sanity Test (FINAL)
# ==========================================================

import torch


print("=" * 70)
print("STEP 20 : EPISODIC DATASET SANITY CHECK")
print("=" * 70)


NUM_TEST_EPISODES = 5


for i in range(
    NUM_TEST_EPISODES
):

    episode = train_dataset[i]

    episode_class = (
        episode[
            "episode_class"
        ].item()
    )

    support_image = (
        episode[
            "support_image"
        ]
    )

    support_target = (
        episode[
            "support_target"
        ]
    )

    query_image = (
        episode[
            "query_image"
        ]
    )

    query_target = (
        episode[
            "query_target"
        ]
    )


    # ======================================================
    # Shape Validation
    # ======================================================

    assert support_image.shape == (
        3,
        CONFIG["image_size"],
        CONFIG["image_size"]
    )

    assert query_image.shape == (
        3,
        CONFIG["image_size"],
        CONFIG["image_size"]
    )


    # ======================================================
    # Support must contain exactly
    # one target-class object
    # ======================================================

    assert (
        len(
            support_target[
                "labels"
            ]
        )
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


    # ======================================================
    # Every query target must belong
    # to episode class
    # ======================================================

    assert (
        query_target[
            "labels"
        ]
        ==
        episode_class
    ).all()


    # ======================================================
    # Independent support/query images
    # ======================================================

    assert (
        support_target[
            "image_id"
        ].item()
        !=
        query_target[
            "image_id"
        ].item()
    )


    # ======================================================
    # Bounding Box Validation
    # ======================================================

    query_boxes = (
        query_target[
            "boxes"
        ]
    )

    assert query_boxes.ndim == 2

    assert query_boxes.shape[1] == 4

    assert torch.all(
        query_boxes >= 0.0
    )

    assert torch.all(
        query_boxes <= 1.0
    )

    assert torch.all(
        query_boxes[:, 2:] > 0
    )


    # ======================================================
    # Print Episode
    # ======================================================

    class_name = CATEGORY_NAMES[
        episode_class
    ]

    print(
        f"Episode {i + 1}"
    )

    print(
        f"  Class          : "
        f"{episode_class} "
        f"({class_name})"
    )

    print(
        f"  Support Image  : "
        f"{support_target['image_id'].item()}"
    )

    print(
        f"  Query Image    : "
        f"{query_target['image_id'].item()}"
    )

    print(
        f"  Support Shape  : "
        f"{tuple(support_image.shape)}"
    )

    print(
        f"  Query Shape    : "
        f"{tuple(query_image.shape)}"
    )

    print(
        f"  Query Objects  : "
        f"{len(query_target['boxes'])}"
    )

    print("-" * 70)


print("✓ ALL EPISODIC TESTS PASSED")
print("=" * 70)
