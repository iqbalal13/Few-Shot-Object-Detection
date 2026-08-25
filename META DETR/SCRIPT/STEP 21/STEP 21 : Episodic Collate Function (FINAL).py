# ==========================================================
# STEP 21 : Episodic Collate Function (FINAL)
# ==========================================================

import torch


def episodic_collate_fn(batch):

    # ======================================================
    # Images
    # All images already resized to the same size:
    # [3, 640, 640]
    # ======================================================

    support_images = torch.stack([
        episode["support_image"]
        for episode in batch
    ])

    query_images = torch.stack([
        episode["query_image"]
        for episode in batch
    ])


    # ======================================================
    # Episode Class
    # ======================================================

    episode_classes = torch.stack([
        episode["episode_class"]
        for episode in batch
    ]).long()


    # ======================================================
    # Targets
    #
    # Keep targets as list[dict]
    # because number of boxes differs per image.
    # ======================================================

    support_targets = [
        episode["support_target"]
        for episode in batch
    ]

    query_targets = [
        episode["query_target"]
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


# Keep old name for compatibility
collate_fn = episodic_collate_fn


print("=" * 70)
print("STEP 21 : EPISODIC COLLATE FUNCTION READY")
print("=" * 70)
