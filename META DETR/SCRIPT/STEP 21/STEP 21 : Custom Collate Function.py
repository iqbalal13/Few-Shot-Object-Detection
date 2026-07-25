# ==========================================================
# STEP 21 : Episodic Collate Function
# ==========================================================

def collate_fn(batch):

    support_images = []
    support_targets = []

    query_images = []
    query_targets = []

    for episode in batch:

        support_images.append(
            episode["support_image"]
        )

        support_targets.append(
            episode["support_target"]
        )

        query_images.append(
            episode["query_image"]
        )

        query_targets.append(
            episode["query_target"]
        )

    return {

        "support_images": support_images,
        "support_targets": support_targets,

        "query_images": query_images,
        "query_targets": query_targets

    }

print("=" * 60)
print("Custom Episodic Collate Function Ready")
print("=" * 60)
