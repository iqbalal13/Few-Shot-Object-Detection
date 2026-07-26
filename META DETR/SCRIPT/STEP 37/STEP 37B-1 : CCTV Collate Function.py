# ==========================================================
# STEP 37B-1 : CCTV Collate Function
# ==========================================================

def cctv_collate_fn(batch):
    """
    Batch:
        [
            (image1, target1),
            (image2, target2),
            ...
        ]

    Return:
        images  -> list[Tensor]
        targets -> list[Dict]
    """

    images, targets = zip(*batch)

    return list(images), list(targets)


print("=" * 60)
print("CCTV Collate Function Ready")
print("=" * 60)
