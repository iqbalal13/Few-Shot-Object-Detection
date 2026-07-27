# ==========================================================
# STEP 38 : Build 1-Shot Support & Query Dataset
# ==========================================================

import random
from collections import defaultdict
from torch.utils.data import Subset

print("=" * 70)
print("STEP 38 : BUILD 1-SHOT SUPPORT & QUERY DATASET")
print("=" * 70)

# ----------------------------------------------------------
# Configuration
# ----------------------------------------------------------

SEED = 42
NUM_SHOTS = 1

random.seed(SEED)

# ----------------------------------------------------------
# Build Class -> Image Index Mapping
# ----------------------------------------------------------

class_to_indices = defaultdict(list)

for idx in range(len(cctv_train_dataset)):

    _, target = cctv_train_dataset[idx]

    labels = target["labels"].tolist()

    for cls in set(labels):
        class_to_indices[cls].append(idx)

print(f"\nDetected {len(class_to_indices)} Classes")

# ----------------------------------------------------------
# Select Support Images
# ----------------------------------------------------------

used_images = set()

support_indices = []

support_mapping = {}

for cls in sorted(class_to_indices.keys()):

    candidates = class_to_indices[cls][:]

    random.shuffle(candidates)

    chosen = []

    # Prioritaskan image yang belum pernah dipakai
    for idx in candidates:

        if idx not in used_images:

            chosen.append(idx)
            used_images.add(idx)

        if len(chosen) == NUM_SHOTS:
            break

    # Jika jumlah image unik kurang,
    # gunakan image yang tersisa (fallback)
    if len(chosen) < NUM_SHOTS:

        remaining = [

            idx

            for idx in candidates

            if idx not in chosen

        ]

        need = NUM_SHOTS - len(chosen)

        if len(remaining) < need:

            raise ValueError(
                f"Class {cls} hanya memiliki "
                f"{len(candidates)} image."
            )

        extra = random.sample(remaining, need)

        chosen.extend(extra)

    support_mapping[cls] = chosen

    support_indices.extend(chosen)

support_indices = sorted(list(set(support_indices)))

# ----------------------------------------------------------
# Build Query Index
# ----------------------------------------------------------

query_indices = [

    idx

    for idx in range(len(cctv_train_dataset))

    if idx not in support_indices

]

# ----------------------------------------------------------
# Validation
# ----------------------------------------------------------

assert len(set(support_indices)) == len(support_indices)

assert len(

    set(support_indices)

    &

    set(query_indices)

) == 0

assert (

    len(support_indices)

    +

    len(query_indices)

) == len(cctv_train_dataset)

# ----------------------------------------------------------
# Build Dataset
# ----------------------------------------------------------

support_dataset = Subset(

    cctv_train_dataset,

    support_indices

)

query_dataset = Subset(

    cctv_train_dataset,

    query_indices

)

# ----------------------------------------------------------
# Statistics
# ----------------------------------------------------------

print("\n" + "=" * 70)
print("Few-Shot Sampling Summary")
print("=" * 70)

print(f"Random Seed        : {SEED}")
print(f"Shots per Class    : {NUM_SHOTS}")
print(f"Number of Classes  : {len(class_to_indices)}")
print(f"Support Images     : {len(support_dataset)}")
print(f"Query Images       : {len(query_dataset)}")

print("=" * 70)

# ----------------------------------------------------------
# Support Mapping
# ----------------------------------------------------------

print("\nSupport Mapping")

for cls in sorted(support_mapping.keys()):

    print(f"\nClass {cls}")

    for img_idx in support_mapping[cls]:

        _, target = cctv_train_dataset[img_idx]

        print(
            f"  Image {img_idx:3d}"
            f" -> Labels {target['labels'].tolist()}"
        )

# ----------------------------------------------------------
# Dataset Validation
# ----------------------------------------------------------

print("\n" + "=" * 70)
print("Validation")
print("=" * 70)

print("Unique Support Images :", len(set(support_indices)))
print("Overlap               :", len(set(support_indices) & set(query_indices)))
print("Total Dataset         :", len(cctv_train_dataset))

print("=" * 70)
print("STEP 38 COMPLETED")
print("=" * 70)

# ----------------------------------------------------------
# Output Variables
# ----------------------------------------------------------
# support_indices
# query_indices
# support_dataset
# query_dataset
# support_mapping

print("=" * 60)
print("STEP 38 Finished")
print("=" * 60)
