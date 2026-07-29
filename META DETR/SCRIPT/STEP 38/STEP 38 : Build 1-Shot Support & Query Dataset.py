# ==========================================================
# STEP 38 : BUILD 1-SHOT SUPPORT & QUERY DATASET
# ==========================================================

import random
import numpy as np
import torch
from collections import defaultdict
from torch.utils.data import Subset

print("=" * 70)
print("STEP 38 : BUILD 1-SHOT SUPPORT & QUERY DATASET")
print("=" * 70)

# ----------------------------------------------------------
# FSOD Configuration
# ----------------------------------------------------------

FSOD_CONFIG = {
    "seed": 42,
    "num_shots": 1
}

random.seed(FSOD_CONFIG["seed"])
np.random.seed(FSOD_CONFIG["seed"])
torch.manual_seed(FSOD_CONFIG["seed"])

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(FSOD_CONFIG["seed"])

print("\nFSOD Configuration")
print("-" * 40)
for k, v in FSOD_CONFIG.items():
    print(f"{k:15s}: {v}")

# ----------------------------------------------------------
# Build Class -> Image Index Mapping
# ----------------------------------------------------------

class_to_indices = defaultdict(list)

for idx in range(len(cctv_train_dataset)):

    _, target = cctv_train_dataset[idx]

    labels = target["labels"].tolist()

    for cls in set(labels):
        class_to_indices[cls].append(idx)

print(f"\nDetected Classes : {len(class_to_indices)}")

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

    # Prioritize unique images
    for idx in candidates:

        if idx not in used_images:

            chosen.append(idx)
            used_images.add(idx)

        if len(chosen) == FSOD_CONFIG["num_shots"]:
            break

    # Fallback if unique images are insufficient
    if len(chosen) < FSOD_CONFIG["num_shots"]:

        remaining = [

            idx

            for idx in candidates

            if idx not in chosen

        ]

        need = FSOD_CONFIG["num_shots"] - len(chosen)

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

expected_support = (
    len(class_to_indices)
    * FSOD_CONFIG["num_shots"]
)

assert len(support_indices) == expected_support

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

print(f"Random Seed       : {FSOD_CONFIG['seed']}")
print(f"Shots per Class   : {FSOD_CONFIG['num_shots']}")
print(f"Detected Classes  : {len(class_to_indices)}")
print(f"Support Images    : {len(support_dataset)}")
print(f"Query Images      : {len(query_dataset)}")

print(
    f"Support Ratio     : "
    f"{len(support_dataset)/len(cctv_train_dataset):.2%}"
)

print(
    f"Query Ratio       : "
    f"{len(query_dataset)/len(cctv_train_dataset):.2%}"
)

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
