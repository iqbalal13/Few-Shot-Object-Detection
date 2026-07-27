# ==========================================================
# STEP 38 : Build 1-Shot Support & Query Dataset
# ==========================================================

import random
from collections import defaultdict
from torch.utils.data import Subset

print("=" * 60)
print("Building 1-Shot Support & Query Dataset")
print("=" * 60)

SEED = 42
NUM_SHOTS = 1

random.seed(SEED)

# ----------------------------------------------------------
# Kelompokkan index berdasarkan class
# ----------------------------------------------------------

class_to_indices = defaultdict(list)

for idx in range(len(cctv_train_dataset)):

    _, target = cctv_train_dataset[idx]

    labels = target["labels"].tolist()

    for cls in set(labels):
        class_to_indices[cls].append(idx)

# ----------------------------------------------------------
# Ambil 1 image per class
# ----------------------------------------------------------

support_indices = set()

for cls in sorted(class_to_indices.keys()):

    candidates = class_to_indices[cls]

    if len(candidates) < NUM_SHOTS:

        raise ValueError(
            f"Class {cls} hanya memiliki {len(candidates)} image."
        )

    chosen = random.sample(candidates, NUM_SHOTS)

    support_indices.update(chosen)

support_indices = sorted(list(support_indices))

# ----------------------------------------------------------
# Query = semua image selain support
# ----------------------------------------------------------

query_indices = [

    idx

    for idx in range(len(cctv_train_dataset))

    if idx not in support_indices

]

# ----------------------------------------------------------
# Build Subset Dataset
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
# Statistik
# ----------------------------------------------------------

print("=" * 60)
print("Few-Shot Sampling Result")
print("=" * 60)

print(f"Number of Classes : {len(class_to_indices)}")
print(f"Shots per Class   : {NUM_SHOTS}")
print(f"Support Images    : {len(support_dataset)}")
print(f"Query Images      : {len(query_dataset)}")

print("=" * 60)

print("Support Index")

print(support_indices)

print("=" * 60)

# ----------------------------------------------------------
# Detail Support
# ----------------------------------------------------------

print("\nSupport Image Detail\n")

for idx in support_indices:

    _, target = cctv_train_dataset[idx]

    print(

        f"Image {idx:3d}"

        f" -> Classes : {target['labels'].tolist()}"

    )

print("=" * 60)
print("STEP 38 Finished")
print("=" * 60)
