# ==========================================================
# STEP 39 : Build Few-Shot DataLoader
# ==========================================================

from torch.utils.data import DataLoader

print("=" * 70)
print("STEP 39 : BUILD FEW-SHOT DATALOADER")
print("=" * 70)

# ----------------------------------------------------------
# Configuration
# ----------------------------------------------------------

SUPPORT_BATCH_SIZE = 1

QUERY_BATCH_SIZE = CONFIG["batch_size"]

# ----------------------------------------------------------
# Support Loader
# ----------------------------------------------------------

support_loader = DataLoader(

    support_dataset,

    batch_size=SUPPORT_BATCH_SIZE,

    shuffle=True,

    collate_fn=collate_fn,

    num_workers=CONFIG["num_workers"]

)

# ----------------------------------------------------------
# Query Loader
# ----------------------------------------------------------

query_loader = DataLoader(

    query_dataset,

    batch_size=QUERY_BATCH_SIZE,

    shuffle=True,

    collate_fn=collate_fn,

    num_workers=CONFIG["num_workers"]

)

# ----------------------------------------------------------
# Validation
# ----------------------------------------------------------

print("\nSupport Loader")

print(f"Batch Size : {SUPPORT_BATCH_SIZE}")

print(f"Images     : {len(support_dataset)}")

print(f"Batches    : {len(support_loader)}")

print("\nQuery Loader")

print(f"Batch Size : {QUERY_BATCH_SIZE}")

print(f"Images     : {len(query_dataset)}")

print(f"Batches    : {len(query_loader)}")

# ----------------------------------------------------------
# Sanity Check
# ----------------------------------------------------------

print("\nRunning DataLoader Sanity Check...\n")

support_images, support_targets = next(iter(support_loader))

query_images, query_targets = next(iter(query_loader))

print("Support Batch")

print(f"Images : {len(support_images)}")

print(f"Targets: {len(support_targets)}")

print()

print("Query Batch")

print(f"Images : {len(query_images)}")

print(f"Targets: {len(query_targets)}")

print()

print("=" * 70)

print("STEP 39 COMPLETED")

print("=" * 70)
