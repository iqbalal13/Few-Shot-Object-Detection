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
# Build Support DataLoader
# ----------------------------------------------------------

support_loader = DataLoader(
    support_dataset,
    batch_size=SUPPORT_BATCH_SIZE,
    shuffle=True,
    num_workers=CONFIG["num_workers"],
    collate_fn=collate_fn,
    pin_memory=True
)

# ----------------------------------------------------------
# Build Query DataLoader
# ----------------------------------------------------------

query_loader = DataLoader(
    query_dataset,
    batch_size=QUERY_BATCH_SIZE,
    shuffle=True,
    num_workers=CONFIG["num_workers"],
    collate_fn=collate_fn,
    pin_memory=True
)

# ----------------------------------------------------------
# Dataset Statistics
# ----------------------------------------------------------

print("\nSupport Dataset")
print("-" * 40)
print(f"Images      : {len(support_dataset)}")
print(f"Batch Size  : {SUPPORT_BATCH_SIZE}")
print(f"Batches     : {len(support_loader)}")

print("\nQuery Dataset")
print("-" * 40)
print(f"Images      : {len(query_dataset)}")
print(f"Batch Size  : {QUERY_BATCH_SIZE}")
print(f"Batches     : {len(query_loader)}")

# ----------------------------------------------------------
# Sanity Check
# ----------------------------------------------------------

print("\nRunning DataLoader Sanity Check...")

support_images, support_targets = next(iter(support_loader))
query_images, query_targets = next(iter(query_loader))

print("\nSupport Batch")
print("-" * 40)
print(f"Images   : {len(support_images)}")
print(f"Targets  : {len(support_targets)}")

for i, target in enumerate(support_targets):
    print(f" Support[{i}] Labels : {target['labels'].tolist()}")

print("\nQuery Batch")
print("-" * 40)
print(f"Images   : {len(query_images)}")
print(f"Targets  : {len(query_targets)}")

for i, target in enumerate(query_targets):
    print(f" Query[{i}] Labels : {target['labels'].tolist()}")

# ----------------------------------------------------------
# Validation
# ----------------------------------------------------------

assert len(support_loader) > 0, "Support loader kosong."
assert len(query_loader) > 0, "Query loader kosong."

print("\nValidation Passed")
print("Support Loader Ready")
print("Query Loader Ready")

print("=" * 70)
print("STEP 39 COMPLETED")
print("=" * 70)

# ----------------------------------------------------------
# Output
# ----------------------------------------------------------
# support_loader
# query_loader
