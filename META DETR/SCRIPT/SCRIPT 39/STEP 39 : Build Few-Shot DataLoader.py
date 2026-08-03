# ==========================================================
# STEP 39 : BUILD FEW-SHOT DATALOADER
# ==========================================================

from torch.utils.data import DataLoader

print("=" * 70)
print("STEP 39 : BUILD FEW-SHOT DATALOADER")
print("=" * 70)

# ----------------------------------------------------------
# Update Configuration
# ----------------------------------------------------------

FSOD_CONFIG.update({
    "support_batch_size": 1,
    "query_batch_size": 2,
    "num_workers": 0,
    "pin_memory": True
})

# ----------------------------------------------------------
# Build Support DataLoader
# ----------------------------------------------------------

support_loader = DataLoader(
    support_dataset,
    batch_size=FSOD_CONFIG["support_batch_size"],
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=FSOD_CONFIG["num_workers"],
    pin_memory=FSOD_CONFIG["pin_memory"],
    drop_last=False
)

# ----------------------------------------------------------
# Build Query DataLoader
# ----------------------------------------------------------

query_loader = DataLoader(
    query_dataset,
    batch_size=FSOD_CONFIG["query_batch_size"],
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=FSOD_CONFIG["num_workers"],
    pin_memory=FSOD_CONFIG["pin_memory"],
    drop_last=False
)

# ----------------------------------------------------------
# DataLoader Information
# ----------------------------------------------------------

print("\nConfiguration")
print("-" * 40)

for k, v in FSOD_CONFIG.items():
    print(f"{k:20s}: {v}")

print("\nSupport Dataset")
print("-" * 40)
print(f"Images        : {len(support_dataset)}")
print(f"Batch Size    : {FSOD_CONFIG['support_batch_size']}")
print(f"Total Batches : {len(support_loader)}")

print("\nQuery Dataset")
print("-" * 40)
print(f"Images        : {len(query_dataset)}")
print(f"Batch Size    : {FSOD_CONFIG['query_batch_size']}")
print(f"Total Batches : {len(query_loader)}")

# ----------------------------------------------------------
# Validation
# ----------------------------------------------------------

assert len(support_loader) > 0, "Support DataLoader kosong."
assert len(query_loader) > 0, "Query DataLoader kosong."

# ----------------------------------------------------------
# Sanity Check
# ----------------------------------------------------------

print("\nRunning DataLoader Sanity Check...")
print("-" * 40)

support_images, support_targets = next(iter(support_loader))
query_images, query_targets = next(iter(query_loader))

print("\nSupport Batch")
print(f"Images  : {len(support_images)}")
print(f"Targets : {len(support_targets)}")

print("\nQuery Batch")
print(f"Images  : {len(query_images)}")
print(f"Targets : {len(query_targets)}")

# ----------------------------------------------------------
# Tensor Shape Check
# ----------------------------------------------------------

print("\nTensor Shape")
print("-" * 40)

print(f"Support Image Shape : {support_images[0].shape}")
print(f"Query Image Shape   : {query_images[0].shape}")

if len(support_targets) > 0:
    print(f"Support Labels      : {support_targets[0]['labels'].shape}")

if len(query_targets) > 0:
    print(f"Query Labels        : {query_targets[0]['labels'].shape}")

print("\n" + "=" * 70)
print("STEP 39 COMPLETED")
print("=" * 70)
