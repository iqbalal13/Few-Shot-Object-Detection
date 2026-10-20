
# ============================================================
# STEP 17B — B1 TARGET ISOLATION & WORKER PREFLIGHT
# NO TRAINING
# ============================================================

import torch
from torch.utils.data import DataLoader, Subset

print("=" * 70)
print("STEP 17B — B1 TARGET ISOLATION TEST")
print("=" * 70)

assert CONFIG["gaussian_radius_mode"] == "v1_3_corrected"

# Synthetic 48x48-pixel bbox => 12x12 output cells.
# V1.2 radius=1, V1.3 radius=0.

synthetic = {
    "boxes": torch.tensor(
        [[200., 200., 248., 248.]],
        dtype=torch.float32
    ),
    "labels": torch.tensor(
        [0], dtype=torch.long
    )
}

v12 = build_centernet_target(
    synthetic,
    gaussian_radius_mode="v1_2_original"
)

v13 = build_centernet_target(
    synthetic,
    gaussian_radius_mode="v1_3_corrected"
)

# Regression targets must remain identical.
for key in (
    "wh", "offset", "indices", "mask", "labels"
):
    assert torch.equal(v12[key], v13[key]), key

for key in (
    "num_objects", "num_candidates", "num_dropped"
):
    assert v12[key] == v13[key], key

assert not torch.equal(
    v12["heatmap"],
    v13["heatmap"]
), "B1 failed: Gaussian targets are identical."

center_index = int(v12["indices"][0])
cy, cx = divmod(center_index, OUTPUT_WIDTH)

assert v12["heatmap"][0, cy, cx] == 1.0
assert v13["heatmap"][0, cy, cx] == 1.0

assert (
    v12["heatmap"][0, cy, cx+1]
    > v13["heatmap"][0, cy, cx+1]
)

print("Synthetic V1.2 vs V1.3: PASSED")


# Test real COCO targets
changed_samples = 0

for i in range(min(64, len(train_dataset))):
    _, raw_target = train_dataset[i]

    t12 = build_centernet_target(
        raw_target,
        gaussian_radius_mode="v1_2_original"
    )

    t13 = build_centernet_target(
        raw_target,
        gaussian_radius_mode="v1_3_corrected"
    )

    for key in (
        "wh", "offset", "indices", "mask", "labels"
    ):
        assert torch.equal(
            t12[key], t13[key]
        ), f"Regression target changed: {key}"

    if not torch.equal(
        t12["heatmap"], t13["heatmap"]
    ):
        changed_samples += 1

print(
    "Real COCO samples with changed heatmap:",
    changed_samples,
    "/ 64"
)


# Worker-encoded preflight
# Uses newly created workers, not old persistent workers.
test_loader = DataLoader(
    Subset(train_dataset, range(8)),
    batch_size=8,
    shuffle=False,
    num_workers=2,
    persistent_workers=False,
    collate_fn=centernet_encoded_collate_fn
)

images17b, raw17b, encoded17b = next(
    iter(test_loader)
)

for i, target in enumerate(raw17b):
    direct = build_centernet_target(
        target,
        gaussian_radius_mode="v1_3_corrected"
    )

    for key in (
        "heatmap", "wh", "offset", "indices", "mask"
    ):
        assert torch.equal(
            encoded17b[key][i],
            direct[key]
        ), f"Worker target mismatch: {key}"

assert encoded17b["heatmap"].shape == (
    8, 80, 160, 160
)

del test_loader, images17b, raw17b, encoded17b

print("DataLoader worker target: PASSED")
print("STEP 17B PASSED — READY FOR B1 TRAINING")
