
# ============================================================
# STEP 25 — A3-v2 LOSS AND GRADIENT SANITY
# NO OPTIMIZER STEP / NO TRAINING
# ============================================================

import torch

# Aktifkan A3, bukan A2-only.
criterion.neighbor_lambda = float(
    CONFIG["a3_neighbor_lambda"]
)

assert abs(criterion.neighbor_lambda - 0.25) < 1e-12
assert CONFIG["wh_loss_type"] == "hybrid_abs_relative"
assert CONFIG["gaussian_radius_mode"] == "v1_2_original"

# ------------------------------------------------------------
# 1. REAL BATCH FORWARD + BACKWARD
# ------------------------------------------------------------

model.train()

images, raw_targets = next(iter(train_loader))

images = images[:1].to(
    device, non_blocking=True
)

targets = build_batch_centernet_targets(
    raw_targets[:1],
    device=device
)

model.zero_grad(set_to_none=True)

outputs = model(images)
losses = criterion(outputs, targets)

assert bool(torch.isfinite(losses["loss_total"]))

losses["loss_total"].backward()

n_grad = 0
n_nonzero = 0

for name, p in model.named_parameters():
    if p.grad is None:
        continue

    assert bool(torch.isfinite(p.grad).all()), (
        f"Bad gradient: {name}"
    )

    n_grad += 1
    n_nonzero += int(
        bool(p.grad.abs().sum() > 0)
    )

assert n_grad > 0
assert n_nonzero > 0

print("Train batch losses:")
for key, value in losses.items():
    print(key, round(float(value.detach()), 6))

print("Finite gradient tensors:", n_grad)
print("Nonzero gradient tensors:", n_nonzero)

model.zero_grad(set_to_none=True)

del images, raw_targets, targets, outputs, losses

# ------------------------------------------------------------
# 2. GUARANTEED A3 NEIGHBOR GRADIENT TEST
# ------------------------------------------------------------

# Synthetic GT berukuran 48x48 piksel.
# Seluruh 8 grid tetangga radius 1 harus valid.

probe_target = {
    "boxes": torch.tensor(
        [[160., 160., 208., 208.]],
        dtype=torch.float32
    ),
    "labels": torch.tensor(
        [0], dtype=torch.long
    )
}

probe_gt = build_batch_centernet_targets(
    [probe_target],
    device=device
)

assert int(
    probe_gt["neighbor_mask"].sum()
) == 8

H = (
    CONFIG["input_size"]
    // CONFIG["output_stride"]
)

probe_wh = torch.zeros(
    1, 2, H, H,
    device=device,
    requires_grad=True
)

probe_pred = {
    "heatmap": torch.zeros(
        1, CONFIG["num_classes"], H, H,
        device=device
    ),
    "wh": probe_wh,
    "offset": torch.zeros(
        1, 2, H, H,
        device=device
    )
}

probe_loss = criterion(
    probe_pred,
    probe_gt
)["loss_wh_neighbor"]

assert float(probe_loss.detach()) > 0

probe_loss.backward()

neighbor_idx = (
    probe_gt["neighbor_indices"][0, 0][
        probe_gt["neighbor_mask"][0, 0]
    ]
)

neighbor_grad = (
    probe_wh.grad[0]
    .reshape(2, -1)[:, neighbor_idx]
)

assert bool(
    (neighbor_grad.abs().sum(dim=0) > 0).all()
), "A3 neighbor gradients disconnected"

print("A3 neighbor lambda:", criterion.neighbor_lambda)
print("8 neighbor gradient paths: PASS")
print("STEP 25 PASS — A3 supervision active")

del (
    probe_target, probe_gt, probe_wh,
    probe_pred, probe_loss
)

if torch.cuda.is_available():
    torch.cuda.empty_cache()
