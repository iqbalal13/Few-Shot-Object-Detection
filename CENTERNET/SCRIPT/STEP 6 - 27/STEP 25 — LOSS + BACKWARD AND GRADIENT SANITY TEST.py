
# ============================================================
# STEP 25 — A3-v2 PEAK-GEOMETRY GRADIENT SANITY
# ============================================================

import torch

assert criterion.neighbor_lambda == 0.25
assert criterion.offset_neighbor_lambda == 0.25
assert CONFIG["a3_neighbor_radius"] == 1

model.train()

images, raw_targets = next(iter(train_loader))

images = images[:1].to(device)

encoded = build_batch_centernet_targets(
    raw_targets[:1],
    device
)

model.zero_grad(set_to_none=True)

outputs = model(images)
losses = criterion(outputs, encoded)

assert bool(torch.isfinite(losses["loss_total"]))

losses["loss_total"].backward()

active, finite = 0, 0

for name, p in model.named_parameters():

    if p.grad is not None:
        active += 1

        assert bool(torch.isfinite(p.grad).all()), (
            f"Bad gradient: {name}"
        )

        finite += int(
            bool(p.grad.abs().sum() > 0)
        )

assert active > 0 and finite > 0

print("Real batch:", {
    k: round(float(v.detach()), 6)
    for k, v in losses.items()
})

model.zero_grad(set_to_none=True)

del images, raw_targets, encoded, outputs, losses

# ------------------------------------------------------------
# SYNTHETIC 8-NEIGHBOR GRADIENT TEST
# ------------------------------------------------------------

probe_raw = [{
    "boxes": torch.tensor([
        [160., 160., 208., 208.]
    ]),
    "labels": torch.tensor(
        [0], dtype=torch.long
    )
}]

probe = build_batch_centernet_targets(
    probe_raw,
    device
)

assert int(probe["neighbor_mask"].sum()) == 8

H = (
    CONFIG["input_size"]
    // CONFIG["output_stride"]
)

probe_wh = torch.zeros(
    1, 2, H, H,
    device=device,
    requires_grad=True
)

probe_off = torch.zeros(
    1, 2, H, H,
    device=device,
    requires_grad=True
)

probe_outputs = {
    "heatmap": torch.zeros(
        1, 80, H, H,
        device=device
    ),
    "wh": probe_wh,
    "offset": probe_off
}

probe_losses = criterion(
    probe_outputs,
    probe
)

assert float(
    probe_losses["loss_wh_neighbor"].detach()
) > 0

assert float(
    probe_losses["loss_offset_neighbor"].detach()
) > 0

# Verify WH gradient on all 8 valid neighbor cells.
probe_losses["loss_wh_neighbor"].backward(
    retain_graph=True
)

ind = probe["neighbor_indices"][0, 0][
    probe["neighbor_mask"][0, 0]
]

wgrad = probe_wh.grad[0].reshape(
    2, -1
)[:, ind]

assert bool(
    (wgrad.abs().sum(0) > 0).all()
), "WH neighbor gradient disconnected"

# Verify OFFSET gradient on all 8 valid neighbor cells.
probe_losses["loss_offset_neighbor"].backward()

ograd = probe_off.grad[0].reshape(
    2, -1
)[:, ind]

assert bool(
    (ograd.abs().sum(0) > 0).all()
), "Offset neighbor gradient disconnected"

center_i = int(probe["indices"][0, 0])

center_xy = (
    center_i % H,
    center_i // H
)

neighbor_xy = [
    (int(v) % H, int(v) // H)
    for v in ind
]

print("GT center feature cell:", center_xy)
print("8 neighbor feature cells:", neighbor_xy)
print("WH / offset neighbor gradient: PASS")
print("STEP 25 PASS — no optimizer step performed")

model.zero_grad(set_to_none=True)

del (
    probe, probe_wh, probe_off,
    probe_outputs, probe_losses
)

if torch.cuda.is_available():
    torch.cuda.empty_cache()
