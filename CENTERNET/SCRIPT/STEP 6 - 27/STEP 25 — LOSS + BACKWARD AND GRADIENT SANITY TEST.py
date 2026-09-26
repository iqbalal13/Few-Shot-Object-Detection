# ============================================================
# STEP 25 — LOSS + BACKWARD / GRADIENT SANITY TEST
# ============================================================

print("=" * 70)
print("STEP 25 — BACKWARD / GRADIENT SANITY TEST")
print("=" * 70)

model.train()

batch_images, batch_targets = next(
    iter(train_loader)
)

# Batch size 1 for safe Colab sanity test
sanity_images = batch_images[:1].to(
    device,
    non_blocking=True
)

sanity_targets_raw = batch_targets[:1]

encoded_targets = \
    build_batch_centernet_targets(
        sanity_targets_raw,
        device=device
    )

model.zero_grad(
    set_to_none=True
)

outputs = model(
    sanity_images
)

losses = criterion(
    outputs,
    encoded_targets
)

total_loss = losses[
    "loss_total"
]

print(
    "Heatmap loss :",
    float(
        losses["loss_heatmap"].detach()
    )
)

print(
    "WH loss      :",
    float(
        losses["loss_wh"].detach()
    )
)

print(
    "Offset loss  :",
    float(
        losses["loss_offset"].detach()
    )
)

print(
    "Total loss   :",
    float(
        total_loss.detach()
    )
)


assert torch.isfinite(
    total_loss
), "Loss contains NaN or Inf"


total_loss.backward()


grad_tensors = 0
finite_grad_tensors = 0
nonzero_grad_tensors = 0

for parameter in model.parameters():

    if parameter.grad is None:
        continue

    grad_tensors += 1

    if torch.isfinite(
        parameter.grad
    ).all():

        finite_grad_tensors += 1

    if parameter.grad.abs().sum() > 0:

        nonzero_grad_tensors += 1


print("\nGradient tensors       :", grad_tensors)
print("Finite gradient tensors:", finite_grad_tensors)
print("Non-zero gradients     :", nonzero_grad_tensors)

assert grad_tensors > 0

assert (
    finite_grad_tensors
    == grad_tensors
), "Non-finite gradients detected"

assert (
    nonzero_grad_tensors > 0
), "No non-zero gradient detected"


model.zero_grad(
    set_to_none=True
)

del outputs
del losses
del total_loss
del encoded_targets
del sanity_images

if torch.cuda.is_available():
    torch.cuda.empty_cache()

print("\nSTEP 25 PASSED")
