# ============================================================
# STEP 23 — FULL MODEL FORWARD SANITY TEST
# ============================================================

print("=" * 70)
print("STEP 23 — FULL FORWARD SANITY TEST")
print("=" * 70)

model.eval()

batch_images, batch_targets = next(
    iter(train_loader)
)

sanity_images = batch_images[:1].to(
    device
)

with torch.no_grad():

    outputs = model(
        sanity_images
    )


print(
    "Input   :",
    sanity_images.shape
)

print(
    "Heatmap :",
    outputs["heatmap"].shape
)

print(
    "WH      :",
    outputs["wh"].shape
)

print(
    "Offset  :",
    outputs["offset"].shape
)


assert outputs["heatmap"].shape == (
    1,
    80,
    160,
    160
)

assert outputs["wh"].shape == (
    1,
    2,
    160,
    160
)

assert outputs["offset"].shape == (
    1,
    2,
    160,
    160
)


for name, tensor in outputs.items():

    assert torch.isfinite(
        tensor
    ).all(), f"{name} contains NaN/Inf"


print(
    "\nHeatmap logits range:",
    float(outputs["heatmap"].min()),
    "→",
    float(outputs["heatmap"].max())
)

print(
    "Heatmap probability max:",
    float(
        torch.sigmoid(
            outputs["heatmap"]
        ).max()
    )
)

print("\nSTEP 23 PASSED")

del sanity_images
del outputs

if torch.cuda.is_available():
    torch.cuda.empty_cache()
