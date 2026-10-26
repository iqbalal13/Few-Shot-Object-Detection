
# STEP 23 — Shape, finite, zero-gate identity
import torch

model.eval()

images, _ = next(iter(train_loader))
x = images[:1].to(device)

with torch.inference_mode():
    fused = model(x)

    features = model.backbone(x)

    old_neck = model.neck.deconv3(
        model.neck.deconv2(
            model.neck.deconv1(features["c5"])
        )
    )
    old = model.heads(old_neck)

for key, channels in [
    ("heatmap", 80),
    ("wh", 2),
    ("offset", 2)
]:
    assert fused[key].shape == (
        1, channels, 160, 160
    )

    assert bool(torch.isfinite(fused[key]).all())

    assert torch.allclose(
        fused[key],
        old[key],
        atol=1e-6,
        rtol=1e-6
    ), key

    print(
        key,
        "max zero-gate difference",
        (fused[key] - old[key]).abs().max().item()
    )

assert (
    model.neck.gate_c3.item()
    == model.neck.gate_c4.item()
    == 0.0
)

print("STEP 23 PASS — B2 output identical to old path at gate=0")

del images, x, fused, features, old_neck, old

if torch.cuda.is_available():
    torch.cuda.empty_cache()
