
# STEP 19 — Multi-scale backbone sanity
import torch

backbone.eval()

with torch.inference_mode():
    x = torch.randn(
        1, 3, 640, 640,
        device=device
    )
    features = backbone(x)

expected = {
    "c3": (1, 512, 80, 80),
    "c4": (1, 1024, 40, 40),
    "c5": (1, 2048, 20, 20)
}

for key, shape in expected.items():
    assert tuple(features[key].shape) == shape, (
        key, features[key].shape
    )
    assert bool(torch.isfinite(features[key]).all())
    print(key, features[key].shape)

del x, features

if torch.cuda.is_available():
    torch.cuda.empty_cache()

print("STEP 19 PASS")
