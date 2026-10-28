
# ============================================================
# STEP 19 — A3-v2 RESNET-101 BACKBONE SANITY
# FIX: A3 outputs C5 Tensor, not B2 C3/C4/C5 dict
# ============================================================

import torch
import gc

assert CONFIG["backbone"] == "resnet101"
assert CONFIG["input_size"] == 640
assert CONFIG["output_stride"] == 4

backbone.eval()

with torch.inference_mode():
    x = torch.randn(
        1, 3,
        CONFIG["input_size"],
        CONFIG["input_size"],
        device=device
    )

    features = backbone(x)

    assert isinstance(features, torch.Tensor), (
        "A3 backbone must return Tensor, not dictionary"
    )

    expected = (
        1, 2048,
        CONFIG["input_size"] // 32,
        CONFIG["input_size"] // 32
    )

    assert tuple(features.shape) == expected, (
        f"Expected {expected}, got {tuple(features.shape)}"
    )

    assert bool(torch.isfinite(features).all())

    print("Input :", tuple(x.shape))
    print("C5    :", tuple(features.shape))

del x, features
gc.collect()

if torch.cuda.is_available():
    torch.cuda.empty_cache()

print("STEP 19 PASS — A3-v2 C5 Tensor")
